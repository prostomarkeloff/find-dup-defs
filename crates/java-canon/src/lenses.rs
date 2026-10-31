//! **Lenses for Java** — the walk, and only the walk.
//!
//! The vocabulary, the stitching, the corpus scoring and the record itself live in
//! [`dup_defs_core::lens`]; the `use` lens's sites are [`crate::canon::collect_sites`]. What is here
//! is how Java answers the other questions, read off the owned tree ([`crate::ir`]).
//!
//! Three have a Java answer worth spelling out:
//!
//! * **`failures`** — Java fails by throwing. What a body fails *with* is the exception type it
//!   constructs (`throw new IOException(…)` → `IOException`'s dotted path), the types it declares it
//!   catches, and the library spellings that turn a precondition into an exception:
//!   `Objects.requireNonNull`, Guava's `Preconditions.check*`, and the `assert` statement.
//! * **`resources`** — what a body holds open is what it declares in a try-with-resources head (each
//!   resource's acquisition — `Files.newInputStream(p)` holds `Files.newInputStream`, `new Socket()`
//!   holds `Socket`) and what it releases on the way out (a `.close()` on a value, read in a
//!   `finally`).
//! * **`decorators`** — the annotations written on the definition, by dotted name, save the two that
//!   are pure noise: `Override` (every override carries it) and `SuppressWarnings` (an IDE directive,
//!   not a role).
//!
//! One reading worth naming: the tree is read *without* a type checker, and [`crate::ir`] folds a
//! static call into the same shape as an instance one — `Files.newInputStream(p)` is a
//! [`crate::ir::Expr::Method`] whose receiver is the name `Files`. The walk tells them apart the way
//! the canonicals do: a receiver that is `this`, a bound name, the file's own class, an imported
//! class, or (by Java's universal convention) an uppercase name is a class — a static call, spelled
//! `scope.name` — anything else is a value, and the *method* is the grammar (`.name`).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use dup_defs_core::lens::{control_tag, Lens, LensFacts};

use crate::canon::{for_each_child, Child};
use crate::ir::{Catch, ClassDecl, ClassMember, Else, Expr, Func, Resource, Stmt, Type};

/// The dotted spelling of a name, resolved through the file's imports: an imported alias becomes its
/// dotted path, an unimported name stays as written.
fn resolve(imports: &HashMap<String, Arc<str>>, name: &str) -> String {
    imports.get(name).map_or_else(|| name.to_owned(), std::string::ToString::to_string)
}

/// Whether a free name reads as a class by Java's convention — an uppercase initial. A static call
/// written without an import (`Files.newInputStream(p)` in a same-package file) is told from an
/// instance call (`file.write(...)`) by nothing else.
fn is_class_name(name: &str) -> bool {
    name.chars().next().is_some_and(|c| c.is_ascii_uppercase())
}

/// Facts gathered in one walk of a body, sliced up per lens afterwards.
#[derive(Default)]
struct Walked {
    /// Every external callee, in call order — the `effects` lens; deduped, it is `outgoing`.
    effects: Vec<String>,
    /// Control-flow tags carrying their nesting.
    control: Vec<String>,
    /// What the body fails with.
    failures: BTreeSet<String>,
    /// What the body holds open for the length of the method.
    resources: Vec<String>,
}

struct Walk<'a> {
    imports: &'a HashMap<String, Arc<str>>,
    /// Names the file itself introduced — its classes — erased, exactly as on the widest rung of the
    /// ladder, so a lens is not held apart by the very identities it exists to see past.
    file_names: &'a HashSet<String>,
    /// Names the callable binds: calling one is calling what it was handed, not a dependency.
    locals: &'a HashSet<String>,
    depth: usize,
    /// Whether the walk is inside a `finally` block — where a `.close()` is a release, not a call.
    in_finally: bool,
    facts: Walked,
}

impl Walk<'_> {
    fn tag(&mut self, tag: &str) {
        self.facts.control.push(control_tag(self.depth, tag));
    }

    fn nested(&mut self, f: impl FnOnce(&mut Self)) {
        self.depth += 1;
        f(self);
        self.depth -= 1;
    }

    /// The name a call names, when it names something the file did not introduce: a method on a free
    /// receiver yields `.m` — the receiver is whatever it is, the *method* is the grammar — a static
    /// `C.m` its resolved scope, a `new C` its resolved class, a bare `f(…)` its resolved name. A
    /// call through a local, or on `this`, or of the file's own class, is internal: none.
    fn callee(&self, e: &Expr) -> Option<String> {
        match e {
            Expr::Method { recv, name, .. } => match recv.as_ref() {
                Expr::Name(n) if n == "this" => None,
                Expr::Name(n) if self.locals.contains(n) => None,
                Expr::Name(n) if self.file_names.contains(n) => None,
                Expr::Name(n) if self.imports.contains_key(n) => Some(format!("{}.{}", resolve(self.imports, n), name)),
                Expr::Name(n) if is_class_name(n) => Some(format!("{n}.{name}")),
                _ => Some(format!(".{name}")),
            },
            Expr::StaticCall { scope, name, .. } => {
                if self.file_names.contains(scope) {
                    return None;
                }
                Some(format!("{}.{}", resolve(self.imports, scope), name))
            }
            Expr::New { class, .. } => {
                if self.file_names.contains(class) {
                    return None;
                }
                Some(resolve(self.imports, class))
            }
            Expr::Call { f, .. } => match f.as_ref() {
                Expr::Name(n) if !self.locals.contains(n) && !self.file_names.contains(n) => Some(resolve(self.imports, n)),
                _ => None,
            },
            _ => None,
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    #[allow(clippy::too_many_lines)] // one arm per statement kind reads better straight-through
    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::If { cond, then, els } => {
                self.expr(cond);
                self.tag("if");
                self.nested(|w| w.stmts(then));
                if let Some(els) = els {
                    self.tag("else");
                    self.nested(|w| match els.as_ref() {
                        Else::Block(b) => w.stmts(b),
                        Else::If(alt) => w.stmt(alt),
                    });
                }
            }
            Stmt::While { cond, body } => {
                self.tag("while");
                self.expr(cond);
                self.nested(|w| w.stmts(body));
            }
            Stmt::DoWhile { body, cond } => {
                self.tag("do");
                self.nested(|w| w.stmts(body));
                self.expr(cond);
            }
            Stmt::For { init, cond, update, body } => {
                self.tag("for");
                if let Some(init) = init {
                    self.stmt(init);
                }
                if let Some(cond) = cond {
                    self.expr(cond);
                }
                for u in update {
                    self.expr(u);
                }
                self.nested(|w| w.stmts(body));
            }
            Stmt::ForEach { iter, body, .. } => {
                self.tag("foreach");
                self.expr(iter);
                self.nested(|w| w.stmts(body));
            }
            Stmt::Switch { value, cases } => {
                self.expr(value);
                self.tag("switch");
                self.nested(|w| {
                    for case in cases {
                        w.tag("case");
                        for v in &case.values {
                            w.expr(v);
                        }
                        w.nested(|w| w.stmts(&case.body));
                    }
                });
            }
            Stmt::Try { resources, body, catches, finally } => {
                self.tag("try");
                self.nested(|w| {
                    for r in resources {
                        w.resource(r);
                    }
                    w.stmts(body);
                    for c in catches {
                        w.tag("catch");
                        w.catch_types(c);
                        w.nested(|w| w.stmts(&c.body));
                    }
                    if let Some(fin) = finally {
                        w.tag("finally");
                        w.nested(|w| {
                            let was = w.in_finally;
                            w.in_finally = true;
                            w.stmts(fin);
                            w.in_finally = was;
                        });
                    }
                });
            }
            Stmt::Synchronized { lock, body } => {
                self.tag("synchronized");
                self.expr(lock);
                self.nested(|w| w.stmts(body));
            }
            Stmt::Throw(e) => {
                self.thrown(e);
                self.expr(e);
            }
            Stmt::Assert(e) => {
                self.facts.failures.insert("assert".to_owned());
                self.expr(e);
            }
            Stmt::Let { init, .. } => {
                if let Some(init) = init {
                    self.expr(init);
                }
            }
            Stmt::Decl(items) => self.stmts(items),
            Stmt::Assign { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            Stmt::Update { e, .. } | Stmt::Expr(e) | Stmt::Return(Some(e)) => self.expr(e),
            Stmt::Block(b) => self.stmts(b),
            Stmt::Labeled { stmt, .. } => self.stmt(stmt),
            Stmt::Return(None) | Stmt::Break(_) | Stmt::Continue(_) | Stmt::NestedItem | Stmt::Unknown(_) => {}
        }
    }

    fn expr(&mut self, e: &Expr) {
        self.failure_of_call(e);
        if let Some(callee) = self.callee(e) {
            self.facts.effects.push(callee);
        }
        // A `.close()` in a `finally` block is a release, not a call — the fact is the release itself.
        if self.in_finally {
            if let Expr::Method { name, .. } = e {
                if name == "close" {
                    self.facts.resources.push("close".to_owned());
                }
            }
        }
        for_each_child(e, &mut |c| match c {
            Child::Expr(x) => self.expr(x),
            Child::Body(b) => self.stmts(b),
        });
    }

    /// The library spellings that turn a precondition into an exception, read as failures, by their
    /// as-written name — `Objects.requireNonNull`, Guava's `Preconditions.check*`.
    fn failure_of_call(&mut self, e: &Expr) {
        let (scope, name) = match e {
            Expr::Method { recv, name, .. } => match recv.as_ref() {
                Expr::Name(scope) => (scope.as_str(), name.as_str()),
                _ => return,
            },
            Expr::StaticCall { scope, name, .. } => (scope.as_str(), name.as_str()),
            _ => return,
        };
        if name == "requireNonNull" && (scope == "Objects" || scope.ends_with(".Objects")) {
            self.facts.failures.insert(format!("{scope}.{name}"));
        }
        if name.starts_with("check") && (scope == "Preconditions" || scope.ends_with(".Preconditions")) {
            self.facts.failures.insert(format!("{scope}.{name}"));
        }
    }

    /// What a `throw` throws: the exception type it constructs, by dotted path.
    fn thrown(&mut self, e: &Expr) {
        if let Expr::New { class, .. } = e {
            self.facts.failures.insert(resolve(self.imports, class));
        }
    }

    /// The exception types a `catch` catches, each by dotted path.
    fn catch_types(&mut self, c: &Catch) {
        for t in &c.types {
            let mut names = Vec::new();
            type_shape(t, self.imports, &[], &mut names);
            for n in names {
                self.facts.failures.insert(format!("catch {n}"));
            }
        }
    }

    /// What a try-with-resources resource acquires: the call or construction its value names.
    fn resource(&mut self, r: &Resource) {
        if let Some(callee) = self.callee(&r.value) {
            self.facts.resources.push(callee);
        }
        self.expr(&r.value);
    }
}

/// The names a type spells, import-resolved, so `List<String>` and `List<Order>` agree on
/// `java.util.List` and say nothing else. A name the definition's own type parameters bind is erased
/// to `_` — the definition's identity, not its vocabulary.
fn type_shape(t: &Type, imports: &HashMap<String, Arc<str>>, erase: &[String], out: &mut Vec<String>) {
    match t {
        Type::Named(n) => out.push(if erase.iter().any(|t| t == n) { "_".to_owned() } else { resolve(imports, n) }),
        Type::Generic(name, args) => {
            out.push(if erase.iter().any(|t| t == name) { "_".to_owned() } else { resolve(imports, name) });
            for a in args {
                type_shape(a, imports, erase, out);
            }
        }
        Type::Array(inner) => {
            out.push("[]".to_owned());
            type_shape(inner, imports, erase, out);
        }
        Type::Variadic(inner) => type_shape(inner, imports, erase, out),
        Type::Wild(opt) => {
            out.push("?".to_owned());
            if let Some((_, bound)) = opt {
                type_shape(bound, imports, erase, out);
            }
        }
        Type::Union(list) | Type::Inter(list) => {
            for t in list {
                type_shape(t, imports, erase, out);
            }
        }
        Type::Other => out.push("?".to_owned()),
    }
}

/// What contract the callable offers: its arity shape and the names in its types.
fn signature_facts(f: &Func, imports: &HashMap<String, Arc<str>>) -> Vec<String> {
    let mut out = vec![format!("arity={}", f.params.len())];
    if f.params.iter().any(|p| p.variadic) {
        out.push("variadic".to_owned());
    }
    for p in &f.params {
        let mut names = Vec::new();
        type_shape(&p.ty, imports, &f.type_params, &mut names);
        out.extend(names.into_iter().map(|n| format!("param:{n}")));
    }
    if let Some(ret) = &f.ret {
        let mut names = Vec::new();
        type_shape(ret, imports, &f.type_params, &mut names);
        out.extend(names.into_iter().map(|n| format!("ret:{n}")));
    }
    out
}

/// Whether an annotation is noise: `Override` rides on every override, `SuppressWarnings` is an IDE
/// directive. Both say nothing about what role the definition plays.
fn is_noise(name: &str) -> bool {
    matches!(name.rsplit('.').next().unwrap_or(name), "Override" | "SuppressWarnings")
}

/// Facts for a callable: everything the lenses can say about a method or constructor. `decorators`
/// are what is written on the definition — its annotations by name.
pub(crate) fn callable_facts(
    f: &Func,
    imports: &HashMap<String, Arc<str>>,
    file_names: &HashSet<String>,
    scope_canonical: String,
    decorators: &[String],
) -> LensFacts {
    let locals = crate::ir::locals_of(f);
    let mut walk = Walk {
        imports,
        file_names,
        locals: &locals,
        depth: 0,
        in_finally: false,
        facts: Walked::default(),
    };
    walk.stmts(&f.body);
    let walked = walk.facts;
    let mut facts = LensFacts::new();
    // A *set*: what it depends on, order irrelevant.
    facts.extend(Lens::Outgoing, walked.effects.iter().cloned().collect::<BTreeSet<_>>());
    // A *sequence*: the protocol it drives, order load-bearing.
    facts.extend(Lens::Effects, walked.effects);
    facts.extend(Lens::Control, walked.control);
    facts.extend(Lens::Failures, walked.failures);
    facts.extend(Lens::Resources, walked.resources);
    facts.extend(Lens::Signature, signature_facts(f, imports));
    let decorators: BTreeSet<String> = decorators.iter().filter(|d| !is_noise(d)).cloned().collect();
    facts.extend(Lens::Decorators, decorators);
    // One fact rather than one per statement: what it asserts is all-or-nothing, and as one rare
    // string it carries the IDF weight that claim deserves.
    facts.push(Lens::Scope, scope_canonical);
    facts
}

/// The shape a class declares: one fact per field — `name: type`, its `static` / `final` options —
/// and one per record component, sorted, because a declaration is a set.
pub(crate) fn class_facts(d: &ClassDecl, imports: &HashMap<String, Arc<str>>) -> LensFacts {
    let mut shapes = Vec::new();
    for m in &d.members {
        if let ClassMember::Field { mods, ty, name, .. } = m {
            let mut names = Vec::new();
            type_shape(ty, imports, &[], &mut names);
            let mut shape = format!("{name}: {}", names.join(" "));
            for opt in ["static", "final"] {
                if mods.iter().any(|m| m == opt) {
                    shape.push(' ');
                    shape.push_str(opt);
                }
            }
            shapes.push(shape);
        }
    }
    for p in &d.components {
        let mut names = Vec::new();
        type_shape(&p.ty, imports, &[], &mut names);
        shapes.push(format!("{}: {}", p.name, names.join(" ")));
    }
    shapes.sort();
    let mut facts = LensFacts::new();
    facts.extend(Lens::Schema, shapes);
    facts
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::Lens;
    use dup_defs_core::lens::stitch;
    use tree_sitter::Node;

    /// Depth-first search for the first node of a grammar kind.
    fn find_kind<'t>(n: Node<'t>, kind: &str) -> Option<Node<'t>> {
        if n.kind() == kind {
            return Some(n);
        }
        for k in crate::tree::kids(n) {
            if let Some(found) = find_kind(k, kind) {
                return Some(found);
            }
        }
        None
    }

    /// The lens's view of the first method in `src`, tags included — through `stitch` rather than
    /// `lens_def`, so a projection below the emission floor is still observable.
    fn project(src: &str, lens: Lens) -> Vec<String> {
        let tree = crate::tree::parse(src).expect("parse");
        let root = tree.root_node();
        let decl = find_kind(root, "method_declaration").expect("method_declaration");
        let name = crate::tree::field(decl, "name").map_or("", |n| crate::tree::text(n, src));
        let func = crate::ir::lower_func(decl, name, src).expect("lower_func");
        let facts = super::callable_facts(&func, &HashMap::new(), &HashSet::new(), String::new(), &[]);
        stitch(&[lens], &facts)
    }

    /// The schema lens's view of the first class-like declaration of `kind` in `src`.
    fn project_class(src: &str, kind: &str) -> Vec<String> {
        let tree = crate::tree::parse(src).expect("parse");
        let root = tree.root_node();
        let decl = find_kind(root, kind).expect(kind);
        let d = crate::ir::lower_class(decl, src).expect("lower_class");
        let facts = super::class_facts(&d, &HashMap::new());
        stitch(&[Lens::Schema], &facts)
    }

    #[test]
    fn outgoing_keeps_external_callees_and_drops_locals_this_and_own_classes() {
        let src = concat!(
            "class A {\n",
            "    void m(Runnable r, java.util.List<String> xs) {\n",
            "        xs.clear();\n",
            "        this.own();\n",
            "        r.run();\n",
            "        obj.visit();\n",
            "        B.helper();\n",
            "        new Socket();\n",
            "        helper();\n",
            "    }\n",
            "}\n",
        );
        let got = project(src, Lens::Outgoing);
        for want in ["outgoing:.visit", "outgoing:B.helper", "outgoing:Socket", "outgoing:helper"] {
            assert!(got.contains(&want.to_owned()), "missing {want:?} in {got:?}");
        }
        // `.clear` / `.run` go through a local, `.own` through `this` — none is a dependency.
        assert!(!got.iter().any(|f| f.contains("clear") || f.contains("run") || f.contains("own")), "{got:?}");
    }

    #[test]
    fn control_records_branching_with_nesting() {
        let src = concat!(
            "class A {\n",
            "    void m(int x) {\n",
            "        if (x > 0) { x++; } else { x--; }\n",
            "        while (x > 0) {\n",
            "            for (int i = 0; i < x; i++) { x--; }\n",
            "        }\n",
            "    }\n",
            "}\n",
        );
        assert_eq!(project(src, Lens::Control), vec!["control:if", "control:else", "control:while", "control:+for"]);
    }

    #[test]
    fn failures_read_throws_catches_and_library_spellings() {
        let src = concat!(
            "class A {\n",
            "    void m(int x) {\n",
            "        if (x < 0) { throw new IllegalArgumentException(\"neg\"); }\n",
            "        try { risky(); } catch (IOException e) { recover(); }\n",
            "        Objects.requireNonNull(x);\n",
            "        Preconditions.checkArgument(x > 0);\n",
            "        assert x > 0;\n",
            "    }\n",
            "}\n",
        );
        let got = project(src, Lens::Failures);
        for want in [
            "failures:IllegalArgumentException",
            "failures:catch IOException",
            "failures:Objects.requireNonNull",
            "failures:Preconditions.checkArgument",
            "failures:assert",
        ] {
            assert!(got.contains(&want.to_owned()), "missing {want:?} in {got:?}");
        }
    }

    #[test]
    fn resources_read_acquisitions_and_finally_releases() {
        let src = concat!(
            "class A {\n",
            "    void m(Path p) throws Exception {\n",
            "        try (InputStream in = Files.newInputStream(p); Socket s = new Socket()) {\n",
            "            read(in);\n",
            "        } finally {\n",
            "            in.close();\n",
            "        }\n",
            "    }\n",
            "}\n",
        );
        assert_eq!(
            project(src, Lens::Resources),
            vec!["resources:Files.newInputStream", "resources:Socket", "resources:close"]
        );
    }

    #[test]
    fn signature_erases_the_definitions_own_type_parameters() {
        let src = "class A {\n    <T> T first(T a, T b) { return a; }\n}\n";
        assert_eq!(
            project(src, Lens::Signature),
            vec!["signature:arity=2", "signature:param:_", "signature:param:_", "signature:ret:_"]
        );
    }

    #[test]
    fn signature_resolves_imported_types_to_dotted_paths() {
        let src = concat!("import java.util.List;\n", "class A {\n", "    List<String> get(String s) { return null; }\n", "}\n");
        let tree = crate::tree::parse(src).expect("parse");
        let root = tree.root_node();
        let decl = find_kind(root, "method_declaration").expect("method_declaration");
        let name = crate::tree::field(decl, "name").map_or("", |n| crate::tree::text(n, src));
        let func = crate::ir::lower_func(decl, name, src).expect("lower_func");
        let mut imports = HashMap::new();
        imports.insert("List".to_owned(), std::sync::Arc::from("java.util.List"));
        let facts = super::callable_facts(&func, &imports, &HashSet::new(), String::new(), &[]);
        let got = stitch(&[Lens::Signature], &facts);
        assert_eq!(
            got,
            vec!["signature:arity=1", "signature:param:String", "signature:ret:java.util.List", "signature:ret:String"]
        );
    }

    #[test]
    fn decorators_drop_override_and_suppress_warnings() {
        let src = "class A {\n    @Override @Deprecated @SuppressWarnings(\"unchecked\") void m() {}\n}\n";
        let tree = crate::tree::parse(src).expect("parse");
        let root = tree.root_node();
        let decl = find_kind(root, "method_declaration").expect("method_declaration");
        let name = crate::tree::field(decl, "name").map_or("", |n| crate::tree::text(n, src));
        let func = crate::ir::lower_func(decl, name, src).expect("lower_func");
        let decorators = ["Override".to_owned(), "Deprecated".to_owned(), "SuppressWarnings".to_owned()];
        let facts = super::callable_facts(&func, &HashMap::new(), &HashSet::new(), String::new(), &decorators);
        assert_eq!(stitch(&[Lens::Decorators], &facts), vec!["decorators:Deprecated"]);
    }

    #[test]
    fn schema_declares_fields_as_a_set_with_modifier_options() {
        let src = concat!(
            "class A {\n",
            "    static final int MAX = 10;\n",
            "    String name;\n",
            "    java.util.List<String> items;\n",
            "}\n",
        );
        assert_eq!(
            project_class(src, "class_declaration"),
            vec!["schema:MAX: int static final", "schema:items: java.util.List String", "schema:name: String"]
        );
    }

    #[test]
    fn schema_includes_record_components() {
        let src = "record P(int x, String name) {}\n";
        assert_eq!(project_class(src, "record_declaration"), vec!["schema:name: String", "schema:x: int"]);
    }
}
