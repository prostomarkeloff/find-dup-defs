//! File-scope definition scan for Java — the "find-*" step of `find-dup-defs`.
//!
//! Parses each file **once** with tree-sitter and lowers every tracked declaration straight to the
//! engine's [`Def`], computing its canonical strings off the syntax tree (no re-parse). Surfaces:
//!
//! * **methods** — every method and constructor with a body, named `Class.method` (a constructor
//!   takes its class's name, so `Foo(int x)` is `Foo`, not `Foo.Foo`). `this` reads as `self`.
//! * **classes** — `class`, `enum` and `record` declarations, named by their package-qualified name
//!   (`com.example.Foo`), nested classes as `Outer.Inner`.
//! * **interfaces** — `interface` and `@interface` (annotation type) declarations.
//! * **constants** — a `static final` field, or a `constant_declaration` (an interface constant),
//!   named `Class.NAME`, compared by the text of its value expression.
//!
//! Only class-body members at file scope: a class or method declared inside a method body is part
//! of that body, the "top-level only" rule every frontend follows. Enum constants are instances,
//! not constants, and are left out; a plain field is not a definition either.
//!
//! A file that declares itself generated (a comment saying `DO NOT EDIT` before its first
//! declaration — the `package` clause, an `import`, or a class-like declaration) yields nothing:
//! its duplication is the generator's, and editing it is not how it would be fixed.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dup_defs_core::lens::{enabled_lenses, lens_def, DefSite, Lens, LensFacts};
use dup_defs_core::{count_loc, Analysis, AnalyzedFn, CanonDialect, Def, Facets, KindSpec, LineMap, ScanOpts};
use tree_sitter::Node;

use crate::canon::{analyze, class_canon, file_imports, reached_paths, scope_canonical};
use crate::decides::{decides, FileScope};
use crate::frontend::{CLASSES, CONSTANTS, INTERFACES, METHODS};
use crate::ir::{annotations_of, lower_class, lower_func, modifiers_of, ClassKind, Expr, Func, Stmt};
use crate::tree::{field, fields, kids, parse, text};

/// Whether a file declares itself generated: a comment saying `DO NOT EDIT` before its first
/// declaration — the `package` clause, an `import`, or a class-like declaration.
///
/// The boundary is the first line that starts a top-level declaration: `package`, `import`, or a
/// class-like keyword, possibly after a modifier (`public class Foo`). Scanning stops there so a
/// `DO NOT EDIT` written later in the file (a warning above a hand-edited method, say) does not
/// discard a file someone still edits.
pub(crate) fn is_generated(src: &str) -> bool {
    for line in src.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if starts_top_decl(t) {
            return false;
        }
        if is_comment_line(t) && t.to_ascii_uppercase().contains("DO NOT EDIT") {
            return true;
        }
    }
    false
}

/// Whether a line is a comment — a line comment, or a block comment's opening or continuation.
fn is_comment_line(t: &str) -> bool {
    t.starts_with("//") || t.starts_with("/*") || t.starts_with('*')
}

/// Whether a line starts a top-level declaration. A class-like declaration may be preceded by a
/// modifier (`public`, `abstract`, `final`, …) or an annotation.
fn starts_top_decl(t: &str) -> bool {
    t.starts_with("package ")
        || t.starts_with("import ")
        || t.starts_with("class ")
        || t.starts_with("interface ")
        || t.starts_with("enum ")
        || t.starts_with("record ")
        || t.starts_with("@interface")
        || t.starts_with("public ")
        || t.starts_with("protected ")
        || t.starts_with("private ")
        || t.starts_with("final ")
        || t.starts_with("abstract ")
        || t.starts_with("sealed ")
        || t.starts_with("non-sealed ")
        || t.starts_with("strictfp ")
        || t.starts_with("static ")
}

/// Whether a grammar kind is a class-like declaration at file scope.
fn is_class_like(kind: &str) -> bool {
    matches!(
        kind,
        "class_declaration"
            | "interface_declaration"
            | "enum_declaration"
            | "record_declaration"
            | "annotation_type_declaration"
    )
}

/// The file's package name — the text of the `package_declaration`'s `scoped_identifier`, or `""`
/// for the default package. The grammar stores the name as a direct child, not a field.
fn package_name(root: Node<'_>, src: &str) -> String {
    kids(root)
        .into_iter()
        .find(|k| k.kind() == "package_declaration")
        .and_then(|p| kids(p).into_iter().next())
        .map_or_else(String::new, |n| dotted(n, src))
}

/// The dotted spelling of a `scoped_identifier` chain (`a.b.C`), read scope first.
fn dotted(n: Node<'_>, src: &str) -> String {
    let mut parts = Vec::new();
    let mut cur = Some(n);
    while let Some(node) = cur {
        if node.kind() == "scoped_identifier" {
            parts.push(field(node, "name").map_or_else(|| "", |name| text(name, src)).to_owned());
            cur = field(node, "scope");
        } else {
            parts.push(text(node, src).to_owned());
            cur = None;
        }
    }
    parts.reverse();
    parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join(".")
}

/// Join a dotted name, dropping empty parts: the package, an enclosing-class chain and a simple
/// name in any combination.
fn join(parts: &[&str]) -> String {
    parts.iter().copied().filter(|p| !p.is_empty()).collect::<Vec<_>>().join(".")
}

// ───────────────────────────── trivial-body filter ─────────────────────────────

/// True when a body is a no-op or a stub: empty; a single `return` of a name or literal only; a
/// single `return;`; a single `throw new UnsupportedOperationException()` (the IDE's override
/// stub); a getter (`return this.x;`); a setter (`this.x = v;`); or a constructor whose every
/// statement assigns a parameter to a field (`this.x = x;`). None of them is a refactor cluster —
/// the accessors and the assigned-every-field constructor are the boilerplate a Java class is
/// mostly made of, and fifty `getId`s are fifty getters, not one duplication. Anything structural
/// falls through and is compared.
fn is_trivial(f: &Func) -> bool {
    if f.body.is_empty() {
        return true;
    }
    // A constructor carries no return type (`ret` `None`), which is how a constructor is told from
    // a `void` method here.
    if f.ret.is_none() && f.body.iter().all(|s| assigns_param_to_field(f, s)) {
        return true;
    }
    match f.body.as_slice() {
        [Stmt::Return(None)] => true,
        [Stmt::Return(Some(e))] => is_atom(e) || is_getter(f, e),
        [Stmt::Throw(Expr::New { class, .. })] => class.rsplit('.').next().unwrap_or(class) == "UnsupportedOperationException",
        [Stmt::Assign { op, lhs, rhs }] => is_setter(f, op, lhs, rhs),
        [Stmt::Expr(Expr::AssignExpr { op, l, r })] => is_setter(f, op, l, r),
        _ => false,
    }
}

fn is_atom(e: &Expr) -> bool {
    matches!(e, Expr::Name(_) | Expr::Lit(..) | Expr::Bool(_) | Expr::Null)
}

/// Whether an expression reads `this`.
fn is_this(e: &Expr) -> bool {
    matches!(e, Expr::Name(n) if n == "this")
}

/// The field an expression reads off `this`, if it is one: `this.x` → `Some("x")`.
fn this_field(e: &Expr) -> Option<&str> {
    match e {
        Expr::Field { base, name } if is_this(base) => Some(name.as_str()),
        _ => None,
    }
}

/// Whether a name is one of the definition's parameters.
fn is_param(f: &Func, name: &str) -> bool {
    f.params.iter().any(|p| p.name == name)
}

/// The accessor suffix of a method name: `getX` / `isX` / `setX` → `Some("X")`. The getter and
/// setter rules ignore whether the accessor is spelled `get`, `is` or `set`, and match the field
/// case-insensitively.
fn accessor_suffix(name: &str) -> Option<&str> {
    ["get", "set", "is"].iter().find_map(|p| name.strip_prefix(p))
}

/// A getter: zero parameters, a single `return this.x;`, and the field `x` matching the method
/// name's suffix (`getX` / `isX`).
fn is_getter(f: &Func, e: &Expr) -> bool {
    f.params.is_empty()
        && this_field(e).is_some_and(|field| accessor_suffix(&f.name).is_some_and(|s| field.eq_ignore_ascii_case(s)))
}

/// A setter: one parameter, a single `this.x = v;` where `v` is that parameter, and the field
/// matching the method name's suffix (`setX`).
fn is_setter(f: &Func, op: &str, lhs: &Expr, rhs: &Expr) -> bool {
    if op != "=" || f.params.len() != 1 {
        return false;
    }
    let Expr::Name(p) = rhs else { return false };
    if !is_param(f, p) {
        return false;
    }
    this_field(lhs).is_some_and(|field| accessor_suffix(&f.name).is_some_and(|s| field.eq_ignore_ascii_case(s)))
}

/// A constructor's building block: `this.x = x;`, assigning a parameter to a field.
fn assigns_param_to_field(f: &Func, s: &Stmt) -> bool {
    let (op, lhs, rhs) = match s {
        Stmt::Assign { op, lhs, rhs } => (op.as_str(), lhs, rhs),
        Stmt::Expr(Expr::AssignExpr { op, l, r }) => (op.as_str(), l.as_ref(), r.as_ref()),
        _ => return false,
    };
    op == "=" && this_field(lhs).is_some() && matches!(rhs, Expr::Name(p) if is_param(f, p))
}

// ───────────────────────────── file-bound names ─────────────────────────────

/// The class names a file declares — every top-level and nested class's simple name plus its
/// qualified chain (`Outer`, `Inner`, `Outer.Inner`) — what every lens erases before it looks.
fn file_bound_names(root: Node<'_>, src: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    for item in kids(root) {
        if is_class_like(item.kind()) {
            collect_class_names(item, src, "", &mut out);
        }
    }
    out
}

fn collect_class_names(node: Node<'_>, src: &str, chain: &str, out: &mut HashSet<String>) {
    let Some(decl) = lower_class(node, src) else { return };
    let qualified = join(&[chain, decl.name.as_str()]);
    out.insert(decl.name.clone());
    out.insert(qualified.clone());
    if let Some(body) = field(node, "body") {
        for m in kids(body) {
            if is_class_like(m.kind()) {
                collect_class_names(m, src, &qualified, out);
            }
        }
    }
}

// ───────────────────────────── the scan ─────────────────────────────

/// What a definition contributes beyond its location: its canonical forms and, for a callable, the
/// facets the perspective passes read.
struct DefCanon {
    cluster: Option<String>,
    analysis: Option<Analysis>,
    facets: Facets,
}

/// One builder so every push site stays uniform.
struct Builder<'a> {
    source: &'a str,
    lines: &'a LineMap<'a>,
    file: &'a Arc<str>,
    package: String,
    imports: HashMap<String, Arc<str>>,
    /// Which lenses this run asked for; empty when it did not ask.
    lenses: Vec<Lens>,
    /// Names the file itself introduced — what every lens erases before it looks.
    file_names: HashSet<String>,
    /// What the file's names stand for, as the decisions pass writes them.
    scope: FileScope,
    /// Whether the run's passes read per-definition facets (converge/decisions/lenses) — gates the
    /// statement stream, decision facts, reached paths and identity when false.
    facets: bool,
    out: Vec<Def>,
}

impl Builder<'_> {
    /// One lens [`Def`] for this definition, when the run asked for lenses and the projection has
    /// enough to say. The record, the thinness floor and the scoring are the shared module's.
    fn push_lens(&mut self, name: &str, span: Node<'_>, facts: &LensFacts) {
        if self.lenses.is_empty() {
            return;
        }
        let start = span.start_byte();
        let (line, col) = self.lines.loc0(start);
        let loc = count_loc(self.source.get(start..span.end_byte()).unwrap_or(""));
        let site = DefSite { name, file: self.file, line, col, loc };
        if let Some(def) = lens_def("java", CanonDialect::Java, &site, &self.lenses, facts) {
            self.out.push(def);
        }
    }

    fn push(&mut self, kind: &'static KindSpec, name: String, span: Node<'_>, args: usize, canon: DefCanon) {
        let DefCanon { cluster, analysis, facets } = canon;
        let text_orig = text(span, self.source).to_owned();
        self.emit(kind, name, span, args, text_orig, cluster, analysis, facets);
    }

    #[allow(clippy::too_many_arguments)] // the parts of one def, each read once
    fn emit(
        &mut self,
        kind: &'static KindSpec,
        name: String,
        span: Node<'_>,
        args: usize,
        text_orig: String,
        cluster: Option<String>,
        analysis: Option<Analysis>,
        facets: Facets,
    ) {
        let start = span.start_byte();
        let (line, col) = self.lines.loc0(start);
        let loc = count_loc(&text_orig);
        self.out.push(Def {
            lang: "java",
            kind,
            name,
            file: Arc::clone(self.file),
            line,
            col,
            loc,
            args,
            text_orig,
            cluster_canonical: cluster,
            analysis,
            thickness: None,
            facets,
        });
    }

    /// A constant: a raw-text kind, compared by the value expression's text rather than the whole
    /// declaration (two `static final int X = 5;` and `Y = 5;` share the `5`, not the name).
    fn push_constant(&mut self, name: String, declarator: Node<'_>, field_node: Node<'_>, value: Option<Node<'_>>) {
        let text_orig = match value {
            Some(v) => text(v, self.source).to_owned(),
            None => text(field_node, self.source).to_owned(),
        };
        let identity = if self.facets { Some(self.scope.identity(&name)) } else { None };
        let facets = Facets { identity, ..Default::default() };
        self.emit(&CONSTANTS, name, declarator, 0, text_orig, None, None, facets);
    }

    /// A class-like declaration: the declaration itself, then its methods, constants and nested
    /// classes. `chain` is the enclosing classes' dotted name (`Outer.Inner`), empty at the top.
    fn walk_class(&mut self, node: Node<'_>, chain: &str) {
        let Some(decl) = lower_class(node, self.source) else { return };
        // The class's name without the package (methods and constants hang off it), and with it.
        let qclass = join(&[chain, decl.name.as_str()]);
        let qualified = join(&[self.package.as_str(), chain, decl.name.as_str()]);

        if !self.lenses.is_empty() {
            let facts = crate::lenses::class_facts(&decl, &self.imports);
            self.push_lens(&qualified, node, &facts);
        }

        let kind = match decl.kind {
            ClassKind::Interface | ClassKind::Annotation => &INTERFACES,
            _ => &CLASSES,
        };
        // `FileScope::identity` prefixes the package itself, so it takes the package-less qualified
        // name (`Outer.Inner`) — the class's report name above is package-qualified, this is not.
        let identity = if self.facets { Some(self.scope.identity(&qclass)) } else { None };
        let facets = Facets { identity, ..Default::default() };
        let canon = DefCanon { cluster: Some(class_canon(&self.imports, &decl)), analysis: None, facets };
        self.push(kind, qualified, node, 0, canon);

        if let Some(body) = field(node, "body") {
            for m in kids(body) {
                self.walk_member(m, &qclass);
            }
        }
    }

    fn walk_member(&mut self, m: Node<'_>, qclass: &str) {
        match m.kind() {
            "method_declaration" | "constructor_declaration" | "compact_constructor_declaration" => {
                self.walk_method(m, qclass);
            }
            "field_declaration" | "constant_declaration" => self.walk_field(m, qclass),
            kind if is_class_like(kind) => self.walk_class(m, qclass),
            _ => {} // enum constants, static initializers, annotation elements — not definitions
        }
    }

    /// A method or constructor with a body: canonical forms, statement stream, what it reaches, and
    /// what it decides on. `qclass` is the owning class's name without the package.
    fn walk_method(&mut self, m: Node<'_>, qclass: &str) {
        let simple = field(m, "name").map_or_else(String::new, |n| text(n, self.source).to_owned());
        // `lower_func` yields `None` for an abstract / native method: a signature, not a body.
        let Some(func) = lower_func(m, &simple, self.source) else { return };
        if is_trivial(&func) {
            return;
        }
        let is_ctor = m.kind() == "constructor_declaration" || m.kind() == "compact_constructor_declaration";
        // A constructor takes its class's name; a method is `Class.method`.
        let name = if is_ctor { qclass.to_owned() } else { join(&[qclass, func.name.as_str()]) };

        let mods = modifiers_of(m);
        // The runtime's entry point: `public static void main(String[] args)`.
        let open = mods.iter().any(|x| x == "static") && func.name == "main" && func.params.len() == 1;

        if !self.lenses.is_empty() {
            let decorators = annotations_of(m, self.source);
            let scope = scope_canonical(&self.imports, &func, &self.file_names);
            let facts = crate::lenses::callable_facts(&func, &self.imports, &self.file_names, scope, &decorators);
            self.push_lens(&name, m, &facts);
        }

        let read = if self.facets { decides(&func, &self.scope, open) } else { dup_defs_core::decide::DecisionFacts::default() };
        let AnalyzedFn { cluster_canonical, xname_canonical, type3_lines, type_refs, type3_units, statements, size } =
            analyze(&self.imports, &name, &func, self.facets);
        let canon = DefCanon {
            cluster: Some(cluster_canonical),
            analysis: Some(Analysis {
                xname_canonical,
                type3_lines,
                type_refs,
                type3_units,
                size,
                canon_dialect: CanonDialect::Java,
            }),
            facets: Facets {
                statements,
                reaches: if self.facets { reached_paths(&func, &self.imports) } else { Vec::new() },
                decides: read.atoms,
                identity: if self.facets { Some(self.scope.identity(&name)) } else { None },
                params: read.params,
                calls: read.calls,
                mentions: read.mentions,
                loose_keywords: read.loose,
            },
        };
        self.push(&METHODS, name, m, func.params.len(), canon);
    }

    /// A `static final` field or an interface `constant_declaration` — one constant per declarator.
    fn walk_field(&mut self, m: Node<'_>, qclass: &str) {
        let declared_const = m.kind() == "constant_declaration";
        let mods = modifiers_of(m);
        let is_constant = declared_const || (mods.iter().any(|x| x == "static") && mods.iter().any(|x| x == "final"));
        if !is_constant {
            return;
        }
        for declarator in fields(m, "declarator") {
            let Some(field_name) = field(declarator, "name") else { continue };
            let fname = text(field_name, self.source);
            let name = join(&[qclass, fname]);
            self.push_constant(name, declarator, m, field(declarator, "value"));
        }
    }
}

/// Scan one Java source string → its definitions as [`Def`]s with canon precomputed. A declaration
/// holding a syntax error is skipped; the rest of the file is still read.
#[must_use]
pub fn scan_source(source: &str, file: &Arc<str>, opts: &ScanOpts) -> Vec<Def> {
    if is_generated(source) {
        return Vec::new();
    }
    let Some(tree) = parse(source) else { return Vec::new() };
    let root = tree.root_node();
    let lines = LineMap::new(source);
    let lenses = enabled_lenses(opts);
    let facets = opts.facets;
    let file_names = if lenses.is_empty() { HashSet::new() } else { file_bound_names(root, source) };
    // The import table feeds the canonicals (class names resolve through it) as well as the facets,
    // so it is always read; only the facets that hang off it are gated.
    let imports = file_imports(root, source);
    let package = package_name(root, source);
    let scope = FileScope::new(&package, &imports);
    let mut b = Builder { source, lines: &lines, file, package, imports, lenses, file_names, scope, facets, out: Vec::new() };
    for item in kids(root) {
        if item.has_error() {
            continue;
        }
        if is_class_like(item.kind()) {
            b.walk_class(item, "");
        }
    }
    b.out
}

#[cfg(test)]
mod tests {
    use super::{is_generated, scan_source};
    use dup_defs_core::{Def, ScanOpts};
    use std::sync::Arc;

    fn scan(src: &str) -> Vec<Def> {
        let f: Arc<str> = Arc::from("t.java");
        scan_source(src, &f, &ScanOpts::default())
    }

    fn defs(src: &str) -> Vec<(String, String)> {
        scan(src).into_iter().map(|d| (d.kind.id.to_owned(), d.name)).collect()
    }

    fn names_of_kind(src: &str, kind: &str) -> Vec<String> {
        defs(src).into_iter().filter(|(k, _)| k == kind).map(|(_, n)| n).collect()
    }

    #[test]
    fn surfaces_each_kind() {
        let src = r"package com.example;

import java.util.List;

class Calculator {
    int add(int a, int b) {
        return a + b;
    }

    static final int MAX = 100;

    int total;
}

interface Shape {
    double area();
}

enum Color {
    RED, GREEN;
}

record Point(int x, int y) {
    int sum() {
        return x + y;
    }
}
";
        assert_eq!(names_of_kind(src, "classes"), ["com.example.Calculator", "com.example.Color", "com.example.Point"]);
        assert_eq!(names_of_kind(src, "interfaces"), ["com.example.Shape"]);
        assert_eq!(names_of_kind(src, "methods"), ["Calculator.add", "Point.sum"]);
        assert_eq!(names_of_kind(src, "constants"), ["Calculator.MAX"]);
    }

    #[test]
    fn nested_classes_take_a_qualified_name() {
        let src = r"class Outer {
    class Inner {
        int run(int x) {
            return x + 1;
        }
    }
}
";
        assert_eq!(names_of_kind(src, "classes"), ["Outer", "Outer.Inner"]);
        assert_eq!(names_of_kind(src, "methods"), ["Outer.Inner.run"]);
    }

    #[test]
    fn methods_carry_an_analysis_and_a_canonical() {
        let src = r"class A {
    int f(int x) {
        return x + 1;
    }
}
";
        let defs = scan(src);
        let method = defs.iter().find(|d| d.name == "A.f").expect("the method");
        assert!(method.cluster_canonical.as_deref().is_some_and(|c| !c.is_empty()));
        let analysis = method.analysis.as_ref().expect("analysis");
        assert_eq!(analysis.canon_dialect, dup_defs_core::CanonDialect::Java);
        assert_ne!(analysis.xname_canonical, "");
    }

    #[test]
    fn trivial_bodies_are_skipped() {
        let src = r"class Getters {
    private int value;

    int getValue() {
        return this.value;
    }

    void setValue(int v) {
        this.value = v;
    }

    Getters(int v) {
        this.value = v;
    }

    int compute(int x) {
        throw new UnsupportedOperationException();
    }

    int real(int x) {
        return this.value + x;
    }
}
";
        assert_eq!(names_of_kind(src, "methods"), ["Getters.real"]);
    }

    #[test]
    fn abstract_and_native_methods_are_not_definitions() {
        let src = r"abstract class Base {
    abstract int go(int x);

    native int nat(int x);

    int body(int x) {
        return x;
    }
}
";
        // `body` returns a bare parameter — an atom — so it is trivial too; nothing survives.
        assert_eq!(names_of_kind(src, "methods"), [] as [String; 0]);
    }

    #[test]
    fn constructor_takes_the_class_name() {
        let src = r"class Foo {
    int x;
    Foo(int x) {
        this.x = x + 1;
    }
}
";
        // The constructor assigns `x + 1`, not the parameter bare — structural, so it survives.
        assert_eq!(names_of_kind(src, "methods"), ["Foo"]);
    }

    #[test]
    fn generated_files_yield_nothing() {
        let src = r"// Code generated by protoc. DO NOT EDIT.
// source: foo.proto

package com.example;

class Gen {
    int f(int x) {
        return x + 1;
    }
}
";
        assert!(scan(src).is_empty());
        assert!(is_generated(src));
    }

    #[test]
    fn a_late_do_not_edit_does_not_discard_the_file() {
        let src = r"package com.example;

class Foo {
    // DO NOT EDIT this method
    int x(int v) {
        return v + 1;
    }
}
";
        assert!(!is_generated(src));
        assert_eq!(names_of_kind(src, "classes"), ["com.example.Foo"]);
    }
}
