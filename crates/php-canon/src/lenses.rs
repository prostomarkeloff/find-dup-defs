//! **Lenses for PHP** — the walk, and only the walk.
//!
//! The vocabulary, the stitching, the corpus scoring and the record itself live in
//! [`dup_defs_core::lens`]; the `use` lens's sites are [`crate::canon::collect_sites`]. What is here is
//! how PHP answers the other questions, read off the owned tree ([`crate::ir`]).
//!
//! Three have a PHP answer worth spelling out:
//!
//! * **`failures`** — PHP fails two ways, and a body may use both. It throws: the classes it throws and
//!   catches, as Python's are read, and a rethrow of what it caught is `?` — the failure handed on, the
//!   fact Rust's `?` records. And it returns a failure value, the convention older PHP and every
//!   plugin API built on it keeps: `false`, `null`, an error object (`new WP_Error(…)`), recorded as what it is;
//!   `exit` / `die` / `wp_die` stop the request outright.
//! * **`resources`** — what a body holds open is what it releases: PHP closes a handle with a call
//!   (`fclose`, `curl_close`, `ob_end_clean`), so the fact is what was *acquired* — `$f = fopen(…); …
//!   fclose($f)` holds `fopen` — and, when the released value was not bound here
//!   (`$this->lock->release()`), the release itself.
//! * **`decorators`** — the modifiers written on the definition (`public`, `static`, `final`) and its
//!   attributes (`#[Route('/x')]`): what role it plays, as far as PHP says.

use std::collections::{BTreeSet, HashMap, HashSet};

use dup_defs_core::lens::{control_tag, Lens, LensFacts};

use crate::canon::{each_stmt_part, for_each_child, type_names, Child, Part};
use crate::ir::{ClassDecl, ClassMember, Else, Expr, Func, Jump, Member, Stmt, Type};

/// Facts gathered in one walk of a body, sliced up per lens afterwards.
#[derive(Default)]
struct Walked {
    /// Every external callee, in call order — the `effects` lens; deduped, it is `outgoing`.
    effects: Vec<String>,
    /// Control-flow tags carrying their nesting.
    control: Vec<String>,
    /// What the body fails with.
    failures: BTreeSet<String>,
    /// What the body holds open for the length of the function.
    resources: Vec<String>,
}

struct Walk<'a> {
    /// Names the file itself introduced — erased, exactly as on the widest rung of the ladder, so a lens
    /// is not held apart by the very identities it exists to see past.
    file_names: &'a HashSet<String>,
    /// What each variable was bound to by a call — `$f = fopen(…)` binds `f` to `fopen` — so a later
    /// `fclose($f)` can say what was held.
    bound_by: HashMap<String, String>,
    /// The variables a `catch` bound: throwing one is handing the failure on.
    caught: HashSet<String>,
    depth: usize,
    facts: Walked,
}

/// Whether a callee releases what its first argument (or its receiver) holds.
fn releases(callee: &str) -> bool {
    let lower = callee.to_ascii_lowercase();
    ["close", "free", "release", "unlock", "destroy", "disconnect", "end_clean", "end_flush", "get_clean", "rollback", "commit", "restore"]
        .iter()
        .any(|word| lower.contains(word))
}

/// The functions that stop the request, or that a body calls to fail outright.
fn aborts(callee: &str) -> bool {
    matches!(callee.to_ascii_lowercase().as_str(), "exit" | "wp_die" | "trigger_error" | "abort" | "dd" | "user_error")
}

/// The last segment of a name: what a class or function is called wherever it is declared.
fn bare(name: &str) -> &str {
    name.rsplit('\\').next().unwrap_or(name)
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

    /// The name a call names, when it names something the file did not introduce: a function `f`, a
    /// method `.m` — the receiver is whatever it is, the method is the grammar — a static one `C::m`.
    fn callee(&self, e: &Expr) -> Option<String> {
        match e {
            Expr::Call { f, .. } => match f.as_ref() {
                Expr::Name(n) => (!self.file_names.contains(bare(n))).then(|| bare(n).to_owned()),
                _ => None,
            },
            Expr::Method { name: Member::Name(m), .. } => (!self.file_names.contains(m)).then(|| format!(".{m}")),
            Expr::StaticCall { scope, name: Member::Name(m), .. } => {
                let class = match scope.as_ref() {
                    Expr::Name(n) if matches!(n.as_str(), "self" | "static" | "parent") => n.clone(),
                    Expr::Name(n) if self.file_names.contains(bare(n)) => "_".to_owned(),
                    Expr::Name(n) => bare(n).to_owned(),
                    _ => "_".to_owned(),
                };
                Some(format!("{class}::{m}"))
            }
            Expr::New { class, .. } => match class.as_ref() {
                Expr::Name(n) => (!self.file_names.contains(bare(n))).then(|| format!("new {}", bare(n))),
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

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::If { cond, then, els } => {
                self.expr(cond);
                self.tag("if");
                self.nested(|w| {
                    w.stmts(then);
                    match els.as_deref() {
                        Some(Else::Block(b)) => w.stmts(b),
                        Some(Else::If(alt)) => w.stmt(alt),
                        None => {}
                    }
                });
            }
            Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                self.tag(if matches!(s, Stmt::DoWhile { .. }) { "do" } else { "while" });
                self.expr(cond);
                self.nested(|w| w.stmts(body));
            }
            Stmt::For { init, cond, update, body } => {
                self.tag(if cond.is_empty() { "loop" } else { "for" });
                for e in init.iter().chain(cond).chain(update) {
                    self.expr(e);
                }
                self.nested(|w| w.stmts(body));
            }
            Stmt::Foreach { subject, body, .. } => {
                self.tag("for");
                self.expr(subject);
                self.nested(|w| w.stmts(body));
            }
            Stmt::Switch { value, cases } => {
                self.expr(value);
                self.tag("switch");
                self.nested(|w| {
                    for case in cases {
                        if let Some(v) = &case.value {
                            w.expr(v);
                        }
                        w.stmts(&case.body);
                    }
                });
            }
            Stmt::Try { body, catches, finally } => {
                self.tag("try");
                self.nested(|w| {
                    w.stmts(body);
                    for c in catches {
                        w.facts.failures.extend(c.types.iter().map(|t| format!("catch {}", bare(t))));
                        if let Some(v) = &c.var {
                            w.caught.insert(v.clone());
                        }
                        w.stmts(&c.body);
                    }
                    if let Some(fin) = finally {
                        w.tag("finally");
                        w.stmts(fin);
                    }
                });
            }
            Stmt::Return(value) => {
                self.tag("return");
                if let Some(v) = value {
                    self.failure_value(v);
                    self.expr(v);
                }
            }
            Stmt::Throw(e) => {
                self.tag("throw");
                self.thrown(e);
                self.expr(e);
            }
            Stmt::Jump { kind, .. } => self.tag(match kind {
                Jump::Break => "break",
                Jump::Continue => "continue",
                Jump::Goto => "goto",
            }),
            Stmt::Assign { lhs, rhs, .. } => {
                if let Expr::Var(name) = lhs {
                    self.note_binding(name, rhs);
                }
                self.expr(lhs);
                self.expr(rhs);
            }
            _ => each_stmt_part(s, &mut |part| match part {
                Part::Stmt(inner) => self.stmt(inner),
                Part::Stmts(list) => self.stmts(list),
                Part::Expr(e) => self.expr(e),
            }),
        }
    }

    fn expr(&mut self, e: &Expr) {
        if let Some(callee) = self.callee(e) {
            if aborts(&callee) {
                self.facts.failures.insert(callee.clone());
            }
            if releases(&callee) {
                let held = match e {
                    Expr::Call { args, .. } => args.first().map(|a| &a.value),
                    Expr::Method { recv, .. } => Some(recv.as_ref()),
                    _ => None,
                };
                self.release(&callee, held);
            }
            self.facts.effects.push(callee);
        }
        if let Expr::Throw(inner) = e {
            self.thrown(inner);
        }
        for_each_child(e, &mut |c| match c {
            Child::Expr(x) => self.expr(x),
            Child::Body(b) => self.stmts(b),
        });
    }

    /// What a `throw` throws: the class it builds, or `?` for a failure it was handed.
    fn thrown(&mut self, e: &Expr) {
        match e {
            Expr::New { class, .. } => {
                if let Expr::Name(n) = class.as_ref() {
                    self.facts.failures.insert(bare(n).to_owned());
                }
            }
            Expr::Var(v) if self.caught.contains(v) => {
                self.facts.failures.insert("?".to_owned());
            }
            _ => {}
        }
    }

    /// What a returned value fails with, when it is a failure value: `false`, `null`, an error object.
    fn failure_value(&mut self, e: &Expr) {
        match e {
            Expr::Bool(false) => {
                self.facts.failures.insert("false".to_owned());
            }
            Expr::Null => {
                self.facts.failures.insert("null".to_owned());
            }
            Expr::New { class, .. } => {
                if let Expr::Name(n) = class.as_ref() {
                    let lower = n.to_ascii_lowercase();
                    if lower.contains("error") || lower.contains("exception") {
                        self.facts.failures.insert(bare(n).to_owned());
                    }
                }
            }
            Expr::Paren(inner) => self.failure_value(inner),
            _ => {}
        }
    }

    /// `$x = f(…)` — remember that `$x` came from `f`.
    fn note_binding(&mut self, name: &str, value: &Expr) {
        let mut value = value;
        while let Expr::Paren(e) | Expr::Silence(e) | Expr::Cast { e, .. } = value {
            value = e;
        }
        if let Some(callee) = self.callee(value) {
            self.bound_by.insert(name.to_owned(), callee);
        }
    }

    /// What a release releases: the call that acquired the value it releases, or the release itself.
    fn release(&mut self, callee: &str, held: Option<&Expr>) {
        let acquired = match held {
            Some(Expr::Var(n)) => self.bound_by.get(n).cloned(),
            _ => None,
        };
        self.facts.resources.push(acquired.unwrap_or_else(|| callee.to_owned()));
    }
}

/// The names a type mentions that the file did not introduce, so `?Order` and `?Invoice` agree on being
/// nullable and say nothing else.
fn type_shape(t: Option<&Type>, file_names: &HashSet<String>, out: &mut Vec<String>) {
    match t {
        None => out.push("?".to_owned()),
        Some(Type::Named(n)) => {
            let mut named = Vec::new();
            type_names(&Type::Named(n.clone()), &mut |c| named.push(c.to_owned()));
            out.push(match named.first() {
                Some(class) if file_names.contains(bare(class)) => "_".to_owned(),
                Some(class) => bare(class).to_owned(),
                None => n.to_ascii_lowercase(),
            });
        }
        Some(Type::Nullable(inner)) => {
            out.push("?".to_owned());
            type_shape(Some(inner), file_names, out);
        }
        Some(Type::Union(list)) => {
            out.push("|".to_owned());
            for t in list {
                type_shape(Some(t), file_names, out);
            }
        }
        Some(Type::Inter(list)) => {
            out.push("&".to_owned());
            for t in list {
                type_shape(Some(t), file_names, out);
            }
        }
    }
}

/// What contract the function offers: its arity shape and the names in its types.
fn signature_facts(f: &Func, method: bool, file_names: &HashSet<String>) -> Vec<String> {
    let mut out = vec![format!("arity={}", f.params.len())];
    if method {
        out.push("method".to_owned());
    }
    let defaults = f.params.iter().filter(|p| p.default.is_some()).count();
    if defaults > 0 {
        out.push(format!("defaults={defaults}"));
    }
    if f.params.iter().any(|p| p.variadic) {
        out.push("variadic".to_owned());
    }
    if f.params.iter().any(|p| p.by_ref) || f.by_ref {
        out.push("by-ref".to_owned());
    }
    for p in &f.params {
        let mut names = Vec::new();
        type_shape(p.ty.as_ref(), file_names, &mut names);
        out.extend(names.into_iter().map(|n| format!("param:{n}")));
    }
    let mut names = Vec::new();
    type_shape(f.ret.as_ref(), file_names, &mut names);
    out.extend(names.into_iter().map(|n| format!("ret:{n}")));
    out
}

/// Facts for a function or method: everything the lenses can say about it. `roles` are what is written
/// on the definition — its modifiers and attribute names.
pub(crate) fn function_facts(f: &Func, method: bool, roles: &[String], file_names: &HashSet<String>, scope_canonical: String) -> LensFacts {
    let mut walk = Walk { file_names, bound_by: HashMap::new(), caught: HashSet::new(), depth: 0, facts: Walked::default() };
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
    facts.extend(Lens::Signature, signature_facts(f, method, file_names));
    facts.extend(Lens::Decorators, roles.iter().cloned());
    // One fact rather than one per statement: what it asserts is all-or-nothing, and as one rare string
    // it carries the IDF weight that claim deserves.
    facts.push(Lens::Scope, scope_canonical);
    facts
}

/// The shape a class declares: one fact per property — its type's names and its modifiers — and one
/// per promoted constructor parameter, sorted, because a declaration is a set.
pub(crate) fn class_facts(c: &ClassDecl, file_names: &HashSet<String>) -> LensFacts {
    let mut shapes: Vec<String> = Vec::new();
    for m in &c.members {
        match m {
            ClassMember::Prop { modifiers, ty, .. } => {
                let mut names = Vec::new();
                type_shape(ty.as_ref(), file_names, &mut names);
                let mut shape = names.join(" ");
                for m in modifiers.iter().filter(|m| matches!(m.as_str(), "static" | "readonly")) {
                    shape.push(' ');
                    shape.push_str(m);
                }
                shapes.push(shape);
            }
            ClassMember::Method { name, sig, .. } if name.eq_ignore_ascii_case("__construct") => {
                for p in sig.params.iter().filter(|p| p.promoted.is_some()) {
                    let mut names = Vec::new();
                    type_shape(p.ty.as_ref(), file_names, &mut names);
                    shapes.push(names.join(" "));
                }
            }
            ClassMember::Case { .. } => shapes.push("case".to_owned()),
            _ => {}
        }
    }
    shapes.sort();
    let mut facts = LensFacts::new();
    facts.extend(Lens::Schema, shapes);
    facts
}
