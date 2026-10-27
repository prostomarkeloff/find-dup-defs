//! **Lenses for C** — the walk, and only the walk.
//!
//! The vocabulary, the stitching, the corpus scoring and the record itself live in
//! [`dup_defs_core::lens`]; the `use` lens's sites are [`crate::canon::collect_sites`]. What is here is
//! how C answers the other questions. It reads the owned tree ([`crate::ir`]) rather than the parser's,
//! so a macro loop is a loop here as it is in the canonicals.
//!
//! Three have a C answer that does not look like the other languages':
//!
//! * **`failures`** — C fails by returning a code. What a body fails *with* is the code it returns or
//!   tests for — an errno constant (`-EINVAL`), a project's own (`NGX_ERROR`, `AVERROR(ENOMEM)`), a
//!   negative number, `NULL` — and what it calls that does not return (`abort`, `exit`). What it does
//!   with a failure it was handed is `if (ret < 0) return ret;` — C's spelling of Rust's `?`, recorded
//!   as the same fact — or `goto` the cleanup at the end, recorded as `goto`.
//! * **`resources`** — what a body holds open is what it releases: C has no scope guard, so the release
//!   is a call, and the fact is what was *acquired* — `p = malloc(n); … free(p)` holds `malloc`,
//!   `f = fopen(…); … fclose(f)` holds `fopen` — and, when the released value was not bound here
//!   (`pthread_mutex_unlock(&s->lock)`), the release itself.
//! * **`decorators`** — the storage class and function specifiers (`static`, `inline`) and the
//!   `__attribute__`s written on the definition: what role it plays, as far as C says.

use std::collections::{BTreeSet, HashMap, HashSet};

use dup_defs_core::lens::{control_tag, Lens, LensFacts};

use crate::canon::{each_stmt_part, for_each_child, Child, Part};
use crate::ir::{Else, Expr, FieldDecl, Func, Jump, Stmt, Type};

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
    /// Names the file itself introduced — erased, exactly as on the widest rung of the ladder, so a
    /// lens is not held apart by the very identities it exists to see past.
    file_names: &'a HashSet<String>,
    /// Names the function binds: calling one is calling what it was handed, not a dependency.
    locals: &'a HashSet<String>,
    /// What each local was bound to by a call — `p = malloc(n)` binds `p` to `malloc` — so a later
    /// `free(p)` can say what was held.
    bound_by: HashMap<String, String>,
    depth: usize,
    facts: Walked,
}

/// Whether a callee releases what its first argument holds.
fn releases(callee: &str) -> bool {
    let lower = callee.to_ascii_lowercase();
    ["free", "close", "unref", "release", "destroy", "unlock", "cleanup", "dispose", "deinit", "unmap", "_put", "delete"]
        .iter()
        .any(|word| lower.contains(word))
}

/// The functions that do not return, or that a body calls to fail outright.
fn aborts(callee: &str) -> bool {
    matches!(callee, "abort" | "exit" | "_exit" | "_Exit" | "quick_exit" | "longjmp" | "siglongjmp" | "panic" | "BUG" | "__builtin_unreachable")
}

/// A name a C program fails with by convention: an errno constant (`EINVAL`), or a code that says so
/// (`NGX_ERROR`, `CURLE_OUT_OF_MEMORY`, `GIT_EUSER`, `ERR_NOMEM`).
fn error_code(name: &str) -> bool {
    let upper = name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
    let errno = name.len() > 1 && name.starts_with('E') && upper;
    errno || (upper && (name.contains("ERR") || name.contains("FAIL") || name.contains("_E")))
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

    /// The name a call names, when it names something the file did not introduce: a bare `f(…)` yields
    /// `f`, a call through a member `.m` — the receiver is whatever it is, the member is the grammar.
    /// A call through a local is a call of what it was handed: none.
    fn callee(&self, f: &Expr) -> Option<String> {
        match f {
            Expr::Name(name) => (!self.file_names.contains(name) && !self.locals.contains(name)).then(|| name.clone()),
            Expr::Field { field, .. } => (!self.file_names.contains(field)).then(|| format!(".{field}")),
            Expr::Paren(inner) => self.callee(inner),
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
                if propagates(cond, then) {
                    self.facts.failures.insert("?".to_owned());
                }
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
                self.tag(if cond.is_none() { "loop" } else { "for" });
                for part in [init, update].into_iter().flatten() {
                    self.stmt(part);
                }
                if let Some(c) = cond {
                    self.expr(c);
                }
                self.nested(|w| w.stmts(body));
            }
            Stmt::MacroLoop { head, body } => {
                self.tag("for");
                self.expr(head);
                self.nested(|w| w.stmts(body));
            }
            Stmt::Switch { value, cases } => {
                self.expr(value);
                self.tag("switch");
                self.nested(|w| {
                    for case in cases {
                        if let Some(v) = &case.value {
                            w.sentinel(v);
                        }
                        w.stmts(&case.body);
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
            Stmt::Jump { kind, .. } => self.tag(match kind {
                Jump::Break => "break",
                Jump::Continue => "continue",
                Jump::Goto => "goto",
            }),
            Stmt::Decl(specs) => {
                for spec in specs {
                    if let Some(init) = &spec.init {
                        self.note_binding(&spec.name, init);
                        self.expr(init);
                    }
                }
            }
            Stmt::Assign { lhs, rhs, .. } => {
                if let Expr::Name(name) = lhs {
                    self.note_binding(name, rhs);
                }
                self.expr(lhs);
                self.expr(rhs);
            }
            _ => each_stmt_part(s, &mut |part| match part {
                Part::Spec(spec) => {
                    if let Some(init) = &spec.init {
                        self.expr(init);
                    }
                }
                Part::Stmt(inner) => self.stmt(inner),
                Part::Stmts(list) => self.stmts(list),
                Part::Expr(e) => self.expr(e),
            }),
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Call { f, args } => {
                if let Some(callee) = self.callee(f) {
                    if aborts(&callee) {
                        self.facts.failures.insert(callee.clone());
                    }
                    if releases(&callee) {
                        self.release(&callee, args.first());
                    }
                    self.facts.effects.push(callee);
                }
            }
            // `ret == -EINVAL`, `rc != NGX_ERROR` — the failure a body tells apart from the rest.
            Expr::Binary { op, l, r } if op == "==" || op == "!=" => {
                self.sentinel(l);
                self.sentinel(r);
            }
            _ => {}
        }
        for_each_child(e, &mut |c| match c {
            Child::Expr(x) => self.expr(x),
            Child::Body(b) => self.stmts(b),
        });
    }

    /// An error code named in a test or a `case`.
    fn sentinel(&mut self, e: &Expr) {
        match e {
            Expr::Name(n) if error_code(n) => {
                self.facts.failures.insert(n.clone());
            }
            Expr::Unary { op, e } if op == "-" => self.sentinel(e),
            Expr::Paren(inner) | Expr::Cast { e: inner, .. } => self.sentinel(inner),
            _ => {}
        }
    }

    /// What a returned value fails with, when it is a failure: an error code, a negative number, `NULL`,
    /// or the code a call builds (`AVERROR(EINVAL)`).
    fn failure_value(&mut self, e: &Expr) {
        match e {
            Expr::Name(n) if error_code(n) => {
                self.facts.failures.insert(n.clone());
            }
            Expr::Unary { op, e: inner } if op == "-" => match inner.as_ref() {
                Expr::Lit(_, t) => {
                    self.facts.failures.insert(format!("-{t}"));
                }
                other => self.failure_value(other),
            },
            Expr::Null => {
                self.facts.failures.insert("NULL".to_owned());
            }
            Expr::Call { f, args } => {
                if let Some(name) = self.callee(f).filter(|c| error_code(c)) {
                    self.facts.failures.insert(name);
                    for a in args {
                        self.sentinel(a);
                    }
                }
            }
            Expr::Paren(inner) | Expr::Cast { e: inner, .. } => self.failure_value(inner),
            _ => {}
        }
    }

    /// `p = f(…)` — remember that `p` came from `f`.
    fn note_binding(&mut self, name: &str, value: &Expr) {
        let mut value = value;
        while let Expr::Cast { e, .. } | Expr::Paren(e) = value {
            value = e;
        }
        if let Expr::Call { f, .. } = value {
            if let Some(callee) = self.callee(f) {
                self.bound_by.insert(name.to_owned(), callee);
            }
        }
    }

    /// What a release releases: the call that acquired the value it releases, or the release itself.
    fn release(&mut self, callee: &str, arg: Option<&Expr>) {
        let held = match arg {
            Some(Expr::Name(n)) => self.bound_by.get(n).cloned(),
            Some(Expr::Unary { op, e }) if op == "&" => match e.as_ref() {
                Expr::Name(n) => self.bound_by.get(n).cloned(),
                _ => None,
            },
            _ => None,
        };
        self.facts.resources.push(held.unwrap_or_else(|| callee.to_owned()));
    }
}

/// `if (ret < 0) return ret;`, `if (err) return err;` — the failure handed back to the caller.
fn propagates(cond: &Expr, then: &[Stmt]) -> bool {
    fn tested(e: &Expr) -> Option<&str> {
        match e {
            Expr::Name(n) => Some(n),
            Expr::Paren(inner) => tested(inner),
            Expr::Binary { op, l, r } if matches!(op.as_str(), "<" | "!=" | ">" | "<=" | ">=") => match (l.as_ref(), r.as_ref()) {
                (Expr::Name(n), Expr::Lit(..) | Expr::Name(_) | Expr::Null) | (Expr::Lit(..) | Expr::Null, Expr::Name(n)) => Some(n),
                _ => None,
            },
            _ => None,
        }
    }
    let Some(checked) = tested(cond) else { return false };
    matches!(then.last(), Some(Stmt::Return(Some(Expr::Name(n)))) if n == checked)
}

/// The names a type mentions that the file did not introduce, so `struct user **` and
/// `struct order **` agree on the pointers and say nothing else.
fn type_shape(t: &Type, file_names: &HashSet<String>, out: &mut Vec<String>) {
    match t {
        Type::Named(name) => {
            let bare = name.rsplit(' ').next().unwrap_or(name);
            out.push(if file_names.contains(bare) { "_".to_owned() } else { name.clone() });
        }
        Type::Ptr(inner) => {
            out.push("*".to_owned());
            type_shape(inner, file_names, out);
        }
        Type::Array(inner, _) => {
            out.push("[]".to_owned());
            type_shape(inner, file_names, out);
        }
        Type::Func(..) => out.push("fn".to_owned()),
        Type::Record { kind, .. } => out.push((*kind).to_owned()),
        Type::Enum { .. } => out.push("enum".to_owned()),
        Type::Other => out.push("?".to_owned()),
    }
}

/// What contract the function offers: its arity shape and the names in its types.
fn signature_facts(f: &Func, file_names: &HashSet<String>) -> Vec<String> {
    let mut out = vec![format!("arity={}", f.params.len())];
    if f.variadic {
        out.push("variadic".to_owned());
    }
    for p in &f.params {
        let mut names = Vec::new();
        type_shape(&p.ty, file_names, &mut names);
        out.extend(names.into_iter().map(|n| format!("param:{n}")));
    }
    let mut names = Vec::new();
    type_shape(&f.ret, file_names, &mut names);
    out.extend(names.into_iter().map(|n| format!("ret:{n}")));
    out
}

/// Facts for a function: everything the lenses can say about it. `specifiers` are what is written on
/// the definition — storage class, function specifiers, attribute names.
pub(crate) fn function_facts(
    f: &Func,
    specifiers: &[String],
    locals: &HashSet<String>,
    file_names: &HashSet<String>,
    scope_canonical: String,
) -> LensFacts {
    let mut walk = Walk { file_names, locals, bound_by: HashMap::new(), depth: 0, facts: Walked::default() };
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
    facts.extend(Lens::Signature, signature_facts(f, file_names));
    facts.extend(Lens::Decorators, specifiers.iter().cloned());
    // One fact rather than one per statement: what it asserts is all-or-nothing, and as one rare
    // string it carries the IDF weight that claim deserves.
    facts.push(Lens::Scope, scope_canonical);
    facts
}

/// The shape a record declares: one fact per member — its type's names, a bit-field's width marked —
/// sorted, because a declaration is a set.
pub(crate) fn record_facts(fields: &[FieldDecl], file_names: &HashSet<String>) -> LensFacts {
    let mut shapes: Vec<String> = fields
        .iter()
        .map(|f| {
            let mut names = Vec::new();
            type_shape(&f.ty, file_names, &mut names);
            let mut shape = names.join(" ");
            if f.bits.is_some() {
                shape.push_str(" :bits");
            }
            if f.name.is_none() {
                shape = format!("embed {shape}");
            }
            shape
        })
        .collect();
    shapes.sort();
    let mut facts = LensFacts::new();
    facts.extend(Lens::Schema, shapes);
    facts
}
