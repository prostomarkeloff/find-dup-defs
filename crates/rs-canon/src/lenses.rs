//! **Lenses for Rust** — the `syn` walk, and only the walk.
//!
//! The vocabulary, the stitching, the corpus scoring and the record itself live in
//! [`dup_defs_core::lens`]. What is here is how Rust answers each of the ten questions.
//!
//! Two of them Rust cannot answer, and they are left silent rather than approximated:
//!
//! * **`decorators`** — attributes are the nearest thing, and they are answered, so this one *is*
//!   filled; what Rust lacks is a decorator that wraps behaviour, which is what the lens is about in
//!   the languages that have them. Attributes are the honest projection of "what role does it play".
//! * **`use`** — a definition's *call sites* need a walk of the whole tree, which the Rust frontend
//!   does not yet make. [`dup_defs_core::lens::merge_use_facts`] simply finds nothing to
//!   merge, and the corpus IDF weighs a lens that never speaks at zero without anyone declaring it
//!   absent.
//!
//! Everything else has a real Rust counterpart, including the one that looks like it does not:
//! `resources` asks what a definition *holds open*, and Rust's answer is not a `with` block but a
//! **guard** — a binding whose value is never read again and whose only purpose is to live until
//! the end of the scope. That is structurally detectable, and it is what this projects.

use std::collections::{BTreeSet, HashSet};

use dup_defs_core::lens::{Lens, LensFacts};
use syn::visit::{self, Visit};
use syn::{Attribute, Block, Expr, Fields, ItemEnum, ItemStruct, Signature, Stmt, Type};

/// Facts gathered in one walk of a body, sliced up per lens afterwards.
#[derive(Default)]
struct Walked {
    /// Every external callee, in call order — the `effects` lens; deduped, it is `outgoing`.
    effects: Vec<String>,
    /// Control-flow tags carrying their nesting.
    control: Vec<String>,
    /// Error types raised and caught.
    failures: BTreeSet<String>,
    /// What the body holds open for the length of a scope.
    resources: Vec<String>,
}

struct Walk<'a> {
    /// Names the file itself introduced — erased, exactly as on the widest rung of the ladder, so a
    /// lens is not held apart by the very identities it exists to see past.
    bound: &'a HashSet<String>,
    /// Names bound by a `let` and read again later. A binding absent from this is a guard.
    read_again: &'a HashSet<String>,
    depth: usize,
    facts: Walked,
}

impl<'a> Walk<'a> {
    fn run(block: &Block, bound: &'a HashSet<String>, read_again: &'a HashSet<String>) -> Walked {
        let mut walk = Walk { bound, read_again, depth: 0, facts: Walked::default() };
        walk.visit_block(block);
        walk.facts
    }

    fn tag(&mut self, tag: &str) {
        self.facts.control.push(dup_defs_core::lens::control_tag(self.depth, tag));
    }

    fn nested(&mut self, f: impl FnOnce(&mut Self)) {
        self.depth += 1;
        f(self);
        self.depth -= 1;
    }

    /// The name a call names, when it names something the file did not introduce. A bare `foo(…)`
    /// yields `foo`; `x.method(…)` yields `.method` — the receiver is whatever it is, the *method* is
    /// the grammar.
    fn callee(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Call(call) => match call.func.as_ref() {
                Expr::Path(path) => {
                    let name = path.path.segments.last()?.ident.to_string();
                    (!self.bound.contains(&name)).then_some(name)
                }
                _ => None,
            },
            Expr::MethodCall(call) => {
                let name = call.method.to_string();
                (!self.bound.contains(&name)).then(|| format!(".{name}"))
            }
            _ => None,
        }
    }
}

impl<'ast> Visit<'ast> for Walk<'_> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Some(callee) = self.callee(expr) {
            self.facts.effects.push(callee);
        }
        match expr {
            Expr::If(node) => {
                // `if let` is a different question from `if`: one tests, the other destructures.
                self.tag(if matches!(node.cond.as_ref(), Expr::Let(_)) { "iflet" } else { "if" });
                self.nested(|w| {
                    w.visit_block(&node.then_branch);
                    if let Some((_, alt)) = &node.else_branch {
                        w.visit_expr(alt);
                    }
                });
                return;
            }
            Expr::Match(node) => {
                self.tag("match");
                self.nested(|w| {
                    for arm in &node.arms {
                        w.visit_arm(arm);
                    }
                });
                self.visit_expr(&node.expr);
                return;
            }
            Expr::ForLoop(node) => {
                self.tag("for");
                self.nested(|w| w.visit_block(&node.body));
                self.visit_expr(&node.expr);
                return;
            }
            Expr::While(node) => {
                self.tag("while");
                self.nested(|w| w.visit_block(&node.body));
                return;
            }
            Expr::Loop(node) => {
                self.tag("loop");
                self.nested(|w| w.visit_block(&node.body));
                return;
            }
            Expr::Unsafe(node) => {
                self.tag("unsafe");
                self.nested(|w| w.visit_block(&node.block));
                return;
            }
            Expr::Async(node) => {
                self.tag("async");
                self.nested(|w| w.visit_block(&node.block));
                return;
            }
            Expr::Return(_) => self.tag("return"),
            Expr::Await(_) => self.tag("await"),
            // `?` is Rust's propagate-or-return, the counterpart of a re-raise.
            Expr::Try(_) => {
                self.tag("try");
                self.facts.failures.insert("?".to_owned());
            }
            Expr::Macro(node) => {
                // `panic!`, `unreachable!`, `todo!`, `bail!` — a macro that ends the path is how Rust
                // raises, and its name is the error it raises under.
                if let Some(name) = node.mac.path.segments.last() {
                    let name = name.ident.to_string();
                    if matches!(name.as_str(), "panic" | "unreachable" | "todo" | "unimplemented" | "bail" | "assert" | "assert_eq" | "assert_ne") {
                        self.facts.failures.insert(format!("{name}!"));
                    }
                }
            }
            Expr::Call(call) => {
                // `Err(SomeError { … })` / `Err(SomeError::Variant)` — the error a body constructs.
                if let Expr::Path(path) = call.func.as_ref() {
                    if path.path.segments.last().is_some_and(|s| s.ident == "Err") {
                        if let Some(inner) = call.args.first() {
                            if let Some(name) = error_name(inner) {
                                self.facts.failures.insert(name);
                            }
                        }
                    }
                }
            }
            Expr::MethodCall(call) => {
                // `.unwrap()` / `.expect()` turn a failure into a panic; that is how this body fails.
                let name = call.method.to_string();
                if matches!(name.as_str(), "unwrap" | "expect" | "unwrap_or_else" | "unwrap_err") {
                    self.facts.failures.insert(format!(".{name}"));
                }
            }
            _ => {}
        }
        visit::visit_expr(self, expr);
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        // 🔴 Rust's answer to "what does it hold open" is a **guard**: a `let` whose value is never
        // read again and whose only job is to live until the end of the scope — a lock, a span, a
        // tracing entry, a temp-dir. That is structurally detectable (bound, never mentioned
        // afterwards), which is why this is a projection rather than a list of blessed names.
        if let Stmt::Local(local) = stmt {
            if let (syn::Pat::Ident(ident), Some(init)) = (&local.pat, &local.init) {
                let name = ident.ident.to_string();
                if !self.read_again.contains(&name) {
                    if let Some(callee) = self.callee(&init.expr) {
                        self.facts.resources.push(callee);
                    }
                }
            }
        }
        visit::visit_stmt(self, stmt);
    }
}

/// The name of the error an `Err(…)` argument constructs.
///
/// A path is kept **whole** (`MyError::Empty`, not `Empty`): in Rust the enum is the failure family
/// and the variant is the specific failure, and a fact that dropped either half would either merge
/// two unrelated `NotFound`s or split one error type across its variants.
fn error_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(path) => Some(
            path.path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>().join("::"),
        ),
        Expr::Call(call) => error_name(&call.func),
        Expr::Struct(node) => node.path.segments.last().map(|s| s.ident.to_string()),
        Expr::MethodCall(call) => Some(format!(".{}", call.method)),
        _ => None,
    }
}

/// Names a block reads more than the once that bound them — the complement of a guard.
fn read_again(block: &Block) -> HashSet<String> {
    #[derive(Default)]
    struct Count {
        bound: HashSet<String>,
        seen: HashSet<String>,
    }
    impl<'ast> Visit<'ast> for Count {
        fn visit_stmt(&mut self, stmt: &'ast Stmt) {
            if let Stmt::Local(local) = stmt {
                if let syn::Pat::Ident(ident) = &local.pat {
                    self.bound.insert(ident.ident.to_string());
                }
                // The initializer is not a *read* of the name it binds, so it is walked separately
                // from the binding — otherwise every guard would look read.
                if let Some(init) = &local.init {
                    self.visit_expr(&init.expr);
                }
                return;
            }
            visit::visit_stmt(self, stmt);
        }

        fn visit_path(&mut self, path: &'ast syn::Path) {
            if let Some(first) = path.segments.first() {
                self.seen.insert(first.ident.to_string());
            }
            visit::visit_path(self, path);
        }
    }
    let mut count = Count::default();
    count.visit_block(block);
    count.bound.intersection(&count.seen).cloned().collect()
}

/// Type names a type mentions that the file did not introduce, so `Result<Foo, E>` and
/// `Result<Bar, E>` agree on `Result` and on `E` and say nothing else.
fn type_shape(ty: &Type, bound: &HashSet<String>, out: &mut Vec<String>) {
    match ty {
        Type::Path(path) => {
            for segment in &path.path.segments {
                let name = segment.ident.to_string();
                out.push(if bound.contains(&name) { "_".to_owned() } else { name });
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    for arg in &args.args {
                        if let syn::GenericArgument::Type(inner) = arg {
                            type_shape(inner, bound, out);
                        }
                    }
                }
            }
        }
        Type::Reference(node) => type_shape(&node.elem, bound, out),
        Type::Slice(node) => {
            out.push("[]".to_owned());
            type_shape(&node.elem, bound, out);
        }
        Type::Tuple(node) => node.elems.iter().for_each(|t| type_shape(t, bound, out)),
        Type::ImplTrait(_) => out.push("impl".to_owned()),
        Type::TraitObject(_) => out.push("dyn".to_owned()),
        _ => out.push("?".to_owned()),
    }
}

/// What contract the callable offers: its arity shape and the names in its types.
fn signature_facts(sig: &Signature, bound: &HashSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    let takes_self = sig.inputs.iter().any(|i| matches!(i, syn::FnArg::Receiver(_)));
    let arity = sig.inputs.len() - usize::from(takes_self);
    out.push(format!("arity={arity}"));
    if sig.asyncness.is_some() {
        out.push("async".to_owned());
    }
    if takes_self {
        out.push("method".to_owned());
    }
    for input in &sig.inputs {
        if let syn::FnArg::Typed(typed) = input {
            let mut names = Vec::new();
            type_shape(&typed.ty, bound, &mut names);
            out.extend(names.into_iter().map(|n| format!("param:{n}")));
        }
    }
    if let syn::ReturnType::Type(_, ty) = &sig.output {
        let mut names = Vec::new();
        type_shape(ty, bound, &mut names);
        out.extend(names.into_iter().map(|n| format!("ret:{n}")));
    }
    out
}

/// What role it plays: the attribute names on the definition.
fn attribute_facts(attrs: &[Attribute]) -> Vec<String> {
    let mut out: Vec<String> = attrs
        .iter()
        .filter_map(|attr| attr.path().segments.last().map(|s| s.ident.to_string()))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The shape a struct or enum declares: one fact per field, sorted.
///
/// A declaration is a **set** — the same struct with its fields reordered is the same struct — where
/// a body's statements are a sequence, so these are sorted and compared unordered.
fn fields_facts(fields: &Fields, bound: &HashSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    for field in fields {
        let mut names = Vec::new();
        type_shape(&field.ty, bound, &mut names);
        out.push(names.join(" "));
    }
    out.sort();
    out
}

/// Facts for a callable: everything the ten lenses can say about a `fn`.
pub(crate) fn callable_facts(
    sig: &Signature,
    body: &Block,
    attrs: &[Attribute],
    bound: &HashSet<String>,
    scope_canonical: String,
) -> LensFacts {
    let read = read_again(body);
    let walked = Walk::run(body, bound, &read);
    let mut facts = LensFacts::new();
    // A *set*: what it depends on, order irrelevant.
    facts.extend(Lens::Outgoing, walked.effects.iter().cloned().collect::<BTreeSet<_>>());
    // A *sequence*: the protocol it drives, order load-bearing.
    facts.extend(Lens::Effects, walked.effects.clone());
    facts.extend(Lens::Control, walked.control.clone());
    facts.extend(Lens::Failures, walked.failures.iter().cloned());
    facts.extend(Lens::Resources, walked.resources.clone());
    facts.extend(Lens::Signature, signature_facts(sig, bound));
    facts.extend(Lens::Decorators, attribute_facts(attrs));
    // One fact rather than one per statement: what it asserts is all-or-nothing — two bodies either
    // reduce to the same shape or they do not — and as one rare string it carries the IDF weight
    // that claim deserves.
    facts.push(Lens::Scope, scope_canonical);
    facts
}

/// Facts for a nominal type: what it declares, and under what attributes.
pub(crate) fn struct_facts(item: &ItemStruct, bound: &HashSet<String>) -> LensFacts {
    let mut facts = LensFacts::new();
    facts.extend(Lens::Schema, fields_facts(&item.fields, bound));
    facts.extend(Lens::Decorators, attribute_facts(&item.attrs));
    facts
}

/// Facts for an enum: every variant's shape, plus the attributes.
pub(crate) fn enum_facts(item: &ItemEnum, bound: &HashSet<String>) -> LensFacts {
    let mut facts = LensFacts::new();
    let mut shapes: Vec<String> = item
        .variants
        .iter()
        .map(|variant| fields_facts(&variant.fields, bound).join(" "))
        .collect();
    shapes.sort();
    facts.extend(Lens::Schema, shapes);
    facts.extend(Lens::Decorators, attribute_facts(&item.attrs));
    facts
}

/// Names the file itself introduced — items, so a lens sees past the identities of this module.
#[must_use]
pub fn file_bound_names(file: &syn::File) -> HashSet<String> {
    let mut out = HashSet::new();
    for item in &file.items {
        match item {
            syn::Item::Fn(node) => {
                out.insert(node.sig.ident.to_string());
            }
            syn::Item::Struct(node) => {
                out.insert(node.ident.to_string());
            }
            syn::Item::Enum(node) => {
                out.insert(node.ident.to_string());
            }
            syn::Item::Union(node) => {
                out.insert(node.ident.to_string());
            }
            syn::Item::Trait(node) => {
                out.insert(node.ident.to_string());
            }
            syn::Item::Type(node) => {
                out.insert(node.ident.to_string());
            }
            syn::Item::Const(node) => {
                out.insert(node.ident.to_string());
            }
            syn::Item::Static(node) => {
                out.insert(node.ident.to_string());
            }
            syn::Item::Mod(node) => {
                out.insert(node.ident.to_string());
            }
            _ => {}
        }
    }
    out
}
