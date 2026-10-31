//! Structural canonicalization of TypeScript definitions over the oxc AST.
//!
//! Mirrors the role `py-canon::canon` plays for Python: produces three strings per callable
//! definition, all walked from one parse — `(cluster_canonical, xname_canonical, lines, size)` —
//! and a names-preserved cluster canonical for any def kind (used by the engine's name-gated
//! pass over body kinds).
//!
//! Unlike `py-canon`, the canonical here doesn't have a byte-exact external reference to match
//! (`CPython`'s `ast.dump` shaped Python; there's no equivalent published reference for TS). We
//! emit a compact, internally-consistent **s-expr** (node name + relevant child fields), which
//! is what difflib's Ratcliff–Obershelp ratio compares for name-gated similarity, and what the
//! cross-name pass's `Eq` checks once locals have been alpha-renamed. Concrete consequences:
//!
//! * **Identifier renaming**: in cluster mode names pass through verbatim; in xname mode bound
//!   locals (params, `let`/`const`/`var` bindings, function / class names declared inside the
//!   body, catch / for-of / for-in targets, destructuring patterns, import specifiers) are
//!   re-numbered to `_v{0..}` by first occurrence — so `function add(a, b) { return a + b }`
//!   alpha-equals `function plus(x, y) { return x + y }`. The top def's own name is blanked to
//!   `_fn` in xname mode.
//! * **Type-only wrappers** (`x as Foo`, `x satisfies Foo`, `x!`, `<Foo>x`, `<T>(...)`) are
//!   emitted with their inner expression preserved — they're real syntax the user authored, not
//!   noise. The accompanying type AST is summarized to a node-kind tag (the TS type system has
//!   ~80 variants; tag-only suffices for cluster detection without committing to byte-exact
//!   type printing).
//! * **JSX nodes** emit as `JSX(tag, [attributes], [children])` / `JSXFragment([children])`, text
//!   whitespace-collapsed — markup is most of what a component says, so it is compared, not elided.
//! * **Long-tail nodes** we haven't explicitly cased emit as `Unknown_<NodeKind>` — deterministic
//!   for any given input, surfacing in `--calibrate` so the next round of tuning sees them.
//!
//! Type-3 lines are per-statement renamed s-expr strings (one statement → one line); the engine
//! tokenizes them via the same regex as `py-canon`'s lines and the IDF/cosine works unchanged.
#![allow(
    clippy::too_many_lines, // top-level match arms enumerate AST variants — splitting them just spreads the same shape over more files
    clippy::doc_markdown, // CPython, TSAs etc. read fine in prose without backticks; clippy is over-eager
    clippy::doc_lazy_continuation, // module docstrings use loose list continuation, intentional
    clippy::self_only_used_in_recursion, // Collect::collect_expr is a visitor — calls into itself, mutation happens via add_binding called from outer dispatch
    clippy::match_same_arms, // visitor's no-op arms enumerate AST variants for clarity
    clippy::needless_pass_by_value, // helpers accept owned Vec<String> by value to consume one allocation
)]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use dup_defs_core::uses::{site_ctx, Construct, Site, Within};
use dup_defs_core::Statement as CoreStatement;
use dup_defs_core::UNFILED;
use dup_defs_core::reach::reach_path;

/// `(cluster_canonical, xname_canonical, type3_lines, node_count)` — the analysis tuple the scan
/// reads to build a `Def`'s cluster canonical + `Analysis`. `ts-canon`'s own type (was shared via
/// `dup-defs-core`, now local since the engine consumes `Def`, not this tuple).
pub use dup_defs_core::AnalyzedFn;
use oxc_allocator::{Allocator, CloneIn};
use oxc_ast::AstBuilder;
use oxc_ast::ast::{
    self, ArrayExpressionElement, Argument, AssignmentTarget, AssignmentTargetMaybeDefault,
    AssignmentTargetPattern, AssignmentTargetProperty, BindingPattern, BindingProperty,
    BindingRestElement, ChainElement, Class, Declaration, ExportDefaultDeclarationKind,
    Expression, ForStatementInit, ForStatementLeft, FormalParameter, FormalParameters, Function,
    MemberExpression, ObjectExpression, ObjectPropertyKind, PropertyKey, SimpleAssignmentTarget,
    Statement, SwitchCase, TSLiteral, TSSignature, TSType, TSTypeName, TemplateLiteral,
    VariableDeclaration, VariableDeclarationKind, VariableDeclarator,
};
use oxc_parser::Parser;
use oxc_span::SourceType;
use rayon::prelude::*;

// ───────────────────────────── bound-locals collector ─────────────────────────────

/// Collect names *bound* anywhere inside the top callable — the rename set for xname mode.
/// Mirrors `py-canon::Collect`: params + var declarations + function/class declarations + catch
/// + for-of/in + destructuring + import bindings. Nested defs' *params* are NOT collected
/// (`CPython` behavior, preserved for TS), only the top function's params plus arrow-fn /
/// inner-fn declared *names* (which ARE bindings in the outer scope).
#[derive(Default)]
struct Collect {
    bound: HashSet<String>,
}

impl Collect {
    fn add_binding(&mut self, pat: &BindingPattern<'_>) {
        match pat {
            BindingPattern::BindingIdentifier(id) => {
                self.bound.insert(id.name.to_string());
            }
            BindingPattern::ArrayPattern(arr) => {
                for p in arr.elements.iter().flatten() {
                    self.add_binding(p);
                }
                if let Some(rest) = &arr.rest {
                    self.add_binding_rest(rest);
                }
            }
            BindingPattern::ObjectPattern(obj) => {
                for prop in &obj.properties {
                    self.add_binding_property(prop);
                }
                if let Some(rest) = &obj.rest {
                    self.add_binding_rest(rest);
                }
            }
            BindingPattern::AssignmentPattern(asgn) => self.add_binding(&asgn.left),
        }
    }

    fn add_binding_property(&mut self, prop: &BindingProperty<'_>) {
        self.add_binding(&prop.value);
    }

    fn add_binding_rest(&mut self, rest: &BindingRestElement<'_>) {
        self.add_binding(&rest.argument);
    }

    fn add_params(&mut self, params: &FormalParameters<'_>) {
        for item in &params.items {
            self.add_binding(&item.pattern);
        }
        if let Some(rest) = &params.rest {
            // `FormalParameterRest` wraps a `BindingRestElement` — the inner one is what binds.
            self.add_binding_rest(&rest.rest);
        }
    }

    fn collect_stmt(&mut self, stmt: &Statement<'_>) {
        match stmt {
            Statement::BlockStatement(b) => {
                for s in &b.body {
                    self.collect_stmt(s);
                }
            }
            Statement::VariableDeclaration(v) => {
                for decl in &v.declarations {
                    self.add_binding(&decl.id);
                    if let Some(init) = &decl.init {
                        self.collect_expr(init);
                    }
                }
            }
            Statement::FunctionDeclaration(f) => {
                if let Some(id) = &f.id {
                    self.bound.insert(id.name.to_string());
                }
                // params of nested defs are NOT collected — only top fn's. Body still walked so a
                // nested var/const inside a closure adds *its* name (rare but valid: `function f
                // () { function g() {}; return g; }` binds `g`).
                if let Some(body) = &f.body {
                    for s in &body.statements {
                        self.collect_stmt(s);
                    }
                }
            }
            Statement::ClassDeclaration(c) => {
                if let Some(id) = &c.id {
                    self.bound.insert(id.name.to_string());
                }
            }
            Statement::IfStatement(i) => {
                self.collect_expr(&i.test);
                self.collect_stmt(&i.consequent);
                if let Some(a) = &i.alternate {
                    self.collect_stmt(a);
                }
            }
            Statement::ForStatement(f) => {
                if let Some(init) = &f.init {
                    self.collect_for_init(init);
                }
                if let Some(test) = &f.test {
                    self.collect_expr(test);
                }
                if let Some(update) = &f.update {
                    self.collect_expr(update);
                }
                self.collect_stmt(&f.body);
            }
            Statement::ForInStatement(f) => {
                self.collect_for_left(&f.left);
                self.collect_expr(&f.right);
                self.collect_stmt(&f.body);
            }
            Statement::ForOfStatement(f) => {
                self.collect_for_left(&f.left);
                self.collect_expr(&f.right);
                self.collect_stmt(&f.body);
            }
            Statement::WhileStatement(w) => {
                self.collect_expr(&w.test);
                self.collect_stmt(&w.body);
            }
            Statement::DoWhileStatement(d) => {
                self.collect_stmt(&d.body);
                self.collect_expr(&d.test);
            }
            Statement::ReturnStatement(r) => {
                if let Some(arg) = &r.argument {
                    self.collect_expr(arg);
                }
            }
            Statement::ThrowStatement(t) => self.collect_expr(&t.argument),
            Statement::TryStatement(t) => {
                for s in &t.block.body {
                    self.collect_stmt(s);
                }
                if let Some(handler) = &t.handler {
                    if let Some(param) = &handler.param {
                        self.add_binding(&param.pattern);
                    }
                    for s in &handler.body.body {
                        self.collect_stmt(s);
                    }
                }
                if let Some(fin) = &t.finalizer {
                    for s in &fin.body {
                        self.collect_stmt(s);
                    }
                }
            }
            Statement::SwitchStatement(s) => {
                self.collect_expr(&s.discriminant);
                for case in &s.cases {
                    if let Some(t) = &case.test {
                        self.collect_expr(t);
                    }
                    for cs in &case.consequent {
                        self.collect_stmt(cs);
                    }
                }
            }
            Statement::LabeledStatement(l) => self.collect_stmt(&l.body),
            Statement::ExpressionStatement(e) => self.collect_expr(&e.expression),
            Statement::ImportDeclaration(imp) => {
                if let Some(specifiers) = &imp.specifiers {
                    for spec in specifiers {
                        match spec {
                            ast::ImportDeclarationSpecifier::ImportSpecifier(s) => {
                                self.bound.insert(s.local.name.to_string());
                            }
                            ast::ImportDeclarationSpecifier::ImportDefaultSpecifier(s) => {
                                self.bound.insert(s.local.name.to_string());
                            }
                            ast::ImportDeclarationSpecifier::ImportNamespaceSpecifier(s) => {
                                self.bound.insert(s.local.name.to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn collect_for_init(&mut self, init: &ForStatementInit<'_>) {
        match init {
            ForStatementInit::VariableDeclaration(v) => {
                for decl in &v.declarations {
                    self.add_binding(&decl.id);
                    if let Some(e) = &decl.init {
                        self.collect_expr(e);
                    }
                }
            }
            other => {
                if let Some(e) = other.as_expression() {
                    self.collect_expr(e);
                }
            }
        }
    }

    fn collect_for_left(&mut self, left: &ForStatementLeft<'_>) {
        if let ForStatementLeft::VariableDeclaration(v) = left {
            for decl in &v.declarations {
                self.add_binding(&decl.id);
            }
        }
        // AssignmentTarget branch — re-assigns existing binding, no new local.
    }

    fn collect_expr(&mut self, expr: &Expression<'_>) {
        // Conservative walk: we only need bound *names*, so we recurse far enough to catch
        // nested var/function declarations (impossible inside an Expression except via
        // arrow/function/class expressions, whose params we deliberately DON'T collect — that
        // matches CPython's "only top fn's params bound" rule). Anything else just bubbles up.
        match expr {
            Expression::ArrowFunctionExpression(_) | Expression::FunctionExpression(_) => {}
            Expression::ClassExpression(_) => {}
            Expression::AssignmentExpression(a) => self.collect_expr(&a.right),
            Expression::BinaryExpression(b) => {
                self.collect_expr(&b.left);
                self.collect_expr(&b.right);
            }
            Expression::LogicalExpression(l) => {
                self.collect_expr(&l.left);
                self.collect_expr(&l.right);
            }
            Expression::UnaryExpression(u) => self.collect_expr(&u.argument),
            Expression::ConditionalExpression(c) => {
                self.collect_expr(&c.test);
                self.collect_expr(&c.consequent);
                self.collect_expr(&c.alternate);
            }
            Expression::CallExpression(c) => {
                self.collect_expr(&c.callee);
                for a in &c.arguments {
                    if let Some(e) = a.as_expression() {
                        self.collect_expr(e);
                    }
                }
            }
            Expression::NewExpression(n) => {
                self.collect_expr(&n.callee);
                for a in &n.arguments {
                    if let Some(e) = a.as_expression() {
                        self.collect_expr(e);
                    }
                }
            }
            Expression::SequenceExpression(s) => {
                for e in &s.expressions {
                    self.collect_expr(e);
                }
            }
            Expression::ArrayExpression(arr) => {
                for elt in &arr.elements {
                    if let ArrayExpressionElement::SpreadElement(s) = elt {
                        self.collect_expr(&s.argument);
                    } else if let Some(e) = elt.as_expression() {
                        self.collect_expr(e);
                    }
                }
            }
            Expression::ObjectExpression(o) => {
                for prop in &o.properties {
                    if let ObjectPropertyKind::ObjectProperty(p) = prop {
                        self.collect_expr(&p.value);
                    }
                }
            }
            Expression::TSAsExpression(a) => self.collect_expr(&a.expression),
            Expression::TSSatisfiesExpression(s) => self.collect_expr(&s.expression),
            Expression::TSNonNullExpression(n) => self.collect_expr(&n.expression),
            Expression::TSTypeAssertion(t) => self.collect_expr(&t.expression),
            Expression::TSInstantiationExpression(i) => self.collect_expr(&i.expression),
            Expression::ParenthesizedExpression(p) => self.collect_expr(&p.expression),
            _ => {}
        }
    }
}

// ─────────────────── small helpers for sum-type member access ───────────────────

// NOTE: oxc's `inherit_variants!` keeps the *parent's* discriminants only for the inherited
// variants — it does NOT renumber other variants to avoid collisions. So a value of type
// `SimpleAssignmentTarget::TSNonNullExpression` (discriminant 3) reinterpret-cast as
// `Expression` would land on `Expression::BigIntLiteral` (also discriminant 3) and segfault on
// access. Every transmute here would be UB. Instead the emitter exposes a per-enum method that
// dispatches via real pattern matching — slightly more verbose, but provably sound.

// ───────────────────────────── s-expr emitter ─────────────────────────────

/// How a dump writes the type annotations of parameters, returns and declarations.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Types {
    /// Left out — the cluster canonical, as the name-gated pass has always compared it.
    Dropped,
    /// As written — the statement stream, where converge parts two definitions on the names they
    /// use, a type among them.
    Written,
    /// Every type reference as a `_T{n}` slot, the names reported apart
    /// ([`AnalyzedFn::type_refs`]) — the exact and near-copy canonicals. TypeScript erases its
    /// annotations before anything runs, so two bodies differing only in the declared types run the
    /// same code: the Python frontend's argument holds here exactly. Renamed consistently, not
    /// erased, so `f(a: X, b: X)` and `f(a: X, b: Y)` stay apart.
    Slots,
}

struct Dump<'a> {
    /// Renaming mode: `None` = cluster canonical (names preserved); `Some(set)` = xname
    /// canonical (every bound name → positional `_v{n}` slot).
    locals: Option<&'a HashSet<String>>,
    /// First-occurrence-of-bound-local → slot number. Numbering is deterministic per parse
    /// (driven by walk order), so two structurally identical functions with different parameter
    /// names produce byte-identical xname canonicals.
    map: HashMap<String, u32>,
    /// In xname mode, blank the *top* def's own name to `_fn` exactly once. Nested defs keep
    /// their renamed names (they're regular bound locals).
    blanked: bool,
    /// In xname mode, the top *binding* name to blank to `_fn` on its first occurrence — the
    /// arrow / function-expression `const`/`let` form (`const foo = () => …`), whose name rides on
    /// the declarator rather than on a `Func`/`Class` node. `function foo` / `class Foo` are blanked
    /// inside [`Dump::function`] / [`Dump::class`] via `is_top` instead, so this stays `None` for
    /// them. Makes two differently-named arrow-consts alpha-equal, exactly like named functions.
    blank: Option<String>,
    /// Count of node-emit calls — used as the cross-name pass's "substance" gate.
    count: usize,
    /// `Some` while writing the Type-3 lines: every sub-expression worth a unit of its own, rendered
    /// as it appears inside its line, with that line's index — [`AnalyzedFn::type3_units`].
    units: Option<Vec<(u32, String)>>,
    /// The index of the line being written, which the units it yields are filed under.
    unit_line: u32,
    /// How many expressions enclose the one being written: `0` for a statement's own expression.
    expr_depth: u32,
    /// How annotations are written — see [`Types`].
    types: Types,
    /// Type reference → `_T{n}`, in [`Types::Slots`].
    tmap: HashMap<String, u32>,
    /// Every type reference turned into a slot, in order of appearance — [`AnalyzedFn::type_refs`].
    type_refs: Vec<String>,
    /// Type parameters the definition declares: slotted like any type, but bound here, so not
    /// vocabulary it shares with anything.
    declared_types: HashSet<String>,
    /// Use-site mode: the definition whose use is being written. Its name becomes the anchor `_t0`,
    /// and its own surface — a property read off it, a key of an object it is called with, an
    /// attribute it is rendered with — comes back marked `#name#` for the profile to number.
    anchor: Option<&'a str>,
    /// Hunt mode: the names that count as anchors. Every one the walk meets is kept in `hunted`.
    known: Option<&'a HashSet<&'a str>>,
    hunted: Vec<String>,
    /// Hunt mode: the methods called (`.name`), for the co-occurrence channel of the `use` lens.
    calls: BTreeSet<String>,
}

impl<'a> Dump<'a> {
    fn new(locals: Option<&'a HashSet<String>>) -> Self {
        Self {
            locals,
            map: HashMap::new(),
            blanked: false,
            blank: None,
            count: 0,
            units: None,
            unit_line: 0,
            expr_depth: 0,
            types: Types::Dropped,
            tmap: HashMap::new(),
            type_refs: Vec::new(),
            declared_types: HashSet::new(),
            anchor: None,
            known: None,
            hunted: Vec::new(),
            calls: BTreeSet::new(),
        }
    }

    /// Whether this dump writes a use site (hunting for, or anchored on, a name).
    fn use_site(&self) -> bool {
        self.known.is_some() || self.anchor.is_some()
    }

    /// Whether `expr` is the anchor itself.
    fn anchored(&self, expr: &Expression<'_>) -> bool {
        matches!(expr, Expression::Identifier(id) if self.anchor == Some(id.name.as_str()))
    }

    /// A name, as this dump writes it: the anchor's `_t0`, or renamed as a local would be; noted
    /// when it is one of the names hunted for.
    fn reference(&mut self, name: &str) -> String {
        if let Some(known) = self.known {
            if known.contains(name) && !self.hunted.iter().any(|h| h == name) {
                self.hunted.push(name.to_owned());
            }
        }
        if self.anchor == Some(name) {
            return "_t0".to_owned();
        }
        self.rename(name)
    }

    /// An identifier of the anchor's own surface — `#name#` when `on_anchor`, as written otherwise.
    fn surface(name: &str, on_anchor: bool) -> String {
        if on_anchor {
            format!("#{name}#")
        } else {
            name.to_owned()
        }
    }

    /// A dump that writes annotations the given way.
    fn with_types(locals: Option<&'a HashSet<String>>, types: Types) -> Self {
        Self { types, ..Self::new(locals) }
    }

    /// A dump that notes which of `known` a line mentions, and the methods it calls.
    fn hunting(locals: &'a HashSet<String>, known: &'a HashSet<&'a str>) -> Self {
        Self { known: Some(known), ..Self::with_types(Some(locals), Types::Slots) }
    }

    /// A dump that writes a line as a use site of `anchor`.
    fn anchoring(locals: &'a HashSet<String>, anchor: &'a str) -> Self {
        Self { anchor: Some(anchor), ..Self::with_types(Some(locals), Types::Slots) }
    }

    fn rename(&mut self, name: &str) -> String {
        dup_defs_core::alpha_rename(&mut self.map, self.locals, name)
    }

    /// A type reference: its `_T{n}` slot in [`Types::Slots`] (reported apart unless the definition
    /// declared it), as written otherwise.
    fn type_ref(&mut self, name: &str) -> String {
        if self.known.is_some() || self.anchor.is_some() {
            // The anchor keeps its own slot even in type position.
            let reference = self.reference(name);
            if reference == "_t0" {
                return reference;
            }
        }
        if self.types != Types::Slots {
            return name.to_owned();
        }
        if !self.declared_types.contains(name) {
            self.type_refs.push(name.to_owned());
        }
        dup_defs_core::type_slot(&mut self.tmap, name)
    }

    /// An annotation, when this dump writes them.
    fn annotation(&mut self, annotation: Option<&ast::TSTypeAnnotation<'_>>) -> Option<String> {
        if self.types == Types::Dropped {
            return None;
        }
        annotation.map(|a| self.ts_type(&a.type_annotation))
    }

    /// A binding with its annotation, when this dump writes them.
    fn annotated(&mut self, binding: String, annotation: Option<&ast::TSTypeAnnotation<'_>>) -> String {
        match self.annotation(annotation) {
            Some(ty) => self.node("Annot", &[binding, ty]),
            None => binding,
        }
    }

    /// The type parameters a definition declares, entered as bound and written as a node — or
    /// nothing, when this dump drops annotations or there are none.
    fn type_params(&mut self, params: Option<&ast::TSTypeParameterDeclaration<'_>>) -> Option<String> {
        let names = self.declare_type_params(params);
        (!names.is_empty()).then(|| self.list("TParams", names))
    }

    /// Enter the type parameters a definition declares as bound, and write each one's name.
    fn declare_type_params(&mut self, params: Option<&ast::TSTypeParameterDeclaration<'_>>) -> Vec<String> {
        let Some(params) = params.filter(|_| self.types != Types::Dropped) else { return Vec::new() };
        params
            .params
            .iter()
            .map(|p| {
                let name = p.name.name.to_string();
                if self.types == Types::Slots {
                    self.declared_types.insert(name.clone());
                }
                self.type_ref(&name)
            })
            .collect()
    }

    /// The header of the Type-3 lines: what the Python frontend's `def _fn(_v0: _T0) -> _T1:` line
    /// says — the parameters, their types, the return type — and nothing else.
    ///
    /// 🔴 Not the `Func(…)` node the stream opens with. The pass reads every function's first line
    /// as context and weighs it like any other; written as an s-expr, the header spells the parse
    /// tree's grammar — `Params`, `Param`, `Bind`, `Annot`, flags — which only headers spell, so
    /// every one of those tokens was rare, and the header outweighed the body of every short
    /// function: two one-liners with one signature scored as copies whatever their bodies did.
    fn compact_header(&mut self, parts: &Parts<'_, '_>) -> String {
        let type_params = self.declare_type_params(parts.type_params);
        let mut out = String::from(if parts.is_async { "async function _fn" } else { "function _fn" });
        if !type_params.is_empty() {
            let _ = write!(out, "<{}>", type_params.join(", "));
        }
        let mut params: Vec<String> = Vec::with_capacity(parts.params.items.len() + 1);
        for p in &parts.params.items {
            let mut param = match &p.pattern {
                BindingPattern::BindingIdentifier(id) => self.rename(id.name.as_str()),
                other => self.binding(other),
            };
            if let Some(ty) = self.annotation(p.type_annotation.as_deref()) {
                let _ = write!(param, ": {ty}");
            }
            if let Some(init) = &p.initializer {
                let value = self.expr(init);
                let _ = write!(param, " = {value}");
            }
            params.push(param);
        }
        if let Some(rest) = &parts.params.rest {
            let inner = self.binding(&rest.rest.argument);
            params.push(format!("...{inner}"));
        }
        let _ = write!(out, "({})", params.join(", "));
        if let Some(ret) = self.annotation(parts.ret) {
            let _ = write!(out, ": {ret}");
        }
        out
    }

    /// Emit one node as `Tag(field1, field2, …)`. Empty trailing fields are NOT trimmed (so
    /// `If(test, body, )` reads differently from `If(test, body)`) — this keeps each variant's
    /// arity stable, making structural differences visible to difflib.
    fn node(&mut self, tag: &str, fields: &[String]) -> String {
        self.count += 1;
        let mut s = String::with_capacity(tag.len() + 2 + fields.iter().map(String::len).sum::<usize>() + fields.len() * 2);
        s.push_str(tag);
        s.push('(');
        for (i, f) in fields.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(f);
        }
        s.push(')');
        s
    }

    fn list(&mut self, tag: &str, items: Vec<String>) -> String {
        let joined = items.join(", ");
        self.node(tag, &[joined])
    }

    fn lit_str(s: &str) -> String {
        // Compact repr — single-quoted, backslash + quote escaped, control chars hex-escaped.
        let mut out = String::with_capacity(s.len() + 2);
        out.push('\'');
        for c in s.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '\'' => out.push_str("\\'"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    let _ = write!(out, "\\x{:02x}", c as u32);
                }
                c => out.push(c),
            }
        }
        out.push('\'');
        out
    }

    fn lit_num(value: f64) -> String {
        if value.is_nan() {
            return "NaN".to_owned();
        }
        if value.is_infinite() {
            return if value < 0.0 { "-Infinity".to_owned() } else { "Infinity".to_owned() };
        }
        // Round-trip shortest — Rust's default Display is good for this.
        format!("{value}")
    }

    // ────── statements ──────

    fn stmt(&mut self, stmt: &Statement<'_>) -> String {
        match stmt {
            Statement::BlockStatement(b) => {
                let items: Vec<String> = b.body.iter().map(|s| self.stmt(s)).collect();
                self.list("Block", items)
            }
            Statement::EmptyStatement(_) => self.node("Empty", &[]),
            Statement::DebuggerStatement(_) => self.node("Debugger", &[]),
            Statement::ExpressionStatement(e) => {
                let v = self.expr(&e.expression);
                self.node("Expr", &[v])
            }
            Statement::IfStatement(i) => {
                let test = self.expr(&i.test);
                let cons = self.stmt(&i.consequent);
                let alt = i.alternate.as_ref().map_or_else(String::new, |a| self.stmt(a));
                self.node("If", &[test, cons, alt])
            }
            Statement::ForStatement(f) => {
                let init = f.init.as_ref().map_or_else(String::new, |i| self.for_init(i));
                let test = f.test.as_ref().map_or_else(String::new, |t| self.expr(t));
                let update = f.update.as_ref().map_or_else(String::new, |u| self.expr(u));
                let body = self.stmt(&f.body);
                self.node("For", &[init, test, update, body])
            }
            Statement::ForInStatement(f) => {
                let left = self.for_left(&f.left);
                let right = self.expr(&f.right);
                let body = self.stmt(&f.body);
                self.node("ForIn", &[left, right, body])
            }
            Statement::ForOfStatement(f) => {
                let tag = if f.r#await { "ForAwaitOf" } else { "ForOf" };
                let left = self.for_left(&f.left);
                let right = self.expr(&f.right);
                let body = self.stmt(&f.body);
                self.node(tag, &[left, right, body])
            }
            Statement::WhileStatement(w) => {
                let test = self.expr(&w.test);
                let body = self.stmt(&w.body);
                self.node("While", &[test, body])
            }
            Statement::DoWhileStatement(d) => {
                let body = self.stmt(&d.body);
                let test = self.expr(&d.test);
                self.node("DoWhile", &[body, test])
            }
            Statement::ReturnStatement(r) => {
                let arg = r.argument.as_ref().map_or_else(String::new, |a| self.expr(a));
                self.node("Return", &[arg])
            }
            Statement::ThrowStatement(t) => {
                let arg = self.expr(&t.argument);
                self.node("Throw", &[arg])
            }
            Statement::TryStatement(t) => {
                let block = self.block_body(&t.block.body);
                let handler = match &t.handler {
                    Some(h) => {
                        let param = h.param.as_ref().map_or_else(String::new, |p| self.binding(&p.pattern));
                        let body = self.block_body(&h.body.body);
                        self.node("Catch", &[param, body])
                    }
                    None => String::new(),
                };
                let fin = match &t.finalizer {
                    Some(f) => self.block_body(&f.body),
                    None => String::new(),
                };
                self.node("Try", &[block, handler, fin])
            }
            Statement::BreakStatement(b) => {
                let label =
                    b.label.as_ref().map_or_else(String::new, |l| Self::lit_str(l.name.as_str()));
                self.node("Break", &[label])
            }
            Statement::ContinueStatement(c) => {
                let label =
                    c.label.as_ref().map_or_else(String::new, |l| Self::lit_str(l.name.as_str()));
                self.node("Continue", &[label])
            }
            Statement::LabeledStatement(l) => {
                let label = Self::lit_str(l.label.name.as_str());
                let body = self.stmt(&l.body);
                self.node("Label", &[label, body])
            }
            Statement::SwitchStatement(s) => {
                let disc = self.expr(&s.discriminant);
                let cases: Vec<String> = s.cases.iter().map(|c| self.switch_case(c)).collect();
                let cs = cases.join(", ");
                self.node("Switch", &[disc, cs])
            }
            Statement::WithStatement(w) => {
                let obj = self.expr(&w.object);
                let body = self.stmt(&w.body);
                self.node("With", &[obj, body])
            }
            Statement::VariableDeclaration(v) => self.variable_decl(v),
            Statement::FunctionDeclaration(f) => self.function(f, false),
            Statement::ClassDeclaration(c) => self.class(c, false),
            Statement::ExportNamedDeclaration(e) => {
                let inner = e.declaration.as_ref().map_or_else(String::new, |d| self.declaration(d));
                self.node("ExportNamed", &[inner])
            }
            Statement::ExportDefaultDeclaration(e) => {
                let inner = match &e.declaration {
                    ExportDefaultDeclarationKind::FunctionDeclaration(f) => self.function(f, false),
                    ExportDefaultDeclarationKind::ClassDeclaration(c) => self.class(c, false),
                    other => match other.as_expression() {
                        Some(e) => self.expr(e),
                        None => self.node("Unknown_ExportDefault", &[]),
                    },
                };
                self.node("ExportDefault", &[inner])
            }
            Statement::ExportAllDeclaration(_) => self.node("ExportAll", &[]),
            Statement::ImportDeclaration(_) => self.node("Import", &[]),
            Statement::TSTypeAliasDeclaration(t) => self.ts_type_alias(t),
            Statement::TSInterfaceDeclaration(i) => self.ts_interface(i),
            Statement::TSEnumDeclaration(e) => {
                let name = self.rename(e.id.name.as_str());
                self.node("TSEnum", &[Self::lit_str(&name)])
            }
            Statement::TSModuleDeclaration(_) => self.node("TSModule", &[]),
            Statement::TSImportEqualsDeclaration(_) => self.node("TSImportEquals", &[]),
            Statement::TSExportAssignment(_) => self.node("TSExportAssign", &[]),
            Statement::TSNamespaceExportDeclaration(_) => self.node("TSNamespaceExport", &[]),
            Statement::TSGlobalDeclaration(_) => self.node("TSGlobal", &[]),
        }
    }

    fn block_body(&mut self, stmts: &[Statement<'_>]) -> String {
        let items: Vec<String> = stmts.iter().map(|s| self.stmt(s)).collect();
        self.list("Block", items)
    }

    fn for_init(&mut self, init: &ForStatementInit<'_>) -> String {
        match init {
            ForStatementInit::VariableDeclaration(v) => self.variable_decl(v),
            other => match other.as_expression() {
                Some(e) => self.expr(e),
                None => self.node("Unknown_ForInit", &[]),
            },
        }
    }

    fn for_left(&mut self, left: &ForStatementLeft<'_>) -> String {
        match left {
            ForStatementLeft::VariableDeclaration(v) => self.variable_decl(v),
            other => match other.as_assignment_target() {
                Some(t) => self.assignment_target(t),
                None => self.node("Unknown_ForLeft", &[]),
            },
        }
    }

    fn switch_case(&mut self, case: &SwitchCase<'_>) -> String {
        let test = case.test.as_ref().map_or_else(String::new, |t| self.expr(t));
        let cons: Vec<String> = case.consequent.iter().map(|s| self.stmt(s)).collect();
        let body = self.list("Block", cons);
        self.node("Case", &[test, body])
    }

    fn variable_decl(&mut self, v: &VariableDeclaration<'_>) -> String {
        let items: Vec<String> = v.declarations.iter().map(|d| self.declarator(d)).collect();
        self.var_node(v.kind, &items)
    }

    /// A `Var` node over declarators already rendered — a declaration's whole list, or the one
    /// declarator a definition is bound by.
    fn var_node(&mut self, kind: VariableDeclarationKind, items: &[String]) -> String {
        let kind = match kind {
            VariableDeclarationKind::Var => "var",
            VariableDeclarationKind::Let => "let",
            VariableDeclarationKind::Const => "const",
            VariableDeclarationKind::Using => "using",
            VariableDeclarationKind::AwaitUsing => "await-using",
        };
        self.node("Var", &[Self::lit_str(kind), bracket_join(items)])
    }

    fn declarator(&mut self, d: &VariableDeclarator<'_>) -> String {
        let id = self.binding(&d.id);
        let id = self.annotated(id, d.type_annotation.as_deref());
        let init = d.init.as_ref().map_or_else(String::new, |i| self.expr(i));
        self.node("Decl", &[id, init])
    }

    fn declaration(&mut self, d: &Declaration<'_>) -> String {
        match d {
            Declaration::VariableDeclaration(v) => self.variable_decl(v),
            Declaration::FunctionDeclaration(f) => self.function(f, false),
            Declaration::ClassDeclaration(c) => self.class(c, false),
            Declaration::TSTypeAliasDeclaration(t) => self.ts_type_alias(t),
            Declaration::TSInterfaceDeclaration(i) => self.ts_interface(i),
            Declaration::TSEnumDeclaration(e) => {
                let name = self.rename(e.id.name.as_str());
                self.node("TSEnum", &[Self::lit_str(&name)])
            }
            Declaration::TSModuleDeclaration(_) => self.node("TSModule", &[]),
            Declaration::TSGlobalDeclaration(_) => self.node("TSGlobal", &[]),
            Declaration::TSImportEqualsDeclaration(_) => self.node("TSImportEquals", &[]),
        }
    }

    fn function(&mut self, f: &Function<'_>, is_top: bool) -> String {
        let name = match (&f.id, is_top) {
            (Some(_), true) if self.locals.is_some() && !self.blanked => {
                self.blanked = true;
                "_fn".to_owned()
            }
            (Some(id), _) => self.rename(id.name.as_str()),
            (None, _) => "<anon>".to_owned(),
        };
        self.func_node(&name, f)
    }

    /// A class method as a callable of its own: the `function` it would be, named by its key. In
    /// xname mode the name blanks to `_fn` as a top-level `function`'s does, so a method and a free
    /// function written alike are one canonical — what the Python frontend gets by stripping the
    /// receiver.
    fn method_fn(&mut self, name: &str, f: &Function<'_>) -> String {
        let name = if self.locals.is_some() && !self.blanked {
            self.blanked = true;
            "_fn"
        } else {
            name
        };
        self.func_node(name, f)
    }

    fn func_node(&mut self, name: &str, f: &Function<'_>) -> String {
        // The signature before the body, so slots number in reading order; the type parameters
        // first of all, since the signature refers to them.
        let type_params = self.type_params(f.type_parameters.as_deref());
        let params = self.formal_params(&f.params);
        let ret = self.annotation(f.return_type.as_deref());
        let body = if self.use_site() {
            // A function written inside a use site is a function of its own: the use walk visits
            // its body, and the site keeps only that one is there.
            String::from("...")
        } else {
            f.body.as_deref().map_or_else(String::new, |b| self.block_body(&b.statements))
        };
        let async_g = format!("async={} gen={}", f.r#async, f.generator);
        let mut fields = vec![Self::lit_str(name), params, body, async_g];
        self.signature_tail(&mut fields, ret, type_params);
        self.node("Func", &fields)
    }

    /// The optional trailing fields a written or slotted signature carries — its return type and
    /// the type parameters it declares — each tagged, so that which one is present is unambiguous.
    fn signature_tail(&mut self, fields: &mut Vec<String>, ret: Option<String>, type_params: Option<String>) {
        if let Some(ret) = ret {
            fields.push(self.node("Ret", &[ret]));
        }
        fields.extend(type_params);
    }

    fn formal_params(&mut self, params: &FormalParameters<'_>) -> String {
        let mut items: Vec<String> = params.items.iter().map(|p| self.param(p)).collect();
        if let Some(rest) = &params.rest {
            // `FormalParameterRest` wraps a `BindingRestElement` as its `rest` field — that
            // inner element carries the binding pattern (`argument`).
            let inner = self.binding(&rest.rest.argument);
            items.push(self.node("Rest", &[inner]));
        }
        self.list("Params", items)
    }

    fn param(&mut self, p: &FormalParameter<'_>) -> String {
        let pat = self.binding(&p.pattern);
        let pat = self.annotated(pat, p.type_annotation.as_deref());
        let init = p.initializer.as_ref().map_or_else(String::new, |i| self.expr(i));
        self.node("Param", &[pat, init])
    }

    fn class(&mut self, c: &Class<'_>, is_top: bool) -> String {
        let name = match (&c.id, is_top) {
            (Some(_), true) if self.locals.is_some() && !self.blanked => {
                self.blanked = true;
                "_fn".to_owned()
            }
            (Some(id), _) => self.rename(id.name.as_str()),
            (None, _) => "<anon>".to_owned(),
        };
        let super_class =
            c.super_class.as_ref().map_or_else(String::new, |s| self.expr(s));
        let body: Vec<String> = c
            .body
            .body
            .iter()
            .map(|el| match el {
                ast::ClassElement::MethodDefinition(m) => {
                    let key = self.property_key(&m.key);
                    let value = self.function(&m.value, false);
                    let kind = match m.kind {
                        ast::MethodDefinitionKind::Method => "method",
                        ast::MethodDefinitionKind::Constructor => "ctor",
                        ast::MethodDefinitionKind::Get => "get",
                        ast::MethodDefinitionKind::Set => "set",
                    };
                    let static_s = if m.r#static { "static" } else { "" };
                    self.node("Method", &[key, Self::lit_str(kind), Self::lit_str(static_s), value])
                }
                ast::ClassElement::PropertyDefinition(p) => {
                    let key = self.property_key(&p.key);
                    let value = p.value.as_ref().map_or_else(String::new, |v| self.expr(v));
                    let static_s = if p.r#static { "static" } else { "" };
                    self.node("Prop", &[key, Self::lit_str(static_s), value])
                }
                ast::ClassElement::AccessorProperty(a) => {
                    let key = self.property_key(&a.key);
                    let value = a.value.as_ref().map_or_else(String::new, |v| self.expr(v));
                    self.node("Accessor", &[key, value])
                }
                ast::ClassElement::StaticBlock(s) => {
                    let body = self.block_body(&s.body);
                    self.node("StaticBlock", &[body])
                }
                ast::ClassElement::TSIndexSignature(_) => self.node("TSIndexSig", &[]),
            })
            .collect();
        self.node("Class", &[Self::lit_str(&name), super_class, bracket_join(&body)])
    }

    fn ts_type_alias(&mut self, t: &ast::TSTypeAliasDeclaration<'_>) -> String {
        let name = self.rename(t.id.name.as_str());
        let ty = self.ts_type(&t.type_annotation);
        self.node("TSTypeAlias", &[Self::lit_str(&name), ty])
    }

    fn ts_interface(&mut self, i: &ast::TSInterfaceDeclaration<'_>) -> String {
        let name = self.rename(i.id.name.as_str());
        let body: Vec<String> = i.body.body.iter().map(|s| self.ts_signature(s)).collect();
        self.node("TSInterface", &[Self::lit_str(&name), bracket_join(&body)])
    }

    fn ts_signature(&mut self, sig: &TSSignature<'_>) -> String {
        match sig {
            TSSignature::TSPropertySignature(p) => {
                let key = self.property_key(&p.key);
                let ty = p
                    .type_annotation
                    .as_ref()
                    .map_or_else(String::new, |a| self.ts_type(&a.type_annotation));
                self.node("TSProp", &[key, ty])
            }
            TSSignature::TSMethodSignature(m) => {
                let key = self.property_key(&m.key);
                self.node("TSMethod", &[key])
            }
            TSSignature::TSCallSignatureDeclaration(_) => self.node("TSCall", &[]),
            TSSignature::TSConstructSignatureDeclaration(_) => self.node("TSCtor", &[]),
            TSSignature::TSIndexSignature(_) => self.node("TSIndex", &[]),
        }
    }

    fn ts_type(&mut self, ty: &TSType<'_>) -> String {
        match ty {
            // The named types — `int` and `str` to the Python frontend, references like any other —
            // and the empty ones, which are values (`None`) and stay.
            TSType::TSStringKeyword(_) => self.type_ref("string"),
            TSType::TSNumberKeyword(_) => self.type_ref("number"),
            TSType::TSBooleanKeyword(_) => self.type_ref("boolean"),
            TSType::TSNullKeyword(_) => "null".to_owned(),
            TSType::TSUndefinedKeyword(_) => "undefined".to_owned(),
            TSType::TSAnyKeyword(_) => self.type_ref("any"),
            TSType::TSUnknownKeyword(_) => self.type_ref("unknown"),
            TSType::TSNeverKeyword(_) => self.type_ref("never"),
            TSType::TSVoidKeyword(_) => "void".to_owned(),
            TSType::TSObjectKeyword(_) => self.type_ref("object"),
            TSType::TSBigIntKeyword(_) => self.type_ref("bigint"),
            TSType::TSSymbolKeyword(_) => self.type_ref("symbol"),
            TSType::TSThisType(_) => "this".to_owned(),
            TSType::TSTypeReference(r) if self.types == Types::Dropped => self.ts_type_name(&r.type_name),
            // Written or slotted: the whole reference, generic arguments included — `Promise<User>`
            // and `Promise<Order>` are two signatures.
            TSType::TSTypeReference(r) => {
                let name = type_name_text(&r.type_name);
                let head = self.type_ref(&name);
                match r.type_arguments.as_deref().filter(|a| !a.params.is_empty()) {
                    Some(args) => {
                        let mut items = vec![head];
                        items.extend(args.params.iter().map(|t| self.ts_type(t)));
                        self.node("TApp", &items)
                    }
                    None => head,
                }
            }
            TSType::TSArrayType(a) => {
                let inner = self.ts_type(&a.element_type);
                self.node("TSArray", &[inner])
            }
            TSType::TSUnionType(u) => {
                let items: Vec<String> = u.types.iter().map(|t| self.ts_type(t)).collect();
                self.list("TSUnion", items)
            }
            TSType::TSIntersectionType(i) => {
                let items: Vec<String> = i.types.iter().map(|t| self.ts_type(t)).collect();
                self.list("TSInter", items)
            }
            TSType::TSLiteralType(l) => match &l.literal {
                TSLiteral::StringLiteral(s) => Self::lit_str(s.value.as_str()),
                TSLiteral::NumericLiteral(n) => Self::lit_num(n.value),
                TSLiteral::BooleanLiteral(b) => if b.value { "true" } else { "false" }.to_owned(),
                _ => self.node("TSLit", &[]),
            },
            other => self.node(&format!("TS_{:?}", std::mem::discriminant(other)), &[]),
        }
    }

    fn ts_type_name(&mut self, name: &TSTypeName<'_>) -> String {
        match name {
            TSTypeName::IdentifierReference(id) => self.rename(id.name.as_str()),
            TSTypeName::QualifiedName(q) => {
                let left = self.ts_type_name(&q.left);
                format!("{left}.{}", q.right.name.as_str())
            }
            TSTypeName::ThisExpression(_) => "this".to_owned(),
        }
    }

    fn binding(&mut self, pat: &BindingPattern<'_>) -> String {
        match pat {
            BindingPattern::BindingIdentifier(id) => {
                // Arrow / fn-expr `const`: blank the top declarator name to `_fn` once (the named
                // `function`/`class` path does this via `is_top`), so two differently-named
                // arrow-consts alpha-equate.
                if self.locals.is_some()
                    && !self.blanked
                    && self.blank.as_deref() == Some(id.name.as_str())
                {
                    self.blanked = true;
                    return self.node("Bind", &[Self::lit_str("_fn")]);
                }
                let n = self.rename(id.name.as_str());
                self.node("Bind", &[Self::lit_str(&n)])
            }
            BindingPattern::ArrayPattern(arr) => {
                let mut items: Vec<String> = Vec::with_capacity(arr.elements.len());
                for elt in &arr.elements {
                    items.push(match elt {
                        Some(p) => self.binding(p),
                        None => "<hole>".to_owned(),
                    });
                }
                if let Some(rest) = &arr.rest {
                    let inner = self.binding(&rest.argument);
                    items.push(self.node("Rest", &[inner]));
                }
                self.list("BindArr", items)
            }
            BindingPattern::ObjectPattern(obj) => {
                let mut items: Vec<String> = obj
                    .properties
                    .iter()
                    .map(|p| {
                        let k = self.property_key(&p.key);
                        let v = self.binding(&p.value);
                        self.node("BindProp", &[k, v])
                    })
                    .collect();
                if let Some(rest) = &obj.rest {
                    let inner = self.binding(&rest.argument);
                    items.push(self.node("Rest", &[inner]));
                }
                self.list("BindObj", items)
            }
            BindingPattern::AssignmentPattern(asgn) => {
                let left = self.binding(&asgn.left);
                let right = self.expr(&asgn.right);
                self.node("BindDefault", &[left, right])
            }
        }
    }


    // ────── expressions ──────

    /// An expression — and, when units are being collected, kept as one if it is a proper part of
    /// its line: the statement's own expression is the line, already a unit, and counting it again
    /// would double the weight of exactly the text an edit changes. A wrapper (parentheses, a type
    /// assertion) is not a level of its own; what it wraps is the part.
    fn expr(&mut self, expr: &Expression<'_>) -> String {
        if is_wrapper(expr) {
            return self.expr_inner(expr);
        }
        self.expr_depth += 1;
        let out = self.expr_inner(expr);
        self.expr_depth -= 1;
        if self.expr_depth > 0 && is_unit(expr) {
            if let Some(units) = self.units.as_mut() {
                units.push((self.unit_line, out.clone()));
            }
        }
        out
    }

    fn expr_inner(&mut self, expr: &Expression<'_>) -> String {
        match expr {
            Expression::Identifier(id) => {
                let n = self.reference(id.name.as_str());
                self.node("Id", &[Self::lit_str(&n)])
            }
            Expression::ThisExpression(_) => self.node("This", &[]),
            Expression::Super(_) => self.node("Super", &[]),
            Expression::BooleanLiteral(b) => {
                self.node("Bool", &[(if b.value { "true" } else { "false" }).to_owned()])
            }
            Expression::NullLiteral(_) => self.node("Null", &[]),
            Expression::NumericLiteral(n) => self.node("Num", &[Self::lit_num(n.value)]),
            Expression::BigIntLiteral(b) => {
                self.node("BigInt", &[Self::lit_str(b.raw.unwrap_or_else(|| "".into()).as_str())])
            }
            Expression::StringLiteral(s) => self.node("Str", &[Self::lit_str(s.value.as_str())]),
            Expression::RegExpLiteral(r) => self.node("Regex", &[Self::lit_str(r.regex.pattern.text.as_str())]),
            Expression::TemplateLiteral(t) => self.template_literal(t),
            Expression::ArrayExpression(arr) => {
                let items: Vec<String> = arr
                    .elements
                    .iter()
                    .map(|e| match e {
                        ArrayExpressionElement::Elision(_) => "<hole>".to_owned(),
                        ArrayExpressionElement::SpreadElement(s) => {
                            let inner = self.expr(&s.argument);
                            self.node("Spread", &[inner])
                        }
                        other => match other.as_expression() {
                            Some(e) => self.expr(e),
                            None => self.node("Unknown_ArrayElt", &[]),
                        },
                    })
                    .collect();
                self.list("Arr", items)
            }
            Expression::ObjectExpression(o) => self.object_expression(o),
            Expression::BinaryExpression(b) => {
                let l = self.expr(&b.left);
                let op = format!("{:?}", b.operator);
                let r = self.expr(&b.right);
                self.node("Bin", &[l, Self::lit_str(&op), r])
            }
            Expression::LogicalExpression(l) => {
                let lhs = self.expr(&l.left);
                let op = format!("{:?}", l.operator);
                let rhs = self.expr(&l.right);
                self.node("Logic", &[lhs, Self::lit_str(&op), rhs])
            }
            Expression::UnaryExpression(u) => {
                let op = format!("{:?}", u.operator);
                let arg = self.expr(&u.argument);
                self.node("Unary", &[Self::lit_str(&op), arg])
            }
            Expression::UpdateExpression(u) => {
                let op = format!("{:?}", u.operator);
                let target_e = self.simple_assignment_target(&u.argument);
                let prefix = if u.prefix { "pre" } else { "post" };
                self.node("Update", &[Self::lit_str(&op), Self::lit_str(prefix), target_e])
            }
            Expression::AssignmentExpression(a) => {
                let op = format!("{:?}", a.operator);
                let lhs = self.assignment_target(&a.left);
                let rhs = self.expr(&a.right);
                self.node("Assign", &[Self::lit_str(&op), lhs, rhs])
            }
            Expression::ConditionalExpression(c) => {
                let test = self.expr(&c.test);
                let cons = self.expr(&c.consequent);
                let alt = self.expr(&c.alternate);
                self.node("Cond", &[test, cons, alt])
            }
            Expression::CallExpression(c) => {
                if let Expression::StaticMemberExpression(m) = &c.callee {
                    if self.known.is_some() {
                        self.calls.insert(format!(".{}", m.property.name));
                    }
                }
                let on_anchor = self.anchored(&c.callee);
                let callee = self.expr(&c.callee);
                let args: Vec<String> = c.arguments.iter().map(|a| self.argument_on(a, on_anchor)).collect();
                let optional = if c.optional { "opt" } else { "" };
                self.node("Call", &[callee, Self::lit_str(optional), bracket_join(&args)])
            }
            Expression::NewExpression(n) => {
                let on_anchor = self.anchored(&n.callee);
                let callee = self.expr(&n.callee);
                let args: Vec<String> = n.arguments.iter().map(|a| self.argument_on(a, on_anchor)).collect();
                self.node("New", &[callee, bracket_join(&args)])
            }
            Expression::StaticMemberExpression(m) => {
                let on_anchor = self.anchored(&m.object);
                let obj = self.expr(&m.object);
                let prop = Self::surface(m.property.name.as_str(), on_anchor);
                let optional = if m.optional { "opt" } else { "" };
                self.node("Member", &[obj, Self::lit_str(&prop), Self::lit_str(optional)])
            }
            Expression::ComputedMemberExpression(m) => {
                let obj = self.expr(&m.object);
                let prop = self.expr(&m.expression);
                let optional = if m.optional { "opt" } else { "" };
                self.node("CMember", &[obj, prop, Self::lit_str(optional)])
            }
            Expression::PrivateFieldExpression(m) => {
                let obj = self.expr(&m.object);
                let prop = format!("#{}", m.field.name.as_str());
                let optional = if m.optional { "opt" } else { "" };
                self.node("PrivateMember", &[obj, Self::lit_str(&prop), Self::lit_str(optional)])
            }
            Expression::ChainExpression(c) => {
                let inner = match &c.expression {
                    ChainElement::CallExpression(call) => {
                        let callee = self.expr(&call.callee);
                        let args: Vec<String> =
                            call.arguments.iter().map(|a| self.argument(a)).collect();
                        self.node("Call", &[callee, Self::lit_str("opt"), bracket_join(&args)])
                    }
                    ChainElement::TSNonNullExpression(e) => {
                        let inner = self.expr(&e.expression);
                        self.node("TSNonNull", &[inner])
                    }
                    other => match other.as_member_expression() {
                        Some(me) => self.member_expression(me),
                        None => self.node("Unknown_Chain", &[]),
                    },
                };
                self.node("Chain", &[inner])
            }
            Expression::SequenceExpression(s) => {
                let items: Vec<String> = s.expressions.iter().map(|e| self.expr(e)).collect();
                self.list("Seq", items)
            }
            Expression::ArrowFunctionExpression(a) => {
                let type_params = self.type_params(a.type_parameters.as_deref());
                let params = self.formal_params(&a.params);
                let ret = self.annotation(a.return_type.as_deref());
                // A callback with a block is a function of its own to a use site — see `func_node`;
                // an expression body is what a Python lambda is, and stays.
                let body =
                    if self.use_site() && !a.expression { String::from("...") } else { self.block_body(&a.body.statements) };
                let async_g = format!("async={} gen=false expr={}", a.r#async, a.expression);
                let mut fields = vec![params, body, async_g];
                self.signature_tail(&mut fields, ret, type_params);
                self.node("Arrow", &fields)
            }
            Expression::FunctionExpression(f) => self.function(f, false),
            Expression::ClassExpression(c) => self.class(c, false),
            Expression::YieldExpression(y) => {
                let arg = y.argument.as_ref().map_or_else(String::new, |a| self.expr(a));
                self.node("Yield", &[arg, (if y.delegate { "*" } else { "" }).to_owned()])
            }
            Expression::AwaitExpression(a) => {
                let arg = self.expr(&a.argument);
                self.node("Await", &[arg])
            }
            Expression::ParenthesizedExpression(p) => self.expr(&p.expression),
            Expression::ImportExpression(i) => {
                let source = self.expr(&i.source);
                self.node("ImportExpr", &[source])
            }
            Expression::MetaProperty(m) => self.node(
                "Meta",
                &[Self::lit_str(m.meta.name.as_str()), Self::lit_str(m.property.name.as_str())],
            ),
            Expression::TaggedTemplateExpression(t) => {
                let tag = self.expr(&t.tag);
                let q = self.template_literal(&t.quasi);
                self.node("TaggedTpl", &[tag, q])
            }
            Expression::PrivateInExpression(p) => {
                let lhs = format!("#{}", p.left.name.as_str());
                let rhs = self.expr(&p.right);
                self.node("PrivateIn", &[Self::lit_str(&lhs), rhs])
            }
            // TS type wrappers — keep inner expr, summarize attached type.
            Expression::TSAsExpression(e) => {
                let inner = self.expr(&e.expression);
                let ty = self.ts_type(&e.type_annotation);
                self.node("TSAs", &[inner, ty])
            }
            Expression::TSSatisfiesExpression(e) => {
                let inner = self.expr(&e.expression);
                let ty = self.ts_type(&e.type_annotation);
                self.node("TSSat", &[inner, ty])
            }
            Expression::TSNonNullExpression(e) => {
                let inner = self.expr(&e.expression);
                self.node("TSNonNull", &[inner])
            }
            Expression::TSTypeAssertion(e) => {
                let inner = self.expr(&e.expression);
                let ty = self.ts_type(&e.type_annotation);
                self.node("TSAssert", &[inner, ty])
            }
            Expression::TSInstantiationExpression(e) => self.expr(&e.expression),
            Expression::JSXElement(j) => self.jsx_element(j),
            Expression::JSXFragment(f) => self.jsx_fragment(f),
            Expression::V8IntrinsicExpression(_) => self.node("V8Intrinsic", &[]),
        }
    }

    // ────── JSX ──────

    /// An element as what it renders: its tag, its attributes in source order and its children.
    ///
    /// 🔴 Not the tag alone. Rendered as `JSX('html')`, every layout returning `<html>` was one
    /// canonical — a thirty-line one with scripts and a theme, and a three-line one around
    /// `{children}` — and on a real React tree the cross-name pass made one cluster of seven hundred.
    /// Markup is most of what a component says; a canonical that drops it compares signatures.
    fn jsx_element(&mut self, j: &ast::JSXElement<'_>) -> String {
        let on_anchor = matches!(
            &j.opening_element.name,
            ast::JSXElementName::IdentifierReference(id) if self.anchor == Some(id.name.as_str())
        );
        let name = self.jsx_element_name(&j.opening_element.name);
        let mut attrs = Vec::with_capacity(j.opening_element.attributes.len());
        for item in &j.opening_element.attributes {
            attrs.push(self.jsx_attribute(item, on_anchor));
        }
        let children = self.jsx_children(&j.children);
        self.node("JSX", &[Self::lit_str(&name), bracket_join(&attrs), children])
    }

    fn jsx_fragment(&mut self, f: &ast::JSXFragment<'_>) -> String {
        let children = self.jsx_children(&f.children);
        self.node("JSXFragment", &[children])
    }

    /// A tag. An intrinsic one (`div`) is the markup's own vocabulary and stays as written; a
    /// component (`Card`, `ui.Card`) is a name like any other, so a local one is renamed.
    fn jsx_element_name(&mut self, name: &ast::JSXElementName<'_>) -> String {
        match name {
            ast::JSXElementName::Identifier(id) => id.name.to_string(),
            ast::JSXElementName::IdentifierReference(id) => self.reference(id.name.as_str()),
            ast::JSXElementName::NamespacedName(n) => format!("{}:{}", n.namespace.name, n.name.name),
            ast::JSXElementName::MemberExpression(m) => self.jsx_member(m),
            ast::JSXElementName::ThisExpression(_) => "this".to_owned(),
        }
    }

    fn jsx_member(&mut self, m: &ast::JSXMemberExpression<'_>) -> String {
        let object = match &m.object {
            ast::JSXMemberExpressionObject::IdentifierReference(id) => self.reference(id.name.as_str()),
            ast::JSXMemberExpressionObject::MemberExpression(inner) => self.jsx_member(inner),
            ast::JSXMemberExpressionObject::ThisExpression(_) => "this".to_owned(),
        };
        format!("{object}.{}", m.property.name)
    }

    /// An attribute; `on_anchor` when the element is the anchor, whose props are its declared
    /// surface.
    fn jsx_attribute(&mut self, item: &ast::JSXAttributeItem<'_>, on_anchor: bool) -> String {
        match item {
            ast::JSXAttributeItem::Attribute(a) => {
                let name = match &a.name {
                    ast::JSXAttributeName::Identifier(id) => Self::surface(id.name.as_str(), on_anchor),
                    ast::JSXAttributeName::NamespacedName(n) => format!("{}:{}", n.namespace.name, n.name.name),
                };
                let value = match &a.value {
                    None => String::new(),
                    Some(ast::JSXAttributeValue::StringLiteral(s)) => self.node("Str", &[Self::lit_str(s.value.as_str())]),
                    Some(ast::JSXAttributeValue::ExpressionContainer(c)) => {
                        c.expression.as_expression().map_or_else(String::new, |e| self.expr(e))
                    }
                    Some(ast::JSXAttributeValue::Element(e)) => self.jsx_element(e),
                    Some(ast::JSXAttributeValue::Fragment(f)) => self.jsx_fragment(f),
                };
                self.node("Attr", &[Self::lit_str(&name), value])
            }
            ast::JSXAttributeItem::SpreadAttribute(s) => {
                let argument = self.expr(&s.argument);
                self.node("SpreadAttr", &[argument])
            }
        }
    }

    /// Children in order. Text is read the way JSX reads it — runs of whitespace collapse, and a run
    /// that is only whitespace (the indentation between two tags) is not a child; an empty `{}` (a
    /// comment, usually) says nothing either.
    fn jsx_children(&mut self, children: &[ast::JSXChild<'_>]) -> String {
        let mut items = Vec::with_capacity(children.len());
        for child in children {
            match child {
                ast::JSXChild::Text(t) => {
                    let text = t.value.split_whitespace().collect::<Vec<_>>().join(" ");
                    if !text.is_empty() {
                        items.push(self.node("Text", &[Self::lit_str(&text)]));
                    }
                }
                ast::JSXChild::Element(e) => items.push(self.jsx_element(e)),
                ast::JSXChild::Fragment(f) => items.push(self.jsx_fragment(f)),
                ast::JSXChild::ExpressionContainer(c) => {
                    if let Some(e) = c.expression.as_expression() {
                        items.push(self.expr(e));
                    }
                }
                ast::JSXChild::Spread(s) => {
                    let e = self.expr(&s.expression);
                    items.push(self.node("SpreadChild", &[e]));
                }
            }
        }
        bracket_join(&items)
    }

    fn member_expression(&mut self, m: &MemberExpression<'_>) -> String {
        match m {
            MemberExpression::StaticMemberExpression(s) => {
                let obj = self.expr(&s.object);
                let prop = s.property.name.as_str().to_owned();
                self.node("Member", &[obj, Self::lit_str(&prop), Self::lit_str("opt")])
            }
            MemberExpression::ComputedMemberExpression(c) => {
                let obj = self.expr(&c.object);
                let prop = self.expr(&c.expression);
                self.node("CMember", &[obj, prop, Self::lit_str("opt")])
            }
            MemberExpression::PrivateFieldExpression(p) => {
                let obj = self.expr(&p.object);
                let prop = format!("#{}", p.field.name.as_str());
                self.node("PrivateMember", &[obj, Self::lit_str(&prop), Self::lit_str("opt")])
            }
        }
    }

    fn template_literal(&mut self, t: &TemplateLiteral<'_>) -> String {
        // Quasis flatten to their `cooked` string; expressions are emitted between them. Result
        // is one logical string interleaving the literal chunks with their interpolations.
        let mut parts: Vec<String> = Vec::with_capacity(t.quasis.len() + t.expressions.len());
        for (i, q) in t.quasis.iter().enumerate() {
            let text = q.value.cooked.as_ref().map_or_else(|| q.value.raw.as_str(), |c| c.as_str());
            parts.push(Self::lit_str(text));
            if let Some(e) = t.expressions.get(i) {
                parts.push(self.expr(e));
            }
        }
        self.list("Tpl", parts)
    }

    fn object_expression(&mut self, o: &ObjectExpression<'_>) -> String {
        self.object_expression_on(o, false)
    }

    /// An object literal; `on_anchor` when it is what the anchor is called with, whose keys then
    /// name the anchor's declared surface — what a keyword argument is to the Python frontend.
    fn object_expression_on(&mut self, o: &ObjectExpression<'_>, on_anchor: bool) -> String {
        let items: Vec<String> = o
            .properties
            .iter()
            .map(|p| match p {
                ObjectPropertyKind::ObjectProperty(p) => {
                    let k = match &p.key {
                        PropertyKey::StaticIdentifier(id) if on_anchor => Self::lit_str(&Self::surface(id.name.as_str(), true)),
                        key => self.property_key(key),
                    };
                    let v = self.expr(&p.value);
                    let kind = match p.kind {
                        ast::PropertyKind::Init => "init",
                        ast::PropertyKind::Get => "get",
                        ast::PropertyKind::Set => "set",
                    };
                    self.node("ObjProp", &[k, v, Self::lit_str(kind)])
                }
                ObjectPropertyKind::SpreadProperty(s) => {
                    let inner = self.expr(&s.argument);
                    self.node("Spread", &[inner])
                }
            })
            .collect();
        self.list("Obj", items)
    }

    fn property_key(&mut self, key: &PropertyKey<'_>) -> String {
        match key {
            PropertyKey::StaticIdentifier(id) => Self::lit_str(id.name.as_str()),
            PropertyKey::PrivateIdentifier(id) => Self::lit_str(&format!("#{}", id.name)),
            other => match other.as_expression() {
                Some(e) => {
                    let v = self.expr(e);
                    self.node("ComputedKey", &[v])
                }
                None => self.node("Unknown_PropertyKey", &[]),
            },
        }
    }

    /// A call's argument; an object literal handed to the anchor itself has its keys marked.
    fn argument_on(&mut self, a: &Argument<'_>, on_anchor: bool) -> String {
        match a {
            Argument::ObjectExpression(o) if on_anchor => self.object_expression_on(o, true),
            other => self.argument(other),
        }
    }

    fn argument(&mut self, a: &Argument<'_>) -> String {
        if let Argument::SpreadElement(s) = a {
            let inner = self.expr(&s.argument);
            return self.node("Spread", &[inner]);
        }
        match a.as_expression() {
            Some(e) => self.expr(e),
            None => self.node("Unknown_Argument", &[]),
        }
    }

    fn assignment_target(&mut self, t: &AssignmentTarget<'_>) -> String {
        if let Some(simple) = t.as_simple_assignment_target() {
            return self.simple_assignment_target(simple);
        }
        if let Some(p) = t.as_assignment_target_pattern() {
            return self.assignment_pattern(p);
        }
        self.node("Unknown_AssignTarget", &[])
    }

    /// `SimpleAssignmentTarget` has four TS-only variants (TSAs / TSSatisfies / TSNonNull /
    /// TSTypeAssertion) at its own low discriminants — these do NOT line up with `Expression`'s
    /// discriminants for the same variant names, so we cannot cast. We emit them inline, then
    /// fall through to `as_member_expression()` (safely-cast inherited region) for the rest.
    fn simple_assignment_target(&mut self, t: &SimpleAssignmentTarget<'_>) -> String {
        match t {
            SimpleAssignmentTarget::AssignmentTargetIdentifier(id) => {
                let n = self.rename(id.name.as_str());
                self.node("Id", &[Self::lit_str(&n)])
            }
            SimpleAssignmentTarget::TSAsExpression(e) => {
                let inner = self.expr(&e.expression);
                let ty = self.ts_type(&e.type_annotation);
                self.node("TSAs", &[inner, ty])
            }
            SimpleAssignmentTarget::TSSatisfiesExpression(e) => {
                let inner = self.expr(&e.expression);
                let ty = self.ts_type(&e.type_annotation);
                self.node("TSSat", &[inner, ty])
            }
            SimpleAssignmentTarget::TSNonNullExpression(e) => {
                let inner = self.expr(&e.expression);
                self.node("TSNonNull", &[inner])
            }
            SimpleAssignmentTarget::TSTypeAssertion(e) => {
                let inner = self.expr(&e.expression);
                let ty = self.ts_type(&e.type_annotation);
                self.node("TSAssert", &[inner, ty])
            }
            other => match other.as_member_expression() {
                Some(me) => self.member_expression(me),
                None => self.node("Unknown_SimpleAssignTarget", &[]),
            },
        }
    }

    fn assignment_pattern(&mut self, p: &AssignmentTargetPattern<'_>) -> String {
        match p {
            AssignmentTargetPattern::ArrayAssignmentTarget(arr) => {
                let mut items: Vec<String> = arr
                    .elements
                    .iter()
                    .map(|elt| match elt {
                        Some(e) => self.assignment_maybe_default(e),
                        None => "<hole>".to_owned(),
                    })
                    .collect();
                if let Some(rest) = &arr.rest {
                    let inner = self.assignment_target(&rest.target);
                    items.push(self.node("Rest", &[inner]));
                }
                self.list("ArrTarget", items)
            }
            AssignmentTargetPattern::ObjectAssignmentTarget(obj) => {
                let mut items: Vec<String> = obj
                    .properties
                    .iter()
                    .map(|p| self.assignment_target_property(p))
                    .collect();
                if let Some(rest) = &obj.rest {
                    let inner = self.assignment_target(&rest.target);
                    items.push(self.node("Rest", &[inner]));
                }
                self.list("ObjTarget", items)
            }
        }
    }

    fn assignment_maybe_default(&mut self, m: &AssignmentTargetMaybeDefault<'_>) -> String {
        match m {
            AssignmentTargetMaybeDefault::AssignmentTargetWithDefault(d) => {
                let lhs = self.assignment_target(&d.binding);
                let rhs = self.expr(&d.init);
                self.node("Default", &[lhs, rhs])
            }
            other => match other.as_assignment_target() {
                Some(t) => self.assignment_target(t),
                None => self.node("Unknown_AssignMaybeDefault", &[]),
            },
        }
    }

    fn assignment_target_property(&mut self, p: &AssignmentTargetProperty<'_>) -> String {
        match p {
            AssignmentTargetProperty::AssignmentTargetPropertyIdentifier(i) => {
                let name = self.rename(i.binding.name.as_str());
                let init = i.init.as_ref().map_or_else(String::new, |e| self.expr(e));
                self.node("TargetProp", &[Self::lit_str(&name), init])
            }
            AssignmentTargetProperty::AssignmentTargetPropertyProperty(p) => {
                let key = self.property_key(&p.name);
                let val = self.assignment_maybe_default(&p.binding);
                self.node("TargetProp", &[key, val])
            }
        }
    }
}

/// A type name as written, qualified with `.`.
fn type_name_text(name: &TSTypeName<'_>) -> String {
    match name {
        TSTypeName::IdentifierReference(id) => id.name.to_string(),
        TSTypeName::QualifiedName(q) => format!("{}.{}", type_name_text(&q.left), q.right.name),
        TSTypeName::ThisExpression(_) => "this".to_owned(),
    }
}

/// Whether an expression is a Type-3 unit of its own: a node with structure — a call, a member
/// chain, an operator application, a function, markup, a non-empty display — not a bare name or a
/// literal, which carry no agreement by themselves. The same rule as the Python frontend's.
fn is_unit(expr: &Expression<'_>) -> bool {
    match expr {
        Expression::CallExpression(_)
        | Expression::NewExpression(_)
        | Expression::StaticMemberExpression(_)
        | Expression::ComputedMemberExpression(_)
        | Expression::PrivateFieldExpression(_)
        | Expression::ChainExpression(_)
        | Expression::BinaryExpression(_)
        | Expression::LogicalExpression(_)
        | Expression::UnaryExpression(_)
        | Expression::UpdateExpression(_)
        | Expression::AssignmentExpression(_)
        | Expression::ConditionalExpression(_)
        | Expression::SequenceExpression(_)
        | Expression::ArrowFunctionExpression(_)
        | Expression::FunctionExpression(_)
        | Expression::ClassExpression(_)
        | Expression::YieldExpression(_)
        | Expression::AwaitExpression(_)
        | Expression::ImportExpression(_)
        | Expression::TaggedTemplateExpression(_)
        | Expression::PrivateInExpression(_)
        | Expression::JSXElement(_) => true,
        Expression::TemplateLiteral(t) => !t.expressions.is_empty(),
        Expression::ArrayExpression(a) => !a.elements.is_empty(),
        Expression::ObjectExpression(o) => !o.properties.is_empty(),
        Expression::JSXFragment(f) => !f.children.is_empty(),
        _ => false,
    }
}

/// Syntax around an expression that is not a part of its own: what it wraps is the part.
fn is_wrapper(expr: &Expression<'_>) -> bool {
    matches!(
        expr,
        Expression::ParenthesizedExpression(_)
            | Expression::TSAsExpression(_)
            | Expression::TSSatisfiesExpression(_)
            | Expression::TSNonNullExpression(_)
            | Expression::TSTypeAssertion(_)
            | Expression::TSInstantiationExpression(_)
    )
}

/// Free helper — bracket-joined list, no node-count side-effects (used to wrap collected items
/// before handing them to a parent `Dump::node` call without double-borrowing `self`).
fn bracket_join(items: &[String]) -> String {
    format!("[{}]", items.join(", "))
}

// ───────────────────────────── public driver ─────────────────────────────

/// Parse a single TS source. Always uses TypeScript mode (`.ts`); the caller's file extension
/// is irrelevant since `text` is already the def's source slice. Returns the parsed program and
/// the allocator (kept alive by the caller) — both needed because the AST is arena-allocated.
fn parse<'a>(allocator: &'a Allocator, text: &'a str) -> Option<oxc_ast::ast::Program<'a>> {
    let st = SourceType::ts();
    let ret = Parser::new(allocator, text, st).parse();
    if ret.panicked {
        return None;
    }
    Some(ret.program)
}

/// Find the leading statement of a parsed program that is a callable / class / type-alias / etc.
/// Unwraps `export` / `export default` wrappers so the inner declaration is what we canonicalize.
fn leading_def_stmt<'a>(prog: &'a oxc_ast::ast::Program<'a>) -> Option<&'a Statement<'a>> {
    prog.body.first()
}

#[must_use]
pub fn ast_canonical(text: &str) -> String {
    let allocator = Allocator::default();
    let Some(prog) = parse(&allocator, text) else {
        return text.to_owned();
    };
    let Some(stmt) = leading_def_stmt(&prog) else {
        return text.to_owned();
    };
    let mut d = Dump::new(None);
    cluster_stmt(&mut d, stmt)
}

/// Cluster canonical of one top-level statement — for top-level callable / class declarations
/// we strip the top def's "outer" wrappers (export, export default) so a `class Foo {}` and
/// `export class Foo {}` produce the same canonical. The bookkeeping for `top_def_pending`
/// (decorator-stripping in py-canon) is handled at the parser level here: the text we receive
/// already excludes top-of-def decorators, so we just emit straight.
fn cluster_stmt(d: &mut Dump<'_>, stmt: &Statement<'_>) -> String {
    match stmt {
        Statement::ExportNamedDeclaration(e) => match e.declaration.as_ref() {
            Some(decl) => d.declaration(decl),
            None => d.stmt(stmt),
        },
        Statement::ExportDefaultDeclaration(e) => match &e.declaration {
            ExportDefaultDeclarationKind::FunctionDeclaration(f) => d.function(f, true),
            ExportDefaultDeclarationKind::ClassDeclaration(c) => d.class(c, true),
            _ => d.stmt(stmt),
        },
        Statement::FunctionDeclaration(f) => d.function(f, true),
        Statement::ClassDeclaration(c) => d.class(c, true),
        _ => d.stmt(stmt),
    }
}

#[must_use]
pub fn ast_canonical_many(texts: &[String]) -> Vec<String> {
    texts.par_iter().map(|t| ast_canonical(t)).collect()
}

/// A callable as the scan found it, in the file's own tree.
///
/// 🔴 The canonical is rendered off this node, never off the definition's source slice parsed
/// again. A slice is not a program: a method's (`load(x) { … }`) parses as nothing, and a `.tsx`
/// body re-parsed as `.ts` reads its JSX as type assertions and fails — and a failed parse used to
/// leave the definition with its raw text for a canonical and no analysis at all. That was every
/// class method, and four in five functions of a React tree: invisible to every pass but the
/// name-gated one, which compared them as text.
pub(crate) enum CallableNode<'a, 'b> {
    /// `function f(…) { … }` — the name is on the node.
    Function(&'a Function<'b>),
    /// `const f = (…) => { … }` / `const f = function (…) { … }` — the name is on the declarator.
    Bound(VariableDeclarationKind, &'a VariableDeclarator<'b>),
    /// A class method — the name is its key, the callable its value.
    Method(&'a str, &'a Function<'b>),
}

/// What a callable is made of, whatever syntax spelled it.
pub(crate) struct Parts<'a, 'b> {
    params: &'a FormalParameters<'b>,
    body: &'a [Statement<'b>],
    is_async: bool,
    ret: Option<&'a ast::TSTypeAnnotation<'b>>,
    type_params: Option<&'a ast::TSTypeParameterDeclaration<'b>>,
}

impl<'a, 'b> Parts<'a, 'b> {
    fn of_function(f: &'a Function<'b>) -> Option<Self> {
        let body = f.body.as_deref()?;
        Some(Parts {
            params: &f.params,
            body: &body.statements,
            is_async: f.r#async,
            ret: f.return_type.as_deref(),
            type_params: f.type_parameters.as_deref(),
        })
    }
}

impl<'a, 'b> CallableNode<'a, 'b> {
    /// What the callable is made of — `None` for one with no body to analyze (an overload, an
    /// ambient declaration) or a declarator bound to something that is not a function.
    fn parts(&self) -> Option<Parts<'a, 'b>> {
        match self {
            Self::Function(f) | Self::Method(_, f) => Parts::of_function(f),
            Self::Bound(_, declarator) => match declarator.init.as_ref()? {
                Expression::ArrowFunctionExpression(a) => Some(Parts {
                    params: &a.params,
                    body: &a.body.statements,
                    is_async: a.r#async,
                    ret: a.return_type.as_deref(),
                    type_params: a.type_parameters.as_deref(),
                }),
                Expression::FunctionExpression(f) => Parts::of_function(f),
                _ => None,
            },
        }
    }

    /// A copy of the callable in `allocator`, its body rewritten to the normal form (`normal.rs`) —
    /// `inline` admitting the single-use-temporary rule.
    fn normalized<'x>(&self, allocator: &'x Allocator, inline: bool) -> Option<CallableNode<'x, 'x>> {
        let ast = AstBuilder::new(allocator);
        Some(match self {
            Self::Function(f) => CallableNode::Function(normalized_function(ast, f, inline)?),
            Self::Method(name, f) => CallableNode::Method(allocator.alloc_str(name), normalized_function(ast, f, inline)?),
            Self::Bound(kind, declarator) => {
                let copy = allocator.alloc(declarator.clone_in(allocator));
                match copy.init.as_mut()? {
                    Expression::ArrowFunctionExpression(a) => {
                        let a = &mut **a;
                        crate::normal::normalize(ast, &a.params, &mut a.body.statements, inline);
                    }
                    Expression::FunctionExpression(f) => {
                        let f = &mut **f;
                        let body = f.body.as_mut()?;
                        crate::normal::normalize(ast, &f.params, &mut body.statements, inline);
                    }
                    _ => return None,
                }
                CallableNode::Bound(*kind, copy)
            }
        })
    }

    /// The node rendered whole. In xname mode the definition's own name blanks to `_fn`.
    fn render(&self, d: &mut Dump<'_>) -> String {
        match self {
            Self::Function(f) => d.function(f, true),
            Self::Bound(kind, declarator) => {
                // The name rides on the declarator rather than on a `Func` node, so it is named for
                // the blank here; a `function`'s is blanked through `is_top`.
                if d.locals.is_some() {
                    if let BindingPattern::BindingIdentifier(id) = &declarator.id {
                        d.blank = Some(id.name.to_string());
                    }
                }
                let item = d.declarator(declarator);
                d.var_node(*kind, &[item])
            }
            Self::Method(name, f) => d.method_fn(name, f),
        }
    }
}

fn normalized_function<'x>(ast: AstBuilder<'x>, f: &Function<'_>, inline: bool) -> Option<&'x Function<'x>> {
    let copy = ast.allocator.alloc(f.clone_in(ast.allocator));
    let body = copy.body.as_mut()?;
    crate::normal::normalize(ast, &copy.params, &mut body.statements, inline);
    Some(copy)
}

/// A shape key for the normal form's orientation rules: the expression rendered with the
/// definition's own names renamed from scratch, so two renamed copies of it have one key.
pub(crate) fn shape_key(expr: &Expression<'_>, locals: &HashSet<String>) -> String {
    Dump::new(Some(locals)).expr(expr)
}

/// The names-preserved cluster canonical of a callable — what the name-gated pass compares.
pub(crate) fn callable_canonical(node: &CallableNode<'_, '_>) -> String {
    node.render(&mut Dump::new(None))
}

/// The cluster canonical of a class, off its node.
pub(crate) fn class_canonical(class: &Class<'_>) -> String {
    Dump::new(None).class(class, true)
}

/// The cluster canonical of an interface, off its node.
pub(crate) fn interface_canonical(interface: &ast::TSInterfaceDeclaration<'_>) -> String {
    Dump::new(None).ts_interface(interface)
}

/// Full callable analysis — the cluster and xname canonicals, the Type-3 lines, the statement stream
/// and the node count — or `None` for a callable with no body.
pub(crate) fn analyze_node(node: &CallableNode<'_, '_>, facets: bool) -> Option<AnalyzedFn> {
    let parts = node.parts()?;
    let cluster_canonical = callable_canonical(node);
    let locals_of = |parts: &Parts<'_, '_>| {
        let mut collect = Collect::default();
        collect.add_params(parts.params);
        for s in parts.body {
            collect.collect_stmt(s);
        }
        collect.bound
    };

    // The statement stream keeps the body as written: converge reads it to say where two definitions
    // stop agreeing, and a rewritten line would point the reader at code that is not there. A run
    // that never reads facets skips the whole walk.
    let statements = if facets { statement_stream(&parts, &locals_of(&parts)) } else { Vec::new() };

    // The name-agnostic canonicals are taken from the normal form (`normal.rs`): what the exact and
    // Type-3 passes compare is the meaning the rules reach, not the spelling. The Type-3 lines come
    // from the form that keeps every statement, the exact canonical from the one that also inlines
    // single-use temporaries.
    let allocator = Allocator::default();
    let statement_form = node.normalized(&allocator, false)?;
    let line_parts = statement_form.parts()?;
    let (type3_lines, type3_units, type_refs) = type3_walk(&line_parts, &locals_of(&line_parts));

    // One Dump with rename mode on — the xname canonical and the node count from one walk — with
    // the types it declares as slots: the exact pass compares what runs.
    let exact_form = statement_form.normalized(&allocator, true)?;
    let exact_locals = locals_of(&exact_form.parts()?);
    let mut xd = Dump::with_types(Some(&exact_locals), Types::Slots);
    let xname = exact_form.render(&mut xd);
    let size = xd.count;
    Some(AnalyzedFn { cluster_canonical, xname_canonical: xname, type3_lines, type_refs, type3_units, statements, size })
}

/// Full callable analysis of a source text, or `None` if the text doesn't parse as a callable (a
/// `function`, or an arrow / function expression assigned to a `const`).
fn analyze_one(text: &str) -> Option<AnalyzedFn> {
    let allocator = Allocator::default();
    let prog = parse(&allocator, text)?;
    let node = match leading_def_stmt(&prog)? {
        Statement::FunctionDeclaration(f) => CallableNode::Function(f),
        Statement::ExportNamedDeclaration(e) => match e.declaration.as_ref() {
            Some(Declaration::FunctionDeclaration(f)) => CallableNode::Function(f),
            _ => return None,
        },
        Statement::ExportDefaultDeclaration(e) => match &e.declaration {
            ExportDefaultDeclarationKind::FunctionDeclaration(f) => CallableNode::Function(f),
            _ => return None,
        },
        Statement::VariableDeclaration(v) => CallableNode::Bound(v.kind, v.declarations.first()?),
        _ => return None,
    };
    analyze_node(&node, true)
}

#[must_use]
pub fn analyze_functions(texts: &[String]) -> Vec<Option<AnalyzedFn>> {
    texts.par_iter().map(|t| analyze_one(t)).collect()
}

/// The body canonicalized with every name the **module** introduced erased — the widest rung of the
/// erasure ladder, which the `scope` lens projects as a single fact.
///
/// One string rather than one per statement: what it asserts is all-or-nothing — two bodies either
/// reduce to the same shape or they do not — and as one rare string it carries the IDF weight that
/// claim deserves.
#[must_use]
pub fn scope_canonical(
    params: &FormalParameters<'_>,
    body: &[Statement<'_>],
    module_names: &HashSet<String>,
) -> String {
    let mut collect = Collect::default();
    collect.add_params(params);
    for stmt in body {
        collect.collect_stmt(stmt);
    }
    let mut locals = collect.bound;
    locals.extend(module_names.iter().cloned());
    let mut dump = Dump::new(Some(&locals));
    let rendered = dump.formal_params(params);
    let rendered_body = dump.block_body(body);
    dump.node("Func", &[Dump::lit_str("_fn"), rendered, rendered_body])
}

// ───────────────────────────── statement stream ─────────────────────────────

/// One line per statement at **every** nesting level, in source order — the engine's
/// [`Facets::statements`](dup_defs_core::Facets::statements).
///
/// A compound statement contributes a **header** naming what it tests (`If(test)`,
/// `Switch(discriminant)`, `ForOf(left, right)`) with its body elided, and the body follows one level
/// deeper — the same shape the Python and Rust frontends produce, so the three streams are
/// comparable rather than merely all present.
///
/// One `Dump` for the whole definition, so a local carries the same `_v{n}` across every line it
/// appears on: here the order *is* the signal. Types are written as they are spelled — converge
/// parts two definitions on the names they use, a type among them.
fn statement_stream(parts: &Parts<'_, '_>, locals: &HashSet<String>) -> Vec<CoreStatement> {
    let mut dump = Dump::with_types(Some(locals), Types::Written);
    let type_params = dump.type_params(parts.type_params);
    let rendered = dump.formal_params(parts.params);
    let ret = dump.annotation(parts.ret);
    let flags = format!("async={}", parts.is_async);
    // The definition's own header opens the stream at depth 0, its body at depth 1 — the contract
    // every frontend agrees on. The name is `_fn` for the same reason the xname canonical blanks it.
    let mut fields = vec![Dump::lit_str("_fn"), rendered, Dump::lit_str(&flags)];
    dump.signature_tail(&mut fields, ret, type_params);
    let head = dump.node("Func", &fields);
    let mut walk = StreamWalk { dump: &mut dump, out: Vec::new() };
    walk.push(0, head);
    walk.stmts(parts.body, 1);
    walk.out
}

/// The Type-3 lines, their parts ([`AnalyzedFn::type3_units`]) and the types they name
/// ([`AnalyzedFn::type_refs`]): the same walk as the statement stream, with the types as slots and
/// the header written the way the Python frontend writes it ([`Dump::compact_header`]).
///
/// The pass compares statements at every depth — a change inside a loop costs the statement it
/// touched, not the loop — and every statement's sub-expressions are where two edited lines still
/// agree. The header comes first because the pass reads every function's first line as its
/// interface: context, not a step.
fn type3_walk(parts: &Parts<'_, '_>, locals: &HashSet<String>) -> (Vec<String>, Vec<(u32, String)>, Vec<String>) {
    let mut dump = Dump::with_types(Some(locals), Types::Slots);
    dump.units = Some(Vec::new());
    dump.unit_line = UNFILED;
    let head = dump.compact_header(parts);
    let mut walk = StreamWalk { dump: &mut dump, out: Vec::new() };
    walk.push(0, head);
    walk.stmts(parts.body, 1);
    let lines = walk.out.into_iter().map(|s| s.line).collect();
    (lines, dump.units.unwrap_or_default(), dump.type_refs)
}


struct StreamWalk<'a, 'b> {
    dump: &'b mut Dump<'a>,
    out: Vec<CoreStatement>,
}

impl StreamWalk<'_, '_> {
    /// A line is rendered whole before it is pushed, and nothing else is pushed in between, so the
    /// units still unfiled when it arrives are exactly its own.
    fn push(&mut self, depth: u16, line: String) {
        let at = u32::try_from(self.out.len()).unwrap_or(UNFILED - 1);
        if let Some(units) = self.dump.units.as_mut() {
            for unit in units.iter_mut().rev().take_while(|u| u.0 == UNFILED) {
                unit.0 = at;
            }
        }
        self.out.push(CoreStatement { line, depth });
    }

    fn stmts(&mut self, stmts: &[Statement<'_>], depth: u16) {
        for stmt in stmts {
            self.stmt(stmt, depth);
        }
    }

    /// A statement's body, whether it is a block or a single statement — `if (c) f();` nests just as
    /// `if (c) { f(); }` does, and a stream that treated the braces as the structure would read two
    /// equivalent sources as different shapes.
    fn body(&mut self, stmt: &Statement<'_>, depth: u16) {
        match stmt {
            Statement::BlockStatement(block) => self.stmts(&block.body, depth),
            other => self.stmt(other, depth),
        }
    }

    fn stmt(&mut self, stmt: &Statement<'_>, depth: u16) {
        match stmt {
            Statement::BlockStatement(block) => {
                let head = self.dump.node("Block", &[]);
                self.push(depth, head);
                self.stmts(&block.body, depth + 1);
            }
            Statement::IfStatement(node) => {
                let test = self.dump.expr(&node.test);
                let head = self.dump.node("If", &[test]);
                self.push(depth, head);
                self.body(&node.consequent, depth + 1);
                if let Some(alt) = &node.alternate {
                    let head = self.dump.node("Else", &[]);
                    self.push(depth, head);
                    match alt {
                        // `else if` is a sibling of its `else`, not a level deeper — what the source
                        // means, and what Python's `elif` and Rust's `else if` render as.
                        Statement::IfStatement(_) => self.stmt(alt, depth),
                        other => self.body(other, depth + 1),
                    }
                }
            }
            Statement::ForStatement(node) => {
                let init = node.init.as_ref().map_or_else(String::new, |i| self.dump.for_init(i));
                let test = node.test.as_ref().map_or_else(String::new, |t| self.dump.expr(t));
                let update = node.update.as_ref().map_or_else(String::new, |u| self.dump.expr(u));
                let head = self.dump.node("For", &[init, test, update]);
                self.push(depth, head);
                self.body(&node.body, depth + 1);
            }
            Statement::ForInStatement(node) => {
                let left = self.dump.for_left(&node.left);
                let right = self.dump.expr(&node.right);
                let head = self.dump.node("ForIn", &[left, right]);
                self.push(depth, head);
                self.body(&node.body, depth + 1);
            }
            Statement::ForOfStatement(node) => {
                let tag = if node.r#await { "ForAwaitOf" } else { "ForOf" };
                let left = self.dump.for_left(&node.left);
                let right = self.dump.expr(&node.right);
                let head = self.dump.node(tag, &[left, right]);
                self.push(depth, head);
                self.body(&node.body, depth + 1);
            }
            Statement::WhileStatement(node) => {
                let test = self.dump.expr(&node.test);
                let head = self.dump.node("While", &[test]);
                self.push(depth, head);
                self.body(&node.body, depth + 1);
            }
            Statement::DoWhileStatement(node) => {
                let test = self.dump.expr(&node.test);
                let head = self.dump.node("DoWhile", &[test]);
                self.push(depth, head);
                self.body(&node.body, depth + 1);
            }
            Statement::SwitchStatement(node) => {
                let disc = self.dump.expr(&node.discriminant);
                let head = self.dump.node("Switch", &[disc]);
                self.push(depth, head);
                for case in &node.cases {
                    let test = case.test.as_ref().map_or_else(String::new, |t| self.dump.expr(t));
                    let head = self.dump.node("Case", &[test]);
                    self.push(depth + 1, head);
                    self.stmts(&case.consequent, depth + 2);
                }
            }
            Statement::TryStatement(node) => {
                let head = self.dump.node("Try", &[]);
                self.push(depth, head);
                self.stmts(&node.block.body, depth + 1);
                if let Some(handler) = &node.handler {
                    let param =
                        handler.param.as_ref().map_or_else(String::new, |p| self.dump.binding(&p.pattern));
                    let head = self.dump.node("Catch", &[param]);
                    self.push(depth, head);
                    self.stmts(&handler.body.body, depth + 1);
                }
                if let Some(finalizer) = &node.finalizer {
                    let head = self.dump.node("Finally", &[]);
                    self.push(depth, head);
                    self.stmts(&finalizer.body, depth + 1);
                }
            }
            // A label names a statement; it is not a block of its own, so it adds no depth.
            Statement::LabeledStatement(node) => self.stmt(&node.body, depth),
            other => {
                let line = self.dump.stmt(other);
                self.push(depth, line);
            }
        }
    }
}

// ───────────────────────────── use sites ─────────────────────────────

/// Every use site in one source — the TypeScript half of the `use` lens
/// ([`dup_defs_core::uses`]): each statement that mentions a known name, written with that name as
/// the anchor. A site is the statement's own line — a compound statement's header, not its body,
/// whose statements are sites of their own — as the statement stream writes it, and inside a
/// function its locals rename as they do there. A callback handed over in an expression stays part
/// of the statement it is handed over in, as a Python lambda does.
pub(crate) fn collect_sites(source: &str, path: &str, known: &HashSet<&str>) -> Vec<Site> {
    let allocator = Allocator::default();
    let source_type = SourceType::from_path(Path::new(path)).unwrap_or_else(|_| SourceType::ts());
    let ret = Parser::new(&allocator, source, source_type).parse();
    if ret.panicked {
        return Vec::new();
    }
    let mut walk = UseWalk { known, out: Vec::new() };
    walk.block(&ret.program.body, &HashSet::new(), (false, false));
    walk.out
}

struct UseWalk<'k> {
    known: &'k HashSet<&'k str>,
    out: Vec<Site>,
}

impl UseWalk<'_> {
    /// A block's statements, each with the methods called beside it — the co-occurrence channel.
    /// `within` is `(in_try, in_loop)`.
    fn block(&mut self, stmts: &[Statement<'_>], locals: &HashSet<String>, within: (bool, bool)) {
        let mut hunt = Dump::hunting(locals, self.known);
        for stmt in stmts {
            let _ = use_line(&mut hunt, stmt);
        }
        let neighbours: Vec<String> = hunt.calls.into_iter().collect();
        for stmt in stmts {
            self.stmt(stmt, locals, within, &neighbours);
        }
    }

    /// One site per known name `render` mentions, each written with that name as the anchor.
    fn site<F>(&mut self, locals: &HashSet<String>, ctx: &'static str, neighbours: &[String], render: F)
    where
        F: Fn(&mut Dump<'_>) -> Option<String>,
    {
        let mut hunt = Dump::hunting(locals, self.known);
        if render(&mut hunt).is_none() {
            return;
        }
        for name in hunt.hunted {
            let mut anchored = Dump::anchoring(locals, &name);
            if let Some(canon) = render(&mut anchored) {
                self.out.push(Site { name, canon, ctx, cooc: neighbours.to_vec() });
            }
        }
    }

    /// A function's body, under its own locals.
    fn function(&mut self, params: &FormalParameters<'_>, body: &[Statement<'_>]) {
        let mut collect = Collect::default();
        collect.add_params(params);
        for stmt in body {
            collect.collect_stmt(stmt);
        }
        self.block(body, &collect.bound, (false, false));
    }

    /// A class: its methods' bodies, and its properties' values as sites of their own. What it
    /// extends or implements is a declaration, not a use.
    fn class(&mut self, class: &Class<'_>) {
        let empty = HashSet::new();
        for element in &class.body.body {
            match element {
                ast::ClassElement::MethodDefinition(m) => {
                    if let Some(body) = &m.value.body {
                        self.function(&m.value.params, &body.statements);
                    }
                }
                ast::ClassElement::PropertyDefinition(p) => match &p.value {
                    Some(Expression::ArrowFunctionExpression(a)) => self.function(&a.params, &a.body.statements),
                    Some(value) => {
                        self.site(&empty, "plain", &[], |d| {
                            let key = d.property_key(&p.key);
                            let value = d.expr(value);
                            Some(d.node("Prop", &[key, value]))
                        });
                        self.functions_in(value);
                    }
                    None => {}
                },
                ast::ClassElement::StaticBlock(b) => self.block(&b.body, &empty, (false, false)),
                _ => {}
            }
        }
    }

    fn declaration(&mut self, decl: &Declaration<'_>, locals: &HashSet<String>, ctx: &'static str, neighbours: &[String]) {
        match decl {
            Declaration::FunctionDeclaration(f) => self.callable(f, locals, neighbours),
            Declaration::ClassDeclaration(c) => self.class(c),
            Declaration::VariableDeclaration(v) => self.variables(v, locals, ctx, neighbours),
            _ => {}
        }
    }

    /// A `function`: its signature is a site when it names a known type, its body is walked.
    fn callable(&mut self, f: &Function<'_>, locals: &HashSet<String>, neighbours: &[String]) {
        self.site(locals, "plain", neighbours, |d| {
            let params = d.formal_params(&f.params);
            let ret = d.annotation(f.return_type.as_deref()).unwrap_or_default();
            Some(d.node("Func", &[params, ret]))
        });
        if let Some(body) = &f.body {
            self.function(&f.params, &body.statements);
        }
    }

    /// A declaration: a function bound to a name is walked as a function; every other declarator is
    /// a site of its own.
    fn variables(&mut self, v: &VariableDeclaration<'_>, locals: &HashSet<String>, ctx: &'static str, neighbours: &[String]) {
        for declarator in &v.declarations {
            match &declarator.init {
                Some(Expression::ArrowFunctionExpression(a)) => self.function(&a.params, &a.body.statements),
                Some(Expression::FunctionExpression(f)) => {
                    if let Some(body) = &f.body {
                        self.function(&f.params, &body.statements);
                    }
                }
                init => {
                    self.site(locals, ctx, neighbours, |d| {
                        let item = d.declarator(declarator);
                        Some(d.var_node(v.kind, &[item]))
                    });
                    if let Some(init) = init {
                        self.functions_in(init);
                    }
                }
            }
        }
    }

    fn nested(&mut self, stmt: &Statement<'_>, locals: &HashSet<String>, within: (bool, bool)) {
        match stmt {
            Statement::BlockStatement(b) => self.block(&b.body, locals, within),
            other => self.block(std::slice::from_ref(other), locals, within),
        }
    }

    fn stmt(&mut self, stmt: &Statement<'_>, locals: &HashSet<String>, within: (bool, bool), neighbours: &[String]) {
        let (in_try, in_loop) = within;
        let construct = match stmt {
            Statement::TryStatement(_) => Construct::Try,
            Statement::ForStatement(_)
            | Statement::ForInStatement(_)
            | Statement::ForOfStatement(_)
            | Statement::WhileStatement(_)
            | Statement::DoWhileStatement(_) => Construct::Loop,
            Statement::IfStatement(_) | Statement::SwitchStatement(_) => Construct::Guard,
            _ => Construct::Plain,
        };
        let ctx = site_ctx(construct, Within { in_try, in_loop });
        match stmt {
            Statement::FunctionDeclaration(f) => self.callable(f, locals, neighbours),
            Statement::ClassDeclaration(c) => self.class(c),
            Statement::VariableDeclaration(v) => self.variables(v, locals, ctx, neighbours),
            Statement::ExportNamedDeclaration(e) => {
                if let Some(decl) = &e.declaration {
                    self.declaration(decl, locals, ctx, neighbours);
                }
            }
            Statement::ExportDefaultDeclaration(e) => match &e.declaration {
                ExportDefaultDeclarationKind::FunctionDeclaration(f) => self.callable(f, locals, neighbours),
                ExportDefaultDeclarationKind::ClassDeclaration(c) => self.class(c),
                other => {
                    if let Some(e) = other.as_expression() {
                        self.site(locals, ctx, neighbours, |d| Some(d.expr(e)));
                        self.functions_in(e);
                    }
                }
            },
            Statement::BlockStatement(b) => self.block(&b.body, locals, within),
            Statement::LabeledStatement(l) => self.stmt(&l.body, locals, within, neighbours),
            Statement::TryStatement(t) => {
                self.block(&t.block.body, locals, (true, in_loop));
                if let Some(handler) = &t.handler {
                    self.block(&handler.body.body, locals, (true, in_loop));
                }
                if let Some(finalizer) = &t.finalizer {
                    self.block(&finalizer.body, locals, (true, in_loop));
                }
            }
            Statement::IfStatement(n) => {
                self.site(locals, ctx, neighbours, |d| use_line(d, stmt));
                self.nested(&n.consequent, locals, within);
                if let Some(alt) = &n.alternate {
                    self.nested(alt, locals, within);
                }
            }
            Statement::SwitchStatement(n) => {
                self.site(locals, ctx, neighbours, |d| use_line(d, stmt));
                for case in &n.cases {
                    if let Some(test) = &case.test {
                        self.site(locals, ctx, neighbours, |d| {
                            let test = d.expr(test);
                            Some(d.node("Case", &[test]))
                        });
                    }
                    self.block(&case.consequent, locals, within);
                }
            }
            Statement::ForStatement(n) => {
                self.site(locals, ctx, neighbours, |d| use_line(d, stmt));
                self.nested(&n.body, locals, (in_try, true));
            }
            Statement::ForInStatement(n) => {
                self.site(locals, ctx, neighbours, |d| use_line(d, stmt));
                self.nested(&n.body, locals, (in_try, true));
            }
            Statement::ForOfStatement(n) => {
                self.site(locals, ctx, neighbours, |d| use_line(d, stmt));
                self.nested(&n.body, locals, (in_try, true));
            }
            Statement::WhileStatement(n) => {
                self.site(locals, ctx, neighbours, |d| use_line(d, stmt));
                self.nested(&n.body, locals, (in_try, true));
            }
            Statement::DoWhileStatement(n) => {
                self.site(locals, ctx, neighbours, |d| use_line(d, stmt));
                self.nested(&n.body, locals, (in_try, true));
            }
            _ => {
                self.site(locals, ctx, neighbours, |d| use_line(d, stmt));
                let own = match stmt {
                    Statement::ExpressionStatement(e) => Some(&e.expression),
                    Statement::ReturnStatement(r) => r.argument.as_ref(),
                    Statement::ThrowStatement(t) => Some(&t.argument),
                    _ => None,
                };
                if let Some(expr) = own {
                    self.functions_in(expr);
                }
            }
        }
    }

    /// The functions written inside an expression — callbacks, object-literal methods — each walked
    /// as a function of its own, as the site they sit in wrote them elided.
    fn functions_in(&mut self, expr: &Expression<'_>) {
        let mut found = Vec::new();
        nested_functions(expr, &mut found);
        for (params, body) in found {
            self.function(params, body);
        }
    }
}

/// A function written inside an expression: its parameters and its body.
type Nested<'x, 'b> = (&'x FormalParameters<'b>, &'x [Statement<'b>]);

/// Every function with a body of statements written inside `expr`, outermost first: a callback, an
/// object literal's method, an immediately-invoked function. Not descended into — what a found
/// function's body holds is that function's to walk. An arrow with an expression body is what a
/// Python lambda is, part of its site; it is looked through for the functions it holds.
fn nested_functions<'x, 'b>(expr: &'x Expression<'b>, out: &mut Vec<Nested<'x, 'b>>) {
    let args = |arguments: &'x oxc_allocator::Vec<'b, Argument<'b>>, out: &mut Vec<Nested<'x, 'b>>| {
        for a in arguments {
            match a {
                Argument::SpreadElement(s) => nested_functions(&s.argument, out),
                other => {
                    if let Some(e) = other.as_expression() {
                        nested_functions(e, out);
                    }
                }
            }
        }
    };
    match expr {
        Expression::ArrowFunctionExpression(a) if a.expression => {
            for stmt in &a.body.statements {
                if let Statement::ExpressionStatement(e) = stmt {
                    nested_functions(&e.expression, out);
                }
            }
        }
        Expression::ArrowFunctionExpression(a) => out.push((&a.params, &a.body.statements)),
        Expression::FunctionExpression(f) => {
            if let Some(body) = &f.body {
                out.push((&f.params, &body.statements));
            }
        }
        Expression::CallExpression(c) => {
            nested_functions(&c.callee, out);
            args(&c.arguments, out);
        }
        Expression::NewExpression(n) => {
            nested_functions(&n.callee, out);
            args(&n.arguments, out);
        }
        Expression::ChainExpression(c) => match &c.expression {
            ChainElement::CallExpression(call) => {
                nested_functions(&call.callee, out);
                args(&call.arguments, out);
            }
            ChainElement::TSNonNullExpression(e) => nested_functions(&e.expression, out),
            other => {
                if let Some(m) = other.as_member_expression() {
                    nested_functions(m.object(), out);
                }
            }
        },
        Expression::StaticMemberExpression(m) => nested_functions(&m.object, out),
        Expression::ComputedMemberExpression(m) => {
            nested_functions(&m.object, out);
            nested_functions(&m.expression, out);
        }
        Expression::PrivateFieldExpression(m) => nested_functions(&m.object, out),
        Expression::ObjectExpression(o) => {
            for p in &o.properties {
                match p {
                    ObjectPropertyKind::ObjectProperty(p) => nested_functions(&p.value, out),
                    ObjectPropertyKind::SpreadProperty(s) => nested_functions(&s.argument, out),
                }
            }
        }
        Expression::ArrayExpression(a) => {
            for e in &a.elements {
                match e {
                    ArrayExpressionElement::SpreadElement(s) => nested_functions(&s.argument, out),
                    other => {
                        if let Some(e) = other.as_expression() {
                            nested_functions(e, out);
                        }
                    }
                }
            }
        }
        Expression::ConditionalExpression(c) => {
            nested_functions(&c.test, out);
            nested_functions(&c.consequent, out);
            nested_functions(&c.alternate, out);
        }
        Expression::LogicalExpression(l) => {
            nested_functions(&l.left, out);
            nested_functions(&l.right, out);
        }
        Expression::BinaryExpression(b) => {
            nested_functions(&b.left, out);
            nested_functions(&b.right, out);
        }
        Expression::AssignmentExpression(a) => nested_functions(&a.right, out),
        Expression::SequenceExpression(seq) => seq.expressions.iter().for_each(|e| nested_functions(e, out)),
        Expression::AwaitExpression(a) => nested_functions(&a.argument, out),
        Expression::YieldExpression(y) => {
            if let Some(a) = &y.argument {
                nested_functions(a, out);
            }
        }
        Expression::UnaryExpression(u) => nested_functions(&u.argument, out),
        Expression::ParenthesizedExpression(p) => nested_functions(&p.expression, out),
        Expression::TSAsExpression(e) => nested_functions(&e.expression, out),
        Expression::TSSatisfiesExpression(e) => nested_functions(&e.expression, out),
        Expression::TSNonNullExpression(e) => nested_functions(&e.expression, out),
        Expression::TSTypeAssertion(e) => nested_functions(&e.expression, out),
        Expression::TSInstantiationExpression(e) => nested_functions(&e.expression, out),
        Expression::TemplateLiteral(t) => t.expressions.iter().for_each(|e| nested_functions(e, out)),
        Expression::TaggedTemplateExpression(t) => {
            nested_functions(&t.tag, out);
            t.quasi.expressions.iter().for_each(|e| nested_functions(e, out));
        }
        Expression::JSXElement(j) => nested_in_jsx(&j.opening_element.attributes, &j.children, out),
        Expression::JSXFragment(f) => nested_in_jsx(&[], &f.children, out),
        _ => {}
    }
}

fn nested_in_jsx<'x, 'b>(
    attributes: &'x [ast::JSXAttributeItem<'b>],
    children: &'x [ast::JSXChild<'b>],
    out: &mut Vec<Nested<'x, 'b>>,
) {
    for item in attributes {
        match item {
            ast::JSXAttributeItem::Attribute(a) => match &a.value {
                Some(ast::JSXAttributeValue::ExpressionContainer(c)) => {
                    if let Some(e) = c.expression.as_expression() {
                        nested_functions(e, out);
                    }
                }
                Some(ast::JSXAttributeValue::Element(e)) => nested_in_jsx(&e.opening_element.attributes, &e.children, out),
                Some(ast::JSXAttributeValue::Fragment(f)) => nested_in_jsx(&[], &f.children, out),
                _ => {}
            },
            ast::JSXAttributeItem::SpreadAttribute(s) => nested_functions(&s.argument, out),
        }
    }
    for child in children {
        match child {
            ast::JSXChild::Element(e) => nested_in_jsx(&e.opening_element.attributes, &e.children, out),
            ast::JSXChild::Fragment(f) => nested_in_jsx(&[], &f.children, out),
            ast::JSXChild::ExpressionContainer(c) => {
                if let Some(e) = c.expression.as_expression() {
                    nested_functions(e, out);
                }
            }
            ast::JSXChild::Spread(s) => nested_functions(&s.expression, out),
            ast::JSXChild::Text(_) => {}
        }
    }
}

/// A statement's own line, as a use site sees it: a compound statement's header (what it tests,
/// what it iterates), a simple statement whole — `None` for what has no line of its own (a block, a
/// declaration of a function or a class, whose insides are walked).
fn use_line(d: &mut Dump<'_>, stmt: &Statement<'_>) -> Option<String> {
    Some(match stmt {
        Statement::IfStatement(n) => {
            let test = d.expr(&n.test);
            d.node("If", &[test])
        }
        Statement::ForStatement(n) => {
            let init = n.init.as_ref().map_or_else(String::new, |i| d.for_init(i));
            let test = n.test.as_ref().map_or_else(String::new, |t| d.expr(t));
            let update = n.update.as_ref().map_or_else(String::new, |u| d.expr(u));
            d.node("For", &[init, test, update])
        }
        Statement::ForInStatement(n) => {
            let left = d.for_left(&n.left);
            let right = d.expr(&n.right);
            d.node("ForIn", &[left, right])
        }
        Statement::ForOfStatement(n) => {
            let left = d.for_left(&n.left);
            let right = d.expr(&n.right);
            d.node(if n.r#await { "ForAwaitOf" } else { "ForOf" }, &[left, right])
        }
        Statement::WhileStatement(n) => {
            let test = d.expr(&n.test);
            d.node("While", &[test])
        }
        Statement::DoWhileStatement(n) => {
            let test = d.expr(&n.test);
            d.node("DoWhile", &[test])
        }
        Statement::SwitchStatement(n) => {
            let disc = d.expr(&n.discriminant);
            d.node("Switch", &[disc])
        }
        Statement::BlockStatement(_)
        | Statement::TryStatement(_)
        | Statement::LabeledStatement(_)
        | Statement::FunctionDeclaration(_)
        | Statement::ClassDeclaration(_)
        | Statement::ExportNamedDeclaration(_)
        | Statement::ExportDefaultDeclaration(_)
        | Statement::EmptyStatement(_) => return None,
        other => d.stmt(other),
    })
}

// ───────────────────────────── reach ─────────────────────────────

/// What each name a module's imports bind stands for, as the engine's dotted path.
///
/// `import { c } from "a/b"` binds `c` to `a.b.c`; `import { c as z }` binds `z` to the same; a
/// default or namespace import binds its name to the module path itself, since that is the whole of
/// what it names. A bare `import "a/b"` binds nothing and is skipped rather than guessed at.
///
/// 🔴 A **relative** specifier is resolved against the importing file's own directory. Left as
/// written, `./a/b` and `../a/b` become paths beginning with a bare `.` or `..`, and every relative
/// import in the tree then shares those as a prefix node — so "both of these import something
/// relative" reads as "both of these are about the same thing". Found by running the pass on a real
/// frontend, where eighteen families formed on exactly that and on nothing else. Resolved, two
/// packages' `./models` are two distinct nodes, which is what the path was recording in the first
/// place.
#[must_use]
pub fn module_imports(prog: &ast::Program<'_>, file: &str) -> Vec<(String, Arc<str>)> {
    let mut out = Vec::new();
    for stmt in &prog.body {
        let Statement::ImportDeclaration(decl) = stmt else { continue };
        let source = resolve_specifier(decl.source.value.as_str(), file);
        let Some(specifiers) = &decl.specifiers else { continue };
        for specifier in specifiers {
            match specifier {
                ast::ImportDeclarationSpecifier::ImportSpecifier(s) => {
                    let mut path = source.clone();
                    path.push(s.imported.name().to_string());
                    out.push((s.local.name.to_string(), reach_path(&path)));
                }
                ast::ImportDeclarationSpecifier::ImportDefaultSpecifier(s) => {
                    out.push((s.local.name.to_string(), reach_path(&source)));
                }
                ast::ImportDeclarationSpecifier::ImportNamespaceSpecifier(s) => {
                    out.push((s.local.name.to_string(), reach_path(&source)));
                }
            }
        }
    }
    out
}

/// A specifier as path segments: absolute ones as written, relative ones walked from the importing
/// file's directory so they name where they actually point.
pub(crate) fn resolve_specifier(specifier: &str, file: &str) -> Vec<String> {
    if !specifier.starts_with('.') {
        return specifier.split('/').map(str::to_owned).collect();
    }
    // The file's directory — everything but the file name itself.
    let mut segments: Vec<String> = file.split('/').map(str::to_owned).collect();
    segments.pop();
    for part in specifier.split('/') {
        match part {
            "." | "" => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other.to_owned()),
        }
    }
    segments
}

/// The dotted paths a definition reaches, read off its own statement stream.
///
/// The stream is already rendered, and the canonical renames *locals* to `_v{n}` while leaving free
/// names verbatim — so an imported name appears in it as itself. Reading identifiers back off those
/// strings costs one pass over text the frontend has already produced, where a second AST visitor
/// would have to keep up with an expression grammar of some eighty variants; anything the unparser
/// learns to render, this sees.
///
/// Not filtered by boundness, for the same reason as the other frontends: a local shadowing an
/// import inside one function is rare, and reading the shadow as "this definition does not reach
/// that module" is a silent false negative, the expensive direction here.
#[must_use]
pub fn reached_paths(
    statements: &[CoreStatement],
    imports: &HashMap<String, Arc<str>>,
) -> Vec<Arc<str>> {
    if imports.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<Arc<str>> = Vec::new();
    let mut token = String::new();
    let take = |token: &mut String, out: &mut Vec<Arc<str>>| {
        if let Some(path) = imports.get(token.as_str()) {
            out.push(Arc::clone(path));
        }
        token.clear();
    };
    for statement in statements {
        for ch in statement.line.chars() {
            if ch.is_alphanumeric() || ch == '_' || ch == '$' {
                token.push(ch);
            } else if !token.is_empty() {
                take(&mut token, &mut out);
            }
        }
        if !token.is_empty() {
            take(&mut token, &mut out);
        }
    }
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_canonical_is_deterministic_for_identical_source() {
        let src = "function foo(x: number): number { return x + 1; }";
        let a = ast_canonical(src);
        let b = ast_canonical(src);
        assert_eq!(a, b);
        assert!(a.contains("Func"));
    }

    #[test]
    fn xname_canonical_alpha_equates_renamed_copies() {
        let a = "function add(x: number, y: number) { return x + y; }";
        let b = "function plus(a: number, b: number) { return a + b; }";
        let aa = analyze_one(a).expect("a parses");
        let bb = analyze_one(b).expect("b parses");
        assert_eq!(aa.xname_canonical, bb.xname_canonical, "xname canonicals should match across alpha-renaming\n  a: {}\n  b: {}", aa.xname_canonical, bb.xname_canonical);
    }

    #[test]
    fn xname_distinguishes_different_bodies() {
        let a = "function add(x: number, y: number) { return x + y; }";
        let b = "function add(x: number, y: number) { return x - y; }";
        let aa = analyze_one(a).expect("a parses");
        let bb = analyze_one(b).expect("b parses");
        assert_ne!(aa.xname_canonical, bb.xname_canonical, "different operators must produce different xname canonicals");
    }

    #[test]
    fn cluster_canonical_preserves_names() {
        let src = "function add(x: number, y: number) { return x + y; }";
        let cc = ast_canonical(src);
        assert!(cc.contains("'add'"), "cluster canonical should keep the def name, got: {cc}");
        assert!(cc.contains("'x'") && cc.contains("'y'"), "param names should be preserved: {cc}");
    }

    #[test]
    fn analyze_lines_are_the_header_then_every_statement() {
        let src = "function f(x: number, y: number) {\n  const z = x + y;\n  return z * 2;\n}";
        let a = analyze_one(src).expect("parses");
        let (lines, size) = (a.type3_lines, a.size);
        // The header first: the Type-3 pass reads every function's first line as its interface.
        assert_eq!(lines.len(), 3, "got lines: {lines:?}");
        assert_eq!(lines[0], "function _fn(_v0: _T0, _v1: _T0)", "{lines:?}");
        assert!(size > 0);
    }

    #[test]
    fn a_nested_statement_is_a_line_of_its_own_and_its_parts_are_units() {
        // An edit deep inside a loop costs the statement it touched, not the loop: the loop is a
        // header line and each statement in it a line, and the sub-expressions of every line are
        // the parts two edited lines can still agree on.
        let src = "function f(rows: R[]) {\n  for (const r of rows) {\n    if (r.ok && r.v > 0) {\n      out.push(r.render(true));\n    }\n  }\n}";
        let a = analyze_one(src).expect("parses");
        assert_eq!(a.type3_lines.len(), 4, "header, for, if, push: {:?}", a.type3_lines);
        let units_of = |line: u32| -> Vec<&str> {
            a.type3_units.iter().filter(|(l, _)| *l == line).map(|(_, u)| u.as_str()).collect()
        };
        // The `if` header's parts: the two comparisons and the member reads inside them — not the
        // whole test, which is the line's own expression.
        let test_parts = units_of(2);
        assert!(test_parts.iter().any(|u| u.starts_with("Bin(")), "{test_parts:?}");
        assert!(!test_parts.iter().any(|u| u.starts_with("Logic(")), "the test itself is the line: {test_parts:?}");
        // The call's parts: the receiver chain and the inner call, filed under the call's own line.
        let call_parts = units_of(3);
        assert!(call_parts.iter().any(|u| u.starts_with("Call(Member(Id('_v1'), 'render'")), "{call_parts:?}");
        assert!(a.type3_units.iter().all(|(l, _)| (*l as usize) < a.type3_lines.len()), "every unit is filed under a line");
    }

    #[test]
    fn export_default_function_canonicalizes_like_plain_function() {
        let plain = "function foo(x: number) { return x + 1; }";
        let exported = "export default function foo(x: number) { return x + 1; }";
        let a = ast_canonical(plain);
        let b = ast_canonical(exported);
        assert_eq!(a, b, "export-default wrapper should be transparent\n  plain: {a}\n  exported: {b}");
    }

    #[test]
    fn arrow_const_name_is_blanked_so_renamed_copies_alpha_equate() {
        // The def name rides on the declarator (not a `Func` node); xname must blank it to `_fn` —
        // like a named `function` — so two differently-named arrow-consts with the same body match.
        let a = "const loadUser = (id: string) => { return db.find(id); };";
        let b = "const fetchOrder = (key: string) => { return db.find(key); };";
        let aa = analyze_one(a).expect("a parses");
        let bb = analyze_one(b).expect("b parses");
        assert!(aa.xname_canonical.contains("Bind('_fn')"), "top name should blank to _fn, got: {}", aa.xname_canonical);
        assert_eq!(aa.xname_canonical, bb.xname_canonical, "renamed arrow-consts should alpha-equate\n  a: {}\n  b: {}", aa.xname_canonical, bb.xname_canonical);
    }

    #[test]
    fn arrow_const_analyzes_as_function() {
        let src = "const fetch = async (x: number): Promise<number> => { return x + 1; };";
        let analysis = analyze_one(src);
        assert!(analysis.is_some(), "arrow-const should analyze");
        let analyzed = analysis.unwrap();
        assert!(analyzed.xname_canonical.contains("Arrow"));
        assert_eq!(analyzed.type3_lines.len(), 2, "the header and the single statement");
    }
}

#[cfg(test)]
mod type_slot_tests {
    use super::{analyze_one, AnalyzedFn};

    fn analyzed(src: &str) -> AnalyzedFn {
        analyze_one(src).expect("a function")
    }

    #[test]
    fn bodies_differing_only_in_declared_types_are_one_canonical() {
        // TypeScript erases its annotations before anything runs: the same code, whatever they say.
        let a = analyzed("function f(p: Token): number | null { const v = p.get('uid'); return v; }");
        let b = analyzed("function g(q: Claims): number | null { const v = q.get('uid'); return v; }");
        assert_eq!(a.xname_canonical, b.xname_canonical);
    }

    #[test]
    fn types_are_renamed_consistently_not_erased() {
        // Parameterized matching: one bijection has to carry the types of one onto the other's —
        // dropping the annotations, as the canonical used to, merged these two.
        let same = analyzed("function f(a: X, b: X): void { g(a, b); }");
        let mixed = analyzed("function f(a: X, b: Y): void { g(a, b); }");
        assert_ne!(same.xname_canonical, mixed.xname_canonical);
    }

    #[test]
    fn type3_lines_slot_the_types_and_report_them_apart() {
        let a = analyzed("function f(p: Token): Promise<Row> { const x: models.Row = load(p); return x; }");
        assert_eq!(a.type_refs, ["Token", "Promise", "Row", "models.Row"]);
        // The header the way the Python frontend writes it, types as slots.
        assert_eq!(a.type3_lines[0], "function _fn(_v0: _T0): TApp(_T1, _T2)");
        assert!(a.type3_lines[1].contains("Annot(Bind('_v1'), _T3)"), "{:?}", a.type3_lines);
        // The statement stream keeps them as written: converge parts definitions on names, a type
        // among them.
        assert!(a.statements[0].line.contains("Annot(Bind('_v0'), Token)"), "{:?}", a.statements[0]);
    }

    #[test]
    fn a_declared_type_parameter_is_bound_not_vocabulary() {
        let a = analyzed("function first<T>(items: Array<T>): T { return items[0]; }");
        assert_eq!(a.type_refs, ["Array"]);
        let b = analyzed("function head<U>(rows: Array<U>): U { return rows[0]; }");
        assert_eq!(a.xname_canonical, b.xname_canonical);
    }
}

#[cfg(test)]
mod use_tests {
    use super::collect_sites;
    use crate::defs::scan_source;
    use dup_defs_core::uses::use_facts;
    use dup_defs_core::ScanOpts;
    use std::collections::HashSet;
    use std::sync::Arc;

    #[test]
    fn a_site_is_the_statement_header_not_its_body() {
        let src = "function f() {\n  for (const row of query(Tbl)) {\n    other(Unrelated);\n  }\n}\n";
        let known: HashSet<&str> = ["Tbl"].into_iter().collect();
        let sites = collect_sites(src, "t.ts", &known);
        assert_eq!(sites.len(), 1, "one site for Tbl");
        assert!(sites[0].canon.starts_with("ForOf("), "the site keeps the statement's own tag: {}", sites[0].canon);
        assert!(!sites[0].canon.contains("other"), "the body is a site of its own, not part of this one");
        assert_eq!(sites[0].ctx, "loop");
    }

    #[test]
    fn the_anchor_and_its_surface_are_slots() {
        let src = "function f(k: string) {\n  const e = new Cache({ key: k, ttl: 5 });\n  return Cache.lookup(e.key);\n}\n";
        let known: HashSet<&str> = ["Cache"].into_iter().collect();
        let sites = collect_sites(src, "t.ts", &known);
        let canons: Vec<&str> = sites.iter().map(|s| s.canon.as_str()).collect();
        assert!(canons.iter().any(|c| c.contains("New(Id('_t0'), [Obj(ObjProp('#key#'")), "{canons:?}");
        assert!(canons.iter().any(|c| c.contains("Member(Id('_t0'), '#lookup#'")), "{canons:?}");
    }

    /// The `use` lens's load-bearing claim, in TypeScript: two definitions whose bodies share
    /// nothing, handled identically, produce the same use-site facts.
    #[test]
    fn divergent_bodies_with_identical_handling_share_their_use_facts() {
        let a = concat!(
            "export class ImageCache { key = ''; blob = new Uint8Array(); deadAt = 0; }\n",
            "export function putImage(s: Session, k: string, v: Uint8Array) { s.add(new ImageCache({ key: k, blob: v, deadAt: soon() })); }\n",
            "export function getImage(s: Session, k: string) { return s.query(ImageCache).filter(ImageCache.key === k).one(); }\n",
            "export function reapImages(s: Session) { s.query(ImageCache).filter(ImageCache.deadAt < now()).delete(); }\n",
        );
        let b = concat!(
            "export class PromptStore { payload: Json = {}; }\n",
            "export function putPrompt(s: Session, k: string, v: Json) { s.add(new PromptStore({ fingerprint: k, payload: v, expires: soon() })); }\n",
            "export function getPrompt(s: Session, k: string) { return s.query(PromptStore).filter(PromptStore.fingerprint === k).one(); }\n",
            "export function reapPrompts(s: Session) { s.query(PromptStore).filter(PromptStore.expires < now()).delete(); }\n",
        );
        let (fa, fb): (Arc<str>, Arc<str>) = (Arc::from("a.ts"), Arc::from("b.ts"));
        let opts = ScanOpts::default();
        let mut defs = scan_source(a, &fa, &opts);
        defs.extend(scan_source(b, &fb, &opts));
        let facts = use_facts(&[(a, "a.ts", 1), (b, "b.ts", 1)], &defs, collect_sites);
        let get = |n: &str| facts.get(n).unwrap_or_else(|| panic!("no use facts for {n}: {facts:?}")).clone();
        assert_eq!(get("ImageCache"), get("PromptStore"), "identical handling ⇒ identical use facts");
    }
}
