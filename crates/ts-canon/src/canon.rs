//! Structural canonicalization of TypeScript definitions over the oxc AST.
//!
//! Mirrors the role `py-canon::canon` plays for Python: produces three strings per callable
//! definition, all walked from one parse — `(cluster_canonical, xname_canonical, lines, size)` —
//! and a names-preserved cluster canonical for any def kind (used by the engine's name-gated
//! pass over body kinds).
//!
//! Unlike `py-canon`, the canonical here doesn't have a byte-exact external reference to match
//! (CPython's `ast.dump` shaped Python; there's no equivalent published reference for TS). We
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
//! * **JSX nodes** emit as `JSXElement(name)` / `JSXFragment` — opaque enough to not confuse
//!   structural matching, conservative enough to not cluster wholesale.
//! * **Long-tail nodes** we haven't explicitly cased emit as `Unknown_<NodeKind>` — deterministic
//!   for any given input, surfacing in `--calibrate` so the next round of tuning sees them.
//!
//! Type-3 lines are per-statement renamed s-expr strings (one statement → one line); the engine
//! tokenizes them via the same regex as `py-canon`'s lines and the IDF/cosine works unchanged.
#![allow(clippy::too_many_lines)] // top-level match arms enumerate AST variants — splitting them just spreads the same shape over more files

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use dup_defs_core::AnalyzedFn;
use oxc_allocator::Allocator;
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
/// (CPython behavior, preserved for TS), only the top function's params plus arrow-fn / inner-fn
/// declared *names* (which ARE bindings in the outer scope).
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
                for elt in &arr.elements {
                    if let Some(p) = elt {
                        self.add_binding(p);
                    }
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
    /// Count of node-emit calls — used as the cross-name pass's "substance" gate.
    count: usize,
}

impl<'a> Dump<'a> {
    fn new(locals: Option<&'a HashSet<String>>) -> Self {
        Self { locals, map: HashMap::new(), blanked: false, count: 0 }
    }

    #[allow(clippy::cast_possible_truncation)] // a function's distinct bound-name count is far below u32::MAX
    fn rename(&mut self, name: &str) -> String {
        if let Some(locals) = self.locals {
            if locals.contains(name) {
                let next = self.map.len() as u32;
                let slot = *self.map.entry(name.to_owned()).or_insert(next);
                return format!("_v{slot}");
            }
        }
        name.to_owned()
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
        let kind = match v.kind {
            VariableDeclarationKind::Var => "var",
            VariableDeclarationKind::Let => "let",
            VariableDeclarationKind::Const => "const",
            VariableDeclarationKind::Using => "using",
            VariableDeclarationKind::AwaitUsing => "await-using",
        };
        let items: Vec<String> = v.declarations.iter().map(|d| self.declarator(d)).collect();
        self.node("Var", &[Self::lit_str(kind), bracket_join(&items)])
    }

    fn declarator(&mut self, d: &VariableDeclarator<'_>) -> String {
        let id = self.binding(&d.id);
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
        let params = self.formal_params(&f.params);
        let body = f.body.as_deref().map_or_else(
            || String::new(),
            |b| self.block_body(&b.statements),
        );
        let async_g = format!("async={} gen={}", f.r#async, f.generator);
        self.node("Func", &[Self::lit_str(&name), params, body, async_g])
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
            TSType::TSStringKeyword(_) => "string".to_owned(),
            TSType::TSNumberKeyword(_) => "number".to_owned(),
            TSType::TSBooleanKeyword(_) => "boolean".to_owned(),
            TSType::TSNullKeyword(_) => "null".to_owned(),
            TSType::TSUndefinedKeyword(_) => "undefined".to_owned(),
            TSType::TSAnyKeyword(_) => "any".to_owned(),
            TSType::TSUnknownKeyword(_) => "unknown".to_owned(),
            TSType::TSNeverKeyword(_) => "never".to_owned(),
            TSType::TSVoidKeyword(_) => "void".to_owned(),
            TSType::TSObjectKeyword(_) => "object".to_owned(),
            TSType::TSBigIntKeyword(_) => "bigint".to_owned(),
            TSType::TSSymbolKeyword(_) => "symbol".to_owned(),
            TSType::TSThisType(_) => "this".to_owned(),
            TSType::TSTypeReference(r) => self.ts_type_name(&r.type_name),
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

    fn expr(&mut self, expr: &Expression<'_>) -> String {
        match expr {
            Expression::Identifier(id) => {
                let n = self.rename(id.name.as_str());
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
                let callee = self.expr(&c.callee);
                let args: Vec<String> =
                    c.arguments.iter().map(|a| self.argument(a)).collect();
                let optional = if c.optional { "opt" } else { "" };
                self.node("Call", &[callee, Self::lit_str(optional), bracket_join(&args)])
            }
            Expression::NewExpression(n) => {
                let callee = self.expr(&n.callee);
                let args: Vec<String> =
                    n.arguments.iter().map(|a| self.argument(a)).collect();
                self.node("New", &[callee, bracket_join(&args)])
            }
            Expression::StaticMemberExpression(m) => {
                let obj = self.expr(&m.object);
                let prop = m.property.name.as_str().to_owned();
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
                let params = self.formal_params(&a.params);
                let body = self.block_body(&a.body.statements);
                let async_g = format!("async={} gen=false expr={}", a.r#async, a.expression);
                self.node("Arrow", &[params, body, async_g])
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
            Expression::JSXElement(j) => {
                let name = jsx_name(&j.opening_element.name);
                self.node("JSX", &[Self::lit_str(&name)])
            }
            Expression::JSXFragment(_) => self.node("JSXFragment", &[]),
            Expression::V8IntrinsicExpression(_) => self.node("V8Intrinsic", &[]),
        }
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
        let items: Vec<String> = o
            .properties
            .iter()
            .map(|p| match p {
                ObjectPropertyKind::ObjectProperty(p) => {
                    let k = self.property_key(&p.key);
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

/// Free helper — bracket-joined list, no node-count side-effects (used to wrap collected items
/// before handing them to a parent `Dump::node` call without double-borrowing `self`).
fn bracket_join(items: &[String]) -> String {
    format!("[{}]", items.join(", "))
}

fn jsx_name(name: &ast::JSXElementName<'_>) -> String {
    match name {
        ast::JSXElementName::Identifier(id) => id.name.to_string(),
        ast::JSXElementName::IdentifierReference(id) => id.name.to_string(),
        ast::JSXElementName::NamespacedName(n) => format!("{}:{}", n.namespace.name, n.name.name),
        ast::JSXElementName::MemberExpression(_) => "<member>".to_owned(),
        ast::JSXElementName::ThisExpression(_) => "this".to_owned(),
    }
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

/// Full callable analysis: `(cluster_canonical, xname_canonical, lines, size)`, or `None` if
/// the text doesn't parse as a callable (FunctionDeclaration or anonymous arrow/function
/// expression assigned to a `const`).
fn analyze_one(text: &str) -> Option<AnalyzedFn> {
    let allocator = Allocator::default();
    let prog = parse(&allocator, text)?;
    let stmt = leading_def_stmt(&prog)?;

    // The leading statement is one of: FunctionDeclaration, export wrappers around it, or
    // a VariableDeclaration whose declarator's init is an arrow/function expression (the
    // arrow-const-as-function form). Locate the Function inside.
    let func = match stmt {
        Statement::FunctionDeclaration(f) => Some(f.as_ref()),
        Statement::ExportNamedDeclaration(e) => match e.declaration.as_ref() {
            Some(Declaration::FunctionDeclaration(f)) => Some(f.as_ref()),
            _ => None,
        },
        Statement::ExportDefaultDeclaration(e) => match &e.declaration {
            ExportDefaultDeclarationKind::FunctionDeclaration(f) => Some(f.as_ref()),
            _ => None,
        },
        _ => None,
    };

    // Two paths: a real Function node, or an arrow/function-expression init in a var decl.
    // Either way, we collect locals from params+body and emit the cluster + xname canonicals
    // off the same parse.
    let (params, body_stmts, cluster_canonical) = if let Some(f) = func {
        let body = f.body.as_deref()?;
        let mut cd = Dump::new(None);
        let cc = cluster_stmt(&mut cd, stmt);
        (Some(&*f.params), Some(&body.statements[..]), cc)
    } else if let Statement::VariableDeclaration(v) = stmt {
        let decl = v.declarations.first()?;
        let init = decl.init.as_ref()?;
        match init {
            Expression::ArrowFunctionExpression(a) => {
                let mut cd = Dump::new(None);
                let cc = cluster_stmt(&mut cd, stmt);
                (Some(&*a.params), Some(&a.body.statements[..]), cc)
            }
            Expression::FunctionExpression(f) => {
                let body = f.body.as_deref()?;
                let mut cd = Dump::new(None);
                let cc = cluster_stmt(&mut cd, stmt);
                (Some(&*f.params), Some(&body.statements[..]), cc)
            }
            _ => return None,
        }
    } else {
        return None;
    };

    let (params, body_stmts) = (params?, body_stmts?);

    let mut collect = Collect::default();
    collect.add_params(params);
    for s in body_stmts {
        collect.collect_stmt(s);
    }
    let locals = collect.bound;

    // Single Dump over the same stmt with rename mode on — produces xname canonical, lines, and
    // the node count from one walk.
    let mut xd = Dump::new(Some(&locals));
    let xname = cluster_stmt(&mut xd, stmt);
    let size = xd.count;

    // Type-3 lines: re-walk the body with a fresh Dump (independent rename map, so the line
    // numbering stays per-line) emitting one renamed sexpr per top-level statement. Cosine
    // matching is order-invariant within the function — same per-statement strings, same IDF
    // weighting, regardless of whether two near-copies emit the same line in slot 4 vs slot 7.
    let mut line_d = Dump::new(Some(&locals));
    let lines: Vec<String> = body_stmts.iter().map(|s| line_d.stmt(s)).collect();

    Some((cluster_canonical, xname, lines, size))
}

#[must_use]
pub fn analyze_functions(texts: &[String]) -> Vec<Option<AnalyzedFn>> {
    texts.par_iter().map(|t| analyze_one(t)).collect()
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
        assert_eq!(aa.1, bb.1, "xname canonicals should match across alpha-renaming\n  a: {}\n  b: {}", aa.1, bb.1);
    }

    #[test]
    fn xname_distinguishes_different_bodies() {
        let a = "function add(x: number, y: number) { return x + y; }";
        let b = "function add(x: number, y: number) { return x - y; }";
        let aa = analyze_one(a).expect("a parses");
        let bb = analyze_one(b).expect("b parses");
        assert_ne!(aa.1, bb.1, "different operators must produce different xname canonicals");
    }

    #[test]
    fn cluster_canonical_preserves_names() {
        let src = "function add(x: number, y: number) { return x + y; }";
        let cc = ast_canonical(src);
        assert!(cc.contains("'add'"), "cluster canonical should keep the def name, got: {cc}");
        assert!(cc.contains("'x'") && cc.contains("'y'"), "param names should be preserved: {cc}");
    }

    #[test]
    fn analyze_lines_are_per_statement() {
        let src = "function f(x: number, y: number) {\n  const z = x + y;\n  return z * 2;\n}";
        let (_, _, lines, size) = analyze_one(src).expect("parses");
        assert_eq!(lines.len(), 2, "got lines: {lines:?}");
        assert!(size > 0);
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
    fn arrow_const_analyzes_as_function() {
        let src = "const fetch = async (x: number): Promise<number> => { return x + 1; };";
        let analysis = analyze_one(src);
        assert!(analysis.is_some(), "arrow-const should analyze");
        let (_, xname, lines, _) = analysis.unwrap();
        assert!(xname.contains("Arrow"));
        assert_eq!(lines.len(), 1, "single-stmt body, one line");
    }
}
