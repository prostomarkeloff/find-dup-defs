//! **Semantic normal form** of a function, taken before its name-agnostic canonicals.
//!
//! The exact pass equates two definitions when their alpha-renamed canonicals are equal, and the
//! Type-3 pass when their statements mostly are. Both read the code as *written*, and the same
//! procedure is written many ways: `if not ok: return err` beside `if ok: ... else: return err`, a
//! loop that appends beside a comprehension, a temporary named and used once beside the expression
//! inlined, `not (a and b)` beside `not a or not b`. Each of those is a Type-4 clone to a reader and
//! an unrelated body to a pass that compares text.
//!
//! This module rewrites a function into one representative of its equivalence class under a fixed
//! set of rules — a *normal form* — so that the passes compare meaning where the rules reach and text
//! elsewhere. The rules are oriented (every rewrite goes one way) and applied to a fixed point, which
//! is the elementary form of the canonicalization that equality saturation (Tate et al. 2009;
//! Willsey et al. 2021) generalizes: no search over a space of equivalent programs, one confluent
//! direction per rule. Type-2 canonicalization of a clone detector (Roy, Cordy & Koschke 2009)
//! already erases names, literals' spelling and layout; this extends it from spelling to structure.
//!
//! ## Rules
//!
//! Expressions, innermost first, to a fixed point:
//! - **Negation normal form**: `not not x` → `x`; `not (a and b)` → `not a or not b` (and dually);
//!   `not (x is y)` → `x is not y`, `not (x in y)` → `x not in y`; `a or (b or c)` → `a or b or c`.
//! - **Comparison orientation**: a symmetric comparison (`==`, `!=`, `is`, `is not`) and its mirror
//!   pair (`<`/`>`, `<=`/`>=`) puts its operands in a fixed order, by their shape with the
//!   definition's own names blanked — so a renamed copy orients the same way.
//! - **Conditional orientation**: `b if not c else a` → `a if c else b`; a test with more negated
//!   than plain operands is flipped. `t if t else b` → `t or b`, `b if t else t` → `t and b`.
//!
//! Statements, in every block:
//! - **Guard form**: `if c: <ends in return/raise/continue/break> else: B` → `if c: …` then `B`;
//!   mirrored when the `else` is the branch that ends.
//! - **Branch orientation** of a two-branch `if` whose test is negated.
//! - **Conditional assignment / return**: `if c: x = a else: x = b` → `x = a if c else b`;
//!   `x = b` then `if c: x = a` likewise; `if c: return a` then `return b` → `return a if c else b`.
//! - **Accumulating loop**: `xs = []` then `for v in it: [if p:] xs.append(e)` → a comprehension
//!   (`set()`/`.add`, `{}`/`xs[k] = v` likewise).
//! - **Single-use temporary**: `t = e` then a statement that reads `t` once and nothing else does
//!   → the statement with `e` in place of `t`.
//! - **`match` on values** → the `if`/`elif` ladder it abbreviates.
//! - `a, b = x, y` → `a = x` then `b = y`; `x += e` → `x = x + e`; `return None` → `return`.
//! - **Dead statements**: `pass` beside other statements, a string expression that is not the
//!   docstring, an assignment of an effect-free value to a name nothing reads.
//!
//! ## What the rules assume
//!
//! The normalized tree is compared, never run, so a rule need only preserve meaning *as a reader
//! judges it*: `x += e` and `x = x + e` are one intent though `__iadd__` may differ; the order in
//! which two adjacent statements evaluate their operands is not what distinguishes two definitions.
//! Both are modelling decisions of the same kind as `is` and `==` being one relation in the
//! decisions pass, and they are listed here so that a surprising equality can be traced to its rule.
//! Everything else is exact under the usual static reading: name, attribute and subscript loads are
//! pure, and `not`/`and`/`or` have their Boolean meaning.
#![allow(clippy::doc_markdown)]

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use ruff_python_ast::comparable::ComparableExpr;
use ruff_python_ast::visitor::transformer::{self as transform, Transformer};
use ruff_python_ast::visitor::{self, Visitor};
use ruff_python_ast::{
    self as ast, AtomicNodeIndex, BoolOp, CmpOp, Expr, ExprContext, Pattern, Singleton, Stmt, UnaryOp,
};
use ruff_text_size::TextRange;

/// Rounds of the rewrite before it is cut off. Every rule either shrinks the tree or fixes an
/// orientation that no rule undoes, so the fixed point comes in a handful of rounds; the bound only
/// guards against a rule pair that would disagree.
const MAX_ROUNDS: usize = 12;

/// The normal form of a definition: a rewritten copy for a function, the node itself otherwise.
///
/// `inline` admits the single-use-temporary rule. The exact pass wants it: a value named and used
/// once is the value, and two definitions differing only in that naming are one. The Type-3 pass
/// does not: its unit is the statement, and folding `t = f(x)` into the assertion that reads `t`
/// turns two short units into one long one, so a near-copy that differs inside the call loses the
/// whole unit instead of a part. Measured on a labelled set of real near-copies, the rule cost a
/// fifth of them; everything else here gained or was neutral. So the two passes read two forms.
#[must_use]
pub(crate) fn normal_form(stmt: &Stmt, inline: bool) -> Stmt {
    let mut copy = stmt.clone();
    if let Stmt::FunctionDef(func) = &mut copy {
        function(func, inline);
    }
    copy
}

/// Continue an already-normalized function to the fixed point that also inlines temporaries —
/// the exact pass's form, from the Type-3 pass's, without starting over.
#[must_use]
pub(crate) fn with_inlining(stmt: &Stmt) -> Stmt {
    normal_form(stmt, true)
}

fn function(func: &mut ast::StmtFunctionDef, inline: bool) {
    nest_elifs(&mut func.body);
    for _ in 0..MAX_ROUNDS {
        let scope = Scope::of(func);
        let mut changed = block(&mut func.body, &scope, true, inline);
        changed |= expressions(&mut func.body, &scope);
        if !changed {
            break;
        }
    }
    if func.body.len() > 1 {
        if let Some(Stmt::Return(ast::StmtReturn { value: None, .. })) = func.body.last() {
            func.body.pop();
        }
    }
    flatten_elifs(&mut func.body);
}

// ---------------------------------------------------------------------------
// Node builders — fresh nodes carry no source position.
// ---------------------------------------------------------------------------

fn range() -> TextRange {
    TextRange::default()
}

fn index() -> AtomicNodeIndex {
    AtomicNodeIndex::default()
}

fn name(id: &str, ctx: ExprContext) -> Expr {
    Expr::Name(ast::ExprName { node_index: index(), range: range(), id: ast::name::Name::new(id), ctx })
}

fn not(operand: Expr) -> Expr {
    Expr::UnaryOp(ast::ExprUnaryOp { node_index: index(), range: range(), op: UnaryOp::Not, operand: Box::new(operand) })
}

fn bool_op(op: BoolOp, values: Vec<Expr>) -> Expr {
    Expr::BoolOp(ast::ExprBoolOp { node_index: index(), range: range(), op, values })
}

fn compare(left: Expr, op: CmpOp, right: Expr) -> Expr {
    Expr::Compare(ast::ExprCompare {
        node_index: index(),
        range: range(),
        left: Box::new(left),
        ops: Box::new([op]),
        comparators: Box::new([right]),
    })
}

fn if_stmt(test: Expr, body: Vec<Stmt>, orelse: Vec<Stmt>) -> Stmt {
    let clauses = if orelse.is_empty() {
        Vec::new()
    } else {
        vec![ast::ElifElseClause { range: range(), node_index: index(), test: None, body: orelse }]
    };
    Stmt::If(ast::StmtIf { node_index: index(), range: range(), test: Box::new(test), body, elif_else_clauses: clauses })
}

fn assign(target: Expr, value: Expr) -> Stmt {
    Stmt::Assign(ast::StmtAssign { node_index: index(), range: range(), targets: vec![target], value: Box::new(value) })
}

fn return_stmt(value: Option<Expr>) -> Stmt {
    Stmt::Return(ast::StmtReturn { node_index: index(), range: range(), value: value.map(Box::new) })
}

/// The same target expression read instead of written.
fn as_load(expr: &Expr) -> Expr {
    let mut copy = expr.clone();
    match &mut copy {
        Expr::Name(n) => n.ctx = ExprContext::Load,
        Expr::Attribute(a) => a.ctx = ExprContext::Load,
        Expr::Subscript(s) => s.ctx = ExprContext::Load,
        _ => {}
    }
    copy
}

// ---------------------------------------------------------------------------
// `elif` ⇄ nested `else: if` — the rules see one shape, the canonical gets the other.
// ---------------------------------------------------------------------------

/// Rewrite every `elif` chain into an `else` holding a nested `if`, recursively.
fn nest_elifs(body: &mut [Stmt]) {
    for stmt in body {
        for_each_block(stmt, &mut |b| nest_elifs(b));
        if let Stmt::If(node) = stmt {
            let clauses = std::mem::take(&mut node.elif_else_clauses);
            let mut orelse: Vec<Stmt> = Vec::new();
            for clause in clauses.into_iter().rev() {
                orelse = match clause.test {
                    None => clause.body,
                    Some(test) => vec![if_stmt(test, clause.body, orelse)],
                };
            }
            if !orelse.is_empty() {
                node.elif_else_clauses = vec![ast::ElifElseClause { range: range(), node_index: index(), test: None, body: orelse }];
            }
        }
    }
}

/// Fold an `else` whose only statement is an `if` back into `elif` clauses, recursively.
fn flatten_elifs(body: &mut [Stmt]) {
    for stmt in body {
        for_each_block(stmt, &mut |b| flatten_elifs(b));
        if let Stmt::If(node) = stmt {
            flatten_if(node);
        }
    }
}

fn flatten_if(node: &mut ast::StmtIf) {
    loop {
        let Some(last) = node.elif_else_clauses.last_mut() else { return };
        if last.test.is_some() || last.body.len() != 1 || !matches!(last.body[0], Stmt::If(_)) {
            return;
        }
        let Some(ast::ElifElseClause { body, .. }) = node.elif_else_clauses.pop() else { return };
        let Some(Stmt::If(inner)) = body.into_iter().next() else { return };
        node.elif_else_clauses.push(ast::ElifElseClause { range: range(), node_index: index(), test: Some(*inner.test), body: inner.body });
        node.elif_else_clauses.extend(inner.elif_else_clauses);
    }
}

/// The `else` body of an `if` in the nested shape (empty when there is none).
fn orelse_of(node: &mut ast::StmtIf) -> &mut Vec<Stmt> {
    if node.elif_else_clauses.is_empty() {
        node.elif_else_clauses.push(ast::ElifElseClause { range: range(), node_index: index(), test: None, body: Vec::new() });
    }
    &mut node.elif_else_clauses[0].body
}

/// Exchange the two branches of an `if` in the nested shape.
fn swap_branches(node: &mut ast::StmtIf) {
    orelse_of(node);
    std::mem::swap(&mut node.body, &mut node.elif_else_clauses[0].body);
}

fn take_orelse(node: &mut ast::StmtIf) -> Vec<Stmt> {
    let out = std::mem::take(orelse_of(node));
    node.elif_else_clauses.clear();
    out
}

fn has_orelse(node: &ast::StmtIf) -> bool {
    node.elif_else_clauses.first().is_some_and(|c| !c.body.is_empty())
}

// ---------------------------------------------------------------------------
// Walking statements
// ---------------------------------------------------------------------------

/// Every statement list directly inside `stmt` — not those of a nested definition, which is a
/// scope of its own and is normalized as one.
fn for_each_block(stmt: &mut Stmt, f: &mut dyn FnMut(&mut Vec<Stmt>)) {
    match stmt {
        Stmt::If(node) => {
            f(&mut node.body);
            for clause in &mut node.elif_else_clauses {
                f(&mut clause.body);
            }
        }
        Stmt::For(node) => {
            f(&mut node.body);
            f(&mut node.orelse);
        }
        Stmt::While(node) => {
            f(&mut node.body);
            f(&mut node.orelse);
        }
        Stmt::With(node) => f(&mut node.body),
        Stmt::Try(node) => {
            f(&mut node.body);
            for ast::ExceptHandler::ExceptHandler(handler) in &mut node.handlers {
                f(&mut handler.body);
            }
            f(&mut node.orelse);
            f(&mut node.finalbody);
        }
        Stmt::Match(node) => {
            for case in &mut node.cases {
                f(&mut case.body);
            }
        }
        _ => {}
    }
}

/// The expressions a statement evaluates itself, before any nested statement runs — the slots the
/// expression rules rewrite and a temporary may be inlined into. Annotations are types, not values,
/// and are left alone.
fn for_each_direct_expr(stmt: &mut Stmt, f: &mut dyn FnMut(&mut Expr)) {
    match stmt {
        Stmt::Assign(node) => {
            f(&mut node.value);
            for t in &mut node.targets {
                f(t);
            }
        }
        Stmt::AnnAssign(node) => {
            if let Some(v) = &mut node.value {
                f(v);
            }
            f(&mut node.target);
        }
        Stmt::AugAssign(node) => {
            f(&mut node.value);
            f(&mut node.target);
        }
        Stmt::Return(node) => {
            if let Some(v) = &mut node.value {
                f(v);
            }
        }
        Stmt::Expr(node) => f(&mut node.value),
        Stmt::If(node) => f(&mut node.test),
        Stmt::While(node) => f(&mut node.test),
        Stmt::For(node) => {
            f(&mut node.iter);
            f(&mut node.target);
        }
        Stmt::With(node) => {
            for item in &mut node.items {
                f(&mut item.context_expr);
                if let Some(v) = &mut item.optional_vars {
                    f(v);
                }
            }
        }
        Stmt::Raise(node) => {
            if let Some(e) = &mut node.exc {
                f(e);
            }
            if let Some(c) = &mut node.cause {
                f(c);
            }
        }
        Stmt::Assert(node) => {
            f(&mut node.test);
            if let Some(m) = &mut node.msg {
                f(m);
            }
        }
        Stmt::Match(node) => {
            f(&mut node.subject);
            for case in &mut node.cases {
                if let Some(g) = &mut case.guard {
                    f(g);
                }
            }
        }
        Stmt::Delete(node) => {
            for t in &mut node.targets {
                f(t);
            }
        }
        _ => {}
    }
}

/// Whether control never falls out of the bottom of `stmt`.
fn is_terminal(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Return(_) | Stmt::Raise(_) | Stmt::Continue(_) | Stmt::Break(_) => true,
        Stmt::If(node) => {
            has_orelse(node)
                && node.body.last().is_some_and(is_terminal)
                && node.elif_else_clauses.iter().all(|c| c.body.last().is_some_and(is_terminal))
        }
        Stmt::With(node) => node.body.last().is_some_and(is_terminal),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Scope facts: what the function binds and reads, counted once per round.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Scope {
    /// Names the function binds anywhere (parameters included).
    locals: HashSet<String>,
    reads: HashMap<String, usize>,
    stores: HashMap<String, usize>,
    /// Reads from inside a lambda, comprehension or nested definition — a closure may see a later
    /// binding, so a temporary read there is not inlined.
    nested_reads: HashSet<String>,
    /// `global`, `nonlocal` and `del` names: nothing is assumed about them.
    pinned: HashSet<String>,
}

impl Scope {
    fn of(func: &ast::StmtFunctionDef) -> Self {
        let mut scope = Scope::default();
        let params = &func.parameters;
        for p in params.posonlyargs.iter().chain(&params.args).chain(&params.kwonlyargs) {
            scope.bind(p.parameter.name.id.as_str());
        }
        for p in params.vararg.iter().chain(&params.kwarg) {
            scope.bind(p.name.id.as_str());
        }
        let mut walker = ScopeWalk { scope: &mut scope, depth: 0 };
        for stmt in &func.body {
            walker.visit_stmt(stmt);
        }
        scope
    }

    fn bind(&mut self, id: &str) {
        self.locals.insert(id.to_owned());
        *self.stores.entry(id.to_owned()).or_insert(0) += 1;
    }

    fn reads(&self, id: &str) -> usize {
        self.reads.get(id).copied().unwrap_or(0)
    }

    fn stores(&self, id: &str) -> usize {
        self.stores.get(id).copied().unwrap_or(0)
    }
}

struct ScopeWalk<'s> {
    scope: &'s mut Scope,
    depth: usize,
}

impl<'a> Visitor<'a> for ScopeWalk<'_> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::FunctionDef(f) => {
                self.scope.bind(f.name.id.as_str());
                self.depth += 1;
                visitor::walk_stmt(self, stmt);
                self.depth -= 1;
                return;
            }
            Stmt::ClassDef(c) => {
                self.scope.bind(c.name.id.as_str());
                self.depth += 1;
                visitor::walk_stmt(self, stmt);
                self.depth -= 1;
                return;
            }
            Stmt::Global(g) => g.names.iter().for_each(|n| {
                self.scope.pinned.insert(n.id.as_str().to_owned());
            }),
            Stmt::Nonlocal(g) => g.names.iter().for_each(|n| {
                self.scope.pinned.insert(n.id.as_str().to_owned());
            }),
            Stmt::Import(node) => {
                for alias in &node.names {
                    let id = alias.asname.as_ref().map_or_else(|| alias.name.id.split('.').next().unwrap_or(""), |a| a.id.as_str());
                    self.scope.bind(id);
                }
            }
            Stmt::ImportFrom(node) => {
                for alias in &node.names {
                    self.scope.bind(alias.asname.as_ref().unwrap_or(&alias.name).id.as_str());
                }
            }
            _ => {}
        }
        visitor::walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr {
            Expr::Name(n) => match n.ctx {
                ExprContext::Load => {
                    *self.scope.reads.entry(n.id.as_str().to_owned()).or_insert(0) += 1;
                    if self.depth > 0 {
                        self.scope.nested_reads.insert(n.id.as_str().to_owned());
                    }
                }
                ExprContext::Store => self.scope.bind(n.id.as_str()),
                ExprContext::Del | ExprContext::Invalid => {
                    self.scope.pinned.insert(n.id.as_str().to_owned());
                }
            },
            Expr::Lambda(_) | Expr::ListComp(_) | Expr::SetComp(_) | Expr::DictComp(_) | Expr::Generator(_) => {
                self.depth += 1;
                visitor::walk_expr(self, expr);
                self.depth -= 1;
                return;
            }
            _ => {}
        }
        visitor::walk_expr(self, expr);
    }

    fn visit_except_handler(&mut self, handler: &'a ast::ExceptHandler) {
        let ast::ExceptHandler::ExceptHandler(h) = handler;
        if let Some(n) = &h.name {
            self.scope.bind(n.id.as_str());
        }
        visitor::walk_except_handler(self, handler);
    }

    fn visit_pattern(&mut self, pattern: &'a Pattern) {
        match pattern {
            Pattern::MatchAs(p) => {
                if let Some(n) = &p.name {
                    self.scope.bind(n.id.as_str());
                }
            }
            Pattern::MatchStar(p) => {
                if let Some(n) = &p.name {
                    self.scope.bind(n.id.as_str());
                }
            }
            Pattern::MatchMapping(p) => {
                if let Some(n) = &p.rest {
                    self.scope.bind(n.id.as_str());
                }
            }
            _ => {}
        }
        visitor::walk_pattern(self, pattern);
    }
}

/// Names read anywhere in `expr`.
fn names_read(expr: &Expr) -> HashSet<String> {
    struct Reads(HashSet<String>);
    impl<'a> Visitor<'a> for Reads {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if let Expr::Name(n) = expr {
                self.0.insert(n.id.as_str().to_owned());
            }
            visitor::walk_expr(self, expr);
        }
    }
    let mut reads = Reads(HashSet::new());
    reads.visit_expr(expr);
    reads.0
}

/// Whether evaluating `expr` could do anything but read: a call, an await, a yield, a walrus. A
/// lambda is a value, not a call, and its body is not looked into.
fn has_effects(expr: &Expr) -> bool {
    struct Effects(bool);
    impl<'a> Visitor<'a> for Effects {
        fn visit_expr(&mut self, expr: &'a Expr) {
            match expr {
                Expr::Call(_) | Expr::Await(_) | Expr::Yield(_) | Expr::YieldFrom(_) | Expr::Named(_) => self.0 = true,
                Expr::Lambda(_) => return,
                _ => {}
            }
            visitor::walk_expr(self, expr);
        }
    }
    let mut effects = Effects(false);
    effects.visit_expr(expr);
    effects.0
}

fn same(a: &Expr, b: &Expr) -> bool {
    ComparableExpr::from(a) == ComparableExpr::from(b)
}

fn simple_target(stmt: &Stmt) -> Option<&str> {
    if let Stmt::Assign(node) = stmt {
        if let [Expr::Name(n)] = node.targets.as_slice() {
            return Some(n.id.as_str());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Expression rules
// ---------------------------------------------------------------------------

/// A shape key with the definition's own names blanked — what the orientation rules order by. Two
/// renamed copies of one expression have one key; a key tie keeps the written order.
fn key(expr: &Expr, locals: &HashSet<String>) -> String {
    struct Blank<'l>(&'l HashSet<String>);
    impl Transformer for Blank<'_> {
        fn visit_expr(&self, expr: &mut Expr) {
            if let Expr::Name(n) = expr {
                if self.0.contains(n.id.as_str()) {
                    n.id = ast::name::Name::new("_");
                }
            }
            transform::walk_expr(self, expr);
        }
    }
    let mut copy = expr.clone();
    Blank(locals).visit_expr(&mut copy);
    format!("{:?}", ComparableExpr::from(&copy))
}

/// Whether a test reads as a negation at its top.
fn is_negative(expr: &Expr) -> bool {
    match expr {
        Expr::UnaryOp(u) => u.op == UnaryOp::Not,
        Expr::Compare(c) => matches!(c.ops.as_ref(), [CmpOp::NotEq | CmpOp::IsNot | CmpOp::NotIn]),
        _ => false,
    }
}

/// The exact inverse of a test, in negation normal form.
fn negate(expr: Expr) -> Expr {
    match expr {
        Expr::UnaryOp(u) if u.op == UnaryOp::Not => *u.operand,
        Expr::Compare(mut c) if c.ops.len() == 1 => {
            let flipped = match c.ops[0] {
                CmpOp::Eq => Some(CmpOp::NotEq),
                CmpOp::NotEq => Some(CmpOp::Eq),
                CmpOp::Is => Some(CmpOp::IsNot),
                CmpOp::IsNot => Some(CmpOp::Is),
                CmpOp::In => Some(CmpOp::NotIn),
                CmpOp::NotIn => Some(CmpOp::In),
                _ => None,
            };
            match flipped {
                Some(op) => {
                    c.ops = Box::new([op]);
                    Expr::Compare(c)
                }
                None => not(Expr::Compare(c)),
            }
        }
        Expr::BoolOp(b) => {
            let op = if b.op == BoolOp::And { BoolOp::Or } else { BoolOp::And };
            bool_op(op, b.values.into_iter().map(negate).collect())
        }
        Expr::BooleanLiteral(mut l) => {
            l.value = !l.value;
            Expr::BooleanLiteral(l)
        }
        other => not(other),
    }
}

/// Whether a two-way test should be flipped: it is a negation, or most of its operands are, or at
/// a tie the flipped form keys lower.
fn prefer_flip(test: &Expr, locals: &HashSet<String>) -> bool {
    if is_negative(test) {
        return true;
    }
    if let Expr::BoolOp(b) = test {
        let negatives = b.values.iter().filter(|v| is_negative(v)).count();
        let positives = b.values.len() - negatives;
        if negatives != positives {
            return negatives > positives;
        }
        return key(&negate(test.clone()), locals) < key(test, locals);
    }
    false
}

struct Exprs<'l> {
    locals: &'l HashSet<String>,
    changed: Cell<bool>,
}

impl Exprs<'_> {
    fn mark(&self) {
        self.changed.set(true);
    }

    fn rewrite(&self, expr: &mut Expr) {
        match expr {
            Expr::UnaryOp(u) if u.op == UnaryOp::Not => {
                let reducible = match u.operand.as_ref() {
                    Expr::UnaryOp(inner) => inner.op == UnaryOp::Not,
                    Expr::Compare(c) => matches!(c.ops.as_ref(), [CmpOp::Eq | CmpOp::NotEq | CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn]),
                    Expr::BoolOp(_) | Expr::BooleanLiteral(_) => true,
                    _ => false,
                };
                if reducible {
                    let operand = std::mem::replace(&mut *u.operand, name("_", ExprContext::Load));
                    *expr = negate(operand);
                    self.mark();
                }
            }
            Expr::BoolOp(b) if b.values.iter().any(|v| matches!(v, Expr::BoolOp(inner) if inner.op == b.op)) => {
                let op = b.op;
                let mut flat = Vec::with_capacity(b.values.len() + 1);
                for v in std::mem::take(&mut b.values) {
                    match v {
                        Expr::BoolOp(inner) if inner.op == op => flat.extend(inner.values),
                        other => flat.push(other),
                    }
                }
                b.values = flat;
                self.mark();
            }
            Expr::Compare(c) if c.ops.len() == 1 => {
                let mirror = match c.ops[0] {
                    CmpOp::Eq | CmpOp::NotEq | CmpOp::Is | CmpOp::IsNot => Some(c.ops[0]),
                    CmpOp::Lt => Some(CmpOp::Gt),
                    CmpOp::Gt => Some(CmpOp::Lt),
                    CmpOp::LtE => Some(CmpOp::GtE),
                    CmpOp::GtE => Some(CmpOp::LtE),
                    CmpOp::In | CmpOp::NotIn => None,
                };
                if let Some(op) = mirror {
                    if key(&c.comparators[0], self.locals) < key(&c.left, self.locals) {
                        std::mem::swap(&mut *c.left, &mut c.comparators[0]);
                        c.ops = Box::new([op]);
                        self.mark();
                    }
                }
            }
            Expr::If(node) => {
                if prefer_flip(&node.test, self.locals) {
                    let test = std::mem::replace(&mut *node.test, name("_", ExprContext::Load));
                    *node.test = negate(test);
                    std::mem::swap(&mut node.body, &mut node.orelse);
                    self.mark();
                }
                if same(&node.test, &node.body) {
                    let test = std::mem::replace(&mut *node.test, name("_", ExprContext::Load));
                    let orelse = std::mem::replace(&mut *node.orelse, name("_", ExprContext::Load));
                    *expr = bool_op(BoolOp::Or, vec![test, orelse]);
                    self.mark();
                } else if same(&node.test, &node.orelse) {
                    let test = std::mem::replace(&mut *node.test, name("_", ExprContext::Load));
                    let body = std::mem::replace(&mut *node.body, name("_", ExprContext::Load));
                    *expr = bool_op(BoolOp::And, vec![test, body]);
                    self.mark();
                }
            }
            _ => {}
        }
    }
}

impl Transformer for Exprs<'_> {
    fn visit_expr(&self, expr: &mut Expr) {
        transform::walk_expr(self, expr);
        self.rewrite(expr);
    }
}

/// Apply the expression rules to every statement's own expressions, in every block.
fn expressions(body: &mut [Stmt], scope: &Scope) -> bool {
    fn walk(body: &mut [Stmt], rules: &Exprs<'_>) {
        for stmt in body {
            if matches!(stmt, Stmt::FunctionDef(_) | Stmt::ClassDef(_)) {
                continue;
            }
            for_each_direct_expr(stmt, &mut |e| rules.visit_expr(e));
            for_each_block(stmt, &mut |b| walk(b, rules));
        }
    }
    let rules = Exprs { locals: &scope.locals, changed: Cell::new(false) };
    walk(body, &rules);
    rules.changed.get()
}

// ---------------------------------------------------------------------------
// Statement rules
// ---------------------------------------------------------------------------

/// One round of the statement rules over a block and the blocks inside it. `top` marks the
/// function's own body, whose first string statement is the docstring and is left for the dump.
fn block(stmts: &mut Vec<Stmt>, scope: &Scope, top: bool, inline: bool) -> bool {
    let mut changed = false;
    for stmt in stmts.iter_mut() {
        match stmt {
            Stmt::FunctionDef(func) => function(func, inline),
            Stmt::ClassDef(class) => {
                for inner in &mut class.body {
                    if let Stmt::FunctionDef(func) = inner {
                        function(func, inline);
                    }
                }
            }
            _ => for_each_block(stmt, &mut |b| changed |= block(b, scope, false, inline)),
        }
    }
    changed |= dead_statements(stmts, top);
    let mut i = 0;
    while i < stmts.len() {
        let rewritten = unread_assignment(stmts, i, scope)
            || expand_conditional(stmts, i)
            || merge_assign_if(stmts, i)
            || guard_form(stmts, i, scope)
            || ladder_to_boolop(stmts, i)
            || orient_guard(stmts, i, scope)
            || accumulating_loop(stmts, i)
            || (inline && inline_temporary(stmts, i, scope))
            || split_tuple_assign(stmts, i)
            || expand_aug_assign(stmts, i)
            || match_to_ladder(stmts, i)
            || bare_return(stmts, i)
            || orient_branches(stmts, i, scope);
        if rewritten {
            changed = true;
        } else {
            i += 1;
        }
    }
    changed
}

/// `pass` beside other statements, a non-docstring string statement. (An effect-free assignment to
/// a name nothing reads goes in [`unread_assignment`], which needs the scope.)
fn dead_statements(stmts: &mut Vec<Stmt>, top: bool) -> bool {
    let before = stmts.len();
    let mut idx = 0;
    stmts.retain(|s| {
        let at = idx;
        idx += 1;
        let dead = match s {
            Stmt::Pass(_) => before > 1,
            Stmt::Expr(e) => matches!(e.value.as_ref(), Expr::StringLiteral(_)) && !(top && at == 0),
            _ => false,
        };
        !dead
    });
    // A block that held statements must keep one; an optional block that was empty stays so.
    if stmts.is_empty() && before > 0 {
        stmts.push(Stmt::Pass(ast::StmtPass { node_index: index(), range: range() }));
    }
    stmts.len() != before
}

/// An effect-free assignment to a name nothing reads.
fn unread_assignment(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope) -> bool {
    let Some(target) = simple_target(&stmts[i]) else { return false };
    let Stmt::Assign(node) = &stmts[i] else { return false };
    let dead = scope.reads(target) == 0 && !scope.pinned.contains(target) && !has_effects(&node.value) && stmts.len() > 1;
    if dead {
        stmts.remove(i);
    }
    dead
}

/// `if c: <terminal> else: B` → `if c: <terminal>` then `B`; when only the `else` ends, the test is
/// negated and the branches change places first; when both end, the branch orientation decides
/// which one becomes the guard.
fn guard_form(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope) -> bool {
    let Stmt::If(node) = &mut stmts[i] else { return false };
    if !has_orelse(node) {
        return false;
    }
    let then_ends = node.body.last().is_some_and(is_terminal);
    let else_ends = node.elif_else_clauses[0].body.last().is_some_and(is_terminal);
    if !then_ends && !else_ends {
        return false;
    }
    let flip = if then_ends && else_ends { prefer_flip(&node.test, &scope.locals) } else { !then_ends };
    if flip {
        let test = std::mem::replace(&mut *node.test, name("_", ExprContext::Load));
        *node.test = negate(test);
        swap_branches(node);
    }
    let tail = take_orelse(node);
    let at = i + 1;
    stmts.splice(at..at, tail);
    true
}

/// `x = b` then `if c: x = a` → `if c: x = a else: x = b`: the two-branch statement is the form the
/// conditional-assignment rules meet at. (`x` read in `c` or `a` would see the old value; left alone.)
fn merge_assign_if(stmts: &mut Vec<Stmt>, i: usize) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Some(x) = simple_target(&stmts[i]).map(str::to_owned) else { return false };
    let Stmt::If(node) = &stmts[i + 1] else { return false };
    if has_orelse(node) || node.body.len() != 1 || simple_target(&node.body[0]) != Some(&x) {
        return false;
    }
    let Stmt::Assign(then) = &node.body[0] else { return false };
    if names_read(&node.test).contains(&x) || names_read(&then.value).contains(&x) {
        return false;
    }
    let Stmt::If(mut node) = stmts.remove(i + 1) else { return false };
    let first = stmts.remove(i);
    *orelse_of(&mut node) = vec![first];
    stmts.insert(i, Stmt::If(node));
    true
}

/// A conditional expression that is a whole statement's value becomes the statement form:
/// `x = a if c else b` → `if c: x = a else: x = b`, `return a if c else b` → `if c: return a else:
/// return b`. The statement form is the normal one because the Type-3 pass counts statements: the
/// guard a reader writes as three lines and the conditional another writes as one must agree on
/// as many units as possible, and three is more than one. A conditional nested inside a larger
/// expression has no statement form and stays.
fn expand_conditional(stmts: &mut [Stmt], i: usize) -> bool {
    let split = |value: Expr| -> Option<(Expr, Expr, Expr)> {
        if let Expr::If(cond) = value {
            Some((*cond.test, *cond.body, *cond.orelse))
        } else {
            None
        }
    };
    match &stmts[i] {
        Stmt::Assign(node) if node.targets.len() == 1 && matches!(node.value.as_ref(), Expr::If(_)) => {
            let Stmt::Assign(mut node) = std::mem::replace(&mut stmts[i], pass()) else { return false };
            let target = node.targets.pop().expect("one target");
            let Some((test, a, b)) = split(*node.value) else { return false };
            stmts[i] = if_stmt(test, vec![assign(target.clone(), a)], vec![assign(target, b)]);
            true
        }
        Stmt::Return(node) if matches!(node.value.as_deref(), Some(Expr::If(_))) => {
            let Stmt::Return(node) = std::mem::replace(&mut stmts[i], pass()) else { return false };
            let Some((test, a, b)) = node.value.and_then(|v| split(*v)) else { return false };
            stmts[i] = if_stmt(test, vec![return_stmt(Some(a))], vec![return_stmt(Some(b))]);
            true
        }
        _ => false,
    }
}

/// `if t: return t` then `return b` → `return t or b`; `if not t: return t` then `return b` →
/// `return t and b`. The ladder spells the short-circuit out; the operator is its one form.
fn ladder_to_boolop(stmts: &mut Vec<Stmt>, i: usize) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Stmt::If(node) = &stmts[i] else { return false };
    if has_orelse(node) || node.body.len() != 1 {
        return false;
    }
    let (Stmt::Return(ast::StmtReturn { value: Some(returned), .. }), Stmt::Return(ast::StmtReturn { value: Some(_), .. })) = (&node.body[0], &stmts[i + 1]) else {
        return false;
    };
    let op = if same(&node.test, returned) {
        BoolOp::Or
    } else if matches!(node.test.as_ref(), Expr::UnaryOp(u) if u.op == UnaryOp::Not && same(&u.operand, returned)) {
        BoolOp::And
    } else {
        return false;
    };
    let Stmt::Return(fallback) = stmts.remove(i + 1) else { return false };
    let Stmt::If(mut node) = stmts.remove(i) else { return false };
    let Stmt::Return(then) = node.body.pop().expect("one statement") else { return false };
    let (Some(first), Some(rest)) = (then.value, fallback.value) else { return false };
    stmts.insert(i, return_stmt(Some(bool_op(op, vec![*first, *rest]))));
    true
}

/// A guard whose remainder also ends is a two-branch `if` written as a guard: `if c: <ends>` then
/// `<ends>` ≡ `if not c: <ends>` then `<ends>`. The branch orientation decides which is the guard.
fn orient_guard(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Stmt::If(node) = &stmts[i] else { return false };
    if has_orelse(node) || !node.body.last().is_some_and(is_terminal) || !stmts.last().is_some_and(is_terminal) {
        return false;
    }
    if !prefer_flip(&node.test, &scope.locals) {
        return false;
    }
    let rest: Vec<Stmt> = stmts.drain(i + 1..).collect();
    let Stmt::If(mut node) = stmts.pop().expect("the guard") else { return false };
    let test = std::mem::replace(&mut *node.test, name("_", ExprContext::Load));
    *node.test = negate(test);
    let body = std::mem::replace(&mut node.body, rest);
    stmts.push(Stmt::If(node));
    stmts.extend(body);
    true
}

fn pass() -> Stmt {
    Stmt::Pass(ast::StmtPass { node_index: index(), range: range() })
}

/// The shape of a loop body that accumulates: which container method, the element(s), the filter.
struct Accumulate<'a> {
    container: &'a str,
    method: &'static str,
    cond: Option<&'a Expr>,
    key: Option<&'a Expr>,
    value: &'a Expr,
}

fn accumulate_step(stmt: &Stmt) -> Option<Accumulate<'_>> {
    let (cond, step) = match stmt {
        Stmt::If(node) if !has_orelse(node) && node.body.len() == 1 => (Some(node.test.as_ref()), &node.body[0]),
        other => (None, other),
    };
    match step {
        Stmt::Expr(e) => {
            let Expr::Call(call) = e.value.as_ref() else { return None };
            let Expr::Attribute(attr) = call.func.as_ref() else { return None };
            let Expr::Name(container) = attr.value.as_ref() else { return None };
            let method = match attr.attr.id.as_str() {
                "append" => "append",
                "add" => "add",
                _ => return None,
            };
            if call.arguments.args.len() != 1 || !call.arguments.keywords.is_empty() {
                return None;
            }
            Some(Accumulate { container: container.id.as_str(), method, cond, key: None, value: &call.arguments.args[0] })
        }
        Stmt::Assign(node) => {
            let [Expr::Subscript(sub)] = node.targets.as_slice() else { return None };
            let Expr::Name(container) = sub.value.as_ref() else { return None };
            Some(Accumulate { container: container.id.as_str(), method: "setitem", cond, key: Some(&sub.slice), value: &node.value })
        }
        _ => None,
    }
}

fn empty_container(expr: &Expr) -> Option<&'static str> {
    match expr {
        Expr::List(l) if l.elts.is_empty() => Some("append"),
        Expr::Dict(d) if d.items.is_empty() => Some("setitem"),
        Expr::Call(c) if c.arguments.args.is_empty() && c.arguments.keywords.is_empty() => match c.func.as_ref() {
            Expr::Name(n) if n.id.as_str() == "set" => Some("add"),
            Expr::Name(n) if n.id.as_str() == "dict" => Some("setitem"),
            _ => None,
        },
        _ => None,
    }
}

/// `xs = []` then `for v in it: [if p:] xs.append(e)` → `xs = [e for v in it if p]`.
fn accumulating_loop(stmts: &mut Vec<Stmt>, i: usize) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Some(container) = simple_target(&stmts[i]) else { return false };
    let Stmt::Assign(init) = &stmts[i] else { return false };
    let Some(method) = empty_container(&init.value) else { return false };
    let Stmt::For(loop_) = &stmts[i + 1] else { return false };
    if loop_.body.len() != 1 || !loop_.orelse.is_empty() {
        return false;
    }
    let Some(step) = accumulate_step(&loop_.body[0]) else { return false };
    if step.container != container || step.method != method {
        return false;
    }
    let mentions = |e: &Expr| names_read(e).contains(container);
    if mentions(&loop_.iter) || mentions(step.value) || step.cond.is_some_and(mentions) || step.key.is_some_and(mentions) {
        return false;
    }
    let Stmt::For(loop_) = stmts.remove(i + 1) else { return false };
    let Stmt::Assign(init) = &mut stmts[i] else { return false };
    let is_async = loop_.is_async;
    let Some(step) = loop_.body.into_iter().next() else { return false };
    let (cond, step) = match step {
        Stmt::If(mut node) => (Some(*node.test), node.body.pop().expect("one statement")),
        other => (None, other),
    };
    let generators = vec![ast::Comprehension { range: range(), node_index: index(), target: *loop_.target, iter: *loop_.iter, ifs: cond.into_iter().collect(), is_async }];
    let value = match step {
        Stmt::Expr(e) => {
            let Expr::Call(call) = *e.value else { return false };
            let elt = Box::new(call.arguments.args.into_vec().pop().expect("one argument"));
            if method == "append" {
                Expr::ListComp(ast::ExprListComp { node_index: index(), range: range(), elt, generators })
            } else {
                Expr::SetComp(ast::ExprSetComp { node_index: index(), range: range(), elt, generators })
            }
        }
        Stmt::Assign(mut node) => {
            let Expr::Subscript(sub) = node.targets.pop().expect("one target") else { return false };
            Expr::DictComp(ast::ExprDictComp { node_index: index(), range: range(), key: sub.slice, value: node.value, generators })
        }
        _ => return false,
    };
    *init.value = value;
    true
}

/// `t = e` then `S(t)`, with `t` bound once and read once, there → `S(e)`.
fn inline_temporary(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Some(t) = simple_target(&stmts[i]).map(str::to_owned) else { return false };
    let Stmt::Assign(node) = &stmts[i] else { return false };
    if scope.stores(&t) != 1 || scope.reads(&t) != 1 || scope.pinned.contains(&t) || scope.nested_reads.contains(&t) {
        return false;
    }
    if matches!(stmts[i + 1], Stmt::While(_) | Stmt::FunctionDef(_) | Stmt::ClassDef(_)) {
        return false;
    }
    if names_read(&node.value).contains(&t) {
        return false;
    }
    let mut direct = 0;
    for_each_direct_expr(&mut stmts[i + 1], &mut |e| {
        direct += names_read_count(e, &t);
    });
    if direct != 1 {
        return false;
    }
    let Stmt::Assign(node) = stmts.remove(i) else { return false };
    let subst = Subst { name: t, value: Cell::new(Some(*node.value)) };
    for_each_direct_expr(&mut stmts[i], &mut |e| subst.visit_expr(e));
    true
}

fn names_read_count(expr: &Expr, id: &str) -> usize {
    struct Count<'n>(&'n str, usize);
    impl<'a> Visitor<'a> for Count<'_> {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if let Expr::Name(n) = expr {
                if n.ctx == ExprContext::Load && n.id.as_str() == self.0 {
                    self.1 += 1;
                }
            }
            visitor::walk_expr(self, expr);
        }
    }
    let mut count = Count(id, 0);
    count.visit_expr(expr);
    count.1
}

struct Subst {
    name: String,
    value: Cell<Option<Expr>>,
}

impl Transformer for Subst {
    fn visit_expr(&self, expr: &mut Expr) {
        if let Expr::Name(n) = expr {
            if n.ctx == ExprContext::Load && n.id.as_str() == self.name {
                if let Some(value) = self.value.take() {
                    *expr = value;
                }
                return;
            }
        }
        transform::walk_expr(self, expr);
    }
}

/// `a, b = x, y` → `a = x` then `b = y`, unless a later value reads an earlier target.
fn split_tuple_assign(stmts: &mut Vec<Stmt>, i: usize) -> bool {
    let Stmt::Assign(node) = &stmts[i] else { return false };
    let [Expr::Tuple(targets)] = node.targets.as_slice() else { return false };
    let Expr::Tuple(values) = node.value.as_ref() else { return false };
    if targets.elts.len() != values.elts.len() || targets.elts.len() < 2 {
        return false;
    }
    let names: Vec<&str> = targets.elts.iter().filter_map(|t| if let Expr::Name(n) = t { Some(n.id.as_str()) } else { None }).collect();
    if names.len() != targets.elts.len() {
        return false;
    }
    for (k, value) in values.elts.iter().enumerate() {
        let reads = names_read(value);
        if names[..k].iter().any(|n| reads.contains(*n)) {
            return false;
        }
    }
    let Stmt::Assign(mut node) = stmts.remove(i) else { return false };
    let Expr::Tuple(targets) = node.targets.pop().expect("one target") else { return false };
    let Expr::Tuple(values) = *node.value else { return false };
    let pairs: Vec<Stmt> = targets.elts.into_iter().zip(values.elts).map(|(t, v)| assign(t, v)).collect();
    stmts.splice(i..i, pairs);
    true
}

/// `x += e` → `x = x + e`.
fn expand_aug_assign(stmts: &mut [Stmt], i: usize) -> bool {
    let Stmt::AugAssign(node) = &stmts[i] else { return false };
    let load = as_load(&node.target);
    let Stmt::AugAssign(node) = std::mem::replace(&mut stmts[i], Stmt::Pass(ast::StmtPass { node_index: index(), range: range() })) else { return false };
    let value = Expr::BinOp(ast::ExprBinOp { node_index: index(), range: range(), left: Box::new(load), op: node.op, right: node.value });
    stmts[i] = assign(*node.target, value);
    true
}

/// `return None` → `return`.
fn bare_return(stmts: &mut [Stmt], i: usize) -> bool {
    if let Stmt::Return(node) = &mut stmts[i] {
        if matches!(node.value.as_deref(), Some(Expr::NoneLiteral(_))) {
            node.value = None;
            return true;
        }
    }
    false
}

/// A two-branch `if` whose test is negated has its branches swapped.
fn orient_branches(stmts: &mut [Stmt], i: usize, scope: &Scope) -> bool {
    let Stmt::If(node) = &mut stmts[i] else { return false };
    if !has_orelse(node) || !prefer_flip(&node.test, &scope.locals) {
        return false;
    }
    let test = std::mem::replace(&mut *node.test, name("_", ExprContext::Load));
    *node.test = negate(test);
    swap_branches(node);
    true
}

/// `match x: case A: … case B | C: … case _: …` → `if x == A: … elif x == B or x == C: … else: …`,
/// when every case is a value, a singleton or an `or` of those, without guards, and only the last
/// may be the wildcard.
fn match_to_ladder(stmts: &mut [Stmt], i: usize) -> bool {
    let Stmt::Match(node) = &stmts[i] else { return false };
    let last = node.cases.len().saturating_sub(1);
    let convertible = node.cases.iter().enumerate().all(|(k, case)| {
        case.guard.is_none() && (value_tests(&case.pattern, &node.subject).is_some() || (k == last && is_wildcard(&case.pattern)))
    });
    if !convertible || node.cases.is_empty() {
        return false;
    }
    let Stmt::Match(node) = std::mem::replace(&mut stmts[i], Stmt::Pass(ast::StmtPass { node_index: index(), range: range() })) else { return false };
    let mut orelse: Vec<Stmt> = Vec::new();
    for case in node.cases.into_iter().rev() {
        if is_wildcard(&case.pattern) {
            orelse = case.body;
            continue;
        }
        let tests = value_tests(&case.pattern, &node.subject).expect("checked above");
        let test = if tests.len() == 1 { tests.into_iter().next().expect("one") } else { bool_op(BoolOp::Or, tests) };
        orelse = vec![if_stmt(test, case.body, orelse)];
    }
    let Some(ladder) = orelse.into_iter().next() else { return false };
    stmts[i] = ladder;
    true
}

fn is_wildcard(pattern: &Pattern) -> bool {
    matches!(pattern, Pattern::MatchAs(p) if p.pattern.is_none() && p.name.is_none())
}

/// The equality tests a value pattern stands for, one per alternative.
fn value_tests(pattern: &Pattern, subject: &Expr) -> Option<Vec<Expr>> {
    match pattern {
        Pattern::MatchValue(p) => Some(vec![compare(subject.clone(), CmpOp::Eq, p.value.as_ref().clone())]),
        Pattern::MatchSingleton(p) => {
            let literal = match p.value {
                Singleton::None => Expr::NoneLiteral(ast::ExprNoneLiteral::default()),
                Singleton::True => Expr::BooleanLiteral(ast::ExprBooleanLiteral { value: true, ..Default::default() }),
                Singleton::False => Expr::BooleanLiteral(ast::ExprBooleanLiteral { value: false, ..Default::default() }),
            };
            Some(vec![compare(subject.clone(), CmpOp::Is, literal)])
        }
        Pattern::MatchOr(p) => {
            let mut out = Vec::new();
            for alt in &p.patterns {
                out.extend(value_tests(alt, subject)?);
            }
            Some(out)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::analyze_functions;

    fn xname(src: &str) -> String {
        analyze_functions(&[src.to_owned()]).pop().flatten().expect("a function").xname_canonical
    }

    fn lines(src: &str) -> Vec<String> {
        analyze_functions(&[src.to_owned()]).pop().flatten().expect("a function").type3_lines
    }

    #[track_caller]
    fn equal(a: &str, b: &str) {
        assert_eq!(xname(a), xname(b), "\n--- a\n{a}\n--- b\n{b}\n--- lines a\n{:#?}\n--- lines b\n{:#?}", lines(a), lines(b));
    }

    #[track_caller]
    fn differ(a: &str, b: &str) {
        assert_ne!(xname(a), xname(b), "\n--- a\n{a}\n--- b\n{b}");
    }

    #[test]
    fn negation_normal_form() {
        equal(
            "def f(order, user):\n    if not (order.open and user.active):\n        return None\n    return order\n",
            "def f(order, user):\n    if not order.open or not user.active:\n        return None\n    return order\n",
        );
        equal("def f(x):\n    return not x is None\n", "def f(x):\n    return x is not None\n");
        equal("def f(x, xs):\n    return not x in xs\n", "def f(x, xs):\n    return x not in xs\n");
        equal("def f(x):\n    return not not x\n", "def f(x):\n    return x\n");
    }

    #[test]
    fn comparison_orientation() {
        equal("def f(order):\n    return order.total > 10\n", "def f(order):\n    return 10 < order.total\n");
        equal("def f(order):\n    return order.status == Status.OPEN\n", "def f(order):\n    return Status.OPEN == order.status\n");
        differ("def f(order):\n    return order.total > 10\n", "def f(order):\n    return order.total < 10\n");
    }

    #[test]
    fn branch_orientation() {
        equal(
            "def f(order):\n    if order.paid:\n        label = 'paid'\n    else:\n        label = 'due'\n    send(label)\n",
            "def f(order):\n    if not order.paid:\n        label = 'due'\n    else:\n        label = 'paid'\n    send(label)\n",
        );
        differ(
            "def f(order):\n    if order.paid:\n        send('paid')\n    else:\n        send('due')\n",
            "def f(order):\n    if order.paid:\n        send('due')\n    else:\n        send('paid')\n",
        );
    }

    #[test]
    fn guard_form() {
        equal(
            "def f(order):\n    if order is None:\n        raise ValueError('no order')\n    total = order.total\n    return total * 2\n",
            "def f(order):\n    if order is not None:\n        total = order.total\n        return total * 2\n    else:\n        raise ValueError('no order')\n",
        );
        equal(
            "def f(order):\n    if order.open:\n        return order\n    return None\n",
            "def f(order):\n    if order.open:\n        return order\n    else:\n        return None\n",
        );
    }

    #[test]
    fn conditional_value_forms() {
        equal(
            "def f(user):\n    role = 'admin' if user.is_admin else 'member'\n    return role\n",
            "def f(user):\n    if user.is_admin:\n        role = 'admin'\n    else:\n        role = 'member'\n    return role\n",
        );
        equal(
            "def f(user):\n    role = 'member'\n    if user.is_admin:\n        role = 'admin'\n    return role\n",
            "def f(user):\n    role = 'admin' if user.is_admin else 'member'\n    return role\n",
        );
        equal(
            "def f(user):\n    if user.is_admin:\n        return 'admin'\n    return 'member'\n",
            "def f(user):\n    return 'admin' if user.is_admin else 'member'\n",
        );
    }

    #[test]
    fn or_ladder_is_an_or_expression() {
        equal(
            "def f(user):\n    if user.nickname:\n        return user.nickname\n    if user.email:\n        return user.email\n    return user.id\n",
            "def f(user):\n    return user.nickname or user.email or user.id\n",
        );
        equal(
            "def f(user):\n    if not user.nickname:\n        return user.nickname\n    return user.email\n",
            "def f(user):\n    return user.nickname and user.email\n",
        );
    }

    #[test]
    fn accumulating_loop_is_a_comprehension() {
        equal(
            "def f(orders):\n    open_ids = []\n    for order in orders:\n        if order.open:\n            open_ids.append(order.id)\n    return open_ids\n",
            "def f(orders):\n    open_ids = [order.id for order in orders if order.open]\n    return open_ids\n",
        );
        equal(
            "def f(orders):\n    by_id = {}\n    for order in orders:\n        by_id[order.id] = order\n    return by_id\n",
            "def f(orders):\n    by_id = {order.id: order for order in orders}\n    return by_id\n",
        );
        equal(
            "def f(orders):\n    seen = set()\n    for order in orders:\n        seen.add(order.user_id)\n    return seen\n",
            "def f(orders):\n    seen = {order.user_id for order in orders}\n    return seen\n",
        );
    }

    #[test]
    fn single_use_temporary_is_inlined() {
        equal(
            "def f(order):\n    message = f'order {order.id} is closed'\n    raise ValueError(message)\n",
            "def f(order):\n    raise ValueError(f'order {order.id} is closed')\n",
        );
        equal(
            "def f(user):\n    profile = load_profile(user.id)\n    return render(profile, user.locale)\n",
            "def f(user):\n    return render(load_profile(user.id), user.locale)\n",
        );
        // read twice: stays a temporary
        differ(
            "def f(user):\n    profile = load_profile(user.id)\n    return render(profile, profile.locale)\n",
            "def f(user):\n    return render(load_profile(user.id), load_profile(user.id).locale)\n",
        );
    }

    #[test]
    fn tuple_assignment_splits_and_aug_assign_expands() {
        equal("def f(a, b):\n    x, y = a + 1, b + 1\n    return x * y\n", "def f(a, b):\n    x = a + 1\n    y = b + 1\n    return x * y\n");
        differ("def f(a, b):\n    a, b = b, a\n    return a - b\n", "def f(a, b):\n    a = b\n    b = a\n    return a - b\n");
        equal("def f(total, item):\n    total += item.price\n    return total\n", "def f(total, item):\n    total = total + item.price\n    return total\n");
    }

    #[test]
    fn dead_statements_vanish() {
        equal(
            "def f(order):\n    pass\n    unused = 0\n    total = order.total\n    return total\n",
            "def f(order):\n    total = order.total\n    return total\n",
        );
        equal("def f(order):\n    return None\n", "def f(order):\n    return\n");
    }

    #[test]
    fn match_on_values_is_a_ladder() {
        equal(
            "def f(event):\n    match event.kind:\n        case 'open':\n            status = 'open'\n        case 'close' | 'cancel':\n            status = 'closed'\n        case _:\n            raise ValueError(event.kind)\n    return status\n",
            "def f(event):\n    if event.kind == 'open':\n        status = 'open'\n    elif event.kind == 'close' or event.kind == 'cancel':\n        status = 'closed'\n    else:\n        raise ValueError(event.kind)\n    return status\n",
        );
    }

    #[test]
    fn elif_and_nested_else_if_agree_and_keep_the_elif_lines() {
        equal(
            "def f(x):\n    if x == 1:\n        a()\n    elif x == 2:\n        b()\n    else:\n        c()\n",
            "def f(x):\n    if x == 1:\n        a()\n    else:\n        if x == 2:\n            b()\n        else:\n            c()\n",
        );
        assert!(lines("def f(x):\n    if x == 1:\n        a()\n    elif x == 2:\n        b()\n").iter().any(|l| l.starts_with("elif ")));
    }

    #[test]
    fn a_guard_and_its_mirror_image_are_one_guard() {
        // Both branches end; which one is the guard is decided by the test's polarity, not by the
        // writer.
        equal(
            "def f(order):\n    if order is None:\n        raise ValueError('no order')\n    return order.total\n",
            "def f(order):\n    if order is not None:\n        return order.total\n    raise ValueError('no order')\n",
        );
    }

    #[test]
    fn a_nested_conditional_orients_and_folds() {
        equal("def f(c, a, b):\n    return g(a if c else b)\n", "def f(c, a, b):\n    return g(b if not c else a)\n");
        equal("def f(x, y):\n    return g(x if x else y)\n", "def f(x, y):\n    return g(x or y)\n");
    }

    #[test]
    fn the_type3_lines_keep_the_temporary_the_canonical_inlines() {
        let named = "def f(order):\n    message = f'order {order.id} is closed'\n    raise ValueError(message)\n";
        let inlined = "def f(order):\n    raise ValueError(f'order {order.id} is closed')\n";
        assert_eq!(xname(named), xname(inlined));
        assert_ne!(lines(named), lines(inlined), "the statement is the Type-3 unit, and two statements stay two");
    }

    #[test]
    fn a_renamed_copy_orients_like_its_original() {
        // Both operands are locals: the key ties and the written order is kept — on both sides.
        equal("def f(a, b):\n    return a < b\n", "def f(p, q):\n    return p < q\n");
        equal("def f(a, b):\n    if b > a:\n        return b\n    return a\n", "def f(x, y):\n    if y > x:\n        return y\n    return x\n");
    }
}
