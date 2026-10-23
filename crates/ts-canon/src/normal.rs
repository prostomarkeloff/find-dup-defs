//! **Semantic normal form** of a callable's body, taken before its name-agnostic canonicals.
//!
//! The exact pass equates two definitions when their alpha-renamed canonicals are equal, and the
//! Type-3 pass when their statements mostly are. Both read the code as *written*, and the same
//! procedure is written many ways: `if (!ok) return e;` beside `if (ok) { … } else { return e; }`, a
//! loop that pushes beside a `filter` / `map` chain, a temporary named and used once beside the
//! expression inlined, `!(a && b)` beside `!a || !b`. Each of those is a Type-4 clone to a reader and
//! an unrelated body to a pass that compares text.
//!
//! This module rewrites a body into one representative of its equivalence class under the rules the
//! Python frontend's normal form applies — TypeScript shares Python's statement-level `if` and
//! `return`, so the rules carry over almost word for word. Oriented, applied to a fixed point, one
//! confluent direction per rule.
//!
//! ## Rules
//!
//! First, **braces**: every `if` branch and loop body is a block — `if (c) return x;` and
//! `if (c) { return x; }` are one statement.
//!
//! Expressions, innermost first, to a fixed point:
//! - **Negation normal form**: `!!x` → `x`; `!(a && b)` → `!a || !b` (and dually); `!(a === b)` →
//!   `a !== b`; `a || (b || c)` → `a || b || c`.
//! - **Comparison orientation**: the equalities and the mirror pairs (`<`/`>`, `<=`/`>=`) put their
//!   operands in a fixed order, by their shape with the definition's own names blanked.
//! - **Conditional orientation**: `!c ? b : a` → `c ? a : b`; `t ? t : b` → `t || b`,
//!   `t ? b : t` → `t && b`.
//!
//! Statements, in every block:
//! - **Guard form**: `if (c) { <ends> } else { B }` → `if (c) { <ends> }` then `B`; mirrored when the
//!   `else` is the branch that ends. **Branch orientation** of a two-branch `if` whose test is negated.
//! - **Conditional assignment / return**: `x = c ? a : b` and `return c ? a : b` take the statement
//!   form; `x = b;` then `if (c) { x = a; }` likewise. `if (t) { return t; }` then `return b;` →
//!   `return t || b;`.
//! - **Accumulating loop**: `const xs = [];` then `for (const v of it) { [if (p) {] xs.push(e); }` →
//!   `const xs = it.filter((v) => p).map((v) => e);` — the chain is TypeScript's comprehension;
//!   `new Set()` and `.add` → `new Set(it.map(…))`.
//! - **Single-use temporary** (the exact form only): `const t = e;` then a statement that reads `t`
//!   once and nothing else does → that statement with `e` in place of `t`.
//! - **`switch` on values** → the `if` / `else if` ladder it abbreviates, when no case falls through
//!   and `default` is last.
//! - `x += e` → `x = x + e`; `return undefined` → `return`; a final bare `return` is dropped.
//! - **Dead statements**: an empty statement, a string expression statement, a declaration of an
//!   effect-free value to a name nothing reads.
//!
//! ## What the rules assume
//!
//! The normalized tree is compared, never run: a rule need only preserve meaning as a reader judges
//! it — `x += e` is `x = x + e`, a loop over an iterable is a `map` over it, `!!x` in a test is `x`.

use std::collections::HashSet;

use oxc_allocator::{CloneIn, TakeIn, Vec as ArenaVec};
use oxc_ast::ast::{
    self, Argument, BindingPattern, Expression, ForStatementLeft, FormalParameterKind, FormalParameters,
    SimpleAssignmentTarget, Statement,
};
use oxc_ast::{AstBuilder, NONE};
use oxc_ast_visit::{walk, walk_mut, Visit, VisitMut};
use oxc_span::{ContentEq, SPAN};
use oxc_syntax::operator::{AssignmentOperator, BinaryOperator, LogicalOperator, UnaryOperator};
use oxc_syntax::scope::ScopeFlags;

use dup_defs_core::normal::{Scope, MAX_ROUNDS};

/// Rewrite `body` to its normal form in place. `inline` admits the single-use-temporary rule: the
/// exact pass wants it, the Type-3 pass does not — its unit is the statement, and folding
/// `const t = f(x);` into the line that reads `t` turns two short units into one long one.
pub(crate) fn normalize<'a>(
    ast: AstBuilder<'a>,
    params: &FormalParameters<'a>,
    body: &mut ArenaVec<'a, Statement<'a>>,
    inline: bool,
) {
    let mut braces = Braces { ast };
    braces.visit_statements(body);
    for _ in 0..MAX_ROUNDS {
        let scope = scope_of(params, body);
        let mut changed = block(ast, body, &scope, inline);
        changed |= expressions(ast, body, &scope);
        if !changed {
            break;
        }
    }
    if body.len() > 1 && matches!(body.last(), Some(Statement::ReturnStatement(r)) if r.argument.is_none()) {
        body.pop();
    }
    Flatten.visit_statements(body);
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

/// Every `if` branch and loop body a block; an `else if` an `else` holding the `if` — every rule then
/// sees two-branch `if`s over blocks only. Nested functions are scopes of their own.
struct Braces<'a> {
    ast: AstBuilder<'a>,
}

fn block_up<'a>(ast: AstBuilder<'a>, stmt: &mut Statement<'a>) {
    if !matches!(stmt, Statement::BlockStatement(_)) {
        let inner = stmt.take_in(ast.allocator);
        *stmt = ast.statement_block(SPAN, ast.vec1(inner));
    }
}

impl<'a> VisitMut<'a> for Braces<'a> {
    fn visit_statement(&mut self, it: &mut Statement<'a>) {
        let ast = self.ast;
        match it {
            Statement::IfStatement(node) => {
                block_up(ast, &mut node.consequent);
                if let Some(alt) = &mut node.alternate {
                    block_up(ast, alt);
                }
            }
            Statement::ForStatement(node) => block_up(ast, &mut node.body),
            Statement::ForInStatement(node) => block_up(ast, &mut node.body),
            Statement::ForOfStatement(node) => block_up(ast, &mut node.body),
            Statement::WhileStatement(node) => block_up(ast, &mut node.body),
            Statement::DoWhileStatement(node) => block_up(ast, &mut node.body),
            _ => {}
        }
        walk_mut::walk_statement(self, it);
    }

    fn visit_function(&mut self, _: &mut ast::Function<'a>, _: ScopeFlags) {}

    fn visit_arrow_function_expression(&mut self, _: &mut ast::ArrowFunctionExpression<'a>) {}

    fn visit_class(&mut self, _: &mut ast::Class<'a>) {}
}

/// An `else` whose only statement is an `if` back to `else if`.
struct Flatten;

impl<'a> VisitMut<'a> for Flatten {
    fn visit_if_statement(&mut self, it: &mut ast::IfStatement<'a>) {
        walk_mut::walk_if_statement(self, it);
        if let Some(Statement::BlockStatement(b)) = &mut it.alternate {
            if matches!(b.body.as_slice(), [Statement::IfStatement(_)]) {
                it.alternate = b.body.pop();
            }
        }
    }

    fn visit_function(&mut self, _: &mut ast::Function<'a>, _: ScopeFlags) {}

    fn visit_arrow_function_expression(&mut self, _: &mut ast::ArrowFunctionExpression<'a>) {}

    fn visit_class(&mut self, _: &mut ast::Class<'a>) {}
}

// ---------------------------------------------------------------------------
// Scope facts: what the function binds and reads, counted once per round.
// ---------------------------------------------------------------------------

/// What the function binds and reads — the shared [`Scope`] counts, filled by this language's walk.
fn scope_of(params: &FormalParameters<'_>, body: &[Statement<'_>]) -> Scope {
    let mut scope = Scope::default();
    let mut walk = ScopeWalk { scope: &mut scope, depth: 0 };
    walk.visit_formal_parameters(params);
    for stmt in body {
        walk.visit_statement(stmt);
    }
    scope
}

struct ScopeWalk<'s> {
    scope: &'s mut Scope,
    depth: usize,
}

impl<'a> Visit<'a> for ScopeWalk<'_> {
    fn visit_binding_identifier(&mut self, it: &ast::BindingIdentifier<'a>) {
        self.scope.bind(it.name.as_str());
    }

    fn visit_identifier_reference(&mut self, it: &ast::IdentifierReference<'a>) {
        self.scope.read(it.name.as_str(), self.depth > 0);
    }

    fn visit_assignment_expression(&mut self, it: &ast::AssignmentExpression<'a>) {
        match it.left.as_simple_assignment_target() {
            Some(SimpleAssignmentTarget::AssignmentTargetIdentifier(id)) => {
                self.scope.store(id.name.as_str());
                // A compound assignment reads what it writes.
                if it.operator != AssignmentOperator::Assign {
                    self.visit_identifier_reference(id);
                }
                self.visit_expression(&it.right);
            }
            _ => walk::walk_assignment_expression(self, it),
        }
    }

    fn visit_update_expression(&mut self, it: &ast::UpdateExpression<'a>) {
        if let SimpleAssignmentTarget::AssignmentTargetIdentifier(id) = &it.argument {
            self.scope.store(id.name.as_str());
        }
        walk::walk_update_expression(self, it);
    }

    fn visit_function(&mut self, it: &ast::Function<'a>, flags: ScopeFlags) {
        self.depth += 1;
        walk::walk_function(self, it, flags);
        self.depth -= 1;
    }

    fn visit_arrow_function_expression(&mut self, it: &ast::ArrowFunctionExpression<'a>) {
        self.depth += 1;
        walk::walk_arrow_function_expression(self, it);
        self.depth -= 1;
    }
}

/// Names read anywhere in `expr`.
fn names_read(expr: &Expression<'_>) -> HashSet<String> {
    struct Reads(HashSet<String>);
    impl<'a> Visit<'a> for Reads {
        fn visit_identifier_reference(&mut self, it: &ast::IdentifierReference<'a>) {
            self.0.insert(it.name.to_string());
        }
    }
    let mut reads = Reads(HashSet::new());
    reads.visit_expression(expr);
    reads.0
}

fn reads_of(expr: &Expression<'_>, name: &str) -> usize {
    struct Count<'n>(&'n str, usize);
    impl<'a> Visit<'a> for Count<'_> {
        fn visit_identifier_reference(&mut self, it: &ast::IdentifierReference<'a>) {
            if it.name == self.0 {
                self.1 += 1;
            }
        }
    }
    let mut count = Count(name, 0);
    count.visit_expression(expr);
    count.1
}

/// Whether evaluating `expr` could do anything but read: a call, a `new`, an `await`, a `yield`, an
/// assignment, an update, a `delete`. A function is a value, not a call, and its body is not looked
/// into.
fn has_effects(expr: &Expression<'_>) -> bool {
    struct Effects(bool);
    impl<'a> Visit<'a> for Effects {
        fn visit_expression(&mut self, it: &Expression<'a>) {
            match it {
                Expression::CallExpression(_)
                | Expression::NewExpression(_)
                | Expression::AwaitExpression(_)
                | Expression::YieldExpression(_)
                | Expression::AssignmentExpression(_)
                | Expression::UpdateExpression(_)
                | Expression::TaggedTemplateExpression(_)
                | Expression::ImportExpression(_) => self.0 = true,
                Expression::UnaryExpression(u) if u.operator == UnaryOperator::Delete => self.0 = true,
                Expression::ArrowFunctionExpression(_) | Expression::FunctionExpression(_) | Expression::ClassExpression(_) => {
                    return;
                }
                _ => {}
            }
            walk::walk_expression(self, it);
        }
    }
    let mut effects = Effects(false);
    effects.visit_expression(expr);
    effects.0
}

// ---------------------------------------------------------------------------
// Expression rules
// ---------------------------------------------------------------------------

fn strip<'x, 'a>(expr: &'x Expression<'a>) -> &'x Expression<'a> {
    match expr {
        Expression::ParenthesizedExpression(p) => strip(&p.expression),
        other => other,
    }
}

fn into_stripped(expr: Expression<'_>) -> Expression<'_> {
    match expr {
        Expression::ParenthesizedExpression(p) => into_stripped(p.unbox().expression),
        other => other,
    }
}

fn same(a: &Expression<'_>, b: &Expression<'_>) -> bool {
    strip(a).content_eq(strip(b))
}

/// A shape key with the definition's own names blanked — what the orientation rules order by.
fn key(expr: &Expression<'_>, locals: &HashSet<String>) -> String {
    crate::canon::shape_key(expr, locals)
}

/// Whether a test reads as a negation at its top.
fn is_negative(expr: &Expression<'_>) -> bool {
    match strip(expr) {
        Expression::UnaryExpression(u) => u.operator == UnaryOperator::LogicalNot,
        Expression::BinaryExpression(b) => matches!(b.operator, BinaryOperator::Inequality | BinaryOperator::StrictInequality),
        _ => false,
    }
}

fn flip_equality(op: BinaryOperator) -> Option<BinaryOperator> {
    Some(match op {
        BinaryOperator::Equality => BinaryOperator::Inequality,
        BinaryOperator::Inequality => BinaryOperator::Equality,
        BinaryOperator::StrictEquality => BinaryOperator::StrictInequality,
        BinaryOperator::StrictInequality => BinaryOperator::StrictEquality,
        _ => return None,
    })
}

/// The operator a comparison is written with when its operands change places — itself for an
/// equality; `None` for anything that is not a comparison.
fn mirror(op: BinaryOperator) -> Option<BinaryOperator> {
    Some(match op {
        BinaryOperator::Equality
        | BinaryOperator::Inequality
        | BinaryOperator::StrictEquality
        | BinaryOperator::StrictInequality => op,
        BinaryOperator::LessThan => BinaryOperator::GreaterThan,
        BinaryOperator::GreaterThan => BinaryOperator::LessThan,
        BinaryOperator::LessEqualThan => BinaryOperator::GreaterEqualThan,
        BinaryOperator::GreaterEqualThan => BinaryOperator::LessEqualThan,
        _ => return None,
    })
}

/// The exact inverse of a test, in negation normal form.
fn negate<'a>(ast: AstBuilder<'a>, expr: Expression<'a>) -> Expression<'a> {
    match into_stripped(expr) {
        Expression::UnaryExpression(u) if u.operator == UnaryOperator::LogicalNot => into_stripped(u.unbox().argument),
        Expression::BinaryExpression(mut b) if flip_equality(b.operator).is_some() => {
            b.operator = flip_equality(b.operator).unwrap_or(b.operator);
            Expression::BinaryExpression(b)
        }
        Expression::LogicalExpression(l) if matches!(l.operator, LogicalOperator::And | LogicalOperator::Or) => {
            let l = l.unbox();
            let op = if l.operator == LogicalOperator::And { LogicalOperator::Or } else { LogicalOperator::And };
            ast.expression_logical(SPAN, negate(ast, l.left), op, negate(ast, l.right))
        }
        Expression::BooleanLiteral(mut b) => {
            b.value = !b.value;
            Expression::BooleanLiteral(b)
        }
        other => ast.expression_unary(SPAN, UnaryOperator::LogicalNot, other),
    }
}

/// The operands of a chain of one `&&` or `||`, left to right.
fn operands<'x, 'a>(expr: &'x Expression<'a>) -> Vec<&'x Expression<'a>> {
    let Expression::LogicalExpression(l) = strip(expr) else { return vec![expr] };
    let same_op = |e: &Expression<'_>| matches!(strip(e), Expression::LogicalExpression(inner) if inner.operator == l.operator);
    let mut out = if same_op(&l.left) { operands(&l.left) } else { vec![&l.left] };
    out.extend(if same_op(&l.right) { operands(&l.right) } else { vec![&l.right] });
    out
}

/// Whether a two-way test should be flipped: it is a negation, or most of its operands are, or at a
/// tie the flipped form keys lower.
fn prefer_flip<'a>(ast: AstBuilder<'a>, test: &Expression<'a>, locals: &HashSet<String>) -> bool {
    if is_negative(test) {
        return true;
    }
    if let Expression::LogicalExpression(l) = strip(test) {
        if matches!(l.operator, LogicalOperator::And | LogicalOperator::Or) {
            let all = operands(test);
            let negatives = all.iter().filter(|v| is_negative(v)).count();
            let positives = all.len() - negatives;
            if negatives != positives {
                return negatives > positives;
            }
            let flipped = negate(ast, test.clone_in(ast.allocator));
            return key(&flipped, locals) < key(test, locals);
        }
    }
    false
}

struct Exprs<'a, 'l> {
    ast: AstBuilder<'a>,
    locals: &'l HashSet<String>,
    changed: bool,
}

impl<'a> Exprs<'a, '_> {
    fn rewrite(&mut self, expr: &mut Expression<'a>) {
        let ast = self.ast;
        match expr {
            Expression::UnaryExpression(u) if u.operator == UnaryOperator::LogicalNot => {
                let reducible = match strip(&u.argument) {
                    Expression::UnaryExpression(inner) => inner.operator == UnaryOperator::LogicalNot,
                    Expression::BinaryExpression(b) => flip_equality(b.operator).is_some(),
                    Expression::LogicalExpression(l) => matches!(l.operator, LogicalOperator::And | LogicalOperator::Or),
                    Expression::BooleanLiteral(_) => true,
                    _ => false,
                };
                if reducible {
                    let operand = u.argument.take_in(ast.allocator);
                    *expr = negate(ast, operand);
                    self.changed = true;
                }
            }
            Expression::LogicalExpression(l)
                if matches!(l.operator, LogicalOperator::And | LogicalOperator::Or)
                    && matches!(strip(&l.right), Expression::LogicalExpression(r) if r.operator == l.operator) =>
            {
                let op = l.operator;
                let left = l.left.take_in(ast.allocator);
                let Expression::LogicalExpression(right) = into_stripped(l.right.take_in(ast.allocator)) else { return };
                let right = right.unbox();
                *expr = ast.expression_logical(SPAN, ast.expression_logical(SPAN, left, op, right.left), op, right.right);
                self.changed = true;
            }
            Expression::BinaryExpression(b) => {
                if let Some(mirrored) = mirror(b.operator) {
                    if key(&b.right, self.locals) < key(&b.left, self.locals) {
                        let b = &mut **b;
                        std::mem::swap(&mut b.left, &mut b.right);
                        b.operator = mirrored;
                        self.changed = true;
                    }
                }
            }
            Expression::ConditionalExpression(c) => {
                if prefer_flip(ast, &c.test, self.locals) {
                    let test = c.test.take_in(ast.allocator);
                    let c = &mut **c;
                    c.test = negate(ast, test);
                    std::mem::swap(&mut c.consequent, &mut c.alternate);
                    self.changed = true;
                }
                let op = if same(&c.test, &c.consequent) {
                    Some(LogicalOperator::Or)
                } else if same(&c.test, &c.alternate) {
                    Some(LogicalOperator::And)
                } else {
                    None
                };
                if let Some(op) = op {
                    let Expression::ConditionalExpression(c) = expr.take_in(ast.allocator) else { return };
                    let c = c.unbox();
                    let rest = if op == LogicalOperator::Or { c.alternate } else { c.consequent };
                    *expr = ast.expression_logical(SPAN, c.test, op, rest);
                    self.changed = true;
                }
            }
            _ => {}
        }
    }
}

impl<'a> VisitMut<'a> for Exprs<'a, '_> {
    fn visit_expression(&mut self, it: &mut Expression<'a>) {
        walk_mut::walk_expression(self, it);
        self.rewrite(it);
    }

    fn visit_function(&mut self, _: &mut ast::Function<'a>, _: ScopeFlags) {}

    fn visit_class(&mut self, _: &mut ast::Class<'a>) {}
}

/// Apply the expression rules to every expression of the body.
fn expressions<'a>(ast: AstBuilder<'a>, body: &mut ArenaVec<'a, Statement<'a>>, scope: &Scope) -> bool {
    let mut rules = Exprs { ast, locals: &scope.locals, changed: false };
    rules.visit_statements(body);
    rules.changed
}

// ---------------------------------------------------------------------------
// Statement rules
// ---------------------------------------------------------------------------

/// Whether control never falls out of the bottom of `stmt`.
fn ends(stmt: &Statement<'_>) -> bool {
    match stmt {
        Statement::ReturnStatement(_)
        | Statement::ThrowStatement(_)
        | Statement::BreakStatement(_)
        | Statement::ContinueStatement(_) => true,
        Statement::IfStatement(node) => node.alternate.as_ref().is_some_and(ends) && ends(&node.consequent),
        Statement::BlockStatement(b) => b.body.last().is_some_and(ends),
        _ => false,
    }
}

/// Every statement list directly inside `stmt` — not those of a nested function, a scope of its own.
fn for_each_block<'a>(stmt: &mut Statement<'a>, f: &mut dyn FnMut(&mut ArenaVec<'a, Statement<'a>>)) {
    match stmt {
        Statement::BlockStatement(b) => f(&mut b.body),
        Statement::IfStatement(node) => {
            for_each_block(&mut node.consequent, f);
            if let Some(alt) = &mut node.alternate {
                for_each_block(alt, f);
            }
        }
        Statement::ForStatement(node) => for_each_block(&mut node.body, f),
        Statement::ForInStatement(node) => for_each_block(&mut node.body, f),
        Statement::ForOfStatement(node) => for_each_block(&mut node.body, f),
        Statement::WhileStatement(node) => for_each_block(&mut node.body, f),
        Statement::DoWhileStatement(node) => for_each_block(&mut node.body, f),
        Statement::LabeledStatement(node) => for_each_block(&mut node.body, f),
        Statement::TryStatement(node) => {
            f(&mut node.block.body);
            if let Some(handler) = &mut node.handler {
                f(&mut handler.body.body);
            }
            if let Some(finalizer) = &mut node.finalizer {
                f(&mut finalizer.body);
            }
        }
        Statement::SwitchStatement(node) => {
            for case in &mut node.cases {
                f(&mut case.consequent);
            }
        }
        _ => {}
    }
}

/// The expressions a statement evaluates itself, before any statement inside it runs — the slots a
/// temporary may be inlined into.
fn for_each_direct_expr<'a>(stmt: &mut Statement<'a>, f: &mut dyn FnMut(&mut Expression<'a>)) {
    match stmt {
        Statement::ExpressionStatement(s) => f(&mut s.expression),
        Statement::ReturnStatement(r) => {
            if let Some(argument) = &mut r.argument {
                f(argument);
            }
        }
        Statement::ThrowStatement(t) => f(&mut t.argument),
        Statement::VariableDeclaration(d) => {
            for declarator in &mut d.declarations {
                if let Some(init) = &mut declarator.init {
                    f(init);
                }
            }
        }
        Statement::IfStatement(node) => f(&mut node.test),
        Statement::ForOfStatement(node) => f(&mut node.right),
        Statement::ForInStatement(node) => f(&mut node.right),
        Statement::SwitchStatement(node) => f(&mut node.discriminant),
        _ => {}
    }
}

/// One round of the statement rules over a block and the blocks inside it.
fn block<'a>(ast: AstBuilder<'a>, stmts: &mut ArenaVec<'a, Statement<'a>>, scope: &Scope, inline: bool) -> bool {
    let mut changed = false;
    for stmt in stmts.iter_mut() {
        for_each_block(stmt, &mut |b| changed |= block(ast, b, scope, inline));
    }
    changed |= dead_statements(stmts);
    let mut i = 0;
    while i < stmts.len() {
        let rewritten = unread_declaration(stmts, i, scope)
            || expand_conditional(ast, stmts, i)
            || merge_assign_if(ast, stmts, i)
            || guard_form(ast, stmts, i, scope)
            || ladder_to_logical(ast, stmts, i)
            || orient_guard(ast, stmts, i, scope)
            || accumulating_loop(ast, stmts, i)
            || (inline && inline_temporary(stmts, i, scope))
            || expand_compound_assign(ast, stmts, i)
            || switch_to_ladder(ast, stmts, i)
            || bare_return(stmts, i)
            || orient_branches(ast, stmts, i, scope);
        if rewritten {
            changed = true;
        } else {
            i += 1;
        }
    }
    changed
}

/// An empty statement, a string expression statement — when the block holds anything else.
fn dead_statements(stmts: &mut ArenaVec<'_, Statement<'_>>) -> bool {
    let before = stmts.len();
    if before < 2 {
        return false;
    }
    stmts.retain(|s| match s {
        Statement::EmptyStatement(_) => false,
        Statement::ExpressionStatement(e) => !matches!(e.expression, Expression::StringLiteral(_)),
        _ => true,
    });
    stmts.len() != before
}

/// The one name a declaration binds and the value it binds it to.
fn single_declaration<'x, 'a>(stmt: &'x Statement<'a>) -> Option<(&'x str, &'x ast::VariableDeclarator<'a>)> {
    let Statement::VariableDeclaration(d) = stmt else { return None };
    let [declarator] = d.declarations.as_slice() else { return None };
    let BindingPattern::BindingIdentifier(id) = &declarator.id else { return None };
    Some((id.name.as_str(), declarator))
}

/// The name a statement assigns, as `x = …`.
fn simple_assign<'x, 'a>(stmt: &'x Statement<'a>) -> Option<(&'x str, &'x Expression<'a>)> {
    let Statement::ExpressionStatement(s) = stmt else { return None };
    let Expression::AssignmentExpression(a) = &s.expression else { return None };
    if a.operator != AssignmentOperator::Assign {
        return None;
    }
    match a.left.as_simple_assignment_target() {
        Some(SimpleAssignmentTarget::AssignmentTargetIdentifier(id)) => Some((id.name.as_str(), &a.right)),
        _ => None,
    }
}

/// A declaration of an effect-free value to a name nothing reads.
fn unread_declaration(stmts: &mut ArenaVec<'_, Statement<'_>>, i: usize, scope: &Scope) -> bool {
    let Some((name, declarator)) = single_declaration(&stmts[i]) else { return false };
    let Some(init) = &declarator.init else { return false };
    let dead = stmts.len() > 1 && scope.reads(name) == 0 && !has_effects(init);
    if dead {
        stmts.remove(i);
    }
    dead
}

fn statement_block<'a>(ast: AstBuilder<'a>, stmt: Statement<'a>) -> Statement<'a> {
    ast.statement_block(SPAN, ast.vec1(stmt))
}

/// A conditional that is a whole statement's value becomes the statement form: `x = c ? a : b` →
/// `if (c) { x = a; } else { x = b; }`, `return c ? a : b` → `if (c) { return a; } else { return b; }`.
/// The statement form is the normal one because the Type-3 pass counts statements.
fn expand_conditional<'a>(ast: AstBuilder<'a>, stmts: &mut ArenaVec<'a, Statement<'a>>, i: usize) -> bool {
    let is_conditional = |e: &Expression<'_>| matches!(strip(e), Expression::ConditionalExpression(_));
    let assigns = matches!(&stmts[i], Statement::ExpressionStatement(s) if matches!(&s.expression, Expression::AssignmentExpression(a) if a.operator == AssignmentOperator::Assign && is_conditional(&a.right)));
    let returns = matches!(&stmts[i], Statement::ReturnStatement(r) if r.argument.as_ref().is_some_and(is_conditional));
    if !assigns && !returns {
        return false;
    }
    let taken = stmts[i].take_in(ast.allocator);
    let replacement = match taken {
        Statement::ExpressionStatement(s) => {
            let Expression::AssignmentExpression(a) = s.unbox().expression else { return false };
            let a = a.unbox();
            let Expression::ConditionalExpression(c) = into_stripped(a.right) else { return false };
            let c = c.unbox();
            let then = ast.statement_expression(SPAN, ast.expression_assignment(SPAN, AssignmentOperator::Assign, a.left.clone_in(ast.allocator), c.consequent));
            let otherwise = ast.statement_expression(SPAN, ast.expression_assignment(SPAN, AssignmentOperator::Assign, a.left, c.alternate));
            ast.statement_if(SPAN, c.test, statement_block(ast, then), Some(statement_block(ast, otherwise)))
        }
        Statement::ReturnStatement(r) => {
            let Some(argument) = r.unbox().argument else { return false };
            let Expression::ConditionalExpression(c) = into_stripped(argument) else { return false };
            let c = c.unbox();
            let then = ast.statement_return(SPAN, Some(c.consequent));
            let otherwise = ast.statement_return(SPAN, Some(c.alternate));
            ast.statement_if(SPAN, c.test, statement_block(ast, then), Some(statement_block(ast, otherwise)))
        }
        _ => return false,
    };
    stmts[i] = replacement;
    true
}

/// `x = b;` then `if (c) { x = a; }` → `if (c) { x = a; } else { x = b; }`: the two-branch statement
/// is the form the conditional-assignment rules meet at. (`x` read in `c` or `a` would see the old
/// value; left alone.)
fn merge_assign_if<'a>(ast: AstBuilder<'a>, stmts: &mut ArenaVec<'a, Statement<'a>>, i: usize) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Some((x, _)) = simple_assign(&stmts[i]) else { return false };
    let Statement::IfStatement(node) = &stmts[i + 1] else { return false };
    let Statement::BlockStatement(then) = &node.consequent else { return false };
    if node.alternate.is_some() || then.body.len() != 1 {
        return false;
    }
    let Some((y, value)) = simple_assign(&then.body[0]) else { return false };
    if x != y || names_read(&node.test).contains(x) || names_read(value).contains(x) {
        return false;
    }
    let Statement::IfStatement(mut node) = stmts.remove(i + 1) else { return false };
    let first = stmts.remove(i);
    node.alternate = Some(statement_block(ast, first));
    stmts.insert(i, Statement::IfStatement(node));
    true
}

fn swap_branches(node: &mut ast::IfStatement<'_>) {
    if let Some(alt) = node.alternate.take() {
        let consequent = std::mem::replace(&mut node.consequent, alt);
        node.alternate = Some(consequent);
    }
}

fn into_statements(stmt: Statement<'_>) -> Vec<Statement<'_>> {
    match stmt {
        Statement::BlockStatement(b) => b.unbox().body.into_iter().collect(),
        other => vec![other],
    }
}

/// `if (c) { <ends> } else { B }` → `if (c) { <ends> }` then `B`; when only the `else` ends, the test
/// is negated and the branches change places first; when both end, the branch orientation decides
/// which one becomes the guard.
fn guard_form<'a>(ast: AstBuilder<'a>, stmts: &mut ArenaVec<'a, Statement<'a>>, i: usize, scope: &Scope) -> bool {
    let Statement::IfStatement(node) = &mut stmts[i] else { return false };
    let Some(alt) = &node.alternate else { return false };
    let then_ends = ends(&node.consequent);
    let else_ends = ends(alt);
    if !then_ends && !else_ends {
        return false;
    }
    let flip = if then_ends && else_ends { prefer_flip(ast, &node.test, &scope.locals) } else { !then_ends };
    if flip {
        let test = node.test.take_in(ast.allocator);
        node.test = negate(ast, test);
        swap_branches(node);
    }
    let Some(alt) = node.alternate.take() else { return false };
    let tail = into_statements(alt);
    let at = i + 1;
    stmts.splice(at..at, tail);
    true
}

/// `if (t) { return t; }` then `return b;` → `return t || b;`; `if (!t) { return t; }` then
/// `return b;` → `return t && b;`. The ladder spells the short-circuit out; the operator is its one
/// form.
fn ladder_to_logical<'a>(ast: AstBuilder<'a>, stmts: &mut ArenaVec<'a, Statement<'a>>, i: usize) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Statement::IfStatement(node) = &stmts[i] else { return false };
    let Statement::BlockStatement(then) = &node.consequent else { return false };
    if node.alternate.is_some() {
        return false;
    }
    let ([Statement::ReturnStatement(first)], Statement::ReturnStatement(fallback)) = (then.body.as_slice(), &stmts[i + 1]) else {
        return false;
    };
    let (Some(returned), Some(_)) = (&first.argument, &fallback.argument) else { return false };
    let op = if same(&node.test, returned) {
        LogicalOperator::Or
    } else if matches!(strip(&node.test), Expression::UnaryExpression(u) if u.operator == UnaryOperator::LogicalNot && same(&u.argument, returned)) {
        LogicalOperator::And
    } else {
        return false;
    };
    let Statement::ReturnStatement(fallback) = stmts.remove(i + 1) else { return false };
    let Statement::IfStatement(node) = stmts.remove(i) else { return false };
    let Statement::BlockStatement(then) = node.unbox().consequent else { return false };
    let Some(Statement::ReturnStatement(first)) = then.unbox().body.pop() else { return false };
    let (Some(first), Some(rest)) = (first.unbox().argument, fallback.unbox().argument) else { return false };
    stmts.insert(i, ast.statement_return(SPAN, Some(ast.expression_logical(SPAN, first, op, rest))));
    true
}

/// A guard whose remainder also ends is a two-branch `if` written as a guard: `if (c) { <ends> }`
/// then `<ends>` ≡ `if (!c) { <ends> }` then `<ends>`. The branch orientation decides which is the
/// guard.
fn orient_guard<'a>(ast: AstBuilder<'a>, stmts: &mut ArenaVec<'a, Statement<'a>>, i: usize, scope: &Scope) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Statement::IfStatement(node) = &stmts[i] else { return false };
    if node.alternate.is_some() || !ends(&node.consequent) || !stmts.last().is_some_and(ends) {
        return false;
    }
    if !prefer_flip(ast, &node.test, &scope.locals) {
        return false;
    }
    let rest: Vec<Statement<'a>> = stmts.drain(i + 1..).collect();
    let Some(Statement::IfStatement(mut node)) = stmts.pop() else { return false };
    let test = node.test.take_in(ast.allocator);
    node.test = negate(ast, test);
    let guard = std::mem::replace(&mut node.consequent, ast.statement_block(SPAN, ast.vec_from_iter(rest)));
    stmts.push(Statement::IfStatement(node));
    for stmt in into_statements(guard) {
        stmts.push(stmt);
    }
    true
}

/// `(v) => value`.
fn arrow<'a>(ast: AstBuilder<'a>, pattern: BindingPattern<'a>, value: Expression<'a>) -> Expression<'a> {
    let params = ast.alloc_formal_parameters(
        SPAN,
        FormalParameterKind::ArrowFormalParameters,
        ast.vec1(ast.plain_formal_parameter(SPAN, pattern)),
        NONE,
    );
    let body = ast.alloc_function_body(SPAN, ast.vec(), ast.vec1(ast.statement_expression(SPAN, value)));
    ast.expression_arrow_function(SPAN, true, false, NONE, params, NONE, body)
}

/// `object.name(argument)`.
fn method_call<'a>(ast: AstBuilder<'a>, object: Expression<'a>, name: &'static str, argument: Expression<'a>) -> Expression<'a> {
    let callee = Expression::from(ast.member_expression_static(SPAN, object, ast.identifier_name(SPAN, name), false));
    ast.expression_call(SPAN, callee, NONE, ast.vec1(Argument::from(argument)), false)
}

/// The step of a loop that accumulates: which container method, the filter, the element.
fn accumulate_step<'x, 'a>(body: &'x Statement<'a>) -> Option<(&'x str, &'x str, Option<&'x Expression<'a>>, &'x Expression<'a>)> {
    let Statement::BlockStatement(b) = body else { return None };
    let [step] = b.body.as_slice() else { return None };
    let (cond, call) = match step {
        Statement::IfStatement(node) if node.alternate.is_none() => match &node.consequent {
            Statement::BlockStatement(inner) if inner.body.len() == 1 => (Some(&node.test), &inner.body[0]),
            _ => return None,
        },
        other => (None, other),
    };
    let Statement::ExpressionStatement(s) = call else { return None };
    let Expression::CallExpression(c) = &s.expression else { return None };
    let Expression::StaticMemberExpression(m) = &c.callee else { return None };
    let Expression::Identifier(container) = &m.object else { return None };
    let [argument] = c.arguments.as_slice() else { return None };
    Some((container.name.as_str(), m.property.name.as_str(), cond, argument.as_expression()?))
}

/// `const xs = [];` then `for (const v of it) { [if (p) {] xs.push(e); }` → `const xs =
/// it.filter((v) => p).map((v) => e);`; a `new Set()` filled by `.add` → `new Set(it.map(…))`.
fn accumulating_loop<'a>(ast: AstBuilder<'a>, stmts: &mut ArenaVec<'a, Statement<'a>>, i: usize) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Some((container, declarator)) = single_declaration(&stmts[i]) else { return false };
    let method = match declarator.init.as_ref().map(strip) {
        Some(Expression::ArrayExpression(a)) if a.elements.is_empty() => "push",
        Some(Expression::NewExpression(n)) if n.arguments.is_empty() && matches!(&n.callee, Expression::Identifier(c) if c.name == "Set") => "add",
        _ => return false,
    };
    let Statement::ForOfStatement(lp) = &stmts[i + 1] else { return false };
    let ForStatementLeft::VariableDeclaration(left) = &lp.left else { return false };
    if lp.r#await || left.declarations.len() != 1 {
        return false;
    }
    let Some((target, called, cond, value)) = accumulate_step(&lp.body) else { return false };
    if target != container || called != method {
        return false;
    }
    let mentions = |e: &Expression<'_>| names_read(e).contains(container);
    if mentions(&lp.right) || mentions(value) || cond.is_some_and(mentions) {
        return false;
    }
    // The element is the loop variable itself: the chain is the filter alone.
    let identity = matches!((&left.declarations[0].id, strip(value)), (BindingPattern::BindingIdentifier(b), Expression::Identifier(v)) if b.name == v.name);

    let Statement::ForOfStatement(lp) = stmts.remove(i + 1) else { return false };
    let lp = lp.unbox();
    let ForStatementLeft::VariableDeclaration(left) = lp.left else { return false };
    let Some(binding) = left.unbox().declarations.into_iter().next() else { return false };
    let pattern = binding.id;
    let Some(step) = into_statements(lp.body).into_iter().next() else { return false };
    let (cond, call) = match step {
        Statement::IfStatement(node) => {
            let node = node.unbox();
            let Some(call) = into_statements(node.consequent).into_iter().next() else { return false };
            (Some(node.test), call)
        }
        other => (None, other),
    };
    let Statement::ExpressionStatement(s) = call else { return false };
    let Expression::CallExpression(c) = s.unbox().expression else { return false };
    let Some(argument) = c.unbox().arguments.into_iter().next() else { return false };
    let element = argument.into_expression();
    let mut chain = lp.right;
    if let Some(cond) = cond {
        chain = method_call(ast, chain, "filter", arrow(ast, pattern.clone_in(ast.allocator), cond));
    }
    if !identity {
        chain = method_call(ast, chain, "map", arrow(ast, pattern, element));
    }
    let init = if method == "add" {
        ast.expression_new(SPAN, ast.expression_identifier(SPAN, "Set"), NONE, ast.vec1(Argument::from(chain)))
    } else {
        chain
    };
    let Statement::VariableDeclaration(d) = &mut stmts[i] else { return false };
    d.declarations[0].init = Some(init);
    true
}

/// `const t = e;` then `S(t)`, with `t` bound once and read once, there → `S(e)`.
fn inline_temporary<'a>(stmts: &mut ArenaVec<'a, Statement<'a>>, i: usize, scope: &Scope) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Some((t, declarator)) = single_declaration(&stmts[i]) else { return false };
    let Some(init) = &declarator.init else { return false };
    if declarator.type_annotation.is_some() {
        return false;
    }
    if scope.stores(t) != 1 || scope.reads(t) != 1 || scope.nested_reads.contains(t) {
        return false;
    }
    if matches!(
        &stmts[i + 1],
        Statement::WhileStatement(_) | Statement::DoWhileStatement(_) | Statement::FunctionDeclaration(_) | Statement::ClassDeclaration(_)
    ) {
        return false;
    }
    if names_read(init).contains(t) {
        return false;
    }
    let t = t.to_owned();
    let mut direct = 0;
    for_each_direct_expr(&mut stmts[i + 1], &mut |e| direct += reads_of(e, &t));
    if direct != 1 {
        return false;
    }
    let Statement::VariableDeclaration(d) = stmts.remove(i) else { return false };
    let Some(value) = d.unbox().declarations.into_iter().next().and_then(|declarator| declarator.init) else { return false };
    let mut subst = Subst { name: t, value: Some(value) };
    for_each_direct_expr(&mut stmts[i], &mut |e| subst.visit_expression(e));
    true
}

struct Subst<'a> {
    name: String,
    value: Option<Expression<'a>>,
}

impl<'a> VisitMut<'a> for Subst<'a> {
    fn visit_expression(&mut self, it: &mut Expression<'a>) {
        if let Expression::Identifier(id) = it {
            if id.name == self.name.as_str() {
                if let Some(value) = self.value.take() {
                    *it = value;
                }
                return;
            }
        }
        walk_mut::walk_expression(self, it);
    }

    fn visit_function(&mut self, _: &mut ast::Function<'a>, _: ScopeFlags) {}

    fn visit_arrow_function_expression(&mut self, _: &mut ast::ArrowFunctionExpression<'a>) {}
}

/// `x += e` → `x = x + e` (and every arithmetic or bitwise compound); the logical ones (`??=`, `||=`,
/// `&&=`) assign only sometimes and are left alone.
fn expand_compound_assign<'a>(ast: AstBuilder<'a>, stmts: &mut ArenaVec<'a, Statement<'a>>, i: usize) -> bool {
    let Statement::ExpressionStatement(s) = &mut stmts[i] else { return false };
    let Expression::AssignmentExpression(a) = &mut s.expression else { return false };
    let Some(op) = a.operator.to_binary_operator() else { return false };
    let read = match a.left.as_simple_assignment_target() {
        Some(SimpleAssignmentTarget::AssignmentTargetIdentifier(id)) => Expression::Identifier(id.clone_in(ast.allocator)),
        Some(SimpleAssignmentTarget::StaticMemberExpression(m)) => Expression::StaticMemberExpression(m.clone_in(ast.allocator)),
        _ => return false,
    };
    let value = a.right.take_in(ast.allocator);
    a.right = ast.expression_binary(SPAN, read, op, value);
    a.operator = AssignmentOperator::Assign;
    true
}

/// Whether a statement list breaks out of the construct around it — a `break` not inside a loop or a
/// `switch` of its own, nor a function.
fn breaks_out(stmts: &[Statement<'_>]) -> bool {
    struct Breaks(bool);
    impl<'a> Visit<'a> for Breaks {
        fn visit_break_statement(&mut self, it: &ast::BreakStatement<'a>) {
            if it.label.is_none() {
                self.0 = true;
            }
        }
        fn visit_for_statement(&mut self, _: &ast::ForStatement<'a>) {}
        fn visit_for_in_statement(&mut self, _: &ast::ForInStatement<'a>) {}
        fn visit_for_of_statement(&mut self, _: &ast::ForOfStatement<'a>) {}
        fn visit_while_statement(&mut self, _: &ast::WhileStatement<'a>) {}
        fn visit_do_while_statement(&mut self, _: &ast::DoWhileStatement<'a>) {}
        fn visit_switch_statement(&mut self, _: &ast::SwitchStatement<'a>) {}
        fn visit_function(&mut self, _: &ast::Function<'a>, _: ScopeFlags) {}
        fn visit_arrow_function_expression(&mut self, _: &ast::ArrowFunctionExpression<'a>) {}
    }
    let mut breaks = Breaks(false);
    for stmt in stmts {
        breaks.visit_statement(stmt);
    }
    breaks.0
}

/// `switch (x) { case A: …; break; case B: case C: …; break; default: … }` → `if (x === A) { … }
/// else if (x === B || x === C) { … } else { … }`, when no case falls into the next and `default`, if
/// any, is last and alone.
fn switch_to_ladder<'a>(ast: AstBuilder<'a>, stmts: &mut ArenaVec<'a, Statement<'a>>, i: usize) -> bool {
    let Statement::SwitchStatement(node) = &stmts[i] else { return false };
    let last = node.cases.len().saturating_sub(1);
    let mut pending = 0usize;
    let mut groups = 0usize;
    for (k, case) in node.cases.iter().enumerate() {
        if case.test.is_none() && (k != last || pending > 0) {
            return false;
        }
        if case.consequent.is_empty() {
            if case.test.is_none() || k == last {
                return false;
            }
            pending += 1;
            continue;
        }
        let body = case.consequent.as_slice();
        let (stripped, had_break) = match body.split_last() {
            Some((Statement::BreakStatement(b), rest)) if b.label.is_none() => (rest, true),
            _ => (body, false),
        };
        if breaks_out(stripped) || !(had_break || k == last || stripped.last().is_some_and(ends)) {
            return false;
        }
        pending = 0;
        if case.test.is_some() {
            groups += 1;
        }
    }
    if groups == 0 {
        return false;
    }
    let Statement::SwitchStatement(node) = stmts[i].take_in(ast.allocator) else { return false };
    let node = node.unbox();
    let discriminant = node.discriminant;
    let mut rungs: Vec<(Vec<Expression<'a>>, Vec<Statement<'a>>)> = Vec::new();
    let mut otherwise: Option<Vec<Statement<'a>>> = None;
    let mut tests: Vec<Expression<'a>> = Vec::new();
    for case in node.cases {
        let mut body: Vec<Statement<'a>> = case.consequent.into_iter().collect();
        let Some(test) = case.test else {
            otherwise = Some(body);
            continue;
        };
        tests.push(ast.expression_binary(SPAN, discriminant.clone_in(ast.allocator), BinaryOperator::StrictEquality, test));
        if body.is_empty() {
            continue;
        }
        if matches!(body.last(), Some(Statement::BreakStatement(b)) if b.label.is_none()) {
            body.pop();
        }
        rungs.push((std::mem::take(&mut tests), body));
    }
    let mut ladder: Option<Statement<'a>> = otherwise.map(|body| ast.statement_block(SPAN, ast.vec_from_iter(body)));
    for (tests, body) in rungs.into_iter().rev() {
        let Some(test) = tests.into_iter().reduce(|a, b| ast.expression_logical(SPAN, a, LogicalOperator::Or, b)) else { return false };
        let alternate = ladder.map(|stmt| match stmt {
            block @ Statement::BlockStatement(_) => block,
            other => statement_block(ast, other),
        });
        ladder = Some(ast.statement_if(SPAN, test, ast.statement_block(SPAN, ast.vec_from_iter(body)), alternate));
    }
    let Some(ladder) = ladder else { return false };
    stmts[i] = ladder;
    true
}

/// `return undefined` → `return`.
fn bare_return(stmts: &mut ArenaVec<'_, Statement<'_>>, i: usize) -> bool {
    if let Statement::ReturnStatement(r) = &mut stmts[i] {
        if matches!(&r.argument, Some(Expression::Identifier(id)) if id.name == "undefined") {
            r.argument = None;
            return true;
        }
    }
    false
}

/// A two-branch `if` whose test is negated has its branches swapped.
fn orient_branches<'a>(ast: AstBuilder<'a>, stmts: &mut ArenaVec<'a, Statement<'a>>, i: usize, scope: &Scope) -> bool {
    let Statement::IfStatement(node) = &mut stmts[i] else { return false };
    if node.alternate.is_none() || !prefer_flip(ast, &node.test, &scope.locals) {
        return false;
    }
    let test = node.test.take_in(ast.allocator);
    node.test = negate(ast, test);
    swap_branches(node);
    true
}

#[cfg(test)]
mod tests {
    use crate::canon::analyze_functions;

    fn analyzed(src: &str) -> dup_defs_core::AnalyzedFn {
        analyze_functions(&[src.to_owned()]).pop().flatten().expect("a function")
    }

    fn xname(src: &str) -> String {
        analyzed(src).xname_canonical
    }

    fn lines(src: &str) -> Vec<String> {
        analyzed(src).type3_lines
    }

    #[test]
    fn braces_do_not_change_a_statement() {
        assert_eq!(xname("function f(x: number) { if (x) return 1; return 2; }"), xname("function f(x: number) { if (x) { return 1; } return 2; }"));
    }

    #[test]
    fn a_guard_and_an_else_are_one_form() {
        let guarded = xname("function f(x: number) { if (x === 0) { return 1; } g(); return 2; }");
        let branched = xname("function f(x: number) { if (x === 0) { return 1; } else { g(); return 2; } }");
        let negated = xname("function f(x: number) { if (x !== 0) { g(); return 2; } else { return 1; } }");
        assert_eq!(guarded, branched);
        assert_eq!(guarded, negated);
    }

    #[test]
    fn a_conditional_return_is_a_guard() {
        assert_eq!(xname("function f(c: boolean) { return c ? 1 : 2; }"), xname("function f(c: boolean) { if (c) { return 1; } return 2; }"));
    }

    #[test]
    fn negations_and_comparisons_have_one_spelling() {
        assert_eq!(xname("function f(a: boolean, b: boolean) { return !(a && b); }"), xname("function f(a: boolean, b: boolean) { return !a || !b; }"));
        assert_eq!(xname("function f(a: number, b: number) { return !(a === b); }"), xname("function f(a: number, b: number) { return a !== b; }"));
        assert_eq!(xname("function f(x: number) { return x > LIMIT; }"), xname("function f(x: number) { return LIMIT < x; }"));
    }

    #[test]
    fn a_loop_that_pushes_is_the_chain_it_spells() {
        let looped = xname("function f(xs: X[]) { const out = []; for (const x of xs) { if (x.ok) { out.push(x.id); } } return out; }");
        let chained = xname("function f(xs: X[]) { const out = xs.filter((x) => x.ok).map((x) => x.id); return out; }");
        assert_eq!(looped, chained);
        let filtered = xname("function f(xs: X[]) { const out = []; for (const x of xs) { if (x.ok) out.push(x); } return out; }");
        assert_eq!(filtered, xname("function f(xs: X[]) { const out = xs.filter((x) => x.ok); return out; }"));
    }

    #[test]
    fn a_switch_on_values_is_the_ladder_it_abbreviates() {
        let switched = xname("function f(k: string) { switch (k) { case 'a': return 1; case 'b': case 'c': g(); break; default: return 3; } return 0; }");
        let ladder = xname("function f(k: string) { if (k === 'a') { return 1; } else if (k === 'b' || k === 'c') { g(); } else { return 3; } return 0; }");
        assert_eq!(switched, ladder);
    }

    #[test]
    fn a_temporary_is_inlined_for_the_exact_pass_and_kept_for_type3() {
        let named = "function f(x: number) { const t = g(x); return h(t); }";
        let inlined = "function f(x: number) { return h(g(x)); }";
        assert_eq!(xname(named), xname(inlined));
        assert_ne!(lines(named), lines(inlined), "Type-3 keeps the statement");
    }

    #[test]
    fn small_spellings_meet() {
        assert_eq!(xname("function f(x: number) { x += 1; return x; }"), xname("function f(x: number) { x = x + 1; return x; }"));
        assert_eq!(xname("function f(x: number) { if (x) { return x; } return 0; }"), xname("function f(x: number) { return x || 0; }"));
        assert_eq!(xname("function f() { const unused = 1; return g(); }"), xname("function f() { return g(); }"));
        assert_eq!(xname("function f() { g(); return undefined; }"), xname("function f() { g(); }"));
    }
}
