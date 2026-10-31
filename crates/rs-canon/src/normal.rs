//! **Semantic normal form** of a callable's body, taken before its name-agnostic canonicals.
//!
//! The exact pass equates two definitions when their alpha-renamed canonicals are equal, and the
//! Type-3 pass when their statements mostly are. Both read the code as *written*, and the same
//! procedure is written many ways: `if !ok { return e; }` beside `if ok { … } else { return e; }`,
//! an `if` that is the body's value beside one that returns early, a temporary named and used once
//! beside the expression inlined, `!(a && b)` beside `!a || !b`. Each of those is a Type-4 clone to a
//! reader and an unrelated body to a pass that compares text.
//!
//! This module rewrites a body into one representative of its equivalence class under the rules the
//! Python frontend's normal form applies, read in Rust's grammar — oriented, applied to a fixed point,
//! one confluent direction per rule. The two languages' forms are not compared with each other; each
//! is the same *kind* of representative, so a pass reads both the same way.
//!
//! ## Rules
//!
//! Expressions, innermost first, to a fixed point:
//! - **Negation normal form**: `!!x` → `x`; `!(a && b)` → `!a || !b` (and dually); `!(a == b)` →
//!   `a != b`; `a || (b || c)` → `a || b || c`.
//! - **Comparison orientation**: `==` / `!=` and the mirror pairs (`<`/`>`, `<=`/`>=`) put their
//!   operands in a fixed order, by their shape with the definition's own names blanked — so a
//!   renamed copy orients the same way.
//! - **Branch orientation**: an `if … else` whose test is negated (or mostly negated) is flipped.
//! - **`match` on values** → the `if` / `else if` ladder it abbreviates, when every arm is a literal
//!   or a path (a constant, a unit variant) or an alternation of those, without guards. A `match` is
//!   exhaustive, so its last arm is the ladder's `else`.
//!
//! Statements, in every block:
//! - **Guard form**: `if c { …; return x; } else { B }` → `if c { …; return x; }` then `B`; mirrored
//!   when the `else` is the branch that ends.
//! - **The body's value is what it returns**: a final `return x;` is the tail `x`; a tail
//!   `if c { a } else { b }` returns `a` early and leaves `b` the tail — Python's
//!   `return a if c else b` becomes the same guard.
//! - **Single-use temporary** (the exact form only): `let t = e;` then a statement that reads `t`
//!   once and nothing else does → that statement with `e` in place of `t`.
//! - `let (a, b) = (x, y);` → `let a = x; let b = y;`; `x += e` → `x = x + e`.
//! - **Dead statements**: a `let` of an effect-free value to a name nothing reads.
//!
//! Not ported: Python's accumulating loop (`xs = []` then `for …: xs.append(e)` → a comprehension).
//! A Rust push loop and the iterator chain that replaces it differ in more than structure — the
//! element type is written on one side only, and `iter` / `into_iter` / `&xs` are the same loop — so
//! the rule could not make them meet without guessing types.
//!
//! ## What the rules assume
//!
//! The normalized tree is compared, never compiled: a rule need only preserve meaning as a reader
//! judges it — `x += e` and `x = x + e` are one intent, `match x { 1 => a, _ => b }` and
//! `if x == 1 { a } else { b }` one decision. A name read inside a macro's tokens (a format string's
//! `{x}` included) is a read nothing here can rewrite, so a binding such a read sees is left alone.

use std::collections::HashSet;

use proc_macro2::{Span, TokenStream, TokenTree};
use quote::ToTokens;
use syn::visit::{self, Visit};
use syn::visit_mut::{self, VisitMut};
use syn::{BinOp, Block, Expr, FnArg, Item, Lit, Pat, Signature, Stmt, Token, UnOp};

use dup_defs_core::normal::{Scope, MAX_ROUNDS};

/// The normal form of a body. `inline` admits the single-use-temporary rule: the exact pass wants it,
/// the Type-3 pass does not — its unit is the statement, and folding `let t = f(x);` into the line
/// that reads `t` turns two short units into one long one (see the Python frontend's normal form for
/// the measurement).
#[must_use]
pub(crate) fn normal_form(sig: &Signature, body: &Block, inline: bool) -> Block {
    let mut copy = body.clone();
    nest_else_ifs(&mut copy);
    for _ in 0..MAX_ROUNDS {
        let scope = scope_of(sig, &copy);
        let mut changed = block(&mut copy.stmts, &scope, true, inline);
        changed |= expressions(&mut copy, &scope);
        if !changed {
            break;
        }
    }
    flatten_else_ifs(&mut copy);
    copy
}

/// Continue an already-normalized body to the fixed point that also inlines temporaries — the exact
/// pass's form, from the Type-3 pass's, without starting over.
#[must_use]
pub(crate) fn with_inlining(sig: &Signature, body: &Block) -> Block {
    normal_form(sig, body, true)
}

// ---------------------------------------------------------------------------
// Node builders
// ---------------------------------------------------------------------------

fn semi() -> Token![;] {
    Token![;](Span::call_site())
}

fn paren(expr: Expr) -> Expr {
    match expr {
        atom @ (Expr::Path(_) | Expr::Lit(_) | Expr::Paren(_) | Expr::MethodCall(_) | Expr::Call(_) | Expr::Field(_)) => atom,
        other => Expr::Paren(syn::ExprParen { attrs: Vec::new(), paren_token: syn::token::Paren::default(), expr: Box::new(other) }),
    }
}

fn not(operand: Expr) -> Expr {
    Expr::Unary(syn::ExprUnary { attrs: Vec::new(), op: UnOp::Not(Token![!](Span::call_site())), expr: Box::new(paren(operand)) })
}

fn binary(left: Expr, op: BinOp, right: Expr) -> Expr {
    Expr::Binary(syn::ExprBinary { attrs: Vec::new(), left: Box::new(left), op, right: Box::new(right) })
}

fn and() -> BinOp {
    BinOp::And(Token![&&](Span::call_site()))
}

fn or() -> BinOp {
    BinOp::Or(Token![||](Span::call_site()))
}

fn eq() -> BinOp {
    BinOp::Eq(Token![==](Span::call_site()))
}

fn block_of(stmts: Vec<Stmt>) -> Block {
    Block { brace_token: syn::token::Brace::default(), stmts }
}

fn block_expr(block: Block) -> Expr {
    Expr::Block(syn::ExprBlock { attrs: Vec::new(), label: None, block })
}

fn returning(value: Expr) -> Stmt {
    let ret = Expr::Return(syn::ExprReturn { attrs: Vec::new(), return_token: Token![return](Span::call_site()), expr: Some(Box::new(value)) });
    Stmt::Expr(ret, Some(semi()))
}

/// An expression with its parentheses and invisible groups taken off.
fn strip(expr: &Expr) -> &Expr {
    match expr {
        Expr::Paren(p) => strip(&p.expr),
        Expr::Group(g) => strip(&g.expr),
        other => other,
    }
}

fn into_stripped(expr: Expr) -> Expr {
    match expr {
        Expr::Paren(p) => into_stripped(*p.expr),
        Expr::Group(g) => into_stripped(*g.expr),
        other => other,
    }
}

fn placeholder() -> Expr {
    Expr::Verbatim(TokenStream::new())
}

/// `else if` → `else { if … }`, everywhere: every rule then sees two-branch `if`s only.
fn nest_else_ifs(body: &mut Block) {
    struct Nest;
    impl VisitMut for Nest {
        fn visit_expr_if_mut(&mut self, node: &mut syn::ExprIf) {
            if let Some((_, alt)) = &mut node.else_branch {
                if matches!(**alt, Expr::If(_)) {
                    let inner = std::mem::replace(&mut **alt, placeholder());
                    **alt = block_expr(block_of(vec![Stmt::Expr(inner, None)]));
                }
            }
            visit_mut::visit_expr_if_mut(self, node);
        }
        fn visit_item_mut(&mut self, _: &mut Item) {}
    }
    Nest.visit_block_mut(body);
}

/// An `else` whose only statement is an `if` back to `else if`.
fn flatten_else_ifs(body: &mut Block) {
    struct Flatten;
    impl VisitMut for Flatten {
        fn visit_expr_if_mut(&mut self, node: &mut syn::ExprIf) {
            visit_mut::visit_expr_if_mut(self, node);
            if let Some((_, alt)) = &mut node.else_branch {
                if let Expr::Block(b) = &mut **alt {
                    if b.attrs.is_empty() && b.label.is_none() && matches!(b.block.stmts.as_slice(), [Stmt::Expr(Expr::If(_), None)]) {
                        if let Some(Stmt::Expr(inner, None)) = b.block.stmts.pop() {
                            **alt = inner;
                        }
                    }
                }
            }
        }
        fn visit_item_mut(&mut self, _: &mut Item) {}
    }
    Flatten.visit_block_mut(body);
}

// ---------------------------------------------------------------------------
// Scope facts: what the body binds and reads, counted once per round.
// ---------------------------------------------------------------------------

/// What the definition binds and reads — the shared [`Scope`] counts, filled by this language's walk.
fn scope_of(sig: &Signature, body: &Block) -> Scope {
    let mut scope = Scope::default();
    let mut walk = ScopeWalk { scope: &mut scope, depth: 0 };
    for input in &sig.inputs {
        if let FnArg::Typed(t) = input {
            walk.visit_pat(&t.pat);
        }
    }
    walk.visit_block(body);
    scope
}

struct ScopeWalk<'s> {
    scope: &'s mut Scope,
    depth: usize,
}

fn single_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(p) if p.qself.is_none() => p.path.get_ident().map(ToString::to_string),
        _ => None,
    }
}

impl<'ast> Visit<'ast> for ScopeWalk<'_> {
    fn visit_pat_ident(&mut self, it: &'ast syn::PatIdent) {
        self.scope.bind(&it.ident.to_string());
        visit::visit_pat_ident(self, it);
    }

    fn visit_expr_path(&mut self, it: &'ast syn::ExprPath) {
        if let Some(name) = it.path.get_ident().filter(|_| it.qself.is_none()) {
            self.scope.read(&name.to_string(), self.depth > 0);
        }
        visit::visit_expr_path(self, it);
    }

    fn visit_expr_assign(&mut self, it: &'ast syn::ExprAssign) {
        match single_name(&it.left) {
            Some(name) => {
                self.scope.store(&name);
                self.visit_expr(&it.right);
            }
            None => visit::visit_expr_assign(self, it),
        }
    }

    fn visit_expr_binary(&mut self, it: &'ast syn::ExprBinary) {
        if compound_base(it.op).is_some() {
            if let Some(name) = single_name(&it.left) {
                self.scope.store(&name);
            }
        }
        visit::visit_expr_binary(self, it);
    }

    fn visit_expr_closure(&mut self, it: &'ast syn::ExprClosure) {
        self.depth += 1;
        visit::visit_expr_closure(self, it);
        self.depth -= 1;
    }

    fn visit_expr_async(&mut self, it: &'ast syn::ExprAsync) {
        self.depth += 1;
        visit::visit_expr_async(self, it);
        self.depth -= 1;
    }

    fn visit_macro(&mut self, it: &'ast syn::Macro) {
        macro_names(&it.tokens, &mut self.scope.pinned);
    }

    fn visit_item(&mut self, _: &'ast Item) {}
}

/// Every name a macro's tokens mention, and every `{name}` a string among them captures.
fn macro_names(tokens: &TokenStream, out: &mut HashSet<String>) {
    for tree in tokens.clone() {
        match tree {
            TokenTree::Ident(id) => {
                out.insert(id.to_string());
            }
            TokenTree::Group(g) => macro_names(&g.stream(), out),
            TokenTree::Literal(lit) => captured_names(&lit.to_string(), out),
            TokenTree::Punct(_) => {}
        }
    }
}

/// The names a format string captures: `{name}`, `{name:?}`, not `{{`.
fn captured_names(text: &str, out: &mut HashSet<String>) {
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '{' {
            continue;
        }
        if chars.peek() == Some(&'{') {
            chars.next();
            continue;
        }
        let mut name = String::new();
        while let Some(&n) = chars.peek() {
            if n.is_alphanumeric() || n == '_' {
                name.push(n);
                chars.next();
            } else {
                break;
            }
        }
        if !name.is_empty() && matches!(chars.peek(), Some('}' | ':')) {
            out.insert(name);
        }
    }
}

/// Names read anywhere in `expr`, macro tokens included.
fn names_read(expr: &Expr) -> HashSet<String> {
    struct Reads(HashSet<String>);
    impl<'ast> Visit<'ast> for Reads {
        fn visit_expr_path(&mut self, it: &'ast syn::ExprPath) {
            if let Some(name) = it.path.get_ident() {
                self.0.insert(name.to_string());
            }
            visit::visit_expr_path(self, it);
        }
        fn visit_macro(&mut self, it: &'ast syn::Macro) {
            macro_names(&it.tokens, &mut self.0);
        }
    }
    let mut reads = Reads(HashSet::new());
    reads.visit_expr(expr);
    reads.0
}

/// Whether evaluating `expr` could do anything but read: a call, a macro, an `.await`, a `?` (which
/// may return), an assignment, a jump. A closure is a value, not a call, and its body is not looked
/// into.
fn has_effects(expr: &Expr) -> bool {
    struct Effects(bool);
    impl<'ast> Visit<'ast> for Effects {
        fn visit_expr(&mut self, it: &'ast Expr) {
            match it {
                Expr::Call(_)
                | Expr::MethodCall(_)
                | Expr::Macro(_)
                | Expr::Await(_)
                | Expr::Try(_)
                | Expr::Assign(_)
                | Expr::Return(_)
                | Expr::Break(_)
                | Expr::Continue(_)
                | Expr::Yield(_)
                | Expr::Unsafe(_) => self.0 = true,
                Expr::Binary(b) if compound_base(b.op).is_some() => self.0 = true,
                Expr::Closure(_) => return,
                _ => {}
            }
            visit::visit_expr(self, it);
        }
    }
    let mut effects = Effects(false);
    effects.visit_expr(expr);
    effects.0
}

fn compound_base(op: BinOp) -> Option<BinOp> {
    let span = Span::call_site();
    Some(match op {
        BinOp::AddAssign(_) => BinOp::Add(Token![+](span)),
        BinOp::SubAssign(_) => BinOp::Sub(Token![-](span)),
        BinOp::MulAssign(_) => BinOp::Mul(Token![*](span)),
        BinOp::DivAssign(_) => BinOp::Div(Token![/](span)),
        BinOp::RemAssign(_) => BinOp::Rem(Token![%](span)),
        BinOp::BitXorAssign(_) => BinOp::BitXor(Token![^](span)),
        BinOp::BitAndAssign(_) => BinOp::BitAnd(Token![&](span)),
        BinOp::BitOrAssign(_) => BinOp::BitOr(Token![|](span)),
        BinOp::ShlAssign(_) => BinOp::Shl(Token![<<](span)),
        BinOp::ShrAssign(_) => BinOp::Shr(Token![>>](span)),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Expression rules
// ---------------------------------------------------------------------------

/// A shape key with the definition's own names blanked — what the orientation rules order by. Two
/// renamed copies of one expression have one key; a key tie keeps the written order.
fn key(expr: &Expr, locals: &HashSet<String>) -> String {
    struct Blank<'l>(&'l HashSet<String>);
    impl VisitMut for Blank<'_> {
        fn visit_expr_path_mut(&mut self, it: &mut syn::ExprPath) {
            if let Some(name) = it.path.get_ident() {
                if self.0.contains(&name.to_string()) {
                    it.path = syn::Path::from(syn::Ident::new("__local", Span::call_site()));
                }
            }
        }
    }
    let mut copy = expr.clone();
    Blank(locals).visit_expr_mut(&mut copy);
    copy.to_token_stream().to_string()
}

/// Whether a test reads as a negation at its top.
fn is_negative(expr: &Expr) -> bool {
    match strip(expr) {
        Expr::Unary(u) => matches!(u.op, UnOp::Not(_)),
        Expr::Binary(b) => matches!(b.op, BinOp::Ne(_)),
        _ => false,
    }
}

/// Whether `expr` binds anything: `if let` tests cannot be negated.
fn binds(expr: &Expr) -> bool {
    struct Lets(bool);
    impl<'ast> Visit<'ast> for Lets {
        fn visit_expr_let(&mut self, _: &'ast syn::ExprLet) {
            self.0 = true;
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
    }
    let mut lets = Lets(false);
    lets.visit_expr(expr);
    lets.0
}

/// The exact inverse of a test, in negation normal form.
fn negate(expr: Expr) -> Expr {
    match into_stripped(expr) {
        Expr::Unary(u) if matches!(u.op, UnOp::Not(_)) => into_stripped(*u.expr),
        Expr::Binary(mut b) if matches!(b.op, BinOp::Eq(_)) => {
            b.op = BinOp::Ne(Token![!=](Span::call_site()));
            Expr::Binary(b)
        }
        Expr::Binary(mut b) if matches!(b.op, BinOp::Ne(_)) => {
            b.op = eq();
            Expr::Binary(b)
        }
        Expr::Binary(b) if matches!(b.op, BinOp::And(_) | BinOp::Or(_)) => {
            let op = if matches!(b.op, BinOp::And(_)) { or() } else { and() };
            binary(paren(negate(*b.left)), op, paren(negate(*b.right)))
        }
        Expr::Lit(mut l) => match &mut l.lit {
            Lit::Bool(v) => {
                v.value = !v.value;
                Expr::Lit(l)
            }
            _ => not(Expr::Lit(l)),
        },
        other => not(other),
    }
}

/// The operands of a chain of one `&&` or `||`, left to right.
fn operands(expr: &Expr) -> Vec<&Expr> {
    let Expr::Binary(b) = strip(expr) else { return vec![expr] };
    let same = |e: &Expr| matches!(strip(e), Expr::Binary(inner) if std::mem::discriminant(&inner.op) == std::mem::discriminant(&b.op));
    let mut out = if same(&b.left) { operands(&b.left) } else { vec![&*b.left] };
    out.extend(if same(&b.right) { operands(&b.right) } else { vec![&*b.right] });
    out
}

/// Whether a two-way test should be flipped: it is a negation, or most of its operands are, or at a
/// tie the flipped form keys lower.
fn prefer_flip(test: &Expr, locals: &HashSet<String>) -> bool {
    if binds(test) {
        return false;
    }
    if is_negative(test) {
        return true;
    }
    if let Expr::Binary(b) = strip(test) {
        if matches!(b.op, BinOp::And(_) | BinOp::Or(_)) {
            let all = operands(test);
            let negatives = all.iter().filter(|v| is_negative(v)).count();
            let positives = all.len() - negatives;
            if negatives != positives {
                return negatives > positives;
            }
            return key(&negate(test.clone()), locals) < key(test, locals);
        }
    }
    false
}

fn is_comparison(op: BinOp) -> bool {
    matches!(op, BinOp::Eq(_) | BinOp::Ne(_) | BinOp::Lt(_) | BinOp::Le(_) | BinOp::Gt(_) | BinOp::Ge(_))
}

fn mirror(op: BinOp) -> BinOp {
    let span = Span::call_site();
    match op {
        BinOp::Lt(_) => BinOp::Gt(Token![>](span)),
        BinOp::Gt(_) => BinOp::Lt(Token![<](span)),
        BinOp::Le(_) => BinOp::Ge(Token![>=](span)),
        BinOp::Ge(_) => BinOp::Le(Token![<=](span)),
        same => same,
    }
}

struct Exprs<'l> {
    locals: &'l HashSet<String>,
    changed: bool,
}

impl Exprs<'_> {
    fn rewrite(&mut self, expr: &mut Expr) {
        match expr {
            Expr::Unary(u) if matches!(u.op, UnOp::Not(_)) => {
                let reducible = match strip(&u.expr) {
                    Expr::Unary(inner) => matches!(inner.op, UnOp::Not(_)),
                    Expr::Binary(b) => matches!(b.op, BinOp::Eq(_) | BinOp::Ne(_) | BinOp::And(_) | BinOp::Or(_)),
                    Expr::Lit(l) => matches!(l.lit, Lit::Bool(_)),
                    _ => false,
                };
                if reducible && !binds(&u.expr) {
                    let operand = std::mem::replace(&mut *u.expr, placeholder());
                    *expr = negate(operand);
                    self.changed = true;
                }
            }
            Expr::Binary(b) if matches!(b.op, BinOp::And(_) | BinOp::Or(_)) => {
                let op = b.op;
                if matches!(strip(&b.right), Expr::Binary(r) if std::mem::discriminant(&r.op) == std::mem::discriminant(&op)) {
                    let Expr::Binary(right) = into_stripped(std::mem::replace(&mut *b.right, placeholder())) else { return };
                    let left = std::mem::replace(&mut *b.left, placeholder());
                    *expr = binary(binary(left, op, *right.left), op, *right.right);
                    self.changed = true;
                }
            }
            Expr::Binary(b) if is_comparison(b.op) && key(&b.right, self.locals) < key(&b.left, self.locals) => {
                std::mem::swap(&mut b.left, &mut b.right);
                b.op = mirror(b.op);
                self.changed = true;
            }
            Expr::If(node) if node.else_branch.is_some() && prefer_flip(&node.cond, self.locals) => {
                let cond = std::mem::replace(&mut *node.cond, placeholder());
                *node.cond = negate(cond);
                swap_branches(node);
                self.changed = true;
            }
            Expr::Match(m) => {
                if let Some(ladder) = match_to_ladder(m) {
                    *expr = ladder;
                    self.changed = true;
                }
            }
            _ => {}
        }
    }
}

impl VisitMut for Exprs<'_> {
    fn visit_expr_mut(&mut self, it: &mut Expr) {
        visit_mut::visit_expr_mut(self, it);
        self.rewrite(it);
    }

    fn visit_item_mut(&mut self, _: &mut Item) {}
}

/// Apply the expression rules to every expression of the body.
fn expressions(body: &mut Block, scope: &Scope) -> bool {
    let mut rules = Exprs { locals: &scope.locals, changed: false };
    rules.visit_block_mut(body);
    rules.changed
}

/// Exchange the two branches of an `if … else { … }` (else-ifs are nested by now).
fn swap_branches(node: &mut syn::ExprIf) {
    let Some((_, alt)) = &mut node.else_branch else { return };
    let Expr::Block(alt_block) = &mut **alt else { return };
    std::mem::swap(&mut node.then_branch, &mut alt_block.block);
}

/// `match x { A => a, B | C => b, _ => c }` → `if x == A { a } else if x == B || x == C { b } else
/// { c }`, when every arm but the last is a value or an alternation of values and none has a guard.
/// The last arm is the `else`: a `match` is exhaustive, so whatever its pattern, it takes the rest.
fn match_to_ladder(node: &syn::ExprMatch) -> Option<Expr> {
    let (last, arms) = node.arms.split_last()?;
    if arms.is_empty() || node.arms.iter().any(|arm| arm.guard.is_some()) {
        return None;
    }
    if !(value_tests(&last.pat, &node.expr).is_some() || matches!(last.pat, Pat::Wild(_))) {
        return None;
    }
    let tests: Vec<Vec<Expr>> = arms.iter().map(|arm| value_tests(&arm.pat, &node.expr)).collect::<Option<_>>()?;
    let as_block = |body: &Expr| match body {
        Expr::Block(b) if b.attrs.is_empty() && b.label.is_none() => b.block.clone(),
        other => block_of(vec![Stmt::Expr(other.clone(), None)]),
    };
    // Built from the last rung up, each rung the `else` block of the one before — the nested shape
    // every rule reads.
    let mut otherwise = as_block(&last.body);
    for (arm, tests) in arms.iter().zip(tests).rev() {
        let cond = tests.into_iter().reduce(|a, b| binary(a, or(), b))?;
        let rung = Expr::If(syn::ExprIf {
            attrs: Vec::new(),
            if_token: Token![if](Span::call_site()),
            cond: Box::new(cond),
            then_branch: as_block(&arm.body),
            else_branch: Some((Token![else](Span::call_site()), Box::new(block_expr(otherwise)))),
        });
        otherwise = block_of(vec![Stmt::Expr(rung, None)]);
    }
    match otherwise.stmts.pop() {
        Some(Stmt::Expr(ladder, None)) => Some(ladder),
        _ => None,
    }
}

/// The equality tests a value pattern stands for, one per alternative.
fn value_tests(pat: &Pat, subject: &Expr) -> Option<Vec<Expr>> {
    let value = match pat {
        Pat::Lit(l) => Expr::Lit(l.clone()),
        Pat::Path(p) => Expr::Path(p.clone()),
        Pat::Ident(i) if i.subpat.is_none() && i.by_ref.is_none() && i.mutability.is_none() && i.ident.to_string().starts_with(char::is_uppercase) => {
            Expr::Path(syn::ExprPath { attrs: Vec::new(), qself: None, path: syn::Path::from(i.ident.clone()) })
        }
        Pat::Paren(p) => return value_tests(&p.pat, subject),
        Pat::Or(o) => {
            let mut out = Vec::new();
            for case in &o.cases {
                out.extend(value_tests(case, subject)?);
            }
            return Some(out);
        }
        _ => return None,
    };
    Some(vec![binary(paren(subject.clone()), eq(), value)])
}

// ---------------------------------------------------------------------------
// Statement rules
// ---------------------------------------------------------------------------

/// Whether control never falls out of the bottom of `stmt`.
fn ends(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Expr(e, _) => expr_ends(e),
        Stmt::Macro(m) => diverging_macro(&m.mac),
        _ => false,
    }
}

fn expr_ends(expr: &Expr) -> bool {
    match strip(expr) {
        Expr::Return(_) | Expr::Break(_) | Expr::Continue(_) => true,
        Expr::Macro(m) => diverging_macro(&m.mac),
        Expr::If(node) => {
            block_ends(&node.then_branch) && node.else_branch.as_ref().is_some_and(|(_, alt)| expr_ends(alt))
        }
        Expr::Block(b) => block_ends(&b.block),
        _ => false,
    }
}

fn block_ends(block: &Block) -> bool {
    block.stmts.last().is_some_and(ends)
}

/// The macros that never return — Python's `raise`.
fn diverging_macro(mac: &syn::Macro) -> bool {
    mac.path.segments.last().is_some_and(|s| matches!(s.ident.to_string().as_str(), "panic" | "unreachable" | "todo" | "unimplemented"))
}

/// Whether `stmt`, the last of the body, hands the body's value back: a tail that is not a control
/// construct (whose value is whatever its branches leave, usually nothing).
fn is_value_tail(stmt: &Stmt) -> bool {
    matches!(stmt, Stmt::Expr(e, None) if !matches!(strip(e), Expr::If(_) | Expr::While(_) | Expr::ForLoop(_) | Expr::Loop(_) | Expr::Match(_) | Expr::Block(_)))
}

/// Whether a block's remainder ends: its last statement does, or — in the body itself — it is the
/// value the body returns.
fn rest_ends(stmts: &[Stmt], top: bool) -> bool {
    stmts.last().is_some_and(|s| ends(s) || (top && is_value_tail(s)))
}

/// Every block directly inside `stmt` — not those of a nested item, which is a scope of its own.
fn for_each_block(stmt: &mut Stmt, f: &mut dyn FnMut(&mut Block)) {
    struct Blocks<'f> {
        f: &'f mut dyn FnMut(&mut Block),
    }
    impl VisitMut for Blocks<'_> {
        fn visit_block_mut(&mut self, it: &mut Block) {
            (self.f)(it);
        }
        fn visit_item_mut(&mut self, _: &mut Item) {}
    }
    Blocks { f }.visit_stmt_mut(stmt);
}

/// The expressions a statement evaluates itself, before any block inside it runs — the slots a
/// temporary may be inlined into.
fn for_each_direct_expr(stmt: &mut Stmt, f: &mut dyn FnMut(&mut Expr)) {
    match stmt {
        Stmt::Local(local) => {
            if let Some(init) = &mut local.init {
                f(&mut init.expr);
            }
        }
        Stmt::Expr(e, _) => match e {
            Expr::If(node) => f(&mut node.cond),
            Expr::ForLoop(node) => f(&mut node.expr),
            Expr::Match(node) => f(&mut node.expr),
            Expr::While(_) | Expr::Loop(_) | Expr::Block(_) | Expr::Unsafe(_) => {}
            other => f(other),
        },
        Stmt::Item(_) | Stmt::Macro(_) => {}
    }
}

/// One round of the statement rules over a block and the blocks inside it. `top` marks the body
/// itself, whose tail is what the definition returns.
fn block(stmts: &mut Vec<Stmt>, scope: &Scope, top: bool, inline: bool) -> bool {
    let mut changed = false;
    for stmt in stmts.iter_mut() {
        for_each_block(stmt, &mut |b| changed |= block(&mut b.stmts, scope, false, inline));
    }
    let mut i = 0;
    while i < stmts.len() {
        let rewritten = unread_let(stmts, i, scope)
            || (top && tail_conditional(stmts, i))
            || guard_form(stmts, i, scope, top)
            || orient_guard(stmts, i, scope, top)
            || (inline && inline_temporary(stmts, i, scope))
            || split_tuple_let(stmts, i)
            || expand_compound_assign(stmts, i)
            || (top && final_return(stmts, i));
        if rewritten {
            changed = true;
        } else {
            i += 1;
        }
    }
    changed
}

/// A `let` of an effect-free value to a name nothing reads.
fn unread_let(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope) -> bool {
    let Stmt::Local(local) = &stmts[i] else { return false };
    let Some(init) = &local.init else { return false };
    if init.diverge.is_some() || has_effects(&init.expr) || stmts.len() < 2 {
        return false;
    }
    let dead = match &local.pat {
        Pat::Wild(_) => true,
        Pat::Ident(id) if id.subpat.is_none() => {
            let name = id.ident.to_string();
            scope.reads(&name) == 0 && !scope.pinned.contains(&name)
        }
        _ => false,
    };
    if dead {
        stmts.remove(i);
    }
    dead
}

/// The body's tail `if c { …; a } else { B }` returns `a` early: `if c { …; return a; } else { B }`,
/// which the guard form then lifts.
fn tail_conditional(stmts: &mut [Stmt], i: usize) -> bool {
    if i + 1 != stmts.len() {
        return false;
    }
    let Stmt::Expr(Expr::If(node), None) = &mut stmts[i] else { return false };
    if node.else_branch.is_none() || binds(&node.cond) {
        return false;
    }
    let Some(last) = node.then_branch.stmts.last_mut() else { return false };
    if !is_value_tail(last) {
        return false;
    }
    let Stmt::Expr(value, None) = std::mem::replace(last, Stmt::Expr(placeholder(), None)) else { return false };
    *last = returning(value);
    true
}

/// `if c { <ends> } else { B }` → `if c { <ends> }` then `B`; when only the `else` ends, the test is
/// negated and the branches change places first; when both end, the branch orientation decides which
/// one becomes the guard.
fn guard_form(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope, top: bool) -> bool {
    let at_tail = i + 1 == stmts.len();
    let Stmt::Expr(Expr::If(node), terminator) = &mut stmts[i] else { return false };
    let Some((_, alt)) = &node.else_branch else { return false };
    let Expr::Block(alt_block) = &**alt else { return false };
    let then_ends = block_ends(&node.then_branch);
    let else_ends = block_ends(&alt_block.block);
    if !then_ends && !else_ends {
        return false;
    }
    let flip = if then_ends && else_ends { prefer_flip(&node.cond, &scope.locals) } else { !then_ends };
    if flip {
        if binds(&node.cond) {
            return false;
        }
        let cond = std::mem::replace(&mut *node.cond, placeholder());
        *node.cond = negate(cond);
        swap_branches(node);
    }
    let Some((_, alt)) = node.else_branch.take() else { return false };
    let Expr::Block(alt) = *alt else { return false };
    let discarded = terminator.is_some();
    *terminator = None;
    let mut tail = alt.block.stmts;
    // An `if` that was a statement had its value discarded; a value its `else` ended in still is.
    if !at_tail || (discarded && !top) {
        if let Some(Stmt::Expr(_, s @ None)) = tail.last_mut() {
            *s = Some(semi());
        }
    }
    let at = i + 1;
    stmts.splice(at..at, tail);
    true
}

/// A guard whose remainder also ends is a two-branch `if` written as a guard: `if c { <ends> }` then
/// `<ends>` ≡ `if !c { <ends> }` then `<ends>`. The branch orientation decides which is the guard.
fn orient_guard(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope, top: bool) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Stmt::Expr(Expr::If(node), _) = &stmts[i] else { return false };
    if node.else_branch.is_some() || !block_ends(&node.then_branch) || !rest_ends(stmts, top) {
        return false;
    }
    if !prefer_flip(&node.cond, &scope.locals) {
        return false;
    }
    let mut rest: Vec<Stmt> = stmts.drain(i + 1..).collect();
    // Moved into a branch, the body's value is what it returns.
    if top {
        if let Some(last) = rest.last_mut() {
            if is_value_tail(last) {
                if let Stmt::Expr(value, None) = std::mem::replace(last, Stmt::Expr(placeholder(), None)) {
                    *last = returning(value);
                }
            }
        }
    }
    let Some(Stmt::Expr(Expr::If(mut node), _)) = stmts.pop() else { return false };
    let cond = std::mem::replace(&mut *node.cond, placeholder());
    *node.cond = negate(cond);
    let guard = std::mem::replace(&mut node.then_branch.stmts, rest);
    stmts.push(Stmt::Expr(Expr::If(node), None));
    stmts.extend(guard);
    true
}

/// `let t = e;` then `S(t)`, with `t` bound once and read once, there → `S(e)`.
fn inline_temporary(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Stmt::Local(local) = &stmts[i] else { return false };
    let Pat::Ident(id) = &local.pat else { return false };
    if id.mutability.is_some() || id.by_ref.is_some() || id.subpat.is_some() {
        return false;
    }
    let Some(init) = &local.init else { return false };
    if init.diverge.is_some() {
        return false;
    }
    let t = id.ident.to_string();
    if scope.stores(&t) != 1 || scope.reads(&t) != 1 || scope.pinned.contains(&t) || scope.nested_reads.contains(&t) {
        return false;
    }
    if matches!(&stmts[i + 1], Stmt::Expr(Expr::While(_) | Expr::Loop(_), _) | Stmt::Item(_) | Stmt::Macro(_)) {
        return false;
    }
    if names_read(&init.expr).contains(&t) {
        return false;
    }
    let mut direct = 0;
    for_each_direct_expr(&mut stmts[i + 1], &mut |e| direct += reads_of(e, &t));
    if direct != 1 {
        return false;
    }
    let Stmt::Local(local) = stmts.remove(i) else { return false };
    let Some(init) = local.init else { return false };
    let mut subst = Subst { name: t, value: Some(*init.expr) };
    for_each_direct_expr(&mut stmts[i], &mut |e| subst.visit_expr_mut(e));
    true
}

fn reads_of(expr: &Expr, name: &str) -> usize {
    struct Count<'n>(&'n str, usize);
    impl<'ast> Visit<'ast> for Count<'_> {
        fn visit_expr_path(&mut self, it: &'ast syn::ExprPath) {
            if it.path.is_ident(self.0) {
                self.1 += 1;
            }
            visit::visit_expr_path(self, it);
        }
    }
    let mut count = Count(name, 0);
    count.visit_expr(expr);
    count.1
}

struct Subst {
    name: String,
    value: Option<Expr>,
}

impl VisitMut for Subst {
    fn visit_expr_mut(&mut self, it: &mut Expr) {
        if let Expr::Path(p) = it {
            if p.qself.is_none() && p.path.is_ident(&self.name) {
                if let Some(value) = self.value.take() {
                    *it = paren(value);
                }
                return;
            }
        }
        visit_mut::visit_expr_mut(self, it);
    }
}

/// `let (a, b) = (x, y);` → `let a = x; let b = y;`, unless a later value reads an earlier name.
fn split_tuple_let(stmts: &mut Vec<Stmt>, i: usize) -> bool {
    let Stmt::Local(local) = &stmts[i] else { return false };
    let Pat::Tuple(targets) = &local.pat else { return false };
    let Some(init) = &local.init else { return false };
    let Expr::Tuple(values) = strip(&init.expr) else { return false };
    if init.diverge.is_some() || targets.elems.len() != values.elems.len() || targets.elems.len() < 2 {
        return false;
    }
    let names: Vec<String> = targets
        .elems
        .iter()
        .filter_map(|t| match t {
            Pat::Ident(id) if id.subpat.is_none() => Some(id.ident.to_string()),
            _ => None,
        })
        .collect();
    if names.len() != targets.elems.len() {
        return false;
    }
    for (k, value) in values.elems.iter().enumerate() {
        let reads = names_read(value);
        if names[..k].iter().any(|n| reads.contains(n)) {
            return false;
        }
    }
    let Stmt::Local(local) = stmts.remove(i) else { return false };
    let Pat::Tuple(targets) = local.pat else { return false };
    let Some(init) = local.init else { return false };
    let Expr::Tuple(values) = into_stripped(*init.expr) else { return false };
    let pairs: Vec<Stmt> = targets
        .elems
        .into_iter()
        .zip(values.elems)
        .map(|(pat, value)| {
            Stmt::Local(syn::Local {
                attrs: Vec::new(),
                let_token: Token![let](Span::call_site()),
                pat,
                init: Some(syn::LocalInit { eq_token: Token![=](Span::call_site()), expr: Box::new(value), diverge: None }),
                semi_token: semi(),
            })
        })
        .collect();
    stmts.splice(i..i, pairs);
    true
}

/// `x += e` → `x = x + e`.
fn expand_compound_assign(stmts: &mut [Stmt], i: usize) -> bool {
    let Stmt::Expr(Expr::Binary(b), _) = &mut stmts[i] else { return false };
    let Some(op) = compound_base(b.op) else { return false };
    let target = std::mem::replace(&mut *b.left, placeholder());
    let value = std::mem::replace(&mut *b.right, placeholder());
    let read = target.clone();
    let Stmt::Expr(e, _) = &mut stmts[i] else { return false };
    *e = Expr::Assign(syn::ExprAssign {
        attrs: Vec::new(),
        left: Box::new(target),
        eq_token: Token![=](Span::call_site()),
        right: Box::new(binary(read, op, paren(value))),
    });
    true
}

/// The body's last `return x;` is its tail `x`; a last bare `return;` is nothing.
fn final_return(stmts: &mut Vec<Stmt>, i: usize) -> bool {
    if i + 1 != stmts.len() {
        return false;
    }
    let Stmt::Expr(Expr::Return(r), _) = &mut stmts[i] else { return false };
    match r.expr.take() {
        Some(value) if !matches!(strip(&value), Expr::Tuple(t) if t.elems.is_empty()) => {
            stmts[i] = Stmt::Expr(*value, None);
            true
        }
        _ if stmts.len() > 1 => {
            stmts.pop();
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::canon::analyze_item_fn;

    fn analyzed(src: &str) -> crate::AnalyzedFn {
        let file = syn::parse_file(src).expect("fixture parses");
        let syn::Item::Fn(f) = &file.items[0] else { panic!("a function") };
        analyze_item_fn(f, true)
    }

    fn xname(src: &str) -> String {
        analyzed(src).xname_canonical
    }

    fn lines(src: &str) -> Vec<String> {
        analyzed(src).type3_lines
    }

    #[test]
    fn a_guard_and_an_else_are_one_form() {
        let guarded = xname("fn f(x: u8) -> u8 { if x == 0 { return 1; } g(); 2 }");
        let branched = xname("fn f(x: u8) -> u8 { if x == 0 { return 1; } else { g(); return 2; } }");
        let negated = xname("fn f(x: u8) -> u8 { if x != 0 { g(); 2 } else { 1 } }");
        assert_eq!(guarded, branched);
        assert_eq!(guarded, negated);
    }

    #[test]
    fn the_body_s_value_and_an_early_return_are_one_form() {
        let value = xname("fn f(c: bool) -> u8 { if c { 1 } else { 2 } }");
        let early = xname("fn f(c: bool) -> u8 { if c { return 1; } 2 }");
        let returned = xname("fn f(c: bool) -> u8 { if c { return 1; } return 2; }");
        assert_eq!(value, early);
        assert_eq!(value, returned);
    }

    #[test]
    fn negations_and_comparisons_have_one_spelling() {
        assert_eq!(xname("fn f(a: bool, b: bool) -> bool { !(a && b) }"), xname("fn f(a: bool, b: bool) -> bool { !a || !b }"));
        assert_eq!(xname("fn f(a: u8, b: u8) -> bool { !(a == b) }"), xname("fn f(a: u8, b: u8) -> bool { a != b }"));
        assert_eq!(xname("fn f(x: u8) -> bool { x > LIMIT }"), xname("fn f(x: u8) -> bool { LIMIT < x }"));
        assert_eq!(xname("fn f(a: u8, b: u8) -> bool { a == b + 1 }"), xname("fn f(a: u8, b: u8) -> bool { b + 1 == a }"));
    }

    #[test]
    fn a_match_on_values_is_the_ladder_it_abbreviates() {
        let matched = xname("fn f(x: u8) -> u8 { match x { 1 => a(), 2 | 3 => b(), _ => c() } }");
        let ladder = xname("fn f(x: u8) -> u8 { if x == 1 { a() } else if x == 2 || x == 3 { b() } else { c() } }");
        assert_eq!(matched, ladder);
    }

    #[test]
    fn a_temporary_is_inlined_for_the_exact_pass_and_kept_for_type3() {
        let named = "fn f(x: u8) -> u8 { let t = g(x); h(t) }";
        let inlined = "fn f(x: u8) -> u8 { h(g(x)) }";
        assert_eq!(xname(named), xname(inlined));
        assert_ne!(lines(named), lines(inlined), "Type-3 keeps the statement");
    }

    #[test]
    fn small_spellings_meet() {
        assert_eq!(xname("fn f(mut x: u8) -> u8 { x += 1; x }"), xname("fn f(mut x: u8) -> u8 { x = x + 1; x }"));
        assert_eq!(xname("fn f() -> u8 { let (a, b) = (g(), h()); a + b }"), xname("fn f() -> u8 { let a = g(); let b = h(); a + b }"));
    }

    #[test]
    fn a_binding_a_macro_reads_is_not_dead() {
        let kept = xname("fn f() { let t = 1; println!(\"{t}\"); }");
        let dropped = xname("fn f() { println!(\"{t}\"); }");
        assert_ne!(kept, dropped);
        assert_eq!(xname("fn f() -> u8 { let unused = 1; g() }"), xname("fn f() -> u8 { g() }"));
    }
}
