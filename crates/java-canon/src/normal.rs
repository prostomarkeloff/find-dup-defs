//! **Semantic normal form** of a Java body, taken before its name-agnostic canonicals.
//!
//! The exact pass equates two definitions when their alpha-renamed canonicals are equal, and the
//! Type-3 pass when their statements mostly are. Both read the code as *written*, and the same
//! procedure is written many ways: a guard beside an `if … else`, `Objects.isNull(x)` beside
//! `x == null`, `"lit".equals(x)` beside `x.equals("lit")`, a temporary named and used once beside
//! the expression inlined, a `switch` beside the `if` ladder it abbreviates, `x++` beside
//! `x = x + 1`. Each of those is a Type-4 clone to a reader and an unrelated body to a pass that
//! compares text.
//!
//! This module rewrites a body into one representative of its equivalence class under the rules the
//! other frontends' normal forms apply, read in Java's grammar — oriented, applied to a fixed point,
//! one confluent direction per rule.
//!
//! ## Rules
//!
//! Expressions, innermost first, to a fixed point:
//! - **Negation normal form**: `!!x` → `x`; `!(a && b)` → `!a || !b` (and dually); `!(a == b)` →
//!   `a != b`; `!(a < b)` → `a >= b`; `a || (b || c)` → `a || b || c`. Parentheses are dropped: the
//!   tree already says what they grouped.
//! - **Comparison orientation**: `==` / `!=` and the mirror pairs put their operands in a fixed order,
//!   by their shape with the definition's own names blanked — so a renamed copy orients the same way.
//!   (`x == null` keeps its spelling: a local sorts before `null`.)
//! - **A null test has one spelling**: `Objects.isNull(x)` is `x == null`, `Objects.nonNull(x)` is
//!   `x != null` — the library's two ways of writing one test.
//! - **`.equals` orientation**: `a.equals(b)` and `a.equalsIgnoreCase(b)` orient by shape, and a
//!   literal operand becomes the *receiver* (`x.equals("lit")` → `"lit".equals(x)`). `Objects.equals`
//!   is symmetric: its two arguments sort by the same shape order. `a == b` stays `==` — identity is
//!   not equality in Java.
//!
//! Statements, in every block:
//! - **A `switch` is the ladder it abbreviates**: `case a, b:` → `if (v == a || v == b)`, `default` →
//!   the last `else` — when every case ends in its own `break`/`return`/`throw`/`continue` (or is
//!   last) and none falls through or `break`s from deeper inside: what a ladder could not say.
//! - **Branch orientation**: an `if … else` that neither branch leaves through, whose test is negated
//!   (or mostly negated), is flipped.
//! - **Guard form**: `if (c) { …; return x; } else { B }` → `if (c) { …; return x; }` then `B`;
//!   mirrored when the `else` is the branch that ends.
//! - **Guard orientation**: when the rest of the block ends too, `if (c) { E }` then `R` is the same
//!   decision as `if (!c) { R }` then `E`. The guard holds the *lighter* branch — the early exit the
//!   method is written around; at equal weight, the branch that *fails* — a `throw`, a `return null`
//!   / `false` / `0` / `-1` / an empty `Optional` / `Collection` — and only then the positive test.
//! - **Single-use temporary** (the exact form only): `T t = e;` then a statement that reads `t` once
//!   and nothing else does → that statement with `e` in place of `t`.
//! - **Truth folds**: `if (c) return true; return false;` → `return c;`; `Boolean.parseBoolean(s)` /
//!   `Boolean.valueOf(s)` in a truth position → `s` — a truth test of the string.
//! - `x += e` and `x++` → `x = x + e`; `while (true)` → `for (;;)`.
//! - A final bare `return;` in a method that returns nothing is falling off its end.
//!
//! ## What the rules assume
//!
//! The normalized tree is compared, never run: a rule need only preserve meaning as a reader judges
//! it. A ladder evaluates the tested value once per test. A name passed to a method, or a target of an
//! in-place update (`x++`), is pinned: inlining it would change what overload the reader infers, or
//! what is mutated.

use std::collections::HashSet;

use dup_defs_core::normal::{complement, mirror, Scope, MAX_ROUNDS};

use crate::canon::{for_each_child, Child};
use crate::ir::{Case, Else, Expr, Func, LambdaBody, Stmt, Type};

/// The normal form of a callable. `inline` admits the single-use-temporary rule: the exact pass wants
/// it, the Type-3 pass does not — its unit is the statement, and folding `T t = e` into the line that
/// reads `t` turns two short units into one long one (the Python frontend's measurement).
#[must_use]
pub(crate) fn normal_form(f: &Func, inline: bool) -> Func {
    let mut out = f.clone();
    // A constructor has no return type; a void method spells its `void`. Both return nothing.
    let void = match &f.ret {
        None => true,
        Some(Type::Named(n)) => n == "void",
        _ => false,
    };
    strip_parens_stmts(&mut out.body);
    nest_else_ifs(&mut out.body);
    for _ in 0..MAX_ROUNDS {
        let scope = scope_of(&out);
        let mut changed = block(&mut out.body, &scope, Ctx { top: true, void, inline });
        changed |= expressions(&mut out.body, &scope);
        if !changed {
            break;
        }
    }
    flatten_else_ifs(&mut out.body);
    out
}

#[derive(Clone, Copy)]
struct Ctx {
    /// The method's own body, whose end is a return.
    top: bool,
    /// The method returns nothing (or is a constructor): falling off its end is a bare `return`.
    void: bool,
    inline: bool,
}

// ---------------------------------------------------------------------------
// Traversal
// ---------------------------------------------------------------------------

/// Every statement list directly inside `s` — its branches and bodies, and the body of every lambda
/// or statement block its expressions hold — not the lists nested in those.
fn nested_lists(s: &mut Stmt, f: &mut dyn FnMut(&mut Vec<Stmt>)) {
    match s {
        Stmt::If { cond, then, els } => {
            func_lits(cond, f);
            f(then);
            match els.as_deref_mut() {
                Some(Else::Block(b)) => f(b),
                Some(Else::If(alt)) => nested_lists(alt, f),
                None => {}
            }
        }
        Stmt::While { cond, body } => {
            func_lits(cond, f);
            f(body);
        }
        Stmt::DoWhile { body, cond } => {
            f(body);
            func_lits(cond, f);
        }
        Stmt::For { init, cond, update, body } => {
            if let Some(init) = init {
                nested_lists(init, f);
            }
            if let Some(c) = cond {
                func_lits(c, f);
            }
            for u in update {
                func_lits(u, f);
            }
            f(body);
        }
        Stmt::ForEach { iter, body, .. } => {
            func_lits(iter, f);
            f(body);
        }
        Stmt::Switch { value, cases } => {
            func_lits(value, f);
            for case in cases {
                for v in &mut case.values {
                    func_lits(v, f);
                }
                f(&mut case.body);
            }
        }
        Stmt::Try { resources, body, catches, finally } => {
            for r in resources {
                func_lits(&mut r.value, f);
            }
            f(body);
            for c in catches {
                f(&mut c.body);
            }
            if let Some(fin) = finally {
                f(fin);
            }
        }
        Stmt::Synchronized { lock, body } => {
            func_lits(lock, f);
            f(body);
        }
        Stmt::Block(b) => f(b),
        Stmt::Labeled { stmt, .. } => nested_lists(stmt, f),
        other => direct_exprs(other, &mut |e| func_lits(e, f)),
    }
}

/// The bodies of the lambdas and statement blocks `e` holds, outermost first.
fn func_lits(e: &mut Expr, f: &mut dyn FnMut(&mut Vec<Stmt>)) {
    match e {
        Expr::Lambda { body: LambdaBody::Block(b), .. } | Expr::StmtBlock(b) => f(b),
        other => for_each_child_mut(other, &mut |c| func_lits(c, f)),
    }
}

/// The expressions a statement evaluates itself, once, before any block inside it runs — the slots a
/// temporary may be inlined into, and the roots the expression rules walk. A loop's condition is not
/// one: it is evaluated again on every pass.
fn direct_exprs(s: &mut Stmt, f: &mut dyn FnMut(&mut Expr)) {
    match s {
        Stmt::Decl(items) => items.iter_mut().for_each(|it| direct_exprs(it, f)),
        Stmt::Assign { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        Stmt::Let { init: Some(e), .. } | Stmt::Update { e, .. } | Stmt::Expr(e) | Stmt::Throw(e) | Stmt::Assert(e) | Stmt::Return(Some(e)) => f(e),
        Stmt::If { cond, .. } => f(cond),
        Stmt::Switch { value, .. } => f(value),
        Stmt::ForEach { iter, .. } => f(iter),
        Stmt::Synchronized { lock, .. } => f(lock),
        _ => {}
    }
}

/// Every expression slot of a body, at every depth, as a root the expression rules rewrite.
fn all_exprs(stmts: &mut [Stmt], f: &mut dyn FnMut(&mut Expr)) {
    for s in stmts.iter_mut() {
        stmt_exprs(s, f);
    }
}

fn stmt_exprs(s: &mut Stmt, f: &mut dyn FnMut(&mut Expr)) {
    match s {
        Stmt::If { cond, then, els } => {
            f(cond);
            all_exprs(then, f);
            match els.as_deref_mut() {
                Some(Else::Block(b)) => all_exprs(b, f),
                Some(Else::If(alt)) => stmt_exprs(alt, f),
                None => {}
            }
        }
        Stmt::While { cond, body } => {
            f(cond);
            all_exprs(body, f);
        }
        Stmt::DoWhile { body, cond } => {
            all_exprs(body, f);
            f(cond);
        }
        Stmt::For { init, cond, update, body } => {
            if let Some(init) = init {
                stmt_exprs(init, f);
            }
            if let Some(c) = cond {
                f(c);
            }
            for u in update {
                f(u);
            }
            all_exprs(body, f);
        }
        Stmt::ForEach { iter, body, .. } => {
            f(iter);
            all_exprs(body, f);
        }
        Stmt::Switch { value, cases } => {
            f(value);
            for case in cases {
                for v in &mut case.values {
                    f(v);
                }
                all_exprs(&mut case.body, f);
            }
        }
        Stmt::Try { resources, body, catches, finally } => {
            for r in resources {
                f(&mut r.value);
            }
            all_exprs(body, f);
            for c in catches {
                all_exprs(&mut c.body, f);
            }
            if let Some(fin) = finally {
                all_exprs(fin, f);
            }
        }
        Stmt::Synchronized { lock, body } => {
            f(lock);
            all_exprs(body, f);
        }
        Stmt::Block(b) => all_exprs(b, f),
        Stmt::Labeled { stmt, .. } => stmt_exprs(stmt, f),
        other => direct_exprs(other, f),
    }
}

/// [`for_each_child`], mutably: every direct sub-expression, and the expressions of a lambda's or a
/// statement block's body.
fn for_each_child_mut(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    match e {
        Expr::Binary { l, r, .. } | Expr::AssignExpr { l, r, .. } => {
            f(l);
            f(r);
        }
        Expr::Unary { e, .. }
        | Expr::Update { e, .. }
        | Expr::Paren(e)
        | Expr::Cast { e, .. }
        | Expr::InstanceOf { e, .. }
        | Expr::Field { base: e, .. }
        | Expr::Throw(e) => f(e),
        Expr::Cond { c, t, f: alt } => {
            f(c);
            f(t);
            f(alt);
        }
        Expr::Call { f: callee, args } => {
            f(callee);
            args.iter_mut().for_each(f);
        }
        Expr::Method { recv, args, .. } => {
            f(recv);
            args.iter_mut().for_each(f);
        }
        Expr::StaticCall { args, .. } | Expr::Super { args } | Expr::New { args, .. } => {
            args.iter_mut().for_each(f);
        }
        Expr::Index { base, i } => {
            f(base);
            f(i);
        }
        Expr::Array { elems } => elems.iter_mut().for_each(f),
        Expr::Lambda { body, .. } => match body {
            LambdaBody::Block(b) => all_exprs(b, f),
            LambdaBody::Expr(inner) => f(inner),
        },
        Expr::MethodRef { target, .. } => f(target),
        Expr::Switch { value, arms } => {
            f(value);
            for arm in arms {
                for c in &mut arm.conds {
                    f(c);
                }
                f(&mut arm.value);
            }
        }
        Expr::StmtBlock(b) => all_exprs(b, f),
        Expr::Name(_)
        | Expr::Lit(..)
        | Expr::Bool(_)
        | Expr::Null
        | Expr::StaticField { .. }
        | Expr::ClassConst { .. }
        | Expr::Unknown(_) => {}
    }
}

fn placeholder() -> Expr {
    Expr::Unknown(String::new())
}

// ---------------------------------------------------------------------------
// Structural passes run once
// ---------------------------------------------------------------------------

/// Parentheses carry nothing the tree does not: the grouping is the tree's shape.
fn strip_parens_stmts(stmts: &mut [Stmt]) {
    all_exprs(stmts, &mut strip_parens);
}

fn strip_parens(e: &mut Expr) {
    while let Expr::Paren(inner) = e {
        let inner = std::mem::replace(&mut **inner, placeholder());
        *e = inner;
    }
    for_each_child_mut(e, &mut strip_parens);
}

/// `else if` → `else { if … }`, everywhere: every rule then sees two-branch `if`s only.
fn nest_else_ifs(stmts: &mut [Stmt]) {
    for s in stmts.iter_mut() {
        if let Stmt::If { els: Some(els), .. } = s {
            if let Else::If(_) = **els {
                let Else::If(inner) = std::mem::replace(&mut **els, Else::Block(Vec::new())) else { continue };
                **els = Else::Block(vec![*inner]);
            }
        }
        nested_lists(s, &mut |list| nest_else_ifs(list));
    }
}

/// An `else` whose only statement is an `if` back to `else if`.
fn flatten_else_ifs(stmts: &mut [Stmt]) {
    for s in stmts.iter_mut() {
        nested_lists(s, &mut |list| flatten_else_ifs(list));
        if let Stmt::If { els: Some(els), .. } = s {
            if let Else::Block(b) = &mut **els {
                if matches!(b.as_slice(), [Stmt::If { .. }]) {
                    if let Some(inner) = b.pop() {
                        **els = Else::If(Box::new(inner));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scope facts: what the body binds and reads, counted once per round.
// ---------------------------------------------------------------------------

fn scope_of(f: &Func) -> Scope {
    let mut scope = Scope::default();
    let mut walk = ScopeWalk { scope: &mut scope, depth: 0 };
    for p in &f.params {
        walk.scope.bind(&p.name);
    }
    walk.stmts(&f.body);
    scope
}

struct ScopeWalk<'s> {
    scope: &'s mut Scope,
    depth: usize,
}

impl ScopeWalk<'_> {
    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    #[allow(clippy::too_many_lines)] // one arm per statement shape reads better straight-through
    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let { name, init, .. } => {
                if let Some(e) = init {
                    self.expr(e);
                }
                self.scope.bind(name);
            }
            Stmt::Decl(items) => self.stmts(items),
            Stmt::Assign { op, lhs, rhs } => {
                self.expr(rhs);
                self.store_target(lhs);
                // A compound assignment reads what it writes.
                if op != "=" {
                    self.expr(lhs);
                }
            }
            Stmt::Update { e, .. } => {
                self.store_target(e);
                self.expr(e);
            }
            Stmt::Expr(e) | Stmt::Return(Some(e)) | Stmt::Throw(e) | Stmt::Assert(e) => self.expr(e),
            Stmt::If { cond, then, els } => {
                self.expr(cond);
                self.stmts(then);
                match els.as_deref() {
                    Some(Else::Block(b)) => self.stmts(b),
                    Some(Else::If(alt)) => self.stmt(alt),
                    None => {}
                }
            }
            Stmt::While { cond, body } => {
                self.expr(cond);
                self.stmts(body);
            }
            Stmt::DoWhile { body, cond } => {
                self.stmts(body);
                self.expr(cond);
            }
            Stmt::For { init, cond, update, body } => {
                if let Some(init) = init {
                    self.stmt(init);
                }
                if let Some(c) = cond {
                    self.expr(c);
                }
                for u in update {
                    self.expr(u);
                }
                self.stmts(body);
            }
            Stmt::ForEach { name, iter, body } => {
                self.expr(iter);
                self.scope.bind(name);
                self.stmts(body);
            }
            Stmt::Switch { value, cases } => {
                self.expr(value);
                for case in cases {
                    for v in &case.values {
                        self.expr(v);
                    }
                    self.stmts(&case.body);
                }
            }
            Stmt::Try { resources, body, catches, finally } => {
                for r in resources {
                    self.expr(&r.value);
                    if let Some(name) = &r.name {
                        self.scope.bind(name);
                    }
                }
                self.stmts(body);
                for c in catches {
                    if let Some(name) = &c.name {
                        self.scope.bind(name);
                    }
                    self.stmts(&c.body);
                }
                if let Some(fin) = finally {
                    self.stmts(fin);
                }
            }
            Stmt::Synchronized { lock, body } => {
                self.expr(lock);
                self.stmts(body);
            }
            Stmt::Labeled { stmt, .. } => self.stmt(stmt),
            Stmt::Block(b) => self.stmts(b),
            Stmt::Return(None) | Stmt::Break(_) | Stmt::Continue(_) | Stmt::NestedItem | Stmt::Unknown(_) => {}
        }
    }

    /// An assignment's target: a plain name is a store, anything else (`a.b`, `m[k]`) a read of what
    /// it reaches into.
    fn store_target(&mut self, e: &Expr) {
        match e {
            Expr::Name(n) => self.scope.store(n),
            other => self.expr(other),
        }
    }

    /// A name that is the direct argument of a call: nothing may be assumed about it (the call may
    /// mutate it, or overload on its declared type).
    fn pin_if_name(&mut self, e: &Expr) {
        if let Expr::Name(n) = e {
            self.scope.pinned.insert(n.clone());
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Name(n) => self.scope.read(n, self.depth > 0),
            // An in-place update reads and writes its target, and needs an lvalue: its name is pinned.
            Expr::Update { e: target, .. } => {
                if let Expr::Name(n) = target.as_ref() {
                    self.scope.pinned.insert(n.clone());
                }
                self.store_target(target);
                self.expr(target);
            }
            // An assignment reads its right side, stores its target, and — compound — reads the target.
            Expr::AssignExpr { op, l, r } => {
                self.expr(r);
                self.store_target(l);
                if op != "=" {
                    self.expr(l);
                }
            }
            Expr::Lambda { params, body } => {
                self.depth += 1;
                for p in params {
                    self.scope.bind(&p.name);
                }
                match body {
                    LambdaBody::Block(b) => self.stmts(b),
                    LambdaBody::Expr(e) => self.expr(e),
                }
                self.depth -= 1;
            }
            Expr::Call { f, args } => {
                self.expr(f);
                for a in args {
                    self.pin_if_name(a);
                    self.expr(a);
                }
            }
            Expr::Method { recv, args, .. } => {
                self.expr(recv);
                for a in args {
                    self.pin_if_name(a);
                    self.expr(a);
                }
            }
            Expr::StaticCall { args, .. } | Expr::Super { args } | Expr::New { args, .. } => {
                for a in args {
                    self.pin_if_name(a);
                    self.expr(a);
                }
            }
            other => for_each_child(other, &mut |child| match child {
                Child::Expr(c) => self.expr(c),
                Child::Body(b) => self.stmts(b),
            }),
        }
    }
}

/// Names read anywhere in `e`, including inside the bodies it holds.
fn names_read(e: &Expr) -> HashSet<String> {
    fn walk(e: &Expr, out: &mut HashSet<String>) {
        if let Expr::Name(n) = e {
            out.insert(n.clone());
        }
        for_each_child(e, &mut |child| match child {
            Child::Expr(c) => walk(c, out),
            Child::Body(body) => {
                for s in body {
                    let mut copy = s.clone();
                    stmt_exprs(&mut copy, &mut |x| walk(x, out));
                }
            }
        });
    }
    let mut out = HashSet::new();
    walk(e, &mut out);
    out
}

fn reads_of(e: &Expr, name: &str) -> usize {
    let mut n = usize::from(matches!(e, Expr::Name(x) if x == name));
    for_each_child(e, &mut |child| match child {
        Child::Expr(c) => n += reads_of(c, name),
        Child::Body(body) => {
            for s in body {
                let mut copy = s.clone();
                stmt_exprs(&mut copy, &mut |x| n += reads_of(x, name));
            }
        }
    });
    n
}

// ---------------------------------------------------------------------------
// Expression rules
// ---------------------------------------------------------------------------

/// A shape key with the definition's own names blanked — what the orientation rules order by. Two
/// renamed copies of one expression have one key; a key tie keeps the written order.
fn key(e: &Expr, locals: &HashSet<String>) -> String {
    fn blank(e: &mut Expr, locals: &HashSet<String>) {
        if let Expr::Name(n) = e {
            if locals.contains(n.as_str()) {
                "__local".clone_into(n);
            }
        }
        for_each_child_mut(e, &mut |c| blank(c, locals));
    }
    let mut copy = e.clone();
    blank(&mut copy, locals);
    format!("{copy:?}")
}

fn is_negative(e: &Expr) -> bool {
    match e {
        Expr::Unary { op, .. } => op == "!",
        Expr::Binary { op, .. } => op == "!=",
        _ => false,
    }
}

fn not(e: Expr) -> Expr {
    Expr::Unary { op: "!".to_owned(), e: Box::new(e) }
}

fn binary(l: Expr, op: &str, r: Expr) -> Expr {
    Expr::Binary { op: op.to_owned(), l: Box::new(l), r: Box::new(r) }
}

/// The exact inverse of a test, in negation normal form.
fn negate(e: Expr) -> Expr {
    match e {
        Expr::Unary { op, e } if op == "!" => *e,
        Expr::Binary { op, l, r } if op == "==" => Expr::Binary { op: "!=".to_owned(), l, r },
        Expr::Binary { op, l, r } if op == "!=" => Expr::Binary { op: "==".to_owned(), l, r },
        // An ordering's complement is an ordering: `!(a < b)` is `a >= b` to a reader, whatever a NaN
        // would make of it — the form is compared, never run.
        Expr::Binary { op, l, r } if complement(&op).is_some() => {
            Expr::Binary { op: complement(&op).unwrap_or_default().to_owned(), l, r }
        }
        Expr::Binary { op, l, r } if op == "&&" || op == "||" => {
            let flipped = if op == "&&" { "||" } else { "&&" };
            binary(negate(*l), flipped, negate(*r))
        }
        Expr::Bool(b) => Expr::Bool(!b),
        other => not(other),
    }
}

/// The operands of a chain of one `&&` or `||`, left to right.
fn operands(e: &Expr) -> Vec<&Expr> {
    let Expr::Binary { op, l, r } = e else { return vec![e] };
    if op != "&&" && op != "||" {
        return vec![e];
    }
    let same = |x: &Expr| matches!(x, Expr::Binary { op: inner, .. } if inner == op);
    let mut out = if same(l) { operands(l) } else { vec![&**l] };
    out.extend(if same(r) { operands(r) } else { vec![&**r] });
    out
}

/// Whether a test leans negative: it is a negation, or most of its operands are, or at a tie its
/// negation keys lower. Exactly one of a test and its negation leans negative, unless they key alike.
fn leans_negative(test: &Expr, locals: &HashSet<String>) -> bool {
    if is_negative(test) {
        return true;
    }
    if let Expr::Binary { op, .. } = test {
        if op == "&&" || op == "||" {
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

/// `Objects.isNull(x)` → `x == null`, `Objects.nonNull(x)` → `x != null`: one null test, one spelling.
fn null_test(e: &mut Expr) -> bool {
    let (name, args) = match e {
        Expr::StaticCall { scope, name, args } if scope == "Objects" => (name.clone(), args),
        Expr::Method { recv, name, args } if matches!(recv.as_ref(), Expr::Name(n) if n == "Objects") => (name.clone(), args),
        _ => return false,
    };
    if args.len() != 1 {
        return false;
    }
    let op = match name.as_str() {
        "isNull" => "==",
        "nonNull" => "!=",
        _ => return false,
    };
    let arg = args.pop().unwrap_or_else(placeholder);
    *e = binary(arg, op, Expr::Null);
    true
}

/// A literal value — `Lit`, `true`/`false`, `null` — the fixed side of a `.equals` orientation.
fn is_literal(e: &Expr) -> bool {
    matches!(e, Expr::Lit(..) | Expr::Bool(_) | Expr::Null)
}

/// `a.equals(b)` / `a.equalsIgnoreCase(b)`: a literal operand becomes the receiver; otherwise the
/// operands orient by shape.
fn orient_equals(recv: &mut Expr, args: &mut [Expr], locals: &HashSet<String>, changed: &mut bool) {
    let arg = &mut args[0];
    let recv_lit = is_literal(recv);
    let arg_lit = is_literal(arg);
    if !recv_lit && (arg_lit || key(arg, locals) < key(recv, locals)) {
        std::mem::swap(recv, arg);
        *changed = true;
    }
}

/// `Boolean.parseBoolean(s)` / `Boolean.valueOf(s)` in a truth position is `s`: a truth test of the
/// string.
fn truth_operand(e: &mut Expr) -> bool {
    let replace = match e {
        Expr::StaticCall { scope, name, args }
            if scope == "Boolean"
                && matches!(name.as_str(), "parseBoolean" | "valueOf")
                && args.len() == 1 =>
        {
            Some(args.pop().unwrap_or_else(placeholder))
        }
        Expr::Method { recv, name, args }
            if matches!(recv.as_ref(), Expr::Name(n) if n == "Boolean")
                && matches!(name.as_str(), "parseBoolean" | "valueOf")
                && args.len() == 1 =>
        {
            Some(args.pop().unwrap_or_else(placeholder))
        }
        _ => None,
    };
    if let Some(r) = replace {
        *e = r;
        true
    } else {
        false
    }
}

/// A condition's truth tests written as their operands, through `!`, `&&` and `||`.
fn rewrite_truth(e: &mut Expr) -> bool {
    let mut changed = truth_operand(e);
    match e {
        Expr::Unary { op, e: inner } if op == "!" => changed |= rewrite_truth(inner),
        Expr::Binary { op, l, r } if op == "&&" || op == "||" => {
            changed |= rewrite_truth(l);
            changed |= rewrite_truth(r);
        }
        _ => {}
    }
    changed
}

/// Apply the expression rules to every expression of the body, innermost first; and to every
/// condition, the truth-test rule.
fn expressions(stmts: &mut [Stmt], scope: &Scope) -> bool {
    let mut changed = false;
    conditions(stmts, &mut |c| changed |= rewrite_truth(c));
    all_exprs(stmts, &mut |root| rewrite_tree(root, &scope.locals, &mut changed));
    changed
}

/// Every condition a statement tests, at every depth.
fn conditions(stmts: &mut [Stmt], f: &mut dyn FnMut(&mut Expr)) {
    for s in stmts.iter_mut() {
        match s {
            Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } | Stmt::For { cond: Some(cond), .. } => f(cond),
            _ => {}
        }
        nested_lists(s, &mut |list| conditions(list, f));
    }
}

fn rewrite_tree(e: &mut Expr, locals: &HashSet<String>, changed: &mut bool) {
    for_each_child_mut(e, &mut |c| rewrite_tree(c, locals, changed));
    rewrite(e, locals, changed);
}

fn rewrite(e: &mut Expr, locals: &HashSet<String>, changed: &mut bool) {
    if null_test(e) {
        *changed = true;
        return;
    }
    match e {
        Expr::Unary { op, e: inner } if op == "!" => {
            let reducible = match inner.as_ref() {
                Expr::Unary { op, .. } => op == "!",
                Expr::Binary { op, .. } => {
                    matches!(op.as_str(), "==" | "!=" | "&&" | "||" | "<" | "<=" | ">" | ">=")
                }
                Expr::Bool(_) => true,
                _ => false,
            };
            if reducible {
                let operand = std::mem::replace(&mut **inner, placeholder());
                *e = negate(operand);
                *changed = true;
            }
        }
        Expr::Binary { op, l, r }
            if (op == "&&" || op == "||") && matches!(r.as_ref(), Expr::Binary { op: ro, .. } if ro == op) =>
        {
            let Expr::Binary { l: rl, r: rr, .. } = std::mem::replace(&mut **r, placeholder()) else { return };
            let left = std::mem::replace(&mut **l, placeholder());
            let op = op.clone();
            *e = binary(binary(left, &op, *rl), &op, *rr);
            *changed = true;
        }
        Expr::Binary { op, l, r } => {
            if let Some(mirrored) = mirror(op) {
                if key(r, locals) < key(l, locals) {
                    std::mem::swap(l, r);
                    mirrored.clone_into(op);
                    *changed = true;
                }
            }
        }
        Expr::AssignExpr { op, l, r } if op != "=" => {
            let base = op.trim_end_matches('=').to_owned();
            let (lhs, rhs) = (std::mem::replace(l, Box::new(placeholder())), std::mem::replace(r, Box::new(placeholder())));
            *e = Expr::AssignExpr { op: "=".to_owned(), l: Box::new((*lhs).clone()), r: Box::new(binary((*lhs).clone(), &base, (*rhs).clone())) };
            *changed = true;
        }
        Expr::Method { recv, name, args }
            if matches!(recv.as_ref(), Expr::Name(n) if n == "Objects") && name == "equals" && args.len() == 2 =>
        {
            if key(&args[1], locals) < key(&args[0], locals) {
                args.swap(0, 1);
                *changed = true;
            }
        }
        Expr::StaticCall { scope, name, args } if scope == "Objects" && name == "equals" && args.len() == 2 => {
            if key(&args[1], locals) < key(&args[0], locals) {
                args.swap(0, 1);
                *changed = true;
            }
        }
        Expr::Method { recv, name, args } if (name == "equals" || name == "equalsIgnoreCase") && args.len() == 1 => {
            orient_equals(recv, args, locals, changed);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Statement rules
// ---------------------------------------------------------------------------

/// Whether control never falls out of the bottom of `s`.
fn ends(s: &Stmt) -> bool {
    match s {
        Stmt::Return(_) | Stmt::Throw(_) | Stmt::Break(_) | Stmt::Continue(_) => true,
        Stmt::If { then, els: Some(els), .. } => {
            block_ends(then)
                && match els.as_ref() {
                    Else::Block(b) => block_ends(b),
                    Else::If(alt) => ends(alt),
                }
        }
        Stmt::Block(b) => block_ends(b),
        Stmt::Labeled { stmt, .. } => ends(stmt),
        _ => false,
    }
}

fn block_ends(stmts: &[Stmt]) -> bool {
    stmts.last().is_some_and(ends)
}

/// Whether a block's remainder ends: its last statement does, or — in the body of a method that
/// returns nothing — it runs off the end, which is a return.
fn rest_ends(stmts: &[Stmt], ctx: Ctx) -> bool {
    block_ends(stmts) || (ctx.top && ctx.void && !stmts.is_empty())
}

/// Whether a value is, by Java's own conventions, a failure: `null`, `false`, zero, a negative number,
/// or an empty `Optional` / `Collection`.
fn errorish(e: &Expr) -> bool {
    match e {
        Expr::Null | Expr::Bool(false) => true,
        Expr::Unary { op, .. } => op == "-",
        Expr::Lit(tag, n) if tag == "Num" => n == "0",
        Expr::Method { recv, name, args } if args.is_empty() => {
            let is_scope = |s: &str| matches!(recv.as_ref(), Expr::Name(n) if n == s);
            (name == "empty" && (is_scope("Optional") || is_scope("OptionalInt")))
                || (name == "emptyList" && is_scope("Collections"))
        }
        Expr::StaticCall { scope, name, args } if args.is_empty() => {
            (name == "empty" && (scope == "Optional" || scope == "OptionalInt"))
                || (name == "emptyList" && scope == "Collections")
        }
        _ => false,
    }
}

/// Whether a branch is the failure path: it throws, or returns a failure as its last value.
fn fails(stmts: &[Stmt]) -> bool {
    match stmts.last() {
        Some(Stmt::Throw(_)) => true,
        Some(Stmt::Return(Some(value))) => errorish(value),
        _ => false,
    }
}

/// How much a run of statements does: one per statement, nested ones included.
fn weight(stmts: &[Stmt]) -> usize {
    stmts
        .iter()
        .map(|s| {
            1 + match s {
                Stmt::If { then, els, .. } => {
                    weight(then)
                        + match els.as_deref() {
                            Some(Else::Block(b)) => weight(b),
                            Some(Else::If(alt)) => weight(std::slice::from_ref(alt)),
                            None => 0,
                        }
                }
                Stmt::While { body, .. }
                | Stmt::DoWhile { body, .. }
                | Stmt::For { body, .. }
                | Stmt::ForEach { body, .. }
                | Stmt::Block(body) => weight(body),
                Stmt::Switch { cases, .. } => cases.iter().map(|c| weight(&c.body)).sum(),
                Stmt::Try { body, catches, finally, .. } => {
                    weight(body)
                        + catches.iter().map(|c| weight(&c.body)).sum::<usize>()
                        + finally.as_deref().map_or(0, weight)
                }
                Stmt::Labeled { stmt, .. } => weight(std::slice::from_ref(stmt)),
                _ => 0,
            }
        })
        .sum()
}

/// Of two branches that both end, which one the guard holds: the lighter — the early exit the method
/// is written around; at equal weight the one that fails; and only then the positive test. `true` when
/// the guard should hold `other` rather than `guard`, tested by `test`.
///
/// Each step decides a pair the same way from either side, so the rule is confluent: once swapped, the
/// same comparison keeps it.
fn guard_should_swap(guard: &[Stmt], other: &[Stmt], test: &Expr, locals: &HashSet<String>) -> bool {
    let (g, o) = (weight(guard), weight(other));
    if g != o {
        return o < g;
    }
    let (guard_fails, other_fails) = (fails(guard), fails(other));
    if guard_fails != other_fails {
        return other_fails;
    }
    leans_negative(test, locals)
}

/// One round of the statement rules over a block and the blocks inside it.
fn block(stmts: &mut Vec<Stmt>, scope: &Scope, ctx: Ctx) -> bool {
    let mut changed = false;
    let inner = Ctx { top: false, ..ctx };
    for s in stmts.iter_mut() {
        nested_lists(s, &mut |b| changed |= block(b, scope, inner));
    }
    let mut i = 0;
    while i < stmts.len() {
        let rewritten = require_guard(stmts, i)
            || truth_fold(stmts, i)
            || forever(stmts, i)
            || switch_to_ladder(stmts, i)
            || orient_branches(stmts, i, scope)
            || guard_form(stmts, i, scope)
            || orient_guard(stmts, i, scope, ctx)
            || (ctx.inline && inline_temporary(stmts, i, scope))
            || expand_compound(stmts, i)
            || final_return(stmts, i, ctx);
        if rewritten {
            changed = true;
        } else {
            i += 1;
        }
    }
    changed
}

/// `if (Objects.requireNonNull(x[, m]) == y) { B }` → `if (x == null) throw new NullPointerException(m);
/// if (y == x) { B }` — the guard form of the library's null check. Only the single-proposition form,
/// only when `x` is re-evaluable without effect.
fn require_guard(stmts: &mut Vec<Stmt>, i: usize) -> bool {
    let Stmt::If { cond, then, els } = &stmts[i] else { return false };
    if !els.is_none() {
        return false;
    }
    let Expr::Binary { op, l, r } = cond else { return false };
    if op != "==" {
        return false;
    }
    let (x, msg, y) = match (require_call(l), require_call(r)) {
        (Some((x, msg)), None) => (x, msg, r.as_ref().clone()),
        (None, Some((x, msg))) => (x, msg, l.as_ref().clone()),
        _ => return false,
    };
    if !is_simple(&x) {
        return false;
    }
    let body = then.clone();
    let null_check = Stmt::If {
        cond: binary(x.clone(), "==", Expr::Null),
        then: vec![Stmt::Throw(Expr::New { class: "NullPointerException".to_owned(), args: msg.into_iter().collect() })],
        els: None,
    };
    let rest = Stmt::If { cond: binary(y, "==", x), then: body, els: None };
    stmts.splice(i..=i, [null_check, rest]);
    true
}

/// The `(x, m)` of `Objects.requireNonNull(x[, m])`, `m` the message when written.
fn require_call(e: &Expr) -> Option<(Expr, Option<Expr>)> {
    let (name, args) = match e {
        Expr::StaticCall { scope, name, args } if scope == "Objects" => (name.as_str(), args),
        Expr::Method { recv, name, args } if matches!(recv.as_ref(), Expr::Name(n) if n == "Objects") => (name.as_str(), args),
        _ => return None,
    };
    if name != "requireNonNull" {
        return None;
    }
    match args.as_slice() {
        [x] => Some((x.clone(), None)),
        [x, m] => Some((x.clone(), Some(m.clone()))),
        _ => None,
    }
}

/// Whether `x` can be evaluated twice with no observable difference — no call, no update.
fn is_simple(e: &Expr) -> bool {
    match e {
        Expr::Name(_) | Expr::Lit(..) | Expr::Bool(_) | Expr::Null | Expr::StaticField { .. } | Expr::ClassConst { .. } => true,
        Expr::Field { base, .. } => is_simple(base),
        Expr::Index { base, i } => is_simple(base) && is_simple(i),
        _ => false,
    }
}

/// `if (c) return true; return false;` → `return c;` and its negation → `return !c;`: the
/// return-true/false pattern IS the condition. Both the two-branch and the guard spelling.
fn truth_fold(stmts: &mut Vec<Stmt>, i: usize) -> bool {
    let keep = match &stmts[i] {
        Stmt::If { then, els: Some(els), .. } if matches!(els.as_ref(), Else::Block(_)) => {
            let Else::Block(alt) = els.as_ref() else { unreachable!() };
            match (then.as_slice(), alt.as_slice()) {
                ([Stmt::Return(Some(Expr::Bool(true)))], [Stmt::Return(Some(Expr::Bool(false)))]) => Some(true),
                ([Stmt::Return(Some(Expr::Bool(false)))], [Stmt::Return(Some(Expr::Bool(true)))]) => Some(false),
                _ => None,
            }
        }
        Stmt::If { then, els: None, .. } if i + 1 < stmts.len() => match (then.as_slice(), &stmts[i + 1]) {
            ([Stmt::Return(Some(Expr::Bool(true)))], Stmt::Return(Some(Expr::Bool(false)))) => Some(true),
            ([Stmt::Return(Some(Expr::Bool(false)))], Stmt::Return(Some(Expr::Bool(true)))) => Some(false),
            _ => None,
        },
        _ => None,
    };
    let Some(keep_cond) = keep else { return false };
    let Stmt::If { cond, els, .. } = &mut stmts[i] else { return false };
    let cond = std::mem::replace(cond, placeholder());
    if els.is_none() {
        stmts.remove(i + 1);
    }
    stmts[i] = Stmt::Return(Some(if keep_cond { cond } else { not(cond) }));
    true
}

/// `while (true)` → `for (;;)`: one loop with no test, written two ways.
fn forever(stmts: &mut [Stmt], i: usize) -> bool {
    let Stmt::While { cond, .. } = &stmts[i] else { return false };
    if !matches!(cond, Expr::Bool(true)) {
        return false;
    }
    let Stmt::While { body, .. } = std::mem::replace(&mut stmts[i], Stmt::Unknown("while".to_owned())) else { return false };
    stmts[i] = Stmt::For { init: None, cond: None, update: Vec::new(), body };
    true
}

/// Whether a clause body leaves its `switch` in a way a ladder cannot say: an unlabeled `break` that
/// would bind to the `switch`, anywhere but inside a loop or `switch` of its own.
fn leaves_switch(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Break(None) => true,
        Stmt::If { then, els, .. } => {
            leaves_switch(then)
                || match els.as_deref() {
                    Some(Else::Block(b)) => leaves_switch(b),
                    Some(Else::If(alt)) => leaves_switch(std::slice::from_ref(alt)),
                    None => false,
                }
        }
        Stmt::Try { body, catches, finally, .. } => {
            leaves_switch(body)
                || catches.iter().any(|c| leaves_switch(&c.body))
                || finally.as_deref().is_some_and(leaves_switch)
        }
        Stmt::Synchronized { body, .. } => leaves_switch(body),
        Stmt::Block(b) => leaves_switch(b),
        Stmt::Labeled { stmt, .. } => leaves_switch(std::slice::from_ref(stmt)),
        // A `break` inside a loop or nested `switch` binds to it, not the outer switch.
        _ => false,
    })
}

/// One rung of a ladder: the case labels that share a body, and the body without the `break` that ends
/// it — or `None` when some case falls into the next or `break`s from deeper inside.
type Rung = (Vec<Expr>, Vec<Stmt>);

fn rungs(cases: &[Case]) -> Option<Vec<Rung>> {
    let mut out = Vec::new();
    for (idx, case) in cases.iter().enumerate() {
        let is_last = idx + 1 == cases.len();
        let mut body = case.body.clone();
        if matches!(body.last(), Some(Stmt::Break(None))) {
            body.pop();
        } else if !is_last && !block_ends(&body) {
            return None; // falls into the next case
        }
        if leaves_switch(&body) {
            return None;
        }
        out.push((case.values.clone(), body));
    }
    Some(out)
}

/// `switch (v) { case a: case b: A; break; default: D; }` → `if (v == a || v == b) { A } else { D }`.
fn switch_to_ladder(stmts: &mut [Stmt], i: usize) -> bool {
    let Stmt::Switch { value, cases } = &stmts[i] else { return false };
    let Some(rungs) = rungs(cases) else { return false };
    let (defaults, tests): (Vec<_>, Vec<_>) = rungs.into_iter().partition(|(values, _)| values.is_empty());
    if tests.is_empty() || defaults.len() > 1 {
        return false;
    }
    let test = |values: &[Expr]| -> Option<Expr> {
        values
            .iter()
            .cloned()
            .map(|v| binary(value.clone(), "==", v))
            .reduce(|a, b| binary(a, "||", b))
    };
    // Built from the last rung up, each rung the `else` of the one before — the nested shape every rule reads.
    let mut otherwise: Option<Vec<Stmt>> = defaults.into_iter().next().map(|(_, body)| body);
    for (values, body) in tests.into_iter().rev() {
        let Some(cond) = test(&values) else { return false };
        let rung = Stmt::If { cond, then: body, els: otherwise.map(|b| Box::new(Else::Block(b))) };
        otherwise = Some(vec![rung]);
    }
    let Some(mut ladder) = otherwise else { return false };
    let Some(rung) = ladder.pop() else { return false };
    stmts[i] = rung;
    true
}

fn swap_branches(then: &mut Vec<Stmt>, els: &mut Else) {
    if let Else::Block(alt) = els {
        std::mem::swap(then, alt);
    }
}

/// An `if … else` that neither branch leaves through is written with its positive test first.
fn orient_branches(stmts: &mut [Stmt], i: usize, scope: &Scope) -> bool {
    let Stmt::If { cond, then, els: Some(els), .. } = &mut stmts[i] else { return false };
    let Else::Block(alt) = els.as_ref() else { return false };
    if block_ends(then) || block_ends(alt) || !leans_negative(cond, &scope.locals) {
        return false;
    }
    let c = std::mem::replace(cond, placeholder());
    *cond = negate(c);
    swap_branches(then, els);
    true
}

/// `if (c) { <ends> } else { B }` → `if (c) { <ends> }` then `B`; when only the `else` ends, the test is
/// negated and the branches change places first; when both end, the guard orientation decides.
fn guard_form(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope) -> bool {
    let Stmt::If { cond, then, els: Some(els), .. } = &mut stmts[i] else { return false };
    let Else::Block(alt) = els.as_ref() else { return false };
    let then_ends = block_ends(then);
    let else_ends = block_ends(alt);
    if !then_ends && !else_ends {
        return false;
    }
    let flip = if then_ends && else_ends { guard_should_swap(then, alt, cond, &scope.locals) } else { !then_ends };
    if flip {
        let c = std::mem::replace(cond, placeholder());
        *cond = negate(c);
        swap_branches(then, els);
    }
    let Stmt::If { els, .. } = &mut stmts[i] else { return false };
    let Some(taken) = els.take() else { return false };
    let Else::Block(tail) = *taken else { return false };
    let at = i + 1;
    stmts.splice(at..at, tail);
    true
}

/// A guard whose remainder also ends is a two-branch `if` written as a guard: `if (c) { <ends> }` then
/// `<ends>` ≡ `if (!c) { <ends> }` then `<ends>`. Java's representative tests the failure.
fn orient_guard(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope, ctx: Ctx) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Stmt::If { cond, then, els: None, .. } = &stmts[i] else { return false };
    if !block_ends(then) || !rest_ends(&stmts[i + 1..], ctx) || !guard_should_swap(then, &stmts[i + 1..], cond, &scope.locals) {
        return false;
    }
    let rest: Vec<Stmt> = stmts.drain(i + 1..).collect();
    let Some(Stmt::If { cond, then, .. }) = stmts.pop() else { return false };
    stmts.push(Stmt::If { cond: negate(cond), then: rest, els: None });
    stmts.extend(then);
    true
}

/// `T t = e;` then `S(t)`, with `t` bound once and read once, there → `S(e)`.
fn inline_temporary(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Stmt::Let { name: t, init: Some(value), .. } = &stmts[i] else { return false };
    if scope.stores(t) != 1
        || scope.reads(t) != 1
        || scope.pinned.contains(t)
        || scope.nested_reads.contains(t)
        || names_read(value).contains(t)
    {
        return false;
    }
    // A loop re-evaluates its test; a compound statement re-evaluates or scopes — a temporary read
    // there is not one single read.
    if matches!(
        &stmts[i + 1],
        Stmt::While { .. }
            | Stmt::DoWhile { .. }
            | Stmt::For { .. }
            | Stmt::ForEach { .. }
            | Stmt::Labeled { .. }
            | Stmt::Block(_)
            | Stmt::Switch { .. }
            | Stmt::Try { .. }
            | Stmt::Synchronized { .. }
    ) {
        return false;
    }
    let t = t.clone();
    let mut direct = 0;
    direct_exprs(&mut stmts[i + 1], &mut |e| direct += reads_of(e, &t));
    if direct != 1 {
        return false;
    }
    let Stmt::Let { init, .. } = stmts.remove(i) else { return false };
    let mut value = init;
    direct_exprs(&mut stmts[i], &mut |e| substitute(e, &t, &mut value));
    true
}

fn substitute(e: &mut Expr, name: &str, value: &mut Option<Expr>) {
    if matches!(e, Expr::Name(n) if n == name) {
        if let Some(v) = value.take() {
            *e = v;
        }
        return;
    }
    if value.is_some() {
        for_each_child_mut(e, &mut |c| substitute(c, name, value));
    }
}

/// `x++` (a bare update statement) → `x = x + 1`; `x--` → `x = x - 1`. A compound assignment
/// (`x += e`) is an `AssignExpr`, rewritten to `x = x + e` by the expression rules in every position;
/// the update is statement-only — in expression position an `Update` stays an `Update`.
fn expand_compound(stmts: &mut [Stmt], i: usize) -> bool {
    let Stmt::Expr(Expr::Update { inc, e, .. }) = &stmts[i] else { return false };
    let base = if *inc { "+" } else { "-" };
    let target = e.as_ref().clone();
    let one = Expr::Lit("Num".to_owned(), "1".to_owned());
    let lhs = target.clone();
    stmts[i] = Stmt::Expr(Expr::AssignExpr { op: "=".to_owned(), l: Box::new(lhs), r: Box::new(binary(target, base, one)) });
    true
}

/// A bare `return;` at the end of a method that returns nothing is falling off its end.
fn final_return(stmts: &mut Vec<Stmt>, i: usize, ctx: Ctx) -> bool {
    if !ctx.top || !ctx.void || i + 1 != stmts.len() || stmts.len() < 2 {
        return false;
    }
    if matches!(&stmts[i], Stmt::Return(None)) {
        stmts.pop();
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use crate::defs::scan_source;
    use dup_defs_core::{Analysis, ScanOpts};
    use std::sync::Arc;

    fn method(body: &str) -> Analysis {
        let src = format!("class T {{\n    {body}\n}}\n");
        let f: Arc<str> = Arc::from("T.java");
        scan_source(&src, &f, &ScanOpts::default()).into_iter().find_map(|d| d.analysis).expect("a method")
    }

    fn xname(body: &str) -> String {
        method(body).xname_canonical
    }

    fn lines(body: &str) -> Vec<String> {
        method(body).type3_lines
    }

    #[test]
    fn negations_have_one_spelling() {
        assert_eq!(
            xname("boolean f(boolean a, boolean b) { return !(a && b); }"),
            xname("boolean f(boolean a, boolean b) { return !a || !b; }")
        );
        assert_eq!(
            xname("boolean f(int a, int b) { return !(a == b); }"),
            xname("boolean f(int a, int b) { return a != b; }")
        );
        assert_eq!(
            xname("boolean f(int a, int b) { return !(a < b); }"),
            xname("boolean f(int a, int b) { return a >= b; }")
        );
        assert_eq!(
            xname("boolean f(boolean a, boolean b, boolean c) { return a || (b || c); }"),
            xname("boolean f(boolean a, boolean b, boolean c) { return a || b || c; }")
        );
    }

    #[test]
    fn a_comparison_orients_by_shape() {
        // `LIMIT` is a free name; the local `x` sorts after it, so the mirrored spellings meet.
        assert_eq!(
            xname("boolean f(int x) { return x > LIMIT; }"),
            xname("boolean f(int x) { return LIMIT < x; }")
        );
        assert_eq!(
            xname("boolean f(int x) { return x >= LIMIT; }"),
            xname("boolean f(int x) { return LIMIT <= x; }")
        );
    }

    #[test]
    fn a_null_test_has_one_spelling() {
        assert_eq!(
            xname("boolean f(Object x) { return Objects.isNull(x); }"),
            xname("boolean f(Object x) { return x == null; }")
        );
        assert_eq!(
            xname("boolean f(Object x) { return Objects.nonNull(x); }"),
            xname("boolean f(Object x) { return x != null; }")
        );
        assert_eq!(
            xname("boolean f(Object x) { return null == x; }"),
            xname("boolean f(Object x) { return x == null; }")
        );
    }

    #[test]
    fn a_require_non_null_guard_is_the_null_check_it_throws() {
        let guarded = "boolean f(Object x, Object y) {\n\tif (Objects.requireNonNull(x, \"msg\") == y) {\n\t\treturn true;\n\t}\n\treturn false;\n}";
        let spelled_out = "boolean f(Object x, Object y) {\n\tif (x == null) {\n\t\tthrow new NullPointerException(\"msg\");\n\t}\n\tif (y == x) {\n\t\treturn true;\n\t}\n\treturn false;\n}";
        assert_eq!(xname(guarded), xname(spelled_out));
    }

    #[test]
    fn an_equals_orients_with_the_literal_as_receiver() {
        assert_eq!(
            xname("boolean f(String x) { return x.equals(\"lit\"); }"),
            xname("boolean f(String x) { return \"lit\".equals(x); }")
        );
        assert_eq!(
            xname("boolean f(String x) { return x.equalsIgnoreCase(\"lit\"); }"),
            xname("boolean f(String x) { return \"lit\".equalsIgnoreCase(x); }")
        );
        assert_eq!(
            xname("boolean f(String x) { return Objects.equals(x, \"lit\"); }"),
            xname("boolean f(String x) { return Objects.equals(\"lit\", x); }")
        );
        // `==` stays identity: it is not rewritten to `equals`.
        assert!(xname("boolean f(Object a, Object b) { return a == b; }").contains("'=='"), "{}", xname("boolean f(Object a, Object b) { return a == b; }"));
    }

    #[test]
    fn a_truth_fold_is_the_condition() {
        // The positive fold's result `return c;` is a single return of a name — trivial by design —
        // so only the negated spelling, whose result `return !c;` survives the filter, is asserted.
        assert_eq!(
            xname("boolean f(boolean c) { if (c) { return false; } return true; }"),
            xname("boolean f(boolean c) { return !c; }")
        );
        assert_eq!(
            xname("boolean f(String s) { if (Boolean.parseBoolean(s)) { g(); } return true; }"),
            xname("boolean f(String s) { if (s) { g(); } return true; }")
        );
    }

    #[test]
    fn a_guard_and_an_else_are_one_form() {
        let guarded = "String f(String x) {\n\tif (x == null) {\n\t\treturn null;\n\t}\n\tuse(x);\n\treturn x;\n}";
        let branched = "String f(String x) {\n\tif (x == null) {\n\t\treturn null;\n\t} else {\n\t\tuse(x);\n\t\treturn x;\n\t}\n}";
        let negated = "String f(String x) {\n\tif (x != null) {\n\t\tuse(x);\n\t\treturn x;\n\t} else {\n\t\treturn null;\n\t}\n}";
        assert_eq!(xname(guarded), xname(branched));
        assert_eq!(xname(guarded), xname(negated));
    }

    #[test]
    fn a_temporary_is_inlined_for_the_exact_pass_and_kept_for_type3() {
        let named = "String f(int x) { String t = g(x); return t; }";
        let inlined = "String f(int x) { return g(x); }";
        assert_eq!(xname(named), xname(inlined));
        assert_ne!(lines(named), lines(inlined), "Type-3 keeps the statement");
        // A temporary passed to a method is pinned: the call may mutate it or overload on its type.
        let pinned = "String f(int x) { String t = g(x); return h(t); }";
        assert!(xname(pinned).contains("Let("), "{}", xname(pinned));
    }

    #[test]
    fn small_spellings_meet() {
        let plus = "int f(int x) { x = x + 1; return g(x); }";
        assert_eq!(xname("int f(int x) { x += 1; return g(x); }"), xname(plus));
        assert_eq!(xname("int f(int x) { x++; return g(x); }"), xname(plus));
        assert_eq!(xname("int f(int x) { ++x; return g(x); }"), xname(plus));
        assert_eq!(xname("int f(int x) { x -= 1; return g(x); }"), xname("int f(int x) { x = x - 1; return g(x); }"));
        assert_eq!(
            xname("void f(int x) { while (true) { g(x); } }"),
            xname("void f(int x) { for (;;) { g(x); } }")
        );
    }

    #[test]
    fn a_switch_is_the_ladder_it_abbreviates() {
        let switched = "int f(int x) {\n\tswitch (x) {\n\tcase 1:\n\t\treturn a();\n\tcase 2:\n\tcase 3:\n\t\treturn b();\n\tdefault:\n\t\treturn c();\n\t}\n}";
        let ladder = "int f(int x) {\n\tif (x == 1) {\n\t\treturn a();\n\t} else if (x == 2 || x == 3) {\n\t\treturn b();\n\t}\n\treturn c();\n}";
        assert_eq!(xname(switched), xname(ladder));
    }

    #[test]
    fn a_switch_that_breaks_out_stays_a_switch() {
        let src = "int f(int x) {\n\tswitch (x) {\n\tcase 1:\n\t\tif (ok()) {\n\t\t\tbreak;\n\t\t}\n\t\tg();\n\t\tbreak;\n\t}\n\treturn x;\n}";
        assert!(xname(src).contains("Switch("), "{}", xname(src));
    }

    #[test]
    fn an_else_if_chain_is_one_shape() {
        let chained = "int f(int x) {\n\tif (x == 1) {\n\t\treturn a();\n\t} else if (x == 2) {\n\t\treturn b();\n\t} else {\n\t\treturn c();\n\t}\n}";
        let nested = "int f(int x) {\n\tif (x == 1) {\n\t\treturn a();\n\t} else {\n\t\tif (x == 2) {\n\t\t\treturn b();\n\t\t} else {\n\t\t\treturn c();\n\t\t}\n\t}\n}";
        assert_eq!(xname(chained), xname(nested));
    }

    #[test]
    fn a_last_bare_return_is_the_end() {
        assert_eq!(
            xname("void f(int x) { g(x); h(x); return; }"),
            xname("void f(int x) { g(x); h(x); }")
        );
    }
}
