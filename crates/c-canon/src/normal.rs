//! **Semantic normal form** of a C body, taken before its name-agnostic canonicals.
//!
//! The exact pass equates two definitions when their alpha-renamed canonicals are equal, and the
//! Type-3 pass when their statements mostly are. Both read the code as *written*, and the same
//! procedure is written many ways: a guard beside an `if … else`, `if (p != NULL)` beside `if (p)`, a
//! temporary named and used once beside the expression inlined, a `switch` beside the `if` ladder it
//! abbreviates, `x++` beside `x += 1` beside `x = x + 1`. Each of those is a Type-4 clone to a reader
//! and an unrelated body to a pass that compares text.
//!
//! This module rewrites a body into one representative of its equivalence class under the rules the
//! other frontends' normal forms apply, read in C's grammar — oriented, applied to a fixed point, one
//! confluent direction per rule.
//!
//! ## Rules
//!
//! Expressions, innermost first, to a fixed point:
//! - **Negation normal form**: `!!x` → `x`; `!(a && b)` → `!a || !b` (and dually); `!(a == b)` →
//!   `a != b`; `!(a < b)` → `a >= b`; `a || (b || c)` → `a || b || c`. Parentheses are dropped: the
//!   tree already says what they grouped.
//! - **A truth test is its operand**: wherever C tests a value for truth — a condition, an operand of
//!   `!`, `&&`, `||`, a `?:`'s test — `x != 0` and `x != NULL` are `x`, and `x == 0`, `x == NULL` are
//!   `!x`. C defines the one as the other.
//! - **Comparison orientation**: `==` / `!=` and the mirror pairs put their operands in a fixed order, by
//!   their shape with the definition's own names blanked — so a renamed copy orients the same way.
//!
//! Statements, in every block:
//! - **A `switch` is the ladder it abbreviates**: `case a: case b: S; break;` → `if (v == a || v == b)
//!   { S }`, `default` → the last `else` — when every case either ends in its own `break` or leaves the
//!   function, and none falls into the next or `break`s from deeper inside: what a ladder could not say.
//! - **Branch orientation**: an `if … else` that neither branch leaves through, whose test is negated
//!   (or mostly negated), is flipped.
//! - **Guard form**: `if (c) { …; return x; } else { B }` → `if (c) { …; return x; }` then `B`;
//!   mirrored when the `else` is the branch that ends.
//! - **Guard orientation**: when the rest of the block ends too, `if (c) { E }` then `R` is the same
//!   decision as `if (!c) { R }` then `E`. 🔴 The representative is the one C is written in: the guard
//!   holds the *lighter* branch — the early exit the function is written around; at equal weight, the
//!   branch that *fails* — a `goto` to the cleanup, a negative or `NULL` return, an error code, an
//!   `abort` — so `if (ret < 0) return ret;` and `if (!p) goto fail;` keep their shape; and only then
//!   the positive test, as the other languages pick (the Go frontend's rule, for its reason).
//! - **Single-use temporary** (the exact form only): `T t = e;` then a statement that reads `t` once
//!   and nothing else does → that statement with `e` in place of `t`.
//! - **A declaration takes its first value**: `T x;` then `x = e;` → `T x = e;`; `T a = 1, b = 2;` →
//!   two declarations.
//! - `x += e` and `x++` → `x = x + e`; `while (1)` → `for (;;)`; `do { B } while (0)` → `B`.
//! - A final bare `return` in a function that returns nothing is falling off its end.
//!
//! ## What the rules assume
//!
//! The normalized tree is compared, never compiled: a rule need only preserve meaning as a reader
//! judges it. Inlining a temporary drops the conversion its declared type implied; a ladder evaluates
//! the tested value once per test. A name whose address is taken (`&x`) is pinned: inlining it would
//! change what the pointer points at.

use std::collections::HashSet;

use dup_defs_core::normal::{complement, mirror, Scope, MAX_ROUNDS};

use crate::canon::{for_each_child, Child};
use crate::ir::{Case, Elem, Else, Expr, Func, Jump, Spec, Stmt, Type};

/// The normal form of a function. `inline` admits the single-use-temporary rule: the exact pass wants
/// it, the Type-3 pass does not — its unit is the statement, and folding `t = f(x)` into the line that
/// reads `t` turns two short units into one long one (the Python frontend's measurement).
#[must_use]
pub(crate) fn normal_form(f: &Func, inline: bool) -> Func {
    let mut out = f.clone();
    let void = matches!(&f.ret, Type::Named(t) if t == "void");
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
    /// The function's own body, whose end is a return.
    top: bool,
    /// The function returns nothing: falling off its end is a bare `return`.
    void: bool,
    inline: bool,
}

// ---------------------------------------------------------------------------
// Traversal
// ---------------------------------------------------------------------------

/// Every statement list directly inside `s` — its branches and bodies, and the body of every statement
/// expression its expressions hold — not the lists nested in those.
fn nested_lists(s: &mut Stmt, f: &mut dyn FnMut(&mut Vec<Stmt>)) {
    match s {
        Stmt::If { cond, then, els } => {
            stmt_exprs_of(cond, f);
            f(then);
            match els.as_deref_mut() {
                Some(Else::Block(b)) => f(b),
                Some(Else::If(alt)) => nested_lists(alt, f),
                None => {}
            }
        }
        Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
            stmt_exprs_of(cond, f);
            f(body);
        }
        Stmt::For { init, cond, update, body } => {
            for part in [init, update].into_iter().flatten() {
                nested_lists(part, f);
            }
            if let Some(c) = cond {
                stmt_exprs_of(c, f);
            }
            f(body);
        }
        Stmt::Switch { value, cases } => {
            stmt_exprs_of(value, f);
            for case in cases {
                f(&mut case.body);
            }
        }
        Stmt::Block(b) => f(b),
        Stmt::Labeled { stmt, .. } => {
            if let Some(inner) = stmt {
                nested_lists(inner, f);
            }
        }
        Stmt::PreIf { branches, .. } => branches.iter_mut().for_each(&mut *f),
        Stmt::MacroLoop { head, body } => {
            stmt_exprs_of(head, f);
            f(body);
        }
        other => direct_exprs(other, &mut |e| stmt_exprs_of(e, f)),
    }
}

/// The bodies of the statement expressions `e` holds, outermost first.
fn stmt_exprs_of(e: &mut Expr, f: &mut dyn FnMut(&mut Vec<Stmt>)) {
    if let Expr::GnuBlock(body) = e {
        f(body);
        return;
    }
    for_each_child_mut(e, &mut |c| stmt_exprs_of(c, f));
}

/// The expressions a statement evaluates itself, once, before any block inside it runs — the slots a
/// temporary may be inlined into, and the roots the expression rules walk. A loop's condition is not
/// one: it is evaluated again on every pass.
fn direct_exprs(s: &mut Stmt, f: &mut dyn FnMut(&mut Expr)) {
    match s {
        Stmt::Decl(specs) => specs.iter_mut().filter_map(|s| s.init.as_mut()).for_each(f),
        Stmt::Assign { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        Stmt::IncDec { target, .. } | Stmt::Expr(target) => f(target),
        Stmt::Return(Some(value)) | Stmt::Switch { value, .. } => f(value),
        Stmt::If { cond, .. } => f(cond),
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
        Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
            f(cond);
            all_exprs(body, f);
        }
        Stmt::For { init, cond, update, body } => {
            for part in [init, update].into_iter().flatten() {
                stmt_exprs(part, f);
            }
            if let Some(c) = cond {
                f(c);
            }
            all_exprs(body, f);
        }
        Stmt::Switch { value, cases } => {
            f(value);
            for case in cases {
                if let Some(v) = &mut case.value {
                    f(v);
                }
                all_exprs(&mut case.body, f);
            }
        }
        Stmt::Block(b) => all_exprs(b, f),
        Stmt::Labeled { stmt, .. } => {
            if let Some(inner) = stmt {
                stmt_exprs(inner, f);
            }
        }
        Stmt::PreIf { branches, .. } => branches.iter_mut().for_each(|b| all_exprs(b, f)),
        Stmt::MacroLoop { head, body } => {
            f(head);
            all_exprs(body, f);
        }
        other => direct_exprs(other, f),
    }
}

/// [`for_each_child`], mutably: every direct sub-expression, and the expressions of a statement
/// expression's body.
fn for_each_child_mut(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    match e {
        Expr::Binary { l, r, .. } | Expr::Assign { l, r, .. } | Expr::Comma(l, r) => {
            f(l);
            f(r);
        }
        Expr::Unary { e, .. }
        | Expr::Update { e, .. }
        | Expr::Paren(e)
        | Expr::Field { e, .. }
        | Expr::Cast { e, .. }
        | Expr::SizeofValue(e)
        | Expr::Generic(e) => f(e),
        Expr::Cond { c, t, f: alt } => {
            f(c);
            if let Some(t) = t {
                f(t);
            }
            f(alt);
        }
        Expr::Call { f: callee, args } => {
            f(callee);
            args.iter_mut().for_each(f);
        }
        Expr::Index { e, i } => {
            f(e);
            f(i);
        }
        Expr::Compound { elems, .. } | Expr::InitList(elems) => {
            for el in elems {
                match el {
                    Elem::Plain(v) | Elem::Field(_, v) => f(v),
                    Elem::Index(i, v) => {
                        f(i);
                        f(v);
                    }
                }
            }
        }
        Expr::GnuBlock(body) => all_exprs(body, f),
        _ => {}
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
                **els = Else::Block(vec![inner]);
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
                        **els = Else::If(inner);
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
    for name in f.params.iter().filter_map(|p| p.name.as_deref()) {
        walk.scope.bind(name);
    }
    walk.stmts(&f.body);
    scope
}

struct ScopeWalk<'s> {
    scope: &'s mut Scope,
    depth: usize,
}

impl ScopeWalk<'_> {
    /// An assignment's target: a plain name is a store, anything else (`a.b`, `p[i]`, `*p`) a read of
    /// what it reaches into.
    fn store_target(&mut self, e: &Expr) {
        match e {
            Expr::Name(n) => self.scope.store(n),
            other => self.expr(other),
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Decl(specs) => {
                for spec in specs {
                    if let Some(init) = &spec.init {
                        self.expr(init);
                    }
                    self.scope.bind(&spec.name);
                }
            }
            Stmt::Assign { op, lhs, rhs } => {
                self.expr(rhs);
                self.store_target(lhs);
                // A compound assignment reads what it writes.
                if op != "=" {
                    self.expr(lhs);
                }
            }
            Stmt::IncDec { target, .. } => {
                self.store_target(target);
                self.expr(target);
            }
            other => crate::canon::each_stmt_part(other, &mut |part| match part {
                crate::canon::Part::Spec(spec) => {
                    if let Some(init) = &spec.init {
                        self.expr(init);
                    }
                    self.scope.bind(&spec.name);
                }
                crate::canon::Part::Stmt(inner) => self.stmt(inner),
                crate::canon::Part::Stmts(list) => self.stmts(list),
                crate::canon::Part::Expr(e) => self.expr(e),
            }),
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Name(n) => self.scope.read(n, self.depth > 0),
            // `&x` — what the pointer points at is `x` itself; nothing may be assumed about it.
            Expr::Unary { op, e: inner } if op == "&" => {
                if let Expr::Name(n) = inner.as_ref() {
                    self.scope.pinned.insert(n.clone());
                }
                self.expr(inner);
            }
            // An assignment or an increment inside an expression stores as well as reads.
            Expr::Assign { l, r, .. } => {
                self.expr(r);
                self.store_target(l);
                self.expr(l);
            }
            Expr::Update { e: inner, .. } => {
                self.store_target(inner);
                self.expr(inner);
            }
            Expr::GnuBlock(body) => {
                self.depth += 1;
                self.stmts(body);
                self.depth -= 1;
            }
            other => for_each_child(other, &mut |child| match child {
                Child::Expr(c) => self.expr(c),
                Child::Body(b) => self.stmts(b),
            }),
        }
    }
}

/// Names read anywhere in `e`.
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

/// Whether a value is zero to C: `0`, `'\0'`, `NULL`, `false`.
fn is_zero(e: &Expr) -> bool {
    match e {
        Expr::Null | Expr::Bool(false) => true,
        Expr::Lit("Int", t) => t.trim_end_matches(['u', 'U', 'l', 'L']) == "0" || t == "0x0",
        Expr::Lit("Char", t) => t == "\\0",
        Expr::Cast { e, .. } => is_zero(e),
        _ => false,
    }
}

/// A value tested for truth, written as its operand: `x != 0` is `x`, `x == NULL` is `!x`.
fn truth_operand(e: &mut Expr) -> bool {
    let Expr::Binary { op, l, r } = e else { return false };
    let negated = match op.as_str() {
        "!=" => false,
        "==" => true,
        _ => return false,
    };
    let operand = if is_zero(r) {
        std::mem::replace(&mut **l, placeholder())
    } else if is_zero(l) {
        std::mem::replace(&mut **r, placeholder())
    } else {
        return false;
    };
    *e = if negated { not(operand) } else { operand };
    true
}

/// Whether a value is, by C's own conventions, a failure: a negative number or error code, `NULL`, a
/// name that says so (`err`, `ret_err`, `EINVAL`), or a call that builds one (`AVERROR(…)`,
/// `ERR_PTR(…)`).
fn errorish(e: &Expr) -> bool {
    let says_error = |name: &str| {
        let lower = name.to_ascii_lowercase();
        lower.contains("err") || lower.contains("fail")
    };
    let errno = |name: &str| name.len() > 1 && name.starts_with('E') && name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
    match e {
        Expr::Unary { op, .. } => op == "-",
        Expr::Null => true,
        Expr::Name(n) => says_error(n) || errno(n),
        Expr::Call { f, .. } => matches!(f.as_ref(), Expr::Name(n) if says_error(n)),
        Expr::Paren(inner) | Expr::Cast { e: inner, .. } => errorish(inner),
        _ => false,
    }
}

/// The functions that do not return.
fn noreturn(callee: &str) -> bool {
    matches!(callee, "abort" | "exit" | "_exit" | "_Exit" | "quick_exit" | "longjmp" | "siglongjmp" | "__builtin_unreachable" | "panic" | "BUG")
}

fn noreturn_call(s: &Stmt) -> bool {
    matches!(s, Stmt::Expr(Expr::Call { f, .. }) if matches!(f.as_ref(), Expr::Name(n) if noreturn(n)))
}

/// Whether a branch is the failure path: it jumps to a label (C's cleanup), returns a failure, or
/// does not return.
fn fails(stmts: &[Stmt]) -> bool {
    match stmts.last() {
        Some(Stmt::Jump { kind: Jump::Goto, .. }) => true,
        Some(Stmt::Return(Some(value))) => errorish(value),
        Some(s) => noreturn_call(s),
        None => false,
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
                | Stmt::Block(body)
                | Stmt::MacroLoop { body, .. } => weight(body),
                Stmt::Switch { cases, .. } => cases.iter().map(|c| weight(&c.body)).sum(),
                Stmt::PreIf { branches, .. } => branches.iter().map(|b| weight(b)).sum(),
                Stmt::Labeled { stmt: Some(inner), .. } => weight(std::slice::from_ref(inner)),
                _ => 0,
            }
        })
        .sum()
}

/// Of two branches that both end, which one the guard holds: the lighter — the early exit the code is
/// written around; at equal weight the one that fails; and only then the positive test, as in the
/// other languages. `true` when the guard should hold `other` rather than `guard`, tested by `test`.
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

fn rewrite_tree(e: &mut Expr, locals: &HashSet<String>, changed: &mut bool) {
    for_each_child_mut(e, &mut |c| rewrite_tree(c, locals, changed));
    rewrite(e, locals, changed);
}

fn rewrite(e: &mut Expr, locals: &HashSet<String>, changed: &mut bool) {
    match e {
        Expr::Unary { op, e: inner } if op == "!" => {
            *changed |= rewrite_truth(inner);
            let reducible = match inner.as_ref() {
                Expr::Unary { op, .. } => op == "!",
                Expr::Binary { op, .. } => matches!(op.as_str(), "==" | "!=" | "&&" | "||" | "<" | "<=" | ">" | ">="),
                Expr::Bool(_) => true,
                _ => false,
            };
            if reducible {
                let operand = std::mem::replace(&mut **inner, placeholder());
                *e = negate(operand);
                *changed = true;
            }
        }
        Expr::Cond { c, .. } => *changed |= rewrite_truth(c),
        Expr::Binary { op, l, r } if (op == "&&" || op == "||") && matches!(r.as_ref(), Expr::Binary { op: ro, .. } if ro == op) => {
            let Expr::Binary { l: rl, r: rr, .. } = std::mem::replace(&mut **r, placeholder()) else { return };
            let left = std::mem::replace(&mut **l, placeholder());
            let op = op.clone();
            *e = binary(binary(left, &op, *rl), &op, *rr);
            *changed = true;
        }
        Expr::Binary { op, l, r } if op == "&&" || op == "||" => {
            *changed |= rewrite_truth(l);
            *changed |= rewrite_truth(r);
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
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Statement rules
// ---------------------------------------------------------------------------

/// Whether control never falls out of the bottom of `s`.
fn ends(s: &Stmt) -> bool {
    match s {
        Stmt::Return(_) | Stmt::Jump { .. } => true,
        Stmt::If { then, els: Some(els), .. } => {
            block_ends(then)
                && match els.as_ref() {
                    Else::Block(b) => block_ends(b),
                    Else::If(alt) => ends(alt),
                }
        }
        Stmt::Block(b) => block_ends(b),
        Stmt::Labeled { stmt: Some(inner), .. } => ends(inner),
        other => noreturn_call(other),
    }
}

fn block_ends(stmts: &[Stmt]) -> bool {
    stmts.last().is_some_and(ends)
}

/// Whether a block's remainder ends: its last statement does, or — in the body of a function that
/// returns nothing — it runs off the end, which is a return.
fn rest_ends(stmts: &[Stmt], ctx: Ctx) -> bool {
    block_ends(stmts) || (ctx.top && ctx.void && !stmts.is_empty())
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
        let rewritten = split_decl(stmts, i)
            || declare_with_value(stmts, i, scope)
            || unwrap_do_once(stmts, i)
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

/// `T a = 1, b = 2;` → `T a = 1;` then `T b = 2;` — each declarator is its own step.
fn split_decl(stmts: &mut Vec<Stmt>, i: usize) -> bool {
    let Stmt::Decl(specs) = &stmts[i] else { return false };
    if specs.len() < 2 {
        return false;
    }
    let Stmt::Decl(specs) = stmts.remove(i) else { return false };
    let singles: Vec<Stmt> = specs.into_iter().map(|s| Stmt::Decl(vec![s])).collect();
    stmts.splice(i..i, singles);
    true
}

/// `T x;` then `x = e;` → `T x = e;`: one binding, written two ways.
fn declare_with_value(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Stmt::Decl(specs) = &stmts[i] else { return false };
    let [Spec { name, init: None, .. }] = specs.as_slice() else { return false };
    let Stmt::Assign { op, lhs: Expr::Name(target), rhs } = &stmts[i + 1] else { return false };
    if op != "=" || target != name || names_read(rhs).contains(name) || scope.pinned.contains(name) {
        return false;
    }
    let Stmt::Assign { rhs, .. } = stmts.remove(i + 1) else { return false };
    if let Stmt::Decl(specs) = &mut stmts[i] {
        specs[0].init = Some(rhs);
    }
    true
}

/// Whether a body leaves its loop or `switch` by an unlabeled jump that binds to it: a `break`, or a
/// `continue` (`continue_too`), anywhere but inside a loop or `switch` of its own.
fn jumps_out(stmts: &[Stmt], continue_too: bool) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Jump { kind: Jump::Break, .. } => true,
        Stmt::Jump { kind: Jump::Continue, .. } => continue_too,
        Stmt::If { then, els, .. } => {
            jumps_out(then, continue_too)
                || match els.as_deref() {
                    Some(Else::Block(b)) => jumps_out(b, continue_too),
                    Some(Else::If(alt)) => jumps_out(std::slice::from_ref(alt), continue_too),
                    None => false,
                }
        }
        Stmt::Block(b) => jumps_out(b, continue_too),
        Stmt::PreIf { branches, .. } => branches.iter().any(|b| jumps_out(b, continue_too)),
        Stmt::Labeled { stmt: Some(inner), .. } => jumps_out(std::slice::from_ref(inner), continue_too),
        // A jump inside these binds to them.
        _ => false,
    })
}

/// `do { B } while (0)` → `B`: the macro writer's block, run once.
fn unwrap_do_once(stmts: &mut Vec<Stmt>, i: usize) -> bool {
    let Stmt::DoWhile { body, cond } = &stmts[i] else { return false };
    if !is_zero(cond) || jumps_out(body, true) {
        return false;
    }
    let Stmt::DoWhile { body, .. } = stmts.remove(i) else { return false };
    stmts.splice(i..i, body);
    true
}

/// `while (1)` → `for (;;)`: one loop with no test, written two ways.
fn forever(stmts: &mut [Stmt], i: usize) -> bool {
    let Stmt::While { cond, .. } = &stmts[i] else { return false };
    let always = matches!(cond, Expr::Bool(true)) || matches!(cond, Expr::Lit("Int", t) if t == "1");
    if !always {
        return false;
    }
    let Stmt::While { body, .. } = std::mem::replace(&mut stmts[i], Stmt::Empty) else { return false };
    stmts[i] = Stmt::For { init: None, cond: None, update: None, body };
    true
}

/// One rung of a ladder: the case labels that share a body (`None` for `default`), and the body.
type Rung = (Vec<Option<Expr>>, Vec<Stmt>);

/// The case labels that share one body (`case 1: case 2: S`), each body without the `break` that ends
/// it — or `None` when some case falls into the next, `break`s from deeper inside, or does neither.
fn rungs(cases: &[Case]) -> Option<Vec<Rung>> {
    let mut out = Vec::new();
    let mut labels: Vec<Option<Expr>> = Vec::new();
    for case in cases {
        labels.push(case.value.clone());
        if case.body.is_empty() {
            continue; // shares the next case's body
        }
        let mut body = case.body.clone();
        if matches!(body.last(), Some(Stmt::Jump { kind: Jump::Break, .. })) {
            body.pop();
        } else if !block_ends(&body) {
            return None; // falls into the next case
        }
        if jumps_out(&body, false) {
            return None;
        }
        out.push((std::mem::take(&mut labels), body));
    }
    labels.is_empty().then_some(out)
}

/// `switch (v) { case a: case b: A; break; default: D; }` → `if (v == a || v == b) { A } else { D }`.
fn switch_to_ladder(stmts: &mut [Stmt], i: usize) -> bool {
    let Stmt::Switch { value, cases } = &stmts[i] else { return false };
    let Some(rungs) = rungs(cases) else { return false };
    let (defaults, tests): (Vec<_>, Vec<_>) = rungs.into_iter().partition(|(labels, _)| labels.iter().any(Option::is_none));
    if tests.is_empty() || defaults.len() > 1 {
        return false;
    }
    let test = |labels: &[Option<Expr>]| -> Option<Expr> {
        labels.iter().flatten().cloned().map(|v| binary(value.clone(), "==", v)).reduce(|a, b| binary(a, "||", b))
    };
    // Built from the last rung up, each rung the `else` of the one before — the nested shape every rule reads.
    let mut otherwise: Option<Vec<Stmt>> = defaults.into_iter().next().map(|(_, body)| body);
    for (labels, body) in tests.into_iter().rev() {
        let Some(cond) = test(&labels) else { return false };
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
    let Stmt::If { cond, then, els: Some(els) } = &mut stmts[i] else { return false };
    let Else::Block(alt) = els.as_ref() else { return false };
    if block_ends(then) || block_ends(alt) || !leans_negative(cond, &scope.locals) {
        return false;
    }
    let c = std::mem::replace(cond, placeholder());
    *cond = negate(c);
    swap_branches(then, els);
    true
}

/// `if (c) { <ends> } else { B }` → `if (c) { <ends> }` then `B`; when only the `else` ends, the test
/// is negated and the branches change places first; when both end, the guard orientation decides.
fn guard_form(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope) -> bool {
    let Stmt::If { cond, then, els: Some(els) } = &mut stmts[i] else { return false };
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
/// `<ends>` ≡ `if (!c) { <ends> }` then `<ends>`. The representative tests the failure.
fn orient_guard(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope, ctx: Ctx) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Stmt::If { cond, then, els: None } = &stmts[i] else { return false };
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
    let Stmt::Decl(specs) = &stmts[i] else { return false };
    let [Spec { name: t, init: Some(value), .. }] = specs.as_slice() else { return false };
    if scope.stores(t) != 1
        || scope.reads(t) != 1
        || scope.pinned.contains(t)
        || scope.nested_reads.contains(t)
        || names_read(value).contains(t)
        || matches!(value, Expr::InitList(_))
    {
        return false;
    }
    // A loop re-evaluates its test; a temporary read there is read every time.
    if !matches!(&stmts[i + 1], Stmt::Decl(_) | Stmt::Assign { .. } | Stmt::IncDec { .. } | Stmt::Expr(_) | Stmt::Return(_) | Stmt::If { .. } | Stmt::Switch { .. }) {
        return false;
    }
    let t = t.clone();
    let mut direct = 0;
    direct_exprs(&mut stmts[i + 1], &mut |e| direct += reads_of(e, &t));
    if direct != 1 {
        return false;
    }
    let Stmt::Decl(mut specs) = stmts.remove(i) else { return false };
    let Some(value) = specs.pop().and_then(|s| s.init) else { return false };
    let mut value = Some(value);
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

/// `x += e` → `x = x + e`; `x++` → `x = x + 1`.
fn expand_compound(stmts: &mut [Stmt], i: usize) -> bool {
    match &mut stmts[i] {
        Stmt::Assign { op, lhs, rhs } if op != "=" => {
            let base = op.trim_end_matches('=').to_owned();
            let target = lhs.clone();
            let value = std::mem::replace(rhs, placeholder());
            stmts[i] = Stmt::Assign { op: "=".to_owned(), lhs: target.clone(), rhs: binary(target, &base, value) };
            true
        }
        Stmt::IncDec { inc, target } => {
            let base = if *inc { "+" } else { "-" };
            let target = std::mem::replace(target, placeholder());
            let one = Expr::Lit("Int", "1".to_owned());
            stmts[i] = Stmt::Assign { op: "=".to_owned(), lhs: target.clone(), rhs: binary(target, base, one) };
            true
        }
        _ => false,
    }
}

/// A bare `return` at the end of a function that returns nothing is falling off its end.
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

    fn analysis(body: &str) -> Analysis {
        let f: Arc<str> = Arc::from("t.c");
        scan_source(body, &f, &ScanOpts::default()).into_iter().find_map(|d| d.analysis).expect("a function")
    }

    fn xname(body: &str) -> String {
        analysis(body).xname_canonical
    }

    fn lines(body: &str) -> Vec<String> {
        analysis(body).type3_lines
    }

    #[test]
    fn a_guard_and_an_else_are_one_form() {
        let guarded = xname("int f(int x) {\n\tif (x == 3) {\n\t\treturn 1;\n\t}\n\tg();\n\treturn 2;\n}");
        let branched = xname("int f(int x) {\n\tif (x == 3) {\n\t\treturn 1;\n\t} else {\n\t\tg();\n\t\treturn 2;\n\t}\n}");
        let negated = xname("int f(int x) {\n\tif (x != 3) {\n\t\tg();\n\t\treturn 2;\n\t} else\n\t\treturn 1;\n}");
        assert_eq!(guarded, branched);
        assert_eq!(guarded, negated);
    }

    #[test]
    fn the_c_guard_tests_the_failure() {
        // The idiom is the representative: `if (ret < 0) return ret;` keeps its shape, and the same
        // decision written inside out meets it.
        let idiom = "int f(void) {\n\tint ret = load();\n\tif (ret < 0)\n\t\treturn ret;\n\tuse(ret);\n\treturn 0;\n}";
        let inside_out = "int f(void) {\n\tint ret = load();\n\tif (ret >= 0) {\n\t\tuse(ret);\n\t\treturn 0;\n\t}\n\treturn ret;\n}";
        assert_eq!(xname(idiom), xname(inside_out));
        let goto = "int f(void *p) {\n\tif (!p)\n\t\tgoto fail;\n\tuse(p);\n\treturn 0;\nfail:\n\treturn -1;\n}";
        assert!(xname(goto).contains("Goto('fail')"), "{}", xname(goto));
    }

    #[test]
    fn a_truth_test_is_its_operand() {
        let spelled = xname("int f(char *p) {\n\tif (p != NULL && *p != '\\0')\n\t\treturn g(p);\n\treturn 0;\n}");
        let bare = xname("int f(char *p) {\n\tif (p && *p)\n\t\treturn g(p);\n\treturn 0;\n}");
        assert_eq!(spelled, bare);
        assert_eq!(
            xname("int f(int *p) {\n\tg();\n\treturn !(p == NULL) ? 1 : 2;\n}"),
            xname("int f(int *p) {\n\tg();\n\treturn p ? 1 : 2;\n}")
        );
    }

    #[test]
    fn negations_and_comparisons_have_one_spelling() {
        assert_eq!(xname("int f(int a, int b) {\n\tg();\n\treturn !(a && b);\n}"), xname("int f(int a, int b) {\n\tg();\n\treturn !a || !b;\n}"));
        assert_eq!(xname("int f(int x) {\n\tg();\n\treturn x > LIMIT;\n}"), xname("int f(int x) {\n\tg();\n\treturn LIMIT < x;\n}"));
    }

    #[test]
    fn a_switch_is_the_ladder_it_abbreviates() {
        let switched = xname("int f(int x) {\n\tswitch (x) {\n\tcase 1:\n\t\treturn a();\n\tcase 2:\n\tcase 3:\n\t\tb();\n\t\tbreak;\n\tdefault:\n\t\treturn c();\n\t}\n\treturn 0;\n}");
        let ladder = xname("int f(int x) {\n\tif (x == 1)\n\t\treturn a();\n\telse if (x == 2 || x == 3)\n\t\tb();\n\telse\n\t\treturn c();\n\treturn 0;\n}");
        assert_eq!(switched, ladder);
        // A case that falls into the next stays a switch.
        let falls = xname("int f(int x) {\n\tswitch (x) {\n\tcase 1:\n\t\ta();\n\tcase 2:\n\t\tb();\n\t\tbreak;\n\t}\n\treturn x;\n}");
        assert!(falls.contains("Switch("), "{falls}");
    }

    #[test]
    fn a_temporary_is_inlined_for_the_exact_pass_and_kept_for_type3() {
        let named = "int f(int x) {\n\tint t = g(x);\n\treturn h(t);\n}";
        let inlined = "int f(int x) {\n\treturn h(g(x));\n}";
        assert_eq!(xname(named), xname(inlined));
        assert_ne!(lines(named), lines(inlined), "Type-3 keeps the statement");
        let taken = "int f(void) {\n\tint t = g();\n\treturn h(&t);\n}";
        assert!(xname(taken).contains("Let("), "{}", xname(taken));
    }

    #[test]
    fn small_spellings_meet() {
        let plus = "int f(int x) {\n\tx = x + 1;\n\treturn g(x);\n}";
        assert_eq!(xname("int f(int x) {\n\tx += 1;\n\treturn g(x);\n}"), xname(plus));
        assert_eq!(xname("int f(int x) {\n\tx++;\n\treturn g(x);\n}"), xname(plus));
        assert_eq!(xname("int f(int x) {\n\t++x;\n\treturn g(x);\n}"), xname(plus));
        assert_eq!(
            xname("int f(void) {\n\tint a = g(), b = h();\n\treturn k(a, b, a, b);\n}"),
            xname("int f(void) {\n\tint a;\n\ta = g();\n\tint b = h();\n\treturn k(a, b, a, b);\n}")
        );
        assert_eq!(xname("void f(int x) {\n\twhile (1) {\n\t\tg(x);\n\t}\n}"), xname("void f(int x) {\n\tfor (;;)\n\t\tg(x);\n}"));
        assert_eq!(xname("void f(int x) {\n\tdo {\n\t\tg(x);\n\t\th(x);\n\t} while (0);\n}"), xname("void f(int x) {\n\tg(x);\n\th(x);\n}"));
    }

    #[test]
    fn a_void_function_s_last_return_is_its_end() {
        assert_eq!(xname("void f(int x) {\n\tg(x);\n\th(x);\n\treturn;\n}"), xname("void f(int x) {\n\tg(x);\n\th(x);\n}"));
    }

    #[test]
    fn the_stream_keeps_the_body_as_written() {
        // Converge points the reader at code: the statement stream is never the normal form.
        let src = "int f(int x) {\n\tswitch (x) {\n\tcase 1:\n\t\treturn a();\n\t}\n\treturn 0;\n}\n";
        let f: Arc<str> = Arc::from("t.c");
        let d = scan_source(src, &f, &ScanOpts::default()).into_iter().next().expect("f");
        assert!(d.facets.statements.iter().any(|s| s.line.starts_with("Switch(")), "{:?}", d.facets.statements);
    }
}
