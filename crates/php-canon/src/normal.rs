//! **Semantic normal form** of a PHP body, taken before its name-agnostic canonicals.
//!
//! The exact pass equates two definitions when their alpha-renamed canonicals are equal, and the
//! Type-3 pass when their statements mostly are. Both read the code as *written*, and the same
//! procedure is written many ways: a guard beside an `if … else`, `null === $x` beside `$x === null`,
//! `isset($x)` beside `$x !== null`, a temporary named and used once beside the expression inlined, a
//! `switch` beside the `if` ladder it abbreviates, `$i++` beside `$i += 1` beside `$i = $i + 1`. Each
//! of those is a Type-4 clone to a reader and an unrelated body to a pass that compares text.
//!
//! This module rewrites a body into one representative of its equivalence class under the rules the
//! other frontends' normal forms apply, read in PHP's grammar — oriented, applied to a fixed point, one
//! confluent direction per rule. (What PHP spells two ways in its *grammar* — `elseif` and `else if`,
//! `array()` and `[]`, `and` and `&&` — is one shape already: [`crate::ir`] lowers it so.)
//!
//! ## Rules
//!
//! Expressions, innermost first, to a fixed point:
//! - **Negation normal form**: `!!x` → `x`; `!(a && b)` → `!a || !b` (and dually); `!($a === $b)` →
//!   `$a !== $b`; `!($a < $b)` → `$a >= $b`; `a || (b || c)` → `a || b || c`. Parentheses are dropped:
//!   the tree already says what they grouped.
//! - **A null test has one spelling**: `isset($x)` is `$x !== null` and `is_null($x)` is
//!   `$x === null` — the interpreter defines the first two alike for a variable, and the second pair
//!   identically.
//! - **Comparison orientation**: `==`, `===`, `!=`, `!==` and the mirror pairs put their operands in a
//!   fixed order, by their shape with the definition's own names blanked — so a renamed copy orients
//!   the same way, and a Yoda condition (`'open' === $status`, a common house style) meets the same
//!   test written the other way round.
//!
//! Statements, in every block:
//! - **A `switch` is the ladder it abbreviates**: `case a: case b: S; break;` → `if ($v == a || $v ==
//!   b) { S }` — `switch` compares loosely, so the ladder does too — `default` → the last `else`, when
//!   every case either ends in its own `break` or leaves the function, and none falls into the next or
//!   `break`s from deeper inside: what a ladder could not say.
//! - **Branch orientation**: an `if … else` that neither branch leaves through, whose test is negated
//!   (or mostly negated), is flipped.
//! - **Guard form**: `if (c) { …; return x; } else { B }` → `if (c) { …; return x; }` then `B`;
//!   mirrored when the `else` is the branch that ends.
//! - **Guard orientation**: when the rest of the block ends too, `if (c) { E }` then `R` is the same
//!   decision as `if (!c) { R }` then `E`. The guard holds the *lighter* branch; at equal weight, the
//!   branch that *fails* — a `throw`, a `return false` / `null`, an error object, an `exit` — and only
//!   then the positive test (the C frontend's rule, for its reason: it is how the code is written).
//! - **Single-use temporary** (the exact form only): `$t = e;` then a statement that reads `$t` once and
//!   nothing else does → that statement with `e` in place of `$t`.
//! - `$x += e`, `$x .= e`, `$x ??= e` and `$x++` → `$x = $x + e` and the like; `while (true)` →
//!   `for (;;)`.
//! - A final bare `return;` is falling off the end of the function: both hand back `null`.
//!
//! ## What the rules assume
//!
//! The normalized tree is compared, never run: a rule need only preserve meaning as a reader judges it.
//! A ladder evaluates the tested value once per test. A variable taken by reference (`&$x`, a `global`
//! or `static` one), or in a body that reads its variables by name (`compact`, `extract`, `$$name`), is
//! pinned: inlining it would change what else sees it.

use std::collections::HashSet;

use dup_defs_core::normal::{complement, Scope, MAX_ROUNDS};

use crate::canon::{each_stmt_part, for_each_child, Child, Part};
use crate::ir::{Arg, Case, Elem, Else, Expr, Func, Jump, Member, Stmt};

/// The normal form of a function. `inline` admits the single-use-temporary rule: the exact pass wants
/// it, the Type-3 pass does not — its unit is the statement, and folding `$t = f($x)` into the line
/// that reads `$t` turns two short units into one long one (the Python frontend's measurement).
#[must_use]
pub(crate) fn normal_form(f: &Func, inline: bool) -> Func {
    let mut out = f.clone();
    strip_parens_stmts(&mut out.body);
    nest_else_ifs(&mut out.body);
    for _ in 0..MAX_ROUNDS {
        let scope = scope_of(&out);
        let mut changed = block(&mut out.body, &scope, Ctx { top: true, inline });
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
    inline: bool,
}

// ---------------------------------------------------------------------------
// Traversal
// ---------------------------------------------------------------------------

/// Every statement list directly inside `s` — its branches and bodies, and the body of every closure
/// its expressions hold — not the lists nested in those.
fn nested_lists(s: &mut Stmt, f: &mut dyn FnMut(&mut Vec<Stmt>)) {
    match s {
        Stmt::If { cond, then, els } => {
            closures_of(cond, f);
            f(then);
            match els.as_deref_mut() {
                Some(Else::Block(b)) => f(b),
                Some(Else::If(alt)) => nested_lists(alt, f),
                None => {}
            }
        }
        Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
            closures_of(cond, f);
            f(body);
        }
        Stmt::For { init, cond, update, body } => {
            for e in init.iter_mut().chain(cond.iter_mut()).chain(update.iter_mut()) {
                closures_of(e, f);
            }
            f(body);
        }
        Stmt::Foreach { subject, body, .. } => {
            closures_of(subject, f);
            f(body);
        }
        Stmt::Switch { value, cases } => {
            closures_of(value, f);
            for case in cases {
                f(&mut case.body);
            }
        }
        Stmt::Try { body, catches, finally } => {
            f(body);
            for c in catches {
                f(&mut c.body);
            }
            if let Some(fin) = finally {
                f(fin);
            }
        }
        Stmt::Block(b) => f(b),
        other => direct_exprs(other, &mut |e| closures_of(e, f)),
    }
}

/// The bodies of the closures `e` holds, outermost first.
fn closures_of(e: &mut Expr, f: &mut dyn FnMut(&mut Vec<Stmt>)) {
    if let Expr::Closure { body, .. } = e {
        f(body);
        return;
    }
    for_each_child_mut(e, &mut |c| closures_of(c, f));
}

/// The expressions a statement evaluates itself, once, before any block inside it runs — the slots a
/// temporary may be inlined into, and the roots the expression rules walk. A loop's condition is not
/// one: it is evaluated again on every pass.
fn direct_exprs(s: &mut Stmt, f: &mut dyn FnMut(&mut Expr)) {
    match s {
        Stmt::Assign { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        Stmt::IncDec { target: e, .. } | Stmt::Expr(e) | Stmt::Throw(e) | Stmt::Return(Some(e)) => f(e),
        Stmt::Switch { value, .. } => f(value),
        Stmt::If { cond, .. } => f(cond),
        Stmt::Echo(list) | Stmt::Unset(list) => list.iter_mut().for_each(f),
        Stmt::Foreach { subject, .. } => f(subject),
        Stmt::Static(vars) => vars.iter_mut().filter_map(|(_, v)| v.as_mut()).for_each(f),
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
            init.iter_mut().chain(cond.iter_mut()).chain(update.iter_mut()).for_each(&mut *f);
            all_exprs(body, f);
        }
        Stmt::Foreach { subject, key, value, body } => {
            f(subject);
            if let Some(k) = key {
                f(k);
            }
            f(value);
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
        Stmt::Try { body, catches, finally } => {
            all_exprs(body, f);
            for c in catches {
                all_exprs(&mut c.body, f);
            }
            if let Some(fin) = finally {
                all_exprs(fin, f);
            }
        }
        Stmt::Block(b) => all_exprs(b, f),
        other => direct_exprs(other, f),
    }
}

fn args_mut(args: &mut [Arg], f: &mut dyn FnMut(&mut Expr)) {
    for a in args {
        f(&mut a.value);
    }
}

/// [`for_each_child`], mutably: every direct sub-expression, and the expressions of a closure's body.
fn for_each_child_mut(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    match e {
        Expr::Binary { l, r, .. } | Expr::Assign { l, r, .. } | Expr::Comma(l, r) => {
            f(l);
            f(r);
        }
        Expr::DynVar(e)
        | Expr::Unary { e, .. }
        | Expr::Update { e, .. }
        | Expr::Paren(e)
        | Expr::Cast { e, .. }
        | Expr::Silence(e)
        | Expr::Clone(e)
        | Expr::Print(e)
        | Expr::Include { e, .. }
        | Expr::YieldFrom(e)
        | Expr::Throw(e)
        | Expr::Spread(e)
        | Expr::StaticProp { scope: e, .. }
        | Expr::ClassConst { scope: e, .. } => f(e),
        Expr::Cond { c, t, f: alt } => {
            f(c);
            if let Some(t) = t {
                f(t);
            }
            f(alt);
        }
        Expr::Call { f: callee, args } => {
            f(callee);
            args_mut(args, f);
        }
        Expr::Method { recv, name, args, .. } => {
            f(recv);
            if let Member::Dyn(d) = name {
                f(d);
            }
            args_mut(args, f);
        }
        Expr::StaticCall { scope, name, args } => {
            f(scope);
            if let Member::Dyn(d) = name {
                f(d);
            }
            args_mut(args, f);
        }
        Expr::Prop { e, name, .. } => {
            f(e);
            if let Member::Dyn(d) = name {
                f(d);
            }
        }
        Expr::Index { e, i } => {
            f(e);
            if let Some(i) = i {
                f(i);
            }
        }
        Expr::Array(elems) | Expr::List(elems) => {
            for el in elems {
                match el {
                    Elem::Plain(v) => f(v),
                    Elem::Pair(k, v) => {
                        f(k);
                        f(v);
                    }
                }
            }
        }
        Expr::New { class, args } => {
            f(class);
            args_mut(args, f);
        }
        Expr::Closure { body, .. } => all_exprs(body, f),
        Expr::Arrow { body, .. } => f(body),
        Expr::Match { subject, arms } => {
            f(subject);
            for arm in arms {
                arm.conds.iter_mut().flatten().for_each(&mut *f);
                f(&mut arm.value);
            }
        }
        Expr::Yield { key, value } => {
            if let Some(k) = key {
                f(k);
            }
            if let Some(v) = value {
                f(v);
            }
        }
        Expr::Interp(parts) | Expr::Shell(parts) => parts.iter_mut().for_each(f),
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

/// The functions that read or write a scope's variables by name.
const SCOPE_READERS: &[&str] = &["compact", "extract", "get_defined_vars", "parse_str", "eval"];

fn scope_of(f: &Func) -> Scope {
    let mut scope = Scope::default();
    let mut walk = ScopeWalk { scope: &mut scope, depth: 0, by_name: false };
    for p in &f.params {
        walk.scope.bind(&p.name);
        if p.by_ref {
            walk.scope.pinned.insert(p.name.clone());
        }
    }
    walk.stmts(&f.body);
    if walk.by_name {
        // Every variable may be read through its name: none is a temporary.
        let all: Vec<String> = scope.locals.iter().cloned().chain(scope.reads.keys().cloned()).collect();
        scope.pinned.extend(all);
    }
    scope
}

struct ScopeWalk<'s> {
    scope: &'s mut Scope,
    depth: usize,
    /// The body reads or writes its variables by name somewhere.
    by_name: bool,
}

impl ScopeWalk<'_> {
    /// An assignment's target: a variable is a store, a destructuring list stores each of its targets,
    /// anything else (`$a->b`, `$a[$i]`) is a read of what it reaches into.
    fn store_target(&mut self, e: &Expr) {
        match e {
            Expr::Var(n) => self.scope.bind(n),
            Expr::List(elems) => {
                for el in elems {
                    match el {
                        Elem::Plain(t) => self.store_target(t),
                        Elem::Pair(k, t) => {
                            self.expr(k);
                            self.store_target(t);
                        }
                    }
                }
            }
            Expr::Unary { op, e: inner } if op == "&" => {
                if let Expr::Var(n) = inner.as_ref() {
                    self.scope.pinned.insert(n.clone());
                }
                self.store_target(inner);
            }
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
            Stmt::Assign { op, lhs, rhs } => {
                self.expr(rhs);
                self.store_target(lhs);
                if op == "=&" {
                    for side in [lhs, rhs] {
                        if let Expr::Var(n) = side {
                            self.scope.pinned.insert(n.clone());
                        }
                    }
                }
                // A compound assignment reads what it writes.
                if op != "=" && op != "=&" {
                    self.expr(lhs);
                }
            }
            Stmt::IncDec { target, .. } => {
                self.store_target(target);
                self.expr(target);
            }
            Stmt::Foreach { subject, key, value, body } => {
                self.expr(subject);
                if let Some(k) = key {
                    self.store_target(k);
                }
                self.store_target(value);
                // A loop variable is bound again on every pass: never a temporary.
                for target in key.iter().chain([value]) {
                    if let Expr::Var(n) = target {
                        self.scope.store(n);
                    }
                }
                self.stmts(body);
            }
            Stmt::Global(names) => {
                for n in names {
                    self.scope.pinned.insert(n.clone());
                }
            }
            Stmt::Static(vars) => {
                for (n, v) in vars {
                    if let Some(v) = v {
                        self.expr(v);
                    }
                    self.scope.bind(n);
                    self.scope.pinned.insert(n.clone());
                }
            }
            Stmt::Try { catches, .. } => {
                for c in catches {
                    if let Some(v) = &c.var {
                        self.scope.bind(v);
                    }
                }
                self.parts(s);
            }
            _ => self.parts(s),
        }
    }

    fn parts(&mut self, s: &Stmt) {
        each_stmt_part(s, &mut |part| match part {
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Var(n) => self.scope.read(n, self.depth > 0),
            Expr::DynVar(inner) => {
                self.by_name = true;
                self.expr(inner);
            }
            Expr::Call { f, .. } if matches!(f.as_ref(), Expr::Name(n) if SCOPE_READERS.iter().any(|r| r.eq_ignore_ascii_case(n))) => {
                self.by_name = true;
                for_each_child(e, &mut |c| match c {
                    Child::Expr(x) => self.expr(x),
                    Child::Body(b) => self.stmts(b),
                });
            }
            // `&$x` — what the reference points at is `$x` itself; nothing may be assumed about it.
            Expr::Unary { op, e: inner } if op == "&" => {
                if let Expr::Var(n) = inner.as_ref() {
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
            Expr::Closure { uses, body, .. } => {
                for (n, by_ref) in uses {
                    self.scope.read(n, true);
                    if *by_ref {
                        self.scope.pinned.insert(n.clone());
                    }
                }
                self.depth += 1;
                self.stmts(body);
                self.depth -= 1;
            }
            Expr::Arrow { body, .. } => {
                self.depth += 1;
                self.expr(body);
                self.depth -= 1;
            }
            other => for_each_child(other, &mut |child| match child {
                Child::Expr(c) => self.expr(c),
                Child::Body(b) => self.stmts(b),
            }),
        }
    }
}

/// Variables read anywhere in `e`.
fn vars_read(e: &Expr) -> HashSet<String> {
    fn walk(e: &Expr, out: &mut HashSet<String>) {
        if let Expr::Var(n) = e {
            out.insert(n.clone());
        }
        if let Expr::Closure { uses, .. } = e {
            out.extend(uses.iter().map(|(n, _)| n.clone()));
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
    let mut n = usize::from(matches!(e, Expr::Var(x) if x == name));
    if let Expr::Closure { uses, .. } = e {
        n += uses.iter().filter(|(u, _)| u == name).count();
    }
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

/// A shape key with the definition's own variables blanked — what the orientation rules order by. Two
/// renamed copies of one expression have one key; a key tie keeps the written order.
fn key(e: &Expr, locals: &HashSet<String>) -> String {
    fn blank(e: &mut Expr, locals: &HashSet<String>) {
        if let Expr::Var(n) = e {
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
        Expr::Binary { op, .. } => op == "!=" || op == "!==",
        _ => false,
    }
}

fn not(e: Expr) -> Expr {
    Expr::Unary { op: "!".to_owned(), e: Box::new(e) }
}

fn binary(l: Expr, op: &str, r: Expr) -> Expr {
    Expr::Binary { op: op.to_owned(), l: Box::new(l), r: Box::new(r) }
}

/// The operator an equality is with its sense inverted.
fn inverted(op: &str) -> Option<&'static str> {
    Some(match op {
        "==" => "!=",
        "!=" => "==",
        "===" => "!==",
        "!==" => "===",
        _ => return None,
    })
}

/// The operator a comparison becomes with its operands swapped. The core's table, with PHP's identity
/// comparisons.
fn mirror(op: &str) -> Option<&'static str> {
    match op {
        "===" => Some("==="),
        "!==" => Some("!=="),
        other => dup_defs_core::normal::mirror(other),
    }
}

/// The exact inverse of a test, in negation normal form.
fn negate(e: Expr) -> Expr {
    match e {
        Expr::Unary { op, e } if op == "!" => *e,
        Expr::Binary { op, l, r } if inverted(&op).is_some() => Expr::Binary { op: inverted(&op).unwrap_or_default().to_owned(), l, r },
        // An ordering's complement is an ordering: `!($a < $b)` is `$a >= $b` to a reader, whatever PHP's
        // comparison of mixed types makes of it — the form is compared, never run.
        Expr::Binary { op, l, r } if complement(&op).is_some() => Expr::Binary { op: complement(&op).unwrap_or_default().to_owned(), l, r },
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

/// The name of a function a call calls, when it is a bare one.
fn callee(e: &Expr) -> Option<&str> {
    match e {
        Expr::Call { f, .. } => match f.as_ref() {
            Expr::Name(n) => Some(n.trim_start_matches('\\')),
            _ => None,
        },
        _ => None,
    }
}

/// `isset($x)` → `$x !== null`, `is_null($x)` → `$x === null`: one null test, one spelling. `isset`
/// of several values is each of them set.
fn null_test(e: &mut Expr) -> bool {
    let Some(name) = callee(e).map(str::to_ascii_lowercase) else { return false };
    let Expr::Call { args, .. } = e else { return false };
    if args.is_empty() || args.iter().any(|a| a.name.is_some() || matches!(a.value, Expr::Spread(_))) {
        return false;
    }
    let op = match name.as_str() {
        "isset" => "!==",
        "is_null" if args.len() == 1 => "===",
        _ => return false,
    };
    let tests: Vec<Expr> = std::mem::take(args).into_iter().map(|a| binary(a.value, op, Expr::Null)).collect();
    if let Some(all) = tests.into_iter().reduce(|a, b| binary(a, "&&", b)) {
        *e = all;
    }
    true
}

/// Apply the expression rules to every expression of the body, innermost first.
fn expressions(stmts: &mut [Stmt], scope: &Scope) -> bool {
    let mut changed = false;
    all_exprs(stmts, &mut |root| rewrite_tree(root, &scope.locals, &mut changed));
    changed
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
                    matches!(op.as_str(), "==" | "!=" | "===" | "!==" | "&&" | "||" | "<" | "<=" | ">" | ">=")
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
        Expr::Binary { op, l, r } if (op == "&&" || op == "||") && matches!(r.as_ref(), Expr::Binary { op: ro, .. } if ro == op) => {
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
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Statement rules
// ---------------------------------------------------------------------------

/// The functions that do not return.
fn noreturn(callee: &str) -> bool {
    matches!(callee.to_ascii_lowercase().as_str(), "exit" | "wp_die" | "abort" | "dd" | "trigger_error")
}

fn noreturn_call(s: &Stmt) -> bool {
    matches!(s, Stmt::Expr(e) if callee(e).is_some_and(noreturn))
}

/// Whether control never falls out of the bottom of `s`.
fn ends(s: &Stmt) -> bool {
    match s {
        Stmt::Return(_) | Stmt::Throw(_) | Stmt::Jump { .. } => true,
        Stmt::If { then, els: Some(els), .. } => {
            block_ends(then)
                && match els.as_ref() {
                    Else::Block(b) => block_ends(b),
                    Else::If(alt) => ends(alt),
                }
        }
        Stmt::Block(b) => block_ends(b),
        other => noreturn_call(other),
    }
}

fn block_ends(stmts: &[Stmt]) -> bool {
    stmts.last().is_some_and(ends)
}

/// Whether a block's remainder ends: its last statement does, or — in the function's own body — it runs
/// off the end, which is a `return;`.
fn rest_ends(stmts: &[Stmt], ctx: Ctx) -> bool {
    block_ends(stmts) || (ctx.top && !stmts.is_empty())
}

/// Whether a value is, by PHP's conventions, a failure: `false`, `null`, a negative number, an error
/// object (`new WP_Error(…)`, `new \RuntimeException(…)`), or a name that says so.
fn errorish(e: &Expr) -> bool {
    let says_error = |name: &str| {
        let lower = name.to_ascii_lowercase();
        lower.contains("err") || lower.contains("fail") || lower.contains("exception")
    };
    match e {
        Expr::Bool(false) | Expr::Null => true,
        Expr::Unary { op, .. } => op == "-",
        Expr::New { class, .. } => matches!(class.as_ref(), Expr::Name(n) if says_error(n)),
        Expr::Name(n) | Expr::Var(n) => says_error(n),
        Expr::Call { f, .. } => matches!(f.as_ref(), Expr::Name(n) if says_error(n)),
        Expr::Paren(inner) => errorish(inner),
        _ => false,
    }
}

/// Whether a branch is the failure path: it throws, returns a failure, or does not return.
fn fails(stmts: &[Stmt]) -> bool {
    match stmts.last() {
        Some(Stmt::Throw(_)) => true,
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
                | Stmt::Foreach { body, .. }
                | Stmt::Block(body) => weight(body),
                Stmt::Switch { cases, .. } => cases.iter().map(|c| weight(&c.body)).sum(),
                Stmt::Try { body, catches, finally } => {
                    weight(body) + catches.iter().map(|c| weight(&c.body)).sum::<usize>() + finally.as_deref().map_or(0, weight)
                }
                _ => 0,
            }
        })
        .sum()
}

/// Of two branches that both end, which one the guard holds: the lighter — the early exit the code is
/// written around; at equal weight the one that fails; and only then the positive test. `true` when the
/// guard should hold `other` rather than `guard`, tested by `test`.
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
        let rewritten = forever(stmts, i)
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

/// Whether a body leaves its loop or `switch` by a jump that binds to it: a `break`, or a `continue`
/// (`continue_too`), anywhere but inside a loop or `switch` of its own — or one that leaves more than
/// one level (`break 2`), which binds to something outside it.
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
        Stmt::Try { body, catches, finally } => {
            jumps_out(body, continue_too)
                || catches.iter().any(|c| jumps_out(&c.body, continue_too))
                || finally.as_deref().is_some_and(|f| jumps_out(f, continue_too))
        }
        // A jump inside these binds to them — unless it leaves more than one level.
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } | Stmt::Foreach { body, .. } => {
            leaves_levels(body)
        }
        Stmt::Switch { cases, .. } => cases.iter().any(|c| leaves_levels(&c.body)),
        _ => false,
    })
}

/// Whether a body holds a `break n` / `continue n` of more than one level.
fn leaves_levels(stmts: &[Stmt]) -> bool {
    let mut found = false;
    for s in stmts {
        if let Stmt::Jump { kind: Jump::Break | Jump::Continue, arg: Some(levels) } = s {
            found |= levels.trim() != "1";
        }
        let mut copy = s.clone();
        nested_lists(&mut copy, &mut |list| found |= leaves_levels(list));
    }
    found
}

/// `while (true)` → `for (;;)`: one loop with no test, written two ways.
fn forever(stmts: &mut [Stmt], i: usize) -> bool {
    let Stmt::While { cond, .. } = &stmts[i] else { return false };
    let always = matches!(cond, Expr::Bool(true)) || matches!(cond, Expr::Lit("Int", t) if t == "1");
    if !always {
        return false;
    }
    let Stmt::While { body, .. } = std::mem::replace(&mut stmts[i], Stmt::Empty) else { return false };
    stmts[i] = Stmt::For { init: Vec::new(), cond: Vec::new(), update: Vec::new(), body };
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
        if matches!(body.last(), Some(Stmt::Jump { kind: Jump::Break, arg: None })) {
            body.pop();
        } else if !block_ends(&body) {
            return None; // falls into the next case
        }
        if jumps_out(&body, true) {
            return None;
        }
        out.push((std::mem::take(&mut labels), body));
    }
    labels.is_empty().then_some(out)
}

/// `switch ($v) { case a: case b: A; break; default: D; }` → `if ($v == a || $v == b) { A } else { D }`.
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

/// `if (c) { <ends> } else { B }` → `if (c) { <ends> }` then `B`; when only the `else` ends, the test is
/// negated and the branches change places first; when both end, the guard orientation decides.
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
/// `<ends>` ≡ `if (!c) { <ends> }` then `<ends>`. The representative guards the lighter branch.
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

/// `$t = e;` then `S($t)`, with `$t` assigned once and read once, there → `S(e)`.
fn inline_temporary(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Stmt::Assign { op, lhs: Expr::Var(t), rhs: value } = &stmts[i] else { return false };
    if op != "="
        || scope.stores(t) != 1
        || scope.reads(t) != 1
        || scope.pinned.contains(t)
        || scope.nested_reads.contains(t)
        || vars_read(value).contains(t)
    {
        return false;
    }
    // A loop re-evaluates its test; a temporary read there is read every time.
    if !matches!(&stmts[i + 1], Stmt::Assign { .. } | Stmt::IncDec { .. } | Stmt::Expr(_) | Stmt::Return(_) | Stmt::Throw(_) | Stmt::Echo(_) | Stmt::If { .. } | Stmt::Switch { .. } | Stmt::Foreach { .. }) {
        return false;
    }
    let t = t.clone();
    let mut direct = 0;
    direct_exprs(&mut stmts[i + 1], &mut |e| direct += reads_of(e, &t));
    if direct != 1 {
        return false;
    }
    let Stmt::Assign { rhs: value, .. } = stmts.remove(i) else { return false };
    let mut value = Some(value);
    direct_exprs(&mut stmts[i], &mut |e| substitute(e, &t, &mut value));
    true
}

fn substitute(e: &mut Expr, name: &str, value: &mut Option<Expr>) {
    if matches!(e, Expr::Var(n) if n == name) {
        if let Some(v) = value.take() {
            *e = v;
        }
        return;
    }
    if value.is_some() {
        for_each_child_mut(e, &mut |c| substitute(c, name, value));
    }
}

/// `$x += e` → `$x = $x + e`; `$x .= e` → `$x = $x . e`; `$x ??= e` → `$x = $x ?? e`; `$i++` →
/// `$i = $i + 1`.
fn expand_compound(stmts: &mut [Stmt], i: usize) -> bool {
    match &mut stmts[i] {
        Stmt::Assign { op, lhs, rhs } if op != "=" && op != "=&" => {
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

/// A bare `return;` at the end of a function is falling off its end: both hand back `null`.
fn final_return(stmts: &mut Vec<Stmt>, i: usize, ctx: Ctx) -> bool {
    if !ctx.top || i + 1 != stmts.len() || stmts.len() < 2 {
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
        let f: Arc<str> = Arc::from("t.php");
        let src = format!("<?php\n{body}\n");
        scan_source(&src, &f, &ScanOpts::default()).into_iter().find_map(|d| d.analysis).expect("a function")
    }

    fn xname(body: &str) -> String {
        analysis(body).xname_canonical
    }

    fn lines(body: &str) -> Vec<String> {
        analysis(body).type3_lines
    }

    #[test]
    fn a_guard_and_an_else_are_one_form() {
        let guarded = xname("function f($x) {\n\tif ($x == 3) {\n\t\treturn 1;\n\t}\n\tg();\n\treturn 2;\n}");
        let branched = xname("function f($x) {\n\tif ($x == 3) {\n\t\treturn 1;\n\t} else {\n\t\tg();\n\t\treturn 2;\n\t}\n}");
        let negated = xname("function f($x) {\n\tif ($x != 3) {\n\t\tg();\n\t\treturn 2;\n\t} else\n\t\treturn 1;\n}");
        assert_eq!(guarded, branched);
        assert_eq!(guarded, negated);
    }

    #[test]
    fn the_guard_tests_the_failure() {
        let idiom = "function f($id) {\n\t$row = load($id);\n\tif (!$row) {\n\t\treturn false;\n\t}\n\tuse_it($row);\n\treturn true;\n}";
        let inside_out = "function f($id) {\n\t$row = load($id);\n\tif ($row) {\n\t\tuse_it($row);\n\t\treturn true;\n\t}\n\treturn false;\n}";
        assert_eq!(xname(idiom), xname(inside_out));
        let thrown = "function f($p) {\n\tif (!$p) {\n\t\tthrow new InvalidArgumentException('p');\n\t}\n\tuse_it($p);\n\treturn 0;\n}";
        assert!(xname(thrown).contains("Throw(New(Path('InvalidArgumentException')"), "{}", xname(thrown));
    }

    #[test]
    fn a_yoda_condition_and_a_null_test_have_one_spelling() {
        assert_eq!(
            xname("function f($s) {\n\tif ('open' === $s) {\n\t\tg();\n\t}\n}"),
            xname("function f($s) {\n\tif ($s === 'open') {\n\t\tg();\n\t}\n}")
        );
        let set = xname("function f($a) {\n\tif (isset($a)) {\n\t\tg($a);\n\t}\n}");
        assert_eq!(set, xname("function f($a) {\n\tif ($a !== null) {\n\t\tg($a);\n\t}\n}"));
        assert_eq!(set, xname("function f($a) {\n\tif (null !== $a) {\n\t\tg($a);\n\t}\n}"));
        assert_eq!(
            xname("function f($a) {\n\tif (!is_null($a)) {\n\t\tg($a);\n\t}\n}"),
            set,
            "`!is_null` is `!==`, through negation"
        );
    }

    #[test]
    fn negations_have_one_spelling() {
        assert_eq!(xname("function f($a, $b) {\n\tg();\n\treturn !($a && $b);\n}"), xname("function f($a, $b) {\n\tg();\n\treturn !$a || !$b;\n}"));
        assert_eq!(xname("function f($a, $b) {\n\tg();\n\treturn $a and $b;\n}"), xname("function f($a, $b) {\n\tg();\n\treturn $a && $b;\n}"));
        assert_eq!(xname("function f($x) {\n\tg();\n\treturn !($x === 1);\n}"), xname("function f($x) {\n\tg();\n\treturn $x !== 1;\n}"));
    }

    #[test]
    fn a_switch_is_the_ladder_it_abbreviates() {
        let switched = xname("function f($x) {\n\tswitch ($x) {\n\tcase 1:\n\t\treturn a();\n\tcase 2:\n\tcase 3:\n\t\tb();\n\t\tbreak;\n\tdefault:\n\t\treturn c();\n\t}\n\treturn 0;\n}");
        let ladder = xname("function f($x) {\n\tif ($x == 1)\n\t\treturn a();\n\telseif ($x == 2 || $x == 3)\n\t\tb();\n\telse\n\t\treturn c();\n\treturn 0;\n}");
        assert_eq!(switched, ladder);
        let falls = xname("function f($x) {\n\tswitch ($x) {\n\tcase 1:\n\t\ta();\n\tcase 2:\n\t\tb();\n\t\tbreak;\n\t}\n\treturn $x;\n}");
        assert!(falls.contains("Switch("), "{falls}");
    }

    #[test]
    fn the_alternative_syntax_is_the_braced_one() {
        assert_eq!(
            xname("function f($xs) {\n\tforeach ($xs as $x):\n\t\tif ($x):\n\t\t\tg($x);\n\t\telse:\n\t\t\th($x);\n\t\tendif;\n\tendforeach;\n}"),
            xname("function f($xs) {\n\tforeach ($xs as $x) {\n\t\tif ($x) {\n\t\t\tg($x);\n\t\t} else {\n\t\t\th($x);\n\t\t}\n\t}\n}")
        );
    }

    #[test]
    fn a_temporary_is_inlined_for_the_exact_pass_and_kept_for_type3() {
        let named = "function f($x) {\n\t$t = g($x);\n\treturn h($t);\n}";
        let inlined = "function f($x) {\n\treturn h(g($x));\n}";
        assert_eq!(xname(named), xname(inlined));
        assert_ne!(lines(named), lines(inlined), "Type-3 keeps the statement");
        let by_name = "function f($x) {\n\t$t = g($x);\n\treturn h(compact('t'));\n}";
        assert!(xname(by_name).contains("Assign("), "a body that reads its variables by name keeps them: {}", xname(by_name));
    }

    #[test]
    fn small_spellings_meet() {
        let plus = "function f($x) {\n\t$x = $x + 1;\n\treturn g($x);\n}";
        assert_eq!(xname("function f($x) {\n\t$x += 1;\n\treturn g($x);\n}"), xname(plus));
        assert_eq!(xname("function f($x) {\n\t$x++;\n\treturn g($x);\n}"), xname(plus));
        assert_eq!(xname("function f($x) {\n\t++$x;\n\treturn g($x);\n}"), xname(plus));
        assert_eq!(
            xname("function f($s) {\n\t$s .= 'x';\n\treturn g($s);\n}"),
            xname("function f($s) {\n\t$s = $s . 'x';\n\treturn g($s);\n}")
        );
        assert_eq!(xname("function f($x) {\n\twhile (true) {\n\t\tg($x);\n\t}\n}"), xname("function f($x) {\n\tfor (;;)\n\t\tg($x);\n}"));
        assert_eq!(xname("function f() {\n\treturn array(1, 2);\n}"), xname("function f() {\n\treturn [1, 2];\n}"));
        assert_eq!(
            xname("function f($x) {\n\tif (\\is_array($x)) {\n\t\treturn \\count($x) . \\PHP_EOL;\n\t}\n}"),
            xname("function f($x) {\n\tif (is_array($x)) {\n\t\treturn count($x) . PHP_EOL;\n\t}\n}"),
            "a global function or constant written qualified is the one an unqualified name reaches"
        );
        assert!(xname("function f($e) {\n\treturn $e instanceof \\Exception;\n}").contains("'\\\\Exception'"), "a class is named as written");
    }

    #[test]
    fn a_last_bare_return_is_the_end() {
        assert_eq!(xname("function f($x) {\n\tg($x);\n\th($x);\n\treturn;\n}"), xname("function f($x) {\n\tg($x);\n\th($x);\n}"));
    }

    #[test]
    fn the_stream_keeps_the_body_as_written() {
        let src = "<?php\nfunction f($x) {\n\tswitch ($x) {\n\tcase 1:\n\t\treturn a();\n\t}\n\treturn 0;\n}\n";
        let f: Arc<str> = Arc::from("t.php");
        let d = scan_source(src, &f, &ScanOpts::default()).into_iter().next().expect("f");
        assert!(d.facets.statements.iter().any(|s| s.line.starts_with("Switch(")), "{:?}", d.facets.statements);
    }
}
