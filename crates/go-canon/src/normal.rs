//! **Semantic normal form** of a Go body, taken before its name-agnostic canonicals.
//!
//! The exact pass equates two definitions when their alpha-renamed canonicals are equal, and the
//! Type-3 pass when their statements mostly are. Both read the code as *written*, and the same
//! procedure is written many ways: a guard beside an `if … else`, `if x := f(); c` beside the same
//! two steps on two lines, a temporary named and used once beside the expression inlined, a `switch`
//! beside the `if` ladder it abbreviates, `x++` beside `x += 1` beside `x = x + 1`. Each of those is a
//! Type-4 clone to a reader and an unrelated body to a pass that compares text.
//!
//! This module rewrites a body into one representative of its equivalence class under the rules the
//! Python and Rust frontends' normal forms apply, read in Go's grammar — oriented, applied to a fixed
//! point, one confluent direction per rule.
//!
//! ## Rules
//!
//! Expressions, innermost first, to a fixed point:
//! - **Negation normal form**: `!!x` → `x`; `!(a && b)` → `!a || !b` (and dually); `!(a == b)` →
//!   `a != b`; `!(a < b)` → `a >= b`; `a || (b || c)` → `a || b || c`. Parentheses are dropped: the
//!   tree already says what they grouped.
//! - **Comparison orientation**: `==` / `!=` and the mirror pairs put their operands in a fixed order,
//!   by their shape with the definition's own names blanked — so a renamed copy orients the same way.
//!   (`err != nil` keeps its spelling: a local sorts before `nil`.)
//!
//! Statements, in every block:
//! - **An initializer is a step**: `if x := f(); c` → `x := f()` then `if c`; the same for a `switch`.
//! - **A `switch` is the ladder it abbreviates**: `case a, b:` → `if v == a || v == b`, a tagless
//!   `switch` → `if a`, `default` → the last `else` — when no clause `break`s or falls through, the two
//!   things a ladder could not say.
//! - **Branch orientation**: an `if … else` that neither branch leaves through, whose test is negated
//!   (or mostly negated), is flipped.
//! - **Guard form**: `if c { …; return x } else { B }` → `if c { …; return x }` then `B`; mirrored when
//!   the `else` is the branch that ends.
//! - **Guard orientation**: when the rest of the block ends too, `if c { E }` then `R` is the same
//!   decision as `if !c { R }` then `E`. 🔴 Go's representative is the one Go is written in: the
//!   guard holds the *lighter* branch — the early exit the function is written around; at equal
//!   weight, the branch that *fails* — returns an error as its last value, or panics — so
//!   `if err != nil { return err }` keeps its shape; and only then the positive test, as the other
//!   languages pick. Read by polarity alone, every `if err != nil` (or every positive guard, the other
//!   way round) turned inside out depending on its spelling. Any confluent choice makes the two
//!   shapes meet; this one rewrites real Go the least, so the form the passes compare — and the
//!   helpers patternology prints from it — still reads like Go.
//! - **Single-use temporary** (the exact form only): `t := e` then a statement that reads `t` once and
//!   nothing else does → that statement with `e` in place of `t`.
//! - `a, b := x, y` → `a := x` then `b := y`; `var x = e` → `x := e`; `x += e` and `x++` → `x = x + e`.
//! - A final bare `return` in a function that returns nothing is falling off its end.
//!
//! ## What the rules assume
//!
//! The normalized tree is compared, never compiled: a rule need only preserve meaning as a reader
//! judges it. Hoisting an initializer widens its scope; a ladder evaluates the tag once per test.
//! A name whose address is taken (`&x`) is pinned: inlining it would change what the pointer points at.

use std::collections::HashSet;

use dup_defs_core::normal::{complement, mirror, Scope, MAX_ROUNDS};

use crate::canon::{for_each_child, Child};
use crate::ir::{Case, Elem, Else, Expr, ForHead, Func, Jump, ParamDecl, Results, Spec, Stmt, Target};

/// The normal form of a callable. `inline` admits the single-use-temporary rule: the exact pass wants
/// it, the Type-3 pass does not — its unit is the statement, and folding `t := f(x)` into the line that
/// reads `t` turns two short units into one long one (the Python frontend's measurement).
#[must_use]
pub(crate) fn normal_form(f: &Func, inline: bool) -> Func {
    let mut out = f.clone();
    let void = matches!(f.results, Results::None);
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

/// Every statement list directly inside `s` — its branches and bodies, and the body of every function
/// literal its expressions hold — not the lists nested in those.
fn nested_lists(s: &mut Stmt, f: &mut dyn FnMut(&mut Vec<Stmt>)) {
    match s {
        Stmt::If { init, cond, then, els } => {
            if let Some(init) = init {
                nested_lists(init, f);
            }
            func_lits(cond, f);
            f(then);
            match els.as_deref_mut() {
                Some(Else::Block(b)) => f(b),
                Some(Else::If(alt)) => nested_lists(alt, f),
                None => {}
            }
        }
        Stmt::For { head, body } => {
            match head {
                ForHead::While(c) => func_lits(c, f),
                ForHead::Range { iter, .. } => func_lits(iter, f),
                ForHead::Clause { init, cond, update } => {
                    for part in [init, update].into_iter().flatten() {
                        nested_lists(part, f);
                    }
                    if let Some(c) = cond {
                        func_lits(c, f);
                    }
                }
                ForHead::Loop => {}
            }
            f(body);
        }
        Stmt::Switch { init, value, cases } => {
            if let Some(init) = init {
                nested_lists(init, f);
            }
            if let Some(v) = value {
                func_lits(v, f);
            }
            for case in cases {
                case.values.iter_mut().flatten().for_each(|e| func_lits(e, f));
                f(&mut case.body);
            }
        }
        Stmt::TypeSwitch { init, value, cases, .. } => {
            if let Some(init) = init {
                nested_lists(init, f);
            }
            func_lits(value, f);
            for case in cases {
                f(&mut case.body);
            }
        }
        Stmt::Select { cases } => {
            for case in cases {
                if let Some(comm) = &mut case.comm {
                    nested_lists(comm, f);
                }
                f(&mut case.body);
            }
        }
        Stmt::Block(b) => f(b),
        Stmt::Labeled { stmt, .. } => nested_lists(stmt, f),
        other => direct_exprs(other, &mut |e| func_lits(e, f)),
    }
}

/// The bodies of the function literals `e` holds, outermost first.
fn func_lits(e: &mut Expr, f: &mut dyn FnMut(&mut Vec<Stmt>)) {
    if let Expr::FuncLit { body, .. } = e {
        f(body);
        return;
    }
    for_each_child_mut(e, &mut |c| func_lits(c, f));
}

/// The expressions a statement evaluates itself, before any block inside it runs — the slots a
/// temporary may be inlined into, and the roots the expression rules walk.
fn direct_exprs(s: &mut Stmt, f: &mut dyn FnMut(&mut Expr)) {
    match s {
        Stmt::Define { lhs, rhs } => {
            for t in lhs {
                if let Target::Expr(e) = t {
                    f(e);
                }
            }
            rhs.iter_mut().for_each(f);
        }
        Stmt::Decl { specs, .. } => specs.iter_mut().flat_map(|s| s.values.iter_mut()).for_each(f),
        Stmt::Assign { lhs, rhs, .. } => lhs.iter_mut().chain(rhs.iter_mut()).for_each(f),
        Stmt::IncDec { target, .. } | Stmt::Expr(target) | Stmt::Go(target) | Stmt::Defer(target) => f(target),
        Stmt::Send { ch, value } => {
            f(ch);
            f(value);
        }
        Stmt::Return(values) => values.iter_mut().for_each(f),
        Stmt::If { cond, .. } => f(cond),
        Stmt::For { head: ForHead::Range { iter, .. }, .. } => f(iter),
        Stmt::Switch { value: Some(v), .. } => f(v),
        Stmt::TypeSwitch { value, .. } => f(value),
        Stmt::Recv { rhs, .. } => f(rhs),
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
        Stmt::If { init, cond, then, els } => {
            if let Some(init) = init {
                stmt_exprs(init, f);
            }
            f(cond);
            all_exprs(then, f);
            match els.as_deref_mut() {
                Some(Else::Block(b)) => all_exprs(b, f),
                Some(Else::If(alt)) => stmt_exprs(alt, f),
                None => {}
            }
        }
        Stmt::For { head, body } => {
            match head {
                ForHead::While(c) => f(c),
                ForHead::Range { lhs, iter, .. } => {
                    for t in lhs.iter_mut().flatten() {
                        if let Target::Expr(e) = t {
                            f(e);
                        }
                    }
                    f(iter);
                }
                ForHead::Clause { init, cond, update } => {
                    for part in [init, update].into_iter().flatten() {
                        stmt_exprs(part, f);
                    }
                    if let Some(c) = cond {
                        f(c);
                    }
                }
                ForHead::Loop => {}
            }
            all_exprs(body, f);
        }
        Stmt::Switch { init, value, cases } => {
            if let Some(init) = init {
                stmt_exprs(init, f);
            }
            if let Some(v) = value {
                f(v);
            }
            for case in cases {
                case.values.iter_mut().flatten().for_each(&mut *f);
                all_exprs(&mut case.body, f);
            }
        }
        Stmt::TypeSwitch { init, value, cases, .. } => {
            if let Some(init) = init {
                stmt_exprs(init, f);
            }
            f(value);
            for case in cases {
                all_exprs(&mut case.body, f);
            }
        }
        Stmt::Select { cases } => {
            for case in cases {
                if let Some(comm) = &mut case.comm {
                    stmt_exprs(comm, f);
                }
                all_exprs(&mut case.body, f);
            }
        }
        Stmt::Recv { lhs, rhs, .. } => {
            for t in lhs.iter_mut().flatten() {
                if let Target::Expr(e) = t {
                    f(e);
                }
            }
            f(rhs);
        }
        Stmt::Block(b) => all_exprs(b, f),
        Stmt::Labeled { stmt, .. } => stmt_exprs(stmt, f),
        other => direct_exprs(other, f),
    }
}

/// [`for_each_child`], mutably: every direct sub-expression, and the expressions of a function
/// literal's body.
fn for_each_child_mut(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    match e {
        Expr::Binary { l, r, .. } => {
            f(l);
            f(r);
        }
        Expr::Unary { e, .. }
        | Expr::Paren(e)
        | Expr::Spread(e)
        | Expr::Assert { e, .. }
        | Expr::Conv { e, .. }
        | Expr::Selector { operand: e, .. } => f(e),
        Expr::Call { f: callee, args, .. } => {
            f(callee);
            args.iter_mut().for_each(f);
        }
        Expr::Index { e, i } => {
            f(e);
            f(i);
        }
        Expr::Slice { e, lo, hi, cap } => {
            f(e);
            for part in [lo, hi, cap].into_iter().flatten() {
                f(part);
            }
        }
        Expr::Composite { elems, .. } => {
            for el in elems {
                match el {
                    Elem::Field(_, v) | Elem::Plain(v) | Elem::KeyedBlank(v) => f(v),
                    Elem::Keyed(k, v) => {
                        f(k);
                        f(v);
                    }
                }
            }
        }
        Expr::FuncLit { body, .. } => all_exprs(body, f),
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
    walk.params(&f.params);
    if let Results::List(list) = &f.results {
        walk.params(list);
    }
    walk.stmts(&f.body);
    scope
}

struct ScopeWalk<'s> {
    scope: &'s mut Scope,
    depth: usize,
}

impl ScopeWalk<'_> {
    fn params(&mut self, decls: &[ParamDecl]) {
        for name in decls.iter().flat_map(|d| d.names.iter()) {
            self.scope.bind(name);
        }
    }

    fn targets(&mut self, targets: &[Target], define: bool) {
        for t in targets {
            match t {
                Target::Name(n) if define => self.scope.bind(n),
                Target::Name(n) => self.scope.store(n),
                Target::Expr(e) => self.store_target(e),
            }
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

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    #[allow(clippy::too_many_lines)] // one arm per statement shape reads better straight-through
    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Define { lhs, rhs } => {
                for e in rhs {
                    self.expr(e);
                }
                self.targets(lhs, true);
            }
            Stmt::Decl { specs, .. } => {
                for spec in specs {
                    spec.values.iter().for_each(|e| self.expr(e));
                    spec.names.iter().for_each(|n| self.scope.bind(n));
                }
            }
            Stmt::Assign { op, lhs, rhs } => {
                for e in rhs {
                    self.expr(e);
                }
                for target in lhs {
                    self.store_target(target);
                    // A compound assignment reads what it writes.
                    if op != "=" {
                        self.expr(target);
                    }
                }
            }
            Stmt::IncDec { target, .. } => {
                self.store_target(target);
                self.expr(target);
            }
            Stmt::Expr(e) | Stmt::Go(e) | Stmt::Defer(e) => self.expr(e),
            Stmt::Send { ch, value } => {
                self.expr(ch);
                self.expr(value);
            }
            Stmt::Return(values) => values.iter().for_each(|e| self.expr(e)),
            Stmt::If { init, cond, then, els } => {
                if let Some(init) = init {
                    self.stmt(init);
                }
                self.expr(cond);
                self.stmts(then);
                match els.as_deref() {
                    Some(Else::Block(b)) => self.stmts(b),
                    Some(Else::If(alt)) => self.stmt(alt),
                    None => {}
                }
            }
            Stmt::For { head, body } => {
                match head {
                    ForHead::While(c) => self.expr(c),
                    ForHead::Range { define, lhs, iter } => {
                        self.expr(iter);
                        if let Some(lhs) = lhs {
                            self.targets(lhs, *define);
                        }
                    }
                    ForHead::Clause { init, cond, update } => {
                        if let Some(i) = init {
                            self.stmt(i);
                        }
                        if let Some(c) = cond {
                            self.expr(c);
                        }
                        if let Some(u) = update {
                            self.stmt(u);
                        }
                    }
                    ForHead::Loop => {}
                }
                self.stmts(body);
            }
            Stmt::Switch { init, value, cases } => {
                if let Some(init) = init {
                    self.stmt(init);
                }
                if let Some(v) = value {
                    self.expr(v);
                }
                for case in cases {
                    case.values.iter().flatten().for_each(|e| self.expr(e));
                    self.stmts(&case.body);
                }
            }
            Stmt::TypeSwitch { init, alias, value, cases } => {
                if let Some(init) = init {
                    self.stmt(init);
                }
                self.expr(value);
                if let Some(alias) = alias {
                    self.targets(alias, true);
                }
                for case in cases {
                    self.stmts(&case.body);
                }
            }
            Stmt::Select { cases } => {
                for case in cases {
                    if let Some(comm) = &case.comm {
                        self.stmt(comm);
                    }
                    self.stmts(&case.body);
                }
            }
            Stmt::Recv { define, lhs, rhs } => {
                self.expr(rhs);
                if let Some(lhs) = lhs {
                    self.targets(lhs, *define);
                }
            }
            Stmt::Block(b) => self.stmts(b),
            Stmt::Labeled { stmt, .. } => self.stmt(stmt),
            Stmt::NestedType | Stmt::Jump { .. } | Stmt::Fallthrough | Stmt::Empty | Stmt::Unknown(_) => {}
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
            Expr::FuncLit { params, body, .. } => {
                self.depth += 1;
                self.params(params);
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

/// An `if … else` is written with its positive test first.
fn prefer_flip(test: &Expr, locals: &HashSet<String>) -> bool {
    leans_negative(test, locals)
}

/// Whether a value is, by Go's own conventions, an error: a name like `err` / `myErr` / `ErrX`, a
/// package's sentinel (`sql.ErrNoRows`), something built by a constructor that says so (`errors.New`,
/// `fmt.Errorf`, `status.Error`, `NewError`), or a literal of a type named for one.
fn errorish(e: &Expr) -> bool {
    let says_error = |name: &str| name.contains("Err") || name.contains("err");
    match e {
        Expr::Name(n) => {
            let lower = n.to_ascii_lowercase();
            lower.ends_with("err") || lower.starts_with("err")
        }
        Expr::Selector { field, .. } => field.starts_with("Err") || field.starts_with("err"),
        Expr::Call { f, .. } => match f.as_ref() {
            Expr::Name(n) => says_error(n),
            Expr::Selector { operand, field } => {
                says_error(field) || matches!(operand.as_ref(), Expr::Name(pkg) if says_error(pkg))
            }
            _ => false,
        },
        Expr::Unary { op, e } if op == "&" => errorish(e),
        Expr::Composite { ty: Some(crate::ir::Type::Named(t)), .. } => t.contains("Error"),
        _ => false,
    }
}

/// Whether a branch is the failure path: it panics, or returns an error as its last value.
fn fails(stmts: &[Stmt]) -> bool {
    match stmts.last() {
        Some(Stmt::Return(values)) => values.last().is_some_and(errorish),
        Some(Stmt::Expr(Expr::Call { f, .. })) => matches!(f.as_ref(), Expr::Name(n) if n == "panic"),
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
                Stmt::For { body, .. } | Stmt::Block(body) => weight(body),
                Stmt::Switch { cases, .. } => cases.iter().map(|c| weight(&c.body)).sum(),
                Stmt::TypeSwitch { cases, .. } => cases.iter().map(|c| weight(&c.body)).sum(),
                Stmt::Select { cases } => cases.iter().map(|c| weight(&c.body)).sum(),
                Stmt::Labeled { stmt, .. } => weight(std::slice::from_ref(stmt)),
                _ => 0,
            }
        })
        .sum()
}

/// Of two branches that both end, which one the guard holds: the lighter — the early exit the code is
/// written around; at equal weight the one that fails; and only then the positive test, as in the
/// other languages. `true` when the guard should hold `other` rather than `guard`, tested by `test`.
///
/// Each step decides a pair the same way from either side, so the rule is confluent: once swapped,
/// the same comparison keeps it.
fn guard_should_swap(guard: &[Stmt], other: &[Stmt], test: &Expr, locals: &HashSet<String>) -> bool {
    let (g, o) = (weight(guard), weight(other));
    if g != o {
        return o < g;
    }
    let (guard_fails, other_fails) = (fails(guard), fails(other));
    if guard_fails != other_fails {
        return other_fails;
    }
    prefer_flip(test, locals)
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
    match e {
        Expr::Unary { op, e: inner } if op == "!" => {
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

/// Whether control never falls out of the bottom of `s`.
fn ends(s: &Stmt) -> bool {
    match s {
        Stmt::Return(_) | Stmt::Jump { .. } => true,
        Stmt::Expr(Expr::Call { f, .. }) => matches!(f.as_ref(), Expr::Name(n) if n == "panic"),
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
        let rewritten = hoist_init(stmts, i)
            || var_to_define(stmts, i)
            || switch_to_ladder(stmts, i)
            || orient_branches(stmts, i, scope)
            || guard_form(stmts, i, scope)
            || orient_guard(stmts, i, scope, ctx)
            || (ctx.inline && inline_temporary(stmts, i, scope))
            || split_define(stmts, i)
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

/// `if x := f(); c` → `x := f()` then `if c`; the same for a `switch` and a type switch.
fn hoist_init(stmts: &mut Vec<Stmt>, i: usize) -> bool {
    let init = match &mut stmts[i] {
        Stmt::If { init, .. } | Stmt::Switch { init, .. } | Stmt::TypeSwitch { init, .. } => init.take(),
        _ => None,
    };
    let Some(init) = init else { return false };
    stmts.insert(i, *init);
    true
}

/// `var x = e` → `x := e`: one binding, written two ways.
fn var_to_define(stmts: &mut [Stmt], i: usize) -> bool {
    let Stmt::Decl { konst: false, specs } = &stmts[i] else { return false };
    let [Spec { names, ty: None, values }] = specs.as_slice() else { return false };
    if names.len() != values.len() || values.is_empty() {
        return false;
    }
    let lhs = names.iter().cloned().map(Target::Name).collect();
    stmts[i] = Stmt::Define { lhs, rhs: values.clone() };
    true
}

/// Whether a clause body leaves its `switch` in a way a ladder cannot say: an unlabeled `break` that
/// would bind to the `switch`, or a `fallthrough`.
fn leaves_switch(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Jump { kind: Jump::Break, label: None } | Stmt::Fallthrough => true,
        Stmt::If { then, els, .. } => {
            leaves_switch(then)
                || match els.as_deref() {
                    Some(Else::Block(b)) => leaves_switch(b),
                    Some(Else::If(alt)) => leaves_switch(std::slice::from_ref(alt)),
                    None => false,
                }
        }
        Stmt::Block(b) => leaves_switch(b),
        Stmt::Labeled { stmt, .. } => leaves_switch(std::slice::from_ref(stmt)),
        // A `break` inside these binds to them.
        _ => false,
    })
}

/// `switch v { case a, b: A; default: D }` → `if v == a || v == b { A } else { D }`; a tagless
/// `switch` tests its cases themselves.
fn switch_to_ladder(stmts: &mut [Stmt], i: usize) -> bool {
    let Stmt::Switch { init: None, value, cases } = &stmts[i] else { return false };
    if cases.iter().any(|c| leaves_switch(&c.body)) {
        return false;
    }
    let (defaults, rungs): (Vec<&Case>, Vec<&Case>) = cases.iter().partition(|c| c.values.is_none());
    if rungs.is_empty() || defaults.len() > 1 || rungs.iter().any(|c| c.values.as_ref().is_some_and(Vec::is_empty)) {
        return false;
    }
    let test = |c: &Case| -> Option<Expr> {
        c.values.as_ref()?.iter().cloned().map(|v| match value {
            Some(tag) => binary(tag.clone(), "==", v),
            None => v,
        }).reduce(|a, b| binary(a, "||", b))
    };
    // Built from the last rung up, each rung the `else` of the one before — the nested shape every rule reads.
    let mut otherwise: Option<Vec<Stmt>> = defaults.first().map(|d| d.body.clone());
    for case in rungs.iter().rev() {
        let Some(cond) = test(case) else { return false };
        let rung = Stmt::If { init: None, cond, then: case.body.clone(), els: otherwise.map(|b| Box::new(Else::Block(b))) };
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
    if block_ends(then) || block_ends(alt) || !prefer_flip(cond, &scope.locals) {
        return false;
    }
    let c = std::mem::replace(cond, placeholder());
    *cond = negate(c);
    swap_branches(then, els);
    true
}

/// `if c { <ends> } else { B }` → `if c { <ends> }` then `B`; when only the `else` ends, the test is
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

/// A guard whose remainder also ends is a two-branch `if` written as a guard: `if c { <ends> }` then
/// `<ends>` ≡ `if !c { <ends> }` then `<ends>`. Go's representative tests the failure.
fn orient_guard(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope, ctx: Ctx) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Stmt::If { cond, then, els: None, init: None } = &stmts[i] else { return false };
    if !block_ends(then) || !rest_ends(&stmts[i + 1..], ctx) || !guard_should_swap(then, &stmts[i + 1..], cond, &scope.locals) {
        return false;
    }
    let rest: Vec<Stmt> = stmts.drain(i + 1..).collect();
    let Some(Stmt::If { cond, then, .. }) = stmts.pop() else { return false };
    stmts.push(Stmt::If { init: None, cond: negate(cond), then: rest, els: None });
    stmts.extend(then);
    true
}

/// `t := e` then `S(t)`, with `t` bound once and read once, there → `S(e)`.
fn inline_temporary(stmts: &mut Vec<Stmt>, i: usize, scope: &Scope) -> bool {
    if i + 1 >= stmts.len() {
        return false;
    }
    let Stmt::Define { lhs, rhs } = &stmts[i] else { return false };
    let ([Target::Name(t)], [value]) = (lhs.as_slice(), rhs.as_slice()) else { return false };
    if t == "_"
        || scope.stores(t) != 1
        || scope.reads(t) != 1
        || scope.pinned.contains(t)
        || scope.nested_reads.contains(t)
        || names_read(value).contains(t)
    {
        return false;
    }
    // A loop re-evaluates its test; a temporary read there is read every time.
    if matches!(&stmts[i + 1], Stmt::For { .. } | Stmt::Labeled { .. } | Stmt::Block(_) | Stmt::Select { .. }) {
        return false;
    }
    let t = t.clone();
    let mut direct = 0;
    direct_exprs(&mut stmts[i + 1], &mut |e| direct += reads_of(e, &t));
    if direct != 1 {
        return false;
    }
    let Stmt::Define { mut rhs, .. } = stmts.remove(i) else { return false };
    let Some(value) = rhs.pop() else { return false };
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

/// `a, b := x, y` → `a := x` then `b := y`, unless a later value reads an earlier name.
fn split_define(stmts: &mut Vec<Stmt>, i: usize) -> bool {
    let Stmt::Define { lhs, rhs } = &stmts[i] else { return false };
    if lhs.len() < 2 || lhs.len() != rhs.len() {
        return false;
    }
    let names: Vec<&str> = lhs
        .iter()
        .filter_map(|t| match t {
            Target::Name(n) if n != "_" => Some(n.as_str()),
            _ => None,
        })
        .collect();
    if names.len() != lhs.len() {
        return false;
    }
    for (k, value) in rhs.iter().enumerate() {
        let reads = names_read(value);
        if names[..k].iter().any(|n| reads.contains(*n)) {
            return false;
        }
    }
    let Stmt::Define { lhs, rhs } = stmts.remove(i) else { return false };
    let pairs: Vec<Stmt> = lhs.into_iter().zip(rhs).map(|(t, v)| Stmt::Define { lhs: vec![t], rhs: vec![v] }).collect();
    stmts.splice(i..i, pairs);
    true
}

/// `x += e` → `x = x + e`; `x++` → `x = x + 1`.
fn expand_compound(stmts: &mut [Stmt], i: usize) -> bool {
    match &mut stmts[i] {
        Stmt::Assign { op, lhs, rhs } if op != "=" && lhs.len() == 1 && rhs.len() == 1 => {
            let base = op.trim_end_matches('=').to_owned();
            let (Some(target), Some(value)) = (lhs.pop(), rhs.pop()) else { return false };
            stmts[i] = Stmt::Assign { op: "=".to_owned(), lhs: vec![target.clone()], rhs: vec![binary(target, &base, value)] };
            true
        }
        Stmt::IncDec { inc, target } => {
            let base = if *inc { "+" } else { "-" };
            let target = std::mem::replace(target, placeholder());
            let one = Expr::Lit("Int", "1".to_owned());
            stmts[i] = Stmt::Assign { op: "=".to_owned(), lhs: vec![target.clone()], rhs: vec![binary(target, base, one)] };
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
    if matches!(&stmts[i], Stmt::Return(values) if values.is_empty()) {
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
        let src = format!("package p\n{body}\n");
        let f: Arc<str> = Arc::from("t.go");
        scan_source(&src, &f, &ScanOpts::default()).into_iter().find_map(|d| d.analysis).expect("a callable")
    }

    fn xname(body: &str) -> String {
        analysis(body).xname_canonical
    }

    fn lines(body: &str) -> Vec<String> {
        analysis(body).type3_lines
    }

    #[test]
    fn a_guard_and_an_else_are_one_form() {
        let guarded = xname("func F(x int) int {\n\tif x == 0 {\n\t\treturn 1\n\t}\n\tg()\n\treturn 2\n}");
        let branched = xname("func F(x int) int {\n\tif x == 0 {\n\t\treturn 1\n\t} else {\n\t\tg()\n\t\treturn 2\n\t}\n}");
        let negated = xname("func F(x int) int {\n\tif x != 0 {\n\t\tg()\n\t\treturn 2\n\t} else {\n\t\treturn 1\n\t}\n}");
        assert_eq!(guarded, branched);
        assert_eq!(guarded, negated);
    }

    #[test]
    fn the_go_guard_tests_the_failure() {
        // The idiom is the representative: `if err != nil { return err }` keeps its shape, and the same
        // decision written inside out meets it.
        let idiom = "func F() (int, error) {\n\tv, err := load()\n\tif err != nil {\n\t\treturn 0, err\n\t}\n\tuse(v)\n\treturn v, nil\n}";
        let inside_out = "func F() (int, error) {\n\tv, err := load()\n\tif err == nil {\n\t\tuse(v)\n\t\treturn v, nil\n\t}\n\treturn 0, err\n}";
        assert_eq!(xname(idiom), xname(inside_out));
        assert!(xname(idiom).contains("If(, Bin(Path('_v1'), '!=', Nil())"), "{}", xname(idiom));
    }

    #[test]
    fn a_positive_guard_keeps_its_shape_and_its_mirror_meets_the_err_idiom() {
        // `if x < lower { return lower }` is not a failure test turned around: it stays as written.
        let clamp = xname("func F(x, lower int) int {\n\tif x < lower {\n\t\treturn lower\n\t}\n\treturn x\n}");
        assert!(clamp.contains("'<'"), "{clamp}");
        assert!(!clamp.contains("Unary('!'"), "{clamp}");
        // The tiny wrapper written inside out meets the idiom at equal weight.
        let idiom = xname("func F() error {\n\terr := g()\n\tif err != nil {\n\t\treturn err\n\t}\n\treturn nil\n}");
        let inside_out = xname("func F() error {\n\terr := g()\n\tif err == nil {\n\t\treturn nil\n\t}\n\treturn err\n}");
        assert_eq!(idiom, inside_out);
        assert_eq!(xname("func F(a, b int) bool {\n\tg()\n\treturn !(a < b)\n}"), xname("func F(a, b int) bool {\n\tg()\n\treturn a >= b\n}"));
    }

    #[test]
    fn an_initializer_is_a_step_of_its_own() {
        let joined = xname("func F() error {\n\tif err := load(); err != nil {\n\t\treturn err\n\t}\n\treturn nil\n}");
        let split = xname("func F() error {\n\terr := load()\n\tif err != nil {\n\t\treturn err\n\t}\n\treturn nil\n}");
        assert_eq!(joined, split);
    }

    #[test]
    fn negations_and_comparisons_have_one_spelling() {
        assert_eq!(xname("func F(a, b bool) bool {\n\tg()\n\treturn !(a && b)\n}"), xname("func F(a, b bool) bool {\n\tg()\n\treturn !a || !b\n}"));
        assert_eq!(xname("func F(a, b int) bool {\n\tg()\n\treturn !(a == b)\n}"), xname("func F(a, b int) bool {\n\tg()\n\treturn a != b\n}"));
        assert_eq!(xname("func F(x int) bool {\n\tg()\n\treturn x > Limit\n}"), xname("func F(x int) bool {\n\tg()\n\treturn Limit < x\n}"));
        assert_eq!(xname("func F(err error) bool {\n\tg()\n\treturn nil != err\n}"), xname("func F(err error) bool {\n\tg()\n\treturn err != nil\n}"));
    }

    #[test]
    fn a_switch_is_the_ladder_it_abbreviates() {
        let switched = xname("func F(x int) int {\n\tswitch x {\n\tcase 1:\n\t\treturn a()\n\tcase 2, 3:\n\t\treturn b()\n\tdefault:\n\t\treturn c()\n\t}\n}");
        let ladder = xname("func F(x int) int {\n\tif x == 1 {\n\t\treturn a()\n\t} else if x == 2 || x == 3 {\n\t\treturn b()\n\t}\n\treturn c()\n}");
        assert_eq!(switched, ladder);
        let tagless = xname("func F(x int) int {\n\tswitch {\n\tcase x > 1:\n\t\tg()\n\tdefault:\n\t\th()\n\t}\n\treturn x\n}");
        let ifs = xname("func F(x int) int {\n\tif x > 1 {\n\t\tg()\n\t} else {\n\t\th()\n\t}\n\treturn x\n}");
        assert_eq!(tagless, ifs);
    }

    #[test]
    fn a_switch_that_breaks_out_stays_a_switch() {
        let src = "func F(x int) int {\n\tswitch x {\n\tcase 1:\n\t\tif ok() {\n\t\t\tbreak\n\t\t}\n\t\tg()\n\t}\n\treturn x\n}";
        assert!(xname(src).contains("Switch("), "{}", xname(src));
    }

    #[test]
    fn a_temporary_is_inlined_for_the_exact_pass_and_kept_for_type3() {
        let named = "func F(x int) int {\n\tt := g(x)\n\treturn h(t)\n}";
        let inlined = "func F(x int) int {\n\treturn h(g(x))\n}";
        assert_eq!(xname(named), xname(inlined));
        assert_ne!(lines(named), lines(inlined), "Type-3 keeps the statement");
    }

    #[test]
    fn small_spellings_meet() {
        let plus = "func F(x int) int {\n\tx = x + 1\n\treturn g(x)\n}";
        assert_eq!(xname("func F(x int) int {\n\tx += 1\n\treturn g(x)\n}"), xname(plus));
        assert_eq!(xname("func F(x int) int {\n\tx++\n\treturn g(x)\n}"), xname(plus));
        assert_eq!(
            xname("func F() int {\n\ta, b := g(), h()\n\treturn a + b\n}"),
            xname("func F() int {\n\ta := g()\n\tb := h()\n\treturn a + b\n}")
        );
        assert_eq!(xname("func F() int {\n\tvar a = g()\n\treturn k(a, a)\n}"), xname("func F() int {\n\ta := g()\n\treturn k(a, a)\n}"));
    }

    #[test]
    fn a_void_function_s_last_return_is_its_end() {
        assert_eq!(xname("func F(x int) {\n\tg(x)\n\th(x)\n\treturn\n}"), xname("func F(x int) {\n\tg(x)\n\th(x)\n}"));
    }

    #[test]
    fn a_taken_address_is_not_inlined() {
        let src = "func F() int {\n\tt := g()\n\treturn h(&t)\n}";
        assert!(xname(src).contains("Let(Bind('_v0'), Call(Path('g'), ))"), "{}", xname(src));
    }

    #[test]
    fn the_stream_keeps_the_body_as_written() {
        // Converge points the reader at code: the statement stream is never the normal form.
        let src = "package p\nfunc F(x int) int {\n\tswitch x {\n\tcase 1:\n\t\treturn a()\n\t}\n\treturn 0\n}\n";
        let f: Arc<str> = Arc::from("t.go");
        let d = scan_source(src, &f, &ScanOpts::default()).into_iter().next().expect("F");
        assert!(d.facets.statements.iter().any(|s| s.line.starts_with("Switch(")), "{:?}", d.facets.statements);
    }
}
