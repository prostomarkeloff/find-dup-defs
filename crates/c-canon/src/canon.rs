//! Structural canonicalization of C definitions, written from the owned tree ([`crate::ir`]).
//!
//! The role `go-canon::canon` plays for Go: a compact, internally consistent **s-expr** per definition
//! that `difflib-fast` compares for name-gated similarity, and that the cross-name pass `Eq`-checks
//! once locals are alpha-renamed. The tag vocabulary is the Rust and Go frontends' wherever the
//! construct is the same one (`Func`, `Params`, `Block`, `Let`, `ExprStmt`, `If`, `While`, `ForC`,
//! `Switch`, `Case`, `Call`, `Method`, `Field`, `Bin`, …), so the engine's s-expr readers — converge's
//! vocabulary split, the unit counter — read C the way they already read the others. Two modes,
//! driven by [`Dump::locals`]:
//!
//! * **cluster** (`locals = None`) — identifiers pass through verbatim; the names-preserved canonical
//!   the name-gated pass clusters on.
//! * **xname** (`locals = Some(set)`) — parameters and local declarations are renumbered `_v{n}` by
//!   first occurrence and the top definition's own name is blanked to `_fn`.
//!
//! A call through a member — `dev->ops->open(dev)` — is written `Method(receiver, 'open', args)`: it is
//! C's method call, dispatched through a table of function pointers, and which member it calls is a
//! selector as a method's name is in the other languages.
//!
//! The canonicals are written from the owned tree rather than from tree-sitter's because the exact and
//! Type-3 canonicals read the body's normal form ([`crate::normal`]), which rewrites it.
#![allow(
    clippy::too_many_lines, // the stmt/expr/type matches enumerate the tree's shapes; splitting just scatters one
    clippy::match_same_arms // distinct shapes intentionally share an emission for clarity
)]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;

use dup_defs_core::uses::{site_ctx, Construct, Site, Within};
use dup_defs_core::{quoted, AnalyzedFn, Statement, UNFILED};

use crate::ir::{Case, Elem, Else, Expr, FieldDecl, Func, Jump, Param, Spec, Stmt, Type};

// ───────────────────────────── bound-locals collector ─────────────────────────────

/// Every object the function binds — the rename set for xname mode, and in every mode the set that
/// tells a local from a name the file or a header declared: parameters and every local declaration,
/// those inside a statement expression included.
#[derive(Default)]
struct Collect {
    bound: HashSet<String>,
}

impl Collect {
    fn params(&mut self, params: &[Param]) {
        for p in params {
            if let Some(name) = &p.name {
                self.bound.insert(name.clone());
            }
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        each_stmt_part(s, &mut |part| match part {
            Part::Spec(spec) => {
                self.bound.insert(spec.name.clone());
                if let Some(init) = &spec.init {
                    self.expr(init);
                }
            }
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    fn expr(&mut self, e: &Expr) {
        for_each_child(e, &mut |child| match child {
            Child::Expr(e) => self.expr(e),
            Child::Body(body) => self.stmts(body),
        });
    }
}

/// A part of a statement: a declarator it declares, a statement or list of them it holds, or an
/// expression it evaluates.
pub(crate) enum Part<'s> {
    Spec(&'s Spec),
    Stmt(&'s Stmt),
    Stmts(&'s [Stmt]),
    Expr(&'s Expr),
}

/// Every part of `s`, in source order — what it declares, what it tests, its bodies.
pub(crate) fn each_stmt_part<'s>(s: &'s Stmt, f: &mut dyn FnMut(Part<'s>)) {
    match s {
        Stmt::Decl(specs) => specs.iter().for_each(|spec| f(Part::Spec(spec))),
        Stmt::Assign { lhs, rhs, .. } => {
            f(Part::Expr(lhs));
            f(Part::Expr(rhs));
        }
        Stmt::IncDec { target: e, .. } | Stmt::Expr(e) => f(Part::Expr(e)),
        Stmt::Return(value) => {
            if let Some(v) = value {
                f(Part::Expr(v));
            }
        }
        Stmt::If { cond, then, els } => {
            f(Part::Expr(cond));
            f(Part::Stmts(then));
            match els.as_deref() {
                Some(Else::Block(b)) => f(Part::Stmts(b)),
                Some(Else::If(alt)) => f(Part::Stmt(alt)),
                None => {}
            }
        }
        Stmt::While { cond, body } => {
            f(Part::Expr(cond));
            f(Part::Stmts(body));
        }
        Stmt::DoWhile { body, cond } => {
            f(Part::Stmts(body));
            f(Part::Expr(cond));
        }
        Stmt::For { init, cond, update, body } => {
            if let Some(i) = init {
                f(Part::Stmt(i));
            }
            if let Some(c) = cond {
                f(Part::Expr(c));
            }
            f(Part::Stmts(body));
            if let Some(u) = update {
                f(Part::Stmt(u));
            }
        }
        Stmt::Switch { value, cases } => {
            f(Part::Expr(value));
            for case in cases {
                if let Some(v) = &case.value {
                    f(Part::Expr(v));
                }
                f(Part::Stmts(&case.body));
            }
        }
        Stmt::Block(b) => f(Part::Stmts(b)),
        Stmt::Labeled { stmt, .. } => {
            if let Some(s) = stmt {
                f(Part::Stmt(s));
            }
        }
        Stmt::PreIf { branches, .. } => branches.iter().for_each(|b| f(Part::Stmts(b))),
        Stmt::MacroLoop { head, body } => {
            f(Part::Expr(head));
            f(Part::Stmts(body));
        }
        Stmt::NestedType | Stmt::Jump { .. } | Stmt::Asm | Stmt::Empty | Stmt::Unknown(_) => {}
    }
}

/// What an expression holds that a walk may want: its sub-expressions, and the body of a statement
/// expression.
pub(crate) enum Child<'e> {
    Expr(&'e Expr),
    Body(&'e [Stmt]),
}

/// Every direct child of an expression, in source order.
pub(crate) fn for_each_child<'e>(e: &'e Expr, f: &mut dyn FnMut(Child<'e>)) {
    match e {
        Expr::Binary { l, r, .. } | Expr::Assign { l, r, .. } | Expr::Comma(l, r) => {
            f(Child::Expr(l));
            f(Child::Expr(r));
        }
        Expr::Unary { e, .. }
        | Expr::Update { e, .. }
        | Expr::Paren(e)
        | Expr::Field { e, .. }
        | Expr::Cast { e, .. }
        | Expr::SizeofValue(e)
        | Expr::Generic(e) => f(Child::Expr(e)),
        Expr::Cond { c, t, f: alt } => {
            f(Child::Expr(c));
            if let Some(t) = t {
                f(Child::Expr(t));
            }
            f(Child::Expr(alt));
        }
        Expr::Call { f: callee, args } => {
            f(Child::Expr(callee));
            for a in args {
                f(Child::Expr(a));
            }
        }
        Expr::Index { e, i } => {
            f(Child::Expr(e));
            f(Child::Expr(i));
        }
        Expr::Compound { elems, .. } | Expr::InitList(elems) => {
            for el in elems {
                match el {
                    Elem::Plain(v) | Elem::Field(_, v) => f(Child::Expr(v)),
                    Elem::Index(i, v) => {
                        f(Child::Expr(i));
                        f(Child::Expr(v));
                    }
                }
            }
        }
        Expr::GnuBlock(body) => f(Child::Body(body)),
        Expr::Name(_)
        | Expr::Lit(..)
        | Expr::Bool(_)
        | Expr::Null
        | Expr::SizeofType(_)
        | Expr::Alignof(_)
        | Expr::Offsetof(..)
        | Expr::Asm
        | Expr::Unknown(_) => {}
    }
}

/// The bound locals of one function.
pub(crate) fn locals_of(f: &Func) -> HashSet<String> {
    let mut collect = Collect::default();
    collect.params(&f.params);
    collect.stmts(&f.body);
    collect.bound
}

/// The names a function reads that it did not bind — the functions it calls, the objects, macros and
/// enumerators it names, and the types it spells — what its file's headers may declare.
pub(crate) fn free_names(f: &Func) -> BTreeSet<String> {
    struct Free<'b> {
        bound: &'b HashSet<String>,
        out: BTreeSet<String>,
    }
    impl Free<'_> {
        fn ty(&mut self, t: &Type) {
            match t {
                Type::Named(n) => {
                    self.out.insert(n.clone());
                }
                Type::Ptr(inner) | Type::Array(inner, _) => self.ty(inner),
                Type::Func(params, _, ret) => {
                    for p in params {
                        self.ty(&p.ty);
                    }
                    self.ty(ret);
                }
                Type::Record { fields, .. } => fields.iter().for_each(|fd| self.ty(&fd.ty)),
                Type::Enum { .. } | Type::Other => {}
            }
        }
        fn stmts(&mut self, stmts: &[Stmt]) {
            for s in stmts {
                each_stmt_part(s, &mut |part| match part {
                    Part::Spec(spec) => {
                        self.ty(&spec.ty);
                        if let Some(init) = &spec.init {
                            self.expr(init);
                        }
                    }
                    Part::Stmt(inner) => self.stmts(std::slice::from_ref(inner)),
                    Part::Stmts(list) => self.stmts(list),
                    Part::Expr(e) => self.expr(e),
                });
            }
        }
        fn expr(&mut self, e: &Expr) {
            match e {
                Expr::Name(n) if !self.bound.contains(n) => {
                    self.out.insert(n.clone());
                }
                Expr::Cast { ty, .. } | Expr::SizeofType(ty) | Expr::Alignof(ty) | Expr::Offsetof(ty, _) | Expr::Compound { ty, .. } => {
                    self.ty(ty);
                }
                _ => {}
            }
            for_each_child(e, &mut |c| match c {
                Child::Expr(x) => self.expr(x),
                Child::Body(b) => self.stmts(b),
            });
        }
    }
    let bound = locals_of(f);
    let mut walk = Free { bound: &bound, out: BTreeSet::new() };
    for p in &f.params {
        walk.ty(&p.ty);
    }
    walk.ty(&f.ret);
    walk.stmts(&f.body);
    walk.out
}

// ───────────────────────────── s-expr emitter ─────────────────────────────

pub(crate) struct Dump<'a> {
    /// Every name the function binds, in every mode: a local shadowing a file-level name is a value.
    bound: &'a HashSet<String>,
    /// `None` = cluster mode (names verbatim); `Some(set)` = xname mode (bound locals → `_v{n}`).
    locals: Option<&'a HashSet<String>>,
    map: HashMap<String, u32>,
    /// Blank the *top* function's own name to `_fn` exactly once (xname mode).
    blanked: bool,
    /// Node-emit count — the cross-name "substance" gate.
    pub(crate) count: usize,
    /// `Some` while writing the Type-3 lines: every sub-expression worth a unit of its own, with the
    /// index of the line it sits in — [`AnalyzedFn::type3_units`].
    units: Option<Vec<(u32, String)>>,
    unit_line: u32,
    /// How many expressions enclose the one being written: `0` for a statement's own expression.
    expr_depth: u32,
    /// `Some` while writing the Type-3 lines: type reference → `_T{n}`. See [`Dump::type_name`].
    tmap: Option<HashMap<String, u32>>,
    /// Every type reference turned into a slot, in order — [`AnalyzedFn::type_refs`].
    type_refs: Vec<String>,
    /// Use-site mode: the definition whose use is being written. A name naming it becomes the anchor
    /// `_t0`, and a designated initializer of a value of it marks the members it sets `#name#` for the
    /// profile to number.
    anchor: Option<&'a str>,
    /// Hunt mode: the names that count as anchors. Every one the walk meets is kept in `hunted`.
    known: Option<&'a HashSet<&'a str>>,
    hunted: Vec<String>,
    /// Hunt mode: the members called through (`.name`), for the co-occurrence channel of the `use` lens.
    calls: BTreeSet<String>,
}

impl<'a> Dump<'a> {
    pub(crate) fn new(bound: &'a HashSet<String>, locals: Option<&'a HashSet<String>>) -> Self {
        Self {
            bound,
            locals,
            map: HashMap::new(),
            blanked: false,
            count: 0,
            units: None,
            unit_line: 0,
            expr_depth: 0,
            tmap: None,
            type_refs: Vec::new(),
            anchor: None,
            known: None,
            hunted: Vec::new(),
            calls: BTreeSet::new(),
        }
    }

    /// A dump that notes which of `known` a line mentions, and the members it calls through.
    fn hunting(bound: &'a HashSet<String>, known: &'a HashSet<&'a str>) -> Self {
        Self { known: Some(known), tmap: Some(HashMap::new()), ..Self::new(bound, Some(bound)) }
    }

    /// A dump that writes a line as a use site of `anchor`.
    fn anchoring(bound: &'a HashSet<String>, anchor: &'a str) -> Self {
        Self { anchor: Some(anchor), tmap: Some(HashMap::new()), ..Self::new(bound, Some(bound)) }
    }

    /// A reference to a file-level name: noted when it is hunted for, and whether it is the anchor.
    fn mention(&mut self, name: &str) -> bool {
        if let Some(known) = self.known {
            if known.contains(name) && !self.hunted.iter().any(|h| h == name) {
                self.hunted.push(name.to_owned());
            }
        }
        self.anchor == Some(name)
    }

    fn rename(&mut self, name: &str) -> String {
        dup_defs_core::alpha_rename(&mut self.map, self.locals, name)
    }

    pub(crate) fn node(&mut self, tag: &str, fields: &[String]) -> String {
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

    fn list(&mut self, tag: &str, items: &[String]) -> String {
        let joined = items.join(", ");
        self.node(tag, &[joined])
    }

    /// A type's name: its `_T{n}` slot while writing the Type-3 lines, as spelled otherwise.
    ///
    /// 🔴 Only the Type-3 lines slot types — the Rust and Go frontends' rule, for their reason. A C
    /// type picks the code that runs: integer or floating arithmetic, the layout a member is read at,
    /// how far a pointer steps. Two bodies differing only in their types may not be one definition, so
    /// the exact pass keeps them apart; the near-copy pass reads a type as vocabulary like any other.
    fn type_name(&mut self, name: &str) -> String {
        // The anchor keeps its own slot in type position too, however it is tagged (`struct foo`).
        let bare = name.rsplit(' ').next().unwrap_or(name);
        if self.mention(bare) {
            return "_t0".to_owned();
        }
        let Some(tmap) = self.tmap.as_mut() else { return quoted(name) };
        self.type_refs.push(name.to_owned());
        dup_defs_core::type_slot(tmap, name)
    }

    /// A binding site: `Bind(_vN)`.
    fn bind(&mut self, name: &str) -> String {
        let renamed = self.rename(name);
        self.node("Bind", &[quoted(&renamed)])
    }

    /// A name read as a value.
    fn ident(&mut self, name: &str) -> String {
        if !self.bound.contains(name) && self.mention(name) {
            return self.node("Path", &[quoted("_t0")]);
        }
        let renamed = self.rename(name);
        self.node("Path", &[quoted(&renamed)])
    }

    // ────── functions ──────

    /// `is_top` blanks the definition's own name in xname mode.
    pub(crate) fn func(&mut self, name: &str, f: &Func, with_body: bool, is_top: bool) -> String {
        let name = if is_top && self.locals.is_some() && !self.blanked {
            self.blanked = true;
            "_fn".to_owned()
        } else {
            self.rename(name)
        };
        let params = self.params(&f.params, f.variadic);
        let ret = self.ty(&f.ret);
        let body = if with_body { self.block(&f.body) } else { String::new() };
        self.node("Func", &[quoted(&name), params, ret, body])
    }

    /// A parameter list, one `Param` per parameter; `...` is a last, nameless one.
    fn params(&mut self, params: &[Param], variadic: bool) -> String {
        let mut items = Vec::new();
        for p in params {
            let ty = self.ty(&p.ty);
            let bound = p.name.as_deref().map_or_else(String::new, |n| self.bind(n));
            items.push(self.node("Param", &[bound, ty]));
        }
        if variadic {
            let rest = self.node("TyVariadic", &[]);
            items.push(self.node("Param", &[String::new(), rest]));
        }
        self.list("Params", &items)
    }

    /// The header of the Type-3 lines, in the language's own order: `Ty(_T0, ) _fn(_v0 Ty(_T1, ))`.
    ///
    /// 🔴 Not the `Func(…)` node the stream opens with. The Type-3 pass reads every function's first
    /// line as context and weighs it like any other; an s-expr header spells the parse tree's grammar
    /// (`Params`, `Param`, `Bind`), which only headers spell, so every one of those tokens was rare and
    /// the header outweighed the body of every short function (the Rust frontend's finding).
    fn compact_header(&mut self, f: &Func) -> String {
        let ret = self.ty(&f.ret);
        let mut items = Vec::new();
        for p in &f.params {
            let ty = self.ty(&p.ty);
            match &p.name {
                Some(n) => {
                    let n = self.rename(n);
                    items.push(format!("{n} {ty}"));
                }
                None => items.push(ty),
            }
        }
        if f.variadic {
            items.push("...".to_owned());
        }
        let mut out = String::new();
        let _ = write!(out, "{ret} _fn({})", items.join(", "));
        out
    }

    // ────── statements / blocks ──────

    pub(crate) fn block(&mut self, stmts: &[Stmt]) -> String {
        let items: Vec<String> = stmts.iter().map(|s| self.stmt(s)).collect();
        self.list("Block", &items)
    }

    /// One declarator: `Let(PatType(Bind(x), ty), init)`.
    fn spec(&mut self, spec: &Spec) -> String {
        let bound = self.bind(&spec.name);
        let ty = self.ty(&spec.ty);
        let pat = self.node("PatType", &[bound, ty]);
        let init = spec.init.as_ref().map_or_else(String::new, |e| self.expr(e));
        self.node("Let", &[pat, init])
    }

    pub(crate) fn stmt(&mut self, s: &Stmt) -> String {
        match s {
            Stmt::Decl(specs) => {
                let mut items: Vec<String> = specs.iter().map(|spec| self.spec(spec)).collect();
                if items.len() == 1 {
                    items.pop().unwrap_or_default()
                } else {
                    self.list("Decls", &items)
                }
            }
            Stmt::NestedType => self.node("NestedItem", &[]),
            Stmt::Assign { op, lhs, rhs } => {
                let left = self.expr(lhs);
                let right = self.expr(rhs);
                if op == "=" {
                    self.node("Assign", &[left, right])
                } else {
                    self.node("Bin", &[left, quoted(op), right])
                }
            }
            Stmt::IncDec { inc, target } => {
                let e = self.expr(target);
                self.node(if *inc { "Inc" } else { "Dec" }, &[e])
            }
            Stmt::Expr(e) => {
                let e = self.expr(e);
                self.node("ExprStmt", &[e])
            }
            Stmt::Return(value) => {
                let value = value.as_ref().map_or_else(String::new, |v| self.expr(v));
                self.node("Return", &[value])
            }
            Stmt::Jump { kind, label } => {
                let tag = match kind {
                    Jump::Break => "Break",
                    Jump::Continue => "Continue",
                    Jump::Goto => "Goto",
                };
                let label = label.as_deref().map_or_else(String::new, quoted);
                self.node(tag, &[label])
            }
            Stmt::If { cond, then, els } => {
                let cond = self.expr(cond);
                let then = self.block(then);
                let els = match els.as_deref() {
                    Some(Else::If(alt)) => self.stmt(alt),
                    Some(Else::Block(b)) => self.block(b),
                    None => String::new(),
                };
                self.node("If", &[cond, then, els])
            }
            Stmt::While { cond, body } => {
                let cond = self.expr(cond);
                let body = self.block(body);
                self.node("While", &[cond, body])
            }
            Stmt::DoWhile { body, cond } => {
                let body = self.block(body);
                let cond = self.expr(cond);
                self.node("DoWhile", &[body, cond])
            }
            Stmt::For { init, cond, update, body } => {
                let (init, cond, update) = self.for_parts(init.as_deref(), cond.as_ref(), update.as_deref());
                let body = self.block(body);
                self.node("ForC", &[init, cond, update, body])
            }
            Stmt::Switch { value, cases } => {
                let value = self.expr(value);
                let mut items = vec![value];
                for case in cases {
                    let item = self.case(case);
                    items.push(item);
                }
                self.node("Switch", &items)
            }
            Stmt::Block(b) => {
                let body = self.block(b);
                self.node("BlockExpr", &[body])
            }
            Stmt::Labeled { label, stmt } => {
                let label = quoted(label);
                let inner = stmt.as_deref().map_or_else(String::new, |s| self.stmt(s));
                self.node("Labeled", &[label, inner])
            }
            Stmt::PreIf { conds, branches } => {
                let items: Vec<String> = conds
                    .iter()
                    .zip(branches)
                    .map(|(cond, body)| {
                        let body = self.block(body);
                        self.node("Branch", &[quoted(cond), body])
                    })
                    .collect();
                self.list("PreIf", &items)
            }
            Stmt::MacroLoop { head, body } => {
                let head = self.expr(head);
                let body = self.block(body);
                self.node("MacroLoop", &[head, body])
            }
            Stmt::Asm => self.node("Asm", &[]),
            Stmt::Empty => self.node("Empty", &[]),
            Stmt::Unknown(kind) => self.node(&format!("Unknown_{kind}"), &[]),
        }
    }

    fn case(&mut self, case: &Case) -> String {
        let body = self.block(&case.body);
        match &case.value {
            Some(v) => {
                let v = self.expr(v);
                self.node("Case", &[v, body])
            }
            None => self.node("Default", &[body]),
        }
    }

    /// `for (init; cond; update)` → its three parts, each possibly empty.
    fn for_parts(&mut self, init: Option<&Stmt>, cond: Option<&Expr>, update: Option<&Stmt>) -> (String, String, String) {
        let init = init.map_or_else(String::new, |i| self.stmt(i));
        let cond = cond.map_or_else(String::new, |c| self.expr(c));
        let update = update.map_or_else(String::new, |u| self.stmt(u));
        (init, cond, update)
    }

    // ────── compound-statement headers ──────
    //
    // A compound statement's own line — what it tests, its body elided. Shared by the statement
    // stream and the use-site walk, which must write a header the same way.

    /// The header of a loop, a `switch` or a macro loop; `None` for any other statement.
    fn header(&mut self, s: &Stmt) -> Option<String> {
        Some(match s {
            Stmt::While { cond, .. } => {
                let cond = self.expr(cond);
                self.node("While", &[cond])
            }
            Stmt::DoWhile { cond, .. } => {
                let cond = self.expr(cond);
                self.node("DoWhile", &[cond])
            }
            Stmt::For { init, cond, update, .. } => {
                let (init, cond, update) = self.for_parts(init.as_deref(), cond.as_ref(), update.as_deref());
                self.node("ForC", &[init, cond, update])
            }
            Stmt::Switch { value, .. } => {
                let value = self.expr(value);
                self.node("Switch", &[value])
            }
            Stmt::MacroLoop { head, .. } => {
                let head = self.expr(head);
                self.node("MacroLoop", &[head])
            }
            _ => return None,
        })
    }

    fn case_head(&mut self, case: &Case) -> String {
        match &case.value {
            Some(v) => {
                let v = self.expr(v);
                self.node("Case", &[v])
            }
            None => self.node("Default", &[]),
        }
    }

    // ────── expressions ──────

    /// An expression — and, when units are being collected, kept as one if it is a proper part of
    /// its line: the statement's own expression is the line, already a unit. A wrapper (parentheses,
    /// `&`) is not a level of its own; what it wraps is the part.
    pub(crate) fn expr(&mut self, e: &Expr) -> String {
        if is_wrapper(e) {
            return self.expr_inner(e);
        }
        self.expr_depth += 1;
        let out = self.expr_inner(e);
        self.expr_depth -= 1;
        if self.expr_depth > 0 && is_unit(e) {
            if let Some(units) = self.units.as_mut() {
                units.push((self.unit_line, out.clone()));
            }
        }
        out
    }

    fn expr_inner(&mut self, e: &Expr) -> String {
        match e {
            Expr::Name(name) => self.ident(name),
            Expr::Lit(tag, text) => self.node(tag, &[quoted(text)]),
            Expr::Bool(b) => self.node("Bool", &[quoted(if *b { "true" } else { "false" })]),
            Expr::Null => self.node("Nil", &[]),
            Expr::Binary { op, l, r } => {
                let l = self.expr(l);
                let r = self.expr(r);
                self.node("Bin", &[l, quoted(op), r])
            }
            Expr::Unary { op, e } => {
                let inner = self.expr(e);
                if op == "&" {
                    self.node("Ref", &[inner])
                } else {
                    self.node("Unary", &[quoted(op), inner])
                }
            }
            Expr::Update { inc, prefix, e } => {
                let inner = self.expr(e);
                let tag = match (inc, prefix) {
                    (true, false) => "Inc",
                    (false, false) => "Dec",
                    (true, true) => "PreInc",
                    (false, true) => "PreDec",
                };
                self.node(tag, &[inner])
            }
            Expr::Assign { op, l, r } => {
                let l = self.expr(l);
                let r = self.expr(r);
                self.node("AssignExpr", &[l, quoted(op), r])
            }
            Expr::Cond { c, t, f } => {
                let c = self.expr(c);
                let t = t.as_deref().map_or_else(String::new, |t| self.expr(t));
                let f = self.expr(f);
                self.node("Cond", &[c, t, f])
            }
            Expr::Comma(l, r) => {
                let l = self.expr(l);
                let r = self.expr(r);
                self.node("Seq", &[l, r])
            }
            Expr::Paren(inner) => self.expr(inner),
            Expr::Call { f, args } => self.call(f, args),
            Expr::Field { e, field } => {
                let base = self.expr(e);
                self.node("Field", &[base, quoted(field)])
            }
            Expr::Index { e, i } => {
                let base = self.expr(e);
                let idx = self.expr(i);
                self.node("Index", &[base, idx])
            }
            Expr::Cast { ty, e } => {
                let e = self.expr(e);
                let t = self.ty(ty);
                self.node("Cast", &[e, t])
            }
            // `sizeof(x)` of a local spells the local, whichever way the parser read the parentheses.
            Expr::SizeofType(Type::Named(n)) if self.bound.contains(n) => {
                let inner = self.ident(n);
                self.node("SizeOf", &[inner])
            }
            Expr::SizeofType(ty) | Expr::Alignof(ty) => {
                let tag = if matches!(e, Expr::Alignof(_)) { "AlignOf" } else { "SizeOf" };
                let t = self.ty(ty);
                self.node(tag, &[t])
            }
            Expr::SizeofValue(inner) => {
                let inner = self.expr(inner);
                self.node("SizeOf", &[inner])
            }
            Expr::Offsetof(ty, member) => {
                let t = self.ty(ty);
                self.node("OffsetOf", &[t, quoted(member)])
            }
            Expr::Compound { ty, elems } => {
                let t = self.ty(ty);
                // A literal of the anchor sets its surface: the members it names are marked for the
                // profile to number.
                let on_anchor = t.starts_with("Ty(_t0");
                let mut items = vec![t];
                items.extend(self.elements(elems, on_anchor));
                self.node("Composite", &items)
            }
            Expr::InitList(elems) => {
                let mut items = vec![String::new()];
                items.extend(self.elements(elems, false));
                self.node("Composite", &items)
            }
            Expr::GnuBlock(body) => {
                let body = self.block(body);
                self.node("StmtExpr", &[body])
            }
            Expr::Generic(inner) => {
                let inner = self.expr(inner);
                self.node("Generic", &[inner])
            }
            Expr::Asm => self.node("Asm", &[]),
            Expr::Unknown(kind) => self.node(&format!("Unknown_{kind}"), &[]),
        }
    }

    /// A call: `Method(receiver, 'name', args)` when it calls through a member — C's method call — and
    /// `Call(callee, args)` otherwise.
    fn call(&mut self, f: &Expr, args: &[Expr]) -> String {
        let items: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
        if let Expr::Field { e, field } = f {
            let recv = self.expr(e);
            if self.known.is_some() {
                self.calls.insert(format!(".{field}"));
            }
            let mut all = vec![recv, quoted(field)];
            all.extend(items);
            return self.node("Method", &all);
        }
        let callee = self.expr(f);
        let mut all = vec![callee];
        all.extend(items);
        self.node("Call", &all)
    }

    /// A brace initializer's elements: `FieldVal('name', v)` for a designated member, its name kept as
    /// spelled — the Rust and Go frontends' struct-literal field — and `KV(index, v)` for a designated
    /// element. The two differ in what a varying key means: another member is another field
    /// (reflection), another index is just another value.
    fn elements(&mut self, elems: &[Elem], on_anchor: bool) -> Vec<String> {
        elems
            .iter()
            .map(|el| match el {
                Elem::Field(name, value) => {
                    let key = if on_anchor { quoted(&format!("#{name}#")) } else { quoted(name) };
                    let value = self.expr(value);
                    self.node("FieldVal", &[key, value])
                }
                Elem::Index(index, value) => {
                    let index = self.expr(index);
                    let value = self.expr(value);
                    self.node("KV", &[index, value])
                }
                Elem::Plain(value) => self.expr(value),
            })
            .collect()
    }

    // ────── types (summarized) ──────

    pub(crate) fn ty(&mut self, t: &Type) -> String {
        match t {
            Type::Named(name) => {
                let name = self.type_name(name);
                self.node("Ty", &[name, String::new()])
            }
            Type::Ptr(inner) => {
                let inner = self.ty(inner);
                self.node("TyPtr", &[inner])
            }
            Type::Array(inner, size) => {
                let inner = self.ty(inner);
                let size = size.as_deref().map_or_else(String::new, |s| self.expr(s));
                self.node("TyArray", &[inner, size])
            }
            Type::Func(params, variadic, ret) => {
                let params = self.params(params, *variadic);
                let ret = self.ty(ret);
                self.node("TyFn", &[params, ret])
            }
            Type::Record { kind, name, fields } => {
                let name = name.as_deref().map_or_else(String::new, |n| self.type_name(&format!("{kind} {n}")));
                let fields = record_fields(self, fields);
                self.node("TyRecord", &[quoted(kind), name, fields])
            }
            Type::Enum { name, items } => {
                let name = name.as_deref().map_or_else(String::new, |n| self.type_name(&format!("enum {n}")));
                let items = enumerators(self, items);
                self.node("TyEnum", &[name, items])
            }
            Type::Other => self.node("Ty_Other", &[]),
        }
    }
}

/// Whether an expression is a Type-3 unit of its own: a node with structure — a call, a member read,
/// an operator application, a cast, an initializer with elements — not a bare name or a literal, which
/// carry no agreement by themselves. The rule every frontend keeps.
fn is_unit(e: &Expr) -> bool {
    match e {
        Expr::Call { .. }
        | Expr::Index { .. }
        | Expr::Field { .. }
        | Expr::Binary { .. }
        | Expr::Unary { .. }
        | Expr::Update { .. }
        | Expr::Assign { .. }
        | Expr::Cond { .. }
        | Expr::Cast { .. }
        | Expr::SizeofValue(_)
        | Expr::GnuBlock(_) => true,
        Expr::Compound { elems, .. } | Expr::InitList(elems) => !elems.is_empty(),
        _ => false,
    }
}

/// Syntax around an expression that is not a part of its own: what it wraps is the part.
fn is_wrapper(e: &Expr) -> bool {
    match e {
        Expr::Paren(_) => true,
        Expr::Unary { op, .. } => op == "&",
        _ => false,
    }
}

// ───────────────────────────── declaration canonicals ─────────────────────────────

/// A record's members, one per declarator; an anonymous member is `Embed(type)`, a bit-field carries
/// its width.
fn record_fields(d: &mut Dump<'_>, fields: &[FieldDecl]) -> String {
    let mut items = Vec::new();
    for decl in fields {
        let ty = d.ty(&decl.ty);
        let item = match (&decl.name, &decl.bits) {
            (None, _) => d.node("Embed", &[ty]),
            (Some(name), None) => d.node("Field", &[quoted(name), ty]),
            (Some(name), Some(bits)) => {
                let bits = d.expr(bits);
                d.node("BitField", &[quoted(name), ty, bits])
            }
        };
        items.push(item);
    }
    d.list("Named", &items)
}

fn enumerators(d: &mut Dump<'_>, items: &[(String, Option<Expr>)]) -> String {
    let items: Vec<String> = items
        .iter()
        .map(|(name, value)| {
            let value = value.as_ref().map_or_else(String::new, |v| d.expr(v));
            d.node("Variant", &[quoted(name), value])
        })
        .collect();
    d.list("Variants", &items)
}

/// Names-preserved structural canonical of a `struct` / `union` — the cluster pass's input for
/// `classes` and `interfaces`.
#[must_use]
pub(crate) fn record_canon(kind: &str, name: &str, fields: &[FieldDecl]) -> String {
    let bound = HashSet::new();
    let mut d = Dump::new(&bound, None);
    let fields = record_fields(&mut d, fields);
    d.node(if kind == "union" { "Union" } else { "Struct" }, &[quoted(name), fields])
}

/// Names-preserved structural canonical of an `enum` — the cluster pass's input for `classes`.
#[must_use]
pub(crate) fn enum_canon(name: &str, items: &[(String, Option<Expr>)]) -> String {
    let bound = HashSet::new();
    let mut d = Dump::new(&bound, None);
    let items = enumerators(&mut d, items);
    d.node("Enum", &[quoted(name), items])
}

// ───────────────────────────── function analysis ─────────────────────────────

/// Every canonical form of one function.
///
/// The cluster canonical and the statement stream read the body as written: the name-gated pass
/// compares what is spelled, and converge reads the stream to say where two definitions stop agreeing
/// — a rewritten line would point the reader at code that is not there. The Type-3 lines read the
/// normal form that keeps every statement, the exact canonical the one that also inlines single-use
/// temporaries ([`crate::normal`]).
pub(crate) fn analyze(name: &str, f: &Func, facets: bool) -> AnalyzedFn {
    let bound = locals_of(f);
    let cluster = Dump::new(&bound, None).func(name, f, true, true);

    // The statement stream keeps the definition as written: converge reads it to say where two
    // definitions stop agreeing. A run that never reads facets skips the whole walk.
    let statements = if facets {
        let mut dump = Dump::new(&bound, Some(&bound));
        // The definition's own header opens the stream at depth 0 and the body starts at depth 1 — the
        // contract every frontend agrees on.
        let head = dump.func(name, f, false, true);
        let mut walk = StreamWalk { dump: &mut dump, out: Vec::new() };
        walk.push(0, head);
        walk.stmts(&f.body, 1);
        walk.out
    } else {
        Vec::new()
    };

    let statement_form = crate::normal::normal_form(f, false);
    let (type3_lines, type3_units, type_refs) = {
        let bound = locals_of(&statement_form);
        let mut dump = Dump::new(&bound, Some(&bound));
        dump.tmap = Some(HashMap::new());
        dump.units = Some(Vec::new());
        dump.unit_line = UNFILED;
        let head = dump.compact_header(&statement_form);
        let mut walk = StreamWalk { dump: &mut dump, out: Vec::new() };
        walk.push(0, head);
        walk.stmts(&statement_form.body, 1);
        let lines = walk.out.into_iter().map(|s| s.line).collect();
        (lines, dump.units.unwrap_or_default(), dump.type_refs)
    };

    let exact_form = crate::normal::normal_form(&statement_form, true);
    let exact_bound = locals_of(&exact_form);
    let mut xd = Dump::new(&exact_bound, Some(&exact_bound));
    let xname = xd.func(name, &exact_form, true, true);
    let size = xd.count;
    AnalyzedFn { cluster_canonical: cluster, xname_canonical: xname, type3_lines, type_refs, type3_units, statements, size }
}

// ───────────────────────────── statement stream ─────────────────────────────

/// One line per statement at **every** nesting level, in source order — the engine's
/// [`Facets::statements`](dup_defs_core::Facets::statements), and with the types slotted, the Type-3
/// lines.
///
/// A compound statement contributes a **header** naming what it tests (`If(cond)`, `While(cond)`,
/// `Switch(value)`, `ForC(init, cond, update)`) with its body one level deeper — the shape every other
/// frontend produces, so the languages' streams compare. A preprocessor conditional is a header per
/// branch (`PreIf('ifdef X')`), its statements under it.
struct StreamWalk<'a, 'b> {
    dump: &'b mut Dump<'a>,
    out: Vec<Statement>,
}

impl StreamWalk<'_, '_> {
    /// A line is rendered whole before it is pushed, so the units still unfiled when it arrives are
    /// exactly its own.
    fn push(&mut self, depth: u16, line: String) {
        let at = u32::try_from(self.out.len()).unwrap_or(UNFILED - 1);
        if let Some(units) = self.dump.units.as_mut() {
            for unit in units.iter_mut().rev().take_while(|u| u.0 == UNFILED) {
                unit.0 = at;
            }
        }
        self.out.push(Statement { line, depth });
    }

    fn stmts(&mut self, stmts: &[Stmt], depth: u16) {
        for s in stmts {
            self.stmt(s, depth);
        }
    }

    fn stmt(&mut self, s: &Stmt, depth: u16) {
        match s {
            Stmt::If { .. } => self.if_chain(s, depth),
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } | Stmt::MacroLoop { body, .. } => {
                if let Some(head) = self.dump.header(s) {
                    self.push(depth, head);
                }
                self.stmts(body, depth + 1);
            }
            Stmt::Switch { cases, .. } => {
                if let Some(head) = self.dump.header(s) {
                    self.push(depth, head);
                }
                for case in cases {
                    let head = self.dump.case_head(case);
                    self.push(depth + 1, head);
                    self.stmts(&case.body, depth + 2);
                }
            }
            Stmt::Block(b) => {
                let head = self.dump.node("Block", &[]);
                self.push(depth, head);
                self.stmts(b, depth + 1);
            }
            Stmt::PreIf { conds, branches } => {
                for (cond, body) in conds.iter().zip(branches) {
                    let head = self.dump.node("PreIf", &[quoted(cond)]);
                    self.push(depth, head);
                    self.stmts(body, depth + 1);
                }
            }
            // A label names a statement; the statement is the step.
            Stmt::Labeled { stmt, .. } => {
                if let Some(inner) = stmt {
                    self.stmt(inner, depth);
                }
            }
            Stmt::Empty => {}
            _ => {
                let line = self.dump.stmt(s);
                self.push(depth, line);
            }
        }
    }

    /// `if` / `else if` / `else`: an `else if` is a sibling of its `else`, not a level deeper — what
    /// the source means, and what Python's `elif` and the other frontends' chains render as.
    fn if_chain(&mut self, s: &Stmt, depth: u16) {
        let Stmt::If { cond, then, els } = s else { return };
        let cond = self.dump.expr(cond);
        let head = self.dump.node("If", &[cond]);
        self.push(depth, head);
        self.stmts(then, depth + 1);
        if let Some(els) = els.as_deref() {
            let head = self.dump.node("Else", &[]);
            self.push(depth, head);
            match els {
                Else::If(alt) => self.if_chain(alt, depth),
                Else::Block(b) => self.stmts(b, depth + 1),
            }
        }
    }
}

// ───────────────────────────── scope / use sites ─────────────────────────────

/// The body canonicalized with every name the **file** introduced erased — the widest rung of the
/// erasure ladder, which the `scope` lens projects as a single fact.
#[must_use]
pub(crate) fn scope_canonical(f: &Func, file_names: &HashSet<String>) -> String {
    let mut locals = locals_of(f);
    locals.extend(file_names.iter().cloned());
    Dump::new(&locals, Some(&locals)).func("_fn", f, true, true)
}

/// What a use-site walk reads off one file: its functions, and its file-scope objects' initializers.
pub(crate) struct FileItems {
    pub funcs: Vec<Func>,
    pub initializers: Vec<Expr>,
}

/// Every use site in one file — the C half of the `use` lens ([`dup_defs_core::uses`]): each statement
/// that mentions a known name, written with that name as the anchor. A site is the statement's own
/// line — a compound statement's header (`If(cond)`, `Switch(value)`, a case's value), not its body,
/// whose statements are sites of their own — as the statement stream writes it, and inside a function
/// its locals rename as they do there.
pub(crate) fn collect_sites(items: &FileItems, known: &HashSet<&str>) -> Vec<Site> {
    let mut walk = UseWalk { known, out: Vec::new() };
    for f in &items.funcs {
        walk.function(f);
    }
    let empty = HashSet::new();
    for init in &items.initializers {
        walk.site(&empty, "plain", &[], |d| {
            let v = d.expr(init);
            Some(d.node("Var", &[v]))
        });
    }
    walk.out
}

struct UseWalk<'k> {
    known: &'k HashSet<&'k str>,
    out: Vec<Site>,
}

impl UseWalk<'_> {
    /// One site per known name `render` mentions, each written with that name as the anchor.
    fn site<F>(&mut self, bound: &HashSet<String>, ctx: &'static str, neighbours: &[String], render: F)
    where
        F: Fn(&mut Dump<'_>) -> Option<String>,
    {
        let mut hunt = Dump::hunting(bound, self.known);
        if render(&mut hunt).is_none() {
            return;
        }
        for name in hunt.hunted {
            let mut anchored = Dump::anchoring(bound, &name);
            if let Some(canon) = render(&mut anchored) {
                self.out.push(Site { name, canon, ctx, cooc: neighbours.to_vec() });
            }
        }
    }

    /// A function: its signature is a site when it names a known type, its body is walked under its
    /// own locals.
    fn function(&mut self, f: &Func) {
        let locals = locals_of(f);
        self.site(&locals, "plain", &[], |d| Some(d.compact_header(f)));
        self.block(&locals, &f.body, Within::default());
    }

    /// A block's statements, each with the members called through beside it — the co-occurrence
    /// channel.
    fn block(&mut self, bound: &HashSet<String>, stmts: &[Stmt], within: Within) {
        let mut hunt = Dump::hunting(bound, self.known);
        for s in stmts {
            let _ = use_line(&mut hunt, s);
        }
        let neighbours: Vec<String> = hunt.calls.into_iter().collect();
        for s in stmts {
            self.stmt(bound, s, within, &neighbours);
        }
    }

    fn stmt(&mut self, bound: &HashSet<String>, s: &Stmt, within: Within, neighbours: &[String]) {
        let plain = site_ctx(Construct::Plain, within);
        let guard = site_ctx(Construct::Guard, within);
        let looped = site_ctx(Construct::Loop, within);
        match s {
            Stmt::If { cond, then, els } => {
                self.site(bound, guard, neighbours, |d| {
                    let cond = d.expr(cond);
                    Some(d.node("If", &[cond]))
                });
                self.block(bound, then, within);
                match els.as_deref() {
                    Some(Else::If(alt)) => self.stmt(bound, alt, within, neighbours),
                    Some(Else::Block(b)) => self.block(bound, b, within),
                    None => {}
                }
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } | Stmt::MacroLoop { body, .. } => {
                self.site(bound, looped, neighbours, |d| d.header(s));
                self.block(bound, body, Within { in_loop: true, ..within });
            }
            Stmt::Switch { cases, .. } => {
                self.site(bound, guard, neighbours, |d| d.header(s));
                for case in cases {
                    self.site(bound, guard, neighbours, |d| Some(d.case_head(case)));
                    self.block(bound, &case.body, within);
                }
            }
            Stmt::Block(b) => self.block(bound, b, within),
            Stmt::PreIf { branches, .. } => {
                for b in branches {
                    self.block(bound, b, within);
                }
            }
            Stmt::Labeled { stmt, .. } => {
                if let Some(inner) = stmt {
                    self.stmt(bound, inner, within, neighbours);
                }
            }
            Stmt::Empty => {}
            _ => self.site(bound, plain, neighbours, |d| use_line(d, s)),
        }
    }
}

/// A statement's own line, as a use site sees it: a compound statement's header, any other statement
/// whole — `None` for a bare block, whose statements are walked.
fn use_line(d: &mut Dump<'_>, s: &Stmt) -> Option<String> {
    match s {
        Stmt::If { cond, .. } => {
            let cond = d.expr(cond);
            Some(d.node("If", &[cond]))
        }
        Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. } | Stmt::Switch { .. } | Stmt::MacroLoop { .. } => d.header(s),
        Stmt::Block(_) | Stmt::Empty | Stmt::PreIf { .. } => None,
        Stmt::Labeled { stmt, .. } => stmt.as_deref().and_then(|inner| use_line(d, inner)),
        _ => Some(d.stmt(s)),
    }
}
