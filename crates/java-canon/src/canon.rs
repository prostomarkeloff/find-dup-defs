//! Structural canonicalization of Java definitions, written from the owned tree ([`crate::ir`]).
//!
//! Mirrors the role `go-canon::canon` plays for Go and `c-canon::canon` for C: a compact, internally
//! consistent **s-expr** per definition that `difflib-fast` compares for name-gated similarity, and
//! that the cross-name pass `Eq`-checks once locals are alpha-renamed. The tag vocabulary is the other
//! frontends' wherever the construct is the same one (`Func`, `Params`, `Block`, `Let`, `Assign`,
//! `ExprStmt`, `If`, `While`, `ForC`, `For`, `Switch`, `Case`, `Try`, `Catch`, `Throw`, `Call`,
//! `Method`, `Field`, `Bin`, …), so the engine's s-expr readers — converge's vocabulary split, the
//! unit counter — read Java the way they already read the others. Two modes, driven by [`Dump::locals`]:
//!
//! * **cluster** (`locals = None`) — identifiers pass through verbatim; the names-preserved
//!   canonical the name-gated pass clusters on.
//! * **xname** (`locals = Some(set)`) — value bindings (parameters, locals, loop variables, catch
//!   bindings, lambda parameters) are renumbered `_v{n}` by first occurrence and the top definition's
//!   own name is blanked to `_fn`.
//!
//! Java's `this` is read as `Expr::Name("this")` and written `self` wherever the body names it — the
//! analog of Go's receiver and the Rust frontend dropping `&self`. `super` reads `Path('super')`.
//! A class name — `new Foo()`, `Foo.m()`, `Foo.field`, `Foo.class` — resolves through the file's
//! import table (an alias `Foo` spells its dotted path) and, in the Type-3 lines, slots `_T{n}` like
//! any other type.
//!
//! The canonicals are written from the owned tree rather than from tree-sitter's because the exact
//! and Type-3 canonicals read the body's normal form ([`crate::normal`]), which rewrites it.
#![allow(
    clippy::too_many_lines,  // the stmt/expr/type matches enumerate the tree's shapes; splitting scatters one
    clippy::match_same_arms  // distinct shapes intentionally share an emission for clarity
)]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use dup_defs_core::uses::{site_ctx, Construct, Site, Within};
use dup_defs_core::{quoted, type_slot, AnalyzedFn, Statement, UNFILED};
use tree_sitter::Node;

use crate::ir::{lower_class, ClassDecl, ClassKind, ClassMember, Else, Expr, Func, LambdaBody, Param, Stmt, Type};
use crate::tree::{field, kids, text};

// ───────────────────────────── traversal ─────────────────────────────

/// What an expression holds that a walk may want: its sub-expressions, and the body of a lambda.
pub(crate) enum Child<'e> {
    Expr(&'e Expr),
    Body(&'e [Stmt]),
}

/// Every direct child of an expression, in source order. A lambda's body is a [`Child::Body`] when it
/// is a block; a switch expression's arms contribute their conditions and values.
pub(crate) fn for_each_child<'e>(e: &'e Expr, f: &mut dyn FnMut(Child<'e>)) {
    match e {
        Expr::Binary { l, r, .. } | Expr::AssignExpr { l, r, .. } => {
            f(Child::Expr(l));
            f(Child::Expr(r));
        }
        Expr::Unary { e, .. } | Expr::Update { e, .. } | Expr::Paren(e) | Expr::Cast { e, .. } | Expr::Throw(e) => f(Child::Expr(e)),
        Expr::Cond { c, t, f: fb } => {
            f(Child::Expr(c));
            f(Child::Expr(t));
            f(Child::Expr(fb));
        }
        Expr::Call { f: callee, args } => {
            f(Child::Expr(callee));
            for a in args {
                f(Child::Expr(a));
            }
        }
        Expr::Method { recv, args, .. } => {
            f(Child::Expr(recv));
            for a in args {
                f(Child::Expr(a));
            }
        }
        Expr::StaticCall { args, .. } | Expr::New { args, .. } | Expr::Super { args } => {
            for a in args {
                f(Child::Expr(a));
            }
        }
        Expr::Field { base, .. } => f(Child::Expr(base)),
        Expr::Index { base, i } => {
            f(Child::Expr(base));
            f(Child::Expr(i));
        }
        Expr::InstanceOf { e, .. } => f(Child::Expr(e)),
        Expr::Array { elems } => {
            for el in elems {
                f(Child::Expr(el));
            }
        }
        Expr::Lambda { body, .. } => match body {
            LambdaBody::Block(b) => f(Child::Body(b)),
            LambdaBody::Expr(e) => f(Child::Expr(e)),
        },
        Expr::MethodRef { target, .. } => f(Child::Expr(target)),
        Expr::Switch { value, arms } => {
            f(Child::Expr(value));
            for arm in arms {
                for c in &arm.conds {
                    f(Child::Expr(c));
                }
                f(Child::Expr(&arm.value));
            }
        }
        Expr::StmtBlock(b) => f(Child::Body(b)),
        Expr::Name(_)
        | Expr::Lit(..)
        | Expr::Bool(_)
        | Expr::Null
        | Expr::StaticField { .. }
        | Expr::ClassConst { .. }
        | Expr::Unknown(_) => {}
    }
}

// ───────────────────────────── s-expr emitter ─────────────────────────────

pub(crate) struct Dump<'a> {
    /// What each name the file imports binds — a class name resolves through this to its dotted path.
    imports: &'a HashMap<String, Arc<str>>,
    /// Every name the callable binds, in every mode: a local shadowing a class name is a value.
    bound: &'a HashSet<String>,
    /// `None` = cluster mode (names verbatim); `Some(set)` = xname mode (bound locals → `_v{n}`).
    locals: Option<&'a HashSet<String>>,
    map: HashMap<String, u32>,
    /// Blank the *top* callable's own name to `_fn` exactly once (xname mode).
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
    /// Type parameters the definition declares: slotted like any type, but bound, not vocabulary.
    declared_types: HashSet<String>,
    /// Use-site mode: the definition whose use is being written. A class name naming it becomes the
    /// anchor `_t0` — whichever package it was reached through.
    anchor: Option<&'a str>,
    /// Hunt mode: the names that count as anchors. Every one the walk meets is kept in `hunted`.
    known: Option<&'a HashSet<&'a str>>,
    hunted: Vec<String>,
    /// Hunt mode: the methods called (`.name`), for the co-occurrence channel of the `use` lens.
    calls: BTreeSet<String>,
}

impl<'a> Dump<'a> {
    pub(crate) fn new(
        imports: &'a HashMap<String, Arc<str>>,
        bound: &'a HashSet<String>,
        locals: Option<&'a HashSet<String>>,
    ) -> Self {
        Self {
            imports,
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
            declared_types: HashSet::new(),
            anchor: None,
            known: None,
            hunted: Vec::new(),
            calls: BTreeSet::new(),
        }
    }

    /// A dump that notes which of `known` a line mentions, and the methods it calls.
    fn hunting(scope: &Scope<'a>, known: &'a HashSet<&'a str>) -> Self {
        Self { known: Some(known), tmap: Some(HashMap::new()), ..scope.dump() }
    }

    /// A dump that writes a line as a use site of `anchor`.
    fn anchoring(scope: &Scope<'a>, anchor: &'a str) -> Self {
        Self { anchor: Some(anchor), tmap: Some(HashMap::new()), ..scope.dump() }
    }

    /// A reference to a class-level name: noted when it is hunted for, and whether it is the anchor.
    fn mention(&mut self, name: &str) -> bool {
        if let Some(known) = self.known {
            if known.contains(name) && !self.hunted.iter().any(|h| h == name) {
                self.hunted.push(name.to_owned());
            }
        }
        self.anchor == Some(name)
    }

    /// Resolve a class/type name through the file's imports: `Foo` → its dotted path, `a.b.C` → itself
    /// when its first segment is no alias, and any unqualified non-alias name passes through.
    fn resolve(&self, name: &str) -> String {
        let first = name.split('.').next().unwrap_or(name);
        match self.imports.get(first) {
            Some(path) if first == name => path.to_string(),
            Some(path) => format!("{path}{}", &name[first.len()..]),
            None => name.to_owned(),
        }
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

    /// A type's name: its `_T{n}` slot while writing the Type-3 lines (reported apart unless the
    /// definition declared it), its import-resolved spelling otherwise.
    ///
    /// 🔴 Only the Type-3 lines slot types — the rule of every frontend. A Java type names a class the
    /// body builds, tests against, or dispatches on, so two bodies differing only in their types may
    /// not be one definition; the exact pass keeps them apart, the near-copy pass reads a type as
    /// vocabulary like any other.
    fn type_name(&mut self, name: &str) -> String {
        let resolved = self.resolve(name);
        let bare = resolved.rsplit('.').next().unwrap_or(&resolved);
        if self.mention(bare) {
            return "_t0".to_owned();
        }
        let Some(tmap) = self.tmap.as_mut() else { return quoted(&resolved) };
        if !self.declared_types.contains(&resolved) {
            self.type_refs.push(resolved.clone());
        }
        type_slot(tmap, &resolved)
    }

    /// A binding site: `Bind(_vN)`.
    fn bind(&mut self, name: &str) -> String {
        let renamed = self.rename(name);
        self.node("Bind", &[quoted(&renamed)])
    }

    /// A name read as a value: `this` → `Path('self')`, `super` → `Path('super')`, an unbound known
    /// name → `Path('_t0')`, otherwise its (renamed) spelling.
    fn ident(&mut self, name: &str) -> String {
        if name == "this" {
            return self.node("Path", &[quoted("self")]);
        }
        if name == "super" {
            return self.node("Path", &[quoted("super")]);
        }
        if !self.bound.contains(name) && self.mention(name) {
            return self.node("Path", &[quoted("_t0")]);
        }
        let renamed = self.rename(name);
        self.node("Path", &[quoted(&renamed)])
    }

    // ────── callables ──────

    /// `is_top` blanks the definition's own name in xname mode. The body sits at slot 3.
    pub(crate) fn func(&mut self, name: &str, f: &Func, with_body: bool, is_top: bool) -> String {
        let name = if is_top && self.locals.is_some() && !self.blanked {
            self.blanked = true;
            "_fn".to_owned()
        } else {
            self.rename(name)
        };
        let params = self.params(&f.params, "Params");
        let ret = f.ret.as_ref().map_or_else(String::new, |t| self.ty(t));
        let body = if with_body { self.block(&f.body) } else { String::new() };
        self.node("Func", &[quoted(&name), params, ret, body])
    }

    /// A parameter list, one `Param` per name — a variadic parameter's type is `TyVariadic`.
    fn params(&mut self, params: &[Param], tag: &str) -> String {
        let mut items = Vec::new();
        for p in params {
            let ty = self.ty(&p.ty);
            let ty = if p.variadic { self.node("TyVariadic", &[ty]) } else { ty };
            let bound = self.bind(&p.name);
            items.push(self.node("Param", &[bound, ty]));
        }
        self.list(tag, &items)
    }

    /// The header of the Type-3 lines, in the language's own order: `Ty(_T0, ) _fn(_v0 Ty(_T1, ))` —
    /// the return type, then `_fn(params)`; a constructor (no return type) opens with `_fn(...)`.
    ///
    /// 🔴 Not the `Func(…)` node the stream opens with: the Type-3 pass reads every function's first
    /// line as context, and an s-expr header spells grammar (`Params`, `Param`, `Bind`) that only
    /// headers spell, so those tokens would outweigh every short body.
    fn compact_header(&mut self, f: &Func) -> String {
        for name in &f.type_params {
            self.declared_types.insert(name.clone());
        }
        let ret = f.ret.as_ref().map_or_else(String::new, |t| self.ty(t));
        let mut items = Vec::new();
        for p in &f.params {
            let ty = self.ty(&p.ty);
            let ty = if p.variadic { self.node("TyVariadic", &[ty]) } else { ty };
            let n = self.rename(&p.name);
            items.push(format!("{n} {ty}"));
        }
        if ret.is_empty() {
            format!("_fn({})", items.join(", "))
        } else {
            format!("{ret} _fn({})", items.join(", "))
        }
    }

    // ────── statements / blocks ──────

    pub(crate) fn block(&mut self, stmts: &[Stmt]) -> String {
        let items: Vec<String> = stmts.iter().map(|s| self.stmt(s)).collect();
        self.list("Block", &items)
    }

    pub(crate) fn stmt(&mut self, s: &Stmt) -> String {
        match s {
            Stmt::Let { ty, name, init } => {
                let bound = self.bind(name);
                let pat = match ty {
                    Some(t) => {
                        let ty = self.ty(t);
                        self.node("PatType", &[bound, ty])
                    }
                    // `var x = e` — no declared type.
                    None => bound,
                };
                let init = init.as_ref().map_or_else(String::new, |e| self.expr(e));
                self.node("Let", &[pat, init])
            }
            Stmt::Decl(items) => {
                let items: Vec<String> = items.iter().map(|s| self.stmt(s)).collect();
                if items.len() == 1 {
                    items.into_iter().next().unwrap_or_default()
                } else {
                    self.list("Decls", &items)
                }
            }
            Stmt::Assign { op, lhs, rhs } => {
                let left = self.expr(lhs);
                let right = self.expr(rhs);
                if op == "=" {
                    self.node("Assign", &[left, right])
                } else {
                    self.node("Bin", &[left, quoted(op), right])
                }
            }
            Stmt::Update { inc, prefix, e } => {
                let inner = self.expr(e);
                let tag = match (inc, prefix) {
                    (true, false) => "Inc",
                    (false, false) => "Dec",
                    (true, true) => "PreInc",
                    (false, true) => "PreDec",
                };
                self.node(tag, &[inner])
            }
            Stmt::Expr(e) => {
                let e = self.expr(e);
                self.node("ExprStmt", &[e])
            }
            Stmt::Return(value) => {
                let value = value.as_ref().map_or_else(String::new, |v| self.expr(v));
                self.node("Return", &[value])
            }
            Stmt::Throw(e) => {
                let e = self.expr(e);
                self.node("Throw", &[e])
            }
            Stmt::Assert(e) => {
                let e = self.expr(e);
                self.node("Assert", &[e])
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
                let cond = self.expr(cond);
                let body = self.block(body);
                self.node("DoWhile", &[body, cond])
            }
            Stmt::For { init, cond, update, body } => {
                let init = init.as_deref().map_or_else(String::new, |i| self.stmt(i));
                let cond = cond.as_ref().map_or_else(String::new, |c| self.expr(c));
                let update: Vec<String> = update.iter().map(|u| self.expr(u)).collect();
                let update = update.join(", ");
                let body = self.block(body);
                self.node("ForC", &[init, cond, update, body])
            }
            Stmt::ForEach { name, iter, body } => {
                let bind = self.bind(name);
                let iter = self.expr(iter);
                let body = self.block(body);
                self.node("For", &[bind, iter, body])
            }
            Stmt::Switch { value, cases } => {
                let value = self.expr(value);
                let cases: Vec<String> = cases
                    .iter()
                    .map(|case| {
                        let body = self.block(&case.body);
                        if case.values.is_empty() {
                            self.node("Default", &[body])
                        } else {
                            let values: Vec<String> = case.values.iter().map(|v| self.expr(v)).collect();
                            self.node("Case", &[values.join(", "), body])
                        }
                    })
                    .collect();
                self.node("Switch", &[value, cases.join(", ")])
            }
            Stmt::Try { resources, body, catches, finally } => {
                let body = self.block(body);
                let catches: Vec<String> = catches
                    .iter()
                    .map(|c| {
                        let types: Vec<String> = c.types.iter().map(|t| self.ty(t)).collect();
                        let types = self.list("Types", &types);
                        let bind = c.name.as_ref().map_or_else(String::new, |n| self.bind(n));
                        let body = self.block(&c.body);
                        self.node("Catch", &[types, bind, body])
                    })
                    .collect();
                let catches = catches.join(", ");
                let finally = finally.as_ref().map_or_else(String::new, |f| self.block(f));
                if resources.is_empty() {
                    self.node("Try", &[body, catches, finally])
                } else {
                    let res: Vec<String> = resources
                        .iter()
                        .map(|r| {
                            let bind = r.name.as_ref().map_or_else(String::new, |n| self.bind(n));
                            let value = self.expr(&r.value);
                            self.node("Resource", &[bind, value])
                        })
                        .collect();
                    let res = self.list("Resources", &res);
                    self.node("TryRes", &[res, body, catches, finally])
                }
            }
            Stmt::Synchronized { lock, body } => {
                let lock = self.expr(lock);
                let body = self.block(body);
                self.node("Synchronized", &[lock, body])
            }
            Stmt::Labeled { label, stmt } => {
                let inner = self.stmt(stmt);
                self.node("Labeled", &[quoted(label), inner])
            }
            Stmt::Break(label) => {
                let label = label.as_deref().map_or_else(String::new, quoted);
                self.node("Break", &[label])
            }
            Stmt::Continue(label) => {
                let label = label.as_deref().map_or_else(String::new, quoted);
                self.node("Continue", &[label])
            }
            Stmt::Block(b) => {
                let body = self.block(b);
                self.node("BlockExpr", &[body])
            }
            Stmt::NestedItem => self.node("NestedItem", &[]),
            Stmt::Unknown(kind) => self.node(&format!("Unknown_{kind}"), &[]),
        }
    }

    /// `ForC(init, cond, update)` — a `for` loop's own line, its body elided. Shared by the statement
    /// stream and the use-site walk.
    fn for_header(&mut self, init: Option<&Stmt>, cond: Option<&Expr>, update: &[Expr]) -> String {
        let init = init.map_or_else(String::new, |i| self.stmt(i));
        let cond = cond.map_or_else(String::new, |c| self.expr(c));
        let update: Vec<String> = update.iter().map(|u| self.expr(u)).collect();
        let update = update.join(", ");
        self.node("ForC", &[init, cond, update])
    }

    // ────── expressions ──────

    /// An expression — and, when units are being collected, kept as one if it is a proper part of its
    /// line: the statement's own expression is the line, already a unit. A wrapper (parentheses) is not
    /// a level of its own; what it wraps is the part.
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
                self.node("Unary", &[quoted(op), inner])
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
            Expr::AssignExpr { op, l, r } => {
                let l = self.expr(l);
                let r = self.expr(r);
                self.node("AssignExpr", &[l, quoted(op), r])
            }
            Expr::Cond { c, t, f } => {
                let c = self.expr(c);
                let t = self.expr(t);
                let f = self.expr(f);
                self.node("Cond", &[c, t, f])
            }
            Expr::Paren(inner) => self.expr(inner),
            Expr::Call { f, args } => {
                let callee = self.expr(f);
                let args: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
                let args = args.join(", ");
                self.node("Call", &[callee, args])
            }
            Expr::Method { recv, name, args } => {
                let recv = self.expr(recv);
                let args: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
                let args = args.join(", ");
                if self.known.is_some() {
                    self.calls.insert(format!(".{name}"));
                }
                self.node("Method", &[recv, quoted(name), args])
            }
            Expr::StaticCall { scope, name, args } => {
                let scope = self.type_name(scope);
                let args: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
                let args = args.join(", ");
                self.node("StaticCall", &[scope, quoted(name), args])
            }
            Expr::New { class, args } => {
                let class = self.type_name(class);
                let args: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
                let args = args.join(", ");
                self.node("New", &[class, args])
            }
            Expr::Super { args } => {
                let args: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
                let args = args.join(", ");
                self.node("Super", &[args])
            }
            Expr::Field { base, name } => {
                let base = self.expr(base);
                self.node("Field", &[base, quoted(name)])
            }
            Expr::StaticField { scope, name } => {
                let scope = self.type_name(scope);
                self.node("StaticField", &[scope, quoted(name)])
            }
            Expr::Index { base, i } => {
                let base = self.expr(base);
                let idx = self.expr(i);
                self.node("Index", &[base, idx])
            }
            Expr::Cast { ty, e } => {
                let e = self.expr(e);
                let t = self.ty(ty);
                self.node("Cast", &[e, t])
            }
            Expr::InstanceOf { e, ty, .. } => {
                let e = self.expr(e);
                let t = self.ty(ty);
                self.node("InstanceOf", &[e, t])
            }
            Expr::Array { elems } => {
                let elems: Vec<String> = elems.iter().map(|e| self.expr(e)).collect();
                let elems = elems.join(", ");
                self.node("Arr", &[elems])
            }
            Expr::Lambda { params, body } => {
                let params = self.params(params, "Params");
                let body = match body {
                    LambdaBody::Block(b) => self.block(b),
                    LambdaBody::Expr(e) => self.expr(e),
                };
                self.node("Lambda", &[params, body])
            }
            Expr::MethodRef { target, name } => {
                let target = self.expr(target);
                self.node("MethodRef", &[target, quoted(name)])
            }
            Expr::ClassConst { scope } => {
                let scope = self.type_name(scope);
                self.node("ClassConst", &[scope])
            }
            Expr::Switch { value, arms } => {
                let value = self.expr(value);
                let arms: Vec<String> = arms
                    .iter()
                    .map(|arm| {
                        let value = self.expr(&arm.value);
                        if arm.conds.is_empty() {
                            self.node("Default", &[value])
                        } else {
                            let conds: Vec<String> = arm.conds.iter().map(|c| self.expr(c)).collect();
                            self.node("Arm", &[conds.join(", "), value])
                        }
                    })
                    .collect();
                let arms = arms.join(", ");
                self.node("SwitchExpr", &[value, arms])
            }
            Expr::StmtBlock(b) => {
                let body = self.block(b);
                self.node("StmtExpr", &[body])
            }
            Expr::Throw(e) => {
                let e = self.expr(e);
                self.node("Throw", &[e])
            }
            Expr::Unknown(kind) => self.node(&format!("Unknown_{kind}"), &[]),
        }
    }

    // ────── types (summarized) ──────

    pub(crate) fn ty(&mut self, t: &Type) -> String {
        match t {
            Type::Named(name) => {
                let name = self.type_name(name);
                self.node("Ty", &[name, String::new()])
            }
            Type::Generic(name, args) => {
                let name = if name.is_empty() { String::new() } else { self.type_name(name) };
                let mut fields = vec![name];
                for a in args {
                    fields.push(self.ty(a));
                }
                self.node("Ty", &fields)
            }
            Type::Array(inner) => {
                let inner = self.ty(inner);
                self.node("TyArray", &[inner])
            }
            Type::Variadic(inner) => {
                let inner = self.ty(inner);
                self.node("TyVariadic", &[inner])
            }
            Type::Wild(bound) => match bound {
                None => self.node("TyWild", &[]),
                Some((kw, ty)) => {
                    let ty = self.ty(ty);
                    self.node("TyWild", &[quoted(kw), ty])
                }
            },
            Type::Union(parts) => {
                let parts: Vec<String> = parts.iter().map(|p| self.ty(p)).collect();
                self.node("TyUnion", &parts)
            }
            Type::Inter(parts) => {
                let parts: Vec<String> = parts.iter().map(|p| self.ty(p)).collect();
                self.node("TyInter", &parts)
            }
            Type::Other => self.node("Ty_Other", &[]),
        }
    }
}

/// Whether an expression is a Type-3 unit of its own: a node with structure — a call, a field read,
/// an operator application, a cast, a lambda, an array with elements, a switch — not a bare name, a
/// class member, or a literal, which carry no agreement by themselves. The rule every frontend keeps.
fn is_unit(e: &Expr) -> bool {
    match e {
        Expr::Call { .. }
        | Expr::Method { .. }
        | Expr::StaticCall { .. }
        | Expr::New { .. }
        | Expr::Super { .. }
        | Expr::Binary { .. }
        | Expr::Unary { .. }
        | Expr::Index { .. }
        | Expr::InstanceOf { .. }
        | Expr::Cast { .. }
        | Expr::Field { .. }
        | Expr::Lambda { .. }
        | Expr::Switch { .. } => true,
        Expr::Array { elems, .. } => !elems.is_empty(),
        _ => false,
    }
}

/// Syntax around an expression that is not a part of its own: what it wraps is the part.
fn is_wrapper(e: &Expr) -> bool {
    matches!(e, Expr::Paren(_))
}

// ───────────────────────────── declaration canonicals ─────────────────────────────

/// Names-preserved structural canonical of a class, interface, enum, record or annotation type — the
/// cluster pass's input for `classes` and `interfaces`: its header and every member, method bodies
/// included.
#[must_use]
pub(crate) fn class_canon(imports: &HashMap<String, Arc<str>>, d: &ClassDecl) -> String {
    let bound = HashSet::new();
    let mut dump = Dump::new(imports, &bound, None);
    class_canon_inner(&mut dump, d)
}

fn class_canon_inner(dump: &mut Dump<'_>, d: &ClassDecl) -> String {
    let name = quoted(&d.name);
    let g_n = format!("g{}", d.type_params);
    let mods: Vec<String> = d.mods.iter().map(|m| quoted(m)).collect();
    let mods = dump.list("Mods", &mods);
    let members: Vec<String> = d.members.iter().filter_map(|m| member_canon(dump, m)).collect();
    let members = dump.list("Members", &members);
    match d.kind {
        ClassKind::Class => {
            let extends = d.extends.as_ref().map_or_else(String::new, |t| dump.ty(t));
            let implements: Vec<String> = d.implements.iter().map(|t| dump.ty(t)).collect();
            let implements = dump.list("Implements", &implements);
            dump.node("Class", &[name, g_n, mods, extends, implements, members])
        }
        ClassKind::Interface | ClassKind::Annotation => {
            let extends: Vec<String> = d.extends_list.iter().map(|t| dump.ty(t)).collect();
            let extends = dump.list("Extends", &extends);
            dump.node("Interface", &[name, g_n, mods, extends, members])
        }
        ClassKind::Enum => {
            let implements: Vec<String> = d.implements.iter().map(|t| dump.ty(t)).collect();
            let implements = dump.list("Implements", &implements);
            dump.node("Enum", &[name, g_n, mods, implements, members])
        }
        ClassKind::Record => {
            let implements: Vec<String> = d.implements.iter().map(|t| dump.ty(t)).collect();
            let implements = dump.list("Implements", &implements);
            let components: Vec<String> = d
                .components
                .iter()
                .map(|p| {
                    let bind = dump.bind(&p.name);
                    let ty = dump.ty(&p.ty);
                    let ty = if p.variadic { dump.node("TyVariadic", &[ty]) } else { ty };
                    dump.node("Component", &[bind, ty])
                })
                .collect();
            let components = dump.list("Components", &components);
            dump.node("Record", &[name, g_n, mods, implements, components, members])
        }
    }
}

fn member_canon(dump: &mut Dump<'_>, m: &ClassMember) -> Option<String> {
    match m {
        ClassMember::Field { mods, ty, name, init, .. } => {
            let mods: Vec<String> = mods.iter().map(|m| quoted(m)).collect();
            let mods = dump.list("Mods", &mods);
            let ty = dump.ty(ty);
            let default = init.as_ref().map_or_else(String::new, |e| dump.expr(e));
            Some(dump.node("Field", &[quoted(name), mods, ty, default]))
        }
        ClassMember::Method { mods, func, abstract_ } => {
            let mods: Vec<String> = mods.iter().map(|m| quoted(m)).collect();
            let mods = dump.list("Mods", &mods);
            let f = dump.func(&func.name, func, !*abstract_, false);
            Some(dump.node("MethodDef", &[mods, f]))
        }
        ClassMember::Nested(n) => {
            let inner = class_canon_inner(dump, n);
            let tag = match n.kind {
                ClassKind::Interface | ClassKind::Annotation => "NestedInterface",
                _ => "NestedClass",
            };
            Some(dump.node(tag, &[inner]))
        }
        ClassMember::Variant { name, args } => {
            let args: Vec<String> = args.iter().map(|a| dump.expr(a)).collect();
            Some(dump.node("Variant", &[quoted(name), args.join(", ")]))
        }
        ClassMember::StaticInit => None,
    }
}

// ───────────────────────────── callable analysis ─────────────────────────────

/// Every canonical form of one callable.
///
/// The cluster canonical and the statement stream read the body as written: the name-gated pass
/// compares what is spelled, and converge reads the stream to say where two definitions stop agreeing.
/// The Type-3 lines read the normal form that keeps every statement, the exact canonical the one that
/// also inlines single-use temporaries ([`crate::normal`]).
pub(crate) fn analyze(imports: &HashMap<String, Arc<str>>, name: &str, f: &Func) -> AnalyzedFn {
    let bound = crate::ir::locals_of(f);
    let cluster = Dump::new(imports, &bound, None).func(name, f, true, true);

    let statements = {
        let mut dump = Dump::new(imports, &bound, Some(&bound));
        // The definition's own header opens the stream at depth 0 and the body starts at depth 1 — the
        // contract every frontend agrees on.
        let head = dump.func(name, f, false, true);
        let mut walk = StreamWalk { dump: &mut dump, out: Vec::new() };
        walk.push(0, head);
        walk.stmts(&f.body, 1);
        walk.out
    };

    let statement_form = crate::normal::normal_form(f, false);
    let (type3_lines, type3_units, type_refs) = {
        let bound = crate::ir::locals_of(&statement_form);
        let mut dump = Dump::new(imports, &bound, Some(&bound));
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
    let exact_bound = crate::ir::locals_of(&exact_form);
    let mut xd = Dump::new(imports, &exact_bound, Some(&exact_bound));
    let xname = xd.func(name, &exact_form, true, true);
    let size = xd.count;
    AnalyzedFn {
        cluster_canonical: cluster,
        xname_canonical: xname,
        type3_lines,
        type_refs,
        type3_units,
        statements,
        size,
    }
}

// ───────────────────────────── statement stream ─────────────────────────────

/// One line per statement at **every** nesting level, in source order — the engine's
/// [`Facets::statements`](dup_defs_core::Facets::statements), and with the types slotted, the Type-3
/// lines.
///
/// A compound statement contributes a **header** naming what it tests (`If(cond)`, `Switch(value)`,
/// `ForC(init, cond, update)`, `For(bind, iter)`, `Try()`, `Catch(types)`) with its body one level
/// deeper — the shape every other frontend produces, so the languages' streams compare.
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
            Stmt::For { init, cond, update, body } => {
                let head = self.dump.for_header(init.as_deref(), cond.as_ref(), update);
                self.push(depth, head);
                self.stmts(body, depth + 1);
            }
            Stmt::ForEach { name, iter, body } => {
                let bind = self.dump.bind(name);
                let iter = self.dump.expr(iter);
                let head = self.dump.node("For", &[bind, iter]);
                self.push(depth, head);
                self.stmts(body, depth + 1);
            }
            Stmt::While { cond, body } => {
                let cond = self.dump.expr(cond);
                let head = self.dump.node("While", &[cond]);
                self.push(depth, head);
                self.stmts(body, depth + 1);
            }
            Stmt::DoWhile { body, cond } => {
                let cond = self.dump.expr(cond);
                let head = self.dump.node("DoWhile", &[cond]);
                self.push(depth, head);
                self.stmts(body, depth + 1);
            }
            Stmt::Switch { value, cases } => {
                let value = self.dump.expr(value);
                let head = self.dump.node("Switch", &[value]);
                self.push(depth, head);
                for case in cases {
                    let case_head = if case.values.is_empty() {
                        self.dump.node("Default", &[])
                    } else {
                        let values: Vec<String> = case.values.iter().map(|v| self.dump.expr(v)).collect();
                        self.dump.node("Case", &[values.join(", ")])
                    };
                    self.push(depth + 1, case_head);
                    self.stmts(&case.body, depth + 2);
                }
            }
            Stmt::Try { resources, body, catches, finally } => {
                let head = if resources.is_empty() {
                    self.dump.node("Try", &[])
                } else {
                    let res: Vec<String> = resources
                        .iter()
                        .map(|r| {
                            let bind = r.name.as_ref().map_or_else(String::new, |n| self.dump.bind(n));
                            let value = self.dump.expr(&r.value);
                            self.dump.node("Resource", &[bind, value])
                        })
                        .collect();
                    let res = self.dump.list("Resources", &res);
                    self.dump.node("TryRes", &[res])
                };
                self.push(depth, head);
                self.stmts(body, depth + 1);
                for c in catches {
                    let types: Vec<String> = c.types.iter().map(|t| self.dump.ty(t)).collect();
                    let types = self.dump.list("Types", &types);
                    let bind = c.name.as_ref().map_or_else(String::new, |n| self.dump.bind(n));
                    let catch_head = self.dump.node("Catch", &[types, bind]);
                    self.push(depth + 1, catch_head);
                    self.stmts(&c.body, depth + 2);
                }
                if let Some(fin) = finally {
                    let head = self.dump.node("Finally", &[]);
                    self.push(depth + 1, head);
                    self.stmts(fin, depth + 2);
                }
            }
            Stmt::Synchronized { lock, body } => {
                let lock = self.dump.expr(lock);
                let head = self.dump.node("Synchronized", &[lock]);
                self.push(depth, head);
                self.stmts(body, depth + 1);
            }
            Stmt::Block(b) => {
                let head = self.dump.node("Block", &[]);
                self.push(depth, head);
                self.stmts(b, depth + 1);
            }
            // A label names a statement; the statement is the step.
            Stmt::Labeled { stmt, .. } => self.stmt(stmt, depth),
            _ => {
                let line = self.dump.stmt(s);
                self.push(depth, line);
            }
        }
    }

    /// `if` / `else if` / `else`: an `else if` is a sibling of its `else`, not a level deeper — what the
    /// source means, and what the other frontends' chains render as.
    fn if_chain(&mut self, s: &Stmt, depth: u16) {
        let Stmt::If { cond, then, els } = s else { return };
        let cond = self.dump.expr(cond);
        let head = self.dump.node("If", &[cond]);
        self.push(depth, head);
        self.stmts(then, depth + 1);
        if let Some(els) = els {
            let head = self.dump.node("Else", &[]);
            self.push(depth, head);
            match els.as_ref() {
                Else::If(alt) => self.if_chain(alt, depth),
                Else::Block(b) => self.stmts(b, depth + 1),
            }
        }
    }
}

// ───────────────────────────── scope / use sites ─────────────────────────────

/// What every dump of one definition shares: the file's imports and the names it binds.
struct Scope<'a> {
    imports: &'a HashMap<String, Arc<str>>,
    bound: &'a HashSet<String>,
}

impl<'a> Scope<'a> {
    fn dump(&self) -> Dump<'a> {
        Dump::new(self.imports, self.bound, Some(self.bound))
    }
}

/// The body canonicalized with every name the **file** introduced erased — the widest rung of the
/// erasure ladder, which the `scope` lens projects as a single fact.
#[must_use]
pub(crate) fn scope_canonical(imports: &HashMap<String, Arc<str>>, f: &Func, file_names: &HashSet<String>) -> String {
    let mut locals = crate::ir::locals_of(f);
    locals.extend(file_names.iter().cloned());
    Dump::new(imports, &locals, Some(&locals)).func("_fn", f, true, true)
}

/// Every use site in one source — the Java half of the `use` lens ([`dup_defs_core::uses`]): each
/// statement that mentions a known name, written with that name as the anchor. A site is the
/// statement's own line — a compound statement's header (`If(cond)`, `Switch(value)`, a case's
/// values), not its body, whose statements are sites of their own — as the statement stream writes it,
/// and inside a method its locals rename as they do there. A lambda stays part of the statement it is
/// written in, as a closure does in the Rust frontend.
///
/// A generated file holds no sites, as it holds no definitions: what its uses would say is the
/// generator's habit.
pub(crate) fn collect_sites(source: &str, _path: &str, known: &HashSet<&str>) -> Vec<Site> {
    if crate::defs::is_generated(source) {
        return Vec::new();
    }
    let Some(tree) = crate::tree::parse(source) else { return Vec::new() };
    let root = tree.root_node();
    let imports = file_imports(root, source);
    let mut walk = UseWalk { imports: &imports, known, out: Vec::new() };
    for item in kids(root) {
        if item.has_error() {
            continue;
        }
        match item.kind() {
            "class_declaration" | "interface_declaration" | "enum_declaration" | "record_declaration"
            | "annotation_type_declaration" => {
                if let Some(decl) = lower_class(item, source) {
                    walk.class(&decl);
                }
            }
            _ => {}
        }
    }
    walk.out
}

struct UseWalk<'k> {
    imports: &'k HashMap<String, Arc<str>>,
    known: &'k HashSet<&'k str>,
    out: Vec<Site>,
}

impl UseWalk<'_> {
    /// One site per known name `render` mentions, each written with that name as the anchor.
    fn site<F>(&mut self, scope: &Scope<'_>, ctx: &'static str, neighbours: &[String], render: F)
    where
        F: Fn(&mut Dump<'_>) -> Option<String>,
    {
        let mut hunt = Dump::hunting(scope, self.known);
        if render(&mut hunt).is_none() {
            return;
        }
        for name in hunt.hunted {
            let mut anchored = Dump::anchoring(scope, &name);
            if let Some(canon) = render(&mut anchored) {
                self.out.push(Site { name, canon, ctx, cooc: neighbours.to_vec() });
            }
        }
    }

    /// A method: its signature is a site when it names a known type, its body is walked under its own
    /// locals.
    fn function(&mut self, f: &Func) {
        let locals = crate::ir::locals_of(f);
        let scope = Scope { imports: self.imports, bound: &locals };
        self.site(&scope, "plain", &[], |d| Some(d.compact_header(f)));
        self.block(&scope, &f.body, Within::default());
    }

    /// A `static final` constant's initializer: a use of what it names.
    fn constant(&mut self, init: &Expr) {
        let empty = HashSet::new();
        let scope = Scope { imports: self.imports, bound: &empty };
        self.site(&scope, "plain", &[], |d| Some(d.expr(init)));
    }

    /// A class-like declaration: its methods' bodies and its constants' initializers, nested classes
    /// included.
    fn class(&mut self, d: &ClassDecl) {
        for m in &d.members {
            match m {
                ClassMember::Method { func, abstract_, .. } => {
                    if !*abstract_ {
                        self.function(func);
                    }
                }
                ClassMember::Field { constant, init, .. } => {
                    if *constant {
                        if let Some(init) = init {
                            self.constant(init);
                        }
                    }
                }
                ClassMember::Nested(n) => self.class(n),
                ClassMember::Variant { .. } | ClassMember::StaticInit => {}
            }
        }
    }

    /// A block's statements, each with the methods called beside it — the co-occurrence channel.
    fn block(&mut self, scope: &Scope<'_>, stmts: &[Stmt], within: Within) {
        let mut hunt = Dump::hunting(scope, self.known);
        for s in stmts {
            let _ = use_line(&mut hunt, s);
        }
        let neighbours: Vec<String> = hunt.calls.into_iter().collect();
        for s in stmts {
            self.stmt(scope, s, within, &neighbours);
        }
    }

    fn stmt(&mut self, scope: &Scope<'_>, s: &Stmt, within: Within, neighbours: &[String]) {
        let plain = site_ctx(Construct::Plain, within);
        let guard = site_ctx(Construct::Guard, within);
        let looped = site_ctx(Construct::Loop, within);
        let tried = site_ctx(Construct::Try, within);
        match s {
            Stmt::If { cond, then, els } => {
                self.site(scope, guard, neighbours, |d| {
                    let cond = d.expr(cond);
                    Some(d.node("If", &[cond]))
                });
                self.block(scope, then, within);
                match els.as_deref() {
                    Some(Else::If(alt)) => self.stmt(scope, alt, within, neighbours),
                    Some(Else::Block(b)) => self.block(scope, b, within),
                    None => {}
                }
            }
            Stmt::For { init, cond, update, body } => {
                self.site(scope, looped, neighbours, |d| Some(d.for_header(init.as_deref(), cond.as_ref(), update)));
                self.block(scope, body, Within { in_loop: true, ..within });
            }
            Stmt::ForEach { name, iter, body } => {
                self.site(scope, looped, neighbours, |d| {
                    let bind = d.bind(name);
                    let iter = d.expr(iter);
                    Some(d.node("For", &[bind, iter]))
                });
                self.block(scope, body, Within { in_loop: true, ..within });
            }
            Stmt::While { cond, body } => {
                self.site(scope, looped, neighbours, |d| {
                    let cond = d.expr(cond);
                    Some(d.node("While", &[cond]))
                });
                self.block(scope, body, Within { in_loop: true, ..within });
            }
            Stmt::DoWhile { body, cond } => {
                self.site(scope, looped, neighbours, |d| {
                    let cond = d.expr(cond);
                    Some(d.node("DoWhile", &[cond]))
                });
                self.block(scope, body, Within { in_loop: true, ..within });
            }
            Stmt::Switch { value, cases } => {
                self.site(scope, guard, neighbours, |d| {
                    let value = d.expr(value);
                    Some(d.node("Switch", &[value]))
                });
                for case in cases {
                    self.site(scope, guard, neighbours, |d| {
                        if case.values.is_empty() {
                            Some(d.node("Default", &[]))
                        } else {
                            let values: Vec<String> = case.values.iter().map(|v| d.expr(v)).collect();
                            Some(d.node("Case", &[values.join(", ")]))
                        }
                    });
                    self.block(scope, &case.body, within);
                }
            }
            Stmt::Try { resources, body, catches, finally } => {
                self.site(scope, tried, neighbours, |d| {
                    if resources.is_empty() {
                        Some(d.node("Try", &[]))
                    } else {
                        let res: Vec<String> = resources
                            .iter()
                            .map(|r| {
                                let bind = r.name.as_ref().map_or_else(String::new, |n| d.bind(n));
                                let value = d.expr(&r.value);
                                d.node("Resource", &[bind, value])
                            })
                            .collect();
                        let res = d.list("Resources", &res);
                        Some(d.node("TryRes", &[res]))
                    }
                });
                self.block(scope, body, Within { in_try: true, ..within });
                for c in catches {
                    self.site(scope, tried, neighbours, |d| {
                        let types: Vec<String> = c.types.iter().map(|t| d.ty(t)).collect();
                        let types = d.list("Types", &types);
                        let bind = c.name.as_ref().map_or_else(String::new, |n| d.bind(n));
                        Some(d.node("Catch", &[types, bind]))
                    });
                    self.block(scope, &c.body, within);
                }
                if let Some(fin) = finally {
                    self.block(scope, fin, within);
                }
            }
            Stmt::Synchronized { lock, body } => {
                self.site(scope, plain, neighbours, |d| {
                    let lock = d.expr(lock);
                    Some(d.node("Synchronized", &[lock]))
                });
                self.block(scope, body, within);
            }
            Stmt::Block(b) => self.block(scope, b, within),
            Stmt::Labeled { stmt, .. } => self.stmt(scope, stmt, within, neighbours),
            _ => self.site(scope, plain, neighbours, |d| use_line(d, s)),
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
        Stmt::While { cond, .. } => {
            let cond = d.expr(cond);
            Some(d.node("While", &[cond]))
        }
        Stmt::DoWhile { cond, .. } => {
            let cond = d.expr(cond);
            Some(d.node("DoWhile", &[cond]))
        }
        Stmt::For { init, cond, update, .. } => Some(d.for_header(init.as_deref(), cond.as_ref(), update)),
        Stmt::ForEach { name, iter, .. } => {
            let bind = d.bind(name);
            let iter = d.expr(iter);
            Some(d.node("For", &[bind, iter]))
        }
        Stmt::Switch { value, .. } => {
            let value = d.expr(value);
            Some(d.node("Switch", &[value]))
        }
        Stmt::Try { resources, .. } => {
            if resources.is_empty() {
                Some(d.node("Try", &[]))
            } else {
                let res: Vec<String> = resources
                    .iter()
                    .map(|r| {
                        let bind = r.name.as_ref().map_or_else(String::new, |n| d.bind(n));
                        let value = d.expr(&r.value);
                        d.node("Resource", &[bind, value])
                    })
                    .collect();
                let res = d.list("Resources", &res);
                Some(d.node("TryRes", &[res]))
            }
        }
        Stmt::Synchronized { lock, .. } => {
            let lock = d.expr(lock);
            Some(d.node("Synchronized", &[lock]))
        }
        Stmt::Block(_) => None,
        Stmt::Labeled { stmt, .. } => use_line(d, stmt),
        _ => Some(d.stmt(s)),
    }
}

// ───────────────────────────── imports / reach ─────────────────────────────

/// What each name a file's imports bind stands for: `import a.b.C;` binds `C` → `a.b.C`,
/// `import static a.b.C.m;` binds `m` → `a.b.C`. On-demand imports (`.*`) bind no enumerable name and
/// are skipped. Dotted names are kept as written.
#[must_use]
pub(crate) fn file_imports(root: Node<'_>, src: &str) -> HashMap<String, Arc<str>> {
    let mut out = HashMap::new();
    for decl in kids(root).into_iter().filter(|k| k.kind() == "import_declaration") {
        if kids(decl).into_iter().any(|k| k.kind() == "asterisk") {
            continue;
        }
        let Some(name_node) = kids(decl).into_iter().next() else { continue };
        let full = dotted_name(name_node, src);
        if full.is_empty() {
            continue;
        }
        let is_static = {
            let mut cursor = decl.walk();
            let any = decl.children(&mut cursor).any(|c| c.kind() == "static");
            any
        };
        if is_static {
            let Some((path, _member)) = full.rsplit_once('.') else { continue };
            if path.is_empty() {
                continue;
            }
            let alias = full.rsplit('.').next().unwrap_or("");
            out.insert(alias.to_owned(), Arc::from(path));
        } else {
            let alias = full.rsplit('.').next().unwrap_or(&full);
            out.insert(alias.to_owned(), Arc::from(full.as_str()));
        }
    }
    out
}

/// A `scoped_identifier`'s flat dotted spelling: `a.b.C`. Left-nested, so the head recurses.
fn dotted_name(n: Node<'_>, src: &str) -> String {
    match n.kind() {
        "scoped_identifier" => {
            let head = field(n, "scope").map_or_else(String::new, |s| dotted_name(s, src));
            let tail = field(n, "name").map_or_else(String::new, |m| text(m, src).to_owned());
            if head.is_empty() {
                tail
            } else if tail.is_empty() {
                head
            } else {
                format!("{head}.{tail}")
            }
        }
        _ => text(n, src).to_owned(),
    }
}

/// The types Java itself spells — never a subject a definition reaches.
const PRIMITIVE_TYPES: &[&str] = &["byte", "short", "int", "long", "float", "double", "boolean", "char", "void"];

/// The dotted paths one callable reaches: every class/type name its body uses that is an import alias
/// or already dotted, resolved through the file's imports — `this` / `super` and the language's own
/// types left out.
///
/// Not filtered by boundness, as in the Rust frontend: a local shadowing a class inside one method is
/// rare, and reading the shadow as "this definition does not reach that class" is a silent false
/// negative, the expensive direction here.
#[must_use]
pub(crate) fn reached_paths(f: &Func, imports: &HashMap<String, Arc<str>>) -> Vec<Arc<str>> {
    let mut out = Vec::new();
    for p in &f.params {
        reach_type(&p.ty, imports, &mut out);
    }
    if let Some(ret) = &f.ret {
        reach_type(ret, imports, &mut out);
    }
    reach_stmts(&f.body, imports, &mut out);
    out.sort();
    out.dedup();
    out
}

fn add_reach(name: &str, imports: &HashMap<String, Arc<str>>, out: &mut Vec<Arc<str>>) {
    if name.is_empty() || name == "this" || name == "super" || PRIMITIVE_TYPES.contains(&name) {
        return;
    }
    let path = if let Some(p) = imports.get(name) {
        Arc::clone(p)
    } else if name.contains('.') {
        Arc::from(name)
    } else {
        return;
    };
    if !out.contains(&path) {
        out.push(path);
    }
}

fn reach_type(t: &Type, imports: &HashMap<String, Arc<str>>, out: &mut Vec<Arc<str>>) {
    match t {
        Type::Named(name) => add_reach(name, imports, out),
        Type::Generic(name, args) => {
            add_reach(name, imports, out);
            for a in args {
                reach_type(a, imports, out);
            }
        }
        Type::Array(inner) | Type::Variadic(inner) => reach_type(inner, imports, out),
        Type::Wild(Some((_, ty))) => reach_type(ty, imports, out),
        Type::Union(parts) | Type::Inter(parts) => {
            for p in parts {
                reach_type(p, imports, out);
            }
        }
        Type::Wild(None) | Type::Other => {}
    }
}

fn reach_expr(e: &Expr, imports: &HashMap<String, Arc<str>>, out: &mut Vec<Arc<str>>) {
    match e {
        Expr::New { class, .. } | Expr::StaticCall { scope: class, .. } | Expr::StaticField { scope: class, .. }
        | Expr::ClassConst { scope: class } => add_reach(class, imports, out),
        Expr::Cast { ty, .. } | Expr::InstanceOf { ty, .. } => reach_type(ty, imports, out),
        Expr::MethodRef { target, .. } => {
            if let Expr::Name(n) = target.as_ref() {
                add_reach(n, imports, out);
            }
        }
        _ => {}
    }
    for_each_child(e, &mut |child| match child {
        Child::Expr(e) => reach_expr(e, imports, out),
        Child::Body(stmts) => reach_stmts(stmts, imports, out),
    });
}

fn reach_stmts(stmts: &[Stmt], imports: &HashMap<String, Arc<str>>, out: &mut Vec<Arc<str>>) {
    for s in stmts {
        match s {
            Stmt::Let { ty, init, .. } => {
                if let Some(t) = ty {
                    reach_type(t, imports, out);
                }
                if let Some(i) = init {
                    reach_expr(i, imports, out);
                }
            }
            Stmt::Decl(items) => reach_stmts(items, imports, out),
            Stmt::Assign { lhs, rhs, .. } => {
                reach_expr(lhs, imports, out);
                reach_expr(rhs, imports, out);
            }
            Stmt::Update { e, .. } => reach_expr(e, imports, out),
            Stmt::Expr(e) | Stmt::Throw(e) | Stmt::Assert(e) => reach_expr(e, imports, out),
            Stmt::Return(Some(e)) => reach_expr(e, imports, out),
            Stmt::If { cond, then, els } => {
                reach_expr(cond, imports, out);
                reach_stmts(then, imports, out);
                match els.as_deref() {
                    Some(Else::Block(b)) => reach_stmts(b, imports, out),
                    Some(Else::If(alt)) => reach_stmts(&[alt.as_ref().clone()], imports, out),
                    None => {}
                }
            }
            Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                reach_expr(cond, imports, out);
                reach_stmts(body, imports, out);
            }
            Stmt::For { init, cond, update, body } => {
                if let Some(i) = init {
                    reach_stmts(&[i.as_ref().clone()], imports, out);
                }
                if let Some(c) = cond {
                    reach_expr(c, imports, out);
                }
                for u in update {
                    reach_expr(u, imports, out);
                }
                reach_stmts(body, imports, out);
            }
            Stmt::ForEach { iter, body, .. } => {
                reach_expr(iter, imports, out);
                reach_stmts(body, imports, out);
            }
            Stmt::Switch { value, cases } => {
                reach_expr(value, imports, out);
                for case in cases {
                    for v in &case.values {
                        reach_expr(v, imports, out);
                    }
                    reach_stmts(&case.body, imports, out);
                }
            }
            Stmt::Try { resources, body, catches, finally } => {
                for r in resources {
                    reach_expr(&r.value, imports, out);
                }
                reach_stmts(body, imports, out);
                for c in catches {
                    for t in &c.types {
                        reach_type(t, imports, out);
                    }
                    reach_stmts(&c.body, imports, out);
                }
                if let Some(fin) = finally {
                    reach_stmts(fin, imports, out);
                }
            }
            Stmt::Synchronized { lock, body } => {
                reach_expr(lock, imports, out);
                reach_stmts(body, imports, out);
            }
            Stmt::Labeled { stmt, .. } => reach_stmts(&[stmt.as_ref().clone()], imports, out),
            Stmt::Block(b) => reach_stmts(b, imports, out),
            Stmt::Return(None) | Stmt::Break(_) | Stmt::Continue(_) | Stmt::NestedItem | Stmt::Unknown(_) => {}
        }
    }
}

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Lower the first method (with a body) of the first class in `src`, returning the file's imports
    /// and the method's [`Func`]. The tests read the canonicals straight off the owned tree.
    fn first_method(src: &str) -> (HashMap<String, Arc<str>>, Func) {
        let tree = crate::tree::parse(src).expect("parse");
        let root = tree.root_node();
        let imports = file_imports(root, src);
        let cls = kids(root).into_iter().find(|k| k.kind() == "class_declaration").expect("class");
        let decl = lower_class(cls, src).expect("lower class");
        for m in &decl.members {
            if let ClassMember::Method { func, abstract_, .. } = m {
                if !*abstract_ {
                    return (imports, func.clone());
                }
            }
        }
        panic!("no method with a body");
    }

    #[test]
    fn xname_blanks_the_top_name_and_renames_locals() {
        let (imports, f) = first_method(
            "class C { int add(int a, int b) { int s = a + b; return s + s; } }",
        );
        let analyzed = analyze(&imports, "C.add", &f);
        assert!(analyzed.xname_canonical.starts_with("Func('_fn', "), "{}", analyzed.xname_canonical);
        // The three value bindings (`a`, `b`, `s`) renumber by first occurrence; `s` is read twice so
        // the exact pass keeps it rather than inlining the single-use temporary.
        assert!(analyzed.xname_canonical.contains("_v0"));
        assert!(analyzed.xname_canonical.contains("_v1"));
        assert!(analyzed.xname_canonical.contains("_v2"));
    }

    #[test]
    fn this_reads_as_self_and_constructor_has_no_return_type() {
        let (imports, f) = first_method(
            "class C { int x; C(int x) { this.x = x; } }",
        );
        let analyzed = analyze(&imports, "C", &f);
        // The constructor's return type is empty, and `this` reads `self`.
        assert!(analyzed.xname_canonical.contains("Path('self')"), "{}", analyzed.xname_canonical);
    }

    #[test]
    fn statements_emit_if_and_loop_headers_at_each_depth() {
        let (imports, f) = first_method(
            "class C { void m(int n) { if (n > 0) { while (n > 0) { n--; } } } }",
        );
        let analyzed = analyze(&imports, "C.m", &f);
        // Depth 0 is the header, body starts at depth 1, the `if` at 1, the `while` at 2.
        assert_eq!(analyzed.statements[0].depth, 0);
        let depths: Vec<u16> = analyzed.statements.iter().map(|s| s.depth).collect();
        assert!(depths.contains(&1), "{depths:?}");
        assert!(depths.contains(&2), "{depths:?}");
        let lines: Vec<&str> = analyzed.statements.iter().map(|s| s.line.as_str()).collect();
        assert!(lines.iter().any(|l| l.starts_with("If(")), "{lines:?}");
        assert!(lines.iter().any(|l| l.starts_with("While(")), "{lines:?}");
    }

    #[test]
    fn type3_lines_slot_types_and_report_references() {
        let (imports, f) = first_method(
            "import java.util.List;\nclass C { List<String> first(List<String> xs) { return xs.get(0); } }",
        );
        let analyzed = analyze(&imports, "C.first", &f);
        // The import-resolved type slots and reports once, whatever its argument.
        assert!(analyzed.type_refs.iter().any(|t| t == "java.util.List"), "{:?}", analyzed.type_refs);
        assert!(analyzed.type3_lines[0].contains("_T0"), "{}", analyzed.type3_lines[0]);
    }

    #[test]
    fn reached_paths_resolve_import_aliases_and_keep_dotted_names() {
        let (imports, f) = first_method(
            "import java.util.List;\nclass C { void m(List x) { java.util.Map e = null; } }",
        );
        let reached = reached_paths(&f, &imports);
        assert!(reached.iter().any(|p| &**p == "java.util.List"), "{reached:?}");
        assert!(reached.iter().any(|p| &**p == "java.util.Map"), "{reached:?}");
    }

    #[test]
    fn file_imports_bind_single_and_static_names_and_skip_on_demand() {
        let src = "import java.util.List;\nimport static java.util.Map.Entry;\nimport java.util.*;\n";
        let tree = crate::tree::parse(src).expect("parse");
        let imports = file_imports(tree.root_node(), src);
        assert_eq!(&*imports["List"], "java.util.List");
        assert_eq!(&*imports["Entry"], "java.util.Map");
        // The on-demand import binds nothing enumerable.
        assert!(!imports.contains_key("Set"));
    }

    #[test]
    fn class_canon_renders_fields_and_method_bodies() {
        let src = "class C { int x = 1; int get() { return x; } }";
        let tree = crate::tree::parse(src).expect("parse");
        let root = tree.root_node();
        let cls = kids(root).into_iter().find(|k| k.kind() == "class_declaration").expect("class");
        let decl = lower_class(cls, src).expect("lower class");
        let canon = class_canon(&HashMap::new(), &decl);
        assert!(canon.starts_with("Class('C', g0, "), "{canon}");
        assert!(canon.contains("Field('x', "), "{canon}");
        assert!(canon.contains("MethodDef("), "{canon}");
    }
}
