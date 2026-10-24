//! Structural canonicalization of Go definitions, written from the owned tree ([`crate::ir`]).
//!
//! Mirrors the role `rs-canon::canon` plays for Rust: a compact, internally consistent **s-expr** per
//! definition that `difflib-fast` compares for name-gated similarity, and that the cross-name pass
//! `Eq`-checks once locals are alpha-renamed. The tag vocabulary is the Rust frontend's wherever the
//! construct is the same one (`Func`, `Params`, `Block`, `Let`, `ExprStmt`, `If`, `For`, `Call`,
//! `Method`, `Field`, `Bin`, …), so the engine's s-expr readers — converge's vocabulary split, the
//! unit counter — read Go the way they already read Rust. Two modes, driven by [`Dump::locals`]:
//!
//! * **cluster** (`locals = None`) — identifiers pass through verbatim; the names-preserved
//!   canonical the name-gated pass clusters on.
//! * **xname** (`locals = Some(set)`) — value bindings (parameters, named results, `:=`, `var`,
//!   `range` and `select` bindings, a type switch's alias) are renumbered `_v{n}` by first
//!   occurrence and the top definition's own name is blanked to `_fn`.
//!
//! A method's receiver is dropped from the parameters and written `self` wherever the body names it,
//! whatever the source called it: `func (s *Server) Close()` and `func (c *Client) Close()` spell the
//! same shape, and a method lines up with an equivalent free function for the cross-name pass — the
//! analog of the Rust frontend dropping `&self`.
//!
//! A selector on an imported package (`logx.Errorf`) is a path, `Path('logx.Errorf')`; on any other
//! value it is a field read or a method call. Go spells both `a.b`, and only the file's import table
//! tells them apart.
//!
//! The canonicals are written from the owned tree rather than from tree-sitter's because the exact
//! and Type-3 canonicals read the body's normal form ([`crate::normal`]), which rewrites it.
#![allow(
    clippy::too_many_lines, // the stmt/expr/type matches enumerate the tree's shapes; splitting just scatters one
    clippy::match_same_arms // distinct shapes intentionally share an emission for clarity
)]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;

use dup_defs_core::uses::{site_ctx, Construct, Site, Within};
use dup_defs_core::{quoted, AnalyzedFn, Statement, UNFILED};
use tree_sitter::Node;

use crate::ir::{
    lower_fields, lower_func, lower_iface, CommCase, Elem, Else, Expr, FieldDecl, ForHead, Func, IfaceItem, Jump,
    ParamDecl, Results, Stmt, Target, Type,
};
use crate::tree::{field, fields, kids, text, unquote};

// ───────────────────────────── the parts of a callable ─────────────────────────────

/// A function or method, as the grammar splits it — what is lowered to a [`Func`], and what the
/// lenses and the reach walk read directly.
#[derive(Clone, Copy)]
pub(crate) struct Callable<'t> {
    /// Whether it is a method — declared on a receiver, named or not.
    pub method: bool,
    /// The receiver's name, for a method that names it (`func (s *S)`); `None` for a function, an
    /// anonymous receiver (`func (*S)`), or `_`.
    pub receiver: Option<&'t str>,
    pub type_params: Option<Node<'t>>,
    pub params: Option<Node<'t>>,
    pub results: Option<Node<'t>>,
    pub body: Node<'t>,
}

// ───────────────────────────── bound-locals collector ─────────────────────────────

/// Every value binding the callable introduces — the rename set for xname mode, and in every mode
/// the set that tells a local from an imported package: parameters, named results, `:=`, `var` and
/// local `const` names, `range` / `select` bindings declared with `:=`, a type switch's alias.
///
/// Inside a function literal its declarations count and its parameters do not — the rule the
/// TypeScript frontend keeps for arrow functions (and the Rust one for closures), so a callback reads
/// the same in either language.
#[derive(Default)]
struct Collect {
    bound: HashSet<String>,
}

impl Collect {
    fn add(&mut self, name: &str) {
        if name != "_" {
            self.bound.insert(name.to_owned());
        }
    }

    fn params(&mut self, decls: &[ParamDecl]) {
        for decl in decls {
            for name in &decl.names {
                self.add(name);
            }
        }
    }

    fn targets(&mut self, targets: &[Target]) {
        for t in targets {
            if let Target::Name(name) = t {
                self.add(name);
            }
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Define { lhs, rhs } => {
                self.targets(lhs);
                for e in rhs {
                    self.expr(e);
                }
            }
            Stmt::Decl { specs, .. } => {
                for spec in specs {
                    for name in &spec.names {
                        self.add(name);
                    }
                    spec.values.iter().for_each(|e| self.expr(e));
                }
            }
            Stmt::Assign { lhs, rhs, .. } => {
                lhs.iter().chain(rhs).for_each(|e| self.expr(e));
            }
            Stmt::IncDec { target, .. } | Stmt::Expr(target) | Stmt::Go(target) | Stmt::Defer(target) => self.expr(target),
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
                    Some(Else::If(s)) => self.stmt(s),
                    None => {}
                }
            }
            Stmt::For { head, body } => {
                match head {
                    ForHead::Loop => {}
                    ForHead::While(c) => self.expr(c),
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
                    ForHead::Range { define, lhs, iter } => {
                        if *define {
                            if let Some(lhs) = lhs {
                                self.targets(lhs);
                            }
                        }
                        self.expr(iter);
                    }
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
                if let Some(alias) = alias {
                    self.targets(alias);
                }
                if let Some(init) = init {
                    self.stmt(init);
                }
                self.expr(value);
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
                if *define {
                    if let Some(lhs) = lhs {
                        self.targets(lhs);
                    }
                }
                self.expr(rhs);
            }
            Stmt::Block(b) => self.stmts(b),
            Stmt::Labeled { stmt, .. } => self.stmt(stmt),
            Stmt::NestedType | Stmt::Jump { .. } | Stmt::Fallthrough | Stmt::Empty | Stmt::Unknown(_) => {}
        }
    }

    fn expr(&mut self, e: &Expr) {
        for_each_child(e, &mut |child| match child {
            Child::Expr(e) => self.expr(e),
            // A function literal's parameters are its own; its body's declarations are walked.
            Child::Body(body) => self.stmts(body),
        });
    }
}

/// What an expression holds that a walk may want: its sub-expressions, and the body of a function
/// literal.
pub(crate) enum Child<'e> {
    Expr(&'e Expr),
    Body(&'e [Stmt]),
}

/// Every direct child of an expression, in source order.
pub(crate) fn for_each_child<'e>(e: &'e Expr, f: &mut dyn FnMut(Child<'e>)) {
    match e {
        Expr::Binary { l, r, .. } => {
            f(Child::Expr(l));
            f(Child::Expr(r));
        }
        Expr::Unary { e, .. } | Expr::Paren(e) | Expr::Spread(e) => f(Child::Expr(e)),
        Expr::Call { f: callee, args, .. } => {
            f(Child::Expr(callee));
            for a in args {
                f(Child::Expr(a));
            }
        }
        Expr::Selector { operand, .. } => f(Child::Expr(operand)),
        Expr::Index { e, i } => {
            f(Child::Expr(e));
            f(Child::Expr(i));
        }
        Expr::Slice { e, lo, hi, cap } => {
            f(Child::Expr(e));
            for part in [lo, hi, cap].into_iter().flatten() {
                f(Child::Expr(part));
            }
        }
        Expr::Assert { e, .. } | Expr::Conv { e, .. } => f(Child::Expr(e)),
        Expr::Composite { elems, .. } => {
            for el in elems {
                match el {
                    Elem::Field(_, v) | Elem::Plain(v) | Elem::KeyedBlank(v) => f(Child::Expr(v)),
                    Elem::Keyed(k, v) => {
                        f(Child::Expr(k));
                        f(Child::Expr(v));
                    }
                }
            }
        }
        Expr::FuncLit { body, .. } => f(Child::Body(body)),
        Expr::Name(_)
        | Expr::Bool(_)
        | Expr::Nil
        | Expr::Iota
        | Expr::Lit(..)
        | Expr::Inst { .. }
        | Expr::Type(_)
        | Expr::Unknown(_) => {}
    }
}

/// The bound locals of one callable (its receiver excluded — that one is written `self`).
pub(crate) fn locals_of(f: &Func) -> HashSet<String> {
    let mut collect = Collect::default();
    collect.params(&f.params);
    if let Results::List(list) = &f.results {
        collect.params(list);
    }
    collect.stmts(&f.body);
    if let Some(receiver) = &f.receiver {
        collect.bound.remove(receiver);
    }
    collect.bound
}

// ───────────────────────────── s-expr emitter ─────────────────────────────

pub(crate) struct Dump<'a> {
    /// What each name the file imports binds — a selector on one of them is a package member.
    imports: &'a HashMap<String, Arc<str>>,
    /// Every name the callable binds, in every mode: a local shadowing a package is a value.
    bound: &'a HashSet<String>,
    /// The receiver's own name, written `self`.
    receiver: Option<&'a str>,
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
    /// Use-site mode: the definition whose use is being written. A name or package member naming it
    /// becomes the anchor `_t0` — whichever package it was reached through, so a use inside its own
    /// package and one from outside read alike — and a struct literal of it marks the fields it sets
    /// `#name#` for the profile to number.
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
        receiver: Option<&'a str>,
        locals: Option<&'a HashSet<String>>,
    ) -> Self {
        Self {
            imports,
            bound,
            receiver,
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

    /// A reference to a package-level name: noted when it is hunted for, and whether it is the anchor.
    fn mention(&mut self, name: &str) -> bool {
        if let Some(known) = self.known {
            if known.contains(name) && !self.hunted.iter().any(|h| h == name) {
                self.hunted.push(name.to_owned());
            }
        }
        self.anchor == Some(name)
    }

    /// Whether a bare name is an imported package rather than a value.
    fn is_package(&self, name: &str) -> bool {
        self.imports.contains_key(name) && !self.bound.contains(name) && self.receiver != Some(name)
    }

    /// The package an expression names, when it is a bare name of one.
    fn package_of<'e>(&self, e: &'e Expr) -> Option<&'e str> {
        match e {
            Expr::Name(name) if self.is_package(name) => Some(name),
            _ => None,
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
    /// definition declared it), as spelled otherwise.
    ///
    /// 🔴 Only the Type-3 lines slot types — the Rust frontend's rule, for the Rust frontend's reason.
    /// A Go type picks the code that runs: the method set a call resolves against, the interface a
    /// value satisfies. Two bodies differing only in their types may not be one definition, so the
    /// exact pass keeps them apart; the near-copy pass reads a type as vocabulary like any other.
    fn type_name(&mut self, name: &str) -> String {
        // The anchor keeps its own slot even in type position, through whichever package it was named.
        let bare = name.rsplit('.').next().unwrap_or(name);
        if self.mention(bare) {
            return "_t0".to_owned();
        }
        let Some(tmap) = self.tmap.as_mut() else { return quoted(name) };
        if !self.declared_types.contains(name) {
            self.type_refs.push(name.to_owned());
        }
        dup_defs_core::type_slot(tmap, name)
    }

    /// A binding site: `Bind(_vN)`, or `Wild()` for the blank identifier.
    fn bind(&mut self, name: &str) -> String {
        if name == "_" {
            return self.node("Wild", &[]);
        }
        let renamed = self.rename(name);
        self.node("Bind", &[quoted(&renamed)])
    }

    /// A name read as a value.
    fn ident(&mut self, name: &str) -> String {
        if name == "_" {
            return self.node("Wild", &[]);
        }
        if self.receiver == Some(name) {
            return self.node("Path", &[quoted("self")]);
        }
        if !self.bound.contains(name) && self.mention(name) {
            return self.node("Path", &[quoted("_t0")]);
        }
        let renamed = self.rename(name);
        self.node("Path", &[quoted(&renamed)])
    }

    // ────── callables ──────

    /// `is_top` blanks the definition's own name in xname mode. The receiver is not a parameter.
    pub(crate) fn func(&mut self, name: &str, f: &Func, with_body: bool, is_top: bool) -> String {
        let name = if is_top && self.locals.is_some() && !self.blanked {
            self.blanked = true;
            "_fn".to_owned()
        } else {
            self.rename(name)
        };
        let params = self.params(&f.params, "Params");
        let results = self.results(&f.results);
        let body = if with_body { self.block(&f.body) } else { String::new() };
        self.node("Func", &[quoted(&name), params, results, body])
    }

    /// A parameter list, one `Param` per name — `a, b int` is two parameters of one type.
    fn params(&mut self, decls: &[ParamDecl], tag: &str) -> String {
        let mut items = Vec::new();
        for decl in decls {
            let ty = decl.ty.as_ref().map_or_else(String::new, |t| self.ty(t));
            let ty = if decl.variadic { self.node("TyVariadic", &[ty]) } else { ty };
            if decl.names.is_empty() {
                items.push(self.node("Param", &[String::new(), ty]));
                continue;
            }
            for name in &decl.names {
                let bound = self.bind(name);
                items.push(self.node("Param", &[bound, ty.clone()]));
            }
        }
        self.list(tag, &items)
    }

    /// A result: one type, or a parenthesized list whose names are bound locals.
    fn results(&mut self, results: &Results) -> String {
        match results {
            Results::None => String::new(),
            Results::List(list) => self.params(list, "Results"),
            Results::Type(ty) => self.ty(ty),
        }
    }

    /// The header of the Type-3 lines, in the language's own order: `func _fn[_T0](_v0 Ty(_T1, )) Ty(_T2, )`.
    ///
    /// 🔴 Not the `Func(…)` node the stream opens with. The Type-3 pass reads every function's first
    /// line as context and weighs it like any other; an s-expr header spells the parse tree's grammar
    /// (`Params`, `Param`, `Bind`, `Results`), which only headers spell, so every one of those tokens
    /// was rare and the header outweighed the body of every short function (the Rust frontend's finding).
    fn compact_header(&mut self, f: &Func) -> String {
        let mut out = String::from("func _fn");
        let mut declared = Vec::new();
        for name in &f.type_params {
            self.declared_types.insert(name.clone());
            declared.push(self.type_name(name));
        }
        if !declared.is_empty() {
            let _ = write!(out, "[{}]", declared.join(", "));
        }
        let params = self.header_params(&f.params);
        let _ = write!(out, "({params})");
        match &f.results {
            Results::None => {}
            Results::List(list) => {
                let results = self.header_params(list);
                let _ = write!(out, " ({results})");
            }
            Results::Type(ty) => {
                let ty = self.ty(ty);
                let _ = write!(out, " {ty}");
            }
        }
        out
    }

    fn header_params(&mut self, decls: &[ParamDecl]) -> String {
        let mut items = Vec::new();
        for decl in decls {
            let ty = decl.ty.as_ref().map_or_else(String::new, |t| self.ty(t));
            let ty = if decl.variadic { format!("...{ty}") } else { ty };
            if decl.names.is_empty() {
                items.push(ty);
                continue;
            }
            for name in &decl.names {
                let name = if name == "_" { "_".to_owned() } else { self.rename(name) };
                items.push(format!("{name} {ty}"));
            }
        }
        items.join(", ")
    }

    // ────── statements / blocks ──────

    pub(crate) fn block(&mut self, stmts: &[Stmt]) -> String {
        let items: Vec<String> = stmts.iter().map(|s| self.stmt(s)).collect();
        self.list("Block", &items)
    }

    /// Binding targets: names become `Bind`s, a list of two or more a tuple.
    fn targets(&mut self, targets: &[Target]) -> String {
        let items: Vec<String> = targets
            .iter()
            .map(|t| match t {
                Target::Name(name) => self.bind(name),
                Target::Expr(e) => self.expr(e),
            })
            .collect();
        self.tuple_of("PatTuple", items)
    }

    /// Expressions read as values, a list of two or more a tuple; nothing at all is an empty field.
    fn exprs(&mut self, list: &[Expr]) -> String {
        if list.is_empty() {
            return String::new();
        }
        let items: Vec<String> = list.iter().map(|e| self.expr(e)).collect();
        self.tuple_of("Tuple", items)
    }

    fn tuple_of(&mut self, tag: &str, mut items: Vec<String>) -> String {
        if items.len() == 1 {
            items.pop().unwrap_or_default()
        } else {
            self.list(tag, &items)
        }
    }

    /// One `var` / `const` spec inside a body: `Let(PatType(binds, ty), values)`.
    fn spec(&mut self, spec: &crate::ir::Spec, tag: &str) -> String {
        let names: Vec<String> = spec.names.iter().map(|n| self.bind(n)).collect();
        let pat = self.tuple_of("PatTuple", names);
        let pat = match &spec.ty {
            Some(t) => {
                let ty = self.ty(t);
                self.node("PatType", &[pat, ty])
            }
            None => pat,
        };
        let value = self.exprs(&spec.values);
        self.node(tag, &[pat, value])
    }

    pub(crate) fn stmt(&mut self, s: &Stmt) -> String {
        match s {
            Stmt::Define { lhs, rhs } => {
                let left = self.targets(lhs);
                let right = self.exprs(rhs);
                self.node("Let", &[left, right])
            }
            Stmt::Decl { konst, specs } => {
                let tag = if *konst { "LocalConst" } else { "Let" };
                let mut items: Vec<String> = specs.iter().map(|spec| self.spec(spec, tag)).collect();
                if items.len() == 1 {
                    items.pop().unwrap_or_default()
                } else {
                    self.list("Decls", &items)
                }
            }
            Stmt::NestedType => self.node("NestedItem", &[]),
            Stmt::Assign { op, lhs, rhs } => {
                let left = self.exprs(lhs);
                let right = self.exprs(rhs);
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
            Stmt::Send { ch, value } => {
                let ch = self.expr(ch);
                let v = self.expr(value);
                self.node("Send", &[ch, v])
            }
            Stmt::Expr(e) => {
                let e = self.expr(e);
                self.node("ExprStmt", &[e])
            }
            Stmt::Return(values) => {
                let values = self.exprs(values);
                self.node("Return", &[values])
            }
            Stmt::Go(e) | Stmt::Defer(e) => {
                let tag = if matches!(s, Stmt::Go(_)) { "Go" } else { "Defer" };
                let e = self.expr(e);
                self.node(tag, &[e])
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
            Stmt::Fallthrough => self.node("Fallthrough", &[]),
            Stmt::If { init, cond, then, els } => {
                let init = init.as_deref().map_or_else(String::new, |i| self.stmt(i));
                let cond = self.expr(cond);
                let then = self.block(then);
                let els = match els.as_deref() {
                    Some(Else::If(alt)) => self.stmt(alt),
                    Some(Else::Block(b)) => self.block(b),
                    None => String::new(),
                };
                self.node("If", &[init, cond, then, els])
            }
            Stmt::For { head, body } => {
                let body = self.block(body);
                match head {
                    ForHead::Loop => self.node("Loop", &[body]),
                    ForHead::Range { .. } => {
                        let (pat, iter) = self.range_parts(head);
                        self.node("For", &[pat, iter, body])
                    }
                    ForHead::Clause { .. } => {
                        let (init, cond, update) = self.clause_parts(head);
                        self.node("ForC", &[init, cond, update, body])
                    }
                    ForHead::While(cond) => {
                        let cond = self.expr(cond);
                        self.node("While", &[cond, body])
                    }
                }
            }
            Stmt::Switch { init, value, cases } => {
                let init = init.as_deref().map_or_else(String::new, |i| self.stmt(i));
                let value = value.as_ref().map_or_else(String::new, |v| self.expr(v));
                let cases: Vec<String> = cases
                    .iter()
                    .map(|case| {
                        if let Some(values) = &case.values {
                            let values = self.exprs(values);
                            let body = self.block(&case.body);
                            self.node("Case", &[values, body])
                        } else {
                            let body = self.block(&case.body);
                            self.node("Default", &[body])
                        }
                    })
                    .collect();
                let cases = cases.join(", ");
                self.node("Switch", &[init, value, cases])
            }
            Stmt::TypeSwitch { init, alias, value, cases } => {
                let init = init.as_deref().map_or_else(String::new, |i| self.stmt(i));
                let alias = alias.as_deref().map_or_else(String::new, |a| self.targets(a));
                let value = self.expr(value);
                let cases: Vec<String> = cases
                    .iter()
                    .map(|case| {
                        if let Some(types) = &case.types {
                            let types = self.types_joined(types);
                            let body = self.block(&case.body);
                            self.node("TypeCase", &[types, body])
                        } else {
                            let body = self.block(&case.body);
                            self.node("Default", &[body])
                        }
                    })
                    .collect();
                let cases = cases.join(", ");
                self.node("TypeSwitch", &[init, alias, value, cases])
            }
            Stmt::Select { cases } => {
                let cases: Vec<String> = cases
                    .iter()
                    .map(|case| {
                        if case.default {
                            let body = self.block(&case.body);
                            self.node("Default", &[body])
                        } else {
                            let comm = case.comm.as_deref().map_or_else(String::new, |m| self.stmt(m));
                            let body = self.block(&case.body);
                            self.node("CommCase", &[comm, body])
                        }
                    })
                    .collect();
                self.list("Select", &cases)
            }
            Stmt::Recv { define, lhs, rhs } => {
                let left = match lhs {
                    Some(l) if *define => self.targets(l),
                    Some(l) => self.target_exprs(l),
                    None => String::new(),
                };
                let right = self.expr(rhs);
                self.node("RecvStmt", &[left, right])
            }
            Stmt::Block(b) => {
                let body = self.block(b);
                self.node("BlockExpr", &[body])
            }
            Stmt::Labeled { label, stmt } => {
                let label = quoted(label);
                let inner = self.stmt(stmt);
                self.node("Labeled", &[label, inner])
            }
            Stmt::Empty => self.node("Empty", &[]),
            Stmt::Unknown(kind) => self.node(&format!("Unknown_{kind}"), &[]),
        }
    }

    /// Targets that assign rather than declare, read as values.
    fn target_exprs(&mut self, targets: &[Target]) -> String {
        let items: Vec<String> = targets
            .iter()
            .map(|t| match t {
                Target::Name(name) => self.ident(name),
                Target::Expr(e) => self.expr(e),
            })
            .collect();
        self.tuple_of("Tuple", items)
    }

    fn types_joined(&mut self, types: &[Type]) -> String {
        let types: Vec<String> = types.iter().map(|t| self.ty(t)).collect();
        types.join(", ")
    }

    /// `for k, v := range xs` → the bindings and what is ranged over.
    fn range_parts(&mut self, head: &ForHead) -> (String, String) {
        let ForHead::Range { define, lhs, iter } = head else { return (String::new(), String::new()) };
        let pat = match lhs {
            Some(l) if *define => self.targets(l),
            Some(l) => self.target_exprs(l),
            None => String::new(),
        };
        let iter = self.expr(iter);
        (pat, iter)
    }

    /// `for init; cond; update` → its three parts, each possibly empty.
    fn clause_parts(&mut self, head: &ForHead) -> (String, String, String) {
        let ForHead::Clause { init, cond, update } = head else { return (String::new(), String::new(), String::new()) };
        let init = init.as_deref().map_or_else(String::new, |i| self.stmt(i));
        let cond = cond.as_ref().map_or_else(String::new, |c| self.expr(c));
        let update = update.as_deref().map_or_else(String::new, |u| self.stmt(u));
        (init, cond, update)
    }

    // ────── compound-statement headers ──────
    //
    // A compound statement's own line — what it tests, its body elided. Shared by the statement
    // stream and the use-site walk, which must write a header the same way.

    /// `For(binds, iter)`, `ForC(init, cond, update)`, `While(cond)` or `Loop()`.
    fn for_header(&mut self, head: &ForHead) -> String {
        match head {
            ForHead::Loop => self.node("Loop", &[]),
            ForHead::Range { .. } => {
                let (pat, iter) = self.range_parts(head);
                self.node("For", &[pat, iter])
            }
            ForHead::Clause { .. } => {
                let (init, cond, update) = self.clause_parts(head);
                self.node("ForC", &[init, cond, update])
            }
            ForHead::While(cond) => {
                let cond = self.expr(cond);
                self.node("While", &[cond])
            }
        }
    }

    /// `Switch(value)`, `TypeSwitch(alias, value)` or `Select()` — its initializer is a step of its
    /// own. `None` for any other statement.
    fn switch_header(&mut self, s: &Stmt) -> Option<String> {
        Some(match s {
            Stmt::TypeSwitch { alias, value, .. } => {
                let alias = alias.as_deref().map_or_else(String::new, |a| self.targets(a));
                let value = self.expr(value);
                self.node("TypeSwitch", &[alias, value])
            }
            Stmt::Select { .. } => self.node("Select", &[]),
            Stmt::Switch { value, .. } => {
                let value = value.as_ref().map_or_else(String::new, |v| self.expr(v));
                self.node("Switch", &[value])
            }
            _ => return None,
        })
    }

    fn comm_head(&mut self, case: &CommCase) -> String {
        if case.default {
            return self.node("Default", &[]);
        }
        let comm = case.comm.as_deref().map_or_else(String::new, |m| self.stmt(m));
        self.node("CommCase", &[comm])
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
        if self.expr_depth > 0 && self.is_unit(e) {
            if let Some(units) = self.units.as_mut() {
                units.push((self.unit_line, out.clone()));
            }
        }
        out
    }

    fn opt_expr(&mut self, e: Option<&Expr>) -> String {
        e.map_or_else(String::new, |e| self.expr(e))
    }

    fn opt_ty(&mut self, t: Option<&Type>) -> String {
        t.map_or_else(String::new, |t| self.ty(t))
    }

    fn expr_inner(&mut self, e: &Expr) -> String {
        match e {
            Expr::Name(name) => self.ident(name),
            Expr::Bool(b) => self.node("Bool", &[quoted(if *b { "true" } else { "false" })]),
            Expr::Nil => self.node("Nil", &[]),
            Expr::Iota => self.node("Iota", &[]),
            Expr::Lit(tag, text) => self.node(tag, &[quoted(text)]),
            Expr::Binary { op, l, r } => {
                let l = self.expr(l);
                let r = self.expr(r);
                self.node("Bin", &[l, quoted(op), r])
            }
            Expr::Unary { op, e } => {
                let inner = self.expr(e);
                match op.as_str() {
                    "&" => self.node("Ref", &[inner]),
                    "<-" => self.node("Recv", &[inner]),
                    op => self.node("Unary", &[quoted(op), inner]),
                }
            }
            Expr::Paren(inner) => self.expr(inner),
            Expr::Call { f, args, targs } => self.call(f, args, targs.as_deref()),
            Expr::Selector { operand, field } => {
                if let Some(pkg) = self.package_of(operand) {
                    let path = if self.mention(field) { "_t0".to_owned() } else { format!("{pkg}.{field}") };
                    return self.node("Path", &[quoted(&path)]);
                }
                let base = self.expr(operand);
                self.node("Field", &[base, quoted(field)])
            }
            Expr::Index { e, i } => {
                let base = self.expr(e);
                let idx = self.expr(i);
                self.node("Index", &[base, idx])
            }
            Expr::Slice { e, lo, hi, cap } => {
                let base = self.expr(e);
                let lo = self.opt_expr(lo.as_deref());
                let hi = self.opt_expr(hi.as_deref());
                let cap = self.opt_expr(cap.as_deref());
                self.node("Slice", &[base, lo, hi, cap])
            }
            Expr::Assert { e, ty } => {
                let e = self.expr(e);
                let t = self.opt_ty(ty.as_ref());
                self.node("Assert", &[e, t])
            }
            Expr::Conv { e, ty } => {
                let e = self.expr(e);
                let t = self.opt_ty(ty.as_ref());
                self.node("Cast", &[e, t])
            }
            Expr::Inst { ty, args } => {
                let t = self.opt_ty(ty.as_ref());
                let args: Vec<String> = args.iter().map(|a| self.ty(a)).collect();
                let args = self.list("TypeArgs", &args);
                self.node("Inst", &[t, args])
            }
            Expr::Composite { ty, elems } => {
                let t = self.opt_ty(ty.as_ref());
                // A literal of the anchor sets its surface: the fields it names are marked for the
                // profile to number.
                let on_anchor = t.starts_with("Ty(_t0");
                let elems = self.elements(elems, on_anchor);
                self.node("Composite", &[t, elems])
            }
            Expr::FuncLit { params, results, body } => {
                let params = self.params(params, "Params");
                let results = self.results(results);
                let body = self.block(body);
                self.node("Closure", &[params, results, body])
            }
            Expr::Spread(inner) => {
                let inner = self.expr(inner);
                self.node("Spread", &[inner])
            }
            Expr::Type(t) => self.ty(t),
            Expr::Unknown(kind) => self.node(&format!("Unknown_{kind}"), &[]),
        }
    }

    /// A call: `Method(receiver, 'name', args)` when it calls through a value — the Rust frontend's
    /// method call — and `Call(callee, args)` when the callee is a name or a package member.
    fn call(&mut self, f: &Expr, args: &[Expr], targs: Option<&[Type]>) -> String {
        let mut items: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
        if let Some(targs) = targs {
            let types: Vec<String> = targs.iter().map(|t| self.ty(t)).collect();
            items.push(self.list("TypeArgs", &types));
        }
        let args = items.join(", ");
        if let Expr::Selector { operand, field } = f {
            if self.package_of(operand).is_none() {
                let recv = self.expr(operand);
                if self.known.is_some() {
                    self.calls.insert(format!(".{field}"));
                }
                return self.node("Method", &[recv, quoted(field), args]);
            }
        }
        let callee = self.expr(f);
        self.node("Call", &[callee, args])
    }

    /// A composite literal's elements. A key that is a bare name is a struct field: `FieldVal('name',
    /// v)`, the Rust frontend's struct-literal field, its name kept as spelled. Any other key is an
    /// expression — a map's — and the element is `KV(key, v)`. The two differ in what a varying key
    /// means: another field is another member (reflection), another map key is just another value.
    fn elements(&mut self, elems: &[Elem], on_anchor: bool) -> String {
        let items: Vec<String> = elems
            .iter()
            .map(|el| match el {
                Elem::Field(name, value) => {
                    let key = if on_anchor { quoted(&format!("#{name}#")) } else { quoted(name) };
                    let value = self.expr(value);
                    self.node("FieldVal", &[key, value])
                }
                // The key is written before the value, so a local in it is numbered in source order.
                Elem::Keyed(key, value) => {
                    let key = self.expr(key);
                    let value = self.expr(value);
                    self.node("KV", &[key, value])
                }
                Elem::KeyedBlank(value) => {
                    let value = self.expr(value);
                    self.node("KV", &[String::new(), value])
                }
                Elem::Plain(value) => self.expr(value),
            })
            .collect();
        items.join(", ")
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
                let args: Vec<String> = args.iter().map(|a| self.ty(a)).collect();
                let args = args.join(", ");
                self.node("Ty", &[name, args])
            }
            Type::Ptr(inner) => {
                let inner = self.ty(inner);
                self.node("TyPtr", &[inner])
            }
            Type::Slice(inner) => {
                let inner = self.ty(inner);
                self.node("TySlice", &[inner])
            }
            Type::Array(inner) => {
                let inner = self.ty(inner);
                self.node("TyArray", &[inner])
            }
            Type::Map(k, v) => {
                let k = self.ty(k);
                let v = self.ty(v);
                self.node("TyMap", &[k, v])
            }
            Type::Chan(dir, v) => {
                let v = self.ty(v);
                self.node("TyChan", &[quoted(dir), v])
            }
            Type::Func(params, results) => {
                let params = self.params(params, "Params");
                let results = self.results(results);
                self.node("TyFn", &[params, results])
            }
            Type::Struct(fields) => {
                let fields = struct_fields(self, fields);
                self.node("TyStruct", &[fields])
            }
            Type::Iface(items) => {
                let items = interface_items(self, items);
                self.node("TyIface", &[items])
            }
            Type::Approx(inner) => {
                let inner = self.ty(inner);
                self.node("TyApprox", &[inner])
            }
            Type::Union(parts) => {
                let parts: Vec<String> = parts.iter().map(|p| self.ty(p)).collect();
                self.list("TyUnion", &parts)
            }
            Type::Other => self.node("Ty_Other", &[]),
        }
    }

    /// Whether an expression is a Type-3 unit of its own: a node with structure — a call, a field
    /// read, an operator application, a literal with elements, a function literal — not a bare name,
    /// a package member, or a literal, which carry no agreement by themselves. The rule the Python
    /// and Rust frontends keep.
    fn is_unit(&self, e: &Expr) -> bool {
        match e {
            Expr::Call { .. }
            | Expr::Index { .. }
            | Expr::Slice { .. }
            | Expr::Binary { .. }
            | Expr::Unary { .. }
            | Expr::Assert { .. }
            | Expr::Conv { .. }
            | Expr::Inst { .. }
            | Expr::FuncLit { .. } => true,
            Expr::Selector { operand, .. } => self.package_of(operand).is_none(),
            Expr::Composite { elems, .. } => !elems.is_empty(),
            _ => false,
        }
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

/// A struct's fields, one per name; an embedded field is `Embed(type)`. Tags are left out: two
/// structs that hold the same fields are the same shape however they serialize.
fn struct_fields(d: &mut Dump<'_>, fields: &[FieldDecl]) -> String {
    let mut items = Vec::new();
    for decl in fields {
        let ty = decl.ty.as_ref().map_or_else(String::new, |t| d.ty(t));
        if decl.names.is_empty() {
            items.push(d.node("Embed", &[ty]));
            continue;
        }
        for name in &decl.names {
            let name = quoted(name);
            items.push(d.node("Field", &[name, ty.clone()]));
        }
    }
    d.list("Named", &items)
}

/// An interface's method set and embedded types, with method signatures only.
fn interface_items(d: &mut Dump<'_>, items: &[IfaceItem]) -> String {
    let items: Vec<String> = items
        .iter()
        .map(|item| match item {
            IfaceItem::Method { name, params, results } => {
                let name = quoted(name);
                let params = d.params(params, "Params");
                let results = d.results(results);
                d.node("IfaceFn", &[name, params, results])
            }
            IfaceItem::Embed(ty) => {
                let ty = d.ty(ty);
                d.node("IfaceEmbed", &[ty])
            }
        })
        .collect();
    items.join(", ")
}

/// `g{n}` — the number of type parameters a declaration takes (their names are not compared).
fn generics_tag(type_params: Option<Node<'_>>) -> String {
    let n: usize = type_params.map(kids).unwrap_or_default().iter().map(|d| fields(*d, "name").len()).sum();
    format!("g{n}")
}

/// Names-preserved structural canonical of a `type X struct { … }` — the cluster pass's input for
/// `classes`.
#[must_use]
pub(crate) fn struct_canon(src: &str, name: &str, type_params: Option<Node<'_>>, struct_type: Node<'_>) -> String {
    let (imports, bound) = (HashMap::new(), HashSet::new());
    let mut d = Dump::new(&imports, &bound, None, None);
    let fields = struct_fields(&mut d, &lower_fields(struct_type, src));
    d.node("Struct", &[quoted(name), generics_tag(type_params), fields])
}

/// Names-preserved structural canonical of a `type X interface { … }` — the cluster pass's input for
/// `interfaces`.
#[must_use]
pub(crate) fn interface_canon(src: &str, name: &str, type_params: Option<Node<'_>>, iface: Node<'_>) -> String {
    let (imports, bound) = (HashMap::new(), HashSet::new());
    let mut d = Dump::new(&imports, &bound, None, None);
    let items = interface_items(&mut d, &lower_iface(iface, src));
    d.node("Interface", &[quoted(name), generics_tag(type_params), items])
}

// ───────────────────────────── callable analysis ─────────────────────────────

/// Every canonical form of one callable.
///
/// The cluster canonical and the statement stream read the body as written: the name-gated pass
/// compares what is spelled, and converge reads the stream to say where two definitions stop
/// agreeing — a rewritten line would point the reader at code that is not there. The Type-3 lines
/// read the normal form that keeps every statement, the exact canonical the one that also inlines
/// single-use temporaries ([`crate::normal`]).
pub(crate) fn analyze(imports: &HashMap<String, Arc<str>>, name: &str, f: &Func) -> AnalyzedFn {
    let bound = locals_of(f);
    let receiver = f.receiver.as_deref();
    let cluster = Dump::new(imports, &bound, receiver, None).func(name, f, true, true);

    let statements = {
        let mut dump = Dump::new(imports, &bound, receiver, Some(&bound));
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
        let bound = locals_of(&statement_form);
        let mut dump = Dump::new(imports, &bound, receiver, Some(&bound));
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
    let mut xd = Dump::new(imports, &exact_bound, receiver, Some(&exact_bound));
    let xname = xd.func(name, &exact_form, true, true);
    let size = xd.count;
    AnalyzedFn { cluster_canonical: cluster, xname_canonical: xname, type3_lines, type_refs, type3_units, statements, size }
}

// ───────────────────────────── statement stream ─────────────────────────────

/// One line per statement at **every** nesting level, in source order — the engine's
/// [`Facets::statements`](dup_defs_core::Facets::statements), and with the types slotted, the Type-3
/// lines.
///
/// A compound statement contributes a **header** naming what it tests (`If(cond)`, `Switch(value)`,
/// `For(binds, iter)`) with its body one level deeper — the shape the Rust and Python frontends
/// produce, so the languages' streams compare. Go's `if x := f(); x != nil` is two steps written on
/// one line: the initializer is emitted as its own statement before the `If`, so it lines up with the
/// same code written `x := f()` on the line above.
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

    fn init(&mut self, init: Option<&Stmt>, depth: u16) {
        if let Some(init) = init {
            let line = self.dump.stmt(init);
            self.push(depth, line);
        }
    }

    fn stmt(&mut self, s: &Stmt, depth: u16) {
        match s {
            Stmt::If { .. } => self.if_chain(s, depth),
            Stmt::For { head, body } => {
                let head = self.dump.for_header(head);
                self.push(depth, head);
                self.stmts(body, depth + 1);
            }
            Stmt::Switch { init, .. } | Stmt::TypeSwitch { init, .. } => {
                self.init(init.as_deref(), depth);
                self.switch(s, depth);
            }
            Stmt::Select { .. } => self.switch(s, depth),
            Stmt::Block(b) => {
                let head = self.dump.node("Block", &[]);
                self.push(depth, head);
                self.stmts(b, depth + 1);
            }
            // A label names a statement; the statement is the step.
            Stmt::Labeled { stmt, .. } => self.stmt(stmt, depth),
            Stmt::Empty => {}
            _ => {
                let line = self.dump.stmt(s);
                self.push(depth, line);
            }
        }
    }

    fn switch(&mut self, s: &Stmt, depth: u16) {
        let Some(head) = self.dump.switch_header(s) else { return };
        self.push(depth, head);
        for (head, body) in clauses(s) {
            let head = head.render(self.dump);
            self.push(depth + 1, head);
            self.stmts(body, depth + 2);
        }
    }

    /// `if` / `else if` / `else`: an `else if` is a sibling of its `else`, not a level deeper — what
    /// the source means, and what Python's `elif` and the Rust frontend's chains render as.
    fn if_chain(&mut self, s: &Stmt, depth: u16) {
        let Stmt::If { init, cond, then, els } = s else { return };
        self.init(init.as_deref(), depth);
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

/// A clause whose header is rendered when the walk reaches it — so its units are filed under its
/// own line, not under the header of the clause before it.
enum ClauseHead<'s> {
    Case(Option<&'s [Expr]>),
    TypeCase(Option<&'s [Type]>),
    Comm(&'s CommCase),
}

impl ClauseHead<'_> {
    fn render(&self, d: &mut Dump<'_>) -> String {
        match self {
            ClauseHead::Case(Some(values)) => {
                let values = d.exprs(values);
                d.node("Case", &[values])
            }
            ClauseHead::TypeCase(Some(types)) => {
                let types = d.types_joined(types);
                d.node("TypeCase", &[types])
            }
            ClauseHead::Case(None) | ClauseHead::TypeCase(None) => d.node("Default", &[]),
            ClauseHead::Comm(case) => d.comm_head(case),
        }
    }
}

/// The clauses of a `switch` / type switch / `select`, headers unrendered.
fn clauses(s: &Stmt) -> Vec<(ClauseHead<'_>, &[Stmt])> {
    match s {
        Stmt::Switch { cases, .. } => {
            cases.iter().map(|c| (ClauseHead::Case(c.values.as_deref()), c.body.as_slice())).collect()
        }
        Stmt::TypeSwitch { cases, .. } => {
            cases.iter().map(|c| (ClauseHead::TypeCase(c.types.as_deref()), c.body.as_slice())).collect()
        }
        Stmt::Select { cases } => cases.iter().map(|c| (ClauseHead::Comm(c), c.body.as_slice())).collect(),
        _ => Vec::new(),
    }
}

// ───────────────────────────── use sites ─────────────────────────────

/// What every dump of one definition shares: the file's imports, the names it binds and its
/// receiver.
#[derive(Clone, Copy)]
pub(crate) struct Scope<'a> {
    pub(crate) imports: &'a HashMap<String, Arc<str>>,
    pub(crate) bound: &'a HashSet<String>,
    pub(crate) receiver: Option<&'a str>,
}

impl<'a> Scope<'a> {
    fn dump(&self) -> Dump<'a> {
        Dump::new(self.imports, self.bound, self.receiver, Some(self.bound))
    }
}

/// The body canonicalized with every name the **file** introduced erased — the widest rung of the
/// erasure ladder, which the `scope` lens projects as a single fact.
#[must_use]
pub(crate) fn scope_canonical(imports: &HashMap<String, Arc<str>>, f: &Func, file_names: &HashSet<String>) -> String {
    let mut locals = locals_of(f);
    locals.extend(file_names.iter().cloned());
    Dump::new(imports, &locals, f.receiver.as_deref(), Some(&locals)).func("_fn", f, true, true)
}

/// Every use site in one source — the Go half of the `use` lens ([`dup_defs_core::uses`]): each
/// statement that mentions a known name, written with that name as the anchor. A site is the
/// statement's own line — a compound statement's header (`If(cond)`, `Switch(value)`, a clause's
/// values), not its body, whose statements are sites of their own — as the statement stream writes
/// it, and inside a function its locals rename as they do there. A function literal stays part of
/// the statement it is written in, as a closure does in the Rust frontend.
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
            "function_declaration" | "method_declaration" => {
                if let Some(c) = crate::defs::callable_of(item, source) {
                    walk.function(&lower_func(&c, source));
                }
            }
            "const_declaration" | "var_declaration" => {
                if let crate::ir::Stmt::Decl { konst, specs } = crate::ir::lower_stmt(item, source) {
                    walk.values(konst, &specs);
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

    /// A function: its signature is a site when it names a known type, its body is walked under its
    /// own locals.
    fn function(&mut self, f: &Func) {
        let locals = locals_of(f);
        let scope = Scope { imports: self.imports, bound: &locals, receiver: f.receiver.as_deref() };
        self.site(&scope, "plain", &[], |d| Some(d.compact_header(f)));
        self.block(&scope, &f.body, Within::default());
    }

    /// A package-level `const` / `var`: its value is a use of what it names.
    fn values(&mut self, konst: bool, specs: &[crate::ir::Spec]) {
        let empty = HashSet::new();
        let scope = Scope { imports: self.imports, bound: &empty, receiver: None };
        let tag = if konst { "Const" } else { "Var" };
        for spec in specs.iter().filter(|s| !s.values.is_empty()) {
            self.site(&scope, "plain", &[], |d| {
                let v = d.exprs(&spec.values);
                Some(d.node(tag, &[v]))
            });
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
        match s {
            Stmt::If { init, cond, then, els } => {
                if let Some(init) = init.as_deref() {
                    self.site(scope, plain, neighbours, |d| Some(d.stmt(init)));
                }
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
            Stmt::For { head, body } => {
                let looped = site_ctx(Construct::Loop, within);
                self.site(scope, looped, neighbours, |d| Some(d.for_header(head)));
                self.block(scope, body, Within { in_loop: true, ..within });
            }
            Stmt::Switch { .. } | Stmt::TypeSwitch { .. } | Stmt::Select { .. } => {
                let init = match s {
                    Stmt::Switch { init, .. } | Stmt::TypeSwitch { init, .. } => init.as_deref(),
                    _ => None,
                };
                if let Some(init) = init {
                    self.site(scope, plain, neighbours, |d| Some(d.stmt(init)));
                }
                self.site(scope, guard, neighbours, |d| d.switch_header(s));
                for (head, body) in clauses(s) {
                    self.site(scope, guard, neighbours, |d| Some(head.render(d)));
                    self.block(scope, body, within);
                }
            }
            Stmt::Block(b) => self.block(scope, b, within),
            Stmt::Labeled { stmt, .. } => self.stmt(scope, stmt, within, neighbours),
            Stmt::Empty => {}
            _ => self.site(scope, plain, neighbours, |d| use_line(d, s)),
        }
    }
}

/// A statement's own line, as a use site sees it: a compound statement's header (and its
/// initializer, rendered for the calls it makes), any other statement whole — `None` for a bare
/// block, whose statements are walked.
fn use_line(d: &mut Dump<'_>, s: &Stmt) -> Option<String> {
    match s {
        Stmt::If { init, cond, .. } => {
            if let Some(init) = init.as_deref() {
                let _ = d.stmt(init);
            }
            let cond = d.expr(cond);
            Some(d.node("If", &[cond]))
        }
        Stmt::For { head, .. } => Some(d.for_header(head)),
        Stmt::Switch { init, .. } | Stmt::TypeSwitch { init, .. } => {
            if let Some(init) = init.as_deref() {
                let _ = d.stmt(init);
            }
            d.switch_header(s)
        }
        Stmt::Select { .. } => d.switch_header(s),
        Stmt::Block(_) | Stmt::Empty => None,
        Stmt::Labeled { stmt, .. } => use_line(d, stmt),
        _ => Some(d.stmt(s)),
    }
}

// ───────────────────────────── reach ─────────────────────────────

/// What each package name a file's imports bind stands for, as the engine's dotted path
/// (`github.com/a/b/logx` → `github.com.a.b.logx`).
///
/// A blank import (`_ "embed"`) binds no name; a dot import (`. "x"`) binds names this cannot
/// enumerate, and guessing would invent reach that is not there — both are skipped. An unaliased
/// import binds the package's own name, which only the imported package's source states; the
/// convention it nearly always follows (the last path element, a major-version element skipped, a
/// `go-` prefix or `.vN` suffix dropped) is read instead, and a package that breaks it simply goes
/// unattributed.
#[must_use]
pub(crate) fn file_imports(root: Node<'_>, src: &str) -> HashMap<String, Arc<str>> {
    let mut out = HashMap::new();
    for decl in kids(root).into_iter().filter(|k| k.kind() == "import_declaration") {
        let mut specs = Vec::new();
        for k in kids(decl) {
            if k.kind() == "import_spec" {
                specs.push(k);
            } else {
                specs.extend(kids(k).into_iter().filter(|s| s.kind() == "import_spec"));
            }
        }
        for spec in specs {
            let Some(path) = field(spec, "path").map(|p| unquote(text(p, src))) else { continue };
            let binding = match field(spec, "name") {
                Some(name) if name.kind() == "package_identifier" => text(name, src).to_owned(),
                Some(_) => continue,
                None => default_package_name(path),
            };
            if binding.is_empty() {
                continue;
            }
            let segments: Vec<&str> = path.split('/').collect();
            out.insert(binding, dup_defs_core::reach::reach_path(&segments));
        }
    }
    out
}

/// The name an unaliased import conventionally binds.
fn default_package_name(path: &str) -> String {
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let is_major = |s: &str| s.len() > 1 && s.starts_with('v') && s[1..].chars().all(|c| c.is_ascii_digit());
    let last = match segments.as_slice() {
        [.., prev, last] if is_major(last) => prev,
        [.., last] => last,
        [] => return String::new(),
    };
    let mut name = *last;
    if let Some((stem, version)) = name.rsplit_once(".v") {
        if !version.is_empty() && version.chars().all(|c| c.is_ascii_digit()) {
            name = stem;
        }
    }
    let name = name.strip_prefix("go-").unwrap_or(name);
    let name = name.strip_suffix("-go").unwrap_or(name);
    name.replace(['-', '.'], "_")
}

/// The dotted paths one callable reaches: every package it names through a selector or a qualified
/// type, resolved through the file's imports.
///
/// Not filtered by boundness, as in the Rust frontend: a local shadowing a package inside one
/// function is rare, and reading the shadow as "this definition does not reach that package" is a
/// silent false negative, the expensive direction here.
pub(crate) fn reached_paths(body: Node<'_>, src: &str, imports: &HashMap<String, Arc<str>>) -> Vec<Arc<str>> {
    fn walk(n: Node<'_>, src: &str, imports: &HashMap<String, Arc<str>>, out: &mut Vec<Arc<str>>) {
        let head = match n.kind() {
            "selector_expression" => field(n, "operand").filter(|o| o.kind() == "identifier"),
            "qualified_type" => field(n, "package"),
            _ => None,
        };
        if let Some(path) = head.and_then(|h| imports.get(text(h, src))) {
            out.push(Arc::clone(path));
        }
        for k in kids(n) {
            walk(k, src, imports, out);
        }
    }
    if imports.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    walk(body, src, imports, &mut out);
    out.sort();
    out.dedup();
    out
}
