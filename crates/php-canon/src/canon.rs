//! Structural canonicalization of PHP definitions, written from the owned tree ([`crate::ir`]).
//!
//! The role `c-canon::canon` plays for C: a compact, internally consistent **s-expr** per definition
//! that `difflib-fast` compares for name-gated similarity, and that the cross-name pass `Eq`-checks
//! once locals are alpha-renamed. The tag vocabulary is the other frontends' wherever the construct is
//! the same one (`Func`, `Params`, `Block`, `Assign`, `ExprStmt`, `If`, `While`, `ForC`, `For`,
//! `Switch`, `Case`, `Try`, `Catch`, `Throw`, `Call`, `Method`, `Field`, `Bin`, …), so the engine's
//! s-expr readers — converge's vocabulary split, the unit counter — read PHP the way they already read
//! the others. Two modes, driven by [`Dump::locals`]:
//!
//! * **cluster** (`locals = None`) — names pass through verbatim; the names-preserved canonical the
//!   name-gated pass clusters on.
//! * **xname** (`locals = Some(set)`) — variables are renumbered `_v{n}` by first occurrence and the
//!   top definition's own name is blanked to `_fn`.
//!
//! PHP keeps two namespaces apart that the other languages share: a variable is `$name`, and a
//! function, constant or class is a bare `name`. `$count = count($items)` names two things, so only a
//! variable is ever renamed — a function a body calls is called whatever its locals are named.
//!
//! **Every variable a function names is its own** — PHP has no enclosing scope a body reads without
//! saying so — save `$this`, written `self` as the Go and Rust frontends write a method's receiver, the
//! superglobals (`$_GET`, `$GLOBALS`, …), and what a `global` statement names, which is the program's.
//!
//! The canonicals are written from the owned tree rather than from tree-sitter's because the exact and
//! Type-3 canonicals read the body's normal form ([`crate::normal`]), which rewrites it.
#![allow(
    clippy::too_many_lines, // the stmt/expr matches enumerate the tree's shapes; splitting just scatters one
    clippy::match_same_arms // distinct shapes intentionally share an emission for clarity
)]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;

use dup_defs_core::uses::{site_ctx, Construct, Site, Within};
use dup_defs_core::{quoted, AnalyzedFn, Statement, UNFILED};

use crate::ir::{Arg, Arm, Case, Catch, ClassDecl, ClassMember, Elem, Else, Expr, Func, Jump, Member, Param, Stmt, Type};

/// The variables PHP defines in every scope: they are the program's, not the function's.
pub(crate) const SUPERGLOBALS: &[&str] =
    &["GLOBALS", "_SERVER", "_GET", "_POST", "_FILES", "_COOKIE", "_SESSION", "_REQUEST", "_ENV", "http_response_header"];

/// Whether a variable is the program's rather than the function's: `$this` or a superglobal.
pub(crate) fn is_ambient(var: &str) -> bool {
    var == "this" || SUPERGLOBALS.contains(&var)
}

// ───────────────────────────── traversal ─────────────────────────────

/// A part of a statement: a statement or list of them it holds, or an expression it evaluates.
pub(crate) enum Part<'s> {
    Stmt(&'s Stmt),
    Stmts(&'s [Stmt]),
    Expr(&'s Expr),
}

/// Every part of `s`, in source order — what it tests, what it evaluates, its bodies.
pub(crate) fn each_stmt_part<'s>(s: &'s Stmt, f: &mut dyn FnMut(Part<'s>)) {
    match s {
        Stmt::Expr(e) | Stmt::Throw(e) | Stmt::IncDec { target: e, .. } => f(Part::Expr(e)),
        Stmt::Assign { lhs, rhs, .. } => {
            f(Part::Expr(lhs));
            f(Part::Expr(rhs));
        }
        Stmt::Echo(list) | Stmt::Unset(list) => list.iter().for_each(|e| f(Part::Expr(e))),
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
            init.iter().chain(cond).for_each(|e| f(Part::Expr(e)));
            f(Part::Stmts(body));
            for e in update {
                f(Part::Expr(e));
            }
        }
        Stmt::Foreach { subject, key, value, body } => {
            f(Part::Expr(subject));
            if let Some(k) = key {
                f(Part::Expr(k));
            }
            f(Part::Expr(value));
            f(Part::Stmts(body));
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
        Stmt::Try { body, catches, finally } => {
            f(Part::Stmts(body));
            for c in catches {
                f(Part::Stmts(&c.body));
            }
            if let Some(fin) = finally {
                f(Part::Stmts(fin));
            }
        }
        Stmt::Block(b) => f(Part::Stmts(b)),
        Stmt::Static(vars) => vars.iter().filter_map(|(_, v)| v.as_ref()).for_each(|v| f(Part::Expr(v))),
        Stmt::Global(_) | Stmt::Jump { .. } | Stmt::Label(_) | Stmt::NestedItem | Stmt::Empty | Stmt::Unknown(_) => {}
    }
}

/// What an expression holds that a walk may want: its sub-expressions, and the body of a closure.
pub(crate) enum Child<'e> {
    Expr(&'e Expr),
    Body(&'e [Stmt]),
}

fn args_children<'e>(args: &'e [Arg], f: &mut dyn FnMut(Child<'e>)) {
    for a in args {
        f(Child::Expr(&a.value));
    }
}

fn elems_children<'e>(elems: &'e [Elem], f: &mut dyn FnMut(Child<'e>)) {
    for el in elems {
        match el {
            Elem::Plain(v) => f(Child::Expr(v)),
            Elem::Pair(k, v) => {
                f(Child::Expr(k));
                f(Child::Expr(v));
            }
        }
    }
}

/// Every direct child of an expression, in source order. A closure's parameter defaults are children
/// too; its body is a [`Child::Body`].
pub(crate) fn for_each_child<'e>(e: &'e Expr, f: &mut dyn FnMut(Child<'e>)) {
    match e {
        Expr::Binary { l, r, .. } | Expr::Assign { l, r, .. } | Expr::Comma(l, r) => {
            f(Child::Expr(l));
            f(Child::Expr(r));
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
        | Expr::ClassConst { scope: e, .. } => f(Child::Expr(e)),
        Expr::Cond { c, t, f: alt } => {
            f(Child::Expr(c));
            if let Some(t) = t {
                f(Child::Expr(t));
            }
            f(Child::Expr(alt));
        }
        Expr::Call { f: callee, args } => {
            f(Child::Expr(callee));
            args_children(args, f);
        }
        Expr::Method { recv, name, args, .. } => {
            f(Child::Expr(recv));
            if let Member::Dyn(d) = name {
                f(Child::Expr(d));
            }
            args_children(args, f);
        }
        Expr::StaticCall { scope, name, args } => {
            f(Child::Expr(scope));
            if let Member::Dyn(d) = name {
                f(Child::Expr(d));
            }
            args_children(args, f);
        }
        Expr::Prop { e, name, .. } => {
            f(Child::Expr(e));
            if let Member::Dyn(d) = name {
                f(Child::Expr(d));
            }
        }
        Expr::Index { e, i } => {
            f(Child::Expr(e));
            if let Some(i) = i {
                f(Child::Expr(i));
            }
        }
        Expr::Array(elems) | Expr::List(elems) => elems_children(elems, f),
        Expr::New { class, args } => {
            f(Child::Expr(class));
            args_children(args, f);
        }
        Expr::Closure { params, body, .. } => {
            params.iter().filter_map(|p| p.default.as_ref()).for_each(|d| f(Child::Expr(d)));
            f(Child::Body(body));
        }
        Expr::Arrow { params, body } => {
            params.iter().filter_map(|p| p.default.as_ref()).for_each(|d| f(Child::Expr(d)));
            f(Child::Expr(body));
        }
        Expr::Match { subject, arms } => {
            f(Child::Expr(subject));
            for arm in arms {
                arm.conds.iter().flatten().for_each(|c| f(Child::Expr(c)));
                f(Child::Expr(&arm.value));
            }
        }
        Expr::Yield { key, value } => {
            if let Some(k) = key {
                f(Child::Expr(k));
            }
            if let Some(v) = value {
                f(Child::Expr(v));
            }
        }
        Expr::Interp(parts) | Expr::Shell(parts) => parts.iter().for_each(|p| f(Child::Expr(p))),
        Expr::Var(_) | Expr::Name(_) | Expr::Lit(..) | Expr::Bool(_) | Expr::Null | Expr::AnonClass | Expr::Unknown(_) => {}
    }
}

// ───────────────────────────── bound variables ─────────────────────────────

/// Every variable a function binds: its parameters and every variable its body names, a closure's
/// included — save `$this`, the superglobals, and what a `global` statement names.
pub(crate) fn locals_of(f: &Func) -> HashSet<String> {
    let mut vars = HashSet::new();
    let mut globals = HashSet::new();
    for p in &f.params {
        vars.insert(p.name.clone());
    }
    collect_vars(&f.body, &mut vars, &mut globals);
    vars.retain(|v| !is_ambient(v) && !globals.contains(v));
    vars
}

fn collect_vars(stmts: &[Stmt], vars: &mut HashSet<String>, globals: &mut HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Global(names) => globals.extend(names.iter().cloned()),
            Stmt::Static(list) => vars.extend(list.iter().map(|(n, _)| n.clone())),
            Stmt::Try { catches, .. } => vars.extend(catches.iter().filter_map(|c| c.var.clone())),
            _ => {}
        }
        each_stmt_part(s, &mut |part| match part {
            Part::Stmt(inner) => collect_vars(std::slice::from_ref(inner), vars, globals),
            Part::Stmts(list) => collect_vars(list, vars, globals),
            Part::Expr(e) => expr_vars(e, vars, globals),
        });
    }
}

fn expr_vars(e: &Expr, vars: &mut HashSet<String>, globals: &mut HashSet<String>) {
    match e {
        Expr::Var(v) => {
            vars.insert(v.clone());
        }
        Expr::Closure { params, uses, .. } => {
            vars.extend(params.iter().map(|p| p.name.clone()));
            vars.extend(uses.iter().map(|(n, _)| n.clone()));
        }
        Expr::Arrow { params, .. } => vars.extend(params.iter().map(|p| p.name.clone())),
        _ => {}
    }
    for_each_child(e, &mut |c| match c {
        Child::Expr(x) => expr_vars(x, vars, globals),
        Child::Body(b) => collect_vars(b, vars, globals),
    });
}

/// The names a function reads that are not variables — the functions it calls, the classes it builds,
/// tests against and declares, the constants it reads — each with what kind of name it is, as spelled.
/// What its file's imports may resolve.
pub(crate) fn free_names(f: &Func) -> BTreeSet<(NameKind, String)> {
    let mut out = BTreeSet::new();
    params_names(&f.params, &mut out);
    if let Some(t) = &f.ret {
        type_names(t, &mut |n| {
            out.insert((NameKind::Class, n.to_owned()));
        });
    }
    stmt_names(&f.body, &mut out);
    out
}

fn params_names(params: &[Param], out: &mut BTreeSet<(NameKind, String)>) {
    for p in params {
        if let Some(t) = &p.ty {
            type_names(t, &mut |n| {
                out.insert((NameKind::Class, n.to_owned()));
            });
        }
        if let Some(d) = &p.default {
            expr_names(d, out);
        }
    }
}

/// What a name names: PHP resolves each kind through its own table of imports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum NameKind {
    Class,
    Function,
    Constant,
}

/// Every class a type names (`?Order`, `A|B`), built-in types left out.
pub(crate) fn type_names(t: &Type, f: &mut dyn FnMut(&str)) {
    match t {
        Type::Named(n) => {
            if !is_builtin_type(n) {
                f(n);
            }
        }
        Type::Nullable(inner) => type_names(inner, f),
        Type::Union(list) | Type::Inter(list) => list.iter().for_each(|t| type_names(t, f)),
    }
}

/// The types PHP itself defines, and the names a scope is read through.
pub(crate) fn is_builtin_type(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "array" | "bool" | "callable" | "false" | "float" | "int" | "iterable" | "mixed" | "null" | "object" | "string" | "true"
            | "void" | "never" | "self" | "static" | "parent"
    )
}

fn stmt_names(stmts: &[Stmt], out: &mut BTreeSet<(NameKind, String)>) {
    for s in stmts {
        if let Stmt::Try { catches, .. } = s {
            for c in catches {
                out.extend(c.types.iter().map(|t| (NameKind::Class, t.clone())));
            }
        }
        each_stmt_part(s, &mut |part| match part {
            Part::Stmt(inner) => stmt_names(std::slice::from_ref(inner), out),
            Part::Stmts(list) => stmt_names(list, out),
            Part::Expr(e) => expr_names(e, out),
        });
    }
}

fn expr_names(e: &Expr, out: &mut BTreeSet<(NameKind, String)>) {
    match e {
        Expr::Name(n) => {
            out.insert((NameKind::Constant, n.clone()));
            return;
        }
        Expr::Call { f, .. } => {
            if let Expr::Name(n) = f.as_ref() {
                out.insert((NameKind::Function, n.clone()));
            }
        }
        Expr::Closure { params, .. } | Expr::Arrow { params, .. } => params_names(params, out),
        _ => {}
    }
    if let Some(Expr::Name(n)) = named_class(e) {
        if !is_builtin_type(n) {
            out.insert((NameKind::Class, n.clone()));
        }
    }
    for_each_child(e, &mut |c| match c {
        // A callee or a class named bare is not a constant read.
        Child::Expr(x) if is_named_slot(e, x) => {}
        Child::Expr(x) => expr_names(x, out),
        Child::Body(b) => stmt_names(b, out),
    });
}

/// The class an expression names in a class position: what it builds, the scope it reads a member
/// off, what it tests an instance against.
pub(crate) fn named_class(e: &Expr) -> Option<&Expr> {
    match e {
        Expr::New { class, .. } => Some(class),
        Expr::StaticCall { scope, .. } | Expr::StaticProp { scope, .. } | Expr::ClassConst { scope, .. } => Some(scope),
        Expr::Binary { op, r, .. } if op == "instanceof" => Some(r),
        _ => None,
    }
}

/// Whether `child` is the bare name `parent` calls or names a class by — not a constant it reads.
fn is_named_slot(parent: &Expr, child: &Expr) -> bool {
    if !matches!(child, Expr::Name(_)) {
        return false;
    }
    match parent {
        Expr::Call { f, .. } => std::ptr::eq(f.as_ref(), child),
        _ => named_class(parent).is_some_and(|c| std::ptr::eq(c, child)),
    }
}

// ───────────────────────────── s-expr emitter ─────────────────────────────

pub(crate) struct Dump<'a> {
    /// `None` = cluster mode (names verbatim); `Some(set)` = xname mode (these variables → `_v{n}`).
    locals: Option<&'a HashSet<String>>,
    /// Scope mode: names (not variables) the file introduced, renamed like variables so the widest rung
    /// of the erasure ladder does not hold two copies apart by their own names.
    erased: Option<&'a HashSet<String>>,
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
    /// `Some` while writing the Type-3 lines: declared type → `_T{n}`. See [`Dump::type_name`].
    tmap: Option<HashMap<String, u32>>,
    /// Every type reference turned into a slot, in order — [`AnalyzedFn::type_refs`].
    type_refs: Vec<String>,
    /// Use-site mode: the definition whose use is being written. A name naming it becomes the anchor
    /// `_t0`, and what is read off it — a constant, a static member, a named constructor argument — is
    /// marked `#name#` for the profile to number.
    anchor: Option<&'a str>,
    /// Hunt mode: the names that count as anchors. Every one the walk meets is kept in `hunted`.
    known: Option<&'a HashSet<&'a str>>,
    hunted: Vec<String>,
    /// Hunt mode: the methods called (`.name`), for the co-occurrence channel of the `use` lens.
    calls: BTreeSet<String>,
}

impl<'a> Dump<'a> {
    pub(crate) fn new(locals: Option<&'a HashSet<String>>) -> Self {
        Self {
            locals,
            erased: None,
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

    /// A dump that notes which of `known` a line mentions, and the methods it calls.
    fn hunting(bound: &'a HashSet<String>, known: &'a HashSet<&'a str>) -> Self {
        Self { known: Some(known), tmap: Some(HashMap::new()), ..Self::new(Some(bound)) }
    }

    /// A dump that writes a line as a use site of `anchor`.
    fn anchoring(bound: &'a HashSet<String>, anchor: &'a str) -> Self {
        Self { anchor: Some(anchor), tmap: Some(HashMap::new()), ..Self::new(Some(bound)) }
    }

    /// A reference to a name the file or the program declares: noted when it is hunted for, and
    /// whether it is the anchor. A qualified name is read by its last segment, as declared.
    fn mention(&mut self, name: &str) -> bool {
        let bare = name.rsplit('\\').next().unwrap_or(name);
        if let Some(known) = self.known {
            if known.contains(bare) && !self.hunted.iter().any(|h| h == bare) {
                self.hunted.push(bare.to_owned());
            }
        }
        self.anchor == Some(bare)
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

    /// A declared type's name: its `_T{n}` slot while writing the Type-3 lines, as spelled otherwise.
    ///
    /// 🔴 Only the Type-3 lines slot types — the rule of every frontend whose language checks them. A
    /// PHP type declaration is enforced at the call: `int $n` coerces or throws, so two bodies differing
    /// only in their declared types do not accept the same calls, and the exact pass keeps them apart;
    /// the near-copy pass reads a type as vocabulary like any other.
    fn type_name(&mut self, name: &str) -> String {
        if self.mention(name) {
            return "_t0".to_owned();
        }
        let Some(tmap) = self.tmap.as_mut() else { return quoted(name) };
        self.type_refs.push(name.to_owned());
        dup_defs_core::type_slot(tmap, name)
    }

    /// A variable: `self` for `$this`, its slot for a local in xname mode, its name otherwise.
    fn var(&mut self, name: &str) -> String {
        if name == "this" {
            return self.node("Path", &[quoted("self")]);
        }
        let renamed = self.rename(name);
        self.node("Path", &[quoted(&renamed)])
    }

    /// A function, constant or class named in a body: never renamed as a variable is, though the scope
    /// rung erases the file's own.
    fn name(&mut self, name: &str) -> String {
        if self.mention(name) {
            return self.node("Path", &[quoted("_t0")]);
        }
        let spelled = match self.erased {
            Some(erased) if erased.contains(name.rsplit('\\').next().unwrap_or(name)) => {
                dup_defs_core::alpha_rename(&mut self.map, Some(erased), name.rsplit('\\').next().unwrap_or(name))
            }
            _ => name.to_owned(),
        };
        self.node("Path", &[quoted(&spelled)])
    }

    /// A binding site: `Bind(_vN)`.
    fn bind(&mut self, name: &str) -> String {
        let renamed = self.rename(name);
        self.node("Bind", &[quoted(&renamed)])
    }

    // ────── functions ──────

    /// `is_top` blanks the definition's own name in xname mode.
    pub(crate) fn func(&mut self, name: &str, f: &Func, with_body: bool, is_top: bool) -> String {
        let name = if is_top && self.locals.is_some() && !self.blanked {
            self.blanked = true;
            "_fn".to_owned()
        } else {
            name.to_owned()
        };
        let params = self.params(&f.params);
        let ret = self.opt_ty(f.ret.as_ref());
        let body = if with_body { self.block(&f.body) } else { String::new() };
        self.node("Func", &[quoted(&name), params, ret, body])
    }

    /// A parameter list, one `Param(Bind, type, default)` per parameter; a variadic one is `Variadic`,
    /// one taken by reference `RefParam`, a promoted constructor parameter carries its visibility.
    fn params(&mut self, params: &[Param]) -> String {
        let mut items = Vec::new();
        for p in params {
            let bound = self.bind(&p.name);
            let ty = self.opt_ty(p.ty.as_ref());
            let default = p.default.as_ref().map_or_else(String::new, |d| self.expr(d));
            let tag = match (p.variadic, p.by_ref) {
                (true, _) => "Variadic",
                (false, true) => "RefParam",
                (false, false) => "Param",
            };
            let mut fields = vec![bound, ty, default];
            if let Some(vis) = &p.promoted {
                fields.push(quoted(vis));
            }
            items.push(self.node(tag, &fields));
        }
        self.list("Params", &items)
    }

    /// The header of the Type-3 lines, in the language's own order: `function _fn(Ty(_T0, ) _v0): Ty(_T1, )`.
    ///
    /// 🔴 Not the `Func(…)` node the stream opens with. The Type-3 pass reads every function's first
    /// line as context and weighs it like any other; an s-expr header spells the parse tree's grammar
    /// (`Params`, `Param`, `Bind`), which only headers spell, so every one of those tokens was rare and
    /// the header outweighed the body of every short function (the Rust frontend's finding).
    fn compact_header(&mut self, f: &Func) -> String {
        let mut items = Vec::new();
        for p in &f.params {
            let ty = self.opt_ty(p.ty.as_ref());
            let n = self.rename(&p.name);
            let mut item = String::new();
            if !ty.is_empty() {
                item.push_str(&ty);
                item.push(' ');
            }
            if p.variadic {
                item.push_str("...");
            }
            if p.by_ref {
                item.push('&');
            }
            item.push_str(&n);
            if let Some(d) = &p.default {
                let d = self.expr(d);
                let _ = write!(item, " = {d}");
            }
            items.push(item);
        }
        let ret = self.opt_ty(f.ret.as_ref());
        let mut out = String::new();
        let _ = write!(out, "function _fn({})", items.join(", "));
        if !ret.is_empty() {
            let _ = write!(out, ": {ret}");
        }
        out
    }

    // ────── statements / blocks ──────

    pub(crate) fn block(&mut self, stmts: &[Stmt]) -> String {
        let items: Vec<String> = stmts.iter().map(|s| self.stmt(s)).collect();
        self.list("Block", &items)
    }

    fn exprs(&mut self, list: &[Expr]) -> String {
        let mut items: Vec<String> = list.iter().map(|e| self.expr(e)).collect();
        match items.len() {
            0 => String::new(),
            1 => items.pop().unwrap_or_default(),
            _ => self.list("Seq", &items),
        }
    }

    pub(crate) fn stmt(&mut self, s: &Stmt) -> String {
        match s {
            Stmt::Expr(e) => {
                let e = self.expr(e);
                self.node("ExprStmt", &[e])
            }
            Stmt::Assign { op, lhs, rhs } => {
                let left = self.expr(lhs);
                let right = self.expr(rhs);
                match op.as_str() {
                    "=" => self.node("Assign", &[left, right]),
                    "=&" => self.node("AssignRef", &[left, right]),
                    _ => self.node("Bin", &[left, quoted(op), right]),
                }
            }
            Stmt::IncDec { inc, target } => {
                let e = self.expr(target);
                self.node(if *inc { "Inc" } else { "Dec" }, &[e])
            }
            Stmt::Echo(list) => {
                let items: Vec<String> = list.iter().map(|e| self.expr(e)).collect();
                self.node("Echo", &items)
            }
            Stmt::Return(value) => {
                let value = value.as_ref().map_or_else(String::new, |v| self.expr(v));
                self.node("Return", &[value])
            }
            Stmt::Throw(e) => {
                let e = self.expr(e);
                self.node("Throw", &[e])
            }
            Stmt::Jump { kind, arg } => {
                let tag = match kind {
                    Jump::Break => "Break",
                    Jump::Continue => "Continue",
                    Jump::Goto => "Goto",
                };
                let arg = arg.as_deref().map_or_else(String::new, quoted);
                self.node(tag, &[arg])
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
                let (init, cond, update) = (self.exprs(init), self.exprs(cond), self.exprs(update));
                let body = self.block(body);
                self.node("ForC", &[init, cond, update, body])
            }
            Stmt::Foreach { subject, key, value, body } => {
                let (pat, iter) = self.foreach_parts(subject, key.as_ref(), value);
                let body = self.block(body);
                self.node("For", &[pat, iter, body])
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
            Stmt::Try { body, catches, finally } => {
                let body = self.block(body);
                let mut items = vec![body];
                for c in catches {
                    let head = self.catch_head(c);
                    let b = self.block(&c.body);
                    items.push(self.node("Catch", &[head, b]));
                }
                let fin = finally.as_deref().map_or_else(String::new, |f| self.block(f));
                items.push(fin);
                self.node("Try", &items)
            }
            Stmt::Block(b) => {
                let body = self.block(b);
                self.node("BlockExpr", &[body])
            }
            Stmt::Global(names) => {
                let items: Vec<String> = names.iter().map(|n| quoted(n)).collect();
                self.node("Global", &items)
            }
            Stmt::Static(vars) => {
                let items: Vec<String> = vars
                    .iter()
                    .map(|(n, v)| {
                        let b = self.bind(n);
                        let v = v.as_ref().map_or_else(String::new, |v| self.expr(v));
                        self.node("Let", &[b, v])
                    })
                    .collect();
                self.list("StaticVars", &items)
            }
            Stmt::Unset(list) => {
                let items: Vec<String> = list.iter().map(|e| self.expr(e)).collect();
                self.node("Unset", &items)
            }
            Stmt::Label(l) => self.node("Labeled", &[quoted(l)]),
            Stmt::NestedItem => self.node("NestedItem", &[]),
            Stmt::Empty => self.node("Empty", &[]),
            Stmt::Unknown(kind) => self.node(&format!("Unknown_{kind}"), &[]),
        }
    }

    /// A `foreach`'s binding and what it iterates: `$k => $v` is `KV(k, v)`.
    fn foreach_parts(&mut self, subject: &Expr, key: Option<&Expr>, value: &Expr) -> (String, String) {
        let iter = self.expr(subject);
        let pat = match key {
            Some(k) => {
                let k = self.expr(k);
                let v = self.expr(value);
                self.node("KV", &[k, v])
            }
            None => self.expr(value),
        };
        (pat, iter)
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

    /// What a `catch` catches and binds: `Types(Ty(…), …)` and the variable.
    fn catch_head(&mut self, c: &Catch) -> String {
        let types: Vec<String> = c
            .types
            .iter()
            .map(|t| {
                let n = self.type_name(t);
                self.node("Ty", &[n, String::new()])
            })
            .collect();
        let types = self.list("Types", &types);
        let var = c.var.as_deref().map_or_else(String::new, |v| self.bind(v));
        format!("{types}, {var}")
    }

    // ────── compound-statement headers ──────
    //
    // A compound statement's own line — what it tests, its body elided. Shared by the statement
    // stream and the use-site walk, which must write a header the same way.

    /// The header of a loop or a `switch`; `None` for any other statement.
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
                let (init, cond, update) = (self.exprs(init), self.exprs(cond), self.exprs(update));
                self.node("ForC", &[init, cond, update])
            }
            Stmt::Foreach { subject, key, value, .. } => {
                let (pat, iter) = self.foreach_parts(subject, key.as_ref(), value);
                self.node("For", &[pat, iter])
            }
            Stmt::Switch { value, .. } => {
                let value = self.expr(value);
                self.node("Switch", &[value])
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

    /// An expression — and, when units are being collected, kept as one if it is a proper part of its
    /// line: the statement's own expression is the line, already a unit. Parentheses are not a level of
    /// their own; what they wrap is the part.
    pub(crate) fn expr(&mut self, e: &Expr) -> String {
        if matches!(e, Expr::Paren(_)) {
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

    fn opt_expr(&mut self, e: Option<&Expr>) -> String {
        e.map_or_else(String::new, |e| self.expr(e))
    }

    fn member_name(&mut self, m: &Member, on_anchor: bool) -> String {
        match m {
            Member::Name(n) if on_anchor => quoted(&format!("#{n}#")),
            Member::Name(n) => quoted(n),
            Member::Dyn(e) => {
                let e = self.expr(e);
                self.node("DynName", &[e])
            }
        }
    }

    /// A call's arguments; a named one is `Kw('name', value)`, its name marked when the call constructs
    /// the anchor.
    fn args(&mut self, args: &[Arg], on_anchor: bool) -> Vec<String> {
        args.iter()
            .map(|a| {
                let v = self.expr(&a.value);
                match &a.name {
                    Some(n) => {
                        let key = if on_anchor { quoted(&format!("#{n}#")) } else { quoted(n) };
                        self.node("Kw", &[key, v])
                    }
                    None => v,
                }
            })
            .collect()
    }

    fn elements(&mut self, elems: &[Elem]) -> Vec<String> {
        elems
            .iter()
            .map(|el| match el {
                Elem::Plain(v) => self.expr(v),
                Elem::Pair(k, v) => {
                    let k = self.expr(k);
                    let v = self.expr(v);
                    self.node("KV", &[k, v])
                }
            })
            .collect()
    }

    /// The scope of `Scope::…`: a class is a name, `self` / `static` / `parent` the class the body is in.
    fn scope(&mut self, scope: &Expr) -> (String, bool) {
        match scope {
            Expr::Name(n) => (self.name(n), self.is_anchor(n)),
            other => (self.expr(other), false),
        }
    }

    /// Whether a name is the anchor of the use site being written.
    fn is_anchor(&self, name: &str) -> bool {
        self.anchor.is_some() && self.anchor == Some(name.rsplit('\\').next().unwrap_or(name))
    }

    fn expr_inner(&mut self, e: &Expr) -> String {
        match e {
            Expr::Var(v) => self.var(v),
            Expr::DynVar(inner) => {
                let inner = self.expr(inner);
                self.node("DynVar", &[inner])
            }
            Expr::Name(n) => self.name(n),
            Expr::Lit(tag, text) => self.node(tag, &[quoted(text)]),
            Expr::Interp(parts) => {
                let items: Vec<String> = parts.iter().map(|p| self.expr(p)).collect();
                self.node("Interp", &items)
            }
            Expr::Bool(b) => self.node("Bool", &[quoted(if *b { "true" } else { "false" })]),
            Expr::Null => self.node("Nil", &[]),
            Expr::Binary { op, l, r } => {
                let l = self.expr(l);
                let r = if op == "instanceof" {
                    match r.as_ref() {
                        Expr::Name(n) => {
                            let n = self.type_name(n);
                            self.node("Ty", &[n, String::new()])
                        }
                        other => self.expr(other),
                    }
                } else {
                    self.expr(r)
                };
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
                let t = self.opt_expr(t.as_deref());
                let f = self.expr(f);
                self.node("Cond", &[c, t, f])
            }
            Expr::Comma(l, r) => {
                let l = self.expr(l);
                let r = self.expr(r);
                self.node("Seq", &[l, r])
            }
            Expr::Paren(inner) => self.expr(inner),
            Expr::Call { f, args } => {
                let callee = self.expr(f);
                let mut all = vec![callee];
                all.extend(self.args(args, false));
                self.node("Call", &all)
            }
            Expr::Method { recv, name, args, nullsafe } => {
                let recv = self.expr(recv);
                if let (Some(_), Member::Name(n)) = (self.known, name) {
                    self.calls.insert(format!(".{n}"));
                }
                let name = self.member_name(name, false);
                let mut all = vec![recv, name];
                all.extend(self.args(args, false));
                self.node(if *nullsafe { "NullsafeMethod" } else { "Method" }, &all)
            }
            Expr::StaticCall { scope, name, args } => {
                let (scope, on_anchor) = self.scope(scope);
                let name = self.member_name(name, on_anchor);
                let mut all = vec![scope, name];
                all.extend(self.args(args, false));
                self.node("StaticCall", &all)
            }
            Expr::Prop { e, name, nullsafe } => {
                let base = self.expr(e);
                let name = self.member_name(name, false);
                self.node(if *nullsafe { "NullsafeField" } else { "Field" }, &[base, name])
            }
            Expr::StaticProp { scope, name } => {
                let (scope, on_anchor) = self.scope(scope);
                let name = if on_anchor { quoted(&format!("#{name}#")) } else { quoted(name) };
                self.node("StaticField", &[scope, name])
            }
            Expr::ClassConst { scope, name } => {
                let (scope, on_anchor) = self.scope(scope);
                let name = if on_anchor && name != "class" { quoted(&format!("#{name}#")) } else { quoted(name) };
                self.node("ClassConst", &[scope, name])
            }
            Expr::Index { e, i } => {
                let base = self.expr(e);
                let idx = self.opt_expr(i.as_deref());
                self.node("Index", &[base, idx])
            }
            Expr::Array(elems) => {
                let items = self.elements(elems);
                self.node("Arr", &items)
            }
            Expr::List(elems) => {
                let items = self.elements(elems);
                self.node("ArrTarget", &items)
            }
            Expr::New { class, args } => {
                let (class, on_anchor) = self.scope(class);
                let mut all = vec![class];
                all.extend(self.args(args, on_anchor));
                self.node("New", &all)
            }
            Expr::AnonClass => self.node("AnonClass", &[]),
            Expr::Closure { params, uses, body } => {
                let params = self.params(params);
                let uses: Vec<String> = uses
                    .iter()
                    .map(|(n, by_ref)| {
                        let v = self.var(n);
                        if *by_ref {
                            self.node("Ref", &[v])
                        } else {
                            v
                        }
                    })
                    .collect();
                let uses = self.list("Uses", &uses);
                let body = self.block(body);
                self.node("Closure", &[params, uses, body])
            }
            Expr::Arrow { params, body } => {
                let params = self.params(params);
                let body = self.expr(body);
                self.node("Arrow", &[params, body])
            }
            Expr::Match { subject, arms } => {
                let subject = self.expr(subject);
                let mut items = vec![subject];
                for Arm { conds, value } in arms {
                    let value = self.expr(value);
                    let arm = match conds {
                        Some(conds) => {
                            let conds: Vec<String> = conds.iter().map(|c| self.expr(c)).collect();
                            let conds = self.list("Conds", &conds);
                            self.node("Arm", &[conds, value])
                        }
                        None => self.node("Default", &[value]),
                    };
                    items.push(arm);
                }
                self.node("Match", &items)
            }
            Expr::Cast { ty, e } => {
                let e = self.expr(e);
                self.node("Cast", &[e, quoted(ty)])
            }
            Expr::Silence(e) => {
                let e = self.expr(e);
                self.node("Silence", &[e])
            }
            Expr::Clone(e) => {
                let e = self.expr(e);
                self.node("Clone", &[e])
            }
            Expr::Print(e) => {
                let e = self.expr(e);
                self.node("Print", &[e])
            }
            Expr::Include { kind, e } => {
                let e = self.expr(e);
                self.node("Include", &[quoted(kind), e])
            }
            Expr::Yield { key, value } => {
                let k = self.opt_expr(key.as_deref());
                let v = self.opt_expr(value.as_deref());
                self.node("Yield", &[k, v])
            }
            Expr::YieldFrom(e) => {
                let e = self.expr(e);
                self.node("YieldFrom", &[e])
            }
            Expr::Throw(e) => {
                let e = self.expr(e);
                self.node("Throw", &[e])
            }
            Expr::Shell(parts) => {
                let items: Vec<String> = parts.iter().map(|p| self.expr(p)).collect();
                self.node("Shell", &items)
            }
            Expr::Spread(e) => {
                let e = self.expr(e);
                self.node("Spread", &[e])
            }
            Expr::Unknown(kind) => self.node(&format!("Unknown_{kind}"), &[]),
        }
    }

    // ────── types ──────

    pub(crate) fn ty(&mut self, t: &Type) -> String {
        match t {
            Type::Named(name) => {
                let name = if is_builtin_type(name) { quoted(name) } else { self.type_name(name) };
                self.node("Ty", &[name, String::new()])
            }
            Type::Nullable(inner) => {
                let inner = self.ty(inner);
                self.node("TyOpt", &[inner])
            }
            Type::Union(list) => {
                let items: Vec<String> = list.iter().map(|t| self.ty(t)).collect();
                self.node("TyUnion", &items)
            }
            Type::Inter(list) => {
                let items: Vec<String> = list.iter().map(|t| self.ty(t)).collect();
                self.node("TyInter", &items)
            }
        }
    }

    fn opt_ty(&mut self, t: Option<&Type>) -> String {
        t.map_or_else(String::new, |t| self.ty(t))
    }
}

/// Whether an expression is a Type-3 unit of its own: a node with structure — a call, a member read,
/// an operator application, a closure, an array with elements — not a bare name or a literal, which
/// carry no agreement by themselves. The rule every frontend keeps.
fn is_unit(e: &Expr) -> bool {
    match e {
        Expr::Call { .. }
        | Expr::Method { .. }
        | Expr::StaticCall { .. }
        | Expr::Prop { .. }
        | Expr::StaticProp { .. }
        | Expr::ClassConst { .. }
        | Expr::Index { .. }
        | Expr::Binary { .. }
        | Expr::Unary { .. }
        | Expr::Update { .. }
        | Expr::Assign { .. }
        | Expr::Cond { .. }
        | Expr::Cast { .. }
        | Expr::New { .. }
        | Expr::Closure { .. }
        | Expr::Arrow { .. }
        | Expr::Match { .. }
        | Expr::Interp(_) => true,
        Expr::Array(elems) | Expr::List(elems) => !elems.is_empty(),
        _ => false,
    }
}

// ───────────────────────────── declaration canonicals ─────────────────────────────

/// Names-preserved structural canonical of a class, trait, enum or interface — the cluster pass's
/// input for `classes` and `interfaces`: its header and every member, method bodies included.
#[must_use]
pub(crate) fn class_canon(c: &ClassDecl) -> String {
    let mut d = Dump::new(None);
    let modifiers: Vec<String> = c.modifiers.iter().map(|m| quoted(m)).collect();
    let modifiers = d.list("Mods", &modifiers);
    let extends: Vec<String> = c.extends.iter().map(|n| d.name(n)).collect();
    let extends = d.list("Extends", &extends);
    let implements: Vec<String> = c.implements.iter().map(|n| d.name(n)).collect();
    let implements = d.list("Implements", &implements);
    let backing = c.backing.as_deref().map_or_else(String::new, quoted);
    let members: Vec<String> = c.members.iter().map(|m| member_canon(&mut d, m)).collect();
    let members = d.list("Members", &members);
    let tag = match c.kind {
        "trait" => "Trait",
        "enum" => "Enum",
        "interface" => "Interface",
        _ => "Class",
    };
    d.node(tag, &[quoted(&c.name), modifiers, extends, implements, backing, members])
}

fn member_canon(d: &mut Dump<'_>, m: &ClassMember) -> String {
    match m {
        ClassMember::Const { name, value } => {
            let v = d.expr(value);
            d.node("Const", &[quoted(name), v])
        }
        ClassMember::Prop { modifiers, ty, name, default } => {
            let mods: Vec<String> = modifiers.iter().map(|m| quoted(m)).collect();
            let mods = d.list("Mods", &mods);
            let ty = d.opt_ty(ty.as_ref());
            let default = d.opt_expr(default.as_ref());
            d.node("Prop", &[quoted(name), mods, ty, default])
        }
        ClassMember::Method { modifiers, name, func, sig } => {
            let mods: Vec<String> = modifiers.iter().map(|m| quoted(m)).collect();
            let mods = d.list("Mods", &mods);
            let f = d.func(name, func.as_ref().unwrap_or(sig), func.is_some(), false);
            d.node("MethodDef", &[mods, f])
        }
        ClassMember::TraitUse(names) => {
            let items: Vec<String> = names.iter().map(|n| d.name(n)).collect();
            d.node("UseTrait", &items)
        }
        ClassMember::Case { name, value } => {
            let v = d.opt_expr(value.as_ref());
            d.node("Variant", &[quoted(name), v])
        }
    }
}

// ───────────────────────────── function analysis ─────────────────────────────

/// Every canonical form of one function or method.
///
/// The cluster canonical and the statement stream read the body as written: the name-gated pass
/// compares what is spelled, and converge reads the stream to say where two definitions stop agreeing
/// — a rewritten line would point the reader at code that is not there. The Type-3 lines read the
/// normal form that keeps every statement, the exact canonical the one that also inlines single-use
/// temporaries ([`crate::normal`]). `facets` false skips the statement stream: a run that never reads
/// facets need not build it.
pub(crate) fn analyze(name: &str, f: &Func, facets: bool) -> AnalyzedFn {
    let cluster = Dump::new(None).func(name, f, true, true);

    let statements = if facets {
        let bound = locals_of(f);
        let mut dump = Dump::new(Some(&bound));
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
        let mut dump = Dump::new(Some(&bound));
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
    let mut xd = Dump::new(Some(&exact_bound));
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
/// `Switch(value)`, `For(pat, iter)`, `Try()`, `Catch(types)`) with its body one level deeper — the
/// shape every other frontend produces, so the languages' streams compare.
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
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } | Stmt::Foreach { body, .. } => {
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
            Stmt::Try { body, catches, finally } => {
                let head = self.dump.node("Try", &[]);
                self.push(depth, head);
                self.stmts(body, depth + 1);
                for c in catches {
                    let types = self.dump.catch_head(c);
                    let head = self.dump.node("Catch", &[types]);
                    self.push(depth, head);
                    self.stmts(&c.body, depth + 1);
                }
                if let Some(fin) = finally {
                    let head = self.dump.node("Finally", &[]);
                    self.push(depth, head);
                    self.stmts(fin, depth + 1);
                }
            }
            Stmt::Block(b) => {
                let head = self.dump.node("Block", &[]);
                self.push(depth, head);
                self.stmts(b, depth + 1);
            }
            Stmt::Empty => {}
            _ => {
                let line = self.dump.stmt(s);
                self.push(depth, line);
            }
        }
    }

    /// `if` / `elseif` / `else`: an `elseif` is a sibling of its `else`, not a level deeper — what the
    /// source means, and what Python's `elif` and the other frontends' chains render as.
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
    let locals = locals_of(f);
    let mut d = Dump::new(Some(&locals));
    d.erased = Some(file_names);
    d.func("_fn", f, true, true)
}

/// Every use site in one file — the PHP half of the `use` lens ([`dup_defs_core::uses`]): each
/// statement that mentions a known name, written with that name as the anchor. A site is the
/// statement's own line — a compound statement's header (`If(cond)`, `Switch(value)`, a case's value),
/// not its body, whose statements are sites of their own — as the statement stream writes it, and
/// inside a function its variables rename as they do there. The file's own top-level code is one more
/// body.
pub(crate) fn collect_sites(funcs: &[Func], known: &HashSet<&str>) -> Vec<Site> {
    let mut walk = UseWalk { known, out: Vec::new() };
    for f in funcs {
        walk.function(f);
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
    /// own variables.
    fn function(&mut self, f: &Func) {
        let locals = locals_of(f);
        self.site(&locals, "plain", &[], |d| Some(d.compact_header(f)));
        self.block(&locals, &f.body, Within::default());
    }

    /// A block's statements, each with the methods called beside it — the co-occurrence channel.
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
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } | Stmt::Foreach { body, .. } => {
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
            Stmt::Try { body, catches, finally } => {
                let inside = Within { in_try: true, ..within };
                self.block(bound, body, inside);
                let ctx = site_ctx(Construct::Try, within);
                for c in catches {
                    self.site(bound, ctx, neighbours, |d| {
                        let head = d.catch_head(c);
                        Some(d.node("Catch", &[head]))
                    });
                    self.block(bound, &c.body, inside);
                }
                if let Some(fin) = finally {
                    self.block(bound, fin, inside);
                }
            }
            Stmt::Block(b) => self.block(bound, b, within),
            Stmt::Empty => {}
            _ => self.site(bound, plain, neighbours, |d| use_line(d, s)),
        }
    }
}

/// A statement's own line, as a use site sees it: a compound statement's header, any other statement
/// whole — `None` for a bare block or a `try`, whose statements are walked.
fn use_line(d: &mut Dump<'_>, s: &Stmt) -> Option<String> {
    match s {
        Stmt::If { cond, .. } => {
            let cond = d.expr(cond);
            Some(d.node("If", &[cond]))
        }
        Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. } | Stmt::Foreach { .. } | Stmt::Switch { .. } => d.header(s),
        Stmt::Block(_) | Stmt::Empty | Stmt::Try { .. } => None,
        _ => Some(d.stmt(s)),
    }
}
