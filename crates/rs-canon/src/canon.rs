//! Structural canonicalization of Rust definitions over the `syn` AST.
//!
//! Mirrors the role `py-canon::canon` / `ts-canon::canon` play: produce a compact, internally
//! consistent **s-expr** per definition (node name + relevant child fields) that `difflib-fast`
//! compares for name-gated similarity, and that the cross-name pass `Eq`-checks once locals are
//! alpha-renamed. Two modes, driven by [`Dump::locals`]:
//!
//! * **cluster** (`locals = None`) — identifiers pass through verbatim; the names-preserved
//!   canonical the name-gated pass clusters on.
//! * **xname** (`locals = Some(set)`) — value bindings (fn params, `let`, `for`/`if let`/
//!   `while let`/`match` arm patterns) are renumbered to `_v{n}` by first occurrence and the top
//!   def's own name is blanked to `_fn`, so `fn add(a,b){a+b}` alpha-equals `fn plus(x,y){x+y}`.
//!
//! A method's `self`/`&self`/`&mut self` receiver is dropped from the emitted parameter list
//! (the analog of Python's `self` strip / TypeScript's `this`), so a method's canonical lines up
//! with an equivalent free function for the cross-name pass. Type annotations are summarized to a
//! structural tag (the path's last segment + generic args); we don't rename type-level generics.
//! Long-tail / `#[non_exhaustive]` AST variants emit as `Unknown_<Kind>` — deterministic for any
//! input and visible in `--calibrate` for the next round of tuning.
#![allow(
    clippy::too_many_lines, // the expr/stmt/pat/type matches enumerate syn variants; splitting just scatters one shape
    clippy::match_same_arms, // distinct variants intentionally share an emission for clarity
    clippy::needless_pass_by_value // emitter helpers take owned Vec<String> to consume one allocation
)]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;

use dup_defs_core::uses::{site_ctx, Construct, Site, Within};
use dup_defs_core::{quoted, Statement, UNFILED};

use syn::punctuated::Punctuated;
use syn::{
    BinOp, Block, Expr, FnArg, Generics, ImplItem, ImplItemFn, Item, ItemEnum, ItemFn, ItemStruct, ItemTrait,
    ItemUnion, Lit, Member, Pat, ReturnType, Signature, Stmt, Token, TraitItem, TraitItemFn, Type, UnOp,
};

/// `(cluster_canonical, xname_canonical, type3_lines, node_count)` — the analysis tuple the scan
/// reads to build a callable `Def`'s cluster canonical + `Analysis`.
pub use dup_defs_core::AnalyzedFn;

// ───────────────────────────── bound-locals collector ─────────────────────────────

/// Collect value bindings introduced anywhere in the callable — the rename set for xname mode.
/// Mirrors `py-canon::Collect` / `ts-canon::Collect`: top fn params + `let` patterns + loop /
/// `if let` / `while let` / `match` arm patterns + their nested blocks. Nested closures' and
/// nested `fn`s' *params* are NOT collected (only the top callable's), matching the other
/// frontends.
#[derive(Default)]
struct Collect {
    bound: HashSet<String>,
}

impl Collect {
    fn add_pat(&mut self, pat: &Pat) {
        match pat {
            Pat::Ident(pi) => {
                self.bound.insert(pi.ident.to_string());
                if let Some((_, sub)) = &pi.subpat {
                    self.add_pat(sub);
                }
            }
            Pat::Reference(r) => self.add_pat(&r.pat),
            Pat::Tuple(t) => t.elems.iter().for_each(|p| self.add_pat(p)),
            Pat::TupleStruct(ts) => ts.elems.iter().for_each(|p| self.add_pat(p)),
            Pat::Slice(s) => s.elems.iter().for_each(|p| self.add_pat(p)),
            Pat::Or(o) => o.cases.iter().for_each(|p| self.add_pat(p)),
            Pat::Paren(p) => self.add_pat(&p.pat),
            Pat::Type(t) => self.add_pat(&t.pat),
            Pat::Struct(s) => {
                for f in &s.fields {
                    self.add_pat(&f.pat);
                }
            }
            _ => {}
        }
    }

    fn add_inputs(&mut self, sig: &Signature) {
        for input in &sig.inputs {
            if let FnArg::Typed(pt) = input {
                self.add_pat(&pt.pat);
            }
            // FnArg::Receiver (self) is not a renameable binding.
        }
    }

    fn visit_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.visit_stmt(stmt);
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Local(local) => {
                self.add_pat(&local.pat);
                if let Some(init) = &local.init {
                    self.visit_expr(&init.expr);
                    if let Some((_, div)) = &init.diverge {
                        self.visit_expr(div);
                    }
                }
            }
            Stmt::Expr(e, _) => self.visit_expr(e),
            // Nested items (fn/struct/...) introduce their own scope; their inner bindings are
            // not the top callable's locals.
            Stmt::Item(_) | Stmt::Macro(_) => {}
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Let(l) => {
                self.add_pat(&l.pat);
                self.visit_expr(&l.expr);
            }
            Expr::ForLoop(f) => {
                self.add_pat(&f.pat);
                self.visit_expr(&f.expr);
                self.visit_block(&f.body);
            }
            Expr::While(w) => {
                self.visit_expr(&w.cond);
                self.visit_block(&w.body);
            }
            Expr::Loop(l) => self.visit_block(&l.body),
            Expr::If(i) => {
                self.visit_expr(&i.cond);
                self.visit_block(&i.then_branch);
                if let Some((_, e)) = &i.else_branch {
                    self.visit_expr(e);
                }
            }
            Expr::Match(m) => {
                self.visit_expr(&m.expr);
                for arm in &m.arms {
                    self.add_pat(&arm.pat);
                    if let Some((_, g)) = &arm.guard {
                        self.visit_expr(g);
                    }
                    self.visit_expr(&arm.body);
                }
            }
            Expr::Block(b) => self.visit_block(&b.block),
            Expr::Unsafe(u) => self.visit_block(&u.block),
            Expr::Async(a) => self.visit_block(&a.block),
            Expr::TryBlock(t) => self.visit_block(&t.block),
            Expr::Paren(p) => self.visit_expr(&p.expr),
            Expr::Group(g) => self.visit_expr(&g.expr),
            Expr::Reference(r) => self.visit_expr(&r.expr),
            Expr::Unary(u) => self.visit_expr(&u.expr),
            Expr::Binary(b) => {
                self.visit_expr(&b.left);
                self.visit_expr(&b.right);
            }
            Expr::Assign(a) => {
                self.visit_expr(&a.left);
                self.visit_expr(&a.right);
            }
            Expr::Return(r) => {
                if let Some(e) = &r.expr {
                    self.visit_expr(e);
                }
            }
            Expr::Call(c) => {
                self.visit_expr(&c.func);
                c.args.iter().for_each(|a| self.visit_expr(a));
            }
            Expr::MethodCall(m) => {
                self.visit_expr(&m.receiver);
                m.args.iter().for_each(|a| self.visit_expr(a));
            }
            Expr::Field(f) => self.visit_expr(&f.base),
            Expr::Index(i) => {
                self.visit_expr(&i.expr);
                self.visit_expr(&i.index);
            }
            Expr::Try(t) => self.visit_expr(&t.expr),
            Expr::Await(a) => self.visit_expr(&a.base),
            Expr::Cast(c) => self.visit_expr(&c.expr),
            Expr::Tuple(t) => t.elems.iter().for_each(|e| self.visit_expr(e)),
            Expr::Array(a) => a.elems.iter().for_each(|e| self.visit_expr(e)),
            // Closures introduce their own param scope — not the top callable's locals.
            _ => {}
        }
    }
}

// ───────────────────────────── s-expr emitter ─────────────────────────────

struct Dump<'a> {
    /// `None` = cluster mode (names verbatim); `Some(set)` = xname mode (bound locals → `_v{n}`).
    locals: Option<&'a HashSet<String>>,
    map: HashMap<String, u32>,
    /// Blank the *top* callable's own name to `_fn` exactly once (xname mode).
    blanked: bool,
    /// Node-emit count — the cross-name "substance" gate.
    count: usize,
    /// `Some` while writing the Type-3 lines: every sub-expression worth a unit of its own, rendered
    /// as it appears inside its line, with that line's index — [`AnalyzedFn::type3_units`].
    units: Option<Vec<(u32, String)>>,
    /// The index of the line being written, which the units it yields are filed under.
    unit_line: u32,
    /// How many expressions enclose the one being written: `0` for a statement's own expression.
    expr_depth: u32,
    /// `Some` while writing the Type-3 lines: type reference → `_T{n}`. `None` writes types as
    /// spelled — every other canonical. See [`Dump::type_ref`] for why only those lines slot them.
    tmap: Option<HashMap<String, u32>>,
    /// Every type reference turned into a slot, in order of appearance — [`AnalyzedFn::type_refs`].
    type_refs: Vec<String>,
    /// Type parameters the definition declares: slotted like any type, but bound here, so not
    /// vocabulary it shares with anything.
    declared_types: HashSet<String>,
    /// Use-site mode: the definition whose use is being written. A path headed by its name becomes
    /// the anchor `_t0`, and its own surface — what the path goes on to name (`Anchor::new`), the
    /// fields a literal of it sets — comes back marked `#name#` for the profile to number.
    anchor: Option<&'a str>,
    /// Hunt mode: the names that count as anchors. Every one the walk meets is kept in `hunted`.
    known: Option<&'a HashSet<&'a str>>,
    hunted: Vec<String>,
    /// Hunt mode: the methods called (`.name`), for the co-occurrence channel of the `use` lens.
    calls: BTreeSet<String>,
}

impl<'a> Dump<'a> {
    fn new(locals: Option<&'a HashSet<String>>) -> Self {
        Self {
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
    fn hunting(locals: &'a HashSet<String>, known: &'a HashSet<&'a str>) -> Self {
        Self { known: Some(known), tmap: Some(HashMap::new()), ..Self::new(Some(locals)) }
    }

    /// A dump that writes a line as a use site of `anchor`.
    fn anchoring(locals: &'a HashSet<String>, anchor: &'a str) -> Self {
        Self { anchor: Some(anchor), tmap: Some(HashMap::new()), ..Self::new(Some(locals)) }
    }

    /// A path naming a definition: in use-site mode, the anchor `_t0` with whatever follows it
    /// marked as its surface (`_t0::#new#`); noted when its head is hunted for; as written
    /// otherwise.
    fn named_path(&mut self, path: &str) -> String {
        let (head, rest) = path.split_once("::").map_or((path, None), |(head, rest)| (head, Some(rest)));
        if let Some(known) = self.known {
            if known.contains(head) && !self.hunted.iter().any(|h| h == head) {
                self.hunted.push(head.to_owned());
            }
        }
        if self.anchor != Some(head) {
            return path.to_owned();
        }
        match rest {
            None => "_t0".to_owned(),
            Some(rest) => {
                let marked: Vec<String> = rest.split("::").map(|segment| format!("#{segment}#")).collect();
                format!("_t0::{}", marked.join("::"))
            }
        }
    }

    /// Whether a path is headed by the anchor.
    fn on_anchor(&self, path: &str) -> bool {
        self.anchor.is_some_and(|anchor| path.split("::").next() == Some(anchor))
    }

    /// A type's name: its `_T{n}` slot while writing the Type-3 lines (reported apart unless the
    /// definition declared it), as spelled otherwise.
    ///
    /// 🔴 Only the Type-3 lines slot types, unlike the Python and TypeScript frontends, whose exact
    /// canonicals do too. Their annotations run nothing; a Rust type picks the code that runs — the
    /// `impl` a method call resolves to, the `From` a `?` converts through — so two bodies differing
    /// only in their types may not be one definition, and the exact pass keeps them apart. The
    /// near-copy pass is a measure of how alike two bodies read, and there a type is vocabulary like
    /// any other: shared, it weighs as shared; different, it costs only its own weight rather than
    /// rewriting every line it appears on.
    fn type_ref(&mut self, name: &str) -> String {
        // The anchor keeps its own slot even in type position.
        if self.named_path(name) == "_t0" {
            return "_t0".to_owned();
        }
        let Some(tmap) = self.tmap.as_mut() else { return quoted(name) };
        if !self.declared_types.contains(name) {
            self.type_refs.push(name.to_owned());
        }
        dup_defs_core::type_slot(tmap, name)
    }

    fn rename(&mut self, name: &str) -> String {
        dup_defs_core::alpha_rename(&mut self.map, self.locals, name)
    }

    fn node(&mut self, tag: &str, fields: &[String]) -> String {
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

    fn list(&mut self, tag: &str, items: Vec<String>) -> String {
        let joined = items.join(", ");
        self.node(tag, &[joined])
    }

    // ────── callables ──────

    /// `is_top` blanks the def's own name in xname mode. Receiver (`self`) params are skipped.
    fn func(&mut self, name: &str, sig: &Signature, body: Option<&Block>, is_top: bool) -> String {
        let name = if is_top && self.locals.is_some() && !self.blanked {
            self.blanked = true;
            "_fn".to_owned()
        } else {
            self.rename(name)
        };
        let params = self.params(sig);
        let output = self.ret_type(&sig.output);
        let flags = format!(
            "async={} unsafe={} const={}",
            u8::from(sig.asyncness.is_some()),
            u8::from(sig.unsafety.is_some()),
            u8::from(sig.constness.is_some()),
        );
        let body = body.map_or_else(String::new, |b| self.block(b));
        self.node("Func", &[quoted(&name), params, output, quoted(&flags), body])
    }

    /// Parameter patterns, with the `self` receiver dropped so methods align with free fns.
    fn params(&mut self, sig: &Signature) -> String {
        let items: Vec<String> = sig
            .inputs
            .iter()
            .filter_map(|input| match input {
                FnArg::Receiver(_) => None,
                FnArg::Typed(pt) => Some({
                    let p = self.pat(&pt.pat);
                    let t = self.ty(&pt.ty);
                    self.node("Param", &[p, t])
                }),
            })
            .collect();
        self.list("Params", items)
    }

    /// The header of the Type-3 lines: what the Python frontend's `def _fn(_v0: _T0) -> _T1:` line
    /// says — the qualifiers that change what a call is, the parameters, their types, the return
    /// type — and nothing else. The receiver is dropped, as everywhere.
    ///
    /// 🔴 Not the `Func(…)` node the stream opens with. The pass reads every function's first line
    /// as context and weighs it like any other; written as an s-expr, the header spells the parse
    /// tree's grammar — `Params`, `Param`, `Bind`, `TyRef`, three flags — which only headers spell,
    /// so every one of those tokens was rare, and the header outweighed the body of every short
    /// function: two one-liners with one signature scored as copies whatever their bodies did.
    fn compact_header(&mut self, sig: &Signature) -> String {
        let mut out = String::new();
        for (on, word) in [(sig.constness.is_some(), "const "), (sig.asyncness.is_some(), "async "), (sig.unsafety.is_some(), "unsafe ")] {
            if on {
                out.push_str(word);
            }
        }
        out.push_str("fn _fn");
        let declared: Vec<String> = sig
            .generics
            .type_params()
            .map(|p| {
                let name = p.ident.to_string();
                self.declared_types.insert(name.clone());
                self.type_ref(&name)
            })
            .collect();
        if !declared.is_empty() {
            let _ = write!(out, "<{}>", declared.join(", "));
        }
        let mut params: Vec<String> = Vec::with_capacity(sig.inputs.len());
        for input in &sig.inputs {
            let FnArg::Typed(pt) = input else { continue };
            let pat = match pt.pat.as_ref() {
                Pat::Ident(id) if id.subpat.is_none() => self.rename(&id.ident.to_string()),
                other => self.pat(other),
            };
            let ty = self.ty(&pt.ty);
            params.push(format!("{pat}: {ty}"));
        }
        let _ = write!(out, "({})", params.join(", "));
        if let ReturnType::Type(_, ty) = &sig.output {
            let ty = self.ty(ty);
            let _ = write!(out, " -> {ty}");
        }
        out
    }

    fn ret_type(&mut self, output: &ReturnType) -> String {
        match output {
            ReturnType::Default => String::new(),
            ReturnType::Type(_, ty) => self.ty(ty),
        }
    }

    // ────── statements / blocks ──────

    fn block(&mut self, block: &Block) -> String {
        let items: Vec<String> = block.stmts.iter().map(|s| self.stmt(s)).collect();
        self.list("Block", items)
    }

    fn stmt(&mut self, stmt: &Stmt) -> String {
        match stmt {
            Stmt::Local(local) => {
                let pat = self.pat(&local.pat);
                let init = local
                    .init
                    .as_ref()
                    .map_or_else(String::new, |i| self.expr(&i.expr));
                self.node("Let", &[pat, init])
            }
            Stmt::Expr(e, semi) => {
                let v = self.expr(e);
                if semi.is_some() {
                    self.node("ExprStmt", &[v])
                } else {
                    self.node("Tail", &[v])
                }
            }
            Stmt::Item(_) => self.node("NestedItem", &[]),
            Stmt::Macro(m) => self.macro_call("MacroStmt", &m.mac),
        }
    }

    // ────── expressions ──────

    /// An expression — and, when units are being collected, kept as one if it is a proper part of
    /// its line: the statement's own expression is the line, already a unit, and counting it again
    /// would double the weight of exactly the text an edit changes. A wrapper (parentheses, a
    /// borrow) is not a level of its own; what it wraps is the part.
    fn expr(&mut self, expr: &Expr) -> String {
        if is_wrapper(expr) {
            return self.expr_inner(expr);
        }
        self.expr_depth += 1;
        let out = self.expr_inner(expr);
        self.expr_depth -= 1;
        if self.expr_depth > 0 && is_unit(expr) {
            if let Some(units) = self.units.as_mut() {
                units.push((self.unit_line, out.clone()));
            }
        }
        out
    }

    fn expr_inner(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Path(p) => {
                let s = path_str(&p.path);
                let named = self.named_path(&s);
                let renamed = if named == s { self.rename(&s) } else { named };
                self.node("Path", &[quoted(&renamed)])
            }
            Expr::Lit(l) => self.lit_expr(&l.lit),
            Expr::Binary(b) => {
                let l = self.expr(&b.left);
                let r = self.expr(&b.right);
                self.node("Bin", &[l, quoted(binop_str(&b.op)), r])
            }
            Expr::Unary(u) => {
                let e = self.expr(&u.expr);
                self.node("Unary", &[quoted(unop_str(&u.op)), e])
            }
            Expr::Assign(a) => {
                let l = self.expr(&a.left);
                let r = self.expr(&a.right);
                self.node("Assign", &[l, r])
            }
            Expr::Call(c) => {
                let f = self.expr(&c.func);
                let args: Vec<String> = c.args.iter().map(|a| self.expr(a)).collect();
                let joined = args.join(", ");
                self.node("Call", &[f, joined])
            }
            Expr::MethodCall(m) => {
                if self.known.is_some() {
                    self.calls.insert(format!(".{}", m.method));
                }
                let recv = self.expr(&m.receiver);
                let method = m.method.to_string();
                let args: Vec<String> = m.args.iter().map(|a| self.expr(a)).collect();
                let joined = args.join(", ");
                // The turbofish picks the code that runs — `parse::<u32>()` and `parse::<f64>()` are
                // two calls — and names the types a call is about (`query::<Order>()`). A trailing
                // field, present only when written, so a call without one reads as it always did.
                match &m.turbofish {
                    Some(tf) => {
                        let types: Vec<String> = tf
                            .args
                            .iter()
                            .filter_map(|a| match a {
                                syn::GenericArgument::Type(t) => Some(self.ty(t)),
                                _ => None,
                            })
                            .collect();
                        let turbofish = self.list("Turbofish", types);
                        self.node("Method", &[recv, quoted(&method), joined, turbofish])
                    }
                    None => self.node("Method", &[recv, quoted(&method), joined]),
                }
            }
            Expr::Field(f) => {
                let base = self.expr(&f.base);
                let member = match &f.member {
                    Member::Named(id) => id.to_string(),
                    Member::Unnamed(idx) => idx.index.to_string(),
                };
                self.node("Field", &[base, quoted(&member)])
            }
            Expr::Index(i) => {
                let base = self.expr(&i.expr);
                let idx = self.expr(&i.index);
                self.node("Index", &[base, idx])
            }
            Expr::If(i) => {
                let cond = self.expr(&i.cond);
                let then = self.block(&i.then_branch);
                let els = i.else_branch.as_ref().map_or_else(String::new, |(_, e)| self.expr(e));
                self.node("If", &[cond, then, els])
            }
            Expr::Match(m) => {
                let scrut = self.expr(&m.expr);
                let arms: Vec<String> = m
                    .arms
                    .iter()
                    .map(|arm| {
                        let p = self.pat(&arm.pat);
                        let guard = arm.guard.as_ref().map_or_else(String::new, |(_, g)| self.expr(g));
                        let body = self.expr(&arm.body);
                        self.node("Arm", &[p, guard, body])
                    })
                    .collect();
                let joined = arms.join(", ");
                self.node("Match", &[scrut, joined])
            }
            Expr::ForLoop(f) => {
                let pat = self.pat(&f.pat);
                let iter = self.expr(&f.expr);
                let body = self.block(&f.body);
                self.node("For", &[pat, iter, body])
            }
            Expr::While(w) => {
                let cond = self.expr(&w.cond);
                let body = self.block(&w.body);
                self.node("While", &[cond, body])
            }
            Expr::Loop(l) => {
                let body = self.block(&l.body);
                self.node("Loop", &[body])
            }
            Expr::Let(l) => {
                let pat = self.pat(&l.pat);
                let e = self.expr(&l.expr);
                self.node("LetExpr", &[pat, e])
            }
            Expr::Block(b) => {
                let body = self.block(&b.block);
                self.node("BlockExpr", &[body])
            }
            Expr::Unsafe(u) => {
                let body = self.block(&u.block);
                self.node("Unsafe", &[body])
            }
            Expr::Async(a) => {
                let body = self.block(&a.block);
                self.node("Async", &[body])
            }
            Expr::TryBlock(t) => {
                let body = self.block(&t.block);
                self.node("TryBlock", &[body])
            }
            Expr::Return(r) => {
                let e = r.expr.as_ref().map_or_else(String::new, |e| self.expr(e));
                self.node("Return", &[e])
            }
            Expr::Break(b) => {
                let e = b.expr.as_ref().map_or_else(String::new, |e| self.expr(e));
                self.node("Break", &[e])
            }
            Expr::Continue(_) => self.node("Continue", &[]),
            Expr::Reference(r) => {
                let e = self.expr(&r.expr);
                let m = if r.mutability.is_some() { "mut" } else { "" };
                self.node("Ref", &[quoted(m), e])
            }
            Expr::Try(t) => {
                let e = self.expr(&t.expr);
                self.node("Try", &[e])
            }
            Expr::Await(a) => {
                let e = self.expr(&a.base);
                self.node("Await", &[e])
            }
            Expr::Cast(c) => {
                let e = self.expr(&c.expr);
                let t = self.ty(&c.ty);
                self.node("Cast", &[e, t])
            }
            Expr::Paren(p) => self.expr(&p.expr),
            Expr::Group(g) => self.expr(&g.expr),
            Expr::Tuple(t) => {
                let items: Vec<String> = t.elems.iter().map(|e| self.expr(e)).collect();
                self.list("Tuple", items)
            }
            Expr::Array(a) => {
                let items: Vec<String> = a.elems.iter().map(|e| self.expr(e)).collect();
                self.list("Array", items)
            }
            Expr::Repeat(r) => {
                let e = self.expr(&r.expr);
                let len = self.expr(&r.len);
                self.node("Repeat", &[e, len])
            }
            Expr::Range(r) => {
                let lo = r.start.as_ref().map_or_else(String::new, |e| self.expr(e));
                let hi = r.end.as_ref().map_or_else(String::new, |e| self.expr(e));
                self.node("Range", &[lo, hi])
            }
            Expr::Struct(s) => {
                let path = path_str(&s.path);
                let on_anchor = self.on_anchor(&path);
                let path = self.named_path(&path);
                let fields: Vec<String> = s
                    .fields
                    .iter()
                    .map(|f| {
                        let member = match &f.member {
                            Member::Named(id) => id.to_string(),
                            Member::Unnamed(idx) => idx.index.to_string(),
                        };
                        let member = if on_anchor { format!("#{member}#") } else { member };
                        let v = self.expr(&f.expr);
                        self.node("FieldVal", &[quoted(&member), v])
                    })
                    .collect();
                let joined = fields.join(", ");
                let rest = s.rest.as_ref().map_or_else(String::new, |e| self.expr(e));
                self.node("StructLit", &[quoted(&path), joined, rest])
            }
            Expr::Closure(c) => {
                let params: Vec<String> = c.inputs.iter().map(|p| self.pat(p)).collect();
                let ps = params.join(", ");
                let body = self.expr(&c.body);
                self.node("Closure", &[ps, body])
            }
            Expr::Macro(m) => self.macro_call("Macro", &m.mac),
            other => self.node(&format!("Unknown_{}", expr_kind(other)), &[]),
        }
    }

    /// A macro invocation: its path, and — when its tokens read as a comma-separated list of
    /// expressions — the arguments, rendered like any call's.
    ///
    /// 🔴 Not the path alone. `vec![…]`, `assert_eq!(…)`, `format!(…)`, `write!(…)` hold most of
    /// what a Rust test or formatter says, and rendered as `Macro('vec')` two tests that build
    /// different values and assert different things were one canonical: on actix-web a pair of
    /// unrelated header tests scored 0.99 as near-copies. Python has no macros; its calls show their
    /// arguments, and these are calls in all but syntax. A body that does not parse as expressions
    /// (`vec![x; n]`, a DSL) stays opaque, as it was.
    fn macro_call(&mut self, tag: &str, mac: &syn::Macro) -> String {
        let path = quoted(&path_str(&mac.path));
        match mac.parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated) {
            Ok(args) => {
                let args: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
                self.node(tag, &[path, format!("[{}]", args.join(", "))])
            }
            Err(_) => self.node(tag, &[path]),
        }
    }

    fn lit_expr(&mut self, lit: &Lit) -> String {
        let (tag, val) = match lit {
            Lit::Str(s) => ("Str", s.value()),
            Lit::ByteStr(_) => ("ByteStr", String::new()),
            Lit::CStr(_) => ("CStr", String::new()),
            Lit::Byte(b) => ("Byte", b.value().to_string()),
            Lit::Char(c) => ("Char", c.value().to_string()),
            Lit::Int(i) => ("Int", i.base10_digits().to_owned()),
            Lit::Float(f) => ("Float", f.base10_digits().to_owned()),
            Lit::Bool(b) => ("Bool", b.value.to_string()),
            _ => ("Lit", String::new()),
        };
        self.node(tag, &[quoted(&val)])
    }

    // ────── patterns ──────

    fn pat(&mut self, pat: &Pat) -> String {
        match pat {
            Pat::Ident(pi) => {
                let n = self.rename(&pi.ident.to_string());
                self.node("Bind", &[quoted(&n)])
            }
            Pat::Wild(_) => self.node("Wild", &[]),
            Pat::Rest(_) => self.node("Rest", &[]),
            Pat::Lit(l) => self.lit_expr(&l.lit),
            Pat::Path(p) => {
                let s = path_str(&p.path);
                let s = self.named_path(&s);
                self.node("PatPath", &[quoted(&s)])
            }
            Pat::Reference(r) => {
                let inner = self.pat(&r.pat);
                self.node("PatRef", &[inner])
            }
            Pat::Tuple(t) => {
                let items: Vec<String> = t.elems.iter().map(|p| self.pat(p)).collect();
                self.list("PatTuple", items)
            }
            Pat::TupleStruct(ts) => {
                let path = path_str(&ts.path);
                let path = self.named_path(&path);
                let items: Vec<String> = ts.elems.iter().map(|p| self.pat(p)).collect();
                let joined = items.join(", ");
                self.node("PatTupleStruct", &[quoted(&path), joined])
            }
            Pat::Struct(s) => {
                let path = path_str(&s.path);
                let path = self.named_path(&path);
                let fields: Vec<String> = s
                    .fields
                    .iter()
                    .map(|f| {
                        let member = match &f.member {
                            Member::Named(id) => id.to_string(),
                            Member::Unnamed(idx) => idx.index.to_string(),
                        };
                        let p = self.pat(&f.pat);
                        self.node("PatField", &[quoted(&member), p])
                    })
                    .collect();
                let joined = fields.join(", ");
                self.node("PatStruct", &[quoted(&path), joined])
            }
            Pat::Slice(s) => {
                let items: Vec<String> = s.elems.iter().map(|p| self.pat(p)).collect();
                self.list("PatSlice", items)
            }
            Pat::Or(o) => {
                let items: Vec<String> = o.cases.iter().map(|p| self.pat(p)).collect();
                self.list("PatOr", items)
            }
            Pat::Paren(p) => self.pat(&p.pat),
            Pat::Type(t) => {
                let p = self.pat(&t.pat);
                let ty = self.ty(&t.ty);
                self.node("PatType", &[p, ty])
            }
            Pat::Range(_) => self.node("PatRange", &[]),
            other => self.node(&format!("Unknown_{}", pat_kind(other)), &[]),
        }
    }

    // ────── types (summarized) ──────

    fn ty(&mut self, ty: &Type) -> String {
        match ty {
            Type::Path(p) => {
                // Last segment + its generic args, structurally; type names are not renamed but in
                // the Type-3 lines, where they are slots.
                let Some(seg) = p.path.segments.last() else { return self.node("Ty", &[]) };
                let name =
                    if self.tmap.is_some() { self.type_ref(&path_str(&p.path)) } else { quoted(&seg.ident.to_string()) };
                let args = match &seg.arguments {
                    syn::PathArguments::AngleBracketed(ab) => {
                        let items: Vec<String> = ab
                            .args
                            .iter()
                            .filter_map(|a| match a {
                                syn::GenericArgument::Type(t) => Some(self.ty(t)),
                                _ => None,
                            })
                            .collect();
                        items.join(", ")
                    }
                    _ => String::new(),
                };
                self.node("Ty", &[name, args])
            }
            Type::Reference(r) => {
                let inner = self.ty(&r.elem);
                let m = if r.mutability.is_some() { "mut" } else { "" };
                self.node("TyRef", &[quoted(m), inner])
            }
            Type::Slice(s) => {
                let inner = self.ty(&s.elem);
                self.node("TySlice", &[inner])
            }
            Type::Array(a) => {
                let inner = self.ty(&a.elem);
                self.node("TyArray", &[inner])
            }
            Type::Tuple(t) => {
                let items: Vec<String> = t.elems.iter().map(|e| self.ty(e)).collect();
                self.list("TyTuple", items)
            }
            Type::Ptr(p) => {
                let inner = self.ty(&p.elem);
                self.node("TyPtr", &[inner])
            }
            Type::Paren(p) => self.ty(&p.elem),
            Type::Group(g) => self.ty(&g.elem),
            Type::Infer(_) => self.node("TyInfer", &[]),
            Type::Never(_) => self.node("TyNever", &[]),
            Type::ImplTrait(_) => self.node("TyImpl", &[]),
            Type::TraitObject(_) => self.node("TyDyn", &[]),
            Type::BareFn(_) => self.node("TyFn", &[]),
            _ => self.node("Ty_Other", &[]),
        }
    }

}

/// Whether an expression is a Type-3 unit of its own: a node with structure — a call, a field
/// chain, an operator application, a closure, a block-valued construct, a macro call, a non-empty
/// literal aggregate — not a bare path or a literal, which carry no agreement by themselves. The
/// same rule as the Python frontend's.
fn is_unit(expr: &Expr) -> bool {
    match expr {
        Expr::Call(_)
        | Expr::Macro(_)
        | Expr::MethodCall(_)
        | Expr::Field(_)
        | Expr::Index(_)
        | Expr::Binary(_)
        | Expr::Unary(_)
        | Expr::Assign(_)
        | Expr::If(_)
        | Expr::Match(_)
        | Expr::Let(_)
        | Expr::Closure(_)
        | Expr::Await(_)
        | Expr::Try(_)
        | Expr::Cast(_)
        | Expr::Range(_)
        | Expr::Repeat(_)
        | Expr::Block(_)
        | Expr::Async(_)
        | Expr::Unsafe(_)
        | Expr::TryBlock(_)
        | Expr::ForLoop(_)
        | Expr::While(_)
        | Expr::Loop(_) => true,
        Expr::Struct(s) => !s.fields.is_empty(),
        Expr::Tuple(t) => !t.elems.is_empty(),
        Expr::Array(a) => !a.elems.is_empty(),
        _ => false,
    }
}

/// Syntax around an expression that is not a part of its own: what it wraps is the part.
fn is_wrapper(expr: &Expr) -> bool {
    matches!(expr, Expr::Paren(_) | Expr::Group(_) | Expr::Reference(_))
}

/// Path → `::`-joined segment idents. The engine renames the whole string in xname mode when it
/// names a bound local (single-segment paths are the common local-variable case).
fn path_str(path: &syn::Path) -> String {
    let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    segs.join("::")
}

fn binop_str(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add(_) => "+",
        BinOp::Sub(_) => "-",
        BinOp::Mul(_) => "*",
        BinOp::Div(_) => "/",
        BinOp::Rem(_) => "%",
        BinOp::And(_) => "&&",
        BinOp::Or(_) => "||",
        BinOp::BitXor(_) => "^",
        BinOp::BitAnd(_) => "&",
        BinOp::BitOr(_) => "|",
        BinOp::Shl(_) => "<<",
        BinOp::Shr(_) => ">>",
        BinOp::Eq(_) => "==",
        BinOp::Lt(_) => "<",
        BinOp::Le(_) => "<=",
        BinOp::Ne(_) => "!=",
        BinOp::Ge(_) => ">=",
        BinOp::Gt(_) => ">",
        BinOp::AddAssign(_) => "+=",
        BinOp::SubAssign(_) => "-=",
        BinOp::MulAssign(_) => "*=",
        BinOp::DivAssign(_) => "/=",
        BinOp::RemAssign(_) => "%=",
        BinOp::BitXorAssign(_) => "^=",
        BinOp::BitAndAssign(_) => "&=",
        BinOp::BitOrAssign(_) => "|=",
        BinOp::ShlAssign(_) => "<<=",
        BinOp::ShrAssign(_) => ">>=",
        _ => "?op",
    }
}

fn unop_str(op: &UnOp) -> &'static str {
    match op {
        UnOp::Deref(_) => "*",
        UnOp::Not(_) => "!",
        UnOp::Neg(_) => "-",
        _ => "?un",
    }
}

/// Variant tag for the `Unknown_<Kind>` fallback (keeps the canonical deterministic + greppable).
fn expr_kind(e: &Expr) -> &'static str {
    match e {
        Expr::Const(_) => "Const",
        Expr::Infer(_) => "Infer",
        Expr::Verbatim(_) => "Verbatim",
        Expr::Yield(_) => "Yield",
        _ => "Expr",
    }
}

fn pat_kind(p: &Pat) -> &'static str {
    match p {
        Pat::Const(_) => "Const",
        Pat::Macro(_) => "Macro",
        Pat::Verbatim(_) => "Verbatim",
        _ => "Pat",
    }
}

// ───────────────────────────── item-level canonicals ─────────────────────────────

/// `where`/generic param *count* only (names aren't compared; presence/shape is the signal).
fn generics_tag(g: &Generics) -> String {
    format!("g{}", g.params.len())
}

/// Names-preserved structural canonical of a `struct` (the cluster pass's input for `classes`).
#[must_use]
pub fn struct_canon(item: &ItemStruct) -> String {
    let mut d = Dump::new(None);
    let fields = fields_canon(&mut d, &item.fields);
    d.node("Struct", &[quoted(&item.ident.to_string()), generics_tag(&item.generics), fields])
}

#[must_use]
pub fn enum_canon(item: &ItemEnum) -> String {
    let mut d = Dump::new(None);
    let variants: Vec<String> = item
        .variants
        .iter()
        .map(|v| {
            let f = fields_canon(&mut d, &v.fields);
            d.node("Variant", &[quoted(&v.ident.to_string()), f])
        })
        .collect();
    let joined = variants.join(", ");
    d.node("Enum", &[quoted(&item.ident.to_string()), generics_tag(&item.generics), joined])
}

#[must_use]
pub fn union_canon(item: &ItemUnion) -> String {
    let mut d = Dump::new(None);
    let items: Vec<String> = item
        .fields
        .named
        .iter()
        .map(|f| {
            let name = f.ident.as_ref().map_or_else(String::new, ToString::to_string);
            let ty = d.ty(&f.ty);
            d.node("Field", &[quoted(&name), ty])
        })
        .collect();
    let joined = items.join(", ");
    d.node("Union", &[quoted(&item.ident.to_string()), joined])
}

fn fields_canon(d: &mut Dump<'_>, fields: &syn::Fields) -> String {
    match fields {
        syn::Fields::Named(named) => {
            let items: Vec<String> = named
                .named
                .iter()
                .map(|f| {
                    let name = f.ident.as_ref().map_or_else(String::new, ToString::to_string);
                    let ty = d.ty(&f.ty);
                    d.node("Field", &[quoted(&name), ty])
                })
                .collect();
            d.list("Named", items)
        }
        syn::Fields::Unnamed(unnamed) => {
            let items: Vec<String> = unnamed.unnamed.iter().map(|f| d.ty(&f.ty)).collect();
            d.list("Tuple", items)
        }
        syn::Fields::Unit => d.node("Unit", &[]),
    }
}

/// Names-preserved structural canonical of a `trait` (the cluster pass's input for `interfaces`):
/// its associated items, with method bodies summarized to signatures.
#[must_use]
pub fn trait_canon(item: &ItemTrait) -> String {
    let mut d = Dump::new(None);
    let items: Vec<String> = item
        .items
        .iter()
        .map(|ti| match ti {
            TraitItem::Fn(f) => {
                let params = d.params(&f.sig);
                let output = d.ret_type(&f.sig.output);
                d.node("TraitFn", &[quoted(&f.sig.ident.to_string()), params, output])
            }
            TraitItem::Const(c) => {
                let ty = d.ty(&c.ty);
                d.node("TraitConst", &[quoted(&c.ident.to_string()), ty])
            }
            TraitItem::Type(t) => d.node("TraitType", &[quoted(&t.ident.to_string())]),
            _ => d.node("TraitOther", &[]),
        })
        .collect();
    let joined = items.join(", ");
    d.node("Trait", &[quoted(&item.ident.to_string()), generics_tag(&item.generics), joined])
}

// ───────────────────────────── callable analysis ─────────────────────────────

/// `(cluster_canonical, xname_canonical, type3_lines, node_count)` for a callable from its
/// signature + body. `name` is the def's own name (blanked to `_fn` in the xname canonical).
fn analyze(name: &str, sig: &Signature, body: &Block) -> AnalyzedFn {
    let cluster = {
        let mut d = Dump::new(None);
        d.func(name, sig, Some(body), true)
    };
    let locals_of = |block: &Block| {
        let mut collect = Collect::default();
        collect.add_inputs(sig);
        collect.visit_block(block);
        collect.bound
    };
    // The statement stream keeps the body as written: converge reads it to say where two definitions
    // stop agreeing, and a rewritten line would point the reader at code that is not there.
    let statements = statement_stream(name, sig, body, &locals_of(body));

    // The name-agnostic canonicals are taken from the normal form (`normal.rs`): what the exact and
    // Type-3 passes compare is the meaning the rules reach, not the spelling. The Type-3 lines come
    // from the form that keeps every statement, the exact canonical from the one that also inlines
    // single-use temporaries.
    let statement_form = crate::normal::normal_form(sig, body, false);
    let (type3_lines, type3_units, type_refs) = type3_walk(sig, &statement_form, &locals_of(&statement_form));

    let exact_form = crate::normal::with_inlining(sig, &statement_form);
    let exact_locals = locals_of(&exact_form);
    let mut xd = Dump::new(Some(&exact_locals));
    let xname = xd.func(name, sig, Some(&exact_form), true);
    let size = xd.count;
    AnalyzedFn { cluster_canonical: cluster, xname_canonical: xname, type3_lines, type_refs, type3_units, statements, size }
}

/// The Type-3 lines, their parts ([`AnalyzedFn::type3_units`]) and the types they name
/// ([`AnalyzedFn::type_refs`]): the statement-stream walk again, with the types as slots and the
/// header written the way the Python frontend writes it ([`Dump::compact_header`]).
///
/// The pass compares statements at every depth — an edit inside an `if` costs the statement it
/// touched, not the `if` — and every statement's sub-expressions are where two edited lines still
/// agree. The header comes first because the pass reads every function's first line as its
/// interface: context, not a step.
fn type3_walk(sig: &Signature, body: &Block, locals: &HashSet<String>) -> (Vec<String>, Vec<(u32, String)>, Vec<String>) {
    let mut dump = Dump::new(Some(locals));
    dump.tmap = Some(HashMap::new());
    dump.units = Some(Vec::new());
    dump.unit_line = UNFILED;
    let head = dump.compact_header(sig);
    let mut walk = StreamWalk { dump: &mut dump, out: Vec::new() };
    walk.push(0, head);
    walk.block(body, 1);
    let lines = walk.out.into_iter().map(|s| s.line).collect();
    (lines, dump.units.unwrap_or_default(), dump.type_refs)
}

/// The body canonicalized with every name the **file** introduced erased — the widest rung of the
/// erasure ladder, which the `scope` lens projects as a single fact.
///
/// One string rather than one per statement: what it asserts is all-or-nothing — two bodies either
/// reduce to the same shape or they do not — and as one rare string it carries the IDF weight that
/// claim deserves.
#[must_use]
pub fn scope_canonical(sig: &Signature, body: &Block, file_names: &HashSet<String>) -> String {
    let mut collect = Collect::default();
    collect.add_inputs(sig);
    collect.visit_block(body);
    let mut locals = collect.bound;
    locals.extend(file_names.iter().cloned());
    let mut dump = Dump::new(Some(&locals));
    dump.func("_fn", sig, Some(body), true)
}

/// Analyze a free `fn` item.
#[must_use]
pub fn analyze_item_fn(f: &ItemFn) -> AnalyzedFn {
    analyze(&f.sig.ident.to_string(), &f.sig, &f.block)
}

/// Analyze an `impl` method (always has a body).
#[must_use]
pub fn analyze_impl_fn(f: &ImplItemFn) -> AnalyzedFn {
    analyze(&f.sig.ident.to_string(), &f.sig, &f.block)
}

/// Analyze a trait method *with a default body* (bodiless signatures are filtered earlier).
#[must_use]
pub fn analyze_trait_fn(f: &TraitItemFn) -> Option<AnalyzedFn> {
    f.default.as_ref().map(|body| analyze(&f.sig.ident.to_string(), &f.sig, body))
}

// ───────────────────────────── statement stream ─────────────────────────────

/// One line per statement at **every** nesting level, in source order — the engine's
/// [`Facets::statements`](dup_defs_core::Facets::statements).
///
/// Rust has no statement-level `if`: control flow is expressions, so the walk descends into the
/// expression forms that carry a block. A compound construct contributes a **header** line naming
/// what it tests (`If(cond)`, `Match(scrutinee)`, `For(pat, iter)`) with its body elided, and the
/// body follows one level deeper — the same shape Python's unparser produces, so the two languages'
/// streams are comparable rather than merely both present.
///
/// One `Dump` for the whole definition, so a local carries the same `_v{n}` across every line it
/// appears on: here the order *is* the signal, and a slot that means something different on each
/// line would destroy it.
fn statement_stream(name: &str, sig: &Signature, body: &Block, locals: &HashSet<String>) -> Vec<Statement> {
    let mut dump = Dump::new(Some(locals));
    // The definition's own header opens the stream at depth 0 and the body starts at depth 1 — the
    // contract every frontend agrees on, so a consumer cannot mistake a missing header for a
    // definition that opens with a statement.
    let head = dump.func(name, sig, None, true);
    let mut walk = StreamWalk { dump: &mut dump, out: Vec::new() };
    walk.push(0, head);
    walk.block(body, 1);
    walk.out
}


struct StreamWalk<'a, 'b> {
    dump: &'b mut Dump<'a>,
    out: Vec<Statement>,
}

impl StreamWalk<'_, '_> {
    /// A line is rendered whole before it is pushed, and nothing else is pushed in between, so the
    /// units still unfiled when it arrives are exactly its own.
    fn push(&mut self, depth: u16, line: String) {
        let at = u32::try_from(self.out.len()).unwrap_or(UNFILED - 1);
        if let Some(units) = self.dump.units.as_mut() {
            for unit in units.iter_mut().rev().take_while(|u| u.0 == UNFILED) {
                unit.0 = at;
            }
        }
        self.out.push(Statement { line, depth });
    }

    fn block(&mut self, block: &Block, depth: u16) {
        for stmt in &block.stmts {
            self.stmt(stmt, depth);
        }
    }

    fn stmt(&mut self, stmt: &Stmt, depth: u16) {
        match stmt {
            // A block-carrying expression *as a statement* is a construct, not a value: walk it.
            // The same expression in value position (`let x = if c { a } else { b };`) stays one
            // line, because there the block is how the value is written, not a step of the body.
            Stmt::Expr(expr, _) if opens_a_block(expr) => self.control(expr, depth),
            _ => {
                let line = self.dump.stmt(stmt);
                self.push(depth, line);
            }
        }
    }

    /// A block-carrying expression: its header, then its body one level deeper.
    fn control(&mut self, expr: &Expr, depth: u16) {
        match expr {
            Expr::If(node) => {
                let cond = self.dump.expr(&node.cond);
                let head = self.dump.node("If", &[cond]);
                self.push(depth, head);
                self.block(&node.then_branch, depth + 1);
                if let Some((_, alt)) = &node.else_branch {
                    let head = self.dump.node("Else", &[]);
                    self.push(depth, head);
                    match alt.as_ref() {
                        // `else if` chains as a sibling of the `else`, not nested under it — which
                        // is what the source means and what Python's `elif` renders as.
                        Expr::If(_) => self.control(alt, depth),
                        Expr::Block(b) => self.block(&b.block, depth + 1),
                        other => self.control_or_line(other, depth + 1),
                    }
                }
            }
            Expr::Match(node) => {
                let scrutinee = self.dump.expr(&node.expr);
                let head = self.dump.node("Match", &[scrutinee]);
                self.push(depth, head);
                for arm in &node.arms {
                    let pat = self.dump.pat(&arm.pat);
                    let guard = arm.guard.as_ref().map_or_else(String::new, |(_, g)| self.dump.expr(g));
                    let head = self.dump.node("Arm", &[pat, guard]);
                    self.push(depth + 1, head);
                    self.control_or_line(&arm.body, depth + 2);
                }
            }
            Expr::While(node) => {
                let cond = self.dump.expr(&node.cond);
                let head = self.dump.node("While", &[cond]);
                self.push(depth, head);
                self.block(&node.body, depth + 1);
            }
            Expr::ForLoop(node) => {
                let pat = self.dump.pat(&node.pat);
                let iter = self.dump.expr(&node.expr);
                let head = self.dump.node("For", &[pat, iter]);
                self.push(depth, head);
                self.block(&node.body, depth + 1);
            }
            Expr::Loop(node) => {
                let head = self.dump.node("Loop", &[]);
                self.push(depth, head);
                self.block(&node.body, depth + 1);
            }
            Expr::Block(node) => {
                let head = self.dump.node("Block", &[]);
                self.push(depth, head);
                self.block(&node.block, depth + 1);
            }
            Expr::Unsafe(node) => {
                let head = self.dump.node("Unsafe", &[]);
                self.push(depth, head);
                self.block(&node.block, depth + 1);
            }
            Expr::TryBlock(node) => {
                let head = self.dump.node("TryBlock", &[]);
                self.push(depth, head);
                self.block(&node.block, depth + 1);
            }
            Expr::Async(node) => {
                let head = self.dump.node("Async", &[]);
                self.push(depth, head);
                self.block(&node.block, depth + 1);
            }
            other => self.control_or_line(other, depth),
        }
    }

    /// Walk `expr` if it opens a block, otherwise emit it as one line.
    fn control_or_line(&mut self, expr: &Expr, depth: u16) {
        if opens_a_block(expr) {
            self.control(expr, depth);
        } else {
            let line = self.dump.expr(expr);
            self.push(depth, line);
        }
    }
}

/// Does this expression carry a block the stream should walk into?
fn opens_a_block(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::If(_)
            | Expr::Match(_)
            | Expr::While(_)
            | Expr::ForLoop(_)
            | Expr::Loop(_)
            | Expr::Block(_)
            | Expr::Unsafe(_)
            | Expr::TryBlock(_)
            | Expr::Async(_)
    )
}

// ───────────────────────────── use sites ─────────────────────────────

/// Every use site in one source — the Rust half of the `use` lens ([`dup_defs_core::uses`]): each
/// statement that mentions a known name, written with that name as the anchor. A site is the
/// statement's own line — a block-carrying construct's header (`If(cond)`, `Match(scrutinee)`, a
/// match arm's pattern), not its body, whose statements are sites of their own — as the statement
/// stream writes it, and inside a function its locals rename as they do there. A closure stays part
/// of the statement it is written in, as a Python lambda does.
pub(crate) fn collect_sites(source: &str, _path: &str, known: &HashSet<&str>) -> Vec<Site> {
    let Ok(file) = syn::parse_file(source) else { return Vec::new() };
    let mut walk = UseWalk { known, out: Vec::new() };
    walk.items(&file.items);
    walk.out
}

struct UseWalk<'k> {
    known: &'k HashSet<&'k str>,
    out: Vec<Site>,
}

impl UseWalk<'_> {
    /// One site per known name `render` mentions, each written with that name as the anchor.
    fn site<F>(&mut self, locals: &HashSet<String>, ctx: &'static str, neighbours: &[String], render: F)
    where
        F: Fn(&mut Dump<'_>) -> Option<String>,
    {
        let mut hunt = Dump::hunting(locals, self.known);
        if render(&mut hunt).is_none() {
            return;
        }
        for name in hunt.hunted {
            let mut anchored = Dump::anchoring(locals, &name);
            if let Some(canon) = render(&mut anchored) {
                self.out.push(Site { name, canon, ctx, cooc: neighbours.to_vec() });
            }
        }
    }

    /// A module's items: the bodies of its functions and methods, and the values of its constants.
    /// A type's own declaration — what it derives, the fields it declares, what it implements — is
    /// a declaration, not a use.
    fn items(&mut self, items: &[Item]) {
        let empty = HashSet::new();
        for item in items {
            match item {
                Item::Fn(f) => self.function(&f.sig, &f.block),
                Item::Impl(i) => {
                    for inner in &i.items {
                        match inner {
                            ImplItem::Fn(f) => self.function(&f.sig, &f.block),
                            ImplItem::Const(c) => self.site(&empty, "plain", &[], |d| {
                                let value = d.expr(&c.expr);
                                Some(d.node("Const", &[value]))
                            }),
                            _ => {}
                        }
                    }
                }
                Item::Trait(t) => {
                    for inner in &t.items {
                        if let TraitItem::Fn(f) = inner {
                            if let Some(body) = &f.default {
                                self.function(&f.sig, body);
                            }
                        }
                    }
                }
                Item::Const(c) => self.site(&empty, "plain", &[], |d| {
                    let value = d.expr(&c.expr);
                    Some(d.node("Const", &[value]))
                }),
                Item::Static(st) => self.site(&empty, "plain", &[], |d| {
                    let value = d.expr(&st.expr);
                    Some(d.node("Static", &[value]))
                }),
                Item::Mod(m) => {
                    if let Some((_, inner)) = &m.content {
                        self.items(inner);
                    }
                }
                _ => {}
            }
        }
    }

    /// A function: its signature is a site when it names a known type, its body is walked under its
    /// own locals.
    fn function(&mut self, sig: &Signature, body: &Block) {
        let mut collect = Collect::default();
        collect.add_inputs(sig);
        collect.visit_block(body);
        let locals = collect.bound;
        self.site(&locals, "plain", &[], |d| Some(d.compact_header(sig)));
        self.block(body, &locals, (false, false));
    }

    /// A block's statements, each with the methods called beside it — the co-occurrence channel.
    /// `within` is `(in_try, in_loop)`.
    fn block(&mut self, block: &Block, locals: &HashSet<String>, within: (bool, bool)) {
        let mut hunt = Dump::hunting(locals, self.known);
        for stmt in &block.stmts {
            let _ = use_line(&mut hunt, stmt);
        }
        let neighbours: Vec<String> = hunt.calls.into_iter().collect();
        for stmt in &block.stmts {
            match stmt {
                Stmt::Item(item) => self.items(std::slice::from_ref(item)),
                Stmt::Expr(expr, _) if opens_a_block(expr) => self.control(expr, locals, within, &neighbours),
                _ => {
                    let ctx = site_ctx(Construct::Plain, Within { in_try: within.0, in_loop: within.1 });
                    self.site(locals, ctx, &neighbours, |d| use_line(d, stmt));
                }
            }
        }
    }

    /// A block-carrying construct: its header is a site, its body is walked.
    fn control(&mut self, expr: &Expr, locals: &HashSet<String>, within: (bool, bool), neighbours: &[String]) {
        let (in_try, in_loop) = within;
        let guard = site_ctx(Construct::Guard, Within { in_try, in_loop });
        let looped = site_ctx(Construct::Loop, Within { in_try, in_loop });
        match expr {
            Expr::If(n) => {
                self.site(locals, guard, neighbours, |d| {
                    let cond = d.expr(&n.cond);
                    Some(d.node("If", &[cond]))
                });
                self.block(&n.then_branch, locals, within);
                if let Some((_, alt)) = &n.else_branch {
                    self.arm_body(alt, locals, within, neighbours);
                }
            }
            Expr::Match(n) => {
                self.site(locals, guard, neighbours, |d| {
                    let scrutinee = d.expr(&n.expr);
                    Some(d.node("Match", &[scrutinee]))
                });
                for arm in &n.arms {
                    self.site(locals, guard, neighbours, |d| {
                        let pat = d.pat(&arm.pat);
                        let guard = arm.guard.as_ref().map_or_else(String::new, |(_, g)| d.expr(g));
                        Some(d.node("Arm", &[pat, guard]))
                    });
                    self.arm_body(&arm.body, locals, within, neighbours);
                }
            }
            Expr::While(n) => {
                self.site(locals, looped, neighbours, |d| {
                    let cond = d.expr(&n.cond);
                    Some(d.node("While", &[cond]))
                });
                self.block(&n.body, locals, (in_try, true));
            }
            Expr::ForLoop(n) => {
                self.site(locals, looped, neighbours, |d| {
                    let pat = d.pat(&n.pat);
                    let iter = d.expr(&n.expr);
                    Some(d.node("For", &[pat, iter]))
                });
                self.block(&n.body, locals, (in_try, true));
            }
            Expr::Loop(n) => self.block(&n.body, locals, (in_try, true)),
            Expr::TryBlock(n) => self.block(&n.block, locals, (true, in_loop)),
            Expr::Block(n) => self.block(&n.block, locals, within),
            Expr::Unsafe(n) => self.block(&n.block, locals, within),
            Expr::Async(n) => self.block(&n.block, locals, within),
            other => {
                let ctx = site_ctx(Construct::Plain, Within { in_try, in_loop });
                self.site(locals, ctx, neighbours, |d| Some(d.expr(other)));
            }
        }
    }

    /// An `else` branch or a match arm's body: a construct is walked, a value is a site.
    fn arm_body(&mut self, expr: &Expr, locals: &HashSet<String>, within: (bool, bool), neighbours: &[String]) {
        if opens_a_block(expr) {
            self.control(expr, locals, within, neighbours);
        } else {
            let ctx = site_ctx(Construct::Plain, Within { in_try: within.0, in_loop: within.1 });
            self.site(locals, ctx, neighbours, |d| Some(d.expr(expr)));
        }
    }
}

/// A statement's own line, as a use site sees it: a block-carrying construct's header, any other
/// statement whole — `None` for a nested item, whose insides are walked.
fn use_line(d: &mut Dump<'_>, stmt: &Stmt) -> Option<String> {
    match stmt {
        Stmt::Item(_) => None,
        Stmt::Expr(expr, _) if opens_a_block(expr) => Some(match expr {
            Expr::If(n) => {
                let cond = d.expr(&n.cond);
                d.node("If", &[cond])
            }
            Expr::Match(n) => {
                let scrutinee = d.expr(&n.expr);
                d.node("Match", &[scrutinee])
            }
            Expr::While(n) => {
                let cond = d.expr(&n.cond);
                d.node("While", &[cond])
            }
            Expr::ForLoop(n) => {
                let pat = d.pat(&n.pat);
                let iter = d.expr(&n.expr);
                d.node("For", &[pat, iter])
            }
            _ => return None,
        }),
        other => Some(d.stmt(other)),
    }
}

// ───────────────────────────── reach ─────────────────────────────

/// What each name a file's `use` items bind stands for, as the engine's dotted path.
///
/// `use a::b::c;` binds `c` to `a.b.c`; `use a::b as z;` binds `z` to `a.b`; `use a::{b, c::d};`
/// binds both leaves. A glob binds no name this can attribute a use to, so it is skipped rather than
/// guessed at.
#[must_use]
pub fn file_imports(file: &syn::File) -> Vec<(String, Arc<str>)> {
    let mut out = Vec::new();
    for item in &file.items {
        if let syn::Item::Use(item) = item {
            walk_use(&item.tree, &mut Vec::new(), &mut out);
        }
    }
    out
}

fn walk_use(tree: &syn::UseTree, prefix: &mut Vec<String>, out: &mut Vec<(String, Arc<str>)>) {
    match tree {
        syn::UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            walk_use(&path.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Name(name) => {
            let leaf = name.ident.to_string();
            prefix.push(leaf.clone());
            out.push((leaf, dup_defs_core::reach::reach_path(prefix)));
            prefix.pop();
        }
        syn::UseTree::Rename(rename) => {
            prefix.push(rename.ident.to_string());
            out.push((rename.rename.to_string(), dup_defs_core::reach::reach_path(prefix)));
            prefix.pop();
        }
        syn::UseTree::Group(group) => {
            for tree in &group.items {
                walk_use(tree, prefix, out);
            }
        }
        // `use a::*` binds names this cannot enumerate; guessing would invent reach that is not there.
        syn::UseTree::Glob(_) => {}
    }
}

/// Every path head this callable mentions — the raw material the frontend intersects with what its
/// file imported.
///
/// Deliberately not filtered by boundness: a local shadowing an import inside one function is rare,
/// and reading the shadow as "this definition does not reach that module" is a silent false
/// negative, the expensive direction here.
#[must_use]
pub fn used_names(block: &Block) -> HashSet<String> {
    #[derive(Default)]
    struct Names {
        seen: HashSet<String>,
    }
    impl<'ast> syn::visit::Visit<'ast> for Names {
        fn visit_path(&mut self, path: &'ast syn::Path) {
            if let Some(first) = path.segments.first() {
                self.seen.insert(first.ident.to_string());
            }
            // A path's own segments are not expressions; keep walking for generic arguments, which
            // can name types this definition genuinely reaches.
            syn::visit::visit_path(self, path);
        }
    }
    let mut names = Names::default();
    syn::visit::Visit::visit_block(&mut names, block);
    names.seen
}

#[cfg(test)]
mod use_tests {
    use super::collect_sites;
    use crate::defs::scan_source;
    use dup_defs_core::uses::use_facts;
    use dup_defs_core::ScanOpts;
    use std::collections::HashSet;
    use std::sync::Arc;

    #[test]
    fn a_site_is_the_construct_header_not_its_body() {
        let src = "fn f(db: &Db) {\n    for row in db.query::<Tbl>() {\n        other(Unrelated);\n    }\n    if let Some(t) = Tbl::find(1) { t.touch(); }\n}\n";
        let known: HashSet<&str> = ["Tbl"].into_iter().collect();
        let sites = collect_sites(src, "t.rs", &known);
        let canons: Vec<&str> = sites.iter().map(|s| s.canon.as_str()).collect();
        assert!(canons.iter().any(|c| c.starts_with("If(LetExpr(") && c.contains("_t0::#find#")), "{canons:?}");
        assert!(canons.iter().all(|c| !c.contains("other")), "a body is a site of its own: {canons:?}");
    }

    /// The `use` lens's load-bearing claim, in Rust: two definitions whose bodies share nothing,
    /// handled identically, produce the same use-site facts. A Rust type's surface is what its path
    /// names — the associated functions it is reached through, the fields a literal of it sets.
    #[test]
    fn divergent_bodies_with_identical_handling_share_their_use_facts() {
        let a = concat!(
            "pub struct ImageCache { key: String, blob: Vec<u8>, dead_at: u64 }\n",
            "pub fn put_image(s: &mut Session, k: String, v: Vec<u8>) { s.add(ImageCache { key: k, blob: v, dead_at: soon() }); }\n",
            "pub fn get_image(s: &Session, k: &str) -> Option<Row> { s.find(ImageCache::by_key(k)) }\n",
            "pub fn reap_images(s: &mut Session) { s.delete(ImageCache::older_than(now())); }\n",
        );
        let b = concat!(
            "pub struct PromptStore { payload: Json }\n",
            "pub fn put_prompt(s: &mut Session, k: String, v: Json) { s.add(PromptStore { fingerprint: k, payload: v, expires: soon() }); }\n",
            "pub fn get_prompt(s: &Session, k: &str) -> Option<Row> { s.find(PromptStore::by_fingerprint(k)) }\n",
            "pub fn reap_prompts(s: &mut Session) { s.delete(PromptStore::expired_before(now())); }\n",
        );
        let (fa, fb): (Arc<str>, Arc<str>) = (Arc::from("a.rs"), Arc::from("b.rs"));
        let opts = ScanOpts::default();
        let mut defs = scan_source(a, &fa, &opts);
        defs.extend(scan_source(b, &fb, &opts));
        let facts = use_facts(&[(a, "a.rs", 1), (b, "b.rs", 1)], &defs, collect_sites);
        let get = |n: &str| facts.get(n).unwrap_or_else(|| panic!("no use facts for {n}: {facts:?}")).clone();
        assert_eq!(get("ImageCache"), get("PromptStore"), "identical handling ⇒ identical use facts");
    }
}
