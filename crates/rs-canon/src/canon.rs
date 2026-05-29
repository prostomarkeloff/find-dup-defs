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

use std::collections::{HashMap, HashSet};

use syn::{
    BinOp, Block, Expr, FnArg, Generics, ImplItemFn, ItemEnum, ItemFn, ItemStruct, ItemTrait,
    ItemUnion, Lit, Member, Pat, ReturnType, Signature, Stmt, TraitItem, TraitItemFn, Type, UnOp,
};

/// `(cluster_canonical, xname_canonical, type3_lines, node_count)` — the analysis tuple the scan
/// reads to build a callable `Def`'s cluster canonical + `Analysis`.
pub type AnalyzedFn = (String, String, Vec<String>, usize);

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
}

impl<'a> Dump<'a> {
    fn new(locals: Option<&'a HashSet<String>>) -> Self {
        Self { locals, map: HashMap::new(), blanked: false, count: 0 }
    }

    #[allow(clippy::cast_possible_truncation)] // a callable's distinct bound-name count is far below u32::MAX
    fn rename(&mut self, name: &str) -> String {
        if let Some(locals) = self.locals {
            if locals.contains(name) {
                let next = self.map.len() as u32;
                let slot = *self.map.entry(name.to_owned()).or_insert(next);
                return format!("_v{slot}");
            }
        }
        name.to_owned()
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

    fn lit(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('\'');
        for c in s.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '\'' => out.push_str("\\'"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c => out.push(c),
            }
        }
        out.push('\'');
        out
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
        self.node("Func", &[Self::lit(&name), params, output, Self::lit(&flags), body])
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
            Stmt::Macro(m) => {
                let path = path_str(&m.mac.path);
                self.node("MacroStmt", &[Self::lit(&path)])
            }
        }
    }

    // ────── expressions ──────

    fn expr(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Path(p) => {
                let s = path_str(&p.path);
                let renamed = self.rename(&s);
                self.node("Path", &[Self::lit(&renamed)])
            }
            Expr::Lit(l) => self.lit_expr(&l.lit),
            Expr::Binary(b) => {
                let l = self.expr(&b.left);
                let r = self.expr(&b.right);
                self.node("Bin", &[l, Self::lit(binop_str(&b.op)), r])
            }
            Expr::Unary(u) => {
                let e = self.expr(&u.expr);
                self.node("Unary", &[Self::lit(unop_str(&u.op)), e])
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
                let recv = self.expr(&m.receiver);
                let method = m.method.to_string();
                let args: Vec<String> = m.args.iter().map(|a| self.expr(a)).collect();
                let joined = args.join(", ");
                self.node("Method", &[recv, Self::lit(&method), joined])
            }
            Expr::Field(f) => {
                let base = self.expr(&f.base);
                let member = match &f.member {
                    Member::Named(id) => id.to_string(),
                    Member::Unnamed(idx) => idx.index.to_string(),
                };
                self.node("Field", &[base, Self::lit(&member)])
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
                self.node("Ref", &[Self::lit(m), e])
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
                let fields: Vec<String> = s
                    .fields
                    .iter()
                    .map(|f| {
                        let member = match &f.member {
                            Member::Named(id) => id.to_string(),
                            Member::Unnamed(idx) => idx.index.to_string(),
                        };
                        let v = self.expr(&f.expr);
                        self.node("FieldVal", &[Self::lit(&member), v])
                    })
                    .collect();
                let joined = fields.join(", ");
                let rest = s.rest.as_ref().map_or_else(String::new, |e| self.expr(e));
                self.node("StructLit", &[Self::lit(&path), joined, rest])
            }
            Expr::Closure(c) => {
                let params: Vec<String> = c.inputs.iter().map(|p| self.pat(p)).collect();
                let ps = params.join(", ");
                let body = self.expr(&c.body);
                self.node("Closure", &[ps, body])
            }
            Expr::Macro(m) => {
                let path = path_str(&m.mac.path);
                self.node("Macro", &[Self::lit(&path)])
            }
            other => self.node(&format!("Unknown_{}", expr_kind(other)), &[]),
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
        self.node(tag, &[Self::lit(&val)])
    }

    // ────── patterns ──────

    fn pat(&mut self, pat: &Pat) -> String {
        match pat {
            Pat::Ident(pi) => {
                let n = self.rename(&pi.ident.to_string());
                self.node("Bind", &[Self::lit(&n)])
            }
            Pat::Wild(_) => self.node("Wild", &[]),
            Pat::Rest(_) => self.node("Rest", &[]),
            Pat::Lit(l) => self.lit_expr(&l.lit),
            Pat::Path(p) => {
                let s = path_str(&p.path);
                self.node("PatPath", &[Self::lit(&s)])
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
                let items: Vec<String> = ts.elems.iter().map(|p| self.pat(p)).collect();
                let joined = items.join(", ");
                self.node("PatTupleStruct", &[Self::lit(&path), joined])
            }
            Pat::Struct(s) => {
                let path = path_str(&s.path);
                let fields: Vec<String> = s
                    .fields
                    .iter()
                    .map(|f| {
                        let member = match &f.member {
                            Member::Named(id) => id.to_string(),
                            Member::Unnamed(idx) => idx.index.to_string(),
                        };
                        let p = self.pat(&f.pat);
                        self.node("PatField", &[Self::lit(&member), p])
                    })
                    .collect();
                let joined = fields.join(", ");
                self.node("PatStruct", &[Self::lit(&path), joined])
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
                // Last segment + its generic args, structurally; type names are not renamed.
                let Some(seg) = p.path.segments.last() else { return self.node("Ty", &[]) };
                let name = seg.ident.to_string();
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
                self.node("Ty", &[Self::lit(&name), args])
            }
            Type::Reference(r) => {
                let inner = self.ty(&r.elem);
                let m = if r.mutability.is_some() { "mut" } else { "" };
                self.node("TyRef", &[Self::lit(m), inner])
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
    d.node("Struct", &[Dump::lit(&item.ident.to_string()), generics_tag(&item.generics), fields])
}

#[must_use]
pub fn enum_canon(item: &ItemEnum) -> String {
    let mut d = Dump::new(None);
    let variants: Vec<String> = item
        .variants
        .iter()
        .map(|v| {
            let f = fields_canon(&mut d, &v.fields);
            d.node("Variant", &[Dump::lit(&v.ident.to_string()), f])
        })
        .collect();
    let joined = variants.join(", ");
    d.node("Enum", &[Dump::lit(&item.ident.to_string()), generics_tag(&item.generics), joined])
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
            d.node("Field", &[Dump::lit(&name), ty])
        })
        .collect();
    let joined = items.join(", ");
    d.node("Union", &[Dump::lit(&item.ident.to_string()), joined])
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
                    d.node("Field", &[Dump::lit(&name), ty])
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
                d.node("TraitFn", &[Dump::lit(&f.sig.ident.to_string()), params, output])
            }
            TraitItem::Const(c) => {
                let ty = d.ty(&c.ty);
                d.node("TraitConst", &[Dump::lit(&c.ident.to_string()), ty])
            }
            TraitItem::Type(t) => d.node("TraitType", &[Dump::lit(&t.ident.to_string())]),
            _ => d.node("TraitOther", &[]),
        })
        .collect();
    let joined = items.join(", ");
    d.node("Trait", &[Dump::lit(&item.ident.to_string()), generics_tag(&item.generics), joined])
}

// ───────────────────────────── callable analysis ─────────────────────────────

/// Per-statement renamed lines for the Type-3 pass (one logical line per body statement),
/// emitted with a fresh `Dump` per line so numbering stays per-line (order-invariant cosine).
fn type3_lines(block: &Block, locals: &HashSet<String>) -> Vec<String> {
    block
        .stmts
        .iter()
        .map(|s| {
            let mut d = Dump::new(Some(locals));
            d.stmt(s)
        })
        .collect()
}

/// `(cluster_canonical, xname_canonical, type3_lines, node_count)` for a callable from its
/// signature + body. `name` is the def's own name (blanked to `_fn` in the xname canonical).
fn analyze(name: &str, sig: &Signature, body: &Block) -> AnalyzedFn {
    let cluster = {
        let mut d = Dump::new(None);
        d.func(name, sig, Some(body), true)
    };
    let mut collect = Collect::default();
    collect.add_inputs(sig);
    collect.visit_block(body);
    let locals = collect.bound;

    let mut xd = Dump::new(Some(&locals));
    let xname = xd.func(name, sig, Some(body), true);
    let size = xd.count;

    let lines = type3_lines(body, &locals);
    (cluster, xname, lines, size)
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
