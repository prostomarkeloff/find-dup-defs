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
use std::sync::Arc;

use dup_defs_core::Statement;

use syn::{
    BinOp, Block, Expr, FnArg, Generics, ImplItemFn, ItemEnum, ItemFn, ItemStruct, ItemTrait,
    ItemUnion, Lit, Member, Pat, ReturnType, Signature, Stmt, TraitItem, TraitItemFn, Type, UnOp,
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
}

impl<'a> Dump<'a> {
    fn new(locals: Option<&'a HashSet<String>>) -> Self {
        Self { locals, map: HashMap::new(), blanked: false, count: 0 }
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
    let statements = statement_stream(name, sig, body, &locals);
    AnalyzedFn { cluster_canonical: cluster, xname_canonical: xname, type3_lines: lines, statements, size }
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
/// Deliberately a second traversal rather than a re-use of [`type3_lines`]. Those are one line per
/// *top-level* block statement, with a nested `if` inlined whole into its parent's line: right for
/// an order-invariant cosine, useless for a pass that asks where two definitions stopped agreeing,
/// since everything interesting is buried inside one string. This walks in.
///
/// Rust has no statement-level `if`: control flow is expressions, so the walk descends into the
/// expression forms that carry a block. A compound construct contributes a **header** line naming
/// what it tests (`If(cond)`, `Match(scrutinee)`, `For(pat, iter)`) with its body elided, and the
/// body follows one level deeper — the same shape Python's unparser produces, so the two languages'
/// streams are comparable rather than merely both present.
///
/// One `Dump` for the whole definition, so a local carries the same `_v{n}` across every line it
/// appears on. `type3_lines` uses a fresh `Dump` per line on purpose (order-invariance); here the
/// order *is* the signal, and a slot that means something different on each line would destroy it.
#[must_use]
pub fn statement_stream(name: &str, sig: &Signature, body: &Block, locals: &HashSet<String>) -> Vec<Statement> {
    let mut dump = Dump::new(Some(locals));
    // The definition's own header opens the stream at depth 0 and the body starts at depth 1 — the
    // contract every frontend agrees on, so a consumer cannot mistake a missing header for a
    // definition that opens with a statement.
    let head = dump.func(name, sig, None, true);
    let mut walk = StreamWalk { dump: &mut dump, out: vec![Statement { line: head, depth: 0 }] };
    walk.block(body, 1);
    walk.out
}

struct StreamWalk<'a, 'b> {
    dump: &'b mut Dump<'a>,
    out: Vec<Statement>,
}

impl StreamWalk<'_, '_> {
    fn push(&mut self, depth: u16, line: String) {
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
