//! **Use profiles** — canonicalizing a definition by its *surroundings* instead of its body.
//!
//! The body kinds answer "what is this thing"; this one answers "**how is it handled**". A
//! definition's profile is the multiset of statements across the whole tree that mention its name,
//! each canonicalized with the definition itself as the anchor. Two subsystems whose *bodies*
//! diverged — a cache on a JSONB column and a cache on typed columns share no AST — but whose
//! *handling* is identical (written by key, read by key, deleted by age) collapse to the same
//! profile canonical. That is the same primitive re-invented, and no body-based pass can see it,
//! because the duplication lives in the composition, not in any one definition.
//!
//! ## Why this needs no resolver
//!
//! The profile is assembled by **name**, with no import resolution and no call graph: a top-level
//! name is a de-facto unique key in a Python project (measured: 2435 distinct names across 2444
//! top-level classes in one production tree). This is not a new assumption — `pass_name_gated`
//! already groups `(kind, name)` across the whole tree without resolving anything. Same anchor,
//! different question.
//!
//! ## The canonical
//!
//! The alpha-rename boundary moves out by exactly one step. In the body canon, bound *locals* are
//! slots and everything else carries behaviour. Here the anchored definition's own identity joins
//! the slots:
//!
//! ```text
//!   select(ImageCache).where(ImageCache.key == k)
//!     → Call(Attribute(Call(Name('_t0'), …), 'where'), [Compare(Attribute(Name('_t0'), '_a0'), …)])
//! ```
//!
//! Nothing is extracted and nothing is filtered — the full `ast.dump` of each site survives, and
//! which parts matter is left to the engine's IDF, exactly as in the body passes. Two canonicals
//! are produced per profile: one keeping the anchor's attribute *names* (the name-gated pass then
//! finds one entity split across two definitions) and one erasing them to positional `_a{n}` (the
//! cross-name pass then finds the same shape under a different domain).
//!
//! ## Determinism
//!
//! Sites arrive in file order, which differs between two instances of the same primitive, so the
//! profile is sorted by its name-keeping canonical first; attribute slots are then numbered in
//! order of first appearance *along that sorted sequence*. Both canonicals are therefore
//! independent of how the tree happens to be laid out.

use std::collections::{BTreeSet, HashSet};

use dup_defs_core::uses::{site_ctx, Construct, Site, Within};
use ruff_python_ast::visitor::{self, Visitor};
use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_python_parser::parse_module;

use crate::canon::{canon_site, collect_locals};

// ── Site collection ────────────────────────────────────────────────────────

/// The *direct* expressions of one statement — those reachable without crossing into a nested
/// statement. A compound statement therefore contributes its header (`for … in <iter>:`,
/// `if <test>:`) and not its body, so a site's canonical stays the shape of the *use*, not of
/// whatever happened to be written under it.
struct DirectExprs<'a> {
    depth: usize,
    out: Vec<&'a Expr>,
}

impl<'a> Visitor<'a> for DirectExprs<'a> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        // depth 0 is the statement we were asked about; anything deeper is a nested statement and
        // belongs to its own site.
        if self.depth == 0 {
            self.depth += 1;
            visitor::walk_stmt(self, stmt);
            self.depth -= 1;
        }
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        // Take the top-level expression whole — no descent, the canonicalizer walks it itself.
        self.out.push(expr);
    }
}

fn direct_exprs(stmt: &Stmt) -> Vec<&Expr> {
    let mut v = DirectExprs { depth: 0, out: Vec::new() };
    v.visit_stmt(stmt);
    v.out
}

/// Every known name mentioned anywhere inside `exprs`.
struct NameHunt<'k> {
    known: &'k HashSet<&'k str>,
    found: Vec<String>,
}

impl<'a> Visitor<'a> for NameHunt<'_> {
    fn visit_expr(&mut self, expr: &'a Expr) {
        if let Expr::Name(n) = expr {
            let id = n.id.as_str();
            if self.known.contains(id) && !self.found.iter().any(|f| f == id) {
                self.found.push(id.to_owned());
            }
        }
        visitor::walk_expr(self, expr);
    }
}

/// The statement's node tag — grammar, not taxonomy: the same label `CPython`'s `ast` gives it.
fn stmt_tag(stmt: &Stmt) -> &'static str {
    match stmt {
        Stmt::FunctionDef(f) if f.is_async => "AsyncFunctionDef",
        Stmt::FunctionDef(_) => "FunctionDef",
        Stmt::ClassDef(_) => "ClassDef",
        Stmt::Return(_) => "Return",
        Stmt::Delete(_) => "Delete",
        Stmt::Assign(_) => "Assign",
        Stmt::AugAssign(_) => "AugAssign",
        Stmt::AnnAssign(_) => "AnnAssign",
        Stmt::For(f) if f.is_async => "AsyncFor",
        Stmt::For(_) => "For",
        Stmt::While(_) => "While",
        Stmt::If(_) => "If",
        Stmt::With(w) if w.is_async => "AsyncWith",
        Stmt::With(_) => "With",
        Stmt::Match(_) => "Match",
        Stmt::Raise(_) => "Raise",
        Stmt::Try(_) => "Try",
        Stmt::Assert(_) => "Assert",
        Stmt::Expr(_) => "Expr",
        Stmt::TypeAlias(_) => "TypeAlias",
        _ => "Stmt",
    }
}

/// Statements nested directly under `stmt` (bodies, else-branches, handlers, match cases).
fn child_stmts(stmt: &Stmt) -> Vec<&Stmt> {
    let mut out: Vec<&Stmt> = Vec::new();
    match stmt {
        Stmt::FunctionDef(n) => out.extend(&n.body),
        Stmt::ClassDef(n) => out.extend(&n.body),
        Stmt::For(n) => {
            out.extend(&n.body);
            out.extend(&n.orelse);
        }
        Stmt::While(n) => {
            out.extend(&n.body);
            out.extend(&n.orelse);
        }
        Stmt::If(n) => {
            out.extend(&n.body);
            for clause in &n.elif_else_clauses {
                out.extend(&clause.body);
            }
        }
        Stmt::With(n) => out.extend(&n.body),
        Stmt::Match(n) => {
            for case in &n.cases {
                out.extend(&case.body);
            }
        }
        Stmt::Try(n) => {
            out.extend(&n.body);
            for h in &n.handlers {
                let ast::ExceptHandler::ExceptHandler(h) = h;
                out.extend(&h.body);
            }
            out.extend(&n.orelse);
            out.extend(&n.finalbody);
        }
        _ => {}
    }
    out
}

/// The external method calls made anywhere in `stmts` — the co-occurrence neighbourhood of any
/// site inside that block.
fn block_calls(stmts: &[&Stmt]) -> Vec<String> {
    struct Calls {
        out: BTreeSet<String>,
    }
    impl<'a> Visitor<'a> for Calls {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if let Expr::Call(c) = expr {
                if let Expr::Attribute(a) = c.func.as_ref() {
                    self.out.insert(format!(".{}", a.attr.id.as_str()));
                }
            }
            visitor::walk_expr(self, expr);
        }
    }
    let mut v = Calls { out: BTreeSet::new() };
    for s in stmts {
        // Only the block's own statements, not what nests under them: a neighbour is something
        // written beside the call, not everything the surrounding function eventually does.
        for e in direct_exprs(s) {
            v.visit_expr(e);
        }
    }
    v.out.into_iter().collect()
}

/// The context a statement puts its expressions in, given what encloses it — see
/// [`dup_defs_core::uses::site_ctx`].
fn stmt_ctx(stmt: &Stmt, in_try: bool, in_loop: bool) -> &'static str {
    let construct = match stmt {
        Stmt::Try(_) => Construct::Try,
        Stmt::For(_) | Stmt::While(_) => Construct::Loop,
        Stmt::If(_) | Stmt::Assert(_) => Construct::Guard,
        _ => Construct::Plain,
    };
    site_ctx(construct, Within { in_try, in_loop })
}

/// Walk one statement tree, recording a site for every known name mentioned in a statement's
/// *direct* expressions. `locals` is the bound-local set of the enclosing function (empty at module
/// level), so a site's locals rename exactly as they would inside the body canon.
/// Everything the walk carries down: what counts as an anchor, what the enclosing function binds,
/// what stands beside the current block, and which enclosures the block sits in.
struct WalkCtx<'a> {
    src: &'a str,
    known: &'a HashSet<&'a str>,
    locals: &'a HashSet<String>,
    neighbours: &'a [String],
    in_try: bool,
    in_loop: bool,
}

fn walk(stmt: &Stmt, ctx: &WalkCtx, out: &mut Vec<Site>) {
    // A `ClassDef`'s direct expressions are its bases, metaclass and decorators — all of which say
    // what the class being *declared* is, not how the anchor is *handled*. Counting them as sites
    // gives every base class the same profile ("I get subclassed"), which is the codebase's
    // background rather than any subsystem's shape. The distinction is positional, the same way
    // `Collect` tells a bound target from a free value: subclassing is a declaration site.
    let exprs = if matches!(stmt, Stmt::ClassDef(_)) { Vec::new() } else { direct_exprs(stmt) };
    if !exprs.is_empty() {
        let mut hunt = NameHunt { known: ctx.known, found: Vec::new() };
        for e in &exprs {
            hunt.visit_expr(e);
        }
        let site_kind = stmt_ctx(stmt, ctx.in_try, ctx.in_loop);
        for name in hunt.found {
            let (canon, _nodes) = canon_site(stmt_tag(stmt), &exprs, ctx.src, ctx.locals, &name);
            out.push(Site { name, canon, ctx: site_kind, cooc: ctx.neighbours.to_vec() });
        }
    }
    // A function introduces a fresh bound-local scope for everything under it.
    let nested;
    let inner_locals = if matches!(stmt, Stmt::FunctionDef(_)) {
        nested = collect_locals(stmt);
        &nested
    } else {
        ctx.locals
    };
    let children = child_stmts(stmt);
    let inner = WalkCtx {
        locals: inner_locals,
        neighbours: &block_calls(&children),
        in_try: ctx.in_try || matches!(stmt, Stmt::Try(_)),
        in_loop: ctx.in_loop || matches!(stmt, Stmt::For(_) | Stmt::While(_)),
        ..*ctx
    };
    for child in &children {
        walk(child, &inner, out);
    }
}

/// Every use site in one source: each statement that mentions a known name, anchored on it. The
/// path is not needed: a Python source parses one way.
pub(crate) fn collect_sites(source: &str, _path: &str, known: &HashSet<&str>) -> Vec<Site> {
    let Ok(parsed) = parse_module(source) else { return Vec::new() };
    let module = parsed.into_syntax();
    let empty: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    let top: Vec<&Stmt> = module.body.iter().collect();
    let ctx = WalkCtx {
        src: source,
        known,
        locals: &empty,
        neighbours: &block_calls(&top),
        in_try: false,
        in_loop: false,
    };
    for stmt in &module.body {
        walk(stmt, &ctx, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::collect_sites;
    use crate::defs::scan_source;
    use dup_defs_core::uses::use_facts;
    use dup_defs_core::ScanOpts;
    use std::collections::HashSet;
    use std::sync::Arc;

    #[test]
    fn a_site_is_the_statement_header_not_its_body() {
        let src = "def f():\n    for row in q(Tbl):\n        other(Unrelated)\n";
        let known: HashSet<&str> = ["Tbl"].into_iter().collect();
        let sites = collect_sites(src, "t.py", &known);
        assert_eq!(sites.len(), 1, "one site for Tbl");
        assert!(sites[0].canon.starts_with("For("), "site keeps the statement tag: {}", sites[0].canon);
        assert!(!sites[0].canon.contains("other"), "the body is a separate site, not part of this one");
    }

    /// The `use` lens's load-bearing claim: two definitions whose *bodies* share nothing, handled
    /// identically, produce the same use-site facts.
    #[test]
    fn divergent_bodies_with_identical_handling_share_their_use_facts() {
        let a = concat!(
            "class ImageCache:\n    key = col(String)\n    blob = col(LargeBinary)\n    dead_at = col(DateTime)\n\n",
            "def put_image(s, k, v):\n    s.add(ImageCache(key=k, blob=v, dead_at=soon()))\n\n",
            "def get_image(s, k):\n    return s.query(ImageCache).filter(ImageCache.key == k).one()\n\n",
            "def reap_images(s):\n    s.query(ImageCache).filter(ImageCache.dead_at < now()).delete()\n",
        );
        let b = concat!(
            "class PromptStore:\n    payload = col(JSONB)\n\n",
            "def put_prompt(s, k, v):\n    s.add(PromptStore(fingerprint=k, payload=v, expires=soon()))\n\n",
            "def get_prompt(s, k):\n    return s.query(PromptStore).filter(PromptStore.fingerprint == k).one()\n\n",
            "def reap_prompts(s):\n    s.query(PromptStore).filter(PromptStore.expires < now()).delete()\n",
        );
        let dir = std::env::temp_dir().join("fdd-use-facts-test");
        let _ = std::fs::create_dir_all(&dir);
        let (pa, pb) = (dir.join("a.py"), dir.join("b.py"));
        std::fs::write(&pa, a).unwrap();
        std::fs::write(&pb, b).unwrap();
        let files: Vec<Arc<str>> =
            vec![Arc::from(pa.to_string_lossy().as_ref()), Arc::from(pb.to_string_lossy().as_ref())];

        let opts = ScanOpts::default();
        let mut defs = scan_source(a, &files[0], &opts);
        defs.extend(scan_source(b, &files[1], &opts));
        // Bodies diverge: the two classes share no structural canonical.
        let body = |n: &str| {
            defs.iter().find(|d| d.name == n && d.kind.id == "classes").unwrap().cluster_canonical.clone()
        };
        assert_ne!(body("ImageCache"), body("PromptStore"), "bodies must diverge for this to be the point");

        let facts = use_facts(&[(a, "a.py", 1), (b, "b.py", 1)], &defs, collect_sites);
        let get = |n: &str| facts.get(n).unwrap_or_else(|| panic!("no use facts for {n}")).clone();
        assert_eq!(get("ImageCache"), get("PromptStore"), "identical handling ⇒ identical use facts");
    }
}
