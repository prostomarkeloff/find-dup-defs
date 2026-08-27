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

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::sync::Arc;

use dup_defs_core::Def;
use rayon::prelude::*;
use ruff_python_ast::visitor::{self, Visitor};
use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_python_parser::parse_module;

use crate::canon::{canon_site, collect_locals};

/// Profiles with fewer sites than this are dropped: a definition used once or twice carries no
/// *pattern of handling*, only an instance of one, and an exact match between two thin profiles is
/// trivially reachable (`raise X(...)` twice). The structural counterpart of the Type-3 pass's
/// `SHINGLE_LINES` floor and patternology's support floor.
const MIN_SITES: usize = 2;

/// Profiles with more sites than this are dropped. A name touched everywhere (a base model, a
/// framework symbol) has a profile that is the codebase's background rather than any one
/// subsystem's shape; it would also dominate the O(n²) work. The structural counterpart of the
/// engine's `--max-name-group`.
const MAX_SITES: usize = 300;

/// Minimum number of *distinct* site shapes a profile must have. A definition called the same way
/// three times (`x = f(y)` ×3) has three sites but one bit of information, and that one shape is a
/// language idiom every codebase repeats — so an exact match between two such profiles is
/// meaningless. Counting distinct shapes instead of sites is what separates "handled in several
/// ways" from "handled the one obvious way, repeatedly".
const MIN_SHAPES: usize = 2;

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

/// One collected use site: which name it anchors on, its canonical (attributes still marked
/// `#name#` for the caller to renumber), the syntactic context it sits in, and the external calls
/// standing beside it in the same block.
struct Site {
    name: String,
    canon: String,
    /// `guard` / `loop` / `try` / `plain` — the shape of the position the anchor is used in.
    /// Splitting the use channel this way follows the four kinds of usage pattern LUPIN mines
    /// (condition check, iteration, error handling, co-occurrence): the full canonical is
    /// all-or-nothing, so two call sites that agree on *being guarded* while differing in the
    /// guard's wording currently agree on nothing at all.
    ctx: &'static str,
    /// Method calls standing in the same block as this site. Not what the anchor calls
    /// (`outgoing`) but what is called *alongside* it — the `fopen`/`fclose` signal, and the one
    /// property here that belongs to neither definition on its own. Only attribute calls are
    /// kept: `.dumps` is grammar every module shares, while a bare name is an identity this
    /// module happens to have imported, and would tell two copies apart rather than together.
    cooc: Vec<String>,
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

/// The context a statement puts its expressions in, given what encloses it. Enclosure wins over
/// the statement's own tag: a plain call inside a `try` is error handling, whatever its shape.
fn site_ctx(stmt: &Stmt, in_try: bool, in_loop: bool) -> &'static str {
    if in_try {
        return "try";
    }
    match stmt {
        Stmt::Try(_) => "try",
        Stmt::For(_) | Stmt::While(_) => "loop",
        Stmt::If(_) | Stmt::Assert(_) => "guard",
        _ if in_loop => "loop",
        _ => "plain",
    }
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
        let site_kind = site_ctx(stmt, ctx.in_try, ctx.in_loop);
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

fn collect_sites(source: &str, known: &HashSet<&str>) -> Vec<Site> {
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

// ── Attribute marker rewriting ─────────────────────────────────────────────

/// Rewrite the `#name#` attribute markers a site canonical carries. `slots = None` restores the
/// bare attribute name (the name-keeping canonical); `Some(map)` replaces each with its positional
/// `_a{n}`, assigning the next free slot on first sight — so the caller controls numbering by the
/// order in which it feeds sites.
fn rewrite_marks(s: &str, slots: Option<&mut HashMap<String, u32>>) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut slots = slots;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'#' {
            // Push the whole run up to the next '#' at once.
            let start = i;
            while i < bytes.len() && bytes[i] != b'#' {
                i += 1;
            }
            out.push_str(&s[start..i]);
            continue;
        }
        // A marker is `#` + a Python identifier + `#`; anything else is literal text.
        let mut j = i + 1;
        if j < bytes.len() && (bytes[j].is_ascii_alphabetic() || bytes[j] == b'_') {
            j += 1;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
        }
        if j > i + 1 && j < bytes.len() && bytes[j] == b'#' {
            let attr = &s[i + 1..j];
            match slots.as_mut() {
                Some(map) => {
                    let next = u32::try_from(map.len()).unwrap_or(u32::MAX);
                    let slot = *map.entry(attr.to_owned()).or_insert(next);
                    let _ = write!(out, "_a{slot}");
                }
                None => out.push_str(attr),
            }
            i = j + 1;
        } else {
            out.push('#');
            i += 1;
        }
    }
    out
}

// ── Profiles ───────────────────────────────────────────────────────────────

/// One profile that cleared the floors: the anchor's name plus its sites in canonical order, in
/// both forms (attribute names kept / erased to `_a{n}`), with per-site node counts.
struct Profile<'a> {
    name: &'a str,
    /// Every fact this definition's use sites contribute, already prefixed by channel:
    /// `use:` the full site canonical, `usectx:` the kind of position it sits in, `cooc:` the
    /// calls standing beside it. Three channels rather than one so that partial agreement is
    /// expressible — two anchors guarded the same way but worded differently share `usectx`
    /// while `use` disagrees, which the single canonical could never say.
    facts: Vec<String>,
}

/// Turn the collected sites into the profiles that survive the floors. Sites are put in canonical
/// order *before* attribute slots are numbered, so neither canonical depends on file layout.
fn build_profiles(
    by_name: BTreeMap<&str, Vec<Site>>,
    min_sites: usize,
    max_sites: usize,
    min_shapes: usize,
) -> Vec<Profile<'_>> {
    by_name
        .into_iter()
        .filter(|(_, sites)| sites.len() >= min_sites && sites.len() <= max_sites)
        .filter_map(|(name, mut sites)| {
            sites.sort_by(|a, b| a.canon.cmp(&b.canon));
            let mut slots: HashMap<String, u32> = HashMap::new();
            let xname: Vec<String> =
                sites.iter().map(|s| rewrite_marks(&s.canon, Some(&mut slots))).collect();
            if xname.iter().collect::<HashSet<_>>().len() < min_shapes {
                return None;
            }
            // Contexts and neighbours are sets: a shape repeated across sites says nothing new,
            // and their whole point is to be coarser than the canonical.
            let ctxs: BTreeSet<&str> = sites.iter().map(|s| s.ctx).collect();
            let cooc: BTreeSet<&str> =
                sites.iter().flat_map(|s| s.cooc.iter().map(String::as_str)).collect();
            let mut facts: Vec<String> = xname.into_iter().map(|f| format!("use:{f}")).collect();
            facts.extend(ctxs.into_iter().map(|c| format!("usectx:{c}")));
            facts.extend(cooc.into_iter().map(|c| format!("cooc:{c}")));
            Some(Profile { name, facts })
        })
        .collect()
}

// ── Entry point ────────────────────────────────────────────────────────────

/// The use-site facts of every definition that has any, keyed by name and already prefixed by
/// channel (`use:` / `usectx:` / `cooc:`), merged into the lens records by
/// [`crate::lenses::merge_use_facts`]. `defs`
/// supplies the anchors (every top-level name the body scan already found); method names are
/// excluded because they are qualified (`Foo.bar`) and never appear as a bare `Name`.
#[must_use]
pub(crate) fn use_facts(files: &[Arc<str>], defs: &[Def]) -> HashMap<String, Vec<String>> {
    // Anchors: top-level names only, first declaration wins on a name collision (the same
    // convention `pass_name_gated` uses when one name covers several definitions).
    let mut decl: BTreeMap<&str, &Def> = BTreeMap::new();
    for d in defs {
        if d.kind.id == "methods" {
            continue;
        }
        decl.entry(d.name.as_str()).or_insert(d);
    }
    let known: HashSet<&str> = decl.keys().copied().collect();
    if known.is_empty() {
        return HashMap::new();
    }

    let per_file: Vec<Vec<Site>> = files
        .par_iter()
        .map(|f| fs::read_to_string(&**f).map_or_else(|_| Vec::new(), |src| collect_sites(&src, &known)))
        .collect();

    // Group sites by anchor. `files` is already sorted, so this is deterministic before the
    // canonical sort inside `build_profiles` even runs.
    let mut by_name: BTreeMap<&str, Vec<Site>> = BTreeMap::new();
    for sites in per_file {
        for site in sites {
            if let Some((k, _)) = decl.get_key_value(site.name.as_str()) {
                by_name.entry(k).or_default().push(site);
            }
        }
    }

    let profiles = build_profiles(by_name, MIN_SITES, MAX_SITES, MIN_SHAPES);
    profiles.into_iter().map(|p| (p.name.to_owned(), p.facts)).collect()
}

#[cfg(test)]
mod tests {
    use super::{collect_sites, rewrite_marks, use_facts};
    use crate::defs::scan_source;
    use dup_defs_core::ScanOpts;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    #[test]
    fn marker_rewrite_keeps_or_numbers_attributes() {
        let s = "Attribute(Name('_t0'), '#key#') Attribute(Name('_t0'), '#ttl#') Attribute(Name('_t0'), '#key#')";
        assert_eq!(
            rewrite_marks(s, None),
            "Attribute(Name('_t0'), 'key') Attribute(Name('_t0'), 'ttl') Attribute(Name('_t0'), 'key')"
        );
        let mut slots: HashMap<String, u32> = HashMap::new();
        assert_eq!(
            rewrite_marks(s, Some(&mut slots)),
            "Attribute(Name('_t0'), '_a0') Attribute(Name('_t0'), '_a1') Attribute(Name('_t0'), '_a0')"
        );
    }

    #[test]
    fn marker_rewrite_leaves_ordinary_hashes_alone() {
        let s = "Constant('a # b') Constant('#not an ident#')";
        assert_eq!(rewrite_marks(s, None), s);
    }

    #[test]
    fn a_site_is_the_statement_header_not_its_body() {
        let src = "def f():\n    for row in q(Tbl):\n        other(Unrelated)\n";
        let known: HashSet<&str> = ["Tbl"].into_iter().collect();
        let sites = collect_sites(src, &known);
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

        let facts = use_facts(&files, &defs);
        let get = |n: &str| facts.get(n).unwrap_or_else(|| panic!("no use facts for {n}")).clone();
        assert_eq!(get("ImageCache"), get("PromptStore"), "identical handling ⇒ identical use facts");
    }
}
