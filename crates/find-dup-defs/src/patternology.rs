//! **Patternology** — detection, clustering and *generalization* of structural meta-patterns
//! (the "repository pattern", the "build-add-flush-return create", the "guard-clause normalizer",
//! …): the same *shape* of code written in textually-divergent ways. This sits one rung more
//! abstract than the Type-3 pass: Type-3 keeps identifiers (renamed-but-present) and joins on
//! name-agnostic *lines*, so a family that drifted its names, its call targets and its statement
//! wording falls apart (one real-world codebase's `create` family fragmented at cos 0.29–0.57). Patternology throws
//! away identifiers, literals, attribute names and call targets entirely and joins on the bare
//! **AST node-type skeleton**, so those same functions land in one family.
//!
//! ## The skeleton is free
//!
//! We do not re-walk the AST. [`dup_defs_core::Analysis::xname_canonical`] is already a complete
//! `ast.dump(..., annotate_fields=False)` of the alpha-renamed function — the full preorder stream
//! of AST node constructors with every identifier/literal already reduced to a positional `_v{n}`
//! or a quoted value. The structural skeleton is exactly the sequence of *node-type constructors*
//! in that string (`FunctionDef`, `Assign`, `Call`, `Await`, `Return`, …), which we recover with a
//! cheap string scan (CapWord immediately followed by `(`, skipping quoted literal text). The
//! q-grams of that sequence are the structural shingles. No new parse, no new field on the
//! frontend contract — purely a re-featurization of data the scan already produced.
//!
//! ## The pipeline
//!
//! 1. **Featurize** — each function → multiset of structural q-grams (preorder node-type bi/tri-grams).
//! 2. **Detect** — IDF-weighted cosine over those grams via the same exact L2AP join
//!    ([`difflib_fast::simjoin`]) the Type-3 pass uses; keep pairs with `cos ≥ θ` whose *canonicals
//!    differ* (identical skeletons are exact dups the cross-name pass already owns).
//! 3. **Cluster** — a **greedy maximal-clique cover** of the similarity graph, NOT single-linkage
//!    connected components. This is the crux: a corpus saturated with one meta-pattern (e.g. a
//!    repository layer of near-identical CRUD skeletons) chains under union-find into one giant
//!    blob whose endpoints share nothing (min-sim → 0), because A~B and B~C edges merge A and C
//!    even when A≁C. A clique requires *every* pair to be an edge, so a family is genuinely
//!    mutually-similar (min-sim ≥ θ by construction). The cover is greedy (seed = highest-degree
//!    unclaimed vertex, grow by neighbors adjacent to all members) rather than exhaustive
//!    Bron–Kerbosch enumeration, so it stays polynomial on exactly those dense components — no
//!    blowup, no fallback. Connected components are computed first only as a cheap parallel partition.
//! 4. **Generalize** — per family, two artifacts. (a) The grams present in *every* member, ranked
//!    by IDF: the shared distinctive subtrees (a quick tag). (b) The **anti-unification template**:
//!    Plotkin's *least general generalization* (LGG) of the members' AST-dump trees, made robust to
//!    arity divergence (see [`lgg`]). Parse each member's `xname_canonical` into a term tree, then
//!    fold pairwise LGG across the family — a position where every member agrees is kept verbatim; a
//!    divergent literal/name becomes a hole `?`; and crucially a *structural* divergence (a node
//!    with an extra trailing field, a body or arg list of different length) no longer collapses the
//!    whole subtree — same-tag nodes anti-unify over their common prefix, and lists are LCS-aligned
//!    with holes only on the inserted/deleted elements. The result is a genuine *template + holes* —
//!    the meta-pattern's shared skeleton with its variation points made explicit, rendered back as
//!    one line of readable pseudo-source (e.g. `def _fn(): _v0 = ?(?, ?); ?.add(_v0); return _v0`
//!    for the build-add-return create, or `def _fn(_v0: ? | None): if _v0 is None: return None;
//!    return _v0.?` for the nullable-getter guard clause) rather than a raw `ast.dump`.
//!
//! Structural similarity is FP-noisier than textual, so the caller keeps these findings advisory
//! (never ERROR) — this is codometry / refactor-surfacing, not a hard gate.
#![allow(
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use difflib_fast::simjoin::{cosine_join_with, Corpus};
use difflib_fast::Concurrency;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::simgraph::{components, cosine};

/// AST node-type constructors that carry no structural signal: they appear 1:1 with their parent
/// (`Name`/`Attribute`/`Subscript` each drag a context node) and only dilute the vectors. Dropped
/// before gramming; everything else (including `arguments`/`arg`, which IDF down-weights) is kept.
const CTX_NOISE: &[&str] = &["Load", "Store", "Del"];

/// Minimum kept node-type-sequence length for a function to participate. A handful of nodes (a
/// `return []` one-liner) has no meta-pattern to speak of; this is the structural analogue of the
/// Type-3 `SHINGLE_LINES` floor.
pub const MIN_SKELETON_NODES: usize = 8;

/// Recover the preorder sequence of AST node-type constructors from an `ast.dump`-style canonical.
///
/// A node constructor is a `[A-Za-z_]\w*` token that **starts uppercase** and is **immediately
/// followed by `(`** — `FunctionDef(`, `Call(`, `Constant(`. Field names are lowercase (`name=`,
/// `args=`) and excluded; literal *values* live inside quotes and are skipped wholesale (so a
/// string constant `'Call(x)'` never injects a phantom `Call`). Renamed locals `_v{n}` start with
/// `_` (not uppercase) and are excluded. `Load`/`Store`/`Del` context nodes are dropped as noise.
#[must_use]
pub fn node_type_seq(canon: &str) -> Vec<&str> {
    let bytes = canon.as_bytes();
    let mut out: Vec<&str> = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        // Skip a quoted literal wholesale (single- or double-quoted, backslash escapes).
        if c == b'\'' || c == b'"' {
            let quote = c;
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == quote {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }
        // An identifier?
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            // Node constructor iff it starts uppercase AND is immediately followed by '('.
            if c.is_ascii_uppercase() && i < bytes.len() && bytes[i] == b'(' {
                let tag = &canon[start..i];
                if !CTX_NOISE.contains(&tag) {
                    out.push(tag);
                }
            }
            continue;
        }
        i += 1;
    }
    out
}

/// Bits per interned node-type id in a packed gram code. The node-type vocabulary is the CPython AST
/// grammar (~120 tags), so 20 bits (1M) is never approached — three ids (trigram) pack into 60 bits,
/// leaving the top 4 for the arity tag.
const TAG_ID_BITS: u32 = 20;

/// Pack a node-type **bi-gram** `(a, b)` losslessly into one `u64` (arity tag `1`). The caller's
/// structural q-grams (preorder node-type bi- and tri-grams) are kept as these codes — never Strings
/// — so featurization allocates nothing per gram. Two distinct bigrams always get distinct codes.
fn pack_bigram(a: u32, b: u32) -> u64 {
    debug_assert!(a < (1 << TAG_ID_BITS) && b < (1 << TAG_ID_BITS), "node-type vocab exceeds 2^20");
    (1u64 << 60) | (u64::from(a) << TAG_ID_BITS) | u64::from(b)
}

/// Pack a node-type **tri-gram** `(a, b, c)` losslessly into one `u64` (arity tag `2`).
fn pack_trigram(a: u32, b: u32, c: u32) -> u64 {
    debug_assert!(
        a < (1 << TAG_ID_BITS) && b < (1 << TAG_ID_BITS) && c < (1 << TAG_ID_BITS),
        "node-type vocab exceeds 2^20"
    );
    (2u64 << 60) | (u64::from(a) << (2 * TAG_ID_BITS)) | (u64::from(b) << TAG_ID_BITS) | u64::from(c)
}

/// Greedy maximal-clique cover of one connected component — the family extractor. Repeatedly: take
/// the highest-degree not-yet-claimed vertex as a seed, grow a clique by adding any unclaimed
/// neighbor adjacent to *all* current members (neighbors visited in degree order), emit it if size
/// ≥ 2, and claim its vertices. Polynomial and deterministic — unlike full Bron–Kerbosch maximal-
/// clique enumeration it cannot blow up on the dense components a repository-pattern-saturated
/// corpus produces (precisely where the single-linkage blob came from). It may not find the
/// *maximum* clique through a vertex, but every emitted family IS a clique, so its min pairwise
/// cosine is ≥ θ by construction — no loose blob can survive, and no budget/fallback is needed.
/// Each vertex lands in at most one family (the per-component contract); degree ties break to the
/// smaller index.
fn clique_families(comp: &[usize], adj: &[FxHashSet<usize>]) -> Vec<Vec<usize>> {
    if comp.len() <= 2 {
        return vec![comp.to_vec()];
    }
    // Seeds and candidates in descending degree (edges never cross components, so `adj[v].len()` is
    // the in-component degree), ties by index — high-degree-first makes the greedy find large cliques.
    let by_degree = |a: &usize, b: &usize| adj[*b].len().cmp(&adj[*a].len()).then(a.cmp(b));
    let mut seeds = comp.to_vec();
    seeds.sort_by(by_degree);

    let mut used: FxHashSet<usize> = FxHashSet::default();
    let mut out: Vec<Vec<usize>> = Vec::new();
    for &seed in &seeds {
        if used.contains(&seed) {
            continue;
        }
        let mut clique = vec![seed];
        let mut cands: Vec<usize> =
            adj[seed].iter().copied().filter(|w| !used.contains(w)).collect();
        cands.sort_by(by_degree);
        for w in cands {
            if clique.iter().all(|&c| adj[w].contains(&c)) {
                clique.push(w);
            }
        }
        // Claim the seed regardless (a size-1 result means it has no unclaimed clique partner);
        // emit only real (≥2) families.
        for &v in &clique {
            used.insert(v);
        }
        if clique.len() >= 2 {
            clique.sort_unstable();
            out.push(clique);
        }
    }
    out
}

/// One detected meta-pattern family: a clique of structurally-similar functions plus the folded
/// anti-unification term that generalizes them. Crate-internal — the public output is the
/// [`HelperCandidate`] derived from it.
#[derive(Clone, Debug)]
pub(crate) struct PatternFamily {
    /// Member indices into the input slices.
    pub members: Vec<usize>,
    /// Exact min pairwise cosine over all intra-family pairs — ≥ θ by construction, since every
    /// family is a clique (every pair is an edge ≥ θ).
    pub min_sim: f64,
    /// The family's folded LGG term (Plotkin anti-unification of the members' AST-dump trees) —
    /// shared skeleton with holes at the variation points. Fed to [`extractable`].
    pub term: Term,
}

/// A surfaced **helper candidate**: a family of functions whose shared shape collapses into one
/// parameterized helper (the DRY-violation framing — see [`extractable`]). The public output of the
/// whole-function and sub-block passes alike.
#[derive(Clone, Debug)]
pub struct HelperCandidate {
    /// Member indices into the input `xname_canonicals` slice — the call sites the helper replaces.
    pub members: Vec<usize>,
    /// Exact min pairwise structural cosine over the family (whole-fn pass); `1.0` for sub-block
    /// motifs (matched by exact statement-skeleton, no cosine).
    pub min_sim: f64,
    /// Distinct functions the motif spans (= `members.len()` for whole-fn; distinct host functions
    /// for sub-block). The codometry **support** count.
    pub support: usize,
    /// Stable signature key (holes `?`, atoms verbatim) — the cross-package `group-by` key.
    pub signature: String,
    /// Human-readable pseudo-source of the proposed helper body.
    pub body: String,
    /// Parameters the helper would take (= expression-holes).
    pub params: usize,
    /// Fixed shared-skeleton node count — proxy for the helper's size / the LOC a collapse absorbs.
    pub fixed: usize,
    /// `"whole-fn"` or `"sub-block"` — which pass produced this candidate.
    pub granularity: &'static str,
}

/// Cluster a set of functions' alpha-renamed canonicals into structurally-similar **clique families**
/// (greedy maximal-clique cover, NOT connected components — see the module doc for why single-linkage
/// blobs), and fold each family's anti-unification term. Input: each function's `xname_canonical`.
/// Output: one [`PatternFamily`] per family (size ≥ 2). The extractability filter and the cross-file
/// contract are applied downstream by [`whole_fn_helpers`] / the caller.
///
/// `theta` is the cosine floor for a structural edge; `concurrency` selects the simjoin backend
/// (same semantics as the Type-3 pass — CPU, or difflib-fast's Metal hybrid under a GPU mode).
#[must_use]
/// Per-function tf·idf row + L2 norm from sorted gram-id sequences. Each `seq` is sorted, so a run
/// of equal ids collapses to one weighted entry (`tf·idf`) and the row comes out in id order for the
/// merge-join — no O(K²) accumulate-scan, no post-sort. (An id's idf is constant, so summing `idf`
/// `tf` times equals `tf·idf`.)
fn tfidf_rows(seqs: &[Vec<u32>], idf: &[f64]) -> (Vec<Vec<(u32, f64)>>, Vec<f64>) {
    seqs.par_iter()
        .with_min_len(64)
        .map(|seq| {
            let mut v: Vec<(u32, f64)> = Vec::new();
            let mut i = 0;
            while i < seq.len() {
                let id = seq[i];
                let mut j = i + 1;
                while j < seq.len() && seq[j] == id {
                    j += 1;
                }
                v.push((id, (j - i) as f64 * idf[id as usize]));
                i = j;
            }
            let norm = v.iter().map(|&(_, p)| p * p).sum::<f64>().sqrt();
            (v, norm)
        })
        .unzip()
}

pub(crate) fn pattern_families(
    xname_canonicals: &[String],
    theta: f64,
    concurrency: Concurrency,
) -> Vec<PatternFamily> {
    let n = xname_canonicals.len();
    if n < 2 {
        return Vec::new();
    }

    // Featurize: each function → its multiset of structural grams (with repetition = tf). A gram is
    // the bi-/tri-gram of preorder node-type tags; we keep it as a *packed u64 code* — never a String
    // — so featurization allocates nothing per gram. The node-type vocabulary is the CPython AST
    // grammar (~120 tags), interned to small dense ids; two ids (bigram) or three (trigram) pack
    // losslessly into one u64 with a 2-bit arity tag (see [`pack_bigram`]/[`pack_trigram`]). The
    // u64→dim interning that follows is a plain bijection (cosine is invariant to which id a gram
    // gets), so this is byte-for-byte the same clustering the old String-gram path produced.
    let tag_seqs: Vec<Vec<&str>> = crate::timed("    pf:grams", || {
        xname_canonicals.par_iter().with_min_len(64).map(|c| node_type_seq(c)).collect()
    });
    let mut tag_id: FxHashMap<&str, u32> = FxHashMap::default();
    for seq in &tag_seqs {
        for &t in seq {
            let next = tag_id.len() as u32;
            tag_id.entry(t).or_insert(next);
        }
    }
    let code_lists: Vec<Vec<u64>> = tag_seqs
        .par_iter()
        .with_min_len(64)
        .map(|seq| {
            let mut g: Vec<u64> = Vec::new();
            for w in seq.windows(2) {
                g.push(pack_bigram(tag_id[w[0]], tag_id[w[1]]));
            }
            for w in seq.windows(3) {
                g.push(pack_trigram(tag_id[w[0]], tag_id[w[1]], tag_id[w[2]]));
            }
            g
        })
        .collect();

    // Intern distinct gram codes → dense dimension ids (any consistent bijection; cosine is invariant
    // to the assignment).
    let mut gram_id: FxHashMap<u64, u32> = FxHashMap::default();
    for codes in &code_lists {
        for &c in codes {
            let next = gram_id.len() as u32;
            gram_id.entry(c).or_insert(next);
        }
    }
    let n_grams = gram_id.len();

    // Per-function gram-id sequences (with repetition), each sorted once up front — the df pass and
    // the tf-vector build below both consume a sorted sequence (run-length over equal ids), so doing
    // it here avoids a per-function clone+sort in each.
    let seqs: Vec<Vec<u32>> = code_lists
        .par_iter()
        .with_min_len(64)
        .map(|codes| {
            let mut s: Vec<u32> = codes.iter().map(|c| gram_id[c]).collect();
            s.sort_unstable();
            s
        })
        .collect();

    // IDF over *document* frequency: df[g] = #functions whose gram-set contains g.
    //
    // Smoothed `idf = ln(n/df) + 1`, NOT the bare `ln(n/df)` the Type-3 text pass uses. For text,
    // a token in every document is a stop-word worth zero. For *structural* grams the opposite is
    // true: a shape shared across the corpus is still the backbone of a pattern, and a family's
    // shared skeleton must keep weight or the cosine would be driven entirely by the members'
    // *divergences* (zeroing the very thing patternology clusters on). The `+1` floor keeps every
    // gram ≥1 while rare/distinctive grams still dominate.
    let mut df: Vec<u32> = vec![0; n_grams];
    for seq in &seqs {
        // `seq` is sorted → each maximal run of equal ids is one distinct gram for this function.
        let mut prev: Option<u32> = None;
        for &id in seq {
            if prev != Some(id) {
                df[id as usize] += 1;
                prev = Some(id);
            }
        }
    }
    let idf: Vec<f64> =
        df.iter().map(|&d| if d == 0 { 0.0 } else { (n as f64 / f64::from(d)).ln() + 1.0 }).collect();

    let (rows, norms) = tfidf_rows(&seqs, &idf);

    // Exact all-pairs weighted-cosine join.
    let corpus = Corpus::from_rows(&rows);
    let pairs = crate::timed("    pf:cosine-join", || cosine_join_with(&corpus, theta, concurrency)); // (j, i, cos), j < i, cos ≥ theta

    // Edges: every structurally-similar pair. We deliberately do NOT drop canonical-identical pairs
    // here (the old patternology did): for *helper extraction* an exact-skeleton family is the
    // EASIEST, highest-value collapse — the cross-name pass reporting it as a "duplicate" is a
    // separate, weaker framing. Keeping them means the helper pass sees the full DRY surface.
    let edges: Vec<(usize, usize)> = pairs
        .into_par_iter()
        .filter(|&(_, _, cos)| cos >= theta)
        .map(|(j, i, _)| (j, i))
        .collect();

    // Adjacency for the clique search; symmetric, only edge-endpoints get a (non-empty) set.
    let mut adj: Vec<FxHashSet<usize>> = (0..n).map(|_| FxHashSet::default()).collect();
    for &(a, b) in &edges {
        adj[a].insert(b);
        adj[b].insert(a);
    }

    // Partition into components (cheap parallel unit), then greedy-clique-cover each component into
    // tight families (every family is a clique → min-sim ≥ θ).
    crate::timed("    pf:clique-fold", || {
    components(n, &edges)
        .into_par_iter()
        .flat_map_iter(|comp| {
            clique_families(&comp, &adj)
                .into_iter()
                .filter_map(|mut members| {
                    if members.len() < 2 {
                        return None;
                    }
                    members.sort_unstable();
                    // Min pairwise cosine over the whole family (≥ θ — every family is a clique).
                    let mut min_sim = 1.0f64;
                    for a in 0..members.len() {
                        for b in (a + 1)..members.len() {
                            let c = cosine(
                                &rows[members[a]],
                                &rows[members[b]],
                                norms[members[a]],
                                norms[members[b]],
                            );
                            min_sim = min_sim.min(c);
                        }
                    }
                    let term = fold_lgg(&members, xname_canonicals);
                    Some(PatternFamily { members, min_sim, term })
                })
                .collect::<Vec<_>>()
        })
        .collect()
    })
}

/// **Whole-function helper candidates** (deliverable A). Cluster the functions into structural
/// clique-families ([`pattern_families`]), then keep only those whose folded shape is *extractable*
/// into one parameterized helper ([`extractable`]) — every other family is the "vaguely-similar /
/// language-idiom" noise we drop. Each surviving family becomes one [`HelperCandidate`] (the helper
/// body + its call sites). This is the highest-value, lowest-risk collapse: N whole functions → one
/// parameterized function.
///
/// `theta` is the structural-cosine floor; `cfg` the extractability thresholds; `concurrency` the
/// simjoin backend.
#[must_use]
pub(crate) fn whole_fn_helpers(
    d: &dyn Dialect,
    xname_canonicals: &[String],
    theta: f64,
    cfg: &ExtractCfg,
    concurrency: Concurrency,
) -> Vec<HelperCandidate> {
    pattern_families(xname_canonicals, theta, concurrency)
        .into_iter()
        .filter_map(|fam| {
            let shape = extractable(d, &fam.term, cfg)?;
            Some(HelperCandidate {
                support: fam.members.len(),
                members: fam.members,
                min_sim: fam.min_sim,
                signature: shape.signature,
                body: shape.body,
                params: shape.params,
                fixed: shape.fixed,
                granularity: "whole-fn",
            })
        })
        .collect()
}

// ── Sub-block idiom mining (deliverable B) ───────────────────────────────────
//
// The whole-function pass only catches a motif when it IS (most of) a function. The dominant case
// the user cares about — an idiom *embedded* in larger, otherwise-different functions (a guard pair,
// a build→add→return run, a try/except-log) — is a recurring *sub-structure*, found by **support**
// (how many functions contain it), not pairwise similarity. We mine contiguous statement-windows:
// key each statement by its tag-skeleton, group identical windows across all functions, keep those
// recurring in ≥ `support_min` distinct functions, fold the instances' LGG, and keep the extractable
// ones. A greedy largest-window-first claim over (function, statement) cells keeps motifs maximal
// (so a 3-statement idiom isn't also reported as its three 2-statement sub-windows).

/// Max statement-window width the sub-block miner considers — real idioms are short; this bounds the
/// O(body·width) window enumeration.
const MAX_SUBBLOCK_WINDOW: usize = 6;

/// The body statement list of a parsed CPython `FunctionDef` term (child slot 2), empty if the term
/// is not a function or has no list body. Superseded in production by [`parse_function_body`] (which
/// skips the discarded signature subtrees); kept as the reference oracle its equivalence test checks.
#[cfg(test)]
fn function_body(term: &Term) -> Vec<Term> {
    match term {
        Term::Node(tag, ch) if base_tag(tag) == "FunctionDef" => match ch.get(2) {
            Some(Term::List(stmts)) => stmts.clone(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// A structural key for one statement: its node-type skeleton with every atom/hole flattened to `?`
/// (names, literals, operators dropped). Two statements share a key iff they have the same tag/arity
/// tree — so the miner groups `x = User(...)` with `y = Foo(...)`, and the LGG holes the difference.
fn skeleton_key(t: &Term) -> String {
    fn children(ch: &[Term], open: char, close: char, out: &mut String) {
        out.push(open);
        for (i, c) in ch.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            push(c, out);
        }
        out.push(close);
    }
    fn push(t: &Term, out: &mut String) {
        match t {
            Term::Hole | Term::Atom(_) => out.push('?'),
            Term::List(ch) => children(ch, '[', ']', out),
            Term::Node(tag, ch) => {
                out.push_str(base_tag(tag));
                if !ch.is_empty() {
                    children(ch, '(', ')', out);
                }
            }
        }
    }
    let mut out = String::new();
    push(t, &mut out);
    out
}

/// One occurrence of a statement-window: `(function index, start statement, end statement)`.
type WindowInst = (usize, usize, usize);

/// Distinct function count among a window's instances — the codometry support metric.
fn distinct_funcs(insts: &[WindowInst]) -> usize {
    let mut fs: Vec<usize> = insts.iter().map(|&(f, _, _)| f).collect();
    fs.sort_unstable();
    fs.dedup();
    fs.len()
}

/// **Sub-block helper candidates** (deliverable B). Mine recurring contiguous statement-windows
/// across the functions' bodies by support, fold each motif's LGG, and keep the extractable ones.
/// `support_min` is the minimum distinct functions a motif must recur in; `cfg` the extractability
/// thresholds.
#[must_use]
pub(crate) fn subblock_helpers(
    d: &dyn Dialect,
    xname_canonicals: &[String],
    support_min: usize,
    cfg: &ExtractCfg,
) -> Vec<HelperCandidate> {
    // Parse each function's body into its statement list; key each statement by its tag-skeleton.
    // `parse_function_body` builds only the body statements, not the discarded signature/annotation
    // trees that `parse_term` would.
    let bodies: Vec<Vec<Term>> =
        xname_canonicals.par_iter().with_min_len(64).map(|c| d.parse_function_body(c)).collect();
    let skel_strs: Vec<Vec<String>> =
        bodies.par_iter().with_min_len(64).map(|b| b.iter().map(skeleton_key).collect()).collect();

    // Intern statement skeleton-keys → dense ids so a window can be grouped/hashed as a cheap `[u32]`
    // slice (`Box<[u32]>`) instead of a freshly `join("|")`-ed String — the dominant allocation here
    // (a window per (function, width, start)). `id_str` keeps the reverse map so the survivor sort can
    // reconstruct the *exact* old joined-string tie-break, keeping the chosen motifs byte-identical.
    let mut key_id: FxHashMap<&str, u32> = FxHashMap::default();
    let mut id_str: Vec<&str> = Vec::new();
    let skels: Vec<Vec<u32>> = skel_strs
        .iter()
        .map(|row| {
            row.iter()
                .map(|k| {
                    let s = k.as_str();
                    if let Some(&id) = key_id.get(s) {
                        id
                    } else {
                        let id = id_str.len() as u32;
                        id_str.push(s);
                        key_id.insert(s, id);
                        id
                    }
                })
                .collect()
        })
        .collect();

    // Window-key → instances (func, start, end). A window is `w` consecutive statements, 2..=MAX.
    let mut groups: FxHashMap<Box<[u32]>, Vec<WindowInst>> = FxHashMap::default();
    for (f, sk) in skels.iter().enumerate() {
        let m = sk.len();
        for w in 2..=MAX_SUBBLOCK_WINDOW.min(m) {
            for start in 0..=(m - w) {
                groups.entry(sk[start..start + w].into()).or_default().push((f, start, start + w));
            }
        }
    }

    // The old String key was `keys.join("|")`; reproduce it from interned ids only where needed (the
    // sort tie-break over surviving keys) so ordering — hence which overlapping motif claims its cells
    // first — is unchanged.
    let joined = |k: &[u32]| -> String {
        let mut s = String::new();
        for (i, &id) in k.iter().enumerate() {
            if i > 0 {
                s.push('|');
            }
            s.push_str(id_str[id as usize]);
        }
        s
    };

    // Window-keys with enough distinct-function support, largest window first (then support, then key
    // for determinism) — so a maximal motif claims its statement cells before its sub-windows.
    let mut keys: Vec<&[u32]> = groups
        .iter()
        .filter(|(_, i)| distinct_funcs(i) >= support_min)
        .map(|(k, _)| k.as_ref())
        .collect();
    keys.sort_by(|a, b| {
        let (ia, ib) = (&groups[*a], &groups[*b]);
        let (wa, wb) = (ia[0].2 - ia[0].1, ib[0].2 - ib[0].1);
        wb.cmp(&wa)
            .then(distinct_funcs(ib).cmp(&distinct_funcs(ia)))
            .then_with(|| joined(a).cmp(&joined(b)))
    });

    let mut claimed: FxHashSet<(usize, usize)> = FxHashSet::default();
    let mut out: Vec<HelperCandidate> = Vec::new();
    for key in keys {
        // Keep only instances whose every statement cell is still unclaimed by a larger motif.
        let fresh: Vec<WindowInst> = groups[key]
            .iter()
            .copied()
            .filter(|&(f, s, e)| (s..e).all(|i| !claimed.contains(&(f, i))))
            .collect();
        let mut funcs: Vec<usize> = fresh.iter().map(|&(f, _, _)| f).collect();
        funcs.sort_unstable();
        funcs.dedup();
        if funcs.len() < support_min {
            continue;
        }
        // Claim the cells, then fold the LGG over the (statement-list) instances.
        let mut acc: Option<Term> = None;
        for &(f, s, e) in &fresh {
            for i in s..e {
                claimed.insert((f, i));
            }
            let block = Term::List(bodies[f][s..e].to_vec());
            acc = Some(match acc {
                None => block,
                Some(a) => lgg(&a, &block),
            });
        }
        let term = Term::Node(STMT_BLOCK_TAG.to_owned(), vec![acc.unwrap_or(Term::Hole)]);
        if let Some(shape) = extractable(d, &term, cfg) {
            out.push(HelperCandidate {
                support: funcs.len(),
                members: funcs,
                min_sim: 1.0,
                signature: shape.signature,
                body: shape.body,
                params: shape.params,
                fixed: shape.fixed,
                granularity: "sub-block",
            });
        }
    }
    out
}

// ── Anti-unification (Plotkin's least general generalization) ────────────────

/// A parsed `ast.dump` term: a node `Tag(child, …)`, a `[child, …]` list, an atom (a quoted
/// literal `'x'` or a bareword `None`/`True`/`42`/`_v0`), or a [`Term::Hole`] introduced by the LGG
/// at a position where the family disagrees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Term {
    Node(String, Vec<Term>),
    List(Vec<Term>),
    Atom(String),
    Hole,
}

/// Recursive-descent parser for the `ast.dump(annotate_fields=False)` shape `py-canon` emits (and
/// the renamed `xname_canonical` variant): `Tag(arg, …)` nodes, `[…]` lists, `'…'` quoted literals,
/// barewords, and `field=value` keyword args (the name is dropped — only the value's *shape*
/// matters for the structural template; node tag vs keyword is told apart by the trailing `(` vs `=`
/// after the identifier, so lowercase node tags like `arg`/`keyword` parse correctly).
struct TermParser<'a> {
    b: &'a [u8],
    s: &'a str,
    i: usize,
}

impl<'a> TermParser<'a> {
    fn new(s: &'a str) -> Self {
        Self { b: s.as_bytes(), s, i: 0 }
    }

    fn skip_ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i] == b' ' {
            self.i += 1;
        }
    }

    fn term(&mut self) -> Term {
        self.skip_ws();
        match self.b.get(self.i) {
            Some(b'\'' | b'"') => self.quoted(),
            Some(b'[') => {
                self.i += 1;
                Term::List(self.args(b']'))
            }
            Some(&c) if c.is_ascii_alphabetic() || c == b'_' => self.ident_term(),
            _ => self.bareword(),
        }
    }

    /// An identifier-led term: `Tag(…)` node, `name=value` keyword (name dropped → parse value), or
    /// a bareword atom (`None`/`True`/…).
    fn ident_term(&mut self) -> Term {
        let start = self.i;
        while self.i < self.b.len()
            && (self.b[self.i].is_ascii_alphanumeric() || self.b[self.i] == b'_')
        {
            self.i += 1;
        }
        let name = &self.s[start..self.i];
        match self.b.get(self.i) {
            Some(b'(') => {
                self.i += 1;
                let mut ch = self.args(b')');
                // Drop context nodes (`Load`/`Store`/`Del`) — pure noise that drags off every
                // `Name`/`Attribute`/`Subscript` (`Name('_v0', Load)`). Same set excluded from the
                // gram sequence; removing them uniformly keeps cross-member arity aligned for LGG
                // and lets `Name(id, Load)` render as just `id`.
                ch.retain(|c| !matches!(c, Term::Node(t, _) if CTX_NOISE.contains(&t.as_str())));
                Term::Node(name.to_owned(), ch)
            }
            Some(b'=') => {
                self.i += 1;
                self.term()
            }
            _ => Term::Atom(name.to_owned()),
        }
    }

    /// Comma-separated terms up to (and consuming) `close`. A per-iteration progress guard makes a
    /// malformed/foreign input terminate rather than spin.
    fn args(&mut self, close: u8) -> Vec<Term> {
        let mut out = Vec::new();
        loop {
            self.skip_ws();
            match self.b.get(self.i) {
                None => break,
                Some(&c) if c == close => {
                    self.i += 1;
                    break;
                }
                _ => {}
            }
            let before = self.i;
            out.push(self.term());
            self.skip_ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(&c) if c == close => {
                    self.i += 1;
                    break;
                }
                None => break,
                _ if self.i == before => self.i += 1, // guarantee progress on unexpected input
                _ => {}
            }
        }
        out
    }

    fn quoted(&mut self) -> Term {
        let q = self.b[self.i];
        let start = self.i;
        self.i += 1;
        while self.i < self.b.len() {
            if self.b[self.i] == b'\\' {
                self.i += 2;
                continue;
            }
            if self.b[self.i] == q {
                self.i += 1;
                break;
            }
            self.i += 1;
        }
        Term::Atom(self.s[start..self.i.min(self.b.len())].to_owned())
    }

    fn bareword(&mut self) -> Term {
        let start = self.i;
        while self.i < self.b.len() && !matches!(self.b[self.i], b',' | b')' | b']' | b' ') {
            self.i += 1;
        }
        if self.i == start {
            self.i += 1; // never stall
        }
        Term::Atom(self.s[start..self.i.min(self.b.len())].to_owned())
    }

    /// Advance past a quoted literal (single/double, backslash escapes) without building a `Term` —
    /// the cursor lands just after the closing quote.
    fn skip_quoted(&mut self) {
        let q = self.b[self.i];
        self.i += 1;
        while self.i < self.b.len() {
            if self.b[self.i] == b'\\' {
                self.i += 2;
                continue;
            }
            if self.b[self.i] == q {
                self.i += 1;
                break;
            }
            self.i += 1;
        }
    }

    /// Skip one comma-separated element (a whole balanced term — nested `()`/`[]`, quoted literals)
    /// of the *current* argument list and consume the trailing `,`, allocating nothing. Stops at the
    /// enclosing `)`/`]` (left unconsumed) or end of input. Used to step over discarded `FunctionDef`
    /// slots (name, arguments, …) when only the body is wanted.
    fn skip_arg(&mut self) {
        self.skip_ws();
        let mut depth = 0i32;
        while self.i < self.b.len() {
            match self.b[self.i] {
                b'\'' | b'"' => {
                    self.skip_quoted();
                    continue;
                }
                b'(' | b'[' => depth += 1,
                b')' | b']' if depth == 0 => return,
                b')' | b']' => depth -= 1,
                b',' if depth == 0 => {
                    self.i += 1;
                    return;
                }
                _ => {}
            }
            self.i += 1;
        }
    }
}

fn parse_term(s: &str) -> Term {
    TermParser::new(s).term()
}

/// Parse **only** the body statement-list of a `FunctionDef`/`AsyncFunctionDef` canonical — skipping
/// the discarded name / `arguments` (with all its annotation subtrees) / decorator / returns slots
/// without ever allocating `Term`s for them. Equal to `function_body(&parse_term(canon))` but it never
/// builds the signature trees, which (annotations and all) are the bulk of the sub-block miner's parse
/// cost. Empty for a non-`FunctionDef` or malformed input. (`PyDialect::parse_function_body`.)
fn py_parse_function_body(canon: &str) -> Vec<Term> {
    let mut p = TermParser::new(canon);
    p.skip_ws();
    let start = p.i;
    while p.i < p.b.len() && (p.b[p.i].is_ascii_alphanumeric() || p.b[p.i] == b'_') {
        p.i += 1;
    }
    if base_tag(&canon[start..p.i]) != "FunctionDef" || p.b.get(p.i) != Some(&b'(') {
        return Vec::new();
    }
    p.i += 1; // past '('
    p.skip_arg(); // slot 0: name
    p.skip_arg(); // slot 1: arguments(...)
    p.skip_ws();
    // slot 2 is the body — always a list literal; anything else is malformed → empty.
    match p.b.get(p.i) {
        Some(b'[') => {
            p.i += 1;
            p.args(b']')
        }
        _ => Vec::new(),
    }
}

/// Anti-unification of two terms — Plotkin's LGG made **robust to arity divergence** so a single
/// extra field / inserted statement no longer collapses an entire subtree to a hole (the dominant
/// cause of degenerate `?` templates on real code):
///
/// * **Same-tag nodes, possibly different arity** (async-insensitive: `AsyncFunctionDef` unifies
///   with `FunctionDef`, etc. — the sync/async mirror case) → recurse over the common *prefix* of
///   children.
///   `ast.dump` only ever omits *trailing* default fields, so the shorter node's children are a
///   prefix of the longer's (field K means the same thing in both); the surplus trailing fields are
///   simply unconstrained. `FunctionDef(name, args, body, decorators, returns)` vs the same without
///   `returns` thus keeps `def …:` + body instead of becoming `?`.
/// * **Lists** (statement bodies, arg lists, comparators) → LCS-aligned (see [`lgg_list`]): matched
///   elements anti-unify pairwise, runs of inserted/deleted elements collapse to a single hole. A
///   body `[A, B, C]` vs `[A, C]` generalizes to `[A, ?, C]`, not `?`.
/// * **Equal atoms** → kept; anything else → a hole.
///
/// Beyond equal-arity this is no longer a *strict* first-order generalization (you'd need insertion,
/// not just substitution, to recover an instance) — it is the least *common-subsequence* skeleton,
/// which is what a readable structural template wants. The anti-unifier of a *set* is the left fold
/// of this over the members; a position survives only if it aligns across EVERY member.
fn lgg(a: &Term, b: &Term) -> Term {
    match (a, b) {
        (Term::Node(ta, ca), Term::Node(tb, cb)) if base_tag(ta) == base_tag(tb) => {
            let n = ca.len().min(cb.len());
            // `AsyncFunctionDef`/`AsyncFor`/`AsyncWith` share field order with their sync twins, so a
            // sync/async mirror (the botocore↔aiobotocore type-5 case) unifies: keep the tag when
            // both agree, else drop to the sync base (asyncness becomes the variation point).
            let tag = if ta == tb { ta.clone() } else { base_tag(ta).to_owned() };
            Term::Node(tag, (0..n).map(|i| lgg(&ca[i], &cb[i])).collect())
        }
        (Term::List(ca), Term::List(cb)) => lgg_list(ca, cb),
        (Term::Atom(x), Term::Atom(y)) if x == y => Term::Atom(x.clone()),
        _ => Term::Hole,
    }
}

/// The async-insensitive node tag: `AsyncFunctionDef`/`AsyncFor`/`AsyncWith` → their sync base, so
/// the anti-unifier treats a sync/async mirror as the same shape.
fn base_tag(tag: &str) -> &str {
    tag.strip_prefix("Async").unwrap_or(tag)
}

/// Max LCS DP cells (`|a|·|b|`) before [`lgg_list`] gives up on alignment and falls back to a cheap
/// prefix zip + trailing hole. Real bodies/arg-lists are tiny; this only guards a pathological giant.
const LIST_ALIGN_MAX_CELLS: usize = 4096;

/// Two terms are *alignable* (count as a match in the list LCS) when their heads agree: same node
/// tag, both lists, equal atoms, or either side already a hole (a hole is a wildcard, so once a
/// position is generalized away in the fold it stays matched and stays a hole).
fn compatible(a: &Term, b: &Term) -> bool {
    match (a, b) {
        // A hole is a wildcard (so a generalized position stays matched through the fold); two lists
        // always align (their elements are aligned one level down).
        (Term::Hole, _) | (_, Term::Hole) | (Term::List(_), Term::List(_)) => true,
        (Term::Node(ta, _), Term::Node(tb, _)) => base_tag(ta) == base_tag(tb),
        (Term::Atom(x), Term::Atom(y)) => x == y,
        _ => false,
    }
}

/// Longest-common-subsequence alignment of two child sequences under [`compatible`], returned as
/// the matched `(i, j)` index pairs in increasing order.
fn lcs_align(ca: &[Term], cb: &[Term]) -> Vec<(usize, usize)> {
    let (n, m) = (ca.len(), cb.len());
    // dp[i][j] = LCS length of ca[i..] vs cb[j..]; extra row/col of zeros at n / m.
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if compatible(&ca[i], &cb[j]) {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let mut pairs = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if compatible(&ca[i], &cb[j]) {
            pairs.push((i, j));
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    pairs
}

/// Anti-unify two child lists. Equal length → positional zip (fast path, exact for the common case).
/// Differing length → LCS-align and anti-unify matched pairs, collapsing each run of unmatched
/// (inserted/deleted) elements into a single hole, so `[A, B, C]` vs `[A, C]` → `[A, ?, C]` rather
/// than `?`. Oversized lists fall back to a prefix zip + one trailing hole.
fn lgg_list(ca: &[Term], cb: &[Term]) -> Term {
    if ca.len() == cb.len() {
        return Term::List(ca.iter().zip(cb).map(|(x, y)| lgg(x, y)).collect());
    }
    if ca.len().saturating_mul(cb.len()) > LIST_ALIGN_MAX_CELLS {
        let n = ca.len().min(cb.len());
        let mut v: Vec<Term> = (0..n).map(|i| lgg(&ca[i], &cb[i])).collect();
        v.push(Term::Hole); // unaligned remainder
        return Term::List(v);
    }
    let pairs = lcs_align(ca, cb);
    let mut out: Vec<Term> = Vec::new();
    let (mut pi, mut pj) = (0usize, 0usize);
    for (i, j) in pairs {
        if i > pi || j > pj {
            out.push(Term::Hole); // a run of inserted/deleted elements
        }
        out.push(lgg(&ca[i], &cb[j]));
        pi = i + 1;
        pj = j + 1;
    }
    if pi < ca.len() || pj < cb.len() {
        out.push(Term::Hole); // trailing remainder
    }
    Term::List(out)
}

/// Node budget for [`render_term`] / [`py`] — templates of a 300-loc function would be unreadable in
/// full, so rendering stops after this many nodes (emitting `…`). The shared core of a real
/// meta-pattern is small; this only truncates pathological giants.
const LGG_NODE_BUDGET: usize = 80;

/// Larger budget for the *signature* serialization ([`signature`]): the signature is a join key, not
/// a display string, so truncation there would collide distinct motifs. Still bounded so a
/// pathological giant can't blow the string up.
const SIGNATURE_NODE_BUDGET: usize = 400;

fn render_children(ch: &[Term], out: &mut String, budget: &mut usize) {
    for (k, c) in ch.iter().enumerate() {
        if k > 0 {
            out.push_str(", ");
        }
        if *budget == 0 {
            out.push('…');
            return;
        }
        render_term(c, out, budget);
    }
}

/// Canonical `Tag(child, …)` serialization of an LGG term — holes as `?`, atoms verbatim, deterministic
/// pre-order. This is the **stable signature key**: two cohorts of the same idiom (in different
/// packages, folded over different members) produce the same string iff their fixed skeletons agree,
/// so an external codometry loop can `group-by` on it. Distinct from [`py`], which renders the same
/// term as human-readable pseudo-source. Bounded by `budget`.
fn render_term(t: &Term, out: &mut String, budget: &mut usize) {
    if *budget == 0 {
        out.push('…');
        return;
    }
    *budget -= 1;
    match t {
        Term::Hole => out.push('?'),
        Term::Atom(s) => out.push_str(s),
        Term::List(ch) => {
            out.push('[');
            render_children(ch, out, budget);
            out.push(']');
        }
        Term::Node(tag, ch) => {
            out.push_str(tag);
            if !ch.is_empty() {
                out.push('(');
                render_children(ch, out, budget);
                out.push(')');
            }
        }
    }
}

/// The stable signature key for an LGG term — [`render_term`] under [`SIGNATURE_NODE_BUDGET`].
fn signature(term: &Term) -> String {
    let mut out = String::new();
    let mut budget = SIGNATURE_NODE_BUDGET;
    render_term(term, &mut out, &mut budget);
    out
}

/// Fold [`lgg`] over a family's parsed canonicals → the anti-unification term (holes at the
/// positions members disagree). Separated from rendering so the same generalization feeds both the
/// readable pseudo-source note ([`py`]) and the tree-form tests.
fn fold_lgg(members: &[usize], canons: &[String]) -> Term {
    let mut acc: Option<Term> = None;
    for &m in members {
        let t = parse_term(&canons[m]);
        acc = Some(match acc {
            None => t,
            Some(a) => lgg(&a, &t),
        });
    }
    acc.unwrap_or(Term::Hole)
}

// ── Extractability — is this folded motif collapsible into a helper? ──────────
//
// A recurring AST *shape* is only worth surfacing if the recurrence is a genuine DRY violation —
// i.e. the members could be replaced by one parameterized helper. That is exactly the structure of
// the folded LGG: the fixed (non-hole) skeleton is the helper *body*, and each hole is a *variation
// point*. The decisive distinction (the user's "схлопывается в хелпер, нет?") is **where** the holes
// sit:
//
//   * an **expression-position hole** (a `Call` arg, a `Name` id, a `Constant` value, a `BinOp`
//     operand) is *bindable* — it becomes a helper parameter. `create(model, **kw)` over a holed
//     model name is a clean helper.
//   * a **statement-position hole** (an element of a body list, or a whole holed body) is *leaky* —
//     you cannot pass a statement as a value; the abstraction would not close. A family whose
//     divergence is "a different statement here" is *not* extractable, it is just vaguely-similar
//     code (the language-idiom noise, `if x is None: return None`, that must be dropped).
//
// So extractability is: a real skeleton (root is a Node), **no** statement-holes, few enough
// expression-holes that the helper has a sane arity, and a high fixed:hole ratio (mostly shared
// skeleton). The fixed part also gives the helper's *size* — and thus the LOC a collapse would save.

/// Which child slots of a compound-statement node tag hold **statement bodies** (lists of statements)
/// rather than expressions. A hole reached through one of these slots is a *statement* hole (leaky);
/// every other hole is an *expression* hole (bindable → helper param). Mirrors the body slots
/// [`py_node`] feeds to [`stmt_body`]. `Async*` tags share their sync twin's field order.
fn py_is_stmt_slot(tag: &str, k: usize) -> bool {
    match base_tag(tag) {
        "Module" => k == 0,
        "FunctionDef" | "For" | "ExceptHandler" => k == 2,
        "While" | "With" => k == 1,
        "If" => k == 1 || k == 2,        // body, orelse
        "Try" => k <= 3,                 // body, handlers, orelse, finalbody
        "ClassDef" => k == 3,            // name, bases, keywords, body, …
        _ => false,
    }
}

/// Synthetic node tag wrapping a sub-block statement-window so it has a single Node root (which
/// [`extractable`] requires) and so [`measure`] / [`is_stmt_slot`] treat its child list's elements as
/// statements. Chosen not to collide with any real CPython `ast` node.
const STMT_BLOCK_TAG: &str = "__stmts__";

/// Lexical context of a position during the extractability walk.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ctx {
    Stmt,
    Expr,
    /// A *name-identity* slot: the `attr` of an `Attribute`, the callee of a `Call`, a kwarg name. A
    /// hole here is the *selector* (which method / function / keyword), not a value — it cannot become
    /// a parameter without `getattr`/reflection, so it disqualifies the helper just like a stmt-hole.
    Selector,
    /// Signature material — a type annotation (`arg`/`AnnAssign`/`FunctionDef` returns) or the synthetic
    /// helper name (`_fn`). Counts toward `fixed` (it renders) but never as an *anchor*: anchors measure
    /// shared *logic*, and a function sharing only its signature shape (`def f(x: str) -> bool: return ?`)
    /// has nothing in its body to collapse.
    Anno,
}

/// Per-language adapter for the (otherwise language-agnostic) patternology engine. One zero-sized
/// impl per frontend dialect; the engine holds `&dyn Dialect` and calls these at the four coupling
/// seams that genuinely differ between languages: slot classification, function-body extraction (two
/// flavors — targeted parse + borrow), and pseudo-source rendering. Everything else (parsing, LGG,
/// clique cover, featurization, extractability arithmetic) is shared.
///
/// Two seams the design originally put here — [`base_tag`] (`Async*` folding) and [`is_ctx_noise`]
/// (`Load`/`Store`/`Del`) — turned out to be *behaviorally shared* across the current frontends
/// (Rust/TS have no `Async*`-prefixed tags nor `Load`/`Store`/`Del` nodes, so the Python rules are
/// no-ops there) and stay free engine functions. If a future dialect ever diverges on those, lift
/// them onto this trait then. `STMT_BLOCK_TAG` is engine-synthetic and handled *before* delegating,
/// so dialects never see it.
pub(crate) trait Dialect: Sync {
    /// Intrinsic lexical context of child slot `k` of node `tag` (before the engine applies the
    /// Anno-stickiness inherited from the parent). Drives the stmt/expr/selector/anno hole split.
    fn slot_ctx(&self, tag: &str, k: usize) -> Ctx;

    /// Parse ONLY the body statement list of a function canonical, skipping the discarded signature
    /// subtrees (the sub-block miner's hot path). Empty for a non-function / malformed input. (The
    /// whole-function path renders the entire term, so it needs no separate body-borrow seam.)
    fn parse_function_body(&self, canon: &str) -> Vec<Term>;

    /// Render a folded term as one line of readable pseudo-source (holes → `?`), bounded by `budget`,
    /// falling back to the raw `Tag(child, …)` form for tags it has no rule for.
    fn render(&self, t: &Term, out: &mut String, budget: &mut usize);
}

/// The Python (`CPython ast.dump`) dialect — the original and, until the Rust dialect lands, only
/// shape the engine ran on. All CPython-specific slot tables, body extraction, and the `def …:`
/// renderer live behind this impl.
pub(crate) struct PyDialect;

impl Dialect for PyDialect {
    fn slot_ctx(&self, tag: &str, k: usize) -> Ctx {
        if py_is_anno_slot(tag, k) {
            Ctx::Anno
        } else if py_is_stmt_slot(tag, k) {
            Ctx::Stmt
        } else if py_is_selector_slot(tag, k) {
            Ctx::Selector
        } else {
            Ctx::Expr
        }
    }

    fn parse_function_body(&self, canon: &str) -> Vec<Term> {
        py_parse_function_body(canon)
    }

    fn render(&self, t: &Term, out: &mut String, budget: &mut usize) {
        py(t, out, budget);
    }
}

/// The Rust dialect — `rs-canon`'s `syn`-derived structural dump. Same `Tag(field, …)` S-shape as
/// Python, but a different vocabulary: a function is `Func(name, Params(…), ret, 'flags', Block(…))`
/// (body at slot 4, a `Block` node whose children are the statements — `rs-canon` *splices* list
/// elements as node children rather than wrapping them in a `[…]` list, so a `Block`/`Params`/`Tuple`
/// never parses to a `Term::List`). Selectors are `Method`/`Field`/`FieldVal` name slots; type/`Param`
/// slots are signature material.
pub(crate) struct RustDialect;

impl Dialect for RustDialect {
    fn slot_ctx(&self, tag: &str, k: usize) -> Ctx {
        match (tag, k) {
            // Every child of a `Block` is a statement (spliced, not a list); If/For/While/Loop bodies
            // are themselves `Block` nodes, so this one rule covers all statement positions.
            ("Block", _) => Ctx::Stmt,
            // Name-identity slots: a varying method/field/struct-field name needs reflection, not a param.
            ("Method" | "Field", 1) | ("FieldVal" | "PatField", 0) => Ctx::Selector,
            // Signature material — parameter / ascription types, the fn name / return type / flags literal.
            ("Param" | "PatType", 1) | ("Func", 0 | 2 | 3) => Ctx::Anno,
            _ => Ctx::Expr,
        }
    }

    fn parse_function_body(&self, canon: &str) -> Vec<Term> {
        rs_parse_function_body(canon)
    }

    fn render(&self, t: &Term, out: &mut String, budget: &mut usize) {
        rs_render(t, out, budget);
    }
}

/// Parse only the body of a Rust `Func(name, Params, ret, 'flags', Block(stmts…))` — skip the four
/// signature slots without building them, then return the `Block` node's children (the statements).
fn rs_parse_function_body(canon: &str) -> Vec<Term> {
    let mut p = TermParser::new(canon);
    p.skip_ws();
    let start = p.i;
    while p.i < p.b.len() && (p.b[p.i].is_ascii_alphanumeric() || p.b[p.i] == b'_') {
        p.i += 1;
    }
    if &canon[start..p.i] != "Func" || p.b.get(p.i) != Some(&b'(') {
        return Vec::new();
    }
    p.i += 1; // past '('
    p.skip_arg(); // name
    p.skip_arg(); // Params(…)
    p.skip_arg(); // ret type
    p.skip_arg(); // 'flags'
    p.skip_ws();
    match p.term() {
        Term::Node(tag, ch) if tag == "Block" => ch, // spliced statement children
        _ => Vec::new(),
    }
}

// ── Rust pseudo-source renderer ──────────────────────────────────────────────

/// An atom/hole in name position (`Path` id, a method/field name, a `Bind` name): strip quotes, hole
/// → `?`.
fn rs_name(t: Option<&Term>) -> String {
    match t {
        Some(Term::Atom(s)) => unq(s).to_owned(),
        _ => "?".to_owned(),
    }
}

/// Comma-join node children `ch[from..]` via [`rs_render`].
fn rs_seq(ch: &[Term], from: usize, out: &mut String, b: &mut usize) {
    for (i, c) in ch.iter().enumerate().skip(from) {
        if i > from {
            out.push_str(", ");
        }
        if *b == 0 {
            out.push('…');
            return;
        }
        rs_render(c, out, b);
    }
}

/// Render a `Block` node's statement children joined with `; ` (a bare body, no braces).
fn rs_block_inner(t: Option<&Term>, out: &mut String, b: &mut usize) {
    match t {
        Some(Term::Node(tag, ch)) if tag == "Block" => {
            for (i, s) in ch.iter().enumerate() {
                if i > 0 {
                    out.push_str("; ");
                }
                if *b == 0 {
                    out.push('…');
                    return;
                }
                rs_render(s, out, b);
            }
        }
        Some(other) => rs_render(other, out, b),
        None => {}
    }
}

/// Render a folded Rust term as one line of pseudo-source. Mirrors [`py`]: bounded by `b`, holes `?`,
/// unknown tags fall back to the raw `Tag(child, …)` form ([`render_term`]).
fn rs_render(t: &Term, out: &mut String, b: &mut usize) {
    if *b == 0 {
        out.push('…');
        return;
    }
    *b -= 1;
    match t {
        Term::Hole => out.push('?'),
        Term::Atom(s) => out.push_str(unq(s)),
        Term::List(items) => {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                rs_render(it, out, b);
            }
            out.push(']');
        }
        Term::Node(tag, ch) => rs_node(tag, ch, out, b),
    }
}

#[allow(clippy::too_many_lines)]
fn rs_node(tag: &str, ch: &[Term], out: &mut String, b: &mut usize) {
    match tag {
        "Func" => {
            out.push_str("fn ");
            out.push_str(&rs_name(ch.first()));
            out.push('(');
            if let Some(Term::Node(pt, pch)) = ch.get(1) {
                if pt == "Params" {
                    rs_seq(pch, 0, out, b);
                }
            }
            out.push(')');
            if let Some(r) = ch.get(2) {
                if matches!(r, Term::Node(..)) {
                    out.push_str(" -> ");
                    rs_render(r, out, b);
                }
            }
            out.push_str(" { ");
            rs_block_inner(ch.get(4), out, b);
            out.push_str(" }");
        }
        "Param" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            if let Some(ty) = ch.get(1) {
                out.push_str(": ");
                rs_render(ty, out, b);
            }
        }
        "Block" => {
            out.push_str("{ ");
            rs_block_inner(Some(&Term::Node(tag.to_owned(), ch.to_vec())), out, b);
            out.push_str(" }");
        }
        // `let pat = init` (statement) and the `if let`/`let … else` expression render identically.
        "Let" | "LetExpr" => {
            out.push_str("let ");
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            if let Some(init) = ch.get(1) {
                out.push_str(" = ");
                rs_render(init, out, b);
            }
        }
        "ExprStmt" | "Tail" | "BlockExpr" => rs_render(ch.first().unwrap_or(&Term::Hole), out, b),
        // Name-position leaves: a path/binding id or a bare numeric/bool literal — render the value.
        "Path" | "PatPath" | "Bind" | "Int" | "Float" | "Bool" | "Byte" => {
            out.push_str(&rs_name(ch.first()));
        }
        "Wild" | "TyInfer" => out.push('_'),
        "Rest" => out.push_str(".."),
        "Str" => {
            out.push('"');
            out.push_str(&rs_name(ch.first()));
            out.push('"');
        }
        "Char" => {
            out.push('\'');
            out.push_str(&rs_name(ch.first()));
            out.push('\'');
        }
        "Bin" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push(' ');
            out.push_str(&rs_name(ch.get(1)));
            out.push(' ');
            rs_render(ch.get(2).unwrap_or(&Term::Hole), out, b);
        }
        "Unary" => {
            out.push_str(&rs_name(ch.first()));
            rs_render(ch.get(1).unwrap_or(&Term::Hole), out, b);
        }
        "Assign" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push_str(" = ");
            rs_render(ch.get(1).unwrap_or(&Term::Hole), out, b);
        }
        "Call" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push('(');
            rs_seq(ch, 1, out, b);
            out.push(')');
        }
        "Method" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push('.');
            out.push_str(&rs_name(ch.get(1)));
            out.push('(');
            rs_seq(ch, 2, out, b);
            out.push(')');
        }
        "Field" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push('.');
            out.push_str(&rs_name(ch.get(1)));
        }
        "Index" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push('[');
            rs_render(ch.get(1).unwrap_or(&Term::Hole), out, b);
            out.push(']');
        }
        "If" => {
            out.push_str("if ");
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push(' ');
            rs_render(ch.get(1).unwrap_or(&Term::Hole), out, b);
            if let Some(els) = ch.get(2) {
                if matches!(els, Term::Node(..)) {
                    out.push_str(" else ");
                    rs_render(els, out, b);
                }
            }
        }
        "Match" => {
            out.push_str("match ");
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push_str(" { ");
            rs_seq(ch, 1, out, b);
            out.push_str(" }");
        }
        "Arm" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            // guard (slot 1) only when it is a real node; the body is the last child.
            if matches!(ch.get(1), Some(Term::Node(..))) {
                out.push_str(" if ");
                rs_render(&ch[1], out, b);
            }
            out.push_str(" => ");
            if let Some(body) = ch.last() {
                rs_render(body, out, b);
            }
        }
        "For" => {
            out.push_str("for ");
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push_str(" in ");
            rs_render(ch.get(1).unwrap_or(&Term::Hole), out, b);
            out.push(' ');
            rs_render(ch.get(2).unwrap_or(&Term::Hole), out, b);
        }
        "While" => {
            out.push_str("while ");
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push(' ');
            rs_render(ch.get(1).unwrap_or(&Term::Hole), out, b);
        }
        "Loop" => {
            out.push_str("loop ");
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
        }
        "Return" => {
            out.push_str("return");
            if let Some(e) = ch.first() {
                out.push(' ');
                rs_render(e, out, b);
            }
        }
        "Break" => {
            out.push_str("break");
            if let Some(e) = ch.first() {
                out.push(' ');
                rs_render(e, out, b);
            }
        }
        "Continue" => out.push_str("continue"),
        // `&expr` / `&mut expr` — value reference and type reference share the `&['mut'] inner` shape.
        "Ref" | "TyRef" => {
            out.push('&');
            if rs_name(ch.first()) == "mut" {
                out.push_str("mut ");
            }
            rs_render(ch.get(1).unwrap_or(&Term::Hole), out, b);
        }
        "Try" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push('?');
        }
        "Await" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push_str(".await");
        }
        "Cast" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push_str(" as ");
            rs_render(ch.get(1).unwrap_or(&Term::Hole), out, b);
        }
        "Unsafe" => {
            out.push_str("unsafe ");
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
        }
        "Async" => {
            out.push_str("async ");
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
        }
        "TryBlock" => {
            out.push_str("try ");
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
        }
        "Tuple" | "PatTuple" | "TyTuple" => {
            out.push('(');
            rs_seq(ch, 0, out, b);
            out.push(')');
        }
        "Array" | "PatSlice" => {
            out.push('[');
            rs_seq(ch, 0, out, b);
            out.push(']');
        }
        "Range" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push_str("..");
            if let Some(hi) = ch.get(1) {
                rs_render(hi, out, b);
            }
        }
        "StructLit" | "PatStruct" => {
            out.push_str(&rs_name(ch.first()));
            out.push_str(" { ");
            rs_seq(ch, 1, out, b);
            out.push_str(" }");
        }
        "FieldVal" | "PatField" => {
            out.push_str(&rs_name(ch.first()));
            out.push_str(": ");
            rs_render(ch.get(1).unwrap_or(&Term::Hole), out, b);
        }
        "Closure" => {
            // `rs-canon` splices the params then the body as one field list — body is the LAST child,
            // the (zero or more) params precede it (an empty-param closure leaves a stray `,` atom we skip).
            out.push('|');
            let n = ch.len();
            let mut first = true;
            for p in ch.iter().take(n.saturating_sub(1)) {
                if matches!(p, Term::Atom(s) if unq(s).trim_matches(',').is_empty()) {
                    continue;
                }
                if !first {
                    out.push_str(", ");
                }
                first = false;
                rs_render(p, out, b);
            }
            out.push_str("| ");
            if let Some(body) = ch.last() {
                rs_render(body, out, b);
            }
        }
        "Macro" | "MacroStmt" => {
            out.push_str(&rs_name(ch.first()));
            out.push_str("!(…)");
        }
        "Ty" => {
            out.push_str(&rs_name(ch.first()));
            if let Some(args) = ch.get(1) {
                if matches!(args, Term::Node(..)) {
                    out.push('<');
                    rs_render(args, out, b);
                    out.push('>');
                }
            }
        }
        "PatRef" => {
            out.push('&');
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
        }
        // Tuple-struct / enum-variant pattern: `Path(p, …)`. Slot 0 is the path name (spliced pats follow).
        "PatTupleStruct" => {
            out.push_str(&rs_name(ch.first()));
            out.push('(');
            rs_seq(ch, 1, out, b);
            out.push(')');
        }
        // Type-ascription pattern: `pat: ty`.
        "PatType" => {
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push_str(": ");
            rs_render(ch.get(1).unwrap_or(&Term::Hole), out, b);
        }
        "TySlice" => {
            out.push('[');
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push(']');
        }
        "TyArray" => {
            out.push('[');
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push_str("; _]");
        }
        "TyPtr" => {
            out.push('*');
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
        }
        "TyImpl" => out.push_str("impl ?"),
        "TyDyn" => out.push_str("dyn ?"),
        "TyNever" => out.push('!'),
        "TyFn" => out.push_str("fn(?)"),
        "Repeat" => {
            out.push('[');
            rs_render(ch.first().unwrap_or(&Term::Hole), out, b);
            out.push_str("; ");
            rs_render(ch.get(1).unwrap_or(&Term::Hole), out, b);
            out.push(']');
        }
        // Unknown / unhandled node: compact `Tag(child, …)` fallback (keeps holes visible).
        _ => {
            out.push_str(tag);
            if !ch.is_empty() {
                out.push('(');
                rs_seq(ch, 0, out, b);
                out.push(')');
            }
        }
    }
}

/// Tallies from walking a folded LGG term: fixed (non-hole) nodes/atoms — the helper body's size —
/// plus holes split by the context they sit in, and the set of distinct *anchor* names (shared
/// identifiers/literals, excluding bound-variable placeholders) that make the motif domain-specific.
#[derive(Default, Clone)]
struct Measure {
    /// Non-hole nodes + atoms: the shared skeleton the helper would keep verbatim.
    fixed: usize,
    /// Holes in expression (value) position — each becomes a helper parameter.
    expr_holes: usize,
    /// Holes in statement position — leaky, cannot be a parameter; their presence kills extractability.
    stmt_holes: usize,
    /// Holes in name-identity (selector) position — `obj.?()`, `?(args)`, `?=val` kwarg. Not bindable
    /// without reflection; their presence kills extractability (a varying *callee* is not one helper).
    selector_holes: usize,
    /// Distinct shared anchor names — fixed identifiers/attrs/literals minus `_vN` LGG placeholders.
    /// A skeleton of pure structure with no shared anchor (`? = ?; ? = ?`) is a syntactic coincidence,
    /// not an extractable idiom.
    anchors: FxHashSet<String>,
}

/// True when a child slot `k` of node `tag` is a name-identity (selector) position — see [`Ctx::Selector`].
/// Note a *callee* hole (`Call(?, …)` / `Call(Name(?), …)`) is deliberately **not** a selector: a
/// varying class/function can be passed as a callable parameter (`def h(factory, x): factory(x)`),
/// which is ordinary Python, not reflection. Only a varying *member* name (`obj.?`) or *kwarg* name
/// (`f(?=v)`) would need `getattr`/`**{name: v}` and so cannot become a clean parameter.
fn py_is_selector_slot(tag: &str, k: usize) -> bool {
    match base_tag(tag) {
        "Attribute" => k == 1, // Attribute(value, attr): attr is the member name
        "keyword" => k == 0,   // keyword(arg, value): arg is the kwarg name
        _ => false,
    }
}

/// True when a child slot `k` of node `tag` is signature material — a type annotation or the synthetic
/// helper name — that should not count as an anchor. See [`Ctx::Anno`].
fn py_is_anno_slot(tag: &str, k: usize) -> bool {
    match base_tag(tag) {
        "FunctionDef" => k == 0 || k == 4, // name, returns (annotation)
        // arg(arg, annotation, type_comment) · AnnAssign(target, annotation, value, simple)
        "arg" | "AnnAssign" => k == 1,
        _ => false,
    }
}

/// Walk a term tallying fixed nodes and context-split holes. `ctx` is the position of `t` itself:
/// the root of a whole function is [`Ctx::Expr`] (the `FunctionDef` node is not inside a body list;
/// its own body slot re-enters [`Ctx::Stmt`]); the root of a bare statement-block is [`Ctx::Stmt`].
fn measure(d: &dyn Dialect, term: &Term, ctx: Ctx, m: &mut Measure) {
    match term {
        Term::Hole => match ctx {
            Ctx::Stmt => m.stmt_holes += 1,
            Ctx::Expr | Ctx::Anno => m.expr_holes += 1,
            Ctx::Selector => m.selector_holes += 1,
        },
        Term::Atom(s) => {
            m.fixed += 1;
            // A shared fixed atom is an anchor only if it is a *named* identifier/attr/literal — it must
            // contain a letter — AND it is real logic, not signature material (`Ctx::Anno`). This drops
            // (a) LGG bound-variable placeholders (`_v0`); (b) bare structural flags
            // (`AnnAssign.simple = 1`) and numerals; (c) type annotations and the synthetic `_fn` name —
            // so `def f(x: str) -> bool: return ?` (no shared *body* logic) no longer clears `min_anchors`.
            let u = unq(s);
            if ctx != Ctx::Anno && !is_placeholder(u) && u.bytes().any(|b| b.is_ascii_alphabetic()) {
                m.anchors.insert(u.to_owned());
            }
        }
        Term::List(items) => {
            for it in items {
                measure(d, it, ctx, m); // a list inherits its slot's context
            }
        }
        Term::Node(tag, ch) => {
            m.fixed += 1;
            for (k, c) in ch.iter().enumerate() {
                // Signature material is sticky: once inside an annotation, stay there so nested type
                // names (`list[int]`) are all excluded from anchors. `STMT_BLOCK_TAG` is the
                // engine-synthetic sub-block root — its slot 0 is the statement window; everything
                // else is the dialect's call.
                let child_ctx = if ctx == Ctx::Anno {
                    Ctx::Anno
                } else if tag == STMT_BLOCK_TAG {
                    if k == 0 { Ctx::Stmt } else { Ctx::Expr }
                } else {
                    d.slot_ctx(tag, k)
                };
                measure(d, c, child_ctx, m);
            }
        }
    }
}

/// True for an LGG bound-variable placeholder atom (`_v0`, `_v1`, …) — the canonical names the fold
/// assigns to a motif's shared variables. Excluded from the anchor set (see [`Measure::anchors`]).
fn is_placeholder(s: &str) -> bool {
    s.strip_prefix("_v").is_some_and(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()))
}

/// Thresholds for [`extractable`]. Defaults are deliberately conservative — patternology is advisory,
/// and a false "you could extract a helper here" is worse than a missed one.
#[derive(Clone, Copy, Debug)]
pub struct ExtractCfg {
    /// Minimum fixed (shared-skeleton) node count — below this the "helper" is too small to bother.
    pub min_fixed: usize,
    /// Maximum expression-holes — a helper with more parameters than this is not a real abstraction.
    pub max_params: usize,
    /// Maximum tolerated statement-holes. Default 0: any leaky statement-divergence disqualifies.
    pub max_stmt_holes: usize,
    /// Maximum tolerated selector-holes (a varying method/callee/kwarg name). Default 0: a helper whose
    /// *target* differs across sites would need `getattr`/reflection — not a clean abstraction.
    pub max_selector_holes: usize,
    /// Minimum distinct anchor names (shared identifiers/literals, sans `_vN` placeholders). Default 2:
    /// below this the motif is pure syntactic structure (`? = ?; ? = ?`), a coincidence not an idiom.
    pub min_anchors: usize,
    /// Minimum fixed / (fixed + expr_holes) ratio — the skeleton must dominate the variation.
    pub min_ratio: f64,
}

impl Default for ExtractCfg {
    fn default() -> Self {
        Self {
            min_fixed: 6,
            max_params: 6,
            max_stmt_holes: 0,
            max_selector_holes: 0,
            min_anchors: 2,
            min_ratio: 0.5,
        }
    }
}

/// A folded motif judged extractable into a helper.
#[derive(Clone, Debug)]
pub struct HelperShape {
    /// Stable signature key (holes `?`, atoms verbatim) — the codometry `group-by` key.
    pub signature: String,
    /// Human-readable pseudo-source of the proposed helper body.
    pub body: String,
    /// Number of parameters the helper would take (= expression-holes).
    pub params: usize,
    /// Fixed (shared-skeleton) node count — a proxy for the helper's size / LOC it would absorb.
    pub fixed: usize,
}

/// Decide whether a folded LGG `term` collapses into a clean helper, and if so describe it. `None`
/// when the motif is not a real skeleton (root holed), leaks statement-holes, has a *selector*-hole (a
/// varying method/callee/kwarg name — `obj.?()`, not bindable without reflection), lacks shared
/// anchors (pure structure like `? = ?; ? = ?`), takes too many parameters, or is mostly variation
/// (low fixed:hole ratio) — i.e. the "vaguely similar / language idiom" noise we deliberately drop.
#[must_use]
fn extractable(d: &dyn Dialect, term: &Term, cfg: &ExtractCfg) -> Option<HelperShape> {
    // A real skeleton must have a node at the root — a holed/atom root is no pattern.
    if !matches!(term, Term::Node(..)) {
        return None;
    }
    let mut m = Measure::default();
    measure(d, term, Ctx::Expr, &mut m);
    if m.stmt_holes > cfg.max_stmt_holes
        || m.selector_holes > cfg.max_selector_holes
        || m.anchors.len() < cfg.min_anchors
        || m.fixed < cfg.min_fixed
        || m.expr_holes > cfg.max_params
    {
        return None;
    }
    let ratio = m.fixed as f64 / (m.fixed + m.expr_holes) as f64;
    if ratio < cfg.min_ratio {
        return None;
    }
    Some(HelperShape {
        signature: signature(term),
        body: render_body(d, term),
        params: m.expr_holes,
        fixed: m.fixed,
    })
}

/// Render a folded term as the proposed helper body via the dialect's renderer. A [`STMT_BLOCK_TAG`]
/// sub-block root has no function header, so its statement window is rendered joined with `; ` (each
/// statement through the dialect renderer); a whole-function term goes straight to [`Dialect::render`].
fn render_body(d: &dyn Dialect, term: &Term) -> String {
    let mut out = String::new();
    let mut budget = LGG_NODE_BUDGET;
    match term {
        Term::Node(tag, ch) if tag == STMT_BLOCK_TAG => match ch.first() {
            Some(Term::List(items)) => {
                if items.is_empty() {
                    out.push_str("pass");
                } else {
                    for (k, s) in items.iter().enumerate() {
                        if k > 0 {
                            out.push_str("; ");
                        }
                        if budget == 0 {
                            out.push('…');
                            break;
                        }
                        d.render(s, &mut out, &mut budget);
                    }
                }
            }
            Some(Term::Hole) | None => out.push('?'),
            Some(other) => d.render(other, &mut out, &mut budget),
        },
        _ => d.render(term, &mut out, &mut budget),
    }
    out
}

// ── Rendering the template as readable pseudo-source ─────────────────────────
//
// The folded LGG term is an `ast.dump` shape — accurate but unreadable as a one-liner
// (`FunctionDef('_fn', arguments([], [arg('_v0', BinOp(Name(?), BitOr, Constant(None)))]), …)`).
// `unparse` walks that term back into compact Python-ish pseudo-source with `?` at every hole
// (`def _fn(_v0: ? | None): if _v0 is None: return None; return _v0.?`). Field order is the CPython
// AST order; trailing default fields are omitted by the dumper, so positional `.get(k)` over a
// known prefix is safe. Anything without a rule falls back to a compact `Tag(child, …)` form.

/// Strip one layer of matching quotes so an identifier atom reads as `name`, not `'name'`. Barewords
/// (`None`, `_v0`, `42`) pass through.
fn unq(s: &str) -> &str {
    let b = s.as_bytes();
    if b.len() >= 2 && (b[0] == b'\'' || b[0] == b'"') && b[b.len() - 1] == b[0] {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Infix symbol for a binary / bool / compare operator node tag (`BitOr` → `|`, `Is` → `is`), or
/// the tag itself if unknown.
fn op_sym(tag: &str) -> &str {
    match tag {
        "Add" => "+", "Sub" => "-", "Mult" => "*", "Div" => "/", "FloorDiv" => "//", "Mod" => "%",
        "Pow" => "**", "LShift" => "<<", "RShift" => ">>", "BitOr" => "|", "BitXor" => "^",
        "BitAnd" => "&", "MatMult" => "@", "And" => "and", "Or" => "or", "Eq" => "==",
        "NotEq" => "!=", "Lt" => "<", "LtE" => "<=", "Gt" => ">", "GtE" => ">=", "Is" => "is",
        "IsNot" => "is not", "In" => "in", "NotIn" => "not in", other => other,
    }
}

fn unary_sym(tag: &str) -> &str {
    match tag {
        "Not" => "not ", "USub" => "-", "UAdd" => "+", "Invert" => "~", other => other,
    }
}

fn is_none(t: &Term) -> bool {
    matches!(t, Term::Atom(s) if s == "None")
}

/// The canon dumps argument-less AST nodes as barewords (`Pass`, `Continue`, the `MatchAs` wildcard)
/// — no `()` — so they parse as `Term::Atom`, not `Term::Node`, and never reach [`py_node`]. Map the
/// statement/pattern keywords to source form; quoted ids (`'Pass'`) and other atoms pass through.
fn stmt_bareword(s: &str) -> &str {
    match s {
        "Pass" => "pass",
        "Break" => "break",
        "Continue" => "continue",
        "MatchAs" => "_",
        other => other,
    }
}

/// Render an atom/hole in identifier position (`Name` id, `Attribute` attr, `arg` name): strip quotes
/// off a literal, hole → `?`, anything unexpected → `?`.
fn ident(t: Option<&Term>) -> String {
    match t {
        Some(Term::Atom(s)) => unq(s).to_owned(),
        _ => "?".to_owned(),
    }
}

/// The operator symbol for an operator child. `ast.dump` emits operators as barewords (`BitOr`,
/// `Is`) → `Term::Atom`; some dumps use `BitOr()` → `Term::Node`. Both map through [`op_sym`]; a hole
/// or anything else → `?`.
fn op_of(t: Option<&Term>) -> String {
    match t {
        Some(Term::Node(tag, _)) => op_sym(tag).to_owned(),
        Some(Term::Atom(s)) => op_sym(unq(s)).to_owned(),
        _ => "?".to_owned(),
    }
}

/// The unary-operator prefix for an operator child (bareword `Term::Atom` or `Term::Node`).
fn unary_of(t: Option<&Term>) -> &str {
    match t {
        Some(Term::Node(tag, _)) => unary_sym(tag),
        Some(Term::Atom(s)) => unary_sym(s),
        _ => "?",
    }
}

/// Append a comma-separated run of expressions / arg nodes drawn from a list child (or a lone hole),
/// continuing an existing argument list via `first`. Missing children contribute nothing.
fn seq_into(
    t: Option<&Term>,
    first: &mut bool,
    out: &mut String,
    b: &mut usize,
    each: impl Fn(&Term, &mut String, &mut usize),
) {
    match t {
        Some(Term::List(items)) => {
            for it in items {
                if !*first {
                    out.push_str(", ");
                }
                *first = false;
                if *b == 0 {
                    out.push('…');
                    return;
                }
                each(it, out, b);
            }
        }
        Some(Term::Hole) => {
            if !*first {
                out.push_str(", ");
            }
            *first = false;
            out.push('?');
        }
        _ => {}
    }
}

/// Render one `arg(name, annotation)` node as `name` or `name: annotation`.
fn arg_one(t: &Term, out: &mut String, b: &mut usize) {
    match t {
        Term::Node(tag, ch) if tag == "arg" => {
            out.push_str(&ident(ch.first()));
            if let Some(ann) = ch.get(1) {
                if !is_none(ann) {
                    out.push_str(": ");
                    py(ann, out, b);
                }
            }
        }
        Term::Hole => out.push('?'),
        other => py(other, out, b),
    }
}

/// True for an `arg(name, annotation)` node — the leaf of a parameter group.
fn is_arg_node(t: &Term) -> bool {
    matches!(t, Term::Node(tag, _) if tag == "arg")
}

/// Emit `, ` between parameters, tracking whether anything has been written yet.
fn param_sep(first: &mut bool, out: &mut String) {
    if !*first {
        out.push_str(", ");
    }
    *first = false;
}

/// Render an `arguments(posonly, args, vararg, kwonly, kw_defaults, kwarg, defaults)` node as a
/// flat parameter list. The canon omits any `None`-valued field (`vararg`/`kwarg`) and empty
/// trailing lists, so `arguments` is *variable-arity* — positional `.get(k)` is unsafe. Instead we
/// classify each child by shape: a list of `arg` nodes is a parameter group (posonly/args/kwonly,
/// all rendered as plain `name: anno`); a bare `arg` node is the `*vararg` / `**kwarg` (the `**`
/// kwarg is the one that follows a non-empty defaults list); a non-arg list is `(kw_)defaults`,
/// whose *shape* is rarely the pattern's point, so it is dropped.
fn params(t: Option<&Term>, out: &mut String, b: &mut usize) {
    match t {
        Some(Term::Node(tag, ch)) if tag == "arguments" => {
            let mut first = true;
            let mut seen_defaults = false;
            for child in ch {
                match child {
                    Term::List(items) if items.iter().any(is_arg_node) => {
                        for it in items {
                            param_sep(&mut first, out);
                            if *b == 0 {
                                out.push('…');
                                return;
                            }
                            arg_one(it, out, b);
                        }
                    }
                    // A non-empty non-arg list is `kw_defaults` / `defaults`; an empty list is the
                    // (dropped) `posonlyargs` slot or empty defaults — neither is rendered, but the
                    // non-empty one marks the boundary past which a bare `arg` is the `**kwarg`.
                    Term::List(items) => seen_defaults |= !items.is_empty(),
                    Term::Node(t2, _) if t2 == "arg" => {
                        param_sep(&mut first, out);
                        out.push_str(if seen_defaults { "**" } else { "*" });
                        arg_one(child, out, b);
                    }
                    _ => {}
                }
            }
        }
        Some(Term::Hole) => out.push('?'),
        Some(other) => py(other, out, b),
        None => {}
    }
}

/// Render a statement-body child (a list of statements) inline, joined with `; ` (`?` for a hole,
/// `pass` for an empty body).
fn stmt_body(t: Option<&Term>, out: &mut String, b: &mut usize) {
    match t {
        Some(Term::List(items)) => {
            if items.is_empty() {
                out.push_str("pass");
                return;
            }
            for (k, s) in items.iter().enumerate() {
                if k > 0 {
                    out.push_str("; ");
                }
                if *b == 0 {
                    out.push('…');
                    return;
                }
                py(s, out, b);
            }
        }
        Some(Term::Hole) | None => out.push('?'),
        Some(other) => py(other, out, b),
    }
}

/// Append the `for … in … [if …]` clauses of a comprehension's generator list (`ListComp`,
/// `GeneratorExp`, `DictComp`, …).
fn comprehensions(gens: Option<&Term>, out: &mut String, b: &mut usize) {
    let Some(Term::List(gs)) = gens else { return };
    for g in gs {
        if let Term::Node(tag, ch) = g {
            if tag == "comprehension" {
                let is_async = matches!(ch.get(3), Some(Term::Atom(s)) if s == "1");
                out.push_str(if is_async { " async for " } else { " for " });
                py_opt(ch.first(), out, b); // target
                out.push_str(" in ");
                py_opt(ch.get(1), out, b); // iter
                if let Some(Term::List(ifs)) = ch.get(2) {
                    for cond in ifs {
                        out.push_str(" if ");
                        py(cond, out, b);
                    }
                }
                continue;
            }
        }
        out.push_str(" for ");
        py(g, out, b);
    }
}

/// Render one piece of an f-string (`JoinedStr`): a literal `Constant` chunk verbatim, a
/// `FormattedValue` as `{expr}`.
fn fstring_part(t: &Term, out: &mut String, b: &mut usize) {
    match t {
        Term::Node(tag, ch) if tag == "Constant" => match ch.first() {
            Some(Term::Atom(s)) => out.push_str(unq(s)),
            _ => out.push('?'),
        },
        Term::Node(tag, ch) if tag == "FormattedValue" => {
            out.push('{');
            py_opt(ch.first(), out, b);
            out.push('}');
        }
        other => {
            out.push('{');
            py(other, out, b);
            out.push('}');
        }
    }
}

/// Render a term as one line of readable pseudo-source. Bounded by node budget `b`.
fn py(t: &Term, out: &mut String, b: &mut usize) {
    if *b == 0 {
        out.push('…');
        return;
    }
    *b -= 1;
    match t {
        Term::Hole => out.push('?'),
        Term::Atom(s) => out.push_str(stmt_bareword(s)),
        Term::List(items) => {
            out.push('[');
            let mut first = true;
            for it in items {
                if !first {
                    out.push_str(", ");
                }
                first = false;
                py(it, out, b);
            }
            out.push(']');
        }
        Term::Node(tag, ch) => py_node(tag, ch, out, b),
    }
}

#[allow(clippy::too_many_lines)]
fn py_node(tag: &str, ch: &[Term], out: &mut String, b: &mut usize) {
    match tag {
        "FunctionDef" | "AsyncFunctionDef" => {
            if tag.starts_with("Async") {
                out.push_str("async ");
            }
            out.push_str("def ");
            out.push_str(&ident(ch.first()));
            out.push('(');
            params(ch.get(1), out, b);
            out.push(')');
            if let Some(r) = ch.get(4) {
                if !matches!(r, Term::List(_)) && !is_none(r) {
                    out.push_str(" -> ");
                    py(r, out, b);
                }
            }
            out.push_str(": ");
            stmt_body(ch.get(2), out, b);
        }
        "Return" => {
            out.push_str("return");
            if let Some(v) = ch.first() {
                out.push(' ');
                py(v, out, b);
            }
        }
        "Pass" => out.push_str("pass"),
        "Break" => out.push_str("break"),
        "Continue" => out.push_str("continue"),
        "Assign" => {
            let mut first = true;
            seq_into(ch.first(), &mut first, out, b, py);
            out.push_str(" = ");
            py_opt(ch.get(1), out, b);
        }
        "AnnAssign" => {
            py_opt(ch.first(), out, b);
            out.push_str(": ");
            py_opt(ch.get(1), out, b);
            if let Some(v) = ch.get(2) {
                out.push_str(" = ");
                py(v, out, b);
            }
        }
        "AugAssign" => {
            py_opt(ch.first(), out, b);
            out.push(' ');
            out.push_str(&op_of(ch.get(1)));
            out.push_str("= ");
            py_opt(ch.get(2), out, b);
        }
        // Both are transparent single-child wrappers: a statement-expression, and (pre-3.9) a
        // subscript-slice wrapper.
        // `MatchValue`/`MatchSingleton` wrap a single literal/expr, like the `Expr`/`Index` wrappers.
        "Expr" | "Index" | "MatchValue" | "MatchSingleton" => py_opt(ch.first(), out, b),
        "If" => {
            out.push_str("if ");
            py_opt(ch.first(), out, b);
            out.push_str(": ");
            stmt_body(ch.get(1), out, b);
            if let Some(Term::List(orelse)) = ch.get(2) {
                if !orelse.is_empty() {
                    out.push_str(" else: ");
                    stmt_body(ch.get(2), out, b);
                }
            }
        }
        "For" | "AsyncFor" => {
            if tag.starts_with("Async") {
                out.push_str("async ");
            }
            out.push_str("for ");
            py_opt(ch.first(), out, b);
            out.push_str(" in ");
            py_opt(ch.get(1), out, b);
            out.push_str(": ");
            stmt_body(ch.get(2), out, b);
        }
        "While" => {
            out.push_str("while ");
            py_opt(ch.first(), out, b);
            out.push_str(": ");
            stmt_body(ch.get(1), out, b);
        }
        "With" | "AsyncWith" => {
            if tag.starts_with("Async") {
                out.push_str("async ");
            }
            out.push_str("with ");
            let mut first = true;
            seq_into(ch.first(), &mut first, out, b, py);
            out.push_str(": ");
            stmt_body(ch.get(1), out, b);
        }
        "withitem" => {
            py_opt(ch.first(), out, b);
            if let Some(v) = ch.get(1) {
                if !is_none(v) {
                    out.push_str(" as ");
                    py(v, out, b);
                }
            }
        }
        "Raise" => {
            out.push_str("raise");
            if let Some(e) = ch.first() {
                if !is_none(e) {
                    out.push(' ');
                    py(e, out, b);
                }
            }
        }
        "Try" => {
            out.push_str("try: ");
            stmt_body(ch.first(), out, b);
            if let Some(Term::List(handlers)) = ch.get(1) {
                for h in handlers {
                    out.push_str("; ");
                    py(h, out, b);
                }
            }
        }
        "ExceptHandler" => {
            out.push_str("except");
            if let Some(ty) = ch.first() {
                if !is_none(ty) {
                    out.push(' ');
                    py(ty, out, b);
                }
            }
            if let Some(name) = ch.get(1) {
                if !is_none(name) {
                    out.push_str(" as ");
                    out.push_str(&ident(Some(name)));
                }
            }
            out.push_str(": ");
            stmt_body(ch.get(2), out, b);
        }
        "Call" => {
            py_opt(ch.first(), out, b);
            out.push('(');
            let mut first = true;
            seq_into(ch.get(1), &mut first, out, b, py); // positional args
            seq_into(ch.get(2), &mut first, out, b, py); // keyword args
            out.push(')');
        }
        "keyword" => {
            if let Some(a) = ch.first() {
                if !is_none(a) {
                    out.push_str(&ident(Some(a)));
                    out.push('=');
                    py_opt(ch.get(1), out, b);
                    return;
                }
            }
            out.push_str("**");
            py_opt(ch.get(1), out, b);
        }
        "Attribute" => {
            py_opt(ch.first(), out, b);
            out.push('.');
            out.push_str(&ident(ch.get(1)));
        }
        "Subscript" => {
            py_opt(ch.first(), out, b);
            out.push('[');
            py_opt(ch.get(1), out, b);
            out.push(']');
        }
        "Name" => out.push_str(&ident(ch.first())),
        "Constant" => match ch.first() {
            Some(Term::Atom(s)) => out.push_str(s),
            Some(other) => py(other, out, b),
            None => out.push('?'),
        },
        "Compare" => {
            py_opt(ch.first(), out, b);
            if let (Some(Term::List(ops)), Some(Term::List(cmps))) = (ch.get(1), ch.get(2)) {
                for (op, c) in ops.iter().zip(cmps) {
                    out.push(' ');
                    out.push_str(op_of(Some(op)).as_str());
                    out.push(' ');
                    py(c, out, b);
                }
            }
        }
        "BinOp" => {
            py_opt(ch.first(), out, b);
            out.push(' ');
            out.push_str(&op_of(ch.get(1)));
            out.push(' ');
            py_opt(ch.get(2), out, b);
        }
        "BoolOp" => {
            let sep = format!(" {} ", op_of(ch.first()));
            if let Some(Term::List(vals)) = ch.get(1) {
                for (k, v) in vals.iter().enumerate() {
                    if k > 0 {
                        out.push_str(&sep);
                    }
                    py(v, out, b);
                }
            }
        }
        "UnaryOp" => {
            out.push_str(unary_of(ch.first()));
            py_opt(ch.get(1), out, b);
        }
        // List/Tuple/Set store their elements as a single list child (`List([a, b])`); the trailing
        // `ctx` is dropped. Iterate that child — wrapping `ch` again would double-bracket (`[]`→`[[]]`).
        // `MatchSequence` is the pattern-position `[...]` and renders identically.
        "List" | "MatchSequence" => {
            out.push('[');
            let mut first = true;
            seq_into(ch.first(), &mut first, out, b, py);
            out.push(']');
        }
        "Tuple" => {
            out.push('(');
            let mut first = true;
            seq_into(ch.first(), &mut first, out, b, py);
            out.push(')');
        }
        "Set" => {
            out.push('{');
            let mut first = true;
            seq_into(ch.first(), &mut first, out, b, py);
            out.push('}');
        }
        "Dict" => {
            out.push('{');
            if let (Some(Term::List(keys)), Some(Term::List(vals))) = (ch.first(), ch.get(1)) {
                for (k, (key, val)) in keys.iter().zip(vals).enumerate() {
                    if k > 0 {
                        out.push_str(", ");
                    }
                    py(key, out, b);
                    out.push_str(": ");
                    py(val, out, b);
                }
            }
            out.push('}');
        }
        "Await" => {
            out.push_str("await ");
            py_opt(ch.first(), out, b);
        }
        "Yield" => {
            out.push_str("yield");
            if let Some(v) = ch.first() {
                out.push(' ');
                py(v, out, b);
            }
        }
        "YieldFrom" => {
            out.push_str("yield from ");
            py_opt(ch.first(), out, b);
        }
        "Starred" => {
            out.push('*');
            py_opt(ch.first(), out, b);
        }
        "Lambda" => {
            out.push_str("lambda ");
            params(ch.first(), out, b);
            out.push_str(": ");
            py_opt(ch.get(1), out, b);
        }
        "IfExp" => {
            py_opt(ch.get(1), out, b);
            out.push_str(" if ");
            py_opt(ch.first(), out, b);
            out.push_str(" else ");
            py_opt(ch.get(2), out, b);
        }
        "ListComp" => {
            out.push('[');
            py_opt(ch.first(), out, b);
            comprehensions(ch.get(1), out, b);
            out.push(']');
        }
        "SetComp" => {
            out.push('{');
            py_opt(ch.first(), out, b);
            comprehensions(ch.get(1), out, b);
            out.push('}');
        }
        "GeneratorExp" => {
            out.push('(');
            py_opt(ch.first(), out, b);
            comprehensions(ch.get(1), out, b);
            out.push(')');
        }
        "DictComp" => {
            out.push('{');
            py_opt(ch.first(), out, b);
            out.push_str(": ");
            py_opt(ch.get(1), out, b);
            comprehensions(ch.get(2), out, b);
            out.push('}');
        }
        "JoinedStr" => {
            out.push_str("f'");
            if let Some(Term::List(parts)) = ch.first() {
                for p in parts {
                    fstring_part(p, out, b);
                }
            }
            out.push('\'');
        }
        "FormattedValue" => {
            out.push('{');
            py_opt(ch.first(), out, b);
            out.push('}');
        }
        "Delete" => {
            out.push_str("del ");
            let mut first = true;
            seq_into(ch.first(), &mut first, out, b, py);
        }
        "Assert" => {
            out.push_str("assert ");
            py_opt(ch.first(), out, b);
            if let Some(m) = ch.get(1) {
                if !is_none(m) {
                    out.push_str(", ");
                    py(m, out, b);
                }
            }
        }
        // `Subscript` index slice: `lower:upper:step`, each omitted when None/absent. The canon drops
        // None fields, so a one-child `Slice(x)` is read as `x:` (lower-only — the common `seq[n:]`).
        "Slice" => {
            if let Some(l) = ch.first() {
                if !is_none(l) {
                    py(l, out, b);
                }
            }
            out.push(':');
            if let Some(u) = ch.get(1) {
                if !is_none(u) {
                    py(u, out, b);
                }
            }
            if let Some(s) = ch.get(2) {
                if !is_none(s) {
                    out.push(':');
                    py(s, out, b);
                }
            }
        }
        "Import" => {
            out.push_str("import ");
            let mut first = true;
            seq_into(ch.first(), &mut first, out, b, py);
        }
        "ImportFrom" => {
            out.push_str("from ");
            if let Some(Term::Atom(lvl)) = ch.get(2) {
                if let Ok(n) = unq(lvl).parse::<usize>() {
                    for _ in 0..n {
                        out.push('.');
                    }
                }
            }
            if let Some(m) = ch.first() {
                if !is_none(m) {
                    out.push_str(&ident(Some(m)));
                }
            }
            out.push_str(" import ");
            let mut first = true;
            seq_into(ch.get(1), &mut first, out, b, py);
        }
        "alias" => {
            out.push_str(&ident(ch.first()));
            if let Some(a) = ch.get(1) {
                if !is_none(a) {
                    out.push_str(" as ");
                    out.push_str(&ident(Some(a)));
                }
            }
        }
        "Match" => {
            out.push_str("match ");
            py_opt(ch.first(), out, b);
            out.push_str(": ");
            if let Some(Term::List(cases)) = ch.get(1) {
                for (k, c) in cases.iter().enumerate() {
                    if k > 0 {
                        out.push_str("; ");
                    }
                    if *b == 0 {
                        out.push('…');
                        break;
                    }
                    py(c, out, b);
                }
            }
        }
        // `match_case(pattern, guard?, body)`: the `None` guard is dropped, so the body is always the
        // last child and a guard exists only when three children remain.
        "match_case" => {
            out.push_str("case ");
            py_opt(ch.first(), out, b);
            if ch.len() >= 3 {
                out.push_str(" if ");
                py_opt(ch.get(1), out, b);
            }
            out.push_str(": ");
            stmt_body(ch.last(), out, b);
        }
        "MatchClass" => {
            py_opt(ch.first(), out, b);
            out.push('(');
            let mut first = true;
            seq_into(ch.get(1), &mut first, out, b, py);
            out.push(')');
        }
        "MatchStar" => {
            out.push('*');
            match ch.first() {
                Some(Term::Atom(s)) => out.push_str(unq(s)),
                _ => out.push('_'),
            }
        }
        // `MatchAs(pattern?, name?)`: bare → wildcard `_`; lone name → capture; `pattern as name`.
        "MatchAs" => match (ch.first(), ch.get(1)) {
            (None, _) => out.push('_'),
            (Some(Term::Atom(s)), None) => out.push_str(unq(s)),
            (Some(p), Some(Term::Atom(s))) => {
                py(p, out, b);
                out.push_str(" as ");
                out.push_str(unq(s));
            }
            (Some(p), _) => py(p, out, b),
        },
        "MatchOr" => {
            if let Some(Term::List(pats)) = ch.first() {
                for (k, p) in pats.iter().enumerate() {
                    if k > 0 {
                        out.push_str(" | ");
                    }
                    py(p, out, b);
                }
            }
        }
        // Unknown / unhandled node: compact `Tag(child, …)` fallback.
        _ => {
            out.push_str(tag);
            if !ch.is_empty() {
                out.push('(');
                let mut first = true;
                seq_into(Some(&Term::List(ch.to_vec())), &mut first, out, b, py);
                out.push(')');
            }
        }
    }
}

/// `py` on an optional child — `?` if the child is missing.
fn py_opt(t: Option<&Term>, out: &mut String, b: &mut usize) {
    match t {
        Some(t) => py(t, out, b),
        None => out.push('?'),
    }
}

/// The family's anti-unification template, rendered as readable one-line pseudo-source: fold
/// [`lgg`] over the members, then render it with [`py`] (holes `?` at the variation points),
/// bounded by [`LGG_NODE_BUDGET`]. Production renders the same string inside [`extractable`]
/// (`HelperShape::body`); this stays as the direct test handle for the `py` renderer.
#[cfg(test)]
fn anti_unify(members: &[usize], canons: &[String]) -> String {
    let term = fold_lgg(members, canons);
    let mut out = String::new();
    let mut budget = LGG_NODE_BUDGET;
    py(&term, &mut out, &mut budget);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_type_seq_picks_constructors_skips_values_and_ctx() {
        // `Name('_v0', Load())` → Name kept, Load dropped (ctx noise), '_v0' is not a constructor.
        let canon = "Assign(targets=[Name('_v0', Load())], value=Call(func=Attribute(value=Name('session', Load()), attr='add'), args=[Name('_v0', Load())]))";
        let seq = node_type_seq(canon);
        assert_eq!(seq, vec!["Assign", "Name", "Call", "Attribute", "Name", "Name"]);
    }

    #[test]
    fn node_type_seq_skips_constructor_lookalikes_inside_strings() {
        // A string literal value that contains `Call(` must not inject a phantom node.
        let canon = "Expr(value=Constant('Call(x) and Return(y)'))";
        assert_eq!(node_type_seq(canon), vec!["Expr", "Constant"]);
    }

    #[test]
    fn gram_packing_is_injective_and_separates_arities() {
        // bi- vs tri-gram codes never collide (distinct arity tag), and distinct id tuples map to
        // distinct codes — the property that makes packed-u64 grams a lossless drop-in for the old
        // `A>B` strings (so the clustering is byte-identical).
        assert_ne!(pack_bigram(1, 2), pack_trigram(1, 2, 0));
        assert_ne!(pack_bigram(1, 2), pack_bigram(2, 1));
        assert_ne!(pack_trigram(1, 2, 3), pack_trigram(3, 2, 1));
        // Node-type ids for the tags in `A(B(C()))` would be e.g. A=0,B=1,C=2 → bigram A>B and B>C
        // and trigram A>B>C are three distinct codes.
        let (a, b, c) = (0u32, 1u32, 2u32);
        let codes = [pack_bigram(a, b), pack_bigram(b, c), pack_trigram(a, b, c)];
        assert_eq!(codes.iter().collect::<FxHashSet<_>>().len(), 3);
    }

    #[test]
    fn identical_skeleton_divergent_text_clusters() {
        // Two "create" shapes: build → add → return, with totally different names/literals.
        let a = "FunctionDef(name='_fn', body=[Assign(targets=[Name('_v0')], value=Call(func=Name('AuthRefreshToken'), args=[Constant('x')])), Expr(value=Call(func=Attribute(value=Name('session'), attr='add'))), Return(value=Name('_v0'))])".to_owned();
        let b = "FunctionDef(name='_fn', body=[Assign(targets=[Name('_v0')], value=Call(func=Name('Clientspace'), args=[Constant('y'), Constant('z')])), Expr(value=Call(func=Attribute(value=Name('db'), attr='add'))), Return(value=Name('_v0'))])".to_owned();
        let fams = pattern_families(&[a, b], 0.7, Concurrency::Cpu);
        assert_eq!(fams.len(), 1, "the two create-shapes should form one family");
        assert_eq!(fams[0].members, vec![0, 1]);
        assert!(matches!(fams[0].term, Term::Node(..)), "family folds to a real skeleton term");
    }

    #[test]
    fn lgg_fold_keeps_skeleton_holes_divergence() {
        // The two create-shapes from above: same build→add→return skeleton, divergent model names,
        // literals and arg lists. The folded LGG term must keep the shared skeleton and hole the
        // divergences (asserted on the tree form, independent of the pseudo-source renderer).
        let a = "FunctionDef(name='_fn', body=[Assign(targets=[Name('_v0')], value=Call(func=Name('AuthRefreshToken'), args=[Constant('x')])), Expr(value=Call(func=Attribute(value=Name('session'), attr='add'))), Return(value=Name('_v0'))])";
        let b = "FunctionDef(name='_fn', body=[Assign(targets=[Name('_v0')], value=Call(func=Name('Clientspace'), args=[Constant('y'), Constant('z')])), Expr(value=Call(func=Attribute(value=Name('db'), attr='add'))), Return(value=Name('_v0'))])";
        let mut tmpl = String::new();
        let mut budget = LGG_NODE_BUDGET;
        render_term(&fold_lgg(&[0, 1], &[a.to_owned(), b.to_owned()]), &mut tmpl, &mut budget);
        // Shared skeleton survives…
        assert!(tmpl.starts_with("FunctionDef("), "got: {tmpl}");
        assert!(tmpl.contains("Return(Name('_v0'))"), "return shape kept: {tmpl}");
        assert!(tmpl.contains("Attribute(Name(?), 'add')"), "session.add kept, receiver holed: {tmpl}");
        // Constructor call: func name holed; arg lists differ in length ([x] vs [y, z]) so the first
        // arg aligns (holed value) and the surplus collapses to one trailing hole.
        assert!(tmpl.contains("Call(Name(?), [Constant(?), ?])"), "call + aligned args: {tmpl}");
        // …and the divergent model names are gone (holed, not in the template).
        assert!(!tmpl.contains("AuthRefreshToken"), "model name must be holed: {tmpl}");
        assert!(!tmpl.contains("Clientspace"), "model name must be holed: {tmpl}");
    }

    #[test]
    fn unparse_renders_readable_pseudocode_with_holes() {
        // The nullable-getter guard clause (a real-world win), two members differing only in the
        // annotation type and the returned attribute name → both holed, skeleton verbatim. The note
        // must read as one line of pseudo-source, not an ast.dump.
        let a = "FunctionDef('_fn', arguments([], [arg('_v0', BinOp(Name('T', Load), BitOr, Constant(None)))]), [If(Compare(Name('_v0', Load), [Is], [Constant(None)]), [Return(Constant(None))], []), Return(Attribute(Name('_v0', Load), 'cfg', Load))], [], None)";
        let b = "FunctionDef('_fn', arguments([], [arg('_v0', BinOp(Name('U', Load), BitOr, Constant(None)))]), [If(Compare(Name('_v0', Load), [Is], [Constant(None)]), [Return(Constant(None))], []), Return(Attribute(Name('_v0', Load), 'env', Load))], [], None)";
        let tmpl = anti_unify(&[0, 1], &[a.to_owned(), b.to_owned()]);
        assert_eq!(
            tmpl,
            "def _fn(_v0: ? | None): if _v0 is None: return None; return _v0.?",
            "got: {tmpl}"
        );
    }

    #[test]
    fn unparse_renders_comprehensions_and_fstrings() {
        // A list-comprehension over an await, plus an f-string — the noisiest residual nodes. Two
        // members differing only in the mapped callable and the literal prefix → both holed.
        let a = "FunctionDef('_fn', arguments([], []), [Return(ListComp(Call(Name('A', Load), [Name('_v0', Load)], []), [comprehension(Name('_v0', Store), Name('rows', Load), [], 0)]))], [], None)";
        let b = "FunctionDef('_fn', arguments([], []), [Return(ListComp(Call(Name('B', Load), [Name('_v0', Load)], []), [comprehension(Name('_v0', Store), Name('rows', Load), [], 0)]))], [], None)";
        let tmpl = anti_unify(&[0, 1], &[a.to_owned(), b.to_owned()]);
        assert_eq!(
            tmpl,
            "def _fn(): return [?(_v0) for _v0 in rows]",
            "got: {tmpl}"
        );

        let c = "FunctionDef('_fn', arguments([], []), [Expr(JoinedStr([Constant('id='), FormattedValue(Name('_v0', Load), -1)]))], [], None)";
        let tmpl2 = anti_unify(&[0], &[c.to_owned()]);
        assert_eq!(tmpl2, "def _fn(): f'id={_v0}'", "got: {tmpl2}");
    }

    #[test]
    fn unparse_renders_create_shape_as_pseudocode() {
        // The build→add→return create-shape, rendered: divergent model + args holed, the rest reads
        // as Python. `Load` context nodes are stripped, so `Name('_v0', Load)` shows as `_v0`.
        let a = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('AuthRefreshToken', Load), [Constant('x')], [])), Expr(Call(Attribute(Name('session', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)";
        let b = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('Clientspace', Load), [Constant('y'), Constant('z')], [])), Expr(Call(Attribute(Name('db', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)";
        let tmpl = anti_unify(&[0, 1], &[a.to_owned(), b.to_owned()]);
        // Arg lists differ ([x] vs [y, z]): first arg aligns (holed), surplus → trailing hole `?, ?`.
        assert_eq!(
            tmpl,
            "def _fn(): _v0 = ?(?, ?); ?.add(_v0); return _v0",
            "got: {tmpl}"
        );
    }

    #[test]
    fn unparse_renders_variadic_arguments_without_leaking_arg_nodes() {
        // The canon omits None vararg/kwarg and trailing empty lists, so `arguments` is variable-arity.
        // A `*vararg` is a bare `arg` node; kwonly args are a list that slid into the vararg slot; a
        // `**kwarg` is a bare `arg` node after a defaults list. None of these may leak as `[arg(...)]`.
        let vararg = "FunctionDef('_fn', arguments([], [arg('_v0', Name('int'))], arg('_v1', Name('str'))), [Pass], [], None)";
        assert_eq!(anti_unify(&[0], &[vararg.to_owned()]), "def _fn(_v0: int, *_v1: str): pass");

        let kwonly = "FunctionDef('_fn', arguments([], [arg('_v0', Name('Path'))], [arg('_v2', Name('str'))], [None]), [Pass], [], None)";
        assert_eq!(anti_unify(&[0], &[kwonly.to_owned()]), "def _fn(_v0: Path, _v2: str): pass");

        let kwarg = "FunctionDef('_fn', arguments([], [arg('_v0', Name('int'))], [arg('_v5', Name('str'))], [None], arg('_v7', Name('Unpack')), [None]), [Pass], [], None)";
        assert_eq!(anti_unify(&[0], &[kwarg.to_owned()]), "def _fn(_v0: int, _v5: str, **_v7: Unpack): pass");
    }

    #[test]
    fn unparse_renders_list_tuple_without_double_brackets() {
        // List/Tuple/Set store elements in a single list child; rendering must not re-wrap it (the
        // `[]`→`[[]]` / `(a, b)`→`([a, b])` bug). Empty list → `[]`, tuple-target → `(a, b)`.
        let f = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0')], List([])), Assign([Tuple([Name('a'), Name('b')])], Name('_v0')), Return(Tuple([Name('a'), Name('b')]))], [], None)";
        assert_eq!(anti_unify(&[0], &[f.to_owned()]), "def _fn(): _v0 = []; (a, b) = _v0; return (a, b)");
    }

    #[test]
    fn unparse_renders_match_import_and_slice() {
        // Three constructs the renderer used to dump raw: `match`/`case` (incl. the bareword `MatchAs`
        // wildcard), `from … import …`, and a subscript `Slice`.
        let m = "FunctionDef('_fn', arguments([], []), [Match(Name('_v0'), [match_case(MatchClass(Name('Ok'), [MatchSequence([MatchAs('x'), MatchAs('y')])]), [Pass]), match_case(MatchAs, [Pass])])], [], None)";
        assert_eq!(
            anti_unify(&[0], &[m.to_owned()]),
            "def _fn(): match _v0: case Ok([x, y]): pass; case _: pass"
        );

        let imp = "FunctionDef('_fn', arguments([], []), [ImportFrom('pkg.mod', [alias('f')], 0), Assign([Name('_v0')], Subscript(Name('_v1'), Slice(Name('_v2'))))], [], None)";
        assert_eq!(
            anti_unify(&[0], &[imp.to_owned()]),
            "def _fn(): from pkg.mod import f; _v0 = _v1[_v2:]"
        );
    }

    #[test]
    fn lgg_renders_hole_for_disagreeing_atoms_and_keeps_agreeing() {
        // Same node, one shared child atom, one divergent → keep tag, keep shared, hole the rest.
        let a = parse_term("Call(Name('foo'), 'shared')");
        let b = parse_term("Call(Name('bar'), 'shared')");
        let g = lgg(&a, &b);
        let mut out = String::new();
        let mut budget = LGG_NODE_BUDGET;
        render_term(&g, &mut out, &mut budget);
        assert_eq!(out, "Call(Name(?), 'shared')");
    }

    #[test]
    fn lgg_aligns_lists_of_differing_length() {
        // A deleted middle element: [A, B, C] vs [A, C] → align A and C, hole the gap → [A, ?, C].
        // (Previously the whole subtree collapsed to a hole — the dominant cause of degenerate `?`.)
        let a = parse_term("G([A(), B(), C()])");
        let b = parse_term("G([A(), C()])");
        let mut out = String::new();
        let mut budget = LGG_NODE_BUDGET;
        render_term(&lgg(&a, &b), &mut out, &mut budget);
        assert_eq!(out, "G([A, ?, C])");
    }

    #[test]
    fn lgg_unifies_async_and_sync_def() {
        // A sync/async mirror (botocore↔aiobotocore type-5): same body, one async one not. They must
        // unify into one template (rendered sync, asyncness dropped) rather than collapsing to `?`.
        let a = "AsyncFunctionDef('_fn', arguments([], []), [Return(Name('_v0', Load))], [], None)";
        let b = "FunctionDef('_fn', arguments([], []), [Return(Name('_v0', Load))], [], None)";
        let tmpl = anti_unify(&[0, 1], &[a.to_owned(), b.to_owned()]);
        assert_eq!(tmpl, "def _fn(): return _v0", "got: {tmpl}");
    }

    #[test]
    fn lgg_prefix_aligns_same_tag_nodes_of_differing_arity() {
        // `ast.dump` omits only trailing default fields, so a same-tag node with fewer children is a
        // prefix — anti-unify the prefix, drop the surplus, instead of holing the whole node.
        let a = parse_term("F('n', 'a', Ret())");
        let b = parse_term("F('n', 'a')");
        let mut out = String::new();
        let mut budget = LGG_NODE_BUDGET;
        render_term(&lgg(&a, &b), &mut out, &mut budget);
        assert_eq!(out, "F('n', 'a')");
    }

    #[test]
    fn parse_function_body_matches_full_parse() {
        // The targeted body parser must reproduce `function_body(&parse_term(c))` exactly while
        // skipping a rich `arguments` node (annotations, kwonly default) and the trailing returns slot.
        let canon = "FunctionDef('_fn', arguments([], [arg('_v0', Name('Path')), arg('_v1', Subscript(Name('list'), Name('int')))], [arg('_v2', Name('str'))], [None]), [Assign([Name('_v3')], Call(Name('open'), [Name('_v0')])), If(Compare(Name('_v3'), [Is], [Constant(None)]), [Return(Constant(None))], []), Return(Name('_v3'))], [], Name('bool'))";
        assert_eq!(py_parse_function_body(canon), function_body(&parse_term(canon)));
        // Async base tag, a quoted `]` that must not close the body list early, and the no-returns form.
        let asy = "AsyncFunctionDef('_fn', arguments([], [arg('_v0', Name('S'))]), [Expr(Constant('a]b')), Return(Name('_v0'))], [], None)";
        assert_eq!(py_parse_function_body(asy), function_body(&parse_term(asy)));
        // Non-FunctionDef → empty, just like `function_body`.
        assert!(py_parse_function_body("Module([Pass])").is_empty());
    }

    #[test]
    fn parse_term_handles_keywords_lists_and_quoted_parens() {
        // Keyword names dropped, lists nested, and a `)` inside a quoted literal must not close a node.
        let t = parse_term("Expr(value=Constant(')(]['))");
        assert_eq!(
            t,
            Term::Node(
                "Expr".to_owned(),
                vec![Term::Node("Constant".to_owned(), vec![Term::Atom("')(]['".to_owned())])]
            )
        );
    }

    /// Two triangles sharing a vertex: {0,1,2} and {2,3,4}. Single-linkage would chain all five
    /// into one component; the greedy clique-cover must split them into two tight families (this is
    /// the whole reason patternology covers by cliques instead of connected components).
    #[test]
    fn clique_cover_splits_chained_triangles() {
        let edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)];
        let n = 5;
        let mut adj: Vec<FxHashSet<usize>> = (0..n).map(|_| FxHashSet::default()).collect();
        for &(a, b) in &edges {
            adj[a].insert(b);
            adj[b].insert(a);
        }
        let comp: Vec<usize> = (0..n).collect();
        let mut fams = clique_families(&comp, &adj);
        fams.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        // Greedy cover: take {0,1,2} (size 3), then {2,3,4} minus the claimed 2 → {3,4}.
        assert_eq!(fams, vec![vec![0, 1, 2], vec![3, 4]]);
    }

    /// A complete graph K4 is a single maximal clique — one family, not six pair-families.
    #[test]
    fn clique_cover_keeps_complete_graph_whole() {
        let n = 4;
        let mut adj: Vec<FxHashSet<usize>> = (0..n).map(|_| FxHashSet::default()).collect();
        for (a, set) in adj.iter_mut().enumerate() {
            for b in 0..n {
                if a != b {
                    set.insert(b);
                }
            }
        }
        let fams = clique_families(&(0..n).collect::<Vec<_>>(), &adj);
        assert_eq!(fams, vec![vec![0, 1, 2, 3]]);
    }

    // ── Extractability ───────────────────────────────────────────────────────

    #[test]
    fn extractable_accepts_clean_create_shape() {
        // build→add→return with the model name and the constant arg holed (expression positions) →
        // a clean helper: shared skeleton dominates, holes are all bindable params, no stmt-holes.
        let a = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('AuthRefreshToken', Load), [Constant('x')], [])), Expr(Call(Attribute(Name('session', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)";
        let b = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('Clientspace', Load), [Constant('y')], [])), Expr(Call(Attribute(Name('session', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)";
        let term = fold_lgg(&[0, 1], &[a.to_owned(), b.to_owned()]);
        let shape = extractable(&PyDialect, &term, &ExtractCfg::default()).expect("create-shape is extractable");
        assert!(shape.params >= 1, "the holed model/constant become params: {shape:?}");
        assert!(shape.fixed >= 6, "skeleton is substantial: {shape:?}");
        // `session` agrees across members → stays verbatim; only the model name and the constant hole.
        assert_eq!(shape.body, "def _fn(): _v0 = ?(?); session.add(_v0); return _v0", "body: {}", shape.body);
    }

    #[test]
    fn extractable_rejects_statement_divergence() {
        // Same head and tail, but the MIDDLE statement differs in kind (Assign vs Raise) → LCS holes
        // a whole statement in the body list → a leaky statement-hole → not a clean helper, dropped.
        let a = "FunctionDef('_fn', arguments([], []), [Expr(Call(Name('a', Load), [], [])), Assign([Name('_v0', Store)], Constant(1)), Return(Name('_v0', Load))], [], None)";
        let b = "FunctionDef('_fn', arguments([], []), [Expr(Call(Name('a', Load), [], [])), Raise(Call(Name('E', Load), [], [])), Return(Name('_v0', Load))], [], None)";
        let term = fold_lgg(&[0, 1], &[a.to_owned(), b.to_owned()]);
        assert!(extractable(&PyDialect, &term, &ExtractCfg::default()).is_none(), "statement-hole must disqualify");
    }

    #[test]
    fn signature_is_stable_across_cohorts() {
        // The SAME idiom mined from two disjoint cohorts (different members, divergence only in a
        // holed leaf) must yield the SAME signature key — the precondition for cross-package codometry
        // group-by. Cohort 1 holes a Constant; cohort 2 holes the same Constant from other members.
        let mk = |lit: &str| format!("FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('M', Load), [Constant('{lit}')], [])), Return(Name('_v0', Load))], [], None)");
        let c1 = [mk("a"), mk("b")];
        let c2 = [mk("p"), mk("q")];
        let s1 = signature(&fold_lgg(&[0, 1], &c1));
        let s2 = signature(&fold_lgg(&[0, 1], &c2));
        assert_eq!(s1, s2, "same idiom, different cohorts → same signature");
        assert!(s1.contains('?'), "the holed constant shows as ?: {s1}");
        assert!(s1.starts_with("FunctionDef("), "signature is the canonical tree form: {s1}");
    }

    #[test]
    fn measure_splits_holes_by_context() {
        // A function with one expression-hole (a holed Call arg) and one statement-hole (a holed body
        // element) — measure must bucket them apart. `parse_term` has no literal hole syntax, so the
        // term (as a fold would produce) is built directly.
        let term = Term::Node(
            "FunctionDef".to_owned(),
            vec![
                Term::Atom("'_fn'".to_owned()),
                Term::Node("arguments".to_owned(), vec![Term::List(vec![]), Term::List(vec![])]),
                Term::List(vec![
                    Term::Node("Expr".to_owned(), vec![Term::Node(
                        "Call".to_owned(),
                        vec![Term::Node("Name".to_owned(), vec![Term::Atom("'f'".to_owned())]), Term::List(vec![Term::Hole]), Term::List(vec![])],
                    )]),
                    Term::Hole, // a holed statement (body-list element)
                ]),
                Term::List(vec![]),
                Term::Atom("None".to_owned()),
            ],
        );
        let mut m = Measure::default();
        measure(&PyDialect, &term, Ctx::Expr, &mut m);
        assert_eq!(m.expr_holes, 1, "the Call arg hole is an expression hole");
        assert_eq!(m.stmt_holes, 1, "the body-list hole is a statement hole");
    }

    #[test]
    fn selector_hole_disqualifies_varying_method_name() {
        // `_v0.?(?); await _v0.flush(); return ?` — the method *name* (the `attr` slot of the leading
        // `Attribute`) is a hole. That is a selector, not a value: you cannot fold `session.add(x)` and
        // `session.merge(x)` into one helper without `getattr`. The fixed-method twin must still pass.
        let stmts = |attr: Term| {
            Term::Node(
                STMT_BLOCK_TAG.to_owned(),
                vec![Term::List(vec![
                    Term::Node("Expr".to_owned(), vec![Term::Node(
                        "Call".to_owned(),
                        vec![
                            Term::Node("Attribute".to_owned(), vec![
                                Term::Node("Name".to_owned(), vec![Term::Atom("'_v0'".to_owned())]),
                                attr,
                            ]),
                            Term::List(vec![Term::Hole]),
                            Term::List(vec![]),
                        ],
                    )]),
                    Term::Node("Return".to_owned(), vec![Term::Node(
                        "Attribute".to_owned(),
                        vec![Term::Node("Name".to_owned(), vec![Term::Atom("'_v0'".to_owned())]), Term::Atom("'committed'".to_owned())],
                    )]),
                ])],
            )
        };
        let mut holed = Measure::default();
        measure(&PyDialect, &stmts(Term::Hole), Ctx::Stmt, &mut holed);
        assert_eq!(holed.selector_holes, 1, "the holed method name is a selector hole");
        assert!(
            extractable(&PyDialect, &stmts(Term::Hole), &ExtractCfg::default()).is_none(),
            "a varying method name must not extract into a helper"
        );

        let mut fixed = Measure::default();
        measure(&PyDialect, &stmts(Term::Atom("'save'".to_owned())), Ctx::Stmt, &mut fixed);
        assert_eq!(fixed.selector_holes, 0, "a fixed method name is an anchor, not a selector hole");
    }

    #[test]
    fn all_value_skeleton_is_rejected_for_lack_of_anchors() {
        // `? = ?; ? = ?` — two assignments, every leaf a hole. Pure syntactic structure, no shared
        // identifier or literal: a coincidence, not an extractable idiom. Anchors == 0 → not extractable.
        let assign = || Term::Node("Assign".to_owned(), vec![Term::List(vec![Term::Hole]), Term::Hole]);
        let term = Term::Node(
            STMT_BLOCK_TAG.to_owned(),
            vec![Term::List(vec![assign(), assign()])],
        );
        let mut m = Measure::default();
        measure(&PyDialect, &term, Ctx::Stmt, &mut m);
        assert_eq!(m.anchors.len(), 0, "no shared identifiers/literals → no anchors");
        assert!(
            extractable(&PyDialect, &term, &ExtractCfg::default()).is_none(),
            "an anchorless all-hole skeleton must not extract"
        );
    }

    #[test]
    fn signature_only_function_is_not_an_anchor() {
        // `def _fn(_v0: str) -> bool: return ?` — the body is a single hole; the only shared names are
        // the type annotations (`str`, `bool`) and the synthetic name (`_fn`). None is shared *logic*,
        // so anchors stays empty and the function does not extract — there is nothing to collapse.
        let term = parse_term("FunctionDef('_fn', arguments([], [arg('_v0', Name('str'))]), [Return(Constant(True))], [], Name('bool'))");
        // Replace the body's Constant(True) with a hole, as a fold of divergent return values would.
        let Term::Node(tag, mut ch) = term else { panic!("node") };
        ch[2] = Term::List(vec![Term::Node("Return".to_owned(), vec![Term::Hole])]);
        let term = Term::Node(tag, ch);
        let mut m = Measure::default();
        measure(&PyDialect, &term, Ctx::Expr, &mut m);
        assert_eq!(m.anchors.len(), 0, "type annotations and `_fn` are signature, not anchors: {:?}", m.anchors);
        assert!(extractable(&PyDialect, &term, &ExtractCfg::default()).is_none(), "a body-less helper must not extract");
    }

    #[test]
    fn whole_fn_helpers_emits_create_collapses_noise() {
        // Two create-shapes (extractable → a HelperCandidate) plus two functions whose only shared
        // shape is a divergent middle statement (not extractable → dropped). The pass returns exactly
        // the one helper, carrying its body, a parameter count, and the call sites.
        let create_a = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('AuthRefreshToken', Load), [Constant('x')], [])), Expr(Call(Attribute(Name('session', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)".to_owned();
        let create_b = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('Clientspace', Load), [Constant('y')], [])), Expr(Call(Attribute(Name('session', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)".to_owned();
        let cands = whole_fn_helpers(&PyDialect, &[create_a, create_b], 0.7, &ExtractCfg::default(), Concurrency::Cpu);
        assert_eq!(cands.len(), 1, "exactly the create-shape helper");
        let c = &cands[0];
        assert_eq!(c.members, vec![0, 1]);
        assert_eq!(c.support, 2);
        assert_eq!(c.granularity, "whole-fn");
        assert_eq!(c.body, "def _fn(): _v0 = ?(?); session.add(_v0); return _v0", "body: {}", c.body);
        assert!(c.params >= 1 && !c.signature.is_empty());
    }

    #[test]
    fn subblock_mines_embedded_motif_by_support() {
        // The get-or-raise guard pair embedded in two otherwise-different functions (surrounded by
        // distinct noise statements that do NOT recur). The miner must surface the 2-statement motif
        // by support=2, hole the divergent *exception class* (a callee — a bindable param), and render
        // the sub-block body — even though neither whole function is similar to the other. The lookup
        // method (`get`) is shared: a *varying* method name would be a selector hole and disqualify
        // (see `selector_hole_disqualifies_varying_method_name`), so the fixture keeps it fixed.
        let fa = "FunctionDef('_fn', arguments([], []), [Expr(Call(Name('log'), [Constant('a')], [])), Assign([Name('_v0', Store)], Call(Attribute(Name('self', Load), 'get', Load), [Name('_v1', Load)], [])), If(Compare(Name('_v0', Load), [Is], [Constant(None)]), [Raise(Call(Name('KeyError', Load), [Name('_v1', Load)], []))], []), Return(Name('_v0', Load))], [], None)".to_owned();
        let fb = "FunctionDef('_fn', arguments([], []), [AugAssign(Name('_v2', Store), Add, Constant(1)), Assign([Name('_v0', Store)], Call(Attribute(Name('self', Load), 'get', Load), [Name('_v1', Load)], [])), If(Compare(Name('_v0', Load), [Is], [Constant(None)]), [Raise(Call(Name('LookupError', Load), [Name('_v1', Load)], []))], []), Pass()], [], None)".to_owned();
        let cands = subblock_helpers(&PyDialect, &[fa, fb], 2, &ExtractCfg::default());
        assert_eq!(cands.len(), 1, "exactly the get-or-raise motif");
        let c = &cands[0];
        assert_eq!(c.granularity, "sub-block");
        assert_eq!(c.support, 2);
        assert_eq!(c.members, vec![0, 1]);
        // The two statements, shared method (`get`) verbatim and the divergent exception
        // (`KeyError`/`LookupError`) holed, rendered as a sub-block (no function header).
        assert_eq!(c.body, "_v0 = self.get(_v1); if _v0 is None: raise ?(_v1)", "body: {}", c.body);
        assert_eq!(c.params, 1, "the holed exception class is the one param");
    }
}
