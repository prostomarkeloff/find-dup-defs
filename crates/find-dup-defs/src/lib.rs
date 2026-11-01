//! `find-dup-defs` library — the cross-file duplicate-definition detection pipeline.
//!
//! The engine is **frontend-agnostic**: it consumes a `Vec<`[`Def`]`>` (each carrying its
//! precomputed canonical strings and a `&'static `[`KindSpec`]) and never names a concrete
//! language crate. The binary owns the [`Frontend`] registry and passes `&[&dyn Frontend]` in;
//! adding a language is a new frontend crate, not an engine edit.
#![allow(
    clippy::struct_excessive_bools // PipelineOpts mirrors CLI flags, not a state machine
)]
//!
//! Three complementary passes (all over the canon the frontend computed in one parse per file):
//!   1. **name-gated** — same-`(kind, name)` defs clustered by exact
//!      Ratcliff–Obershelp similarity (via `difflib-fast`).
//!   2. **cross-name** — renamed copy-paste: alpha-renamed canonical bucketed
//!      with ≥2 distinct names across ≥2 sites.
//!   3. **Type-3** (`ECScan`) — IDF-weighted cosine over name-agnostic lines;
//!      edited renamed copies the exact pass misses.
//!
//! Each cluster is graded ERROR / WARNING / INFO, with optional thickness-based
//! demotion/escalation passes the caller can request via [`PipelineOpts`].

mod bicm;
pub mod converge;
pub mod decide;
pub mod patternology;
mod simgraph;
pub mod type3;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use dup_defs_core::{CanonDialect, Def, Frontend, KindSpec, ScanOpts};
use rayon::prelude::*;
use serde::Serialize;

/// Wall-time a pipeline phase to stderr when `FDD_TIMING` is set; a transparent pass-through
/// otherwise. Output-neutral (stderr only, behind the env flag) so it never affects findings.
pub fn timed<T>(label: &str, f: impl FnOnce() -> T) -> T {
    if std::env::var_os("FDD_TIMING").is_none() {
        return f();
    }
    let t = Instant::now();
    let r = f();
    eprintln!("[timing] {label:<12} {:>8.1} ms", t.elapsed().as_secs_f64() * 1000.0);
    r
}

// ── Constants ──────────────────────────────────────────────────────────────

/// Cross-name: ERROR only when the alpha-renamed canonical has ≥ this many AST nodes. Avoids
/// escalating "two `return []` one-liners" to ERROR purely on a renamed-exact match. A policy on
/// how much copied text gates a build; measured as an alternative, the rarity of the copy under
/// the corpus's null escalated every rare three-liner on a small tree and tripled the ERRORs on a
/// large one, so the count stays and the null's p-value rides along as a note.
pub const SUBSTANCE_NODES: usize = 20;
/// Type-3 cluster's min soft cosine ≥ this → ERROR (else WARNING): the tier is a policy on how
/// alike two functions must be to gate a build, and so a number the reader owns, like
/// `--error-thickness`.
pub const TYPE3_ERROR_THETA: f64 = 0.9;
/// Patternology family's min structural cosine ≥ this → WARNING (else INFO). Patternology never
/// reaches ERROR — structural similarity is advisory, not a gate.
pub const PATTERN_WARNING_THETA: f64 = 0.92;
/// Section offset for patternology findings. The existing per-pass offsets (name 0 / cross-name 1 /
/// type-3 2) pack kinds tightly (functions 1/2/3, methods 4/5/6, classes 7, …), leaving no slot for
/// a 4th pass without colliding with the next kind. Patternology is a new advisory layer, so its
/// sections are filed as an APPENDIX after every existing section via this large base-relative
/// offset — no existing index moves, so the legacy report ordering is byte-identical.
pub const PATTERN_SECTION_OFFSET: usize = 1000;

/// Default `--converge-top`: how many divergences of each kind are worth reading in one sitting.
/// Not a threshold on what is a finding — the pass has none — but on how much of a ranking a report
/// should be.
pub const DEFAULT_CONVERGE_TOP: usize = 50;

/// Directory-name blacklist for source discovery — virtualenvs, package
/// caches, build artefacts, vendored tooling, JS bundler outputs.
const SKIP_DIRS: &[&str] = &[
    // Python ecosystem
    ".venv", "venv", "venv2", "venv3", "env", ".env",
    "__pycache__", ".tox", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".ipynb_checkpoints", "site-packages",
    // JS / TS ecosystem
    "node_modules", "dist", "out", "build",
    ".next", ".nuxt", ".turbo", ".cache", "coverage",
    // VCS / editors / build artefacts
    ".git", "target", ".idea", ".vscode", ".direnv",
];

fn is_excluded_dir(name: &str) -> bool {
    SKIP_DIRS.contains(&name) || name.ends_with(".egg-info")
}

/// Every entry under `dir`, walked in parallel, with excluded directories pruned whole.
///
/// 🔴 Stat'ing the tree was 10% of the run's CPU and all of it on one thread — a sequential walker
/// over fifty thousand files while eleven cores waited. The result is collected into a set, so the
/// ORDER the tree comes back in is not observable and the walk has nothing to serialize on.
///
/// Directories are yielded as entries too, and an unreadable one is skipped rather than fatal —
/// both matching the sequential walker this replaces, whose `filter_entry` likewise pruned an
/// excluded directory without descending into it or reporting it.
fn walk_entries(dir: &Path) -> Vec<PathBuf> {
    let Ok(reader) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let entries: Vec<std::fs::DirEntry> = reader.filter_map(Result::ok).collect();
    entries
        .par_iter()
        .flat_map(|entry| {
            let path = entry.path();
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                if is_excluded_dir(&entry.file_name().to_string_lossy()) {
                    return Vec::new();
                }
                let mut under = walk_entries(&path);
                under.push(path);
                under
            } else {
                vec![path]
            }
        })
        .collect()
}

// ── Severity ───────────────────────────────────────────────────────────────

/// Cluster severity. The pipeline emits ERROR / WARNING / INFO; the consumer
/// (CLI or Python wrapper) maps to whatever wire shape it wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Severity {
    Error,
    Warning,
    /// Low-confidence finding: name-collision constants whose bodies differ,
    /// mass-demoted WARNINGs below `warning_thickness`, or directive-chained
    /// de-escalations from WARNING.
    Info,
}

impl Severity {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Severity::Error => "ERROR",
            Severity::Warning => "WARNING",
            Severity::Info => "INFO",
        }
    }

    /// 0/1/2 ladder for directive-driven stepping — `escalate` goes UP
    /// (toward 0=ERROR), `de-escalate` goes DOWN (toward 2=INFO).
    #[must_use]
    pub fn to_index(self) -> i32 {
        match self {
            Severity::Error => 0,
            Severity::Warning => 1,
            Severity::Info => 2,
        }
    }

    #[must_use]
    pub fn from_index(i: i32) -> Self {
        match i.clamp(0, 2) {
            0 => Severity::Error,
            1 => Severity::Warning,
            _ => Severity::Info,
        }
    }
}

// ── Finding ────────────────────────────────────────────────────────────────

/// Structured patternology payload — the machine-readable form of a pattern finding's `notes`
/// strings (the anti-unification template, its stable cross-package signature, and the collapse
/// economics). `Some` only for `pass == "pattern"` findings; the textual passes leave it `None`.
/// Consumers (e.g. a type-aware fusion layer) group by `signature` and read `template` / `support`
/// without re-parsing the human note strings.
#[derive(Clone, Debug, Serialize)]
pub struct PatternInfo {
    /// Pseudo-source of the proposed parameterized helper (the LGG template body, holes `?`).
    pub template: String,
    /// Stable signature key (the LGG tree, holes `?`, atoms verbatim) — the cross-package group-by key.
    pub signature: String,
    /// Parameters the helper would take (= expression-holes).
    pub params: usize,
    /// `"whole-fn"` | `"sub-block"` — which pass produced it.
    pub granularity: &'static str,
    /// Distinct call sites the family spans (the codometry support count).
    pub support: usize,
    /// Estimated LOC removed by collapsing the family into the helper.
    pub loc_saved: usize,
}

/// One reported cluster of duplicate definitions.
#[derive(Clone, Debug)]
pub struct Finding {
    /// Which pass produced the finding: `"name"` / `"cross-name"` / `"type-3"`.
    pub pass: &'static str,
    /// The kind of the clustered definitions, as declared by the frontend.
    pub kind: &'static KindSpec,
    pub name: String,
    pub severity: Severity,
    /// Min pairwise similarity inside the cluster. `None` for name-only kinds.
    pub min_sim: Option<f64>,
    /// Max non-blank-line count across cluster members.
    pub loc: usize,
    /// Max parameter count across members (0 for non-callable kinds).
    pub args: usize,
    /// Normalized [0, 1] "GET ME REFACTORED" score — see [`thickness`].
    pub thickness: f64,
    /// Pre-strip source of one representative member, for calibration display.
    pub snippet: String,
    /// Notes attached by matching directives.
    pub notes: Vec<String>,
    /// `(file, line 1-indexed, col 0-indexed)` for every member of the cluster.
    pub members: Vec<(String, usize, usize)>,
    /// Structured patternology payload (template + signature + economics); `Some` only for the
    /// `"pattern"` pass. Mirrors the `notes` strings in machine-readable form.
    pub pattern: Option<PatternInfo>,
    /// Per-facet agreement: `(facet, shared facts)` for every facet the cluster's members agree on,
    /// strongest first. Empty unless the frontend tags its `type3_lines` as `facet:fact` — see
    /// [`facet_votes`]. What it answers is "*why* did these cluster": a definition can look like
    /// another through one perspective and not through six, and only the tally distinguishes the
    /// two cases.
    pub facets: Vec<(String, usize)>,
}

/// A [`Finding`] is a [`directiva`] directive target: its qualifier is the kind id (so a
/// `<methods>` filter matches), its names are the cluster name plus each `/`-joined alias (so
/// `Foo.bar` lands on a cross-name `Foo.bar/Baz.bar`), and its scopes are the member file paths
/// (any member match wins).
impl directiva::Target for Finding {
    fn qualifier(&self) -> Option<&str> {
        Some(self.kind.id)
    }
    fn matches_name(&self, pat: &directiva::Pattern) -> bool {
        self.name.split('/').any(|alias| pat.matches(alias)) || pat.matches(&self.name)
    }
    fn matches_scope(&self, pat: &directiva::Pattern) -> bool {
        self.members.iter().any(|(file, _, _)| pat.matches(file))
    }
}

/// How much a frontend-supplied [`Def::thickness`] is discounted by weak agreement: at `min_sim`
/// 0 a cluster keeps this fraction of its score, at 1 it keeps all of it. A frontend score says how
/// *interesting* a member is on its own — how rare its facts are — which is a potential, not a
/// finding. Whether the cluster realized that potential is what `min_sim` measures, so the two are
/// combined multiplicatively rather than added: a member full of rare facts that only half-matches
/// its cluster should not outrank one that matched exactly.
const AGREEMENT_FLOOR: f64 = 0.6;

/// The cluster's refactor-payoff score: the minimum of the members' frontend-supplied
/// [`Def::thickness`], scaled by how tightly the cluster agrees, or the engine's default
/// [`thickness`] formula when any member lacks a score. Taking the minimum keeps a cluster from
/// reading as thicker than its thinnest member — the same conservative convention as `min_sim`
/// itself. (The default formula already folds `sim` in at 0.2 weight, hence the scaling here
/// applying only to the override path.)
fn cluster_thickness(
    defs: &[Def],
    members: impl IntoIterator<Item = usize>,
    min_sim: f64,
    fallback: impl FnOnce() -> f64,
) -> f64 {
    let mut best: Option<f64> = None;
    for i in members {
        match defs[i].thickness {
            Some(t) => best = Some(best.map_or(t, |b: f64| b.min(t))),
            None => return fallback(),
        }
    }
    match best {
        Some(t) => t * (AGREEMENT_FLOOR + (1.0 - AGREEMENT_FLOOR) * min_sim.clamp(0.0, 1.0)),
        None => fallback(),
    }
}

/// Normalized [0, 1] "GET ME REFACTORED" score. Driven by three dimensions,
/// each saturated independently with `1 - exp(-x/k)`:
///
/// * `volume = (n_members - 1) * loc` — the lines you'd actually delete by
///   extracting one shared helper.
/// * `args` — wide signatures push the score up marginally.
/// * `sim` — `1.0` for normalized-exact/cross-name passes, the cluster's min
///   pairwise ratio for name-gated body kinds.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn thickness(loc: usize, args: usize, n_members: usize, sim: f64) -> f64 {
    let volume = (loc as f64) * (n_members.saturating_sub(1) as f64);
    let volume_score = 1.0 - (-volume / 30.0).exp();
    let args_score = 1.0 - (-(args as f64) / 5.0).exp();
    0.7 * volume_score + 0.1 * args_score + 0.2 * sim
}

/// Which facets a cluster agrees on, and by how many facts each.
///
/// A frontend that projects a definition through several perspectives tags each fact with the one
/// it came from (`control:if`, `outgoing:.commit`). The engine stays ignorant of what those tags
/// mean; it intersects the members' facts and counts what survives per tag. A cluster carried by
/// one facet and a cluster every facet agrees on are very different findings, and the similarity
/// score alone cannot tell them apart — it says how close, never through what.
///
/// Empty when the facts carry no tags, so an untagged frontend pays nothing and reports nothing.
fn facet_tag(fact: &str) -> Option<&str> {
    let (head, _) = fact.split_once(':')?;
    let tagged = !head.is_empty()
        && head.len() <= 16
        && head.bytes().all(|b| b.is_ascii_lowercase() || b == b'-');
    tagged.then_some(head)
}

#[must_use]
fn facet_votes(defs: &[Def], members: impl IntoIterator<Item = usize>) -> Vec<(String, usize)> {
    let mut shared: Option<BTreeSet<&str>> = None;
    for i in members {
        let Some(a) = &defs[i].analysis else { return Vec::new() };
        let facts: BTreeSet<&str> = a.type3_lines.iter().map(String::as_str).collect();
        shared = Some(match shared {
            None => facts,
            Some(prev) => prev.intersection(&facts).copied().collect(),
        });
    }
    let mut per_facet: BTreeMap<&str, usize> = BTreeMap::new();
    for fact in shared.unwrap_or_default() {
        // A tag is a bare lowercase word before the colon. Ordinary source lines contain colons
        // too (`def _fn(_v0):`, `for _v0 in _v1:`), so anything else is an untagged fact and is
        // counted under no facet rather than under a bogus one.
        if let Some(tag) = facet_tag(fact) {
            *per_facet.entry(tag).or_insert(0) += 1;
        }
    }
    let mut out: Vec<(String, usize)> =
        per_facet.into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

// ── File collection ────────────────────────────────────────────────────────

/// Index of the first frontend that claims `path`'s extension, if any.
fn frontend_for(frontends: &[&dyn Frontend], path: &Path) -> Option<usize> {
    let ext = path.extension()?.to_str()?;
    frontends.iter().position(|f| f.extensions().contains(&ext))
}

/// Walk `paths` once, route each file to the frontend that claims its extension, and scan.
/// Replaces the per-language `WalkDir` passes: the tree is traversed a single time regardless
/// of how many frontends are active. Per-frontend file lists are gathered into a `BTreeSet` so
/// they are deduplicated and sorted before scanning (the engine re-sorts defs afterwards, so
/// this is defense-in-depth for determinism).
/// `(the file already parsed, the file that repeats its bytes)`, per copy.
type Twins = Vec<(Arc<str>, Arc<str>)>;

/// Split the discovered files into the ones worth parsing and the ones that are a copy of another.
///
/// 🔴 85% of the files in a real monorepo are byte-identical to another one — vendored trees,
/// generated clients, a package present in several lockfile-pinned copies. Parsing is the single
/// most expensive thing this tool does per file, and it was doing it seven times over for the same
/// bytes. Content is compared in full, not by digest: a hash collision here would silently attach
/// one file's definitions to another file's path.
///
/// Returns `(files to parse, (source, twin) pairs)`. The file kept for each content is the first in
/// the discovered order, which is sorted, so which one is parsed does not vary between runs. A file
/// that cannot be read is kept as its own group and left to the frontend to fail on as before.
fn one_per_content(files: &[Arc<str>]) -> (Vec<Arc<str>>, Twins) {
    // 🔴 Size-bucket first. A file is only worth reading+digesting for twin detection when its
    // byte count is shared with at least one other file — equal content implies equal size. A
    // unique size cannot have a twin, so its bytes are not read here at all; the frontend reads it
    // exactly once when parsing. A file whose stat fails is treated the same way (kept, no twin),
    // matching the old read-failure behavior.
    let sizes: Vec<Option<u64>> = files
        .par_iter()
        .map(|f| std::fs::metadata(f.as_ref()).ok().map(|m| m.len()))
        .collect();
    let mut size_count: rustc_hash::FxHashMap<u64, u32> = rustc_hash::FxHashMap::default();
    for sz in sizes.iter().flatten() {
        *size_count.entry(*sz).or_default() += 1;
    }

    // Read and digested in parallel (only the files that share a size); the sequential pass below
    // then compares CONTENT only against files that already agree on their digest — one comparison,
    // where keying an ordered map by the bytes themselves meant a dozen full-file compares per lookup.
    let read: Vec<Option<(u64, Vec<u8>)>> = files
        .par_iter()
        .enumerate()
        .map(|(i, f)| {
            let shared = sizes[i].is_some_and(|sz| size_count.get(&sz).copied().unwrap_or(0) >= 2);
            if !shared {
                return None;
            }
            std::fs::read(f.as_ref()).ok().map(|bytes| {
                use std::hash::{BuildHasher, Hasher};
                let mut h = rustc_hash::FxBuildHasher.build_hasher();
                h.write(&bytes);
                (h.finish(), bytes)
            })
        })
        .collect();

    let mut by_digest: rustc_hash::FxHashMap<u64, Vec<usize>> = rustc_hash::FxHashMap::default();
    let (mut to_parse, mut twins) = (Vec::new(), Vec::new());
    for (i, file) in files.iter().enumerate() {
        let Some((digest, content)) = read[i].as_ref() else {
            to_parse.push(Arc::clone(file));
            continue;
        };
        // The digest narrows it; the bytes decide it. A collision must not attach one file's
        // definitions to another file's path, so the comparison stays exact.
        let seen = by_digest.entry(*digest).or_default();
        if let Some(&source) = seen.iter().find(|&&r| read[r].as_ref().is_some_and(|(_, c)| c == content)) {
            twins.push((Arc::clone(&files[source]), Arc::clone(file)));
        } else {
            seen.push(i);
            to_parse.push(Arc::clone(file));
        }
    }
    (to_parse, twins)
}

#[must_use]
pub fn collect_defs(frontends: &[&dyn Frontend], paths: &[PathBuf], opts: &ScanOpts) -> Vec<Def> {
    // 🔴 Sorted vectors, not `BTreeSet`s. The set was only ever used to order and deduplicate, and
    // then handed on as a slice — but building it meant inserting fifty thousand paths one at a
    // time, on one thread, each insert a walk down a tree of string compares. Routing and sorting
    // are both per path and both parallel; a sorted, deduplicated vector iterates in exactly the
    // order the set did.
    let per_frontend: Vec<Vec<Arc<str>>> = timed("discovery", || {
        let mut candidates: Vec<PathBuf> = Vec::new();
        for p in paths {
            candidates.push(p.clone());
            if p.is_dir() {
                // The root itself is an entry, as it was for the sequential walker.
                candidates.extend(walk_entries(p));
            }
        }
        let routed: Vec<(usize, Arc<str>)> = candidates
            .par_iter()
            .filter_map(|path| {
                frontend_for(frontends, path).map(|fi| (fi, Arc::from(path.to_string_lossy().as_ref())))
            })
            .collect();
        let mut per_frontend: Vec<Vec<Arc<str>>> = vec![Vec::new(); frontends.len()];
        for (fi, file) in routed {
            per_frontend[fi].push(file);
        }
        per_frontend.par_iter_mut().for_each(|files| {
            files.sort_unstable();
            files.dedup();
        });
        per_frontend
    });
    // Recursive-descent parsers (syn especially) have no built-in nesting limit, and a deeply
    // nested file — rustc's parser-stress fixtures, generated code — can exhaust a worker
    // thread's default 2 MiB stack during parse or canonicalization. Run the scan on a pool with
    // a generous per-worker stack so realistically-deep input is handled; the few inputs nested
    // beyond this would also blow a compiler's own limits.
    let scan = move || {
        // 🔴 NOT for a frontend whose output is defined ACROSS the set — the Python `use` lens
        // counts "how this definition is used elsewhere" over the files the scan was given, and
        // shortening that list changed the profile and with it the lens thickness (a ±0.01 drift
        // on four clusters, which is the parity gate earning its keep). The frontend says so
        // through `scans_across_files`, and is handed everything. For every other frontend and
        // kind a copy of the bytes is a copy of the definitions; the one corpus-relative number a
        // per-file frontend produces, the lens score, is taken again below over the replayed set.
        let mut defs = Vec::new();
        for (fi, files) in per_frontend.into_iter().enumerate() {
            let per_file_only = !frontends[fi].scans_across_files(opts);
            let (to_parse, twins) =
                if per_file_only { one_per_content(&files) } else { (files.clone(), Vec::new()) };
            let mut parsed = frontends[fi].scan(&to_parse, opts);
            // Replay each parsed file's definitions onto the paths that spell the same bytes. A
            // definition is a function of the source and where in it the definition sits — the path
            // is recorded, never read — so this is the same `Vec<Def>` the parse would have
            // produced, at the cost of a clone instead of a parse.
            if !twins.is_empty() {
                let mut by_file: rustc_hash::FxHashMap<&str, Vec<&Def>> = rustc_hash::FxHashMap::default();
                for def in &parsed {
                    by_file.entry(def.file.as_ref()).or_default().push(def);
                }
                // A `Def` is a dozen strings, and there are hundreds of thousands to copy: pure
                // per twin, so the copying runs on the pool. `collect` keeps the twin order.
                let copies: Vec<Def> = twins
                    .par_iter()
                    .flat_map_iter(|(source, twin)| {
                        by_file.get(source.as_ref()).into_iter().flatten().map(move |def| {
                            let mut copy = (*def).clone();
                            copy.file = Arc::clone(twin);
                            copy
                        })
                    })
                    .collect();
                parsed.extend(copies);
                // The score is corpus-relative: the frontend took it over the parsed files, and
                // the corpus is the parsed files plus their copies. Re-scoring is a pure function
                // of the record set, so this is the number the frontend would have produced had
                // it been handed every copy.
                if opts.wants("lenses") {
                    dup_defs_core::lens::score_lens_defs(&mut parsed);
                }
            }
            defs.extend(parsed);
        }
        defs
    };
    // Run the scan on a pool with a generous per-worker stack, falling back to the global pool if
    // the pool can't be built (only fails on OS thread-spawn exhaustion).
    timed("scan", || match rayon::ThreadPoolBuilder::new().stack_size(64 * 1024 * 1024).build() {
        Ok(pool) => pool.install(scan),
        Err(_) => scan(),
    })
}

// ── Cluster helpers ─────────────────────────────────────────────────────────

/// `(file, line 1-indexed, col 0-indexed)` for a def member of a cluster.
#[must_use]
pub fn member(defs: &[Def], i: usize) -> (String, usize, usize) {
    (defs[i].file.to_string(), defs[i].line + 1, defs[i].col)
}

/// Pick the `&'static KindSpec` to label a cross-name / Type-3 cluster of callables: METHOD if
/// every member is a method, otherwise FUNCTION (a mixed function/method cluster reports as a
/// function, matching the historical behavior). All frontends use identical `KindSpec` fields
/// for a given `id`, so taking a member's own spec is correct regardless of language.
fn callable_kind(defs: &[Def], members: &[usize]) -> &'static KindSpec {
    if members.iter().all(|&p| defs[p].kind.id == "methods") {
        defs[members[0]].kind
    } else {
        let p = members.iter().copied().find(|&p| defs[p].kind.id == "functions").unwrap_or(members[0]);
        defs[p].kind
    }
}

// ── Section index (for stable, reproducible cluster sort) ──────────────────

/// Printed-section index — a kind's `section` base plus a per-pass offset for callables
/// (`name` 0 / `cross-name` 1 / `type-3` 2). Reproduces the historical fixed ordering:
/// constants 0, functions 1/2/3, methods 4/5/6, classes 7, interfaces 8, type-aliases 9.
#[must_use]
pub fn section_index(f: &Finding) -> usize {
    let base = f.kind.section as usize;
    let offset = if f.kind.fn_like {
        match f.pass {
            "cross-name" => 1,
            "type-3" => 2,
            "pattern" => PATTERN_SECTION_OFFSET,
            "converge" => converge::CONVERGE_SECTION_OFFSET,
            "converge-family" => converge::FAMILY_SECTION_OFFSET,
            "decision" => decide::DECISION_SECTION_OFFSET,
            "decision-family" => decide::DECISION_FAMILY_SECTION_OFFSET,
            _ => 0,
        }
    } else {
        0
    };
    base + offset
}

// ── Passes ─────────────────────────────────────────────────────────────────

/// Exact single-linkage clustering of a name group, over its DISTINCT canonicals only.
///
/// 🔴 87% of the definitions in a name group share their canonical with another member of it — on a
/// real tree that is not a quirk, it is the thing a duplicate detector is looking for. The
/// clustering underneath builds one suffix automaton per input and then compares pairs, so handing
/// it the group as-is meant 369k automata over 4.65M pairs where 48k over 52k say the same thing:
/// the pair count is quadratic in the multiplicity that carries no information.
///
/// **Exact, not an approximation.** Two identical strings have a Ratcliff–Obershelp ratio of
/// exactly 1.0 (`2M/T` with `M` = length and `T` = twice it), so:
///   - they are always joined, at any threshold ≤ 1 — a duplicate never lands in another cluster;
///   - adding them to a cluster cannot lower its minimum pairwise ratio, which is the figure
///     reported, so the min over the whole group equals the min over its distinct representatives.
///
/// 🔴 **Two representatives per canonical, not one** — the first occurrence and the last.
/// Ratcliff–Obershelp is **asymmetric**: the recursion indexes one side, so `ratio(x, y)` and
/// `ratio(y, x)` can differ, and the clustering computes the pair in the orientation its two
/// indices happen to have. When a canonical occurs at several positions, the group therefore
/// contains BOTH orientations of some pair, and the reported minimum is the lower of the two.
/// Collapsing to a single representative computes only one orientation and reports a minimum that
/// is too high — measured at +0.005 on four clusters, with membership unchanged, which is how this
/// was found. Keeping the first and last occurrence preserves the orientation set exactly:
/// `x → y` is realizable in the group iff `first(x) < last(y)`, which is what the two survivors
/// still say. A canonical spelled once contributes one representative.
///
/// A canonical shared by several definitions needs no special case: its two representatives are
/// identical, so they are an edge at any threshold and come back as a cluster of their own.
fn cluster_distinct(
    rationer: &difflib_fast::Rationer,
    canons: &[&str],
    threshold: f64,
) -> Vec<(Vec<usize>, f64)> {
    let mut seen: rustc_hash::FxHashMap<&str, usize> = rustc_hash::FxHashMap::default();
    // Distinct index → the positions in `canons` that spell it, ascending by construction.
    let mut spelled_by: Vec<Vec<usize>> = Vec::new();
    for (pos, &s) in canons.iter().enumerate() {
        if let Some(&d) = seen.get(s) {
            spelled_by[d].push(pos);
        } else {
            seen.insert(s, spelled_by.len());
            spelled_by.push(vec![pos]);
        }
    }

    // 🔴 A group that spells ONE canonical needs no clustering at all, and most groups are that:
    // 43 504 groups hold 47 733 distinct canonicals between them. Every pair is a pair of identical
    // strings, so the answer is the whole group at ratio 1.0 — and the machinery below would reach
    // it only after building a suffix automaton, which was 26% of the run's CPU. Nothing to compare
    // means nothing to build.
    if spelled_by.len() == 1 {
        return if canons.len() >= 2 { vec![((0..canons.len()).collect(), 1.0)] } else { Vec::new() };
    }

    // Kept in position order, so a survivor pair's orientation is the one it had in the group.
    //
    // The second representative is only needed when something else can sit BETWEEN this canonical's
    // first and last occurrence — that is what makes both orientations of a pair reachable. If the
    // occurrences are CONTIGUOUS nothing does, every other canonical lies wholly before or wholly
    // after them, and one representative at the first occurrence answers both questions: `x → y` is
    // reachable iff some `y` follows the block, `y → x` iff some `y` precedes it, and `first(x)`
    // decides each. So a contiguous run of copies costs one automaton, not two.
    let mut reps: Vec<(usize, usize)> = Vec::new();
    for (d, positions) in spelled_by.iter().enumerate() {
        reps.push((positions[0], d));
        let last = positions[positions.len() - 1];
        let contiguous = last - positions[0] + 1 == positions.len();
        if positions.len() > 1 && !contiguous {
            reps.push((last, d));
        }
    }
    reps.sort_unstable();
    // The representatives' canonicals as borrowed `&str` — `candidates` reads lengths and counts
    // straight off them (one histogram pass), so `Vec<char>` is only materialized for the strings
    // that survive its bounds and actually reach the clusterer, never for the singletons the bounds
    // rule out. A clone-and-sort plus an allocation each, saved on the hot path.
    let rep_strs: Vec<&str> = reps.iter().map(|&(pos, _)| canons[pos]).collect();

    // 🔴 Only the representatives that can be in an edge at all are handed to the clusterer. Its
    // bounds are exact upper bounds on the ratio, so a string without a partner under them is a
    // singleton whatever the clusterer computes, and leaving it out changes no edge, no cluster, no
    // minimum.
    let active = candidates(&rep_strs, threshold);
    let clustered = if active.len() == rep_strs.len() {
        let chars: Vec<Vec<char>> = rep_strs.iter().map(|s| s.chars().collect()).collect();
        rationer.cluster_canonicals_chars(&chars, threshold)
    } else if active.len() < 2 {
        Vec::new()
    } else {
        let subset: Vec<Vec<char>> = active.iter().map(|&i| rep_strs[i].chars().collect()).collect();
        rationer
            .cluster_canonicals_chars(&subset, threshold)
            .into_iter()
            .map(|(members, min_sim)| (members.into_iter().map(|m| active[m]).collect(), min_sim))
            .collect()
    };

    let mut out: Vec<(Vec<usize>, f64)> = Vec::new();
    // A canonical's two representatives always land in the same cluster, so each contributes its
    // positions once; clusters are disjoint, so one flag vector serves the whole loop.
    let mut taken = vec![false; spelled_by.len()];
    for (members, min_sim) in clustered {
        let mut expanded: Vec<usize> = Vec::new();
        for r in members {
            let d = reps[r].1;
            if !taken[d] {
                taken[d] = true;
                expanded.extend_from_slice(&spelled_by[d]);
            }
        }
        expanded.sort_unstable();
        out.push((expanded, min_sim));
    }
    // A canonical reduced to ONE representative (its copies are contiguous) that clustered with
    // nothing is still a cluster of its copies, pairwise 1.0 — the call saw a single string and
    // dropped it as a singleton, so it is added back. With two representatives this could not
    // happen: they are identical, hence an edge, hence already a cluster.
    for (d, positions) in spelled_by.iter().enumerate() {
        if !taken[d] && positions.len() >= 2 {
            out.push((positions.clone(), 1.0));
        }
    }
    // Same order the undeduplicated call produced: members ascending, clusters by their first.
    out.sort_by(|a, b| a.0[0].cmp(&b.0[0]));
    out
}

/// Sorted `(char, count)` multiset of a canonical — the same shape the clusterer computes for its
/// own `quick_ratio` bound. ASCII (canonical code, nearly always) is one histogram pass with no
/// allocation; non-ASCII falls back to sort-and-count, byte-identical to the previous behavior.
fn char_counts_str(s: &str) -> Vec<(char, u32)> {
    let mut hist = [0u32; 128];
    if s.chars().all(|c| {
        let u = c as usize;
        if u < 128 {
            hist[u] += 1;
            true
        } else {
            false
        }
    }) {
        return hist
            .iter()
            .enumerate()
            .filter(|&(_, &k)| k > 0)
            .map(|(i, &k)| (char::from(u8::try_from(i).expect("ascii")), k))
            .collect();
    }
    let mut v: Vec<char> = s.chars().collect();
    v.sort_unstable();
    let mut out: Vec<(char, u32)> = Vec::new();
    for c in v {
        match out.last_mut() {
            Some(last) if last.0 == c => last.1 += 1,
            _ => out.push((c, 1)),
        }
    }
    out
}

/// The strings that have at least one partner under the two exact upper bounds the clusterer
/// filters pairs by: `ratio ≤ 2·min(|a|,|b|)/(|a|+|b|)` (lengths) and
/// `ratio ≤ 2·Σ min(count_a(c), count_b(c))/(|a|+|b|)` (character multisets). Same formulas, same
/// floats; the pairs admitted are a superset of the clusterer's, never a subset. Length-sorted so
/// each string only scans the window its length admits — the clusterer's own blocking.
///
/// Reads the canonicals borrowed (no `Vec<char>`), counting each in one histogram pass instead of
/// a clone-and-sort; the caller materializes `Vec<char>` only for the strings this admits.
fn candidates(strings: &[&str], threshold: f64) -> Vec<usize> {
    // `threshold ≤ 0` admits every pair (both bounds are `≥ 0`), so every string is a partner and
    // the blocking below is pure overhead — skip it. Exactly what the loop would conclude.
    if threshold <= 0.0 {
        return (0..strings.len()).collect();
    }
    let lengths: Vec<usize> = strings.iter().map(|s| s.chars().count()).collect();
    #[allow(clippy::cast_precision_loss)]
    let length_bound = |a: usize, b: usize| -> f64 {
        let total = a + b;
        if total == 0 { 1.0 } else { 2.0 * (a.min(b) as f64) / (total as f64) }
    };
    let counts: Vec<Vec<(char, u32)>> = strings.par_iter().map(|s| char_counts_str(s)).collect();
    #[allow(clippy::cast_precision_loss)]
    let multiset_bound = |i: usize, j: usize| -> f64 {
        let (ca, cb) = (&counts[i], &counts[j]);
        let (mut x, mut y, mut matches) = (0usize, 0usize, 0u32);
        while x < ca.len() && y < cb.len() {
            match ca[x].0.cmp(&cb[y].0) {
                std::cmp::Ordering::Less => x += 1,
                std::cmp::Ordering::Greater => y += 1,
                std::cmp::Ordering::Equal => {
                    matches += ca[x].1.min(cb[y].1);
                    x += 1;
                    y += 1;
                }
            }
        }
        let total = lengths[i] + lengths[j];
        if total == 0 { 1.0 } else { 2.0 * f64::from(matches) / total as f64 }
    };
    let mut order: Vec<usize> = (0..strings.len()).collect();
    order.sort_by_key(|&i| lengths[i]);
    let mut has_partner = vec![false; strings.len()];
    for (p, &i) in order.iter().enumerate() {
        for &j in &order[p + 1..] {
            if length_bound(lengths[i], lengths[j]) < threshold {
                break; // lengths only grow: every remaining partner fails the bound too
            }
            if !(has_partner[i] && has_partner[j]) && multiset_bound(i, j) >= threshold {
                has_partner[i] = true;
                has_partner[j] = true;
            }
        }
    }
    (0..strings.len()).filter(|&i| has_partner[i]).collect()
}

/// Pass 1 — name-gated: same-named body-kind defs clustered by structural-canonical
/// similarity; same-named raw-text kinds (constants / type-aliases) compared by `text_orig`.
///
/// `max_group` (the CLI `--max-name-group`) optionally skips any `(kind, name)` group with more
/// than that many members. Off by default (`None`) — behavior is unchanged unless the caller asks
/// for it. It exists because a name shared by hundreds of definitions (`fn main` across thousands
/// of test fixtures, `new` / `default`) is a convention, not a refactor cluster, and the
/// within-group O(n²) Ratcliff–Obershelp comparison can dominate runtime on huge monorepos;
/// renamed-identical copies among the members still surface via the cross-name pass (O(n)).
#[must_use]
pub fn pass_name_gated(
    defs: &[Def],
    threshold: f64,
    error: f64,
    min_size: usize,
    max_group: Option<usize>,
    rationer: &difflib_fast::Rationer,
) -> Vec<Finding> {
    let mut groups: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
    for (i, d) in defs.iter().enumerate() {
        groups.entry((d.kind.id, d.name.as_str())).or_default().push(i);
    }
    let groups: Vec<((&str, &str), Vec<usize>)> = groups
        .into_iter()
        .filter(|(_, v)| v.len() >= 2 && max_group.is_none_or(|c| v.len() <= c))
        .collect();

    groups
        .par_iter()
        .flat_map_iter(|((_, name), idxs)| {
            // All members of a `(kind.id, name)` group share a kind; any member's spec labels it.
            let kind = defs[idxs[0]].kind;
            if !kind.body {
                let canons: Vec<&str> = idxs.iter().map(|&i| defs[i].text_orig.as_str()).collect();
                let clusters = cluster_distinct(rationer, &canons, 0.0);
                return clusters
                    .into_iter()
                    .filter(|(c, _)| c.len() >= min_size)
                    .map(|(c, min_sim)| {
                        let loc = c.iter().map(|&k| defs[idxs[k]].loc).max().unwrap_or(0);
                        let args = c.iter().map(|&k| defs[idxs[k]].args).max().unwrap_or(0);
                        let severity = if min_sim >= error {
                            Severity::Error
                        } else if min_sim >= threshold {
                            Severity::Warning
                        } else {
                            Severity::Info
                        };
                        Finding {
                            pass: "name",
                            kind,
                            name: (*name).to_owned(),
                            severity,
                            min_sim: Some(min_sim),
                            loc,
                            args,
                            thickness: cluster_thickness(defs, c.iter().map(|&k| idxs[k]), min_sim, || {
                                thickness(loc, args, c.len(), min_sim)
                            }),
                            snippet: defs[idxs[c[0]]].text_orig.clone(),
                            notes: Vec::new(),
                            members: c.iter().map(|&k| member(defs, idxs[k])).collect(),
                            pattern: None,
                            facets: facet_votes(defs, c.iter().map(|&k| idxs[k])),
                        }
                    })
                    .collect::<Vec<_>>();
            }
            let canons: Vec<&str> =
                idxs.iter().map(|&i| defs[i].cluster_canonical.as_deref().unwrap_or_default()).collect();
            cluster_distinct(rationer, &canons, threshold)
                .into_iter()
                .filter(|(c, _)| c.len() >= min_size)
                .map(|(c, min_sim)| {
                    let loc = c.iter().map(|&k| defs[idxs[k]].loc).max().unwrap_or(0);
                    let args = c.iter().map(|&k| defs[idxs[k]].args).max().unwrap_or(0);
                    Finding {
                        pass: "name",
                        kind,
                        name: (*name).to_owned(),
                        severity: if min_sim >= error { Severity::Error } else { Severity::Warning },
                        min_sim: Some(min_sim),
                        loc,
                        args,
                        thickness: cluster_thickness(defs, c.iter().map(|&k| idxs[k]), min_sim, || {
                            thickness(loc, args, c.len(), min_sim)
                        }),
                        snippet: defs[idxs[c[0]]].text_orig.clone(),
                        notes: Vec::new(),
                        members: c.iter().map(|&k| member(defs, idxs[k])).collect(),
                        pattern: None,
                        facets: facet_votes(defs, c.iter().map(|&k| idxs[k])),
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Pass 2 — cross-name: callables with identical alpha-renamed canonicals but
/// ≥2 distinct names across ≥2 files.
#[must_use]
pub fn pass_cross_name(defs: &[Def], min_size: usize) -> Vec<Finding> {
    // Bucketed by hash, then only the buckets that qualify are put in canonical order — the order
    // the ordered map produced, at the cost of sorting a few hundred strings instead of walking a
    // tree of long-string compares for every definition.
    let mut buckets: rustc_hash::FxHashMap<&str, Vec<usize>> = rustc_hash::FxHashMap::default();
    for (i, d) in defs.iter().enumerate() {
        if d.kind.fn_like {
            if let Some(a) = &d.analysis {
                buckets.entry(a.xname_canonical.as_str()).or_default().push(i);
            }
        }
    }
    // The distinct-name set is built ONCE per bucket and carried through to the output, where the
    // same set already spelled the `/`-joined `name` — the previous form rebuilt it per qualifying
    // bucket (once to test `≥2` names, once to join them).
    let mut qualifying: Vec<(&str, Vec<usize>, BTreeSet<&str>)> = buckets
        .into_iter()
        .filter_map(|(k, ps)| {
            if ps.len() < min_size {
                return None;
            }
            let names: BTreeSet<&str> = ps.iter().map(|&p| defs[p].name.as_str()).collect();
            (names.len() >= 2).then_some((k, ps, names))
        })
        .collect();
    qualifying.par_sort_unstable_by(|a, b| a.0.cmp(b.0));
    let mut out = Vec::new();
    for (_, ps, names) in qualifying {
        let size = defs[ps[0]].analysis.as_ref().map_or(0, |a| a.size);
        let kind = callable_kind(defs, &ps);
        let loc = ps.iter().map(|&p| defs[p].loc).max().unwrap_or(0);
        let args = ps.iter().map(|&p| defs[p].args).max().unwrap_or(0);
        out.push(Finding {
            pass: "cross-name",
            kind,
            name: names.iter().copied().collect::<Vec<_>>().join("/"),
            severity: if size >= SUBSTANCE_NODES { Severity::Error } else { Severity::Warning },
            min_sim: None,
            loc,
            args,
            // Cross-name is an exact match by construction, so agreement is total.
            thickness: cluster_thickness(defs, ps.iter().copied(), 1.0, || {
                thickness(loc, args, ps.len(), 1.0)
            }),
            snippet: defs[ps[0]].text_orig.clone(),
            notes: Vec::new(),
            members: ps.iter().map(|&p| member(defs, p)).collect(),
            pattern: None,
            facets: facet_votes(defs, ps.iter().copied()),
        });
    }
    out
}

/// What the Type-3 pass joins — every `fn_like` def with a canonical. Exposed so a perf bench can
/// snapshot just this (tiny, plain-string) slice instead of the whole `Vec<Def>`.
#[derive(Debug, Default)]
pub struct Type3Input {
    /// Per function, its units: each canonical line with its slots renumbered **within the line**.
    ///
    /// A frontend numbers slots in order of first appearance across the whole body, which is right
    /// for the exact pass and wrong for this one. A Type-3 clone is a copy with statements added,
    /// removed or changed (Roy, Cordy & Koschke, 2009), and the pass compares bags of lines to
    /// tolerate exactly that — but with body-wide numbering one added parameter shifts every later
    /// `_v{n}`, so a single insertion rewrote every line and two functions agreeing on nine lines
    /// out of nine shared none. A unit's canonical form has to depend on the unit alone.
    ///
    /// Each type reference the frontend reports apart ([`dup_defs_core::Analysis::type_refs`]) follows
    /// as a unit of its own, `: Name` — a shared type weighs as shared vocabulary, a different one
    /// only its own weight.
    ///
    /// There is no floor on how many lines a function needs to take part. There was one — three —
    /// because a cosine over two lines says nothing; the null model says what two shared lines are
    /// worth given how rare they are, and a two-line function sharing two rare lines is a finding.
    ///
    pub line_lists: Vec<Vec<String>>,
    /// Per function, the sub-expressions of its lines ([`dup_defs_core::Analysis::type3_units`]),
    /// each with the index of its line in `line_lists`, renumbered within themselves like lines. A
    /// statement is one unit and a token changed inside it costs the whole statement; its
    /// sub-expressions are the parts the change did not reach, and two lines agree in part through
    /// them — see [`type3`] for how they are read.
    pub part_lists: Vec<Vec<(u32, String)>>,
    /// Per function, its name.
    pub names: Vec<String>,
    /// Per function, its cross-name bucket: equal ids ⇔ equal alpha-renamed canonicals, the pairs
    /// the exact pass reports and this one leaves to it.
    pub owners: Vec<u32>,
}

/// [`Type3Input`] plus, per function, its index in `defs`.
fn type3_rows(defs: &[Def]) -> (Type3Input, Vec<usize>) {
    let rows: Vec<(usize, &dup_defs_core::Analysis)> = defs
        .iter()
        .enumerate()
        .filter_map(|(i, d)| d.analysis.as_ref().filter(|a| d.kind.fn_like && !a.type3_lines.is_empty()).map(|a| (i, a)))
        .collect();
    let line_lists: Vec<Vec<String>> = rows
        .par_iter()
        .with_min_len(128)
        .map(|(_, a)| {
            let statements = a.type3_lines.iter().map(|line| converge::line_slots_local(line, a.canon_dialect));
            statements.chain(a.type_refs.iter().map(|name| format!(": {name}"))).collect()
        })
        .collect();
    let part_lists: Vec<Vec<(u32, String)>> = rows
        .par_iter()
        .with_min_len(128)
        .map(|(_, a)| {
            a.type3_units.iter().map(|(line, unit)| (*line, converge::line_slots_local(unit, a.canon_dialect))).collect()
        })
        .collect();
    let mut bucket: rustc_hash::FxHashMap<&str, u32> = rustc_hash::FxHashMap::default();
    let owners: Vec<u32> = rows
        .iter()
        .map(|(_, a)| {
            let next = u32::try_from(bucket.len()).unwrap_or(u32::MAX);
            *bucket.entry(a.xname_canonical.as_str()).or_insert(next)
        })
        .collect();
    let names: Vec<String> = rows.iter().map(|&(i, _)| defs[i].name.clone()).collect();
    let def_of: Vec<usize> = rows.iter().map(|&(i, _)| i).collect();
    (Type3Input { line_lists, part_lists, names, owners }, def_of)
}

/// The Type-3 pass's input over `defs` — see [`Type3Input`].
#[must_use]
pub fn type3_inputs(defs: &[Def]) -> Type3Input {
    type3_rows(defs).0
}

/// Pass 3 — Type-3 (`ECScan`): renamed near-copy callables — a soft cosine over name-agnostic lines.
/// `theta` is the join's threshold and the pass's; `gpu` selects the simjoin backend for the
/// all-pairs join (`settings:gpu=on` → GPU hybrid).
#[must_use]
pub fn pass_type3(defs: &[Def], theta: f64, gpu: GpuMode) -> Vec<Finding> {
    let (input, def_of) = type3_rows(defs);
    if input.names.len() < 2 {
        return Vec::new();
    }
    let line_lists: Vec<&[String]> = input.line_lists.iter().map(Vec::as_slice).collect();
    let part_lists: Vec<&[(u32, String)]> = input.part_lists.iter().map(Vec::as_slice).collect();
    let prepared = timed("  type3-prep", || type3::Prepared::new(&line_lists, &part_lists));
    let (input, def_of, prepared) = (&input, &def_of, &prepared);
    let names: Vec<&str> = input.names.iter().map(String::as_str).collect();
    type3::type3_clusters(prepared, &names, &input.owners, theta, gpu.to_concurrency())
        .into_iter()
        .filter_map(|cluster| {
            let distinct: BTreeSet<&str> = cluster.members.iter().map(|&c| names[c]).collect();
            if distinct.len() < 2 {
                return None;
            }
            let min_sim = cluster.min_sim;
            let members: Vec<usize> = cluster.members.iter().map(|&c| def_of[c]).collect();
            let kind = callable_kind(defs, &members);
            let loc = members.iter().map(|&i| defs[i].loc).max().unwrap_or(0);
            let args = members.iter().map(|&i| defs[i].args).max().unwrap_or(0);
            Some(Finding {
                pass: "type-3",
                kind,
                name: distinct.iter().copied().collect::<Vec<_>>().join("/"),
                severity: if min_sim >= TYPE3_ERROR_THETA { Severity::Error } else { Severity::Warning },
                min_sim: Some(min_sim),
                loc,
                args,
                thickness: cluster_thickness(defs, members.iter().copied(), min_sim, || {
                    thickness(loc, args, members.len(), min_sim)
                }),
                snippet: defs[members[0]].text_orig.clone(),
                notes: vec![format!("evidence: p = {:.1e} under the bipartite configuration model", cluster.worst_p)],
                members: members.iter().map(|&i| member(defs, i)).collect(),
                pattern: None,
                facets: facet_votes(defs, members.iter().copied()),
            })
        })
        .collect()
}

/// Pass 4 — **patternology / helper candidates**: collapsible structural duplication. Re-featurizes
/// every `fn_like` def's alpha-renamed canonical into AST node-type q-grams (see [`patternology`]),
/// clusters mutually-similar shapes, and keeps only the families whose anti-unification template is
/// *extractable into one parameterized helper* — the holes are bindable expression-parameters, not
/// leaky statement-divergences. Each surviving family is a [`patternology::HelperCandidate`] carrying
/// the proposed helper body, a stable cross-package signature, and the call sites it would replace.
/// Advisory only — never ERROR — this surfaces DRY violations to refactor, it is not a gate.
///
/// `theta` is the structural cosine floor; `support_min` the minimum distinct functions a sub-block
/// motif must recur in; `gpu` selects the simjoin backend.
/// One patternology dialect group: (dialect impl, short tag for timing, that group's canonicals,
/// group-local cluster index → global `defs` index map).
type PatGroup<'a> = (&'a dyn patternology::Dialect, &'a str, &'a [String], &'a [usize]);

#[must_use]
pub fn pass_patternology(defs: &[Def], theta: f64, support_min: usize, gpu: GpuMode) -> Vec<Finding> {
    // Patternology's helper-extractor is dialect-specific (the slot tables + pseudo-source renderer
    // are shaped per frontend). Partition the fn-like defs by `CanonDialect`, route each to its
    // `Dialect` impl, and run the engine once per group — never anti-unifying across languages. A
    // dialect the engine has no impl for is skipped rather than mis-walked.
    let dialects: [(&dyn patternology::Dialect, &str); 7] = [
        (&patternology::PyDialect, "py"),
        (&patternology::RustDialect, "rs"),
        (&patternology::TsDialect, "ts"),
        (&patternology::GoDialect, "go"),
        (&patternology::CDialect, "c"),
        (&patternology::PhpDialect, "php"),
        (&patternology::JavaDialect, "java"),
    ];
    // Per dialect above: its canonicals, and the group-local index → global `defs` index map.
    let mut buckets: [(Vec<String>, Vec<usize>); 7] = Default::default();
    for (i, d) in defs.iter().enumerate() {
        if !d.kind.fn_like {
            continue;
        }
        let Some(a) = &d.analysis else { continue };
        let slot = match a.canon_dialect {
            CanonDialect::CPythonAst => 0,
            CanonDialect::Rust => 1,
            CanonDialect::Other => 2, // the TypeScript frontend
            CanonDialect::Go => 3,
            CanonDialect::C => 4,
            CanonDialect::Php => 5,
            CanonDialect::Java => 6,
            // `CanonDialect` is `#[non_exhaustive]`: a future dialect without an engine impl is
            // skipped rather than mis-walked.
            _ => continue,
        };
        let (canons, def_of) = &mut buckets[slot];
        if patternology::node_type_seq(&a.xname_canonical).len() >= patternology::MIN_SKELETON_NODES {
            canons.push(a.xname_canonical.clone());
            def_of.push(i);
        }
    }
    let cfg = patternology::ExtractCfg::default();

    // One helper-candidate → one Finding, granularity-aware. Whole-function candidates size the LOC
    // saved from the members' own length; sub-block candidates from the motif's statement count (the
    // helper body), since only that slice — not the whole host function — collapses. `def_of` maps a
    // group-local cluster index back to the global `defs` index.
    let to_finding = |cand: patternology::HelperCandidate, def_of: &[usize]| {
        let members: Vec<usize> = cand.members.iter().map(|&c| def_of[c]).collect();
        let distinct_names: BTreeSet<&str> = members.iter().map(|&i| defs[i].name.as_str()).collect();
        let kind = callable_kind(defs, &members);
        let args = members.iter().map(|&i| defs[i].args).max().unwrap_or(0);
        let sub_block = cand.granularity == "sub-block";
        // Per-site LOC the collapse removes: a sub-block motif is its statement count (`;`-joined
        // body); a whole-function helper is the function's length.
        let per_site = if sub_block {
            cand.body.matches(';').count() + 1
        } else {
            members.iter().map(|&i| defs[i].loc).max().unwrap_or(0)
        };
        // A collapse keeps one helper + `support` one-line call sites, removing `support` copies of
        // `per_site` lines: saved ≈ (support − 1)·per_site. A conservative, readable proxy.
        let loc_saved = cand.support.saturating_sub(1) * per_site;
        let notes = vec![
            format!("{}helper: {} ({} param{})", if sub_block { "sub-block " } else { "" }, cand.body, cand.params, if cand.params == 1 { "" } else { "s" }),
            format!("collapses {} sites, ~{loc_saved} loc saved", cand.support),
            format!("sig: {}", cand.signature),
        ];
        Finding {
            pass: "pattern",
            kind,
            name: distinct_names.iter().copied().collect::<Vec<_>>().join("/"),
            // Advisory: structural similarity is FP-noisier, so patternology never escalates to
            // ERROR. A very tight family is WARNING; the rest is INFO.
            severity: if cand.min_sim >= PATTERN_WARNING_THETA { Severity::Warning } else { Severity::Info },
            min_sim: Some(cand.min_sim),
            loc: per_site,
            args,
            thickness: thickness(per_site, args, members.len(), cand.min_sim),
            snippet: defs[members[0]].text_orig.clone(),
            notes,
            members: members.iter().map(|&i| member(defs, i)).collect(),
            facets: Vec::new(),
            pattern: Some(PatternInfo {
                template: cand.body.clone(),
                signature: cand.signature.clone(),
                params: cand.params,
                granularity: cand.granularity,
                support: cand.support,
                loc_saved,
            }),
        }
    };

    // Run the engine once per dialect group and merge. The whole-vs-sub subset filter is per group
    // (cluster indices are group-local), applied before mapping through that group's `def_of`.
    let groups: Vec<PatGroup> =
        dialects.iter().zip(&buckets).map(|(&(d, tag), (canons, def_of))| (d, tag, canons.as_slice(), def_of.as_slice())).collect();
    let mut findings: Vec<Finding> = Vec::new();
    for (dialect, tag, canons, def_of) in groups {
        if canons.len() < 2 {
            continue;
        }
        if std::env::var_os("FDD_TIMING").is_some() {
            eprintln!("[timing]   pat:N-{tag:<9} {:>6}", canons.len());
        }
        let whole = timed("  pat:whole-fn", || {
            patternology::whole_fn_helpers(dialect, canons, theta, &cfg, gpu.to_concurrency())
        });
        // A sub-block motif whose host functions are all already a whole-function family is redundant
        // — the whole-function helper subsumes it. Keep only sub-blocks spanning functions the
        // whole-fn pass did NOT collapse (the embedded-idiom case the sub-block miner exists for).
        let whole_sets: Vec<BTreeSet<usize>> =
            whole.iter().map(|c| c.members.iter().copied().collect()).collect();
        let sub: Vec<_> = timed("  pat:subblock", || {
            patternology::subblock_helpers(dialect, canons, support_min, &cfg)
        })
        .into_iter()
        .filter(|c| {
            let s: BTreeSet<usize> = c.members.iter().copied().collect();
            !whole_sets.iter().any(|w| s.is_subset(w))
        })
        .collect();
        findings.extend(whole.into_iter().chain(sub).map(|c| to_finding(c, def_of)));
    }
    findings
}

// ── Backend selection ──────────────────────────────────────────────────────

/// Backend for the name-gated Ratcliff–Obershelp clustering ([`pass_name_gated`]).
///
/// `Cpu` (default) is the historical path. `Gpu` / `GpuPlusCpu` ask `difflib-fast` to offload the
/// large same-name groups to its Metal backend — but only when this crate is built with
/// `--features gpu` *and* running on macOS with a usable Metal device. Without those, the
/// [`difflib_fast::Rationer`] transparently degrades to CPU with byte-identical output, so the mode
/// is always safe to request. GPU only engages where it measured a net win (a single group past
/// `difflib-fast`'s size cutoff, all-ASCII); smaller groups stay on CPU regardless.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuMode {
    /// Pure CPU. The only mode with any effect on a non-`gpu` build.
    #[default]
    Cpu,
    /// GPU-only (`settings:gpu=only`): take the pure-GPU path where one exists (Type-3's f32 cosine
    /// join), short-circuiting to the GPU+CPU hybrid on paths that have none (`difflib-fast` routes
    /// those identically). Fastest; the join's f32 score differs from exact f64 by ≤1 pair in millions.
    Gpu,
    /// GPU+CPU hybrid — the recommended GPU mode, what `settings:gpu=on` maps to. Exact (byte-identical
    /// to CPU): the GPU filters and the CPU re-scores, with rayon-parallel overlap on the output.
    #[serde(rename = "gpu+cpu")]
    GpuPlusCpu,
}

impl GpuMode {
    /// Map to `difflib-fast`'s backend selector.
    fn to_concurrency(self) -> difflib_fast::Concurrency {
        match self {
            GpuMode::Cpu => difflib_fast::Concurrency::Cpu,
            GpuMode::Gpu => difflib_fast::Concurrency::Gpu,
            GpuMode::GpuPlusCpu => difflib_fast::Concurrency::GpuPlusCpu,
        }
    }
}

/// Build the shared clustering handle once per run. A [`difflib_fast::Rationer`] owns the long-lived
/// backend resources (Metal device + power-boost assertion under a GPU mode; nothing under `Cpu`)
/// and is reused across every per-group `cluster_canonicals` call instead of rebuilding per call.
/// `Concurrency::Cpu` acquires no Metal device, so the default mode keeps the historical zero
/// startup cost.
#[must_use]
fn build_rationer(mode: GpuMode) -> difflib_fast::Rationer {
    difflib_fast::Rationer::builder().concurrency(mode.to_concurrency()).build()
}

// ── Pipeline orchestration ────────────────────────────────────────────────

/// All the knobs the pipeline takes. Defaults via [`PipelineOpts::with_paths`].
#[derive(Clone, Debug, Serialize)]
pub struct PipelineOpts {
    pub paths: Vec<PathBuf>,
    /// Name-gated clustering floor (default `0.5`).
    pub threshold: f64,
    /// Name-gated ERROR floor (default `0.85`).
    pub error_threshold: f64,
    /// Type-3 soft-cosine detection floor (default `0.7`).
    pub type3_theta: f64,
    /// Run the patternology pass (structural meta-pattern families). Opt-in (default `false`): it is
    /// informational/codometry, not part of the duplicate gate.
    pub patternology: bool,
    /// Patternology structural-cosine detection floor (default `0.85`). Clique-grouping (not
    /// single-linkage) means a moderate floor no longer risks a chained mega-blob, so this favors
    /// recall while the clique requirement keeps each family tight (min-sim ≥ this).
    pub pattern_theta: f64,
    /// Minimum distinct functions a **sub-block** motif must recur in to be surfaced (default `3`).
    /// The codometry support floor — below it a "recurring idiom" is just a coincidence.
    pub pattern_support: usize,
    /// Drop patternology candidates whose `thickness` is below this (default `0.0` = off, nothing
    /// implicit). The directive-driven calibration knob: `--calibrate` proposes a value and the user
    /// applies it explicitly via `-D settings:pattern-min-thickness=…`. Only patternology (`pass ==
    /// "pattern"`) findings are affected; the duplicate gate is untouched.
    pub pattern_min_thickness: f64,
    /// Run the converge pass (divergence, by two anchors). Opt-in (default `false`) and always
    /// advisory: its output is a ranked list with no threshold, which is not a thing to gate on.
    pub converge: bool,
    /// How many converge findings of each kind to report, strongest first; `0` reports all of them.
    /// See [`converge::pass_converge`] for why this pass is capped when no other is.
    pub converge_top: usize,
    /// How many places a shared statement (or a shared subject) may occur in before converge treats
    /// it as an idiom rather than a coincidence — `settings:converge-cap=N`.
    ///
    /// The pass's cost knob as well as its meaning knob: the work it admits is quadratic in this, so
    /// dropping it from the default 60 to 20 took converge from 5.2 s to 1.5 s on a 371k-definition
    /// tree. It reports a shorter ranking in exchange, which is why it is a knob and not a change.
    pub converge_cap: usize,
    /// Run the decisions pass: definitions that decide on the same facts, however they are written.
    /// Opt-in (default `false`) and advisory. Its only cut is learned from the corpus — see
    /// [`decide::pass_decisions`].
    pub decisions: bool,
    /// The decisions pass's false-discovery rate and statistic (`settings:decision-fdr`,
    /// `settings:decision-statistic`).
    pub decision: decide::DecisionOpts,
    /// Minimum cluster size (default `2`).
    pub min_size: usize,
    /// De-escalate ERRORs whose `thickness` is below this to WARNING (default
    /// `0.0` = off).
    pub error_thickness: f64,
    /// De-escalate WARNINGs whose `thickness` is below this to INFO (default
    /// `0.0` = off).
    pub warning_thickness: f64,
    /// Escalate non-ERROR clusters whose `thickness` ≥ this to ERROR. Applied
    /// after the de-escalation knobs (default `0.0` = off).
    pub escalate_thickness: f64,
    /// Restrict to a specific subset of definition kinds, by `KindSpec::id`. `None` = all.
    pub kinds: Option<Vec<String>>,
    /// Skip the cross-name pass (default `false`).
    pub no_cross_name: bool,
    /// Skip the Type-3 pass (default `false`).
    pub no_type3: bool,
    /// Skip name-gated clustering for `(kind, name)` groups larger than this. `None` (default) =
    /// no cap, behavior unchanged. See [`pass_name_gated`].
    pub max_name_group: Option<usize>,
    /// Backend for the name-gated clustering (default [`GpuMode::Cpu`]). See [`GpuMode`].
    pub gpu: GpuMode,
}

impl PipelineOpts {
    /// Construct with reasonable defaults and the given source paths.
    #[must_use]
    pub fn with_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            threshold: 0.5,
            error_threshold: 0.85,
            type3_theta: 0.7,
            patternology: false,
            converge: false,
            converge_top: DEFAULT_CONVERGE_TOP,
            converge_cap: converge::SEED_CAP,
            decisions: false,
            decision: decide::DecisionOpts::default(),
            pattern_theta: 0.85,
            pattern_support: 3,
            pattern_min_thickness: 0.0,
            min_size: 2,
            error_thickness: 0.0,
            warning_thickness: 0.0,
            escalate_thickness: 0.0,
            kinds: None,
            no_cross_name: false,
            no_type3: false,
            max_name_group: None,
            gpu: GpuMode::Cpu,
        }
    }
}

/// Run the whole detection pipeline end-to-end:
///
/// 1. Walk `paths` once and scan every file via the matching `frontend` (single parse per file;
///    canon precomputed inside `scan`).
/// 2. Run name-gated, cross-name (unless `no_cross_name`), and Type-3 (unless
///    `no_type3`) passes over the resulting `Def`s.
/// 3. Apply `error_thickness` / `warning_thickness` demotion and
///    `escalate_thickness` escalation, if any is non-zero.
///
/// Whether the pipeline will read per-definition facets under `opts` — the converge and decisions
/// passes consume the statement stream and the decide facts, and the lenses consume the reached
/// paths. A default run reads none of them, so the scan may skip their computation.
#[must_use]
pub fn facets_needed(opts: &PipelineOpts) -> bool {
    opts.converge
        || opts.decisions
        || opts.kinds.as_ref().is_some_and(|ks| ks.iter().any(|k| k == "lenses"))
}

/// Returns the unsorted findings. The caller sorts (typically by
/// [`section_index`] + name + first member) and renders.
#[must_use]
pub fn scan_and_cluster(opts: &PipelineOpts, frontends: &[&dyn Frontend]) -> Vec<Finding> {
    let scan_opts = ScanOpts { kinds: opts.kinds.as_deref(), facets: facets_needed(opts) };
    cluster(collect_defs(frontends, &opts.paths, &scan_opts), opts)
}

/// Group definitions by `(kind, name)` and return the groups with at least `min_members`,
/// sorted by descending size. A name shared by very many definitions is a convention or entry
/// point (`fn main`, `async_setup_entry`) rather than a refactor cluster — this is the cheap
/// (O(n)) signal the directive-inferrer uses to suggest a `settings:max-name-group` cap, and it
/// is independent of clustering, so it's reported even when the cap skips those groups.
#[must_use]
pub fn large_name_groups(defs: &[Def], min_members: usize) -> Vec<(&'static KindSpec, String, usize)> {
    // Counted by hash; the result is sorted below, so the map's order never shows.
    let mut counts: rustc_hash::FxHashMap<(&str, &str), (&'static KindSpec, usize)> = rustc_hash::FxHashMap::default();
    for d in defs {
        let entry = counts.entry((d.kind.id, d.name.as_str())).or_insert((d.kind, 0));
        entry.1 += 1;
    }
    let mut out: Vec<(&'static KindSpec, String, usize)> = counts
        .into_iter()
        .filter(|(_, (_, n))| *n >= min_members)
        .map(|((_, name), (kind, n))| (kind, name.to_owned(), n))
        .collect();
    out.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.1.cmp(&b.1)));
    out
}

/// Cluster a pre-collected `Vec<Def>` (the three passes + thickness demotion/escalation). Split
/// out of [`scan_and_cluster`] so a caller (the CLI's `--calibrate`) can also derive
/// [`large_name_groups`] from the same single scan without re-walking the tree.
#[must_use]
pub fn cluster(mut defs: Vec<Def>, opts: &PipelineOpts) -> Vec<Finding> {
    if let Some(kinds) = &opts.kinds {
        defs.retain(|d| kinds.iter().any(|k| k == d.kind.id));
    }
    timed("sort", || {
        defs.sort_by(|a, b| {
            (a.file.as_ref(), a.line, a.col).cmp(&(b.file.as_ref(), b.line, b.col))
        });
    });

    // One shared clustering handle for the whole run — built once (acquiring the Metal device only
    // under a GPU mode), reused across every per-group `cluster_canonicals` call below.
    let rationer = build_rationer(opts.gpu);
    let mut findings = timed("pass1-name", || {
        pass_name_gated(
            &defs,
            opts.threshold,
            opts.error_threshold,
            opts.min_size,
            opts.max_name_group,
            &rationer,
        )
    });
    if !opts.no_cross_name {
        timed("pass2-xname", || findings.extend(pass_cross_name(&defs, opts.min_size)));
    }
    if !opts.no_type3 {
        timed("pass3-type3", || findings.extend(pass_type3(&defs, opts.type3_theta, opts.gpu)));
    }
    if opts.patternology {
        timed("pass4-pattern", || findings.extend(pass_patternology(&defs, opts.pattern_theta, opts.pattern_support, opts.gpu)));
    }
    if opts.converge {
        timed("pass5-converge", || findings.extend(converge::pass_converge(&defs, opts.converge_top, opts.converge_cap)));
    }
    if opts.decisions {
        // After the body passes, which it reads: a pair they already cluster is a copy, theirs to report.
        let found = timed("pass6-decisions", || decide::pass_decisions(&defs, &findings, &opts.decision));
        findings.extend(found);
    }

    // Directive-driven patternology calibration: drop advisory candidates below the explicit
    // thickness floor (`set:pattern-min-thickness`, proposed by `--calibrate`). Default `0.0` → no-op,
    // nothing implicit. Touches only `pass == "pattern"`; the duplicate gate is untouched.
    if opts.pattern_min_thickness > 0.0 {
        findings.retain(|f| f.pass != "pattern" || f.thickness >= opts.pattern_min_thickness);
    }

    if opts.error_thickness > 0.0 {
        for f in &mut findings {
            if f.severity == Severity::Error && f.thickness < opts.error_thickness {
                f.severity = Severity::Warning;
            }
        }
    }
    if opts.warning_thickness > 0.0 {
        for f in &mut findings {
            if f.severity == Severity::Warning && f.thickness < opts.warning_thickness {
                f.severity = Severity::Info;
            }
        }
    }
    if opts.escalate_thickness > 0.0 {
        for f in &mut findings {
            if f.severity != Severity::Error && f.thickness >= opts.escalate_thickness {
                f.severity = Severity::Error;
            }
        }
    }
    findings
}

#[cfg(test)]
mod type3_tests {
    use std::sync::Arc;

    use dup_defs_core::{Analysis, CanonDialect, Def, Facets};

    use super::{pass_type3, GpuMode};

    fn def(name: &str, src: &str) -> Def {
        let a = py_canon::analyze_functions(&[src.to_owned()]).pop().flatten().expect("a function definition");
        Def {
            lang: "py",
            kind: &py_canon::FUNCTIONS,
            name: name.to_owned(),
            file: Arc::from(format!("{name}.py")),
            line: 1,
            col: 0,
            loc: src.lines().count(),
            args: 1,
            text_orig: src.to_owned(),
            cluster_canonical: Some(a.cluster_canonical),
            analysis: Some(Analysis {
                xname_canonical: a.xname_canonical,
                type3_lines: a.type3_lines,
                type_refs: a.type_refs,
                type3_units: a.type3_units,
                size: a.size,
                canon_dialect: CanonDialect::CPythonAst,
            }),
            thickness: None,
            facets: Facets::default(),
        }
    }

    /// Unrelated bodies, so the weighting has a corpus to call a line common or rare against — and
    /// enough of them, with enough of their own, that two functions sharing twenty units is more
    /// than the null expects. In a corpus of eight, two functions holding a third of every unit
    /// there is would share most of them by chance, and the null says so.
    fn corpus(defs: &mut Vec<Def>) {
        for i in 0..40 {
            let src = format!(
                "def other{i}(rows, limit):\n    kept = [r for r in rows if r.rank > {i}]\n    \
                 total = measure_{i}(kept, limit)\n    if total > {i}:\n        kept.sort(key=lambda r: r.field_{i})\n    \
                 return kept[:limit], total\n"
            );
            defs.push(def(&format!("other{i}"), &src));
        }
    }

    fn pairs(defs: &[Def]) -> Vec<String> {
        pass_type3(defs, 0.7, GpuMode::Cpu).into_iter().map(|f| f.name).collect()
    }

    #[test]
    fn one_added_parameter_does_not_rewrite_every_line() {
        // Body-wide numbering shifted every `_v{n}` after the extra parameter, so these two agreed on
        // nine statements and shared no unit at all.
        let body = "    raw = identity.payload\n    name = raw.get('username')\n    if isinstance(name, str):\n        \
                    out['username'] = name\n    first = raw.get('first_name')\n    if isinstance(first, str):\n        \
                    out['first_name'] = first\n    return out\n";
        let mut defs = vec![
            def("build_payload", &format!("def build_payload(identity, out):\n{body}")),
            def("build_context", &format!("def build_context(base, identity, out):\n{body}")),
        ];
        corpus(&mut defs);
        assert_eq!(pairs(&defs), ["build_context/build_payload"]);
    }

    #[test]
    fn a_pair_the_exact_pass_owns_is_left_to_it() {
        let mut defs = vec![
            def("from_jwt", "def from_jwt(p: Jwt) -> int | None:\n    v = p.get('uid')\n    if isinstance(v, int):\n        return v\n    return None\n"),
            def("from_refresh", "def from_refresh(p: Refresh) -> int | None:\n    v = p.get('uid')\n    if isinstance(v, int):\n        return v\n    return None\n"),
        ];
        corpus(&mut defs);
        assert_eq!(
            defs[0].analysis.as_ref().map(|a| &a.xname_canonical),
            defs[1].analysis.as_ref().map(|a| &a.xname_canonical),
            "the two canonicals agree"
        );
        assert!(pairs(&defs).is_empty(), "equal canonicals are the cross-name pass's finding");
    }

    #[test]
    fn equal_units_with_different_canonicals_are_still_reported() {
        // Line for line alike, yet one returns what it read first and the other what it read second:
        // the exact pass does not have this pair, so this pass may not hand it over.
        let mut defs = vec![
            def("first", "def first(a, b):\n    x = load(a)\n    y = load(b)\n    return x\n"),
            def("second", "def second(a, b):\n    x = load(a)\n    y = load(b)\n    return y\n"),
        ];
        corpus(&mut defs);
        assert_eq!(pairs(&defs), ["first/second"]);
    }

    #[test]
    fn a_token_changed_inside_a_long_line_costs_only_its_share() {
        // Four lines agree and the fifth — the long one, most of the function's weight — differs by
        // one keyword argument. As whole units the pair shares four lines in five with the heavy one
        // lost, and the plain cosine sits under the threshold; the soft cosine credits the long line
        // for the sub-expressions the edit did not touch, and the join finds the pair on them.
        let body = "    rows = load_rows(orders)\n    kept = [r for r in rows if r.ok]\n    total = sum(r.amount for r in kept)\n    label = describe(owner)\n";
        let long = "    return render(template=template_for(owner.locale), owner=owner, label=label, items=[describe(r) for r in kept], total=total, count=len(kept), currency=owner.currency.code, mode=";
        let left = format!("def summary_for_user(owner, orders):\n{body}{long}'full')\n");
        let right = format!("def summary_for_group(owner, orders):\n{body}{long}'brief')\n");
        let mut defs = vec![def("summary_for_user", &left), def("summary_for_group", &right)];
        corpus(&mut defs);
        assert_eq!(pairs(&defs), ["summary_for_group/summary_for_user"]);
    }

    #[test]
    fn a_shared_signature_without_a_shared_step_is_not_a_copy() {
        // The header and the types it names count toward the similarity, and here they are most of
        // it; but the one thing each function does, the other does not, and two definitions with
        // one interface are not one definition written twice.
        let mut defs = vec![
            def("rows_a", "def rows_a(self) -> list[tuple[Row, ...]]:\n    return self.rows\n"),
            def("rows_b", "def rows_b(self) -> list[tuple[Row, ...]]:\n    return self.items\n"),
        ];
        corpus(&mut defs);
        assert!(pairs(&defs).is_empty(), "one interface over two different steps is not one definition written twice");
    }
}

#[cfg(test)]
mod pipeline_unit_tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use dup_defs_core::{Analysis, CanonDialect, Def, Facets, KindSpec};

    use super::{
        build_rationer, callable_kind, candidates, cluster, cluster_distinct, is_excluded_dir,
        large_name_groups, one_per_content, pass_name_gated, pass_patternology, section_index,
        type3_inputs, Finding, GpuMode, PipelineOpts, Severity, PATTERN_SECTION_OFFSET,
    };

    fn method_kind() -> &'static KindSpec {
        &dup_defs_core::kinds::METHODS
    }

    fn function_kind() -> &'static KindSpec {
        &dup_defs_core::kinds::FUNCTIONS
    }

    fn use_profile_kind() -> &'static KindSpec {
        &dup_defs_core::kinds::USE_PROFILES
    }

    /// A body-kind `Def` with an alpha-renamed canonical, enough for the name-gated and thickness
    /// machinery to cluster it and compute a thickness.
    fn body_def(name: &str, kind: &'static KindSpec, canonical: &str, loc: usize, args: usize) -> Def {
        Def {
            lang: "test",
            kind,
            name: name.to_owned(),
            file: Arc::from(format!("{name}.py")),
            line: 0,
            col: 0,
            loc,
            args,
            text_orig: format!("def {name}(): pass"),
            cluster_canonical: Some(canonical.to_owned()),
            analysis: None,
            thickness: None,
            facets: Facets::default(),
        }
    }

    // ── is_excluded_dir ───────────────────────────────────────────────────

    #[test]
    fn excluded_dirs_are_the_blacklist_plus_egg_info() {
        assert!(is_excluded_dir("node_modules"), "blacklisted dir is excluded");
        assert!(is_excluded_dir(".git"), "blacklisted vcs dir is excluded");
        assert!(is_excluded_dir("pkg.egg-info"), "egg-info dirs are excluded by suffix");
        assert!(!is_excluded_dir("src"), "a plain source dir is not excluded");
        assert!(!is_excluded_dir("mynode_modules"), "a bare suffix match is not exclusion");
    }

    // ── Severity::from_index ──────────────────────────────────────────────

    #[test]
    fn severity_index_round_trips_and_clamps() {
        assert_eq!(Severity::from_index(0), Severity::Error);
        assert_eq!(Severity::from_index(1), Severity::Warning);
        assert_eq!(Severity::from_index(2), Severity::Info);
        assert_eq!(Severity::from_index(7), Severity::Info, "above the ladder clamps to INFO");
        assert_eq!(Severity::from_index(-4), Severity::Error, "below the ladder clamps to ERROR");
    }

    // ── section_index ─────────────────────────────────────────────────────

    #[test]
    fn section_index_offsets_each_pass() {
        let kind = function_kind();
        let mut f = Finding {
            pass: "name",
            kind,
            name: "n".to_owned(),
            severity: Severity::Warning,
            min_sim: Some(1.0),
            loc: 1,
            args: 0,
            thickness: 0.0,
            snippet: String::new(),
            notes: Vec::new(),
            members: Vec::new(),
            pattern: None,
            facets: Vec::new(),
        };
        assert_eq!(section_index(&f), 1, "name pass on a function has offset 0");
        f.pass = "pattern";
        assert_eq!(section_index(&f), 1 + PATTERN_SECTION_OFFSET, "patternology files as an appendix");
        f.pass = "decision-family";
        assert_eq!(
            section_index(&f),
            1 + crate::decide::DECISION_FAMILY_SECTION_OFFSET,
            "decision families file past every other pass"
        );
        f.pass = "name";
        f.kind = &dup_defs_core::kinds::CONSTANTS;
        assert_eq!(section_index(&f), 0, "a non-fn_like kind never takes a pass offset");
    }

    // ── callable_kind ─────────────────────────────────────────────────────

    #[test]
    fn callable_kind_labels_all_method_groups_as_method() {
        let defs = vec![body_def("m1", method_kind(), "a", 1, 0), body_def("m2", method_kind(), "a", 1, 0)];
        assert_eq!(callable_kind(&defs, &[0, 1]).id, "methods");
    }

    #[test]
    fn callable_kind_labels_a_mixed_group_as_function() {
        let defs = vec![body_def("m", method_kind(), "a", 1, 0), body_def("f", function_kind(), "a", 1, 0)];
        assert_eq!(callable_kind(&defs, &[0, 1]).id, "functions", "a method+function mix reports as a function");
    }

    #[test]
    fn callable_kind_does_not_take_the_first_member_as_a_function_fallback() {
        // A non-method, non-function kind first in the list: the function arm must still win over
        // the first member's own kind.
        let defs = vec![body_def("u", use_profile_kind(), "a", 1, 0), body_def("f", function_kind(), "a", 1, 0)];
        assert_eq!(callable_kind(&defs, &[0, 1]).id, "functions");
    }

    // ── candidates ────────────────────────────────────────────────────────

    #[test]
    fn candidates_admits_partners_under_both_bounds() {
        assert_eq!(candidates(&[], 0.5), Vec::<usize>::new(), "nothing has a partner");
        assert_eq!(candidates(&["a", "b"], 0.0), vec![0, 1], "a non-positive threshold admits every pair");
        assert_eq!(candidates(&["ab", "ab"], 0.5), vec![0, 1], "identical strings partner");
        assert_eq!(candidates(&["", ""], 0.5), vec![0, 1], "two empty strings share the trivial bound");
        assert_eq!(candidates(&["ab", "cd"], 0.9), Vec::<usize>::new(), "disjoint multisets never partner");
        assert_eq!(candidates(&["abc", "abd"], 0.6), vec![0, 1], "a shared prefix passes the multiset bound");
        assert_eq!(candidates(&["abc", "abd"], 0.9), Vec::<usize>::new(), "below the multiset bound there is no partner");
        assert_eq!(candidates(&["a", "abcdefgh"], 0.9), Vec::<usize>::new(), "the length bound rejects a lopsided pair");
        assert_eq!(candidates(&["ab", "ac", "xy"], 0.5), vec![0, 1], "only the shared character is a partner");
        assert_eq!(candidates(&["bc", "ac"], 0.5), vec![0, 1], "the greater arm of the multiset merge is walked");
    }

    // ── cluster_distinct ──────────────────────────────────────────────────

    fn cd(canons: &[&str], threshold: f64) -> Vec<(Vec<usize>, f64)> {
        cluster_distinct(&build_rationer(GpuMode::Cpu), canons, threshold)
    }

    #[test]
    fn cluster_distinct_handles_a_single_canonical() {
        assert_eq!(cd(&["foo", "foo"], 0.5), vec![(vec![0, 1], 1.0)]);
        assert_eq!(cd(&["foo"], 0.5), Vec::<(Vec<usize>, f64)>::new());
    }

    #[test]
    fn cluster_distinct_clusters_two_similar_canonicals() {
        let got = cd(&["foobar", "foobaz"], 0.5);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, vec![0, 1]);
    }

    #[test]
    fn cluster_distinct_clusters_three_similar_canonicals() {
        let got = cd(&["foobar", "foobaz", "foobat"], 0.5);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, vec![0, 1, 2]);
    }

    #[test]
    fn cluster_distinct_keeps_both_orientations_of_a_repeated_canonical() {
        // `aaba` occurs non-contiguously at positions 0 and 2, so its second representative must be
        // kept: the reported minimum is the lower of ratio("aaba","baaa") = 0.75 and the reverse
        // orientation 0.5.
        assert_eq!(cd(&["aaba", "baaa", "aaba"], 0.5), vec![(vec![0, 1, 2], 0.5)]);
    }

    // ── large_name_groups ─────────────────────────────────────────────────

    #[test]
    fn large_name_groups_counts_by_kind_and_name() {
        let defs = vec![
            body_def("f", function_kind(), "a", 1, 0),
            body_def("f", function_kind(), "b", 1, 0),
            body_def("f", function_kind(), "c", 1, 0),
            body_def("g", function_kind(), "a", 1, 0),
            body_def("m", method_kind(), "a", 1, 0),
        ];
        let groups = large_name_groups(&defs, 2);
        assert_eq!(groups.len(), 1, "only the tripled (functions, f) group reaches the floor");
        assert_eq!(groups[0].1, "f");
        assert_eq!(groups[0].2, 3);
    }

    // ── pass_name_gated ───────────────────────────────────────────────────

    #[test]
    fn max_name_group_drops_an_overlarge_group() {
        let defs = vec![
            body_def("f", function_kind(), "same", 2, 0),
            body_def("f", function_kind(), "same", 2, 0),
            body_def("f", function_kind(), "same", 2, 0),
        ];
        let rationer = build_rationer(GpuMode::Cpu);
        let capped = pass_name_gated(&defs, 0.5, 0.85, 2, Some(2), &rationer);
        assert!(capped.is_empty(), "a group above the cap is skipped");
        let exact = pass_name_gated(&defs, 0.5, 0.85, 2, Some(3), &rationer);
        assert_eq!(exact.len(), 1, "a group at the cap is clustered");
    }

    // ── type3_inputs (type3_rows) ─────────────────────────────────────────

    #[test]
    fn type3_inputs_excludes_fn_like_defs_without_lines() {
        let a = py_canon::analyze_functions(&["def f():\n    return 1\n".to_owned()])
            .pop()
            .flatten()
            .expect("a function");
        let mut empty = body_def("f", function_kind(), "x", 2, 0);
        empty.analysis = Some(Analysis {
            xname_canonical: a.xname_canonical.clone(),
            type3_lines: Vec::new(), // fn-like but no body lines
            type_refs: Vec::new(),
            type3_units: Vec::new(),
            size: a.size,
            canon_dialect: CanonDialect::CPythonAst,
        });
        let inputs = type3_inputs(&[empty]);
        assert!(inputs.names.is_empty(), "a fn-like def with no type-3 lines is not a row");
    }

    // ── cluster (thickness demotion / escalation) ─────────────────────────

    fn opts() -> PipelineOpts {
        PipelineOpts::with_paths(vec![PathBuf::from(".")])
    }

    fn name_finding(defs: &[Def], opts: &PipelineOpts) -> Vec<Finding> {
        cluster(defs.to_vec(), opts)
    }

    /// Two identical body defs form one name-gated ERROR cluster with a small (low-volume) thickness.
    fn duplicated_defs() -> Vec<Def> {
        vec![body_def("f", function_kind(), "canon", 2, 0), body_def("f", function_kind(), "canon", 2, 0)]
    }

    #[test]
    fn error_thickness_demotes_a_thin_error_to_warning() {
        let mut o = opts();
        o.error_thickness = 0.5;
        let findings = name_finding(&duplicated_defs(), &o);
        assert_eq!(findings.len(), 1);
        assert!(findings[0].thickness < 0.5, "the duplicate is a thin two-liner");
        assert_eq!(findings[0].severity, Severity::Warning, "a thin ERROR is demoted to WARNING");
    }

    #[test]
    fn error_thickness_leaves_a_thick_error_alone() {
        let mut o = opts();
        o.error_thickness = 0.1;
        let findings = name_finding(&duplicated_defs(), &o);
        assert_eq!(findings[0].severity, Severity::Error, "a duplicate above the floor stays ERROR");
    }

    #[test]
    fn warning_thickness_demotes_a_thin_warning_to_info() {
        // Near-copies: similar but not identical canonicals cluster as WARNING.
        let defs = vec![body_def("f", function_kind(), "foobar", 2, 0), body_def("f", function_kind(), "foobaz", 2, 0)];
        let mut o = opts();
        o.warning_thickness = 0.5;
        let findings = name_finding(&defs, &o);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Info, "a thin WARNING is demoted to INFO");
    }

    #[test]
    fn escalate_thickness_promotes_a_thick_warning_to_error() {
        let defs = vec![body_def("f", function_kind(), "foobar", 2, 0), body_def("f", function_kind(), "foobaz", 2, 0)];
        let mut o = opts();
        o.escalate_thickness = 0.01;
        let findings = name_finding(&defs, &o);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Error, "a thick non-ERROR is escalated to ERROR");
    }

    #[test]
    fn pattern_min_thickness_drops_only_thin_pattern_findings() {
        let a = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('AuthRefreshToken', Load), [Constant('x')], [])), Expr(Call(Attribute(Name('session', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)";
        let b = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('Clientspace', Load), [Constant('y')], [])), Expr(Call(Attribute(Name('session', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)";
        let defs = vec![
            pattern_def("create", a, CanonDialect::CPythonAst),
            pattern_def("create", b, CanonDialect::CPythonAst),
        ];
        let mut o = opts();
        o.patternology = true;
        o.pattern_min_thickness = 0.5;
        let findings = cluster(defs, &o);
        assert!(findings.iter().any(|f| f.pass == "name"), "the name-gated finding is not a pattern candidate and survives");
        assert!(findings.iter().all(|f| f.pass != "pattern"), "a thin pattern candidate is dropped below the floor");
    }

    // ── pass_patternology ─────────────────────────────────────────────────

    fn pattern_def(name: &str, canonical: &str, dialect: CanonDialect) -> Def {
        let mut d = body_def(name, function_kind(), "unused", 3, 0);
        d.analysis = Some(Analysis {
            xname_canonical: canonical.to_owned(),
            type3_lines: Vec::new(),
            type_refs: Vec::new(),
            type3_units: Vec::new(),
            size: 0,
            canon_dialect: dialect,
        });
        d
    }

    fn pattern_findings(defs: &[Def]) -> Vec<Finding> {
        pass_patternology(defs, 0.7, 2, GpuMode::Cpu)
    }

    #[test]
    fn patternology_emits_a_whole_function_helper() {
        let a = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('AuthRefreshToken', Load), [Constant('x')], [])), Expr(Call(Attribute(Name('session', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)";
        let b = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('Clientspace', Load), [Constant('y')], [])), Expr(Call(Attribute(Name('session', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)";
        let defs = vec![
            pattern_def("create_a", a, CanonDialect::CPythonAst),
            pattern_def("create_b", b, CanonDialect::CPythonAst),
        ];
        let findings = pattern_findings(&defs);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].pass, "pattern");
        // A tight family (two near-identical shapes) clears the WARNING theta.
        assert_eq!(findings[0].severity, Severity::Warning);
        let pat = findings[0].pattern.as_ref().expect("a pattern payload");
        assert_eq!(pat.granularity, "whole-fn");
        assert_eq!(pat.support, 2);
        assert_eq!(findings[0].loc, 3, "a whole-fn helper removes the member's own length");
        assert_eq!(pat.loc_saved, 3, "(support - 1) * per-site LOC");
    }

    #[test]
    fn patternology_files_a_three_way_family() {
        let a = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('A1', Load), [Constant('x')], [])), Expr(Call(Attribute(Name('session', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)";
        let b = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('A2', Load), [Constant('y')], [])), Expr(Call(Attribute(Name('session', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)";
        let c = "FunctionDef('_fn', arguments([], []), [Assign([Name('_v0', Store)], Call(Name('A3', Load), [Constant('z')], [])), Expr(Call(Attribute(Name('session', Load), 'add', Load), [Name('_v0', Load)], [])), Return(Name('_v0', Load))], [], None)";
        let defs = vec![
            pattern_def("a", a, CanonDialect::CPythonAst),
            pattern_def("b", b, CanonDialect::CPythonAst),
            pattern_def("c", c, CanonDialect::CPythonAst),
        ];
        let findings = pattern_findings(&defs);
        assert!(!findings.is_empty(), "three mutually-similar shapes form a family");
    }

    #[test]
    fn patternology_mines_a_subblock_motif_and_sizes_it_by_statement() {
        let fa = "FunctionDef('_fn', arguments([], []), [Expr(Call(Name('log'), [Constant('a')], [])), Assign([Name('_v0', Store)], Call(Attribute(Name('self', Load), 'get', Load), [Name('_v1', Load)], [])), If(Compare(Name('_v0', Load), [Is], [Constant(None)]), [Raise(Call(Name('KeyError', Load), [Name('_v1', Load)], []))], []), Return(Name('_v0', Load))], [], None)";
        let fb = "FunctionDef('_fn', arguments([], []), [AugAssign(Name('_v2', Store), Add, Constant(1)), Assign([Name('_v0', Store)], Call(Attribute(Name('self', Load), 'get', Load), [Name('_v1', Load)], [])), If(Compare(Name('_v0', Load), [Is], [Constant(None)]), [Raise(Call(Name('LookupError', Load), [Name('_v1', Load)], []))], []), Pass()], [], None)";
        let defs = vec![
            pattern_def("get_a", fa, CanonDialect::CPythonAst),
            pattern_def("get_b", fb, CanonDialect::CPythonAst),
        ];
        let findings = pattern_findings(&defs);
        assert_eq!(findings.len(), 1);
        let pat = findings[0].pattern.as_ref().expect("a pattern payload");
        assert_eq!(pat.granularity, "sub-block");
        assert_eq!(pat.support, 2);
        // per-site LOC = the motif's statement count (one `;`), not the host function's length.
        assert_eq!(findings[0].loc, 2);
        assert_eq!(pat.loc_saved, 2);
    }

    #[test]
    fn patternology_routes_the_typescript_dialect() {
        let make_user = "Func('_fn', Params(Param(Bind('_v0'))), Block(Var('const', [Decl(Bind('_v1'), New(Id('User'), [Id('_v0')]))]), Expr(Call(Member(Id('store'), 'add', ''), '', [Id('_v1')])), Return(Id('_v1'))), async=false gen=false)";
        let make_order = "Func('_fn', Params(Param(Bind('_v0'))), Block(Var('const', [Decl(Bind('_v1'), New(Id('Order'), [Id('_v0')]))]), Expr(Call(Member(Id('store'), 'add', ''), '', [Id('_v1')])), Return(Id('_v1'))), async=false gen=false)";
        let defs = vec![
            pattern_def("make_user", make_user, CanonDialect::Other),
            pattern_def("make_order", make_order, CanonDialect::Other),
        ];
        let findings = pattern_findings(&defs);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].pattern.as_ref().expect("payload").granularity, "whole-fn");
    }

    // ── one_per_content ───────────────────────────────────────────────────

    #[test]
    fn one_per_content_parses_identical_files_once() {
        let dir = std::env::temp_dir().join(format!("fdd-onepc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let a = dir.join("a.py");
        let b = dir.join("b.py");
        std::fs::write(&a, b"def f():\n    return 1\n").expect("write a");
        std::fs::write(&b, b"def f():\n    return 1\n").expect("write b");
        let files: Vec<Arc<str>> = vec![Arc::from(a.to_string_lossy().as_ref()), Arc::from(b.to_string_lossy().as_ref())];
        let (to_parse, twins) = one_per_content(&files);
        assert_eq!(to_parse.len(), 1, "one representative is parsed");
        assert_eq!(twins.len(), 1, "the copy is recorded as a twin");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
