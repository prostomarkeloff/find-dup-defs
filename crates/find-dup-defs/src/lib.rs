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

pub mod type3;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use dup_defs_core::{Def, Frontend, KindSpec};
use rayon::prelude::*;
use serde::Serialize;
use walkdir::WalkDir;

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

/// Cross-name: ERROR only when the alpha-renamed canonical has ≥ this many AST
/// nodes. Avoids escalating "two `return []` one-liners" to ERROR purely on a
/// renamed-exact match.
pub const SUBSTANCE_NODES: usize = 20;
/// Type-3 N-line shingle window.
pub const SHINGLE_LINES: usize = 3;
/// Type-3 IDF gate: drop a shingle present in > this ratio of functions.
pub const COMMON_RATIO: f64 = 0.007;
/// Type-3 cluster's min-cosine ≥ this → ERROR (else WARNING).
pub const TYPE3_ERROR_THETA: f64 = 0.9;

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
#[must_use]
pub fn collect_defs(frontends: &[&dyn Frontend], paths: &[PathBuf]) -> Vec<Def> {
    let per_frontend: Vec<BTreeSet<Arc<str>>> = timed("discovery", || {
        let mut per_frontend: Vec<BTreeSet<Arc<str>>> = vec![BTreeSet::new(); frontends.len()];
        let mut route = |path: &Path| {
            if let Some(fi) = frontend_for(frontends, path) {
                per_frontend[fi].insert(Arc::from(path.to_string_lossy().as_ref()));
            }
        };
        for p in paths {
            if p.is_dir() {
                let walker = WalkDir::new(p).into_iter().filter_entry(|e| {
                    if e.depth() == 0 {
                        return true;
                    }
                    if e.file_type().is_dir() {
                        return !is_excluded_dir(&e.file_name().to_string_lossy());
                    }
                    true
                });
                for entry in walker.filter_map(Result::ok) {
                    route(entry.path());
                }
            } else {
                route(p);
            }
        }
        per_frontend
    });
    // Recursive-descent parsers (syn especially) have no built-in nesting limit, and a deeply
    // nested file — rustc's parser-stress fixtures, generated code — can exhaust a worker
    // thread's default 2 MiB stack during parse or canonicalization. Run the scan on a pool with
    // a generous per-worker stack so realistically-deep input is handled; the few inputs nested
    // beyond this would also blow a compiler's own limits.
    let scan = move || {
        let mut defs = Vec::new();
        for (fi, files) in per_frontend.into_iter().enumerate() {
            let files: Vec<Arc<str>> = files.into_iter().collect();
            defs.extend(frontends[fi].scan(&files));
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
            _ => 0,
        }
    } else {
        0
    };
    base + offset
}

// ── Passes ─────────────────────────────────────────────────────────────────

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
                let canons: Vec<String> = idxs.iter().map(|&i| defs[i].text_orig.clone()).collect();
                let clusters = rationer.cluster_canonicals(&canons, 0.0);
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
                            thickness: thickness(loc, args, c.len(), min_sim),
                            snippet: defs[idxs[c[0]]].text_orig.clone(),
                            notes: Vec::new(),
                            members: c.iter().map(|&k| member(defs, idxs[k])).collect(),
                        }
                    })
                    .collect::<Vec<_>>();
            }
            let canons: Vec<String> =
                idxs.iter().map(|&i| defs[i].cluster_canonical.clone().unwrap_or_default()).collect();
            rationer
                .cluster_canonicals(&canons, threshold)
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
                        thickness: thickness(loc, args, c.len(), min_sim),
                        snippet: defs[idxs[c[0]]].text_orig.clone(),
                        notes: Vec::new(),
                        members: c.iter().map(|&k| member(defs, idxs[k])).collect(),
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
    let mut buckets: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, d) in defs.iter().enumerate() {
        if d.kind.fn_like {
            if let Some(a) = &d.analysis {
                buckets.entry(a.xname_canonical.as_str()).or_default().push(i);
            }
        }
    }
    let mut out = Vec::new();
    for (_, ps) in buckets {
        if ps.len() < min_size {
            continue;
        }
        let names: BTreeSet<&str> = ps.iter().map(|&p| defs[p].name.as_str()).collect();
        if names.len() < 2 {
            continue;
        }
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
            thickness: thickness(loc, args, ps.len(), 1.0),
            snippet: defs[ps[0]].text_orig.clone(),
            notes: Vec::new(),
            members: ps.iter().map(|&p| member(defs, p)).collect(),
        });
    }
    out
}

/// The `(line_lists, names)` inputs the Type-3 pass feeds to [`type3::type3_clusters`] — the
/// `fn_like` defs with ≥ `SHINGLE_LINES` lines. Exposed so a perf bench can snapshot just this
/// (tiny, plain-string) slice instead of the whole `Vec<Def>`.
#[must_use]
pub fn type3_inputs(defs: &[Def]) -> (Vec<Vec<String>>, Vec<String>) {
    let (mut line_lists, mut names) = (Vec::new(), Vec::new());
    for d in defs {
        if d.kind.fn_like {
            if let Some(a) = &d.analysis {
                if a.type3_lines.len() >= SHINGLE_LINES {
                    line_lists.push(a.type3_lines.clone());
                    names.push(d.name.clone());
                }
            }
        }
    }
    (line_lists, names)
}

/// Pass 3 — Type-3 (`ECScan`): renamed near-copy callables via IDF-weighted
/// cosine over name-agnostic lines.
#[must_use]
pub fn pass_type3(defs: &[Def], theta: f64) -> Vec<Finding> {
    let (mut line_lists, mut names, mut def_of) = (Vec::new(), Vec::new(), Vec::new());
    for (i, d) in defs.iter().enumerate() {
        if d.kind.fn_like {
            if let Some(a) = &d.analysis {
                if a.type3_lines.len() >= SHINGLE_LINES {
                    line_lists.push(a.type3_lines.clone());
                    names.push(d.name.clone());
                    def_of.push(i);
                }
            }
        }
    }
    if names.len() < 2 {
        return Vec::new();
    }
    type3::type3_clusters(&line_lists, &names, theta, SHINGLE_LINES, COMMON_RATIO)
        .into_iter()
        .filter_map(|(cluster, min_sim)| {
            let distinct: BTreeSet<&str> = cluster.iter().map(|&c| names[c].as_str()).collect();
            if distinct.len() < 2 {
                return None;
            }
            let members: Vec<usize> = cluster.iter().map(|&c| def_of[c]).collect();
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
                thickness: thickness(loc, args, cluster.len(), min_sim),
                snippet: defs[def_of[cluster[0]]].text_orig.clone(),
                notes: Vec::new(),
                members: members.iter().map(|&i| member(defs, i)).collect(),
            })
        })
        .collect()
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
    /// Allow the Metal GPU on the paths where it wins; everything else stays on CPU.
    Gpu,
    /// Like [`GpuMode::Gpu`] plus rayon-parallel CPU overlap on the GPU output. `difflib-fast`'s
    /// own recommended default; the value `settings:gpu=on` maps to.
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
    /// Type-3 cosine detection floor (default `0.7`).
    pub type3_theta: f64,
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
/// Returns the unsorted findings. The caller sorts (typically by
/// [`section_index`] + name + first member) and renders.
#[must_use]
pub fn scan_and_cluster(opts: &PipelineOpts, frontends: &[&dyn Frontend]) -> Vec<Finding> {
    cluster(collect_defs(frontends, &opts.paths), opts)
}

/// Group definitions by `(kind, name)` and return the groups with at least `min_members`,
/// sorted by descending size. A name shared by very many definitions is a convention or entry
/// point (`fn main`, `async_setup_entry`) rather than a refactor cluster — this is the cheap
/// (O(n)) signal the directive-inferrer uses to suggest a `settings:max-name-group` cap, and it
/// is independent of clustering, so it's reported even when the cap skips those groups.
#[must_use]
pub fn large_name_groups(defs: &[Def], min_members: usize) -> Vec<(&'static KindSpec, String, usize)> {
    let mut counts: BTreeMap<(&str, &str), (&'static KindSpec, usize)> = BTreeMap::new();
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
        timed("pass3-type3", || findings.extend(pass_type3(&defs, opts.type3_theta)));
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
