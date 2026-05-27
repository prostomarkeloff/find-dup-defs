//! `find-dup-defs` library — the cross-file duplicate-definition detection pipeline
//! exposed as a Rust API (consumed by the `find-dup-defs` CLI binary AND by
//! `iilint._native` via the iilint-py PyO3 bindings).
#![allow(
    clippy::doc_markdown, // PyO3 etc. mentioned in prose
    clippy::struct_excessive_bools // PipelineOpts mirrors CLI flags, not a state machine
)]
//!
//! Three complementary passes (all over one native parse per definition):
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

use dup_defs_core::{AnalyzedFn, Language, ModuleDef};
use rayon::prelude::*;
use serde::Serialize;
use walkdir::WalkDir;

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
/// Language codes accepted by the CLI's `--only` (and the Python wrapper).
pub const KNOWN_LANGS: &[&str] = &["py", "ts"];

/// Directory-name blacklist for `.py`/`.ts` discovery — virtualenvs, package
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
    /// Definition kind: `"functions"` / `"methods"` / `"classes"` /
    /// `"interfaces"` / `"constants"` / `"type-aliases"`.
    pub kind: String,
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

/// Walk `paths` and return every matching source file. `matches` decides which
/// extensions count for one language frontend.
pub fn collect_files(paths: &[PathBuf], matches: impl Fn(&Path) -> bool) -> Vec<String> {
    let mut files: BTreeSet<String> = BTreeSet::new();
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
                let path = entry.path();
                if matches(path) {
                    files.insert(path.to_string_lossy().into_owned());
                }
            }
        } else if matches(p) {
            files.insert(p.to_string_lossy().into_owned());
        }
    }
    files.into_iter().collect()
}

/// Walk for `.py` files under `paths`.
#[must_use]
pub fn collect_py_files(paths: &[PathBuf]) -> Vec<String> {
    collect_files(paths, |p| p.extension().is_some_and(|e| e == "py"))
}

/// Walk for `.ts` / `.tsx` / `.mts` / `.cts` files under `paths`.
#[must_use]
pub fn collect_ts_files(paths: &[PathBuf]) -> Vec<String> {
    collect_files(paths, |p| {
        p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e, "ts" | "tsx" | "mts" | "cts"))
    })
}

// ── Kind helpers ───────────────────────────────────────────────────────────

#[must_use]
pub fn is_body_kind(kind: &str) -> bool {
    matches!(kind, "functions" | "classes" | "methods" | "interfaces")
}

#[must_use]
pub fn is_fn_like(kind: &str) -> bool {
    matches!(kind, "functions" | "methods")
}

/// Per-language dispatch for the names-preserved cluster canonical.
#[must_use]
pub fn cluster_canonical_for(def: &ModuleDef) -> String {
    match def.lang {
        Language::Python => py_canon::ast_canonical(&def.text),
        Language::TypeScript => ts_canon::ast_canonical(&def.text),
    }
}

/// Per-language dispatch for the full callable analysis.
#[must_use]
pub fn analyze_for(def: &ModuleDef) -> Option<AnalyzedFn> {
    let texts = [def.text.clone()];
    match def.lang {
        Language::Python => py_canon::analyze_functions(&texts).into_iter().next().flatten(),
        Language::TypeScript => ts_canon::analyze_functions(&texts).into_iter().next().flatten(),
    }
}

#[must_use]
pub fn member(defs: &[ModuleDef], i: usize) -> (String, usize, usize) {
    (defs[i].file.clone(), defs[i].line + 1, defs[i].col)
}

// ── Passes ─────────────────────────────────────────────────────────────────

/// Pass 1 — name-gated: same-named functions/classes clustered by body
/// similarity; same-named constants/type-aliases compared by raw text.
#[must_use]
pub fn pass_name_gated(
    defs: &[ModuleDef],
    canon_of: &[Option<String>],
    threshold: f64,
    error: f64,
    min_size: usize,
) -> Vec<Finding> {
    let mut groups: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
    for (i, d) in defs.iter().enumerate() {
        groups.entry((d.kind.as_str(), d.name.as_str())).or_default().push(i);
    }
    let groups: Vec<((&str, &str), Vec<usize>)> =
        groups.into_iter().filter(|(_, v)| v.len() >= 2).collect();

    groups
        .par_iter()
        .flat_map_iter(|((kind, name), idxs)| {
            if !is_body_kind(kind) {
                let canons: Vec<String> = idxs.iter().map(|&i| defs[i].text.clone()).collect();
                let clusters = difflib_fast::cluster_canonicals(&canons, 0.0);
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
                            kind: (*kind).to_owned(),
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
                idxs.iter().map(|&i| canon_of[i].clone().unwrap_or_default()).collect();
            difflib_fast::cluster_canonicals(&canons, threshold)
                .into_iter()
                .filter(|(c, _)| c.len() >= min_size)
                .map(|(c, min_sim)| {
                    let loc = c.iter().map(|&k| defs[idxs[k]].loc).max().unwrap_or(0);
                    let args = c.iter().map(|&k| defs[idxs[k]].args).max().unwrap_or(0);
                    Finding {
                        pass: "name",
                        kind: (*kind).to_owned(),
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

/// Pass 2 — cross-name: functions with identical alpha-renamed canonicals but
/// ≥2 distinct names across ≥2 files.
#[must_use]
pub fn pass_cross_name(
    defs: &[ModuleDef],
    fn_idx: &[usize],
    analyses: &[Option<AnalyzedFn>],
    min_size: usize,
) -> Vec<Finding> {
    let mut buckets: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (p, a) in analyses.iter().enumerate() {
        if let Some((_, xname, _, _)) = a {
            buckets.entry(xname.as_str()).or_default().push(p);
        }
    }
    let mut out = Vec::new();
    for (_, ps) in buckets {
        if ps.len() < min_size {
            continue;
        }
        let names: BTreeSet<&str> = ps.iter().map(|&p| defs[fn_idx[p]].name.as_str()).collect();
        if names.len() < 2 {
            continue;
        }
        let size = analyses[ps[0]].as_ref().map_or(0, |a| a.3);
        let kind = if ps.iter().all(|&p| defs[fn_idx[p]].kind == "methods") {
            "methods"
        } else {
            "functions"
        };
        let loc = ps.iter().map(|&p| defs[fn_idx[p]].loc).max().unwrap_or(0);
        let args = ps.iter().map(|&p| defs[fn_idx[p]].args).max().unwrap_or(0);
        out.push(Finding {
            pass: "cross-name",
            kind: kind.to_owned(),
            name: names.iter().copied().collect::<Vec<_>>().join("/"),
            severity: if size >= SUBSTANCE_NODES { Severity::Error } else { Severity::Warning },
            min_sim: None,
            loc,
            args,
            thickness: thickness(loc, args, ps.len(), 1.0),
            snippet: defs[fn_idx[ps[0]]].text_orig.clone(),
            notes: Vec::new(),
            members: ps.iter().map(|&p| member(defs, fn_idx[p])).collect(),
        });
    }
    out
}

/// Pass 3 — Type-3 (`ECScan`): renamed near-copy functions via IDF-weighted
/// cosine over name-agnostic lines.
#[must_use]
pub fn pass_type3(
    defs: &[ModuleDef],
    fn_idx: &[usize],
    analyses: &[Option<AnalyzedFn>],
    theta: f64,
) -> Vec<Finding> {
    let (mut line_lists, mut names, mut def_of) = (Vec::new(), Vec::new(), Vec::new());
    for (p, a) in analyses.iter().enumerate() {
        if let Some((_, _, lines, _)) = a {
            if lines.len() >= SHINGLE_LINES {
                line_lists.push(lines.clone());
                names.push(defs[fn_idx[p]].name.clone());
                def_of.push(fn_idx[p]);
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
            let kind = if cluster.iter().all(|&c| defs[def_of[c]].kind == "methods") {
                "methods"
            } else {
                "functions"
            };
            let loc = cluster.iter().map(|&c| defs[def_of[c]].loc).max().unwrap_or(0);
            let args = cluster.iter().map(|&c| defs[def_of[c]].args).max().unwrap_or(0);
            Some(Finding {
                pass: "type-3",
                kind: kind.to_owned(),
                name: distinct.iter().copied().collect::<Vec<_>>().join("/"),
                severity: if min_sim >= TYPE3_ERROR_THETA { Severity::Error } else { Severity::Warning },
                min_sim: Some(min_sim),
                loc,
                args,
                thickness: thickness(loc, args, cluster.len(), min_sim),
                snippet: defs[def_of[cluster[0]]].text_orig.clone(),
                notes: Vec::new(),
                members: cluster.iter().map(|&c| member(defs, def_of[c])).collect(),
            })
        })
        .collect()
}

// ── Section index (for stable, reproducible cluster sort) ──────────────────

/// Printed-section index — functions come first by pass, then methods by pass
/// (same three-pass order), then classes, interfaces (TS-only kind, slotted
/// right after classes since both are body-bearing nominal types),
/// type-aliases.
#[must_use]
pub fn section_index(f: &Finding) -> usize {
    match (f.kind.as_str(), f.pass) {
        ("constants", _) => 0,
        ("functions", "name") => 1,
        ("functions", "cross-name") => 2,
        ("functions", "type-3") => 3,
        ("methods", "name") => 4,
        ("methods", "cross-name") => 5,
        ("methods", "type-3") => 6,
        ("classes", _) => 7,
        ("interfaces", _) => 8,
        ("type-aliases", _) => 9,
        _ => 10,
    }
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
    /// Restrict to a specific subset of definition kinds (functions / methods /
    /// classes / interfaces / constants / type-aliases). `None` = all.
    pub kinds: Option<Vec<String>>,
    /// Scan Python files (default `true`).
    pub scan_py: bool,
    /// Scan TypeScript files (default `true`).
    pub scan_ts: bool,
    /// Skip the cross-name pass (default `false`).
    pub no_cross_name: bool,
    /// Skip the Type-3 pass (default `false`).
    pub no_type3: bool,
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
            scan_py: true,
            scan_ts: true,
            no_cross_name: false,
            no_type3: false,
        }
    }
}

/// Run the whole detection pipeline end-to-end:
///
/// 1. Walk `paths` for `.py` (if `scan_py`) and `.ts`/`.tsx`/`.mts`/`.cts`
///    (if `scan_ts`) and parse each via py-canon / ts-canon.
/// 2. Compute the names-preserved cluster canonical for every body-kind def
///    (rayon-parallel).
/// 3. Run name-gated, cross-name (unless `no_cross_name`), and Type-3 (unless
///    `no_type3`) passes.
/// 4. Apply `error_thickness` / `warning_thickness` demotion and
///    `escalate_thickness` escalation, if any is non-zero.
///
/// Returns the unsorted findings. The caller sorts (typically by
/// [`section_index`] + name + first member) and renders.
#[must_use]
pub fn scan_and_cluster(opts: &PipelineOpts) -> Vec<Finding> {
    let py_files = if opts.scan_py { collect_py_files(&opts.paths) } else { Vec::new() };
    let ts_files = if opts.scan_ts { collect_ts_files(&opts.paths) } else { Vec::new() };
    let mut defs = py_canon::find_module_defs(&py_files);
    defs.extend(ts_canon::find_module_defs(&ts_files));
    if let Some(kinds) = &opts.kinds {
        defs.retain(|d| kinds.iter().any(|k| k == &d.kind));
    }
    defs.sort_by(|a, b| (&a.file, a.line, a.col).cmp(&(&b.file, b.line, b.col)));

    let body_idx: Vec<usize> = (0..defs.len()).filter(|&i| is_body_kind(&defs[i].kind)).collect();
    let body_canon: Vec<String> =
        body_idx.par_iter().map(|&i| cluster_canonical_for(&defs[i])).collect();
    let mut canon_of: Vec<Option<String>> = vec![None; defs.len()];
    for (k, &i) in body_idx.iter().enumerate() {
        canon_of[i] = Some(body_canon[k].clone());
    }

    let fn_idx: Vec<usize> = (0..defs.len()).filter(|&i| is_fn_like(&defs[i].kind)).collect();
    let analyses: Vec<Option<AnalyzedFn>> =
        fn_idx.par_iter().map(|&i| analyze_for(&defs[i])).collect();

    let mut findings =
        pass_name_gated(&defs, &canon_of, opts.threshold, opts.error_threshold, opts.min_size);
    if !opts.no_cross_name {
        findings.extend(pass_cross_name(&defs, &fn_idx, &analyses, opts.min_size));
    }
    if !opts.no_type3 {
        findings.extend(pass_type3(&defs, &fn_idx, &analyses, opts.type3_theta));
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
