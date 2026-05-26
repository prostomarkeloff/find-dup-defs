//! `find-dup-defs` — find duplicate / near-duplicate top-level definitions across a Python codebase.
//!
//! Three complementary passes (all over one native parse per definition, `py-canon`):
//!   1. **name-gated** — same-named functions/classes clustered by exact Ratcliff–Obershelp body
//!      similarity (`difflib-fast`); same-named constants/type-aliases flagged by name alone.
//!   2. **cross-name** — *renamed* copy-paste: functions with byte-identical alpha-renamed canonicals
//!      but ≥2 distinct names across ≥2 files.
//!   3. **Type-3** (`ECScan`) — *renamed near-copies*: IDF-weighted cosine over name-agnostic lines,
//!      catching edited renamed copies the exact cross-name pass misses.
//!
//! Each cluster is graded ERROR / WARNING. Ported from the iilint dup-defs analyzer.

mod type3;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use clap::Parser;
use py_canon::{analyze_functions, ast_canonical_many, find_module_defs, ModuleDef};
use rayon::prelude::*;
use serde::Serialize;
use walkdir::WalkDir;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const SUBSTANCE_NODES: usize = 20; // cross-name: ERROR only when the canonical has ≥ this many nodes
const SHINGLE_LINES: usize = 3; // Type-3: N-line shingle window
const COMMON_RATIO: f64 = 0.007; // Type-3: drop a shingle present in > 0.7% of functions
const TYPE3_ERROR_THETA: f64 = 0.9; // Type-3: cluster min-cosine ≥ this → ERROR

#[derive(Clone, Copy, PartialEq, Eq)]
enum Severity {
    Error,
    Warning,
    /// Low-confidence finding: name-collision constants whose bodies differ, mass-demoted
    /// WARNINGs below `--warning-thickness`, or directive-chained de-escalations from WARNING.
    /// Hidden from the human report by default; surface with `--show-info` or read JSON.
    Info,
}
impl Severity {
    fn label(self) -> &'static str {
        match self {
            Severity::Error => "ERROR",
            Severity::Warning => "WARNING",
            Severity::Info => "INFO",
        }
    }
    /// 0/1/2 ladder for directive-driven stepping — `escalate` goes UP (toward 0=ERROR),
    /// `de-escalate` goes DOWN (toward 2=INFO). Clamped so chained directives can't overshoot.
    fn to_index(self) -> i32 {
        match self {
            Severity::Error => 0,
            Severity::Warning => 1,
            Severity::Info => 2,
        }
    }
    fn from_index(i: i32) -> Self {
        match i.clamp(0, 2) {
            0 => Severity::Error,
            1 => Severity::Warning,
            _ => Severity::Info,
        }
    }
}

/// One reported cluster of duplicate definitions.
#[derive(Clone)]
struct Finding {
    pass: &'static str, // "name" | "cross-name" | "type-3"
    kind: String,
    name: String,
    severity: Severity,
    min_sim: Option<f64>,
    // "How fat is this cluster" triage signals. `loc`/`args` are the MAX across members
    // (representative of the biggest copy you'd dedup); `thickness` is a [0,1] composite — see
    // [`thickness`]. JSON exposes all three so external consumers can sort/filter.
    loc: usize,
    args: usize,
    thickness: f64,
    // Pre-strip source of one representative member, for calibration-mode display (and any
    // future code-preview use). Kept off the normal report output so per-finding lines stay
    // scannable — only the calibrate path opts in.
    snippet: String,
    /// Notes attached by matching directives — shown on the report line so a reader sees the
    /// "why" of any override without needing to look up the directive list. Accumulates if
    /// multiple directives match (e.g. a `de-escalate` with reason A and a `note` with reason B).
    notes: Vec<String>,
    members: Vec<(String, usize, usize)>, // (file, line 1-indexed, col 0-indexed)
}

/// Normalized [0, 1] "GET ME REFACTORED" score. Driven by three dimensions, each saturated
/// independently with `1 - exp(-x/k)`:
///
/// * `volume = (n_members - 1) * loc` — the lines you'd actually delete by extracting one
///   shared helper. A 2-copy 30-liner and a 4-copy 10-liner have the same dedup volume (30
///   lines) and read as equally "loud". This is the dominant signal.
/// * `args` — wide signatures push the score up marginally; a 6-arg method that repeats is
///   architecturally chunkier than a 2-arg one even at equal line count.
/// * `sim` — `1.0` for normalized-exact/cross-name passes, the cluster's min pairwise ratio
///   for name-gated body kinds, and `0.5` for name-only constant/type-alias matches (we
///   didn't compare bodies, so the duplicate is unverified).
///
/// Architecture stays extensible: add another dimension + characteristic constant when needed.
#[allow(clippy::cast_precision_loss)] // loc/args/n_members are always small (line/parameter/cluster sizes).
fn thickness(loc: usize, args: usize, n_members: usize, sim: f64) -> f64 {
    let volume = (loc as f64) * (n_members.saturating_sub(1) as f64);
    let volume_score = 1.0 - (-volume / 30.0).exp();
    let args_score = 1.0 - (-(args as f64) / 5.0).exp();
    0.7 * volume_score + 0.1 * args_score + 0.2 * sim
}

#[derive(Parser)]
#[command(about, version)]
#[allow(clippy::struct_excessive_bools)] // CLI flags, not a state machine
struct Cli {
    /// Files or directories to scan (directories are walked for `*.py`).
    #[arg(required = true)]
    paths: Vec<PathBuf>,
    /// Name-gated clustering floor: same-named defs cluster if their exact RO ratio is ≥ this.
    #[arg(short, long, default_value_t = 0.5)]
    threshold: f64,
    /// Name-gated ERROR floor: a cluster's min pairwise ratio ≥ this gates as ERROR (else WARNING).
    #[arg(short, long, default_value_t = 0.85)]
    error_threshold: f64,
    /// De-escalate any cluster whose `thickness` < this from ERROR to WARNING — the calibration
    /// knob for "what counts as a real refactor candidate" per codebase. WARNINGs stay WARNINGs
    /// at this stage (`--warning-thickness` handles their tier); findings are NEVER dropped.
    /// `0.0` (default) leaves severities untouched. Try `0.3` to mute 2-3 line copy-pastes,
    /// `0.5` for genuinely fat candidates only.
    #[arg(long, default_value_t = 0.0)]
    error_thickness: f64,
    /// De-escalate any WARNING below this thickness to INFO — symmetric to `--error-thickness`,
    /// keeps WARNING meaningful by routing low-confidence stuff to the INFO tier instead of
    /// letting it pile up. Default `0.0` leaves WARNINGs untouched.
    #[arg(long, default_value_t = 0.0)]
    warning_thickness: f64,
    /// Escalate any non-ERROR cluster whose `thickness` ≥ this to ERROR — symmetric inverse of
    /// `--error-thickness`. Catches the "fat cluster that landed in WARNING because of mid-sim
    /// or small-canonical heuristics, but is actually a big-mass refactor target." Default
    /// `0.0` disabled. Applied LAST so it overrides the de-escalation knobs above.
    #[arg(long, default_value_t = 0.0)]
    escalate_thickness: f64,
    /// Include INFO-severity findings in the human-readable report. JSON output always
    /// contains them. Default hides INFO so the normal report stays focused on the actionable
    /// ERROR/WARNING list.
    #[arg(long)]
    show_info: bool,
    /// Repo root for relative paths in the report (paths under it are shown repo-relative).
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    /// Type-3 cosine detection floor (candidate edge when cosine > this).
    #[arg(long, default_value_t = 0.7)]
    type3_theta: f64,
    /// Only report clusters with at least this many definitions.
    #[arg(long, default_value_t = 2)]
    min_size: usize,
    /// Restrict to these kinds (comma-separated: functions,methods,classes,constants,type-aliases).
    #[arg(long, value_delimiter = ',')]
    kinds: Option<Vec<String>>,
    /// Skip the cross-name (renamed-identical) pass.
    #[arg(long)]
    no_cross_name: bool,
    /// Skip the Type-3 (renamed near-copy) pass.
    #[arg(long)]
    no_type3: bool,
    /// Only report ERROR-severity clusters.
    #[arg(long)]
    errors_only: bool,
    /// Emit JSON instead of the human-readable report.
    #[arg(long)]
    json: bool,
    /// Print a thickness-calibration report instead of the normal duplicate list — distribution
    /// of current ERROR thicknesses + three percentile-anchored `--error-thickness` candidates
    /// (`permissive`/`balanced`/`strict` at p50/p75/p90). Pairs with `--json` for machine
    /// output. Respects `--kinds` / `--min-size` so you can calibrate against a focused subset.
    #[arg(long)]
    calibrate: bool,
    /// Filter findings by a compact glob-rule. Repeatable. Format:
    ///   `ACTION:[KIND:]NAME[@PATH][=NOTE]`
    ///
    /// `ACTION` ∈ `suppress` (drop entirely) | `de-escalate` (ERROR → WARNING) | `escalate`
    ///            (WARNING → ERROR) | `note` (attach text, no severity change). Same vocabulary
    ///            as iilint's `[tool.iilint].directives`.
    /// `KIND`   ∈ `METHOD` | `FUNCTION` | `CLASS` | `CONSTANT` | `TYPE_ALIAS` (optional).
    /// `NAME`   glob on the cluster's dup name (`Class.method` or `a/b/c` for cross-name).
    ///          `*` matches any chars, `?` matches one. Tested against each `/`-separated alias.
    /// `PATH`   glob on member file paths (any member match wins).
    /// `NOTE`   free-form annotation surfaced next to the finding (required for `note`,
    ///          optional self-documentation for the other three).
    ///
    /// Examples:
    ///   `-D de-escalate:Plugin.get_*_hook=intentional plugin no-op API`
    ///   `-D suppress:FUNCTION:spawn@*mypyc/lib-rt/*=bootstrap copy`
    ///   `-D de-escalate:METHOD:*.test_*@*/test/*=parametrize candidate`
    ///   `-D escalate:METHOD:Lock.*@*/storage/*=Lock dups block this release`
    ///   `-D note:METHOD:For*.begin_body=v2 refactor target`
    #[arg(long = "directive", short = 'D', value_name = "DIRECTIVE")]
    directives: Vec<String>,
}

// ───────────────────────────── directives (glob filters) ─────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum DirectiveAction {
    /// Drop the finding from the report entirely.
    Suppress,
    /// ERROR → WARNING. WARNING stays WARNING.
    Deescalate,
    /// WARNING → ERROR. ERROR stays ERROR.
    Escalate,
    /// Pure annotation — attach a note to the finding, no severity change.
    Note,
}

/// One compact directive: `ACTION:[KIND:]NAME[@PATH]`. See the `--directive` CLI doc for the
/// grammar and examples. Parsed once into globs + a kind filter; matched against each Finding.
struct Directive {
    action: DirectiveAction,
    /// Internal kind tag (`functions` / `methods` / `classes` / `constants` / `type-aliases`).
    /// `None` matches any kind.
    kind: Option<&'static str>,
    name_pat: String,
    path_pat: Option<String>,
    /// Self-documentation attached after `=` in the spec — surfaced alongside the finding so
    /// the next reviewer sees WHY this directive was added without grepping the directive file.
    /// Required for `note`; optional but encouraged for `suppress`/`de-escalate`/`escalate`.
    note: Option<String>,
}

impl Directive {
    fn matches(&self, f: &Finding) -> bool {
        if let Some(k) = self.kind {
            if f.kind != k {
                return false;
            }
        }
        // Cross-name passes join aliases with '/'. Match any individual alias OR the joined
        // form, so a pattern like `Foo.bar` lands on `Foo.bar/Baz.bar` without needing wildcards.
        let any_alias = f.name.split('/').any(|alias| glob_match(&self.name_pat, alias));
        if !(any_alias || glob_match(&self.name_pat, &f.name)) {
            return false;
        }
        if let Some(pp) = &self.path_pat {
            if !f.members.iter().any(|(file, _, _)| glob_match(pp, file)) {
                return false;
            }
        }
        true
    }
}

/// Parse `ACTION:[KIND:]NAME[@PATH]` into a [`Directive`]. The KIND segment is optional — if the
/// first `:`-delimited chunk after the action matches a known kind label (`METHOD`/`FUNCTION`/
/// `CLASS`/`CONSTANT`/`TYPE_ALIAS`, case-insensitive, `_` or `-` for the alias), it's consumed;
/// otherwise the whole remainder is treated as the NAME glob.
fn parse_directive(spec: &str) -> Result<Directive, String> {
    // Strip the optional `=NOTE` tail first — note text is free-form (may contain `:` or `@`).
    let (head, note) = match spec.split_once('=') {
        Some((h, n)) => (h, Some(n.trim().to_owned())),
        None => (spec, None),
    };
    let (action_str, rest) = head
        .split_once(':')
        .ok_or_else(|| format!("expected `ACTION:…` in directive {spec:?}"))?;
    let action = match action_str.trim().to_ascii_lowercase().replace('-', "").as_str() {
        "suppress" => DirectiveAction::Suppress,
        "deescalate" => DirectiveAction::Deescalate,
        "escalate" => DirectiveAction::Escalate,
        "note" => DirectiveAction::Note,
        other => {
            return Err(format!(
                "unknown action {other:?} in directive {spec:?} \
                 (expected `suppress` / `de-escalate` / `escalate` / `note`)"
            ));
        }
    };
    if action == DirectiveAction::Note && note.is_none() {
        return Err(format!(
            "`note` directive requires note text after `=` (directive: {spec:?})"
        ));
    }
    let (kind, after_kind) = match rest.split_once(':') {
        Some((maybe_kind, after)) => {
            let token = maybe_kind.trim().to_ascii_uppercase().replace('-', "_");
            match token.as_str() {
                // `*` is an explicit "any kind" — consume the segment, no kind filter.
                "*" => (None, after),
                "METHOD" | "METHODS" => (Some("methods"), after),
                "FUNCTION" | "FUNCTIONS" => (Some("functions"), after),
                "CLASS" | "CLASSES" => (Some("classes"), after),
                "CONSTANT" | "CONSTANTS" => (Some("constants"), after),
                "TYPE_ALIAS" | "TYPE_ALIASES" => (Some("type-aliases"), after),
                _ => (None, rest), // not a kind — the whole `rest` is NAME[@PATH]
            }
        }
        None => (None, rest),
    };
    let (name, path) = match after_kind.split_once('@') {
        Some((n, p)) => (n.trim(), Some(p.trim())),
        None => (after_kind.trim(), None),
    };
    if name.is_empty() {
        return Err(format!("empty name pattern in directive {spec:?}"));
    }
    Ok(Directive {
        action,
        kind,
        name_pat: name.to_owned(),
        path_pat: path.map(str::to_owned),
        note,
    })
}

/// Minimal glob matcher — `*` (any run), `?` (single char). Recursive-backtracking; cheap for
/// the short patterns directives carry. No character classes / brace expansion — keep CLI
/// grammar tiny; users wanting heavier matching can add a second directive.
fn glob_match(pat: &str, s: &str) -> bool {
    fn go(p: &[u8], pi: usize, t: &[u8], ti: usize) -> bool {
        if pi == p.len() {
            return ti == t.len();
        }
        match p[pi] {
            b'*' => {
                // Collapse runs of `*` so `**name` doesn't blow up branching.
                let mut j = pi + 1;
                while j < p.len() && p[j] == b'*' {
                    j += 1;
                }
                for k in ti..=t.len() {
                    if go(p, j, t, k) {
                        return true;
                    }
                }
                false
            }
            b'?' => ti < t.len() && go(p, pi + 1, t, ti + 1),
            c => ti < t.len() && t[ti] == c && go(p, pi + 1, t, ti + 1),
        }
    }
    go(pat.as_bytes(), 0, s.as_bytes(), 0)
}

/// Directory-name blacklist for `.py` discovery — virtualenvs, package caches, build artefacts
/// and vendored tooling that's never project source. Matches what `ruff` / `pyright` skip by
/// default; the goal is "what `cloc`-style tools count as the project" not "every byte on disk".
/// `.egg-info` matched by suffix, not in the set.
const SKIP_DIRS: &[&str] = &[
    ".venv", "venv", "venv2", "venv3", "env", ".env",
    "__pycache__", "node_modules", ".git",
    "build", "dist", ".tox", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".ipynb_checkpoints", "site-packages",
    "target", ".idea", ".vscode", ".direnv",
];

fn is_excluded_dir(name: &str) -> bool {
    SKIP_DIRS.contains(&name) || name.ends_with(".egg-info")
}

fn collect_py_files(paths: &[PathBuf]) -> Vec<String> {
    let mut files: BTreeSet<String> = BTreeSet::new();
    for p in paths {
        if p.is_dir() {
            // `filter_entry` prunes excluded directories from the walk (vs filtering each
            // file after-the-fact) — a `venv/` with thousands of vendored `.py` files takes
            // milliseconds to skip this way instead of seconds to walk and discard.
            let walker = WalkDir::new(p).into_iter().filter_entry(|e| {
                // Always keep the root entry, otherwise drop directories named like venvs/etc.
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
                if path.extension().is_some_and(|e| e == "py") {
                    files.insert(path.to_string_lossy().into_owned());
                }
            }
        } else if p.extension().is_some_and(|e| e == "py") {
            files.insert(p.to_string_lossy().into_owned());
        }
    }
    files.into_iter().collect()
}

fn is_body_kind(kind: &str) -> bool {
    matches!(kind, "functions" | "classes" | "methods")
}

/// Function-like kinds: drive cross-name and Type-3 passes (both treat any callable body the
/// same way). Mixed clusters across the two kinds are still reported, with a kind label derived
/// from cluster members in each pass.
fn is_fn_like(kind: &str) -> bool {
    matches!(kind, "functions" | "methods")
}

fn member(defs: &[ModuleDef], i: usize) -> (String, usize, usize) {
    (defs[i].file.clone(), defs[i].line + 1, defs[i].col)
}

/// Pass 1 — name-gated: same-named functions/classes clustered by body similarity; same-named
/// constants/type-aliases flagged by name alone.
fn pass_name_gated(defs: &[ModuleDef], canon_of: &[Option<String>], threshold: f64, error: f64, min_size: usize) -> Vec<Finding> {
    let mut groups: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
    for (i, d) in defs.iter().enumerate() {
        groups.entry((d.kind.as_str(), d.name.as_str())).or_default().push(i);
    }
    let groups: Vec<((&str, &str), Vec<usize>)> = groups.into_iter().filter(|(_, v)| v.len() >= 2).collect();

    groups
        .par_iter()
        .flat_map_iter(|((kind, name), idxs)| {
            if !is_body_kind(kind) {
                // constants / type-aliases: compare BODIES too, not just names. A blind
                // name-match used to false-positive on idiomatic tokens like `T = TypeVar("T")`
                // or `CLASS = "..."` defined in many modules with different contents. Now we
                // run difflib over the texts and three-way severity off the min pairwise sim:
                //   sim ≥ error_threshold → ERROR (real content dup)
                //   sim ≥ threshold       → WARNING (related, partial overlap)
                //   else                   → INFO    (name collides, content actually differs)
                // Using `cluster_canonicals(_, 0.0)` is the cheapest way to read min pairwise
                // sim — at threshold 0 every member joins one cluster, and we read its min_sim.
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
            let canons: Vec<String> = idxs.iter().map(|&i| canon_of[i].clone().unwrap_or_default()).collect();
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

/// Pass 2 — cross-name: functions with identical alpha-renamed canonicals but ≥2 distinct names
/// across ≥2 files (renamed copy-paste the name-gated pass cannot see).
fn pass_cross_name(defs: &[ModuleDef], fn_idx: &[usize], analyses: &[Option<py_canon::AnalyzedFn>], min_size: usize) -> Vec<Finding> {
    let mut buckets: BTreeMap<&str, Vec<usize>> = BTreeMap::new(); // xname canonical → fn-local positions
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
            continue; // ≥2 distinct names is the "cross-name" semantic; intra-file is fine
        }
        let size = analyses[ps[0]].as_ref().map_or(0, |a| a.3);
        // Pure-method clusters land in the methods sections; functions-only and mixed go to
        // functions (the broader category, so a method↔top-level-function dup is visible).
        let kind = if ps.iter().all(|&p| defs[fn_idx[p]].kind == "methods") { "methods" } else { "functions" };
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
            // Cross-name is byte-exact on alpha-renamed canonicals → sim = 1.0.
            thickness: thickness(loc, args, ps.len(), 1.0),
            snippet: defs[fn_idx[ps[0]]].text_orig.clone(),
            notes: Vec::new(),
            members: ps.iter().map(|&p| member(defs, fn_idx[p])).collect(),
        });
    }
    out
}

/// Pass 3 — Type-3 (ECScan): renamed near-copy functions via IDF-weighted cosine over name-agnostic
/// lines; ≥2 distinct names across ≥2 files.
fn pass_type3(defs: &[ModuleDef], fn_idx: &[usize], analyses: &[Option<py_canon::AnalyzedFn>], theta: f64) -> Vec<Finding> {
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
                return None; // ≥2 distinct names; intra-file allowed (e.g., 2 classes/1 file)
            }
            // Same kind-label rule as cross-name: pure-method cluster → methods section,
            // anything else → functions section.
            let kind = if cluster.iter().all(|&c| defs[def_of[c]].kind == "methods") { "methods" } else { "functions" };
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

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn main() {
    let cli = Cli::parse();
    let files = collect_py_files(&cli.paths);
    let mut defs = find_module_defs(&files);
    if let Some(kinds) = &cli.kinds {
        defs.retain(|d| kinds.iter().any(|k| k == &d.kind));
    }
    // global stable order so every pass's member indices are deterministic (RO ratio is arg-order-
    // sensitive, clustering is single-linkage — a fixed order makes results reproducible).
    defs.sort_by(|a, b| (&a.file, a.line, a.col).cmp(&(&b.file, b.line, b.col)));

    // names-preserved cluster canonical for body kinds (functions + classes) — the name-gated key.
    let body_idx: Vec<usize> = (0..defs.len()).filter(|&i| is_body_kind(&defs[i].kind)).collect();
    let body_texts: Vec<String> = body_idx.iter().map(|&i| defs[i].text.clone()).collect();
    let body_canon = ast_canonical_many(&body_texts);
    let mut canon_of: Vec<Option<String>> = vec![None; defs.len()];
    for (k, &i) in body_idx.iter().enumerate() {
        canon_of[i] = Some(body_canon[k].clone());
    }

    // rename-invariant analysis (xname canonical, name-agnostic lines, node count) for any
    // callable body — both top-level functions and class methods, since a method copy-pasted as
    // a free function (or vice versa) is still a duplicate worth flagging.
    let fn_idx: Vec<usize> = (0..defs.len()).filter(|&i| is_fn_like(&defs[i].kind)).collect();
    let fn_texts: Vec<String> = fn_idx.iter().map(|&i| defs[i].text.clone()).collect();
    let analyses = analyze_functions(&fn_texts);

    let mut findings = pass_name_gated(&defs, &canon_of, cli.threshold, cli.error_threshold, cli.min_size);
    if !cli.no_cross_name {
        findings.extend(pass_cross_name(&defs, &fn_idx, &analyses, cli.min_size));
    }
    if !cli.no_type3 {
        findings.extend(pass_type3(&defs, &fn_idx, &analyses, cli.type3_theta));
    }
    // Thickness-based mass de-escalation. Two-stage so ERROR → WARNING and WARNING → INFO are
    // independent knobs (a finding could fall through both if both thresholds are set high).
    if cli.error_thickness > 0.0 {
        for f in &mut findings {
            if f.severity == Severity::Error && f.thickness < cli.error_thickness {
                f.severity = Severity::Warning;
            }
        }
    }
    if cli.warning_thickness > 0.0 {
        for f in &mut findings {
            if f.severity == Severity::Warning && f.thickness < cli.warning_thickness {
                f.severity = Severity::Info;
            }
        }
    }
    if cli.escalate_thickness > 0.0 {
        // Last word — overrides any de-escalation above so a high-T cluster that was demoted
        // by sim-based heuristics still surfaces as ERROR. One-step lift: WARNING/INFO → ERROR.
        for f in &mut findings {
            if f.severity != Severity::Error && f.thickness >= cli.escalate_thickness {
                f.severity = Severity::Error;
            }
        }
    }
    // User-authored directives (the manual override layer). Parsed once; mismatched action /
    // kind tokens fail loud with exit-2 so a typo in CI doesn't silently keep emitting errors.
    let directives: Vec<Directive> = cli
        .directives
        .iter()
        .map(|s| {
            parse_directive(s).unwrap_or_else(|e| {
                eprintln!("find-dup-defs: invalid --directive: {e}");
                std::process::exit(2);
            })
        })
        .collect();
    if !directives.is_empty() {
        // Attach notes from EVERY matching directive first, even ones whose action will drop
        // the finding — but we run suppress last, so a `suppress` with a `=note` still has its
        // note visible if some non-suppressing directive also matches. Order of effects:
        //   1. Notes accumulate from every match.
        //   2. Severity adjusts: escalate before de-escalate (so a conflicting pair lands at
        //      ERROR, the louder of the two — matches "the user explicitly asked to raise it").
        //   3. Suppress drops findings entirely.
        for f in &mut findings {
            for d in &directives {
                if d.matches(f) {
                    if let Some(n) = &d.note {
                        f.notes.push(n.clone());
                    }
                }
            }
            // Each matching directive contributes one step on the severity ladder — same
            // semantic as iilint's `severity_steps`. Multiple `de-escalate`s chain (ERROR →
            // WARNING → INFO); `escalate` cancels out (1 escalate + 1 de-escalate = no-op).
            let step = directives
                .iter()
                .filter(|d| d.matches(f))
                .map(|d| match d.action {
                    DirectiveAction::Deescalate => 1,
                    DirectiveAction::Escalate => -1,
                    _ => 0,
                })
                .sum::<i32>();
            if step != 0 {
                f.severity = Severity::from_index(f.severity.to_index() + step);
            }
        }
        findings.retain(|f| {
            !directives
                .iter()
                .any(|d| d.action == DirectiveAction::Suppress && d.matches(f))
        });
    }
    if cli.errors_only {
        findings.retain(|f| f.severity == Severity::Error);
    }
    // Detection/section order (constants, fn-name-gated, fn-cross-name, fn-Type-3, classes,
    // type-aliases), then within a section by name and first member — deterministic + reproducible.
    findings.sort_by(|a, b| {
        section_index(a)
            .cmp(&section_index(b))
            .then(a.name.cmp(&b.name))
            .then(a.members[0].cmp(&b.members[0]))
    });

    if cli.calibrate {
        // Calibration is informational — never exits non-zero, never prints the dup list. Runs
        // AFTER demotion/escalation knobs so each invocation answers "given my current
        // configuration, what's the next nudge worth?". Both ladders surface: ERROR drives
        // `--error-thickness` suggestions, WARNING drives `--warning-thickness`.
        let errs: Vec<&Finding> = findings.iter().filter(|f| f.severity == Severity::Error).collect();
        let warns: Vec<&Finding> = findings.iter().filter(|f| f.severity == Severity::Warning).collect();
        let report = if cli.json {
            render_calibration_json(&errs, &warns, &findings, &cli.repo_root)
        } else {
            format_calibration(&errs, &warns, &findings, &cli.repo_root)
        };
        print!("{report}");
        return;
    }

    let report = if cli.json {
        // JSON consumers get every severity unconditionally — it's their job to filter.
        render_json(&findings, &cli.repo_root)
    } else {
        // Human report hides INFO by default — that's the whole point of the tier. JSON path
        // unchanged so downstream tooling never loses data.
        let visible: Vec<Finding> = if cli.show_info {
            findings.clone()
        } else {
            findings.iter().filter(|f| f.severity != Severity::Info).cloned().collect()
        };
        format_report(&visible, cli.threshold, cli.error_threshold, &cli.repo_root)
    };
    print!("{report}");

    if findings.iter().any(|f| f.severity == Severity::Error) {
        std::process::exit(1);
    }
}

// ───────────────────────────── thickness calibration ─────────────────────────────

/// Linear-interpolated percentile on a value list, *sorted ascending*. `p` in `[0, 1]`. Returns
/// `0.0` on empty input — the format/JSON paths handle the "no errors to calibrate" case by
/// reading the list length, so the value here just needs to be a stable sentinel.
#[allow(
    clippy::cast_precision_loss, // cluster counts fit in f64 mantissa
    clippy::cast_possible_truncation, // floor/ceil already discrete and bounded by sorted.len()
    clippy::cast_sign_loss, // p ∈ [0, 1] and len ≥ 0 → rank is non-negative
)]
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn median_usize_sorted(sorted: &[usize]) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    sorted[sorted.len() / 2]
}

#[derive(Serialize)]
struct CalibSuggestion {
    label: &'static str,
    percentile: u8,
    error_thickness: f64,
    errors_kept: usize,
    median_loc: usize,
    median_args: usize,
    /// "What does this threshold actually catch?" — smallest cluster whose `thickness ≥
    /// error_thickness`, i.e. the threshold's lower-edge exemplar. Pairs with `example_snippet`.
    example_name: String,
    example_thickness: f64,
    example_loc: usize,
    example_args: usize,
    /// `<short-path>:<line>` for every member, so the user can jump to any copy. Full member
    /// list (not truncated) — JSON consumer can render however they want; the human formatter
    /// trims to a few entries inline.
    example_members: Vec<String>,
    /// Full source text of one representative member (pre-strip for methods). JSON keeps it
    /// verbatim; the human formatter dedents Python class-method indentation and box-frames it.
    example_snippet: String,
}

#[derive(Serialize)]
struct CalibHistBin {
    thickness_lo: f64,
    thickness_hi: f64,
    count: usize,
}

/// Per-severity calibration block — emitted twice in the JSON report, once for ERROR
/// (driving `--error-thickness`) and once for WARNING (driving `--warning-thickness`).
#[derive(Serialize)]
struct CalibTier {
    total: usize,
    /// CLI flag the suggestions would set — `error-thickness` for the ERROR block,
    /// `warning-thickness` for the WARNING block.
    target_flag: &'static str,
    histogram: Vec<CalibHistBin>,
    suggestions: Vec<CalibSuggestion>,
}

#[derive(Serialize)]
struct CalibReport {
    error: CalibTier,
    warning: CalibTier,
    /// Auto-detected noise patterns + ready-to-paste directives. Each entry quotes the exact
    /// `-D` string the user can copy. See [`infer_directives`] for what's checked.
    inferred_directives: Vec<InferredDirective>,
}

/// One auto-discovered noise pattern: a CLI-ready directive string + rationale + effect size,
/// so the calibrate user can decide whether to paste it into their CI invocation.
#[derive(Serialize, Clone)]
struct InferredDirective {
    /// Exact `-D <…>` string the user can paste verbatim.
    directive: String,
    /// One-sentence "why this matches your codebase" explanation, grounded in counts.
    rationale: String,
    /// How many existing findings the directive would touch (ERROR + WARNING + INFO combined).
    affects_total: usize,
    affects_error: usize,
    affects_warning: usize,
    affects_info: usize,
}

/// 10 bins of width 0.1 over `[0.0, 1.0]`; the top bin captures anything ≥ 1.0 (saturated
/// thickness never *exceeds* 1, but we collapse the boundary to make the upper-edge bucket
/// non-empty for huge defs).
fn thickness_histogram(values: &[f64]) -> Vec<CalibHistBin> {
    let bins: i32 = 10;
    let step = 1.0 / f64::from(bins);
    (0..bins)
        .map(|i| {
            let lo = f64::from(i) * step;
            let hi = f64::from(i + 1) * step;
            let count = values.iter().filter(|&&v| v >= lo && (v < hi || (i == bins - 1 && v >= hi))).count();
            CalibHistBin { thickness_lo: lo, thickness_hi: hi, count }
        })
        .collect()
}

/// Build the three percentile-anchored suggestions. For each anchor we report what the user
/// would actually get: how many ERROR clusters survive `T >= anchor`, the median size of those
/// survivors, AND the cluster sitting at the threshold's lower edge — its name + a snippet of
/// its source — so the user can SEE what kind of dup would still be ERROR after dialing this
/// knob. That's the difference between picking a number blindly and picking one against a
/// concrete example.
fn calibration_suggestions(errs: &[&Finding], repo_root: &Path) -> Vec<CalibSuggestion> {
    if errs.is_empty() {
        return Vec::new();
    }
    let mut ts: Vec<f64> = errs.iter().map(|f| f.thickness).collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    [("permissive", 50u8), ("balanced", 75), ("strict", 90)]
        .into_iter()
        .map(|(label, p)| {
            let t = percentile_sorted(&ts, f64::from(p) / 100.0);
            let kept: Vec<&&Finding> = errs.iter().filter(|f| f.thickness >= t).collect();
            let mut locs: Vec<usize> = kept.iter().map(|f| f.loc).collect();
            let mut args: Vec<usize> = kept.iter().map(|f| f.args).collect();
            locs.sort_unstable();
            args.sort_unstable();
            // The "smallest survivor" — the cluster closest to (but not below) the threshold.
            // Tiebreak by thickness ascending, then by name for determinism.
            let example = kept
                .iter()
                .min_by(|a, b| {
                    a.thickness
                        .partial_cmp(&b.thickness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.name.cmp(&b.name))
                })
                .copied();
            let (ex_name, ex_t, ex_loc, ex_args, ex_members, ex_snippet) = match example {
                Some(f) => {
                    let members: Vec<String> = f
                        .members
                        .iter()
                        .map(|(file, line, _)| format!("{}:{}", short_path(file, repo_root), line))
                        .collect();
                    (f.name.clone(), f.thickness, f.loc, f.args, members, f.snippet.clone())
                }
                None => (String::new(), 0.0, 0, 0, Vec::new(), String::new()),
            };
            CalibSuggestion {
                label,
                percentile: p,
                error_thickness: t,
                errors_kept: kept.len(),
                median_loc: median_usize_sorted(&locs),
                median_args: median_usize_sorted(&args),
                example_name: ex_name,
                example_thickness: ex_t,
                example_loc: ex_loc,
                example_args: ex_args,
                example_members: ex_members,
                example_snippet: ex_snippet,
            }
        })
        .collect()
}

/// Normalize Python class-method indentation for standalone display: the `def` line lives at
/// column 0 (we strip leading whitespace at extraction time), but the body still carries the
/// class-level extra 4-space step. We dedent the body to 4 spaces so a method snippet reads as
/// a regular function would in isolation. Top-level functions and bodies already at ≤4-space
/// indent are left untouched.
fn dedent_python_def_body(s: &str) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() < 2 {
        return s.to_owned();
    }
    let body_min = lines[1..]
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.chars().take_while(|c| *c == ' ').count())
        .min()
        .unwrap_or(0);
    if body_min <= 4 {
        return s.to_owned();
    }
    let strip = body_min - 4;
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    out.push(lines[0].to_owned());
    for l in &lines[1..] {
        if l.len() >= strip && l.chars().take(strip).all(|c| c == ' ') {
            out.push(l[strip..].to_owned());
        } else {
            out.push((*l).to_owned());
        }
    }
    out.join("\n")
}

/// Render a snippet inside a left-bar box (`│ …`) under the suggestion bullet — the visual
/// frame separates code from prose without compounding indentation. Long bodies collapse to
/// `max_lines` rows + an elision marker so the calibration view stays one screen.
fn snippet_box(snippet: &str, max_lines: usize) -> String {
    let cleaned = dedent_python_def_body(snippet);
    let lines: Vec<&str> = cleaned.lines().collect();
    let shown_count = lines.len().min(max_lines);
    let extra = lines.len().saturating_sub(shown_count);
    let mut out = String::new();
    out.push_str("    ┌──\n");
    for l in &lines[..shown_count] {
        out.push_str("    │ ");
        out.push_str(l);
        out.push('\n');
    }
    if extra > 0 {
        let _ = writeln!(out, "    │ … (+{extra} more lines)");
    }
    out.push_str("    └──");
    out
}

// ───────────────────────────── directive inference ─────────────────────────────

fn path_is_i18n(p: &str) -> bool {
    let lo = p.to_ascii_lowercase();
    lo.contains("/locale/") || lo.contains("/locales/") || lo.contains("/i18n/") || lo.contains("/translations/")
}
fn path_is_test(p: &str) -> bool {
    let lo = p.to_ascii_lowercase();
    if lo.contains("/tests/") || lo.contains("/test/") {
        return true;
    }
    p.rsplit('/').next().is_some_and(|fname| fname.starts_with("test_") || fname.ends_with("_test.py"))
}
fn path_is_generated(p: &str) -> bool {
    p.ends_with("_pb2.py")
        || p.ends_with("_pb2_grpc.py")
        || p.ends_with("_grpc.py")
        || p.ends_with(".gen.py")
        || p.contains("/_generated/")
        || p.contains("/generated/")
}
fn path_is_migration(p: &str) -> bool {
    p.contains("/migrations/") || p.contains("/alembic/versions/")
}
/// Tutorial / docs / example code — each snippet is a self-contained illustration that
/// reasonably shares boilerplate with siblings (same `app = FastAPI()` scaffold, same login
/// example, …). Refactoring them into shared helpers would defeat the "this snippet is
/// runnable on its own" point of doc examples.
fn path_is_doc_example(p: &str) -> bool {
    p.contains("/docs_src/")
        || p.contains("/docs/src/")
        || p.contains("/examples/")
        || p.contains("/example/")
        || p.contains("/tutorial/")
        || p.contains("/tutorials/")
        || p.contains("/samples/")
}

/// Build one suggestion from findings matching `predicate`; returns `None` below `min_clusters`
/// so the inferred-directives section only surfaces patterns with real evidence.
fn build_suggestion(
    findings: &[Finding],
    predicate: impl Fn(&Finding) -> bool,
    min_clusters: usize,
    directive: &str,
    rationale_template: impl Fn(usize) -> String,
) -> Option<InferredDirective> {
    let matched: Vec<&Finding> = findings.iter().filter(|f| predicate(f)).collect();
    if matched.len() < min_clusters {
        return None;
    }
    let by_sev = |s: Severity| matched.iter().filter(|f| f.severity == s).count();
    Some(InferredDirective {
        directive: directive.to_owned(),
        rationale: rationale_template(matched.len()),
        affects_total: matched.len(),
        affects_error: by_sev(Severity::Error),
        affects_warning: by_sev(Severity::Warning),
        affects_info: by_sev(Severity::Info),
    })
}

/// Pattern-match across findings to emit ready-to-paste `-D` directives for the recurring noise
/// shapes (i18n locales, all-test clusters, generated `_pb2`/`_grpc`, schema migrations). The
/// user still owns the final list; we shortcut the first 80% of project-specific tuning.
fn infer_directives(findings: &[Finding]) -> Vec<InferredDirective> {
    let mut out: Vec<InferredDirective> = Vec::new();
    if let Some(s) = build_suggestion(
        findings,
        |f| {
            f.kind == "constants" && {
                // Majority-in-locale heuristic — Django-style codebases have one canonical
                // declaration in `global_settings.py` plus the per-locale overrides; requiring
                // ALL members in locale paths would miss exactly that legit case.
                #[allow(clippy::cast_precision_loss)] // member counts always small
                {
                    let n = f.members.len() as f64;
                    let in_locale = f.members.iter().filter(|(p, _, _)| path_is_i18n(p)).count() as f64;
                    in_locale / n >= 0.8
                }
            }
        },
        5,
        "suppress:CONSTANT:*@*locale*=i18n locale tables, duplication is by design",
        |n| format!("{n} CONSTANT clusters are ≥80% inside locale/i18n paths"),
    ) {
        out.push(s);
    }
    if let Some(s) = build_suggestion(
        findings,
        |f| f.members.iter().all(|(p, _, _)| path_is_test(p)),
        10,
        "de-escalate:*:*@*tests/*=test parametrize/fixture candidates — review for conftest",
        |n| format!("{n} clusters live entirely in test paths — parametrize/conftest candidates"),
    ) {
        out.push(s);
    }
    if let Some(s) = build_suggestion(
        findings,
        |f| f.members.iter().any(|(p, _, _)| path_is_generated(p)),
        3,
        "suppress:*:*@*_pb2*=generated protobuf/gRPC code, suppress wholesale",
        |n| format!("{n} clusters touch `*_pb2*`/`*_grpc*` files — generated code"),
    ) {
        out.push(s);
    }
    if let Some(s) = build_suggestion(
        findings,
        |f| f.members.iter().all(|(p, _, _)| path_is_migration(p)),
        3,
        "suppress:*:*@*migrations/*=schema migrations are snapshots, not refactor targets",
        |n| format!("{n} clusters live entirely under migrations/ — schema-history files"),
    ) {
        out.push(s);
    }
    // Tutorial / docs / example code — fastapi/docs_src, sklearn/examples, etc.
    if let Some(s) = build_suggestion(
        findings,
        |f| f.members.iter().all(|(p, _, _)| path_is_doc_example(p)),
        5,
        "de-escalate:*:*@*docs_src/*=tutorial/doc-example code — snippet duplication is expected",
        |n| format!("{n} clusters live entirely under docs_src/ / examples/ / tutorial/ paths"),
    ) {
        out.push(s);
    }
    out
}

/// Compact "where are the duplicates?" line. Consecutive members sharing the same file collapse
/// to `path:lineA, :lineB, :lineC` so multi-copy intra-file clusters don't smear the same path
/// 3×. Truncates the tail with `(+N more)` after `max` entries.
fn fmt_member_locations(members: &[String], max: usize) -> String {
    if members.is_empty() {
        return String::new();
    }
    let take = members.len().min(max);
    let mut parts: Vec<String> = Vec::with_capacity(take);
    let mut last_file: Option<&str> = None;
    for entry in &members[..take] {
        let (file, line) = match entry.rsplit_once(':') {
            Some((f, l)) => (f, l),
            None => (entry.as_str(), ""),
        };
        if last_file == Some(file) {
            parts.push(format!(":{line}"));
        } else {
            parts.push(entry.clone());
            last_file = Some(file);
        }
    }
    let mut s = parts.join(", ");
    let extra = members.len() - take;
    if extra > 0 {
        let _ = write!(s, " (+{extra} more)");
    }
    s
}

fn render_calibration_json(errs: &[&Finding], warns: &[&Finding], all: &[Finding], repo_root: &Path) -> String {
    let report = CalibReport {
        error: CalibTier {
            total: errs.len(),
            target_flag: "error-thickness",
            histogram: thickness_histogram(&errs.iter().map(|f| f.thickness).collect::<Vec<_>>()),
            suggestions: calibration_suggestions(errs, repo_root),
        },
        warning: CalibTier {
            total: warns.len(),
            target_flag: "warning-thickness",
            histogram: thickness_histogram(&warns.iter().map(|f| f.thickness).collect::<Vec<_>>()),
            suggestions: calibration_suggestions(warns, repo_root),
        },
        inferred_directives: infer_directives(all),
    };
    serde_json::to_string_pretty(&report).unwrap_or_default() + "\n"
}

/// Render one severity tier — histogram + percentile-anchored suggestions + a sample snippet
/// per suggestion. Both `--error-thickness` and `--warning-thickness` are calibrated this way;
/// `tier_label` / `target_flag` switch the prose, the rest is identical so the two blocks read
/// as a parallel pair.
fn format_calibration_tier(
    out: &mut String,
    tier_label: &str,
    target_flag: &str,
    findings: &[&Finding],
    repo_root: &Path,
) {
    if findings.is_empty() {
        let _ = writeln!(out, "=== thickness calibration ({tier_label}): 0 clusters — skip.\n");
        return;
    }
    let _ = writeln!(
        out,
        "=== thickness calibration ({tier_label}): {} clusters analyzed ===\n",
        findings.len()
    );
    let hist = thickness_histogram(&findings.iter().map(|f| f.thickness).collect::<Vec<_>>());
    let max_count = hist.iter().map(|b| b.count).max().unwrap_or(1).max(1);
    let bar_max = 30usize;
    let _ = writeln!(out, "distribution (each ▇ ≈ one {tier_label} cluster, scaled to fit):");
    for b in &hist {
        let bar_len = (b.count * bar_max).div_ceil(max_count);
        let bar = "▇".repeat(bar_len);
        let _ = writeln!(
            out,
            "  T [{:.1}, {:.1})  {bar} {}",
            b.thickness_lo, b.thickness_hi, b.count
        );
    }
    out.push('\n');
    let _ = writeln!(
        out,
        "suggested thresholds (p50/p75/p90 of current {tier_label} thickness distribution):"
    );
    for s in calibration_suggestions(findings, repo_root) {
        out.push('\n');
        let _ = writeln!(
            out,
            "  {:<11}  --{target_flag} {:.2}  → {} {tier_label} remain  (median dup: {} loc, {} args)",
            s.label, s.error_thickness, s.errors_kept, s.median_loc, s.median_args
        );
        let _ = writeln!(
            out,
            "    e.g. {}  [T={:.2}, loc={}, args={}]",
            s.example_name, s.example_thickness, s.example_loc, s.example_args
        );
        let locs = fmt_member_locations(&s.example_members, 3);
        if !locs.is_empty() {
            let _ = writeln!(out, "         {locs}");
        }
        out.push_str(&snippet_box(&s.example_snippet, 15));
        out.push('\n');
    }
    out.push('\n');
}

fn format_calibration(errs: &[&Finding], warns: &[&Finding], all: &[Finding], repo_root: &Path) -> String {
    if errs.is_empty() && warns.is_empty() {
        return "=== thickness calibration: 0 findings — nothing to calibrate against. ===\n".to_owned();
    }
    let mut out = String::new();
    format_calibration_tier(&mut out, "ERROR", "error-thickness", errs, repo_root);
    format_calibration_tier(&mut out, "WARNING", "warning-thickness", warns, repo_root);

    let inferred = infer_directives(all);
    if !inferred.is_empty() {
        out.push_str("=== inferred directives (auto-detected noise patterns) ===\n\n");
        for d in &inferred {
            let _ = writeln!(out, "  → -D '{}'", d.directive);
            let _ = writeln!(out, "    rationale: {}", d.rationale);
            let _ = writeln!(
                out,
                "    affects: {} total ({} ERROR, {} WARNING, {} INFO)",
                d.affects_total, d.affects_error, d.affects_warning, d.affects_info
            );
            out.push('\n');
        }
        out.push_str("(Paste any of these into your CI invocation — patterns matched repeatably,\n");
        out.push_str("not heuristics on individual clusters. Review the rationale before applying.)\n\n");
    }
    out.push_str("workflow: dial `--error-thickness` to focus the gate; dial `--warning-thickness`\n");
    out.push_str("to control how much low-confidence noise stays in WARNING vs falls to INFO.\n");
    out
}

// ───────────────────────── report (identical to the Python reference) ─────────────────────────

/// Report label for a kind — `DupKind.value` in the reference (uppercase, singular).
fn dup_kind(kind: &str) -> &'static str {
    match kind {
        "functions" => "FUNCTION",
        "methods" => "METHOD",
        "classes" => "CLASS",
        "constants" => "CONSTANT",
        "type-aliases" => "TYPE_ALIAS",
        _ => "UNKNOWN",
    }
}

/// A group is "cross-name" when found by a name-agnostic pass (renamed copy-paste).
fn is_cross_name(pass: &str) -> bool {
    pass == "cross-name" || pass == "type-3"
}

/// Printed-section index — functions come first by pass, then methods by pass (same three-pass
/// order), then classes / type-aliases. Keeping methods adjacent to functions reads naturally
/// since both pass families are callable-body comparisons.
fn section_index(f: &Finding) -> usize {
    match (f.kind.as_str(), f.pass) {
        ("constants", _) => 0,
        ("functions", "name") => 1,
        ("functions", "cross-name") => 2,
        ("functions", "type-3") => 3,
        ("methods", "name") => 4,
        ("methods", "cross-name") => 5,
        ("methods", "type-3") => 6,
        ("classes", _) => 7,
        ("type-aliases", _) => 8,
        _ => 9,
    }
}

/// Best-effort repo-relative path; raw string when not under `repo_root` (mirrors `short_path`).
fn short_path(file: &str, repo_root: &Path) -> String {
    let p = std::fs::canonicalize(file).unwrap_or_else(|_| PathBuf::from(file));
    let root = std::fs::canonicalize(repo_root).unwrap_or_else(|_| repo_root.to_path_buf());
    p.strip_prefix(&root).map_or_else(|_| file.to_owned(), |rel| rel.to_string_lossy().into_owned())
}

/// The trailing marker on a `DUPLICATE` line: similarity / pass tag + thickness triage signals
/// (`T=…`, `loc=…`, `args=…`). `args` is dropped when 0 (constants, type-aliases, classes) to
/// keep the line scannable. Constants / type-aliases with no similarity tag still get a `loc`
/// hint so the user can tell a multi-line constant from a one-line alias at a glance.
fn group_suffix(f: &Finding) -> String {
    let tag: Option<String> = if is_cross_name(f.pass) && f.min_sim.is_none() {
        Some("normalized-exact".to_owned())
    } else {
        f.min_sim.map(|s| format!("ast sim {s:.2}"))
    };
    let n = f.members.len();
    let metrics = if f.args > 0 {
        format!("T={:.2}, n={}, loc={}, args={}", f.thickness, n, f.loc, f.args)
    } else {
        format!("T={:.2}, n={}, loc={}", f.thickness, n, f.loc)
    };
    let base = match tag {
        Some(t) => format!("  [{t}, {metrics}]"),
        None => format!("  [{metrics}]"),
    };
    // Notes from matching directives — visible right next to the finding so the next reviewer
    // sees the override reason without grepping. Multiple notes join with `; ` to stay on one
    // line; the JSON form keeps them as a structured array.
    if f.notes.is_empty() {
        base
    } else {
        format!("{base}  # {}", f.notes.join("; "))
    }
}

/// Human-readable per-section report — byte-for-byte the Python `format_report`.
fn format_report(findings: &[Finding], warn: f64, error: f64, repo_root: &Path) -> String {
    let sim = format!("AST sim warn={warn} error={error}");
    let sections: [(String, usize); 9] = [
        ("duplicate constants (cross-file, by name)".to_owned(), 0),
        (format!("duplicate functions (cross-file, {sim})"), 1),
        ("duplicate functions (cross-name, exact AST-normalized)".to_owned(), 2),
        ("duplicate functions (cross-name Type-3, IDF-weighted cosine)".to_owned(), 3),
        (format!("duplicate methods (cross-file, {sim})"), 4),
        ("duplicate methods (cross-name, exact AST-normalized)".to_owned(), 5),
        ("duplicate methods (cross-name Type-3, IDF-weighted cosine)".to_owned(), 6),
        (format!("duplicate classes (cross-file, {sim})"), 7),
        ("duplicate type aliases (cross-file, by name)".to_owned(), 8),
    ];

    let mut lines: Vec<String> = Vec::new();
    for (index, (header, sect)) in sections.iter().enumerate() {
        if index > 0 {
            lines.push(String::new());
        }
        lines.push(format!("--- {header} ---"));
        for f in findings.iter().filter(|&f| section_index(f) == *sect) {
            lines.push(format!("DUPLICATE {} [{}]: {}{}", dup_kind(&f.kind), f.severity.label(), f.name, group_suffix(f)));
            for (file, line, _col) in &f.members {
                lines.push(format!("  {}:{}", short_path(file, repo_root), line));
            }
            lines.push(String::new());
        }
    }

    if findings.is_empty() {
        lines.push("No cross-file duplicates.".to_owned());
        return lines.join("\n") + "\n";
    }
    let errs = findings.iter().filter(|f| f.severity == Severity::Error).count();
    let warns = findings.len() - errs;
    lines.push(format!("# summary: {errs} ERROR, {warns} WARNING groups"));
    lines.join("\n") + "\n"
}

#[derive(Serialize)]
struct JsonMember {
    file: String,
    line: usize,
}

#[derive(Serialize)]
struct JsonGroup {
    kind: String,
    name: String,
    severity: String,
    min_sim: Option<f64>,
    cross_name: bool,
    /// Composite [0,1] "fat function" score — see [`thickness`].
    thickness: f64,
    /// Max non-blank-line count across cluster members.
    loc: usize,
    /// Max parameter count across cluster members (0 for non-callable kinds).
    args: usize,
    members: Vec<JsonMember>,
    allowlist_key: String,
    notes: Vec<String>,
}

#[derive(Serialize)]
struct JsonReport {
    groups: Vec<JsonGroup>,
    summary: serde_json::Map<String, serde_json::Value>,
}

/// Machine-readable groups + summary — byte-for-byte the Python `render_json` (indent=2).
fn render_json(findings: &[Finding], repo_root: &Path) -> String {
    let groups: Vec<JsonGroup> = findings
        .iter()
        .map(|f| {
            let cross = is_cross_name(f.pass);
            let rule = if cross { "dup-xname".to_owned() } else { format!("dup-{}", dup_kind(&f.kind).to_ascii_lowercase()) };
            JsonGroup {
                kind: dup_kind(&f.kind).to_owned(),
                name: f.name.clone(),
                severity: f.severity.label().to_owned(),
                min_sim: f.min_sim,
                cross_name: cross,
                thickness: f.thickness,
                loc: f.loc,
                args: f.args,
                members: f.members.iter().map(|(file, line, _)| JsonMember { file: short_path(file, repo_root), line: *line }).collect(),
                allowlist_key: format!("{rule} {}", f.name),
                notes: f.notes.clone(),
            }
        })
        .collect();

    // summary: counts in first-seen severity order (matches the reference dict), then total.
    let mut summary = serde_json::Map::new();
    for f in findings {
        let key = f.severity.label();
        let n = summary.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0) + 1;
        summary.insert(key.to_owned(), serde_json::Value::from(n));
    }
    summary.insert("total".to_owned(), serde_json::Value::from(findings.len()));

    let report = JsonReport { groups, summary };
    serde_json::to_string_pretty(&report).unwrap_or_default() + "\n"
}
