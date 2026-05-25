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
use std::path::PathBuf;

use clap::Parser;
use py_canon::{analyze_functions, ast_canonical_many, find_module_defs, ModuleDef};
use rayon::prelude::*;
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
}
impl Severity {
    fn label(self) -> &'static str {
        match self {
            Severity::Error => "ERROR",
            Severity::Warning => "WARNING",
        }
    }
    fn rank(self) -> u8 {
        match self {
            Severity::Error => 0,
            Severity::Warning => 1,
        }
    }
}

/// One reported cluster of duplicate definitions.
struct Finding {
    pass: &'static str, // "name" | "cross-name" | "type-3"
    kind: String,
    name: String,
    severity: Severity,
    min_sim: Option<f64>,
    members: Vec<(String, usize, usize)>, // (file, line 1-indexed, col 0-indexed)
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
    #[arg(short, long, default_value_t = 0.9)]
    error_threshold: f64,
    /// Type-3 cosine detection floor (candidate edge when cosine > this).
    #[arg(long, default_value_t = 0.7)]
    type3_theta: f64,
    /// Only report clusters with at least this many definitions.
    #[arg(long, default_value_t = 2)]
    min_size: usize,
    /// Restrict to these kinds (comma-separated: functions,classes,constants,type-aliases).
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
}

fn collect_py_files(paths: &[PathBuf]) -> Vec<String> {
    let mut files: BTreeSet<String> = BTreeSet::new();
    for p in paths {
        if p.is_dir() {
            for entry in WalkDir::new(p).into_iter().filter_map(Result::ok) {
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
    kind == "functions" || kind == "classes"
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
                // constants / type-aliases: any same-name reuse is suspicious → ERROR, name-only.
                return vec![Finding {
                    pass: "name",
                    kind: (*kind).to_owned(),
                    name: (*name).to_owned(),
                    severity: Severity::Error,
                    min_sim: None,
                    members: idxs.iter().map(|&i| member(defs, i)).collect(),
                }];
            }
            let canons: Vec<String> = idxs.iter().map(|&i| canon_of[i].clone().unwrap_or_default()).collect();
            difflib_fast::cluster_canonicals(&canons, threshold)
                .into_iter()
                .filter(|(c, _)| c.len() >= min_size)
                .map(|(c, min_sim)| Finding {
                    pass: "name",
                    kind: (*kind).to_owned(),
                    name: (*name).to_owned(),
                    severity: if min_sim >= error { Severity::Error } else { Severity::Warning },
                    min_sim: Some(min_sim),
                    members: c.iter().map(|&k| member(defs, idxs[k])).collect(),
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
        let files: BTreeSet<&str> = ps.iter().map(|&p| defs[fn_idx[p]].file.as_str()).collect();
        if names.len() < 2 || files.len() < 2 {
            continue; // cross-FILE contract: ≥2 distinct names AND files
        }
        let size = analyses[ps[0]].as_ref().map_or(0, |a| a.3);
        out.push(Finding {
            pass: "cross-name",
            kind: "functions".to_owned(),
            name: names.iter().copied().collect::<Vec<_>>().join("/"),
            severity: if size >= SUBSTANCE_NODES { Severity::Error } else { Severity::Warning },
            min_sim: None,
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
            let files: BTreeSet<&str> = cluster.iter().map(|&c| defs[def_of[c]].file.as_str()).collect();
            if distinct.len() < 2 || files.len() < 2 {
                return None;
            }
            Some(Finding {
                pass: "type-3",
                kind: "functions".to_owned(),
                name: distinct.iter().copied().collect::<Vec<_>>().join("/"),
                severity: if min_sim >= TYPE3_ERROR_THETA { Severity::Error } else { Severity::Warning },
                min_sim: Some(min_sim),
                members: cluster.iter().map(|&c| member(defs, def_of[c])).collect(),
            })
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
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
    let total_defs = defs.len();

    // names-preserved cluster canonical for body kinds (functions + classes) — the name-gated key.
    let body_idx: Vec<usize> = (0..defs.len()).filter(|&i| is_body_kind(&defs[i].kind)).collect();
    let body_texts: Vec<String> = body_idx.iter().map(|&i| defs[i].text.clone()).collect();
    let body_canon = ast_canonical_many(&body_texts);
    let mut canon_of: Vec<Option<String>> = vec![None; defs.len()];
    for (k, &i) in body_idx.iter().enumerate() {
        canon_of[i] = Some(body_canon[k].clone());
    }

    // rename-invariant analysis (xname canonical, name-agnostic lines, node count) for functions.
    let fn_idx: Vec<usize> = (0..defs.len()).filter(|&i| defs[i].kind == "functions").collect();
    let fn_texts: Vec<String> = fn_idx.iter().map(|&i| defs[i].text.clone()).collect();
    let analyses = analyze_functions(&fn_texts);

    let mut findings = pass_name_gated(&defs, &canon_of, cli.threshold, cli.error_threshold, cli.min_size);
    if !cli.no_cross_name {
        findings.extend(pass_cross_name(&defs, &fn_idx, &analyses, cli.min_size));
    }
    if !cli.no_type3 {
        findings.extend(pass_type3(&defs, &fn_idx, &analyses, cli.type3_theta));
    }
    if cli.errors_only {
        findings.retain(|f| f.severity == Severity::Error);
    }
    // ERROR first, then larger / more-similar / earlier-named clusters
    findings.sort_by(|a, b| {
        a.severity
            .rank()
            .cmp(&b.severity.rank())
            .then(b.members.len().cmp(&a.members.len()))
            .then(b.min_sim.unwrap_or(0.0).total_cmp(&a.min_sim.unwrap_or(0.0)))
            .then(a.name.cmp(&b.name))
    });

    if cli.json {
        print_json(&findings);
        return;
    }

    let (mut errs, mut warns) = (0usize, 0usize);
    for f in &findings {
        match f.severity {
            Severity::Error => errs += 1,
            Severity::Warning => warns += 1,
        }
        let tag = match f.pass {
            "cross-name" => " (cross-name)",
            "type-3" => " (type-3)",
            _ => "",
        };
        let sim = f.min_sim.map_or_else(String::new, |s| format!(", min similarity {s:.3}"));
        println!("\n[{}] {} '{}'{} — {} definitions{}:", f.severity.label(), f.kind, f.name, tag, f.members.len(), sim);
        for (file, line, col) in &f.members {
            println!("    {}:{}:{}", rel(file), line, col + 1);
        }
    }
    eprintln!(
        "\nscanned {} files, {} top-level defs → {} clusters ({} ERROR, {} WARNING)",
        files.len(),
        total_defs,
        findings.len(),
        errs,
        warns
    );
    if errs > 0 {
        std::process::exit(1);
    }
}

/// Display `file` relative to the current directory when it's under it (cosmetic).
fn rel(file: &str) -> &str {
    thread_local! {
        static CWD: String = std::env::current_dir().ok().and_then(|d| d.to_str().map(|s| format!("{s}/"))).unwrap_or_default();
    }
    CWD.with(|cwd| if !cwd.is_empty() && file.starts_with(cwd.as_str()) { &file[cwd.len()..] } else { file })
}

fn json_escape(s: &str) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn print_json(findings: &[Finding]) {
    println!("[");
    for (fi, f) in findings.iter().enumerate() {
        let members: Vec<String> = f
            .members
            .iter()
            .map(|(file, line, col)| format!("{{\"file\":\"{}\",\"line\":{line},\"col\":{}}}", json_escape(file), col + 1))
            .collect();
        let sim = f.min_sim.map_or_else(|| "null".to_owned(), |s| format!("{s:.6}"));
        let comma = if fi + 1 < findings.len() { "," } else { "" };
        println!(
            "  {{\"pass\":\"{}\",\"severity\":\"{}\",\"kind\":\"{}\",\"name\":\"{}\",\"min_sim\":{sim},\"members\":[{}]}}{comma}",
            f.pass,
            f.severity.label(),
            json_escape(&f.kind),
            json_escape(&f.name),
            members.join(",")
        );
    }
    println!("]");
}
