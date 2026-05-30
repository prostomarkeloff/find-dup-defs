//! Dev-only perf snapshots: serialize the pipeline's intermediate data (the scanned `Vec<Def>`,
//! the clustered `Vec<Finding>`) to JSON next to the corpus, so a per-stage micro-bench can reload
//! a stage's input in ~a second and loop the function under test — instead of re-running the whole
//! 25s pipeline for every measurement.
//!
//! `&'static` fields (`KindSpec`, `Def::lang`, `Finding::pass`) are reconstructed on load by leaking
//! interned copies (deduped by id) — acceptable for a short-lived bench process, and it means the
//! loader needs no frontend registry.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use dup_defs_core::{Analysis, Def, KindSpec};
use serde::{Deserialize, Serialize};

use crate::{Finding, Severity};

#[derive(Serialize, Deserialize)]
struct KindSnap {
    id: String,
    label: String,
    noun_plural: String,
    section: u16,
    body: bool,
    fn_like: bool,
}

#[derive(Serialize, Deserialize)]
struct AnalysisSnap {
    xname_canonical: String,
    type3_lines: Vec<String>,
    size: usize,
}

#[derive(Serialize, Deserialize)]
struct DefSnap {
    lang: String,
    kind: KindSnap,
    name: String,
    file: String,
    line: usize,
    col: usize,
    loc: usize,
    args: usize,
    text_orig: String,
    cluster_canonical: Option<String>,
    analysis: Option<AnalysisSnap>,
}

#[derive(Serialize, Deserialize)]
struct FindingSnap {
    pass: String,
    kind: KindSnap,
    name: String,
    severity: u8,
    min_sim: Option<f64>,
    loc: usize,
    args: usize,
    thickness: f64,
    notes: Vec<String>,
    members: Vec<(String, usize, usize)>,
    // NB: `snippet` is intentionally dropped — only the render bench consumes findings, and
    // `format_report` never reads it; keeping it would bloat the snapshot with full source.
}

fn kind_snap(k: &KindSpec) -> KindSnap {
    KindSnap {
        id: k.id.to_owned(),
        label: k.label.to_owned(),
        noun_plural: k.noun_plural.to_owned(),
        section: k.section,
        body: k.body,
        fn_like: k.fn_like,
    }
}

/// Interner: id → leaked `&'static KindSpec`, so repeated kinds share one static (and one leak).
#[derive(Default)]
struct KindInterner(HashMap<String, &'static KindSpec>);

impl KindInterner {
    fn get(&mut self, k: &KindSnap) -> &'static KindSpec {
        if let Some(s) = self.0.get(&k.id) {
            return s;
        }
        let leaked: &'static KindSpec = Box::leak(Box::new(KindSpec {
            id: Box::leak(k.id.clone().into_boxed_str()),
            label: Box::leak(k.label.clone().into_boxed_str()),
            noun_plural: Box::leak(k.noun_plural.clone().into_boxed_str()),
            section: k.section,
            body: k.body,
            fn_like: k.fn_like,
        }));
        self.0.insert(k.id.clone(), leaked);
        leaked
    }
}

fn leak_pass(p: &str) -> &'static str {
    match p {
        "name" => "name",
        "cross-name" => "cross-name",
        "type-3" => "type-3",
        other => Box::leak(other.to_owned().into_boxed_str()),
    }
}

/// Serialize `defs` to `<dir>/defs.json`.
pub fn dump_defs(dir: &Path, defs: &[Def]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let snaps: Vec<DefSnap> = defs
        .iter()
        .map(|d| DefSnap {
            lang: d.lang.to_owned(),
            kind: kind_snap(d.kind),
            name: d.name.clone(),
            file: d.file.to_string(),
            line: d.line,
            col: d.col,
            loc: d.loc,
            args: d.args,
            text_orig: d.text_orig.clone(),
            cluster_canonical: d.cluster_canonical.clone(),
            analysis: d.analysis.as_ref().map(|a| AnalysisSnap {
                xname_canonical: a.xname_canonical.clone(),
                type3_lines: a.type3_lines.clone(),
                size: a.size,
            }),
        })
        .collect();
    let f = std::fs::File::create(dir.join("defs.json"))?;
    serde_json::to_writer(std::io::BufWriter::new(f), &snaps)?;
    Ok(())
}

/// Reload `defs` from `<dir>/defs.json` (leaking interned `&'static` kinds/langs).
pub fn load_defs(dir: &Path) -> std::io::Result<Vec<Def>> {
    let f = std::fs::File::open(dir.join("defs.json"))?;
    let snaps: Vec<DefSnap> = serde_json::from_reader(std::io::BufReader::new(f))?;
    let mut interner = KindInterner::default();
    let mut langs: HashMap<String, &'static str> = HashMap::new();
    let out = snaps
        .into_iter()
        .map(|s| {
            let lang = *langs
                .entry(s.lang.clone())
                .or_insert_with(|| Box::leak(s.lang.clone().into_boxed_str()));
            Def {
                lang,
                kind: interner.get(&s.kind),
                name: s.name,
                file: Arc::from(s.file.as_str()),
                line: s.line,
                col: s.col,
                loc: s.loc,
                args: s.args,
                text_orig: s.text_orig,
                cluster_canonical: s.cluster_canonical,
                analysis: s.analysis.map(|a| Analysis {
                    xname_canonical: a.xname_canonical,
                    type3_lines: a.type3_lines,
                    size: a.size,
                }),
            }
        })
        .collect();
    Ok(out)
}

/// Serialize `findings` to `<dir>/findings.json` (sans `snippet`).
pub fn dump_findings(dir: &Path, findings: &[Finding]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let snaps: Vec<FindingSnap> = findings
        .iter()
        .map(|f| FindingSnap {
            pass: f.pass.to_owned(),
            kind: kind_snap(f.kind),
            name: f.name.clone(),
            severity: u8::try_from(f.severity.to_index()).unwrap_or(0),
            min_sim: f.min_sim,
            loc: f.loc,
            args: f.args,
            thickness: f.thickness,
            notes: f.notes.clone(),
            members: f.members.clone(),
        })
        .collect();
    let file = std::fs::File::create(dir.join("findings.json"))?;
    serde_json::to_writer(std::io::BufWriter::new(file), &snaps)?;
    Ok(())
}

/// Serialize the Type-3 inputs `(line_lists, names)` to `<dir>/type3.json` — plain strings, tiny
/// next to the full `defs.json`, so a Type-3 bench loads in milliseconds.
pub fn dump_type3(dir: &Path, line_lists: &[Vec<String>], names: &[String]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let file = std::fs::File::create(dir.join("type3.json"))?;
    serde_json::to_writer(std::io::BufWriter::new(file), &(line_lists, names))?;
    Ok(())
}

/// Reload the Type-3 inputs `(line_lists, names)` from `<dir>/type3.json`.
pub fn load_type3(dir: &Path) -> std::io::Result<(Vec<Vec<String>>, Vec<String>)> {
    let file = std::fs::File::open(dir.join("type3.json"))?;
    let v = serde_json::from_reader(std::io::BufReader::new(file))?;
    Ok(v)
}

/// Reload `findings` from `<dir>/findings.json` (snippet comes back empty — render ignores it).
pub fn load_findings(dir: &Path) -> std::io::Result<Vec<Finding>> {
    let file = std::fs::File::open(dir.join("findings.json"))?;
    let snaps: Vec<FindingSnap> = serde_json::from_reader(std::io::BufReader::new(file))?;
    let mut interner = KindInterner::default();
    let out = snaps
        .into_iter()
        .map(|s| Finding {
            pass: leak_pass(&s.pass),
            kind: interner.get(&s.kind),
            name: s.name,
            severity: Severity::from_index(i32::from(s.severity)),
            min_sim: s.min_sim,
            loc: s.loc,
            args: s.args,
            thickness: s.thickness,
            snippet: String::new(),
            notes: s.notes,
            members: s.members,
        })
        .collect();
    Ok(out)
}
