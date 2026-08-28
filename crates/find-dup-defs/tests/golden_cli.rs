//! Byte-for-byte CLI golden test — the parity gate for the engine/frontend-contract refactor.
//!
//! Runs the built binary against the committed `tests/fixtures/mixed` tree (mixed Python +
//! TypeScript + TSX, exercising every report section) and asserts stdout matches the captured
//! goldens exactly. The default-run goldens (`report.txt` / `report.json` / `calibrate.txt`)
//! were captured from `main` before the refactor, so they pin output across the rewrite; the
//! `--only` goldens reflect the intended active-only section behavior.
//!
//! `--repo-root` is pinned to the fixtures dir so member paths are repo-relative and stable
//! across machines (no absolute paths leak into the goldens).

use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_find-dup-defs");
const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/mixed");
/// A second tree for the converge goldens. Separate on purpose: the pass needs definitions that
/// reach imports and word one shape differently, and adding those to `mixed` would move every
/// existing golden for a pass they have nothing to do with.
const FIX_CONVERGE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/converge");

fn run(extra: &[&str]) -> String {
    run_in(FIX, extra)
}

fn run_in(fixture: &str, extra: &[&str]) -> String {
    let out = Command::new(BIN)
        .arg(fixture)
        .arg("--repo-root")
        .arg(fixture)
        .args(extra)
        .output()
        .expect("spawn find-dup-defs");
    String::from_utf8(out.stdout).expect("stdout is utf-8")
}

#[track_caller]
fn assert_golden(got: &str, golden: &str, name: &str) {
    if got == golden {
        return;
    }
    let first_diff = got
        .lines()
        .zip(golden.lines())
        .enumerate()
        .find(|(_, (a, b))| a != b);
    let detail = match first_diff {
        Some((i, (a, b))) => format!("first diff at line {}:\n  got:    {a:?}\n  golden: {b:?}", i + 1),
        None => format!(
            "outputs share a common prefix but differ in length (got {} lines, golden {} lines)",
            got.lines().count(),
            golden.lines().count()
        ),
    };
    panic!("CLI golden mismatch for {name}\n{detail}\n\nRe-run with the binary and diff tests/golden/{name} if this change is intended.");
}

#[test]
fn report_default_matches_golden() {
    assert_golden(&run(&["--show-info"]), include_str!("golden/report.txt"), "report.txt");
}

#[test]
fn json_default_matches_golden() {
    assert_golden(&run(&["--json"]), include_str!("golden/report.json"), "report.json");
}

#[test]
fn calibrate_default_matches_golden() {
    assert_golden(&run(&["--calibrate"]), include_str!("golden/calibrate.txt"), "calibrate.txt");
}

#[test]
fn report_only_py_matches_golden() {
    assert_golden(&run(&["--only", "py", "--show-info"]), include_str!("golden/report.py.txt"), "report.py.txt");
}

#[test]
fn report_only_ts_matches_golden() {
    assert_golden(&run(&["--only", "ts", "--show-info"]), include_str!("golden/report.ts.txt"), "report.ts.txt");
}

#[test]
fn lenses_matches_golden() {
    // Locks the lens record across two frontends at once — the fixture holds Python and TypeScript,
    // and the ten questions are answered by each language's own walk.
    assert_golden(&run(&["--kinds", "lenses", "--show-info"]), include_str!("golden/lenses.txt"), "lenses.txt");
}

#[test]
fn converge_matches_golden() {
    // Locks both anchors, the family rubric and the per-dialect vocabulary: the fixture holds a
    // Python pair that meets only on a subject, a Python family worded three ways, and a Rust and a
    // TypeScript pair whose canonicals are s-expr dumps rather than source-like.
    assert_golden(
        &run_in(FIX_CONVERGE, &["--converge", "--show-info"]),
        include_str!("golden/converge.txt"),
        "converge.txt",
    );
}

// ───────────────────────── directive reporting ─────────────────────────
//
// `--json` reports what every `-D` matched. The point is the zero: a directive that suppresses
// nothing can only be found by asking the tool, because the matcher (globs × kind gating × path
// scoping) lives here and a second implementation elsewhere would be a second answer.

fn run_status(fixture: &str, extra: &[&str]) -> (String, String, i32) {
    let out = Command::new(BIN)
        .arg(fixture)
        .arg("--repo-root")
        .arg(fixture)
        .args(extra)
        .output()
        .expect("spawn find-dup-defs");
    (
        String::from_utf8(out.stdout).expect("stdout is utf-8"),
        String::from_utf8(out.stderr).expect("stderr is utf-8"),
        out.status.code().unwrap_or(-1),
    )
}

/// `directives[]` of a `--json` run, as `(directive, matched, hit-count)` triples.
fn directive_report(extra: &[&str]) -> Vec<(String, Option<i64>, usize)> {
    let doc: serde_json::Value = serde_json::from_str(&run(extra)).expect("stdout is json");
    doc["directives"]
        .as_array()
        .expect("directives[] present")
        .iter()
        .map(|d| {
            (
                d["directive"].as_str().expect("directive text").to_owned(),
                d["matched"].as_i64(),
                d["findings"].as_array().expect("findings[]").len(),
            )
        })
        .collect()
}

#[test]
fn json_reports_a_directive_that_matched() {
    let report = directive_report(&["--json", "-D", "suppress:<constants>DEFAULT_TIMEOUT=live"]);
    assert_eq!(report.len(), 1);
    assert_eq!(report[0].0, "suppress:<constants>DEFAULT_TIMEOUT=live", "canonical text round-trips");
    assert_eq!(report[0].1, Some(1));
    assert_eq!(report[0].2, 1, "the matched finding is named, not just counted");
}

#[test]
fn json_reports_a_dead_directive_as_zero() {
    let report = directive_report(&["--json", "-D", "suppress:<functions>__no_such_definition__=dead"]);
    assert_eq!(report[0].1, Some(0));
    assert!(report[0].2 == 0, "a dead directive names nothing");
}

#[test]
fn set_directive_is_never_dead() {
    // `set:` is pipeline config, not a finding filter. Reporting `0` would read as dead to a
    // consumer auditing a directive file, so it reports null instead.
    let report = directive_report(&["--json", "-D", "set:max-name-group=256"]);
    assert_eq!(report[0].1, None);
}

#[test]
fn json_without_directives_has_no_directives_key() {
    // This is what keeps the default document byte-identical to earlier releases — the golden
    // above would catch a regression, but only obscurely, so state the contract directly.
    let doc: serde_json::Value = serde_json::from_str(&run(&["--json"])).expect("stdout is json");
    assert!(doc.get("directives").is_none());
}

#[test]
fn directive_from_a_file_reports_its_line() {
    // A consumer auditing a directive FILE needs to be told which line it is looking at.
    let path = std::env::temp_dir().join("fdd-directive-origin-test.directives");
    std::fs::write(&path, "# comment\nsuppress:<constants>DEFAULT_TIMEOUT=live\n\nsuppress:<functions>__none__=dead\n")
        .expect("write directive file");
    let doc: serde_json::Value =
        serde_json::from_str(&run(&["--json", "-D", &format!("@{}", path.display())])).expect("stdout is json");
    let origins: Vec<&str> = doc["directives"]
        .as_array()
        .expect("directives[]")
        .iter()
        .map(|d| d["origin"].as_str().expect("origin"))
        .collect();
    let base = path.display().to_string();
    assert_eq!(origins, vec![format!("{base}:2"), format!("{base}:4")], "comments and blanks do not shift the line");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn unknown_kind_exits_nonzero() {
    // Previously an unrecognized `--kinds` value selected no kind at all: the scan collected
    // nothing and the run reported zero findings — green having looked at nothing.
    let (stdout, stderr, code) = run_status(FIX, &["--kinds", "all"]);
    assert_eq!(code, 2, "unknown kind must fail loudly");
    assert!(stderr.contains("unknown kind \"all\""), "stderr names the offending value: {stderr:?}");
    assert!(stderr.contains("functions"), "stderr lists the vocabulary: {stderr:?}");
    assert!(stdout.is_empty(), "no report is emitted for an invalid selection");
}

#[test]
fn known_kind_still_scans() {
    let (stdout, _, _) = run_status(FIX, &["--kinds", "constants", "--json"]);
    let doc: serde_json::Value = serde_json::from_str(&stdout).expect("stdout is json");
    assert!(!doc["groups"].as_array().expect("groups[]").is_empty(), "a valid kind still finds things");
}
