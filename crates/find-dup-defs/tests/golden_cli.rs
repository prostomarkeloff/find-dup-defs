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
