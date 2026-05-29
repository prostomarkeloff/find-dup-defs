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

fn run(extra: &[&str]) -> String {
    let out = Command::new(BIN)
        .arg(FIX)
        .arg("--repo-root")
        .arg(FIX)
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
