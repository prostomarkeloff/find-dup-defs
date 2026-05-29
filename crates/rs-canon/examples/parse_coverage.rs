//! Parse-coverage + robustness probe over a corpus: walks `<dir>` for `.rs` files (sorted) and,
//! for each, prints its path to stderr (flushed) BEFORE parsing — so if `syn` stack-overflows on
//! a pathological file the last stderr line names the culprit — then counts parse-OK / parse-FAIL
//! and total defs.
//!
//! Usage: `cargo run --release --example parse_coverage -- <dir>`
#![allow(clippy::cast_precision_loss)] // a coverage percentage doesn't need full f64 precision
use std::io::Write;
use std::sync::Arc;

use rs_canon::Rust;
use walkdir::WalkDir;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: parse_coverage <dir>");
    let mut files: Vec<String> = WalkDir::new(&dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "rs"))
        .map(|e| e.path().to_string_lossy().into_owned())
        .collect();
    files.sort();

    let (mut ok, mut fail) = (0u32, 0u32);
    let stderr = std::io::stderr();
    for f in &files {
        // Flush the path first so an abort (stack overflow is not catchable) leaves it visible.
        let _ = writeln!(stderr.lock(), "PARSE {f}");
        let _ = stderr.lock().flush();
        match std::fs::read_to_string(f) {
            Ok(src) => match syn::parse_file(&src) {
                Ok(_) => ok += 1,
                Err(_) => fail += 1,
            },
            Err(_) => fail += 1,
        }
    }
    // Count defs via the real frontend (skips files that fail to parse).
    let arcs: Vec<Arc<str>> = files.iter().map(|f| Arc::from(f.as_str())).collect();
    let defs = <Rust as dup_defs_core::Frontend>::scan(&Rust, &arcs).len();

    println!(
        "files={} parse_ok={ok} parse_fail={fail} ({:.2}% fail) defs={defs}",
        files.len(),
        100.0 * f64::from(fail) / files.len().max(1) as f64
    );
}
