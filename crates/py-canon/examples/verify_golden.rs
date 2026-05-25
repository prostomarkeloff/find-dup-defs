//! Cross-check: the function canonicals this (ported) crate produces must match, byte-for-byte, the
//! golden corpus dumped by the original iilint `canon.rs` (same source, pyo3 stripped). Validates the
//! port didn't change canonicalization.
//!
//! Usage: `cargo run --release --example verify_golden -- <repo_dir> <golden.canon.bin>`
use py_canon::{ast_canonical_many, find_module_defs};
use walkdir::WalkDir;

fn main() {
    let repo = std::env::args().nth(1).expect("usage: verify_golden <repo_dir> <golden.canon.bin>");
    let golden_path = std::env::args().nth(2).expect("usage: verify_golden <repo_dir> <golden.canon.bin>");

    let mut files: Vec<String> = WalkDir::new(&repo)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "py"))
        .map(|e| e.path().to_string_lossy().into_owned())
        .collect();
    files.sort();

    let defs = find_module_defs(&files);
    let texts: Vec<String> = defs.iter().filter(|d| d.kind == "functions").map(|d| d.text.clone()).collect();
    let mut canon: Vec<String> = ast_canonical_many(&texts).into_iter().filter(|c| !c.is_empty()).collect();

    let golden_bytes = std::fs::read(&golden_path).expect("read golden");
    let mut golden: Vec<String> = golden_bytes
        .split(|&b| b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect();

    canon.sort();
    golden.sort();
    let ok = canon == golden;
    println!("py-canon functions: {}   golden: {}   multiset identical: {ok}", canon.len(), golden.len());
    if !ok {
        let first = canon.iter().zip(&golden).position(|(a, b)| a != b);
        println!("first differing index (sorted): {first:?}");
        let only_ours = canon.iter().filter(|c| !golden.contains(c)).count();
        let only_golden = golden.iter().filter(|c| !canon.contains(c)).count();
        println!("canonicals only in ours: {only_ours}, only in golden: {only_golden}");
        std::process::exit(1);
    }
}
