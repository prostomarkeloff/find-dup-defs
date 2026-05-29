//! Scan one file and print per-def progress (kind / name / location / canon length), so a
//! segfault or panic narrows to a specific definition.

use std::env;
use std::sync::Arc;

use dup_defs_core::Frontend;
use ts_canon::TypeScript;

fn main() {
    let file = env::args().nth(1).expect("usage: canon_one <file>");
    let files = vec![Arc::<str>::from(file.as_str())];
    let defs = TypeScript.scan(&files);
    eprintln!("found {} defs in {file}", defs.len());
    for (i, d) in defs.iter().enumerate() {
        let canon_len = d.cluster_canonical.as_deref().map_or(0, str::len);
        eprintln!("[{i}] {} {} {}:{} loc={} canon_len={canon_len}", d.kind.id, d.name, d.file, d.line + 1, d.loc);
    }
    eprintln!("done");
}
