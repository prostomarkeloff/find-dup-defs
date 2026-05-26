//! Run `find_module_defs` + `ast_canonical` on each emitted def's text, printing per-def
//! progress so a segfault narrows to a specific def.

use std::env;

use ts_canon::{ast_canonical, find_module_defs};

fn main() {
    let file = env::args().nth(1).expect("usage: canon_one <file>");
    let defs = find_module_defs(&[file.clone()]);
    eprintln!("found {} defs in {}", defs.len(), file);
    for (i, d) in defs.iter().enumerate() {
        eprintln!("[{i}] {} {} {}:{} loc={}", d.kind, d.name, d.file, d.line + 1, d.loc);
        let c = ast_canonical(&d.text);
        eprintln!("    canon_len={}", c.len());
    }
    eprintln!("done");
}
