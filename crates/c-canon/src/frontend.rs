//! The [`Frontend`] implementation for C — kind declarations + the file-reading driver.
//!
//! The scan (single tree-sitter parse per file → [`Def`]s with canon precomputed off the syntax tree)
//! lives in [`crate::defs::scan_file`]; this module owns the [`KindSpec`] vocabulary and the `C`
//! registry entry. Section bases match the other frontends so a mixed-language report keeps one
//! consistent section order.
//!
//! A run reads its files twice: once together, for what they say about each other — the attribute
//! macros the corpus defines, what its headers declare ([`crate::defs::Corpus`]) — and then each on its
//! own. Both run on threads with deep stacks ([`dup_defs_core::tree::on_deep_stacks`]): lowering is
//! depth-limited, but machine-written C still nests a statement or an expression far deeper than any
//! hand-written code in the other languages, and a walk that recurses must have room to.

use std::fs;
use std::sync::Arc;

use dup_defs_core::{Def, Frontend, KindSpec, ScanOpts};
use rayon::prelude::*;

use crate::build::{alternative_unit, collapse_alternatives};
use crate::defs::{file_sites, scan_file, Corpus};

// The shared `KindSpec` vocabulary. C has no methods: a function called through a member is a call
// through a function pointer, and the table it sits in is the interface.
pub use dup_defs_core::kinds::{CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, LENSES, TYPE_ALIASES};

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &CLASSES, &INTERFACES, &CONSTANTS, &TYPE_ALIASES];

/// The kind list with the opt-in `lenses` kind appended — never in the default set, so the default
/// run stays byte-identical (see the Rust frontend).
static KINDS_WITH_LENSES: &[&KindSpec] = &[&FUNCTIONS, &CLASSES, &INTERFACES, &CONSTANTS, &TYPE_ALIASES, &LENSES];

/// C frontend over the tree-sitter C grammar.
pub struct C;

impl Frontend for C {
    fn lang(&self) -> &'static str {
        "c"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["c", "h"]
    }
    fn kinds(&self, opts: &ScanOpts) -> &'static [&'static KindSpec] {
        if opts.wants("lenses") { KINDS_WITH_LENSES } else { KINDS }
    }
    /// The `use` lens counts a name's mentions across the whole set, so in lens mode this frontend
    /// takes every file — and reads each content once (`scan_with_uses`).
    fn scans_across_files(&self, opts: &ScanOpts) -> bool {
        opts.wants("lenses")
    }
    /// Each file is read on its own against what the run's files say about each other; then, among
    /// the files built only in some configurations, a name two of one unit define is reported once
    /// (see [`crate::build`]). A file's unit is read off its path alone, so a file whose bytes repeat
    /// another's — parsed once in lens mode, its definitions replayed onto it — has its own.
    fn scan(&self, files: &[Arc<str>], opts: &ScanOpts) -> Vec<Def> {
        let run = || {
            let corpus = Corpus::build(files);
            let read = |src: &str, f: &Arc<str>| scan_file(src, f, opts, &corpus);
            let defs = if opts.wants("lenses") {
                dup_defs_core::uses::scan_with_uses(files, read, |src, _path, known| file_sites(src, known, &corpus))
            } else {
                files.par_iter().flat_map(|f| fs::read_to_string(&**f).map_or_else(|_| Vec::new(), |src| read(&src, f))).collect()
            };
            collapse_alternatives(defs, alternative_unit)
        };
        dup_defs_core::tree::on_deep_stacks(run)
    }
}

#[cfg(test)]
mod tests {
    use super::C;
    use dup_defs_core::{Frontend, ScanOpts};
    use std::sync::Arc;

    #[test]
    fn registry_metadata() {
        assert_eq!(C.lang(), "c");
        assert_eq!(C.extensions(), &["c", "h"]);
        assert_eq!(C.kinds(&ScanOpts::default()).len(), 5);
        assert!(C.kinds(&ScanOpts::default()).iter().all(|k| k.id != "lenses"));
    }

    #[test]
    fn a_run_learns_its_attribute_macros_resolves_its_headers_and_collapses_platform_files() {
        let dir = std::env::temp_dir().join(format!("c-canon-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for sub in ["include", "src/os/unix", "src/os/win32"] {
            std::fs::create_dir_all(dir.join(sub)).expect("temp dir");
        }
        let body = "{\n\tint fd = sys_open(path);\n\treturn fd + 1;\n}\n";
        let files = [
            ("include/attr.h", "#ifdef __GNUC__\n#define my_cold __attribute__((cold))\n#else\n#define my_cold\n#endif\n".to_owned()),
            ("include/mem.h", "void *xmalloc(unsigned long n);\n".to_owned()),
            ("src/a.c", "#include \"mem.h\"\n#include \"attr.h\"\nstatic my_cold int setup(int n)\n{\n\tchar *p = xmalloc(n);\n\treturn use(p);\n}\n".to_owned()),
            ("src/os/unix/file.c", format!("int open_file(const char *path)\n{body}")),
            ("src/os/win32/file.c", format!("int open_file(const char *path)\n{body}")),
        ];
        let paths: Vec<Arc<str>> = files
            .iter()
            .map(|(name, src)| {
                let path = dir.join(name);
                std::fs::write(&path, src).expect("fixture");
                Arc::from(path.to_string_lossy().as_ref())
            })
            .collect();
        let defs = C.scan(&paths, &ScanOpts::default());
        assert_eq!(defs.iter().filter(|d| d.name == "open_file").count(), 1, "one per platform unit");
        let setup = defs.iter().find(|d| d.name == "setup").expect("setup");
        assert!(setup.cluster_canonical.as_deref().is_some_and(|c| c.contains("Ty('int', )")), "the learned attribute was erased");
        let reaches: Vec<&str> = setup.facets.reaches.iter().map(|r| &**r).collect();
        assert_eq!(reaches, vec!["include.mem"]);
        // In lens mode every file is handed over and read against the same corpus: the lens records
        // come out alongside the body definitions, scored against the tree.
        let kinds = vec!["lenses".to_owned()];
        let lensed = C.scan(&paths, &ScanOpts { kinds: Some(&kinds) });
        let record = lensed.iter().find(|d| d.kind.id == "lenses" && d.name == "setup").expect("a lens record for setup");
        assert!(record.thickness.is_some(), "scored against the corpus");
        assert_eq!(lensed.iter().filter(|d| d.kind.id == "functions" && d.name == "open_file").count(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
