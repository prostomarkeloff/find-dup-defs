//! The [`Frontend`] implementation for Go — kind declarations + the file-reading driver.
//!
//! The scan (single tree-sitter parse per file → [`Def`]s with canon precomputed off the syntax tree)
//! lives in [`crate::defs::scan_source`]; this module owns the [`KindSpec`] vocabulary and the `Go`
//! registry entry. Section bases match the other frontends so a mixed-language report keeps one
//! consistent section order.

use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex};

use dup_defs_core::{Def, Frontend, KindSpec, ScanOpts};
use rayon::prelude::*;

use crate::build::{alternative_unit, collapse_alternatives};
use crate::defs::scan_source;

// The shared `KindSpec` vocabulary. Go supports all six body/text kinds (`interfaces` = `interface`).
pub use dup_defs_core::kinds::{CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, LENSES, METHODS, TYPE_ALIASES};

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &INTERFACES, &CONSTANTS, &TYPE_ALIASES];

/// The kind list with the opt-in `lenses` kind appended — never in the default set, so the default
/// run stays byte-identical (see the Rust frontend).
static KINDS_WITH_LENSES: &[&KindSpec] =
    &[&FUNCTIONS, &METHODS, &CLASSES, &INTERFACES, &CONSTANTS, &TYPE_ALIASES, &LENSES];

/// What each file read so far was built as: the package it is one alternative within, or `None` for a
/// file every build compiles.
type Units = HashMap<Arc<str>, Option<(String, String)>>;

/// Go frontend over the tree-sitter Go grammar.
pub struct Go;

impl Frontend for Go {
    fn lang(&self) -> &'static str {
        "go"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["go"]
    }
    fn kinds(&self, opts: &ScanOpts) -> &'static [&'static KindSpec] {
        if opts.wants("lenses") { KINDS_WITH_LENSES } else { KINDS }
    }
    /// The `use` lens counts a name's mentions across the whole set, so in lens mode this frontend
    /// takes every file — and reads each content once (`scan_with_uses`).
    fn scans_across_files(&self, opts: &ScanOpts) -> bool {
        opts.wants("lenses")
    }
    /// Each file is read on its own; then, among the files built only in some configurations, a
    /// name a package declares in two of them is reported once (see [`crate::build`]).
    ///
    /// A file's unit needs its source (its build constraint is a comment in it), so it is noted as each
    /// file is read — and a file the lens mode never reads, because its bytes repeat another's and its
    /// definitions were replayed onto it, has its source read for that alone.
    fn scan(&self, files: &[Arc<str>], opts: &ScanOpts) -> Vec<Def> {
        let units: Mutex<Units> = Mutex::new(HashMap::new());
        let read = |src: &str, f: &Arc<str>| {
            let unit = alternative_unit(src, f);
            units.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(Arc::clone(f), unit);
            scan_source(src, f, opts)
        };
        let defs = if opts.wants("lenses") {
            dup_defs_core::uses::scan_with_uses(files, read, crate::canon::collect_sites)
        } else {
            files.par_iter().flat_map(|f| fs::read_to_string(&**f).map_or_else(|_| Vec::new(), |src| read(&src, f))).collect()
        };
        let units = units.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
        collapse_alternatives(defs, |file| match units.get(file) {
            Some(unit) => unit.clone(),
            None => fs::read_to_string(file).ok().and_then(|src| alternative_unit(&src, file)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Go;
    use dup_defs_core::{Frontend, ScanOpts};
    use std::sync::Arc;

    #[test]
    fn registry_metadata() {
        let go = Go;
        assert_eq!(go.lang(), "go");
        assert_eq!(go.extensions(), &["go"]);
        assert_eq!(go.kinds(&ScanOpts::default()).len(), 6);
        assert!(go.kinds(&ScanOpts::default()).iter().all(|k| k.id != "lenses"));
    }

    #[test]
    fn a_name_two_build_alternatives_declare_is_one_definition() {
        let dir = std::env::temp_dir().join(format!("go-canon-build-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("cmd")).expect("temp dir");
        let body = "func Default(method string) int {\n\tn := len(method)\n\treturn n + 1\n}\n\nfunc init() {\n\tregister(1)\n}\n";
        let files = [
            ("binding.go", format!("//go:build !nomsgpack\n\npackage binding\n\n{body}")),
            ("binding_nomsgpack.go", format!("//go:build nomsgpack\n\npackage binding\n\n{body}")),
            ("poll_windows.go", format!("package binding\n\n{body}")),
            // Two alternatives whose bytes are one: lens mode parses the first and replays the second.
            ("twin_linux.go", "package binding\n\nfunc Twin(x int) int {\n\ty := x * 3\n\treturn y\n}\n".to_owned()),
            ("twin_darwin.go", "package binding\n\nfunc Twin(x int) int {\n\ty := x * 3\n\treturn y\n}\n".to_owned()),
            // Every build compiles these two, so a repeated name is a real copy.
            ("plain.go", "package binding\n\nfunc Copied(x int) int {\n\ty := x * 2\n\treturn y\n}\n".to_owned()),
            ("plain2.go", "package binding\n\nfunc Copied(x int) int {\n\ty := x * 2\n\treturn y\n}\n".to_owned()),
            // `go run` programs, one per file: no build includes them, none is another's alternative.
            ("cmd/a.go", "//go:build ignore\n\npackage main\n\nfunc main() {\n\tx := run()\n\tprint(x)\n}\n".to_owned()),
            ("cmd/b.go", "//go:build ignore\n\npackage main\n\nfunc main() {\n\tx := run()\n\tprint(x)\n}\n".to_owned()),
        ];
        let paths: Vec<Arc<str>> = files
            .iter()
            .map(|(name, src)| {
                let path = dir.join(name);
                std::fs::write(&path, src).expect("fixture");
                Arc::from(path.to_string_lossy().as_ref())
            })
            .collect();
        let defs = Go.scan(&paths, &ScanOpts::default());
        let count = |name: &str| defs.iter().filter(|d| d.name == name).count();
        assert_eq!(count("Default"), 1, "three build alternatives, one definition");
        assert_eq!(count("init"), 3, "a package holds as many `init`s as it likes");
        assert_eq!(count("Copied"), 2);
        assert_eq!(count("main"), 2);
        // In lens mode a file whose bytes repeat another's is parsed once and replayed: the
        // alternatives still collapse.
        let kinds = vec!["lenses".to_owned()];
        let lensed = Go.scan(&paths, &ScanOpts { kinds: Some(&kinds), facets: true });
        assert_eq!(lensed.iter().filter(|d| d.kind.id == "functions" && d.name == "Default").count(), 1);
        assert_eq!(count("Twin"), 1);
        assert_eq!(lensed.iter().filter(|d| d.kind.id == "functions" && d.name == "Twin").count(), 1, "a replayed twin is an alternative too");
        std::fs::remove_dir_all(&dir).ok();
    }
}
