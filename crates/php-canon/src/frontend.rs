//! The [`Frontend`] implementation for PHP — kind declarations + the file-reading driver.
//!
//! The scan (single tree-sitter parse per file → [`Def`]s with canon precomputed off the syntax tree)
//! lives in [`crate::defs::scan_source`]; this module owns the [`KindSpec`] vocabulary and the `Php`
//! registry entry. Section bases match the other frontends so a mixed-language report keeps one
//! consistent section order.
//!
//! The scan runs on threads with deep stacks ([`dup_defs_core::tree::on_deep_stacks`]): lowering is
//! depth-limited, but machine-written PHP — a generated array of every locale, a concatenation of a
//! whole template — nests an expression far deeper than hand-written code, and a walk that recurses
//! must have room to.

use std::fs;
use std::sync::Arc;

use dup_defs_core::{Def, Frontend, KindSpec, ScanOpts};
use rayon::prelude::*;

use crate::defs::{file_sites, scan_source};

// The shared `KindSpec` vocabulary. PHP has no type aliases.
pub use dup_defs_core::kinds::{CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, LENSES, METHODS};

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &INTERFACES, &CONSTANTS];

/// The kind list with the opt-in `lenses` kind appended — never in the default set, so the default run
/// stays byte-identical (see the Rust frontend).
static KINDS_WITH_LENSES: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &INTERFACES, &CONSTANTS, &LENSES];

/// PHP frontend over the tree-sitter PHP grammar.
pub struct Php;

impl Frontend for Php {
    fn lang(&self) -> &'static str {
        "php"
    }
    /// `.php`, a template's `.phtml`, and the extensions PHP applications give their code besides —
    /// Drupal's `.module` / `.install` / `.theme` / `.profile`, and the `.inc` of an included file. A
    /// file of another language under one of those names has no `<?php` in it, and yields nothing.
    fn extensions(&self) -> &'static [&'static str] {
        &["php", "phtml", "inc", "module", "install", "theme", "profile"]
    }
    fn kinds(&self, opts: &ScanOpts) -> &'static [&'static KindSpec] {
        if opts.wants("lenses") { KINDS_WITH_LENSES } else { KINDS }
    }
    /// The `use` lens counts a name's mentions across the whole set, so in lens mode this frontend takes
    /// every file — and reads each content once (`scan_with_uses`).
    fn scans_across_files(&self, opts: &ScanOpts) -> bool {
        opts.wants("lenses")
    }
    fn scan(&self, files: &[Arc<str>], opts: &ScanOpts) -> Vec<Def> {
        let run = || {
            let read = |src: &str, f: &Arc<str>| scan_source(src, f, opts);
            if opts.wants("lenses") {
                dup_defs_core::uses::scan_with_uses(files, read, file_sites)
            } else {
                files.par_iter().flat_map(|f| fs::read_to_string(&**f).map_or_else(|_| Vec::new(), |src| read(&src, f))).collect()
            }
        };
        dup_defs_core::tree::on_deep_stacks(run)
    }
}

#[cfg(test)]
mod tests {
    use super::Php;
    use dup_defs_core::{Frontend, ScanOpts};
    use std::sync::Arc;

    #[test]
    fn registry_metadata() {
        assert_eq!(Php.lang(), "php");
        assert!(Php.extensions().contains(&"php") && Php.extensions().contains(&"inc"));
        assert_eq!(Php.kinds(&ScanOpts::default()).len(), 5);
        assert!(Php.kinds(&ScanOpts::default()).iter().all(|k| k.id != "lenses"));
    }

    #[test]
    fn a_run_reads_every_file_and_lens_mode_scores_against_the_tree() {
        let dir = std::env::temp_dir().join(format!("php-canon-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("lib")).expect("temp dir");
        let body = "{\n\t$fd = sys_open($path);\n\treturn $fd + 1;\n}\n";
        let files = [
            ("lib/a.php", format!("<?php\nfunction open_file($path)\n{body}")),
            ("lib/b.php", format!("<?php\nfunction open_file($path)\n{body}")),
            ("lib/c.inc", "not php at all: SECTION .text\n".to_owned()),
        ];
        let paths: Vec<Arc<str>> = files
            .iter()
            .map(|(name, src)| {
                let path = dir.join(name);
                std::fs::write(&path, src).expect("fixture");
                Arc::from(path.to_string_lossy().as_ref())
            })
            .collect();
        let defs = Php.scan(&paths, &ScanOpts::default());
        assert_eq!(defs.iter().filter(|d| d.name == "open_file").count(), 2, "two copies, two definitions");
        let kinds = vec!["lenses".to_owned()];
        let lensed = Php.scan(&paths, &ScanOpts { kinds: Some(&kinds), facets: true });
        let record = lensed.iter().find(|d| d.kind.id == "lenses" && d.name == "open_file").expect("a lens record");
        assert!(record.thickness.is_some(), "scored against the corpus");
        std::fs::remove_dir_all(&dir).ok();
    }
}
