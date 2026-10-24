//! The [`Frontend`] implementation for Go — kind declarations + the file-reading driver.
//!
//! The scan (single tree-sitter parse per file → [`Def`]s with canon precomputed off the syntax tree)
//! lives in [`crate::defs::scan_source`]; this module owns the [`KindSpec`] vocabulary and the `Go`
//! registry entry. Section bases match the other frontends so a mixed-language report keeps one
//! consistent section order.

use std::fs;
use std::sync::Arc;

use dup_defs_core::{Def, Frontend, KindSpec, ScanOpts};
use rayon::prelude::*;

use crate::defs::scan_source;

// The shared `KindSpec` vocabulary. Go supports all six body/text kinds (`interfaces` = `interface`).
pub use dup_defs_core::kinds::{CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, LENSES, METHODS, TYPE_ALIASES};

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &INTERFACES, &CONSTANTS, &TYPE_ALIASES];

/// The kind list with the opt-in `lenses` kind appended — never in the default set, so the default
/// run stays byte-identical (see the Rust frontend).
static KINDS_WITH_LENSES: &[&KindSpec] =
    &[&FUNCTIONS, &METHODS, &CLASSES, &INTERFACES, &CONSTANTS, &TYPE_ALIASES, &LENSES];

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
    fn scan(&self, files: &[Arc<str>], opts: &ScanOpts) -> Vec<Def> {
        if opts.wants("lenses") {
            return dup_defs_core::uses::scan_with_uses(files, |src, f| scan_source(src, f, opts), crate::canon::collect_sites);
        }
        files
            .par_iter()
            .flat_map(|f| fs::read_to_string(&**f).map_or_else(|_| Vec::new(), |src| scan_source(&src, f, opts)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Go;
    use dup_defs_core::{Frontend, ScanOpts};

    #[test]
    fn registry_metadata() {
        let go = Go;
        assert_eq!(go.lang(), "go");
        assert_eq!(go.extensions(), &["go"]);
        assert_eq!(go.kinds(&ScanOpts::default()).len(), 6);
        assert!(go.kinds(&ScanOpts::default()).iter().all(|k| k.id != "lenses"));
    }
}
