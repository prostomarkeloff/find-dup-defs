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
pub use dup_defs_core::kinds::{CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS, TYPE_ALIASES};

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &INTERFACES, &CONSTANTS, &TYPE_ALIASES];

/// Go frontend over the tree-sitter Go grammar.
pub struct Go;

impl Frontend for Go {
    fn lang(&self) -> &'static str {
        "go"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["go"]
    }
    fn kinds(&self, _opts: &ScanOpts) -> &'static [&'static KindSpec] {
        KINDS
    }
    fn scan(&self, files: &[Arc<str>], opts: &ScanOpts) -> Vec<Def> {
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
