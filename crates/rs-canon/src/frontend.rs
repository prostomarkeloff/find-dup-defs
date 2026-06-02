//! The [`Frontend`] implementation for Rust — kind declarations + the file-reading driver.
//!
//! The scan (single `syn` parse per file → [`Def`]s with canon precomputed off the AST node)
//! lives in [`crate::defs::scan_source`]; this module owns the [`KindSpec`] vocabulary and the
//! `Rust` registry entry. Section bases match the other frontends so a mixed-language report
//! keeps one consistent section order.

use std::fs;
use std::sync::Arc;

use dup_defs_core::{Def, Frontend, KindSpec};
use rayon::prelude::*;

use crate::defs::scan_source;

// The `KindSpec` vocabulary is shared across frontends — re-exported from `canon-core` so callers
// (`crate::frontend::METHODS`, …) are unchanged. Rust supports all six kinds (`interfaces` = `trait`).
pub use canon_core::kinds::{CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS, TYPE_ALIASES};

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &INTERFACES, &CONSTANTS, &TYPE_ALIASES];

/// Rust frontend over the `syn` parser.
pub struct Rust;

impl Frontend for Rust {
    fn lang(&self) -> &'static str {
        "rs"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["rs"]
    }
    fn kinds(&self) -> &'static [&'static KindSpec] {
        KINDS
    }
    fn scan(&self, files: &[Arc<str>]) -> Vec<Def> {
        files
            .par_iter()
            .flat_map(|f| fs::read_to_string(&**f).map_or_else(|_| Vec::new(), |src| scan_source(&src, f)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Rust;
    use dup_defs_core::Frontend;

    #[test]
    fn registry_metadata() {
        let rs = Rust;
        assert_eq!(rs.lang(), "rs");
        assert_eq!(rs.extensions(), &["rs"]);
        assert_eq!(rs.kinds().len(), 6);
    }
}
