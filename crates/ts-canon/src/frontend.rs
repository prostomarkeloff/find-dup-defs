//! The [`Frontend`] implementation for TypeScript — kind declarations + the file-reading driver.
//!
//! The scan (single oxc parse per file → [`Def`]s with canon precomputed) lives in
//! [`crate::defs::scan_source`]; this module owns the [`KindSpec`] vocabulary and the
//! `TypeScript` registry entry. TypeScript declares six kinds (the shared five plus
//! `interfaces`).

use std::fs;
use std::sync::Arc;

use dup_defs_core::{Def, Frontend, KindSpec};
use rayon::prelude::*;

use crate::defs::scan_source;

// The `KindSpec` vocabulary is shared across frontends — re-exported from `find-dup-defs-canon` so callers
// (`crate::frontend::METHODS`, …) are unchanged. TypeScript supports all six kinds.
pub use find_dup_defs_canon::kinds::{CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS, TYPE_ALIASES};

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &INTERFACES, &CONSTANTS, &TYPE_ALIASES];

/// TypeScript frontend over the oxc parser.
pub struct TypeScript;

impl Frontend for TypeScript {
    fn lang(&self) -> &'static str {
        "ts"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["ts", "tsx", "mts", "cts"]
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
    use super::TypeScript;
    use dup_defs_core::Frontend;

    #[test]
    fn registry_metadata() {
        let ts = TypeScript;
        assert_eq!(ts.lang(), "ts");
        assert_eq!(ts.extensions(), &["ts", "tsx", "mts", "cts"]);
        assert_eq!(ts.kinds().len(), 6);
        assert!(ts.kinds().iter().any(|k| k.id == "interfaces"), "interfaces is a TS kind");
    }
}
