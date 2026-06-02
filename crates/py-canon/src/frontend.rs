//! The [`Frontend`] implementation for Python — kind declarations + the file-reading driver.
//!
//! The actual scan (single Ruff parse per file → [`Def`]s with canon precomputed off the AST
//! nodes) lives in [`crate::defs::scan_source`]; this module owns the [`KindSpec`] vocabulary
//! and the `Python` registry entry. Python declares five kinds; `interfaces` is TypeScript-only.

use std::fs;
use std::sync::Arc;

use dup_defs_core::{Def, Frontend, KindSpec};
use rayon::prelude::*;

use crate::defs::scan_source;

// The `KindSpec` vocabulary is shared across frontends — re-exported from `canon-core` so callers
// (`crate::frontend::METHODS`, …) are unchanged. Python declares five kinds; `interfaces` is TS-only.
pub use canon_core::kinds::{CLASSES, CONSTANTS, FUNCTIONS, METHODS, TYPE_ALIASES};

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &CONSTANTS, &TYPE_ALIASES];

/// Map the extraction's kind string to its `&'static KindSpec`. Internal to the frontend — the
/// engine never does this; it reads `KindSpec` fields directly.
pub(crate) fn kind_spec(id: &str) -> &'static KindSpec {
    match id {
        "functions" => &FUNCTIONS,
        "methods" => &METHODS,
        "classes" => &CLASSES,
        "constants" => &CONSTANTS,
        "type-aliases" => &TYPE_ALIASES,
        other => unreachable!("py-canon emitted unknown kind {other:?}"),
    }
}

/// Python frontend over Ruff's parser.
pub struct Python;

impl Frontend for Python {
    fn lang(&self) -> &'static str {
        "py"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["py"]
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
    use super::Python;
    use dup_defs_core::Frontend;

    #[test]
    fn registry_metadata() {
        let py = Python;
        assert_eq!(py.lang(), "py");
        assert_eq!(py.extensions(), &["py"]);
        assert_eq!(py.kinds().len(), 5);
        assert!(py.kinds().iter().all(|k| k.id != "interfaces"), "interfaces is TS-only");
    }
}
