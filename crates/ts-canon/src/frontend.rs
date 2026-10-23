//! The [`Frontend`] implementation for TypeScript — kind declarations + the file-reading driver.
//!
//! The scan (single oxc parse per file → [`Def`]s with canon precomputed) lives in
//! [`crate::defs::scan_source`]; this module owns the [`KindSpec`] vocabulary and the
//! `TypeScript` registry entry. TypeScript declares six kinds (the shared five plus
//! `interfaces`).

use std::fs;
use std::sync::Arc;

use dup_defs_core::{Def, Frontend, KindSpec, ScanOpts};
use rayon::prelude::*;

use crate::defs::scan_source;

// The `KindSpec` vocabulary is shared across frontends — re-exported from `find-dup-defs-canon` so callers
// (`crate::frontend::METHODS`, …) are unchanged. TypeScript supports all six kinds.
pub use dup_defs_core::kinds::{CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, LENSES, METHODS, TYPE_ALIASES};

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &INTERFACES, &CONSTANTS, &TYPE_ALIASES];

/// The kind list with the opt-in `lenses` kind appended. Never in the default set: it costs a second
/// projection per definition and answers a question most runs do not ask. Requested by name
/// (`--kinds lenses`), so the choice is a CLI argument rather than a side channel and the default run
/// stays byte-identical.
static KINDS_WITH_LENSES: &[&KindSpec] =
    &[&FUNCTIONS, &METHODS, &CLASSES, &INTERFACES, &CONSTANTS, &TYPE_ALIASES, &LENSES];

/// TypeScript frontend over the oxc parser.
pub struct TypeScript;

impl Frontend for TypeScript {
    fn lang(&self) -> &'static str {
        "ts"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["ts", "tsx", "mts", "cts"]
    }
    fn kinds(&self, opts: &ScanOpts) -> &'static [&'static KindSpec] {
        if opts.wants("lenses") { KINDS_WITH_LENSES } else { KINDS }
    }
    /// The `use` lens counts a name's mentions across the whole set, so in lens mode this frontend
    /// takes every file — and reads each content once, see `scan_with_uses`.
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
    use super::TypeScript;
    use dup_defs_core::{Frontend, ScanOpts};

    #[test]
    fn registry_metadata() {
        let ts = TypeScript;
        assert_eq!(ts.lang(), "ts");
        assert_eq!(ts.extensions(), &["ts", "tsx", "mts", "cts"]);
        assert_eq!(ts.kinds(&ScanOpts::default()).len(), 6);
        // The opt-in kind is absent from the default set, so no empty section is advertised.
        assert!(ts.kinds(&ScanOpts::default()).iter().all(|k| k.id != "lenses"));
        assert!(ts.kinds(&ScanOpts::default()).iter().any(|k| k.id == "interfaces"), "interfaces is a TS kind");
    }
}
