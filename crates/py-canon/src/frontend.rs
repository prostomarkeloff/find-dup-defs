//! The [`Frontend`] implementation for Python — kind declarations + the file-reading driver.
//!
//! The actual scan (single Ruff parse per file → [`Def`]s with canon precomputed off the AST
//! nodes) lives in [`crate::defs::scan_source`]; this module owns the [`KindSpec`] vocabulary
//! and the `Python` registry entry. Python declares five kinds; `interfaces` is TypeScript-only.

use std::fs;
use std::sync::Arc;

use dup_defs_core::{Def, Frontend, KindSpec, ScanOpts};
use rayon::prelude::*;

use crate::defs::scan_source;

// The `KindSpec` vocabulary is shared across frontends — re-exported from `find-dup-defs-canon` so callers
// (`crate::frontend::METHODS`, …) are unchanged. Python declares five kinds; `interfaces` is TS-only.
pub use dup_defs_core::kinds::{
    CLASSES, CONSTANTS, FUNCTIONS, LENSES, METHODS, TYPE_ALIASES,
};

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &CONSTANTS, &TYPE_ALIASES];

/// The kind list with the opt-in `lenses` kind appended. It is never in the default set: it costs a
/// second walk of every file plus a per-definition projection, and it answers a question most runs
/// do not ask. Requested by name (`--kinds lenses`), so the choice is a CLI argument rather than a
/// side channel, and the default run stays byte-identical.
static KINDS_WITH_LENSES: &[&KindSpec] =
    &[&FUNCTIONS, &METHODS, &CLASSES, &CONSTANTS, &TYPE_ALIASES, &LENSES];

// Which lenses vote, and the record they stitch into, are the shared module's — re-exported here so
// the frontend's own call sites read the same as they did.
pub(crate) use dup_defs_core::lens::enabled_lenses;

/// Map the extraction's kind string to its `&'static KindSpec`. Internal to the frontend — the
/// engine never does this; it reads `KindSpec` fields directly. Delegates to the shared vocabulary
/// rather than re-listing it: a second copy of the same id→spec table drifts the moment a kind is
/// added to one and not the other.
pub(crate) fn kind_spec(id: &str) -> &'static KindSpec {
    dup_defs_core::kind_spec(id)
        .unwrap_or_else(|| unreachable!("py-canon emitted unknown kind {id:?}"))
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
    fn kinds(&self, opts: &ScanOpts) -> &'static [&'static KindSpec] {
        if opts.wants("lenses") {
            KINDS_WITH_LENSES
        } else {
            KINDS
        }
    }
    fn scan(&self, files: &[Arc<str>], opts: &ScanOpts) -> Vec<Def> {
        let mut defs: Vec<Def> = files
            .par_iter()
            .flat_map(|f| {
                fs::read_to_string(&**f).map_or_else(|_| Vec::new(), |src| scan_source(&src, f, opts))
            })
            .collect();
        // Phase two: the body scan above supplies the anchors (every top-level name it found), so
        // the profile pass needs no resolver — only a second walk of the same files.
        if opts.wants("lenses") {
            // The `use` lens sees a definition from outside, so its facts exist only once every
            // file has been read; scoring is corpus-relative for the same reason.
            let facts = crate::uses::use_facts(files, &defs);
            dup_defs_core::lens::merge_use_facts(&mut defs, facts);
            dup_defs_core::lens::score_lens_defs(&mut defs);
        }
        defs
    }
}

#[cfg(test)]
mod tests {
    use super::Python;
    use dup_defs_core::{Frontend, ScanOpts};

    #[test]
    fn registry_metadata() {
        let py = Python;
        assert_eq!(py.lang(), "py");
        assert_eq!(py.extensions(), &["py"]);
        // Default run: the opt-in profile kind is absent, so no empty section is advertised.
        assert_eq!(py.kinds(&ScanOpts::default()).len(), 5);
        assert!(py.kinds(&ScanOpts::default()).iter().all(|k| k.id != "interfaces"), "interfaces is TS-only");
    }
}
