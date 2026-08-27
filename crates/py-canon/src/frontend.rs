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
pub use find_dup_defs_canon::kinds::{
    CLASSES, CONSTANTS, FUNCTIONS, LENSES, METHODS, TYPE_ALIASES,
};

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &CONSTANTS, &TYPE_ALIASES];

/// The kind list with the opt-in `lenses` kind appended. It is never in the default set: it costs a
/// second walk of every file plus a per-definition projection, and it answers a question most runs
/// do not ask. Requested by name (`--kinds lenses`), so the choice is a CLI argument rather than a
/// side channel, and the default run stays byte-identical.
static KINDS_WITH_LENSES: &[&KindSpec] =
    &[&FUNCTIONS, &METHODS, &CLASSES, &CONSTANTS, &TYPE_ALIASES, &LENSES];

/// Lenses that vote when the `lenses` kind is asked for. All of them: a lens is a weight on one
/// scale rather than a separate question, and the corpus IDF already silences the ones a given tree
/// has nothing to say through.
pub(crate) fn enabled_lenses(opts: &ScanOpts) -> Vec<crate::lenses::Lens> {
    if opts.wants("lenses") {
        crate::lenses::Lens::all().to_vec()
    } else {
        Vec::new()
    }
}

/// Map the extraction's kind string to its `&'static KindSpec`. Internal to the frontend — the
/// engine never does this; it reads `KindSpec` fields directly. Delegates to the shared vocabulary
/// rather than re-listing it: a second copy of the same id→spec table drifts the moment a kind is
/// added to one and not the other.
pub(crate) fn kind_spec(id: &str) -> &'static KindSpec {
    find_dup_defs_canon::kind_spec(id)
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
            crate::lenses::merge_use_facts(&mut defs, facts);
            crate::lenses::score_lens_defs(&mut defs);
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
