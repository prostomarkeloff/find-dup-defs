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

// Section bases reproduce the engine's historical ordering: constants(0), functions(1→1/2/3 via
// pass offset), methods(4→4/5/6), classes(7), type-aliases(9). `interfaces`(8) is TS-only.
/// `def foo(...)` — top-level functions.
pub static FUNCTIONS: KindSpec =
    KindSpec { id: "functions", label: "FUNCTION", noun_plural: "functions", section: 1, body: true, fn_like: true };
/// Class methods, qualified `Class.method`.
pub static METHODS: KindSpec =
    KindSpec { id: "methods", label: "METHOD", noun_plural: "methods", section: 4, body: true, fn_like: true };
/// `class Foo: ...`.
pub static CLASSES: KindSpec =
    KindSpec { id: "classes", label: "CLASS", noun_plural: "classes", section: 7, body: true, fn_like: false };
/// Module-level `UPPER_CASE = ...` constants.
pub static CONSTANTS: KindSpec =
    KindSpec { id: "constants", label: "CONSTANT", noun_plural: "constants", section: 0, body: false, fn_like: false };
/// PEP 695 `type X = ...` aliases (note the space in `noun_plural`, distinct from the `id`).
pub static TYPE_ALIASES: KindSpec =
    KindSpec { id: "type-aliases", label: "TYPE_ALIAS", noun_plural: "type aliases", section: 9, body: false, fn_like: false };

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
