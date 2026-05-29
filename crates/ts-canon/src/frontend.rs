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

/// `function foo(...)` / `const foo = (...) => {}`.
pub static FUNCTIONS: KindSpec =
    KindSpec { id: "functions", label: "FUNCTION", noun_plural: "functions", section: 1, body: true, fn_like: true };
/// Class methods, qualified `Class.method`.
pub static METHODS: KindSpec =
    KindSpec { id: "methods", label: "METHOD", noun_plural: "methods", section: 4, body: true, fn_like: true };
/// `class Foo { ... }`.
pub static CLASSES: KindSpec =
    KindSpec { id: "classes", label: "CLASS", noun_plural: "classes", section: 7, body: true, fn_like: false };
/// `interface X { ... }` — first-class kind so directives can target it independently.
pub static INTERFACES: KindSpec =
    KindSpec { id: "interfaces", label: "INTERFACE", noun_plural: "interfaces", section: 8, body: true, fn_like: false };
/// Module-level `const NAME = ...` with an `UPPER_SNAKE` name and a non-function initializer.
pub static CONSTANTS: KindSpec =
    KindSpec { id: "constants", label: "CONSTANT", noun_plural: "constants", section: 0, body: false, fn_like: false };
/// `type X = ...` (note the space in `noun_plural`, distinct from the hyphenated `id`).
pub static TYPE_ALIASES: KindSpec =
    KindSpec { id: "type-aliases", label: "TYPE_ALIAS", noun_plural: "type aliases", section: 9, body: false, fn_like: false };

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
