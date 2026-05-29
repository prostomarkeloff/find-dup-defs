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

/// Free `fn foo(...)`.
pub static FUNCTIONS: KindSpec =
    KindSpec { id: "functions", label: "FUNCTION", noun_plural: "functions", section: 1, body: true, fn_like: true };
/// `impl` methods and trait default methods, qualified `Type::method`.
pub static METHODS: KindSpec =
    KindSpec { id: "methods", label: "METHOD", noun_plural: "methods", section: 4, body: true, fn_like: true };
/// `struct` / `enum` / `union` — body-bearing nominal types.
pub static CLASSES: KindSpec =
    KindSpec { id: "classes", label: "CLASS", noun_plural: "classes", section: 7, body: true, fn_like: false };
/// `trait` — its associated-item shape (the interface analog).
pub static INTERFACES: KindSpec =
    KindSpec { id: "interfaces", label: "INTERFACE", noun_plural: "interfaces", section: 8, body: true, fn_like: false };
/// `const` / `static` with an `UPPER_SNAKE` name.
pub static CONSTANTS: KindSpec =
    KindSpec { id: "constants", label: "CONSTANT", noun_plural: "constants", section: 0, body: false, fn_like: false };
/// `type X = ...` (note the space in `noun_plural`, distinct from the hyphenated `id`).
pub static TYPE_ALIASES: KindSpec =
    KindSpec { id: "type-aliases", label: "TYPE_ALIAS", noun_plural: "type aliases", section: 9, body: false, fn_like: false };

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
