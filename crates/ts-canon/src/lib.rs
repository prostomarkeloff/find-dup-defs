//! `ts-canon` — TypeScript frontend for `find-dup-defs`. Mirrors `py-canon`'s contract over
//! [`oxc_parser`]: file walking + module-level definition scan (`.ts` / `.tsx` / `.mts` / `.cts`)
//! plus a structural canonicalization of each definition's source. The same shapes
//! ([`ModuleDef`], [`AnalyzedFn`]) feed the engine's three duplicate-detection passes uniformly.

mod canon;
mod defs;
mod frontend;

pub use canon::{analyze_functions, ast_canonical, ast_canonical_many};
pub use defs::find_module_defs;
pub use dup_defs_core::{AnalyzedFn, Language, LineMap, ModuleDef};
pub use frontend::{TypeScript, CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS, TYPE_ALIASES};

#[cfg(test)]
mod test_helpers {
    // Kept here so per-module tests don't need to duplicate the helper.
}

