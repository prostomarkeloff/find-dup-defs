//! `ts-canon` — the TypeScript [`Frontend`](dup_defs_core::Frontend) for `find-dup-defs`, over
//! [`oxc_parser`]. [`TypeScript`] scans `.ts` / `.tsx` / `.mts` / `.cts` files and lowers every
//! definition to a [`Def`](dup_defs_core::Def) with a structural s-expr canonical precomputed.
//! [`ast_canonical`] / [`analyze_functions`] expose that canonicalization over a source string;
//! [`AnalyzedFn`] is the supporting analysis tuple.

mod canon;
mod defs;
mod frontend;

pub use canon::{analyze_functions, ast_canonical, ast_canonical_many, AnalyzedFn};
pub use dup_defs_core::LineMap;
pub use frontend::{TypeScript, CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS, TYPE_ALIASES};

#[cfg(test)]
mod test_helpers {
    // Kept here so per-module tests don't need to duplicate the helper.
}

