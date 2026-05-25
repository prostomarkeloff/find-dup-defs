//! `py-canon` — `CPython` `ast.dump`-shape **canonicalization** of Python source, plus a parallel
//! **module-level definition scan**, both over Ruff's native parser (modern syntax: PEP 695 / 701).
//!
//! - [`find_module_defs`] walks a set of files → top-level functions / classes / constants /
//!   type-aliases as [`ModuleDef`]s (kind, name, file, location, source text).
//! - [`ast_canonical`] / [`ast_canonical_many`] reduce a definition's source to a name-preserving,
//!   docstring-stripped structural canonical — the representation `difflib-fast` clusters to find
//!   near-duplicate definitions.

mod canon;
mod defs;
mod loc;

pub use canon::{analyze_functions, ast_canonical, ast_canonical_many, normalize_functions, AnalyzedFn};
pub use defs::{find_module_defs, ModuleDef};
pub use loc::LineMap;
