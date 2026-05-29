//! `py-canon` — the Python [`Frontend`](dup_defs_core::Frontend) for `find-dup-defs`, over Ruff's
//! native parser (modern syntax: PEP 695 / 701).
//!
//! [`Python`] walks each file once and lowers every top-level function / class / constant /
//! type-alias and class method to a [`Def`](dup_defs_core::Def), computing its canonical strings
//! off the AST node — a `CPython` `ast.dump`-shaped, name-preserving, docstring-stripped
//! structural canonical (the representation `difflib-fast` clusters). [`ast_canonical`] /
//! [`analyze_functions`] expose that canonicalization over a source string for tooling /
//! golden checks; [`LineMap`] and [`AnalyzedFn`] are the supporting source-location / analysis
//! types.

mod canon;
mod defs;
mod frontend;

pub use canon::{analyze_functions, ast_canonical, ast_canonical_many, normalize_functions, AnalyzedFn};
pub use dup_defs_core::LineMap;
pub use frontend::{Python, CLASSES, CONSTANTS, FUNCTIONS, METHODS, TYPE_ALIASES};
