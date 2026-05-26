//! `py-canon` — `CPython` `ast.dump`-shape **canonicalization** of Python source, plus a parallel
//! **module-level definition scan**, both over Ruff's native parser (modern syntax: PEP 695 / 701).
//!
//! - [`find_module_defs`] walks a set of files → top-level functions / classes / constants /
//!   type-aliases as [`ModuleDef`]s (kind, name, file, location, source text).
//! - [`ast_canonical`] / [`ast_canonical_many`] reduce a definition's source to a name-preserving,
//!   docstring-stripped structural canonical — the representation `difflib-fast` clusters to find
//!   near-duplicate definitions.
//!
//! Shared types ([`ModuleDef`], [`AnalyzedFn`], [`LineMap`]) live in the language-agnostic
//! `dup-defs-core` crate so the TypeScript frontend (`ts-canon`) produces the exact same shapes
//! and the engine can route by [`dup_defs_core::Language`] without per-frontend adapters. They
//! are re-exported here so existing `py_canon::ModuleDef` / `py_canon::LineMap` import paths keep
//! working.

mod canon;
mod defs;

pub use canon::{analyze_functions, ast_canonical, ast_canonical_many, normalize_functions};
pub use defs::find_module_defs;
pub use dup_defs_core::{AnalyzedFn, LineMap, ModuleDef};
