//! Shared types for `find-dup-defs` frontends — language-agnostic. Both `py-canon` (Python) and
//! `ts-canon` (TypeScript) depend on this crate and produce the same [`ModuleDef`] / [`AnalyzedFn`]
//! shapes, so the `find-dup-defs` engine can treat all definitions uniformly while dispatching
//! canonicalization back to the right frontend by [`Language`] tag.

mod loc;
mod types;

pub use loc::LineMap;
pub use types::{Analysis, AnalyzedFn, Def, Frontend, KindSpec, Language, ModuleDef};
