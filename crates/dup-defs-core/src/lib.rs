//! The `find-dup-defs` frontend↔engine contract. Every frontend (`py-canon`, `ts-canon`, …)
//! depends on this crate, declares its kinds as [`KindSpec`]s, and lowers each definition to a
//! [`Def`] via the [`Frontend`] trait — so the engine clusters duplicates without knowing any
//! language's construct types. [`ModuleDef`] / [`AnalyzedFn`] are the frontends' shared
//! extraction intermediate, not part of the engine contract.

mod loc;
mod types;

pub use loc::LineMap;
pub use types::{Analysis, AnalyzedFn, Def, Frontend, KindSpec, ModuleDef};
