//! The `find-dup-defs` shared ground: the frontend↔engine contract, and the pieces every frontend
//! and the engine would otherwise each hand-maintain a copy of.
//!
//! A frontend parses each file once, classifies its definitions, and lowers each to a [`Def`] — a
//! flat feature record carrying the precomputed canonical strings the clustering engine consumes.
//! The engine never sees a frontend's rich per-language representation and never matches on a fixed
//! kind vocabulary: each frontend declares its own kinds as `&'static` [`KindSpec`]s.
//!
//! This used to be two crates — a pure contract and a `find-dup-defs-canon` sitting between it and
//! the language frontends — on the reasoning that the engine, which depends on the contract, should
//! not also pull in frontend implementation. The perspective passes ended that: the engine reads
//! [`reach::prefixes`] to walk the module tree the frontends' [`Facets::reaches`] names, and
//! [`lens`] needs [`Def`] to build a record. The boundary had holes in both directions, and a
//! boundary that does not hold is a version to bump and a publish order to get right for nothing.
//!
//! The per-language *unparsers* (the `syn` / Ruff / oxc walkers) stay in their own crates; only the
//! language-agnostic pieces live here.

mod canon;
mod loc;
mod types;

pub mod kinds;

pub mod lens;
pub mod reach;

pub use canon::{alpha_rename, count_loc, is_upper_snake, AnalyzedFn};
pub use kinds::kind_spec;
pub use loc::LineMap;
pub use types::{Analysis, CanonDialect, Def, Facets, Frontend, KindSpec, ScanOpts, Statement};
