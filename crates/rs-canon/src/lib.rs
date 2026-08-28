//! `rs-canon` — the Rust [`Frontend`](dup_defs_core::Frontend) for `find-dup-defs`, over `syn`.
//!
//! [`Rust`] scans `.rs` files and lowers every free `fn`, `impl`/trait method, `struct`/`enum`/
//! `union`, `trait`, `const`/`static`, and `type` alias to a [`Def`](dup_defs_core::Def) with a
//! structural s-expr canonical precomputed off the AST node. [`AnalyzedFn`] is the supporting
//! analysis record.
//!
//! The `lenses` kind is opt-in (`--kinds lenses`) and answers the ten perspective questions off the
//! same `syn` walk; the vocabulary and the record it stitches into are shared with every other
//! frontend in `find-dup-defs-canon`.

mod canon;
mod defs;
mod frontend;
mod lenses;

pub use canon::AnalyzedFn;
pub use frontend::{Rust, CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS, TYPE_ALIASES};
