//! `rs-canon` — the Rust [`Frontend`](dup_defs_core::Frontend) for `find-dup-defs`, over `syn`.
//!
//! [`Rust`] scans `.rs` files and lowers every free `fn`, `impl`/trait method, `struct`/`enum`/
//! `union`, `trait`, `const`/`static`, and `type` alias to a [`Def`](dup_defs_core::Def) with a
//! structural s-expr canonical precomputed off the AST node. [`AnalyzedFn`] is the supporting
//! analysis tuple.

mod canon;
mod defs;
mod frontend;

pub use canon::AnalyzedFn;
pub use frontend::{Rust, CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS, TYPE_ALIASES};
