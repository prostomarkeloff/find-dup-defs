//! `go-canon` — the Go [`Frontend`](dup_defs_core::Frontend) for `find-dup-defs`, over tree-sitter.
//!
//! [`Go`] scans `.go` files and lowers every package-level function, method (`Type.method`), struct,
//! interface, constant and other type declaration to a [`Def`](dup_defs_core::Def) with a structural
//! s-expr canonical precomputed off the syntax tree.

mod build;
mod canon;
mod decides;
mod defs;
mod frontend;
mod ir;
mod lenses;
mod normal;
mod tree;

pub use defs::scan_source;
pub use frontend::{Go, CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS, TYPE_ALIASES};
