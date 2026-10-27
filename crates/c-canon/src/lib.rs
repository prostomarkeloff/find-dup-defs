//! `c-canon` — the C [`Frontend`](dup_defs_core::Frontend) for `find-dup-defs`, over tree-sitter.
//!
//! [`C`] scans `.c` and `.h` files and lowers every file-scope function, record (`struct` / `union` /
//! `enum`), table of function pointers, constant (`#define` with a value, `const` object with an
//! initializer) and other `typedef` to a [`Def`](dup_defs_core::Def) with a structural s-expr canonical
//! precomputed off the syntax tree.

mod build;
mod canon;
mod decides;
mod defs;
mod frontend;
mod headers;
mod ir;
mod lenses;
mod normal;
mod prep;
mod tree;

pub use defs::scan_source;
pub use frontend::{C, CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, TYPE_ALIASES};
