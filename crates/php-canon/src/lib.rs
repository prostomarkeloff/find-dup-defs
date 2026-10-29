//! `php-canon` — the PHP [`Frontend`](dup_defs_core::Frontend) for `find-dup-defs`, over tree-sitter.
//!
//! [`Php`] scans `.php` files (and the other extensions PHP code is kept under) and lowers every
//! file-scope function, every method of a class, trait or enum, every class-like declaration, and every
//! constant (`const`, `define`) to a [`Def`](dup_defs_core::Def) with a structural s-expr canonical
//! precomputed off the syntax tree.

mod canon;
mod decides;
mod defs;
mod frontend;
mod ir;
mod lenses;
mod names;
mod normal;
mod tree;

pub use defs::scan_source;
pub use frontend::{Php, CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS};
