//! `java-canon` — the Java [`Frontend`](dup_defs_core::Frontend) for `find-dup-defs`, over tree-sitter.
//!
//! [`Java`] scans `.java` files and lowers every method and constructor, every class, interface,
//! enum, record and annotation declaration, and every constant (`static final` field) to a
//! [`Def`](dup_defs_core::Def) with a structural s-expr canonical precomputed off the syntax tree.

mod canon;
mod decides;
mod defs;
mod frontend;
mod ir;
mod lenses;
mod normal;
mod tree;

pub use defs::scan_source;
pub use frontend::{Java, CLASSES, CONSTANTS, INTERFACES, METHODS};
