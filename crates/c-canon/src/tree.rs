//! tree-sitter plumbing: one parser per thread, and the handful of node accessors every walk in this
//! crate goes through.
//!
//! The grammar's comments are *extras* — they sit between any two nodes as named siblings — so every
//! accessor that hands out children drops them. A walk that forgot to would read a doc comment as a
//! statement.

use std::cell::RefCell;

use tree_sitter::{Node, Parser, Tree};

thread_local! {
    /// A `Parser` is not `Sync`; the scan runs on a rayon pool, so each worker keeps its own.
    static PARSER: RefCell<Option<Parser>> = const { RefCell::new(None) };
}

/// Parse one C source. `None` only when the grammar cannot be loaded: tree-sitter recovers from a
/// syntax error *inside* the tree (an `ERROR` node), so a bad declaration costs that declaration
/// rather than the file.
pub(crate) fn parse(src: &str) -> Option<Tree> {
    PARSER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_c::LANGUAGE.into()).ok()?;
            *slot = Some(parser);
        }
        slot.as_mut()?.parse(src, None)
    })
}

pub(crate) use dup_defs_core::tree::{field, fields, kids, text, unquote};

/// A node's text with every run of whitespace written as one space — how a name a macro builds
/// (`FUNC( put )`) or a multi-word type (`unsigned   long`) is spelled the same wherever it is written.
pub(crate) fn squashed(n: Node<'_>, src: &str) -> String {
    text(n, src).split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether a statement was cut short by a `;` the parser had to invent: a macro used as a statement
/// keyword (`list_for_each(p, head) { … }`) leaves exactly that.
pub(crate) fn missing_semicolon(n: Node<'_>) -> bool {
    let count = n.child_count();
    count > 0 && n.child(count - 1).is_some_and(|last| last.is_missing())
}
