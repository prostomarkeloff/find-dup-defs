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

/// Parse one Java source. `None` only when the grammar cannot be loaded: tree-sitter recovers from a
/// syntax error *inside* the tree (an `ERROR` node), so a bad declaration costs that declaration
/// rather than the file.
pub(crate) fn parse(src: &str) -> Option<Tree> {
    PARSER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_java::LANGUAGE.into()).ok()?;
            *slot = Some(parser);
        }
        slot.as_mut()?.parse(src, None)
    })
}

pub(crate) use dup_defs_core::tree::{field, text, unquote};

/// The Java grammar's comment kinds — two, where Go's grammar names one. The shared `kids` drops a
/// `comment` node; a walk over a Java tree must drop `line_comment` and `block_comment` too, or a doc
/// comment reads as a statement.
const EXTRAS: &[&str] = &["comment", "line_comment", "block_comment"];

/// The named children of `n`, the grammar's comments dropped.
pub(crate) fn kids(n: Node<'_>) -> Vec<Node<'_>> {
    dup_defs_core::tree::kids_except(n, EXTRAS)
}

/// Every named child stored under a grammar field, the grammar's comments dropped.
pub(crate) fn fields<'t>(n: Node<'t>, name: &str) -> Vec<Node<'t>> {
    dup_defs_core::tree::fields_except(n, name, EXTRAS)
}

/// The statements a block holds, with empty statements dropped. A `switch` clause's body is its
/// `switch_block_statement_group` children — the labels are walked by the caller, not here.
pub(crate) fn statements(n: Node<'_>) -> Vec<Node<'_>> {
    kids(n).into_iter().filter(|k| k.kind() != "empty_statement").collect()
}
