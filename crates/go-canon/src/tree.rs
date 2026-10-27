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

/// Parse one Go source. `None` only when the grammar cannot be loaded: tree-sitter recovers from a
/// syntax error *inside* the tree (an `ERROR` node), so a bad declaration costs that declaration
/// rather than the file.
pub(crate) fn parse(src: &str) -> Option<Tree> {
    PARSER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_go::LANGUAGE.into()).ok()?;
            *slot = Some(parser);
        }
        slot.as_mut()?.parse(src, None)
    })
}

pub(crate) use dup_defs_core::tree::{field, fields, kids, text, unquote};

/// Whether `n` holds an anonymous token spelled `tok` — how the grammar tells `for k := range` from
/// `for k = range`, and `case v := <-ch` from `case v = <-ch`: the operator is not a field.
pub(crate) fn has_token(n: Node<'_>, tok: &str) -> bool {
    let mut cursor = n.walk();
    let found = n.children(&mut cursor).any(|k| !k.is_named() && k.kind() == tok);
    found
}

/// The statements a block or a `case` clause holds — through its `statement_list` — with empty
/// statements dropped. A clause with no statements has no list at all.
pub(crate) fn statements(n: Node<'_>) -> Vec<Node<'_>> {
    kids(n)
        .into_iter()
        .filter(|k| k.kind() == "statement_list")
        .flat_map(kids)
        .filter(|k| k.kind() != "empty_statement")
        .collect()
}
