//! tree-sitter plumbing: one parser per thread, and the node accessors every walk in this crate goes
//! through.
//!
//! The PHP grammar has two kinds of *extras* — nodes that may sit between any two others as named
//! siblings: comments, and the stretch of a template between `?>` and the next `<?php`
//! (`text_interpolation`). The shared accessors drop the first; the ones here drop both, because a walk
//! handed a template stretch as a child would read it as an operand. The one place a template stretch
//! means something — between two statements, where it is output — reads it on purpose
//! ([`crate::ir`]).

use std::cell::RefCell;

use tree_sitter::{Node, Parser, Tree};

pub(crate) use dup_defs_core::tree::{field, text, unquote};
use dup_defs_core::tree::{fields_except, kids_except};

thread_local! {
    /// A `Parser` is not `Sync`; the scan runs on a rayon pool, so each worker keeps its own.
    static PARSER: RefCell<Option<Parser>> = const { RefCell::new(None) };
}

/// Parse one PHP source — a file as the interpreter reads it, HTML around `<?php … ?>` included.
/// `None` only when the grammar cannot be loaded: tree-sitter recovers from a syntax error *inside* the
/// tree (an `ERROR` node), so a bad declaration costs that declaration rather than the file.
pub(crate) fn parse(src: &str) -> Option<Tree> {
    PARSER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_php::LANGUAGE_PHP.into()).ok()?;
            *slot = Some(parser);
        }
        slot.as_mut()?.parse(src, None)
    })
}

/// The extras every walk reads past: comments, and template stretches.
const EXTRAS: &[&str] = &["comment", "text_interpolation"];

/// The named children of `n`, extras dropped.
pub(crate) fn kids(n: Node<'_>) -> Vec<Node<'_>> {
    kids_except(n, EXTRAS)
}

/// The named children of `n` with its template stretches kept — a statement list, where one is output.
pub(crate) use dup_defs_core::tree::kids as statements_of;

/// Every named child stored under a grammar field, extras dropped.
pub(crate) fn fields<'t>(n: Node<'t>, name: &str) -> Vec<Node<'t>> {
    fields_except(n, name, EXTRAS)
}

/// The text of the first named child of a kind.
pub(crate) fn child_of_kind<'t>(n: Node<'t>, kind: &str) -> Option<Node<'t>> {
    kids(n).into_iter().find(|k| k.kind() == kind)
}

/// A variable's name without its `$`: `$order` → `order`. A dynamic one (`$$name`, `${expr}`) has none.
pub(crate) fn var_name<'s>(n: Node<'_>, src: &'s str) -> Option<&'s str> {
    (n.kind() == "variable_name").then(|| text(n, src).trim_start_matches('$'))
}

/// A name as written, with whatever whitespace older PHP allowed between its segments taken out
/// (`App \ Models` is `App\Models`) and the leading `\` of a fully qualified name kept.
pub(crate) fn name_of(n: Node<'_>, src: &str) -> String {
    text(n, src).chars().filter(|c| !c.is_whitespace()).collect()
}
