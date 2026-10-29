//! The node accessors every tree-sitter frontend reads its parse through, and the limits every one of
//! them walks under.
//!
//! The Go, C and PHP frontends all parse with tree-sitter and walk the result the same way: named
//! children with the grammar's comments dropped, a child by field, the text a node spans. The grammars
//! differ; how a node is read does not, so it is written once. Behind the `tree-sitter` feature, so a
//! frontend over another parser pulls in nothing.
//!
//! The grammar's comments are *extras* — they sit between any two nodes as named siblings — so every
//! accessor that hands out children drops them. A walk that forgot to would read a doc comment as a
//! statement. A grammar may have extras of its own (PHP's template stretches); the `_except` accessors
//! take the list.
//!
//! Machine-written C and PHP nest a statement or an expression far deeper than any hand-written code —
//! a table's sum, an `else if` ladder of every opcode, a generated array of every locale — and every walk
//! downstream of a parse recurses. So lowering is depth-limited ([`Level`]), and the scan runs on threads
//! with room for the depth it allows ([`on_deep_stacks`]).

use std::cell::Cell;
use std::sync::LazyLock;

use tree_sitter::Node;

/// The named children of `n`, comments dropped.
#[must_use]
pub fn kids(n: Node<'_>) -> Vec<Node<'_>> {
    kids_except(n, &["comment"])
}

/// The named children of `n`, the grammar's extras of the kinds in `extras` dropped.
#[must_use]
pub fn kids_except<'t>(n: Node<'t>, extras: &[&str]) -> Vec<Node<'t>> {
    let mut cursor = n.walk();
    n.named_children(&mut cursor).filter(|k| !extras.contains(&k.kind())).collect()
}

/// The child stored under a grammar field.
#[must_use]
pub fn field<'t>(n: Node<'t>, name: &str) -> Option<Node<'t>> {
    n.child_by_field_name(name)
}

/// Every named child stored under a grammar field (`name` repeats in Go's `a, b int`, `declarator` in
/// C's `int a, *b;`).
#[must_use]
pub fn fields<'t>(n: Node<'t>, name: &str) -> Vec<Node<'t>> {
    fields_except(n, name, &["comment"])
}

/// Every named child stored under a grammar field, the grammar's extras of the kinds in `extras`
/// dropped.
#[must_use]
pub fn fields_except<'t>(n: Node<'t>, name: &str, extras: &[&str]) -> Vec<Node<'t>> {
    let mut cursor = n.walk();
    n.children_by_field_name(name, &mut cursor).filter(|k| k.is_named() && !extras.contains(&k.kind())).collect()
}

/// The source text a node spans.
#[must_use]
pub fn text<'s>(n: Node<'_>, src: &'s str) -> &'s str {
    src.get(n.start_byte()..n.end_byte()).unwrap_or("")
}

/// A literal's contents without its delimiters: `"a"`, `` `a` `` and `'a'` → `a`, and the encoding
/// prefixes off first — C's (`L"a"`, `u8"a"`) and PHP's binary one (`b"a"`). Text that is no quoted
/// literal is handed back whole: a prefix is one only before a quote.
#[must_use]
pub fn unquote(s: &str) -> &str {
    let body = s.trim_start_matches(['L', 'u', 'U', '8', 'b', 'B']);
    let mut chars = body.chars();
    match (chars.next(), chars.next_back()) {
        (Some(open @ ('"' | '`' | '\'')), Some(close)) if open == close && body.len() >= 2 => &body[1..body.len() - 1],
        _ => s,
    }
}

/// How deep a frontend's lowering follows nested statements and expressions before writing the rest as
/// one opaque node.
pub const MAX_DEPTH: usize = 200;

thread_local! {
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// One level of lowering depth, released when dropped; [`Level::enter`] refuses past [`MAX_DEPTH`].
pub struct Level;

impl Level {
    /// Enter one more level, or `None` when the walk is already [`MAX_DEPTH`] deep.
    #[must_use]
    pub fn enter() -> Option<Self> {
        DEPTH.with(|d| {
            if d.get() >= MAX_DEPTH {
                None
            } else {
                d.set(d.get() + 1);
                Some(Level)
            }
        })
    }
}

impl Drop for Level {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// The stack each deep-walking thread gets: room for a walk [`MAX_DEPTH`] levels deep, every level a few
/// frames of a recursive lowering and the canonicals written from it.
const STACK: usize = 64 << 20;

/// The pool the deep walks run on — as many threads as the global pool, each with [`STACK`].
static POOL: LazyLock<Option<rayon::ThreadPool>> =
    LazyLock::new(|| rayon::ThreadPoolBuilder::new().stack_size(STACK).thread_name(|i| format!("deep-walk-{i}")).build().ok());

/// Run `work` on the deep-stack pool — every `par_iter` inside it on threads with room for the walk —
/// or on the caller's pool when one could not be built.
pub fn on_deep_stacks<R: Send>(work: impl FnOnce() -> R + Send) -> R {
    match POOL.as_ref() {
        Some(pool) => pool.install(work),
        None => work(),
    }
}

#[cfg(test)]
mod tests {
    use super::{unquote, Level, MAX_DEPTH};

    #[test]
    fn a_literal_loses_its_delimiters_and_its_encoding_prefix() {
        assert_eq!(unquote("\"a\""), "a");
        assert_eq!(unquote("`raw`"), "raw");
        assert_eq!(unquote("'x'"), "x");
        assert_eq!(unquote("u8\"wide\""), "wide");
        assert_eq!(unquote("L'w'"), "w");
        assert_eq!(unquote("b'bytes'"), "bytes");
        assert_eq!(unquote("bare"), "bare");
        assert_eq!(unquote("Uber8"), "Uber8", "a word that starts like a prefix is no literal");
    }

    #[test]
    fn a_level_is_refused_past_the_limit_and_released_when_dropped() {
        let held: Vec<Level> = (0..MAX_DEPTH).map_while(|_| Level::enter()).collect();
        assert_eq!(held.len(), MAX_DEPTH);
        assert!(Level::enter().is_none(), "one past the limit");
        drop(held);
        assert!(Level::enter().is_some(), "every level was released");
    }
}
