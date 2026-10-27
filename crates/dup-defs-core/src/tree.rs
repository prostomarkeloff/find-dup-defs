//! The node accessors every tree-sitter frontend reads its parse through.
//!
//! The Go and C frontends both parse with tree-sitter and walk the result the same way: named
//! children with the grammar's comments dropped, a child by field, the text a node spans. The grammars
//! differ; how a node is read does not, so it is written once. Behind the `tree-sitter` feature, so a
//! frontend over another parser pulls in nothing.
//!
//! The grammar's comments are *extras* — they sit between any two nodes as named siblings — so every
//! accessor that hands out children drops them. A walk that forgot to would read a doc comment as a
//! statement.

use tree_sitter::Node;

/// The named children of `n`, comments dropped.
#[must_use]
pub fn kids(n: Node<'_>) -> Vec<Node<'_>> {
    let mut cursor = n.walk();
    n.named_children(&mut cursor).filter(|k| k.kind() != "comment").collect()
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
    let mut cursor = n.walk();
    n.children_by_field_name(name, &mut cursor).filter(|k| k.is_named() && k.kind() != "comment").collect()
}

/// The source text a node spans.
#[must_use]
pub fn text<'s>(n: Node<'_>, src: &'s str) -> &'s str {
    src.get(n.start_byte()..n.end_byte()).unwrap_or("")
}

/// A literal's contents without its delimiters: `"a"`, `` `a` `` and `'a'` → `a`, and C's encoding
/// prefixes (`L"a"`, `u8"a"`) off first.
#[must_use]
pub fn unquote(s: &str) -> &str {
    let s = s.trim_start_matches(['L', 'u', 'U', '8']);
    let mut chars = s.chars();
    match (chars.next(), chars.next_back()) {
        (Some(open @ ('"' | '`' | '\'')), Some(close)) if open == close && s.len() >= 2 => &s[1..s.len() - 1],
        _ => s,
    }
}

#[cfg(test)]
mod tests {
    use super::unquote;

    #[test]
    fn a_literal_loses_its_delimiters_and_its_encoding_prefix() {
        assert_eq!(unquote("\"a\""), "a");
        assert_eq!(unquote("`raw`"), "raw");
        assert_eq!(unquote("'x'"), "x");
        assert_eq!(unquote("u8\"wide\""), "wide");
        assert_eq!(unquote("L'w'"), "w");
        assert_eq!(unquote("bare"), "bare");
    }
}
