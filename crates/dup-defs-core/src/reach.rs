//! Building the dotted paths a definition **reaches**, in the one form the engine reads.
//!
//! Every language spells a path its own way — `a.b.c`, `crate::a::b`, `./a/b` — and every language's
//! frontend is the only thing that knows which of its names are paths at all. What the engine needs
//! is a single spelling it can take prefixes of, because the prefix lattice over the module tree is
//! how "imported the module" and "imported a member of it" are made to meet.
//!
//! So the split is: the frontend decides what the segments *are*, this decides how they are written.
//! Nothing here parses a path — passing a whole `a::b` through as one segment would produce a node
//! no other language's path can ever match, which is precisely the drift this exists to prevent.

use std::sync::Arc;

/// Join path segments into the engine's dotted form.
///
/// Empty segments are dropped rather than producing `a..b`: a leading `::` in Rust and a leading `./`
/// in TypeScript both split into an empty first segment, and neither means a nameless module.
#[must_use]
pub fn reach_path<S: AsRef<str>>(segments: &[S]) -> Arc<str> {
    let joined: Vec<&str> =
        segments.iter().map(AsRef::as_ref).filter(|s| !s.is_empty()).collect();
    Arc::from(joined.join("."))
}

/// Every prefix of a dotted path, longest first — the module tree read as a lattice.
///
/// `from a.b import c` and `from a.b.c import d` name different paths and never meet as strings; as
/// prefix sets they meet at `a.b.c`, which is the thing they are both about. Which nodes matter is
/// not decided here: the corpus decides, by how many definitions reach each one.
pub fn prefixes(path: &str) -> impl Iterator<Item = &str> {
    let mut at = Some(path.len());
    std::iter::from_fn(move || {
        let end = at?;
        let slice = &path[..end];
        at = slice.rfind('.').filter(|dot| *dot > 0);
        Some(slice)
    })
}

#[cfg(test)]
mod tests {
    use super::{prefixes, reach_path};

    #[test]
    fn segments_join_with_dots_whatever_the_language_used() {
        assert_eq!(&*reach_path(&["a", "b", "c"]), "a.b.c");
    }

    #[test]
    fn an_empty_segment_is_dropped_not_written() {
        // `::a::b` and `./a/b` both split with an empty head; neither means a nameless module.
        assert_eq!(&*reach_path(&["", "a", "b"]), "a.b");
        assert_eq!(&*reach_path(&["a", "", "b"]), "a.b");
    }

    #[test]
    fn prefixes_walk_the_tree_from_the_leaf_up() {
        assert_eq!(prefixes("a.b.c").collect::<Vec<_>>(), vec!["a.b.c", "a.b", "a"]);
        assert_eq!(prefixes("a").collect::<Vec<_>>(), vec!["a"]);
        assert_eq!(prefixes("").collect::<Vec<_>>(), vec![""]);
    }

    #[test]
    fn a_leading_dot_is_not_a_prefix_boundary() {
        // A relative Python import keeps its dots so two packages' `.models` stay distinct; the walk
        // must not slice one of those into an empty node.
        assert_eq!(prefixes(".pkg.x").collect::<Vec<_>>(), vec![".pkg.x", ".pkg"]);
    }
}
