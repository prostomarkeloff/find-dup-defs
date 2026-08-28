//! The shared [`KindSpec`] vocabulary — the kinds every frontend may emit, declared once.
//!
//! Each frontend exposes the subset it supports (Python has no `interfaces`) via its own `KINDS`
//! slice, but the specs themselves are identical, so they live here rather than in three copies that
//! drift the moment a kind is added to one and not the others.

use crate::KindSpec;

/// Top-level functions (`def f`, `fn f`, `function f`).
pub static FUNCTIONS: KindSpec =
    KindSpec { id: "functions", label: "FUNCTION", noun_plural: "functions", section: 1, body: true, fn_like: true };
/// Methods, qualified `Type.method` / `Type::method`.
pub static METHODS: KindSpec =
    KindSpec { id: "methods", label: "METHOD", noun_plural: "methods", section: 4, body: true, fn_like: true };
/// Body-bearing nominal types (`class` / `struct` / `enum` / `union`).
pub static CLASSES: KindSpec =
    KindSpec { id: "classes", label: "CLASS", noun_plural: "classes", section: 7, body: true, fn_like: false };
/// The interface analog (TypeScript `interface`, Rust `trait`) — not emitted by `py-canon`.
pub static INTERFACES: KindSpec =
    KindSpec { id: "interfaces", label: "INTERFACE", noun_plural: "interfaces", section: 8, body: true, fn_like: false };
/// `UPPER_SNAKE` module/namespace constants (`const` / `static`).
pub static CONSTANTS: KindSpec =
    KindSpec { id: "constants", label: "CONSTANT", noun_plural: "constants", section: 0, body: false, fn_like: false };
/// **Use profile** — a definition canonicalized by its *use sites* rather than its body: the
/// multiset of statements across the tree that mention its name, alpha-renamed with the
/// definition itself as the anchor (`_t0`) and its attribute accesses as positional `_a{n}`.
/// Where the body kinds answer "what is this thing", this one answers "how is it handled" —
/// so two subsystems whose *bodies* diverged but whose *handling* is identical (the same
/// primitive re-invented) cluster together. `fn_like` so the cross-name and Type-3 passes run
/// over it; `section` sits after every body kind.
pub static USE_PROFILES: KindSpec =
    KindSpec { id: "use-profiles", label: "USE_PROFILE", noun_plural: "use profiles", section: 10, body: true, fn_like: true };
/// **Lens consensus** — one definition seen through every enabled lens at once, stitched into a
/// single record. Each lens contributes its facts under its own prefix (`control:if`,
/// `outgoing:.commit`), so the Type-3 pass's IDF-weighted cosine *is* the vote: agreeing through
/// several lenses raises the score, agreeing through one barely moves it, and a fact the whole
/// corpus shares (`control:return`) is weighted to nothing without anyone naming it as noise.
/// A cross-name exact match here means every lens agreed at once.
pub static LENSES: KindSpec = KindSpec {
    id: "lenses",
    label: "LENSES",
    noun_plural: "lens consensus",
    section: 20,
    body: true,
    fn_like: true,
};
/// `type X = …` aliases (note the space in `noun_plural`, distinct from the hyphenated `id`).
pub static TYPE_ALIASES: KindSpec =
    KindSpec { id: "type-aliases", label: "TYPE_ALIAS", noun_plural: "type aliases", section: 9, body: false, fn_like: false };

/// Every kind id [`kind_spec`] resolves — the vocabulary a caller may name (`--kinds`). Kept
/// adjacent to the match below so the two are edited together; `kind_ids_all_resolve` fails if an
/// id here stops resolving.
pub const KIND_IDS: &[&str] =
    &["functions", "methods", "classes", "interfaces", "constants", "type-aliases", "lenses"];

/// Map a kind `id` to its shared [`KindSpec`]. A frontend that doesn't support a kind simply never
/// passes its id here.
#[must_use]
pub fn kind_spec(id: &str) -> Option<&'static KindSpec> {
    Some(match id {
        "functions" => &FUNCTIONS,
        "methods" => &METHODS,
        "classes" => &CLASSES,
        "interfaces" => &INTERFACES,
        "constants" => &CONSTANTS,
        "type-aliases" => &TYPE_ALIASES,
        "lenses" => &LENSES,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{kind_spec, KIND_IDS};

    #[test]
    fn kind_ids_all_resolve() {
        for id in KIND_IDS {
            assert!(kind_spec(id).is_some(), "KIND_IDS lists {id:?} but kind_spec does not resolve it");
        }
    }

    #[test]
    fn unknown_kind_does_not_resolve() {
        // The vocabulary is closed: `all` reads like a wildcard but is not one, and resolving it
        // to nothing is exactly how a caller ends up scanning an empty kind set.
        for id in ["all", "*", "function", "Functions", ""] {
            assert!(kind_spec(id).is_none(), "{id:?} must not resolve to a kind");
        }
    }
}
