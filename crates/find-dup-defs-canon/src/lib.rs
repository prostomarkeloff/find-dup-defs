//! Shared canonicalization helpers reused verbatim by every `*-canon` frontend (`py-canon`,
//! `rs-canon`, `ts-canon`): the [`KindSpec`] vocabulary, the alpha-rename map, and a couple of
//! string utilities.
//!
//! This crate sits *between* the contract ([`dup_defs_core`]) and the language frontends: it
//! `depends on` the contract and is `depended on by` the frontends, so the three of them don't each
//! hand-maintain an identical copy — without polluting the pure contract crate (which the engine
//! also depends on) with frontend implementation. The per-language *unparsers* (the `syn`/Ruff/oxc
//! AST walkers) stay in their own crates; only the language-agnostic pieces live here.

use std::collections::{HashMap, HashSet};

use dup_defs_core::KindSpec;

/// The result of analyzing one callable: `(cluster-canonical, xname-canonical, type3-lines, size)`.
/// Identical across the three frontends — they differ only in how they *produce* the strings.
pub type AnalyzedFn = (String, String, Vec<String>, usize);

/// Non-blank line count of a def's source text — the simplest "how big" metric the report surfaces.
/// Blank/whitespace-only lines (including the line after a multi-line signature) are excluded so a
/// def with a deliberately spaced-out body doesn't read as twice as big as an equivalent dense one.
#[must_use]
pub fn count_loc(text: &str) -> usize {
    text.lines().filter(|l| !l.trim().is_empty()).count()
}

/// True for a non-empty `UPPER_SNAKE` name — `[A-Z0-9_]+` — the constant-naming convention shared by
/// Python module constants, Rust `const`/`static`, and TypeScript `const`.
#[must_use]
pub fn is_upper_snake(name: &str) -> bool {
    !name.is_empty()
        && name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// Alpha-rename a *bound local* to its canonical `_vN` slot (stable within one definition), leaving
/// free names (types, called functions, imported constants) verbatim. `map` tracks the per-def
/// name→slot assignment; `locals` is the set of names treated as bound (params, `let`/`for`/pattern
/// bindings). The single rename rule the three unparsers used to each re-implement.
#[must_use]
#[allow(clippy::cast_possible_truncation)] // a def's distinct bound-name count is far below u32::MAX
#[allow(clippy::implicit_hasher)] // always called with the frontends' std-hasher maps
pub fn alpha_rename(map: &mut HashMap<String, u32>, locals: Option<&HashSet<String>>, name: &str) -> String {
    if let Some(locals) = locals {
        if locals.contains(name) {
            let next = map.len() as u32;
            let slot = *map.entry(name.to_owned()).or_insert(next);
            return format!("_v{slot}");
        }
    }
    name.to_owned()
}

/// The shared [`KindSpec`] vocabulary. Each frontend exposes the subset it supports (Python has no
/// `interfaces`) via its own `KINDS` slice, but the specs themselves are identical, so they live
/// here once. `section` reproduces the engine's historical report ordering.
pub mod kinds {
    use dup_defs_core::KindSpec;

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
}

/// Map a kind `id` to its shared [`KindSpec`]. A frontend that doesn't support a kind simply never
/// passes its id here.
#[must_use]
pub fn kind_spec(id: &str) -> Option<&'static KindSpec> {
    Some(match id {
        "functions" => &kinds::FUNCTIONS,
        "methods" => &kinds::METHODS,
        "classes" => &kinds::CLASSES,
        "interfaces" => &kinds::INTERFACES,
        "constants" => &kinds::CONSTANTS,
        "type-aliases" => &kinds::TYPE_ALIASES,
        "lenses" => &kinds::LENSES,
        _ => return None,
    })
}
