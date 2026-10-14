//! Canonicalization helpers reused verbatim by every frontend: the analysis record every one of
//! them produces, the alpha-rename rule, and a couple of string utilities.

use std::collections::{HashMap, HashSet};

use crate::Statement;

/// The result of analyzing one callable. Identical across the three frontends — they differ only in
/// how they *produce* the fields.
///
/// It was a bare 4-tuple, which is why `statements` did not exist: adding a fifth slot to an
/// anonymous tuple reads as noise at every destructuring site, so the block structure the walk
/// already knew was thrown away and every pass that needed it had to re-derive or do without.
/// Named, the field is a question the type asks each frontend.
#[derive(Clone, Debug, Default)]
pub struct AnalyzedFn {
    /// Names-preserved structural canonical — what the name-gated pass clusters on.
    pub cluster_canonical: String,
    /// Alpha-renamed structural canonical — what the cross-name pass buckets on.
    pub xname_canonical: String,
    /// Per-statement renamed lines — the Type-3 pass's shingles.
    pub type3_lines: Vec<String>,
    /// The type references the annotations name, in order — filled only by a frontend whose
    /// language attaches no behaviour to an annotation, which writes them as `_T{n}` slots in the
    /// lines above ([`type_slot`]).
    ///
    /// A statement line used to carry two facts at once — what the statement does and which types
    /// its annotations name — so a changed annotation made the whole line a different unit, and two
    /// near-copies typed differently shared nothing on their signature. Erasing the types instead
    /// threw away the agreement of two copies that do share them, which was often most of what they
    /// shared. Reported apart, each is its own fact: the statement agrees whatever the types, and the
    /// engine weighs a shared type as shared vocabulary and a different one at its own weight.
    pub type_refs: Vec<String>,
    /// The sub-expressions of the lines above — every call, attribute chain, subscript, operator
    /// application, comprehension, lambda and display the statements contain, rendered the same way
    /// the lines are, each with the index (into `type3_lines`) of the line it sits in. Empty for a
    /// frontend that reports only lines.
    ///
    /// A line is one unit, so a near-copy that differs by one token inside a long statement loses
    /// the whole statement's agreement at once, and in a short function that is most of it. The
    /// sub-expressions are the parts of that statement the edit did not touch: through them two
    /// lines agree *in part*, on everything but the subtrees the token sits in. This is the
    /// characteristic-vector idea of DECKARD (Jiang et al. 2007) and the path-context idea of
    /// code2vec (Alon et al. 2019), taken at the granularity a bag-of-units measure can use.
    pub type3_units: Vec<(u32, String)>,
    /// Every statement of the body at every nesting level, in source order — a *different* unit
    /// from [`Self::type3_lines`], which each frontend shingles as suits the Type-3 pass. Empty
    /// when the frontend does not walk statements. Rides to the engine as
    /// [`dup_defs_core::Facets::statements`].
    pub statements: Vec<Statement>,
    /// Node count of the alpha-renamed canonical — the cross-name "substance" gate.
    pub size: usize,
}

impl AnalyzedFn {
    /// Build one from the four fields every frontend already had, with no statement stream and no
    /// sub-expression units.
    ///
    /// The honest constructor for a frontend that has not yet been taught to walk statements: it
    /// says "nothing to report" rather than fabricating a flat stream, and every pass reading them
    /// is self-gating on the empty vector.
    #[must_use]
    pub fn flat(cluster_canonical: String, xname_canonical: String, type3_lines: Vec<String>, size: usize) -> Self {
        Self {
            cluster_canonical,
            xname_canonical,
            type3_lines,
            type_refs: Vec::new(),
            type3_units: Vec::new(),
            statements: Vec::new(),
            size,
        }
    }
}

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

/// The spelling every name-agnostic canonical gives a renamed **type reference**: `_T{n}`.
pub const TYPE_SLOT_PREFIX: &str = "_T";

/// Rename a type reference to its `_T{n}` slot, stable within one definition — the type-position
/// counterpart of [`alpha_rename`].
///
/// Why types are renamed at all: a Type-2 clone is *syntactically identical except for variations in
/// identifiers, literals and types* (Roy, Cordy & Koschke, 2009). Keeping the annotations verbatim
/// made two bodies that agree line for line, and differ only in the declared type of their argument,
/// two different canonicals — so neither the exact pass nor the near-copy pass saw them.
///
/// Renamed **consistently** rather than erased, which is parameterized matching (Baker, 1995): two
/// definitions agree only if one bijection maps the types of the first onto the types of the second,
/// so `f(a: X, b: X)` and `f(a: X, b: Y)` stay apart. Only frontends whose language attaches no
/// behaviour to an annotation may call this; where types select behaviour, renaming them would merge
/// definitions that run differently.
#[must_use]
#[allow(clippy::cast_possible_truncation)] // a def's distinct type-reference count is far below u32::MAX
#[allow(clippy::implicit_hasher)] // always called with the frontends' std-hasher maps
pub fn type_slot(map: &mut HashMap<String, u32>, name: &str) -> String {
    let next = map.len() as u32;
    let slot = *map.entry(name.to_owned()).or_insert(next);
    format!("{TYPE_SLOT_PREFIX}{slot}")
}

