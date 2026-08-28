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
    /// Every statement of the body at every nesting level, in source order — a *different* unit
    /// from [`Self::type3_lines`], which each frontend shingles as suits the Type-3 pass. Empty
    /// when the frontend does not walk statements. Rides to the engine as
    /// [`dup_defs_core::Facets::statements`].
    pub statements: Vec<Statement>,
    /// Node count of the alpha-renamed canonical — the cross-name "substance" gate.
    pub size: usize,
}

impl AnalyzedFn {
    /// Build one from the four fields every frontend already had, with no statement stream.
    ///
    /// The honest constructor for a frontend that has not yet been taught to walk statements: it
    /// says "nothing to report" rather than fabricating a flat stream, and every pass reading them
    /// is self-gating on the empty vector.
    #[must_use]
    pub fn flat(cluster_canonical: String, xname_canonical: String, type3_lines: Vec<String>, size: usize) -> Self {
        Self { cluster_canonical, xname_canonical, type3_lines, statements: Vec::new(), size }
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

