//! Definition records and analysis tuple shared by every frontend.
//!
//! Two generations of the frontend↔engine contract live here side by side during the
//! migration to the [`Frontend`] trait:
//!
//! * **New ([`Def`] / [`KindSpec`] / [`Analysis`] / [`Frontend`]).** A frontend parses each
//!   file once, classifies its definitions, and lowers each to a [`Def`] — a flat *feature
//!   record* carrying the precomputed canonical strings the clustering engine consumes. The
//!   engine never sees a frontend's rich per-language representation and never matches on a
//!   fixed kind vocabulary: each frontend declares its own kinds as `&'static` [`KindSpec`]s.
//! * **Old ([`ModuleDef`] / [`AnalyzedFn`] / [`Language`]).** The pre-trait shape, where the
//!   engine held `Vec<ModuleDef>` and dispatched canonicalization back to a frontend by the
//!   [`ModuleDef::lang`] tag. Retained until every frontend and the engine are ported, then
//!   removed.

use std::sync::Arc;

/// Source language of a [`ModuleDef`]. Stamped by the frontend during extraction so the engine
/// can route canonicalization back to the right per-language implementation without re-deriving
/// from the file extension. Default = [`Language::Python`] so existing JSON dumps and
/// `ModuleDef`-typed APIs created before the multi-language split still deserialize cleanly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Language {
    #[default]
    #[serde(rename = "python")]
    Python,
    #[serde(rename = "typescript")]
    TypeScript,
}

/// One module-level definition found by a frontend scan (kind = `functions` / `classes` /
/// `constants` / `type-aliases` / `interfaces` / `methods`; `line`/`col` 0-indexed). `loc` and
/// `args` are "how fat is this" signals surfaced in the dup-defs report so a user reading a flat
/// list of clusters can immediately tell a 50-line copy-paste from a 3-liner. `loc` is the count
/// of non-blank lines in the def's original source text (including the signature line); `args`
/// is the parameter count for functions/methods (including a `self`/`cls` receiver in Python or
/// a `this` parameter in TypeScript — the count the user actually typed) and `0` for
/// non-callable kinds. For methods, `loc` reflects the **original** source, NOT the
/// post-receiver-strip text — the user is looking up code they wrote, not the synthetic form
/// the canonicalizer ingests.
///
/// `text` is the canonicalization-input form (post-`self`/`cls`/`this`-strip for methods);
/// `text_orig` is what the user actually wrote and is used for snippet display (calibration
/// view, etc). For every kind other than `methods` the two are identical clones.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModuleDef {
    pub kind: String,
    pub name: String,
    pub file: String,
    pub line: usize,
    pub col: usize,
    pub text: String,
    #[serde(default)]
    pub text_orig: String,
    #[serde(default)]
    pub loc: usize,
    #[serde(default)]
    pub args: usize,
    /// Source language — drives engine-side dispatch of per-language canonicalization. Defaults
    /// to [`Language::Python`] for backward compatibility with pre-multi-language JSON dumps.
    #[serde(default)]
    pub lang: Language,
}

/// Full dup-defs analysis of one callable definition:
/// `(cluster_canonical, xname_canonical, lines, size)`.
///
/// * `cluster_canonical` — names-preserved structural canonical, used by the name-gated pass.
/// * `xname_canonical` — alpha-renamed structural canonical (bound locals → positional `_v{n}`,
///   top def name blanked to `_fn`), used by the cross-name pass.
/// * `lines` — per-statement renamed lines (one logical line per statement, equivalent to
///   `ast.unparse` line splits in Python), used by the Type-3 (`ECScan`) cosine pass.
/// * `size` — node count of the alpha-renamed canonical, used as a "substance" gate so a
///   3-line accessor doesn't escalate to ERROR purely on a renamed-exact match.
pub type AnalyzedFn = (String, String, Vec<String>, usize);

// ── New frontend↔engine contract (Frontend trait) ───────────────────────────

/// Engine-facing metadata about one *kind* of definition. Each frontend declares its own kinds
/// as `&'static` consts (e.g. `py_canon::FUNCTIONS`, `ts_canon::INTERFACES`) and stamps the
/// matching `&'static KindSpec` onto every [`Def`] it emits. The engine treats a kind as opaque
/// grouping / ordering data and reads only these fields — it never matches on a fixed string
/// vocabulary, so a new language's constructs need no engine edit.
///
/// * `id` — stable machine tag, the name-gated grouping key and the `KIND:` directive match
///   target (e.g. `"functions"`, `"struct"`). Frontends that want a kind to cluster *across*
///   languages share an `id`; distinct ids keep languages in separate buckets.
/// * `label` — uppercase report / JSON tag (e.g. `"FUNCTION"`).
/// * `noun_plural` — pluralized noun for the report section header (e.g. `"functions"`,
///   `"type aliases"` — note the space, distinct from the hyphenated `id`).
/// * `section` — base ordering slot for this kind in the report; the engine adds a per-pass
///   offset for `fn_like` kinds (`name` 0 / `cross-name` 1 / `type-3` 2).
/// * `body` — body-bearing: clustered by structural canonical similarity (else by raw text).
/// * `fn_like` — callable: participates in the cross-name and Type-3 passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KindSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub noun_plural: &'static str,
    pub section: u16,
    pub body: bool,
    pub fn_like: bool,
}

/// Full callable analysis precomputed by the frontend — the cross-name + Type-3 inputs the
/// engine needs. Named replacement for the legacy [`AnalyzedFn`] tuple's last three fields
/// (the cluster canonical now lives on [`Def::cluster_canonical`]).
///
/// * `xname_canonical` — alpha-renamed structural canonical (bound locals → positional
///   `_v{n}`, top def name blanked); the cross-name pass buckets on this.
/// * `type3_lines` — per-statement renamed lines for the Type-3 IDF-cosine pass.
/// * `size` — node count of the alpha-renamed canonical, the cross-name "substance" gate.
#[derive(Clone, Debug)]
pub struct Analysis {
    pub xname_canonical: String,
    pub type3_lines: Vec<String>,
    pub size: usize,
}

/// One definition lowered to the engine's feature record. Produced by [`Frontend::scan`] with
/// the canonical strings already computed (single parse per file). `line`/`col` are 0-indexed;
/// `loc`/`args` mirror [`ModuleDef`]'s semantics.
///
/// `cluster_canonical` is `Some` for body kinds (the names-preserved structural canonical the
/// name-gated pass clusters); `None` for raw-text kinds. `analysis` is `Some` only for
/// `fn_like` kinds — but may still be `None` for a callable that failed to analyze (e.g. an
/// un-reparseable receiver-stripped method), which the cross-name / Type-3 passes skip.
#[derive(Clone, Debug)]
pub struct Def {
    pub lang: &'static str,
    pub kind: &'static KindSpec,
    pub name: String,
    pub file: Arc<str>,
    pub line: usize,
    pub col: usize,
    pub loc: usize,
    pub args: usize,
    pub text_orig: String,
    pub cluster_canonical: Option<String>,
    pub analysis: Option<Analysis>,
}

/// A language frontend: walks a set of files and lowers each definition to a [`Def`], computing
/// its canonical strings during the single parse. The engine consumes `&[&dyn Frontend]` and
/// never names a concrete frontend crate — the binary owns the registry.
pub trait Frontend: Sync {
    /// Short language code, matching the CLI `--only` vocabulary (e.g. `"py"`, `"ts"`).
    fn lang(&self) -> &'static str;
    /// File extensions this frontend claims (without the dot), e.g. `["ts", "tsx"]`.
    fn extensions(&self) -> &'static [&'static str];
    /// Parse each file once and return its definitions as [`Def`]s with canon precomputed.
    fn scan(&self, files: &[Arc<str>]) -> Vec<Def>;
}
