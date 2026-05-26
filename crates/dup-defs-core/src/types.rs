//! Definition records and analysis tuple shared by every frontend.
//!
//! The frontend (Python: [`py-canon`], TypeScript: [`ts-canon`]) parses files, classifies
//! top-level statements and class methods, and emits a [`ModuleDef`] per definition. The
//! `find-dup-defs` engine then runs language-agnostic similarity / cluster passes over the
//! resulting `Vec<ModuleDef>`, dispatching canonicalization back to the right frontend by the
//! [`ModuleDef::lang`] tag.

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
