//! The frontend↔engine contract: [`Def`] / [`KindSpec`] / [`Analysis`] / [`Frontend`]. A
//! frontend parses each file once, classifies its definitions, and lowers each to a [`Def`] — a
//! flat *feature record* carrying the precomputed canonical strings the clustering engine
//! consumes. The engine never sees a frontend's rich per-language representation and never
//! matches on a fixed kind vocabulary: each frontend declares its own kinds as `&'static`
//! [`KindSpec`]s. (Each `*-canon` crate keeps its own extraction intermediate internally.)

use std::sync::Arc;

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

/// The structural *dialect* of an [`Analysis::xname_canonical`] — which frontend's unparser shaped
/// it. Dialect-specific engine passes (the patternology helper-extractor walks `CPython` `ast.dump`
/// field order) read this to skip canon they cannot soundly analyze, rather than silently
/// mis-walking a foreign tree. The engine matches on the *capability*, not a language name.
/// New dialects are added as the engine grows language support, so downstream `match`es must carry a
/// wildcard arm — adding a variant is not a breaking change.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonDialect {
    /// `CPython` `ast.dump(annotate_fields=False)` tree shape (the Python frontend).
    CPythonAst,
    /// The Rust frontend's `syn`-derived structural dump (`Func`/`Block`/`Let`/`Method`/… tags) —
    /// the patternology engine has a `RustDialect` for it.
    Rust,
    /// The TypeScript frontend's `oxc`-derived structural dump (`Func`/`Block`/`Var`/`Call`/… tags)
    /// — the patternology engine has a `TsDialect` for it. (Also the catch-all for any future
    /// frontend whose canonical the dialect-specific passes don't yet recognize.)
    Other,
}

/// Full callable analysis precomputed by the frontend — the cross-name + Type-3 inputs the
/// engine needs (the cluster canonical lives separately on [`Def::cluster_canonical`]).
///
/// * `xname_canonical` — alpha-renamed structural canonical (bound locals → positional
///   `_v{n}`, top def name blanked); the cross-name pass buckets on this.
/// * `type3_lines` — per-statement renamed lines for the Type-3 IDF-cosine pass.
/// * `size` — node count of the alpha-renamed canonical, the cross-name "substance" gate.
/// * `canon_dialect` — the [`CanonDialect`] of `xname_canonical`, for dialect-specific passes.
#[derive(Clone, Debug)]
pub struct Analysis {
    pub xname_canonical: String,
    pub type3_lines: Vec<String>,
    pub size: usize,
    pub canon_dialect: CanonDialect,
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
    /// Frontend-supplied refactor-payoff score in `[0, 1]`, overriding the engine's default
    /// [`crate::Def`]-agnostic [`thickness`](../find_dup_defs/fn.thickness.html) formula for
    /// clusters of this def. `None` (every body kind) ⇒ the default formula, unchanged.
    ///
    /// The default is `volume = (n − 1) · loc` — lines a refactor deletes — which assumes a bigger
    /// cluster is a bigger win. That assumption inverts for units whose "body" is derived rather
    /// than written: a use profile shared by fifty definitions is not fifty times the payoff, it is
    /// evidence the profile is a language idiom. A frontend that knows its unit's economics scores
    /// it here; the engine takes the cluster's minimum, so a cluster is never thicker than its
    /// thinnest member.
    pub thickness: Option<f64>,
}

/// What the caller asked this run to produce. Some kinds cost real work — a second walk of every
/// file, an extra canonicalization per definition — and emitting them unasked would both slow the
/// default run and print sections nobody wanted. A frontend consults this to decide which of its
/// kinds are worth computing, so the choice lives in the CLI (`--kinds`) rather than in a side
/// channel.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScanOpts<'a> {
    /// Kind ids the caller named (`--kinds functions,lenses`), or `None` for "the default set".
    pub kinds: Option<&'a [String]>,
}

impl ScanOpts<'_> {
    /// True when the caller named this kind explicitly. `None` (no `--kinds`) means the default
    /// set, which never includes the opt-in kinds — hence `false` rather than `true` here.
    #[must_use]
    pub fn wants(&self, id: &str) -> bool {
        self.kinds.is_some_and(|ks| ks.iter().any(|k| k == id))
    }
}

/// A language frontend: walks a set of files and lowers each definition to a [`Def`], computing
/// its canonical strings during the single parse. The engine consumes `&[&dyn Frontend]` and
/// never names a concrete frontend crate — the binary owns the registry.
pub trait Frontend: Sync {
    /// Short language code, matching the CLI `--only` vocabulary (e.g. `"py"`, `"ts"`).
    fn lang(&self) -> &'static str;
    /// File extensions this frontend claims (without the dot), e.g. `["ts", "tsx"]`.
    fn extensions(&self) -> &'static [&'static str];
    /// Every kind this frontend emits *for this run*. The binary unions these across the selected
    /// frontends to build the report's section list, so `--only py` prints only Python's sections
    /// and an opt-in kind contributes a section exactly when it was asked for.
    fn kinds(&self, opts: &ScanOpts) -> &'static [&'static KindSpec];
    /// Parse each file once and return its definitions as [`Def`]s with canon precomputed.
    fn scan(&self, files: &[Arc<str>], opts: &ScanOpts) -> Vec<Def>;
}
