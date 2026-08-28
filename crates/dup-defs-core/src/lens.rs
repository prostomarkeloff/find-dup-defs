//! **Lenses** — the perspective axis, shared by every frontend.
//!
//! The erasure ladder (locals → own name → module names → everything) always canonicalizes the same
//! text: the definition's body. A lens canonicalizes a *different projection* of the same
//! definition, so two lenses are not a wider and a narrower version of one another — measured on an
//! application corpus, the use-profile lens overlaps the body passes by under 10%.
//!
//! Each lens answers one question about a definition and throws the rest away:
//!
//! | lens | question | keeps |
//! |---|---|---|
//! | `outgoing`  | what does it depend on?      | the *set* of external callees |
//! | `effects`   | what protocol does it drive? | the *sequence* of external callees |
//! | `control`   | how does it branch?          | the branching skeleton, with nesting |
//! | `failures`  | how does it fail?            | raised and caught error types |
//! | `resources` | what does it hold open?      | scoped-resource acquisitions |
//! | `signature` | what contract does it offer? | arity shape + annotation names |
//! | `decorators`| what role does it play?      | decorator / attribute names |
//! | `schema`    | what shape does it declare?  | declared field types and options, as a *set* |
//! | `scope`     | what does its body do?       | the body with every name its module introduced erased |
//! | `use`       | how is it handled?           | the statements elsewhere that mention it |
//!
//! ## What is shared and what is not
//!
//! The ten *questions* are language-independent; only the answers are not. So everything in this
//! module is the part that would otherwise be re-implemented per language — the vocabulary, the
//! stitching, the corpus scoring, the merge of facts that can only be known once the whole tree is
//! read — and a frontend contributes exactly one thing: an AST walk that fills a [`LensFacts`].
//!
//! That split is the point. Lenses were Python-only not because the other languages lacked the
//! notion of "what does this call" or "how does this fail", but because the machinery around the
//! notion lived inside the Python frontend. Moved here, a new language needs the walk and nothing
//! else.
//!
//! ## Why one stitched record rather than ten kinds
//!
//! The lenses are stitched into **one** record. Each contributes its facts under its own prefix
//! (`control:if`, `outgoing:.commit`), and the Type-3 pass's IDF-weighted cosine over those lines
//! *is* the vote: agreement through several lenses raises the score, agreement through one barely
//! moves it, and a fact the whole corpus shares is weighted to nothing without anyone having to
//! declare it noise. A cross-name exact match means every lens agreed at once.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::{Analysis, CanonDialect, Def, Facets, ScanOpts};

use crate::kinds::LENSES;

/// A projection with fewer facts than this carries no shape — two definitions that each call one
/// external thing match trivially. The lens counterpart of the Type-3 shingle-count floor.
pub const MIN_FACTS: usize = 3;

/// Saturation constant for a stitched record's information mass — what counts as "a substantial
/// record". Calibrated so a handful of distinctive facts clears half scale.
const LENS_MASS_K: f64 = 10.0;

/// The perspectives. Each is one question a definition can be asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lens {
    /// What does it depend on — the *set* of things it calls that its module did not define.
    Outgoing,
    /// What protocol does it drive — the same callees in *call order*.
    Effects,
    /// How does it branch — the control skeleton, each tag carrying its nesting.
    Control,
    /// How does it fail — the error types it raises and catches.
    Failures,
    /// What does it hold open — scoped resource acquisitions (`with`, RAII guards, `using`).
    Resources,
    /// What contract does it offer — arity shape and the names in its annotations.
    Signature,
    /// What role does it play — decorators, attributes, annotations attached to the definition.
    Decorators,
    /// What shape does it declare — declared field types and their options, as a set.
    Schema,
    /// The body itself with every name the module introduced erased — the widest rung of the
    /// erasure ladder, seated here as a perspective among the rest.
    Scope,
    /// The definition's *use sites*: statements elsewhere that mention its name. Unlike every other
    /// lens this cannot be computed from the definition alone, so its facts arrive through
    /// [`merge_use_facts`] once the whole tree has been walked.
    Use,
}

impl Lens {
    /// The prefix this lens stamps on its facts, so the stitched record stays attributable and two
    /// lenses can never accidentally agree on the same string.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Lens::Outgoing => "outgoing",
            Lens::Effects => "effects",
            Lens::Control => "control",
            Lens::Failures => "failures",
            Lens::Resources => "resources",
            Lens::Signature => "signature",
            Lens::Decorators => "decorators",
            Lens::Schema => "schema",
            Lens::Scope => "scope",
            Lens::Use => "use",
        }
    }

    /// Every lens, in report order.
    #[must_use]
    pub fn all() -> [Lens; 10] {
        [
            Lens::Outgoing,
            Lens::Effects,
            Lens::Control,
            Lens::Failures,
            Lens::Resources,
            Lens::Signature,
            Lens::Decorators,
            Lens::Schema,
            Lens::Scope,
            Lens::Use,
        ]
    }
}

/// Which lenses vote in this run.
///
/// All of them, or none: a lens is a weight on one scale rather than a separate question, and the
/// corpus IDF already silences the ones a given tree has nothing to say through. The kind is
/// opt-in (`--kinds lenses`) because it costs a second walk of every file, so the choice is a CLI
/// argument rather than a side channel and the default run stays byte-identical.
#[must_use]
pub fn enabled_lenses(opts: &ScanOpts) -> Vec<Lens> {
    if opts.wants("lenses") { Lens::all().to_vec() } else { Vec::new() }
}

/// How deep a control-flow fact sits, rendered into the fact itself.
///
/// 🔴 Shared because the three frontends must **agree**, not merely to avoid a copy. A `control`
/// fact is compared across languages by string, so if one walk capped the depth marker at four and
/// another at five, the same nesting would produce different facts and two genuinely alike
/// procedures would be held apart by a constant neither of them is about. The cap exists because
/// past a few levels "deeper still" stops discriminating; where exactly it sits matters far less
/// than that one place decides it.
///
/// Found by running this tool on itself: the lens pass reported `FactWalk::tag` / `Walk::tag` as one
/// thing under three names, which is exactly what it was.
#[must_use]
pub fn control_tag(depth: usize, tag: &str) -> String {
    const MAX_MARKED_DEPTH: usize = 4;
    format!("{}{tag}", "+".repeat(depth.min(MAX_MARKED_DEPTH)))
}

/// One definition's answers, one bucket per lens.
///
/// A frontend fills this during its own walk and hands it over; ordering inside a bucket is the
/// frontend's business (`Effects` is a sequence, `Outgoing` a set), because only it knows whether
/// the order it saw is the order that happened.
#[derive(Clone, Debug, Default)]
pub struct LensFacts {
    buckets: Vec<(Lens, Vec<String>)>,
}

impl LensFacts {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one fact under one lens. Facts are stored unprefixed; [`stitch`] applies the tag.
    pub fn push(&mut self, lens: Lens, fact: impl Into<String>) {
        let fact = fact.into();
        match self.buckets.iter_mut().find(|(l, _)| *l == lens) {
            Some((_, out)) => out.push(fact),
            None => self.buckets.push((lens, vec![fact])),
        }
    }

    /// Record many facts under one lens, in the order given.
    pub fn extend<I>(&mut self, lens: Lens, facts: I)
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        for fact in facts {
            self.push(lens, fact);
        }
    }

    /// The facts one lens holds, in insertion order.
    #[must_use]
    pub fn get(&self, lens: Lens) -> &[String] {
        self.buckets.iter().find(|(l, _)| *l == lens).map_or(&[], |(_, v)| v.as_slice())
    }

    /// Total facts across every lens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buckets.iter().map(|(_, v)| v.len()).sum()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The stitched record: every enabled lens's facts, each under its own prefix, in lens order.
#[must_use]
pub fn stitch(enabled: &[Lens], facts: &LensFacts) -> Vec<String> {
    let mut out = Vec::new();
    for &lens in enabled {
        let tag = lens.tag();
        out.extend(facts.get(lens).iter().map(|fact| format!("{tag}:{fact}")));
    }
    out
}

/// Where a definition sits — the location fields a lens [`Def`] carries over unchanged.
#[derive(Clone, Debug)]
pub struct DefSite<'a> {
    pub name: &'a str,
    pub file: &'a Arc<str>,
    pub line: usize,
    pub col: usize,
    pub loc: usize,
}

/// One [`Def`] per definition, carrying every enabled lens's view of it.
///
/// `None` when no lens is enabled or the record is too thin to carry shape ([`MIN_FACTS`]).
/// `thickness` is left unset: it is corpus-relative and filled by [`score_lens_defs`] once the whole
/// tree is known.
#[must_use]
pub fn lens_def(
    lang: &'static str,
    dialect: CanonDialect,
    site: &DefSite<'_>,
    enabled: &[Lens],
    facts: &LensFacts,
) -> Option<Def> {
    if enabled.is_empty() {
        return None;
    }
    let stitched = stitch(enabled, facts);
    if stitched.len() < MIN_FACTS {
        return None;
    }
    let canonical = stitched.join(" ");
    Some(Def {
        lang,
        kind: &LENSES,
        name: site.name.to_owned(),
        file: Arc::clone(site.file),
        line: site.line,
        col: site.col,
        loc: site.loc,
        args: stitched.len(),
        text_orig: stitched.join("\n"),
        cluster_canonical: Some(canonical.clone()),
        analysis: Some(Analysis {
            xname_canonical: canonical,
            size: stitched.len(),
            type3_lines: stitched,
            canon_dialect: dialect,
        }),
        thickness: None,
        // A lens record is a projection, not a statement stream, and it reaches nothing of its own —
        // the perspective passes read the body def for that.
        facets: Facets::default(),
    })
}

/// Fold each definition's use sites into its lens record.
///
/// These are the only facts that cannot be computed from the definition alone — they live in every
/// *other* file — so they are merged once the tree has been walked rather than during the per-file
/// scan. They arrive already prefixed by channel, which is why nothing is tagged here.
#[allow(clippy::implicit_hasher)] // always called with the frontends' std-hasher maps
pub fn merge_use_facts(defs: &mut [Def], mut facts: HashMap<String, Vec<String>>) {
    for def in defs.iter_mut() {
        if def.kind.id != LENSES.id {
            continue;
        }
        let Some(extra) = facts.remove(&def.name) else { continue };
        let Some(analysis) = def.analysis.as_mut() else { continue };
        analysis.type3_lines.extend(extra);
        analysis.type3_lines.sort();
        analysis.size = analysis.type3_lines.len();
        let canonical = analysis.type3_lines.join(" ");
        analysis.xname_canonical.clone_from(&canonical);
        def.text_orig = analysis.type3_lines.join("\n");
        def.args = analysis.type3_lines.len();
        def.cluster_canonical = Some(canonical);
    }
}

/// Score every lens record against the corpus, once scanning is done.
///
/// A fact's weight is its IDF over the corpus of lens records, so what counts as signal is decided
/// by the tree rather than by a list: `control:return` is in nearly every record and weighs nothing,
/// while a rare callee weighs a lot. The count of *lenses that actually spoke* rides alongside,
/// because a record whose facts all come from one lens is one opinion, not a consensus.
pub fn score_lens_defs(defs: &mut [Def]) {
    let mut df: HashMap<&str, usize> = HashMap::new();
    let mut corpus = 0usize;
    for def in defs.iter() {
        if def.kind.id != LENSES.id {
            continue;
        }
        corpus += 1;
        if let Some(analysis) = &def.analysis {
            for fact in analysis.type3_lines.iter().map(String::as_str).collect::<BTreeSet<_>>() {
                *df.entry(fact).or_insert(0) += 1;
            }
        }
    }
    if corpus == 0 {
        return;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = corpus as f64;
    let scores: Vec<Option<f64>> = defs
        .iter()
        .map(|def| {
            if def.kind.id != LENSES.id {
                return None;
            }
            let analysis = def.analysis.as_ref()?;
            let mut mass = 0.0f64;
            let mut lenses_heard: BTreeSet<&str> = BTreeSet::new();
            for fact in analysis.type3_lines.iter().map(String::as_str).collect::<BTreeSet<_>>() {
                #[allow(clippy::cast_precision_loss)]
                let idf = (n / df.get(fact).copied().unwrap_or(1).max(1) as f64).ln();
                mass += idf.max(0.0);
                if let Some((tag, _)) = fact.split_once(':') {
                    lenses_heard.insert(tag);
                }
            }
            let mass_score = 1.0 - (-mass / LENS_MASS_K).exp();
            #[allow(clippy::cast_precision_loss)]
            let breadth = 1.0 - (-(lenses_heard.len() as f64) / 2.0).exp();
            Some(0.7f64.mul_add(mass_score, 0.3 * breadth))
        })
        .collect();
    for (def, score) in defs.iter_mut().zip(scores) {
        if let Some(s) = score {
            def.thickness = Some(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{lens_def, stitch, DefSite, Lens, LensFacts, MIN_FACTS};
    use crate::CanonDialect;
    use std::sync::Arc;

    fn site(file: &Arc<str>) -> DefSite<'_> {
        DefSite { name: "f", file, line: 1, col: 0, loc: 4 }
    }

    #[test]
    fn nesting_is_marked_identically_whatever_language_walked_it() {
        // The point of sharing this: a `control` fact is compared across languages by string, so a
        // depth cap that differed between two walks would hold alike procedures apart by a constant
        // neither is about.
        assert_eq!(super::control_tag(0, "if"), "if");
        assert_eq!(super::control_tag(2, "for"), "++for");
        assert_eq!(super::control_tag(9, "try"), super::control_tag(4, "try"), "the cap is one place's decision");
    }

    #[test]
    fn facts_are_tagged_by_their_lens_and_ordered_by_it() {
        let mut facts = LensFacts::new();
        facts.push(Lens::Control, "if");
        facts.push(Lens::Outgoing, ".commit");
        // Order follows the lens list, not the order facts were recorded — otherwise two records
        // holding the same facts would stitch to different strings.
        let out = stitch(&[Lens::Outgoing, Lens::Control], &facts);
        assert_eq!(out, vec!["outgoing:.commit", "control:if"]);
    }

    #[test]
    fn a_lens_keeps_the_order_its_frontend_saw() {
        // `effects` is a sequence: the frontend's order is the protocol, and stitching must not
        // sort it away.
        let mut facts = LensFacts::new();
        facts.extend(Lens::Effects, [".begin", ".write", ".commit"]);
        assert_eq!(
            stitch(&[Lens::Effects], &facts),
            vec!["effects:.begin", "effects:.write", "effects:.commit"]
        );
    }

    #[test]
    fn a_thin_record_is_no_record() {
        let file: Arc<str> = Arc::from("a.py");
        let mut facts = LensFacts::new();
        for i in 0..MIN_FACTS - 1 {
            facts.push(Lens::Control, format!("t{i}"));
        }
        assert!(lens_def("py", CanonDialect::CPythonAst, &site(&file), &Lens::all(), &facts).is_none());
        facts.push(Lens::Control, "one more");
        assert!(lens_def("py", CanonDialect::CPythonAst, &site(&file), &Lens::all(), &facts).is_some());
    }

    #[test]
    fn no_enabled_lens_means_no_record_however_many_facts() {
        let file: Arc<str> = Arc::from("a.py");
        let mut facts = LensFacts::new();
        facts.extend(Lens::Control, ["if", "for", "try", "return"]);
        assert!(lens_def("py", CanonDialect::CPythonAst, &site(&file), &[], &facts).is_none());
    }

    #[test]
    fn a_lens_record_carries_the_frontends_own_dialect_and_language() {
        let file: Arc<str> = Arc::from("a.rs");
        let mut facts = LensFacts::new();
        facts.extend(Lens::Control, ["if", "for", "match"]);
        let def = lens_def("rs", CanonDialect::Rust, &site(&file), &Lens::all(), &facts).expect("record");
        assert_eq!(def.lang, "rs");
        assert_eq!(def.analysis.expect("analysis").canon_dialect, CanonDialect::Rust);
        // A projection reaches nothing of its own; the body def answers that.
        assert!(def.facets.reaches.is_empty());
    }
}
