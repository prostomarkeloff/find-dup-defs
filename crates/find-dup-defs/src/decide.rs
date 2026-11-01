//! **Decisions** — two places that decide on the same facts, however differently they are written.
//!
//! Every other pass compares how definitions are *written*: their bodies, their statements, the
//! modules they reach. A decision made in two places independently shares none of that by
//! construction — one copy raises where the other returns a sentinel, filters a loop where the other
//! refuses a request, reads a snapshot where the other reads the row, splits one test into three — and
//! the textual passes see two unrelated functions. What the two keep in common is the **support** of
//! the decision: the set of facts it depends on, which [`Facets::decides`] lists and which is
//! invariant under all of that.
//!
//! ## A projection, validated
//!
//! Definitions and the facts they test form a bipartite graph; two definitions deciding on the same
//! facts are a pair in its projection. Which pairs are more than chance is the question of validated
//! projections, and the pass answers it the way that literature does ([`crate::bicm`]): the null is
//! the bipartite configuration model fitted to both degree sequences — every definition keeps its
//! number of facts and every fact its number of definitions, in expectation, and nothing else is
//! assumed — each pair gets a p-value under it, and Benjamini–Hochberg holds the false-discovery rate
//! over every pair of definitions.
//!
//! Two statistics, one null (`-D settings:decision-statistic=…`):
//!
//! - `count` — the number of shared facts, with its exact Poisson-binomial tail. The published one.
//!   It weighs a fact half the tree tests the same as one two definitions test, so a few rare facts
//!   shared next to many common ones are lost in the count; the tests pin a corpus where exactly that
//!   happens.
//! - `surprise` (default) — `Σ −ln q` over the shared facts, `q` the null probability that both hold
//!   the fact: the improbability of the shared *set*. Its tail is a saddlepoint approximation.
//!
//! The sizes are in the null, not in a weighting: a definition testing forty facts shares some with
//! anything, and the model expects it to, which is what lets a check copied into the middle of a long
//! function surface at all where a resemblance score would drown it in the host.
//!
//! ## What is not reported
//!
//! A pair a body pass already clusters is a copy, and that pass reports it; this one exists for the
//! pairs those passes cannot see. Facts are not inlined across calls: two definitions calling one
//! predicate share that call, which is evidence they decide the same thing, while the predicate's own
//! tests stay its own — reuse is not duplication. A column filtered by a key the caller chose, or
//! joined to another table, is a filter, not a fact ([`decide::is_fact`]): what every query on one
//! table has in common is not a decision, and the pass does not read it as one.
//!
//! Findings are advisory ([`Severity::Info`]), like converge's: the pass says where a decision lives
//! twice, and whether that is one rule or two that look alike is a reading, not a gate.
//!
//! [`Facets::decides`]: dup_defs_core::Facets::decides

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dup_defs_core::decide::{self, has_identity, params_in, resolve_params, rewrite_locals, CallSite};
use dup_defs_core::{reach, Def};
use rayon::prelude::*;
use serde::Serialize;

use crate::bicm::{bh_threshold, count_survival, expected_count, Bicm, Surprise};
use crate::{member, Finding, Severity};

/// Report-section offset for decision pairs, past every other pass's.
pub const DECISION_SECTION_OFFSET: usize = 4000;
/// Report-section offset for decision families.
pub const DECISION_FAMILY_SECTION_OFFSET: usize = 5000;


/// The statistic a pair is weighed by; the null is the same for both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Statistic {
    /// How many facts the two share — the statistic of the validated-projection literature.
    Count,
    /// How improbable the shared set is: `Σ −ln q` over the shared facts.
    Surprise,
}

/// The pass's options. `fdr` is the false-discovery rate the reader accepts — a convention of the
/// reader's, not a property of the code, and the one number the pass takes from outside.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DecisionOpts {
    pub fdr: f64,
    pub statistic: Statistic,
    /// Read the corpus as a closed world: a parameter every visible call fills with one identity is
    /// that identity (`settings:decision-closed-world=on`). Off by default, because it is not a
    /// property of the language but an assumption about the corpus — that the calls it shows are all
    /// there are — which an application mostly satisfies and a library mostly does not. Findings
    /// that lean on it say so.
    pub closed_world: bool,
}

impl Default for DecisionOpts {
    fn default() -> Self {
        Self { fdr: 0.05, statistic: Statistic::Surprise, closed_world: false }
    }
}

// ---------------------------------------------------------------------------
// The corpus: definitions as sets of fact ids
// ---------------------------------------------------------------------------

struct Corpus {
    /// Definition index (into `defs`) for each member of the population.
    at: Vec<usize>,
    /// Each member's facts that some other member also tests, as ids, rarest first.
    facts: Vec<Vec<u32>>,
    /// Each member's degree: every fact it tests, including those no one else does.
    size: Vec<usize>,
    /// Fact frequency: how many members test it.
    df: Vec<u32>,
    /// Each fact as the frontend wrote it, local paths renamed — the structure [`operands`] reads.
    atoms: Vec<Arc<str>>,
    /// Each fact rendered for the reader.
    text: Vec<String>,
    /// Per member, the facts it holds only through a parameter the closed-world reading filled.
    inferred: Vec<HashSet<u32>>,
}

/// What the tree calls a module whose file path a frontend reported.
///
/// A file at `src/pkg/m.py` is imported as `pkg.m`: the prefix the tree drops from the file path is
/// its source root, and the corpus shows it — every module some import names absolutely attests the
/// root its path sits under. A module nothing imports absolutely (a leaf of private helpers, one
/// reached only relatively) is named under the root its neighbours attest, so its locals read
/// `@pkg.m.X` and not as the path the file was found at, and two such modules with one file name
/// stay apart. Roots are weighed by how many modules attest them, which keeps a relative import
/// read as a top-level module from outvoting the tree. With nothing attested a path stays as it is.
fn module_names(defs: &[Def]) -> impl Fn(&str) -> String {
    let attested: HashSet<String> = defs
        .iter()
        .flat_map(|d| d.facets.reaches.iter())
        .flat_map(|path| reach::prefixes(path).map(str::to_owned).collect::<Vec<_>>())
        .collect();
    let modules: HashSet<String> = defs
        .iter()
        .filter_map(|d| d.facets.identity.as_deref())
        .filter_map(|identity| decide::local_modules(identity).next().map(str::to_owned))
        .collect();
    let mut support: HashMap<String, usize> = HashMap::new();
    for module in &modules {
        let segments: Vec<&str> = module.split('.').collect();
        if let Some(drop) = (0..segments.len()).find(|&drop| attested.contains(&segments[drop..].join("."))) {
            *support.entry(segments[..drop].join(".")).or_insert(0) += 1;
        }
    }
    move |module: &str| {
        let segments: Vec<&str> = module.split('.').collect();
        // The best-attested root this path sits under, the longest among equals.
        let root = (0..segments.len())
            .filter_map(|cut| support.get(&segments[..cut].join(".")).map(|&n| (n, cut)))
            .max();
        match root {
            Some((_, cut)) => segments[cut..].join("."),
            None => module.to_owned(),
        }
    }
}

/// Every definition's atoms with parameters resolved — to what the closed-world reading inferred, or
/// to holes — and local paths named the way the tree names them, keeping those that state a fact
/// ([`decide::is_fact`]): resolved, an atom may have lost its identity to a hole, or turned out to
/// filter rather than decide, as a column does against a key no caller fixes. Alongside, per
/// definition, the atoms that hold only through an inferred parameter or field.
struct Resolved {
    atoms: Vec<Vec<Arc<str>>>,
    via: Vec<HashSet<Arc<str>>>,
}

/// What the closed-world reading inferred: each callable's identity, and what each of its parameters
/// (a function's) or fields (a record's) holds, where the corpus says.
#[derive(Default)]
struct Inference {
    ids: HashMap<String, usize>,
    values: HashMap<(usize, String), String>,
}

impl Inference {
    fn param(&self, def: usize, name: &str) -> Option<String> {
        self.values.get(&(def, name.to_owned())).cloned()
    }

    fn field(&self, class: &str, field: &str, named: &impl Fn(&str) -> String) -> Option<String> {
        let at = self.ids.get(&*rewrite_locals(class, |module| named(module)))?;
        self.values.get(&(*at, field.to_owned())).cloned()
    }
}

/// A term or atom as the corpus names it: typed fields and parameters filled by the inference (or
/// erased), local paths named the way the tree imports them.
fn fill(text: &str, def: usize, inference: &Inference, named: &impl Fn(&str) -> String) -> String {
    let typed = decide::resolve_typed_fields(text, |class, field| inference.field(class, field, named));
    let filled = resolve_params(&typed, |p| inference.param(def, p));
    rewrite_locals(&filled, |module| named(module)).to_string()
}

fn resolved_atoms(defs: &[Def], closed_world: bool) -> Resolved {
    let name_of = module_names(defs);
    let named = |module: &str| name_of(module);
    let inference = if closed_world { infer(defs, &named) } else { Inference::default() };
    let mut all = Vec::with_capacity(defs.len());
    let mut via = Vec::with_capacity(defs.len());
    for (i, def) in defs.iter().enumerate() {
        let mut out: Vec<Arc<str>> = Vec::with_capacity(def.facets.decides.len());
        let mut leaned: HashSet<Arc<str>> = HashSet::new();
        for atom in &def.facets.decides {
            let filled = decide::canonical(&fill(atom, i, &inference, &named));
            if !decide::is_fact(&filled) {
                continue;
            }
            let filled: Arc<str> = Arc::from(filled);
            let through_param = params_in(atom).any(|p| inference.param(i, p).is_some());
            let through_field =
                decide::typed_fields_in(atom).any(|(class, field)| inference.field(class, field, &named).is_some());
            if through_param || through_field {
                leaned.insert(Arc::clone(&filled));
            }
            out.push(filled);
        }
        out.sort_unstable();
        out.dedup();
        all.push(out);
        via.push(leaned);
    }
    Resolved { atoms: all, via }
}

/// What each parameter and record field holds, where the corpus can say: interprocedural constant
/// propagation over the calls the corpus shows, under the closed-world reading.
///
/// A parameter holds an identity when every call to its definition passes it, explicitly, as that
/// identity — possibly a parameter of the caller, or a field of a record the caller built, that itself
/// resolves; hence the fixed point. A record's field is a parameter of its constructor and is read the
/// same way, with one more condition: a field is written only by its constructor. A value can reach an
/// instance past it — a keyword to a callee the corpus cannot name (`replace(x, f=…)`), an attribute
/// assigned — so a field name any such write uses is inferred for no record.
///
/// A callable read as a value anywhere (handed on, an attribute of it read, a class queried or passed
/// to a validator), or called once with arguments that cannot be mapped (`*args`, `**kwargs`), is
/// reached from places no call site shows, and nothing of it is inferred.
fn infer(defs: &[Def], named: &impl Fn(&str) -> String) -> Inference {
    let local = |text: &str| rewrite_locals(text, |module| named(module)).to_string();
    let mut ids: HashMap<String, usize> = HashMap::new();
    for (i, def) in defs.iter().enumerate() {
        if let Some(identity) = &def.facets.identity {
            ids.insert(local(identity), i);
        }
    }
    let mut escaped: HashSet<usize> = HashSet::new();
    let mut loose: HashSet<&str> = HashSet::new();
    let mut sites: HashMap<usize, Vec<(usize, &CallSite)>> = HashMap::new();
    for (caller, def) in defs.iter().enumerate() {
        loose.extend(def.facets.loose_keywords.iter().map(|k| &**k));
        for mention in &def.facets.mentions {
            let read = local(mention);
            escaped.extend(reach::prefixes(&read).filter_map(|prefix| ids.get(prefix).copied()));
        }
        for site in &def.facets.calls {
            let callee = local(&site.callee);
            if let Some(&target) = ids.get(&callee) {
                sites.entry(target).or_default().push((caller, site));
            } else {
                // `f.attribute(…)` reads `f` as an object; and whatever an unnamed callee is handed by
                // keyword may land in a record's field of that name.
                escaped.extend(reach::prefixes(&callee).skip(1).filter_map(|prefix| ids.get(prefix).copied()));
                loose.extend(site.args.iter().filter_map(|(keyword, _)| keyword.as_deref()));
            }
        }
    }
    let mut targets: Vec<usize> = sites.keys().copied().filter(|t| !escaped.contains(t)).collect();
    targets.sort_unstable();
    let mut inference = Inference { ids, values: HashMap::new() };
    loop {
        let mut changed = false;
        for &target in &targets {
            let is_record = defs[target].kind.id == "classes";
            for (position, param) in defs[target].facets.params.iter().enumerate() {
                let key = (target, param.to_string());
                if inference.values.contains_key(&key) || (is_record && loose.contains(&**param)) {
                    continue;
                }
                let mut agreed: Option<String> = None;
                let mut unanimous = true;
                for (caller, site) in &sites[&target] {
                    let passed = site
                        .args
                        .iter()
                        .find(|(keyword, _)| keyword.as_deref() == Some(&**param))
                        .or_else(|| site.args.iter().filter(|(keyword, _)| keyword.is_none()).nth(position))
                        .map(|(_, value)| value);
                    let Some(passed) = passed.filter(|_| !site.open) else {
                        unanimous = false;
                        break;
                    };
                    let value = fill(passed, *caller, &inference, named);
                    if !has_identity(&value) || agreed.as_ref().is_some_and(|a| *a != value) {
                        unanimous = false;
                        break;
                    }
                    agreed = Some(value);
                }
                if let (true, Some(value)) = (unanimous, agreed) {
                    inference.values.insert(key, value);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    inference
}

fn corpus_of(defs: &[Def], resolved: &[Vec<Arc<str>>], via: &[HashSet<Arc<str>>]) -> Corpus {
    let mut ids: HashMap<Arc<str>, u32> = HashMap::new();
    let mut text: Vec<String> = Vec::new();
    let mut atoms: Vec<Arc<str>> = Vec::new();
    let mut at = Vec::new();
    let mut facts: Vec<Vec<u32>> = Vec::new();
    let mut inferred: Vec<HashSet<u32>> = Vec::new();
    for (i, def_atoms) in resolved.iter().enumerate() {
        if def_atoms.is_empty() {
            continue;
        }
        let _ = &defs[i];
        let mut own: Vec<u32> = Vec::with_capacity(def_atoms.len());
        let mut leaned: HashSet<u32> = HashSet::new();
        for atom in def_atoms {
            let next = u32::try_from(text.len()).unwrap_or(u32::MAX);
            let id = *ids.entry(Arc::clone(atom)).or_insert_with_key(|key| {
                text.push(decide::render(key));
                atoms.push(Arc::clone(key));
                next
            });
            if via[i].contains(atom) {
                leaned.insert(id);
            }
            own.push(id);
        }
        own.sort_unstable();
        own.dedup();
        at.push(i);
        facts.push(own);
        inferred.push(leaned);
    }
    let mut df = vec![0u32; text.len()];
    for own in &facts {
        for &f in own {
            df[f as usize] += 1;
        }
    }
    let size: Vec<usize> = facts.iter().map(Vec::len).collect();
    // A fact no other member tests can never be shared: it stays in the degree, not in the lists.
    for own in &mut facts {
        own.retain(|&f| df[f as usize] > 1);
        own.sort_unstable_by_key(|&f| (df[f as usize], f));
    }
    Corpus { at, facts, size, df, atoms, text, inferred }
}

// ---------------------------------------------------------------------------
// The null, per pair of degree classes
// ---------------------------------------------------------------------------

/// What the null says about every pair whose rows fall in one pair of degree classes — computed once
/// per class pair, since the model gives such pairs identical laws.
struct ClassNull {
    /// The column classes as (how many facts, the probability both rows hold one).
    classes: Vec<(usize, f64)>,
    /// `P(V ≥ t)` for the shared count, as far as computed.
    survival: Vec<f64>,
    /// The smallest count whose tail is at most the rate, or `usize::MAX` when none is.
    count_crit: usize,
    /// The surprise law, and the smallest surprise whose tail is at most the rate.
    surprise: Surprise,
    surprise_crit: f64,
}

impl ClassNull {
    fn new(model: &Bicm, a: usize, b: usize, alpha: f64, statistic: Statistic) -> Self {
        let classes = model.pair_classes(a, b);
        let most = model.row_degree(a).min(model.row_degree(b));
        let (mut survival, mut count_crit) = (vec![1.0], usize::MAX);
        if statistic == Statistic::Count {
            // Grow the computed tail until it crosses the rate or reaches what the pair can share.
            let mut upto = 4.min(most);
            loop {
                survival = count_survival(&classes, upto);
                if let Some(t) = survival.iter().position(|&p| p <= alpha) {
                    count_crit = t;
                    break;
                }
                if upto >= most {
                    break;
                }
                upto = (upto * 2).min(most);
            }
        }
        let surprise = Surprise::new(&classes);
        let surprise_crit =
            if statistic == Statistic::Surprise { surprise.critical(alpha) } else { f64::INFINITY };
        Self { classes, survival, count_crit, surprise, surprise_crit }
    }

    fn count_tail(&self, shared: usize) -> f64 {
        match self.survival.get(shared) {
            Some(&p) => p,
            None => count_survival(&self.classes, shared)[shared],
        }
    }
}

/// A pair that passed its class pair's critical value at the rate. `cp` is the pair's class pair,
/// carried so the step-up and the tail below never recompute it (two row-class lookups and a
/// min/max per overlap, per iteration).
struct Overlap {
    a: usize,
    b: usize,
    cp: (usize, usize),
    shared: usize,
    surprise: f64,
}

/// An overlap the step-up could reject, with its p-value.
struct Candidate {
    a: usize,
    b: usize,
    shared: usize,
    surprise: f64,
    p: f64,
}

/// Every pair of members sharing a fact, weighed, keeping those the rate could admit at all.
///
/// One pass over an inverted index (fact → members), dense counters per row. No pair is dropped for
/// sharing a common fact: the cut is each class pair's critical value at the rate itself, which no
/// rejected pair can be under — the Benjamini–Hochberg threshold never exceeds the rate.
#[allow(clippy::cast_possible_truncation)] // member indices fit u32 by construction
fn overlaps(
    c: &Corpus,
    model: &Bicm,
    nulls: &HashMap<(usize, usize), ClassNull>,
    row_of: &[usize],
    col_of: &[usize],
    statistic: Statistic,
) -> Vec<Overlap> {
    let n = c.facts.len();
    let mut index: Vec<Vec<u32>> = vec![Vec::new(); c.df.len()];
    for (m, list) in c.facts.iter().enumerate() {
        for &f in list {
            index[f as usize].push(m as u32);
        }
    }
    let mut out: Vec<Overlap> = (0..n)
        .into_par_iter()
        .map_init(
            || (vec![0u32; n], vec![0.0f64; n], Vec::<u32>::new()),
            |(count, weight, touched), a| {
                let ra = row_of[a];
                for &f in &c.facts[a] {
                    // The edge probability depends on the row class and the fact's column class
                    // alone, so `a`'s own term is loop-invariant over the members sharing the fact:
                    // it is taken once here, not once per member.
                    let cf = col_of[f as usize];
                    let own = -model.edge(ra, cf).ln();
                    for &b in &index[f as usize] {
                        let b_at = b as usize;
                        if b_at <= a {
                            continue;
                        }
                        if count[b_at] == 0 {
                            touched.push(b);
                        }
                        count[b_at] += 1;
                        weight[b_at] += own - model.edge(row_of[b_at], cf).ln();
                    }
                }
                let mut found = Vec::new();
                for b in touched.drain(..) {
                    let b_at = b as usize;
                    let (shared, w) = (count[b_at] as usize, weight[b_at]);
                    count[b_at] = 0;
                    weight[b_at] = 0.0;
                    let cp = class_pair(row_of, a, b_at);
                    let null = &nulls[&cp];
                    let admitted = match statistic {
                        Statistic::Count => shared >= null.count_crit,
                        Statistic::Surprise => w >= null.surprise_crit,
                    };
                    if admitted {
                        found.push(Overlap { a, b: b_at, cp, shared, surprise: w });
                    }
                }
                found
            },
        )
        .flatten()
        .collect();
    out.sort_unstable_by_key(|o| (o.a, o.b));
    out
}

fn class_pair(row_of: &[usize], a: usize, b: usize) -> (usize, usize) {
    (row_of[a].min(row_of[b]), row_of[a].max(row_of[b]))
}

/// The overlaps the step-up could reject, each with its p-value.
///
/// The step-up rejects no p-value above `R·α/m`, and `R` is at most the number of pairs that reach
/// that level — so the level a pair must reach shrinks with the count reaching it, down to a fixed
/// point. A saddlepoint tail is paid only past it (the critical value taken at twice the level, so a
/// tail's last digits cannot hide a pair): the threshold over these is the threshold over all, since
/// every p-value under it is among them. A count tail is a table lookup, paid for every overlap.
#[allow(clippy::cast_precision_loss)] // a pair count, far below 2^52
fn candidates(
    overlaps: Vec<Overlap>,
    nulls: &HashMap<(usize, usize), ClassNull>,
    statistic: Statistic,
    tests: f64,
    alpha: f64,
) -> Vec<Candidate> {
    let mut kept = overlaps;
    if statistic == Statistic::Surprise {
        // The class pairs that hold an overlap: `critical` is paid only for these, not the whole
        // row-class × row-class grid, and only the step-up's survivors ever read one. `kept` is
        // narrowed by the loop and never re-scanned, so the set is collected once, up front.
        let mut seen: rustc_hash::FxHashSet<(usize, usize)> = rustc_hash::FxHashSet::default();
        seen.reserve(nulls.len());
        seen.extend(kept.iter().map(|o| o.cp));
        let pairs: Vec<(usize, usize)> = seen.into_iter().collect();
        let mut level = alpha;
        loop {
            let next = kept.len() as f64 * alpha / tests;
            if next >= level {
                break;
            }
            level = next;
            let crit: HashMap<(usize, usize), f64> = pairs
                .par_iter()
                .map(|&cp| (cp, nulls[&cp].surprise.critical(2.0 * level)))
                .collect();
            kept.retain(|o| o.surprise >= crit[&o.cp]);
        }
    }
    kept.into_par_iter()
        .map(|o| {
            let null = &nulls[&o.cp];
            let p = match statistic {
                Statistic::Count => null.count_tail(o.shared),
                Statistic::Surprise => null.surprise.tail(o.surprise),
            };
            Candidate { a: o.a, b: o.b, shared: o.shared, surprise: o.surprise, p }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Findings
// ---------------------------------------------------------------------------

/// What an atom is *about* — its terms that carry an identity, without the relation. Two atoms that
/// test one operand differently are the place two decisions part.
fn operands(atom: &str) -> Vec<&str> {
    decide::parts(atom).1.into_iter().filter(|term| decide::has_identity(term)).collect()
}

/// The facts two members share, rarest first.
fn shared_facts(c: &Corpus, a: usize, b: usize) -> Vec<u32> {
    let theirs: HashSet<u32> = c.facts[b].iter().copied().collect();
    c.facts[a].iter().copied().filter(|f| theirs.contains(f)).collect()
}

/// One shared fact, marked when a member holds it only through a parameter the closed-world reading
/// filled — the reader is told which findings lean on that assumption.
fn fact_line(c: &Corpus, fact: u32, members: &[usize]) -> String {
    if members.iter().any(|&m| c.inferred[m].contains(&fact)) {
        format!("  · {}   (through a parameter every visible call fills with it)", c.text[fact as usize])
    } else {
        format!("  · {}", c.text[fact as usize])
    }
}

/// The verdict line every finding opens with: what was observed, what the null expected, and where
/// the cut fell.
struct Verdict<'a> {
    opts: &'a DecisionOpts,
    threshold: f64,
    tests: f64,
}

fn pair_notes(c: &Corpus, defs: &[Def], model: &Bicm, row_of: &[usize], cand: &Candidate, verdict: &Verdict<'_>) -> Vec<String> {
    let (a, b) = (cand.a, cand.b);
    let shared = shared_facts(c, a, b);
    let classes = model.pair_classes(row_of[a], row_of[b]);
    let statistic = match verdict.opts.statistic {
        Statistic::Count => format!("{} shared", cand.shared),
        Statistic::Surprise => format!("surprise {:.1} nats (null mean {:.1})", cand.surprise, Surprise::new(&classes).mean()),
    };
    let mut notes = vec![format!(
        "decide on {} shared fact(s), {:.2} expected by chance; {statistic}; p = {:.1e} under the bipartite configuration model, Benjamini–Hochberg cut {:.1e} at FDR {} over {:.0} pairs",
        shared.len(),
        expected_count(&classes),
        cand.p,
        verdict.threshold,
        verdict.opts.fdr,
        verdict.tests,
    )];
    notes.extend(shared.iter().map(|&f| fact_line(c, f, &[a, b])));
    // Where they part: a fact one side tests about an operand both already test.
    let common: HashSet<&str> = shared.iter().flat_map(|&f| operands(&c.atoms[f as usize])).collect();
    let shared_set: HashSet<u32> = shared.iter().copied().collect();
    for (side, other) in [(a, b), (b, a)] {
        let mut only: Vec<&str> = c.facts[side]
            .iter()
            .filter(|f| !shared_set.contains(f) && !c.facts[other].contains(f))
            .filter(|&&f| operands(&c.atoms[f as usize]).iter().any(|o| common.contains(o)))
            .map(|&f| c.text[f as usize].as_str())
            .collect();
        only.sort_unstable();
        if !only.is_empty() {
            notes.push(format!("only {}:", defs[c.at[side]].name));
            notes.extend(only.iter().map(|t| format!("  ~ {t}")));
        }
    }
    notes
}

/// The family's notes and the size of its core — the facts every member decides on.
fn family_notes(c: &Corpus, defs: &[Def], members: &[usize], weakest: f64, verdict: &Verdict<'_>) -> (Vec<String>, usize) {
    let mut held: HashMap<u32, Vec<usize>> = HashMap::new();
    for &m in members {
        for &f in &c.facts[m] {
            held.entry(f).or_default().push(m);
        }
    }
    let mut core: Vec<u32> = held.iter().filter(|(_, who)| who.len() == members.len()).map(|(&f, _)| f).collect();
    core.sort_by(|x, y| c.df[*x as usize].cmp(&c.df[*y as usize]).then(x.cmp(y)));
    let mut notes = vec![format!(
        "{} definitions decide on {} common fact(s); every pair of them significant, the weakest at p = {weakest:.1e} (Benjamini–Hochberg cut {:.1e} at FDR {})",
        members.len(),
        core.len(),
        verdict.threshold,
        verdict.opts.fdr,
    )];
    notes.extend(core.iter().map(|&f| fact_line(c, f, members)));
    let mut partial: Vec<(u32, &Vec<usize>)> =
        held.iter().filter(|(_, who)| who.len() > 1 && who.len() < members.len()).map(|(&f, who)| (f, who)).collect();
    partial.sort_by(|x, y| y.1.len().cmp(&x.1.len()).then(x.0.cmp(&y.0)));
    if !partial.is_empty() {
        notes.push("not all of them:".to_owned());
        for (f, who) in partial {
            let names: Vec<&str> = who.iter().map(|&m| defs[c.at[m]].name.as_str()).collect();
            notes.push(format!("  ~ {}  [{} of {}: {}]", c.text[f as usize], who.len(), members.len(), names.join(", ")));
        }
    }
    (notes, core.len())
}

fn finding(defs: &[Def], c: &Corpus, pass: &'static str, members: &[usize], thickness: f64, shared: usize, notes: Vec<String>) -> Finding {
    let idx: Vec<usize> = members.iter().map(|&m| c.at[m]).collect();
    let first = &defs[idx[0]];
    Finding {
        pass,
        kind: first.kind,
        name: idx.iter().map(|&i| defs[i].name.as_str()).collect::<Vec<_>>().join(" / "),
        severity: Severity::Info,
        min_sim: None,
        loc: idx.iter().map(|&i| defs[i].loc).max().unwrap_or(0),
        args: idx.iter().map(|&i| defs[i].args).max().unwrap_or(0),
        thickness,
        snippet: first.text_orig.clone(),
        notes,
        members: idx.iter().map(|&i| member(defs, i)).collect(),
        pattern: None,
        facets: vec![("decides".to_owned(), shared), ("members".to_owned(), members.len())],
    }
}

/// How far below the cut a p-value lies, in `[0, 1]`: the share of its evidence (in log-probability)
/// beyond what the cut asks for. Zero at the cut, a half at its square.
fn thickness_of(p: f64, threshold: f64) -> f64 {
    if p <= 0.0 {
        1.0
    } else if p >= threshold || threshold >= 1.0 {
        0.0
    } else {
        1.0 - threshold.ln() / p.ln()
    }
}

/// Find definitions that decide on the same facts. `already` are the findings the body passes made
/// this run: a pair one of them clusters is a copy, reported there.
#[must_use]
#[allow(clippy::cast_precision_loss)] // a pair count, far below 2^52
pub fn pass_decisions(defs: &[Def], already: &[Finding], opts: &DecisionOpts) -> Vec<Finding> {
    let Resolved { atoms: resolved, via } = resolved_atoms(defs, opts.closed_world);
    let c = corpus_of(defs, &resolved, &via);
    let n = c.facts.len();
    if n < 2 {
        return Vec::new();
    }
    if let Some(path) = std::env::var_os("FDD_DUMP_DECIDES") {
        // Dev hook, like the snapshot dumps: every member's atoms as JSON lines `[file, line, name,
        // atoms]`, named the way findings name them — what an evaluation labels pairs against.
        let mut dump = String::new();
        for &i in &c.at {
            let (file, line, _) = member(defs, i);
            let atoms: Vec<String> = resolved[i].iter().map(|a| decide::render(a)).collect();
            dump.push_str(&serde_json::to_string(&(file, line, &defs[i].name, atoms)).unwrap_or_default());
            dump.push('\n');
        }
        let _ = std::fs::write(path, dump);
    }
    let model = Bicm::fit(&c.size, &c.df);
    let row_of: Vec<usize> = c.size.iter().map(|&k| model.row_class(k)).collect();
    let col_of: Vec<usize> = c.df.iter().map(|&d| model.col_class(d)).collect();
    let class_pairs: Vec<(usize, usize)> =
        (0..model.row_classes()).flat_map(|a| (a..model.row_classes()).map(move |b| (a, b))).collect();
    let nulls: HashMap<(usize, usize), ClassNull> = class_pairs
        .par_iter()
        .map(|&(a, b)| ((a, b), ClassNull::new(&model, a, b, opts.fdr, opts.statistic)))
        .collect();
    let admitted = overlaps(&c, &model, &nulls, &row_of, &col_of, opts.statistic);
    let admitted_count = admitted.len();
    let tests = n as f64 * (n as f64 - 1.0) / 2.0;
    let found = candidates(admitted, &nulls, opts.statistic, tests, opts.fdr);
    let p_values: Vec<f64> = found.iter().map(|cand| cand.p).collect();
    let threshold = bh_threshold(&p_values, tests, opts.fdr);
    if std::env::var_os("FDD_TIMING").is_some() {
        eprintln!(
            "[decisions] members={n} facts={} row classes={} fit residual={:.1e} candidates={admitted_count} weighed={} BH threshold={threshold:?}",
            c.df.len(),
            model.row_classes(),
            model.residual,
            found.len(),
        );
    }
    let Some(observed) = threshold else { return Vec::new() };
    // The cut a reader is shown is the step-up line at the last rejection, `R·α/m` — what the procedure
    // compared against — rather than the p-value that happened to sit under it: measured from the
    // latter, the weakest finding would always read as having no evidence to spare.
    let rejected = p_values.iter().filter(|&&p| p <= observed).count();
    let threshold = rejected as f64 * opts.fdr / tests;
    let verdict = Verdict { opts, threshold, tests };

    // The body-pass groups each member sits in: a pair is a copy when its two lists meet.
    let mut groups_of: HashMap<(String, usize, usize), Vec<usize>> = HashMap::new();
    for (group, f) in already.iter().filter(|f| matches!(f.pass, "name" | "cross-name" | "type-3")).enumerate() {
        for key in &f.members {
            groups_of.entry(key.clone()).or_default().push(group);
        }
    }
    let groups_at = |m: usize| groups_of.get(&member(defs, c.at[m])).map_or(&[][..], Vec::as_slice);
    let is_copy = |a: usize, b: usize| {
        let theirs = groups_at(b);
        groups_at(a).iter().any(|g| theirs.contains(g))
    };
    let pairs: Vec<&Candidate> = found.iter().filter(|cand| cand.p <= observed && !is_copy(cand.a, cand.b)).collect();

    // Families: greedy maximal cliques over the significant pairs — every member significant with
    // every other, so the family's claim holds pairwise, not by a chain.
    let mut edges: rustc_hash::FxHashMap<u32, rustc_hash::FxHashSet<u32>> = rustc_hash::FxHashMap::default();
    let mut p_of: HashMap<(usize, usize), f64> = HashMap::new();
    for cand in &pairs {
        let (a, b) = (u32::try_from(cand.a).unwrap_or(u32::MAX), u32::try_from(cand.b).unwrap_or(u32::MAX));
        edges.entry(a).or_default().insert(b);
        edges.entry(b).or_default().insert(a);
        p_of.insert((cand.a, cand.b), cand.p);
    }
    let mut vertices: Vec<u32> = edges.keys().copied().collect();
    vertices.sort_unstable();
    let mut in_family: HashMap<usize, usize> = HashMap::new();
    let mut out = Vec::new();
    for clique in crate::converge::clique_cover(&vertices, &edges) {
        if clique.len() < crate::converge::MIN_FAMILY {
            continue;
        }
        let mut members: Vec<usize> = clique.iter().map(|&v| v as usize).collect();
        members.sort_unstable();
        let id = out.len();
        for &m in &members {
            in_family.insert(m, id);
        }
        let weakest = members
            .iter()
            .enumerate()
            .flat_map(|(i, &x)| members[i + 1..].iter().map(move |&y| (x, y)))
            .filter_map(|key| p_of.get(&key).copied())
            .fold(0.0_f64, f64::max);
        let (notes, common) = family_notes(&c, defs, &members, weakest, &verdict);
        out.push(finding(defs, &c, "decision-family", &members, thickness_of(weakest, threshold), common, notes));
    }
    for cand in &pairs {
        if in_family.get(&cand.a).is_some_and(|fa| in_family.get(&cand.b) == Some(fa)) {
            continue;
        }
        let notes = pair_notes(&c, defs, &model, &row_of, cand, &verdict);
        let shared = shared_facts(&c, cand.a, cand.b).len();
        out.push(finding(defs, &c, "decision", &[cand.a, cand.b], thickness_of(cand.p, threshold), shared, notes));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{operands, pass_decisions, thickness_of, DecisionOpts, Statistic};
    use dup_defs_core::decide::CallSite;
    use dup_defs_core::kinds::FUNCTIONS;
    use dup_defs_core::{Def, Facets};
    use std::sync::Arc;

    fn def(name: &str, file: &str, atoms: &[&str]) -> Def {
        Def {
            lang: "test",
            kind: &FUNCTIONS,
            name: name.to_owned(),
            file: Arc::from(file),
            line: 0,
            col: 0,
            loc: 3,
            args: 1,
            text_orig: String::new(),
            cluster_canonical: None,
            analysis: None,
            thickness: None,
            facets: Facets {
                decides: atoms.iter().map(|a| Arc::from(a.replace(' ', "\u{1f}"))).collect(),
                ..Facets::default()
            },
        }
    }

    /// Unrelated definitions, each testing a fact of its own, one most of the tree tests, and one it
    /// shares with three others — plus one pair deciding on the same five rare facts, written apart.
    /// Five, because twenty thousand pairs are tested: three shared rare facts happen to some pair by
    /// chance often enough that a false-discovery rate of 5% rightly declines to call them a finding.
    fn corpus() -> Vec<Def> {
        let mut defs: Vec<Def> = (0..200)
            .map(|i| {
                let own = format!("= _.f{i} {i}");
                let near = format!("? _.g{}", i % 50);
                def(&format!("noise{i}"), &format!("n{i}.py"), &["= _ None", &own, &near])
            })
            .collect();
        let rule = ["= @app.State.OPEN _.state", "? _.paid", "in _.code _.allowed", "< _.total @app.LIMIT", "? _.verified"];
        let mut refuse = rule.to_vec();
        refuse.push("= _ None");
        let mut offer: Vec<&str> = rule.iter().rev().copied().collect();
        offer.push("? _.extra");
        defs.push(def("refuse", "a.py", &refuse));
        defs.push(def("offer", "b.py", &offer));
        defs
    }

    fn opts(statistic: Statistic) -> DecisionOpts {
        DecisionOpts { fdr: 0.05, statistic, closed_world: false }
    }

    fn callable(name: &str, atoms: &[&str], params: &[&str], calls: Vec<CallSite>, mentions: &[&str]) -> Def {
        let mut d = def(name, &format!("{name}.py"), atoms);
        d.facets.identity = Some(Arc::from(format!("@app.{name}")));
        d.facets.params = params.iter().map(|p| Arc::from(*p)).collect();
        d.facets.calls = calls;
        d.facets.mentions = mentions.iter().map(|m| Arc::from(*m)).collect();
        d
    }

    fn call(callee: &str, args: &[(Option<&str>, &str)]) -> CallSite {
        CallSite {
            callee: Arc::from(callee),
            args: args.iter().map(|(k, v)| (k.map(Arc::from), Arc::from(*v))).collect(),
            open: false,
        }
    }

    fn atoms_of(defs: &[Def], closed_world: bool, name: &str) -> Vec<String> {
        let resolved = super::resolved_atoms(defs, closed_world).atoms;
        let at = defs.iter().position(|d| d.name == name).unwrap_or(0);
        resolved[at].iter().map(|a| dup_defs_core::decide::render(a)).collect()
    }

    #[test]
    fn a_parameter_every_call_fills_alike_is_that_value_only_in_a_closed_world() {
        let defs = vec![
            callable("helper", &["? $flag", "= _.kind @app.K.A"], &["flag"], Vec::new(), &[]),
            callable("one", &[], &[], vec![call("@app.helper", &[(None, "@app.cfg.on")])], &[]),
            callable("two", &[], &[], vec![call("@app.helper", &[(Some("flag"), "@app.cfg.on")])], &[]),
        ];
        assert_eq!(atoms_of(&defs, true, "helper"), vec!["@app.K.A = _.kind", "? @app.cfg.on"]);
        // Without the assumption the parameter is the hole it always was, and its atom says nothing.
        assert_eq!(atoms_of(&defs, false, "helper"), vec!["@app.K.A = _.kind"]);
    }

    fn record(name: &str, fields: &[&str]) -> Def {
        let mut d = def(name, &format!("{name}.py"), &[]);
        d.kind = &dup_defs_core::kinds::CLASSES;
        d.facets.identity = Some(Arc::from(format!("@app.{name}")));
        d.facets.params = fields.iter().map(|f| Arc::from(*f)).collect();
        d
    }

    #[test]
    fn a_record_field_every_constructor_fills_alike_is_that_value() {
        let defs = vec![
            record("Snap", &["tariff", "mode"]),
            callable("build", &[], &[], vec![call("@app.Snap", &[(Some("tariff"), "_.subtype"), (Some("mode"), "@app.M.A")])], &[]),
            callable("other", &[], &[], vec![call("@app.Snap", &[(None, "_.subtype"), (None, "@app.M.B")])], &[]),
            callable("reader", &["= $code %@~app~Snap#tariff", "= %@~app~Snap#mode @app.M.A"], &[], Vec::new(), &[]),
        ];
        // `tariff` is the same at both constructions; `mode` is not, and stays the erased field. The
        // reader names the class as a local path, the way a module that defines it does.
        assert_eq!(atoms_of(&defs, true, "reader"), vec!["@app.M.A = _.mode", "_ = _.subtype"]);
        assert_eq!(atoms_of(&defs, false, "reader"), vec!["@app.M.A = _.mode", "_ = _.tariff"]);
    }

    #[test]
    fn a_field_written_past_its_constructor_is_not_inferred() {
        let mut writer = callable("writer", &[], &[], Vec::new(), &[]);
        writer.facets.loose_keywords = vec![Arc::from("tariff")];
        let defs = vec![
            record("Snap", &["tariff"]),
            callable("build", &[], &[], vec![call("@app.Snap", &[(Some("tariff"), "_.subtype")])], &[]),
            writer,
            callable("reader", &["= $code %@app.Snap#tariff"], &[], Vec::new(), &[]),
        ];
        assert_eq!(atoms_of(&defs, true, "reader"), vec!["_ = _.tariff"]);
    }

    #[test]
    fn disagreeing_or_escaping_callers_infer_nothing() {
        let differ = vec![
            callable("helper", &["? $flag"], &["flag"], Vec::new(), &[]),
            callable("one", &[], &[], vec![call("@app.helper", &[(None, "@app.cfg.on")])], &[]),
            callable("two", &[], &[], vec![call("@app.helper", &[(None, "@app.cfg.off")])], &[]),
        ];
        let inferred = atoms_of(&differ, true, "helper");
        assert!(inferred.is_empty(), "callers disagree on the argument, yet it inferred: {inferred:?}");
        let escapes = vec![
            callable("helper", &["? $flag"], &["flag"], Vec::new(), &[]),
            callable("one", &[], &[], vec![call("@app.helper", &[(None, "@app.cfg.on")])], &[]),
            callable("registry", &[], &[], Vec::new(), &["@app.helper"]),
        ];
        let inferred = atoms_of(&escapes, true, "helper");
        assert!(inferred.is_empty(), "the callable escapes into a registry, yet it inferred: {inferred:?}");
    }

    #[test]
    fn a_value_passed_through_a_callers_parameter_resolves_at_the_fixed_point() {
        let defs = vec![
            callable("inner", &["? $flag"], &["flag"], Vec::new(), &[]),
            callable("outer", &[], &["mode"], vec![call("@app.inner", &[(None, "$mode")])], &[]),
            callable("entry", &[], &[], vec![call("@app.outer", &[(None, "@app.cfg.on")])], &[]),
        ];
        assert_eq!(atoms_of(&defs, true, "inner"), vec!["? @app.cfg.on"]);
    }

    #[test]
    fn two_queries_on_one_table_share_its_keys_and_decide_on_nothing() {
        // Two repository functions filtering one table by its keys and a date, joining its lines and
        // keeping the open rows: the key filters, the bound and the join select rows and decide
        // nothing, so one fact is left to share and the pair is no finding — written as five facts it
        // would be, as `refuse / offer` is.
        let query = [
            "= @app.db.Order.shop_id _",
            "= @app.db.Order.customer_id _",
            "< @app.db.Order.placed_at _",
            "= @app.db.Line.order_id @app.db.Order.id",
            "= @app.db.Order.status @app.Status.OPEN",
        ];
        let mut defs = corpus();
        defs.push(def("open_orders", "r.py", &query));
        defs.push(def("count_open", "r.py", &query));
        assert_eq!(atoms_of(&defs, false, "open_orders"), vec!["@app.Status.OPEN = @app.db.Order.status"]);
        let names: Vec<String> = pass_decisions(&defs, &[], &opts(Statistic::Surprise)).iter().map(|f| f.name.clone()).collect();
        assert_eq!(names, vec!["refuse / offer"]);
    }

    #[test]
    fn a_module_nothing_imports_is_named_under_the_root_its_neighbours_attest() {
        // Scanned at `src/app/…`: `app.models` is imported by that name, so `src` is the root the
        // tree drops, and the helpers module nothing imports is named under it too.
        let mut models = def("State", "src/app/models.py", &[]);
        models.facets.identity = Some(Arc::from("@~src.app.models~State"));
        let mut check = callable("check", &["= _.state @~src.app.models~State.OPEN"], &[], Vec::new(), &[]);
        check.facets.identity = Some(Arc::from("@~src.app.rules~check"));
        check.facets.reaches = vec![Arc::from("app.models")];
        let mut helper = callable("helper", &["= _.kind @~src.app.helpers~KIND"], &[], Vec::new(), &[]);
        helper.facets.identity = Some(Arc::from("@~src.app.helpers~helper"));
        let defs = vec![models, check, helper];
        assert_eq!(atoms_of(&defs, false, "check"), vec!["@app.models.State.OPEN = _.state"]);
        assert_eq!(atoms_of(&defs, false, "helper"), vec!["@app.helpers.KIND = _.kind"]);
    }

    #[test]
    fn a_decision_written_twice_is_found_and_chance_overlap_is_not() {
        let found = pass_decisions(&corpus(), &[], &opts(Statistic::Surprise));
        let names: Vec<&str> = found.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["refuse / offer"]);
        assert!(found[0].notes.iter().any(|n| n.contains("@app.State.OPEN = _.state")));
    }

    #[test]
    fn the_count_statistic_cannot_tell_a_rare_fact_from_a_common_one() {
        // `offer` does not test the fact nearly every definition tests, so the five rare facts it
        // shares with `refuse` are, by count, not far from what the null expects any pair to share —
        // the rarity that makes them evidence is exactly what a count throws away.
        assert!(pass_decisions(&corpus(), &[], &opts(Statistic::Count)).is_empty(), "by count the rare facts are not evidence");
    }

    #[test]
    fn a_pair_a_body_pass_already_clusters_is_left_to_it() {
        let defs = corpus();
        let mut already = pass_decisions(&defs, &[], &opts(Statistic::Surprise));
        for f in &mut already {
            f.pass = "type-3";
        }
        assert!(pass_decisions(&defs, &already, &opts(Statistic::Surprise)).is_empty(), "a pair the body pass already holds is not reported twice");
    }

    #[test]
    fn the_run_reproduces() {
        let defs = corpus();
        let run = || -> Vec<String> {
            pass_decisions(&defs, &[], &opts(Statistic::Surprise)).iter().flat_map(|f| f.notes.clone()).collect()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn thickness_is_the_share_of_evidence_past_the_cut() {
        assert!((thickness_of(1e-12, 1e-6) - 0.5).abs() < 1e-12);
        assert!(thickness_of(1e-6, 1e-6).abs() < 1e-12);
        assert!((thickness_of(0.0, 1e-6) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_atom_is_about_its_identities_not_its_relation() {
        let sep = |s: &str| s.replace(' ', "\u{1f}");
        assert_eq!(operands(&sep("= @app.State.OPEN _.state")), vec!["@app.State.OPEN", "_.state"]);
        assert_eq!(operands(&sep("= _ None")), vec!["None"]);
        // A literal holding the characters a rendering is written with stays one operand.
        assert_eq!(operands("=\u{1f}_.note\u{1f}\"a = b < c\""), vec!["_.note", "\"a = b < c\""]);
    }

    // ── inference: module naming ──────────────────────────────────────────

    #[test]
    fn a_module_named_under_two_roots_takes_the_better_attested_one() {
        // `src` is attested by two modules, `src.app` by one. A helper under `src.app.helpers`
        // (nothing imports it) must be named under `src`, the root its neighbours attest more.
        let mut m1 = def("m1", "src/app/models.py", &[]);
        m1.facets.identity = Some(Arc::from("@~src.app.models~M1"));
        m1.facets.reaches = vec![Arc::from("app.models")];
        let mut m2 = def("m2", "src/app/rules.py", &[]);
        m2.facets.identity = Some(Arc::from("@~src.app.rules~M2"));
        m2.facets.reaches = vec![Arc::from("app.rules")];
        let mut m3 = def("m3", "src/app/special.py", &[]);
        m3.facets.identity = Some(Arc::from("@~src.app.special~M3"));
        m3.facets.reaches = vec![Arc::from("special")];
        let mut helper = callable("helper", &["= _.kind @~src.app.helpers~KIND"], &[], Vec::new(), &[]);
        helper.facets.identity = Some(Arc::from("@~src.app.helpers~helper"));
        let defs = vec![m1, m2, m3, helper];
        assert_eq!(atoms_of(&defs, false, "helper"), vec!["@app.helpers.KIND = _.kind"]);
    }

    // ── inference: through-parameter / through-field marking ───────────────

    #[test]
    fn an_atom_filled_through_a_parameter_alone_is_marked_as_inferred() {
        let defs = vec![
            callable("helper", &["? $flag"], &["flag"], Vec::new(), &[]),
            callable("one", &[], &[], vec![call("@app.helper", &[(None, "@app.cfg.on")])], &[]),
        ];
        let resolved = super::resolved_atoms(&defs, true);
        let at = defs.iter().position(|d| d.name == "helper").unwrap();
        assert_eq!(resolved.via[at].len(), 1, "the param-filled atom is held only through inference");
    }

    // ── corpus construction ───────────────────────────────────────────────

    #[test]
    fn corpus_of_drops_facts_only_one_member_tests() {
        let defs = vec![
            def("a", "a.py", &["= @app.State.OPEN _.state", "= @app.State.CLOSED _.state"]),
            def("b", "b.py", &["= @app.State.OPEN _.state"]),
        ];
        let resolved = super::resolved_atoms(&defs, false);
        let c = super::corpus_of(&defs, &resolved.atoms, &resolved.via);
        for (m, own) in c.facts.iter().enumerate() {
            for &f in own {
                assert!(c.df[f as usize] > 1, "member {m} still holds a fact nothing else tests");
            }
        }
    }

    // ── thickness ─────────────────────────────────────────────────────────

    #[test]
    fn thickness_is_zero_past_the_cut_or_a_unit_threshold() {
        assert_eq!(thickness_of(0.9, 0.5), 0.0, "a p-value above the cut has no evidence to spare");
        assert_eq!(thickness_of(0.3, 1.5), 0.0, "a threshold at or above one is not a cut at all");
    }

    // ── notes ─────────────────────────────────────────────────────────────

    #[test]
    fn pair_notes_lists_where_each_side_parts_on_a_common_operand() {
        // Both decide on `_.state`; `a` also rules the CLOSED value of it (which a third member
        // tests too, so the fact is shared enough to stay in the corpus), so the pair note names
        // where the two part — the fact `a` alone tests about an operand they both already test.
        let defs = vec![
            def("a", "a.py", &["= @app.State.OPEN _.state", "= @app.State.CLOSED _.state"]),
            def("b", "b.py", &["= @app.State.OPEN _.state"]),
            def("d", "d.py", &["= @app.State.CLOSED _.state"]),
        ];
        let resolved = super::resolved_atoms(&defs, false);
        let c = super::corpus_of(&defs, &resolved.atoms, &resolved.via);
        let model = crate::bicm::Bicm::fit(&c.size, &c.df);
        let row_of: Vec<usize> = c.size.iter().map(|&k| model.row_class(k)).collect();
        let cand = super::Candidate { a: 0, b: 1, shared: 1, surprise: 0.5, p: 1e-3 };
        let verdict = super::Verdict { opts: &DecisionOpts::default(), threshold: 1e-6, tests: 1.0 };
        let notes = super::pair_notes(&c, &defs, &model, &row_of, &cand, &verdict);
        assert!(notes.iter().any(|n| n.contains("only a:")), "got: {notes:?}");
    }

    #[test]
    fn family_notes_counts_the_core_and_marks_partial_facts() {
        // OPEN is decided by all three; CLOSED by two of them — one common fact, one partial.
        let defs = vec![
            def("a", "a.py", &["= @app.State.OPEN _.state", "= @app.State.CLOSED _.state"]),
            def("b", "b.py", &["= @app.State.OPEN _.state", "= @app.State.CLOSED _.state"]),
            def("c", "c.py", &["= @app.State.OPEN _.state"]),
        ];
        let resolved = super::resolved_atoms(&defs, false);
        let c = super::corpus_of(&defs, &resolved.atoms, &resolved.via);
        let verdict = super::Verdict { opts: &DecisionOpts::default(), threshold: 1e-6, tests: 3.0 };
        let (notes, core) = super::family_notes(&c, &defs, &[0, 1, 2], 1e-6, &verdict);
        assert_eq!(core, 1, "exactly one fact is held by every member");
        assert!(notes.iter().any(|n| n.contains("· @app.State.OPEN = _.state")), "the OPEN fact is the core: {notes:?}");
        assert!(notes.iter().any(|n| n.contains("not all of them:")), "the CLOSED fact is partial: {notes:?}");
    }
}
