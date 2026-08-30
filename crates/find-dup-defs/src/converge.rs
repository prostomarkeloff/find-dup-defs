//! **Converge** — one definition's *divergence* from another, by two anchors on one currency.
//!
//! Every other pass here answers "are these two the same". This one answers "these two are about
//! the same thing — *where do they stop agreeing*", and reports the step rather than the cluster.
//!
//! ## Two anchors, because one is structurally blind
//!
//! A pass keyed on a **shared statement** can only see divergence that grew out of textual
//! agreement: a copy that drifted, or two paths that converged. It cannot reach the opposite case —
//! two places written independently about the same thing, with no line in common. Measured on an
//! application corpus that blind spot is real: a pair of functions answering one question ("does
//! this channel fit the plan") shared exactly one name between them, and that name was `int`.
//!
//! What such a pair does share is a **subject**: both reach the same module.
//! [`Facets::reaches`](dup_defs_core::Facets::reaches) is the frontend's answer to that, and the
//! prefix lattice over the module tree is how "imported the module" and "imported a member of it"
//! are made to meet.
//!
//! - **statement anchor** — same words, different names ⇒ one decision made in two ways;
//! - **subject anchor** — same entity and same shape, different words ⇒ one procedure written twice.
//!
//! ## One currency
//!
//! The seed decides only what the report points at, never how a pair is weighed:
//!
//! ```text
//! score = (E_text + E_shape + E_subject) * D * sharpness * novelty / members
//! ```
//!
//! **E** is how surprising the coincidence is, in nats, over the three ways two definitions can
//! evidently be one thing done twice: the run they share line for line, the shapes they share among
//! the lines they word differently, and the rarity of the deepest module both reach. **D** is the
//! rarity of the rarest name they part on. **novelty** is `1` when they found each other again after
//! the gap and `1 - jaccard` when they parted for good — for a drifted copy alikeness is the premise,
//! for a permanent fork it means a similarity pass already has the pair. **members** divides by how
//! many places share the run: read against real code, a divergence between exactly two places was
//! worth acting on 74% of the time and one among three or more 16%.
//!
//! Every input is a [`Facets`](dup_defs_core::Facets) field, so this is language-blind: a frontend
//! that fills them lights the pass up with no edit here and no list of supported languages anywhere.
//!
//! ## Advisory, always
//!
//! Findings are [`Severity::Info`]. The output is a ranked list with no threshold, which is not a
//! thing to gate a build on — the pass exists to be read, and a gate that fires on the tail of a
//! ranking teaches people to ignore it.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use rayon::prelude::*;

use dup_defs_core::{CanonDialect, Def, KindSpec};
use dup_defs_core::reach::prefixes;

use crate::{member, Finding, Severity};

/// Report slot for this pass's pair sections, past the body passes and the patternology appendix.
pub const CONVERGE_SECTION_OFFSET: usize = 2000;

/// Report slot for the **family** sections — the same pass answering a different question, so its own
/// slot rather than a mixed section a reader has to sort by eye.
pub const FAMILY_SECTION_OFFSET: usize = 3000;

/// Below this a group is a pair, and a pair is what the rest of the pass reports. Three is where
/// "these two" becomes "these all", which is the whole distinction the family rubric exists to draw.
const MIN_FAMILY: usize = 3;

/// What bounds the report: how many findings of each kind it prints, and how many places a shared
/// statement may occur in before the pass calls it an idiom.
#[derive(Clone, Copy)]
struct Limits {
    top: usize,
    cap: usize,
}

/// A shared line that occurs in more places than this is an idiom, and pairing all of its sites is
/// quadratic for nothing. The same reasoning, and the same cap, applies to a subject everything
/// reaches: that is infrastructure, not a thing two definitions are *about*.
///
/// It is also the pass's dominant cost knob — the work it admits is QUADRATIC in it — and it is
/// exposed as `settings:converge-cap=N`. **Sixty is the right default and lowering it is not a free
/// speedup**, which took two measurements to establish because the first one lied:
///
/// | cap | converge (mono) | findings lost, mono | findings lost, mixed |
/// |-----|-----------------|---------------------|----------------------|
/// |  60 |         5478 ms |                  0% |                   0% |
/// |  40 |         3249 ms |                0.0% |               38.9% |
/// |  30 |         1873 ms |                0.0% |               53.4% |
/// |  20 |         1412 ms |                0.0% |               69.9% |
///
/// 🔴 On a duplication-heavy monorepo the cap almost never binds — 86% of bodies there are copies of
/// another, so a shared statement has few DISTINCT sites and the count stays at 44 777 whatever the
/// cap. Read alone, that says the cap costs nothing and buys 3.9x. It is a property of that corpus,
/// not of the cap. On ordinary code — a standard library, a framework — statements genuinely occur
/// in twenty to sixty places, the cap bites, and cutting it to 20 deletes **seven findings in ten**.
///
/// The lesson is about the measurement, not the constant: a knob whose cost depends on the shape of
/// the input has to be priced on more than one shape.
pub const SEED_CAP: usize = 60;

/// How far apart two streams may drift before we stop believing they are still the same block.
const GAP_MAX: usize = 3;

/// Definitions longer than this are not compared: the alignment is quadratic, and past a few hundred
/// statements a "divergence" is a whole other function rather than a step.
const MAX_STATEMENTS: usize = 200;

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

/// One token of a canonical line.
#[derive(PartialEq, Eq, Clone, Copy)]
enum Tok<'a> {
    /// A name the corpus itself chose.
    Name(&'a str),
    /// `_v{n}` / `_s{n}` / `_fn` / `_` — grammar, not identity.
    Slot(&'a str),
    /// A quoted or numeric literal, taken whole.
    Lit(&'a str),
    /// Operators, brackets, punctuation.
    Punct(&'a str),
}

fn is_slot(token: &str) -> bool {
    token == "_"
        || token == "_fn"
        || token
            .strip_prefix("_v")
            .or_else(|| token.strip_prefix("_s"))
            .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
}

/// Which vocabulary a canonical is written in — what counts as a *name the corpus chose* rather than
/// grammar.
///
/// 🔴 This has to be dialect-aware, and getting it wrong is silent. Python's canonical is source-like:
/// identifiers are bare and quoted runs are strings. The Rust and TypeScript canonicals are s-expr
/// dumps where it is the other way round — `Let(Bind('_v1'), Call(Path('open_session'), Path('_v0')))`
/// — so the bare words are AST node tags and the identifiers live inside the quotes.
///
/// Read with one rule, the two disagree about everything that matters. On Rust the "names they parted
/// on" came out as `Path`, `Method`, `Bind`, `Ref` — node tags, every finding parting on the same
/// dozen. On Python, with the keyword list dropped, they came out as `if`, `return`, `None`, `await`.
/// Both are the grammar reported as the decision.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Vocab {
    /// Source-like: bare identifiers are names, keywords are grammar, quotes are literals.
    Source,
    /// S-expression dump: bare words are node tags, and the identifiers are inside the quotes.
    SExpr,
}

impl Vocab {
    fn of(dialect: CanonDialect) -> Self {
        match dialect {
            CanonDialect::CPythonAst => Vocab::Source,
            _ => Vocab::SExpr,
        }
    }
}

/// Python's reserved words — the fixed grammar of the source-like dialect, not a list of things
/// deemed uninteresting.
const KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "case", "class", "continue", "def",
    "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
    "match", "nonlocal", "not", "or", "pass", "raise", "return", "try", "type", "while", "with", "yield",
];

/// Is this the content of a quoted run an identifier rather than a message?
///
/// The s-expr dialects quote both — `Path('open_session')` and `Str('failed to reach {}')` — and only
/// the shape of the content tells them apart. A message has spaces or punctuation; an identifier does
/// not. A quoted run that *is* identifier-shaped but was written as a string constant (a sentinel like
/// `'contacts'`) reads as a name, which is right: a sentinel carries identity exactly as a name does.
fn quoted_is_identifier(body: &str) -> bool {
    !body.is_empty()
        && body.bytes().next().is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && body.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b':' || b == b'.')
}

/// Split a canonical line into tokens under one vocabulary.
///
/// A quoted run is **one** token either way: in the source dialect what a message says is not what a
/// statement does, and in the s-expr dialect the quotes are how an identifier is spelled.
fn tokens(line: &str, vocab: Vocab) -> Vec<Tok<'_>> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        match bytes[i] {
            b'\'' | b'"' => {
                let quote = bytes[i];
                i += 1;
                let body_start = i;
                let mut escaped = false;
                while i < bytes.len() {
                    let ch = bytes[i];
                    i += 1;
                    if escaped {
                        escaped = false;
                    } else if ch == b'\\' {
                        escaped = true;
                    } else if ch == quote {
                        break;
                    }
                }
                let body_end = if i > body_start && bytes[i - 1] == quote { i - 1 } else { i };
                let body = &line[body_start..body_end];
                // In the source dialect a quoted run is always a message; in the s-expr dialect it
                // is how an identifier — or a slot — is spelled, and only content that is not
                // identifier-shaped is a message after all.
                out.push(match vocab {
                    Vocab::SExpr if is_slot(body) => Tok::Slot(body),
                    Vocab::SExpr if quoted_is_identifier(body) => Tok::Name(body),
                    Vocab::Source | Vocab::SExpr => Tok::Lit(&line[start..i]),
                });
            }
            b'0'..=b'9' => {
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'.') {
                    i += 1;
                }
                out.push(Tok::Lit(&line[start..i]));
            }
            b if b.is_ascii_alphabetic() || b == b'_' => {
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let token = &line[start..i];
                // In the s-expr dialect every bare word is a node tag; in the source dialect it is a
                // name unless the language reserved it.
                out.push(match vocab {
                    Vocab::SExpr => Tok::Slot(token),
                    Vocab::Source if is_slot(token) || KEYWORDS.contains(&token) => Tok::Slot(token),
                    Vocab::Source => Tok::Name(token),
                });
            }
            _ => {
                i += line[i..].chars().next().map_or(1, char::len_utf8);
                out.push(Tok::Punct(&line[start..i]));
            }
        }
    }
    out
}

/// A line with its slot numbers renumbered from zero, in order of first appearance.
///
/// A frontend numbers a definition's locals in binding order across the whole body, so one statement
/// is `except E as _v0:` where nothing was bound before it and `_v5` five bindings later. Right for a
/// definition, wrong for an index of statements across definitions. This is only the **index** key:
/// two lines that parameterized-match always normalize alike, so it never misses a match, and
/// [`Renaming`] rejects what it over-matches.
fn slot_normalize(line: &str, vocab: Vocab) -> String {
    let mut out = String::with_capacity(line.len());
    let mut map: HashMap<&str, usize> = HashMap::default();
    for token in tokens(line, vocab) {
        match token {
            Tok::Slot(slot) if is_slot(slot) => {
                let next = map.len();
                let idx = *map.entry(slot).or_insert(next);
                out.push_str("_s");
                out.push_str(&idx.to_string());
            }
            Tok::Name(t) | Tok::Slot(t) | Tok::Lit(t) | Tok::Punct(t) => out.push_str(t),
        }
    }
    out
}

/// **Control skeleton**: the line with all vocabulary holed out, leaving the grammar.
///
/// An attribute chain collapses into one hole: `a.b.c` navigates to a single value, and the dots are
/// not a step of the procedure. Two definitions answering one question in different words agree here
/// and nowhere else, which is exactly the signal the statement index cannot carry — it keys on words.
fn skeleton(line: &str, vocab: Vocab) -> String {
    let toks = tokens(line, vocab);
    let mut out = String::with_capacity(line.len());
    let mut last_hole = false;
    let mut i = 0;
    while i < toks.len() {
        let hole = matches!(toks[i], Tok::Name(_) | Tok::Lit(_) | Tok::Slot(_));
        if matches!(toks[i], Tok::Punct(".")) && last_hole && toks.get(i + 1).is_some_and(|t| matches!(t, Tok::Name(_) | Tok::Lit(_) | Tok::Slot(_))) {
            i += 2;
            continue;
        }
        if hole {
            out.push('_');
        } else {
            let (Tok::Name(t) | Tok::Slot(t) | Tok::Lit(t) | Tok::Punct(t)) = toks[i];
            out.push_str(t);
        }
        last_hole = hole;
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Rarity
// ---------------------------------------------------------------------------

fn rarity(seen: usize, total: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let ratio = total.max(1) as f64 / seen.max(1) as f64;
    ratio.ln()
}

/// 🔴 The `k`th repetition of one shape is not a `k`th fact. Two definitions that each write twelve
/// `except _ as _:` lines agree on ONE idiom unrolled, not on twelve independent facts, and counting
/// them linearly hands a framework's error-translation boilerplate the top of the ranking by sheer
/// length. Saturating the multiplicity is the standard correction for that dependence.
fn saturate(count: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = count.max(1) as f64;
    1.0 + n.ln()
}

/// Every count the weighing needs, taken once over the corpus.
///
/// Keyed by `&str` borrowed from the views rather than by owned `String`: the counts are taken over
/// every line of every definition, and cloning each key and each shape to use it as a map key was
/// two allocations per statement in the corpus for tables that never outlive the views they count.
struct Corpus<'a> {
    names: HashMap<&'a str, usize>,
    name_total: usize,
    lines: HashMap<&'a str, usize>,
    shapes: HashMap<&'a str, usize>,
    line_total: usize,
    subjects: HashMap<u32, usize>,
    def_total: usize,
}

impl Corpus<'_> {
    fn name(&self, name: &str) -> f64 {
        rarity(self.names.get(name).copied().unwrap_or(1), self.name_total)
    }
    fn line(&self, key: &str) -> f64 {
        rarity(self.lines.get(key).copied().unwrap_or(1), self.line_total)
    }
    fn shape(&self, shape: &str) -> f64 {
        rarity(self.shapes.get(shape).copied().unwrap_or(1), self.line_total)
    }
    fn subject(&self, node: u32) -> f64 {
        rarity(self.subjects.get(&node).copied().unwrap_or(1), self.def_total)
    }
}

/// The rarest name in a set.
///
/// The **max**, not the sum: a fork is characterized by the rarest thing it turns on, not by how many
/// tokens happen to differ. Summing made a pair that parted in thirty places outrank a pair that
/// parted on one rare field — backwards, since the first two are simply different code and the second
/// two are one decision made twice.
fn peak(corpus: &Corpus, names: &[String]) -> f64 {
    names.iter().map(|n| corpus.name(n)).fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// The per-definition view this pass runs on
// ---------------------------------------------------------------------------

/// One definition, in the forms the two anchors need — all derived from its [`Facets`].
///
/// [`Facets`]: dup_defs_core::Facets
struct View {
    /// Index into the caller's `defs`.
    at: usize,
    /// Canonical statement lines, in source order.
    lines: Vec<String>,
    /// Slot-normalized lines: the statement index key, and the measure of shared *text*.
    keys: Vec<String>,
    /// Control skeleton per line: the measure of shared *shape*.
    shape: Vec<String>,
    /// Nesting, parallel to `lines`.
    depths: Vec<u16>,
    /// Interned nodes of the module tree this definition reaches.
    subjects: HashSet<u32>,
    /// Which vocabulary its canonical is written in.
    vocab: Vocab,
}

/// A view with everything but its subjects — the part that is a pure function of one definition.
struct Shaped {
    at: usize,
    lines: Vec<String>,
    keys: Vec<String>,
    shape: Vec<String>,
    depths: Vec<u16>,
    vocab: Vocab,
}

fn views(defs: &[Def]) -> (Vec<View>, Vec<String>) {
    // Split along what can be computed independently, as the seeding passes are. Normalizing and
    // skeletonizing every line is per definition and pure; interning the module tree is a shared
    // table whose ORDER is load-bearing — the id doubles as the tiebreak that decides which subject
    // a pair is reported under, so it has to stay "first appearance in `defs`".
    let shaped: Vec<Shaped> = defs
        .par_iter()
        .enumerate()
        .filter_map(|(at, def)| {
            // The definition's own header is a declaration, not a step: two definitions sharing a
            // signature shape have not agreed on *doing* anything. The contract puts it first, so it
            // is dropped here rather than by each anchor.
            let body = def.facets.statements.get(1..).unwrap_or(&[]);
            if body.len() < 2 || body.len() > MAX_STATEMENTS {
                return None;
            }
            let vocab = def.analysis.as_ref().map_or(Vocab::Source, |a| Vocab::of(a.canon_dialect));
            let lines: Vec<String> = body.iter().map(|s| s.line.clone()).collect();
            let keys: Vec<String> = lines.iter().map(|l| slot_normalize(l, vocab)).collect();
            let shape: Vec<String> = lines.iter().map(|l| skeleton(l, vocab)).collect();
            let depths: Vec<u16> = body.iter().map(|s| s.depth).collect();
            Some(Shaped { at, lines, keys, shape, depths, vocab })
        })
        .collect();

    let mut node_ids: HashMap<String, u32> = HashMap::default();
    let mut node_names: Vec<String> = Vec::new();
    let out: Vec<View> = shaped
        .into_iter()
        .map(|s| {
            let mut subjects: HashSet<u32> = HashSet::default();
            for path in &defs[s.at].facets.reaches {
                for node in prefixes(path) {
                    let next = u32::try_from(node_names.len()).unwrap_or(u32::MAX);
                    let id = *node_ids.entry(node.to_owned()).or_insert_with(|| {
                        node_names.push(node.to_owned());
                        next
                    });
                    subjects.insert(id);
                }
            }
            View {
                at: s.at,
                lines: s.lines,
                keys: s.keys,
                shape: s.shape,
                depths: s.depths,
                subjects,
                vocab: s.vocab,
            }
        })
        .collect();
    (out, node_names)
}

/// One view's lines, tokenized once into a flat arena.
///
/// 🔴 The block walk asks whether two lines parameterized-match, and it asks it millions of times:
/// once per neighbour per candidate pair, and the candidate pairs are quadratic in a shared line's
/// occurrences. Tokenizing inside that question made `tokens` 54% of the pass's CPU and its
/// `Vec<Tok>` growth another 11% — the same handful of lines re-lexed for every pair that touches
/// them. The lines do not change, so this is computed once and indexed.
///
/// Flat rather than a `Vec` per line: two allocations per definition instead of one per statement,
/// and the walk reads neighbouring lines in order, which is what makes the token compares cheap.
struct Lexed<'a> {
    toks: Vec<Tok<'a>>,
    /// `starts[i]..starts[i + 1]` bounds line `i`; length is `lines.len() + 1`.
    starts: Vec<u32>,
    /// Per line, the signature [`can_match`] rules pairs out by.
    sigs: Vec<u64>,
}

impl<'a> Lexed<'a> {
    fn of(view: &'a View) -> Self {
        let mut toks = Vec::new();
        let mut starts = Vec::with_capacity(view.lines.len() + 1);
        let mut sigs = Vec::with_capacity(view.lines.len());
        for line in &view.lines {
            starts.push(u32::try_from(toks.len()).unwrap_or(u32::MAX));
            let from = toks.len();
            toks.extend(tokens(line, view.vocab));
            sigs.push(match_sig(&toks[from..]));
        }
        starts.push(u32::try_from(toks.len()).unwrap_or(u32::MAX));
        Self { toks, starts, sigs }
    }

    /// Line `i`'s tokens, or empty when `i` is past the end — the walk probes off the end of a
    /// definition routinely, and an empty slice is what a nonexistent line matches nothing as.
    fn line(&self, i: usize) -> &[Tok<'a>] {
        let (Some(&from), Some(&to)) = (self.starts.get(i), self.starts.get(i + 1)) else {
            return &[];
        };
        &self.toks[from as usize..to as usize]
    }
}

/// Hash of a line under everything [`Renaming::accepts`] requires *pointwise*: the token count,
/// every non-binding token verbatim, and every binding slot blanked to one marker.
///
/// Two lines that parameterized-match are necessarily equal under this — `accepts` demands equal
/// tokens everywhere except where both sides bind a slot — so an unequal signature settles the
/// pair without walking it.
///
/// 🔴 Blanked, NOT renumbered: the statement key renumbers, and it is *not* a valid filter here.
/// `_v0 _v1 _v0` and `_v5 _v6 _v7` match — the first line of a run commits no mapping, so nothing
/// conflicts — yet they normalize to `_s0 _s1 _s0` and `_s0 _s1 _s2`. Filtering on the key cut
/// those runs short and moved the report.
fn match_sig(toks: &[Tok<'_>]) -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = rustc_hash::FxBuildHasher.build_hasher();
    for tok in toks {
        match *tok {
            Tok::Slot(s) if is_slot(s) => h.write_u8(0),
            Tok::Name(s) => {
                h.write_u8(1);
                h.write(s.as_bytes());
            }
            Tok::Slot(s) => {
                h.write_u8(2);
                h.write(s.as_bytes());
            }
            Tok::Lit(s) => {
                h.write_u8(3);
                h.write(s.as_bytes());
            }
            Tok::Punct(s) => {
                h.write_u8(4);
                h.write(s.as_bytes());
            }
        }
    }
    h.finish()
}

/// Can these two lines possibly parameterized-match? A hash compare, and only ever a filter: a
/// collision falls through to [`Renaming::accepts`], which is the real answer.
fn can_match(left: &Lexed<'_>, right: &Lexed<'_>, i: usize, j: usize) -> bool {
    match (left.sigs.get(i), right.sigs.get(j)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// A view together with its tokenized lines. The block walk needs both at every step, and passing
/// them separately is how they drift apart.
#[derive(Clone, Copy)]
struct Side<'a> {
    v: &'a View,
    lex: &'a Lexed<'a>,
}

/// The four count tables, as one value so they can be folded across threads.
#[derive(Default)]
struct Counts<'a> {
    names: HashMap<&'a str, usize>,
    lines: HashMap<&'a str, usize>,
    shapes: HashMap<&'a str, usize>,
    subjects: HashMap<u32, usize>,
    name_total: usize,
    line_total: usize,
}

impl Counts<'_> {
    fn absorb(&mut self, other: Self) {
        for (k, v) in other.names {
            *self.names.entry(k).or_default() += v;
        }
        for (k, v) in other.lines {
            *self.lines.entry(k).or_default() += v;
        }
        for (k, v) in other.shapes {
            *self.shapes.entry(k).or_default() += v;
        }
        for (k, v) in other.subjects {
            *self.subjects.entry(k).or_default() += v;
        }
        self.name_total += other.name_total;
        self.line_total += other.line_total;
    }
}

fn corpus_of(views: &[View]) -> Corpus<'_> {
    // Counting is a sum, and a sum does not care who added what in which order — every value here
    // is a `usize`, so unlike the weighing's floats this folds across threads with nothing to
    // preserve. The tables are per-thread and merged at the end.
    let Counts { names, lines, shapes, subjects, name_total, line_total } = views
        .par_iter()
        .fold(Counts::default, |mut acc, view| {
            for (key, shape) in view.keys.iter().zip(&view.shape) {
                *acc.lines.entry(key.as_str()).or_default() += 1;
                *acc.shapes.entry(shape.as_str()).or_default() += 1;
                acc.line_total += 1;
                for token in tokens(key, view.vocab) {
                    if let Tok::Name(name) = token {
                        *acc.names.entry(name).or_default() += 1;
                        acc.name_total += 1;
                    }
                }
            }
            for node in &view.subjects {
                *acc.subjects.entry(*node).or_default() += 1;
            }
            acc
        })
        .reduce(Counts::default, |mut a, b| {
            a.absorb(b);
            a
        });

    Corpus {
        names,
        name_total: name_total.max(1),
        lines,
        shapes,
        line_total: line_total.max(1),
        subjects,
        def_total: views.len().max(1),
    }
}

// ---------------------------------------------------------------------------
// Anti-unification of a fork
// ---------------------------------------------------------------------------

/// Names that fell into the holes of the least general generalization of two lines, per side.
///
/// Which side a name came from is what tells a role filled twice apart from a name and its own
/// derivative sitting in one expression.
struct Fork {
    holes_a: Vec<String>,
    holes_b: Vec<String>,
    aligned: usize,
}

/// Plotkin's least general generalization of two flat token sequences: aligned tokens form the
/// skeleton, everything else falls into holes. Flat rather than tree-shaped, because at this point
/// the two sides are canonical statements, not nodes.
fn anti_unify(a: &str, b: &str, vocab: Vocab) -> Fork {
    // Beyond this the quadratic table is not worth it; the multiset split is a sound stand-in — it
    // can only understate the alignment, never invent one.
    const LCS_CAP: usize = 160;
    let (left, right) = (tokens(a, vocab), tokens(b, vocab));
    if left.len() > LCS_CAP || right.len() > LCS_CAP {
        let names = |t: &[Tok<'_>]| -> HashSet<String> {
            t.iter().filter_map(|tok| if let Tok::Name(n) = tok { Some((*n).to_owned()) } else { None }).collect()
        };
        let (l, r) = (names(&left), names(&right));
        let aligned = l.intersection(&r).count() * 2;
        let mut holes_a: Vec<String> = l.difference(&r).cloned().collect();
        let mut holes_b: Vec<String> = r.difference(&l).cloned().collect();
        holes_a.sort();
        holes_b.sort();
        return Fork { holes_a, holes_b, aligned };
    }
    let (rows, cols) = (left.len(), right.len());
    let stride = cols + 1;
    // The LCS table is up to `LCS_CAP²` cells and this runs once per seed, so it comes from a
    // per-thread scratch buffer rather than a fresh zeroed allocation each time.
    LCS_SCRATCH.with_borrow_mut(|table| {
    table.clear();
    table.resize((rows + 1) * stride, 0u16);
    for row in (0..rows).rev() {
        for col in (0..cols).rev() {
            table[row * stride + col] = if left[row] == right[col] {
                table[(row + 1) * stride + col + 1] + 1
            } else {
                table[(row + 1) * stride + col].max(table[row * stride + col + 1])
            };
        }
    }
    let (mut holes_a, mut holes_b) = (Vec::new(), Vec::new());
    let mut aligned = 0usize;
    let (mut row, mut col) = (0usize, 0usize);
    let push = |out: &mut Vec<String>, tok: Tok<'_>| {
        if let Tok::Name(name) = tok {
            out.push(name.to_owned());
        }
    };
    while row < rows && col < cols {
        if left[row] == right[col] {
            aligned += 2;
            row += 1;
            col += 1;
        } else if table[(row + 1) * stride + col] >= table[row * stride + col + 1] {
            push(&mut holes_a, left[row]);
            row += 1;
        } else {
            push(&mut holes_b, right[col]);
            col += 1;
        }
    }
    for token in &left[row..] {
        push(&mut holes_a, *token);
    }
    for token in &right[col..] {
        push(&mut holes_b, *token);
    }
    holes_a.sort();
    holes_a.dedup();
    holes_b.sort();
    holes_b.dedup();
    Fork { holes_a, holes_b, aligned }
    })
}

thread_local! {
    /// Reused LCS table for [`anti_unify`]; see the comment at its allocation.
    static LCS_SCRATCH: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------------------
// Seed one: a shared statement
// ---------------------------------------------------------------------------

/// A bijection between two sides' slots, built as the run grows.
///
/// Two little association lists, not two hash maps. A run binds a handful of slots — the vectors
/// are a dozen entries at their worst — and the gap probe below copies the whole renaming for every
/// gap it tries. Copying a short vector is a memcpy; copying two `HashMap`s was two allocations and
/// a rehash, and every binding was a hash of a `_v12`-sized string to find a bucket that a linear
/// scan reaches sooner.
#[derive(Default, Clone)]
struct Renaming<'a> {
    ab: Vec<(&'a str, &'a str)>,
    ba: Vec<(&'a str, &'a str)>,
}

/// Last write wins, as the map it replaces did.
fn bind<'a>(map: &mut Vec<(&'a str, &'a str)>, key: &'a str, val: &'a str) {
    if let Some(slot) = map.iter_mut().find(|(k, _)| *k == key) {
        slot.1 = val;
    } else {
        map.push((key, val));
    }
}

fn bound<'a>(map: &[(&'a str, &'a str)], key: &str) -> Option<&'a str> {
    map.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

impl<'a> Renaming<'a> {
    /// Baker's parameterized match, checked incrementally instead of through a p-suffix tree:
    /// constants must be equal and parameters must correspond one-to-one **across the whole run**. A
    /// per-line renaming — which is all the index key does — matches `_s0 = f(_s1)` against itself
    /// even when one line means `x = f(y)` and the other `y = f(x)`.
    ///
    /// Takes the two lines already tokenized — see [`Lexed`].
    ///
    /// Validated first, committed second, over the same token walk twice. The mappings a line
    /// proposes must not be visible to the rest of that same line — a line that maps one slot two
    /// ways is ACCEPTED, and its last mapping wins — which the previous version bought with a
    /// `Vec` of staged pairs allocated on every call. Two passes buy the same rule for nothing: the
    /// first pass changes no state, so it reads the committed prefix by construction.
    fn accepts(&mut self, ta: &[Tok<'a>], tb: &[Tok<'a>]) -> bool {
        if ta.len() != tb.len() {
            return false;
        }
        for (x, y) in ta.iter().zip(tb) {
            match (x, y) {
                (Tok::Slot(p), Tok::Slot(q)) if is_slot(p) && is_slot(q) => {
                    if bound(&self.ab, p).is_some_and(|m| m != *q) || bound(&self.ba, q).is_some_and(|m| m != *p) {
                        return false;
                    }
                }
                _ if x == y => {}
                _ => return false,
            }
        }
        for (x, y) in ta.iter().zip(tb) {
            if let (Tok::Slot(p), Tok::Slot(q)) = (x, y) {
                if is_slot(p) && is_slot(q) {
                    bind(&mut self.ab, p, q);
                    bind(&mut self.ba, q, p);
                }
            }
        }
        true
    }
}

/// What a seed weighs, once the fork is anti-unified: the names each side parted by, how much of
/// the block still aligns, and the rarest name the agreement rests on.
type Weighed = (Vec<String>, Vec<String>, f64, Option<String>);

/// A run, the lines the two sides took differently, and — when they found each other again — the run
/// after that.
///
/// The second run is the whole point. A gap **bounded on both sides by agreement** is the shape the
/// inconsistent-clone literature reports faults in; an open tail is only "they started alike".
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Run {
    a_at: usize,
    b_at: usize,
    len: usize,
    gap_a: usize,
    gap_b: usize,
    run2: usize,
}

fn same_shape(a: &View, b: &View, i: usize, j: usize, base: isize) -> bool {
    let (Some(&da), Some(&db)) = (a.depths.get(i), b.depths.get(j)) else { return false };
    #[allow(clippy::cast_possible_wrap)]
    let delta = da as isize - db as isize;
    delta == base
}

fn run_forward<'a>(a: Side<'a>, b: Side<'a>, a_from: usize, b_from: usize, base: isize, renaming: &mut Renaming<'a>) -> usize {
    let mut len = 0usize;
    while same_shape(a.v, b.v, a_from + len, b_from + len, base)
        && can_match(a.lex, b.lex, a_from + len, b_from + len)
        && renaming.accepts(a.lex.line(a_from + len), b.lex.line(b_from + len))
    {
        len += 1;
    }
    len
}

/// Grow the match around a seed while one renaming and one block shape cover all of it, then find the
/// smallest gap after which the two find each other again.
///
/// Probing in order of total gap size prefers the tightest explanation — one line changed over two
/// deleted and two inserted. Each probe runs on a **copy** of the renaming, so a probe that leads
/// nowhere leaves no mappings behind. Carrying one renaming across the gap is the block-level
/// property Clone Digger's third phase is about: the same variable is the same placeholder on both
/// sides of the hole, not a fresh one per statement.
fn extend_block<'a>(a: Side<'a>, b: Side<'a>, a_pos: usize, b_pos: usize) -> Option<Run> {
    let mut renaming = Renaming::default();
    if !renaming.accepts(a.lex.line(a_pos), b.lex.line(b_pos)) {
        return None; // the index key matched but the slots do not correspond
    }
    #[allow(clippy::cast_possible_wrap)]
    let base = a.v.depths[a_pos] as isize - b.v.depths[b_pos] as isize;
    let (mut a_at, mut b_at, mut len) = (a_pos, b_pos, 1usize);
    // Backwards first, so every seed inside one run yields the same start and the caller's dedup
    // collapses them.
    while a_at > 0
        && b_at > 0
        && same_shape(a.v, b.v, a_at - 1, b_at - 1, base)
        && can_match(a.lex, b.lex, a_at - 1, b_at - 1)
        && renaming.accepts(a.lex.line(a_at - 1), b.lex.line(b_at - 1))
    {
        a_at -= 1;
        b_at -= 1;
        len += 1;
    }
    while same_shape(a.v, b.v, a_at + len, b_at + len, base)
        && can_match(a.lex, b.lex, a_at + len, b_at + len)
        && renaming.accepts(a.lex.line(a_at + len), b.lex.line(b_at + len))
    {
        len += 1;
    }
    // `renaming` already IS the renaming over `[a_at, a_at + len)`: those are exactly the lines it
    // accepted, and a rejected line commits nothing. Replaying them into a second, empty `Renaming`
    // rebuilt the identical map at the cost of another `len` matches per candidate pair.
    let carried = renaming;
    let (end_a, end_b) = (a_at + len, b_at + len);
    let mut best: Option<(usize, usize, usize)> = None;
    'outer: for total in 1..=GAP_MAX * 2 {
        for gap_a in 0..=total.min(GAP_MAX) {
            let gap_b = total - gap_a;
            if gap_b > GAP_MAX || (gap_a == 0 && gap_b == 0) {
                continue;
            }
            if end_a + gap_a >= a.v.lines.len() || end_b + gap_b >= b.v.lines.len() {
                continue;
            }
            // The probe needs its own copy of the renaming, so check first what `run_forward`
            // would check first anyway: a shape mismatch makes `run2` zero, and paying for a copy
            // of both tables to discover that is the common case, not the rare one.
            if !same_shape(a.v, b.v, end_a + gap_a, end_b + gap_b, base)
                || !can_match(a.lex, b.lex, end_a + gap_a, end_b + gap_b)
            {
                continue;
            }
            let mut probe = carried.clone();
            let run2 = run_forward(a, b, end_a + gap_a, end_b + gap_b, base, &mut probe);
            if run2 > 0 {
                best = Some((gap_a, gap_b, run2));
                break 'outer;
            }
        }
    }
    let (gap_a, gap_b, run2) = best.unwrap_or((1, 1, 0));
    Some(Run { a_at, b_at, len, gap_a, gap_b, run2 })
}

/// `n` lines from `from`, clamped to what the definition actually has.
fn slice(lines: &[String], from: usize, n: usize) -> &[String] {
    if n == 0 || from >= lines.len() {
        return &[];
    }
    &lines[from..(from + n).min(lines.len())]
}

// ---------------------------------------------------------------------------
// The pair, and its weighing
// ---------------------------------------------------------------------------

struct RunFork {
    run: Run,
    holes_a: Vec<String>,
    holes_b: Vec<String>,
    sharpness: f64,
    /// The rarest name the run holds — what identifies it, and what the cluster is keyed by.
    anchor: Option<String>,
}

struct SubjectFork {
    node: u32,
    sites: usize,
    holes: Vec<String>,
}

#[derive(Default)]
struct Pair {
    run: Option<RunFork>,
    subject: Option<SubjectFork>,
}

/// How surprising it is that these two coincide, over the evidence the finding actually rests on.
///
/// 🔴 Local to the finding, not global to the pair. Summed over the two definitions whole, the
/// textual term is unbounded in their length and swallows everything else: in a top fifty its median
/// came out eighteen times the corpus median while the fork term barely moved, which is to say the
/// ranking had quietly become "which two files overlap most" — a plain clone detector, and the one
/// thing neither anchor is for.
struct Evidence {
    text: f64,
    shape: f64,
    subject: f64,
    jaccard: f64,
}

/// The per-view facts the weighing needs, taken once instead of once per pair.
///
/// 🔴 A view appears in as many pairs as it has partners, which is the whole point of the pass —
/// and `evidence` was rebuilding the multiset of its keys, the set of them, and re-tokenizing every
/// line of its run, separately for each of those pairs. None of that depends on the partner.
struct Tally<'a> {
    /// Multiset of the view's statement keys.
    count: HashMap<&'a str, usize>,
    /// Whether each key names anything at all, positional with `keys`.
    named: Vec<bool>,
    /// The view's distinct control skeletons, sorted — the weighing walks the two views' shapes in
    /// sorted order, and merging two sorted lists costs what sorting their concatenation cost per
    /// pair.
    shapes: Vec<&'a str>,
}

impl<'a> Tally<'a> {
    fn of(view: &'a View) -> Self {
        let mut count: HashMap<&str, usize> = HashMap::default();
        for key in &view.keys {
            *count.entry(key.as_str()).or_default() += 1;
        }
        // A line that names nothing weighs zero however rare its text is: identity lives in the free
        // names, and a nameless line is grammar.
        let named = view
            .keys
            .iter()
            .map(|key| tokens(key, view.vocab).iter().any(|t| matches!(t, Tok::Name(_))))
            .collect();
        let mut shapes: Vec<&str> = view.shape.iter().map(String::as_str).collect();
        shapes.sort_unstable();
        shapes.dedup();
        Self { count, named, shapes }
    }
}

/// The half of the evidence that depends on the two BODIES and nothing else — how much shape mass
/// they hold in common, and how much of their statement text coincides.
///
/// 🔴 Split out because it is the expensive half and it is not per pair. Two definitions with the
/// same body give the same answer to it whoever their partner is, and 86% of views share a body, so
/// the pairs ask this roughly eight times more often than there are answers. Computed once per
/// distinct body pair (see `score_pairs`) and looked up.
///
/// `subject` and `text` deliberately stay out: the first reads `subjects`, which is per view —
/// identical bodies can still reach different modules — and the second depends on the pair's run.
fn body_evidence<'a>(a: &'a View, b: &'a View, ta: &Tally<'a>, tb: &Tally<'a>, corpus: &Corpus) -> (f64, f64) {
    let count_a = &ta.count;
    let mut shared_text: HashMap<&str, usize> = HashMap::default();
    for key in &b.keys {
        if let Some(left) = count_a.get(key.as_str()) {
            let entry = shared_text.entry(key.as_str()).or_default();
            if *entry < *left {
                *entry += 1;
            }
        }
    }
    // 🔴 Shapes are counted only over the lines the two do NOT already write identically. A line both
    // sides spell the same way is already the statement anchor's evidence, and counting it here would
    // rank near-clones — which that anchor reports with their exact divergence point — above the only
    // pairs this anchor can reach. Subtracted, not penalized by a factor.
    let bag = |view: &'a View| -> HashMap<&'a str, usize> {
        let mut out: HashMap<&str, usize> = HashMap::default();
        for (key, shape) in view.keys.iter().zip(&view.shape) {
            if !shared_text.contains_key(key.as_str()) {
                *out.entry(shape.as_str()).or_default() += 1;
            }
        }
        out
    };
    let (left, right) = (bag(a), bag(b));
    // 🔴 Summed in a fixed order. Float addition is not associative, and these terms come out of a
    // `HashMap`: iterating it directly made the total differ in its last bits between runs, which was
    // enough to reorder equally-scoring pairs and hand the same tree a different report every time.
    // A ranking whose ties are decided by the hasher is not reproducible, and reproducibility is the
    // difference between a finding and a coincidence.
    // Same sequence the sort produced — the sorted distinct union of both views' shapes — merged
    // from the per-view sorted lists instead of re-sorted for every pair. Shapes that survive in
    // neither bag contribute nothing to either sum, so skipping them changes no total.
    let (mut ia, mut ib) = (0usize, 0usize);
    let (mut shared, mut total) = (0.0, 0.0);
    while ia < ta.shapes.len() || ib < tb.shapes.len() {
        let shape = match (ta.shapes.get(ia), tb.shapes.get(ib)) {
            (Some(x), Some(y)) => match x.cmp(y) {
                std::cmp::Ordering::Less => {
                    ia += 1;
                    *x
                }
                std::cmp::Ordering::Greater => {
                    ib += 1;
                    *y
                }
                std::cmp::Ordering::Equal => {
                    ia += 1;
                    ib += 1;
                    *x
                }
            },
            (Some(x), None) => {
                ia += 1;
                *x
            }
            (None, Some(y)) => {
                ib += 1;
                *y
            }
            (None, None) => break,
        };
        let (in_a, in_b) = (left.get(shape).copied(), right.get(shape).copied());
        if in_a.is_none() && in_b.is_none() {
            continue;
        }
        let weight = corpus.shape(shape);
        if let Some(n) = in_a {
            total += saturate(n) * weight;
        }
        if let Some(n) = in_b {
            total += saturate(n) * weight;
        }
        if let (Some(n), Some(m)) = (in_a, in_b) {
            shared += 2.0 * saturate(n.min(m)) * weight;
        }
    }
    let cover = if total > 0.0 { shared / total } else { 0.0 };

    // The two key SETS are the key tables' domains, so the sets themselves need not be built:
    // |A ∪ B| = |A| + |B| − |A ∩ B|, and the intersection is counted by probing the larger table
    // with the smaller one's keys.
    let (na, nb) = (count_a.len(), tb.count.len());
    let inter = if na <= nb {
        count_a.keys().filter(|k| tb.count.contains_key(*k)).count()
    } else {
        tb.count.keys().filter(|k| count_a.contains_key(*k)).count()
    };
    let union = na + nb - inter;
    #[allow(clippy::cast_precision_loss)]
    let jaccard = if union == 0 { 0.0 } else { inter as f64 / union as f64 };

    (shared * cover, jaccard)
}

/// The whole evidence for one pair: the shared half looked up, plus the two terms that are the
/// pair's own.
fn evidence(
    a: &View,
    b: &View,
    ta: &Tally<'_>,
    corpus: &Corpus,
    run: Option<&Run>,
    shared: (f64, f64),
) -> Evidence {
    // Index ranges rather than slices, so `named` can be read off the precomputed vector by
    // position; the clamping is what `slice` does.
    let span = |from: usize, n: usize| -> std::ops::Range<usize> {
        if n == 0 || from >= a.keys.len() {
            0..0
        } else {
            from..(from + n).min(a.keys.len())
        }
    };
    let text: f64 = run.map_or(0.0, |r| {
        span(r.a_at, r.len)
            .chain(span(r.a_at + r.len + r.gap_a, r.run2))
            .filter(|&i| ta.named[i])
            .map(|i| corpus.line(&a.keys[i]))
            .sum()
    });
    // Per view, not per body: two definitions can spell the same body and still reach different
    // modules, which is exactly what this term is about. Memoizing it on the pair of subject SETS
    // was tried and measured neutral — numbering the sets costs what the intersection costs.
    let subject = a
        .subjects
        .iter()
        .filter(|node| b.subjects.contains(*node))
        .map(|node| corpus.subject(*node))
        .fold(0.0, f64::max);
    Evidence { text, shape: shared.0, subject, jaccard: shared.1 }
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

/// Find divergences, by both anchors, on one currency. Every finding is advisory.
///
/// `top` caps each of the two kinds of finding — pairs and families — at its strongest `top`, or
/// reports all of them when zero.
///
/// 🔴 Capped by default, unlike every other pass here. The others report a **set**: a cluster either
/// is a duplicate or is not, and cutting the set off would drop findings that are just as true as
/// the ones kept. This one reports a **ranking** with no threshold anywhere in it, and a ranking's
/// tail is not a set of weaker findings — it is the part the ordering exists to push away. Left
/// uncapped it emitted forty-four thousand pairs on a mid-sized tree, which is not a report anyone
/// reads; `--converge-top 0` still prints every one for a tool that wants them.
#[must_use]
pub fn pass_converge(defs: &[Def], top: usize, cap: usize) -> Vec<Finding> {
    let limits = Limits { top, cap };
    let (views, node_names) = views(defs);
    if views.len() < 2 {
        return Vec::new();
    }
    let corpus = corpus_of(&views);
    let (of_view, representative) = body_ids(&views);
    let bodies = Bodies { of_view: &of_view, representative: &representative };
    // Tokenized once per BODY rather than inside the block walk, or once per view — see [`Lexed`]
    // and [`body_ids`]. Pure, so it parallelizes with nothing to decide; `collect` on an indexed
    // parallel iterator keeps position, which is what makes `lexed[bodies[i]]` mean view `i`.
    let lexed: Vec<Lexed<'_>> = bodies.representative.par_iter().map(|&at| Lexed::of(&views[at])).collect();
    let mut pairs: HashMap<(usize, usize), Pair> = HashMap::default();
    seed_by_statement(&views, &lexed, bodies, &corpus, cap, &mut pairs);
    seed_by_subject(&views, bodies, &corpus, cap, &mut pairs);
    weigh(defs, &views, bodies, &corpus, &node_names, pairs, limits)
}

/// Кандидат посева, у которого посчитан только БЛОК согласия — до дорогого разбора развилки.
struct Seed {
    a: usize,
    b: usize,
    run: Run,
}

/// Number each view by the body it has, so two views spelling the same thing share a number.
///
/// 🔴 86% of views share a body with another view. Everything this pass derives from a body alone —
/// its tokens, its key multiset, its shape bag, its first line per shape — was being built once per
/// VIEW, which is seven times more often than there are distinct answers. Numbering the bodies lets
/// each of those be built once and looked up.
///
/// A body is what those derivations read, and nothing more: the lines, their nesting, and the
/// vocabulary they are written in. Views agreeing on all three are interchangeable to them. It is
/// deliberately NOT the whole view — two definitions with identical bodies can still reach
/// different modules, so `subjects` stays per view and nothing keyed on a body may consult it.
///
/// The numbering itself: which body each view has, and one representative view per body.
#[derive(Clone, Copy)]
struct Bodies<'a> {
    of_view: &'a [u32],
    representative: &'a [usize],
}

impl Bodies<'_> {
    /// The index into anything built per body, for a given view.
    fn at(self, view: usize) -> usize {
        self.of_view[view] as usize
    }
}

fn body_ids(views: &[View]) -> (Vec<u32>, Vec<usize>) {
    let mut seen: HashMap<(&[String], &[u16], Vocab), u32> = HashMap::default();
    let mut ids = Vec::with_capacity(views.len());
    let mut first: Vec<usize> = Vec::new();
    for (at, view) in views.iter().enumerate() {
        let key = (view.lines.as_slice(), view.depths.as_slice(), view.vocab);
        let next = u32::try_from(seen.len()).unwrap_or(u32::MAX);
        let id = *seen.entry(key).or_insert(next);
        if id as usize == first.len() {
            first.push(at);
        }
        ids.push(id);
    }
    (ids, first)
}

/// Every run one shared statement seeds, in the order its sites pair up.
///
/// 🔴 86% of views share their body with another view — the same duplication the name-gated pass
/// sees, and for the same reason. `extend_block` reads nothing but the two bodies and the two
/// positions, so its answer is a function of `(body, position)` twice over: a key whose sites are
/// copies of a few bodies asks one question many times and gets a few answers. The answers are
/// held in a flat table indexed by the sites' distinct `(body, position)` identities — `SEED_CAP`
/// bounds how many sites a key has, so the table is small and finding an identity by scanning it
/// beats hashing one.
///
/// Walked in the original site order regardless: the seed SEQUENCE decides which run a pair keeps
/// when several are the same length.
fn seeds_for_key<'a>(
    sites: &[(usize, usize)],
    bodies: Bodies<'_>,
    side: &impl Fn(usize) -> Side<'a>,
) -> Vec<Seed> {
    let mut identity: Vec<usize> = Vec::with_capacity(sites.len());
    let mut distinct: Vec<(usize, usize)> = Vec::new();
    for &(view, pos) in sites {
        let want = (bodies.at(view), pos);
        identity.push(distinct.iter().position(|&had| had == want).unwrap_or_else(|| {
            distinct.push(want);
            distinct.len() - 1
        }));
    }
    let width = distinct.len();
    let mut memo: Vec<Option<Option<Run>>> = vec![None; width * width];

    let mut out: Vec<Seed> = Vec::new();
    for (i, &(a, a_pos)) in sites.iter().enumerate() {
        for (off, &(b, b_pos)) in sites[i + 1..].iter().enumerate() {
            // One file does not disqualify a pair — a module that gathers one concern is exactly
            // where its near-copies collect. Being the same definition does: one line repeated
            // inside one body diverges from nothing.
            if a == b {
                continue;
            }
            let cell = identity[i] * width + identity[i + 1 + off];
            let run = if let Some(hit) = memo[cell] {
                hit
            } else {
                let computed = extend_block(side(a), side(b), a_pos, b_pos);
                memo[cell] = Some(computed);
                computed
            };
            if let Some(run) = run {
                out.push(Seed { a, b, run });
            }
        }
    }
    out
}

fn seed_by_statement(
    views: &[View],
    lexed: &[Lexed<'_>],
    bodies: Bodies<'_>,
    corpus: &Corpus,
    cap: usize,
    pairs: &mut HashMap<(usize, usize), Pair>,
) {
    let side = |i: usize| Side { v: &views[i], lex: &lexed[bodies.at(i)] };
    // Sequential on purpose — a parallel `fold`/`reduce` over the views was tried and measured
    // SLOWER (seed-stmt 1754 -> 2134 ms): the key space is millions of distinct statements, so
    // merging a table per chunk rehashes every one of them at every level of the reduction, which
    // costs more than the inserts it spread out.
    let mut occurrences: HashMap<&str, Vec<(usize, usize)>> = HashMap::default();
    for (idx, view) in views.iter().enumerate() {
        for (pos, key) in view.keys.iter().enumerate() {
            occurrences.entry(key.as_str()).or_default().push((idx, pos));
        }
    }

    // 🔴 Seeds are walked in a total order, not `HashMap` order. A pair can share several runs, and
    // whichever is registered last would otherwise win — so which agreement a divergence is reported
    // against changed between runs of the same tool over the same tree.
    let mut keys: Vec<&&str> = occurrences.keys().collect();
    keys.sort_unstable();

    // 🔴 Три фазы вместо одного цикла, и делятся они ровно по тому, что можно считать независимо.
    // Растяжение блока и разбор развилки — работа на пару, общего состояния у них нет; отсев же
    // («этот прогон уже посеян другим утверждением», «у пары остаётся ДЛИННЕЙШИЙ прогон») читает и
    // пишет одну таблицу, и порядок в нём несущий. Поэтому считаем параллельно, а решаем
    // последовательно — по заранее отсортированному списку ключей, а не по обходу хеш-таблицы.
    let seeds: Vec<Seed> = keys
        .par_iter()
        .flat_map_iter(|key| {
            let sites: &[(usize, usize)] = &occurrences[**key];
            let usable = sites.len() >= 2 && sites.len() <= cap;
            let sites: &[(usize, usize)] = if usable { sites } else { &[] };
            seeds_for_key(sites, bodies, &side)
        })
        .collect();

    // Отсев прогонов, уже посеянных раньше тем же блоком: последовательный и в том же порядке, что
    // был у однопоточного цикла, — иначе «раньше» перестаёт быть определённым.
    let mut seen: HashSet<(usize, usize, usize, usize)> = HashSet::default();
    let fresh: Vec<Seed> = seeds
        .into_iter()
        .filter(|s| {
            seen.insert((s.a.min(s.b), s.a.max(s.b), s.run.a_at.min(s.run.b_at), s.run.a_at.max(s.run.b_at)))
        })
        .collect();
    // What a seed weighs depends on the two BODIES and the run, not on which definitions spell
    // them — anti-unifying the fork was the pass's last big per-pair cost, and eight pairs in nine
    // ask it a question already answered. Computed over the distinct signatures, then read.
    let mut signatures: Vec<(u32, u32, Run)> =
        fresh.iter().map(|s| (bodies.of_view[s.a], bodies.of_view[s.b], s.run)).collect();
    signatures.sort_unstable();
    signatures.dedup();
    let weighed: HashMap<(u32, u32, Run), Option<Weighed>> = signatures
        .par_iter()
        .map(|&(ba, bb, run)| {
            let (va, vb) = (&views[bodies.representative[ba as usize]], &views[bodies.representative[bb as usize]]);
            let (pa, pb) = (
                slice(&va.lines, run.a_at + run.len, run.gap_a).join("; "),
                slice(&vb.lines, run.b_at + run.len, run.gap_b).join("; "),
            );
            if pa.is_empty() && pb.is_empty() {
                return ((ba, bb, run), None);
            }
            let fork = anti_unify(&pa, &pb, va.vocab);
            let holes: Vec<String> = fork.holes_a.iter().chain(&fork.holes_b).cloned().collect();
            if peak(corpus, &holes) <= 0.0 {
                return ((ba, bb, run), None); // they part on nothing named
            }
            let counted = |lex: &Lexed<'_>, lines: &[String], from: usize, n: usize| -> usize {
                (from..(from + n).min(lines.len())).map(|i| lex.line(i).len()).sum()
            };
            let agreed: usize = counted(&lexed[ba as usize], &va.lines, run.a_at, run.len)
                + counted(&lexed[bb as usize], &vb.lines, run.b_at, run.len);
            let gap = tokens(&pa, va.vocab).len() + tokens(&pb, vb.vocab).len();
            #[allow(clippy::cast_precision_loss)]
            let sharpness =
                if agreed + gap == 0 { 0.0 } else { (agreed as f64 + fork.aligned as f64) / (agreed + gap) as f64 };
            let anchor = slice(&va.keys, run.a_at, run.len)
                .iter()
                .flat_map(|line| tokens(line, va.vocab))
                .filter_map(|t| if let Tok::Name(n) = t { Some(n.to_owned()) } else { None })
                .max_by(|x, y| corpus.name(x).partial_cmp(&corpus.name(y)).unwrap_or(std::cmp::Ordering::Equal));
            ((ba, bb, run), Some((fork.holes_a, fork.holes_b, sharpness, anchor)))
        })
        .collect();

    let scored: Vec<(usize, usize, Run, RunFork)> = fresh
        .into_par_iter()
        .filter_map(|Seed { a, b, run }| {
            let Some((holes_a, holes_b, sharpness, anchor)) =
                &weighed[&(bodies.of_view[a], bodies.of_view[b], run)]
            else {
                return None;
            };
            let flip = a > b;
            let held = RunFork {
                run: Run {
                    a_at: if flip { run.b_at } else { run.a_at },
                    b_at: if flip { run.a_at } else { run.b_at },
                    len: run.len,
                    gap_a: if flip { run.gap_b } else { run.gap_a },
                    gap_b: if flip { run.gap_a } else { run.gap_b },
                    run2: run.run2,
                },
                holes_a: if flip { holes_b.clone() } else { holes_a.clone() },
                holes_b: if flip { holes_a.clone() } else { holes_b.clone() },
                sharpness: *sharpness,
                anchor: anchor.clone(),
            };
            Some((a, b, run, held))
        })
        .collect();
    for (a, b, run, held) in scored {
        let entry = pairs.entry((a.min(b), a.max(b))).or_default();
        // The longest agreement is the strongest evidence, so a pair keeps its longest run
        // rather than its last-seen one — and "longest" is a property of the code, where
        // "last seen" was a property of the hasher.
        if entry.run.as_ref().is_some_and(|kept| kept.run.len >= run.len) {
            continue;
        }
        entry.run = Some(held);
    }
}

fn seed_by_subject(
    views: &[View],
    bodies: Bodies<'_>,
    corpus: &Corpus,
    cap: usize,
    pairs: &mut HashMap<(usize, usize), Pair>,
) {
    let mut index: HashMap<u32, Vec<usize>> = HashMap::default();
    for (idx, view) in views.iter().enumerate() {
        for node in &view.subjects {
            index.entry(*node).or_default().push(idx);
        }
    }
    // Rarest node first: a pair meeting at `a.b.c` also meets at `a.b` and at `a`, and only the most
    // specific of those says anything. Claiming pairs as they are taken gives each its sharpest
    // subject and reports it once.
    // 🔴 The tiebreak is not cosmetic. Nodes come out of a `HashMap` in whatever order it hashes
    // them, and equally-rare nodes are common (every node in a package reached by the same set of
    // definitions ties). Without a total order, which node a pair gets claimed at — and therefore
    // which subject the finding is reported under — varies run to run. Interning order is by first
    // appearance in `defs`, so the id is a stable second key.
    let mut nodes: Vec<u32> = index.keys().copied().collect();
    nodes.sort_by(|x, y| {
        corpus
            .subject(*y)
            .partial_cmp(&corpus.subject(*x))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.cmp(y))
    });
    // Захват пары за узлом — решение ПОРЯДКА («кто первый, за тем и субъект»), и потому идёт
    // последовательно; разбор развилки захваченной пары ни от кого не зависит и уходит в параллель.
    // Разделение сохраняет и то, и другое: список захватов строится в том же порядке узлов, что и
    // раньше, а считается по нему уже безразлично кем.
    let mut claimed: HashSet<(usize, usize)> = HashSet::default();
    let mut taken: Vec<(u32, usize, (usize, usize))> = Vec::new();
    for node in nodes {
        let sites = &index[&node];
        if sites.len() < 2 || sites.len() > cap {
            continue;
        }
        for (i, &a) in sites.iter().enumerate() {
            for &b in &sites[i + 1..] {
                let key = (a.min(b), a.max(b));
                if claimed.insert(key) {
                    taken.push((node, sites.len(), key));
                }
            }
        }
    }

    // The first line at each shape, per view. It depends on one view only, and building it inside
    // the pair loop rebuilt the whole table of `b` once for every partner `b` has.
    let by_shape_of: Vec<HashMap<&str, &str>> = bodies.representative
        .par_iter()
        .map(|&at| {
            let view = &views[at];
            let mut out: HashMap<&str, &str> = HashMap::default();
            for (shape, line) in view.shape.iter().zip(&view.lines) {
                out.entry(shape.as_str()).or_insert(line.as_str());
            }
            out
        })
        .collect();

    // The fork of a subject seed reads the two bodies and nothing else, so it is computed once per
    // distinct body pair and cloned out — the same eight-to-one redundancy the statement seeding and
    // the weighing both pay for.
    let mut body_pairs: Vec<(usize, usize)> =
        taken.iter().map(|&(_, _, key)| (bodies.at(key.0), bodies.at(key.1))).collect();
    body_pairs.sort_unstable();
    body_pairs.dedup();
    let holes_of: HashMap<(usize, usize), Vec<String>> = body_pairs
        .par_iter()
        .map(|&(x, y)| {
            let va = &views[bodies.representative[x]];
            let by_shape = &by_shape_of[y];
            let mut holes: Vec<String> = Vec::new();
            for (shape, line) in va.shape.iter().zip(&va.lines) {
                if let Some(other) = by_shape.get(shape.as_str()) {
                    if *other != line.as_str() {
                        let fork = anti_unify(line, other, va.vocab);
                        holes.extend(fork.holes_a);
                        holes.extend(fork.holes_b);
                    }
                }
            }
            ((x, y), holes)
        })
        .collect();

    let forks: Vec<((usize, usize), SubjectFork)> = taken
        .into_par_iter()
        .filter_map(|(node, sites, key)| {
            // The fork of this seed: steps the two take alike and word differently. Comparing the
            // bodies line for line where their shapes agree is enough to find them — the ordering
            // of the two streams is what the statement anchor is for.
            let holes = &holes_of[&(bodies.at(key.0), bodies.at(key.1))];
            if peak(corpus, holes) <= 0.0 {
                return None;
            }
            Some((key, SubjectFork { node, sites, holes: holes.clone() }))
        })
        .collect();

    for (key, fork) in forks {
        pairs.entry(key).or_default().subject = Some(fork);
    }
}

/// The cluster a finding belongs to: the run's rarest name when there is a run, the pair itself
/// otherwise.
///
/// 🔴 Clustering is for RUNS only. A shared run really is one thing with many consumers — one
/// prologue came back as six reports because its variants differ textually while naming the same two
/// functions, and keying by the rarest name merges them. A shared subject is not that: twenty-one
/// definitions reaching one module are twenty-one different procedures, and collapsing their pairs
/// printed the best and hid the rest. The fan-in correction those pairs need is already in
/// `E_subject`, which is the rarity of the node; dividing by cluster size too would charge twice.
/// 🔴 A key, not a rendering of one. A pair with no run is its own cluster, so this was formatting
/// `"pair:12:34"` — a fresh `String`, then a hash of it, for every one of hundreds of thousands of
/// such pairs, to build a group of exactly one. The variants below partition identically to the
/// strings they replace: an anchor is a `Name` token, so it always starts with a letter or an
/// underscore and can never read as the run length it would otherwise collide with.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
enum Cluster {
    /// A shared run, grouped by its rarest name.
    Anchored(String),
    /// A shared run that names nothing, grouped by its length.
    Bare(usize),
    /// No run: the pair stands alone.
    #[default]
    Alone,
}

fn cluster_key(pair: &Pair) -> Cluster {
    match &pair.run {
        Some(run) => run.anchor.clone().map_or(Cluster::Bare(run.run.len), Cluster::Anchored),
        None => Cluster::Alone,
    }
}

/// Находка вместе с тем, что понадобится, только если она переживёт потолок.
///
/// Сортировка от отложенного разбора не страдает: ни `notes`, ни `snippet` в её ключи не входят.
struct Pending {
    finding: Finding,
    render: Option<(Pair, usize, usize)>,
    snippet_def: usize,
}

/// Поджатая клика: её состав, общая масса формы и рендер этих форм для читателя.
type TightFamily = (Vec<usize>, f64, Vec<String>);

/// A set of definitions that all reach one subject and all take the same shape.
///
/// 🔴 The rubric this exists for. Twenty-two definitions reaching one module produce two hundred and
/// thirty-one pairs, and reporting them as pairs says the same thing two hundred and thirty-one
/// times while burying it: the finding is not "these two are alike", it is "**this family exists**".
/// Measured on two corpora, that is what the top of the pair ranking was actually made of.
///
/// Grouped by **greedy maximal clique**, not by connected component. A corpus saturated with one
/// shape chains under single-linkage into one blob whose ends share nothing — `A~B` and `B~C` merge
/// `A` and `C` even when `A≁C`. A clique requires *every* pair to be an edge, so a family really is
/// mutually alike. This is the same answer the patternology pass reached for the same problem, and
/// reaching a different one here would mean one of the two is wrong.
struct Family {
    node: u32,
    members: Vec<usize>,
    /// The shape bag every member holds in common, weighted — what makes them a family.
    shared: f64,
    /// One member's rendering of those shared shapes, for the reader.
    shapes: Vec<String>,
    score: f64,
}

/// Greedy maximal-clique cover of a graph given as an adjacency map.
///
/// Greedy (seed = highest-degree unclaimed vertex, grow by neighbours adjacent to every member)
/// rather than exhaustive enumeration: the groups here are bounded by [`SEED_CAP`], but the shape is
/// exactly the dense component where exhaustive enumeration blows up, and a family that is *a*
/// maximal clique rather than *the* largest one answers the question just as well.
fn clique_cover(vertices: &[usize], edges: &HashMap<usize, HashSet<usize>>) -> Vec<Vec<usize>> {
    let mut unclaimed: HashSet<usize> = vertices.iter().copied().collect();
    let mut out = Vec::new();
    while !unclaimed.is_empty() {
        let degree = |v: usize| edges.get(&v).map_or(0, |n| n.iter().filter(|u| unclaimed.contains(u)).count());
        // Ties break on the vertex itself: the seed decides the whole clique, and a seed chosen by
        // the hasher is a family that changes between runs.
        let Some(&seed) = unclaimed.iter().max_by(|x, y| degree(**x).cmp(&degree(**y)).then(y.cmp(x))) else {
            break;
        };
        let mut clique = vec![seed];
        let mut candidates: Vec<usize> =
            edges.get(&seed).map_or_else(Vec::new, |n| n.iter().copied().filter(|u| unclaimed.contains(u)).collect());
        candidates.sort_unstable();
        for candidate in candidates {
            if clique.iter().all(|m| edges.get(m).is_some_and(|n| n.contains(&candidate))) {
                clique.push(candidate);
            }
        }
        for member in &clique {
            unclaimed.remove(member);
        }
        out.push(clique);
    }
    out
}

/// The shape bag every member of a group holds in common, weighted and **covered**, plus a
/// rendering of it.
///
/// The intersection over *all* members, not a pairwise average: a family's claim is that every one
/// of them takes this shape, and a shape two of five share is not part of it.
///
/// 🔴 Normalized by what the members do NOT share, exactly as the pair term is. Raw, the intersection
/// over three arbitrary functions survives on `_()` and `_(_)` — "these three call something" — and
/// the first families this produced were made of nothing else. Cover is what says whether the shared
/// shape is most of what these definitions are, or the residue of any three procedures.
fn common_shapes(members: &[usize], views: &[View], corpus: &Corpus) -> (f64, Vec<String>) {
    let mut counts: HashMap<&str, usize> = HashMap::default();
    let Some((&first, rest)) = members.split_first() else { return (0.0, Vec::new()) };
    for shape in &views[first].shape {
        *counts.entry(shape.as_str()).or_default() += 1;
    }
    for &member in rest {
        let mut here: HashMap<&str, usize> = HashMap::default();
        for shape in &views[member].shape {
            *here.entry(shape.as_str()).or_default() += 1;
        }
        counts.retain(|shape, n| {
            let m = here.get(shape).copied().unwrap_or(0);
            *n = (*n).min(m);
            *n > 0
        });
    }
    let mut shapes: Vec<&str> = counts.keys().copied().collect();
    shapes.sort_unstable();
    let shared: f64 = shapes.iter().map(|shape| saturate(counts[shape]) * corpus.shape(shape)).sum();
    // Shown rarest first: the intersection legitimately contains `_()` — every procedure calls
    // something — and leading with it makes a family read as though that is what it is about.
    shapes.sort_by(|x, y| {
        corpus.shape(y).partial_cmp(&corpus.shape(x)).unwrap_or(std::cmp::Ordering::Equal).then(x.cmp(y))
    });
    let mut whole: f64 = 0.0;
    for &member in members {
        let mut here: HashMap<&str, usize> = HashMap::default();
        for shape in &views[member].shape {
            *here.entry(shape.as_str()).or_default() += 1;
        }
        let mut keys: Vec<&&str> = here.keys().collect();
        keys.sort_unstable();
        for shape in keys {
            whole += saturate(here[*shape]) * corpus.shape(shape);
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let cover = if whole > 0.0 { shared * members.len() as f64 / whole } else { 0.0 };
    (shared * cover, shapes.into_iter().map(str::to_owned).collect())
}

/// Drop members until no further removal makes the family stronger.
///
/// 🔴 The clique proposes, the evidence disposes. Adjacency here is weak — "this pair scored, and
/// both reach this node" — so a greedy clique happily grows past the real family, and one stray
/// member drags the intersection down to what any two procedures share. Left alone, a family of five
/// siblings came back as a family of six whose common shape was nothing.
///
/// The score is what decides, so the pruning needs no threshold and no notion of which member is the
/// "odd" one: whichever removal helps most is taken, and when none helps the family is what is left.
/// Below [`MIN_FAMILY`] there is no family — that is a pair, and the pair report already has it.
fn tighten(
    mut members: Vec<usize>,
    views: &[View],
    corpus: &Corpus,
) -> Option<TightFamily> {
    members.sort_unstable();
    let mut best = common_shapes(&members, views, corpus);
    while members.len() > MIN_FAMILY {
        let mut improved: Option<(usize, (f64, Vec<String>))> = None;
        for drop in 0..members.len() {
            let mut candidate = members.clone();
            candidate.remove(drop);
            let scored = common_shapes(&candidate, views, corpus);
            // Strictly better, and ties keep the larger family: more places is the finding.
            if scored.0 > improved.as_ref().map_or(best.0, |(_, s)| s.0) {
                improved = Some((drop, scored));
            }
        }
        let Some((drop, scored)) = improved else { break };
        members.remove(drop);
        best = scored;
    }
    (best.0 > 0.0).then_some((members, best.0, best.1))
}

/// Group subject-seeded pairs into families, and say which pairs a family absorbed.
///
/// A pair inside a family is not reported separately: it *is* the family, said once for each of its
/// members' partners. What is left over — a pair whose group never reached three — stays a pair.
fn families(
    pairs: &[(usize, usize)],
    views: &[View],
    corpus: &Corpus,
    cap: usize,
) -> (Vec<Family>, HashSet<(usize, usize)>) {
    // 🔴 An edge is registered at EVERY node its two members share, not at the one the pair was
    // claimed on. Claiming a pair at its rarest common node is right for reporting *that pair* — it
    // is the most specific thing the two are both about — and wrong for finding a family, which needs
    // the node the whole group meets on. Built on claims, a group of six sibling functions fragmented
    // into pairs scattered over six different nodes and no family formed at all.
    let mut by_node: HashMap<u32, Vec<(usize, usize)>> = HashMap::default();
    for &(a, b) in pairs {
        let (small, large) = if views[a].subjects.len() <= views[b].subjects.len() {
            (&views[a].subjects, &views[b].subjects)
        } else {
            (&views[b].subjects, &views[a].subjects)
        };
        for node in small.iter().filter(|node| large.contains(*node)) {
            by_node.entry(*node).or_default().push((a, b));
        }
    }
    // Rarest node first: the same clique forms at every ancestor of the node it really meets on, and
    // the most specific of those is the one worth reporting. Later duplicates are dropped by the
    // member set they carry.
    let mut nodes: Vec<u32> = by_node.keys().copied().collect();
    nodes.sort_by(|x, y| {
        corpus.subject(*y).partial_cmp(&corpus.subject(*x)).unwrap_or(std::cmp::Ordering::Equal).then(x.cmp(y))
    });
    // 🔴 Клики считаются параллельно, а отбираются последовательно. Поиск клик и их поджатие —
    // работа внутри ОДНОГО узла, соседи ей не нужны; а вот «эту же семью уже напечатали на более
    // точном субъекте» — решение о порядке, и оно обязано приниматься по списку узлов, отсортированному
    // от редкого к частому. Смешай их — и семья выходила бы то под одним субъектом, то под другим.
    let per_node: Vec<Vec<TightFamily>> = nodes
        .par_iter()
        .map(|node| {
            // 🔴 The same cap the pair seeding uses, applied here too. A node reached by more
            // definitions than this is infrastructure — the framework, the store, the directory every
            // file lives in — not a thing a handful of definitions are *about*. Without it the family
            // index re-admitted exactly what the pair index excludes, and families formed around
            // "these files are in the same tree".
            if corpus.subjects.get(node).copied().unwrap_or(0) > cap {
                return Vec::new();
            }
            let edges_list = &by_node[node];
            let mut edges: HashMap<usize, HashSet<usize>> = HashMap::default();
            let mut vertices: Vec<usize> = Vec::new();
            for &(a, b) in edges_list {
                edges.entry(a).or_default().insert(b);
                edges.entry(b).or_default().insert(a);
                vertices.push(a);
                vertices.push(b);
            }
            vertices.sort_unstable();
            vertices.dedup();
            clique_cover(&vertices, &edges)
                .into_iter()
                .filter(|clique| clique.len() >= MIN_FAMILY)
                .filter_map(|clique| tighten(clique, views, corpus))
                .collect()
        })
        .collect();

    let mut out = Vec::new();
    let mut absorbed: HashSet<(usize, usize)> = HashSet::default();
    let mut seen_members: HashSet<Vec<usize>> = HashSet::default();
    for (node, cliques) in nodes.iter().zip(per_node) {
        for (clique, shared, shapes) in cliques {
            if !seen_members.insert(clique.clone()) {
                continue; // the same family, already reported at a more specific subject
            }
            for i in 0..clique.len() {
                for j in i + 1..clique.len() {
                    absorbed.insert((clique[i], clique[j]));
                }
            }
            // A family's value grows with how many places it holds and how much shape they all
            // share — no hub penalty, because being many places is the finding, not a discount on
            // it. That penalty belongs to a pair, where a third carrier means the primitive exists.
            #[allow(clippy::cast_precision_loss)]
            let score = corpus.subject(*node) * shared * (clique.len() as f64).ln();
            out.push(Family { node: *node, members: clique, shared, shapes, score });
        }
    }
    out.sort_by(|x, y| {
        y.score.partial_cmp(&x.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| x.members.cmp(&y.members))
    });
    (out, absorbed)
}

/// Взвешенная пара — до того, как из неё сделали находку.
struct Scored {
    key: (usize, usize),
    pair: Pair,
    ev: Evidence,
    divergence: f64,
    score: f64,
    /// Ключ кластера считается в ПАРАЛЛЕЛЬНОЙ фазе, а не в последовательной группировке.
    cluster: Cluster,
}

fn score_pairs(
    views: &[View],
    bodies: Bodies<'_>,
    corpus: &Corpus,
    pairs: HashMap<(usize, usize), Pair>,
) -> Vec<Scored> {
    // 🔴 Взвешивание идёт ПАРАЛЛЕЛЬНО, но по заранее упорядоченному списку, а не по обходу
    // `HashMap`. Порядок тут не косметика: `collect` индексированного параллельного итератора
    // сохраняет позиции, и потому результат не зависит ни от числа ядер, ни от того, какой воркер
    // успел первым. Обход хеш-таблицы дал бы каждому прогону свой порядок групп — а группы решают,
    // какая пара станет представителем кластера, то есть какое расхождение будет напечатано.
    // Per body, not per view: a tally reads only what a body spells.
    let tallies: Vec<Tally<'_>> = bodies.representative.par_iter().map(|&at| Tally::of(&views[at])).collect();
    let mut entries: Vec<((usize, usize), Pair)> = pairs.into_iter().collect();
    entries.sort_unstable_by_key(|(key, _)| *key);

    // The body half of the evidence, once per distinct body pair instead of once per pair — see
    // [`body_evidence`]. Two parallel passes rather than one with a shared table: the answers are
    // computed over a deduplicated list, then read.
    let mut body_pairs: Vec<(usize, usize)> =
        entries.iter().map(|(key, _)| (bodies.at(key.0), bodies.at(key.1))).collect();
    body_pairs.sort_unstable();
    body_pairs.dedup();
    let shared_of: HashMap<(usize, usize), (f64, f64)> = body_pairs
        .par_iter()
        .map(|&(x, y)| {
            let (va, vb) = (&views[bodies.representative[x]], &views[bodies.representative[y]]);
            ((x, y), body_evidence(va, vb, &tallies[x], &tallies[y], corpus))
        })
        .collect();
    let out: Vec<Scored> = entries
        .into_par_iter()
        .filter_map(|(key, pair)| {
        let (a, b) = (&views[key.0], &views[key.1]);
        let shared = shared_of[&(bodies.at(key.0), bodies.at(key.1))];
        let ev = evidence(a, b, &tallies[bodies.at(key.0)], corpus, pair.run.as_ref().map(|r| &r.run), shared);
        let total = ev.text + ev.shape + ev.subject;
        if total <= 0.0 {
            return None;
        }
        // The rarest name across all three hole lists, without concatenating them into a fourth.
        // `peak` is a maximum and every term is a rarity, which is non-negative, so the maximum over
        // the parts IS the maximum over the whole — and cloning hundreds of thousands of little
        // string vectors to take one number was the pass's largest remaining memmove.
        let divergence = pair
            .run
            .as_ref()
            .map_or(0.0, |r| peak(corpus, &r.holes_a).max(peak(corpus, &r.holes_b)))
            .max(pair.subject.as_ref().map_or(0.0, |s| peak(corpus, &s.holes)));
        if divergence <= 0.0 {
            return None;
        }
        // Sharpness applies only where it was measured — inside a run, where "how much of the block
        // still aligns" is defined. A subject-seeded pair has no block, and its counterpart is the
        // share of shape it holds in common, which is already in the evidence.
        let sharpness = pair.run.as_ref().map_or(1.0, |r| r.sharpness);
        // Rejoined means the two found each other again: a drifted copy, where alikeness is the
        // premise. Parted for good means a similarity pass already has the pair and only the fork is
        // news.
        let rejoined = pair.run.as_ref().is_some_and(|r| r.run.run2 > 0);
        let novelty = if rejoined { 1.0 } else { 1.0 - ev.jaccard };
        let score = total * divergence * sharpness * novelty;
        let cluster = cluster_key(&pair);
        Some(Scored { key, pair, ev, divergence, score, cluster })
        })
        .collect();
    out
}

fn weigh(
    defs: &[Def],
    views: &[View],
    bodies: Bodies<'_>,
    corpus: &Corpus,
    node_names: &[String],
    pairs: HashMap<(usize, usize), Pair>,
    limits: Limits,
) -> Vec<Finding> {
    let Limits { top, cap } = limits;
    let scored = score_pairs(views, bodies, corpus, pairs);

    // 🔴 Families are built from **scored** pairs, not from the raw claims. An edge has to mean "these
    // two are genuinely alike" — the seed only established that they meet on a subject and part on
    // something named. Built on raw claims, a clique guaranteed pairwise adjacency that guaranteed
    // nothing, and the families that came out shared `_()` and `_(_)`.
    //
    // A group of N definitions around one subject produces N(N−1)/2 pairs, and reporting those says
    // one fact once per pair while burying it. What is left over — a pair whose group never reached
    // three — is still a pair.
    let subject_pairs: Vec<(usize, usize)> = scored
        .iter()
        .filter(|s| s.ev.shape > 0.0 && s.pair.subject.is_some())
        .map(|s| s.key)
        .collect();
    let (families, absorbed) = families(&subject_pairs, views, corpus, cap);
    let mut out: Vec<Pending> = families
        .iter()
        .map(|family| Pending {
            snippet_def: views[family.members[0]].at,
            render: None,
            finding: family_finding(defs, views, node_names, family),
        })
        .collect();
    let mut by_cluster: HashMap<(Cluster, usize, usize), Vec<Scored>> = HashMap::default();
    for mut entry in scored {
        // A pair a family already accounts for is that family, said once per partner. A pair that
        // also grew a run stays: the run is a fact about those two that the family does not carry.
        if absorbed.contains(&entry.key) && entry.pair.run.is_none() {
            continue;
        }
        // A standalone pair is grouped by ITSELF, which the string form spelled into its key; the
        // pair's indices ride alongside so `Alone` does not collapse every such pair into one group.
        let alone = matches!(entry.cluster, Cluster::Alone);
        let (l, r) = if alone { entry.key } else { (0, 0) };
        by_cluster.entry((std::mem::take(&mut entry.cluster), l, r)).or_default().push(entry);
    }

    for (_, mut group) in by_cluster {
        // The representative is the strongest pair; ties break on the pair itself, because the group
        // was collected from a `HashMap` and an unstable order would pick a different exemplar — and
        // therefore print a different divergence — between runs.
        group.sort_by(|x, y| {
            y.score.partial_cmp(&x.score).unwrap_or(std::cmp::Ordering::Equal).then(x.key.cmp(&y.key))
        });
        let members: HashSet<usize> = group.iter().flat_map(|s| [s.key.0, s.key.1]).collect();
        #[allow(clippy::cast_precision_loss)]
        let hub = members.len().max(2) as f64;
        let Some(best) = group.first() else { continue };
        let (a, b) = (&views[best.key.0], &views[best.key.1]);
        let (da, db) = (&defs[a.at], &defs[b.at]);
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let facets = vec![
            ("text".to_owned(), best.ev.text.max(0.0) as usize),
            ("shape".to_owned(), best.ev.shape.max(0.0) as usize),
            ("subject".to_owned(), best.ev.subject.max(0.0) as usize),
            ("fork".to_owned(), best.divergence.max(0.0) as usize),
        ];
        let score = best.score / hub;
        let (key_a, key_b) = (best.key.0, best.key.1);
        let pair_for_render = group.into_iter().next().map(|s| s.pair);
        out.push(Pending {
            snippet_def: a.at,
            render: pair_for_render.map(|pair| (pair, key_a, key_b)),
            finding: Finding {
            pass: "converge",
            kind: kind_of(da, db),
            name: format!("{} / {}", da.name, db.name),
            // Advisory, always: the output is a ranked list with no threshold, and a gate that fires
            // on the tail of a ranking teaches people to ignore it.
            severity: Severity::Info,
            min_sim: None,
            loc: da.loc.max(db.loc),
            args: da.args.max(db.args),
            // Squashed to [0, 1] so it sorts alongside every other pass's thickness without pretending
            // to be the same quantity; the raw nats ride in `facets`.
            thickness: 1.0 - (-score / 400.0).exp(),
            snippet: String::new(),
            notes: Vec::new(),
            members: vec![member(defs, a.at), member(defs, b.at)],
            facets,
            pattern: None,
            },
        });
    }
    // Same reasoning as the node order: clusters come out of a `HashMap`, and equal scores tie often
    // enough that an unstable order would shuffle the report between runs.
    // 🔴 Ordered down to the members, not just the name and the first of them. The report's own sort
    // is stable and breaks ties on `(name, members[0])`, so two findings that share both — the same
    // definition diverging from two different partners — would keep whatever order this handed over,
    // which came out of a `HashMap`. The same tool over the same tree printed a different report each
    // run, in a section whose whole value is its order.
    out.sort_by(|x, y| {
        y.finding
            .thickness
            .partial_cmp(&x.finding.thickness)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.finding.name.cmp(&y.finding.name))
            .then_with(|| x.finding.members.cmp(&y.finding.members))
    });
    cap_and_explain(out, top, defs, views, node_names)
}

/// Обрезать отчёт по потолку и разобрать ТОЛЬКО то, что его пережило.
///
/// 🔴 Порядок обязателен именно такой. Разбор находки и её сниппет — самая дорогая часть прохода, и
/// до потолка они не нужны ни одной находке: на среднем дереве пар набирается сто с лишним тысяч, а
/// печатается несколько сотен. Считая всё сразу, проход рендерил и копировал исходники ста тысяч
/// определений, чтобы тут же их выбросить.
fn cap_and_explain(
    mut out: Vec<Pending>,
    top: usize,
    defs: &[Def],
    views: &[View],
    node_names: &[String],
) -> Vec<Finding> {
    if top > 0 {
        // Каждый вид держит свою голову: семейство не соревнуется с парой за один слот, и общий
        // потолок дал бы тому, кто на этом дереве набрал больше, заглушить второго целиком.
        let mut kept = 0;
        let mut kept_family = 0;
        out.retain(|p| {
            let seen = if p.finding.pass == "converge-family" { &mut kept_family } else { &mut kept };
            *seen += 1;
            *seen <= top
        });
    }
    out.into_iter()
        .map(|mut p| {
            if let Some((pair, a, b)) = &p.render {
                p.finding.notes = render(pair, &views[*a], &views[*b], node_names);
            }
            p.finding.snippet.clone_from(&defs[p.snippet_def].text_orig);
            p.finding
        })
        .collect()
}

/// What the reader sees: the run the two agreed on, the step where they parted and the names they
/// parted by, and the subject they meet on.
///
/// Separate from the weighing because they answer different questions — one decides whether this is
/// worth printing at all, the other decides what printing it should say — and mixing them is how a
/// scoring function grows a rendering it cannot be read without.
fn render(pair: &Pair, a: &View, b: &View, node_names: &[String]) -> Vec<String> {
    let mut notes = Vec::new();
    if let Some(run) = &pair.run {
        notes.push(format!("agreed on {} statement(s):", run.run.len));
        notes.extend(slice(&a.lines, run.run.a_at, run.run.len).iter().map(|l| format!("  = {l}")));
        notes.push("parted:".to_owned());
        for (view, at, gap, holes) in [
            (a, run.run.a_at, run.run.gap_a, &run.holes_a),
            (b, run.run.b_at, run.run.gap_b, &run.holes_b),
        ] {
            for line in slice(&view.lines, at + run.run.len, gap) {
                notes.push(format!("  {line}"));
            }
            if !holes.is_empty() {
                notes.push(format!("  by: {}", holes.join(", ")));
            }
        }
        if run.run.run2 > 0 {
            notes.push("and agreed again".to_owned());
        }
    }
    if let Some(subject) = &pair.subject {
        notes.push(format!(
            "subject: {} (reached by {} definitions)",
            node_names.get(subject.node as usize).map_or("?", String::as_str),
            subject.sites
        ));
    }
    notes
}

/// One family, as the report sees it.
fn family_finding(defs: &[Def], views: &[View], node_names: &[String], family: &Family) -> Finding {
    let members: Vec<usize> = family.members.iter().map(|&v| views[v].at).collect();
    let first = &defs[members[0]];
    let names: Vec<&str> = members.iter().map(|&i| defs[i].name.as_str()).collect();
    let subject = node_names.get(family.node as usize).map_or("?", String::as_str);
    let mut notes = vec![
        format!("{} definitions around {subject}, all taking one shape:", family.members.len()),
    ];
    // The shapes every member holds, not one member's body: what the family *is* rather than what one
    // of them happens to look like.
    notes.extend(family.shapes.iter().map(|shape| format!("  ~ {shape}")));
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let facets = vec![
        ("shape".to_owned(), family.shared.max(0.0) as usize),
        ("members".to_owned(), family.members.len()),
    ];
    Finding {
        pass: "converge-family",
        kind: first.kind,
        name: names.join(" / "),
        severity: Severity::Info,
        min_sim: None,
        loc: members.iter().map(|&i| defs[i].loc).max().unwrap_or(0),
        args: members.iter().map(|&i| defs[i].args).max().unwrap_or(0),
        thickness: 1.0 - (-family.score / 400.0).exp(),
        snippet: first.text_orig.clone(),
        notes,
        members: members.iter().map(|&i| member(defs, i)).collect(),
        facets,
        pattern: None,
    }
}

/// The kind a pair is reported under.
///
/// A mixed pair — a free function diverging from a method — is real and worth reporting; it is filed
/// under the first member's kind, and the members name both. Picking by a rule (say, always the
/// method) would only move the arbitrariness somewhere less visible.
fn kind_of(a: &Def, _b: &Def) -> &'static KindSpec {
    a.kind
}

#[cfg(test)]
mod tests {
    use super::{clique_cover, pass_converge, tokens, Tok, Vocab};
    use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
    use dup_defs_core::{Analysis, CanonDialect, Def, Facets, KindSpec, Statement};
    use std::sync::Arc;

    static FUNCTIONS: KindSpec = KindSpec {
        id: "functions",
        label: "FUNCTION",
        noun_plural: "functions",
        section: 1,
        body: true,
        fn_like: true,
    };

    fn def(name: &str, file: &str, lines: &[(&str, u16)], reaches: &[&str]) -> Def {
        let statements: Vec<Statement> =
            lines.iter().map(|(l, d)| Statement { line: (*l).to_owned(), depth: *d }).collect();
        Def {
            lang: "test",
            kind: &FUNCTIONS,
            name: name.to_owned(),
            file: Arc::from(file),
            line: 1,
            col: 0,
            loc: statements.len(),
            args: 1,
            text_orig: String::new(),
            cluster_canonical: None,
            analysis: Some(Analysis {
                xname_canonical: String::new(),
                type3_lines: Vec::new(),
                size: statements.len(),
                // The fixture lines below are source-like, so they must be read with the source
                // vocabulary — `Other` would say the bare words are AST node tags.
                canon_dialect: CanonDialect::CPythonAst,
            }),
            thickness: None,
            facets: Facets {
                statements,
                reaches: reaches.iter().map(|r| Arc::from(*r)).collect(),
            },
        }
    }

    /// A body whose first entry is the definition's header, as every frontend reports it.
    fn body<'a>(lines: &[(&'a str, u16)]) -> Vec<(&'a str, u16)> {
        let mut out = vec![("def _fn(_v0):", 0u16)];
        out.extend_from_slice(lines);
        out
    }

    #[test]
    fn the_cap_keeps_the_head_and_each_kind_keeps_its_own() {
        // Two independent groups, each around its own subject, so more than one family exists to
        // choose between — with only one, a cap of one is indistinguishable from no cap.
        let mut defs = Vec::new();
        for (group, subject, call) in
            [(0, "billing.plans.policy_get", "policy_get"), (1, "billing.quota.limit_get", "limit_get")]
        {
            for i in 0..4 {
                defs.push(def(
                    &format!("f{group}_{i}"),
                    &format!("f{group}_{i}.py"),
                    &body(&[
                        (Box::leak(format!("_v1 = {call}(_v0)").into_boxed_str()), 1),
                        ("if _v1 is None:", 1),
                        (if i % 2 == 0 { "return True" } else { "return ALLOWED" }, 2),
                    ]),
                    &[subject],
                ));
            }
        }
        for i in 0..8 {
            defs.push(def(&format!("x{i}"), "x.py", &body(&[("_v9 = other()", 1), ("return _v9", 1)]), &["elsewhere"]));
        }
        let all = pass_converge(&defs, 0, super::SEED_CAP);
        let families = all.iter().filter(|f| f.pass == "converge-family").count();
        assert!(families > 1, "the fixture needs more than one family to cap between: {families}");

        let capped = pass_converge(&defs, 1, super::SEED_CAP);
        assert_eq!(capped.iter().filter(|f| f.pass == "converge-family").count(), 1);
        // The head, not the tail.
        let strongest =
            all.iter().filter(|f| f.pass == "converge-family").map(|f| f.thickness).fold(f64::MIN, f64::max);
        let kept = capped.iter().find(|f| f.pass == "converge-family").expect("kept a family");
        assert!((kept.thickness - strongest).abs() < 1e-9, "kept {} of {strongest}", kept.thickness);
    }

    #[test]
    fn each_dialect_is_read_in_its_own_vocabulary() {
        // Source-like: bare identifiers are names, reserved words are grammar.
        let src = "if _v0 is None:";
        let names = |line: &str, vocab| {
            tokens(line, vocab)
                .into_iter()
                .filter_map(|t| if let Tok::Name(n) = t { Some(n.to_owned()) } else { None })
                .collect::<Vec<_>>()
        };
        assert!(names(src, Vocab::Source).is_empty(), "`if`/`is`/`None` are grammar: {:?}", names(src, Vocab::Source));
        assert_eq!(names("_v0 = resolve_plan(_v1)", Vocab::Source), vec!["resolve_plan"]);

        // S-expr: the bare words are node tags and the identifiers live inside the quotes.
        let sexpr = "Let(Bind('_v1'), Call(Path('open_session'), Path('_v0')))";
        assert_eq!(
            names(sexpr, Vocab::SExpr),
            vec!["open_session"],
            "node tags are grammar, the quoted identifier is the name"
        );
        // A quoted run that is not identifier-shaped is a message, whichever dialect wrote it.
        assert!(names("Str('failed to reach {}')", Vocab::SExpr).is_empty());
    }

    fn graph(edges: &[(usize, usize)]) -> HashMap<usize, HashSet<usize>> {
        let mut out: HashMap<usize, HashSet<usize>> = HashMap::default();
        for &(a, b) in edges {
            out.entry(a).or_default().insert(b);
            out.entry(b).or_default().insert(a);
        }
        out
    }

    #[test]
    fn a_clique_is_not_a_connected_component() {
        // 🔴 The whole reason this is a clique cover. Under single-linkage `1-2-3-4` is one group of
        // four whose ends share no edge; a family has to be mutually alike, or "family" means only
        // "reachable from".
        let chain = graph(&[(1, 2), (2, 3), (3, 4)]);
        let cover = clique_cover(&[1, 2, 3, 4], &chain);
        assert!(cover.iter().all(|c| c.len() <= 2), "a chain is not a family: {cover:?}");
    }

    #[test]
    fn clique_cover_keeps_a_complete_graph_whole() {
        let complete = graph(&[(1, 2), (1, 3), (2, 3)]);
        let mut cover = clique_cover(&[1, 2, 3], &complete);
        for c in &mut cover {
            c.sort_unstable();
        }
        assert_eq!(cover, vec![vec![1, 2, 3]]);
    }

    #[test]
    fn the_same_graph_always_covers_the_same_way() {
        // The seed decides the whole clique, and a seed picked out of a `HashSet` is a family that
        // changes between runs of the same tool over the same tree.
        let g = graph(&[(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (3, 5)]);
        let first = clique_cover(&[1, 2, 3, 4, 5], &g);
        for _ in 0..8 {
            assert_eq!(clique_cover(&[1, 2, 3, 4, 5], &g), first);
        }
    }

    #[test]
    fn a_shared_run_and_a_fork_is_a_finding() {
        let a = def(
            "a",
            "one.py",
            &body(&[("_v1 = resolutions_prune(_v0)", 1), ("if _v1.channel_id is None:", 1), ("return", 2)]),
            &[],
        );
        let b = def("b", "two.py", &body(&[("_v1 = resolutions_prune(_v0)", 1), ("return _v1", 1)]), &[]);
        let found = pass_converge(&[a, b], 0, super::SEED_CAP);
        assert_eq!(found.len(), 1, "{found:?}", found = found.iter().map(|f| &f.name).collect::<Vec<_>>());
        let f = &found[0];
        assert_eq!(f.pass, "converge");
        // Always advisory: the output is a ranking with no threshold, and a gate on the tail of a
        // ranking teaches people to ignore it.
        assert_eq!(f.severity, crate::Severity::Info);
        assert!(f.notes.iter().any(|n| n.contains("channel_id")), "the fork is named: {:?}", f.notes);
    }

    #[test]
    fn two_definitions_sharing_nothing_are_not_a_finding() {
        let a = def("a", "one.py", &body(&[("_v1 = alpha(_v0)", 1), ("return _v1", 1)]), &[]);
        let b = def("b", "two.py", &body(&[("_v1 = beta(_v0)", 1), ("raise Boom", 1)]), &[]);
        assert!(pass_converge(&[a, b], 0, super::SEED_CAP).is_empty());
    }

    #[test]
    fn a_shared_subject_reaches_a_pair_that_shares_no_word() {
        // The blind spot the second anchor exists for: same subject, same shape, not one line alike.
        let a = def(
            "fits",
            "one.py",
            &body(&[("_v1 = person_limit_get(_v0)", 1), ("if _v1 is None:", 1), ("return True", 2)]),
            &["billing.plans.person_limit_get"],
        );
        let b = def(
            "decide",
            "two.py",
            &body(&[("_v2 = import_policy_get(_v0)", 1), ("if _v2 is None:", 1), ("return ALLOWED", 2)]),
            &["billing.plans.import_policy_get"],
        );
        // A corpus of two makes every shared subject certain, and a certain coincidence is worth
        // nothing — `ln(total/seen)` is zero when everything reaches it. The filler is what makes
        // `billing.plans` rare, which is the whole basis of the subject term.
        let mut defs = vec![a, b];
        for i in 0..8 {
            defs.push(def(
                &format!("filler{i}"),
                "filler.py",
                &body(&[("_v9 = unrelated()", 1), ("return _v9", 1)]),
                &["somewhere.else"],
            ));
        }
        let found = pass_converge(&defs, 0, super::SEED_CAP);
        let pair = found
            .iter()
            .find(|f| f.name == "fits / decide")
            .expect("the subject anchor reaches it");
        // They meet on the prefix both paths share, not on either path itself.
        assert!(
            pair.notes.iter().any(|n| n.contains("billing.plans")),
            "reported under the deepest shared node: {:?}",
            pair.notes
        );
        let subject = pair.facets.iter().find(|(name, _)| name == "subject").expect("subject vote");
        assert!(subject.1 > 0, "the subject carries the evidence: {:?}", pair.facets);
    }

    #[test]
    fn a_definition_does_not_diverge_from_itself() {
        let a = def("a", "one.py", &body(&[("_v1 = f(_v0)", 1), ("_v1 = f(_v0)", 1), ("return _v1", 1)]), &[]);
        assert!(pass_converge(&[a], 0, super::SEED_CAP).is_empty());
    }
}
