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

use dup_defs_core::{CanonDialect, Def, KindSpec, Statement};
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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    tokens_into(line, vocab, &mut out);
    out
}

/// [`tokens`], appended to `out`. Returns whether the line is **closed**: every quote it opened was
/// shut before the line ended. A closed line tokenizes the same whatever follows it, which is what
/// lets a joined pair of lines be lexed as the two lines' tokens with the separator between them.
fn tokens_into<'a>(line: &'a str, vocab: Vocab, out: &mut Vec<Tok<'a>>) -> bool {
    let bytes = line.as_bytes();
    let mut i = 0;
    let mut closed = true;
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
                let shut = i > body_start && bytes[i - 1] == quote;
                closed &= shut;
                let body_end = if shut { i - 1 } else { i };
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
    closed
}

/// A line — given lexed, with its text length for the buffer — with its slot numbers renumbered
/// from zero, in order of first appearance.
///
/// A frontend numbers a definition's locals in binding order across the whole body, so one statement
/// is `except E as _v0:` where nothing was bound before it and `_v5` five bindings later. Right for a
/// definition, wrong for an index of statements across definitions. This is only the **index** key:
/// two lines that parameterized-match always normalize alike, so it never misses a match, and
/// [`Renaming`] rejects what it over-matches.
fn slot_normalize(toks: &[Tok<'_>], len: usize) -> String {
    let mut out = String::with_capacity(len);
    let mut map: HashMap<&str, usize> = HashMap::default();
    for &token in toks {
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

/// **Control skeleton**: the line — given lexed — with all vocabulary holed out, leaving the grammar.
///
/// An attribute chain collapses into one hole: `a.b.c` navigates to a single value, and the dots are
/// not a step of the procedure. Two definitions answering one question in different words agree here
/// and nowhere else, which is exactly the signal the statement index cannot carry — it keys on words.
fn skeleton(toks: &[Tok<'_>], len: usize) -> String {
    let mut out = String::with_capacity(len);
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

/// Every rarity the weighing reads, precomputed per interned id.
///
/// 🔴 The previous form was four `&str`-keyed hash tables, probed from inside the pair loops: a
/// string hash and a `ln` per probe, and the probes ran to the millions — `peak` alone was 1.6% of
/// the whole run's CPU, on one thread. Every input to a rarity is now an id, so a rarity is an
/// array read. The floats are the same floats: `rarity` over the same two counts.
struct Corpus {
    key: Vec<f64>,
    shape: Vec<f64>,
    name: Vec<f64>,
    /// What a name no key spells weighs — `rarity(1, total)`, as the missing-key default did.
    unknown_name: f64,
    name_id: HashMap<String, u32>,
    subject: Vec<f64>,
    /// How many definitions reach each node; the family index compares it against the cap.
    subject_count: Vec<usize>,
}

impl Corpus {
    /// The rarity of a name by its spelling — for the names a fork parts on, which come out of the
    /// anti-unifier as text. A name that no key of the corpus spells is as rare as one seen once.
    fn name_of(&self, name: &str) -> f64 {
        self.name_id.get(name).map_or(self.unknown_name, |&id| self.name[id as usize])
    }
}

/// `usize` → `u32` for the ids and counts this pass keeps small; saturating rather than truncating.
fn u32_of(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// A position or length inside one body: bounded by [`MAX_STATEMENTS`], so it fits in a `u16`.
fn u16_of(n: usize) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}

/// Where each run of equal keys starts in a sorted slice, plus the slice's length at the end — so
/// `windows(2)` over the result walks the runs. Computed in parallel: the test is between neighbours.
fn run_starts<T: Sync>(items: &[T], same: impl Fn(&T, &T) -> bool + Sync) -> Vec<usize> {
    let mut starts: Vec<usize> =
        (0..items.len()).into_par_iter().filter(|&i| i == 0 || !same(&items[i - 1], &items[i])).collect();
    starts.push(items.len());
    starts
}

// ---------------------------------------------------------------------------
// Bodies and views
// ---------------------------------------------------------------------------

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
    /// Per line, whether it shut every quote it opened — see [`tokens_into`].
    closed: Vec<bool>,
}

impl<'a> Lexed<'a> {
    fn of(lines: &[&'a str], vocab: Vocab) -> Self {
        let mut toks = Vec::new();
        let mut starts = Vec::with_capacity(lines.len() + 1);
        let mut sigs = Vec::with_capacity(lines.len());
        let mut closed = Vec::with_capacity(lines.len());
        for line in lines {
            starts.push(u32_of(toks.len()));
            let from = toks.len();
            closed.push(tokens_into(line, vocab, &mut toks));
            sigs.push(match_sig(&toks[from..]));
        }
        starts.push(u32_of(toks.len()));
        Self { toks, starts, sigs, closed }
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

/// One distinct body — the lines, their nesting, the vocabulary they are read in — and everything
/// the pass derives from those three alone.
///
/// 🔴 This is the unit the pass runs on, not the definition. 86% of definitions spell a body that
/// another definition already spelled, and every table below — the tokens, the statement keys, the
/// shapes, the rarities, the tallies — is a function of the text. Built per definition, each was
/// built seven times over; built here, once. A definition is then a [`View`]: which body, which
/// module nodes, nothing else. Nothing keyed on a body may consult a view's subjects — two
/// definitions with one body can still reach different modules.
///
/// Statement keys, shapes and names are **ids**, numbered in lexicographic order of the text they
/// stand for. The order is load-bearing twice: the seeding walks keys in key order, and the weighing
/// sums shape terms in shape order, and both of those were string orders before. An id assigned in
/// sorted order compares as its string did.
struct Body<'a> {
    lines: Vec<&'a str>,
    /// Each line's text as an id — what the fork memo keys on.
    line_ids: Vec<u32>,
    depths: Vec<u16>,
    vocab: Vocab,
    /// Slot-normalized line, as an id: the statement index key, and the measure of shared *text*.
    keys: Vec<u32>,
    /// Control skeleton per line, as an id: the measure of shared *shape*.
    shapes: Vec<u32>,
    lexed: Lexed<'a>,
    /// The names each line's key spells, with repetition — `name_ids[name_starts[i]..name_starts[i + 1]]`.
    name_starts: Vec<u32>,
    name_ids: Vec<u32>,
    /// The views spelling this body, ascending.
    views: Vec<u32>,
    // Everything below reads the corpus, so it is filled by `derive` once the counts exist.
    /// Whether each line names anything at all: a nameless line is grammar and weighs nothing.
    named: Vec<bool>,
    /// Per line, the rarity of its key.
    key_rarity: Vec<f64>,
    /// Per line, its rarest name — the last of the rarest, as `max_by` picks — with that rarity.
    anchor: Vec<Option<(f64, u32)>>,
    /// Distinct keys, sorted.
    key_set: Vec<u32>,
    /// Distinct shapes, sorted, with how often each occurs and the first line that has it.
    shape_set: Vec<u32>,
    shape_counts: Vec<u32>,
    first_of_shape: Vec<u16>,
    /// Per line, its shape's position in `shape_set`.
    shape_slot: Vec<u16>,
}

/// One definition, as the pass sees it — see [`Body`] for why there is nothing textual here.
struct View {
    /// Index into the caller's `defs`.
    at: u32,
    body: u32,
    /// Index into the shared subject-set table: the interned module nodes this definition reaches.
    subjects: u32,
}

/// Everything shaped from the input, before any counting.
struct Shaped<'a> {
    bodies: Vec<Body<'a>>,
    views: Vec<View>,
    /// Distinct reach sets, each as sorted node ids. Views share them: twins reach alike.
    subject_sets: Vec<Vec<u32>>,
    /// Node id → its dotted name, for the report.
    node_names: Vec<String>,
    /// Shape id → its text, for the family report.
    shape_text: Vec<String>,
    /// Name id → its spelling, for the corpus's lookup table.
    name_text: Vec<String>,
    /// Line id → its text: the gap lines a fork is anti-unified over.
    line_text: Vec<&'a str>,
    n_keys: usize,
}

fn body_of_def(def: &Def) -> &[Statement] {
    // The definition's own header is a declaration, not a step: two definitions sharing a
    // signature shape have not agreed on *doing* anything. The contract puts it first, so it is
    // dropped here rather than by each anchor.
    def.facets.statements.get(1..).unwrap_or(&[])
}

fn vocab_of(def: &Def) -> Vocab {
    def.analysis.as_ref().map_or(Vocab::Source, |a| Vocab::of(a.canon_dialect))
}

/// Hash of what a body is. Only a first cut: equal digests are compared in full before they share
/// an id, so a collision costs a comparison and never merges two bodies.
fn digest(body: &[Statement], vocab: Vocab) -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = rustc_hash::FxBuildHasher.build_hasher();
    h.write_u8(u8::from(vocab == Vocab::SExpr));
    for s in body {
        h.write(s.line.as_bytes());
        h.write_u8(0xff);
        h.write_u16(s.depth);
    }
    h.finish()
}

/// The text of one body, before interning.
struct Text<'a> {
    lines: Vec<&'a str>,
    depths: Vec<u16>,
    vocab: Vocab,
}

/// Distinct strings, and every occurrence mapped to its number.
///
/// No hash table: occurrences are sorted by the hash of their text — one machine word, so the sort
/// is cheap — and grouped by exact text within equal hashes, so a collision costs a comparison and
/// never merges two strings. With `sorted`, the distinct spellings are then ordered by text and the
/// ids compare as the strings did; without it the numbering is by hash, which is all an id that is
/// only ever an identity needs.
///
/// Returns the distinct strings in id order, and the id of every occurrence, in input order.
fn intern<'s>(occurrences: &[&'s str], sorted: bool) -> (Vec<&'s str>, Vec<u32>) {
    use std::hash::{BuildHasher, Hasher};
    let mut order: Vec<(u64, u32)> = occurrences
        .par_iter()
        .enumerate()
        .map(|(i, text)| {
            let mut h = rustc_hash::FxBuildHasher.build_hasher();
            h.write(text.as_bytes());
            (h.finish(), u32_of(i))
        })
        .collect();
    order.par_sort_unstable_by(|x, y| {
        x.0.cmp(&y.0).then_with(|| occurrences[x.1 as usize].cmp(occurrences[y.1 as usize]))
    });
    // Where a new string starts: a test between neighbours, so it runs on the pool; the running
    // count over it is integers only and runs in a blink.
    let fresh: Vec<bool> = (0..order.len())
        .into_par_iter()
        .map(|k| {
            k == 0
                || order[k - 1].0 != order[k].0
                || occurrences[order[k - 1].1 as usize] != occurrences[order[k].1 as usize]
        })
        .collect();
    let mut group_at: Vec<u32> = Vec::with_capacity(order.len());
    let mut groups = 0u32;
    for &f in &fresh {
        groups += u32::from(f);
        group_at.push(groups - 1);
    }
    let distinct: Vec<&'s str> =
        (0..order.len()).into_par_iter().filter(|&k| fresh[k]).map(|k| occurrences[order[k].1 as usize]).collect();
    let mut placed: Vec<(u32, u32)> = order.par_iter().zip(&group_at).map(|(&(_, i), &g)| (i, g)).collect();
    placed.par_sort_unstable();
    let group_of: Vec<u32> = placed.into_par_iter().map(|(_, g)| g).collect();
    if !sorted {
        return (distinct, group_of);
    }
    let mut by_text: Vec<u32> = (0..u32_of(distinct.len())).collect();
    by_text.par_sort_unstable_by(|&x, &y| distinct[x as usize].cmp(distinct[y as usize]));
    let mut rank = vec![0u32; distinct.len()];
    for (r, &d) in by_text.iter().enumerate() {
        rank[d as usize] = u32_of(r);
    }
    let text: Vec<&'s str> = by_text.iter().map(|&d| distinct[d as usize]).collect();
    let ids: Vec<u32> = group_of.par_iter().map(|&d| rank[d as usize]).collect();
    (text, ids)
}

/// Shape the input: number the bodies, build each once, intern every string the pass compares by.
#[allow(clippy::too_many_lines)]
fn shape(defs: &[Def]) -> Option<Shaped<'_>> {
    // Which definitions take part, in `defs` order — that order is the view order, and view indices
    // are the pair keys and the tiebreaks everywhere below.
    let eligible: Vec<(u64, u32)> = defs
        .par_iter()
        .enumerate()
        .filter_map(|(at, def)| {
            let body = body_of_def(def);
            if body.len() < 2 || body.len() > MAX_STATEMENTS {
                return None;
            }
            Some((digest(body, vocab_of(def)), u32_of(at)))
        })
        .collect();
    if eligible.len() < 2 {
        return None;
    }

    // Number the bodies: equal digests are candidates, equal text is the verdict. Sorted by
    // `(digest, at)`, every run of one digest lists its views ascending, which is the order each
    // body's view list is kept in.
    let mut order: Vec<u32> = (0..u32_of(eligible.len())).collect();
    order.par_sort_unstable_by_key(|&v| eligible[v as usize]);
    let starts = run_starts(&order, |&x, &y| eligible[x as usize].0 == eligible[y as usize].0);
    let numbered: Vec<Vec<Vec<u32>>> = starts
        .par_windows(2)
        .map(|w| {
            let mut groups: Vec<Vec<u32>> = Vec::new();
            for &v in &order[w[0]..w[1]] {
                let def = &defs[eligible[v as usize].1 as usize];
                let same = groups.iter_mut().find(|g| {
                    let rep = &defs[eligible[g[0] as usize].1 as usize];
                    vocab_of(rep) == vocab_of(def) && body_of_def(rep) == body_of_def(def)
                });
                match same {
                    Some(g) => g.push(v),
                    None => groups.push(vec![v]),
                }
            }
            groups
        })
        .collect();
    let mut body_of = vec![u32::MAX; eligible.len()];
    let mut rep_def: Vec<u32> = Vec::new();
    let mut views_of: Vec<Vec<u32>> = Vec::new();
    for groups in numbered {
        for members in groups {
            let id = u32_of(rep_def.len());
            for &v in &members {
                body_of[v as usize] = id;
            }
            rep_def.push(eligible[members[0] as usize].1);
            views_of.push(members);
        }
    }

    let texts: Vec<Text<'_>> = rep_def
        .par_iter()
        .map(|&at| {
            let def = &defs[at as usize];
            let body = body_of_def(def);
            Text {
                lines: body.iter().map(|s| s.line.as_str()).collect(),
                depths: body.iter().map(|s| s.depth).collect(),
                vocab: vocab_of(def),
            }
        })
        .collect();

    // Where each body's lines start in the flattened occurrence lists.
    let mut line_start: Vec<usize> = Vec::with_capacity(texts.len() + 1);
    line_start.push(0);
    for t in &texts {
        line_start.push(line_start[line_start.len() - 1] + t.lines.len());
    }
    // Lines first. A statement key, a shape and the names a line spells are all functions of the
    // line's text and the vocabulary it is read in — so they are computed once per distinct such
    // pair, not once per line of every body: half the lines of a tree of distinct bodies still
    // repeat some other body's line.
    let line_occ: Vec<&str> = texts.iter().flat_map(|t| t.lines.iter().copied()).collect();
    let (line_text, line_ids) = intern(&line_occ, false);
    let unit_occ: Vec<(u32, Vocab)> = texts
        .iter()
        .enumerate()
        .flat_map(|(b, t)| line_ids[line_start[b]..line_start[b + 1]].iter().map(move |&id| (id, t.vocab)))
        .collect();
    let mut units: Vec<(u32, Vocab)> = unit_occ.clone();
    units.par_sort_unstable();
    units.dedup();
    let unit_of_occ: Vec<u32> =
        unit_occ.par_iter().map(|u| u32_of(units.binary_search(u).unwrap_or(0))).collect();
    // One lexing per unit serves all three derivations. The names a KEY spells are the names the
    // line spells in the source vocabulary and nothing in the s-expr one — normalizing rewrites
    // slots only, and an s-expr key re-reads its unquoted identifiers as node tags — so they are
    // read off the same tokens rather than off a re-lexed key.
    let derived: Vec<(String, String, Vec<&str>)> = units
        .par_iter()
        .map(|&(id, vocab)| {
            let line = line_text[id as usize];
            let toks = tokens(line, vocab);
            let names = match vocab {
                Vocab::Source => {
                    toks.iter().filter_map(|tok| if let Tok::Name(n) = tok { Some(*n) } else { None }).collect()
                }
                Vocab::SExpr => Vec::new(),
            };
            (slot_normalize(&toks, line.len()), skeleton(&toks, line.len()), names)
        })
        .collect();
    let unit_keys: Vec<&str> = derived.iter().map(|(key, _, _)| key.as_str()).collect();
    let unit_shapes: Vec<&str> = derived.iter().map(|(_, shape, _)| shape.as_str()).collect();
    let (key_text, key_of_unit) = intern(&unit_keys, true);
    let (shape_text, shape_of_unit) = intern(&unit_shapes, true);
    let key_ids: Vec<u32> = unit_of_occ.par_iter().map(|&u| key_of_unit[u as usize]).collect();
    let shape_ids: Vec<u32> = unit_of_occ.par_iter().map(|&u| shape_of_unit[u as usize]).collect();

    let unit_names: Vec<&[&str]> = derived.iter().map(|(_, _, names)| names.as_slice()).collect();
    let name_occ: Vec<&str> = unit_names.iter().flat_map(|names| names.iter().copied()).collect();
    let (name_text, name_occ_ids) = intern(&name_occ, false);
    let mut unit_name_start: Vec<usize> = Vec::with_capacity(units.len() + 1);
    unit_name_start.push(0);
    for names in &unit_names {
        unit_name_start.push(unit_name_start[unit_name_start.len() - 1] + names.len());
    }
    // Per body: each line's names, as the unit's slice of the interned occurrence list.
    let names_of: Vec<(Vec<u32>, Vec<u32>)> = (0..texts.len())
        .into_par_iter()
        .map(|b| {
            let mut starts = Vec::with_capacity(texts[b].lines.len() + 1);
            let mut ids = Vec::new();
            for &u in &unit_of_occ[line_start[b]..line_start[b + 1]] {
                starts.push(u32_of(ids.len()));
                ids.extend_from_slice(&name_occ_ids[unit_name_start[u as usize]..unit_name_start[u as usize + 1]]);
            }
            starts.push(u32_of(ids.len()));
            (starts, ids)
        })
        .collect();

    // Interning the module tree is a shared table whose ORDER is load-bearing — the id doubles as
    // the tiebreak that decides which subject a pair is reported under, so it has to stay "first
    // appearance in `defs`". Sequential, then, but over distinct reach sets rather than views: a
    // set already seen has every node interned, exactly where the per-view walk would have found
    // them interned too.
    let reach_digest: Vec<u64> = eligible
        .par_iter()
        .map(|&(_, at)| {
            use std::hash::{BuildHasher, Hasher};
            let mut h = rustc_hash::FxBuildHasher.build_hasher();
            for path in &defs[at as usize].facets.reaches {
                h.write(path.as_bytes());
                h.write_u8(0xff);
            }
            h.finish()
        })
        .collect();
    // Group the views by reach set — digest first, bytes decide — on the pool; then intern the
    // nodes of each distinct set in order of the set's first view, which is the order the per-view
    // walk met them in: a set's nodes were interned when its first view was walked, and by then
    // every earlier set's nodes already were.
    let reaches_of = |v: u32| defs[eligible[v as usize].1 as usize].facets.reaches.as_slice();
    let mut by_digest: Vec<(u64, u32)> = reach_digest.par_iter().enumerate().map(|(v, &d)| (d, u32_of(v))).collect();
    by_digest.par_sort_unstable();
    let digest_runs = run_starts(&by_digest, |x, y| x.0 == y.0);
    let mut sets: Vec<Vec<u32>> = digest_runs
        .par_windows(2)
        .flat_map_iter(|w| {
            let mut groups: Vec<Vec<u32>> = Vec::new();
            for &(_, v) in &by_digest[w[0]..w[1]] {
                match groups.iter_mut().find(|g| reaches_of(g[0]) == reaches_of(v)) {
                    Some(g) => g.push(v),
                    None => groups.push(vec![v]),
                }
            }
            groups
        })
        .collect();
    sets.par_sort_unstable_by_key(|members| members[0]);
    let mut node_ids: HashMap<&str, u32> = HashMap::default();
    let mut node_names: Vec<String> = Vec::new();
    let mut set_of_view = vec![0u32; eligible.len()];
    let subject_sets: Vec<Vec<u32>> = sets
        .iter()
        .enumerate()
        .map(|(set, members)| {
            for &v in members {
                set_of_view[v as usize] = u32_of(set);
            }
            let mut nodes: Vec<u32> = Vec::new();
            for path in reaches_of(members[0]) {
                for node in prefixes(path) {
                    let next = u32_of(node_names.len());
                    let id = *node_ids.entry(node).or_insert_with(|| {
                        node_names.push(node.to_owned());
                        next
                    });
                    nodes.push(id);
                }
            }
            nodes.sort_unstable();
            nodes.dedup();
            nodes
        })
        .collect();
    let views: Vec<View> =
        eligible.iter().enumerate().map(|(v, &(_, at))| View { at, body: body_of[v], subjects: set_of_view[v] }).collect();

    let bodies: Vec<Body<'_>> = (0..texts.len())
        .into_par_iter()
        .map(|b| {
            let (t, (name_starts, name_ids_here)) = (&texts[b], &names_of[b]);
            let lexed = Lexed::of(&t.lines, t.vocab);
            let (from, to) = (line_start[b], line_start[b + 1]);
            Body {
                keys: key_ids[from..to].to_vec(),
                shapes: shape_ids[from..to].to_vec(),
                line_ids: line_ids[from..to].to_vec(),
                name_ids: name_ids_here.clone(),
                name_starts: name_starts.clone(),
                lexed,
                lines: t.lines.clone(),
                depths: t.depths.clone(),
                vocab: t.vocab,
                views: views_of[b].clone(),
                named: Vec::new(),
                key_rarity: Vec::new(),
                anchor: Vec::new(),
                key_set: Vec::new(),
                shape_set: Vec::new(),
                shape_counts: Vec::new(),
                first_of_shape: Vec::new(),
                shape_slot: Vec::new(),
            }
        })
        .collect();

    Some(Shaped {
        bodies,
        views,
        subject_sets,
        node_names,
        shape_text: shape_text.into_iter().map(str::to_owned).collect(),
        name_text: name_text.into_iter().map(str::to_owned).collect(),
        line_text,
        n_keys: key_text.len(),
    })
}

/// Every count the weighing needs, taken once over the corpus — per body, times how many
/// definitions spell it. A count is a sum, and a sum over views grouped by body is the same sum.
fn corpus_of(s: &Shaped<'_>) -> Corpus {
    let mut key_count = vec![0usize; s.n_keys];
    let mut shape_count = vec![0usize; s.shape_text.len()];
    let mut name_count = vec![0usize; s.name_text.len()];
    let (mut line_total, mut name_total) = (0usize, 0usize);
    for body in &s.bodies {
        let mult = body.views.len();
        line_total += body.lines.len() * mult;
        name_total += body.name_ids.len() * mult;
        for &k in &body.keys {
            key_count[k as usize] += mult;
        }
        for &sh in &body.shapes {
            shape_count[sh as usize] += mult;
        }
        for &n in &body.name_ids {
            name_count[n as usize] += mult;
        }
    }
    let mut subject_count = vec![0usize; s.node_names.len()];
    for view in &s.views {
        for &node in &s.subject_sets[view.subjects as usize] {
            subject_count[node as usize] += 1;
        }
    }
    let (line_total, name_total, def_total) = (line_total.max(1), name_total.max(1), s.views.len().max(1));
    Corpus {
        key: key_count.iter().map(|&c| rarity(c, line_total)).collect(),
        shape: shape_count.iter().map(|&c| rarity(c, line_total)).collect(),
        name: name_count.iter().map(|&c| rarity(c, name_total)).collect(),
        unknown_name: rarity(1, name_total),
        name_id: s.name_text.iter().enumerate().map(|(i, n)| (n.clone(), u32_of(i))).collect(),
        subject: subject_count.iter().map(|&c| rarity(c, def_total)).collect(),
        subject_count,
    }
}

/// Fill in what a body derives from the corpus: per-line rarities and anchors, and the sorted
/// distinct tables the weighing merges.
fn derive(body: &mut Body<'_>, corpus: &Corpus) {
    let n = body.lines.len();
    body.named = (0..n).map(|i| body.name_starts[i] != body.name_starts[i + 1]).collect();
    body.key_rarity = body.keys.iter().map(|&k| corpus.key[k as usize]).collect();
    body.anchor = (0..n)
        .map(|i| {
            let mut best: Option<(f64, u32)> = None;
            for &id in &body.name_ids[body.name_starts[i] as usize..body.name_starts[i + 1] as usize] {
                let r = corpus.name[id as usize];
                // `>=`: among equally rare names the LAST wins, as `max_by` picks it.
                if best.is_none_or(|(held, _)| r >= held) {
                    best = Some((r, id));
                }
            }
            best
        })
        .collect();
    let mut key_set = body.keys.clone();
    key_set.sort_unstable();
    key_set.dedup();
    body.key_set = key_set;
    let mut shape_set = body.shapes.clone();
    shape_set.sort_unstable();
    shape_set.dedup();
    let mut counts = vec![0u32; shape_set.len()];
    let mut first = vec![u16::MAX; shape_set.len()];
    let mut slot = Vec::with_capacity(n);
    for (i, &sh) in body.shapes.iter().enumerate() {
        let at = shape_set.binary_search(&sh).unwrap_or(0);
        counts[at] += 1;
        if first[at] == u16::MAX {
            first[at] = u16_of(i);
        }
        slot.push(u16_of(at));
    }
    body.shape_set = shape_set;
    body.shape_counts = counts;
    body.first_of_shape = first;
    body.shape_slot = slot;
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
}

/// What the weighing reads of a fork: how much aligned, how many tokens each side's gap has, and
/// the rarest name in the holes.
///
/// The rarest, not a sum: a fork is characterized by the rarest thing it turns on, not by how many
/// tokens happen to differ. Summing made a pair that parted in thirty places outrank a pair that
/// parted on one rare field — backwards, since the first two are simply different code and the
/// second two are one decision made twice. Not the holes themselves — those are text, and text is for the
/// hundred findings that print, which anti-unify again from their lines. Every other fork is a
/// number; on a tree of mostly distinct bodies there were seven hundred thousand of them, each
/// carrying two lists of strings for nobody.
#[derive(Clone, Copy)]
struct ForkTerm {
    aligned: usize,
    tokens_a: usize,
    tokens_b: usize,
    peak: f64,
}

/// [`anti_unify`] over lines already lexed, keeping only the term.
///
/// The same alignment the string form computes, on the same tokens: a closed line lexes the same
/// whatever follows it, so a joined pair of lines is the two lines' tokens with the separator's two
/// between them — see [`tokens_into`]. Names that fall into holes weigh in as they are met, without
/// being collected.
fn fork_term(left: &[Tok<'_>], right: &[Tok<'_>], corpus: &Corpus) -> ForkTerm {
    const LCS_CAP: usize = 160;
    let (tokens_a, tokens_b) = (left.len(), right.len());
    if left.len() > LCS_CAP || right.len() > LCS_CAP {
        fn names<'t>(t: &[Tok<'t>]) -> HashSet<&'t str> {
            t.iter().filter_map(|tok| if let Tok::Name(n) = tok { Some(*n) } else { None }).collect()
        }
        let (l, r) = (names(left), names(right));
        let aligned = l.intersection(&r).count() * 2;
        let peak = l.symmetric_difference(&r).map(|n| corpus.name_of(n)).fold(0.0, f64::max);
        return ForkTerm { aligned, tokens_a, tokens_b, peak };
    }
    let (rows, cols) = (left.len(), right.len());
    let stride = cols + 1;
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
        let mut aligned = 0usize;
        let mut peak = 0.0f64;
        let (mut row, mut col) = (0usize, 0usize);
        let mut hole = |tok: Tok<'_>| {
            if let Tok::Name(name) = tok {
                peak = peak.max(corpus.name_of(name));
            }
        };
        while row < rows && col < cols {
            if left[row] == right[col] {
                aligned += 2;
                row += 1;
                col += 1;
            } else if table[(row + 1) * stride + col] >= table[row * stride + col + 1] {
                hole(left[row]);
                row += 1;
            } else {
                hole(right[col]);
                col += 1;
            }
        }
        for token in &left[row..] {
            hole(*token);
        }
        for token in &right[col..] {
            hole(*token);
        }
        ForkTerm { aligned, tokens_a, tokens_b, peak }
    })
}

/// [`fork_term`] from the text, for a pair of lines that cannot be read off a lexed body: a line
/// that did not shut a quote, or two sides in different vocabularies.
fn fork_term_of_text(a: &str, b: &str, vocab_a: Vocab, vocab_b: Vocab, corpus: &Corpus) -> ForkTerm {
    let (left, right) = (tokens(a, vocab_a), tokens(b, vocab_a));
    let mut term = fork_term(&left, &right, corpus);
    if vocab_b != vocab_a {
        term.tokens_b = tokens(b, vocab_b).len();
    }
    term
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
        let mut holes_a: Vec<String> = l.difference(&r).cloned().collect();
        let mut holes_b: Vec<String> = r.difference(&l).cloned().collect();
        holes_a.sort();
        holes_b.sort();
        return Fork { holes_a, holes_b };
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
        let (mut row, mut col) = (0usize, 0usize);
        let push = |out: &mut Vec<String>, tok: Tok<'_>| {
            if let Tok::Name(name) = tok {
                out.push(name.to_owned());
            }
        };
        while row < rows && col < cols {
            if left[row] == right[col] {
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
        Fork { holes_a, holes_b }
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

/// A run, the lines the two sides took differently, and — when they found each other again — the run
/// after that.
///
/// The second run is the whole point. A gap **bounded on both sides by agreement** is the shape the
/// inconsistent-clone literature reports faults in; an open tail is only "they started alike".
///
/// Sixteen-bit fields: every one is a position or a length inside a body of at most
/// [`MAX_STATEMENTS`] lines, and a seed is held in the millions.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct Run {
    a_at: u16,
    b_at: u16,
    len: u16,
    gap_a: u16,
    gap_b: u16,
    run2: u16,
}

fn same_shape(a: &Body<'_>, b: &Body<'_>, i: usize, j: usize, base: isize) -> bool {
    let (Some(&da), Some(&db)) = (a.depths.get(i), b.depths.get(j)) else { return false };
    #[allow(clippy::cast_possible_wrap)]
    let delta = da as isize - db as isize;
    delta == base
}

fn run_forward<'a>(a: &Body<'a>, b: &Body<'a>, a_from: usize, b_from: usize, base: isize, renaming: &mut Renaming<'a>) -> usize {
    let mut len = 0usize;
    while same_shape(a, b, a_from + len, b_from + len, base)
        && can_match(&a.lexed, &b.lexed, a_from + len, b_from + len)
        && renaming.accepts(a.lexed.line(a_from + len), b.lexed.line(b_from + len))
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
fn extend_block<'a>(a: &Body<'a>, b: &Body<'a>, a_pos: usize, b_pos: usize) -> Option<Run> {
    let mut renaming = Renaming::default();
    if !renaming.accepts(a.lexed.line(a_pos), b.lexed.line(b_pos)) {
        return None; // the index key matched but the slots do not correspond
    }
    #[allow(clippy::cast_possible_wrap)]
    let base = a.depths[a_pos] as isize - b.depths[b_pos] as isize;
    let (mut a_at, mut b_at, mut len) = (a_pos, b_pos, 1usize);
    // Backwards first, so every seed inside one run yields the same start and the caller's dedup
    // collapses them.
    while a_at > 0
        && b_at > 0
        && same_shape(a, b, a_at - 1, b_at - 1, base)
        && can_match(&a.lexed, &b.lexed, a_at - 1, b_at - 1)
        && renaming.accepts(a.lexed.line(a_at - 1), b.lexed.line(b_at - 1))
    {
        a_at -= 1;
        b_at -= 1;
        len += 1;
    }
    while same_shape(a, b, a_at + len, b_at + len, base)
        && can_match(&a.lexed, &b.lexed, a_at + len, b_at + len)
        && renaming.accepts(a.lexed.line(a_at + len), b.lexed.line(b_at + len))
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
            if end_a + gap_a >= a.lines.len() || end_b + gap_b >= b.lines.len() {
                continue;
            }
            // The probe needs its own copy of the renaming, so check first what `run_forward`
            // would check first anyway: a shape mismatch makes `run2` zero, and paying for a copy
            // of both tables to discover that is the common case, not the rare one.
            if !same_shape(a, b, end_a + gap_a, end_b + gap_b, base)
                || !can_match(&a.lexed, &b.lexed, end_a + gap_a, end_b + gap_b)
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
    Some(Run {
        a_at: u16_of(a_at),
        b_at: u16_of(b_at),
        len: u16_of(len),
        gap_a: u16_of(gap_a),
        gap_b: u16_of(gap_b),
        run2: u16_of(run2),
    })
}

/// `n` lines from `from`, clamped to what the definition actually has.
fn slice<T>(lines: &[T], from: usize, n: usize) -> &[T] {
    if n == 0 || from >= lines.len() {
        return &[];
    }
    &lines[from..(from + n).min(lines.len())]
}

/// A seed candidate: the two definitions and the block they agree on — before the expensive
/// anti-unification of the fork, which is done once per distinct signature.
#[derive(Clone, Copy)]
struct Seed {
    a: u32,
    b: u32,
    run: Run,
}

/// What a fork weighs, once anti-unified: the names each side parted by, how much of the block
/// still aligns, the rarest name the agreement rests on, and the rarest name it parts on.
///
/// Keyed by `(body, body, run)`: what a seed weighs depends on the two BODIES and the run, not on
/// which definitions spell them, and eight seeds in nine ask a question already answered.
struct Weighed {
    sharpness: f64,
    anchor: Option<u32>,
    /// The fork term this seed contributes to the divergence.
    peak: f64,
}

/// What a fork is a function of: the gap lines of each side, as ids, and the vocabulary each side
/// is read in. Two signatures parting on the same lines have the same fork, and on a tree where
/// most bodies are distinct that is the eight-to-one the body memo could not reach.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ForkKey {
    vocab_a: Vocab,
    vocab_b: Vocab,
    a: [u32; GAP_MAX],
    b: [u32; GAP_MAX],
}

fn fork_key(bodies: &[Body<'_>], ba: u32, bb: u32, run: Run) -> ForkKey {
    let (va, vb) = (&bodies[ba as usize], &bodies[bb as usize]);
    let gap = |body: &Body<'_>, at: u16, n: u16| -> [u32; GAP_MAX] {
        let mut out = [u32::MAX; GAP_MAX];
        for (slot, &id) in slice(&body.line_ids, usize::from(at), usize::from(n)).iter().enumerate() {
            out[slot] = id;
        }
        out
    };
    ForkKey {
        vocab_a: va.vocab,
        vocab_b: vb.vocab,
        a: gap(va, run.a_at + run.len, run.gap_a),
        b: gap(vb, run.b_at + run.len, run.gap_b),
    }
}

/// The gap lines of one side of a run: which body, from where, how many.
#[derive(Clone, Copy)]
struct Gap {
    body: u32,
    from: usize,
    n: usize,
}

impl Gap {
    fn lines<'s, 'a>(self, bodies: &'s [Body<'a>]) -> &'s [&'a str] {
        slice(&bodies[self.body as usize].lines, self.from, self.n)
    }

    /// The side's tokens as the joined text would lex — or `None` when a line did not shut a
    /// quote, in which case only lexing the text itself is exact.
    fn tokens<'a>(self, bodies: &[Body<'a>], out: &mut Vec<Tok<'a>>) -> bool {
        let body = &bodies[self.body as usize];
        let lines = self.lines(bodies);
        for (k, _) in lines.iter().enumerate() {
            let i = self.from + k;
            if !body.lexed.closed[i] {
                return false;
            }
            if k > 0 {
                out.push(Tok::Punct(";"));
                out.push(Tok::Punct(" "));
            }
            out.extend_from_slice(body.lexed.line(i));
        }
        true
    }
}

/// Weigh one distinct fork from a signature that has it; `None` when neither side has a gap or the
/// sides part on nothing named — the two reasons a seed weighs nothing.
fn fork_of(bodies: &[Body<'_>], corpus: &Corpus, ba: u32, bb: u32, run: Run) -> Option<ForkTerm> {
    let (a_at, b_at, len) = (usize::from(run.a_at), usize::from(run.b_at), usize::from(run.len));
    let ga = Gap { body: ba, from: a_at + len, n: usize::from(run.gap_a) };
    let gb = Gap { body: bb, from: b_at + len, n: usize::from(run.gap_b) };
    if ga.lines(bodies).is_empty() && gb.lines(bodies).is_empty() {
        return None;
    }
    let (vocab_a, vocab_b) = (bodies[ba as usize].vocab, bodies[bb as usize].vocab);
    // Both sides are read in `a`'s vocabulary, so `b`'s lexed tokens serve only when the two agree.
    let term = if vocab_a == vocab_b {
        let (mut left, mut right) = (Vec::new(), Vec::new());
        (ga.tokens(bodies, &mut left) && gb.tokens(bodies, &mut right)).then(|| fork_term(&left, &right, corpus))
    } else {
        None
    };
    let term = term.unwrap_or_else(|| {
        let (pa, pb) = (ga.lines(bodies).join("; "), gb.lines(bodies).join("; "));
        fork_term_of_text(&pa, &pb, vocab_a, vocab_b, corpus)
    });
    if term.peak <= 0.0 {
        return None; // they part on nothing named
    }
    Some(term)
}



/// A pair's run, as the pass keeps it: the block and its weighing.
#[derive(Clone, Copy)]
struct RunFork {
    run: Run,
    /// Whether the seed's sides were the other way round from the pair's: the fork was then
    /// anti-unified with the pair's `b` on the left, and printing it has to do the same.
    flip: bool,
    sharpness: f64,
    anchor: Option<u32>,
    peak: f64,
}

/// A pair's subject seed: the node they meet on, how many reach it, and the fork term of the lines
/// they take alike and word differently — which depends on the two bodies only, so it is taken once
/// per body pair and copied here.
#[derive(Clone, Copy)]
struct SubjectFork {
    node: u32,
    sites: u32,
    peak: f64,
}

/// One candidate pair, with whichever of the two anchors reached it.
struct Pair {
    a: u32,
    b: u32,
    run: Option<RunFork>,
    subject: Option<SubjectFork>,
}

/// Every run one shared statement seeds, in the order its sites pair up.
///
/// `extend_block` reads nothing but the two bodies and the two positions, so its answer is a
/// function of `(body, position)` twice over: a key whose sites are copies of a few bodies asks one
/// question many times and gets a few answers. The answers are held in a flat table indexed by the
/// sites' distinct `(body, position)` identities — `SEED_CAP` bounds how many sites a key has, so
/// the table is small.
///
/// Walked in the site order regardless: the seed SEQUENCE decides which run a pair keeps when
/// several are the same length.
fn seeds_for_key(sites: &[(u32, u16, u32)], width: usize, views: &[View], bodies: &[Body<'_>], out: &mut Vec<Seed>) {
    SEED_SCRATCH.with_borrow_mut(|memo| {
    memo.clear();
    memo.resize(width * width, None);
    for (i, &(a, a_pos, ia)) in sites.iter().enumerate() {
        for &(b, b_pos, ib) in &sites[i + 1..] {
            // One file does not disqualify a pair — a module that gathers one concern is exactly
            // where its near-copies collect. Being the same definition does: one line repeated
            // inside one body diverges from nothing.
            if a == b {
                continue;
            }
            let cell = ia as usize * width + ib as usize;
            let run = if let Some(hit) = memo[cell] {
                hit
            } else {
                let computed = extend_block(
                    &bodies[views[a as usize].body as usize],
                    &bodies[views[b as usize].body as usize],
                    a_pos as usize,
                    b_pos as usize,
                );
                memo[cell] = Some(computed);
                computed
            };
            if let Some(run) = run {
                out.push(Seed { a, b, run });
            }
        }
    }
    });
}

thread_local! {
    /// The per-key memo of [`seeds_for_key`], reused across keys instead of allocated per key.
    /// Outer `None`: not computed yet; inner `None`: computed, and the slots did not correspond.
    #[allow(clippy::option_option)]
    static SEED_SCRATCH: std::cell::RefCell<Vec<Option<Option<Run>>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Weigh one distinct `(body, body, run)` signature over its fork.
fn weigh_signature(bodies: &[Body<'_>], fork: ForkTerm, ba: u32, bb: u32, run: Run) -> Weighed {
    let (va, vb) = (&bodies[ba as usize], &bodies[bb as usize]);
    let (a_at, b_at, len) = (usize::from(run.a_at), usize::from(run.b_at), usize::from(run.len));
    let counted = |body: &Body<'_>, from: usize| -> usize {
        (from..(from + len).min(body.lines.len())).map(|i| body.lexed.line(i).len()).sum()
    };
    let agreed = counted(va, a_at) + counted(vb, b_at);
    let gap = fork.tokens_a + fork.tokens_b;
    #[allow(clippy::cast_precision_loss)]
    let sharpness = if agreed + gap == 0 { 0.0 } else { (agreed as f64 + fork.aligned as f64) / (agreed + gap) as f64 };
    // The rarest name the run holds — the last of the rarest, as the `max_by` this replaces picked.
    let mut anchor: Option<(f64, u32)> = None;
    for i in a_at..(a_at + len).min(va.lines.len()) {
        if let Some((r, id)) = va.anchor[i] {
            if anchor.is_none_or(|(held, _)| r >= held) {
                anchor = Some((r, id));
            }
        }
    }
    Weighed { sharpness, anchor: anchor.map(|(_, id)| id), peak: fork.peak }
}


/// Seed by shared statement: every pair of sites of every statement that occurs in at least two and
/// at most `cap` places, extended into a block, weighed once per distinct signature, and reduced to
/// one run per pair — the longest, and among equals the first seeded.
///
/// 🔴 Три фазы вместо одного цикла, и делятся они ровно по тому, что можно считать независимо.
/// Растяжение блока и разбор развилки — работа на пару, общего состояния у них нет; отсев же
/// («этот прогон уже посеян другим утверждением», «у пары остаётся ДЛИННЕЙШИЙ прогон») читает и
/// пишет одну таблицу на пару, и порядок в нём несущий. Поэтому считаем параллельно, а решаем —
/// тоже параллельно, но ПО ПАРАМ: посевы сортируются по (пара, порядковый номер), и внутри одной
/// пары решение принимается ровно в том порядке, в каком его принимал однопоточный цикл. Между
/// парами общего состояния нет, так что параллельность здесь ничего не меняет.
#[allow(clippy::too_many_lines)]
fn seed_by_statement(s: &Shaped<'_>, corpus: &Corpus, cap: usize) -> Vec<(u32, u32, RunFork)> {
    let (bodies, views) = (&s.bodies, &s.views);
    // The statement index, per BODY: `sites[start[k]..start[k + 1]]` lists the `(body, position)`
    // pairs spelling key `k`, bodies ascending. Keys are ids in lexicographic order, so walking
    // them in id order is the sorted walk the seeding always did.
    let mut start = vec![0u32; s.n_keys + 1];
    for body in bodies {
        for &k in &body.keys {
            start[k as usize + 1] += 1;
        }
    }
    for k in 0..s.n_keys {
        start[k + 1] += start[k];
    }
    let mut fill = start.clone();
    let mut sites: Vec<(u32, u16)> = vec![(0, 0); start[s.n_keys] as usize];
    for (b, body) in bodies.iter().enumerate() {
        for (pos, &k) in body.keys.iter().enumerate() {
            let at = &mut fill[k as usize];
            sites[*at as usize] = (u32_of(b), u16_of(pos));
            *at += 1;
        }
    }

    // Extend every seed, in parallel over keys. The view sites of a key are its body sites
    // expanded through each body's view list and put back in `(view, position)` order — the order
    // the per-view index listed them in.
    let seeds: Vec<Seed> = (0..s.n_keys)
        .into_par_iter()
        .flat_map_iter(|k| {
            let body_sites = &sites[start[k] as usize..start[k + 1] as usize];
            let total: usize = body_sites.iter().map(|&(b, _)| bodies[b as usize].views.len()).sum();
            let mut out = Vec::new();
            if total < 2 || total > cap {
                return out;
            }
            let mut view_sites: Vec<(u32, u16, u32)> = Vec::with_capacity(total);
            for (identity, &(b, pos)) in body_sites.iter().enumerate() {
                for &v in &bodies[b as usize].views {
                    view_sites.push((v, pos, u32_of(identity)));
                }
            }
            view_sites.sort_unstable();
            seeds_for_key(&view_sites, body_sites.len(), views, bodies, &mut out);
            out
        })
        .collect();

    // Weigh once per distinct signature; then find each seed's signature.
    let mut signatures: Vec<(u32, u32, Run)> =
        seeds.par_iter().map(|sd| (views[sd.a as usize].body, views[sd.b as usize].body, sd.run)).collect();
    signatures.par_sort_unstable();
    signatures.dedup();
    let sig_id: HashMap<(u32, u32, Run), u32> =
        signatures.iter().enumerate().map(|(i, &sig)| (sig, u32_of(i))).collect();
    // The fork of a signature is a function of its gap lines alone, so it is anti-unified once per
    // distinct gap pair; the signature then only counts tokens and picks its anchor.
    let mut keyed: Vec<(ForkKey, u32)> = signatures
        .par_iter()
        .enumerate()
        .map(|(i, &(ba, bb, run))| (fork_key(bodies, ba, bb, run), u32_of(i)))
        .collect();
    keyed.par_sort_unstable();
    // One representative signature per distinct fork: its lexed gap lines are the fork's text.
    let mut fork_rep: Vec<u32> = Vec::new();
    let mut fork_of_sig = vec![0u32; signatures.len()];
    for (k, &(key, sig)) in keyed.iter().enumerate() {
        if k == 0 || keyed[k - 1].0 != key {
            fork_rep.push(sig);
        }
        fork_of_sig[sig as usize] = u32_of(fork_rep.len() - 1);
    }
    let forks: Vec<Option<ForkTerm>> = fork_rep
        .par_iter()
        .map(|&sig| {
            let (ba, bb, run) = signatures[sig as usize];
            fork_of(bodies, corpus, ba, bb, run)
        })
        .collect();
    let weighed: Vec<Option<Weighed>> = signatures
        .par_iter()
        .enumerate()
        .map(|(i, &(ba, bb, run))| forks[fork_of_sig[i] as usize].map(|fork| weigh_signature(bodies, fork, ba, bb, run)))
        .collect();
    let sig_of: Vec<u32> = seeds
        .par_iter()
        .map(|sd| sig_id[&(views[sd.a as usize].body, views[sd.b as usize].body, sd.run)])
        .collect();

    // Decide per pair. Sorted by `(pair, seed order)`, one pair's seeds are contiguous and in the
    // order the sequential loop met them; the two rules it applied — drop a run already seeded by
    // another statement, keep the longest run and the first among equals — read and write nothing
    // outside the pair.
    // Packed: the pair as one word and the seed's place as another, so the sort never reads a seed.
    let mut order: Vec<(u64, u32)> = seeds
        .par_iter()
        .enumerate()
        .map(|(i, sd)| ((u64::from(sd.a) << 32) | u64::from(sd.b), u32_of(i)))
        .collect();
    order.par_sort_unstable();
    let starts = run_starts(&order, |x, y| x.0 == y.0);
    let runs: Vec<(u32, u32, RunFork)> = starts
        .par_windows(2)
        .filter_map(|w| {
            let chunk = &order[w[0]..w[1]];
            let mut seen: Vec<(u16, u16)> = Vec::new();
            let mut best: Option<(u16, u32)> = None;
            for &(_, i) in chunk {
                let sd = &seeds[i as usize];
                let key = (sd.run.a_at.min(sd.run.b_at), sd.run.a_at.max(sd.run.b_at));
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
                if weighed[sig_of[i as usize] as usize].is_none() {
                    continue;
                }
                // The longest agreement is the strongest evidence, so a pair keeps its longest run
                // rather than its last-seen one — and "longest" is a property of the code.
                if best.is_none_or(|(len, _)| sd.run.len > len) {
                    best = Some((sd.run.len, i));
                }
            }
            let (_, i) = best?;
            let sd = &seeds[i as usize];
            let w = weighed[sig_of[i as usize] as usize].as_ref()?;
            let flip = sd.a > sd.b;
            let run = Run {
                a_at: if flip { sd.run.b_at } else { sd.run.a_at },
                b_at: if flip { sd.run.a_at } else { sd.run.b_at },
                len: sd.run.len,
                gap_a: if flip { sd.run.gap_b } else { sd.run.gap_a },
                gap_b: if flip { sd.run.gap_a } else { sd.run.gap_b },
                run2: sd.run.run2,
            };
            Some((
                sd.a.min(sd.b),
                sd.a.max(sd.b),
                RunFork { run, flip, sharpness: w.sharpness, anchor: w.anchor, peak: w.peak },
            ))
        })
        .collect();
    runs
}

// ---------------------------------------------------------------------------
// Seed two: a shared subject
// ---------------------------------------------------------------------------

/// The line pairs a subject fork is made of: for every line of `x`, the first line of `y` with the
/// same shape when the two are worded differently. Comparing the bodies line for line where their
/// shapes agree is enough to find the fork — the ordering of the two streams is what the statement
/// anchor is for.
fn subject_line_pairs(bodies: &[Body<'_>], x: u32, y: u32, out: &mut Vec<(u32, u32, Vocab)>) {
    let (bx, by) = (&bodies[x as usize], &bodies[y as usize]);
    for (i, &shape) in bx.shapes.iter().enumerate() {
        if let Ok(at) = by.shape_set.binary_search(&shape) {
            let other = usize::from(by.first_of_shape[at]);
            if by.lines[other] != bx.lines[i] {
                out.push((bx.line_ids[i], by.line_ids[other], bx.vocab));
            }
        }
    }
}

/// The fork term of one line against another, read in `vocab`.
fn line_pair_term(s: &Shaped<'_>, a: u32, b: u32, vocab: Vocab, corpus: &Corpus) -> f64 {
    // A single line lexes as its lexed form by definition; the join concern does not arise. The
    // text is only re-lexed when the other side's body was read in another vocabulary.
    let (la, lb) = (&s.line_text[a as usize], &s.line_text[b as usize]);
    fork_term_of_text(la, lb, vocab, vocab, corpus).peak
}

/// The fork term of every distinct body pair — the rarest name among the holes where their shapes
/// agree and their words do not. Only the term survives: the holes of a subject seed feed the score
/// and are never printed.
///
/// The term is a maximum over the line pairs, and a line pair recurs across body pairs far more
/// than a body pair recurs, so each distinct line pair is anti-unified once and the maximum is
/// taken over lookups.
fn subject_peaks(s: &Shaped<'_>, corpus: &Corpus, body_pairs: &[(u32, u32)]) -> Vec<f64> {
    let bodies = &s.bodies;
    let mut line_pairs: Vec<(u32, u32, Vocab)> = body_pairs
        .par_iter()
        .flat_map_iter(|&(x, y)| {
            let mut out = Vec::new();
            subject_line_pairs(bodies, x, y, &mut out);
            out
        })
        .collect();
    line_pairs.par_sort_unstable();
    line_pairs.dedup();
    let peaks: Vec<f64> = line_pairs.par_iter().map(|&(a, b, vocab)| line_pair_term(s, a, b, vocab, corpus)).collect();
    body_pairs
        .par_iter()
        .map(|&(x, y)| {
            let mut out = Vec::new();
            subject_line_pairs(bodies, x, y, &mut out);
            out.iter()
                .map(|pair| peaks[line_pairs.binary_search(pair).unwrap_or(0)])
                .fold(0.0, f64::max)
        })
        .collect()
}

fn seed_by_subject(s: &Shaped<'_>, corpus: &Corpus, cap: usize) -> Vec<(u32, u32, SubjectFork)> {
    let views = &s.views;
    let mut index: Vec<Vec<u32>> = vec![Vec::new(); s.node_names.len()];
    for (v, view) in views.iter().enumerate() {
        for &node in &s.subject_sets[view.subjects as usize] {
            index[node as usize].push(u32_of(v));
        }
    }
    // Rarest node first: a pair meeting at `a.b.c` also meets at `a.b` and at `a`, and only the most
    // specific of those says anything. Claiming pairs as they are taken gives each its sharpest
    // subject and reports it once.
    // 🔴 The tiebreak is not cosmetic. Equally-rare nodes are common (every node in a package
    // reached by the same set of definitions ties). Without a total order, which node a pair gets
    // claimed at — and therefore which subject the finding is reported under — varies run to run.
    // Interning order is by first appearance in `defs`, so the id is a stable second key.
    let mut nodes: Vec<u32> = (0..u32_of(index.len())).filter(|&n| !index[n as usize].is_empty()).collect();
    nodes.sort_by(|x, y| {
        corpus.subject[*y as usize]
            .partial_cmp(&corpus.subject[*x as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.cmp(y))
    });
    // Захват пары за узлом — решение ПОРЯДКА («кто первый, за тем и субъект»), и потому идёт
    // последовательно; разбор развилки захваченной пары ни от кого не зависит и уходит в параллель.
    let mut claimed: HashSet<u64> = HashSet::default();
    let mut taken: Vec<(u32, u32, u32, u32)> = Vec::new();
    for node in nodes {
        let sites = &index[node as usize];
        if sites.len() < 2 || sites.len() > cap {
            continue;
        }
        for (i, &a) in sites.iter().enumerate() {
            for &b in &sites[i + 1..] {
                let (lo, hi) = (a.min(b), a.max(b));
                if claimed.insert((u64::from(lo) << 32) | u64::from(hi)) {
                    taken.push((node, u32_of(sites.len()), lo, hi));
                }
            }
        }
    }

    // The fork of a subject seed reads the two bodies and nothing else, so it is computed once per
    // distinct body pair and indexed — the same eight-to-one redundancy the statement seeding and
    // the weighing both pay for.
    let mut body_pairs: Vec<(u32, u32)> =
        taken.par_iter().map(|&(_, _, a, b)| (views[a as usize].body, views[b as usize].body)).collect();
    body_pairs.par_sort_unstable();
    body_pairs.dedup();
    let peak_of: Vec<f64> = subject_peaks(s, corpus, &body_pairs);
    let peak_id: HashMap<(u32, u32), u32> = body_pairs.iter().enumerate().map(|(i, &p)| (p, u32_of(i))).collect();
    let mut forks: Vec<(u32, u32, SubjectFork)> = taken
        .into_par_iter()
        .filter_map(|(node, sites, a, b)| {
            let peak = peak_of[peak_id[&(views[a as usize].body, views[b as usize].body)] as usize];
            if peak <= 0.0 {
                return None;
            }
            Some((a, b, SubjectFork { node, sites, peak }))
        })
        .collect();
    forks.par_sort_unstable_by_key(|&(a, b, _)| (a, b));
    forks
}

/// The two seedings' pairs, merged into one list sorted by pair — both inputs already are.
fn merge_pairs(runs: &[(u32, u32, RunFork)], forks: &[(u32, u32, SubjectFork)]) -> Vec<Pair> {
    let mut out: Vec<Pair> = Vec::with_capacity(runs.len() + forks.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < runs.len() || j < forks.len() {
        let next_run = runs.get(i).map(|r| (r.0, r.1));
        let next_fork = forks.get(j).map(|f| (f.0, f.1));
        match (next_run, next_fork) {
            (Some(r), Some(f)) if r == f => {
                out.push(Pair { a: r.0, b: r.1, run: Some(runs[i].2), subject: Some(forks[j].2) });
                i += 1;
                j += 1;
            }
            (Some(r), Some(f)) if r < f => {
                out.push(Pair { a: r.0, b: r.1, run: Some(runs[i].2), subject: None });
                i += 1;
            }
            (Some(r), None) => {
                out.push(Pair { a: r.0, b: r.1, run: Some(runs[i].2), subject: None });
                i += 1;
            }
            (_, Some(f)) => {
                out.push(Pair { a: f.0, b: f.1, run: None, subject: Some(forks[j].2) });
                j += 1;
            }
            (None, None) => break,
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The pair, and its weighing
// ---------------------------------------------------------------------------

/// How surprising it is that these two coincide, over the evidence the finding actually rests on.
///
/// 🔴 Local to the finding, not global to the pair. Summed over the two definitions whole, the
/// textual term is unbounded in their length and swallows everything else: in a top fifty its median
/// came out eighteen times the corpus median while the fork term barely moved, which is to say the
/// ranking had quietly become "which two files overlap most" — a plain clone detector, and the one
/// thing neither anchor is for.
#[derive(Clone, Copy)]
struct Evidence {
    text: f64,
    shape: f64,
    subject: f64,
    jaccard: f64,
}

/// The half of the evidence that depends on the two BODIES and nothing else — how much shape mass
/// they hold in common, and how much of their statement text coincides.
///
/// 🔴 Split out because it is the expensive half and it is not per pair. Two definitions with the
/// same body give the same answer to it whoever their partner is, and 86% of views share a body, so
/// the pairs ask this roughly eight times more often than there are answers. Computed once per
/// distinct body pair (see `score_pairs`) and looked up.
///
/// `subject` and `text` deliberately stay out: the first reads the subjects, which are per view —
/// identical bodies can still reach different modules — and the second depends on the pair's run.
fn body_evidence(a: &Body<'_>, b: &Body<'_>, corpus: &Corpus) -> (f64, f64) {
    // 🔴 Shapes are counted only over the lines the two do NOT already write identically. A line both
    // sides spell the same way is already the statement anchor's evidence, and counting it here would
    // rank near-clones — which that anchor reports with their exact divergence point — above the only
    // pairs this anchor can reach. Subtracted, not penalized by a factor.
    let bag = |view: &Body<'_>, other: &Body<'_>| -> Vec<u32> {
        let mut out = vec![0u32; view.shape_set.len()];
        for (i, &key) in view.keys.iter().enumerate() {
            if other.key_set.binary_search(&key).is_err() {
                out[view.shape_slot[i] as usize] += 1;
            }
        }
        out
    };
    let (left, right) = (bag(a, b), bag(b, a));
    // 🔴 Summed in a fixed order. Float addition is not associative, and a total that came out of a
    // hash-table walk differed in its last bits between runs, which was enough to reorder
    // equally-scoring pairs and hand the same tree a different report every time. The order is the
    // sorted distinct union of both bodies' shapes — shape ids ARE that order — merged from the two
    // sorted lists. Shapes that survive in neither bag contribute nothing to either sum.
    let (mut ia, mut ib) = (0usize, 0usize);
    let (mut shared, mut total) = (0.0, 0.0);
    loop {
        let (shape, in_a, in_b) = match (a.shape_set.get(ia), b.shape_set.get(ib)) {
            (Some(&x), Some(&y)) => match x.cmp(&y) {
                std::cmp::Ordering::Less => {
                    ia += 1;
                    (x, left[ia - 1], 0)
                }
                std::cmp::Ordering::Greater => {
                    ib += 1;
                    (y, 0, right[ib - 1])
                }
                std::cmp::Ordering::Equal => {
                    ia += 1;
                    ib += 1;
                    (x, left[ia - 1], right[ib - 1])
                }
            },
            (Some(&x), None) => {
                ia += 1;
                (x, left[ia - 1], 0)
            }
            (None, Some(&y)) => {
                ib += 1;
                (y, 0, right[ib - 1])
            }
            (None, None) => break,
        };
        if in_a == 0 && in_b == 0 {
            continue;
        }
        let weight = corpus.shape[shape as usize];
        if in_a > 0 {
            total += saturate(in_a as usize) * weight;
        }
        if in_b > 0 {
            total += saturate(in_b as usize) * weight;
        }
        if in_a > 0 && in_b > 0 {
            shared += 2.0 * saturate(in_a.min(in_b) as usize) * weight;
        }
    }
    let cover = if total > 0.0 { shared / total } else { 0.0 };

    // |A ∪ B| = |A| + |B| − |A ∩ B|, over the two sorted key sets.
    let (na, nb) = (a.key_set.len(), b.key_set.len());
    let (mut i, mut j, mut inter) = (0usize, 0usize, 0usize);
    while i < na && j < nb {
        match a.key_set[i].cmp(&b.key_set[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
        }
    }
    let union = na + nb - inter;
    #[allow(clippy::cast_precision_loss)]
    let jaccard = if union == 0 { 0.0 } else { inter as f64 / union as f64 };

    (shared * cover, jaccard)
}

/// The rarest node two sorted subject sets share, or zero when they share none.
fn shared_subject(a: &[u32], b: &[u32], corpus: &Corpus) -> f64 {
    let (mut i, mut j, mut best) = (0usize, 0usize, 0.0f64);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                best = best.max(corpus.subject[a[i] as usize]);
                i += 1;
                j += 1;
            }
        }
    }
    best
}

/// The whole evidence for one pair: the shared half looked up, plus the two terms that are the
/// pair's own.
fn evidence(a: &Body<'_>, sa: &[u32], sb: &[u32], corpus: &Corpus, run: Option<&Run>, shared: (f64, f64)) -> Evidence {
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
        let (a_at, len) = (r.a_at as usize, r.len as usize);
        span(a_at, len)
            .chain(span(a_at + len + r.gap_a as usize, r.run2 as usize))
            .filter(|&i| a.named[i])
            .map(|i| a.key_rarity[i])
            .sum()
    });
    // Per view, not per body: two definitions can spell the same body and still reach different
    // modules, which is exactly what this term is about.
    let subject = shared_subject(sa, sb, corpus);
    Evidence { text, shape: shared.0, subject, jaccard: shared.1 }
}

/// Взвешенная пара — до того, как из неё сделали находку.
struct Scored {
    key: (u32, u32),
    pair: Pair,
    ev: Evidence,
    divergence: f64,
    score: f64,
    /// The cluster this pair reports under, as a number — see [`group_of`].
    group: u64,
}

/// The cluster a pair belongs to: the run's rarest name when there is a run, the run's length when
/// it names nothing, the pair itself otherwise.
///
/// 🔴 Clustering is for RUNS only. A shared run really is one thing with many consumers — one
/// prologue came back as six reports because its variants differ textually while naming the same two
/// functions, and keying by the rarest name merges them. A shared subject is not that: twenty-one
/// definitions reaching one module are twenty-one different procedures, and collapsing their pairs
/// printed the best and hid the rest. The fan-in correction those pairs need is already in
/// `E_subject`, which is the rarity of the node; dividing by cluster size too would charge twice.
///
/// A number, not a string: the three cases partition the id space — names first, then one slot per
/// possible run length, then one per pair — exactly as the `enum` they replace partitioned by
/// variant. An anchor is a name id, so it can never collide with a length.
fn group_of(pair: &Pair, index: usize, n_names: usize) -> u64 {
    match &pair.run {
        Some(r) => match r.anchor {
            Some(name) => u64::from(name),
            None => (n_names as u64) + u64::from(r.run.len),
        },
        None => (n_names as u64) + (1 << 16) + index as u64,
    }
}

fn score_pairs(s: &Shaped<'_>, corpus: &Corpus, pairs: Vec<Pair>) -> Vec<Scored> {
    let (bodies, views, sets) = (&s.bodies, &s.views, &s.subject_sets);
    // The body half of the evidence, once per distinct body pair instead of once per pair — see
    // [`body_evidence`]. Two parallel passes rather than one with a shared table: the answers are
    // computed over a deduplicated list, then read.
    let mut body_pairs: Vec<(u32, u32)> =
        pairs.par_iter().map(|p| (views[p.a as usize].body, views[p.b as usize].body)).collect();
    body_pairs.par_sort_unstable();
    body_pairs.dedup();
    let shared_of: Vec<(f64, f64)> = body_pairs
        .par_iter()
        .map(|&(x, y)| body_evidence(&bodies[x as usize], &bodies[y as usize], corpus))
        .collect();
    let shared_id: HashMap<(u32, u32), u32> = body_pairs.iter().enumerate().map(|(i, &p)| (p, u32_of(i))).collect();
    // 🔴 Взвешивание идёт ПАРАЛЛЕЛЬНО, но по заранее упорядоченному списку, а не по обходу
    // хеш-таблицы: `collect` индексированного параллельного итератора сохраняет позиции, и потому
    // результат не зависит ни от числа ядер, ни от того, какой воркер успел первым.
    let n_names = corpus.name.len();
    let out: Vec<Scored> = pairs
        .into_par_iter()
        .enumerate()
        .filter_map(|(index, pair)| {
            let (va, vb) = (&views[pair.a as usize], &views[pair.b as usize]);
            let shared = shared_of[shared_id[&(va.body, vb.body)] as usize];
            let ev = evidence(
                &bodies[va.body as usize],
                &sets[va.subjects as usize],
                &sets[vb.subjects as usize],
                corpus,
                pair.run.as_ref().map(|r| &r.run),
                shared,
            );
            let total = ev.text + ev.shape + ev.subject;
            if total <= 0.0 {
                return None;
            }
            // The rarest name across all the hole lists. `peak` is a maximum and every term is a
            // rarity, which is non-negative, so the maximum over the parts IS the maximum over the
            // whole — and each part was taken once, where its list was built.
            let divergence = pair.run.as_ref().map_or(0.0, |r| r.peak).max(pair.subject.as_ref().map_or(0.0, |f| f.peak));
            if divergence <= 0.0 {
                return None;
            }
            // Sharpness applies only where it was measured — inside a run, where "how much of the
            // block still aligns" is defined. A subject-seeded pair has no block, and its
            // counterpart is the share of shape it holds in common, which is already in the evidence.
            let sharpness = pair.run.as_ref().map_or(1.0, |r| r.sharpness);
            // Rejoined means the two found each other again: a drifted copy, where alikeness is the
            // premise. Parted for good means a similarity pass already has the pair and only the
            // fork is news.
            let rejoined = pair.run.as_ref().is_some_and(|r| r.run.run2 > 0);
            let novelty = if rejoined { 1.0 } else { 1.0 - ev.jaccard };
            let score = total * divergence * sharpness * novelty;
            let group = group_of(&pair, index, n_names);
            Some(Scored { key: (pair.a, pair.b), pair, ev, divergence, score, group })
        })
        .collect();
    out
}

// ---------------------------------------------------------------------------
// Families
// ---------------------------------------------------------------------------

/// Поджатая клика: её состав, общая масса формы и общие формы (как id) для читателя.
type TightFamily = (Vec<u32>, f64, Vec<u32>);

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
    members: Vec<u32>,
    /// The shape bag every member holds in common, weighted — what makes them a family.
    shared: f64,
    /// Those shared shapes, rarest first, for the reader.
    shapes: Vec<u32>,
    score: f64,
}

/// Greedy maximal-clique cover of a graph given as an adjacency map.
///
/// Greedy (seed = highest-degree unclaimed vertex, grow by neighbours adjacent to every member)
/// rather than exhaustive enumeration: the groups here are bounded by [`SEED_CAP`], but the shape is
/// exactly the dense component where exhaustive enumeration blows up, and a family that is *a*
/// maximal clique rather than *the* largest one answers the question just as well.
fn clique_cover(vertices: &[u32], edges: &HashMap<u32, HashSet<u32>>) -> Vec<Vec<u32>> {
    let mut unclaimed: HashSet<u32> = vertices.iter().copied().collect();
    let mut out = Vec::new();
    while !unclaimed.is_empty() {
        let degree = |v: u32| edges.get(&v).map_or(0, |n| n.iter().filter(|u| unclaimed.contains(u)).count());
        // Ties break on the vertex itself: the seed decides the whole clique, and a seed chosen by
        // the hasher is a family that changes between runs.
        let Some(&seed) = unclaimed.iter().max_by(|x, y| degree(**x).cmp(&degree(**y)).then(y.cmp(x))) else {
            break;
        };
        let mut clique = vec![seed];
        let mut candidates: Vec<u32> =
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

/// The shape bag every member of a group holds in common, weighted and **covered**, plus the shapes
/// themselves, rarest first.
///
/// The intersection over *all* members, not a pairwise average: a family's claim is that every one
/// of them takes this shape, and a shape two of five share is not part of it.
///
/// 🔴 Normalized by what the members do NOT share, exactly as the pair term is. Raw, the intersection
/// over three arbitrary functions survives on `_()` and `_(_)` — "these three call something" — and
/// the first families this produced were made of nothing else. Cover is what says whether the shared
/// shape is most of what these definitions are, or the residue of any three procedures.
fn common_shapes(members: &[u32], s: &Shaped<'_>, corpus: &Corpus) -> (f64, Vec<u32>) {
    let body = |v: u32| &s.bodies[s.views[v as usize].body as usize];
    let Some((&first, rest)) = members.split_first() else { return (0.0, Vec::new()) };
    let b0 = body(first);
    let mut counts: Vec<(u32, u32)> = b0.shape_set.iter().copied().zip(b0.shape_counts.iter().copied()).collect();
    for &member in rest {
        let here = body(member);
        let mut kept = Vec::with_capacity(counts.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < counts.len() && j < here.shape_set.len() {
            match counts[i].0.cmp(&here.shape_set[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    let n = counts[i].1.min(here.shape_counts[j]);
                    if n > 0 {
                        kept.push((counts[i].0, n));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        counts = kept;
    }
    // Summed in shape order — the string order the ids were numbered in.
    let shared: f64 = counts.iter().map(|&(shape, n)| saturate(n as usize) * corpus.shape[shape as usize]).sum();
    // Shown rarest first: the intersection legitimately contains `_()` — every procedure calls
    // something — and leading with it makes a family read as though that is what it is about.
    let mut shapes: Vec<u32> = counts.iter().map(|&(shape, _)| shape).collect();
    shapes.sort_by(|x, y| {
        corpus.shape[*y as usize]
            .partial_cmp(&corpus.shape[*x as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.cmp(y))
    });
    let mut whole: f64 = 0.0;
    for &member in members {
        let here = body(member);
        for (&shape, &n) in here.shape_set.iter().zip(&here.shape_counts) {
            whole += saturate(n as usize) * corpus.shape[shape as usize];
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let cover = if whole > 0.0 { shared * members.len() as f64 / whole } else { 0.0 };
    (shared * cover, shapes)
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
fn tighten(mut members: Vec<u32>, s: &Shaped<'_>, corpus: &Corpus) -> Option<TightFamily> {
    members.sort_unstable();
    let mut best = common_shapes(&members, s, corpus);
    while members.len() > MIN_FAMILY {
        let mut improved: Option<(usize, (f64, Vec<u32>))> = None;
        for drop in 0..members.len() {
            let mut candidate = members.clone();
            candidate.remove(drop);
            let scored = common_shapes(&candidate, s, corpus);
            // Strictly better, and ties keep the larger family: more places is the finding.
            if scored.0 > improved.as_ref().map_or(best.0, |(_, sc)| sc.0) {
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
fn families(pairs: &[(u32, u32)], s: &Shaped<'_>, corpus: &Corpus, cap: usize) -> (Vec<Family>, HashSet<(u32, u32)>) {
    let (views, sets) = (&s.views, &s.subject_sets);
    // 🔴 An edge is registered at EVERY node its two members share, not at the one the pair was
    // claimed on. Claiming a pair at its rarest common node is right for reporting *that pair* — it
    // is the most specific thing the two are both about — and wrong for finding a family, which needs
    // the node the whole group meets on. Built on claims, a group of six sibling functions fragmented
    // into pairs scattered over six different nodes and no family formed at all.
    let mut flat: Vec<(u32, u32, u32)> = pairs
        .par_iter()
        .flat_map_iter(|&(a, b)| {
            let (sa, sb) = (&sets[views[a as usize].subjects as usize], &sets[views[b as usize].subjects as usize]);
            let mut out = Vec::new();
            let (mut i, mut j) = (0usize, 0usize);
            while i < sa.len() && j < sb.len() {
                match sa[i].cmp(&sb[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        out.push((sa[i], a, b));
                        i += 1;
                        j += 1;
                    }
                }
            }
            out
        })
        .collect();
    flat.par_sort_unstable();
    let starts = run_starts(&flat, |x, y| x.0 == y.0);
    // Rarest node first: the same clique forms at every ancestor of the node it really meets on, and
    // the most specific of those is the one worth reporting. Later duplicates are dropped by the
    // member set they carry.
    let mut nodes: Vec<(u32, usize, usize)> =
        starts.windows(2).map(|w| (flat[w[0]].0, w[0], w[1])).collect();
    nodes.sort_by(|x, y| {
        corpus.subject[y.0 as usize]
            .partial_cmp(&corpus.subject[x.0 as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.0.cmp(&y.0))
    });
    // 🔴 Клики считаются параллельно, а отбираются последовательно. Поиск клик и их поджатие —
    // работа внутри ОДНОГО узла, соседи ей не нужны; а вот «эту же семью уже напечатали на более
    // точном субъекте» — решение о порядке, и оно обязано приниматься по списку узлов, отсортированному
    // от редкого к частому. Смешай их — и семья выходила бы то под одним субъектом, то под другим.
    let per_node: Vec<Vec<TightFamily>> = nodes
        .par_iter()
        .map(|&(node, from, to)| {
            // 🔴 The same cap the pair seeding uses, applied here too. A node reached by more
            // definitions than this is infrastructure — the framework, the store, the directory every
            // file lives in — not a thing a handful of definitions are *about*. Without it the family
            // index re-admitted exactly what the pair index excludes, and families formed around
            // "these files are in the same tree".
            if corpus.subject_count[node as usize] > cap {
                return Vec::new();
            }
            let mut edges: HashMap<u32, HashSet<u32>> = HashMap::default();
            let mut vertices: Vec<u32> = Vec::new();
            for &(_, a, b) in &flat[from..to] {
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
                .filter_map(|clique| tighten(clique, s, corpus))
                .collect()
        })
        .collect();

    let mut out = Vec::new();
    let mut absorbed: HashSet<(u32, u32)> = HashSet::default();
    let mut seen_members: HashSet<Vec<u32>> = HashSet::default();
    for (&(node, _, _), cliques) in nodes.iter().zip(per_node) {
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
            let score = corpus.subject[node as usize] * shared * (clique.len() as f64).ln();
            out.push(Family { node, members: clique, shared, shapes, score });
        }
    }
    out.sort_by(|x, y| {
        y.score.partial_cmp(&x.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| x.members.cmp(&y.members))
    });
    (out, absorbed)
}

// ---------------------------------------------------------------------------
// The report: rank, cap, explain
// ---------------------------------------------------------------------------

/// One cluster of scored pairs, reduced to what the ranking needs: its strongest pair and how many
/// places share the run.
struct Group {
    best: u32,
    hub: usize,
}

/// A finding before it is one: enough to rank it, nothing more.
///
/// 🔴 Разбор находки и её сниппет — самая дорогая часть прохода, и до потолка они не нужны ни одной
/// находке: на среднем дереве кластеров набирается четыреста тысяч, а печатается сотня. Считая всё
/// сразу, проход собирал четыреста тысяч `Finding` — имя, участники, фасеты, — сортировал их и тут
/// же выбрасывал. Здесь кандидат — это число и ссылка; строится только то, что переживёт потолок.
#[derive(Clone, Copy)]
struct Candidate {
    thickness: f64,
    /// A family index, or a group index.
    family: Option<u32>,
    group: u32,
}

/// Находка вместе с тем, что понадобится, только если она переживёт потолок.
struct Pending {
    finding: Finding,
    render: Option<u32>,
    family: Option<u32>,
    snippet_def: u32,
}

/// Which candidates can still be in the head of the ranking: everything at or above the `top`-th
/// thickness of its kind. The full order is thickness first, so nothing below that line can be in
/// the first `top` of its kind, and everything on or above it might — ties are settled later, on
/// the full key. Each kind keeps its own head: a family does not compete with a pair for a slot.
fn shortlist(candidates: Vec<Candidate>, top: usize) -> Vec<Candidate> {
    if top == 0 {
        return candidates;
    }
    let floor = |family: bool| -> Option<f64> {
        let mut t: Vec<f64> = candidates.iter().filter(|c| c.family.is_some() == family).map(|c| c.thickness).collect();
        if t.len() <= top {
            return None;
        }
        t.par_sort_unstable_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        Some(t[top - 1])
    };
    let (pairs_floor, families_floor) = (floor(false), floor(true));
    candidates
        .into_par_iter()
        .filter(|c| {
            let floor = if c.family.is_some() { families_floor } else { pairs_floor };
            floor.is_none_or(|f| c.thickness >= f)
        })
        .collect()
}

/// Turn the scored pairs into the ranked report.
#[allow(clippy::too_many_lines)]
fn weigh(defs: &[Def], s: &Shaped<'_>, corpus: &Corpus, pairs: Vec<Pair>, limits: Limits) -> Vec<Finding> {
    let Limits { top, cap } = limits;
    let views = &s.views;
    let scored = score_pairs(s, corpus, pairs);

    // 🔴 Families are built from **scored** pairs, not from the raw claims. An edge has to mean "these
    // two are genuinely alike" — the seed only established that they meet on a subject and part on
    // something named. Built on raw claims, a clique guaranteed pairwise adjacency that guaranteed
    // nothing, and the families that came out shared `_()` and `_(_)`.
    let subject_pairs: Vec<(u32, u32)> =
        scored.iter().filter(|sc| sc.ev.shape > 0.0 && sc.pair.subject.is_some()).map(|sc| sc.key).collect();
    let (families, absorbed) = families(&subject_pairs, s, corpus, cap);

    // Group by cluster: sorted by group id, one cluster's pairs are contiguous, and each cluster is
    // reduced on its own — nothing about a cluster depends on another.
    let mut kept: Vec<(u64, u32)> = scored
        .par_iter()
        .enumerate()
        .filter_map(|(i, sc)| {
            // A pair a family already accounts for is that family, said once per partner. A pair that
            // also grew a run stays: the run is a fact about those two that the family does not carry.
            (!(absorbed.contains(&sc.key) && sc.pair.run.is_none())).then_some((sc.group, u32_of(i)))
        })
        .collect();
    kept.par_sort_unstable();
    let starts = run_starts(&kept, |x, y| x.0 == y.0);
    let groups: Vec<Group> = starts
        .par_windows(2)
        .map(|w| {
            let chunk: Vec<u32> = kept[w[0]..w[1]].iter().map(|&(_, i)| i).collect();
            let chunk = chunk.as_slice();
            // The representative is the strongest pair; ties break on the pair itself, so that a
            // different worker order could never pick a different exemplar — and therefore print a
            // different divergence.
            let best = chunk
                .iter()
                .copied()
                .min_by(|&x, &y| {
                    let (p, q) = (&scored[x as usize], &scored[y as usize]);
                    q.score.partial_cmp(&p.score).unwrap_or(std::cmp::Ordering::Equal).then(p.key.cmp(&q.key))
                })
                .unwrap_or(chunk[0]);
            let mut members: Vec<u32> = chunk.iter().flat_map(|&i| [scored[i as usize].key.0, scored[i as usize].key.1]).collect();
            members.sort_unstable();
            members.dedup();
            Group { best, hub: members.len() }
        })
        .collect();

    // Rank. Squashed to [0, 1] so it sorts alongside every other pass's thickness without
    // pretending to be the same quantity; the raw nats ride in `facets`.
    let mut candidates: Vec<Candidate> = families
        .iter()
        .enumerate()
        .map(|(i, f)| Candidate { thickness: 1.0 - (-f.score / 400.0).exp(), family: Some(u32_of(i)), group: 0 })
        .collect();
    candidates.par_extend(groups.par_iter().enumerate().map(|(i, g)| {
        #[allow(clippy::cast_precision_loss)]
        let hub = g.hub.max(2) as f64;
        let score = scored[g.best as usize].score / hub;
        Candidate { thickness: 1.0 - (-score / 400.0).exp(), family: None, group: u32_of(i) }
    }));
    let candidates = shortlist(candidates, top);

    // Build the findings that can still be printed, then order them the way the whole ranking was
    // ordered — by thickness, then name, then members, so that a definition diverging from two
    // partners lands in one order every run — and apply the cap in that order.
    let mut out: Vec<Pending> = candidates
        .into_par_iter()
        .map(|c| {
            if let Some(fi) = c.family {
                let family = &families[fi as usize];
                Pending {
                    snippet_def: views[family.members[0] as usize].at,
                    render: None,
                    family: Some(fi),
                    finding: family_finding(defs, s, family, c.thickness),
                }
            } else {
                let best = &scored[groups[c.group as usize].best as usize];
                let (a, b) = (&views[best.key.0 as usize], &views[best.key.1 as usize]);
                let (da, db) = (&defs[a.at as usize], &defs[b.at as usize]);
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let facets = vec![
                    ("text".to_owned(), best.ev.text.max(0.0) as usize),
                    ("shape".to_owned(), best.ev.shape.max(0.0) as usize),
                    ("subject".to_owned(), best.ev.subject.max(0.0) as usize),
                    ("fork".to_owned(), best.divergence.max(0.0) as usize),
                ];
                Pending {
                    snippet_def: a.at,
                    render: Some(groups[c.group as usize].best),
                    family: None,
                    finding: Finding {
                        pass: "converge",
                        kind: kind_of(da, db),
                        name: format!("{} / {}", da.name, db.name),
                        // Advisory, always: the output is a ranked list with no threshold, and a
                        // gate that fires on the tail of a ranking teaches people to ignore it.
                        severity: Severity::Info,
                        min_sim: None,
                        loc: da.loc.max(db.loc),
                        args: da.args.max(db.args),
                        thickness: c.thickness,
                        snippet: String::new(),
                        notes: Vec::new(),
                        members: vec![member(defs, a.at as usize), member(defs, b.at as usize)],
                        facets,
                        pattern: None,
                    },
                }
            }
        })
        .collect();
    out.par_sort_by(|x, y| {
        y.finding
            .thickness
            .partial_cmp(&x.finding.thickness)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.finding.name.cmp(&y.finding.name))
            .then_with(|| x.finding.members.cmp(&y.finding.members))
    });
    if top > 0 {
        // Каждый вид держит свою голову: семейство не соревнуется с парой за один слот, и общий
        // потолок дал бы тому, кто на этом дереве набрал больше, заглушить второго целиком.
        let mut kept = 0;
        let mut kept_family = 0;
        out.retain(|p| {
            let seen = if p.family.is_some() { &mut kept_family } else { &mut kept };
            *seen += 1;
            *seen <= top
        });
    }

    // Explain ONLY what survived the cap: the notes and the snippet are the expensive part of a
    // finding, and nothing below the cap needs them.
    out.into_par_iter()
        .map(|mut p| {
            match (p.render, p.family) {
                (Some(best), _) => {
                    let sc = &scored[best as usize];
                    p.finding.notes = render(&sc.pair, s);
                }
                (None, Some(fi)) => {
                    p.finding.notes = family_notes(s, &families[fi as usize]);
                }
                (None, None) => {}
            }
            p.finding.snippet.clone_from(&defs[p.snippet_def as usize].text_orig);
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
fn render(pair: &Pair, shaped: &Shaped<'_>) -> Vec<String> {
    let body = |view: u32| &shaped.bodies[shaped.views[view as usize].body as usize];
    let (left, right) = (body(pair.a), body(pair.b));
    let mut notes = Vec::new();
    if let Some(run) = &pair.run {
        let block = run.run;
        let (a_at, b_at, len) = (usize::from(block.a_at), usize::from(block.b_at), usize::from(block.len));
        notes.push(format!("agreed on {} statement(s):", block.len));
        notes.extend(slice(&left.lines, a_at, len).iter().map(|line| format!("  = {line}")));
        notes.push("parted:".to_owned());
        // The names the two parted by, anti-unified again from the lines: the weighing kept only
        // the term, and this is one of the hundred findings that print. In the seed's own
        // orientation and vocabulary — the alignment's tie-breaks are not symmetric.
        let (pa, pb) = (
            slice(&left.lines, a_at + len, usize::from(block.gap_a)).join("; "),
            slice(&right.lines, b_at + len, usize::from(block.gap_b)).join("; "),
        );
        let fork = if run.flip { anti_unify(&pb, &pa, right.vocab) } else { anti_unify(&pa, &pb, left.vocab) };
        let (holes_a, holes_b) = if run.flip { (&fork.holes_b, &fork.holes_a) } else { (&fork.holes_a, &fork.holes_b) };
        for (side, at, gap, holes) in [
            (left, a_at, usize::from(block.gap_a), holes_a),
            (right, b_at, usize::from(block.gap_b), holes_b),
        ] {
            for line in slice(&side.lines, at + len, gap) {
                notes.push(format!("  {line}"));
            }
            if !holes.is_empty() {
                notes.push(format!("  by: {}", holes.join(", ")));
            }
        }
        if block.run2 > 0 {
            notes.push("and agreed again".to_owned());
        }
    }
    if let Some(subject) = &pair.subject {
        notes.push(format!(
            "subject: {} (reached by {} definitions)",
            shaped.node_names.get(subject.node as usize).map_or("?", String::as_str),
            subject.sites
        ));
    }
    notes
}

/// The family's notes: the shapes every member holds, not one member's body — what the family *is*
/// rather than what one of them happens to look like.
fn family_notes(s: &Shaped<'_>, family: &Family) -> Vec<String> {
    let subject = s.node_names.get(family.node as usize).map_or("?", String::as_str);
    let mut notes = vec![format!("{} definitions around {subject}, all taking one shape:", family.members.len())];
    notes.extend(family.shapes.iter().map(|&shape| format!("  ~ {}", s.shape_text[shape as usize])));
    notes
}

/// One family, as the report sees it — without its notes and snippet, which are added if it prints.
fn family_finding(defs: &[Def], s: &Shaped<'_>, family: &Family, thickness: f64) -> Finding {
    let members: Vec<usize> = family.members.iter().map(|&v| s.views[v as usize].at as usize).collect();
    let first = &defs[members[0]];
    let names: Vec<&str> = members.iter().map(|&i| defs[i].name.as_str()).collect();
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
        thickness,
        snippet: String::new(),
        notes: Vec::new(),
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
    let Some(mut shaped) = shape(defs) else {
        return Vec::new();
    };
    let corpus = corpus_of(&shaped);
    shaped.bodies.par_iter_mut().for_each(|body| derive(body, &corpus));
    let runs = seed_by_statement(&shaped, &corpus, cap);
    let subjects = seed_by_subject(&shaped, &corpus, cap);
    let pairs = merge_pairs(&runs, &subjects);
    weigh(defs, &shaped, &corpus, pairs, limits)
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

    fn graph(edges: &[(u32, u32)]) -> HashMap<u32, HashSet<u32>> {
        let mut out: HashMap<u32, HashSet<u32>> = HashMap::default();
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
