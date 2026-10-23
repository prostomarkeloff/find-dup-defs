//! **Use profiles** — canonicalizing a definition by its *surroundings* instead of its body: the
//! language-blind half of the `use` lens.
//!
//! The body kinds answer "what is this thing"; this one answers "**how is it handled**". A
//! definition's profile is the multiset of statements across the whole tree that mention its name,
//! each canonicalized with the definition itself as the anchor. Two subsystems whose *bodies*
//! diverged — a cache on a JSONB column and a cache on typed columns share no AST — but whose
//! *handling* is identical (written by key, read by key, deleted by age) collapse to the same
//! profile canonical. That is the same primitive re-invented, and no body-based pass can see it,
//! because the duplication lives in the composition, not in any one definition.
//!
//! A frontend contributes one thing: the [`Site`]s of one file — every statement that mentions a
//! known name, rendered with that name as the anchor `_t0` and the anchor's own surface (an
//! attribute read off it, a keyword it is called with) marked `#name#`. Everything after that —
//! the profiles, their floors, the attribute slots, the facts, and reading a tree whose files
//! repeat each other — is here, once.
//!
//! ## Why this needs no resolver
//!
//! The profile is assembled by **name**, with no import resolution and no call graph: a top-level
//! name is a de-facto unique key in an application tree (measured: 2435 distinct names across 2444
//! top-level classes in one production tree). This is not a new assumption — the name-gated pass
//! already groups `(kind, name)` across the whole tree without resolving anything. Same anchor,
//! different question.
//!
//! ## Determinism
//!
//! Sites arrive in file order, which differs between two instances of the same primitive, so the
//! profile is sorted by its name-keeping canonical first; attribute slots are then numbered in
//! order of first appearance *along that sorted sequence*. The facts are therefore independent of
//! how the tree happens to be laid out.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::sync::Arc;

use rayon::prelude::*;

use crate::Def;

/// Profiles with fewer sites than this are dropped: a definition used once or twice carries no
/// *pattern of handling*, only an instance of one, and an exact match between two thin profiles is
/// trivially reachable (`raise X(...)` twice).
const MIN_SITES: usize = 2;

/// Profiles with more sites than this are dropped. A name touched everywhere (a base model, a
/// framework symbol) has a profile that is the codebase's background rather than any one
/// subsystem's shape; it would also dominate the O(n²) work. The structural counterpart of the
/// engine's `--max-name-group`.
const MAX_SITES: usize = 300;

/// Minimum number of *distinct* site shapes a profile must have. A definition called the same way
/// three times (`x = f(y)` ×3) has three sites but one bit of information, and that one shape is a
/// language idiom every codebase repeats — so an exact match between two such profiles is
/// meaningless. Counting distinct shapes instead of sites is what separates "handled in several
/// ways" from "handled the one obvious way, repeatedly".
const MIN_SHAPES: usize = 2;

/// One collected use site: which name it anchors on, its canonical (attributes still marked
/// `#name#` for the profile to renumber), the syntactic context it sits in, and the external calls
/// standing beside it in the same block.
pub struct Site {
    pub name: String,
    pub canon: String,
    /// `guard` / `loop` / `try` / `plain` — the shape of the position the anchor is used in.
    /// Splitting the use channel this way follows the four kinds of usage pattern LUPIN mines
    /// (condition check, iteration, error handling, co-occurrence): the full canonical is
    /// all-or-nothing, so two call sites that agree on *being guarded* while differing in the
    /// guard's wording would otherwise agree on nothing at all.
    pub ctx: &'static str,
    /// Method calls standing in the same block as this site, each `.name`. Not what the anchor
    /// calls (`outgoing`) but what is called *alongside* it — the `fopen`/`fclose` signal, and the
    /// one property here that belongs to neither definition on its own. Only method calls are kept:
    /// `.dumps` is grammar every module shares, while a bare name is an identity this module
    /// happens to have imported, and would tell two copies apart rather than together.
    pub cooc: Vec<String>,
}

/// What a statement holding a use site is — the part of the context it decides itself.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Construct {
    /// Error handling: a `try`.
    Try,
    /// Iteration: a loop's header.
    Loop,
    /// A condition: an `if`, an `assert`, a `match`.
    Guard,
    /// Anything else.
    Plain,
}

/// What encloses a statement holding a use site.
#[derive(Clone, Copy, Default)]
pub struct Within {
    pub in_try: bool,
    pub in_loop: bool,
}

/// The context a statement puts its expressions in, given what encloses it. Enclosure wins over
/// the statement's own kind: a plain call inside a `try` is error handling, whatever its shape.
#[must_use]
pub fn site_ctx(construct: Construct, within: Within) -> &'static str {
    if within.in_try || construct == Construct::Try {
        return "try";
    }
    match construct {
        Construct::Loop => "loop",
        Construct::Guard => "guard",
        _ if within.in_loop => "loop",
        _ => "plain",
    }
}

/// Rewrite the `#name#` attribute markers a site canonical carries. `slots = None` restores the
/// bare attribute name (the name-keeping canonical); `Some(map)` replaces each with its positional
/// `_a{n}`, assigning the next free slot on first sight — so the caller controls numbering by the
/// order in which it feeds sites.
#[must_use]
#[allow(clippy::implicit_hasher)] // always called with this module's own std-hasher map
pub fn rewrite_marks(s: &str, slots: Option<&mut HashMap<String, u32>>) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut slots = slots;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'#' {
            // Push the whole run up to the next '#' at once.
            let start = i;
            while i < bytes.len() && bytes[i] != b'#' {
                i += 1;
            }
            out.push_str(&s[start..i]);
            continue;
        }
        // A marker is `#` + an identifier + `#`; anything else is literal text.
        let mut j = i + 1;
        if j < bytes.len() && (bytes[j].is_ascii_alphabetic() || bytes[j] == b'_') {
            j += 1;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
        }
        if j > i + 1 && j < bytes.len() && bytes[j] == b'#' {
            let attr = &s[i + 1..j];
            match slots.as_mut() {
                Some(map) => {
                    let next = u32::try_from(map.len()).unwrap_or(u32::MAX);
                    let slot = *map.entry(attr.to_owned()).or_insert(next);
                    let _ = write!(out, "_a{slot}");
                }
                None => out.push_str(attr),
            }
            i = j + 1;
        } else {
            out.push('#');
            i += 1;
        }
    }
    out
}

/// Turn the collected sites into the facts of the profiles that survive the floors. Sites are put
/// in canonical order *before* attribute slots are numbered, so no fact depends on file layout.
///
/// Each site comes with how many files spell it — a site found in a file that exists in five copies
/// is five sites. The profile is a multiset, and the count is carried rather than the site cloned:
/// a repeated site sorts next to itself, numbers no new slot, and contributes its fact that many
/// times, which is exactly what five copies of it would have done.
///
/// Three channels rather than one, so that partial agreement is expressible: `use:` the full site
/// canonical, `usectx:` the kind of position it sits in, `cooc:` the calls standing beside it. Two
/// anchors guarded the same way but worded differently share `usectx` while `use` disagrees, which
/// the single canonical could never say.
fn build_profiles(by_name: BTreeMap<&str, Vec<(Site, usize)>>) -> Vec<(String, Vec<String>)> {
    by_name
        .into_par_iter()
        .filter(|(_, sites)| {
            let n: usize = sites.iter().map(|(_, copies)| copies).sum();
            (MIN_SITES..=MAX_SITES).contains(&n)
        })
        .filter_map(|(name, mut sites)| {
            sites.sort_by(|a, b| a.0.canon.cmp(&b.0.canon));
            let mut slots: HashMap<String, u32> = HashMap::new();
            let xname: Vec<String> =
                sites.iter().map(|(s, _)| rewrite_marks(&s.canon, Some(&mut slots))).collect();
            if xname.iter().collect::<HashSet<_>>().len() < MIN_SHAPES {
                return None;
            }
            // Contexts and neighbours are sets: a shape repeated across sites says nothing new,
            // and their whole point is to be coarser than the canonical.
            let ctxs: BTreeSet<&str> = sites.iter().map(|(s, _)| s.ctx).collect();
            let cooc: BTreeSet<&str> =
                sites.iter().flat_map(|(s, _)| s.cooc.iter().map(String::as_str)).collect();
            let mut facts: Vec<String> = Vec::new();
            for (fact, (_, copies)) in xname.into_iter().zip(&sites) {
                let fact = format!("use:{fact}");
                for _ in 1..*copies {
                    facts.push(fact.clone());
                }
                facts.push(fact);
            }
            facts.extend(ctxs.into_iter().map(|c| format!("usectx:{c}")));
            facts.extend(cooc.into_iter().map(|c| format!("cooc:{c}")));
            Some((name.to_owned(), facts))
        })
        .collect()
}

/// The use-site facts of every definition that has any, keyed by name and already prefixed by
/// channel, for [`crate::lens::merge_use_facts`]. `defs` supplies the anchors — every top-level
/// name the body scan found, the first declaration winning on a collision (the same convention the
/// name-gated pass uses when one name covers several definitions); methods are left out, being
/// qualified (`Foo.bar`) and never mentioned bare.
///
/// `sources` is each distinct file's text and path with how many files spell it: a site found in
/// a file that exists in five copies is five sites, exactly as walking the five copies would have
/// found it five times — and a profile is a multiset, so the count is what it keeps. `collect` is
/// the frontend's: the sites of one source (its path says how to parse it), given the names that
/// count as anchors.
#[must_use]
pub fn use_facts<C>(sources: &[(&str, &str, usize)], defs: &[Def], collect: C) -> HashMap<String, Vec<String>>
where
    C: Fn(&str, &str, &HashSet<&str>) -> Vec<Site> + Sync,
{
    let mut decl: BTreeMap<&str, &Def> = BTreeMap::new();
    for d in defs {
        if d.kind.id == "methods" {
            continue;
        }
        decl.entry(d.name.as_str()).or_insert(d);
    }
    let known: HashSet<&str> = decl.keys().copied().collect();
    if known.is_empty() {
        return HashMap::new();
    }

    let per_file: Vec<Vec<Site>> = sources.par_iter().map(|&(src, path, _)| collect(src, path, &known)).collect();

    // Group sites by anchor. The order sites arrive in is not observable past this point: the
    // profile sorts them by canonical, and two sites with one canonical are interchangeable to
    // everything that reads a profile.
    let mut by_name: BTreeMap<&str, Vec<(Site, usize)>> = BTreeMap::new();
    for (&(_, _, copies), sites) in sources.iter().zip(per_file) {
        for site in sites {
            if let Some((k, _)) = decl.get_key_value(site.name.as_str()) {
                by_name.entry(k).or_default().push((site, copies));
            }
        }
    }
    build_profiles(by_name).into_iter().collect()
}

/// A content digest that only narrows — the bytes decide — so it is the cheapest word-at-a-time
/// multiply-and-rotate rather than a keyed hash: over a quarter-gigabyte tree the default hasher
/// was a measurable share of the scan.
fn digest(bytes: &[u8]) -> u64 {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    let mut h: u64 = bytes.len() as u64;
    let (words, rest) = bytes.as_chunks::<8>();
    for word in words {
        h = (h.rotate_left(5) ^ u64::from_le_bytes(*word)).wrapping_mul(K);
    }
    for &b in rest {
        h = (h.rotate_left(5) ^ u64::from(b)).wrapping_mul(K);
    }
    h
}

/// A frontend's scan in lens mode: every file's definitions, with the `use` lens's facts merged in
/// and the lens records scored against the corpus.
///
/// 🔴 In lens mode the engine hands over EVERY file, twins included: the `use` lens counts how many
/// places mention a name across the whole set, which is a fact about the tree and not about any one
/// file, so the engine cannot thin the list for it. That does not mean every file has to be parsed.
/// A twin is the same bytes again, and the same bytes produce the same definitions and the same use
/// sites; so the content is read once, parsed once, its definitions replayed onto each twin's path,
/// and its use sites counted once per copy. On a real monorepo that is one parse in six.
///
/// `scan` is the frontend's per-file scan; `collect` its use-site collector (see [`use_facts`]).
pub fn scan_with_uses<S, C>(files: &[Arc<str>], scan: S, collect: C) -> Vec<Def>
where
    S: Fn(&str, &Arc<str>) -> Vec<Def> + Sync,
    C: Fn(&str, &str, &HashSet<&str>) -> Vec<Site> + Sync,
{
    let sources: Vec<Option<String>> = files.par_iter().map(|f| fs::read_to_string(&**f).ok()).collect();
    let digests: Vec<u64> =
        sources.par_iter().map(|src| digest(src.as_deref().unwrap_or("").as_bytes())).collect();
    // The digest narrows it; the bytes decide it. Which copy is parsed does not vary between runs:
    // the first in the (sorted) list. A file that cannot be read is its own group, and the scan
    // skips it as it always did.
    let mut by_digest: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut parse: Vec<usize> = Vec::new();
    let mut copies_of: Vec<Vec<usize>> = vec![Vec::new(); files.len()];
    for (i, src) in sources.iter().enumerate() {
        let Some(content) = src else {
            parse.push(i);
            continue;
        };
        let seen = by_digest.entry(digests[i]).or_default();
        if let Some(&source) = seen.iter().find(|&&r| sources[r].as_deref() == Some(content.as_str())) {
            copies_of[source].push(i);
        } else {
            seen.push(i);
            parse.push(i);
        }
    }
    let parsed: Vec<Vec<Def>> =
        parse.par_iter().map(|&i| sources[i].as_ref().map_or_else(Vec::new, |src| scan(src, &files[i]))).collect();
    // Definitions in file order for the parsed files, then each twin's copies. The order matters to
    // exactly one consumer — `merge_use_facts` attaches a name's use facts to the FIRST definition
    // bearing it — and a twin's source always precedes the twin, so the first bearer is a parsed
    // file's definition in either order.
    let copies: Vec<Def> = parse
        .par_iter()
        .zip(&parsed)
        .flat_map_iter(|(&i, found)| {
            copies_of[i].iter().flat_map(move |&twin| {
                found.iter().map(move |def| {
                    let mut copy = def.clone();
                    copy.file = Arc::clone(&files[twin]);
                    copy
                })
            })
        })
        .collect();
    let mut defs: Vec<Def> = parsed.into_iter().flatten().collect();
    defs.extend(copies);
    // Phase two: the body scan above supplies the anchors (every top-level name it found), so the
    // profile pass needs no resolver — only a second walk of the same files. The `use` lens sees a
    // definition from outside, so its facts exist only once every file has been read; scoring is
    // corpus-relative for the same reason.
    let distinct: Vec<(&str, &str, usize)> = parse
        .iter()
        .filter_map(|&i| sources[i].as_deref().map(|src| (src, &*files[i], 1 + copies_of[i].len())))
        .collect();
    let facts = use_facts(&distinct, &defs, collect);
    crate::lens::merge_use_facts(&mut defs, facts);
    crate::lens::score_lens_defs(&mut defs);
    defs
}

#[cfg(test)]
mod tests {
    use super::rewrite_marks;
    use std::collections::HashMap;

    #[test]
    fn marker_rewrite_keeps_or_numbers_attributes() {
        let s = "Attribute(Name('_t0'), '#key#') Attribute(Name('_t0'), '#ttl#') Attribute(Name('_t0'), '#key#')";
        assert_eq!(
            rewrite_marks(s, None),
            "Attribute(Name('_t0'), 'key') Attribute(Name('_t0'), 'ttl') Attribute(Name('_t0'), 'key')"
        );
        let mut slots: HashMap<String, u32> = HashMap::new();
        assert_eq!(
            rewrite_marks(s, Some(&mut slots)),
            "Attribute(Name('_t0'), '_a0') Attribute(Name('_t0'), '_a1') Attribute(Name('_t0'), '_a0')"
        );
    }

    #[test]
    fn marker_rewrite_leaves_ordinary_hashes_alone() {
        let s = "Constant('a # b') Constant('#not an ident#')";
        assert_eq!(rewrite_marks(s, None), s);
    }
}
