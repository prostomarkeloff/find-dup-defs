//! **Type-3** (ECScan) cross-name near-copy detection — a soft cosine over statements, judged
//! against a null model of the corpus, with difflib-fast's exact weighted-cosine join as the
//! candidate generator.
//!
//! Given each function's name-agnostic normalized lines (from `py-canon`), the sub-expressions of
//! those lines, and its name: build IDF-weighted vectors, hand them to [`difflib_fast::simjoin`] —
//! an **exact all-pairs weighted-cosine join** (every pair with `cos ≥ θ`) on the SOTA **L2AP**
//! algorithm (an inverted index with a Cauchy–Schwarz prefix bound), asserted bit-identical to an
//! `O(n²)` brute force — then score every candidate pair by the **soft cosine** of its two line
//! vectors (Sidorov, Gelbukh, Gómez-Adorno & Pinto 2014), drop the pairs other passes own (same
//! name, sync/async twins, equal alpha-renamed canonicals) and the pairs that share no step beyond
//! their signature, union-find the survivors into clusters, and report the exact min pairwise soft
//! cosine per cluster together with what the shared units are worth under a null model of the
//! corpus ([`Significance`]). The cross-file policy (≥2 distinct names AND files) is applied by the
//! caller.
//!
//! ## What decides, and what was measured not to
//!
//! Two things decide: the proportion — the soft cosine over θ — and one shared step: a pair whose
//! only agreement is its header and the types it names is two definitions with one interface, not
//! one definition written twice. The second replaces the line floor the pass used to have (three
//! lines to take part at all): a function of any length takes part, and what keeps two one-liners
//! that do different things apart is that they do different things.
//!
//! Every finding also carries a p-value: how often two definitions of these sizes share a set of
//! units this improbable under the bipartite configuration model fitted to the corpus (Saracco et
//! al. 2015; [`crate::bicm`]) — the null the decisions pass is built on. It was tried as the verdict,
//! in three forms, and measured on a production tree against a labelled set of near-copies:
//! significance alone (Benjamini–Hochberg over every pair) reported forty times the pairs, with half
//! of the labelled non-copies among them — at that corpus size almost any pair the join admits
//! shares more than chance, because chance there is very little; counting the multiplicity over the
//! candidates instead made the cut vanish into the rate; and on a tree of forty definitions the null
//! can call nothing significant, and dropped a family of three near-copies the pass had always
//! found. Sharing more than chance is necessary for a copy; it is not what a copy is, which is
//! sharing most. So the proportion decides, and the evidence is reported, not ruled on.
//!
//! ## Why soft, and soft how
//!
//! The unit is the statement, and a statement either matches or it does not: a near-copy that
//! differs by one token inside a long line loses that whole line's agreement, and in a short
//! function that is most of the agreement there is. The soft cosine keeps the statement as the unit
//! and lets two statements agree *in part*: the similarity of two line vectors is `Σ s(x, y)·a_x·b_y`
//! over all line pairs, normalized the same way, where `s` is a kernel between lines. With `s` the
//! identity this is the plain cosine, so nothing a hard match found is lost; what is gained is the
//! partial credit for the lines an edit touched.
//!
//! The kernel is the Jaccard of the two lines' **parts** — the line itself and every call,
//! attribute chain, operator application, comprehension and display inside it — not of their
//! tokens. Tokens were measured first: a kernel that credits a line for nine tokens in ten calls two
//! functions near-copies when every line of each differs from the other's by one literal, and on a
//! real tree that was fifty times the pairs, most of them tests that differ in their values. A
//! changed token invalidates every part it sits in, the line included, so the credit a line keeps
//! falls with the depth of the edit, and a short line changed anywhere keeps none: that is the
//! decay a reader applies.
//!
//! The join cannot see partial agreement, so it runs twice, over the lines and over the
//! sub-expressions alone (united): an edited line still shares most of its parts with the
//! original, and the pair reaches the join's threshold on those where the line's own weight — the
//! text the edit changed — would have sunk it. Those vectors are candidate generators only —
//! nested parts overlap and would overweight deep idioms if they were scored (measured: three
//! times the pairs, half of a labelled set of non-copies reported).
#![allow(
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use difflib_fast::simjoin::{cosine_join_with, Corpus};
use difflib_fast::Concurrency;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::bicm::{Bicm, Surprise};
use crate::simgraph::components;

/// Tokenize one line like Python's `re.compile(r"[A-Za-z_]\w*|\d+|\S")` — Unicode-aware, matching
/// `findall`. An identifier starts on an ASCII `[A-Za-z_]` then greedily takes Python's `\w`
/// (Unicode word chars: `is_alphanumeric` or `_`); a run of ASCII digits is `\d+`; any other
/// non-whitespace code point is a single `\S`; whitespace is skipped. Left-to-right, non-overlapping.
fn tokenize(line: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut it = line.char_indices().peekable();
    while let Some(&(start, c)) = it.peek() {
        if c.is_ascii_alphabetic() || c == '_' {
            it.next();
            let mut end = start + c.len_utf8();
            while let Some(&(i, cc)) = it.peek() {
                if cc.is_alphanumeric() || cc == '_' {
                    end = i + cc.len_utf8();
                    it.next();
                } else {
                    break;
                }
            }
            out.push(&line[start..end]);
        } else if c.is_ascii_digit() {
            it.next();
            let mut end = start + 1;
            while let Some(&(i, cc)) = it.peek() {
                if cc.is_ascii_digit() {
                    end = i + 1;
                    it.next();
                } else {
                    break;
                }
            }
            out.push(&line[start..end]);
        } else if c.is_whitespace() {
            it.next();
        } else {
            it.next();
            out.push(&line[start..start + c.len_utf8()]);
        }
    }
    out
}

/// `a == f"a{b}"` or `b == f"a{a}"` — the sync/async naming twin convention (owned by another pass).
fn is_sync_async(a: &str, b: &str) -> bool {
    (a.len() == b.len() + 1 && a.as_bytes().first() == Some(&b'a') && &a[1..] == b)
        || (b.len() == a.len() + 1 && b.as_bytes().first() == Some(&b'a') && &b[1..] == a)
}

/// One function's weighted vector over unit ids, sorted by id, and its L2 norm.
///
/// Counted first, then emitted in FIRST-APPEARANCE order — which is what the linear scan this
/// replaces produced, and the order the norm is summed in, so the float total is unchanged. The
/// weight for one unit id is the same number every time it occurs, so adding it `k` times is
/// order-independent; it is still ADDED `k` times rather than multiplied, because in floating point
/// those are not the same. The scan was quadratic in a function's distinct units; this is linear.
fn weighted_row(seq: &[u32], unit_weight: &[f64]) -> (Vec<(u32, f64)>, f64) {
    let mut count: FxHashMap<u32, u32> = FxHashMap::default();
    for &id in seq {
        *count.entry(id).or_insert(0) += 1;
    }
    let mut v: Vec<(u32, f64)> = Vec::with_capacity(count.len());
    for &id in seq {
        if let Some(k) = count.remove(&id) {
            let w = unit_weight[id as usize];
            let mut acc = 0.0;
            for _ in 0..k {
                acc += w;
            }
            v.push((id, acc));
        }
    }
    let norm = v.iter().map(|&(_, p)| p * p).sum::<f64>().sqrt();
    v.sort_unstable_by_key(|&(id, _)| id);
    (v, norm)
}

/// Jaccard of two sorted sets of part ids: what both lines are made of over what either is.
fn jaccard(left: &[u32], right: &[u32]) -> f64 {
    let (mut li, mut ri, mut both) = (0usize, 0usize, 0usize);
    while li < left.len() && ri < right.len() {
        match left[li].cmp(&right[ri]) {
            std::cmp::Ordering::Less => li += 1,
            std::cmp::Ordering::Greater => ri += 1,
            std::cmp::Ordering::Equal => {
                both += 1;
                li += 1;
                ri += 1;
            }
        }
    }
    let either = left.len() + right.len() - both;
    if either == 0 { 0.0 } else { both as f64 / either as f64 }
}

/// The line kernel: `1` for one line, the Jaccard of their parts (each line's sub-expressions and
/// the line itself) for two.
fn kernel(x: u32, y: u32, line_parts: &[Vec<u32>]) -> f64 {
    if x == y { 1.0 } else { jaccard(&line_parts[x as usize], &line_parts[y as usize]) }
}

/// `Σ_x Σ_y c_x c_y w_x w_y k(x, y)` over the distinct lines of two functions (counts and weights
/// folded into each vector's entries) — the soft inner product.
fn soft_dot(a: &[(u32, f64)], b: &[(u32, f64)], line_parts: &[Vec<u32>]) -> f64 {
    let mut acc = 0.0;
    for &(x, wx) in a {
        for &(y, wy) in b {
            let k = kernel(x, y, line_parts);
            if k > 0.0 {
                acc += wx * wy * k;
            }
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// The null: a definition's units against the corpus's
// ---------------------------------------------------------------------------

/// The bipartite configuration model over definitions × units, and what it says a pair's shared
/// units are worth. Units are the lines and their parts, each counted once per definition.
pub struct Significance {
    model: Bicm,
    /// Per function, its distinct unit ids, ascending.
    units: Vec<Vec<u32>>,
    /// Per function, its degree class.
    row_of: Vec<usize>,
    /// Per unit id, its degree class.
    col_of: Vec<usize>,
}

impl Significance {
    fn fit(units: Vec<Vec<u32>>, n_units: usize) -> Self {
        let mut df = vec![0u32; n_units];
        for own in &units {
            for &u in own {
                df[u as usize] += 1;
            }
        }
        let size: Vec<usize> = units.iter().map(Vec::len).collect();
        let model = Bicm::fit(&size, &df);
        let row_of = size.iter().map(|&k| model.row_class(k)).collect();
        let col_of = df.iter().map(|&d| model.col_class(d)).collect();
        Self { model, units, row_of, col_of }
    }

    /// The surprise of the units two functions share: `Σ −ln q` over them, `q` the null probability
    /// that both hold the unit.
    fn shared_surprise(&self, first: usize, second: usize) -> f64 {
        let (ra, rb) = (self.row_of[first], self.row_of[second]);
        let (left, right) = (&self.units[first], &self.units[second]);
        let (mut li, mut ri, mut surprise) = (0usize, 0usize, 0.0f64);
        while li < left.len() && ri < right.len() {
            match left[li].cmp(&right[ri]) {
                std::cmp::Ordering::Less => li += 1,
                std::cmp::Ordering::Greater => ri += 1,
                std::cmp::Ordering::Equal => {
                    let col = self.col_of[left[li] as usize];
                    let q = self.model.edge(ra, col) * self.model.edge(rb, col);
                    if q < 1.0 {
                        surprise -= q.ln();
                    }
                    li += 1;
                    ri += 1;
                }
            }
        }
        surprise
    }

    /// The law of the surprise for a pair of degree classes.
    fn law(&self, ra: usize, rb: usize) -> Surprise {
        Surprise::new(&self.model.pair_classes(ra, rb))
    }

    /// The p-value of a pair: how often two definitions of these sizes share a set this improbable.
    fn p_value(&self, a: usize, b: usize, law: &Surprise) -> f64 {
        law.tail(self.shared_surprise(a, b))
    }

    /// The p-value of a definition's *whole* unit set being shared — what an exact copy is worth.
    #[must_use]
    pub fn exact_p(&self, a: usize) -> f64 {
        let r = self.row_of[a];
        let mut w = 0.0;
        for &u in &self.units[a] {
            let q = self.model.edge(r, self.col_of[u as usize]).powi(2);
            if q < 1.0 {
                w -= q.ln();
            }
        }
        self.law(r, r).tail(w)
    }

}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

/// What makes two functions indistinguishable to the pass: lines, parts, name, owner.
type RepKey<'a> = (&'a [u32], &'a [(u32, u32)], &'a str, u32);

/// A weighed candidate pair: the pair, its p-value, its soft cosine and whether it shares a step.
type Scored = ((usize, usize), f64, (f64, bool));

/// One cluster of near-copies: its members (function indices), the least soft cosine among them,
/// and the weakest evidence among its pairs.
#[derive(Clone, Debug)]
pub struct Type3Cluster {
    pub members: Vec<usize>,
    pub min_sim: f64,
    pub worst_p: f64,
}

/// Everything the pass derives from the units before any pair is looked at: the interned units,
/// their weights, what each line is made of, and the null. Built once; the cross-name pass reads
/// the null too.
pub struct Prepared {
    seqs: Vec<Vec<u32>>,
    parts: Vec<Vec<(u32, u32)>>,
    line_parts: Vec<Vec<u32>>,
    unit_weight: Vec<f64>,
    /// Per unit id, whether it is context rather than a step: a definition's header line, or a type
    /// its annotations name. Both count toward the similarity — two copies share their interface
    /// too — but a pair that shares nothing else is two definitions with one signature, not one
    /// definition written twice.
    context: Vec<bool>,
    /// The null over definitions × units.
    pub significance: Significance,
}

impl Prepared {
    /// Intern the units, weigh them, read each line's parts, and fit the null.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new(line_lists: &[&[String]], part_lists: &[&[(u32, String)]]) -> Self {
        // Intern distinct units — lines and parts in one vocabulary — → ids (lexicographic / byte
        // order, stable), then per-function id sequences. Sorted and deduplicated on the pool rather
        // than fed one at a time into a set: on a lens run that is thirteen million lines, and the
        // set was the pass's one sequential stretch.
        let mut id_text: Vec<&str> = line_lists
            .par_iter()
            .zip(part_lists.par_iter())
            .flat_map_iter(|(lines, parts)| lines.iter().map(String::as_str).chain(parts.iter().map(|(_, p)| p.as_str())))
            .collect();
        id_text.par_sort_unstable();
        id_text.dedup();
        let unit_id: FxHashMap<&str, u32> = id_text.iter().enumerate().map(|(i, &t)| (t, i as u32)).collect();
        let seqs: Vec<Vec<u32>> = line_lists
            .par_iter()
            .with_min_len(128)
            .map(|lines| lines.iter().map(|l| unit_id[l.as_str()]).collect())
            .collect();
        let parts: Vec<Vec<(u32, u32)>> = part_lists
            .par_iter()
            .with_min_len(128)
            .map(|parts| parts.iter().map(|(line, p)| (*line, unit_id[p.as_str()])).collect())
            .collect();
        let total_lines: usize = seqs.iter().map(Vec::len).sum();

        // What each distinct line is made of: the sorted set of its parts' ids, read off any one
        // occurrence — the same text always yields the same parts — and the line itself, its largest
        // part. Without that, two lines whose sub-expressions all agree but whose own text differs
        // (a literal argument to the outermost call) would be one line to the kernel; with it they
        // agree on `k` parts out of `k + 2`, and a line with no sub-expressions agrees with nothing
        // but itself.
        let mut line_parts: Vec<Vec<u32>> = (0..id_text.len() as u32).map(|id| vec![id]).collect();
        for (seq, fn_parts) in seqs.iter().zip(&parts) {
            for &(line_idx, part) in fn_parts {
                if let Some(&line) = seq.get(line_idx as usize) {
                    line_parts[line as usize].push(part);
                }
            }
        }
        line_parts.par_iter_mut().with_min_len(512).for_each(|ps| {
            ps.sort_unstable();
            ps.dedup();
        });

        // IDF over LINES: df[token] = #line-occurrences containing the token (counted once per
        // distinct line via a set). A statement's vocabulary is the corpus's; the parts are weighed
        // against it, never counted into it, or every call would count twice.
        let mut occ: Vec<u32> = vec![0; id_text.len()];
        for seq in &seqs {
            for &id in seq {
                occ[id as usize] += 1;
            }
        }
        // A sum of counts, so it folds across threads: per-thread tables, merged at the end.
        let df: FxHashMap<&str, u64> = id_text
            .par_iter()
            .enumerate()
            .with_min_len(512)
            .fold(FxHashMap::default, |mut df: FxHashMap<&str, u64>, (id, &text)| {
                let count = u64::from(occ[id]);
                if count > 0 {
                    let mut toks: Vec<&str> = tokenize(text);
                    toks.sort_unstable();
                    toks.dedup();
                    for t in toks {
                        *df.entry(t).or_insert(0) += count;
                    }
                }
                df
            })
            .reduce(FxHashMap::default, |mut a, b| {
                for (t, c) in b {
                    *a.entry(t).or_insert(0) += c;
                }
                a
            });
        let idf: FxHashMap<&str, f64> = if total_lines == 0 {
            FxHashMap::default()
        } else {
            df.iter().map(|(&t, &c)| (t, (total_lines as f64 / c as f64).ln())).collect()
        };
        // Per distinct unit: weight = Σ idf(token) over findall(unit) (with repetition, left-to-right).
        let unit_weight: Vec<f64> = id_text
            .par_iter()
            .with_min_len(256)
            .map(|&t| tokenize(t).iter().map(|tok| idf.get(tok).copied().unwrap_or(0.0)).sum())
            .collect();

        // The null is fitted over what each definition holds, once each: its lines and their parts.
        let units: Vec<Vec<u32>> = seqs
            .par_iter()
            .zip(parts.par_iter())
            .map(|(seq, fn_parts)| {
                let mut own: Vec<u32> = seq.iter().copied().chain(fn_parts.iter().map(|&(_, p)| p)).collect();
                own.sort_unstable();
                own.dedup();
                own
            })
            .collect();
        let significance = Significance::fit(units, id_text.len());
        // The header is every function's first line; a type reference is written `: Name`.
        let mut context: Vec<bool> = id_text.iter().map(|t| t.starts_with(": ")).collect();
        for seq in &seqs {
            if let Some(&head) = seq.first() {
                context[head as usize] = true;
            }
        }
        Self { seqs, parts, line_parts, unit_weight, context, significance }
    }
}

/// Type-3 clusters of renamed near-copies over a [`Prepared`] corpus. Input: each function's name
/// and its cross-name owner (equal ⇔ equal alpha-renamed canonicals). Output: one
/// [`Type3Cluster`] per connected component of significant pairs (size ≥ 2). The caller applies the
/// cross-file contract (≥2 distinct names AND files) and builds the severity-tagged groups.
///
/// 🔴 A pair is left to the exact pass by **its** criterion — the owner — not by equal line
/// sequences. The two used to coincide; with units renumbered per line they do not, and a rule
/// keyed on sequences would drop pairs the exact pass never reports: the gap a pass hands to another
/// must be a gap that pass fills.
///
/// `theta` is the join's threshold — the cost knob. `concurrency` selects the simjoin backend for the
/// all-pairs join (`settings:gpu=on` → `GpuPlusCpu`, the exact f64 CPU+GPU hybrid; `=gpu` → `Gpu`,
/// GPU-dominant f32; default `Cpu`). On a non-`gpu` build or with no Metal device it transparently
/// runs on CPU — the GPU modes are always safe to ask.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn type3_clusters(
    prepared: &Prepared,
    names: &[&str],
    owners: &[u32],
    theta: f64,
    concurrency: Concurrency,
) -> Vec<Type3Cluster> {
    let Prepared { seqs, parts, line_parts, unit_weight, context, significance } = prepared;
    let n = seqs.len();
    if n < 2 {
        return Vec::new();
    }

    // 🔴 The join runs over the DISTINCT `(line sequence, parts, name, owner)` tuples, not over every
    // function. 86% of the functions here repeat another one exactly — the same duplication the rest
    // of the tool trades on — and the join is quadratic, so that is a fifty-fold factor on its input.
    //
    // The IDF above is deliberately NOT deduplicated: `df` counts how many functions contain a
    // line, which is a property of the corpus, and thinning it would reweight every vector.
    //
    // Sound because two functions agreeing on all four keys are indistinguishable to everything
    // below: the same rows, hence the same scores to anything; the same name, hence the same
    // `names[i] != names[j]` and `is_sync_async` verdicts; the same owner, hence the same verdict on
    // whether the exact pass has the pair. A representative that joins nothing expands to a group
    // sharing ONE name, and the caller drops any cluster with fewer than two distinct names — exactly
    // as it dropped the singletons this replaces.
    let mut seen: FxHashMap<RepKey<'_>, usize> = FxHashMap::default();
    let mut reps: Vec<usize> = Vec::new();
    let mut spoken_by: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let key = (seqs[i].as_slice(), parts[i].as_slice(), names[i], owners[i]);
        if let Some(&r) = seen.get(&key) {
            spoken_by[r].push(i);
        } else {
            seen.insert(key, reps.len());
            reps.push(i);
            spoken_by.push(vec![i]);
        }
    }

    // Per-representative vectors: the line vector the score reads, and the parts vector the second
    // join reads (distinct units, weight accumulated by repeated `+= w`, sorted by id).
    let line_rows: Vec<Vec<(u32, f64)>> =
        reps.par_iter().with_min_len(128).map(|&i| weighted_row(&seqs[i], unit_weight).0).collect();
    let part_rows: Vec<Vec<(u32, f64)>> = reps
        .par_iter()
        .with_min_len(128)
        .map(|&i| {
            let only: Vec<u32> = parts[i].iter().map(|&(_, p)| p).collect();
            weighted_row(&only, unit_weight).0
        })
        .collect();
    // The soft self-product of every line vector — the normalizer the soft cosine divides by.
    let self_soft: Vec<f64> =
        line_rows.par_iter().with_min_len(64).map(|row| soft_dot(row, row, line_parts)).collect();
    // The steps alone — the lines that are neither the header nor a type — for the one question the
    // similarity cannot answer: do the two agree on anything they *do*?
    let body_rows: Vec<Vec<(u32, f64)>> = line_rows
        .par_iter()
        .with_min_len(128)
        .map(|row| row.iter().copied().filter(|&(id, _)| !context[id as usize]).collect())
        .collect();

    // Exact all-pairs weighted-cosine joins — the candidates — on the selected backend: CPU, or
    // difflib-fast's Metal GPU hybrid when `settings:gpu=on`. Two joins, united: over the lines,
    // which is every pair the plain cosine would have scored, and over the parts alone, which
    // reaches the pairs whose agreement is inside edited lines — a line's own weight is exactly what
    // the edit changes, so it is kept out of that vector.
    let mut candidates = cosine_join_with(&Corpus::from_rows(&line_rows), theta, concurrency); // (j, i, cos), j < i
    candidates.extend(cosine_join_with(&Corpus::from_rows(&part_rows), theta, concurrency));
    candidates.par_sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    candidates.dedup_by(|a, b| (a.0, a.1) == (b.0, b.1));

    // Keep cross-name, non-twin pairs the exact pass does not own (other passes own same-name /
    // sync-async / equal-canonical clones), then weigh each: its p-value under the null and its soft
    // cosine. The law of the surprise depends only on the two degree classes, so it is built once
    // per class pair that occurs among the candidates.
    let sig = significance;
    let pairs: Vec<(usize, usize)> = candidates
        .into_iter()
        .filter(|&(j, i, _)| {
            let (ri, rj) = (reps[i], reps[j]);
            names[ri] != names[rj] && !is_sync_async(names[ri], names[rj]) && owners[ri] != owners[rj]
        })
        .map(|(j, i, _)| (j, i))
        .collect();
    let mut by_class: FxHashMap<(usize, usize), Vec<(usize, usize)>> = FxHashMap::default();
    for &(j, i) in &pairs {
        let (ra, rb) = (sig.row_of[reps[j]], sig.row_of[reps[i]]);
        by_class.entry((ra.min(rb), ra.max(rb))).or_default().push((j, i));
    }
    // The soft cosine, and whether the two agree on any step at all.
    let soft = |a: usize, b: usize| -> (f64, bool) {
        let dot = soft_dot(&line_rows[a], &line_rows[b], line_parts);
        let denom = (self_soft[a] * self_soft[b]).sqrt();
        let steps = soft_dot(&body_rows[a], &body_rows[b], line_parts) > 0.0;
        (if denom == 0.0 { 0.0 } else { dot / denom }, steps)
    };
    let (reps_ref, soft_ref) = (&reps, &soft);
    let scored: Vec<Scored> = by_class
        .into_par_iter()
        .flat_map_iter(|((ra, rb), group)| {
            let law = sig.law(ra, rb);
            group
                .into_iter()
                .map(move |(j, i)| ((j, i), sig.p_value(reps_ref[j], reps_ref[i], &law), soft_ref(j, i)))
                .collect::<Vec<_>>()
        })
        .collect();
    if std::env::var_os("FDD_TIMING").is_some() {
        eprintln!(
            "[type-3] functions={n} reps={} candidates={} row classes={} fit residual={:.1e}",
            reps.len(),
            scored.len(),
            sig.model.row_classes(),
            sig.model.residual,
        );
    }
    // The verdict: a pair is a near-copy when it is alike in proportion (soft cosine over θ) and
    // shares a step. The p-value rides along as evidence; see the module doc for why it is not the
    // verdict.
    let mut score: FxHashMap<(usize, usize), (f64, f64)> = FxHashMap::default();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for &(key, p, (s, steps)) in &scored {
        if s > theta && steps {
            score.insert(key, (p, s));
            edges.push(key);
        }
    }

    // Components → exact min soft cosine and weakest p-value over ALL intra-component pairs
    // (single-linkage's conservative figures; a pair not directly joined is scored on the spot).
    // Taken over the representatives: a pair of twins scores exactly 1.0 and so can never be the
    // minimum.
    //
    // Measured against a greedy clique cover on a labelled set of real near-copies: the cover gave
    // up a tenth of them for one false pair fewer in twenty — the chains it cuts are mostly families
    // whose members sit just under the cut of each other.
    components(reps.len(), &edges)
        .into_par_iter()
        .map(|mut group| {
            group.sort_unstable();
            let (mut min_sim, mut worst_p) = (f64::INFINITY, 0.0f64);
            for a in 0..group.len() {
                for b in (a + 1)..group.len() {
                    let key = (group[a], group[b]);
                    let (p, s) = score.get(&key).copied().unwrap_or_else(|| {
                        let (ra, rb) = (sig.row_of[reps[key.0]], sig.row_of[reps[key.1]]);
                        (sig.p_value(reps[key.0], reps[key.1], &sig.law(ra.min(rb), ra.max(rb))), soft(key.0, key.1).0)
                    });
                    min_sim = min_sim.min(s);
                    worst_p = worst_p.max(p);
                }
            }
            let mut members: Vec<usize> = group.iter().flat_map(|&r| spoken_by[r].iter().copied()).collect();
            members.sort_unstable();
            Type3Cluster { members, min_sim: if min_sim.is_finite() { min_sim } else { 1.0 }, worst_p }
        })
        .collect()
}
