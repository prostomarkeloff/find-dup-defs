//! **Type-3** (ECScan) cross-name near-copy detection — on difflib-fast's exact weighted-cosine join.
//!
//! Given each function's name-agnostic normalized lines (from `py-canon`) plus its name: build
//! IDF-weighted per-line vectors, then hand them to [`difflib_fast::simjoin`] — an **exact all-pairs
//! weighted-cosine join** (every pair with `cos ≥ θ`) on the SOTA **L2AP** algorithm (an inverted
//! index with a Cauchy–Schwarz prefix bound), asserted bit-identical to an `O(n²)` brute force. We
//! then drop the pairs other passes own (same name, sync/async twins, byte-identical sequences),
//! union-find the
//! survivors into clusters, and report the exact min pairwise cosine per cluster (single-linkage's
//! conservative figure, which drives ERROR/WARNING severity). The cross-file policy (≥2 distinct
//! names AND files) is applied by the caller.
//!
//! This replaces the previous hand-rolled rare-3-line-shingle candidate generation + Python-bit-exact
//! Neumaier cosine. simjoin is **exact all-pairs** — no shingle recall loss — and computes the same
//! IDF-cosine metric (L2-normalised dot); scores shift by ~1e-15 vs the old Neumaier path, immaterial
//! beside the recall change and below the ERROR/WARNING boundary except for a cluster sitting exactly
//! on it. Vector construction (line interning, IDF, weights) is rayon-parallel; maps are `FxHash`.
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

use crate::simgraph::{components, cosine};

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

/// Type-3 clusters of renamed near-copies. Input: each function's normalized lines + name. Output:
/// `(member indices, min pairwise cosine)` per connected component (size ≥ 2). The caller applies the
/// cross-file contract (≥2 distinct names AND files) and builds the severity-tagged groups.
///
/// `concurrency` selects the simjoin backend for the all-pairs join (`settings:gpu=on` → `GpuPlusCpu`,
/// the exact f64 CPU+GPU hybrid; `=gpu` → `Gpu`, GPU-dominant f32; default `Cpu`). On a non-`gpu`
/// build or with no Metal device it transparently runs on CPU — the GPU modes are always safe to ask.
#[must_use]
/// One function's weighted line vector, sorted by line id, and its L2 norm.
///
/// Counted first, then emitted in FIRST-APPEARANCE order — which is what the linear scan this
/// replaces produced, and the order the norm is summed in, so the float total is unchanged. The
/// weight for one line id is the same number every time it occurs, so adding it `k` times is
/// order-independent; it is still ADDED `k` times rather than multiplied, because in floating point
/// those are not the same. The scan was quadratic in a function's distinct lines; this is linear.
fn weighted_row(seq: &[u32], line_weight: &[f64]) -> (Vec<(u32, f64)>, f64) {
    let mut count: FxHashMap<u32, u32> = FxHashMap::default();
    for &id in seq {
        *count.entry(id).or_insert(0) += 1;
    }
    let mut v: Vec<(u32, f64)> = Vec::with_capacity(count.len());
    for &id in seq {
        if let Some(k) = count.remove(&id) {
            let w = line_weight[id as usize];
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

pub fn type3_clusters(
    line_lists: &[&[String]],
    names: &[&str],
    theta: f64,
    concurrency: Concurrency,
) -> Vec<(Vec<usize>, f64)> {
    let n = line_lists.len();
    if n < 2 {
        return Vec::new();
    }
    // Intern distinct lines → ids (lexicographic / byte order, stable), then per-function id sequences.
    // Sorted and deduplicated on the pool rather than fed one at a time into a set: on a lens run
    // that is thirteen million lines, and the set was the pass's one sequential stretch.
    let mut id_text: Vec<&str> = line_lists.par_iter().flat_map_iter(|lines| lines.iter().map(String::as_str)).collect();
    id_text.par_sort_unstable();
    id_text.dedup();
    let line_id: FxHashMap<&str, u32> = id_text.iter().enumerate().map(|(i, &t)| (t, i as u32)).collect();
    let seqs: Vec<Vec<u32>> = line_lists
        .par_iter()
        .with_min_len(128)
        .map(|lines| lines.iter().map(|line| line_id[line.as_str()]).collect())
        .collect();
    let total_lines: usize = seqs.iter().map(Vec::len).sum();


    // IDF: df[token] = #line-occurrences containing token (counted once per distinct line via a set).
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
            let mut toks: Vec<&str> = tokenize(text);
            toks.sort_unstable();
            toks.dedup();
            for t in toks {
                *df.entry(t).or_insert(0) += count;
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
    // Per distinct line: weight = Σ idf(token) over findall(line) (with repetition, left-to-right).
    let line_weight: Vec<f64> = id_text
        .par_iter()
        .with_min_len(256)
        .map(|&t| tokenize(t).iter().map(|tok| idf.get(tok).copied().unwrap_or(0.0)).sum())
        .collect();

    // 🔴 The join runs over the DISTINCT `(line sequence, name)` pairs, not over every function.
    // 86% of the functions here repeat another one exactly — the same duplication the rest of the
    // tool trades on — and the join is quadratic, so that is a fifty-fold factor on its input.
    //
    // The IDF above is deliberately NOT deduplicated: `df` counts how many functions contain a
    // line, which is a property of the corpus, and thinning it would reweight every vector.
    //
    // Sound because two functions agreeing on both keys are indistinguishable to everything below:
    // the same row and norm, hence the same cosine to anything; the same name, hence the same
    // `names[i] != names[j]` and `is_sync_async` verdicts; the same sequence, hence the same
    // byte-identical rejection. A representative that joins nothing expands to a group sharing ONE
    // name, and the caller drops any cluster with fewer than two distinct names — exactly as it
    // dropped the singletons this replaces.
    let mut seen: FxHashMap<(&[u32], &str), usize> = FxHashMap::default();
    let mut reps: Vec<usize> = Vec::new();
    let mut spoken_by: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let key = (seqs[i].as_slice(), names[i]);
        if let Some(&r) = seen.get(&key) {
            spoken_by[r].push(i);
        } else {
            seen.insert(key, reps.len());
            reps.push(i);
            spoken_by.push(vec![i]);
        }
    }

    // Per-representative vector: distinct lines, weight accumulated by repeated `+= w`, then sorted
    // by id (for the dot merge and for simjoin). Also the L2 norm, for the min_sim cosine.
    let (rows, norms): (Vec<Vec<(u32, f64)>>, Vec<f64>) =
        reps.par_iter().with_min_len(128).map(|&i| weighted_row(&seqs[i], &line_weight)).unzip();

    // Exact all-pairs weighted-cosine join (replaces shingle candidate-gen + per-pair verify), on the
    // selected backend — CPU, or difflib-fast's Metal GPU hybrid when `settings:gpu=on`.
    let corpus = Corpus::from_rows(&rows);
    let pairs = cosine_join_with(&corpus, theta, concurrency); // (j, i, cos), j < i, cos ≥ theta

    // Edges: keep cross-name, non-twin, non-byte-identical pairs strictly above θ (other passes own
    // same-name / sync-async / byte-identical clones).
    let edges: Vec<(usize, usize)> = pairs
        .into_par_iter()
        .filter(|&(j, i, cos)| {
            let (ri, rj) = (reps[i], reps[j]);
            cos > theta
                && names[ri] != names[rj]
                && !is_sync_async(names[ri], names[rj])
                && seqs[ri] != seqs[rj]
        })
        .map(|(j, i, _)| (j, i))
        .collect();

    // Components → exact min cosine over ALL intra-component pairs (single-linkage's conservative
    // figure; can be < θ for a chain A–B–C where A,C aren't directly joined). Taken over the
    // representatives: a pair of twins scores exactly 1.0 and so can never be the minimum.
    components(reps.len(), &edges)
        .into_par_iter()
        .map(|mut group| {
            group.sort_unstable();
            let mut min_sim = theta;
            let mut first = true;
            for a in 0..group.len() {
                for b in (a + 1)..group.len() {
                    let c = cosine(&rows[group[a]], &rows[group[b]], norms[group[a]], norms[group[b]]);
                    if first || c < min_sim {
                        min_sim = c;
                        first = false;
                    }
                }
            }
            let mut members: Vec<usize> = group.iter().flat_map(|&r| spoken_by[r].iter().copied()).collect();
            members.sort_unstable();
            (members, min_sim)
        })
        .collect()
}
