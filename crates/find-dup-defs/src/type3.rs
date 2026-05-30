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
use rustc_hash::{FxHashMap, FxHashSet};

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

/// Cosine of two sparse line vectors given their precomputed norms: sorted-merge dot of the shared
/// line ids over `norm_a · norm_b`. Same IDF-cosine metric as the join; used for the per-cluster
/// `min_sim` (over all intra-cluster pairs, including the sub-θ ones the join doesn't return).
fn cosine(a: &[(u32, f64)], b: &[(u32, f64)], na: f64, nb: f64) -> f64 {
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    let (mut i, mut j) = (0usize, 0usize);
    let mut dot = 0.0f64;
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    dot / (na * nb)
}

/// Union-find over edge endpoints → connected components.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn components(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..n).collect();
    let mut seen = vec![false; n];
    for &(x, y) in edges {
        seen[x] = true;
        seen[y] = true;
        let (rx, ry) = (uf_find(&mut parent, x), uf_find(&mut parent, y));
        if rx != ry {
            parent[rx] = ry;
        }
    }
    let mut groups: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (i, &s) in seen.iter().enumerate() {
        if s {
            let r = uf_find(&mut parent, i);
            groups.entry(r).or_default().push(i);
        }
    }
    groups.into_values().collect()
}

/// Type-3 clusters of renamed near-copies. Input: each function's normalized lines + name. Output:
/// `(member indices, min pairwise cosine)` per connected component (size ≥ 2). The caller applies the
/// cross-file contract (≥2 distinct names AND files) and builds the severity-tagged groups.
///
/// `concurrency` selects the simjoin backend for the all-pairs join (`settings:gpu=on` → `GpuPlusCpu`,
/// the exact f64 CPU+GPU hybrid; `=gpu` → `Gpu`, GPU-dominant f32; default `Cpu`). On a non-`gpu`
/// build or with no Metal device it transparently runs on CPU — the GPU modes are always safe to ask.
#[must_use]
pub fn type3_clusters(
    line_lists: &[Vec<String>],
    names: &[String],
    theta: f64,
    concurrency: Concurrency,
) -> Vec<(Vec<usize>, f64)> {
    let n = line_lists.len();
    if n < 2 {
        return Vec::new();
    }
    // Intern distinct lines → ids (lexicographic / byte order, stable), then per-function id sequences.
    let mut id_text: Vec<&str> = {
        let mut seen: FxHashSet<&str> = FxHashSet::default();
        let mut distinct: Vec<&str> = Vec::new();
        for lines in line_lists {
            for line in lines {
                if seen.insert(line.as_str()) {
                    distinct.push(line.as_str());
                }
            }
        }
        distinct
    };
    id_text.sort_unstable();
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
    let mut df: FxHashMap<&str, u64> = FxHashMap::default();
    for (id, &text) in id_text.iter().enumerate() {
        let count = u64::from(occ[id]);
        let mut toks: Vec<&str> = tokenize(text);
        toks.sort_unstable();
        toks.dedup();
        for t in toks {
            *df.entry(t).or_insert(0) += count;
        }
    }
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

    // Per-function vector: distinct lines, weight accumulated by repeated `+= w`, then sorted by id
    // (for the dot merge and for simjoin). Also the L2 norm, for the min_sim cosine.
    let (rows, norms): (Vec<Vec<(u32, f64)>>, Vec<f64>) = seqs
        .par_iter()
        .with_min_len(128)
        .map(|seq| {
            let mut v: Vec<(u32, f64)> = Vec::new();
            for &id in seq {
                let w = line_weight[id as usize];
                if let Some(slot) = v.iter_mut().find(|(k, _)| *k == id) {
                    slot.1 += w;
                } else {
                    v.push((id, w));
                }
            }
            let norm = v.iter().map(|&(_, p)| p * p).sum::<f64>().sqrt();
            v.sort_unstable_by_key(|&(id, _)| id);
            (v, norm)
        })
        .unzip();

    // Exact all-pairs weighted-cosine join (replaces shingle candidate-gen + per-pair verify), on the
    // selected backend — CPU, or difflib-fast's Metal GPU hybrid when `settings:gpu=on`.
    let corpus = Corpus::from_rows(&rows);
    let pairs = cosine_join_with(&corpus, theta, concurrency); // (j, i, cos), j < i, cos ≥ theta

    // Edges: keep cross-name, non-twin, non-byte-identical pairs strictly above θ (other passes own
    // same-name / sync-async / byte-identical clones).
    let edges: Vec<(usize, usize)> = pairs
        .into_par_iter()
        .filter(|&(j, i, cos)| {
            cos > theta
                && names[i] != names[j]
                && !is_sync_async(&names[i], &names[j])
                && seqs[i] != seqs[j]
        })
        .map(|(j, i, _)| (j, i))
        .collect();

    // Components → exact min cosine over ALL intra-component pairs (single-linkage's conservative
    // figure; can be < θ for a chain A–B–C where A,C aren't directly joined).
    components(n, &edges)
        .into_par_iter()
        .map(|mut members| {
            members.sort_unstable();
            let mut min_sim = theta;
            let mut first = true;
            for a in 0..members.len() {
                for b in (a + 1)..members.len() {
                    let c = cosine(&rows[members[a]], &rows[members[b]], norms[members[a]], norms[members[b]]);
                    if first || c < min_sim {
                        min_sim = c;
                        first = false;
                    }
                }
            }
            (members, min_sim)
        })
        .collect()
}
