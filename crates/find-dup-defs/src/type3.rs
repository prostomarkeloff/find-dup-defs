//! Native compute core for the dup-defs **Type-3** (ECScan) pass — the O(n²) part that hung in
//! Python. Given each function's name-agnostic normalized lines (produced Python-side by
//! `normalize.normalized_lines`), this does the whole numeric pipeline natively:
//!   IDF over line tokens → per-function line-weight vectors → rare-shingle candidate generation →
//!   IDF-weighted cosine verify → union-find → exact min-cosine per cluster.
//! It mirrors `type3.py` bit-for-bit (token regex, IDF, sorted-shared dot, insertion-order norm) so
//! `min_sim` — which drives ERROR/WARNING severity — is identical. The cross-file *policy* (≥2 distinct
//! names AND files, severity, `DupGroup` building) stays in Python; this returns raw clusters.
//!
//! Perf shape (all results-invariant — clusters + `min_sim` are bit-identical regardless): the setup
//! (seqs interning, `line_weight`, `FnVec` build) is rayon-parallel; the internal maps use `FxHash`
//! not SipHash; each `FnVec` is a sorted `Vec` not a per-function `HashMap`; the whole `_native` crate
//! runs on mimalloc. Profile it with `src/bin/t3prof.rs` + `benchmarks/samply_selftime.py`.
#![allow(clippy::doc_markdown)]

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Tokenize one line like Python's `re.compile(r"[A-Za-z_]\w*|\d+|\S")`: at each position take an
/// identifier (`[A-Za-z_][A-Za-z0-9_]*`), else a run of digits, else a single non-whitespace char;
/// whitespace is skipped. Left-to-right, non-overlapping — same matches `findall` yields.
fn tokenize(line: &str) -> Vec<&str> {
    let bytes = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let n = bytes.len();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    while i < n {
        let b = bytes[i];
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            i += 1;
            while i < n && is_word(bytes[i]) {
                i += 1;
            }
            out.push(&line[start..i]);
        } else if b.is_ascii_digit() {
            let start = i;
            i += 1;
            while i < n && bytes[i].is_ascii_digit() {
                i += 1;
            }
            out.push(&line[start..i]);
        } else if b.is_ascii_whitespace() {
            i += 1;
        } else {
            // a single non-whitespace, non-word char. Advance by one UTF-8 char so multi-byte
            // (non-ASCII) symbols are taken whole, matching `\S` (one code point).
            let ch_len = line[i..].chars().next().map_or(1, char::len_utf8);
            out.push(&line[i..i + ch_len]);
            i += ch_len;
        }
    }
    out
}

/// One function reduced to what the cosine needs: its distinct lines as `(line id, aggregated IDF
/// weight)` **sorted by id**, plus the precomputed norm. A sorted vector (not a `HashMap`) so the dot
/// is a linear merge and construction allocates one small Vec instead of a per-function hash table —
/// the per-function `HashMap<u32,f64>` was the profile's dominant cost (alloc + memmove churn). Bit-
/// exactness is preserved: weights still accumulate by repeated `+= w` in line order, and the norm
/// still sums in first-occurrence (insertion) order before the sort. (The raw line sequence for
/// shingles / byte-identical comparison lives in the shared `seqs`.)
struct FnVec {
    sorted: Vec<(u32, f64)>, // (line id, aggregated weight), ascending by id — for the dot merge
    norm: f64,
}

/// `a == f"a{b}"` or `b == f"a{a}"` — the sync/async naming twin convention (owned by another pass).
fn is_sync_async(a: &str, b: &str) -> bool {
    (a.len() == b.len() + 1 && a.as_bytes().first() == Some(&b'a') && &a[1..] == b)
        || (b.len() == a.len() + 1 && b.as_bytes().first() == Some(&b'a') && &b[1..] == a)
}

/// IDF-weighted cosine over two functions' line vectors. Dot is summed over **sorted** shared keys
/// and each norm over **insertion order** — exactly as `type3.py` does, so the score is reproducible
/// and bit-identical (it can otherwise flip a cluster across the ERROR/WARNING threshold).
fn cosine(a: &FnVec, b: &FnVec) -> f64 {
    if a.norm == 0.0 || b.norm == 0.0 {
        return 0.0;
    }
    // Dot over shared line ids, summed in ascending-id order — same deterministic float accumulation
    // as the reference's sorted-shared-keys sum, now via a linear merge of the two sorted vectors.
    let (mut i, mut j) = (0usize, 0usize);
    let mut dot = 0.0;
    while i < a.sorted.len() && j < b.sorted.len() {
        let (ka, va) = a.sorted[i];
        let (kb, vb) = b.sorted[j];
        match ka.cmp(&kb) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot += va * vb;
                i += 1;
                j += 1;
            }
        }
    }
    dot / (a.norm * b.norm)
}

/// Union-find over edge endpoints → connected components (mirrors `type3._cluster`).
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
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
#[must_use]
pub fn type3_clusters(
    line_lists: &[Vec<String>],
    names: &[String],
    theta: f64,
    shingle_lines: usize,
    common_ratio: f64,
) -> Vec<(Vec<usize>, f64)> {
    let n = line_lists.len();
    if n < 2 {
        return Vec::new();
    }
    // Intern distinct lines → ids. Phase 1 (serial): assign each distinct line a stable id in
    // first-occurrence order — HashMap inserts only, no per-function allocation, same order as before
    // so ids are bit-identical. Phase 2 (parallel): build each function's id sequence by lookup. The
    // 15k per-function `Vec<u32>` allocations are independent, so rayon spreads them over the idle
    // workers (the profile showed this serial map was ~all on one thread).
    let mut line_id: FxHashMap<&str, u32> = FxHashMap::default();
    let mut id_text: Vec<&str> = Vec::new();
    for lines in line_lists {
        for line in lines {
            line_id.entry(line.as_str()).or_insert_with(|| {
                id_text.push(line.as_str());
                (id_text.len() - 1) as u32
            });
        }
    }
    // `with_min_len`: each rayon job takes a run of functions, not one — the per-item work (a few
    // HashMap lookups) is tiny, so without this the join/work-steal coordination dwarfs it.
    let seqs: Vec<Vec<u32>> = line_lists
        .par_iter()
        .with_min_len(128)
        .map(|lines| lines.iter().map(|line| line_id[line.as_str()]).collect())
        .collect();
    let total_lines: usize = seqs.iter().map(Vec::len).sum();

    // IDF: df[token] = #line-occurrences containing token (a token counts once per line via the set).
    let mut occ: Vec<u32> = vec![0; id_text.len()]; // occurrences of each distinct line
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
        toks.dedup(); // set(tokens(line)) — distinct tokens of this line
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
    // Parallel: each line's weight is independent and order is preserved, so this stays bit-exact.
    let line_weight: Vec<f64> = id_text
        .par_iter()
        .with_min_len(256)
        .map(|&t| tokenize(t).iter().map(|tok| idf.get(tok).copied().unwrap_or(0.0)).sum())
        .collect();

    // Per-function vector: distinct lines in first-occurrence order with weight accumulated by
    // repeated `+= w` (bit-identical to the old HashMap), then sorted by id for the dot merge.
    // Most functions have few distinct lines (avg ~14), so the linear `find` dedup beats a hash
    // table and its allocation — the whole point of this representation.
    let vecs: Vec<FnVec> = seqs
        .par_iter()
        .with_min_len(128)
        .map(|seq| {
            // Build directly into one Vec in first-occurrence order (linear `find` dedup), accumulating
            // weight by repeated `+= w` exactly as the reference HashMap did — one allocation per
            // function instead of three (the profile is allocator-bound, so transients matter).
            let mut sorted: Vec<(u32, f64)> = Vec::new();
            for &id in seq {
                let w = line_weight[id as usize];
                if let Some(slot) = sorted.iter_mut().find(|(k, _)| *k == id) {
                    slot.1 += w;
                } else {
                    sorted.push((id, w));
                }
            }
            // norm sums in first-occurrence order — `sorted` is still insertion-ordered here, so this
            // matches Python's `sum(v*v for v in dict.values())` bit-for-bit; THEN sort for the merge.
            let norm = sorted.iter().map(|&(_, p)| p * p).sum::<f64>().sqrt();
            sorted.sort_unstable_by_key(|&(id, _)| id);
            FnVec { sorted, norm }
        })
        .collect();

    // Candidate pairs: functions sharing a rare N-line shingle (drop shingles in > max_fns functions).
    let max_fns = (common_ratio * n as f64) as usize;
    let max_fns = max_fns.max(2);
    let mut shingle_map: FxHashMap<&[u32], Vec<usize>> = FxHashMap::default();
    for (fi, seq) in seqs.iter().enumerate() {
        if seq.len() >= shingle_lines {
            for w in seq.windows(shingle_lines) {
                shingle_map.entry(w).or_default().push(fi);
            }
        }
    }
    let mut cand: FxHashSet<(usize, usize)> = FxHashSet::default();
    for fns in shingle_map.values() {
        let mut uniq: Vec<usize> = fns.clone();
        uniq.sort_unstable();
        uniq.dedup();
        if uniq.len() >= 2 && uniq.len() <= max_fns {
            for a in 0..uniq.len() {
                for b in (a + 1)..uniq.len() {
                    cand.insert((uniq[a], uniq[b]));
                }
            }
        }
    }

    // Verify candidates with cosine (parallel); keep edges with cos > theta, skipping pairs other
    // passes own: same name, sync/async twins, byte-identical line sequences.
    let cand: Vec<(usize, usize)> = cand.into_iter().collect();
    let edges: Vec<(usize, usize)> = cand
        .par_iter()
        .filter(|&&(i, j)| {
            names[i] != names[j]
                && !is_sync_async(&names[i], &names[j])
                && seqs[i] != seqs[j]
                && cosine(&vecs[i], &vecs[j]) > theta
        })
        .copied()
        .collect();

    // Components → exact min cosine over ALL intra-component pairs (single-linkage's conservative figure).
    components(n, &edges)
        .into_par_iter()
        .map(|mut members| {
            members.sort_unstable();
            let mut min_sim = theta;
            let mut first = true;
            for a in 0..members.len() {
                for b in (a + 1)..members.len() {
                    let c = cosine(&vecs[members[a]], &vecs[members[b]]);
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
