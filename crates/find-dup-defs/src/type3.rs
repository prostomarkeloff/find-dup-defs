//! **Type-3** (ECScan) cross-name near-copy detection — the O(n²) numeric core.
//!
//! Given each function's name-agnostic normalized lines (from `py-canon`) plus its name, this runs the
//! whole pipeline: IDF over line tokens → per-function line-weight vectors → rare-shingle candidate
//! generation → IDF-weighted cosine verify → union-find → exact min-cosine per cluster. It reproduces
//! the Python reference bit-for-bit — same token regex, IDF, sorted-shared dot, insertion-order norm,
//! and Neumaier-compensated summation — so `min_sim` (which drives ERROR/WARNING severity) is
//! identical. The cross-file policy (≥2 distinct names AND files) is applied by the caller.
//!
//! Perf shape (all results-invariant — clusters + `min_sim` are bit-identical regardless): the setup
//! (line interning, `line_weight`, `FnVec` build) is rayon-parallel; the internal maps use `FxHash`
//! not SipHash; each `FnVec` is a sorted `Vec` not a per-function `HashMap`; the crate runs on mimalloc.
#![allow(clippy::doc_markdown)]

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Tokenize one line like Python's `re.compile(r"[A-Za-z_]\w*|\d+|\S")` — Unicode-aware, matching
/// `findall`. An identifier starts on an ASCII `[A-Za-z_]` then greedily takes Python's `\w`
/// (Unicode word chars: `is_alphanumeric` or `_` — so an ASCII letter followed by Cyrillic/accented
/// letters in a string literal, e.g. `nВключи`, stays ONE token, exactly as Python's `\w*` does); a
/// run of ASCII digits is `\d+`; any other non-whitespace code point is a single `\S`; whitespace
/// (Unicode `\s`) is skipped. Left-to-right, non-overlapping.
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

/// Kahan–Babuška–**Neumaier** compensated summation — exactly what CPython's `sum()` does for floats
/// since 3.12. The Python reference sums every float total (line weights, dot, norm) with `sum()`, so
/// the score is only bit-identical if we compensate the same way; naive left-to-right summation drifts
/// by ~1 ULP and can flip a cluster across the ERROR/WARNING line.
fn neumaier_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for v in values {
        let t = sum + v;
        if sum.abs() >= v.abs() {
            c += (sum - t) + v;
        } else {
            c += (v - t) + sum;
        }
        sum = t;
    }
    sum + c
}

/// `a == f"a{b}"` or `b == f"a{a}"` — the sync/async naming twin convention (owned by another pass).
fn is_sync_async(a: &str, b: &str) -> bool {
    (a.len() == b.len() + 1 && a.as_bytes().first() == Some(&b'a') && &a[1..] == b)
        || (b.len() == a.len() + 1 && b.as_bytes().first() == Some(&b'a') && &b[1..] == a)
}

/// IDF-weighted cosine over two functions' line vectors. Dot is summed over **sorted** shared keys
/// and each norm over **insertion order**, both with Neumaier compensation — exactly as the Python
/// reference does, so the score is reproducible and bit-identical (it can otherwise flip a cluster
/// across the ERROR/WARNING threshold).
fn cosine(a: &FnVec, b: &FnVec) -> f64 {
    if a.norm == 0.0 || b.norm == 0.0 {
        return 0.0;
    }
    // Dot over shared line ids, summed in ascending-id (= line-text, since ids are assigned in
    // lexicographic order) order with Neumaier compensation — same keys, same order, same algorithm as
    // the reference's `sum(a[k]*b[k] for k in sorted(shared))`, so the dot is bit-identical.
    let (mut i, mut j) = (0usize, 0usize);
    let (mut sum, mut comp) = (0.0f64, 0.0f64);
    while i < a.sorted.len() && j < b.sorted.len() {
        let (ka, wa) = a.sorted[i];
        let (kb, wb) = b.sorted[j];
        match ka.cmp(&kb) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                let prod = wa * wb;
                let next = sum + prod;
                if sum.abs() >= prod.abs() {
                    comp += (sum - next) + prod;
                } else {
                    comp += (prod - next) + sum;
                }
                sum = next;
                i += 1;
                j += 1;
            }
        }
    }
    (sum + comp) / (a.norm * b.norm)
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
    // Intern distinct lines → ids, assigned in **lexicographic (byte) order of the line text**. For
    // valid UTF-8 byte order == code-point order == Python's `sorted()`, so when the cosine later sums
    // the dot by ascending id it sums over shared keys in the exact order Python's `sorted(shared)`
    // does — making the (order-sensitive) float dot bit-identical. (The norm is summed earlier, in
    // first-occurrence order, so it is unaffected by this id ordering.)
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
        .map(|&t| neumaier_sum(tokenize(t).iter().map(|tok| idf.get(tok).copied().unwrap_or(0.0))))
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
            // matches Python's `sqrt(sum(v*v for v in dict.values()))` bit-for-bit (Neumaier); THEN
            // sort for the merge.
            let norm = neumaier_sum(sorted.iter().map(|&(_, p)| p * p)).sqrt();
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
