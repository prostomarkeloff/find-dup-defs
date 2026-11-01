//! Shared sparse-vector + similarity-graph primitives used by every cosine-join pass (`type3`,
//! `patternology`): the sorted-merge cosine over precomputed norms, and union-find connected
//! components over an edge list. These were byte-for-byte duplicated across the passes (the tool
//! flagged its own copies as `DUPLICATE FUNCTION [ERROR]`) — they live here once.
#![allow(clippy::cast_precision_loss)]

use rustc_hash::FxHashMap;

/// Cosine of two sparse vectors given precomputed L2 norms: sorted-merge dot over shared component
/// ids / (na·nb). Both `a` and `b` must be sorted by id (the join and the per-cluster `min_sim`
/// both rely on this). Zero norm → 0.0.
#[must_use]
pub fn cosine(a: &[(u32, f64)], b: &[(u32, f64)], na: f64, nb: f64) -> f64 {
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

/// Path-halving find for the union-find over edge endpoints.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Union-find connected components over an edge list on `n` vertices. Only edge-touched vertices are
/// emitted (isolated vertices form no component), one `Vec` per component, in unspecified order.
#[must_use]
pub fn components(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
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

#[cfg(test)]
mod tests {
    use super::cosine;

    fn assert_cos(a: &[(u32, f64)], b: &[(u32, f64)], na: f64, nb: f64, expected: f64) {
        let got = cosine(a, b, na, nb);
        assert!(
            (got - expected).abs() < 1e-12,
            "cosine({a:?}, {b:?}, na={na}, nb={nb}) = {got}, expected {expected}"
        );
    }

    #[test]
    fn zero_norms_short_circuit_to_zero() {
        // One zero norm returns 0.0 whatever the shared ids — the dot would divide by zero.
        assert_cos(&[(1, 1.0)], &[(1, 1.0)], 0.0, 1.0, 0.0);
        assert_cos(&[(1, 1.0)], &[(1, 1.0)], 1.0, 0.0, 0.0);
    }

    #[test]
    fn a_single_shared_id_is_its_product_over_the_norms() {
        // dot = 2·3 = 6, na·nb = 2·3 = 6 → exactly 1.0. The division and both multiplications
        // must be the ones above or the answer moves.
        assert_cos(&[(1, 2.0)], &[(1, 3.0)], 2.0, 3.0, 1.0);
    }

    #[test]
    fn the_merge_advances_both_cursors_only_on_shared_ids() {
        // ids 1 and 2 shared: dot = 2·3 + 4·5 = 26 over na·nb = 30.
        assert_cos(&[(1, 2.0), (2, 4.0)], &[(1, 3.0), (2, 5.0)], 5.0, 6.0, 26.0 / 30.0);
    }

    #[test]
    fn the_merge_skips_unmatched_ids_on_either_side() {
        // Only id 3 is shared (Less arm advances a, Greater arm advances b): dot = 4·6 = 24 over 20.
        assert_cos(&[(1, 2.0), (3, 4.0)], &[(2, 5.0), (3, 6.0)], 4.0, 5.0, 24.0 / 20.0);
    }

    #[test]
    fn a_repeated_shared_id_is_counted_once_per_match() {
        // b carries id 1 twice; the Equal arm must advance i (and only then j) so the pair is
        // counted once: dot = 2·3 = 6 over 2·5 = 10.
        assert_cos(&[(1, 2.0)], &[(1, 3.0), (1, 4.0)], 2.0, 5.0, 6.0 / 10.0);
        // Mirror: a carries id 1 twice — the Equal arm must advance j to stop re-matching.
        assert_cos(&[(1, 2.0), (1, 5.0)], &[(1, 3.0)], 5.0, 3.0, 6.0 / 15.0);
    }
}
