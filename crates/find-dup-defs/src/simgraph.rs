//! Shared similarity-graph primitives used by the cosine-join passes (`type3`, `patternology`):
//! union-find connected components over an edge list. These were byte-for-byte duplicated across
//! the passes (the tool flagged its own copies as `DUPLICATE FUNCTION [ERROR]`) — they live here
//! once. The sorted-merge cosine used to live here too; the exact join now carries every edge's
//! score, so the per-pair recomputation it served is gone with it.
#![allow(clippy::cast_precision_loss)]

use rustc_hash::FxHashMap;

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
