//! The **bipartite configuration model** (`BiCM`) and the tests that validate a projection of it.
//!
//! The decisions pass projects a bipartite graph — definitions on one side, the facts they test on
//! the other — onto definition pairs, and must say which pairs share more than chance explains. That
//! question has a literature, and this module follows it rather than inventing a statistic:
//!
//! - **The null** is the `BiCM` (Saracco, Di Clemente, Gabrielli, Squartini 2015): the maximum-entropy
//!   ensemble of bipartite graphs whose *expected* degrees equal the observed ones on both sides. An
//!   edge between a row of fitness `x` and a column of fitness `y` exists independently with
//!   probability `xy / (1 + xy)`, the fitnesses solving the degree equations. Rows with one degree
//!   share one fitness, as do columns — so the model is fitted over degree *classes*, a few hundred
//!   unknowns however large the corpus.
//! - **The count test** is the one projections are validated with in that literature (Saracco et al.
//!   2017, "Inferring monopartite projections of bipartite networks"; Neal 2014, the stochastic degree
//!   sequence model): the number of columns two rows share is a sum of independent Bernoullis, a
//!   Poisson-binomial, and its upper tail is the pair's p-value — computed here exactly.
//! - **The surprise test** keeps the same null and changes only the statistic: the improbability of
//!   the shared *set*, `Σ −ln q` over the shared columns, instead of their number. The count gives a
//!   column every pair is likely to share the same vote as one almost none share; the surprise weighs
//!   each by how unlikely sharing it was, which is the likelihood of the observation under the null.
//!   Its law is a weighted Poisson-binomial with no closed tail; the tail is taken by the
//!   Lugannani–Rice saddlepoint formula (1980), checked against exact enumeration in the tests.
//! - **Multiplicity** is Benjamini–Hochberg (1995) false-discovery control over every pair of rows.
//!   It assumes the p-values independent or positively dependent; pairs sharing a row are neither
//!   proven to be, which is the caveat this inherits from the literature it follows.

use std::collections::HashMap;

/// Iterations the fixed point is allowed before its residual is reported as it stands. A solver
/// bound, not a model parameter: the degree equations either converge well inside it or the fitted
/// model says how far off it is.
const MAX_ITERATIONS: usize = 100_000;

/// Relative degree residual at which the fit is taken as solved.
const TOLERANCE: f64 = 1e-10;

/// The fitted model over degree classes.
pub(crate) struct Bicm {
    /// Distinct row degrees, ascending.
    rows: Vec<usize>,
    x: Vec<f64>,
    /// Distinct column degrees, ascending, and how many columns have each.
    cols: Vec<u32>,
    col_count: Vec<usize>,
    y: Vec<f64>,
    /// A row joined to every column, or a column joined to every row, has its edges for certain and
    /// no finite fitness; it is held at probability one and kept out of the fixed point.
    row_full: Vec<bool>,
    col_full: Vec<bool>,
    /// The largest relative gap between an expected and an observed degree after fitting.
    pub(crate) residual: f64,
}

fn classes<T: Copy + Ord + std::hash::Hash>(values: impl Iterator<Item = T>) -> (Vec<T>, Vec<usize>) {
    let mut counts: HashMap<T, usize> = HashMap::new();
    for v in values {
        *counts.entry(v).or_insert(0) += 1;
    }
    let mut keys: Vec<T> = counts.keys().copied().collect();
    keys.sort_unstable();
    let mult = keys.iter().map(|k| counts[k]).collect();
    (keys, mult)
}

#[allow(clippy::cast_precision_loss)] // degrees and class sizes, far below 2^52
impl Bicm {
    /// Fit the model to observed degrees. `row_degrees` holds one entry per row, `col_degrees` one
    /// per column.
    pub(crate) fn fit(row_degrees: &[usize], col_degrees: &[u32]) -> Self {
        let (rows, row_mult) = classes(row_degrees.iter().copied());
        let (cols, col_count) = classes(col_degrees.iter().copied());
        let n_rows = row_degrees.len();
        let n_cols = col_degrees.len();
        let edges: f64 = row_degrees.iter().sum::<usize>() as f64;
        let row_full: Vec<bool> = rows.iter().map(|&k| k >= n_cols).collect();
        let col_full: Vec<bool> = cols.iter().map(|&d| d as usize >= n_rows).collect();
        let scale = edges.sqrt().max(1.0);
        let mut x: Vec<f64> = rows.iter().map(|&k| k as f64 / scale).collect();
        let mut y: Vec<f64> = cols.iter().map(|&d| f64::from(d) / scale).collect();
        // What the certain edges already supply, taken out of each class's target.
        let certain_cols: f64 = cols.iter().zip(&col_count).zip(&col_full).filter(|(_, f)| **f).map(|((_, n), _)| *n as f64).sum();
        let certain_rows: f64 = rows.iter().zip(&row_mult).zip(&row_full).filter(|(_, f)| **f).map(|((_, m), _)| *m as f64).sum();
        let mut residual = f64::INFINITY;
        let edge = |x: &[f64], y: &[f64], r: usize, c: usize| {
            if row_full[r] || col_full[c] {
                1.0
            } else {
                x[r] * y[c] / (1.0 + x[r] * y[c])
            }
        };
        for _ in 0..MAX_ITERATIONS {
            for r in 0..rows.len() {
                if row_full[r] {
                    continue;
                }
                let target = rows[r] as f64 - certain_cols;
                let denom: f64 = (0..cols.len())
                    .filter(|&c| !col_full[c])
                    .map(|c| col_count[c] as f64 * y[c] / (1.0 + x[r] * y[c]))
                    .sum();
                x[r] = if target > 0.0 && denom > 0.0 { target / denom } else { 0.0 };
            }
            for c in 0..cols.len() {
                if col_full[c] {
                    continue;
                }
                let target = f64::from(cols[c]) - certain_rows;
                let denom: f64 = (0..rows.len())
                    .filter(|&r| !row_full[r])
                    .map(|r| row_mult[r] as f64 * x[r] / (1.0 + x[r] * y[c]))
                    .sum();
                y[c] = if target > 0.0 && denom > 0.0 { target / denom } else { 0.0 };
            }
            // The largest relative gap between an expected and an observed degree, on either side.
            let mut worst: f64 = 0.0;
            for (r, &k) in rows.iter().enumerate() {
                let expected: f64 = (0..cols.len()).map(|c| col_count[c] as f64 * edge(&x, &y, r, c)).sum();
                worst = worst.max((expected - k as f64).abs() / k.max(1) as f64);
            }
            for (c, &d) in cols.iter().enumerate() {
                let expected: f64 = (0..rows.len()).map(|r| row_mult[r] as f64 * edge(&x, &y, r, c)).sum();
                worst = worst.max((expected - f64::from(d)).abs() / f64::from(d.max(1)));
            }
            residual = worst;
            if residual < TOLERANCE {
                break;
            }
        }
        Self { rows, x, cols, col_count, y, row_full, col_full, residual }
    }

    /// The class of a row of degree `k`.
    pub(crate) fn row_class(&self, k: usize) -> usize {
        self.rows.binary_search(&k).unwrap_or_else(|at| at.min(self.rows.len().saturating_sub(1)))
    }

    /// The class of a column of degree `d`.
    pub(crate) fn col_class(&self, d: u32) -> usize {
        self.cols.binary_search(&d).unwrap_or_else(|at| at.min(self.cols.len().saturating_sub(1)))
    }

    /// How many row classes there are.
    pub(crate) fn row_classes(&self) -> usize {
        self.rows.len()
    }

    /// The degree a row class stands for.
    pub(crate) fn row_degree(&self, r: usize) -> usize {
        self.rows[r]
    }

    /// Probability of an edge between a row of class `r` and a column of class `c`.
    pub(crate) fn edge(&self, r: usize, c: usize) -> f64 {
        if self.row_full[r] || self.col_full[c] {
            return 1.0;
        }
        let xy = self.x[r] * self.y[c];
        xy / (1.0 + xy)
    }

    /// For a pair of rows of classes `a` and `b`: each column class as (how many columns, the
    /// probability both rows hold one of them).
    pub(crate) fn pair_classes(&self, a: usize, b: usize) -> Vec<(usize, f64)> {
        (0..self.cols.len()).map(|c| (self.col_count[c], self.edge(a, c) * self.edge(b, c))).collect()
    }
}

// ---------------------------------------------------------------------------
// The count test: an exact Poisson-binomial upper tail
// ---------------------------------------------------------------------------

/// `pmf[s] = P(B = s)` for `s = 0..=upto` and `upper[t] = P(B ≥ t)` for `t = 0..=upto`, `B ~
/// Binomial(n, q)`. Logs throughout, and the tail summed from its own terms whenever subtracting from
/// one would cancel — so a tail of `1e-60` comes out as one, not as zero.
#[allow(clippy::cast_precision_loss)]
fn binomial_head(n: usize, q: f64, upto: usize) -> (Vec<f64>, Vec<f64>) {
    let mut pmf = vec![0.0; upto + 1];
    let mut upper = vec![0.0; upto + 2];
    if q >= 1.0 {
        if n <= upto {
            pmf[n] = 1.0;
        }
        for (t, u) in upper.iter_mut().enumerate().take(upto + 1) {
            *u = if t <= n { 1.0 } else { 0.0 };
        }
        upper.truncate(upto + 1);
        return (pmf, upper);
    }
    let (ln_q, ln_not) = (q.ln(), (-q).ln_1p());
    // ln P(B = s + 1) from ln P(B = s).
    let ln_term = |s: usize, previous: f64| previous + ((n - s) as f64).ln() - ((s + 1) as f64).ln() + ln_q - ln_not;
    let top = upto.min(n);
    let mut ln_pmf = vec![f64::NEG_INFINITY; top + 1];
    ln_pmf[0] = n as f64 * ln_not;
    for s in 0..top {
        ln_pmf[s + 1] = ln_term(s, ln_pmf[s]);
    }
    for (p, ln_p) in pmf.iter_mut().zip(&ln_pmf) {
        *p = ln_p.exp();
    }
    if upto <= n {
        let below: f64 = pmf[..upto].iter().sum();
        upper[upto] = if below <= 0.5 {
            1.0 - below
        } else {
            // Past the median the terms only shrink: sum them until they stop mattering, in logs
            // from where the head left off, so a tail below the smallest double is not a zero.
            let mut sum = pmf[upto];
            let mut ln_s = if upto < n { ln_term(upto, ln_pmf[upto]) } else { f64::NEG_INFINITY };
            let mode = ((n + 1) as f64 * q).floor();
            let mut s = upto + 1;
            while s <= n {
                let term = ln_s.exp();
                sum += term;
                if term < sum * 1e-18 && s as f64 > mode {
                    break;
                }
                if s < n {
                    ln_s = ln_term(s, ln_s);
                }
                s += 1;
            }
            sum
        };
    }
    for t in (0..upto).rev() {
        upper[t] = (upper[t + 1] + pmf[t]).min(1.0);
    }
    upper.truncate(upto + 1);
    (pmf, upper)
}

/// `P(V ≥ t)` for `t = 0..=upto`, where `V` is the number of columns two rows share: a sum over
/// column classes of `Binomial(n_c, q_c)`, independent. Exact up to rounding.
///
/// Carried as a survival function rather than a mass function so every step adds positive terms:
/// `P(V + B ≥ t) = Σ_{s<t} P(B = s)·P(V ≥ t − s) + P(B ≥ t)`.
pub(crate) fn count_survival(classes: &[(usize, f64)], upto: usize) -> Vec<f64> {
    let mut survival = vec![0.0; upto + 1];
    survival[0] = 1.0;
    for &(n, q) in classes {
        if n == 0 || q <= 0.0 {
            continue;
        }
        let (pmf, upper) = binomial_head(n, q, upto);
        let mut next = vec![0.0; upto + 1];
        next[0] = 1.0;
        for t in 1..=upto {
            let mut acc = upper[t];
            for s in 0..t {
                acc += pmf[s] * survival[t - s];
            }
            next[t] = acc.min(1.0);
        }
        survival = next;
    }
    survival
}

/// The expected number of shared columns, `Σ n_c q_c`.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn expected_count(classes: &[(usize, f64)]) -> f64 {
    classes.iter().map(|&(n, q)| n as f64 * q).sum()
}

// ---------------------------------------------------------------------------
// The surprise test: a saddlepoint tail
// ---------------------------------------------------------------------------

/// `ln erfc(x)` for `x ≥ 0`: the Chebyshev fit of Numerical Recipes (`erfcc`, fractional error below
/// 1.2e-7 everywhere), which is `t·exp(·)` in form — so its logarithm never underflows, however far
/// into the tail.
fn ln_erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    t.ln() + poly
}

/// `ln P(Z ≥ r)` for a standard normal `Z` and `r ≥ 0`.
fn ln_normal_tail(r: f64) -> f64 {
    std::f64::consts::LN_2.mul_add(-1.0, ln_erfc(r / std::f64::consts::SQRT_2))
}

/// The law of `W = Σ_columns w·[shared]`, `w = −ln q`: how improbable the shared set is under the
/// null, as a random variable of the null itself.
pub(crate) struct Surprise {
    /// `(n, q, ln q)` per column class with `0 < q < 1` — a certain column carries no surprise.
    terms: Vec<(f64, f64, f64)>,
}

#[allow(clippy::cast_precision_loss)]
impl Surprise {
    pub(crate) fn new(classes: &[(usize, f64)]) -> Self {
        let terms = classes
            .iter()
            .filter(|&&(n, q)| n > 0 && q > 0.0 && q < 1.0)
            .map(|&(n, q)| (n as f64, q, q.ln()))
            .collect();
        Self { terms }
    }

    /// The cumulant generating function and its first two derivatives at `theta`.
    fn cgf(&self, theta: f64) -> (f64, f64, f64) {
        let (mut k0, mut k1, mut k2) = (0.0, 0.0, 0.0);
        for &(n, q, ln_q) in &self.terms {
            // q·e^{θw} with w = −ln q is q^{1−θ}.
            let t = ((1.0 - theta) * ln_q).exp();
            let base = 1.0 - q;
            let denom = base + t;
            k0 += n * (t - q).ln_1p();
            k1 += n * (-ln_q) * t / denom;
            k2 += n * ln_q * ln_q * t * base / (denom * denom);
        }
        (k0, k1, k2)
    }

    /// The mean of `W`.
    pub(crate) fn mean(&self) -> f64 {
        self.cgf(0.0).1
    }

    /// The largest value `W` can take: every column shared.
    fn max(&self) -> f64 {
        self.terms.iter().map(|&(n, _, ln_q)| -n * ln_q).sum()
    }

    /// The Lugannani–Rice tail at the saddlepoint `theta`, together with the `w` it is the tail of.
    fn tail_at(&self, theta: f64) -> (f64, f64) {
        let (k0, k1, k2) = self.cgf(theta);
        let r = (2.0 * theta.mul_add(k1, -k0)).max(0.0).sqrt();
        let u = theta * k2.sqrt();
        let ln_phi = -0.5 * r * r - 0.5 * (2.0 * std::f64::consts::PI).ln();
        let ln_q = ln_normal_tail(r);
        let mills = (ln_q - ln_phi).exp();
        let correction = mills + 1.0 / u - 1.0 / r;
        let p = if correction > 0.0 && correction.is_finite() { (ln_phi + correction.ln()).exp() } else { ln_q.exp() };
        (p.clamp(0.0, 1.0), k1)
    }

    /// The saddlepoint `θ` with `K'(θ) = w`, for `w` above the mean.
    fn saddle(&self, w: f64) -> f64 {
        let (mut lo, mut hi) = (0.0, 1.0);
        while self.cgf(hi).1 < w {
            lo = hi;
            hi *= 2.0;
            if hi > 1e6 {
                break;
            }
        }
        let mut theta = lo.midpoint(hi);
        for _ in 0..200 {
            let (_, k1, k2) = self.cgf(theta);
            // Converged is checked before stepping: at the root a Newton step lands on the bracket
            // edge, and the bisection fallback would carry the answer away from it.
            if (k1 - w).abs() < 1e-12 * w.max(1.0) || (hi - lo) < 1e-14 * hi.max(1.0) {
                break;
            }
            if k1 < w {
                lo = theta;
            } else {
                hi = theta;
            }
            let newton = theta - (k1 - w) / k2;
            theta = if newton > lo && newton < hi && k2 > 0.0 { newton } else { lo.midpoint(hi) };
        }
        theta
    }

    /// `P(W ≥ w)`. One-sided: at or below the mean there is no evidence to weigh, and the answer
    /// is one.
    pub(crate) fn tail(&self, w: f64) -> f64 {
        if self.terms.is_empty() || w <= self.mean() {
            return 1.0;
        }
        if w >= self.max() {
            // Every column shared: the probability of that one outcome.
            return self.terms.iter().map(|&(n, _, ln_q)| n * ln_q).sum::<f64>().exp();
        }
        self.tail_at(self.saddle(w)).0
    }

    /// The smallest `w` whose tail is at most `alpha` — the critical value, searched along the
    /// saddlepoint `θ` itself, where the tail is monotone and each step costs one pass over the
    /// classes.
    pub(crate) fn critical(&self, alpha: f64) -> f64 {
        if self.terms.is_empty() {
            return f64::INFINITY;
        }
        let (mut lo, mut hi) = (0.0, 1.0);
        while self.tail_at(hi).0 > alpha {
            lo = hi;
            hi *= 2.0;
            if hi > 1e6 {
                return self.max();
            }
        }
        for _ in 0..100 {
            let mid = lo.midpoint(hi);
            if self.tail_at(mid).0 > alpha {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        self.tail_at(hi).1
    }
}

// ---------------------------------------------------------------------------
// Multiplicity
// ---------------------------------------------------------------------------

/// The Benjamini–Hochberg rejection threshold at false-discovery rate `alpha`, over `tests`
/// hypotheses of which `p` are the ones that could be rejected at all (every other p-value is known
/// to exceed `alpha`, so it never enters the step-up). `None` when nothing is rejected.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn bh_threshold(p: &[f64], tests: f64, alpha: f64) -> Option<f64> {
    let mut sorted: Vec<f64> = p.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut threshold = None;
    for (i, &value) in sorted.iter().enumerate() {
        if value <= (i + 1) as f64 * alpha / tests {
            threshold = Some(value);
        }
    }
    threshold
}

#[cfg(test)]
mod tests {
    use super::{bh_threshold, binomial_head, count_survival, expected_count, ln_erfc, Bicm, Surprise};

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs())
    }

    #[test]
    fn the_fit_reproduces_both_degree_sequences() {
        let rows = [1, 1, 2, 3, 3, 5, 8, 2, 4, 1];
        let cols = [1, 4, 2, 2, 7, 3, 1, 5, 2, 3];
        let model = Bicm::fit(&rows, &cols);
        assert!(model.residual < 1e-8, "residual {}", model.residual);
    }

    #[test]
    fn a_binomial_tail_far_below_rounding_is_still_a_number() {
        // P(B ≥ 3), B ~ Bin(1000, 1e-6) — the reference is exact rational arithmetic.
        let (_, upper) = binomial_head(1000, 1e-6, 3);
        assert!(close(upper[3], 1.660_427_981_138_186_8e-10, 1e-12), "{}", upper[3]);
    }

    #[test]
    fn the_count_survival_matches_a_bernoulli_by_bernoulli_convolution() {
        let classes = [(3, 0.2), (2, 0.05), (4, 0.6)];
        let mut pmf = vec![1.0];
        for &(n, q) in &classes {
            for _ in 0..n {
                let mut next = vec![0.0; pmf.len() + 1];
                for (k, p) in pmf.iter().enumerate() {
                    next[k] += p * (1.0 - q);
                    next[k + 1] += p * q;
                }
                pmf = next;
            }
        }
        let survival = count_survival(&classes, 9);
        for t in 0..=9 {
            let exact: f64 = pmf[t..].iter().sum();
            assert!(close(survival[t], exact, 1e-12), "t={t}: {} vs {exact}", survival[t]);
        }
        assert!(close(expected_count(&classes), 3.0 * 0.2 + 2.0 * 0.05 + 4.0 * 0.6, 1e-12));
    }

    #[test]
    fn erfc_matches_reference_values() {
        for (x, value) in [(1.0, 0.157_299_207_050_285_1), (3.0, 2.209_049_699_858_544e-5), (5.0, 1.537_459_794_428_035e-12)] {
            assert!(close(ln_erfc(x).exp(), value, 2e-7), "erfc({x})");
        }
    }

    /// The exact law of `W` by enumerating every count vector — feasible for small classes.
    fn exact_surprise_tail(classes: &[(usize, f64)], w: f64) -> f64 {
        let mut states: Vec<(f64, f64)> = vec![(0.0, 1.0)];
        for &(n, q) in classes {
            let weight = -q.ln();
            let mut binom = vec![0.0; n + 1];
            for (k, b) in binom.iter_mut().enumerate() {
                let mut c = 1.0;
                for i in 0..k {
                    c = c * f64::from(u32::try_from(n - i).unwrap_or(0)) / f64::from(u32::try_from(i + 1).unwrap_or(1));
                }
                *b = c * q.powi(i32::try_from(k).unwrap_or(0)) * (1.0 - q).powi(i32::try_from(n - k).unwrap_or(0));
            }
            let mut next = Vec::with_capacity(states.len() * (n + 1));
            for &(s, p) in &states {
                for (k, b) in binom.iter().enumerate() {
                    next.push((s + f64::from(u32::try_from(k).unwrap_or(0)) * weight, p * b));
                }
            }
            states = next;
        }
        states.iter().filter(|(s, _)| *s >= w - 1e-12).map(|(_, p)| p).sum()
    }

    #[test]
    fn the_saddlepoint_tail_tracks_the_exact_law() {
        let classes = [(40, 0.01), (30, 0.03), (20, 0.1)];
        let law = Surprise::new(&classes);
        // Tails between 1e-2 and 1e-9, at the values W can actually take.
        let (w1, w2, w3) = (-(0.01_f64).ln(), -(0.03_f64).ln(), -(0.1_f64).ln());
        let mut worst: f64 = 0.0;
        for (a, b, c) in [(2, 1, 3), (3, 2, 4), (4, 3, 5), (5, 4, 6), (6, 5, 7)] {
            let w = f64::from(a) * w1 + f64::from(b) * w2 + f64::from(c) * w3;
            let exact = exact_surprise_tail(&classes, w);
            let approx = law.tail(w);
            worst = worst.max((approx / exact).ln().abs());
        }
        // Measured: the formula sits 14–21% under the exact law from 5e-2 down to 1e-10 — a
        // continuous approximation of a discrete law, accurate to the factor that ranking and a
        // false-discovery cut need, not to three digits.
        assert!(worst < 0.3, "worst log-ratio {worst}");
    }

    #[test]
    fn the_critical_value_is_where_the_tail_reaches_alpha() {
        let law = Surprise::new(&[(500, 0.001), (50, 0.02)]);
        let w = law.critical(1e-6);
        assert!(close(law.tail(w), 1e-6, 1e-3), "tail at critical {}", law.tail(w));
    }

    #[test]
    fn benjamini_hochberg_steps_up() {
        // The textbook shape: the largest p(i) under i·α/m decides, even past a larger one before it.
        let p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216];
        assert_eq!(bh_threshold(&p, 10.0, 0.05), Some(0.008));
        assert_eq!(bh_threshold(&[0.9], 10.0, 0.05), None);
    }
}
