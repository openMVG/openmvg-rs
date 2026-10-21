// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cycle reliability and the per-edge prior (steps 3 and 4 of the conditioner).
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                     Cycle reliability and prior                      │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: triangles, kernel scales, edge evidence
//!                                   │
//!                                   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: consistency_t = Cauchy(error_t / scale) per triangle         │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                   │
//!                                   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Repeat CYCLE_ROUNDS times:                                   │
//! │         w_t = min reliability of the triangle's edges                │
//! │         blame_e = sum of w_t (1 - consistency_t) over its triangles  │
//! │         reliability_e moves toward 1 - blame_e / total_e             │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                   │
//!                                   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: prior_e = reliability_e^g * evidence factor_e                │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                   │
//!                                   ▼
//! Output: reliability, evidence factor, prior per edge
//! ```
//!
//! ## IterBP blame propagation
//!
//! A triangle of three pair rotations should close to the identity. When it does not, at least one
//! of its three edges is wrong, but the closure error alone does not say which. IterBP resolves
//! that by iterating between two quantities: the reliability of each edge, and the blame each
//! triangle hands to its edges. A triangle's vote is weighted by its least reliable edge, so a
//! triangle containing a known-bad edge cannot condemn the two good edges it shares; edges
//! that keep appearing in inconsistent triangles lose reliability, and the rounds concentrate the
//! blame on them. Here the usual hard indicator "triangle is consistent" is replaced by a smooth
//! Cauchy kernel `1 / (1 + (error / scale)^2)`, so a slightly noisy edge is down-weighted rather
//! than removed.

use super::triangles::Triangle;
use super::{AdaptiveEdge, quantile, sorted_values};

/// Blame-propagation rounds.
const CYCLE_ROUNDS: usize = 10;
/// Exponential-moving-average step of the reliability update.
const CYCLE_ALPHA: f64 = 0.5;
/// Exponent of the cycle reliability in the prior.
const RELIABILITY_EXPONENT: f64 = 0.5;
/// Exponent of the evidence factor.
const EVIDENCE_EXPONENT: f64 = 0.5;
/// Smallest and largest value of the evidence factor.
const EVIDENCE_FACTOR_RANGE: (f64, f64) = (0.2, 5.0);
/// Blame totals below this carry no information and leave the reliability unchanged.
const MIN_BLAME_TOTAL: f64 = 1e-12;

fn cauchy(x: f64) -> f64 {
    1.0 / (1.0 + x * x)
}

/// IterBP blame propagation with a smooth consistency kernel in place of the hard indicator.
///
/// Each round, a triangle weighted by its least reliable edge adds `1 - consistency` blame to each
/// of its edges; an edge's reliability moves halfway toward one minus its weighted blame share.
/// Edges in no triangle keep a reliability of one. The translation-triplet residual judges a
/// triangle when `translation_scale` is known and the triangle has one, the rotation closure
/// otherwise.
pub(super) fn soft_cycle_reliability(
    edge_count: usize,
    triangles: &[Triangle],
    rotation_scale: f64,
    translation_scale: Option<f64>,
) -> Vec<f64> {
    let consistency: Vec<f64> = triangles
        .iter()
        .map(|t| match (translation_scale, t.translation_error_deg()) {
            (Some(scale), Some(error)) => cauchy(error / scale),
            _ => cauchy(t.rotation_error_deg() / rotation_scale),
        })
        .collect();
    let mut reliability = vec![1.0; edge_count];
    let mut blame = vec![0.0; edge_count];
    let mut total = vec![0.0; edge_count];
    for _ in 0..CYCLE_ROUNDS {
        blame.fill(0.0);
        total.fill(0.0);
        for (t, c) in triangles.iter().zip(&consistency) {
            let edges = t.edges();
            let w = edges
                .iter()
                .map(|&e| reliability[e])
                .fold(f64::INFINITY, f64::min);
            for e in edges {
                blame[e] += w * (1.0 - c);
                total[e] += w;
            }
        }
        for e in 0..edge_count {
            if total[e] > MIN_BLAME_TOTAL {
                reliability[e] = (1.0 - CYCLE_ALPHA) * reliability[e]
                    + CYCLE_ALPHA * (1.0 - blame[e] / total[e]);
            }
        }
    }
    reliability
}

/// `(evidence / median evidence)^0.5`, clamped to `[0.2, 5]`; evidence below one counts as one.
pub(super) fn evidence_factor(edges: &[AdaptiveEdge]) -> Vec<f64> {
    let sorted = sorted_values(edges.iter().map(|e| e.evidence.max(1.0)));
    let median = quantile(&sorted, 0.5).unwrap_or(1.0).max(1.0);
    let (lo, hi) = EVIDENCE_FACTOR_RANGE;
    edges
        .iter()
        .map(|e| {
            (e.evidence.max(1.0) / median)
                .powf(EVIDENCE_EXPONENT)
                .clamp(lo, hi)
        })
        .collect()
}

/// Per-edge prior of the graduated solve: `reliability^0.5 * evidence factor`.
pub(super) fn prior(reliability: &[f64], evidence: &[f64]) -> Vec<f64> {
    reliability
        .iter()
        .zip(evidence)
        .map(|(r, e)| r.powf(RELIABILITY_EXPONENT) * e)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{edge, truth};
    use super::super::triangles::enumerate_triangles;
    use super::*;

    fn clique(n: usize, bad: Option<(usize, usize)>) -> Vec<AdaptiveEdge> {
        let t = truth(n);
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let mut e = edge(&t, i, j, 0.0, 50.0);
                if bad == Some((i, j)) {
                    e.rotation =
                        super::super::test_support::rotation([0.0, 1.0, 0.0], 30.0) * e.rotation;
                }
                edges.push(e);
            }
        }
        edges
    }

    #[test]
    fn a_consistent_clique_keeps_full_reliability() {
        let edges = clique(7, None);
        let triangles = enumerate_triangles(&edges).unwrap();
        let reliability = soft_cycle_reliability(edges.len(), &triangles, 1.0, None);
        assert!(reliability.iter().all(|r| *r > 0.999), "{reliability:?}");
    }

    #[test]
    fn the_one_bad_edge_is_blamed_more_than_every_other() {
        let edges = clique(8, Some((2, 5)));
        let bad = edges
            .iter()
            .position(|e| (e.view_i.index(), e.view_j.index()) == (2, 5))
            .unwrap();
        let triangles = enumerate_triangles(&edges).unwrap();
        let reliability = soft_cycle_reliability(edges.len(), &triangles, 1.0, None);
        let worst_good = reliability
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != bad)
            .map(|(_, r)| *r)
            .fold(f64::INFINITY, f64::min);
        assert!(
            reliability[bad] < 0.5 * worst_good,
            "bad {} vs worst good {worst_good}",
            reliability[bad]
        );
    }

    #[test]
    fn edges_outside_every_triangle_keep_a_reliability_of_one() {
        let reliability = soft_cycle_reliability(3, &[], 1.0, None);
        assert_eq!(reliability, vec![1.0; 3]);
    }

    #[test]
    fn evidence_factor_is_one_at_the_median_and_clamped() {
        let t = truth(2);
        let mut edges = vec![edge(&t, 0, 1, 0.0, 100.0); 5];
        edges[0].evidence = 1.0e6;
        edges[1].evidence = 0.001;
        let factor = evidence_factor(&edges);
        assert_eq!(factor[0], EVIDENCE_FACTOR_RANGE.1);
        // Evidence below one counts as one: 1 / 100 under a square root is 0.1, floored at 0.2.
        assert_eq!(factor[1], EVIDENCE_FACTOR_RANGE.0);
        assert!((factor[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_prior_multiplies_the_reliability_root_by_the_evidence() {
        let p = prior(&[1.0, 0.25], &[2.0, 4.0]);
        assert!((p[0] - 2.0).abs() < 1e-12);
        assert!((p[1] - 2.0).abs() < 1e-12);
    }
}
