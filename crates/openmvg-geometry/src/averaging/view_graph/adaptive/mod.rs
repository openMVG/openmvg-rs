// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adaptive soft conditioning of a rotation graph: cycle reliability plus a graduated robust solve.
//!
//! Its teaching docs are on [`condition_edges_adaptive`], where docs.rs shows them.
//!
//! The evaluation below is contributor context; the public docs keep only the method.
//!
//! ## Evaluation
//!
//! Scored offline on 287 cached scenes (BlendedMVS, ETH3D, Strecha, MegaDepth) against the
//! ground-truth rotations: a scene passes when at least 90% of the cameras a valid core could
//! hold end under 5 degrees, their p90 is at most 6 degrees and their median at most 1.5 times the
//! oracle median plus 0.25 degrees. Pass rates: uniform weights 72.1%, a private re-implementation
//! of the former fixed 5 degree residual cut 87.1% (87.5% for the original), this method 93.4%
//! (scored on its confidence times inliers through one more solve), a ground-truth oracle 99.0%.
//! The paired gain over the original cut is 5.9 points (95% CI 2.8 to 9.1). Of the 18 failures
//! where the oracle passes, 14 prefer the wrong solution even when refined inside the
//! ground-truth basin, so only 3D evidence can arbitrate them. Tried and rejected: a-contrario thresholds, an EM scale mixture,
//! Tukey, per-edge graduation (CycleGrad), two-candidate synchronization (MixSync-lite), the
//! original IterBP thresholds and soft confidences for translation averaging. The harness, its
//! notes and the legacy conditioner live at the git tag `rotation-lab`.

mod graduated;
mod reliability;
mod triangles;

use std::collections::BTreeMap;

use nalgebra::{Rotation3, Vector3};
use openmvg_core::model::{EstimationResult, RelativeRotation, ViewId, rotation_angle_deg};
use thiserror::Error;

use crate::averaging::RotationEstimate;
use graduated::{Solver, effective_weight, solve_graduated};
use reliability::{evidence_factor, prior, soft_cycle_reliability};
use triangles::{Triangle, enumerate_triangles};

/// Smallest scale (degrees) the cycle kernels may use.
const MIN_KERNEL_SCALE_DEG: f64 = 0.3;
/// Floor of the closure-error quantile (degrees), so noise-free input cannot give a zero scale.
const MIN_QUANTILE_DEG: f64 = 0.05;
/// Fewer finite translation residuals than this and the translation scale is not estimated.
const MIN_TRANSLATION_SAMPLES: usize = 8;
/// Cycle-kernel scale as a multiple of the 25th percentile of the closure errors.
const CYCLE_SCALE_FACTOR: f64 = 2.0;

/// One verified pair as seen by the conditioner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveEdge {
    /// First view; must be smaller than `view_j`.
    pub view_i: ViewId,
    /// Second view.
    pub view_j: ViewId,
    /// Measured relative rotation with `R_j = rotation * R_i`.
    pub rotation: Rotation3<f64>,
    /// Positive evidence behind the pair, normally the inlier count.
    pub evidence: f64,
    /// Direction of the second camera's center in the first camera's frame, when known.
    /// Any positive norm; a zero or non-finite vector counts as unknown.
    pub baseline: Option<Vector3<f64>>,
}

/// Failure of [`condition_edges_adaptive`].
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum AdaptiveConditioningError {
    /// An input edge is malformed.
    #[error("invalid adaptive-conditioning edge: {0}")]
    InvalidEdge(String),
    /// The injected rotation solver failed.
    #[error("rotation solver failed during adaptive conditioning: {0}")]
    Solver(String),
}

/// Outcome of [`condition_edges_adaptive`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveConditioning {
    /// Final edge weights, evidence included, in input order, as the final rotation solve received
    /// them: floored at `1e-6`, so a suppressed edge never reads as exactly zero.
    pub weights: Vec<f64>,
    /// Weights with the evidence factor removed and the maximum scaled to one: the part of the
    /// weight a consumer that multiplies by the inlier count itself should apply.
    pub confidence: Vec<f64>,
    /// Rotations solved with [`Self::weights`].
    pub rotations: BTreeMap<ViewId, Rotation3<f64>>,
    /// Disagreement of each edge with [`Self::rotations`], degrees, in input order.
    pub residuals_deg: Vec<f64>,
    /// Scale of the last graduated solve, degrees.
    pub final_scale_deg: f64,
    /// Triangles seen by the cycle prior.
    pub triangle_count: usize,
    /// Rotation solves performed.
    pub solve_count: usize,
}

/// Condition `edges` and return per-edge weights and the rotations they produce.
///
/// `solve` is the rotation solver, called with the edges weighted by the current weights; it must
/// return one rotation per view that has an edge.
///
/// # How it works
///
/// The kernel scales are derived from the data within fixed bounds, so there is no per-dataset
/// threshold, and no edge is cut: the output is one weight per edge and the rotations those
/// weights produce.
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │                 Adaptive rotation-graph conditioning                 │
/// └──────────────────────────────────────────────────────────────────────┘
/// Input: edges (R_ij, inliers, optional baseline direction)
///                                   │
///                                   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 1: Triangles i<j<k: closure error R_ik^-1 R_jk R_ij (degrees),  │
/// │         translation-triplet residual when baselines exist            │
/// └──────────────────────────────────────────────────────────────────────┘
///                                   │
///                                   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 2: Scale = 2 x 25th percentile of each error distribution       │
/// └──────────────────────────────────────────────────────────────────────┘
///                                   │
///                                   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 3: Soft cycle reliability r_e: IterBP blame propagation with    │
/// │         a smooth Cauchy kernel over the triangles                    │
/// └──────────────────────────────────────────────────────────────────────┘
///                                   │
///                                   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 4: Prior_e = r_e^g * (inliers_e / median inliers)^b, clamped    │
/// └──────────────────────────────────────────────────────────────────────┘
///                                   │
///                                   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 5: Graduated solve: scale 30 deg -> 2.5 x p25 over k re-solves, │
/// │         w = prior * Cauchy(residual / scale)                         │
/// └──────────────────────────────────────────────────────────────────────┘
///                                   │
///                                   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 6: Final rounds: scale = 2.5 x median residual of trusted edges │
/// └──────────────────────────────────────────────────────────────────────┘
///                                   │
///                                   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 7: One more solve; confidence = weight / evidence factor        │
/// └──────────────────────────────────────────────────────────────────────┘
///                                   │
///                                   ▼
/// Output: weights, rotations, confidence
/// ```
///
/// IterBP, iterative blame propagation, finds the wrong edges of the triangles that do not close:
/// each such triangle hands blame to its three edges, weighted by its least reliable edge, and the
/// edge reliabilities and the blame are refined together over a few rounds, so a triangle that
/// holds a known-bad edge cannot condemn the two good edges it shares. Here a smooth kernel
/// replaces the hard closed-or-not test. The graduated solve is graduated non-convexity (Blake
/// and Zisserman) on the robust rotation-averaging objective: it starts with a wide kernel and
/// narrows it, and the cycle prior never sees the contaminated estimate.
///
/// # Errors
///
/// [`AdaptiveConditioningError::InvalidEdge`] for an edge with `view_i >= view_j`, a duplicate
/// pair, non-positive evidence or a non-finite rotation, and [`AdaptiveConditioningError::Solver`]
/// if `solve` fails.
pub fn condition_edges_adaptive(
    edges: &[AdaptiveEdge],
    solve: &dyn Fn(&[RelativeRotation]) -> EstimationResult<RotationEstimate>,
) -> Result<AdaptiveConditioning, AdaptiveConditioningError> {
    let triangles = enumerate_triangles(edges)?;
    let scales = Scales::from_triangles(&triangles);
    let reliability = soft_cycle_reliability(
        edges.len(),
        &triangles,
        scales.cycle_deg,
        scales.translation_deg,
    );
    let evidence = evidence_factor(edges);
    let prior = prior(&reliability, &evidence);

    let mut solver = Solver::new(edges, solve);
    let fit = solve_graduated(&mut solver, &reliability, &prior, scales.cycle_quantile)?;

    let mut confidence: Vec<f64> = fit
        .weights
        .iter()
        .zip(&evidence)
        .map(|(w, e)| w / e)
        .collect();
    let top = confidence.iter().copied().fold(0.0_f64, f64::max);
    if top > 0.0 {
        for c in &mut confidence {
            *c /= top;
        }
    }
    let residuals_deg = edges
        .iter()
        .map(|e| residual_deg(&fit.rotations, e))
        .collect();
    Ok(AdaptiveConditioning {
        weights: fit.weights.iter().map(|&w| effective_weight(w)).collect(),
        confidence,
        rotations: fit.rotations,
        residuals_deg,
        final_scale_deg: fit.scale_deg,
        triangle_count: triangles.len(),
        solve_count: solver.count(),
    })
}

/// Kernel scales derived from the closure-error distributions (step 2).
struct Scales {
    /// 25th percentile of the rotation closure errors, degrees, floored.
    cycle_quantile: f64,
    /// Cauchy scale of the rotation closure kernel, degrees.
    cycle_deg: f64,
    /// Cauchy scale of the translation-triplet kernel, degrees; `None` without enough residuals.
    translation_deg: Option<f64>,
}

impl Scales {
    fn from_triangles(triangles: &[Triangle]) -> Self {
        let rotation_errors = sorted_values(triangles.iter().map(Triangle::rotation_error_deg));
        let translation_errors =
            sorted_values(triangles.iter().filter_map(Triangle::translation_error_deg));
        let cycle_quantile = quantile(&rotation_errors, 0.25)
            .unwrap_or(2.0)
            .max(MIN_QUANTILE_DEG);
        let translation_quantile = (translation_errors.len() >= MIN_TRANSLATION_SAMPLES)
            .then(|| quantile(&translation_errors, 0.25))
            .flatten();
        let kernel = |q: f64| (CYCLE_SCALE_FACTOR * q).max(MIN_KERNEL_SCALE_DEG);
        Self {
            cycle_quantile,
            cycle_deg: kernel(cycle_quantile),
            translation_deg: translation_quantile.map(kernel),
        }
    }
}

/// Angle of the residual `(R_ij R_i)^-1 R_j` in degrees; infinite when a view has no rotation.
fn residual_deg(rotations: &BTreeMap<ViewId, Rotation3<f64>>, edge: &AdaptiveEdge) -> f64 {
    match (rotations.get(&edge.view_i), rotations.get(&edge.view_j)) {
        (Some(ri), Some(rj)) => rotation_angle_deg(&((edge.rotation * ri).inverse() * rj)),
        _ => f64::INFINITY,
    }
}

/// Finite values in ascending order.
fn sorted_values(values: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut v: Vec<f64> = values.filter(|x| x.is_finite()).collect();
    v.sort_by(f64::total_cmp);
    v
}

/// Value at `fraction` of a sorted slice (lower index), `None` when empty.
fn quantile(sorted: &[f64], fraction: f64) -> Option<f64> {
    let last = sorted.len().checked_sub(1)?;
    sorted.get((last as f64 * fraction) as usize).copied()
}

#[cfg(test)]
mod test_support {
    use super::*;
    use crate::averaging::{RotationConfig, average_rotations_dense};
    use nalgebra::Unit;

    pub(super) fn rotation(axis: [f64; 3], angle_deg: f64) -> Rotation3<f64> {
        Rotation3::from_axis_angle(
            &Unit::new_normalize(Vector3::new(axis[0], axis[1], axis[2])),
            angle_deg.to_radians(),
        )
    }

    /// Deterministic absolute rotations spread over the sphere.
    pub(super) fn truth(n: usize) -> Vec<Rotation3<f64>> {
        (0..n)
            .map(|i| {
                let a = i as f64;
                rotation(
                    [1.0 + a.sin(), 0.5 + a.cos(), 0.3 + (a * 0.7).sin()],
                    7.0 * a + 3.0,
                )
            })
            .collect()
    }

    /// The edge `(i, j)` of `truth`, bent by `noise_deg` about a fixed axis.
    pub(super) fn edge(
        truth: &[Rotation3<f64>],
        i: usize,
        j: usize,
        noise_deg: f64,
        evidence: f64,
    ) -> AdaptiveEdge {
        let exact = truth[j] * truth[i].inverse();
        let jitter = rotation([0.3, 0.9, -0.2], noise_deg);
        AdaptiveEdge {
            view_i: ViewId::new(i),
            view_j: ViewId::new(j),
            rotation: jitter * exact,
            evidence,
            baseline: None,
        }
    }

    pub(super) fn solver() -> impl Fn(&[RelativeRotation]) -> EstimationResult<RotationEstimate> {
        move |edges| average_rotations_dense(edges, &RotationConfig::default())
    }

    pub(super) fn max_error_deg(
        truth: &[Rotation3<f64>],
        estimate: &BTreeMap<ViewId, Rotation3<f64>>,
    ) -> f64 {
        // Right gauge: estimate_i = truth_i * H.
        let h = truth[0].inverse() * estimate[&ViewId::new(0)];
        truth
            .iter()
            .enumerate()
            .map(|(i, t)| rotation_angle_deg(&((*t * h).inverse() * estimate[&ViewId::new(i)])))
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{edge, max_error_deg, rotation, solver, truth};
    use super::*;

    #[test]
    fn reported_weights_are_the_weights_the_final_solve_received() {
        let n = 8;
        let t = truth(n);
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let mut e = edge(&t, i, j, 0.05, 100.0);
                if (i, j) == (2, 6) {
                    e.rotation = rotation([0.0, 1.0, 0.0], 120.0) * e.rotation;
                }
                edges.push(e);
            }
        }
        let inner = solver();
        let last = std::cell::RefCell::new(Vec::new());
        let spy = |relatives: &[RelativeRotation]| {
            *last.borrow_mut() = relatives.iter().map(RelativeRotation::weight).collect();
            inner(relatives)
        };
        let result = condition_edges_adaptive(&edges, &spy).unwrap();
        assert_eq!(result.weights, *last.borrow());
        assert!(result.weights.iter().all(|w| *w >= graduated::MIN_WEIGHT));
        let outlier = edges
            .iter()
            .position(|e| (e.view_i.index(), e.view_j.index()) == (2, 6))
            .unwrap();
        assert_eq!(result.weights[outlier], graduated::MIN_WEIGHT);
        assert!(result.confidence[outlier] < 1e-6);
    }

    #[test]
    fn scales_follow_the_closure_quantile_and_respect_the_floors() {
        let triangle = |error: f64| Triangle::new([0, 1, 2], error, None);
        let none = Scales::from_triangles(&[]);
        assert_eq!(none.cycle_quantile, 2.0);
        assert_eq!(none.cycle_deg, 4.0);
        assert!(none.translation_deg.is_none());
        let tiny = Scales::from_triangles(&[triangle(0.0), triangle(0.0)]);
        assert_eq!(tiny.cycle_quantile, MIN_QUANTILE_DEG);
        assert_eq!(tiny.cycle_deg, MIN_KERNEL_SCALE_DEG);
        let spread: Vec<Triangle> = (0..9).map(|k| triangle(f64::from(k))).collect();
        let s = Scales::from_triangles(&spread);
        assert_eq!(s.cycle_quantile, 2.0);
        assert_eq!(s.cycle_deg, 4.0);
    }

    #[test]
    fn clean_graph_is_recovered_with_uniform_confidence() {
        let n = 12;
        let t = truth(n);
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n.min(i + 4) {
                edges.push(edge(&t, i, j, 0.05 * ((i + j) % 3) as f64, 100.0));
            }
        }
        let result = condition_edges_adaptive(&edges, &solver()).unwrap();
        assert!(max_error_deg(&t, &result.rotations) < 0.5);
        assert!(
            result.confidence.iter().all(|c| *c > 0.3),
            "{:?}",
            result.confidence
        );
        assert!(result.triangle_count > 0);
    }

    #[test]
    fn planted_outliers_receive_far_lower_confidence() {
        let n = 16;
        let t = truth(n);
        let mut edges = Vec::new();
        let mut planted = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n.min(i + 5) {
                let bad = (i * 5 + j * 3) % 7 == 0;
                let mut e = edge(&t, i, j, 0.1, 100.0);
                if bad {
                    e.rotation = rotation([0.2, 1.0, 0.4], 40.0 + (i + j) as f64) * e.rotation;
                }
                planted.push(bad);
                edges.push(e);
            }
        }
        let result = condition_edges_adaptive(&edges, &solver()).unwrap();
        let mean = |want: bool| {
            let v: Vec<f64> = result
                .confidence
                .iter()
                .zip(&planted)
                .filter(|(_, p)| **p == want)
                .map(|(c, _)| *c)
                .collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        assert!(
            mean(true) < 0.25 * mean(false),
            "{} vs {}",
            mean(true),
            mean(false)
        );
        assert!(max_error_deg(&t, &result.rotations) < 1.0);
    }

    #[test]
    fn conflicting_bridges_are_decided_by_evidence() {
        // Two cliques tied by two good bridges (strong evidence) and two bad bridges (weak).
        let n = 10;
        let t = truth(n);
        let mut edges = Vec::new();
        for (lo, hi) in [(0, 5), (5, 10)] {
            for i in lo..hi {
                for j in (i + 1)..hi {
                    edges.push(edge(&t, i, j, 0.05, 100.0));
                }
            }
        }
        for (i, j, bad, evidence) in [
            (0, 5, false, 400.0),
            (2, 7, false, 300.0),
            (1, 6, true, 40.0),
            (3, 9, true, 30.0),
        ] {
            let mut e = edge(&t, i, j, 0.05, evidence);
            if bad {
                e.rotation = rotation([0.0, 1.0, 0.0], 25.0) * e.rotation;
            }
            edges.push(e);
        }
        let result = condition_edges_adaptive(&edges, &solver()).unwrap();
        assert!(max_error_deg(&t, &result.rotations) < 2.0);
    }

    #[test]
    fn result_is_deterministic_and_independent_of_edge_order() {
        let n = 12;
        let t = truth(n);
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n.min(i + 4) {
                edges.push(edge(
                    &t,
                    i,
                    j,
                    0.1 * ((i * j) % 4) as f64,
                    50.0 + (i + j) as f64,
                ));
            }
        }
        let first = condition_edges_adaptive(&edges, &solver()).unwrap();
        let second = condition_edges_adaptive(&edges, &solver()).unwrap();
        assert_eq!(first.weights, second.weights);
        let mut reversed = edges.clone();
        reversed.reverse();
        let permuted = condition_edges_adaptive(&reversed, &solver()).unwrap();
        for (k, w) in first.weights.iter().enumerate() {
            let other = permuted.weights[edges.len() - 1 - k];
            assert!((w - other).abs() <= 1e-6 * w.max(other), "{w} vs {other}");
        }
    }

    #[test]
    fn a_tree_without_triangles_still_yields_finite_weights() {
        let n = 6;
        let t = truth(n);
        let edges: Vec<AdaptiveEdge> = (0..n - 1).map(|i| edge(&t, i, i + 1, 0.1, 50.0)).collect();
        let result = condition_edges_adaptive(&edges, &solver()).unwrap();
        assert_eq!(result.triangle_count, 0);
        assert!(result.weights.iter().all(|w| w.is_finite() && *w > 0.0));
    }

    #[test]
    fn malformed_input_is_rejected() {
        let t = truth(3);
        let good = edge(&t, 0, 1, 0.0, 10.0);
        let mut reversed = good;
        reversed.view_i = ViewId::new(1);
        reversed.view_j = ViewId::new(0);
        assert!(matches!(
            condition_edges_adaptive(&[reversed], &solver()),
            Err(AdaptiveConditioningError::InvalidEdge(_))
        ));
        let mut no_evidence = good;
        no_evidence.evidence = 0.0;
        assert!(matches!(
            condition_edges_adaptive(&[no_evidence], &solver()),
            Err(AdaptiveConditioningError::InvalidEdge(_))
        ));
        assert!(matches!(
            condition_edges_adaptive(&[good, good], &solver()),
            Err(AdaptiveConditioningError::InvalidEdge(_))
        ));
    }

    #[test]
    fn exactly_fitting_edges_never_produce_non_finite_weights() {
        let n = 8;
        let t = truth(n);
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push(edge(&t, i, j, 0.0, 20.0));
            }
        }
        let result = condition_edges_adaptive(&edges, &solver()).unwrap();
        assert!(result.weights.iter().all(|w| w.is_finite()));
        assert!(result.confidence.iter().all(|w| w.is_finite()));
    }
}
