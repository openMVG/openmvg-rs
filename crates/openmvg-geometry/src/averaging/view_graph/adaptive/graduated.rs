// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graduated robust solve (steps 5 to 7 of the conditioner).
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                        Graduated robust solve                        │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: prior, reliability, closure-error quantile p25
//!                                   │
//!                                   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Anneal: scale 30 deg -> 2.5 x p25 over STEPS re-solves,      │
//! │         weights = prior / (1 + (residual / scale)^2)                 │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                   │
//!                                   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Follow the residuals, FINAL_ROUNDS times:                    │
//! │         scale = 2.5 x median residual of trusted edges, clamped      │
//! │         weights = annealed weights / (1 + (residual / scale)^2)      │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                   │
//!                                   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: One last solve with the final weights                        │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                   │
//!                                   ▼
//! Output: weights, rotations, final scale
//! ```
//!
//! This is a continuation heuristic in the spirit of graduated non-convexity (Blake and
//! Zisserman): the wide first scale lets the early re-weightings tolerate large residuals, so a
//! coherent wrong cluster is less likely to lock in before the cycle prior has had its say, and
//! the prior itself never sees the contaminated estimate. There is no convexity guarantee, the
//! injected solver is itself a robust IRLS, and the nine solves always run (no convergence test).
//! The residual-following rounds start from the annealed weights as their prior, so the final
//! weights keep the suppression of the last annealing step and add one more Cauchy factor.

use std::collections::BTreeMap;

use nalgebra::Rotation3;
use openmvg_core::model::{EstimationResult, RelativeRotation, ViewId};

use super::{AdaptiveConditioningError, AdaptiveEdge, quantile, residual_deg};
use crate::averaging::RotationEstimate;

/// Starting scale of the graduated solve, degrees.
const START_SCALE_DEG: f64 = 30.0;
/// Final scale of the graduated solve as a multiple of the closure-error 25th percentile.
const END_SCALE_FACTOR: f64 = 2.5;
/// Re-solves of the graduated schedule.
const STEPS: usize = 5;
/// Rounds that re-estimate the scale from the residuals of the trusted edges.
const FINAL_ROUNDS: usize = 3;
/// Reliability above which an edge is trusted when estimating the final scale.
const TRUSTED_RELIABILITY: f64 = 0.5;
/// The following scale is this multiple of the median residual of the trusted edges.
const TRUSTED_MEDIAN_FACTOR: f64 = 2.5;
/// The final-round scale stays within `[LOW, HIGH] x` the end scale of the annealing.
const FINAL_SCALE_RANGE: (f64, f64) = (0.7, 4.0);
/// Weight handed to the solver is never below this, so every view stays in the system.
pub(super) const MIN_WEIGHT: f64 = 1e-6;
/// A returned rotation may deviate from orthonormality by this much (Frobenius norm and
/// determinant) before the solver output is rejected.
const ROTATION_TOLERANCE: f64 = 1e-6;

// The schedule divides by `STEPS - 1` and reports the scale of its last round.
const _: () = assert!(STEPS > 1 && FINAL_ROUNDS > 0);

/// The weight the solver receives for a requested weight: floored, and finite.
pub(super) fn effective_weight(weight: f64) -> f64 {
    if weight.is_finite() {
        weight.max(MIN_WEIGHT)
    } else {
        MIN_WEIGHT
    }
}

/// Whether `rotation` is finite, orthonormal and right-handed.
fn is_proper_rotation(rotation: &Rotation3<f64>) -> bool {
    let m = rotation.matrix();
    m.iter().all(|x| x.is_finite())
        && (m.transpose() * m - nalgebra::Matrix3::identity()).norm() <= ROTATION_TOLERANCE
        && (m.determinant() - 1.0).abs() <= ROTATION_TOLERANCE
}

/// The injected rotation solver with a call counter. Its output is checked on every call.
pub(super) struct Solver<'a> {
    edges: &'a [AdaptiveEdge],
    /// Every view an edge touches: the solver must return a rotation for each.
    views: Vec<ViewId>,
    solve: &'a dyn Fn(&[RelativeRotation]) -> EstimationResult<RotationEstimate>,
    count: usize,
}

impl<'a> Solver<'a> {
    pub(super) fn new(
        edges: &'a [AdaptiveEdge],
        solve: &'a dyn Fn(&[RelativeRotation]) -> EstimationResult<RotationEstimate>,
    ) -> Self {
        let mut views: Vec<ViewId> = edges.iter().flat_map(|e| [e.view_i, e.view_j]).collect();
        views.sort_unstable();
        views.dedup();
        Self {
            edges,
            views,
            solve,
            count: 0,
        }
    }

    /// Rotation solves performed so far.
    pub(super) fn count(&self) -> usize {
        self.count
    }

    /// Solve with `weights`, one per edge, floored by [`effective_weight`]. Fails when the solver
    /// errors or does not return a proper rotation for every view.
    fn rotations(
        &mut self,
        weights: &[f64],
    ) -> Result<BTreeMap<ViewId, Rotation3<f64>>, AdaptiveConditioningError> {
        let relatives = self
            .edges
            .iter()
            .zip(weights)
            .map(|(e, &w)| {
                RelativeRotation::try_new(e.view_i, e.view_j, e.rotation, effective_weight(w))
                    .map_err(|error| AdaptiveConditioningError::InvalidEdge(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.count += 1;
        let rotations = (self.solve)(&relatives)
            .map(|estimate| estimate.rotations)
            .map_err(|error| AdaptiveConditioningError::Solver(error.to_string()))?;
        if let Some(view) = self
            .views
            .iter()
            .find(|view| !rotations.get(view).is_some_and(is_proper_rotation))
        {
            return Err(AdaptiveConditioningError::Solver(format!(
                "the rotation solver returned no proper rotation for view {view}"
            )));
        }
        Ok(rotations)
    }
}

/// Result of the graduated solve.
pub(super) struct Graduated {
    /// Final edge weights before flooring, evidence included.
    pub(super) weights: Vec<f64>,
    /// Rotations solved with [`Self::weights`] floored by [`effective_weight`].
    pub(super) rotations: BTreeMap<ViewId, Rotation3<f64>>,
    /// Scale of the last re-weighting, degrees.
    pub(super) scale_deg: f64,
}

/// Run the annealed solves, the residual-following rounds and the final solve.
///
/// `reliability` selects the trusted edges of the final rounds, `prior` seeds the weights and
/// `cycle_quantile` (degrees) sets the end scale.
pub(super) fn solve_graduated(
    solver: &mut Solver<'_>,
    reliability: &[f64],
    prior: &[f64],
    cycle_quantile: f64,
) -> Result<Graduated, AdaptiveConditioningError> {
    // A graph whose closures are wider than the start scale would otherwise anneal upwards.
    let end_scale = (END_SCALE_FACTOR * cycle_quantile).min(START_SCALE_DEG);
    let weights = anneal(solver, prior, end_scale)?;
    let (weights, scale_deg) = follow_residuals(solver, reliability, weights, end_scale)?;
    let rotations = solver.rotations(&weights)?;
    Ok(Graduated {
        weights,
        rotations,
        scale_deg,
    })
}

/// Step 5: re-solve `STEPS` times while the scale shrinks geometrically to `end_scale`.
fn anneal(
    solver: &mut Solver<'_>,
    prior: &[f64],
    end_scale: f64,
) -> Result<Vec<f64>, AdaptiveConditioningError> {
    let mut weights = prior.to_vec();
    for step in 0..STEPS {
        let t = step as f64 / (STEPS - 1) as f64;
        let scale = START_SCALE_DEG * (end_scale / START_SCALE_DEG).powf(t);
        let rotations = solver.rotations(&weights)?;
        weights = reweight(solver.edges, prior, &rotations, scale);
    }
    Ok(weights)
}

/// Step 6: the scale follows the median residual of the trusted edges. Returns the weights and
/// the scale of the last round.
fn follow_residuals(
    solver: &mut Solver<'_>,
    reliability: &[f64],
    mut weights: Vec<f64>,
    end_scale: f64,
) -> Result<(Vec<f64>, f64), AdaptiveConditioningError> {
    let lo = FINAL_SCALE_RANGE.0 * end_scale;
    let hi = FINAL_SCALE_RANGE.1 * end_scale;
    let stage_prior = weights.clone();
    let mut scale = (lo * hi).sqrt();
    for _ in 0..FINAL_ROUNDS {
        let rotations = solver.rotations(&weights)?;
        let mut trusted: Vec<f64> = solver
            .edges
            .iter()
            .zip(reliability)
            .filter(|(_, r)| **r > TRUSTED_RELIABILITY)
            .map(|(e, _)| residual_deg(&rotations, e))
            .filter(|x| x.is_finite())
            .collect();
        trusted.sort_by(f64::total_cmp);
        if let Some(median) = quantile(&trusted, 0.5) {
            scale = (TRUSTED_MEDIAN_FACTOR * median).clamp(lo, hi);
        }
        weights = reweight(solver.edges, &stage_prior, &rotations, scale);
    }
    Ok((weights, scale))
}

/// `prior / (1 + (residual / scale)^2)` per edge.
fn reweight(
    edges: &[AdaptiveEdge],
    prior: &[f64],
    rotations: &BTreeMap<ViewId, Rotation3<f64>>,
    scale: f64,
) -> Vec<f64> {
    edges
        .iter()
        .zip(prior)
        .map(|(edge, prior)| {
            let x = residual_deg(rotations, edge) / scale;
            prior / (1.0 + x * x)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{edge, solver, truth};
    use super::*;
    use openmvg_core::model::EstimationError;

    fn ring(n: usize) -> Vec<AdaptiveEdge> {
        let t = truth(n);
        (0..n)
            .map(|i| edge(&t, i, (i + 1) % n.max(2), 0.0, 10.0))
            .collect()
    }

    #[test]
    fn reweight_applies_the_cauchy_kernel_of_the_residual() {
        let t = truth(3);
        let mut e = edge(&t, 0, 1, 0.0, 10.0);
        // A residual of exactly one scale halves the weight.
        e.rotation = super::super::test_support::rotation([0.0, 0.0, 1.0], 2.0) * e.rotation;
        let rotations: BTreeMap<ViewId, Rotation3<f64>> =
            (0..3).map(|i| (ViewId::new(i), t[i])).collect();
        let w = reweight(&[e], &[8.0], &rotations, 2.0);
        assert!((w[0] - 4.0).abs() < 1e-9, "{w:?}");
    }

    #[test]
    fn a_view_without_a_rotation_gets_a_vanishing_weight() {
        let t = truth(3);
        let e = edge(&t, 0, 1, 0.0, 10.0);
        let w = reweight(&[e], &[1.0], &BTreeMap::new(), 2.0);
        assert!(w[0] == 0.0, "{w:?}");
    }

    #[test]
    fn the_solver_counts_calls_and_floors_bad_weights() {
        let n = 4;
        let edges: Vec<AdaptiveEdge> = {
            let t = truth(n);
            (0..n - 1).map(|i| edge(&t, i, i + 1, 0.0, 10.0)).collect()
        };
        let seen = std::cell::RefCell::new(Vec::new());
        let inner = solver();
        let spy = |relatives: &[RelativeRotation]| {
            seen.borrow_mut()
                .extend(relatives.iter().map(RelativeRotation::weight));
            inner(relatives)
        };
        let mut s = Solver::new(&edges, &spy);
        s.rotations(&[f64::NAN, 0.0, 3.0]).unwrap();
        assert_eq!(s.count(), 1);
        assert_eq!(*seen.borrow(), vec![MIN_WEIGHT, MIN_WEIGHT, 3.0]);
    }

    fn bogus_estimate(
        rotations: BTreeMap<ViewId, Rotation3<f64>>,
    ) -> EstimationResult<RotationEstimate> {
        Ok(RotationEstimate::new(
            rotations,
            1,
            openmvg_core::AngularResidual::try_new(0.0).unwrap(),
        ))
    }

    #[test]
    fn malformed_solver_output_is_rejected_before_it_is_used() {
        let edges = ring(4);
        let views = (0..4).map(ViewId::new);
        let scaled = |factor: f64| {
            views
                .clone()
                .map(|v| {
                    let m = nalgebra::Matrix3::identity() * factor;
                    (v, Rotation3::from_matrix_unchecked(m))
                })
                .collect::<BTreeMap<_, _>>()
        };
        let mut missing_one = scaled(1.0);
        missing_one.remove(&ViewId::new(2));
        for rotations in [
            BTreeMap::new(),
            missing_one,
            scaled(f64::NAN),
            scaled(2.0),
            scaled(-1.0),
        ] {
            let solve = |_: &[RelativeRotation]| bogus_estimate(rotations.clone());
            let mut s = Solver::new(&edges, &solve);
            assert!(matches!(
                solve_graduated(&mut s, &[1.0; 4], &[1.0; 4], 1.0),
                Err(AdaptiveConditioningError::Solver(_))
            ));
        }
    }

    #[test]
    fn a_graph_wider_than_the_start_scale_does_not_anneal_upwards() {
        let n = 5;
        let t = truth(n);
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push(edge(&t, i, j, 0.05, 30.0));
            }
        }
        let m = edges.len();
        let solve = solver();
        let mut s = Solver::new(&edges, &solve);
        let out = solve_graduated(&mut s, &vec![1.0; m], &vec![1.0; m], 60.0).unwrap();
        assert!(
            out.scale_deg <= FINAL_SCALE_RANGE.1 * START_SCALE_DEG + 1e-9,
            "{}",
            out.scale_deg
        );
    }

    #[test]
    fn a_failing_solver_surfaces_as_a_solver_error() {
        let edges = ring(3);
        let failing = |_: &[RelativeRotation]| -> EstimationResult<RotationEstimate> {
            Err(EstimationError::InsufficientData {
                required: 1,
                provided: 0,
            })
        };
        let mut s = Solver::new(&edges, &failing);
        assert!(matches!(
            solve_graduated(&mut s, &[1.0; 3], &[1.0; 3], 1.0),
            Err(AdaptiveConditioningError::Solver(_))
        ));
    }

    #[test]
    fn the_schedule_runs_the_documented_number_of_solves() {
        let n = 5;
        let t = truth(n);
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push(edge(&t, i, j, 0.05, 30.0));
            }
        }
        let m = edges.len();
        let solve = solver();
        let mut s = Solver::new(&edges, &solve);
        let out = solve_graduated(&mut s, &vec![1.0; m], &vec![1.0; m], 0.5).unwrap();
        // STEPS annealed solves, FINAL_ROUNDS following rounds and the last solve.
        assert_eq!(s.count(), STEPS + FINAL_ROUNDS + 1);
        assert_eq!(out.weights.len(), m);
        let (lo, hi) = FINAL_SCALE_RANGE;
        let end = END_SCALE_FACTOR * 0.5;
        assert!(out.scale_deg >= lo * end - 1e-12 && out.scale_deg <= hi * end + 1e-12);
    }
}
