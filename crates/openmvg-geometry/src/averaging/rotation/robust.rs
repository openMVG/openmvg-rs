// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic IRLS (Iteratively Reweighted Least Squares) wrapper for robust
//! rotation averaging.
//!
//! Its teaching docs are on [`average_rotations_with_robust_loss`], where docs.rs shows them.

use nalgebra::Rotation3;
#[cfg(any(feature = "faer", test))]
use openmvg_ba::{ResidualScale, RobustLoss};
use openmvg_core::model::{EstimationError, EstimationResult, RelativeRotation, ViewId};
use std::collections::BTreeMap;

use crate::averaging::{RotationConfig, RotationEstimate, RotationSolver};

// ============================================================================
// Robust kernels
// ============================================================================

/// Robust kernel function for IRLS weighting.
///
/// Each variant computes a weight w(r) ∈ [0, 1] that downweights
/// edges with large residuals. The weighted cost Σ w_ij · κ_ij · ‖…‖²
/// approximates the robust cost Σ ρ(‖…‖).
#[derive(Debug, Clone, Copy)]
pub(crate) enum RobustKernel {
    /// Standard L2 (no robustness). Baseline for comparison.
    // Constructed only by the `faer` loss selection and the tests.
    #[cfg_attr(not(any(feature = "faer", test)), allow(dead_code))]
    L2,
    /// Huber kernel: ρ(r) = r²/2 if r ≤ c, else c·r − c²/2.
    /// Weight: w(r) = min(1, c/r). Mild outlier rejection.
    // Constructed only by the `faer` loss selection and the tests.
    #[cfg_attr(not(any(feature = "faer", test)), allow(dead_code))]
    Huber { c: f64 },
    /// Cauchy (Lorentzian): ρ(r) = (c²/2) ln(1 + r²/c²).
    /// Weight: w(r) = 1/(1 + r²/c²). Moderate outlier rejection.
    Cauchy { c: f64 },
}

impl RobustKernel {
    /// Map a public robust loss onto the chordal IRLS kernel.
    ///
    /// The IRLS residual is the chordal distance `‖R_j R_iᵀ − R̄_ij‖_F`, which
    /// equals `2√2·sin(θ/2)` for a rotation error θ, so an angular scale θ maps
    /// exactly to the chordal scale `2√2·sin(θ/2)`. With `c` equal to that chord,
    /// both kernels reproduce the weights `openmvg-ba` applies for the same
    /// [`RobustLoss`] on the same residual: Huber `min(1, c/r)`, and Cauchy
    /// `1/(1 + r²/c²)` (ba evaluates it as Barron's α = 0 kernel at scale
    /// `c/√2`). A pixel scale has no meaning for rotation residuals and is
    /// rejected.
    #[cfg(any(feature = "faer", test))]
    pub(crate) fn try_from_loss(loss: RobustLoss) -> EstimationResult<Self> {
        fn chord(scale: ResidualScale) -> EstimationResult<f64> {
            match scale {
                ResidualScale::Angular(angle) => {
                    Ok(2.0 * std::f64::consts::SQRT_2 * (0.5 * angle.value()).sin())
                }
                other => Err(EstimationError::InvalidInput(format!(
                    "rotation averaging needs an angular robust-loss scale, got {other:?}"
                ))),
            }
        }
        match loss {
            RobustLoss::Squared => Ok(Self::L2),
            RobustLoss::Huber(scale) => Ok(Self::Huber { c: chord(scale)? }),
            RobustLoss::Cauchy(scale) => Ok(Self::Cauchy { c: chord(scale)? }),
            other => Err(EstimationError::InvalidInput(format!(
                "unsupported robust loss for rotation averaging: {other:?}"
            ))),
        }
    }

    /// Compute the IRLS weight for a given residual r ≥ 0.
    #[inline]
    pub(crate) fn weight(self, r: f64) -> f64 {
        match self {
            Self::L2 => 1.0,
            Self::Huber { c } => {
                if r <= c {
                    1.0
                } else {
                    c / r.max(1e-15)
                }
            }
            Self::Cauchy { c } => {
                let c2 = c * c;
                c2 / (c2 + r * r)
            }
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the robust IRLS wrapper.
#[derive(Debug, Clone)]
pub(crate) struct RobustConfig {
    /// The robust kernel to use.
    pub(crate) kernel: RobustKernel,
    /// Maximum number of IRLS iterations.
    pub(crate) max_irls_iter: usize,
    /// Convergence tolerance on max |w_new − w_old|.
    pub(crate) weight_tol: f64,
    /// Minimum weight floor to prevent complete edge removal.
    pub(crate) min_weight: f64,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self {
            kernel: RobustKernel::Cauchy { c: 1.0 },
            max_irls_iter: 20,
            weight_tol: 1e-3,
            min_weight: 1e-8,
        }
    }
}

impl RobustConfig {
    /// Create a config with a specific kernel and default iteration parameters.
    #[cfg(any(feature = "faer", test))]
    pub(crate) fn with_kernel(kernel: RobustKernel) -> Self {
        Self {
            kernel,
            ..Default::default()
        }
    }
}

// ============================================================================
// Solver
// ============================================================================

/// Generic robust rotation averaging solver via IRLS.
///
/// Wraps any [`RotationSolver`] in an iteratively-reweighted loop.
/// The inner solver sees modified `RelativeRotation` weights each iteration.
#[derive(Debug, Clone)]
pub struct RobustRotationSolver<S: RotationSolver> {
    inner: S,
    config: RobustConfig,
}

impl<S: RotationSolver> RobustRotationSolver<S> {
    pub fn new(inner: S, config: RobustConfig) -> Self {
        Self { inner, config }
    }

    /// Compute per-edge Frobenius residual against the solver's convention R_ij = R_j·R_iᵀ:
    /// r_ij = ‖R_j·R_iᵀ − R̄_ij‖_F. This is zero at the exact solution (R_j = R_ij·R_i) for
    /// *any* rotations. The `R_iᵀ·R_j` form is a conjugate that is only zero when R_i and
    /// R_ij commute, so it would down-weight good edges on general 3D rotations.
    fn compute_edge_residuals(
        rotations: &BTreeMap<ViewId, Rotation3<f64>>,
        relative_rotations: &[RelativeRotation],
        residuals: &mut [f64],
    ) {
        for (idx, rel) in relative_rotations.iter().enumerate() {
            residuals[idx] = match (rotations.get(&rel.view_i()), rotations.get(&rel.view_j())) {
                (Some(ri), Some(rj)) => {
                    let diff = *rj * ri.inverse();
                    (diff.matrix() - rel.rotation().matrix()).norm()
                }
                _ => 0.0,
            };
        }
    }
}

impl<S: RotationSolver> RotationSolver for RobustRotationSolver<S> {
    type Workspace = ();

    fn solve(
        &self,
        relative_rotations: &[RelativeRotation],
        config: &RotationConfig,
    ) -> EstimationResult<RotationEstimate> {
        config.validate()?;
        if relative_rotations.is_empty() {
            return Err(EstimationError::InsufficientData {
                required: 1,
                provided: 0,
            });
        }
        super::ensure_connected(relative_rotations)?;

        let m = relative_rotations.len();
        let kernel = self.config.kernel;
        let min_weight = self.config.min_weight;

        // Store original weights (small Vec<f64>, no matrices)
        let original_weights: Vec<f64> = relative_rotations.iter().map(|r| r.weight()).collect();

        // Working copy of measurements — only .weight is mutated each iteration
        let mut working: Vec<RelativeRotation> = relative_rotations.to_vec();

        // Pre-allocated buffers
        let mut weights: Vec<f64> = vec![1.0; m];
        let mut residuals: Vec<f64> = vec![0.0; m];
        // Only the weights change between iterations, so the inner solver may reuse
        // its structure work.
        let mut workspace = S::Workspace::default();

        let mut last_result: Option<RotationEstimate> = None;
        let mut iteration_count = 0;
        let max_iterations = self.config.max_irls_iter.min(config.max_iterations());

        for _iter in 0..max_iterations {
            // Apply current weights to measurements
            for (idx, meas) in working.iter_mut().enumerate() {
                *meas = meas.with_weight(original_weights[idx] * weights[idx])?;
            }

            // Solve with inner solver
            let result = self.inner.solve_in(&working, config, &mut workspace)?;
            iteration_count += 1;

            // Compute per-edge residuals
            Self::compute_edge_residuals(&result.rotations, relative_rotations, &mut residuals);

            // Fused weight update: new weight + track max change
            let mut max_weight_change: f64 = 0.0;
            for idx in 0..m {
                let new_w = kernel.weight(residuals[idx]).max(min_weight);
                let change = (new_w - weights[idx]).abs();
                if change > max_weight_change {
                    max_weight_change = change;
                }
                weights[idx] = new_w;
            }

            last_result = Some(result);

            // Check convergence
            if max_weight_change < self.config.weight_tol {
                break;
            }
        }

        let result = last_result
            .ok_or_else(|| EstimationError::Numerical("IRLS produced no result".to_string()))?;

        // Report the public result in radians; the chordal residual buffer above
        // remains deliberate because the configured IRLS kernels use that scale.
        let avg_residual = super::mean_angular_residual(&result.rotations, relative_rotations);

        Ok(RotationEstimate {
            rotations: result.rotations,
            iteration_count,
            residual: crate::averaging::checked_angular_residual(avg_residual)?,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Unit, Vector3};
    use openmvg_ba::numerics::refine::Loss;
    use openmvg_core::{AngularThreshold, PixelThreshold};
    use std::f64::consts::SQRT_2;

    #[test]
    fn robust_rotation_honors_outer_iteration_limit_and_count() {
        use std::cell::Cell;
        struct Counted<'a>(&'a Cell<usize>);
        impl RotationSolver for Counted<'_> {
            type Workspace = ();
            fn solve(
                &self,
                _: &[RelativeRotation],
                _: &RotationConfig,
            ) -> EstimationResult<RotationEstimate> {
                self.0.set(self.0.get() + 1);
                Ok(RotationEstimate::new(
                    BTreeMap::from([
                        (ViewId::new(0), Rotation3::identity()),
                        (ViewId::new(1), Rotation3::identity()),
                    ]),
                    99,
                    openmvg_core::AngularResidual::ZERO,
                ))
            }
        }
        let edge = RelativeRotation::try_new(
            ViewId::new(0),
            ViewId::new(1),
            Rotation3::from_euler_angles(0.0, 0.0, 0.3),
            1.0,
        )
        .unwrap();
        let calls = Cell::new(0);
        let solver = RobustRotationSolver::new(
            Counted(&calls),
            RobustConfig::with_kernel(RobustKernel::Cauchy { c: 0.1 }),
        );
        let invalid = RotationConfig::default().with_max_iterations(0);
        assert!(matches!(
            solver.solve(std::slice::from_ref(&edge), &invalid),
            Err(EstimationError::InvalidInput(_))
        ));
        assert_eq!(calls.get(), 0);
        let one = solver
            .solve(
                std::slice::from_ref(&edge),
                &RotationConfig::default().with_max_iterations(1),
            )
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(one.iteration_count, 1);
        calls.set(0);
        let two = solver
            .solve(&[edge], &RotationConfig::default().with_max_iterations(2))
            .unwrap();
        assert_eq!(calls.get(), 2);
        assert_eq!(two.iteration_count, 2);
    }

    /// Chordal residual `‖R − I‖_F` of a rotation by `angle` radians.
    fn chordal_residual(angle: f64) -> f64 {
        let axis = Unit::new_normalize(Vector3::new(0.3, -0.5, 0.8));
        (Rotation3::from_axis_angle(&axis, angle).matrix() - Matrix3::identity()).norm()
    }

    #[test]
    fn chordal_residual_is_two_root_two_sine_half_angle() {
        for angle in [1e-4, 0.01, 0.3, 1.0, 2.5, std::f64::consts::PI] {
            let expected = 2.0 * SQRT_2 * (0.5 * angle).sin();
            assert!((chordal_residual(angle) - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn robust_loss_weights_match_ba_kernel_on_chordal_residuals() {
        let residuals = [0.0, 1e-3, 0.02, 0.07, 0.1, 0.25, 0.8, 1.9, 2.8];
        for degrees in [0.5, 2.0, 5.0, 15.0, 90.0, 180.0] {
            let angle = AngularThreshold::from_degrees(degrees).unwrap();
            let scale = ResidualScale::Angular(angle);
            // The same `RobustLoss` as ba's kernel sees it on a chordal residual
            // (`RobustLoss::kernel_loss`: Huber delta = s, Cauchy scale = s/√2),
            // with `s` the chord of the angular scale.
            let chord = chordal_residual(angle.value());
            let cases = [
                (RobustLoss::Squared, Loss::L2),
                (RobustLoss::Huber(scale), Loss::Huber { delta: chord }),
                (
                    RobustLoss::Cauchy(scale),
                    Loss::Cauchy {
                        scale: chord / SQRT_2,
                    },
                ),
            ];
            for (loss, ba_kernel) in cases {
                let kernel = RobustKernel::try_from_loss(loss).unwrap();
                for residual in residuals {
                    let ba_weight = ba_kernel.relative_sqrt_weight(residual).powi(2);
                    let weight = kernel.weight(residual);
                    assert!(
                        (weight - ba_weight).abs() <= 1e-12,
                        "{loss:?} at r = {residual}: geometry {weight}, ba {ba_weight}"
                    );
                }
            }
        }
    }

    #[test]
    fn robust_loss_rejects_pixel_scale() {
        let pixels = ResidualScale::Pixels(PixelThreshold::ONE_PIXEL);
        for loss in [RobustLoss::Huber(pixels), RobustLoss::Cauchy(pixels)] {
            assert!(matches!(
                RobustKernel::try_from_loss(loss),
                Err(EstimationError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn cauchy_angle_for_chordal_scale_one_tenth() {
        // The rotation-filter benchmark's kernel `c = 0.1` (chordal
        // units) is the angle `θ = 2·asin(0.1 / (2√2))`, because `c = 2√2·sin(θ/2)`.
        let angle = AngularThreshold::try_new(2.0 * (0.1 / (2.0 * SQRT_2)).asin()).unwrap();
        let RobustKernel::Cauchy { c } =
            RobustKernel::try_from_loss(RobustLoss::Cauchy(ResidualScale::Angular(angle))).unwrap()
        else {
            panic!("a Cauchy loss maps to the Cauchy kernel");
        };
        assert!((c - 0.1).abs() < 1e-15, "c = {c}");

        // `θ = 2·asin(0.05)` instead makes ba's internal Barron scale
        // `chord/√2 = 2·sin(θ/2)` equal 0.1, which is the classical Cauchy
        // `c = 0.1·√2` used here.
        let angle = AngularThreshold::try_new(2.0 * 0.05_f64.asin()).unwrap();
        let RobustKernel::Cauchy { c } =
            RobustKernel::try_from_loss(RobustLoss::Cauchy(ResidualScale::Angular(angle))).unwrap()
        else {
            panic!("a Cauchy loss maps to the Cauchy kernel");
        };
        assert!((c / SQRT_2 - 0.1).abs() < 1e-15, "c = {c}");
    }

    #[test]
    fn test_kernel_weights() {
        let l2 = RobustKernel::L2;
        assert!((l2.weight(0.0) - 1.0).abs() < 1e-10);
        assert!((l2.weight(100.0) - 1.0).abs() < 1e-10);

        let huber = RobustKernel::Huber { c: 1.0 };
        assert!((huber.weight(0.5) - 1.0).abs() < 1e-10);
        assert!((huber.weight(2.0) - 0.5).abs() < 1e-10);

        let cauchy = RobustKernel::Cauchy { c: 1.0 };
        assert!((cauchy.weight(0.0) - 1.0).abs() < 1e-10);
        assert!((cauchy.weight(1.0) - 0.5).abs() < 1e-10);
    }
}
