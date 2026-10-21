// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! FusedP3P: Ding pencil, polished depths, and cheap `SO(3)` recovery.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                         Fused P3P                                    │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: 3 world points Xᵢ + 3 unit bearings bᵢ
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Ding pencil and cubic with a multiple-root test              │
//! └──────────────────────────────────────────────────────────────────────┘
//! Build the Ding pencil, polish the cubic root, and extract line factors.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Depth charts and SO(3) recovery                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! Evaluate both depth quadratics, polish λᵢ, and recover R with the
//! Newton-Schulz orthogonalization path.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Fallback and validation                                      │
//! └──────────────────────────────────────────────────────────────────────┘
//! Run Ke, then LambdaTwist, only when the cubic is near a multiple root or
//! the primary path finds no pose, and keep only the poses that differ from
//! those already found. Every candidate passes positive-depth and fit checks.
//!   │
//!   ▼
//! Output: up to 4 validated positive-depth pose candidates
//! ```
//!
//! FusedP3P is original to OpenMVG-rs: it adds a depth polish, a rotation clean-up and two
//! fallbacks to the Ding solver, so that near-degenerate samples still give accurate poses.

use crate::solver_traits::{MultiSolutionSolver, Solver};
use crate::{SolverError, SolverResult};
use nalgebra::{Matrix3, Point3, UnitVector3};

use super::{
    common::{
        P3pInput, P3pSolutions, PROJECTED_RAY_TOLERANCE, SolverP3pSolutions, angular_residual,
        exact_indexed_sample, exact_sample, refine_depths, rotation_from_edges,
    },
    ding::DingProblem,
    ke::{KeP3pConfig, KeP3pSolver},
    lambdatwist::LambdaTwistP3pSolver,
};
use crate::{
    pose::{BearingToPoint, SolverPose, SolverPoseExt},
    shared::polynomial::polished_cubic_root,
};

/// Options for [`FusedP3pSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FusedP3pConfig {
    /// Enable the Ke, then LambdaTwist, fallback near a multiple cubic root or when the primary path finds no pose.
    pub(crate) guarded_fallback: bool,
}

impl Default for FusedP3pConfig {
    fn default() -> Self {
        Self {
            guarded_fallback: true,
        }
    }
}

/// Calibrated P3P solver: Ding pencil, depth polish, and a Ke/LambdaTwist fallback near multiple roots.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FusedP3pSolver;

impl FusedP3pSolver {
    /// Solve one exact minimal sample without heap allocation.
    pub(crate) fn solve_minimal(
        world: [Point3<f64>; 3],
        bearings: [UnitVector3<f64>; 3],
        config: FusedP3pConfig,
    ) -> P3pSolutions {
        Self::try_solve_minimal(world, bearings, config)
            .map(P3pSolutions::from_solver)
            .unwrap_or_default()
    }

    fn try_solve_minimal(
        world: [Point3<f64>; 3],
        bearings: [UnitVector3<f64>; 3],
        config: FusedP3pConfig,
    ) -> SolverResult<SolverP3pSolutions> {
        let input = P3pInput::new(world, bearings).ok_or(SolverError::InvalidInput {
            reason: "P3P points and bearings must be finite",
        })?;
        let (output, guard_activated) = solve_validated(&input, config.guarded_fallback);
        if !config.guarded_fallback || (!guard_activated && !output.is_empty()) {
            return Ok(output);
        }
        Ok(guarded_fallback(&input, world, bearings, output))
    }
}

impl Solver for FusedP3pSolver {
    type Datum = BearingToPoint;
    type Model = SolverPose<f64>;
    type Config = FusedP3pConfig;
    type Unit = openmvg_core::Radians;

    fn min_sample_size() -> usize {
        3
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        angular_residual(model, datum)
    }
}

impl MultiSolutionSolver for FusedP3pSolver {
    fn solve(data: &[Self::Datum], config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        let (world, bearings) = exact_sample(data)?;
        Ok(Self::try_solve_minimal(world, bearings, *config)?.into_vec())
    }

    fn solve_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Vec<Self::Model>> {
        let (world, bearings) = exact_indexed_sample(indices, data)?;
        Ok(Self::try_solve_minimal(world, bearings, *config)?.into_vec())
    }

    fn for_each_solution_from_indices<F>(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
        mut emit: F,
    ) -> SolverResult<usize>
    where
        F: FnMut(Self::Model),
    {
        let (world, bearings) = exact_indexed_sample(indices, data)?;
        let solutions = Self::try_solve_minimal(world, bearings, *config)?;
        let count = solutions.len();
        for pose in solutions {
            emit(pose);
        }
        Ok(count)
    }
}

fn solve_validated(input: &P3pInput, check_guard: bool) -> (SolverP3pSolutions, bool) {
    let mut output = SolverP3pSolutions::default();
    let Some(problem) = DingProblem::new(input) else {
        return (output, false);
    };
    let [cubic_b, cubic_c, cubic_d] = problem.pencil_cubic();
    let guard_activated = check_guard && near_multiple_root(cubic_b, cubic_c, cubic_d);
    let Some(root) = polished_cubic_root(cubic_b, cubic_c, cubic_d) else {
        return (output, guard_activated);
    };
    let Some(factors) = problem.line_factors(root) else {
        return (output, guard_activated);
    };

    for factor in factors {
        for mut depths in problem.depths(factor).into_iter().flatten() {
            refine_depths(
                &mut depths,
                problem.squared_edges,
                problem.bearing_cosines,
                2,
            );
            if depths
                .iter()
                .any(|depth| !depth.is_finite() || *depth <= 0.0)
            {
                continue;
            }
            let camera0 = depths[0] * problem.bearings[0];
            let camera1 = depths[1] * problem.bearings[1];
            let camera2 = depths[2] * problem.bearings[2];
            let algebraic_rotation =
                rotation_from_edges(camera0 - camera1, camera0 - camera2, &problem.world_inverse);
            let (rotation, certified, correction_bound_squared) =
                newton_schulz_step(algebraic_rotation);
            let translation = camera0 - rotation * problem.world[0];
            if certified {
                // Let E bound the camera-point displacement introduced by
                // the polar step. E <= tau / (1 + tau) * depth proves both
                // positive corrected depth and tan(angle) <= tau.
                let fit_margin = PROJECTED_RAY_TOLERANCE / (1.0 + PROJECTED_RAY_TOLERANCE);
                let squared_fit_margin = fit_margin * fit_margin;
                let minimal_sample_fit_proven = correction_bound_squared * problem.squared_edges[0]
                    <= squared_fit_margin * depths[1] * depths[1]
                    && correction_bound_squared * problem.squared_edges[1]
                        <= squared_fit_margin * depths[2] * depths[2];
                input.push_certified_edge_basis_pose_from_positive_depths(
                    &mut output,
                    rotation,
                    translation,
                    minimal_sample_fit_proven,
                );
            } else {
                input.push_edge_basis_pose_from_positive_depths(&mut output, rotation, translation);
            }
        }
    }
    (output, guard_activated)
}

#[inline]
fn near_multiple_root(b: f64, c: f64, d: f64) -> bool {
    if !b.is_finite() || !c.is_finite() || !d.is_finite() {
        return true;
    }
    let b3 = b / 3.0;
    let p = c - b * b3;
    let q = d + b3 * (2.0 * b3 * b3 - c);
    let positive = 0.25 * q * q;
    let negative = p * p * p / 27.0;
    let magnitude = positive + negative.abs();
    let discriminant = (positive + negative).abs();
    if magnitude.is_finite() && discriminant.is_finite() {
        return magnitude == 0.0 || discriminant < 1e-10 * magnitude;
    }

    // Only exceptional coefficient powers need a scale-safe second pass.
    let scale = b.abs().max(c.abs().sqrt()).max(d.abs().cbrt());
    if scale == 0.0 || !scale.is_finite() {
        return true;
    }
    let scaled_b = b / scale;
    let scaled_c = c / scale / scale;
    let scaled_d = d / scale / scale / scale;
    let b3 = scaled_b / 3.0;
    let p = scaled_c - scaled_b * b3;
    let q = scaled_d + b3 * (2.0 * b3 * b3 - scaled_c);
    let positive = 0.25 * q * q;
    let negative = p * p * p / 27.0;
    let magnitude = positive + negative.abs();
    magnitude == 0.0 || (positive + negative).abs() < 1e-10 * magnitude
}

/// One order-two Newton-Schulz polar step and a conservative `SO(3)` certificate.
#[inline]
fn newton_schulz_step(rotation: Matrix3<f64>) -> (Matrix3<f64>, bool, f64) {
    let r00 = rotation[(0, 0)] * rotation[(0, 0)]
        + rotation[(1, 0)] * rotation[(1, 0)]
        + rotation[(2, 0)] * rotation[(2, 0)];
    let r11 = rotation[(0, 1)] * rotation[(0, 1)]
        + rotation[(1, 1)] * rotation[(1, 1)]
        + rotation[(2, 1)] * rotation[(2, 1)];
    let r22 = rotation[(0, 2)] * rotation[(0, 2)]
        + rotation[(1, 2)] * rotation[(1, 2)]
        + rotation[(2, 2)] * rotation[(2, 2)];
    let r01 = rotation[(0, 0)] * rotation[(0, 1)]
        + rotation[(1, 0)] * rotation[(1, 1)]
        + rotation[(2, 0)] * rotation[(2, 1)];
    let r02 = rotation[(0, 0)] * rotation[(0, 2)]
        + rotation[(1, 0)] * rotation[(1, 2)]
        + rotation[(2, 0)] * rotation[(2, 2)];
    let r12 = rotation[(0, 1)] * rotation[(0, 2)]
        + rotation[(1, 1)] * rotation[(1, 2)]
        + rotation[(2, 1)] * rotation[(2, 2)];
    // For G = R^T R = I + E, the corrected Gram matrix is exactly
    // I - 3/4 E^2 + 1/4 E^3. This bound leaves ample rounding margin below
    // the common 1e-10 near-rotation tolerance and avoids recomputing R^T R.
    let gram_error = (r00 - 1.0).abs()
        + (r11 - 1.0).abs()
        + (r22 - 1.0).abs()
        + r01.abs()
        + r02.abs()
        + r12.abs();
    let corrected_is_certified = gram_error <= 1e-6;
    let correction_bound_squared = 0.5 * (r00 + r11 + r22) * gram_error * gram_error;
    let corrected = rotation
        * Matrix3::new(
            1.5 - 0.5 * r00,
            -0.5 * r01,
            -0.5 * r02,
            -0.5 * r01,
            1.5 - 0.5 * r11,
            -0.5 * r12,
            -0.5 * r02,
            -0.5 * r12,
            1.5 - 0.5 * r22,
        );
    (corrected, corrected_is_certified, correction_bound_squared)
}

#[cold]
#[inline(never)]
fn guarded_fallback(
    input: &P3pInput,
    world: [Point3<f64>; 3],
    bearings: [UnitVector3<f64>; 3],
    primary: SolverP3pSolutions,
) -> SolverP3pSolutions {
    let scene_scale = (world[0] - world[1])
        .norm()
        .max((world[0] - world[2]).norm())
        .max((world[1] - world[2]).norm());
    let mut output = SolverP3pSolutions::default();
    append_distinct(
        &mut output,
        &input.fitting_candidates(&primary),
        scene_scale,
    );
    let ke = KeP3pSolver::solve_solver_minimal(world, bearings, KeP3pConfig);
    let previous_len = output.len();
    append_distinct(&mut output, &input.fitting_candidates(&ke), scene_scale);
    if output.len() == previous_len {
        let lambda = LambdaTwistP3pSolver::solve_solver_minimal(world, bearings);
        append_distinct(&mut output, &input.fitting_candidates(&lambda), scene_scale);
    }
    output
}

fn append_distinct(
    output: &mut SolverP3pSolutions,
    fallback: &SolverP3pSolutions,
    scene_scale: f64,
) {
    for candidate in fallback {
        if output.len() == super::common::MAX_P3P_SOLUTIONS {
            break;
        }
        let candidate_center = candidate.camera_center();
        let duplicate = output.iter().any(|existing| {
            existing.rotation.angle_to(&candidate.rotation) < 1e-7
                && (existing.camera_center() - candidate_center).norm()
                    <= scene_scale.max(f64::MIN_POSITIVE) * 1e-6
        });
        if !duplicate {
            output.push(*candidate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{PERMUTATIONS, assert_recovers_truth};
    use super::*;
    use nalgebra::{Isometry3, UnitQuaternion, Vector3};
    use proptest::prelude::*;

    const UNGUARDED: FusedP3pConfig = FusedP3pConfig {
        guarded_fallback: false,
    };

    fn assert_primary_recovers(
        world: [Point3<f64>; 3],
        bearings: [UnitVector3<f64>; 3],
        truth: &Isometry3<f64>,
    ) {
        let candidates = FusedP3pSolver::solve_minimal(world, bearings, UNGUARDED);
        assert_recovers_truth("FusedP3P primary", &candidates, &world, &bearings, truth);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        // Counterpart of the public dispatcher property test with the fallback disabled.
        #[test]
        fn unguarded_primary_recovers_well_conditioned_scenes(
            rotation_vector in prop::array::uniform3(-1.0_f64..1.0),
            translation in prop::array::uniform3(-5.0_f64..5.0),
            scale_log2 in -30_i32..31,
            perturbation in prop::array::uniform3(-0.25_f64..0.25),
            permutation_index in 0_usize..6,
        ) {
            let rotation = UnitQuaternion::from_scaled_axis(Vector3::from(rotation_vector));
            let scale = 2.0_f64.powi(scale_log2);
            let pose = Isometry3::from_parts((Vector3::from(translation) * scale).into(), rotation);
            let camera_points = [
                Point3::new((-1.2 + perturbation[0]) * scale, -0.5 * scale, 4.0 * scale),
                Point3::new(1.1 * scale, (-0.8 + perturbation[1]) * scale, 5.5 * scale),
                Point3::new(0.2 * scale, (1.3 + perturbation[2]) * scale, 6.5 * scale),
            ];
            let world = camera_points.map(|point| pose.inverse_transform_point(&point));
            let bearings = camera_points.map(|point| UnitVector3::new_normalize(point.coords));
            let permutation = PERMUTATIONS[permutation_index];
            assert_primary_recovers(
                permutation.map(|index| world[index]),
                permutation.map(|index| bearings[index]),
                &pose,
            );
        }
    }

    #[test]
    fn unguarded_primary_recovers_reference_scenes() {
        let scenes = [
            [
                Point3::new(-1.290_876_283_094_080_6, -0.5, 4.0),
                Point3::new(1.1, -1.045_265_169_156_624_6, 5.5),
                Point3::new(0.2, 1.3, 6.728_130_915_886_78),
            ],
            [
                Point3::new(-1.317_456_026_700_417_4, -0.5, 4.0),
                Point3::new(1.1, -0.627_585_641_893_859_9, 5.5),
                Point3::new(0.2, 1.3, 6.670_492_730_828_727),
            ],
        ];
        for scene in scenes {
            for scale_log2 in [-30, -28, -9, 0, 30] {
                let scale = 2.0_f64.powi(scale_log2);
                let camera = scene.map(|point| point * scale);
                let bearings = camera.map(|point| UnitVector3::new_normalize(point.coords));
                for permutation in PERMUTATIONS {
                    assert_primary_recovers(
                        permutation.map(|index| camera[index]),
                        permutation.map(|index| bearings[index]),
                        &Isometry3::identity(),
                    );
                }
            }
        }

        let scale = 0.5;
        let truth = Isometry3::from_parts(
            (Vector3::new(0.0, -2.212_220_062_337_294_3, 4.030_728_176_765_445) * scale).into(),
            UnitQuaternion::identity(),
        );
        let camera = [
            Point3::new(-1.333_921_544_049_885_5, -0.5, 4.0) * scale,
            Point3::new(1.1, -0.722_830_053_520_761_9, 5.5) * scale,
            Point3::new(0.2, 1.3, 6.567_597_775_961_706) * scale,
        ];
        let world = camera.map(|point| truth.inverse_transform_point(&point));
        let bearings = camera.map(|point| UnitVector3::new_normalize(point.coords));
        let permutation = [0, 2, 1];
        assert_primary_recovers(
            permutation.map(|index| world[index]),
            permutation.map(|index| bearings[index]),
            &truth,
        );

        let rotation = UnitQuaternion::from_scaled_axis(Vector3::new(0.2, -0.4, 0.3));
        let local_translation = Vector3::new(0.3, -0.2, 0.5);
        for magnitude in [1.0, 1.0e3, 1.0e6, 1.0e7] {
            let offset = Vector3::new(magnitude, -2.0 * magnitude, 3.0 * magnitude);
            let translation = local_translation - rotation * offset;
            let truth = Isometry3::from_parts(translation.into(), rotation);
            let world = [
                Point3::from(offset + Vector3::new(-1.2, -0.5, 4.0)),
                Point3::from(offset + Vector3::new(1.1, -0.8, 5.5)),
                Point3::from(offset + Vector3::new(0.2, 1.3, 6.5)),
            ];
            let bearings =
                world.map(|point| UnitVector3::new_normalize(truth.transform_point(&point).coords));
            assert_primary_recovers(world, bearings, &truth);
        }
    }

    #[test]
    fn guard_detects_exact_triple_root() {
        assert!(near_multiple_root(3.0, 3.0, 1.0));
    }

    #[test]
    fn newton_certificate_implies_common_rotation_tolerance() {
        let rotation = Matrix3::new(1.0, 2e-7, -1e-7, -2e-7, 1.0, 3e-7, 1e-7, -3e-7, 1.0);
        let (corrected, certified, _) = newton_schulz_step(rotation);
        assert!(certified);

        let gram = corrected.transpose() * corrected;
        let error = (gram[(0, 0)] - 1.0).abs()
            + (gram[(1, 1)] - 1.0).abs()
            + (gram[(2, 2)] - 1.0).abs()
            + gram[(0, 1)].abs()
            + gram[(0, 2)].abs()
            + gram[(1, 2)].abs();
        assert!(
            error <= super::super::common::NEAR_ROTATION_TOLERANCE,
            "certified Gram error was {error:e}"
        );
    }

    #[test]
    fn danger_cylinder_activates_the_guard() {
        let world = [
            Point3::new(0.0, 0.0, 4.0),
            Point3::new(1.0, 2.0, 4.0),
            Point3::new(2.0, 0.0, 4.0),
        ];
        let bearings = world.map(|point| UnitVector3::new_normalize(point.coords));
        let input = P3pInput::new(world, bearings).unwrap();

        let (primary, guard_activated) = solve_validated(&input, true);

        assert!(guard_activated);
        let candidates = guarded_fallback(&input, world, bearings, primary);

        assert!(!candidates.is_empty());
        assert_eq!(
            input.fitting_candidates(&candidates).len(),
            candidates.len()
        );
        assert!(
            candidates
                .iter()
                .any(|pose| pose.rotation.angle() < 1e-7
                    && pose.camera_center().coords.norm() < 1e-6)
        );
    }
}
