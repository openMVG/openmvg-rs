// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! LambdaTwist P3P solver (Persson and Nordberg, ECCV 2018).
//!
//! Reference: M. Persson, K. Nordberg, "Lambda Twist: An Accurate Fast Robust
//! Perspective Three Point (P3P) Solver", ECCV 2018.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                       LambdaTwist P3P                                │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: 3 world points + 3 unit bearings
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Distance cubic and eigensystem                               │
//! └──────────────────────────────────────────────────────────────────────┘
//! Solve the distance cubic, polish its root, and diagonalize the singular
//! 3×3 system used by the LambdaTwist parameterization.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Two depth charts and pose                                    │
//! └──────────────────────────────────────────────────────────────────────┘
//! Evaluate both quadratic charts, refine λᵢ, and recover R,t from the
//! resulting world/camera point triples.
//!   │
//!   ▼
//! Output: up to 4 rigid pose candidates after cheirality checks
//! ```
//!
//! This follows the analytical known-zero eigensystem and both depth charts of
//! the reference implementation. The second chart is essential: omitting it
//! makes results depend strongly on the ordering of the three observations.

use crate::SolverResult;
use crate::solver_traits::{MultiSolutionSolver, Solver};
use nalgebra::{Matrix3, Point3, UnitVector3, Vector3};

use super::common::{
    P3pInput, P3pSolutions, SolverP3pSolutions, angular_residual, exact_indexed_sample,
    exact_sample, refine_depths, rotation_from_edges, world_basis_inverse,
};
use crate::{
    pose::{BearingToPoint, SolverPose},
    shared::polynomial::{polished_cubic_root, quadratic_real_monic},
};

/// LambdaTwist calibrated three-point pose solver.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LambdaTwistP3pSolver;

impl LambdaTwistP3pSolver {
    /// Solve one exact minimal sample without heap allocation.
    pub(crate) fn solve_minimal(
        world: [Point3<f64>; 3],
        bearings: [UnitVector3<f64>; 3],
    ) -> P3pSolutions {
        P3pSolutions::from_solver(Self::solve_solver_minimal(world, bearings))
    }

    pub(super) fn solve_solver_minimal(
        world: [Point3<f64>; 3],
        bearings: [UnitVector3<f64>; 3],
    ) -> SolverP3pSolutions {
        let Some(input) = P3pInput::new(world, bearings) else {
            return SolverP3pSolutions::default();
        };
        solve_validated(&input)
    }
}

impl Solver for LambdaTwistP3pSolver {
    type Datum = BearingToPoint;
    type Model = SolverPose<f64>;
    type Config = ();
    type Unit = openmvg_core::Radians;

    #[inline]
    fn min_sample_size() -> usize {
        3
    }

    #[inline]
    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        angular_residual(model, datum)
    }
}

impl MultiSolutionSolver for LambdaTwistP3pSolver {
    fn solve(data: &[Self::Datum], _config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        let (world, bearings) = exact_sample(data)?;
        Ok(Self::solve_solver_minimal(world, bearings).into_vec())
    }

    fn solve_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Vec<Self::Model>> {
        let (world, bearings) = exact_indexed_sample(indices, data)?;
        Ok(Self::solve_solver_minimal(world, bearings).into_vec())
    }
}

fn solve_validated(input: &P3pInput) -> SolverP3pSolutions {
    solve_ordered(input, input.world, input.bearings, true)
}

fn solve_ordered(
    input: &P3pInput,
    world: [Vector3<f64>; 3],
    bearings: [Vector3<f64>; 3],
    retry_singular_chart: bool,
) -> SolverP3pSolutions {
    let mut output = SolverP3pSolutions::default();
    let [p0, p1, p2] = world;
    let [f0, f1, f2] = bearings;
    let d01 = p0 - p1;
    let d02 = p0 - p2;
    let d12 = p1 - p2;
    let a01 = d01.norm_squared();
    let a02 = d02.norm_squared();
    let a12 = d12.norm_squared();
    let c01 = f0.dot(&f1).clamp(-1.0, 1.0);
    let c12 = f1.dot(&f2).clamp(-1.0, 1.0);
    let c20 = f2.dot(&f0).clamp(-1.0, 1.0);
    let triple = c01 * c12 * c20 - 1.0;
    let s20_squared = 1.0 - c20 * c20;
    let s12_squared = 1.0 - c12 * c12;
    let s01_squared = 1.0 - c01 * c01;
    let b01 = -2.0 * c01;
    let b02 = -2.0 * c20;
    let b12 = -2.0 * c12;

    let cubic3 = a02 * (a12 * s20_squared - a02 * s12_squared);
    let cubic2 = 2.0 * triple * a12 * a02
        + a02 * (2.0 * a01 + a02) * s12_squared
        + a12 * (a12 - a01) * s20_squared;
    let cubic1 = a12 * (a02 - a12) * s01_squared
        - a01 * a01 * s12_squared
        - 2.0 * a01 * (triple * a12 + a02 * s12_squared);
    let cubic0 = a01 * (a01 * s12_squared - a12 * s01_squared);
    let coefficient_scale = cubic3
        .abs()
        .max(cubic2.abs())
        .max(cubic1.abs())
        .max(cubic0.abs());
    if !coefficient_scale.is_finite() || cubic3.abs() <= coefficient_scale * f64::EPSILON * 64.0 {
        if retry_singular_chart {
            return solve_best_conditioned_permutation(input);
        }
        return output;
    }
    let Some(g) = polished_cubic_root(cubic2 / cubic3, cubic1 / cubic3, cubic0 / cubic3) else {
        return output;
    };

    let matrix = Matrix3::new(
        a12 * (1.0 - g),
        0.5 * a12 * b01,
        -0.5 * a12 * b02 * g,
        0.5 * a12 * b01,
        a12 - a01 + a02 * g,
        0.5 * b12 * (a02 * g - a01),
        -0.5 * a12 * b02 * g,
        0.5 * b12 * (a02 * g - a01),
        g * (a02 - a12) - a01,
    );
    let Some((eigenvectors, eigenvalues)) = eig_with_known_zero(&matrix) else {
        return output;
    };
    let ratio_squared = -eigenvalues[1] / eigenvalues[0];
    if !ratio_squared.is_finite() || ratio_squared < -f64::EPSILON * 64.0 {
        return output;
    }
    let ratio = ratio_squared.max(0.0).sqrt();

    let mut depths = [Vector3::zeros(); 4];
    let mut depth_count = 0;
    for sign in [ratio, -ratio] {
        let null_vector = eigenvectors.column(0) - sign * eigenvectors.column(1);
        let [u0, u1, u2] = [null_vector[0], null_vector[1], null_vector[2]];
        if u0.abs() < u1.abs() {
            let denominator = (a12 - a01) * u2 * u2 - a01 * u1 * u1 + a01 * b12 * u1 * u2;
            if denominator == 0.0 || !denominator.is_finite() {
                continue;
            }
            let quadratic_b = (2.0 * (a12 - a01) * u0 * u2 + a01 * b12 * u0 * u1
                - a12 * b01 * u1 * u2)
                / denominator;
            let quadratic_c =
                ((a12 - a01) * u0 * u0 + a12 * u1 * u1 - a12 * b01 * u0 * u1) / denominator;
            let Some(roots) = quadratic_real_monic(quadratic_b, quadratic_c) else {
                continue;
            };
            for tau in roots {
                let depth_denominator = tau * (tau + b02) + 1.0;
                if tau <= 0.0 || depth_denominator <= 0.0 {
                    continue;
                }
                let depth0 = (a02 / depth_denominator).sqrt();
                let depth2 = tau * depth0;
                let depth1 = -(u0 * depth0 + u2 * depth2) / u1;
                if depth1 > 0.0 && depth_count < depths.len() {
                    depths[depth_count] = Vector3::new(depth0, depth1, depth2);
                    depth_count += 1;
                }
            }
        } else {
            if u0 == 0.0 || !u0.is_finite() {
                continue;
            }
            let inverse = -u0.recip();
            let w0 = u1 * inverse;
            let w1 = u2 * inverse;
            let denominator = (a02 - a01) * w1 * w1 - a01 * b02 * w1 - a01;
            if denominator == 0.0 || !denominator.is_finite() {
                continue;
            }
            let quadratic_b =
                (a02 * b01 * w1 - a01 * b02 * w0 - 2.0 * w0 * w1 * (a01 - a02)) / denominator;
            let quadratic_c = ((a02 - a01) * w0 * w0 + a02 * b01 * w0 + a02) / denominator;
            let Some(roots) = quadratic_real_monic(quadratic_b, quadratic_c) else {
                continue;
            };
            for tau in roots {
                let depth_denominator = tau * (b12 + tau) + 1.0;
                if tau <= 0.0 || depth_denominator <= 0.0 {
                    continue;
                }
                let depth1 = (a12 / depth_denominator).sqrt();
                let depth2 = tau * depth1;
                let depth0 = w0 * depth1 + w1 * depth2;
                if depth0 > 0.0 && depth_count < depths.len() {
                    depths[depth_count] = Vector3::new(depth0, depth1, depth2);
                    depth_count += 1;
                }
            }
        }
    }

    let Some(world_inverse) = world_basis_inverse(d01, d02) else {
        return output;
    };
    for depth in depths.iter_mut().take(depth_count) {
        refine_depths(depth, [a01, a02, a12], [c01, c20, c12], 5);
        if depth
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            continue;
        }
        let camera0 = depth[0] * f0;
        let camera1 = depth[1] * f1;
        let camera2 = depth[2] * f2;
        let rotation = rotation_from_edges(camera0 - camera1, camera0 - camera2, &world_inverse);
        let translation = camera0 - rotation * p0;
        input.push_edge_basis_pose_from_positive_depths(&mut output, rotation, translation);
    }
    output
}

fn solve_best_conditioned_permutation(input: &P3pInput) -> SolverP3pSolutions {
    let mut best = None;
    for permutation in [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
        let world = permutation.map(|index| input.world[index]);
        let bearings = permutation.map(|index| input.bearings[index]);
        let condition = leading_cubic_condition(world, bearings);
        if best
            .as_ref()
            .is_none_or(|(best_condition, _, _)| condition > *best_condition)
        {
            best = Some((condition, world, bearings));
        }
    }
    let Some((condition, world, bearings)) = best else {
        return SolverP3pSolutions::default();
    };
    if condition <= 64.0 * f64::EPSILON {
        return SolverP3pSolutions::default();
    }
    solve_ordered(input, world, bearings, false)
}

fn leading_cubic_condition(world: [Vector3<f64>; 3], bearings: [Vector3<f64>; 3]) -> f64 {
    let a01 = (world[0] - world[1]).norm_squared();
    let a02 = (world[0] - world[2]).norm_squared();
    let a12 = (world[1] - world[2]).norm_squared();
    let c01 = bearings[0].dot(&bearings[1]).clamp(-1.0, 1.0);
    let c12 = bearings[1].dot(&bearings[2]).clamp(-1.0, 1.0);
    let c20 = bearings[2].dot(&bearings[0]).clamp(-1.0, 1.0);
    let s01_squared = 1.0 - c01 * c01;
    let s12_squared = 1.0 - c12 * c12;
    let s20_squared = 1.0 - c20 * c20;
    let triple = c01 * c12 * c20 - 1.0;
    let coefficients = [
        a02 * (a12 * s20_squared - a02 * s12_squared),
        2.0 * triple * a12 * a02
            + a02 * (2.0 * a01 + a02) * s12_squared
            + a12 * (a12 - a01) * s20_squared,
        a12 * (a02 - a12) * s01_squared
            - a01 * a01 * s12_squared
            - 2.0 * a01 * (triple * a12 + a02 * s12_squared),
        a01 * (a01 * s12_squared - a12 * s01_squared),
    ];
    let scale = coefficients
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        0.0
    } else {
        coefficients[0].abs() / scale
    }
}

/// Analytical eigendecomposition for a symmetric matrix with one known zero
/// eigenvalue. Returns nonzero eigenvalues ordered by decreasing magnitude.
fn eig_with_known_zero(matrix: &Matrix3<f64>) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut zero_vector = Vector3::new(
        matrix[(0, 1)] * matrix[(1, 2)] - matrix[(0, 2)] * matrix[(1, 1)],
        matrix[(0, 2)] * matrix[(0, 1)] - matrix[(1, 2)] * matrix[(0, 0)],
        matrix[(1, 1)] * matrix[(0, 0)] - matrix[(0, 1)] * matrix[(0, 1)],
    );
    let zero_norm_squared = zero_vector.norm_squared();
    let matrix_scale = matrix.norm_squared();
    if zero_norm_squared <= 64.0 * f64::EPSILON * matrix_scale * matrix_scale {
        zero_vector = eigenvector_from_row_crosses(matrix, 0.0)?;
    } else {
        zero_vector /= zero_norm_squared.sqrt();
    }

    let off_diagonal_squared = matrix[(0, 1)] * matrix[(0, 1)];
    let trace_coefficient = -matrix.trace();
    let pair_coefficient =
        -off_diagonal_squared - matrix[(0, 2)] * matrix[(0, 2)] - matrix[(1, 2)] * matrix[(1, 2)]
            + matrix[(0, 0)] * (matrix[(1, 1)] + matrix[(2, 2)])
            + matrix[(1, 1)] * matrix[(2, 2)];
    let [mut first_value, mut second_value] =
        quadratic_real_monic(trace_coefficient, pair_coefficient)?;
    if first_value.abs() < second_value.abs() {
        std::mem::swap(&mut first_value, &mut second_value);
    }
    if first_value == 0.0 || second_value == 0.0 {
        return None;
    }

    let product00_11 = -matrix[(0, 0)] * matrix[(1, 1)];
    let precomputed0 = matrix[(0, 1)] * matrix[(1, 2)] - matrix[(0, 2)] * matrix[(1, 1)];
    let precomputed1 = matrix[(0, 1)] * matrix[(0, 2)] - matrix[(0, 0)] * matrix[(1, 2)];
    let eigenvector = |eigenvalue: f64| -> Option<Vector3<f64>> {
        let denominator = eigenvalue * (matrix[(0, 0)] + matrix[(1, 1)]) + product00_11
            - eigenvalue * eigenvalue
            + off_diagonal_squared;
        let denominator_scale = (eigenvalue * (matrix[(0, 0)] + matrix[(1, 1)])).abs()
            + product00_11.abs()
            + eigenvalue * eigenvalue
            + off_diagonal_squared;
        if denominator.abs() <= 64.0 * f64::EPSILON * denominator_scale.max(f64::MIN_POSITIVE)
            || !denominator.is_finite()
        {
            return eigenvector_from_row_crosses(matrix, eigenvalue);
        }
        let first = -(eigenvalue * matrix[(0, 2)] + precomputed0) / denominator;
        let second = -(eigenvalue * matrix[(1, 2)] + precomputed1) / denominator;
        let inverse_norm = (first * first + second * second + 1.0).sqrt().recip();
        let vector = Vector3::new(first * inverse_norm, second * inverse_norm, inverse_norm);
        vector
            .iter()
            .all(|value| value.is_finite())
            .then_some(vector)
    };
    Some((
        Matrix3::from_columns(&[
            eigenvector(first_value)?,
            eigenvector(second_value)?,
            zero_vector,
        ]),
        Vector3::new(first_value, second_value, 0.0),
    ))
}

fn eigenvector_from_row_crosses(matrix: &Matrix3<f64>, eigenvalue: f64) -> Option<Vector3<f64>> {
    let row0 = Vector3::new(matrix[(0, 0)] - eigenvalue, matrix[(0, 1)], matrix[(0, 2)]);
    let row1 = Vector3::new(matrix[(1, 0)], matrix[(1, 1)] - eigenvalue, matrix[(1, 2)]);
    let row2 = Vector3::new(matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)] - eigenvalue);
    let candidates = [row0.cross(&row1), row0.cross(&row2), row1.cross(&row2)];
    let vector = candidates
        .into_iter()
        .max_by(|left, right| left.norm_squared().total_cmp(&right.norm_squared()))?;
    let norm = vector.norm();
    (norm > f64::MIN_POSITIVE && norm.is_finite()).then_some(vector / norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SolverError;
    use nalgebra::{Rotation3, Unit, UnitQuaternion};

    fn scene(
        scale: f64,
        offset: Vector3<f64>,
    ) -> ([Point3<f64>; 3], [UnitVector3<f64>; 3], SolverPose<f64>) {
        let rotation = UnitVector3::new_normalize(Vector3::new(0.2, -0.4, 0.7));
        let rotation = Rotation3::from_axis_angle(&Unit::new_unchecked(rotation.into_inner()), 0.6);
        let translation = Vector3::new(0.3, -0.2, 0.5) * scale - rotation * offset;
        let pose = SolverPose::from_parts(
            translation.into(),
            UnitQuaternion::from_rotation_matrix(&rotation),
        );
        let world = [
            Point3::from(offset + scale * Vector3::new(-1.0, -0.4, 4.0)),
            Point3::from(offset + scale * Vector3::new(1.2, -0.8, 5.5)),
            Point3::from(offset + scale * Vector3::new(0.1, 1.4, 6.2)),
        ];
        let bearings =
            world.map(|point| UnitVector3::new_normalize(pose.transform_point(&point).coords));
        (world, bearings, pose)
    }

    fn best_error(solutions: &SolverP3pSolutions, truth: &SolverPose<f64>) -> f64 {
        solutions
            .iter()
            .map(|pose| {
                pose.rotation.angle_to(&truth.rotation)
                    + (pose.translation.vector - truth.translation.vector).norm()
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn recovers_pose_for_every_point_permutation() {
        let (world, bearings, truth) = scene(1.0, Vector3::zeros());
        for permutation in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let permuted_world = permutation.map(|index| world[index]);
            let permuted_bearings = permutation.map(|index| bearings[index]);
            let solutions =
                LambdaTwistP3pSolver::solve_solver_minimal(permuted_world, permuted_bearings);
            assert!(
                best_error(&solutions, &truth) < 1e-8,
                "permutation {permutation:?}"
            );
        }
    }

    #[test]
    fn handles_large_offset_and_scale() {
        let offset = Vector3::new(1.0e6, -2.0e6, 3.0e6);
        let (world, bearings, truth) = scene(1.0, offset);
        let solutions = LambdaTwistP3pSolver::solve_solver_minimal(world, bearings);
        let error = best_error(&solutions, &truth);
        assert!(
            error < 1e-3,
            "pose error {error}, {} candidates",
            solutions.len()
        );
    }

    #[test]
    fn exact_solver_rejects_oversized_input() {
        let (world, bearings, _) = scene(1.0, Vector3::zeros());
        let mut correspondences: Vec<_> = world
            .into_iter()
            .zip(bearings)
            .map(|(point, bearing)| BearingToPoint::new(point, bearing))
            .collect();
        correspondences.push(correspondences[0]);
        assert!(matches!(
            LambdaTwistP3pSolver::solve(&correspondences, &()),
            Err(SolverError::InvalidSampleSize { .. })
        ));
    }
}
