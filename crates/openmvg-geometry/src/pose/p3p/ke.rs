// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ke and Roumeliotis P3P solver (CVPR 2017).
//!
//! Reference: T. Ke, S. I. Roumeliotis, "An Efficient Algebraic Solution to the
//! Perspective-Three-Point Problem", CVPR 2017.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                           Ke P3P                                     │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: world triangle and bearing frame
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Build and solve conic quartic                                │
//! └──────────────────────────────────────────────────────────────────────┘
//! Form the Ke conic equations, solve the conditioned quartic, and retain
//! finite real roots for the angle parameter.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Recover pose and check depth                                 │
//! └──────────────────────────────────────────────────────────────────────┘
//! Enforce cos²θ + sin²θ = 1, recover R and t, then reject candidates with
//! a non-positive depth.
//!   │
//!   ▼
//! Output: up to 4 SO(3) poses after cheirality validation
//! ```
//!
//! The normalized and original algebraic readouts share one implementation.
//! Enforcing `cos^2 + sin^2 = 1` is the default because public poses must lie
//! on `SO(3)`, and finite-precision quartic roots do not satisfy it exactly.

use crate::SolverResult;
use crate::solver_traits::{MultiSolutionSolver, Solver};
use nalgebra::{Matrix3, Point3, UnitVector3};

use super::common::{
    P3pInput, P3pSolutions, SolverP3pSolutions, angular_residual, exact_indexed_sample,
    exact_sample,
};
use crate::{
    pose::{BearingToPoint, SolverPose},
    shared::polynomial::quartic_real_conditioned,
};

/// Options for [`KeP3pSolver`]; the solver has none.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct KeP3pConfig;

/// Ke and Roumeliotis' conic-quartic calibrated P3P solver.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct KeP3pSolver;

impl KeP3pSolver {
    /// Solve one exact minimal sample without heap allocation.
    pub(crate) fn solve_minimal(
        world: [Point3<f64>; 3],
        bearings: [UnitVector3<f64>; 3],
        config: KeP3pConfig,
    ) -> P3pSolutions {
        P3pSolutions::from_solver(Self::solve_solver_minimal(world, bearings, config))
    }

    pub(super) fn solve_solver_minimal(
        world: [Point3<f64>; 3],
        bearings: [UnitVector3<f64>; 3],
        _config: KeP3pConfig,
    ) -> SolverP3pSolutions {
        let Some(input) = P3pInput::new(world, bearings) else {
            return SolverP3pSolutions::default();
        };
        solve_validated(&input)
    }
}

impl Solver for KeP3pSolver {
    type Datum = BearingToPoint;
    type Model = SolverPose<f64>;
    type Config = KeP3pConfig;
    type Unit = openmvg_core::Radians;

    fn min_sample_size() -> usize {
        3
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        angular_residual(model, datum)
    }
}

impl MultiSolutionSolver for KeP3pSolver {
    fn solve(data: &[Self::Datum], config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        let (world, bearings) = exact_sample(data)?;
        Ok(Self::solve_solver_minimal(world, bearings, *config).into_vec())
    }

    fn solve_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Vec<Self::Model>> {
        let (world, bearings) = exact_indexed_sample(indices, data)?;
        Ok(Self::solve_solver_minimal(world, bearings, *config).into_vec())
    }
}

/// Recovers the second angle pair of Ke's equations 37-38 as a normalized
/// sine/cosine pair before pose construction.
pub(super) fn solve_validated(input: &P3pInput) -> SolverP3pSolutions {
    let mut output = SolverP3pSolutions::default();
    let [world0, world1, world2] = input.world;
    let [bearing0, bearing1, bearing2] = input.bearings;

    let world_edge = world0 - world1;
    let edge_norm_squared = world_edge.norm_squared();
    if edge_norm_squared <= 0.0 || !edge_norm_squared.is_finite() {
        return output;
    }
    let edge_norm = edge_norm_squared.sqrt();
    let k1 = world_edge / edge_norm;
    let bearing_cross = bearing0.cross(&bearing1);
    let bearing_cross_norm = bearing_cross.norm();
    if bearing_cross_norm <= f64::EPSILON.sqrt() {
        return output;
    }
    let k3 = bearing_cross / bearing_cross_norm;
    let tangent = bearing0.cross(&k3);
    let bearing0_cross_2 = bearing0.cross(&bearing2);
    let bearing1_cross_2 = bearing1.cross(&bearing2);
    let world_edge2 = world0 - world2;
    let projection = world_edge2.dot(&k1);
    let k3_dot_bearing2 = k3.dot(&bearing2);
    if k3_dot_bearing2.abs() <= f64::EPSILON.sqrt() {
        return output;
    }

    let mut f11 = k3_dot_bearing2;
    let mut f13 = k3.dot(&bearing0_cross_2);
    let f15 = -projection * f11;
    let normal_unnormalized = world_edge2.cross(&k1);
    let normal_norm_squared = normal_unnormalized.norm_squared();
    let geometry_scale_squared = edge_norm_squared.max(world_edge2.norm_squared());
    if !normal_norm_squared.is_finite()
        || normal_norm_squared <= 64.0 * f64::EPSILON * geometry_scale_squared
    {
        return output;
    }
    let delta = normal_norm_squared.sqrt();
    let normal = normal_unnormalized / delta;
    f11 *= delta;
    f13 *= delta;

    let projection2 = projection - edge_norm;
    let mut f21 = tangent.dot(&bearing1_cross_2);
    let mut f22 = bearing_cross_norm * k3_dot_bearing2;
    let mut f23 = k3.dot(&bearing1_cross_2);
    let f24 = projection2 * f22;
    let f25 = -projection2 * f21;
    f21 *= delta;
    f22 *= delta;
    f23 *= delta;

    let g1 = f13 * f22;
    let g2 = f13 * f25 - f15 * f23;
    let g3 = f11 * f23 - f13 * f21;
    let g4 = -f13 * f24;
    let g5 = f11 * f22;
    let g6 = f11 * f25 - f15 * f21;
    let g7 = -f15 * f24;
    let quartic4 = g5 * g5 + g1 * g1 + g3 * g3;
    let quartic3 = 2.0 * (g5 * g6 + g1 * g2 + g3 * g4);
    let quartic2 = g6 * g6 + 2.0 * g5 * g7 + g2 * g2 + g4 * g4 - g1 * g1 - g3 * g3;
    let quartic1 = 2.0 * (g6 * g7 - g1 * g2 - g3 * g4);
    let quartic0 = g7 * g7 - g2 * g2 - g4 * g4;

    let mut roots = [0.0; 4];
    let root_count = quartic_real_conditioned(
        [quartic4, quartic3, quartic2, quartic1, quartic0],
        &mut roots,
    );
    let world_frame = Matrix3::from_columns(&[k1, normal, k1.cross(&normal)]);
    let bearing_frame =
        Matrix3::from_rows(&[bearing0.transpose(), k3.transpose(), tangent.transpose()]);
    let translated_bearing = bearing2 * (delta / k3_dot_bearing2);
    let sine_sign = k3_dot_bearing2.signum();

    for cosine1 in roots.into_iter().take(root_count) {
        if cosine1.abs() > 1.0 + f64::EPSILON * 64.0 {
            continue;
        }
        let cosine1 = cosine1.clamp(-1.0, 1.0);
        let sine1 = sine_sign * (1.0 - cosine1 * cosine1).sqrt();
        if sine1 == 0.0 {
            continue;
        }
        let cosine_numerator = g1 * cosine1 + g2;
        let sine_numerator = g3 * cosine1 + g4;
        let squared_norm = cosine_numerator * cosine_numerator + sine_numerator * sine_numerator;
        let denominator = (g5 * cosine1 + g6) * cosine1 + g7;
        if squared_norm <= 0.0
            || !squared_norm.is_finite()
            || denominator == 0.0
            || !denominator.is_finite()
        {
            continue;
        }
        let sign = if denominator * sine1 >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let inverse = sign / squared_norm.sqrt();
        let (cosine3, sine3) = (cosine_numerator * inverse, sine_numerator * inverse);
        let angle_matrix = Matrix3::new(
            cosine3,
            0.0,
            -sine3,
            sine1 * sine3,
            cosine1,
            sine1 * cosine3,
            cosine1 * sine3,
            -sine1,
            cosine1 * cosine3,
        );
        let rotation = (world_frame * angle_matrix * bearing_frame).transpose();
        let translation = translated_bearing * sine1 - rotation * world2;
        // `R * world2 + t` is `bearing2 * delta * |sin(theta)| /
        // |k3.dot(bearing2)|`, so observation 2 is strictly cheiral.
        input.push_orthonormal_pose_with_positive_third_depth(&mut output, rotation, translation);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::super::test_support::assert_recovers_all_permutations;
    use super::*;

    #[test]
    fn recovery_is_physical_for_every_input_order() {
        let world = [
            Point3::new(-1.2, -0.5, 4.0),
            Point3::new(1.1, -0.8, 5.5),
            Point3::new(0.2, 1.3, 6.5),
        ];
        assert_recovers_all_permutations("Ke", world, |world, bearings| {
            KeP3pSolver::solve_minimal(world, bearings, KeP3pConfig)
        });
    }
}
