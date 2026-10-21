// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Nakano P3P solver (BMVC 2019).
//!
//! Reference: G. Nakano, "A Simple Direct Solution to the Perspective-Three-Point
//! Problem", BMVC 2019.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                          Nakano P3P                                  │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: 3 world points + 3 unit bearings
//! Longest-side permutation -> orthonormal world frame
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Build constraints and solve quartic                          │
//! └──────────────────────────────────────────────────────────────────────┘
//! Build the two Nakano polynomial constraints and solve the shared scaled
//! quartic with conditioned Ferrari/root polishing.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Recover depths and pose                                      │
//! └──────────────────────────────────────────────────────────────────────┘
//! Recover depths rationally, polish the constraints, and construct R,t.
//! Output: up to 4 rigid poses after positive-depth validation
//! ```

use crate::SolverResult;
use crate::solver_traits::{MultiSolutionSolver, Solver};
use nalgebra::{Point3, UnitVector3, Vector3};

use super::common::{
    P3pInput, P3pSolutions, SolverP3pSolutions, angular_residual, exact_indexed_sample,
    exact_sample, refine_depths, rotation_from_edges, world_basis_inverse,
};
use crate::{
    pose::{BearingToPoint, SolverPose},
    shared::polynomial::{quadratic_real_monic, quartic_real_conditioned},
};

/// Nakano's quartic calibrated P3P solver.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NakanoP3pSolver;

impl NakanoP3pSolver {
    /// Solve one exact minimal sample without heap allocation.
    pub(crate) fn solve_minimal(
        world: [Point3<f64>; 3],
        bearings: [UnitVector3<f64>; 3],
    ) -> P3pSolutions {
        P3pSolutions::from_solver(Self::solve_solver_minimal(world, bearings))
    }

    fn solve_solver_minimal(
        world: [Point3<f64>; 3],
        bearings: [UnitVector3<f64>; 3],
    ) -> SolverP3pSolutions {
        let Some(input) = P3pInput::new(world, bearings) else {
            return SolverP3pSolutions::default();
        };
        solve_validated(&input)
    }
}

impl Solver for NakanoP3pSolver {
    type Datum = BearingToPoint;
    type Model = SolverPose<f64>;
    type Config = ();
    type Unit = openmvg_core::Radians;

    fn min_sample_size() -> usize {
        3
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        angular_residual(model, datum)
    }
}

impl MultiSolutionSolver for NakanoP3pSolver {
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
    let mut output = SolverP3pSolutions::default();
    let mut world = input.world;
    let mut bearings = input.bearings;
    let squared_side_lengths = [
        (world[1] - world[0]).norm_squared(),
        (world[2] - world[1]).norm_squared(),
        (world[0] - world[2]).norm_squared(),
    ];
    let longest = squared_side_lengths
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0);
    match longest {
        1 => {
            world = [world[1], world[2], world[0]];
            bearings = [bearings[1], bearings[2], bearings[0]];
        }
        2 => {
            world.swap(1, 2);
            bearings.swap(1, 2);
        }
        _ => {}
    }

    let edge21 = world[1] - world[0];
    let edge31 = world[2] - world[0];
    let edge_norm_squared = edge21.norm_squared();
    if edge_norm_squared <= 0.0 || !edge_norm_squared.is_finite() {
        return output;
    }
    let edge_norm = edge_norm_squared.sqrt();
    let frame_x = edge21 / edge_norm;
    let frame_z_unnormalized = frame_x.cross(&edge31);
    let frame_z_norm_squared = frame_z_unnormalized.norm_squared();
    let geometry_scale_squared = edge_norm_squared.max(edge31.norm_squared());
    if !frame_z_norm_squared.is_finite()
        || frame_z_norm_squared <= 64.0 * f64::EPSILON * geometry_scale_squared
    {
        return output;
    }
    let frame_z_norm = frame_z_norm_squared.sqrt();
    let frame_z = frame_z_unnormalized / frame_z_norm;
    let frame_y = frame_z.cross(&frame_x);
    let a = frame_x.dot(&edge21);
    let b = frame_x.dot(&edge31);
    let c = frame_y.dot(&edge31);
    if a == 0.0 || c == 0.0 || !a.is_finite() || !c.is_finite() {
        return output;
    }
    let m12 = bearings[0].dot(&bearings[1]).clamp(-1.0, 1.0);
    let m13 = bearings[0].dot(&bearings[2]).clamp(-1.0, 1.0);
    let m23 = bearings[1].dot(&bearings[2]).clamp(-1.0, 1.0);
    let squared_edges = [
        (world[0] - world[1]).norm_squared(),
        (world[0] - world[2]).norm_squared(),
        (world[1] - world[2]).norm_squared(),
    ];
    let bearing_cosines = [m12, m13, m23];
    let Some(world_inverse) = world_basis_inverse(world[0] - world[1], world[0] - world[2]) else {
        return output;
    };
    let p = b / a;
    let q = (b * b + c * c) / (a * a);
    let f1 = p;
    let f2 = -m23;
    let f4 = -m12 * (2.0 * p - 1.0);
    let f5 = m13;
    let f6 = p - 1.0;
    let g1 = q;
    let g4 = -2.0 * m12 * q;
    let g5 = 2.0 * m13;
    let g6 = q - 1.0;

    let h1 = -f1 * f1 + g1 * f2 * f2;
    let h2 = f2 * f2 * g4 - 2.0 * f1 * f4 - 2.0 * f1 * f2 * f5 + 2.0 * f2 * f5 * g1;
    let h3 = f5 * f5 * g1 - 2.0 * f1 * f5 * f5 - 2.0 * f1 * f6 + f2 * f2 * g6
        - f4 * f4
        - 2.0 * f2 * f4 * f5
        + 2.0 * f2 * f5 * g4;
    let h4 =
        f5 * f5 * g4 - 2.0 * f4 * f5 * f5 - 2.0 * f4 * f6 - 2.0 * f2 * f5 * f6 + 2.0 * f2 * f5 * g6;
    let h5 = -2.0 * f5 * f5 * f6 + g6 * f5 * f5 - f6 * f6;
    let mut roots = [0.0; 4];
    let root_count = if f2 == 0.0 && f5 == 0.0 {
        // The first conic is independent of y. Its squared resultant has
        // repeated roots, so solve the conic itself before recovering y.
        if f1 == 0.0 {
            if f4 == 0.0 {
                return output;
            }
            roots[0] = -f6 / f4;
            1
        } else if let Some(pair) = quadratic_real_monic(f4 / f1, f6 / f1) {
            roots[..2].copy_from_slice(&pair);
            if pair[0] == pair[1] { 1 } else { 2 }
        } else {
            0
        }
    } else {
        quartic_real_conditioned([h1, h2, h3, h4, h5], &mut roots)
    };
    for x in roots.into_iter().take(root_count) {
        if x <= 0.0 {
            continue;
        }
        let denominator = f5 + f2 * x;
        if !denominator.is_finite() {
            continue;
        }
        let numerator = (f1 * x + f4) * x + f6;
        let candidates_y = if denominator == 0.0 {
            let scale = f1.abs() * x * x + f4.abs() * x.abs() + f6.abs();
            if !numerator.is_finite()
                || numerator.abs() > 64.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE)
            {
                continue;
            }
            // At a pole of y=-A/B, A=0 and the second conic determines y.
            let constant = (g1 * x + g4) * x + g6;
            let Some(pair) = quadratic_real_monic(-g5, -constant) else {
                continue;
            };
            pair
        } else {
            [-numerator / denominator, f64::NAN]
        };
        for mut y in candidates_y {
            if !y.is_finite() {
                continue;
            }
            let mut x = x;
            let x_squared = x * x;
            let xy = x * y;
            let y_squared = y * y;
            let f_value = f1 * x_squared + f2 * xy + f4 * x + f5 * y + f6;
            let g_value = g1 * x_squared - y_squared + g4 * x + g5 * y + g6;
            let dfdx = 2.0 * f1 * x + f2 * y + f4;
            let dfdy = f2 * x + f5;
            let dgdx = 2.0 * g1 * x + g4;
            let dgdy = -2.0 * y + g5;
            let jacobian_determinant = dfdx * dgdy - dfdy * dgdx;
            if jacobian_determinant.abs() > 64.0 * f64::EPSILON {
                x -= (dgdy * f_value - dfdy * g_value) / jacobian_determinant;
                y -= (-dgdx * f_value + dfdx * g_value) / jacobian_determinant;
            }
            if !x.is_finite() || !y.is_finite() {
                continue;
            }

            let first_constraint = bearings[1] * x - bearings[0];
            let depth_scale = first_constraint.norm() / a;
            if depth_scale <= 0.0 || !depth_scale.is_finite() {
                continue;
            }
            let inverse_depth_scale = depth_scale.recip();
            let mut depths = Vector3::new(
                inverse_depth_scale,
                x * inverse_depth_scale,
                y * inverse_depth_scale,
            );
            refine_depths(&mut depths, squared_edges, bearing_cosines, 1);
            if depths
                .iter()
                .any(|depth| !depth.is_finite() || *depth <= 0.0)
            {
                continue;
            }
            let camera0 = depths[0] * bearings[0];
            let camera1 = depths[1] * bearings[1];
            let camera2 = depths[2] * bearings[2];
            let rotation =
                rotation_from_edges(camera0 - camera1, camera0 - camera2, &world_inverse);
            let translation = camera0 - rotation * world[0];
            input.push_edge_basis_pose_from_positive_depths(&mut output, rotation, translation);
        }
    }
    output
}

#[cfg(test)]
mod orthogonal_ray_contract_tests {
    use super::*;
    use crate::pose::p3p::{P3pMethod, solve};

    #[test]
    fn nakano_recovers_observable_orthogonal_forward_rays() {
        let bearings = [
            Vector3::new(2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0),
            Vector3::new(-1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
            Vector3::new(-2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0),
        ]
        .map(UnitVector3::new_normalize);
        let depths = [3.0, 4.0, 5.0];
        let world =
            std::array::from_fn(|index| Point3::from(bearings[index].into_inner() * depths[index]));
        let poses = solve(world, bearings, P3pMethod::Nakano);
        assert!(
            poses.iter().any(|pose| {
                pose.center().coords.norm() < 1e-6 && pose.rotation().angle() < 1e-6
            }),
            "the three squared-depth equations have the unique positive solution (3,4,5)"
        );
        for pose in poses {
            for (point, bearing) in world.iter().zip(&bearings) {
                let camera = pose.project_to_camera(point).coords;
                assert!(camera.dot(bearing.as_ref()) > 0.0);
                assert!(
                    camera
                        .cross(bearing.as_ref())
                        .norm()
                        .atan2(camera.dot(bearing.as_ref()))
                        < 2e-6
                );
            }
        }
    }
}
