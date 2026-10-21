// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ding et al. P3P solver (CVPR 2023).
//!
//! Reference: Y. Ding, J. Yang, V. Larsson, C. Olsson, K. Åström, "Revisiting
//! the P3P Problem", CVPR 2023.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                         Ding P3P                                     │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: 3 world points Xᵢ and 3 unit bearings bᵢ
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Order sides and build conic pencil                           │
//! └──────────────────────────────────────────────────────────────────────┘
//! Choose the longest world-triangle side for conditioning. Eliminate pose
//! variables to obtain the conic-pencil cubic and its line factors.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Solve depth charts                                           │
//! └──────────────────────────────────────────────────────────────────────┘
//! Each line factor yields a quadratic depth chart. Solve both charts,
//! refine λᵢ in bᵢ λᵢ = R Xᵢ + t, and recover the rigid pose.
//!   │
//!   ▼
//! Output: up to 4 positive-depth rigid pose candidates
//!
//! Complexity: fixed-size polynomial work; no dynamic sample allocation.
//! ```

use crate::SolverResult;
use crate::solver_traits::{MultiSolutionSolver, Solver};
use nalgebra::{Matrix3, Point3, UnitVector3, Vector3};

use super::common::{
    P3pInput, P3pSolutions, SolverP3pSolutions, angular_residual, exact_indexed_sample,
    exact_sample, refine_depths, rotation_from_edges, world_basis_inverse,
};
use crate::{
    pose::{BearingToPoint, SolverPose},
    shared::polynomial::{cubic_real_branch, quadratic_real_monic},
};

/// Ding's conic-cubic calibrated P3P solver.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DingP3pSolver;

impl DingP3pSolver {
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

impl Solver for DingP3pSolver {
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

impl MultiSolutionSolver for DingP3pSolver {
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
    let Some(problem) = DingProblem::new(input) else {
        return output;
    };
    let [cubic_b, cubic_c, cubic_d] = problem.pencil_cubic();
    let Some((root, single_real)) = cubic_real_branch(cubic_b, cubic_c, cubic_d) else {
        return output;
    };
    let Some(factors) = problem.line_factors(root) else {
        return output;
    };

    for factor in factors {
        for mut depths in problem.depths(factor).into_iter().flatten() {
            refine_depths(
                &mut depths,
                problem.squared_edges,
                problem.bearing_cosines,
                5,
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
            let rotation =
                rotation_from_edges(camera0 - camera1, camera0 - camera2, &problem.world_inverse);
            let translation = camera0 - rotation * problem.world[0];
            input.push_edge_basis_pose_from_positive_depths(&mut output, rotation, translation);
        }
        // The single-real-root branch yields one relevant line factor once it
        // has produced a pose (Ding et al.'s reference dispatch).
        if single_real && !output.is_empty() {
            break;
        }
    }
    output
}

/// Shared algebraic front end for Ding and FusedP3P.
pub(super) struct DingProblem {
    pub(super) world: [Vector3<f64>; 3],
    pub(super) bearings: [Vector3<f64>; 3],
    pub(super) squared_edges: [f64; 3],
    pub(super) bearing_cosines: [f64; 3],
    pub(super) world_inverse: Matrix3<f64>,
    normalized_edges: [f64; 2],
}

impl DingProblem {
    #[inline]
    pub(super) fn new(input: &P3pInput) -> Option<Self> {
        let mut world = input.world;
        let mut bearings = input.bearings;
        let mut edge01 = world[0] - world[1];
        let mut edge02 = world[0] - world[2];
        let edge12 = world[1] - world[2];
        let mut squared_edges = [
            edge01.norm_squared(),
            edge02.norm_squared(),
            edge12.norm_squared(),
        ];

        // Put the longest world side at index 12, as required by the pencil.
        if squared_edges[0] > squared_edges[1] {
            if squared_edges[0] > squared_edges[2] {
                world.swap(0, 2);
                bearings.swap(0, 2);
                squared_edges.swap(0, 2);
                edge01 = -edge12;
                edge02 = -edge02;
            }
        } else if squared_edges[1] > squared_edges[2] {
            world.swap(0, 1);
            bearings.swap(0, 1);
            squared_edges.swap(1, 2);
            edge01 = -edge01;
            edge02 = edge12;
        }
        let longest = squared_edges[2];
        if longest <= 0.0 || !longest.is_finite() {
            return None;
        }
        let world_inverse = world_basis_inverse(edge01, edge02)?;
        let bearing_cosines = [
            bearings[0].dot(&bearings[1]).clamp(-1.0, 1.0),
            bearings[0].dot(&bearings[2]).clamp(-1.0, 1.0),
            bearings[1].dot(&bearings[2]).clamp(-1.0, 1.0),
        ];
        Some(Self {
            world,
            bearings,
            squared_edges,
            bearing_cosines,
            world_inverse,
            normalized_edges: [squared_edges[0] / longest, squared_edges[1] / longest],
        })
    }

    #[inline]
    pub(super) fn pencil_cubic(&self) -> [f64; 3] {
        let [a, b] = self.normalized_edges;
        let [m01, m02, m12] = self.bearing_cosines;
        let m12_squared = 1.0 - m12 * m12;
        let m02_squared = m02 * m02 - 1.0;
        let m01_squared = m01 * m01 - 1.0;
        let ab = a * b;
        let triple = -2.0 + 2.0 * m01 * m02 * m12;
        let denominator = b * b * m12_squared + b * m02_squared;
        let inverse = denominator.recip();
        [
            ((a - 1.0) * m02_squared + 2.0 * ab * m12_squared + b * b * m12_squared + b * triple)
                * inverse,
            (a * a * m12_squared + 2.0 * ab * m12_squared + a * triple + (b - 1.0) * m01_squared)
                * inverse,
            (a * a * m12_squared + a * m01_squared) * inverse,
        ]
    }

    #[inline]
    pub(super) fn line_factors(&self, root: f64) -> Option<[Vector3<f64>; 2]> {
        let [a, b] = self.normalized_edges;
        let [m01, m02, m12] = self.bearing_cosines;
        let c00 = -a + root * (1.0 - b);
        let c01 = -m02 * root;
        let c02 = a * m12 + b * m12 * root;
        let c11 = root + 1.0;
        let c12 = -m01;
        let c22 = -a - b * root + 1.0;

        let adjugate00 = c12 * c12 - c11 * c22;
        let adjugate11 = c02 * c02 - c00 * c22;
        let adjugate22 = c01 * c01 - c00 * c11;
        let adjugate01 = c01 * c22 - c02 * c12;
        let adjugate02 = c02 * c11 - c01 * c12;
        let adjugate12 = c00 * c12 - c02 * c01;
        let (diagonal, column) = if adjugate00 > adjugate11 {
            if adjugate00 > adjugate22 {
                (adjugate00, Vector3::new(adjugate00, adjugate01, adjugate02))
            } else {
                (adjugate22, Vector3::new(adjugate02, adjugate12, adjugate22))
            }
        } else if adjugate11 > adjugate22 {
            (adjugate11, Vector3::new(adjugate01, adjugate11, adjugate12))
        } else {
            (adjugate22, Vector3::new(adjugate02, adjugate12, adjugate22))
        };
        if diagonal <= 0.0 || !diagonal.is_finite() {
            return None;
        }
        let vector = column / diagonal.sqrt();
        Some([
            Vector3::new(c00, c01 + vector[2], c02 - vector[1]),
            Vector3::new(c00, c01 - vector[2], c02 + vector[1]),
        ])
    }

    #[inline]
    pub(super) fn depths(&self, factor: Vector3<f64>) -> [Option<Vector3<f64>>; 2] {
        let [a, b] = self.normalized_edges;
        let [a01, _, a12] = self.squared_edges;
        let [m01, m02, m12] = self.bearing_cosines;
        let use_second_chart = factor[0].abs() <= factor[1].abs();
        let (w0, w1, quadratic_b, quadratic_c) = if use_second_chart {
            if factor[1] == 0.0 {
                return [None, None];
            }
            let w0 = -factor[0] / factor[1];
            let w1 = -factor[2] / factor[1];
            let denominator = w1 * w1 - b;
            (
                w0,
                w1,
                2.0 * (b * m12 - m02 * w1 + w0 * w1) / denominator,
                (w0 * w0 - 2.0 * m02 * w0 - b + 1.0) / denominator,
            )
        } else {
            if factor[0] == 0.0 {
                return [None, None];
            }
            let w0 = -factor[1] / factor[0];
            let w1 = -factor[2] / factor[0];
            let denominator = -a * w1 * w1 + 2.0 * a * m12 * w1 - a + 1.0;
            (
                w0,
                w1,
                2.0 * (a * m12 * w0 - m01 - a * w0 * w1) / denominator,
                (1.0 - a * w0 * w0) / denominator,
            )
        };
        let Some(roots) = quadratic_real_monic(quadratic_b, quadratic_c) else {
            return [None, None];
        };
        roots.map(|tau| {
            if tau <= 0.0 {
                return None;
            }
            let depths = if use_second_chart {
                let denominator = tau * (tau - 2.0 * m12) + 1.0;
                if denominator <= 0.0 {
                    return None;
                }
                let depth2 = (a12 / denominator).sqrt();
                let depth1 = tau * depth2;
                Vector3::new(w0 * depth2 + w1 * depth1, depth1, depth2)
            } else {
                let denominator = tau * (tau - 2.0 * m01) + 1.0;
                if denominator <= 0.0 {
                    return None;
                }
                let depth0 = (a01 / denominator).sqrt();
                let depth1 = tau * depth0;
                Vector3::new(depth0, depth1, w0 * depth0 + w1 * depth1)
            };
            depths
                .iter()
                .all(|depth| depth.is_finite() && *depth > 0.0)
                .then_some(depths)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SolverError;

    #[test]
    fn indexed_p3p_reports_out_of_bounds_samples() {
        let world = [
            Point3::new(-1.2, -0.5, 4.0),
            Point3::new(1.1, -0.8, 5.5),
            Point3::new(0.2, 1.3, 6.5),
        ];
        let bearings = world.map(|point| UnitVector3::new_normalize(point.coords));
        let data: Vec<_> = world
            .into_iter()
            .zip(bearings)
            .map(|(point, bearing)| BearingToPoint::try_new(point, bearing).unwrap())
            .collect();
        let error = DingP3pSolver::solve_from_indices(&[0, 1, 3], &data, &()).unwrap_err();
        assert_eq!(
            error,
            SolverError::InvalidDataIndex {
                index: 3,
                data_len: 3,
            }
        );
    }
}
