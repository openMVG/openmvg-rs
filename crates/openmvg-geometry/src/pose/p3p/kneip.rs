// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Kneip, Scaramuzza and Siegwart's frame P3P solver (CVPR 2011).
//!
//! Reference: L. Kneip, D. Scaramuzza, R. Siegwart, "A Novel Parametrization of
//! the Perspective-Three-Point Problem for a Direct Computation of Absolute
//! Camera Position and Orientation", CVPR 2011.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                         Kneip P3P                                    │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: camera-bearing frame + world-triangle frame
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Solve the cos(θ) quartic                                     │
//! └──────────────────────────────────────────────────────────────────────┘
//! Use the frame reduction to solve the conditioned quartic in cos(θ), then
//! recover the camera center C in the world triangle frame.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Construct pose and test cheirality                           │
//! └──────────────────────────────────────────────────────────────────────┘
//! Construct R and t = −RC, validate bᵢᵀ(RXᵢ+t) > 0, and retain finite roots.
//! Output: up to 4 physically valid poses
//! ```

use crate::SolverResult;
use crate::solver_traits::{MultiSolutionSolver, Solver};
use nalgebra::{Matrix3, Point3, UnitVector3, Vector3};

use super::common::{
    P3pInput, P3pSolutions, SolverP3pSolutions, angular_residual, exact_indexed_sample,
    exact_sample,
};
use crate::{
    pose::{BearingToPoint, SolverPose},
    shared::polynomial::quartic_real_conditioned,
};

/// Kneip et al.'s calibrated P3P frame solver.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct KneipP3pSolver;

impl KneipP3pSolver {
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

impl Solver for KneipP3pSolver {
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

impl MultiSolutionSolver for KneipP3pSolver {
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
    if let Some(chart) = ordered_camera_chart(input.world, input.bearings)
        && chart_is_usable(&chart)
    {
        let output = solve_chart(input, chart);
        if !output.is_empty() {
            return output;
        }
    }

    solve_alternate_charts(input)
}

#[cold]
#[inline(never)]
fn solve_alternate_charts(input: &P3pInput) -> SolverP3pSolutions {
    for permutation in [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
        let Some(chart) = ordered_camera_chart(
            permutation.map(|index| input.world[index]),
            permutation.map(|index| input.bearings[index]),
        ) else {
            continue;
        };
        if chart_is_usable(&chart) {
            let output = solve_chart(input, chart);
            if !output.is_empty() {
                return output;
            }
        }
    }
    SolverP3pSolutions::default()
}

#[inline]
fn chart_is_usable(chart: &CameraChart) -> bool {
    chart.local[2].abs() > f64::EPSILON.sqrt() && chart.local[1].abs() > f64::EPSILON.sqrt()
}

#[inline]
fn solve_chart(input: &P3pInput, chart: CameraChart) -> SolverP3pSolutions {
    let mut output = SolverP3pSolutions::default();
    let [world0, world1, world2] = chart.world;
    let [bearing0, bearing1, _] = chart.bearings;
    let camera_axes = chart.axes;
    let bearing2_local = chart.local;

    let world_edge = world1 - world0;
    let edge_length_squared = world_edge.norm_squared();
    if edge_length_squared <= 0.0 || !edge_length_squared.is_finite() {
        return output;
    }
    let edge_length = edge_length_squared.sqrt();
    let world_axis0 = world_edge / edge_length;
    let world_to_third = world2 - world0;
    let world_normal = world_axis0.cross(&world_to_third);
    let normal_length_squared = world_normal.norm_squared();
    let geometry_scale_squared = edge_length_squared.max(world_to_third.norm_squared());
    if !normal_length_squared.is_finite()
        || normal_length_squared <= 64.0 * f64::EPSILON * geometry_scale_squared
    {
        return output;
    }
    let normal_length = normal_length_squared.sqrt();
    let world_axis2 = world_normal / normal_length;
    let world_axis1 = world_axis2.cross(&world_axis0);
    let point_x = world_axis0.dot(&world_to_third);
    let point_y = world_axis1.dot(&world_to_third);

    let bearing_x = bearing2_local[0] / bearing2_local[2];
    let bearing_y = bearing2_local[1] / bearing2_local[2];
    if bearing_y.abs() <= f64::EPSILON.sqrt() {
        return output;
    }
    let cosine_beta = bearing0.dot(&bearing1).clamp(-1.0, 1.0);
    let sine_beta_squared = 1.0 - cosine_beta * cosine_beta;
    if sine_beta_squared <= f64::EPSILON {
        return output;
    }
    let cotangent_beta = cosine_beta / sine_beta_squared.sqrt();

    let bearing_x2 = bearing_x * bearing_x;
    let bearing_y2 = bearing_y * bearing_y;
    let point_x2 = point_x * point_x;
    let point_x3 = point_x2 * point_x;
    let point_x4 = point_x3 * point_x;
    let point_y2 = point_y * point_y;
    let point_y3 = point_y2 * point_y;
    let point_y4 = point_y3 * point_y;
    let edge_length2 = edge_length * edge_length;
    let cotangent_beta2 = cotangent_beta * cotangent_beta;

    let quartic4 = -point_y4 * (bearing_y2 + bearing_x2 + 1.0);
    let quartic3 = 2.0 * point_y3 * edge_length * cotangent_beta * (1.0 + bearing_y2)
        - 2.0 * bearing_y * point_y3 * bearing_x * edge_length;
    let quartic2 = -bearing_y2 * point_y2 * point_x2
        - bearing_y2 * point_y2 * edge_length2 * cotangent_beta2
        - bearing_y2 * point_y2 * edge_length2
        + bearing_y2 * point_y4
        + point_y4 * bearing_x2
        + 2.0 * point_x * point_y2 * edge_length
        + 2.0 * bearing_x * bearing_y * point_x * point_y2 * edge_length * cotangent_beta
        - point_y2 * point_x2 * bearing_x2
        + 2.0 * point_x * point_y2 * bearing_y2 * edge_length
        - point_y2 * edge_length2 * cotangent_beta2
        - 2.0 * point_x2 * point_y2;
    let quartic1 = 2.0 * point_x2 * point_y * edge_length * cotangent_beta
        + 2.0 * bearing_y * point_y3 * bearing_x * edge_length
        - 2.0 * bearing_y2 * point_y3 * edge_length * cotangent_beta
        - 2.0 * point_x * point_y * edge_length2 * cotangent_beta;
    let quartic0 = -2.0 * bearing_y * point_y2 * bearing_x * point_x * edge_length * cotangent_beta
        + bearing_y2 * point_y2 * edge_length2
        + 2.0 * point_x3 * edge_length
        - point_x2 * edge_length2
        + bearing_y2 * point_y2 * point_x2
        - point_x4
        - 2.0 * bearing_y2 * point_y2 * point_x * edge_length
        + point_y2 * bearing_x2 * point_x2
        + bearing_y2 * point_y2 * edge_length2 * cotangent_beta2;

    let mut roots = [0.0; 4];
    let root_count = quartic_real_conditioned(
        [quartic4, quartic3, quartic2, quartic1, quartic0],
        &mut roots,
    );
    let bearing_x_over_y = bearing_x / bearing_y;
    for cosine_theta in roots.into_iter().take(root_count) {
        if cosine_theta.abs() > 1.0 + 64.0 * f64::EPSILON {
            continue;
        }
        let cosine_theta = cosine_theta.clamp(-1.0, 1.0);
        let sine_theta = (1.0 - cosine_theta * cosine_theta).sqrt();
        let denominator = -bearing_x_over_y * cosine_theta * point_y + point_x - edge_length;
        if denominator == 0.0 || !denominator.is_finite() {
            continue;
        }
        let cotangent_alpha = (-bearing_x_over_y * point_x - cosine_theta * point_y
            + edge_length * cotangent_beta)
            / denominator;
        if !cotangent_alpha.is_finite() {
            continue;
        }
        let inverse_norm = cotangent_alpha.hypot(1.0).recip();
        let sine_alpha = inverse_norm;
        let cosine_alpha = cotangent_alpha * inverse_norm;
        let center_scale = edge_length * (sine_alpha * cotangent_beta + cosine_alpha);
        let center_local = Vector3::new(
            cosine_alpha * center_scale,
            cosine_theta * sine_alpha * center_scale,
            sine_theta * sine_alpha * center_scale,
        );
        let camera_center = world0
            + world_axis0 * center_local[0]
            + world_axis1 * center_local[1]
            + world_axis2 * center_local[2];

        let [camera_axis0, camera_axis1, camera_axis2] = camera_axes;
        let mixed_axis0 = -cosine_alpha * camera_axis0 + sine_alpha * camera_axis1;
        let mixed_axis1 = -sine_alpha * cosine_theta * camera_axis0
            - cosine_alpha * cosine_theta * camera_axis1
            - sine_theta * camera_axis2;
        let mixed_axis2 = -sine_alpha * sine_theta * camera_axis0
            - cosine_alpha * sine_theta * camera_axis1
            + cosine_theta * camera_axis2;
        let rotation = compose_frames(
            [mixed_axis0, mixed_axis1, mixed_axis2],
            [world_axis0, world_axis1, world_axis2],
        );
        let translation = -rotation * camera_center;
        input.push_orthonormal_pose(&mut output, rotation, translation);
    }
    output
}

struct CameraChart {
    world: [Vector3<f64>; 3],
    bearings: [Vector3<f64>; 3],
    axes: [Vector3<f64>; 3],
    local: Vector3<f64>,
}

#[inline]
fn ordered_camera_chart(
    mut world: [Vector3<f64>; 3],
    mut bearings: [Vector3<f64>; 3],
) -> Option<CameraChart> {
    let mut normal = bearings[0].cross(&bearings[1]);
    if normal.dot(&bearings[2]) > 0.0 {
        world.swap(0, 1);
        bearings.swap(0, 1);
        normal = -normal;
    }
    let normal_length = normal.norm();
    if normal_length <= f64::EPSILON.sqrt() || !normal_length.is_finite() {
        return None;
    }
    let axis2 = normal / normal_length;
    let axis1 = axis2.cross(&bearings[0]);
    let local = Vector3::new(
        bearings[0].dot(&bearings[2]),
        axis1.dot(&bearings[2]),
        axis2.dot(&bearings[2]),
    );
    Some(CameraChart {
        world,
        bearings,
        axes: [bearings[0], axis1, axis2],
        local,
    })
}

/// Multiply the frame matrices without materializing two generic matrices.
#[inline]
fn compose_frames(camera: [Vector3<f64>; 3], world: [Vector3<f64>; 3]) -> Matrix3<f64> {
    Matrix3::from_columns(&[
        camera[0] * world[0][0] + camera[1] * world[1][0] + camera[2] * world[2][0],
        camera[0] * world[0][1] + camera[1] * world[1][1] + camera[2] * world[2][1],
        camera[0] * world[0][2] + camera[1] * world[1][2] + camera[2] * world[2][2],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_frame_composition_matches_matrix_product() {
        let camera = [
            Vector3::new(0.8, -0.2, 0.5),
            Vector3::new(0.1, 0.9, -0.3),
            Vector3::new(-0.4, 0.2, 0.7),
        ];
        let world = [
            Vector3::new(0.7, 0.3, -0.1),
            Vector3::new(-0.2, 0.8, 0.4),
            Vector3::new(0.5, -0.3, 0.6),
        ];
        let expected = Matrix3::from_columns(&camera)
            * Matrix3::from_rows(&[
                world[0].transpose(),
                world[1].transpose(),
                world[2].transpose(),
            ]);

        assert!((compose_frames(camera, world) - expected).norm() < 1e-15);
    }
}
