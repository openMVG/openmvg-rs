// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Two-view solves on the relative-pose kernel.

use super::gauge::gauge_plan;
use super::reporting::two_view_solve_report;
use super::world_rebase::WorldRebase;
use super::{pinhole, solver_options};
use crate::kernel::two_view::{TwoViewMeasurements, refine_ba_2view};
use crate::{BundleProblem, ResidualSpace, Solution, SolveConfig, SolveError};

/// Select the relative-pose fast path only when its complete variable/track model fits.
pub(super) fn can_use_two_view(problem: &BundleProblem) -> bool {
    if problem.poses().len() != 2
        || problem.points().iter().any(|point| point.is_fixed())
        || problem.observations().len() != problem.points().len().saturating_mul(2)
        || super::has_free_intrinsics(problem)
        || super::has_repeated_pose_points(problem)
    {
        return false;
    }
    let Some(intrinsic_indices) = super::single_pose_intrinsics(problem) else {
        return false;
    };
    let anchor = match problem.gauge() {
        crate::Gauge::FixedBaseline { first, .. } => first,
        crate::Gauge::Free => {
            let Some(anchor) = problem
                .poses()
                .iter()
                .position(|pose| pose.variables().is_fixed())
            else {
                return false;
            };
            anchor
        }
    };
    if !problem.poses()[anchor].variables().is_fixed()
        || problem.poses()[1 - anchor].variables() != crate::PoseVariables::FREE
    {
        return false;
    }
    matches!(problem.residual_space(), ResidualSpace::Angular)
        || intrinsic_indices.iter().all(|&intrinsic| {
            pinhole(problem.intrinsics()[intrinsic].intrinsics().camera()).is_ok()
        })
}

pub(super) fn solve_two_view(
    problem: &BundleProblem,
    config: &SolveConfig,
) -> Result<Solution, SolveError> {
    let intrinsic_indices = super::single_pose_intrinsics(problem).ok_or_else(|| {
        SolveError::Unsupported("two-view lowering requires one calibration per pose".to_owned())
    })?;
    if super::has_free_intrinsics(problem) {
        return Err(SolveError::Unsupported(
            "direct two-view lowering requires fixed intrinsics".to_owned(),
        ));
    }
    if problem.points().iter().any(|point| point.is_fixed()) {
        return Err(SolveError::Unsupported(
            "direct two-view lowering cannot preserve fixed point blocks".to_owned(),
        ));
    }
    let gauge = gauge_plan(problem)?;
    let variable = (0..2).find(|index| *index != gauge.anchor).ok_or_else(|| {
        SolveError::Internal("two-view problem has no variable camera".to_owned())
    })?;
    if problem.poses()[variable].variables() != crate::PoseVariables::FREE
        || !problem.poses()[gauge.anchor].variables().is_fixed()
    {
        return Err(SolveError::Unsupported(
            "direct two-view lowering requires one fixed anchor and one full-pose camera"
                .to_owned(),
        ));
    }
    let rebase = WorldRebase::try_new(problem.poses()[gauge.anchor].pose(), gauge.scale)?;
    let initial_pose = rebase.pose_to_native(problem.poses()[variable].pose())?;
    let mut points = problem
        .points()
        .iter()
        .map(|point| rebase.point_to_native(point.position()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut observations = [
        vec![None; problem.points().len()],
        vec![None; problem.points().len()],
    ];
    for observation in problem.observations() {
        let view = if observation.pose() == gauge.anchor {
            0
        } else if observation.pose() == variable {
            1
        } else {
            return Err(SolveError::Internal(
                "two-view observation references an unknown camera".to_owned(),
            ));
        };
        if observations[view][observation.point()]
            .replace([observation.pixel().x, observation.pixel().y])
            .is_some()
        {
            return Err(SolveError::Internal(
                "two-view problem contains a duplicate observation".to_owned(),
            ));
        }
    }
    let complete = |view: usize| {
        observations[view]
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                SolveError::Unsupported(
                    "direct two-view lowering requires complete two-view tracks".to_owned(),
                )
            })
    };
    let observations = [complete(0)?, complete(1)?];
    let report = match problem.residual_space() {
        ResidualSpace::Pixels => {
            let cameras = [
                pinhole(
                    problem.intrinsics()[intrinsic_indices[gauge.anchor]]
                        .intrinsics()
                        .camera(),
                )?,
                pinhole(
                    problem.intrinsics()[intrinsic_indices[variable]]
                        .intrinsics()
                        .camera(),
                )?,
            ];
            refine_ba_2view(
                &initial_pose,
                &mut points,
                TwoViewMeasurements::Pixel {
                    observations1: &observations[0],
                    observations2: &observations[1],
                    cameras,
                },
                solver_options(problem, config)?,
                crate::numerics::refine::NoiseModel::Unit,
            )
        }
        ResidualSpace::Angular => {
            let bearings = [
                two_view_bearings(problem, intrinsic_indices[gauge.anchor], &observations[0])?,
                two_view_bearings(problem, intrinsic_indices[variable], &observations[1])?,
            ];
            refine_ba_2view(
                &initial_pose,
                &mut points,
                TwoViewMeasurements::Angular {
                    bearings1: &bearings[0],
                    bearings2: &bearings[1],
                },
                solver_options(problem, config)?,
                crate::numerics::refine::NoiseModel::Unit,
            )
        }
    }
    .map_err(|error| SolveError::Numerical(error.to_string()))?;
    let mut poses = problem
        .poses()
        .iter()
        .map(|camera| camera.pose().clone())
        .collect::<Vec<_>>();
    poses[variable] = rebase.pose_from_native(report.pose)?;
    let points = points
        .into_iter()
        .map(|point| rebase.point_from_native(point))
        .collect::<Result<Vec<_>, _>>()?;
    let intrinsics = problem
        .intrinsics()
        .iter()
        .map(|intrinsic| intrinsic.intrinsics().clone())
        .collect();
    let report = two_view_solve_report(&report)?
        .with_unconstrained_intrinsics(super::unobserved_free_intrinsics(problem));
    Solution::try_new(problem, poses, intrinsics, points, report)
        .map_err(super::reporting::map_solution_error)
}

fn two_view_bearings(
    problem: &BundleProblem,
    intrinsic: usize,
    pixels: &[[f64; 2]],
) -> Result<Vec<[f64; 3]>, SolveError> {
    let model = problem.intrinsics()[intrinsic].intrinsics().camera();
    pixels
        .iter()
        .map(|pixel| {
            let bearing = model
                .bearing(&nalgebra::Point2::from(*pixel))
                .map_err(|error| SolveError::Numerical(error.to_string()))?;
            Ok([bearing.x, bearing.y, bearing.z])
        })
        .collect()
}
