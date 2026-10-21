// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Calibrated pixel and angular solves over fixed intrinsics.

use super::gauge::calibrated_gauge_plan;
use super::reporting::{map_native_error, solve_report};
use super::two_view::{can_use_two_view, solve_two_view};
use super::world_rebase::WorldRebase;
use super::{
    NativePreflight, check_workspace_budget, has_free_intrinsics, pinhole, solver_options,
    workspace_options,
};
use crate::kernel::multi_view::sparse::{
    SparseBundleAdjustmentReport, SparseBundleAdjustmentWorkspace,
    refine_ba_multi_view_sparse_with_factors,
};
use crate::kernel::multi_view::{
    AngularObservation, BundleAdjustmentPriors, CameraPoseVariables, FixedVariableBlocks,
    MultiViewMeasurements, PixelObservation, ScaleGauge,
};
use crate::kernel::{PinholeCamera, RelativePose};
use crate::{BundleProblem, PoseVariables, ResidualSpace, Solution, SolveConfig, SolveError};
use openmvg_core::camera::Camera;
use openmvg_core::model::WorldToCameraPose;

struct PreparedAngularProblem {
    rebase: WorldRebase,
    poses: Vec<RelativePose>,
    points: Vec<[f64; 3]>,
    observations: Vec<AngularObservation>,
    point_offsets: Vec<usize>,
    camera_source_order: Vec<usize>,
    fixed_poses: Vec<bool>,
    fixed_points: Vec<bool>,
    gauge: ScaleGauge,
    pose_masks: Vec<PoseVariables>,
}

impl PreparedAngularProblem {
    fn try_new(problem: &BundleProblem) -> Result<Self, SolveError> {
        if has_free_intrinsics(problem) {
            return Err(SolveError::Unsupported(
                "angular BA requires fixed intrinsics".to_owned(),
            ));
        }
        let gauge_plan = calibrated_gauge_plan(problem)?;
        let rebase =
            WorldRebase::try_new(problem.poses()[gauge_plan.anchor].pose(), gauge_plan.scale)?;
        let camera_source_order = std::iter::once(gauge_plan.anchor)
            .chain((0..problem.poses().len()).filter(|index| *index != gauge_plan.anchor))
            .collect::<Vec<_>>();
        let mut camera_destination = vec![0; problem.poses().len()];
        for (destination, source) in camera_source_order.iter().copied().enumerate() {
            camera_destination[source] = destination;
        }
        let poses = lower_poses(problem, &rebase, &camera_source_order)?;
        let points = lower_points(problem, &rebase)?;
        let mut observation_source_order = (0..problem.observations().len()).collect::<Vec<_>>();
        observation_source_order.sort_by_key(|&source| {
            let observation = &problem.observations()[source];
            (observation.point(), source)
        });
        let observations = observation_source_order
            .iter()
            .map(|&source| {
                let observation = &problem.observations()[source];
                let model = problem.intrinsics()[observation.intrinsic()]
                    .intrinsics()
                    .camera();
                let bearing = model
                    .bearing(observation.pixel())
                    .map_err(|error| SolveError::Numerical(error.to_string()))?;
                Ok(AngularObservation::new(
                    camera_destination[observation.pose()],
                    [bearing.x, bearing.y, bearing.z],
                ))
            })
            .collect::<Result<Vec<_>, SolveError>>()?;
        let mut point_offsets = vec![0; problem.points().len() + 1];
        for observation in problem.observations() {
            point_offsets[observation.point() + 1] += 1;
        }
        for point in 0..problem.points().len() {
            point_offsets[point + 1] += point_offsets[point];
        }
        let fixed_poses = camera_source_order
            .iter()
            .map(|&source| gauge_plan.pose_masks[source].is_fixed())
            .collect();
        let pose_masks = camera_source_order
            .iter()
            .map(|&source| gauge_plan.pose_masks[source])
            .collect();
        Ok(Self {
            rebase,
            poses,
            points,
            observations,
            point_offsets,
            camera_source_order,
            fixed_poses,
            fixed_points: problem
                .points()
                .iter()
                .map(|point| point.is_fixed())
                .collect(),
            gauge: match gauge_plan.native {
                ScaleGauge::FixedBaseline { camera } => ScaleGauge::FixedBaseline {
                    camera: camera_destination[camera],
                },
                ScaleGauge::FixedTranslationComponent { camera, component } => {
                    ScaleGauge::FixedTranslationComponent {
                        camera: camera_destination[camera],
                        component,
                    }
                }
                ScaleGauge::Free => ScaleGauge::Free,
            },
            pose_masks,
        })
    }
}

pub(super) fn solve_angular(
    problem: &BundleProblem,
    config: &SolveConfig,
) -> Result<Solution, SolveError> {
    if can_use_two_view(problem) {
        return solve_two_view(problem, config);
    }
    let mut prepared = PreparedAngularProblem::try_new(problem)?;
    let measurements = MultiViewMeasurements::Angular {
        observations: &prepared.observations,
        point_offsets: &prepared.point_offsets,
    };
    let fixed = FixedVariableBlocks {
        camera_poses: &prepared.fixed_poses,
        points: &prepared.fixed_points,
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    let mut workspace = SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks_and_pose_masks(
        prepared.poses.len(),
        measurements,
        prepared.gauge,
        fixed,
        &prepared.pose_masks,
        workspace_options(config),
    )
    .map_err(map_native_error)?;
    check_workspace_budget(
        "native angular BA",
        workspace.execution_decision().selected_workspace_bytes(),
        config,
    )?;
    let report = refine_ba_multi_view_sparse_with_factors(
        &mut prepared.poses,
        &mut prepared.points,
        measurements,
        prepared.gauge,
        fixed,
        BundleAdjustmentPriors::NONE,
        &mut workspace,
        solver_options(problem, config)?,
        crate::numerics::refine::NoiseModel::Unit,
    )
    .map_err(map_native_error)?;
    raise_solution(
        &prepared.rebase,
        problem,
        &prepared.poses,
        &prepared.points,
        &prepared.camera_source_order,
        &report,
        None,
    )
}

pub(super) struct PreparedCalibratedProblem {
    pub(super) rebase: WorldRebase,
    pub(super) poses: Vec<RelativePose>,
    pub(super) points: Vec<[f64; 3]>,
    pub(super) cameras: Vec<PinholeCamera>,
    pub(super) fixed_perspective_cameras: Vec<Camera>,
    pub(super) observations: Vec<PixelObservation>,
    pub(super) observation_intrinsics: Vec<usize>,
    pub(super) point_offsets: Vec<usize>,
    pub(super) camera_source_order: Vec<usize>,
    #[cfg(test)]
    pub(super) observation_source_order: Vec<usize>,
    pub(super) fixed_poses: Vec<bool>,
    pub(super) fixed_points: Vec<bool>,
    pub(super) gauge: ScaleGauge,
    pub(super) pose_variables: CameraPoseVariables,
    pub(super) pose_masks: Vec<PoseVariables>,
}

impl PreparedCalibratedProblem {
    pub(super) fn try_new(problem: &BundleProblem) -> Result<Self, SolveError> {
        if !matches!(problem.residual_space(), ResidualSpace::Pixels) {
            return Err(SolveError::Unsupported(
                "direct native lowering supports pixel residuals only".to_owned(),
            ));
        }
        if has_free_intrinsics(problem) {
            return Err(SolveError::Unsupported(
                "direct native lowering does not support self-calibration".to_owned(),
            ));
        }
        if problem.observations().iter().any(|observation| {
            matches!(
                problem.intrinsics()[observation.intrinsic()]
                    .intrinsics()
                    .camera(),
                Camera::Fisheye(_) | Camera::Spherical(_)
            )
        }) {
            return Err(SolveError::Unsupported(
                "pixel BA supports fixed pinhole and Brown-Conrady cameras; use angular residuals for wide-angle cameras"
                    .to_owned(),
            ));
        }

        let gauge_plan = calibrated_gauge_plan(problem)?;
        let pose_variables = if gauge_plan
            .pose_masks
            .iter()
            .filter(|mask| !mask.is_fixed())
            .all(|mask| !mask.rotation())
        {
            CameraPoseVariables::TranslationOnly
        } else {
            CameraPoseVariables::RotationAndTranslation
        };
        let rebase =
            WorldRebase::try_new(problem.poses()[gauge_plan.anchor].pose(), gauge_plan.scale)?;
        let camera_source_order = std::iter::once(gauge_plan.anchor)
            .chain((0..problem.poses().len()).filter(|index| *index != gauge_plan.anchor))
            .collect::<Vec<_>>();
        let mut camera_destination = vec![0; problem.poses().len()];
        for (destination, source) in camera_source_order.iter().copied().enumerate() {
            camera_destination[source] = destination;
        }
        let poses = lower_poses(problem, &rebase, &camera_source_order)?;
        let points = lower_points(problem, &rebase)?;
        let observed = super::gauge::observed_cameras(problem);
        let pose_intrinsics = super::single_pose_intrinsics(problem);
        let indexed_models = pose_intrinsics.is_none();
        let uses_full_model = indexed_models
            || problem.observations().iter().any(|row| {
                matches!(
                    problem.intrinsics()[row.intrinsic()].intrinsics().camera(),
                    Camera::BrownConrady(camera) if camera.has_distortion()
                )
            });
        let mut model_destination = vec![usize::MAX; problem.intrinsics().len()];
        let fixed_perspective_cameras = if indexed_models {
            let used = problem
                .observations()
                .iter()
                .map(|row| row.intrinsic())
                .collect::<std::collections::BTreeSet<_>>();
            used.into_iter()
                .enumerate()
                .map(|(destination, source)| {
                    model_destination[source] = destination;
                    problem.intrinsics()[source].intrinsics().camera().clone()
                })
                .collect()
        } else if uses_full_model {
            let mapping = pose_intrinsics.as_ref().ok_or_else(|| {
                SolveError::Internal("missing per-pose calibration mapping".to_owned())
            })?;
            let fallback = problem
                .observations()
                .first()
                .ok_or_else(|| {
                    SolveError::Internal("calibrated problem has no observations".to_owned())
                })?
                .intrinsic();
            camera_source_order
                .iter()
                .map(|&source| {
                    let intrinsic = if observed[source] {
                        mapping[source]
                    } else {
                        fallback
                    };
                    problem.intrinsics()[intrinsic]
                        .intrinsics()
                        .camera()
                        .clone()
                })
                .collect()
        } else {
            Vec::new()
        };
        let cameras = if uses_full_model {
            Vec::new()
        } else {
            let mapping = pose_intrinsics.as_ref().ok_or_else(|| {
                SolveError::Internal("missing per-pose calibration mapping".to_owned())
            })?;
            camera_source_order
                .iter()
                .map(|&source| {
                    if observed[source] {
                        pinhole(problem.intrinsics()[mapping[source]].intrinsics().camera())
                    } else {
                        // This pose contributes no projection. Its declared variables
                        // still remain in the final information and rank assessment.
                        Ok(PinholeCamera::new(1.0, 1.0, 0.0, 0.0))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut observation_source_order = (0..problem.observations().len()).collect::<Vec<_>>();
        observation_source_order.sort_by_key(|&source| {
            let observation = &problem.observations()[source];
            (observation.point(), source)
        });
        let observation_intrinsics = if indexed_models {
            observation_source_order
                .iter()
                .map(|&source| model_destination[problem.observations()[source].intrinsic()])
                .collect()
        } else {
            Vec::new()
        };
        let observations = observation_source_order
            .iter()
            .map(|&source| {
                let observation = &problem.observations()[source];
                PixelObservation::new(
                    camera_destination[observation.pose()],
                    [observation.pixel().x, observation.pixel().y],
                )
            })
            .collect();
        let mut point_offsets = vec![0; problem.points().len() + 1];
        for observation in problem.observations() {
            point_offsets[observation.point() + 1] += 1;
        }
        for point in 0..problem.points().len() {
            point_offsets[point + 1] += point_offsets[point];
        }
        let fixed_poses = camera_source_order
            .iter()
            .map(|&source| gauge_plan.pose_masks[source].is_fixed())
            .collect();
        let fixed_points = problem
            .points()
            .iter()
            .map(|point| point.is_fixed())
            .collect();

        Ok(Self {
            rebase,
            poses,
            points,
            cameras,
            fixed_perspective_cameras,
            observations,
            observation_intrinsics,
            point_offsets,
            #[cfg(test)]
            observation_source_order,
            fixed_poses,
            fixed_points,
            gauge: match gauge_plan.native {
                ScaleGauge::FixedBaseline { camera } => ScaleGauge::FixedBaseline {
                    camera: camera_destination[camera],
                },
                ScaleGauge::FixedTranslationComponent { camera, component } => {
                    ScaleGauge::FixedTranslationComponent {
                        camera: camera_destination[camera],
                        component,
                    }
                }
                ScaleGauge::Free => ScaleGauge::Free,
            },
            pose_variables,
            pose_masks: camera_source_order
                .iter()
                .map(|&source| gauge_plan.pose_masks[source])
                .collect(),
            camera_source_order,
        })
    }
}

fn lower_poses(
    problem: &BundleProblem,
    rebase: &WorldRebase,
    source_order: &[usize],
) -> Result<Vec<RelativePose>, SolveError> {
    let observed = super::gauge::observed_cameras(problem);
    source_order
        .iter()
        .map(|&source| {
            if observed[source] {
                rebase.pose_to_native(problem.poses()[source].pose())
            } else {
                // Inactive poses are copied on output and never enter the native objective.
                Ok(RelativePose::new(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [0.0; 3],
                ))
            }
        })
        .collect()
}

fn lower_points(
    problem: &BundleProblem,
    rebase: &WorldRebase,
) -> Result<Vec<[f64; 3]>, SolveError> {
    let mut observed = vec![false; problem.points().len()];
    for observation in problem.observations() {
        observed[observation.point()] = true;
    }
    problem
        .points()
        .iter()
        .enumerate()
        .map(|(point, block)| {
            if block.is_fixed() && !observed[point] {
                // Survey inventory outside the objective needs no coordinate conversion.
                Ok([0.0; 3])
            } else {
                rebase.point_to_native(block.position())
            }
        })
        .collect()
}

pub(super) fn raise_solution(
    rebase: &WorldRebase,
    problem: &BundleProblem,
    native_poses: &[RelativePose],
    native_points: &[[f64; 3]],
    camera_source_order: &[usize],
    native_report: &SparseBundleAdjustmentReport,
    observability: Option<NativePreflight>,
) -> Result<Solution, SolveError> {
    let mut poses = problem
        .poses()
        .iter()
        .map(|camera| camera.pose().clone())
        .collect::<Vec<_>>();
    let observed = super::gauge::observed_cameras(problem);
    for (native, &source) in native_poses.iter().copied().zip(camera_source_order) {
        let input = &problem.poses()[source];
        if input.variables().is_fixed() || !observed[source] {
            continue;
        }
        let candidate = rebase.pose_from_native(native)?;
        poses[source] = WorldToCameraPose::from_rotation_center(
            if input.variables().rotation() {
                *candidate.rotation()
            } else {
                *input.pose().rotation()
            },
            if input.variables().translation() {
                *candidate.center()
            } else {
                *input.pose().center()
            },
        )
        .map_err(|error| SolveError::Numerical(error.to_string()))?;
    }
    let points = native_points
        .iter()
        .copied()
        .enumerate()
        .map(|(index, point)| {
            if problem.points()[index].is_fixed() {
                Ok(*problem.points()[index].position())
            } else {
                rebase.point_from_native(point)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let intrinsics = problem
        .intrinsics()
        .iter()
        .map(|intrinsic| intrinsic.intrinsics().clone())
        .collect();
    let report = solve_report(native_report, observability, problem.gauge())?
        .with_unconstrained_intrinsics(super::unobserved_free_intrinsics(problem));
    Solution::try_new(problem, poses, intrinsics, points, report)
        .map_err(super::reporting::map_solution_error)
}
