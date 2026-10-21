// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free dense-Schur optimizer for BAL-style camera blocks.

#[cfg(feature = "unstable-internals")]
use crate::numerics::refine::nonlinear::{LmRunState, StepRejection, final_stage_index, stages};
#[cfg(feature = "unstable-internals")]
use crate::numerics::refine::{NoiseModel, Options};

use super::super::camera_kernel::CameraKernel;
use super::super::model::ObservationModel;
use super::super::optimizer::undamped_model_reduction;
#[cfg(feature = "unstable-internals")]
use super::super::{
    BundleAdjustmentPriors, FixedVariableBlocks, MultiViewBundleAdjustmentError,
    MultiViewBundleAdjustmentReport, ScaleGauge, dense,
    optimizer::{norm, retract_poses, scene_scale, squared_norm},
    schur::{self, GaugeState},
    validation,
};
#[cfg(feature = "unstable-internals")]
use super::SelfCalibrationWorkspace;
use super::{RadialCamera, SelfCalibratingMeasurements};
#[cfg(feature = "unstable-internals")]
use crate::kernel::RelativePose;

const INTRINSIC_DIMENSION: usize = 3;

pub(in crate::kernel::multi_view) fn exact_model_reduction(
    point_model_reduction: f64,
    reduced_gradient: &[f64],
    diagonal: &[f64],
    step: &[f64],
    damping: f64,
    point_damping_norm: f64,
) -> f64 {
    let damped_reduction = point_model_reduction
        - 0.5
            * reduced_gradient
                .iter()
                .zip(step)
                .map(|(gradient, step)| gradient * step)
                .sum::<f64>();
    undamped_model_reduction(
        damped_reduction,
        diagonal,
        step,
        damping,
        point_damping_norm,
    )
}

/// Jointly refine camera poses, BAL-style intrinsics, and 3D points.
///
/// Camera 0's pose is fixed, but its focal length and radial coefficients are
/// refined. Focal length uses a log retraction; `k1` and `k2` use additive
/// updates. The caller-selected [`ScaleGauge`] has the same semantics as in
/// calibrated multi-view BA. Workspace construction may allocate; the solve
/// itself does not.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "unstable-internals")]
pub fn refine_ba_multi_view_self_calibrating_dense(
    poses: &mut [RelativePose],
    points: &mut [[f64; 3]],
    cameras: &mut [RadialCamera],
    measurements: SelfCalibratingMeasurements<'_>,
    gauge: ScaleGauge,
    workspace: &mut SelfCalibrationWorkspace,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentReport, MultiViewBundleAdjustmentError> {
    refine(
        poses,
        points,
        cameras,
        None,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        BundleAdjustmentPriors::NONE,
        workspace,
        options,
        noise,
    )
}

/// Jointly refine per-view radial cameras with exact block fixes and priors.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "unstable-internals")]
pub fn refine_ba_multi_view_self_calibrating_dense_with_factors(
    poses: &mut [RelativePose],
    points: &mut [[f64; 3]],
    cameras: &mut [RadialCamera],
    measurements: SelfCalibratingMeasurements<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut SelfCalibrationWorkspace,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentReport, MultiViewBundleAdjustmentError> {
    refine(
        poses,
        points,
        cameras,
        None,
        measurements,
        gauge,
        fixed,
        priors,
        workspace,
        options,
        noise,
    )
}

/// Jointly refine poses, 3D points, and shared BAL-style intrinsic groups.
///
/// `camera_to_group` contains one group index per pose. Multiple views may
/// reference the same entry of `intrinsic_groups`; each group is then a single
/// three-parameter `(log focal, k1, k2)` block in the reduced system. Every
/// group must be used. The solve remains allocation-free with a sufficiently
/// large [`SelfCalibrationWorkspace`].
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "unstable-internals")]
pub fn refine_ba_multi_view_self_calibrating_grouped_dense(
    poses: &mut [RelativePose],
    points: &mut [[f64; 3]],
    intrinsic_groups: &mut [RadialCamera],
    camera_to_group: &[usize],
    measurements: SelfCalibratingMeasurements<'_>,
    gauge: ScaleGauge,
    workspace: &mut SelfCalibrationWorkspace,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentReport, MultiViewBundleAdjustmentError> {
    refine(
        poses,
        points,
        intrinsic_groups,
        Some(camera_to_group),
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        BundleAdjustmentPriors::NONE,
        workspace,
        options,
        noise,
    )
}

/// Jointly refine grouped radial intrinsics with exact block fixes and priors.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "unstable-internals")]
pub fn refine_ba_multi_view_self_calibrating_grouped_dense_with_factors(
    poses: &mut [RelativePose],
    points: &mut [[f64; 3]],
    intrinsic_groups: &mut [RadialCamera],
    camera_to_group: &[usize],
    measurements: SelfCalibratingMeasurements<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut SelfCalibrationWorkspace,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentReport, MultiViewBundleAdjustmentError> {
    refine(
        poses,
        points,
        intrinsic_groups,
        Some(camera_to_group),
        measurements,
        gauge,
        fixed,
        priors,
        workspace,
        options,
        noise,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "unstable-internals")]
fn refine(
    poses: &mut [RelativePose],
    points: &mut [[f64; 3]],
    cameras: &mut [RadialCamera],
    camera_to_group: Option<&[usize]>,
    measurements: SelfCalibratingMeasurements<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut SelfCalibrationWorkspace,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentReport, MultiViewBundleAdjustmentError> {
    options.try_validate()?;
    if !(workspace.dense.initial_damping.is_finite() && workspace.dense.initial_damping > 0.0) {
        return Err(MultiViewBundleAdjustmentError::InitialDampingInvalid);
    }
    let model = radial_model(measurements, cameras, camera_to_group);
    noise.try_validate(model.observation_count())?;
    validation::problem(poses, points, model, gauge, &workspace.dense)?;
    validation::factors(poses, points, model, gauge, fixed, priors)?;
    let dimension = if camera_to_group.is_some() {
        schur::configure_grouped_layout_with_fixed(
            &mut workspace.dense,
            poses.len(),
            gauge,
            cameras.len(),
            fixed,
        )
    } else {
        schur::configure_layout_with_fixed(
            &mut workspace.dense,
            poses.len(),
            gauge,
            INTRINSIC_DIMENSION,
            fixed,
        )
    };
    let model = radial_model(measurements, cameras, camera_to_group);
    validation::projection_domain(model, poses, points)?;
    let normalization_scale = scene_scale(poses, points, gauge);
    let (observation_cost, initial_rmse) =
        schur::metrics(model, poses, points, options.loss, noise);
    let initial_cost =
        observation_cost + schur::prior_cost(model, poses, points, priors).unwrap_or(f64::INFINITY);
    let fixed_baseline = match gauge {
        ScaleGauge::Free => None,
        ScaleGauge::FixedBaseline { camera } => Some(norm(&poses[camera].translation)),
        ScaleGauge::FixedTranslationComponent { .. } => None,
    };
    let fixed_translation_component = match gauge {
        ScaleGauge::FixedTranslationComponent { camera, component } => {
            Some(poses[camera].translation[component.index()])
        }
        ScaleGauge::Free | ScaleGauge::FixedBaseline { .. } => None,
    };
    let schedule = stages(options);
    let target_stage = final_stage_index(&schedule);
    let mut run = LmRunState::with_initial_damping(workspace.dense.initial_damping);

    'optimization: for (stage_index, stage) in schedule.into_iter().enumerate() {
        for _ in 0..stage.max_iterations {
            run.begin_iteration();
            let gauge_state = GaugeState::new(gauge, poses, normalization_scale);
            let assembly = schur::assemble_problem(
                schur::BaProblemView {
                    model: radial_model(measurements, cameras, camera_to_group),
                    poses,
                    points,
                    fixed,
                    priors,
                },
                schur::LinearizationPolicy {
                    gauge: gauge_state,
                    loss: stage.loss,
                    noise,
                    damping: run.damping(),
                    prior_sqrt_scale: 1.0,
                },
                dimension,
                &mut workspace.dense,
            );
            if !assembly.valid {
                run.reject_failure(StepRejection::Linearization);
                continue;
            }
            let current_cost = assembly.cost;
            if stage_index == target_stage
                && run.gradient_converged(
                    assembly.scaled_gradient_inf_norm,
                    current_cost,
                    options.convergence,
                )
            {
                break 'optimization;
            }
            if !dense::solve_camera_step(&mut workspace.dense, dimension) {
                run.reject_failure(StepRejection::StepSolve);
                continue;
            }
            retract_poses(
                poses,
                &mut workspace.dense.candidate_poses[..poses.len()],
                gauge_state,
                &workspace.dense.camera_offsets,
                &workspace.dense.camera_pose_dimensions,
                &workspace.dense.step[..dimension],
                0,
            );
            retract_cameras(
                cameras,
                &mut workspace.candidate_cameras[..cameras.len()],
                &workspace.dense.intrinsic_offsets,
                &workspace.dense.intrinsic_dimensions,
                &workspace.dense.step[..dimension],
            );
            let Some((point_step_rms, point_damping_norm, candidate_cost)) = candidate_points(
                measurements,
                cameras,
                camera_to_group,
                poses,
                points,
                gauge_state,
                stage.loss,
                noise,
                run.damping(),
                dimension,
                workspace,
                fixed,
                priors,
            ) else {
                run.reject_failure(StepRejection::Candidate);
                continue;
            };
            let predicted_reduction = exact_model_reduction(
                assembly.point_model_reduction,
                &workspace.dense.gradient[..dimension],
                &workspace.dense.diagonal[..dimension],
                &workspace.dense.step[..dimension],
                run.damping(),
                point_damping_norm,
            );
            let reduction = current_cost - candidate_cost;
            if run.accept_gain(reduction, predicted_reduction) {
                poses.copy_from_slice(&workspace.dense.candidate_poses[..poses.len()]);
                points.copy_from_slice(&workspace.dense.candidate_points[..points.len()]);
                cameras.copy_from_slice(&workspace.candidate_cameras[..cameras.len()]);
                let camera_step_rms = (squared_norm(&workspace.dense.step[..dimension])
                    / dimension.max(1) as f64)
                    .sqrt();
                let step_norm =
                    (camera_step_rms * camera_step_rms + point_step_rms * point_step_rms).sqrt();
                let is_target = stage_index == target_stage;
                let candidate_convergence = run.record_accepted_step(
                    step_norm,
                    reduction,
                    current_cost,
                    is_target,
                    options.convergence,
                );
                if is_target && run.certify_convergence(candidate_convergence) {
                    break 'optimization;
                }
            }
        }
    }

    let expected_gauge_nullity = schur::expected_gauge_nullity(gauge, poses, points, fixed, priors);
    let (gradient_inf_norm, diagnostics, final_system_valid) =
        if workspace.dense.final_diagnostics() {
            let final_assembly = schur::assemble_problem(
                schur::BaProblemView {
                    model: radial_model(measurements, cameras, camera_to_group),
                    poses,
                    points,
                    fixed,
                    priors,
                },
                schur::LinearizationPolicy {
                    gauge: GaugeState::new(gauge, poses, normalization_scale),
                    loss: options.loss,
                    noise,
                    damping: 0.0,
                    prior_sqrt_scale: 1.0,
                },
                dimension,
                &mut workspace.dense,
            );
            let gradient_inf_norm = if final_assembly.valid {
                final_assembly.scaled_gradient_inf_norm
            } else {
                f64::INFINITY
            };
            let diagnostics = schur::diagnostics_with_expected_nullity(
                &mut workspace.dense,
                dimension,
                expected_gauge_nullity,
                &final_assembly,
            );
            (gradient_inf_norm, diagnostics, final_assembly.valid)
        } else {
            (
                f64::NAN,
                schur::unassessed_diagnostics(dimension, expected_gauge_nullity),
                true,
            )
        };
    let status = run.status(final_system_valid && diagnostics.is_well_constrained());
    let final_model = radial_model(measurements, cameras, camera_to_group);
    let (observation_cost, final_rmse) =
        schur::metrics(final_model, poses, points, options.loss, noise);
    let final_cost = observation_cost
        + schur::prior_cost(final_model, poses, points, priors).unwrap_or(f64::INFINITY);

    Ok(MultiViewBundleAdjustmentReport {
        loss: options.loss,
        status,
        scale_gauge: gauge,
        normalization_scale,
        fixed_baseline,
        fixed_translation_component,
        pose_count: poses.len(),
        point_count: points.len(),
        observation_count: measurements.observation_count(),
        noninvertible_measurement_count: final_model.noninvertible_measurement_count(),
        iteration_count: run.iterations(),
        accepted_step_count: run.accepted_steps(),
        rejected_step_count: run.rejected_steps(),
        step_rejections: run.rejections(),
        initial_cost,
        final_cost,
        initial_rmse,
        final_rmse,
        gradient_inf_norm,
        step_norm: run.step_norm(),
        initial_damping: workspace.dense.initial_damping,
        final_damping: run.damping(),
        last_gain_ratio: run.last_gain_ratio(),
        diagnostics,
    })
}

pub(in crate::kernel::multi_view) fn radial_model<'a>(
    measurements: SelfCalibratingMeasurements<'a>,
    cameras: &'a [RadialCamera],
    camera_to_group: Option<&'a [usize]>,
) -> ObservationModel<'a> {
    match (camera_to_group, measurements.signed_depth()) {
        (Some(camera_to_group), false) => ObservationModel::radial_grouped(
            measurements.observations,
            measurements.point_offsets,
            cameras,
            camera_to_group,
        ),
        (Some(camera_to_group), true) => ObservationModel::radial_grouped_signed_depth(
            measurements.observations,
            measurements.point_offsets,
            cameras,
            camera_to_group,
        ),
        (None, false) => ObservationModel::radial(
            measurements.observations,
            measurements.point_offsets,
            cameras,
        ),
        (None, true) => ObservationModel::radial_signed_depth(
            measurements.observations,
            measurements.point_offsets,
            cameras,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "unstable-internals")]
fn candidate_points(
    measurements: SelfCalibratingMeasurements<'_>,
    cameras: &[RadialCamera],
    camera_to_group: Option<&[usize]>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    gauge: GaugeState,
    loss: crate::numerics::refine::Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    dimension: usize,
    workspace: &mut SelfCalibrationWorkspace,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
) -> Option<(f64, f64, f64)> {
    let mut squared = 0.0;
    let mut point_damping_norm = 0.0;
    let mut candidate_cost = 0.0;
    let current_model = radial_model(measurements, cameras, camera_to_group);
    let candidate_model = radial_model(
        measurements,
        &workspace.candidate_cameras[..cameras.len()],
        camera_to_group,
    );
    let layout = schur::VariableLayout {
        offsets: &workspace.dense.camera_offsets,
        pose_dimensions: &workspace.dense.camera_pose_dimensions,
        intrinsic_offsets: &workspace.dense.intrinsic_offsets,
        intrinsic_dimensions: &workspace.dense.intrinsic_dimensions,
        pose_masks: &[],
        #[cfg(feature = "faer")]
        pose_variables: super::super::CameraPoseVariables::RotationAndTranslation,
    };
    let mut point_prior_index = 0;
    for (point_index, point) in points.iter().enumerate() {
        let point_prior = priors
            .points
            .get(point_prior_index)
            .filter(|prior| prior.point == point_index);
        if point_prior.is_some() {
            point_prior_index += 1;
        }
        if !fixed.points.is_empty() && fixed.points[point_index] {
            workspace.dense.candidate_points[point_index] = *point;
            candidate_cost += schur::point_cost(
                candidate_model,
                &workspace.dense.candidate_poses[..poses.len()],
                point,
                point_index,
                loss,
                noise,
                gauge.scene_scale,
            )?;
            continue;
        }
        let (step, damping_norm) = schur::point_step_with_prior_and_damping_norm(
            current_model,
            poses,
            point,
            point_index,
            gauge,
            loss,
            noise,
            damping,
            &workspace.dense.step[..dimension],
            layout,
            &mut workspace.dense.cross,
            &mut workspace.dense.active_parameters,
            &mut workspace.dense.parameter_marks,
            point_prior,
        )?;
        let evaluated = schur::candidate_point(
            candidate_model,
            poses,
            &workspace.dense.candidate_poses[..poses.len()],
            point,
            std::array::from_fn(|coordinate| {
                point[coordinate] + gauge.scene_scale * step[coordinate]
            }),
            point_index,
            loss,
            noise,
            gauge.scene_scale,
        )?;
        workspace.dense.candidate_points[point_index] = evaluated.position;
        candidate_cost += evaluated.cost;
        if evaluated.moved {
            point_damping_norm += damping_norm;
            squared += squared_norm(&step);
        }
    }
    candidate_cost += schur::prior_cost(
        candidate_model,
        &workspace.dense.candidate_poses[..poses.len()],
        &workspace.dense.candidate_points[..points.len()],
        priors,
    )?;
    Some((
        (squared / points.len() as f64).sqrt(),
        point_damping_norm,
        candidate_cost,
    ))
}

pub(in crate::kernel::multi_view) fn retract_cameras(
    cameras: &[RadialCamera],
    candidates: &mut [RadialCamera],
    intrinsic_offsets: &[usize],
    intrinsic_dimensions: &[usize],
    step: &[f64],
) {
    for camera in 0..cameras.len() {
        if intrinsic_dimensions[camera] == 0 {
            candidates[camera] = cameras[camera];
            continue;
        }
        // Parameters beyond the free prefix stay at their input values.
        let intrinsic_offset = intrinsic_offsets[camera];
        let mut delta = [0.0; INTRINSIC_DIMENSION];
        delta[..intrinsic_dimensions[camera]].copy_from_slice(
            &step[intrinsic_offset..intrinsic_offset + intrinsic_dimensions[camera]],
        );
        candidates[camera] =
            <RadialCamera as CameraKernel<INTRINSIC_DIMENSION>>::plus(cameras[camera], &delta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_reduction_restores_camera_and_point_damping_energy() {
        let predicted =
            exact_model_reduction(3.0, &[2.0, -4.0], &[5.0, 7.0], &[-0.5, 0.25], 0.2, 11.0);
        let damped = 3.0 - 0.5 * (2.0 * -0.5 + -4.0 * 0.25);
        let damping_norm = 5.0 * 0.25 + 7.0 * 0.0625 + 11.0;
        assert!((predicted - (damped + 0.1 * damping_norm)).abs() < 1e-14);
    }

    #[test]
    fn camera_retraction_preserves_existing_arithmetic_bits() {
        let cameras = [
            RadialCamera::new(730.0, 0.08, -0.025),
            RadialCamera::new(410.0, -0.03, 0.004),
        ];
        let mut candidates = [RadialCamera::new(1.0, 1.0, 1.0); 2];
        let step = [9.0, 0.013, -0.004, 0.0007];
        retract_cameras(&cameras, &mut candidates, &[0, 1], &[0, 3], &step);
        let expected = RadialCamera::new(
            cameras[1].focal * step[1].exp(),
            cameras[1].k1 + step[2],
            cameras[1].k2 + step[3],
        );

        assert_eq!(candidates[0], cameras[0]);
        assert_eq!(candidates[1].focal.to_bits(), expected.focal.to_bits());
        assert_eq!(candidates[1].k1.to_bits(), expected.k1.to_bits());
        assert_eq!(candidates[1].k2.to_bits(), expected.k2.to_bits());
    }
}
