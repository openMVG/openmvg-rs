// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error-atomic sparse solve regressions.

use super::super::*;
use super::fixture::fixture;
use crate::kernel::multi_view::self_calibration::{RadialCamera, SelfCalibratingMeasurements};

#[test]
fn calibrated_exact_rank_storage_failure_precedes_caller_mutation() {
    let fixture = fixture(4, 24, 4);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let original_poses = fixture.poses.clone();
    let original_points = fixture.points.clone();
    let mut poses = original_poses.clone();
    let mut points = original_points.clone();
    let mut workspace = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
        poses.len(),
        measurements,
        gauge,
        IterativeSchurOptions::new(),
    )
    .expect("fixture topology is valid");
    workspace.core.fail_next_direct_storage_for_test = true;

    let error = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut workspace,
        Options::l2(4),
        NoiseModel::Unit,
    )
    .expect_err("injected exact-rank storage failure must be reported");

    assert_eq!(
        error,
        SparseBundleAdjustmentError::NumericalWorkspaceAllocation
    );
    assert_eq!(poses, original_poses);
    assert_eq!(points, original_points);
}

#[test]
fn calibrated_lazy_fallback_storage_failure_is_a_rejected_attempt() {
    let fixture = fixture(4, 24, 4);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let original_poses = fixture.poses.clone();
    let original_points = fixture.points.clone();
    let mut poses = original_poses.clone();
    let mut points = original_points.clone();
    let mut workspace = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
        poses.len(),
        measurements,
        gauge,
        IterativeSchurOptions::new()
            .with_matvec(RetainedSchurMatvec::Recomputed)
            .with_maximum_iterations(1)
            .with_relative_tolerance(1e-14)
            .with_exact_camera_rank_diagnostics(false),
    )
    .expect("fixture topology is valid");
    workspace.core.fail_next_direct_storage_for_test = true;

    let report = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut workspace,
        Options::l2(1),
        NoiseModel::Unit,
    )
    .expect("an unavailable lazy fallback is not a post-mutation error");

    assert_eq!(report.accepted_step_count, 0, "{report:#?}");
    assert_eq!(report.rejected_step_count, 1, "{report:#?}");
    assert_eq!(report.direct_fallback_steps, 0, "{report:#?}");
    assert_eq!(report.unavailable_direct_fallback_steps, 1, "{report:#?}");
    assert_eq!(report.iterative_inner_iterations(), 1, "{report:#?}");
    assert!(
        report
            .maximum_rejected_iterative_relative_residual
            .is_some(),
        "{report:#?}"
    );
    assert_eq!(workspace.direct_numeric_bytes(), 0);
    assert_eq!(poses, original_poses);
    assert_eq!(points, original_points);
}

#[test]
fn grouped_self_calibration_exact_rank_failure_precedes_all_mutation() {
    let fixture = fixture(4, 24, 4);
    let measurements = SelfCalibratingMeasurements::new(&fixture.observations, &fixture.offsets);
    let camera_to_group = [0, 0, 1, 1];
    let original_groups = [
        RadialCamera::new(500.0, 0.01, -0.001),
        RadialCamera::new(510.0, -0.01, 0.001),
    ];
    let original_poses = fixture.poses.clone();
    let original_points = fixture.points.clone();
    let mut poses = original_poses.clone();
    let mut points = original_points.clone();
    let mut groups = original_groups;
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let mut workspace =
        SparseSelfCalibrationWorkspace::try_new_grouped_with_iterative_schur_options(
            poses.len(),
            groups.len(),
            &camera_to_group,
            measurements,
            gauge,
            FixedVariableBlocks::NONE,
            IterativeSchurOptions::new(),
        )
        .expect("grouped fixture topology is valid");
    workspace.sparse.fail_next_direct_storage_for_test = true;

    let error = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
        &mut poses,
        &mut points,
        &mut groups,
        &camera_to_group,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        BundleAdjustmentPriors::NONE,
        &mut workspace,
        Options::l2(4),
        NoiseModel::Unit,
    )
    .expect_err("injected exact-rank storage failure must be reported");

    assert_eq!(
        error,
        SparseBundleAdjustmentError::NumericalWorkspaceAllocation
    );
    assert_eq!(poses, original_poses);
    assert_eq!(points, original_points);
    assert_eq!(groups, original_groups);
}

#[test]
fn grouped_self_calibration_lazy_fallback_failure_is_a_rejected_attempt() {
    let fixture = fixture(4, 24, 4);
    let measurements = SelfCalibratingMeasurements::new(&fixture.observations, &fixture.offsets);
    let camera_to_group = [0, 0, 1, 1];
    let original_groups = [
        RadialCamera::new(500.0, 0.01, -0.001),
        RadialCamera::new(510.0, -0.01, 0.001),
    ];
    let original_poses = fixture.poses.clone();
    let original_points = fixture.points.clone();
    let mut poses = original_poses.clone();
    let mut points = original_points.clone();
    let mut groups = original_groups;
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let mut workspace =
        SparseSelfCalibrationWorkspace::try_new_grouped_with_iterative_schur_options(
            poses.len(),
            groups.len(),
            &camera_to_group,
            measurements,
            gauge,
            FixedVariableBlocks::NONE,
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::Recomputed)
                .with_maximum_iterations(1)
                .with_relative_tolerance(1e-14)
                .with_exact_camera_rank_diagnostics(false),
        )
        .expect("grouped fixture topology is valid");
    workspace.sparse.fail_next_direct_storage_for_test = true;

    let report = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
        &mut poses,
        &mut points,
        &mut groups,
        &camera_to_group,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        BundleAdjustmentPriors::NONE,
        &mut workspace,
        Options::l2(1),
        NoiseModel::Unit,
    )
    .expect("an unavailable lazy fallback is not a post-mutation error");

    assert_eq!(report.accepted_step_count, 0, "{report:#?}");
    assert_eq!(report.rejected_step_count, 1, "{report:#?}");
    assert_eq!(report.direct_fallback_steps, 0, "{report:#?}");
    assert_eq!(report.unavailable_direct_fallback_steps, 1, "{report:#?}");
    assert_eq!(report.iterative_inner_iterations(), 1, "{report:#?}");
    assert!(
        report
            .maximum_rejected_iterative_relative_residual
            .is_some(),
        "{report:#?}"
    );
    assert_eq!(workspace.direct_numeric_bytes(), 0);
    assert_eq!(poses, original_poses);
    assert_eq!(points, original_points);
    assert_eq!(groups, original_groups);
}
