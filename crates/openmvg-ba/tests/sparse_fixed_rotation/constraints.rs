// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed blocks, pose priors, validation, and covariance rejection.

use super::*;

#[test]
fn translation_only_preserves_fixed_pose_and_point_blocks_bit_exactly() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let fixed_poses = [false, false, true, false];
    let mut fixed_points = vec![false; fixture.initial_points.len()];
    fixed_points[4] = true;
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_poses,
        points: &fixed_points,
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    let mut workspace = translation_only_workspace_with_fixed(
        fixture.initial_poses.len(),
        measurements,
        gauge,
        fixed,
    );
    let mut poses = fixture.initial_poses.clone();
    let before_poses = poses.clone();
    let mut points = fixture.initial_points.clone();
    let before_point = points[4];
    let report = refine_ba_multi_view_sparse_with_factors(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        fixed,
        BundleAdjustmentPriors::NONE,
        &mut workspace,
        Options::l2(30),
        NoiseModel::Unit,
    )
    .expect("translation-only solve with fixed blocks succeeds");

    assert!(report.final_cost < report.initial_cost);
    assert_eq!(poses[2], before_poses[2]);
    assert_eq!(points[4], before_point);
    for (pose, before) in poses.iter().zip(before_poses) {
        assert_eq!(pose.rotation, before.rotation);
    }
}

#[test]
fn translation_only_pose_priors_reject_fixed_targets_and_support_free_cameras() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let fixed_poses = [false, false, true, false];
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_poses,
        points: &[],
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    let workspace_options = SparseWorkspaceOptions::iterative_schur(
        IterativeSchurOptions::new()
            .with_matvec(RetainedSchurMatvec::Fused)
            .with_exact_camera_rank_diagnostics(false),
    );
    let make_workspace = || {
        SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks_and_pose_variables(
            fixture.initial_poses.len(),
            measurements,
            gauge,
            fixed,
            CameraPoseVariables::TranslationOnly,
            workspace_options,
        )
        .expect("translation-only retained workspace with fixed cameras is valid")
    };
    let sqrt_information: [[f64; 6]; 6] =
        std::array::from_fn(|row| std::array::from_fn(|column| f64::from(row == column)));

    let fixed_prior = [PosePrior {
        camera: 0,
        mean: fixture.initial_poses[0],
        sqrt_information,
    }];
    let mut rejected_workspace = make_workspace();
    let mut rejected_poses = fixture.initial_poses.clone();
    let mut rejected_points = fixture.initial_points.clone();
    let rejected = refine_ba_multi_view_sparse_with_factors(
        &mut rejected_poses,
        &mut rejected_points,
        measurements,
        gauge,
        fixed,
        BundleAdjustmentPriors {
            poses: &fixed_prior,
            points: &[],
            intrinsics: &[],
        },
        &mut rejected_workspace,
        Options::l2(5),
        NoiseModel::Unit,
    );
    assert!(matches!(
        rejected,
        Err(SparseBundleAdjustmentError::Problem(
            MultiViewBundleAdjustmentError::InvalidPriorTarget {
                block: BundleAdjustmentBlock::WorldToCameraPose,
                index: 0,
            }
        ))
    ));
    assert_eq!(rejected_poses, fixture.initial_poses);
    assert_eq!(rejected_points, fixture.initial_points);

    let mut free_mean = fixture.initial_poses[3];
    free_mean.translation[0] -= 0.02;
    free_mean.translation[1] += 0.01;
    free_mean.translation[2] -= 0.015;
    let free_prior = [PosePrior {
        camera: 3,
        mean: free_mean,
        sqrt_information,
    }];
    let priors = BundleAdjustmentPriors {
        poses: &free_prior,
        points: &[],
        intrinsics: &[],
    };
    let mut workspace = make_workspace();
    let mut poses = fixture.initial_poses.clone();
    let rotations = poses.iter().map(|pose| pose.rotation).collect::<Vec<_>>();
    let mut points = fixture.initial_points.clone();
    let report = refine_ba_multi_view_sparse_with_factors(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        fixed,
        priors,
        &mut workspace,
        Options::l2(30),
        NoiseModel::Unit,
    )
    .expect("a translation prior on a free camera is supported");

    assert!(report.final_cost < report.initial_cost);
    assert_eq!(poses[2], fixture.initial_poses[2]);
    for (pose, rotation) in poses.iter().zip(rotations) {
        assert_eq!(pose.rotation, rotation);
    }
}

#[test]
fn translation_only_validation_errors_are_atomic() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let mut workspace =
        translation_only_workspace(fixture.initial_poses.len(), measurements, gauge);
    let mut poses = fixture.initial_poses.clone();
    let mut points = fixture.initial_points.clone();
    let before_poses = poses.clone();
    let before_points = points.clone();

    let zero_iteration = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut workspace,
        Options::l2(0),
        NoiseModel::Unit,
    )
    .expect("zero iterations is a valid no-op solve");
    assert_eq!(zero_iteration.iteration_count, 0);
    assert_eq!(poses, before_poses);
    assert_eq!(points, before_points);

    let invalid_options = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut workspace,
        Options::huber(-1.0, 10),
        NoiseModel::Unit,
    );
    assert!(invalid_options.is_err());
    assert_eq!(poses, before_poses);
    assert_eq!(points, before_points);

    let topology_mismatch = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: TranslationComponent::X,
        },
        &mut workspace,
        Options::l2(10),
        NoiseModel::Unit,
    );
    assert!(matches!(
        topology_mismatch,
        Err(SparseBundleAdjustmentError::TopologyMismatch)
    ));
    assert_eq!(poses, before_poses);
    assert_eq!(points, before_points);
}

#[test]
fn translation_only_covariance_rejects_with_a_typed_error() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let mut workspace =
        translation_only_workspace(fixture.initial_poses.len(), measurements, gauge);
    let result = selected_marginal_covariances_ba_multi_view_sparse(
        &fixture.initial_poses,
        &fixture.initial_points,
        measurements,
        gauge,
        &[2],
        &[3],
        &mut workspace,
        Loss::L2,
        NoiseModel::Unit,
    );
    assert!(matches!(
        result,
        Err(SparseBundleAdjustmentCovarianceError::TranslationOnlyUnsupported)
    ));
}
