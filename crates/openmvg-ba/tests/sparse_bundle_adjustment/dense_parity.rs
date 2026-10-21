// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sparse solves against the dense reference, damping, fixed blocks, and priors.

use super::*;

#[test]
fn calibrated_structure_only_dense_and_sparse_solves_support_zero_camera_dimension() {
    let fixture = fixture();
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::Free;
    let fixed_poses = [true; 4];
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_poses,
        points: &[],
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    let (_, initial_points) = perturbed(&fixture.truth_poses, &fixture.truth_points);
    let mut dense_poses = fixture.truth_poses.clone();
    let mut sparse_poses = fixture.truth_poses.clone();
    let mut dense_points = initial_points.clone();
    let mut sparse_points = initial_points;
    let mut dense_workspace =
        DenseBundleAdjustmentWorkspace::try_new(4, fixture.truth_points.len()).unwrap();
    let mut sparse_workspace = SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks(
        4,
        measurements,
        gauge,
        fixed,
        SparseWorkspaceOptions::direct(),
    )
    .unwrap();
    assert_eq!(sparse_workspace.camera_dimension(), 0);

    let dense = refine_ba_multi_view_dense_with_factors(
        &mut dense_poses,
        &mut dense_points,
        measurements,
        gauge,
        fixed,
        BundleAdjustmentPriors::NONE,
        &mut dense_workspace,
        Options::l2(30),
        NoiseModel::Unit,
    )
    .unwrap();
    let sparse = refine_ba_multi_view_sparse_with_factors(
        &mut sparse_poses,
        &mut sparse_points,
        measurements,
        gauge,
        fixed,
        BundleAdjustmentPriors::NONE,
        &mut sparse_workspace,
        Options::l2(30),
        NoiseModel::Unit,
    )
    .unwrap();

    assert_eq!(dense_poses, fixture.truth_poses);
    assert_eq!(sparse_poses, fixture.truth_poses);
    assert!(dense.final_cost < dense.initial_cost, "{dense:#?}");
    assert!((dense.final_cost - sparse.final_cost).abs() <= 1e-12);
    for (dense, sparse) in dense_points
        .iter()
        .flatten()
        .zip(sparse_points.iter().flatten())
    {
        assert!((dense - sparse).abs() <= 1e-9);
    }
    assert!(sparse.diagnostics.is_well_constrained(), "{sparse:#?}");
}

#[test]
fn calibrated_sparse_zero_budget_reports_its_initial_damping() {
    let fixture = fixture();
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let (mut poses, mut points) = perturbed(&fixture.truth_poses, &fixture.truth_points);
    let mut workspace = SparseBundleAdjustmentWorkspace::try_new(4, measurements, gauge)
        .expect("fixture topology is valid");

    let report = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut workspace,
        Options::l2(0),
        NoiseModel::Unit,
    )
    .expect("zero-budget diagnostics are valid");

    assert_eq!(report.initial_damping, 1e-4);
    assert_eq!(report.final_damping, 1e-4);
    assert_eq!(report.last_gain_ratio, None);
    assert_eq!(
        report.solver_working_set_bytes,
        workspace.solver_specific_working_set_bytes()
    );
}

#[test]
fn calibrated_dense_initial_damping_override_is_reported_and_validated() {
    let fixture = fixture();
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let (mut poses, mut points) = perturbed(&fixture.truth_poses, &fixture.truth_points);
    let mut workspace = DenseBundleAdjustmentWorkspace::try_new(poses.len(), points.len())
        .expect("fixture workspace must fit")
        .with_initial_damping(0.025);
    let report = refine_ba_multi_view_dense(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut workspace,
        Options::l2(0),
        NoiseModel::Unit,
    )
    .expect("zero-budget diagnostics are valid");
    assert_eq!(report.initial_damping, 0.025);
    assert_eq!(report.final_damping, 0.025);

    let mut invalid_workspace = DenseBundleAdjustmentWorkspace::try_new(poses.len(), points.len())
        .expect("fixture workspace must fit")
        .with_initial_damping(0.0);
    assert!(matches!(
        refine_ba_multi_view_dense(
            &mut poses,
            &mut points,
            measurements,
            gauge,
            &mut invalid_workspace,
            Options::l2(0),
            NoiseModel::Unit,
        ),
        Err(MultiViewBundleAdjustmentError::InitialDampingInvalid)
    ));
}

#[test]
fn sparse_and_dense_fixed_gauge_solutions_agree() {
    let Fixture {
        truth_poses,
        truth_points,
        observations,
        offsets,
        cameras,
    } = fixture();
    let measurements = MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &offsets,
        cameras: &cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let (initial_poses, initial_points) = perturbed(&truth_poses, &truth_points);
    let mut dense_poses = initial_poses.clone();
    let mut dense_points = initial_points.clone();
    let mut sparse_poses = initial_poses;
    let mut sparse_points = initial_points;
    let mut dense_workspace = DenseBundleAdjustmentWorkspace::try_new(4, truth_points.len())
        .expect("fixture dimensions fit");
    let mut sparse_workspace = SparseBundleAdjustmentWorkspace::try_new(4, measurements, gauge)
        .expect("fixture topology is valid");

    let dense = refine_ba_multi_view_dense(
        &mut dense_poses,
        &mut dense_points,
        measurements,
        gauge,
        &mut dense_workspace,
        Options::l2(60),
        NoiseModel::Unit,
    )
    .expect("dense fixture is valid");
    let sparse = refine_ba_multi_view_sparse(
        &mut sparse_poses,
        &mut sparse_points,
        measurements,
        gauge,
        &mut sparse_workspace,
        Options::l2(60),
        NoiseModel::Unit,
    )
    .expect("sparse fixture is valid");

    assert!(dense.final_rmse < 1e-7, "{dense:#?}");
    assert!(sparse.final_rmse < 1e-7, "{sparse:#?}");
    assert!((dense.final_cost - sparse.final_cost).abs() < 1e-10);
    assert_eq!(
        sparse.diagnostics.pose_rank,
        SparseCameraRankAssessment::FullRank
    );
    assert!(sparse.diagnostics.is_well_constrained(), "{sparse:#?}");
    assert!(sparse_workspace.reduced_nonzero_count() < 17 * 17);
}

#[test]
fn calibrated_sparse_fixed_blocks_and_priors_match_dense() {
    let fixture = fixture();
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let (initial_poses, initial_points) = perturbed(&fixture.truth_poses, &fixture.truth_points);
    let fixed_pose_mask = [false, false, false, true];
    let mut fixed_point_mask = vec![false; fixture.truth_points.len()];
    fixed_point_mask[0] = true;
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_pose_mask,
        points: &fixed_point_mask,
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    let pose_priors = [PosePrior {
        camera: 2,
        mean: fixture.truth_poses[2],
        sqrt_information: std::array::from_fn(|row| {
            std::array::from_fn(|column| if row == column { 4.0 } else { 0.0 })
        }),
    }];
    let point_priors = [PointPrior {
        point: 1,
        mean: fixture.truth_points[1],
        sqrt_information: [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
    }];
    let priors = BundleAdjustmentPriors {
        poses: &pose_priors,
        points: &point_priors,
        intrinsics: &[],
    };
    let mut dense_poses = initial_poses.clone();
    let mut dense_points = initial_points.clone();
    let mut sparse_poses = initial_poses.clone();
    let mut sparse_points = initial_points.clone();
    let fixed_pose = initial_poses[3];
    let fixed_point = initial_points[0];
    let mut dense_workspace =
        DenseBundleAdjustmentWorkspace::try_new(4, fixture.truth_points.len()).unwrap();
    let mut sparse_workspace = SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks(
        4,
        measurements,
        gauge,
        fixed,
        SparseWorkspaceOptions::direct(),
    )
    .unwrap();

    let dense = refine_ba_multi_view_dense_with_factors(
        &mut dense_poses,
        &mut dense_points,
        measurements,
        gauge,
        fixed,
        priors,
        &mut dense_workspace,
        Options::l2(60),
        NoiseModel::Unit,
    )
    .unwrap();
    let sparse = refine_ba_multi_view_sparse_with_factors(
        &mut sparse_poses,
        &mut sparse_points,
        measurements,
        gauge,
        fixed,
        priors,
        &mut sparse_workspace,
        Options::l2(60),
        NoiseModel::Unit,
    )
    .unwrap();

    assert_eq!(dense_poses[3], fixed_pose);
    assert_eq!(sparse_poses[3], fixed_pose);
    assert_eq!(dense_points[0], fixed_point);
    assert_eq!(sparse_points[0], fixed_point);
    assert!(
        (dense.final_cost - sparse.final_cost).abs() <= 1e-9,
        "{dense:#?}\n{sparse:#?}"
    );
    for (dense, sparse) in dense_poses.iter().zip(&sparse_poses) {
        for row in 0..3 {
            for column in 0..3 {
                let error = (dense.rotation[row][column] - sparse.rotation[row][column]).abs();
                assert!(error <= 1e-6, "rotation error {error:e}");
            }
            let error = (dense.translation[row] - sparse.translation[row]).abs();
            assert!(error <= 1e-6, "translation error {error:e}");
        }
    }
    for (dense, sparse) in dense_points.iter().zip(&sparse_points) {
        for coordinate in 0..3 {
            let error = (dense[coordinate] - sparse[coordinate]).abs();
            assert!(error <= 1e-6, "point error {error:e}");
        }
    }
}
