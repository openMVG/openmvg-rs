// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-view and grouped sparse self-calibration against the dense reference.

use super::*;

#[test]
fn radial_structure_only_dense_and_sparse_solves_support_zero_reduced_dimension() {
    let (truth_poses, truth_points) = scene();
    let truth_cameras = [
        RadialCamera::new(680.0, 0.055, -0.022),
        RadialCamera::new(725.0, -0.041, 0.017),
        RadialCamera::new(695.0, 0.032, -0.013),
        RadialCamera::new(740.0, -0.048, 0.020),
    ];
    let (observations, offsets) =
        observations(&truth_poses, &truth_points, |camera| truth_cameras[camera]);
    let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
    let fixed_poses = [true; CAMERA_COUNT];
    let fixed_intrinsics = [true; CAMERA_COUNT];
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_poses,
        points: &[],
        intrinsics: &fixed_intrinsics,
        intrinsic_dimensions: &[],
    };
    let (_, initial_points) = perturbed_scene(&truth_poses, &truth_points);
    let mut dense_poses = truth_poses;
    let mut sparse_poses = truth_poses;
    let mut dense_points = initial_points.clone();
    let mut sparse_points = initial_points;
    let mut dense_cameras = truth_cameras;
    let mut sparse_cameras = truth_cameras;
    let mut dense_workspace = SelfCalibrationWorkspace::try_new(CAMERA_COUNT, truth_points.len())
        .expect("dense workspace must fit the fixture");
    let camera_to_group = [0, 1, 2, 3];
    let mut sparse_workspace = SparseSelfCalibrationWorkspace::try_new_grouped(
        CAMERA_COUNT,
        CAMERA_COUNT,
        &camera_to_group,
        measurements,
        ScaleGauge::Free,
        fixed,
    )
    .expect("zero-dimensional sparse topology must be valid");
    assert_eq!(sparse_workspace.reduced_dimension(), 0);
    assert_eq!(
        sparse_workspace.initial_damping(),
        SparseCameraSolver::DirectLlt.default_initial_damping()
    );

    let dense = refine_ba_multi_view_self_calibrating_dense_with_factors(
        &mut dense_poses,
        &mut dense_points,
        &mut dense_cameras,
        measurements,
        ScaleGauge::Free,
        fixed,
        BundleAdjustmentPriors::NONE,
        &mut dense_workspace,
        Options::l2(30),
        NoiseModel::Unit,
    )
    .expect("dense structure-only self-calibration must be valid");
    let sparse = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
        &mut sparse_poses,
        &mut sparse_points,
        &mut sparse_cameras,
        &camera_to_group,
        measurements,
        ScaleGauge::Free,
        fixed,
        BundleAdjustmentPriors::NONE,
        &mut sparse_workspace,
        Options::l2(30),
        NoiseModel::Unit,
    )
    .expect("sparse structure-only self-calibration must be valid");

    assert_eq!(dense_poses, truth_poses);
    assert_eq!(sparse_poses, truth_poses);
    assert_eq!(dense_cameras, truth_cameras);
    assert_eq!(sparse_cameras, truth_cameras);
    assert!(dense.final_cost < dense.initial_cost, "{dense:#?}");
    assert!((dense.final_cost - sparse.final_cost).abs() <= 1e-12);
    for (dense, sparse) in dense_points
        .iter()
        .flatten()
        .zip(sparse_points.iter().flatten())
    {
        assert!((dense - sparse).abs() <= 1e-9);
    }
    assert_eq!(dense.diagnostics.pose_dimension, 0);
    assert_eq!(dense.diagnostics.pose_rank, 0);
    assert_eq!(sparse.diagnostics.pose_dimension, 0);
    assert_eq!(
        sparse.diagnostics.pose_rank,
        SparseCameraRankAssessment::FullRank
    );
}

#[test]
fn per_view_sparse_matches_dense_objective_and_state() {
    let (truth_poses, truth_points) = scene();
    let truth_cameras = [
        RadialCamera::new(680.0, 0.055, -0.022),
        RadialCamera::new(725.0, -0.041, 0.017),
        RadialCamera::new(695.0, 0.032, -0.013),
        RadialCamera::new(740.0, -0.048, 0.020),
    ];
    let (observations, offsets) =
        observations(&truth_poses, &truth_points, |camera| truth_cameras[camera]);
    let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let (initial_poses, initial_points) = perturbed_scene(&truth_poses, &truth_points);
    let initial_cameras = [
        RadialCamera::new(688.0, 0.058, -0.023),
        RadialCamera::new(718.0, -0.038, 0.016),
        RadialCamera::new(701.0, 0.030, -0.012),
        RadialCamera::new(733.0, -0.045, 0.019),
    ];

    let mut dense_poses = initial_poses;
    let mut dense_points = initial_points.clone();
    let mut dense_cameras = initial_cameras;
    let mut dense_workspace = SelfCalibrationWorkspace::try_new(CAMERA_COUNT, truth_points.len())
        .expect("dense workspace must fit the fixture");
    let dense = refine_ba_multi_view_self_calibrating_dense(
        &mut dense_poses,
        &mut dense_points,
        &mut dense_cameras,
        measurements,
        gauge,
        &mut dense_workspace,
        Options::l2(80),
        NoiseModel::Unit,
    )
    .expect("dense reference problem must be valid");

    let mut sparse_poses = initial_poses;
    let mut sparse_points = initial_points.clone();
    let mut sparse_cameras = initial_cameras;
    let mut sparse_workspace = SparseSelfCalibrationWorkspace::try_new(
        CAMERA_COUNT,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
    )
    .expect("sparse workspace topology must be valid");
    assert!(sparse_workspace.reduced_nonzero_count() > 0);
    assert!(sparse_workspace.direct_pattern_bytes() > 0);
    let sparse = refine_ba_multi_view_self_calibrating_sparse(
        &mut sparse_poses,
        &mut sparse_points,
        &mut sparse_cameras,
        measurements,
        gauge,
        &mut sparse_workspace,
        Options::l2(80),
        NoiseModel::Unit,
    )
    .expect("sparse self-calibration must be valid");

    let mut retained_poses = initial_poses;
    let mut retained_points = initial_points.clone();
    let mut retained_cameras = initial_cameras;
    let mut retained_workspace =
        SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
            CAMERA_COUNT,
            measurements,
            gauge,
            FixedVariableBlocks::NONE,
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::Fused)
                .with_maximum_iterations(300),
        )
        .expect("retained per-view topology must be valid");
    let retained = refine_ba_multi_view_self_calibrating_sparse(
        &mut retained_poses,
        &mut retained_points,
        &mut retained_cameras,
        measurements,
        gauge,
        &mut retained_workspace,
        Options::l2(80),
        NoiseModel::Unit,
    )
    .expect("retained self-calibration must be valid");

    assert!(dense.final_rmse < 1e-6, "{dense:#?}");
    assert!(sparse.final_rmse < 1e-6, "{sparse:#?}");
    assert!(retained.final_rmse < 1e-6, "{retained:#?}");
    assert!(
        matches!(
            retained.camera_solve_quality,
            SparseCameraSolveQuality::ResidualBoundedIterativeSchur
                | SparseCameraSolveQuality::IterativeSchurWithDirectFallback
        ),
        "{retained:#?}"
    );
    assert!(retained.retained_camera_steps > 0, "{retained:#?}");
    assert_objective_close(dense.final_cost, sparse.final_cost);
    assert_objective_close(dense.final_cost, retained.final_cost);
    assert_state_close(
        &dense_poses,
        &dense_points,
        &dense_cameras,
        &sparse_poses,
        &sparse_points,
        &sparse_cameras,
    );
    assert_state_close(
        &dense_poses,
        &dense_points,
        &dense_cameras,
        &retained_poses,
        &retained_points,
        &retained_cameras,
    );
}

#[test]
fn grouped_sparse_matches_dense_with_fixed_blocks_priors_and_repeated_group() {
    let (truth_poses, truth_points) = scene();
    // Both pairs share one intrinsic block. Consequently every point track
    // references each group twice, exercising unique grouped back-substitution.
    let camera_to_group = [0, 0, 1, 1];
    let truth_groups = [
        RadialCamera::new(685.0, 0.052, -0.021),
        RadialCamera::new(735.0, -0.043, 0.018),
    ];
    let (observations, offsets) = observations(&truth_poses, &truth_points, |camera| {
        truth_groups[camera_to_group[camera]]
    });
    let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
    assert_eq!(
        camera_to_group[observations[0].camera],
        camera_to_group[observations[1].camera]
    );

    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let (mut initial_poses, mut initial_points) = perturbed_scene(&truth_poses, &truth_points);
    initial_poses[3] = truth_poses[3];
    initial_points[0] = truth_points[0];
    let initial_groups = [truth_groups[0], RadialCamera::new(726.0, -0.039, 0.016)];
    let fixed_poses = [false, false, false, true];
    let mut fixed_points = vec![false; truth_points.len()];
    fixed_points[0] = true;
    let fixed_intrinsics = [true, false];
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_poses,
        points: &fixed_points,
        intrinsics: &fixed_intrinsics,
        intrinsic_dimensions: &[],
    };
    let pose_priors = [PosePrior {
        camera: 2,
        mean: truth_poses[2],
        sqrt_information: diagonal::<6>(0.5),
    }];
    let point_priors = [PointPrior {
        point: 1,
        mean: truth_points[1],
        sqrt_information: diagonal::<3>(1.0),
    }];
    let intrinsic_priors = [RadialIntrinsicPrior {
        intrinsic: 1,
        focal: truth_groups[1].focal,
        k1: truth_groups[1].k1,
        k2: truth_groups[1].k2,
        sqrt_information: [[0.1, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
    }];
    let priors = BundleAdjustmentPriors {
        poses: &pose_priors,
        points: &point_priors,
        intrinsics: &intrinsic_priors,
    };

    let mut dense_poses = initial_poses;
    let mut dense_points = initial_points.clone();
    let mut dense_groups = initial_groups;
    let mut dense_workspace = SelfCalibrationWorkspace::try_new(CAMERA_COUNT, truth_points.len())
        .expect("dense grouped workspace must fit");
    let dense = refine_ba_multi_view_self_calibrating_grouped_dense_with_factors(
        &mut dense_poses,
        &mut dense_points,
        &mut dense_groups,
        &camera_to_group,
        measurements,
        gauge,
        fixed,
        priors,
        &mut dense_workspace,
        Options::l2(80),
        NoiseModel::Unit,
    )
    .expect("dense grouped reference must be valid");

    let mut sparse_poses = initial_poses;
    let mut sparse_points = initial_points.clone();
    let mut sparse_groups = initial_groups;
    let mut sparse_workspace = SparseSelfCalibrationWorkspace::try_new_grouped(
        CAMERA_COUNT,
        truth_groups.len(),
        &camera_to_group,
        measurements,
        gauge,
        fixed,
    )
    .expect("grouped sparse topology must be valid");
    let sparse = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
        &mut sparse_poses,
        &mut sparse_points,
        &mut sparse_groups,
        &camera_to_group,
        measurements,
        gauge,
        fixed,
        priors,
        &mut sparse_workspace,
        Options::l2(80),
        NoiseModel::Unit,
    )
    .expect("factor-aware grouped sparse solve must be valid");

    #[cfg(feature = "parallel")]
    {
        let mut auto_observations = Vec::new();
        let mut auto_offsets = Vec::with_capacity(129);
        auto_offsets.push(0);
        for point in 0..128 {
            let source = point % truth_points.len();
            auto_observations
                .extend_from_slice(&observations[offsets[source]..offsets[source + 1]]);
            auto_offsets.push(auto_observations.len());
        }
        let auto_workspace = SparseSelfCalibrationWorkspace::try_new_grouped_with_direct_assembly(
            CAMERA_COUNT,
            truth_groups.len(),
            &camera_to_group,
            SelfCalibratingMeasurements::new(&auto_observations, &auto_offsets),
            gauge,
            FixedVariableBlocks::NONE,
            DirectSchurAssembly::Auto,
        )
        .expect("auto grouped direct topology must be valid");
        assert!(auto_workspace.direct_schur_parallel_threads().is_some());

        let mut one_poses = initial_poses;
        let mut one_points = initial_points.clone();
        let mut one_groups = initial_groups;
        let mut one_workspace =
            SparseSelfCalibrationWorkspace::try_new_grouped_with_direct_assembly(
                CAMERA_COUNT,
                truth_groups.len(),
                &camera_to_group,
                measurements,
                gauge,
                fixed,
                DirectSchurAssembly::Parallel { threads: 1 },
            )
            .expect("one-thread grouped direct topology must be valid");
        let one = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
            &mut one_poses,
            &mut one_points,
            &mut one_groups,
            &camera_to_group,
            measurements,
            gauge,
            fixed,
            priors,
            &mut one_workspace,
            Options::l2(80),
            NoiseModel::Unit,
        )
        .expect("one-thread grouped direct solve must be valid");

        let mut four_poses = initial_poses;
        let mut four_points = initial_points.clone();
        let mut four_groups = initial_groups;
        let mut four_workspace =
            SparseSelfCalibrationWorkspace::try_new_grouped_with_direct_assembly(
                CAMERA_COUNT,
                truth_groups.len(),
                &camera_to_group,
                measurements,
                gauge,
                fixed,
                DirectSchurAssembly::Parallel { threads: 4 },
            )
            .expect("four-thread grouped direct topology must be valid");
        let four = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
            &mut four_poses,
            &mut four_points,
            &mut four_groups,
            &camera_to_group,
            measurements,
            gauge,
            fixed,
            priors,
            &mut four_workspace,
            Options::l2(80),
            NoiseModel::Unit,
        )
        .expect("four-thread grouped direct solve must be valid");

        assert_eq!(one_workspace.direct_schur_parallel_threads(), Some(1));
        assert_eq!(four_workspace.direct_schur_parallel_threads(), Some(4));
        assert!(four_workspace.direct_schur_assembly_bytes() > 0);
        assert!(four_workspace.retains_cross_rows());
        assert_eq!(one_poses, four_poses);
        assert_eq!(one_points, four_points);
        assert_eq!(one_groups, four_groups);
        assert_eq!(one.final_cost.to_bits(), four.final_cost.to_bits());
        assert_eq!(one.final_rmse.to_bits(), four.final_rmse.to_bits());
        assert_objective_close(sparse.final_cost, one.final_cost);
        assert_state_close(
            &sparse_poses,
            &sparse_points,
            &sparse_groups,
            &one_poses,
            &one_points,
            &one_groups,
        );
    }

    let mut retained_poses = initial_poses;
    let mut retained_points = initial_points.clone();
    let mut retained_groups = initial_groups;
    let mut retained_workspace =
        SparseSelfCalibrationWorkspace::try_new_grouped_with_iterative_schur_options(
            CAMERA_COUNT,
            truth_groups.len(),
            &camera_to_group,
            measurements,
            gauge,
            fixed,
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::PointMajorParallel { threads: 2 })
                .with_maximum_iterations(300),
        )
        .expect("retained grouped topology must be valid");
    let retained = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
        &mut retained_poses,
        &mut retained_points,
        &mut retained_groups,
        &camera_to_group,
        measurements,
        gauge,
        fixed,
        priors,
        &mut retained_workspace,
        Options::l2(80),
        NoiseModel::Unit,
    )
    .expect("retained grouped solve must be valid");

    assert!(dense.final_rmse < 1e-6, "{dense:#?}");
    assert!(sparse.final_rmse < 1e-6, "{sparse:#?}");
    assert!(retained.final_rmse < 1e-6, "{retained:#?}");
    assert!(retained.retained_camera_steps > 0, "{retained:#?}");
    assert_objective_close(dense.final_cost, sparse.final_cost);
    assert_objective_close(dense.final_cost, retained.final_cost);
    assert_state_close(
        &dense_poses,
        &dense_points,
        &dense_groups,
        &sparse_poses,
        &sparse_points,
        &sparse_groups,
    );
    assert_state_close(
        &dense_poses,
        &dense_points,
        &dense_groups,
        &retained_poses,
        &retained_points,
        &retained_groups,
    );
    assert_eq!(sparse_poses[3], initial_poses[3]);
    assert_eq!(sparse_points[0], initial_points[0]);
    assert_eq!(sparse_groups[0], initial_groups[0]);
}

#[test]
fn grouped_prefix_intrinsic_dimensions_refine_only_the_declared_parameters() {
    let (truth_poses, truth_points) = scene();
    let camera_to_group = [0, 0, 1, 1];
    let truth_groups = [
        RadialCamera::new(685.0, 0.052, -0.021),
        RadialCamera::new(735.0, -0.043, 0.018),
    ];
    let (observations, offsets) = observations(&truth_poses, &truth_points, |camera| {
        truth_groups[camera_to_group[camera]]
    });
    let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let (initial_poses, initial_points) = perturbed_scene(&truth_poses, &truth_points);
    // Group 0 refines its focal length only and group 1 adds k1; every trailing
    // parameter starts at the truth and must come back bit-identical.
    let initial_groups = [
        RadialCamera::new(699.0, truth_groups[0].k1, truth_groups[0].k2),
        RadialCamera::new(722.0, -0.031, truth_groups[1].k2),
    ];
    let intrinsic_dimensions = [1, 2];
    let fixed = FixedVariableBlocks {
        intrinsic_dimensions: &intrinsic_dimensions,
        ..FixedVariableBlocks::NONE
    };

    let mut dense_poses = initial_poses;
    let mut dense_points = initial_points.clone();
    let mut dense_groups = initial_groups;
    let mut dense_workspace = SelfCalibrationWorkspace::try_new(CAMERA_COUNT, truth_points.len())
        .expect("dense grouped workspace must fit");
    let dense = refine_ba_multi_view_self_calibrating_grouped_dense_with_factors(
        &mut dense_poses,
        &mut dense_points,
        &mut dense_groups,
        &camera_to_group,
        measurements,
        gauge,
        fixed,
        BundleAdjustmentPriors::NONE,
        &mut dense_workspace,
        Options::l2(80),
        NoiseModel::Unit,
    )
    .expect("dense prefix-dimension solve must be valid");

    let mut sparse_poses = initial_poses;
    let mut sparse_points = initial_points.clone();
    let mut sparse_groups = initial_groups;
    let mut sparse_workspace = SparseSelfCalibrationWorkspace::try_new_grouped(
        CAMERA_COUNT,
        truth_groups.len(),
        &camera_to_group,
        measurements,
        gauge,
        fixed,
    )
    .expect("prefix-dimension sparse topology must be valid");
    let sparse = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
        &mut sparse_poses,
        &mut sparse_points,
        &mut sparse_groups,
        &camera_to_group,
        measurements,
        gauge,
        fixed,
        BundleAdjustmentPriors::NONE,
        &mut sparse_workspace,
        Options::l2(80),
        NoiseModel::Unit,
    )
    .expect("prefix-dimension sparse solve must be valid");

    let mut retained_poses = initial_poses;
    let mut retained_points = initial_points.clone();
    let mut retained_groups = initial_groups;
    let mut retained_workspace =
        SparseSelfCalibrationWorkspace::try_new_grouped_with_iterative_schur_options(
            CAMERA_COUNT,
            truth_groups.len(),
            &camera_to_group,
            measurements,
            gauge,
            fixed,
            IterativeSchurOptions::new().with_maximum_iterations(300),
        )
        .expect("prefix-dimension retained topology must be valid");
    let retained = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
        &mut retained_poses,
        &mut retained_points,
        &mut retained_groups,
        &camera_to_group,
        measurements,
        gauge,
        fixed,
        BundleAdjustmentPriors::NONE,
        &mut retained_workspace,
        Options::l2(80),
        NoiseModel::Unit,
    )
    .expect("prefix-dimension retained solve must be valid");

    assert!(dense.final_rmse < 1e-6, "{dense:#?}");
    assert!(sparse.final_rmse < 1e-6, "{sparse:#?}");
    assert!(retained.final_rmse < 1e-6, "{retained:#?}");
    for groups in [&dense_groups, &sparse_groups, &retained_groups] {
        assert!((groups[0].focal / truth_groups[0].focal - 1.0).abs() < 1e-8);
        assert!((groups[1].focal / truth_groups[1].focal - 1.0).abs() < 1e-8);
        assert!((groups[1].k1 - truth_groups[1].k1).abs() < 1e-8);
        assert_eq!(groups[0].k1.to_bits(), initial_groups[0].k1.to_bits());
        assert_eq!(groups[0].k2.to_bits(), initial_groups[0].k2.to_bits());
        assert_eq!(groups[1].k2.to_bits(), initial_groups[1].k2.to_bits());
    }
    assert_objective_close(dense.final_cost, sparse.final_cost);
    assert_objective_close(dense.final_cost, retained.final_cost);

    // Declaring all three parameters free is the empty mask, bit for bit.
    let all_free = [3, 3];
    let solve = |fixed: FixedVariableBlocks<'_>| {
        let mut poses = initial_poses;
        let mut points = initial_points.clone();
        let mut groups = initial_groups;
        let mut workspace = SparseSelfCalibrationWorkspace::try_new_grouped(
            CAMERA_COUNT,
            truth_groups.len(),
            &camera_to_group,
            measurements,
            gauge,
            fixed,
        )
        .expect("full-dimension sparse topology must be valid");
        let report = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
            &mut poses,
            &mut points,
            &mut groups,
            &camera_to_group,
            measurements,
            gauge,
            fixed,
            BundleAdjustmentPriors::NONE,
            &mut workspace,
            Options::l2(80),
            NoiseModel::Unit,
        )
        .expect("full-dimension sparse solve must be valid");
        (poses, points, groups, report.final_cost.to_bits())
    };
    assert_eq!(
        solve(FixedVariableBlocks {
            intrinsic_dimensions: &all_free,
            ..FixedVariableBlocks::NONE
        }),
        solve(FixedVariableBlocks::NONE)
    );
}
