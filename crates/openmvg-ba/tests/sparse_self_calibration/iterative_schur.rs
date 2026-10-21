// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained Schur-PCG self-calibration, storage, and determinism.

use super::*;

#[test]
fn self_calibrating_iterative_initial_damping_override_is_reported() {
    let (truth_poses, truth_points) = scene();
    let mut cameras = [RadialCamera::new(700.0, 0.01, -0.002); CAMERA_COUNT];
    let (observations, offsets) =
        observations(&truth_poses, &truth_points, |camera| cameras[camera]);
    let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let (mut poses, mut points) = perturbed_scene(&truth_poses, &truth_points);
    let mut workspace = SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
        CAMERA_COUNT,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        IterativeSchurOptions::new(),
    )
    .expect("fixture topology is valid")
    .with_initial_damping(0.025);

    let report = refine_ba_multi_view_self_calibrating_sparse(
        &mut poses,
        &mut points,
        &mut cameras,
        measurements,
        gauge,
        &mut workspace,
        Options::l2(0),
        NoiseModel::Unit,
    )
    .expect("zero-budget diagnostics are valid");

    assert_eq!(report.initial_damping, 0.025);
    assert_eq!(report.final_damping, 0.025);
}

#[cfg(feature = "parallel")]
#[test]
fn iterative_self_calibration_is_bitwise_stable_across_thread_counts() {
    const MULTI_SHARD_POINT_COUNT: usize = 14_337;
    let (truth_poses, seed_points) = scene();
    // 57,348 observations force eight topology-defined logical shards. This
    // exercises genuinely different executor widths instead of proving only
    // that a single shard is stable when installed on differently sized pools.
    let truth_points = (0..MULTI_SHARD_POINT_COUNT)
        .map(|index| {
            let mut point = seed_points[index % seed_points.len()];
            let phase = index as f64;
            point[0] += 1e-4 * (0.17 * phase).sin();
            point[1] += 1e-4 * (0.11 * phase).cos();
            point[2] += 1e-4 * (0.07 * phase).sin();
            point
        })
        .collect::<Vec<_>>();
    let truth_cameras = [
        RadialCamera::new(680.0, 0.055, -0.022),
        RadialCamera::new(725.0, -0.041, 0.017),
        RadialCamera::new(695.0, 0.032, -0.013),
        RadialCamera::new(740.0, -0.048, 0.020),
    ];
    let (observations, offsets) =
        observations(&truth_poses, &truth_points, |camera| truth_cameras[camera]);
    let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
    let camera_to_group = [0, 1, 2, 3];
    let fixed_poses = [true, false, false, false];
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_poses,
        points: &[],
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    let (initial_poses, initial_points) = perturbed_scene(&truth_poses, &truth_points);
    let initial_cameras = [
        RadialCamera::new(688.0, 0.058, -0.023),
        RadialCamera::new(718.0, -0.038, 0.016),
        RadialCamera::new(701.0, 0.030, -0.012),
        RadialCamera::new(733.0, -0.045, 0.019),
    ];

    let run = |threads| {
        let mut poses = initial_poses;
        let mut points = initial_points.clone();
        let mut cameras = initial_cameras;
        let mut workspace =
            SparseSelfCalibrationWorkspace::try_new_grouped_with_iterative_schur_options(
                CAMERA_COUNT,
                camera_to_group.len(),
                &camera_to_group,
                measurements,
                ScaleGauge::Free,
                fixed,
                IterativeSchurOptions::new()
                    .with_matvec(RetainedSchurMatvec::PointMajorParallel { threads })
                    .with_maximum_iterations(300)
                    .with_exact_camera_rank_diagnostics(false),
            )
            .expect("iterative self-calibration topology is valid");
        assert_eq!(workspace.retained_schur_parallel_threads(), Some(threads));
        let report = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
            &mut poses,
            &mut points,
            &mut cameras,
            &camera_to_group,
            measurements,
            ScaleGauge::Free,
            fixed,
            BundleAdjustmentPriors::NONE,
            &mut workspace,
            Options::l2(1),
            NoiseModel::Unit,
        )
        .expect("iterative self-calibration solve is valid");
        assert_eq!(report.direct_fallback_steps, 0, "{report:#?}");
        assert!(report.retained_camera_steps > 0, "{report:#?}");
        (exact_state_bits(&poses, &points, &cameras), report)
    };

    let expected = run(1);
    for threads in [2, 4, 8] {
        let actual = run(threads);
        assert_eq!(actual.0, expected.0, "{threads}-thread state drifted");
        assert_eq!(actual.1.status, expected.1.status);
        assert_eq!(actual.1.iteration_count, expected.1.iteration_count);
        assert_eq!(actual.1.accepted_step_count, expected.1.accepted_step_count);
        assert_eq!(actual.1.rejected_step_count, expected.1.rejected_step_count);
        assert_eq!(
            actual.1.retained_inner_iterations,
            expected.1.retained_inner_iterations
        );
        assert_eq!(
            actual.1.initial_cost.to_bits(),
            expected.1.initial_cost.to_bits()
        );
        assert_eq!(
            actual.1.final_cost.to_bits(),
            expected.1.final_cost.to_bits()
        );
        assert_eq!(
            actual.1.final_rmse.to_bits(),
            expected.1.final_rmse.to_bits()
        );
        assert_eq!(
            actual
                .1
                .maximum_retained_relative_residual
                .map(f64::to_bits),
            expected
                .1
                .maximum_retained_relative_residual
                .map(f64::to_bits)
        );
    }
}

#[test]
fn retained_only_self_calibration_avoids_direct_numeric_storage() {
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
    let (mut recomputed_poses, mut recomputed_points) =
        perturbed_scene(&truth_poses, &truth_points);
    let mut recomputed_cameras = truth_cameras;
    let mut recomputed_workspace =
        SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
            CAMERA_COUNT,
            measurements,
            gauge,
            FixedVariableBlocks::NONE,
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::Recomputed)
                .with_exact_camera_rank_diagnostics(false),
        )
        .expect("recomputed self-calibration topology is valid");
    assert!(!recomputed_workspace.retains_cross_rows());
    assert_eq!(recomputed_workspace.retained_cross_precision(), None);
    assert_eq!(recomputed_workspace.retained_cross_row_bytes(), 0);
    assert_eq!(recomputed_workspace.direct_numeric_bytes(), 0);
    let recomputed = refine_ba_multi_view_self_calibrating_sparse(
        &mut recomputed_poses,
        &mut recomputed_points,
        &mut recomputed_cameras,
        measurements,
        gauge,
        &mut recomputed_workspace,
        Options::l2(1),
        NoiseModel::Unit,
    )
    .expect("recomputed self-calibration step is valid");
    assert_eq!(recomputed.direct_fallback_steps, 0, "{recomputed:#?}");
    assert_eq!(recomputed_workspace.direct_numeric_bytes(), 0);
    let (mut poses, mut points) = perturbed_scene(&truth_poses, &truth_points);
    let mut cameras = truth_cameras;
    let mut workspace = SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
        CAMERA_COUNT,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        IterativeSchurOptions::new()
            .with_matvec(RetainedSchurMatvec::Fused)
            .with_exact_camera_rank_diagnostics(false),
    )
    .expect("retained-only self-calibration topology is valid");
    assert_eq!(workspace.direct_numeric_bytes(), 0);
    let lazy_pattern_bytes = workspace.direct_pattern_bytes();
    assert_eq!(workspace.reduced_nonzero_count(), 0);

    let report = refine_ba_multi_view_self_calibrating_sparse(
        &mut poses,
        &mut points,
        &mut cameras,
        measurements,
        gauge,
        &mut workspace,
        Options::l2(1),
        NoiseModel::Unit,
    )
    .expect("the retained-only step is valid");

    assert_eq!(report.direct_fallback_steps, 0, "{report:#?}");
    assert_eq!(
        report.diagnostics.reduced_nonzero_count,
        workspace.reduced_nonzero_count()
    );
    assert_eq!(workspace.direct_numeric_bytes(), 0);
    assert_eq!(workspace.direct_pattern_bytes(), lazy_pattern_bytes);
    assert_eq!(workspace.reduced_nonzero_count(), 0);
    assert_eq!(
        report.solver_working_set_bytes,
        workspace.solver_specific_working_set_bytes()
    );
    assert_eq!(
        report.diagnostics.pose_rank,
        SparseCameraRankAssessment::NotAssessed
    );
    assert_objective_close(report.final_cost, recomputed.final_cost);
    assert_state_close(
        &poses,
        &points,
        &cameras,
        &recomputed_poses,
        &recomputed_points,
        &recomputed_cameras,
    );

    let (mut poses, mut points) = perturbed_scene(&truth_poses, &truth_points);
    let mut cameras = truth_cameras;
    let mut guarded_workspace =
        SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
            CAMERA_COUNT,
            measurements,
            gauge,
            FixedVariableBlocks::NONE,
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::Recomputed)
                .with_maximum_iterations(1)
                .with_exact_camera_rank_diagnostics(false),
        )
        .expect("guarded retained topology is valid");
    assert_eq!(guarded_workspace.direct_numeric_bytes(), 0);
    assert_eq!(guarded_workspace.direct_pattern_bytes(), lazy_pattern_bytes);
    assert_eq!(guarded_workspace.reduced_nonzero_count(), 0);
    let guarded = refine_ba_multi_view_self_calibrating_sparse(
        &mut poses,
        &mut points,
        &mut cameras,
        measurements,
        gauge,
        &mut guarded_workspace,
        Options::l2(1),
        NoiseModel::Unit,
    )
    .expect("a failed finite PCG guard must fall back safely");
    assert_eq!(guarded.direct_fallback_steps, 1, "{guarded:#?}");
    assert!(guarded_workspace.direct_numeric_bytes() > 0);
    assert!(guarded_workspace.direct_pattern_bytes() > lazy_pattern_bytes);
    assert!(guarded_workspace.reduced_nonzero_count() > 0);
}

#[test]
fn retained_only_self_calibration_materializes_direct_pattern_on_demand() {
    let (truth_poses, truth_points) = scene();
    let truth_cameras = [RadialCamera::new(700.0, 0.01, -0.002); CAMERA_COUNT];
    let (observations, offsets) =
        observations(&truth_poses, &truth_points, |camera| truth_cameras[camera]);
    let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
    let mut workspace = SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
        CAMERA_COUNT,
        measurements,
        ScaleGauge::FixedBaseline { camera: 1 },
        FixedVariableBlocks::NONE,
        IterativeSchurOptions::new().with_exact_camera_rank_diagnostics(false),
    )
    .expect("retained-only self-calibration topology is valid");
    let lazy_pattern_bytes = workspace.direct_pattern_bytes();

    assert_eq!(workspace.reduced_nonzero_count(), 0);
    assert_eq!(workspace.direct_numeric_bytes(), 0);

    workspace
        .try_reserve_direct_storage()
        .expect("the deferred direct pattern is materialized on demand");

    assert!(workspace.reduced_nonzero_count() > 0);
    assert!(workspace.direct_pattern_bytes() > lazy_pattern_bytes);
    assert!(workspace.direct_numeric_bytes() > 0);
}

#[test]
fn self_calibration_f32_cross_rows_halve_storage_and_preserve_the_solve() {
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
    let common = IterativeSchurOptions::new()
        .with_maximum_iterations(300)
        .with_exact_camera_rank_diagnostics(false);
    let mut f64_workspace = SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
        CAMERA_COUNT,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        common,
    )
    .unwrap();
    let mut f32_workspace = SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
        CAMERA_COUNT,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        common.with_cross_precision(RetainedCrossPrecision::F32),
    )
    .unwrap();
    assert_eq!(
        f32_workspace.retained_cross_precision(),
        Some(RetainedCrossPrecision::F32)
    );
    assert_eq!(
        2 * f32_workspace.retained_cross_row_bytes(),
        f64_workspace.retained_cross_row_bytes()
    );

    let (initial_poses, initial_points) = perturbed_scene(&truth_poses, &truth_points);
    let mut f64_poses = initial_poses;
    let mut f64_points = initial_points.clone();
    let mut f64_cameras = truth_cameras;
    let mut f32_poses = initial_poses;
    let mut f32_points = initial_points;
    let mut f32_cameras = truth_cameras;
    let f64_report = refine_ba_multi_view_self_calibrating_sparse(
        &mut f64_poses,
        &mut f64_points,
        &mut f64_cameras,
        measurements,
        gauge,
        &mut f64_workspace,
        Options::l2(4),
        NoiseModel::Unit,
    )
    .unwrap();
    let f32_report = refine_ba_multi_view_self_calibrating_sparse(
        &mut f32_poses,
        &mut f32_points,
        &mut f32_cameras,
        measurements,
        gauge,
        &mut f32_workspace,
        Options::l2(4),
        NoiseModel::Unit,
    )
    .unwrap();
    assert_eq!(f32_report.direct_fallback_steps, 0, "{f32_report:#?}");
    let relative_cost = (f32_report.final_cost - f64_report.final_cost).abs()
        / f64_report.final_cost.abs().max(1.0);
    assert!(
        relative_cost <= 1e-6,
        "relative cost drift {relative_cost:e}"
    );
}

#[test]
fn retained_self_calibration_confirms_cost_convergence_without_direct_fallback() {
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
    let (mut poses, mut points) = perturbed_scene(&truth_poses, &truth_points);
    let mut cameras = truth_cameras;
    let mut workspace = SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
        CAMERA_COUNT,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        IterativeSchurOptions::new()
            .with_relative_tolerance(0.5)
            .with_exact_camera_rank_diagnostics(false),
    )
    .expect("the retained self-calibration convergence fixture is valid");
    let options = Options::l2(4).with_convergence(ConvergenceCriteria::new(1e-30, 1e-30, 1.0));

    let report = refine_ba_multi_view_self_calibrating_sparse(
        &mut poses,
        &mut points,
        &mut cameras,
        measurements,
        gauge,
        &mut workspace,
        options,
        NoiseModel::Unit,
    )
    .expect("retained self-calibration convergence confirmation is valid");

    assert_eq!(
        report.status,
        RefinementStatus::Converged(ConvergenceCriterion::Cost),
        "{report:#?}"
    );
    assert!(report.iteration_count >= 2, "{report:#?}");
    assert!(
        report
            .iterative_convergence_confirmation_relative_residual()
            .is_some_and(|residual| residual <= 1e-6),
        "{report:#?}"
    );
    assert_eq!(report.direct_fallback_steps, 0, "{report:#?}");
    assert_eq!(workspace.direct_numeric_bytes(), 0);
}
