// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained and recomputed Schur-PCG solves and their policies.

use super::*;

#[test]
fn calibrated_iterative_schur_supports_fixed_blocks_and_priors() {
    let fixture = fixture();
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let fixed_pose_mask = [false, false, false, true];
    let mut fixed_point_mask = vec![false; fixture.truth_points.len()];
    fixed_point_mask[0] = true;
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_pose_mask,
        points: &fixed_point_mask,
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    let point_priors = [PointPrior {
        point: 1,
        mean: fixture.truth_points[1],
        sqrt_information: [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
    }];
    let priors = BundleAdjustmentPriors {
        poses: &[],
        points: &point_priors,
        intrinsics: &[],
    };
    let iterative_options = IterativeSchurOptions::new()
        .with_relative_tolerance(1e-10)
        .with_exact_camera_rank_diagnostics(false);
    let mut workspace = SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks(
        4,
        measurements,
        gauge,
        fixed,
        SparseWorkspaceOptions::iterative_schur(iterative_options),
    )
    .unwrap();
    let (mut poses, mut points) = perturbed(&fixture.truth_poses, &fixture.truth_points);
    let fixed_pose = poses[3];
    let fixed_point = points[0];

    let report = refine_ba_multi_view_sparse_with_factors(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        fixed,
        priors,
        &mut workspace,
        Options::l2(20),
        NoiseModel::Unit,
    )
    .unwrap();

    assert_eq!(poses[3], fixed_pose);
    assert_eq!(points[0], fixed_point);
    assert!(report.final_cost < report.initial_cost, "{report:#?}");
    assert_eq!(report.direct_fallback_steps, 0, "{report:#?}");
    assert!(report.retained_camera_steps > 0, "{report:#?}");
}

#[test]
fn cached_and_recomputed_point_back_substitution_agree() {
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
    let mut cached_poses = initial_poses.clone();
    let mut cached_points = initial_points.clone();
    let mut recomputed_poses = initial_poses;
    let mut recomputed_points = initial_points;
    let mut cached = SparseBundleAdjustmentWorkspace::try_new(4, measurements, gauge)
        .expect("fixture topology is valid");
    let mut recomputed = SparseBundleAdjustmentWorkspace::try_new_with_point_back_substitution(
        4,
        measurements,
        gauge,
        SparsePointBackSubstitution::Recompute,
    )
    .expect("fixture topology is valid");
    let cached_report = refine_ba_multi_view_sparse(
        &mut cached_poses,
        &mut cached_points,
        measurements,
        gauge,
        &mut cached,
        Options::l2(60),
        NoiseModel::Unit,
    )
    .expect("cached fixture is valid");
    let recomputed_report = refine_ba_multi_view_sparse(
        &mut recomputed_poses,
        &mut recomputed_points,
        measurements,
        gauge,
        &mut recomputed,
        Options::l2(60),
        NoiseModel::Unit,
    )
    .expect("recomputed fixture is valid");

    assert!(cached.point_linearization_cache_bytes() > 0);
    // This fixture retains 24 validity flags, two [f64; 3] arrays per point,
    // 25 point offsets, and 360 transformed cross rows. Rust Vec<bool> stores
    // one byte per element; it is not a packed bit vector.
    let expected_cache_bytes = 24 * std::mem::size_of::<bool>()
        + 2 * 24 * std::mem::size_of::<[f64; 3]>()
        + 25 * std::mem::size_of::<usize>()
        + 360 * std::mem::size_of::<[f64; 3]>();
    assert_eq!(
        cached.point_linearization_cache_bytes(),
        expected_cache_bytes
    );
    assert_eq!(recomputed.point_linearization_cache_bytes(), 0);
    assert_eq!(
        cached.point_back_substitution(),
        SparsePointBackSubstitution::Cached
    );
    assert_eq!(
        recomputed.point_back_substitution(),
        SparsePointBackSubstitution::Recompute
    );
    assert_eq!(cached_report.status, recomputed_report.status);
    assert_eq!(
        cached_report.iteration_count,
        recomputed_report.iteration_count
    );
    assert_eq!(
        cached_report.accepted_step_count,
        recomputed_report.accepted_step_count
    );
    assert_eq!(
        cached_report.rejected_step_count,
        recomputed_report.rejected_step_count
    );
    assert!((cached_report.final_cost - recomputed_report.final_cost).abs() < 1e-10);
    for (cached, recomputed) in cached_poses.iter().zip(&recomputed_poses) {
        for row in 0..3 {
            for column in 0..3 {
                assert!(
                    (cached.rotation[row][column] - recomputed.rotation[row][column]).abs() < 1e-10
                );
            }
            assert!((cached.translation[row] - recomputed.translation[row]).abs() < 1e-10);
        }
    }
    for (cached, recomputed) in cached_points.iter().zip(&recomputed_points) {
        for coordinate in 0..3 {
            assert!((cached[coordinate] - recomputed[coordinate]).abs() < 1e-10);
        }
    }
}

#[test]
fn opt_in_f32_retained_cross_cache_preserves_the_bounded_solve() {
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
    let options = IterativeSchurOptions::new().with_exact_camera_rank_diagnostics(false);
    let mut f64_workspace = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
        4,
        measurements,
        gauge,
        options,
    )
    .unwrap();
    let mut f32_workspace = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
        4,
        measurements,
        gauge,
        options.with_cross_precision(RetainedCrossPrecision::F32),
    )
    .unwrap();
    assert_eq!(
        f64_workspace.retained_cross_precision(),
        Some(RetainedCrossPrecision::F64)
    );
    assert_eq!(
        f32_workspace.retained_cross_precision(),
        Some(RetainedCrossPrecision::F32)
    );
    assert!(
        f32_workspace.point_linearization_cache_bytes()
            < f64_workspace.point_linearization_cache_bytes()
    );
    assert_eq!(
        2 * f32_workspace.retained_cross_row_bytes(),
        f64_workspace.retained_cross_row_bytes()
    );

    let (initial_poses, initial_points) = perturbed(&truth_poses, &truth_points);
    let mut f64_poses = initial_poses.clone();
    let mut f64_points = initial_points.clone();
    let mut f32_poses = initial_poses;
    let mut f32_points = initial_points;
    let solve_options = Options::l2(12);
    let f64_report = refine_ba_multi_view_sparse(
        &mut f64_poses,
        &mut f64_points,
        measurements,
        gauge,
        &mut f64_workspace,
        solve_options,
        NoiseModel::Unit,
    )
    .unwrap();
    let f32_report = refine_ba_multi_view_sparse(
        &mut f32_poses,
        &mut f32_points,
        measurements,
        gauge,
        &mut f32_workspace,
        solve_options,
        NoiseModel::Unit,
    )
    .unwrap();
    assert_eq!(f32_report.direct_fallback_steps, 0);
    assert_eq!(
        f32_report.accepted_step_count,
        f64_report.accepted_step_count
    );
    assert_eq!(
        f32_report.rejected_step_count,
        f64_report.rejected_step_count
    );
    let relative_cost = (f32_report.final_cost - f64_report.final_cost).abs()
        / f64_report.final_cost.abs().max(1.0);
    assert!(
        relative_cost <= 1e-7,
        "relative cost drift {relative_cost:e}"
    );
}

#[test]
fn recomputed_schur_rows_match_retained_pcg_without_a_cross_cache() {
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
    let common = IterativeSchurOptions::new().with_exact_camera_rank_diagnostics(false);
    let mut retained = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
        4,
        measurements,
        gauge,
        common,
    )
    .unwrap();
    let mut recomputed = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
        4,
        measurements,
        gauge,
        common.with_matvec(RetainedSchurMatvec::Recomputed),
    )
    .unwrap();
    let mut parallel = SparseBundleAdjustmentWorkspace::try_new_with_options(
        4,
        measurements,
        gauge,
        SparseWorkspaceOptions::low_memory(2),
    )
    .unwrap();
    assert!(retained.retains_cross_rows());
    assert!(!recomputed.retains_cross_rows());
    assert_eq!(recomputed.retained_cross_precision(), None);
    assert_eq!(recomputed.retained_cross_row_bytes(), 0);
    assert!(!parallel.retains_cross_rows());
    assert_eq!(parallel.retained_cross_precision(), None);
    assert_eq!(parallel.retained_cross_row_bytes(), 0);
    assert_eq!(parallel.direct_numeric_bytes(), 0);
    assert!(!parallel.exact_camera_rank_diagnostics());
    let topology_offset_bytes = (recomputed.point_count() + 1) * std::mem::size_of::<usize>();
    assert_eq!(
        recomputed.point_linearization_cache_bytes(),
        topology_offset_bytes
    );
    assert_eq!(
        parallel.point_linearization_cache_bytes(),
        topology_offset_bytes
    );
    assert_eq!(parallel.retained_schur_parallel_threads(), Some(2));
    assert_eq!(
        parallel.point_back_substitution(),
        SparsePointBackSubstitution::Recompute
    );
    assert_eq!(
        recomputed.point_back_substitution(),
        SparsePointBackSubstitution::Recompute
    );

    let (initial_poses, initial_points) = perturbed(&truth_poses, &truth_points);
    let mut retained_poses = initial_poses.clone();
    let mut retained_points = initial_points.clone();
    let mut recomputed_poses = initial_poses;
    let mut recomputed_points = initial_points;
    let mut parallel_poses = recomputed_poses.clone();
    let mut parallel_points = recomputed_points.clone();
    let options = Options::l2(8);
    let retained_report = refine_ba_multi_view_sparse(
        &mut retained_poses,
        &mut retained_points,
        measurements,
        gauge,
        &mut retained,
        options,
        NoiseModel::Unit,
    )
    .unwrap();
    let recomputed_report = refine_ba_multi_view_sparse(
        &mut recomputed_poses,
        &mut recomputed_points,
        measurements,
        gauge,
        &mut recomputed,
        options,
        NoiseModel::Unit,
    )
    .unwrap();
    let parallel_report = refine_ba_multi_view_sparse(
        &mut parallel_poses,
        &mut parallel_points,
        measurements,
        gauge,
        &mut parallel,
        options,
        NoiseModel::Unit,
    )
    .unwrap();
    assert_eq!(recomputed_report.direct_fallback_steps, 0);
    assert_eq!(
        recomputed_report.accepted_step_count,
        retained_report.accepted_step_count
    );
    assert_eq!(
        recomputed_report.rejected_step_count,
        retained_report.rejected_step_count
    );
    assert_eq!(
        recomputed_report.iterative_inner_iterations(),
        retained_report.iterative_inner_iterations()
    );
    let relative_cost = (recomputed_report.final_cost - retained_report.final_cost).abs()
        / retained_report.final_cost.abs().max(1.0);
    assert!(
        relative_cost <= 1e-12,
        "relative cost drift {relative_cost:e}"
    );
    assert_eq!(
        parallel_report.camera_solve_quality,
        SparseCameraSolveQuality::ResidualBoundedIterativeSchur
    );
    let maximum_residual = parallel_report
        .maximum_iterative_relative_residual()
        .expect("the low-memory solve reports its true PCG residual");
    assert!(maximum_residual.is_finite());
    assert!(
        maximum_residual
            <= IterativeSchurOptions::DEFAULT_RELATIVE_TOLERANCE * (1.0 + 32.0 * f64::EPSILON)
    );
    assert_eq!(parallel_report.direct_fallback_steps, 0);
    assert_eq!(
        parallel_report.diagnostics.pose_rank,
        SparseCameraRankAssessment::NotAssessed
    );
    assert_eq!(
        parallel_report.accepted_step_count,
        retained_report.accepted_step_count
    );
    assert_eq!(
        parallel_report.rejected_step_count,
        retained_report.rejected_step_count
    );
    assert_eq!(
        parallel_report.iterative_inner_iterations(),
        retained_report.iterative_inner_iterations()
    );
    let parallel_relative_cost = (parallel_report.final_cost - retained_report.final_cost).abs()
        / retained_report.final_cost.abs().max(1.0);
    assert!(
        parallel_relative_cost <= 1e-12,
        "parallel relative cost drift {parallel_relative_cost:e}"
    );

    let (mut fallback_poses, mut fallback_points) = perturbed(&truth_poses, &truth_points);
    let mut fallback = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
        4,
        measurements,
        gauge,
        IterativeSchurOptions::new()
            .with_matvec(RetainedSchurMatvec::Recomputed)
            .with_maximum_iterations(1)
            .with_exact_camera_rank_diagnostics(false),
    )
    .unwrap();
    assert_eq!(fallback.direct_numeric_bytes(), 0);
    let fallback_report = refine_ba_multi_view_sparse(
        &mut fallback_poses,
        &mut fallback_points,
        measurements,
        gauge,
        &mut fallback,
        Options::l2(1),
        NoiseModel::Unit,
    )
    .unwrap();
    assert_eq!(fallback_report.direct_fallback_steps, 1);
    assert_eq!(
        fallback_report.camera_solve_quality,
        SparseCameraSolveQuality::IterativeSchurWithDirectFallback
    );
    assert!(fallback.direct_numeric_bytes() > 0);
    assert_eq!(
        fallback_report.solver_working_set_bytes,
        fallback.solver_specific_working_set_bytes()
    );
    assert!(
        fallback_report.solver_working_set_bytes
            > fallback_report
                .execution_decision
                .selected_workspace_bytes()
    );
}

#[test]
fn iterative_schur_is_explicit_opt_in_and_converges_without_required_fallback() {
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
    let mut direct_poses = initial_poses.clone();
    let mut direct_points = initial_points.clone();
    let mut power_poses = initial_poses;
    let mut power_points = initial_points;
    let mut direct = SparseBundleAdjustmentWorkspace::try_new(4, measurements, gauge)
        .expect("fixture topology is valid");
    let mut power = SparseBundleAdjustmentWorkspace::try_new_with_camera_solver(
        4,
        measurements,
        gauge,
        SparseCameraSolver::IterativeSchur,
    )
    .expect("fixture topology is valid");
    assert_eq!(power.direct_numeric_bytes(), 0);
    let options = Options::l2(60);
    let direct_report = refine_ba_multi_view_sparse(
        &mut direct_poses,
        &mut direct_points,
        measurements,
        gauge,
        &mut direct,
        options,
        NoiseModel::Unit,
    )
    .expect("direct solve is valid");
    let power_report = refine_ba_multi_view_sparse(
        &mut power_poses,
        &mut power_points,
        measurements,
        gauge,
        &mut power,
        options,
        NoiseModel::Unit,
    )
    .expect("iterative Schur solve is valid");

    assert_eq!(power.camera_solver(), SparseCameraSolver::IterativeSchur);
    assert_eq!(
        power.initial_damping(),
        SparseCameraSolver::IterativeSchur.default_initial_damping()
    );
    assert!(power.retained_schur_bytes() > 0);
    assert!(power.direct_numeric_bytes() > 0);
    assert_eq!(power_report.initial_damping, 1e-1);
    assert!(power_report.iterative_camera_steps() > 0);
    assert!(power_report.final_damping.is_finite() && power_report.final_damping > 0.0);
    assert!(power_report.last_gain_ratio.is_none_or(f64::is_finite));
    assert_eq!(power_report.direct_fallback_steps, 0);
    assert!(power_report.iterative_inner_iterations() >= power_report.direct_fallback_steps);
    assert!(power_report.maximum_iterative_relative_residual().is_some());
    assert_eq!(
        power_report.camera_solve_quality,
        SparseCameraSolveQuality::ResidualBoundedIterativeSchur
    );
    assert!(
        power_report.final_cost <= 1.01 * direct_report.final_cost.max(1e-16),
        "direct={direct_report:#?}\npower={power_report:#?}"
    );
    assert_eq!(
        power_report.diagnostics.pose_rank,
        SparseCameraRankAssessment::FullRank
    );
    let (mut free_poses, mut free_points) = perturbed(&truth_poses, &truth_points);
    let mut free = SparseBundleAdjustmentWorkspace::try_new_with_camera_solver(
        4,
        measurements,
        ScaleGauge::Free,
        SparseCameraSolver::IterativeSchur,
    )
    .expect("retained Schur supports the damped free-scale system");
    let free_report = refine_ba_multi_view_sparse(
        &mut free_poses,
        &mut free_points,
        measurements,
        ScaleGauge::Free,
        &mut free,
        options,
        NoiseModel::Unit,
    )
    .expect("free-scale iterative Schur solve is valid");
    assert!(free_report.final_rmse < 1e-7, "{free_report:#?}");
    assert_eq!(
        free_report.diagnostics.pose_rank,
        SparseCameraRankAssessment::OnlyExpectedScaleGauge
    );
}

#[test]
fn retained_schur_matvec_policy_is_explicit_and_auto_is_conservative() {
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
    let automatic_tolerance = 5e-2;
    assert!(matches!(
        SparseWorkspaceOptions::auto_with_iterative_relative_tolerance(
            SparseWorkspaceMemoryBudget::Auto,
            automatic_tolerance,
        ),
        SparseWorkspaceOptions::Auto {
            iterative_relative_tolerance_bits,
            ..
        } if f64::from_bits(iterative_relative_tolerance_bits) == automatic_tolerance
    ));
    let mut fused = SparseBundleAdjustmentWorkspace::try_new_with_camera_solver(
        4,
        measurements,
        gauge,
        SparseCameraSolver::IterativeSchur,
    )
    .expect("fused iterative Schur topology is valid");
    let mut point_major = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_matvec(
        4,
        measurements,
        gauge,
        RetainedSchurMatvec::PointMajorParallel { threads: 2 },
    )
    .expect("parallel point-major iterative Schur topology is valid");
    let automatic = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_matvec(
        4,
        measurements,
        gauge,
        RetainedSchurMatvec::Auto,
    )
    .expect("automatic iterative Schur topology is valid");
    let tuned = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
        4,
        measurements,
        gauge,
        IterativeSchurOptions::new()
            .with_maximum_iterations(321)
            .with_relative_tolerance(1e-4)
            .with_matvec(RetainedSchurMatvec::Fused),
    )
    .expect("explicit iterative Schur options are valid");
    let mut retained_only = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
        4,
        measurements,
        gauge,
        IterativeSchurOptions::new().with_exact_camera_rank_diagnostics(false),
    )
    .expect("retained-only iterative Schur options are valid");

    assert_eq!(
        fused.retained_schur_matvec(),
        Some(RetainedSchurMatvec::Fused)
    );
    assert_eq!(fused.retained_schur_parallel_threads(), None);
    assert_eq!(
        fused.retained_working_set_bytes(),
        fused.retained_schur_bytes() + fused.point_linearization_cache_bytes()
    );
    assert_eq!(
        fused.solver_specific_working_set_bytes(),
        fused.retained_working_set_bytes() + fused.direct_numeric_bytes()
    );
    assert_eq!(
        point_major.retained_schur_matvec(),
        Some(RetainedSchurMatvec::PointMajorParallel { threads: 2 })
    );
    assert_eq!(point_major.retained_schur_parallel_threads(), Some(2));
    assert_eq!(
        automatic.retained_schur_matvec(),
        Some(RetainedSchurMatvec::Auto)
    );
    assert_eq!(automatic.retained_schur_parallel_threads(), None);
    assert_eq!(tuned.iterative_schur_maximum_iterations(), Some(321));
    assert_eq!(tuned.iterative_schur_relative_tolerance(), Some(1e-4));
    assert_eq!(
        tuned.initial_damping(),
        SparseCameraSolver::IterativeSchur.default_initial_damping()
    );
    let tuned = tuned.with_initial_damping(0.025);
    assert_eq!(tuned.initial_damping(), 0.025);
    assert!(!retained_only.exact_camera_rank_diagnostics());
    assert_eq!(retained_only.direct_numeric_bytes(), 0);
    let lazy_pattern_bytes = retained_only.direct_pattern_bytes();
    assert!(lazy_pattern_bytes < fused.direct_pattern_bytes());
    assert!(retained_only.reduced_nonzero_count() > 0);
    assert!(matches!(
        SparseBundleAdjustmentWorkspace::try_new_with_options(
            4,
            measurements,
            gauge,
            SparseWorkspaceOptions::low_memory(0),
        ),
        Err(SparseBundleAdjustmentError::RetainedSchurThreadsMustBePositive)
    ));
    assert!(matches!(
        SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_matvec(
            4,
            measurements,
            gauge,
            RetainedSchurMatvec::PointMajorParallel { threads: 0 },
        ),
        Err(SparseBundleAdjustmentError::RetainedSchurThreadsMustBePositive)
    ));
    assert!(matches!(
        SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
            4,
            measurements,
            gauge,
            IterativeSchurOptions::new().with_maximum_iterations(0),
        ),
        Err(SparseBundleAdjustmentError::IterativeSchurIterationsMustBePositive)
    ));
    assert!(matches!(
        SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
            4,
            measurements,
            gauge,
            IterativeSchurOptions::new().with_relative_tolerance(f64::NAN),
        ),
        Err(SparseBundleAdjustmentError::IterativeSchurToleranceInvalid)
    ));
    let (mut fused_poses, mut fused_points) = perturbed(&truth_poses, &truth_points);
    let fused_report = refine_ba_multi_view_sparse(
        &mut fused_poses,
        &mut fused_points,
        measurements,
        gauge,
        &mut fused,
        Options::l2(4),
        NoiseModel::Unit,
    )
    .expect("fused iterative Schur solve is valid");
    let (mut point_major_poses, mut point_major_points) = perturbed(&truth_poses, &truth_points);
    let point_major_report = refine_ba_multi_view_sparse(
        &mut point_major_poses,
        &mut point_major_points,
        measurements,
        gauge,
        &mut point_major,
        Options::l2(4),
        NoiseModel::Unit,
    )
    .expect("parallel point-major iterative Schur solve is valid");
    assert_eq!(
        point_major_report.accepted_step_count,
        fused_report.accepted_step_count
    );
    assert_eq!(
        point_major_report.rejected_step_count,
        fused_report.rejected_step_count
    );
    assert!((point_major_report.final_cost - fused_report.final_cost).abs() <= 1e-10);

    let (mut poses, mut points) = perturbed(&truth_poses, &truth_points);
    let report = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut retained_only,
        Options::l2(4),
        NoiseModel::Unit,
    )
    .expect("retained-only iterative Schur solve is valid");
    assert_eq!(
        report.diagnostics.pose_rank,
        SparseCameraRankAssessment::NotAssessed
    );
    assert_eq!(retained_only.direct_numeric_bytes(), 0);
    assert_eq!(retained_only.direct_pattern_bytes(), lazy_pattern_bytes);
    assert!(!matches!(report.status, RefinementStatus::Degenerate));
}

#[test]
fn iterative_schur_confirms_cost_convergence_without_direct_fallback() {
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
    let (mut poses, mut points) = perturbed(&truth_poses, &truth_points);
    let mut workspace = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
        poses.len(),
        measurements,
        gauge,
        IterativeSchurOptions::new()
            .with_relative_tolerance(0.5)
            .with_exact_camera_rank_diagnostics(false),
    )
    .expect("the retained convergence fixture is valid");
    let options = Options::l2(4).with_convergence(ConvergenceCriteria::new(1e-30, 1e-30, 1.0));

    let report = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut workspace,
        options,
        NoiseModel::Unit,
    )
    .expect("residual-bounded convergence confirmation is valid");

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
