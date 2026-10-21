// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Translation-only Schur-PCG solves, fallbacks, and determinism.

use super::*;

#[test]
fn translation_only_cached_and_recomputed_pcg_are_supported() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    for (matvec, expected_backend) in [
        (
            RetainedSchurMatvec::Fused,
            SparseExecutionBackend::CachedIterativeSchur,
        ),
        (
            RetainedSchurMatvec::Recomputed,
            SparseExecutionBackend::RecomputedIterativeSchur,
        ),
    ] {
        let options = IterativeSchurOptions::new()
            .with_matvec(matvec)
            .with_relative_tolerance(1e-8)
            .with_exact_camera_rank_diagnostics(false);
        let mut workspace = translation_only_workspace_with_options(
            fixture.initial_poses.len(),
            measurements,
            gauge,
            SparseWorkspaceOptions::iterative_schur(options),
        );
        assert_eq!(
            workspace.execution_decision().selected_backend(),
            expected_backend
        );
        let mut poses = fixture.initial_poses.clone();
        let rotations = poses.iter().map(|pose| pose.rotation).collect::<Vec<_>>();
        let mut points = fixture.initial_points.clone();
        let report = refine_ba_multi_view_sparse(
            &mut poses,
            &mut points,
            measurements,
            gauge,
            &mut workspace,
            Options::l2(30),
            NoiseModel::Unit,
        )
        .expect("translation-only retained Schur solve succeeds");

        assert!(report.final_cost < report.initial_cost);
        assert_eq!(
            report.camera_pose_variables,
            CameraPoseVariables::TranslationOnly
        );
        assert_eq!(
            report.execution_decision.selected_backend(),
            expected_backend
        );
        assert!(report.retained_camera_steps > 0);
        assert_eq!(
            report.camera_solve_quality,
            SparseCameraSolveQuality::ResidualBoundedIterativeSchur
        );
        for (pose, rotation) in poses.iter().zip(rotations) {
            assert_eq!(pose.rotation, rotation);
        }
    }
}

#[test]
fn translation_only_llt_and_pcg_have_matching_solutions() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let solve = |workspace_options| {
        let mut workspace = translation_only_workspace_with_options(
            fixture.initial_poses.len(),
            measurements,
            gauge,
            workspace_options,
        );
        let mut poses = fixture.initial_poses.clone();
        let mut points = fixture.initial_points.clone();
        let report = refine_ba_multi_view_sparse(
            &mut poses,
            &mut points,
            measurements,
            gauge,
            &mut workspace,
            Options::l2(40),
            NoiseModel::Unit,
        )
        .expect("translation-only sparse solve succeeds");
        (poses, points, report)
    };
    let direct = solve(SparseWorkspaceOptions::direct());
    for matvec in [RetainedSchurMatvec::Fused, RetainedSchurMatvec::Recomputed] {
        let iterative = solve(SparseWorkspaceOptions::iterative_schur(
            IterativeSchurOptions::new()
                .with_matvec(matvec)
                .with_maximum_iterations(1_000)
                .with_relative_tolerance(1e-10),
        ));
        assert_close(direct.2.final_cost, iterative.2.final_cost, 1e-7);
        for (direct_pose, iterative_pose) in direct.0.iter().zip(&iterative.0) {
            assert_eq!(direct_pose.rotation, iterative_pose.rotation);
            for (direct_coordinate, iterative_coordinate) in direct_pose
                .translation
                .iter()
                .zip(iterative_pose.translation.iter())
            {
                assert_close(*direct_coordinate, *iterative_coordinate, 2e-6);
            }
        }
        for (direct_point, iterative_point) in direct.1.iter().zip(&iterative.1) {
            for (direct_coordinate, iterative_coordinate) in
                direct_point.iter().zip(iterative_point)
            {
                assert_close(*direct_coordinate, *iterative_coordinate, 2e-6);
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn translation_only_parallel_pcg_policies_match_direct() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let solve = |workspace_options| {
        let mut workspace = translation_only_workspace_with_options(
            fixture.initial_poses.len(),
            measurements,
            gauge,
            workspace_options,
        );
        let selected_backend = workspace.execution_decision().selected_backend();
        let retained_threads = workspace.retained_schur_parallel_threads();
        let mut poses = fixture.initial_poses.clone();
        let mut points = fixture.initial_points.clone();
        let report = refine_ba_multi_view_sparse(
            &mut poses,
            &mut points,
            measurements,
            gauge,
            &mut workspace,
            Options::l2(40),
            NoiseModel::Unit,
        )
        .expect("parallel translation-only sparse solve succeeds");
        (poses, points, report, selected_backend, retained_threads)
    };
    let direct = solve(SparseWorkspaceOptions::direct());
    for (matvec, expected_backend) in [
        (
            RetainedSchurMatvec::PointMajorParallel { threads: 2 },
            SparseExecutionBackend::CachedIterativeSchur,
        ),
        (
            RetainedSchurMatvec::RecomputedParallel { threads: 2 },
            SparseExecutionBackend::RecomputedIterativeSchur,
        ),
    ] {
        let iterative = solve(SparseWorkspaceOptions::iterative_schur(
            IterativeSchurOptions::new()
                .with_matvec(matvec)
                .with_maximum_iterations(1_000)
                .with_relative_tolerance(1e-10),
        ));
        assert_eq!(iterative.3, expected_backend);
        assert_eq!(iterative.4, Some(2));
        assert_close(direct.2.final_cost, iterative.2.final_cost, 1e-7);
        for (direct_pose, iterative_pose) in direct.0.iter().zip(&iterative.0) {
            assert_eq!(direct_pose.rotation, iterative_pose.rotation);
            for (direct_coordinate, iterative_coordinate) in direct_pose
                .translation
                .iter()
                .zip(iterative_pose.translation.iter())
            {
                assert_close(*direct_coordinate, *iterative_coordinate, 2e-6);
            }
        }
        for (direct_point, iterative_point) in direct.1.iter().zip(&iterative.1) {
            for (direct_coordinate, iterative_coordinate) in
                direct_point.iter().zip(iterative_point)
            {
                assert_close(*direct_coordinate, *iterative_coordinate, 2e-6);
            }
        }
    }
}

#[test]
fn translation_only_serial_pcg_is_deterministic_and_preserves_rotations() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let run = || {
        let mut workspace = translation_only_workspace_with_options(
            fixture.initial_poses.len(),
            measurements,
            gauge,
            SparseWorkspaceOptions::iterative_schur(
                IterativeSchurOptions::new()
                    .with_matvec(RetainedSchurMatvec::Fused)
                    .with_relative_tolerance(1e-9),
            ),
        );
        let mut poses = fixture.initial_poses.clone();
        let mut points = fixture.initial_points.clone();
        let report = refine_ba_multi_view_sparse(
            &mut poses,
            &mut points,
            measurements,
            gauge,
            &mut workspace,
            Options::l2(30),
            NoiseModel::Unit,
        )
        .expect("deterministic translation-only PCG solve succeeds");
        (poses, points, report)
    };

    let first = run();
    let second = run();
    assert_eq!(first, second);
    for (pose, initial) in first.0.iter().zip(&fixture.initial_poses) {
        assert_eq!(pose.rotation, initial.rotation);
    }
}

#[test]
fn translation_only_pcg_uses_guarded_direct_fallback() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let mut workspace = translation_only_workspace_with_options(
        fixture.initial_poses.len(),
        measurements,
        gauge,
        SparseWorkspaceOptions::iterative_schur(
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::Fused)
                .with_maximum_iterations(1)
                .with_relative_tolerance(1e-14),
        ),
    );
    workspace
        .try_reserve_direct_storage()
        .expect("guarded exact fallback storage is available");
    let mut poses = fixture.initial_poses.clone();
    let rotations = poses.iter().map(|pose| pose.rotation).collect::<Vec<_>>();
    let mut points = fixture.initial_points.clone();
    let report = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut workspace,
        Options::l2(10),
        NoiseModel::Unit,
    )
    .expect("translation-only PCG falls back to direct LLT");

    assert!(report.direct_fallback_steps > 0);
    assert!(report.linear_guard_fallback_steps > 0);
    assert_eq!(
        report.camera_solve_quality,
        SparseCameraSolveQuality::IterativeSchurWithDirectFallback
    );
    for (pose, rotation) in poses.iter().zip(rotations) {
        assert_eq!(pose.rotation, rotation);
    }
}
