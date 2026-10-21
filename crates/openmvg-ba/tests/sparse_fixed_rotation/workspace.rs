// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Translation-only backend selection, memory receipts, and scale.

use super::*;

#[test]
fn translation_only_auto_selection_and_memory_receipts_are_consistent() {
    let fixture = chain_fixture(300, 2);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.pixels,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let unbounded = translation_only_workspace_with_options(
        fixture.poses.len(),
        measurements,
        gauge,
        SparseWorkspaceOptions::auto_with_memory_budget(SparseWorkspaceMemoryBudget::Unbounded),
    );
    let unbounded_decision = unbounded.execution_decision();
    assert!(unbounded_decision.is_automatic());
    assert_eq!(
        unbounded_decision.selected_backend(),
        SparseExecutionBackend::DirectLlt
    );
    assert_eq!(
        unbounded_decision.selected_workspace_bytes(),
        unbounded.solver_specific_working_set_bytes()
    );

    let cached_budget = unbounded_decision
        .estimated_cached_retained_bytes()
        .expect("Auto reports its cached retained estimate");
    let cached = translation_only_workspace_with_options(
        fixture.poses.len(),
        measurements,
        gauge,
        SparseWorkspaceOptions::auto_with_memory_budget(SparseWorkspaceMemoryBudget::Bytes(
            cached_budget,
        )),
    );
    assert_eq!(
        cached.execution_decision().selected_backend(),
        SparseExecutionBackend::CachedIterativeSchur
    );
    assert!(cached.retains_cross_rows());
    assert!(cached.retained_cross_row_bytes() > 0);

    let recomputed = translation_only_workspace_with_options(
        fixture.poses.len(),
        measurements,
        gauge,
        SparseWorkspaceOptions::auto_with_memory_budget(SparseWorkspaceMemoryBudget::Bytes(1)),
    );
    assert_eq!(
        recomputed.execution_decision().selected_backend(),
        SparseExecutionBackend::RecomputedIterativeSchur
    );
    assert!(!recomputed.retains_cross_rows());
    assert_eq!(recomputed.retained_cross_row_bytes(), 0);
    assert!(
        recomputed.solver_specific_working_set_bytes()
            <= cached.solver_specific_working_set_bytes()
    );
    assert_eq!(
        cached.camera_pose_variables(),
        CameraPoseVariables::TranslationOnly
    );
    assert_eq!(cached.camera_dimension(), 3 * (fixture.poses.len() - 2) + 2);
    assert!(cached.reduced_nonzero_count() >= cached.camera_dimension());
}

#[test]
fn translation_only_sparse_workspace_scales_to_two_thousand_cameras() {
    const CAMERA_COUNT: usize = 2_000;
    let fixture = chain_fixture(CAMERA_COUNT, 2);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.pixels,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let mut workspace = translation_only_workspace_with_options(
        fixture.poses.len(),
        measurements,
        gauge,
        SparseWorkspaceOptions::auto(),
    );
    // The dense fixed-rotation path materializes about eight (3N)^2 f64 matrices.
    let dense_bytes = (3 * CAMERA_COUNT as u128).pow(2) * 8 * 8;
    let sparse_bytes = workspace.solver_specific_working_set_bytes();

    assert_eq!(workspace.camera_dimension(), 3 * (CAMERA_COUNT - 2) + 2);
    assert!(workspace.reduced_nonzero_count() >= workspace.camera_dimension());
    assert!((sparse_bytes as u128).saturating_mul(10) < dense_bytes);
    assert_eq!(
        workspace.execution_decision().selected_workspace_bytes(),
        sparse_bytes
    );

    let rotations = fixture
        .poses
        .iter()
        .map(|pose| pose.rotation)
        .collect::<Vec<_>>();
    let mut poses = fixture.poses;
    let mut points = fixture.points;
    let report = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut workspace,
        Options::l2(0),
        NoiseModel::Unit,
    )
    .expect("2k-camera translation-only sparse admission remains finite");

    assert_eq!(report.pose_count, CAMERA_COUNT);
    assert_eq!(
        report.camera_pose_variables,
        CameraPoseVariables::TranslationOnly
    );
    assert_eq!(report.iteration_count, 0);
    for (pose, rotation) in poses.iter().zip(rotations) {
        assert_eq!(pose.rotation, rotation);
    }
}
