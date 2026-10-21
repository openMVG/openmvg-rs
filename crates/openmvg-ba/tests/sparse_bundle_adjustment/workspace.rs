// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Workspace selection, memory budgets, and topology checks.

use super::*;

#[cfg(feature = "parallel")]
#[test]
fn direct_solve_is_bitwise_stable_across_ambient_pool_widths() {
    let fixture = fixture();
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let (initial_poses, initial_points) = perturbed(&fixture.truth_poses, &fixture.truth_points);
    let run = |threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("test pool is constructible");
        pool.install(|| {
            let mut poses = initial_poses.clone();
            let mut points = initial_points.clone();
            let mut workspace = SparseBundleAdjustmentWorkspace::try_new_with_options(
                poses.len(),
                measurements,
                gauge,
                SparseWorkspaceOptions::direct(),
            )
            .expect("fixture topology is valid");
            let report = refine_ba_multi_view_sparse(
                &mut poses,
                &mut points,
                measurements,
                gauge,
                &mut workspace,
                Options::l2(20),
                NoiseModel::Unit,
            )
            .expect("direct solve succeeds");
            (poses, points, report)
        })
    };

    let expected = run(1);
    for threads in [2, 4, 8] {
        assert_eq!(run(threads), expected, "{threads}-thread solve drifted");
    }
}

#[test]
fn borrowed_problem_and_unified_sparse_options_preserve_workspace_policy() {
    let fixture = fixture();
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let problem = CalibratedBundleAdjustmentProblem::new(
        measurements,
        ScaleGauge::FixedBaseline { camera: 1 },
    );
    let workspace = problem
        .try_sparse_workspace(
            fixture.truth_poses.len(),
            SparseWorkspaceOptions::direct_with_point_back_substitution(
                SparsePointBackSubstitution::Recompute,
            ),
        )
        .expect("fixture topology is valid");

    assert_eq!(problem.point_count(), fixture.truth_points.len());
    assert_eq!(problem.observation_count(), fixture.observations.len());
    assert_eq!(workspace.camera_solver(), SparseCameraSolver::DirectLlt);
    assert_eq!(
        workspace.initial_damping(),
        SparseCameraSolver::DirectLlt.default_initial_damping()
    );
    assert_eq!(
        workspace.point_back_substitution(),
        SparsePointBackSubstitution::Recompute
    );
    assert_eq!(workspace.point_linearization_cache_bytes(), 0);
}

#[test]
fn automatic_workspace_selection_is_visible_and_budget_monotonic() {
    let fixture = fixture();
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let direct = SparseBundleAdjustmentWorkspace::try_new_with_options(
        fixture.truth_poses.len(),
        measurements,
        gauge,
        SparseWorkspaceOptions::auto(),
    )
    .expect("the default automatic budget supports the exact fixture");
    let direct_decision = direct.execution_decision();
    assert!(direct_decision.is_automatic());
    assert_eq!(
        direct_decision.selected_backend(),
        SparseExecutionBackend::DirectLlt
    );
    assert_eq!(
        direct_decision.reason(),
        SparseExecutionReason::DirectFitsBudget
    );
    assert_eq!(
        direct.initial_damping(),
        SparseCameraSolver::DirectLlt.default_initial_damping()
    );
    assert_eq!(
        direct_decision.selected_workspace_bytes(),
        direct.solver_specific_working_set_bytes()
    );

    let mut recomputed = SparseBundleAdjustmentWorkspace::try_new_with_options(
        fixture.truth_poses.len(),
        measurements,
        gauge,
        SparseWorkspaceOptions::auto_with_memory_budget(SparseWorkspaceMemoryBudget::Bytes(1)),
    )
    .expect("the soft budget selects the lowest-memory calibrated backend");
    assert_eq!(
        recomputed.execution_decision().selected_backend(),
        SparseExecutionBackend::RecomputedIterativeSchur
    );
    assert_eq!(
        recomputed.execution_decision().reason(),
        SparseExecutionReason::CachedRetainedExceedsBudget
    );
    assert_eq!(
        recomputed.initial_damping(),
        SparseCameraSolver::IterativeSchur.default_initial_damping()
    );
    let expected_decision = recomputed.execution_decision();
    let (mut poses, mut points) = perturbed(&fixture.truth_poses, &fixture.truth_points);
    let report = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut recomputed,
        Options::l2(5),
        NoiseModel::Unit,
    )
    .expect("the automatically selected recomputed backend executes");
    assert_eq!(report.execution_decision, expected_decision);
    assert_eq!(
        recomputed.try_reserve_direct_storage(),
        Err(SparseBundleAdjustmentError::DirectStorageExceedsMemoryBudget),
        "lazy direct storage must respect the budget that excluded it"
    );

    assert!(matches!(
        SparseBundleAdjustmentWorkspace::try_new_with_options(
            fixture.truth_poses.len(),
            measurements,
            gauge,
            SparseWorkspaceOptions::auto_with_memory_budget(SparseWorkspaceMemoryBudget::Bytes(0),),
        ),
        Err(SparseBundleAdjustmentError::WorkspaceMemoryBudgetMustBePositive)
    ));
}

#[test]
fn sparse_workspace_rejects_a_changed_visibility_graph() {
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
    let mut workspace = SparseBundleAdjustmentWorkspace::try_new(
        4,
        measurements,
        ScaleGauge::FixedBaseline { camera: 1 },
    )
    .expect("fixture topology is valid");
    let mut changed = observations.clone();
    changed[0].camera = 1;
    let changed_measurements = MultiViewMeasurements::Pixel {
        observations: &changed,
        point_offsets: &offsets,
        cameras: &cameras,
    };
    let (mut poses, mut points) = perturbed(&truth_poses, &truth_points);
    let result = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        changed_measurements,
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut workspace,
        Options::l2(1),
        NoiseModel::Unit,
    );
    assert_eq!(result, Err(SparseBundleAdjustmentError::TopologyMismatch));
}
