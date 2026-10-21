// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend selection, storage accounting, and workspace topology checks.

use super::*;

#[test]
fn self_calibrating_auto_uses_the_lowest_memory_supported_backend() {
    let (truth_poses, truth_points) = scene();
    let truth_cameras = [RadialCamera::new(700.0, 0.01, -0.002); CAMERA_COUNT];
    let (observations, offsets) =
        observations(&truth_poses, &truth_points, |camera| truth_cameras[camera]);
    let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
    let mut workspace = SparseSelfCalibrationWorkspace::try_new_auto(
        CAMERA_COUNT,
        measurements,
        ScaleGauge::FixedBaseline { camera: 1 },
        FixedVariableBlocks::NONE,
        SparseWorkspaceMemoryBudget::Bytes(1),
    )
    .expect("the soft budget selects retained self-calibration");
    assert_eq!(
        workspace.execution_decision().selected_backend(),
        SparseExecutionBackend::CachedIterativeSchur
    );
    assert_eq!(
        workspace.initial_damping(),
        SparseCameraSolver::IterativeSchur.default_initial_damping()
    );
    assert_eq!(
        workspace.execution_decision().reason(),
        SparseExecutionReason::LowestMemoryBackend
    );
    assert_eq!(
        workspace.execution_decision().selected_workspace_bytes(),
        workspace.solver_specific_working_set_bytes()
    );
    assert!(workspace.reduced_nonzero_count() > 0);
    assert_eq!(
        workspace.try_reserve_direct_storage(),
        Err(SparseBundleAdjustmentError::DirectStorageExceedsMemoryBudget),
        "lazy direct storage must respect the budget that excluded it"
    );
}

#[cfg(feature = "parallel")]
#[test]
fn candidate_metric_storage_is_counted_and_gated_by_fixed_point_topology() {
    let (truth_poses, truth_points) = scene();
    let truth_cameras = [RadialCamera::new(700.0, 0.01, -0.002); CAMERA_COUNT];
    let (observations, offsets) =
        observations(&truth_poses, &truth_points, |camera| truth_cameras[camera]);
    let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
    let options = IterativeSchurOptions::new()
        .with_matvec(RetainedSchurMatvec::PointMajorParallel { threads: 2 })
        .with_exact_camera_rank_diagnostics(false);
    let eligible = SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
        CAMERA_COUNT,
        measurements,
        ScaleGauge::FixedBaseline { camera: 1 },
        FixedVariableBlocks::NONE,
        options,
    )
    .expect("all-free candidate topology is valid");
    // An explicit mask fixing nothing is the same topology as no mask.
    let explicit_free_points = vec![false; truth_points.len()];
    let explicit_free = SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
        CAMERA_COUNT,
        measurements,
        ScaleGauge::FixedBaseline { camera: 1 },
        FixedVariableBlocks {
            camera_poses: &[],
            points: &explicit_free_points,
            intrinsics: &[],
            intrinsic_dimensions: &[],
        },
        options,
    )
    .expect("explicit all-free topology is valid");
    assert_eq!(
        explicit_free.retained_schur_bytes(),
        eligible.retained_schur_bytes()
    );
    assert_eq!(
        explicit_free.point_linearization_cache_bytes(),
        eligible.point_linearization_cache_bytes()
    );
    let mut one_fixed_point = explicit_free_points;
    one_fixed_point[0] = true;
    let ineligible = SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
        CAMERA_COUNT,
        measurements,
        ScaleGauge::FixedBaseline { camera: 1 },
        FixedVariableBlocks {
            camera_poses: &[],
            points: &one_fixed_point,
            intrinsics: &[],
            intrinsic_dimensions: &[],
        },
        options,
    )
    .expect("explicit fixed-point topology is valid");
    let metric_bytes = truth_points
        .len()
        .saturating_mul(std::mem::size_of::<[f64; 3]>());

    // A genuinely fixed point disables the candidate metrics (and shrinks the
    // retained operator), so at least the metric storage disappears.
    assert!(
        ineligible
            .retained_schur_bytes()
            .saturating_add(metric_bytes)
            <= eligible.retained_schur_bytes()
    );
    assert!(
        ineligible
            .solver_specific_working_set_bytes()
            .saturating_add(metric_bytes)
            <= eligible.solver_specific_working_set_bytes()
    );
}

#[test]
fn grouped_sparse_workspace_rejects_a_changed_intrinsic_assignment() {
    let (truth_poses, truth_points) = scene();
    let camera_to_group = [0, 0, 1, 1];
    let changed_assignment = [0, 1, 0, 1];
    let truth_groups = [
        RadialCamera::new(685.0, 0.052, -0.021),
        RadialCamera::new(735.0, -0.043, 0.018),
    ];
    let (observations, offsets) = observations(&truth_poses, &truth_points, |camera| {
        truth_groups[camera_to_group[camera]]
    });
    let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let mut workspace = SparseSelfCalibrationWorkspace::try_new_grouped(
        CAMERA_COUNT,
        truth_groups.len(),
        &camera_to_group,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
    )
    .expect("the original grouped topology is valid");
    let (mut poses, mut points) = perturbed_scene(&truth_poses, &truth_points);
    let mut groups = truth_groups;
    let error = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
        &mut poses,
        &mut points,
        &mut groups,
        &changed_assignment,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        BundleAdjustmentPriors::NONE,
        &mut workspace,
        Options::l2(1),
        NoiseModel::Unit,
    )
    .expect_err("changing shared-intrinsic ownership requires a new workspace");
    assert_eq!(error, SparseBundleAdjustmentError::TopologyMismatch);
}

#[test]
fn grouped_workspace_rejects_impossible_group_count_before_allocation() {
    let (truth_poses, truth_points) = scene();
    let truth_camera = RadialCamera::new(700.0, 0.0, 0.0);
    let (observations, offsets) = observations(&truth_poses, &truth_points, |_| truth_camera);
    let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
    let error = SparseSelfCalibrationWorkspace::try_new_grouped(
        CAMERA_COUNT,
        usize::MAX,
        &[0; CAMERA_COUNT],
        measurements,
        ScaleGauge::FixedBaseline { camera: 1 },
        FixedVariableBlocks::NONE,
    )
    .err()
    .expect("an impossible group count must be rejected before allocation");
    assert_eq!(
        error,
        SparseBundleAdjustmentError::Problem(
            openmvg_ba::kernel::multi_view::MultiViewBundleAdjustmentError::InvalidIntrinsicGroupCount {
                cameras: CAMERA_COUNT,
                groups: usize::MAX,
            }
        )
    );
}
