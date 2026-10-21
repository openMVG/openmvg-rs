// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct Schur assembly determinism, rank-probe restoration, and auto-selection policy.

use super::super::*;
use super::fixture::fixture;
#[cfg(feature = "parallel")]
use super::fixture::relative_norm_error;
#[cfg(feature = "parallel")]
use crate::kernel::multi_view::{PointPrior, PosePrior};

#[cfg(feature = "parallel")]
#[test]
fn direct_assembly_is_thread_deterministic_and_matches_serial_numerically() {
    let fixture = fixture(32, 256, 8);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let model = ObservationModel::from(measurements);
    let scene_scale = optimizer::scene_scale(&fixture.poses, &fixture.points, gauge);
    let gauge_state = GaugeState::new(gauge, &fixture.poses, scene_scale);
    let mut fixed_poses = vec![false; fixture.poses.len()];
    fixed_poses[7] = true;
    let mut fixed_points = vec![false; fixture.points.len()];
    fixed_points[3] = true;
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_poses,
        points: &fixed_points,
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    let mut pose_mean = fixture.poses[2];
    pose_mean.translation[0] += 0.01;
    let pose_priors = [PosePrior {
        camera: 2,
        mean: pose_mean,
        sqrt_information: std::array::from_fn(|row| {
            std::array::from_fn(|column| if row == column { 0.5 } else { 0.0 })
        }),
    }];
    let point_priors = [PointPrior {
        point: 4,
        mean: [
            fixture.points[4][0] + 0.02,
            fixture.points[4][1] - 0.01,
            fixture.points[4][2] + 0.03,
        ],
        sqrt_information: std::array::from_fn(|row| {
            std::array::from_fn(|column| if row == column { 0.75 } else { 0.0 })
        }),
    }];
    let priors = BundleAdjustmentPriors {
        poses: &pose_priors,
        points: &point_priors,
        intrinsics: &[],
    };
    let standard_deviations = (0..fixture.observations.len())
        .map(|observation| 0.8 + 0.05 * (observation % 7) as f64)
        .collect::<Vec<_>>();
    let mut serial = SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks(
        fixture.poses.len(),
        measurements,
        gauge,
        fixed,
        SparseWorkspaceOptions::direct(),
    )
    .expect("serial direct topology is valid");
    let mut one = SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks(
        fixture.poses.len(),
        measurements,
        gauge,
        fixed,
        SparseWorkspaceOptions::direct_parallel(1),
    )
    .expect("one-thread direct topology is valid");
    let mut four = SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks(
        fixture.poses.len(),
        measurements,
        gauge,
        fixed,
        SparseWorkspaceOptions::direct_parallel(4),
    )
    .expect("four-thread direct topology is valid");
    for damping in [1e-8, 1e-4, 1.0] {
        let serial_assembly = assemble_sparse(
            model,
            &fixture.poses,
            &fixture.points,
            gauge_state,
            Loss::Huber { delta: 2.0 },
            NoiseModel::per_observation(&standard_deviations),
            damping,
            false,
            fixed,
            priors,
            &mut serial,
        );
        let one_assembly = assemble_sparse(
            model,
            &fixture.poses,
            &fixture.points,
            gauge_state,
            Loss::Huber { delta: 2.0 },
            NoiseModel::per_observation(&standard_deviations),
            damping,
            false,
            fixed,
            priors,
            &mut one,
        );
        let four_assembly = assemble_sparse(
            model,
            &fixture.poses,
            &fixture.points,
            gauge_state,
            Loss::Huber { delta: 2.0 },
            NoiseModel::per_observation(&standard_deviations),
            damping,
            false,
            fixed,
            priors,
            &mut four,
        );
        assert!(serial_assembly.valid && one_assembly.valid && four_assembly.valid);
        assert_eq!(one_assembly.valid, four_assembly.valid);
        assert_eq!(
            one_assembly.weak_point_count,
            four_assembly.weak_point_count
        );
        assert_eq!(
            one_assembly.minimum_point_rcond.to_bits(),
            four_assembly.minimum_point_rcond.to_bits()
        );
        assert_eq!(
            one_assembly.scaled_gradient_inf_norm.to_bits(),
            four_assembly.scaled_gradient_inf_norm.to_bits()
        );
        assert_eq!(
            one_assembly.point_model_reduction.to_bits(),
            four_assembly.point_model_reduction.to_bits()
        );
        assert_eq!(one_assembly.cost.to_bits(), four_assembly.cost.to_bits());
        assert_eq!(
            one_assembly.squared_residual.to_bits(),
            four_assembly.squared_residual.to_bits()
        );

        assert_eq!(one.core.values, four.core.values);
        assert_eq!(one.core.gradient, four.core.gradient);
        assert_eq!(one.core.full_gradient, four.core.full_gradient);
        assert_eq!(one.core.diagonal, four.core.diagonal);
        assert_eq!(one.core.point_cache_valid, four.core.point_cache_valid);
        assert_eq!(
            one.core.point_inverse_gradients,
            four.core.point_inverse_gradients
        );
        assert_eq!(
            one.core.point_damping_diagonals,
            four.core.point_damping_diagonals
        );
        assert_eq!(
            one.core.point_inverse_cross_values.as_f64(),
            four.core.point_inverse_cross_values.as_f64()
        );
        assert!(relative_norm_error(&one.core.values, &serial.core.values) <= 1e-12);
        assert!(relative_norm_error(&one.core.gradient, &serial.core.gradient) <= 1e-15);
    }
}

#[test]
fn free_gauge_rank_probe_restores_sparse_values_bitwise() {
    let fixture = fixture(4, 32, 3);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::Free;
    let model = ObservationModel::from(measurements);
    let scene_scale = optimizer::scene_scale(&fixture.poses, &fixture.points, gauge);
    let mut workspace =
        SparseBundleAdjustmentWorkspace::try_new(fixture.poses.len(), measurements, gauge)
            .expect("fixture topology is valid");
    let assembly = assemble_sparse(
        model,
        &fixture.poses,
        &fixture.points,
        GaugeState::new(gauge, &fixture.poses, scene_scale),
        Loss::L2,
        NoiseModel::Unit,
        0.0,
        false,
        FixedVariableBlocks::NONE,
        BundleAdjustmentPriors::NONE,
        &mut workspace,
    );
    assert!(assembly.valid);
    let before: Vec<u64> = workspace
        .core
        .values
        .iter()
        .map(|value| value.to_bits())
        .collect();

    let _ = assess_camera_rank_with_expected_nullity(
        &mut workspace.core,
        workspace.backend.direct_executor(),
        &fixture.poses,
        1,
    );

    let after: Vec<u64> = workspace
        .core
        .values
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(after, before);
}

#[cfg(feature = "parallel")]
#[test]
fn direct_auto_is_topology_and_memory_gated() {
    let build = |camera_count, point_count, track_length| {
        let fixture = fixture(camera_count, point_count, track_length);
        let measurements = MultiViewMeasurements::Pixel {
            observations: &fixture.observations,
            point_offsets: &fixture.offsets,
            cameras: &fixture.cameras,
        };
        SparseBundleAdjustmentWorkspace::try_new_with_options(
            camera_count,
            measurements,
            ScaleGauge::FixedBaseline { camera: 1 },
            SparseWorkspaceOptions::direct_auto(),
        )
        .expect("auto direct topology is valid")
    };

    let small = build(8, 32, 4);
    assert_eq!(small.direct_schur_parallel_threads(), None);

    let indexed = build(64, 32, 8);
    assert_eq!(indexed.direct_schur_parallel_threads(), Some(1));

    let observation_driven = build(64, 256, 2);
    assert_eq!(observation_driven.direct_schur_parallel_threads(), Some(1));

    let policy_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("policy test pool is valid");
    let large = policy_pool.install(|| build(64, 4096, 8));
    assert_eq!(large.direct_schur_parallel_threads(), Some(4));
    assert!(large.direct_schur_assembly_bytes() <= 256 * 1024 * 1024);
}
