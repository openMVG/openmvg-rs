// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained Schur representation regression tests.

use super::super::*;
use super::fixture::fixture;

#[test]
fn retained_materialization_matches_direct_schur() {
    let fixture = fixture(8, 96, 5);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let model = ObservationModel::from(measurements);
    let scene_scale = optimizer::scene_scale(&fixture.poses, &fixture.points, gauge);
    for damping in [0.0, 1e-4, 0.3] {
        let mut direct =
            SparseBundleAdjustmentWorkspace::try_new(fixture.poses.len(), measurements, gauge)
                .expect("direct topology is valid");
        let mut retained = SparseBundleAdjustmentWorkspace::try_new_with_camera_solver(
            fixture.poses.len(),
            measurements,
            gauge,
            SparseCameraSolver::IterativeSchur,
        )
        .expect("retained topology is valid");
        let gauge_state = GaugeState::new(gauge, &fixture.poses, scene_scale);
        let direct_assembly = assemble_sparse(
            model,
            &fixture.poses,
            &fixture.points,
            gauge_state,
            Loss::L2,
            NoiseModel::Unit,
            damping,
            false,
            FixedVariableBlocks::NONE,
            BundleAdjustmentPriors::NONE,
            &mut direct,
        );
        let retained_assembly = assemble_sparse(
            model,
            &fixture.poses,
            &fixture.points,
            gauge_state,
            Loss::L2,
            NoiseModel::Unit,
            damping,
            true,
            FixedVariableBlocks::NONE,
            BundleAdjustmentPriors::NONE,
            &mut retained,
        );
        assert!(direct_assembly.valid && retained_assembly.valid);
        assert!(
            materialize_retained_schur(model, &mut retained)
                .expect("direct fallback storage is available")
        );

        let matrix_scale = maximum_absolute_value(&direct.core.values);
        let matrix_error = maximum_absolute_difference(&direct.core.values, &retained.core.values)
            / matrix_scale.max(f64::MIN_POSITIVE);
        let gradient_scale = maximum_absolute_value(&direct.core.gradient);
        let gradient_error =
            maximum_absolute_difference(&direct.core.gradient, &retained.core.gradient)
                / gradient_scale.max(f64::MIN_POSITIVE);
        assert!(
            matrix_error < 2e-12,
            "damping={damping} error={matrix_error:e}"
        );
        assert!(
            gradient_error < 2e-14,
            "damping={damping} error={gradient_error:e}"
        );
    }
}

#[test]
fn packed_retained_storage_is_smaller_than_the_previous_layout() {
    let camera_count = 32;
    let point_count = 128;
    let fixture = fixture(camera_count, point_count, 5);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let workspace = SparseBundleAdjustmentWorkspace::try_new_with_camera_solver(
        camera_count,
        measurements,
        ScaleGauge::FixedBaseline { camera: 1 },
        SparseCameraSolver::IterativeSchur,
    )
    .expect("retained topology is valid");
    let dimension = workspace.camera_dimension();
    let previous_bytes = 632 * camera_count + 72 * point_count + 48 * dimension;
    assert!(workspace.retained_schur_bytes() < previous_bytes);
}

fn maximum_absolute_value(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn maximum_absolute_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}
