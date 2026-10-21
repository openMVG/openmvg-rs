// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::tests::fixture::{Fixture, fixture, relative_norm_error};
use super::matvec::calibrated_point_major_worker_bytes;
use super::operator::*;
use super::point_major::*;
use super::*;
use crate::kernel::multi_view::schur::GaugeState;
#[cfg(feature = "parallel")]
use crate::kernel::multi_view::sparse::api::RetainedCrossPrecision;
use crate::kernel::multi_view::sparse::workspace::SparseBundleAdjustmentWorkspace;
#[cfg(feature = "parallel")]
use crate::kernel::multi_view::sparse::{SparseWorkspaceOptions, refine_ba_multi_view_sparse};
use crate::kernel::multi_view::{MultiViewMeasurements, ScaleGauge};
#[cfg(feature = "parallel")]
use crate::numerics::refine::Options;
use crate::numerics::refine::{Loss, NoiseModel};

fn retained_fixture(
    fixture: &Fixture,
    damping: f64,
) -> (
    ObservationModel<'_>,
    SparseBundleAdjustmentWorkspace,
    RetainedSchurWorkspace,
) {
    retained_fixture_with_gauge(fixture, damping, ScaleGauge::FixedBaseline { camera: 1 })
}

fn retained_fixture_with_gauge(
    fixture: &Fixture,
    damping: f64,
    gauge: ScaleGauge,
) -> (
    ObservationModel<'_>,
    SparseBundleAdjustmentWorkspace,
    RetainedSchurWorkspace,
) {
    retained_fixture_with_options(fixture, damping, gauge, IterativeSchurOptions::new())
}

fn retained_fixture_with_options(
    fixture: &Fixture,
    damping: f64,
    gauge: ScaleGauge,
    options: IterativeSchurOptions,
) -> (
    ObservationModel<'_>,
    SparseBundleAdjustmentWorkspace,
    RetainedSchurWorkspace,
) {
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let model = ObservationModel::from(measurements);
    let scene_scale = super::super::optimizer::scene_scale(&fixture.poses, &fixture.points, gauge);
    let mut workspace = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
        fixture.poses.len(),
        measurements,
        gauge,
        options,
    )
    .expect("retained fixture topology is valid");
    let assembly = super::super::assemble_sparse(
        model,
        &fixture.poses,
        &fixture.points,
        GaugeState::new(gauge, &fixture.poses, scene_scale),
        Loss::L2,
        NoiseModel::Unit,
        damping,
        true,
        super::super::FixedVariableBlocks::NONE,
        super::super::BundleAdjustmentPriors::NONE,
        &mut workspace,
    );
    assert!(assembly.valid);
    let retained = workspace
        .take_retained_for_test()
        .expect("retained Schur workspace is present");
    (model, workspace, retained)
}

const TEST_CAMERA_BLOCK_DIMENSION: usize = 9;
const TEST_CAMERA_BLOCK_STORAGE: usize = 45;
type TestCameraBlock =
    PackedSymmetricCameraBlock<TEST_CAMERA_BLOCK_DIMENSION, TEST_CAMERA_BLOCK_STORAGE>;
type TestCameraBlockFactor =
    PackedCameraBlockFactor<TEST_CAMERA_BLOCK_DIMENSION, TEST_CAMERA_BLOCK_STORAGE>;

fn packed_spd_block(dimension: usize) -> (TestCameraBlock, Vec<Vec<f64>>) {
    let mut lower = vec![vec![0.0; dimension]; dimension];
    for (row, values) in lower.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate().take(row + 1) {
            *value = if row == column {
                1.25 + 0.125 * row as f64
            } else {
                0.025 * (1 + row + 2 * column) as f64
            };
        }
    }
    let dense: Vec<Vec<f64>> = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| {
                    (0..dimension.min(row + 1).min(column + 1))
                        .map(|inner| lower[row][inner] * lower[column][inner])
                        .sum()
                })
                .collect()
        })
        .collect();
    let mut packed = TestCameraBlock::default();
    for (row, values) in dense.iter().enumerate() {
        for (column, &value) in values.iter().enumerate().take(row + 1) {
            packed.add(row, column, value);
        }
    }
    (packed, dense)
}

#[test]
fn packed_camera_block_matvec_supports_dimensions_through_nine() {
    assert_eq!(TEST_CAMERA_BLOCK_STORAGE, 45);
    for dimension in [3, 5, 6, 8, 9] {
        let (packed, dense) = packed_spd_block(dimension);
        let input: Vec<_> = (0..dimension)
            .map(|index| 0.25 - 0.075 * index as f64)
            .collect();
        let expected: Vec<_> = dense
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&input)
                    .map(|(left, right)| left * right)
                    .sum::<f64>()
            })
            .collect();
        let mut actual = vec![0.0; dimension];
        packed.multiply_add(&input, &mut actual, dimension);
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = 64.0 * f64::EPSILON * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "dimension {dimension}, row {index}: {actual:e} != {expected:e}"
            );
        }
    }
}

#[test]
fn camera_block_factor_solves_dimensions_through_nine() {
    for dimension in [3, 5, 6, 8, 9] {
        let (packed, dense) = packed_spd_block(dimension);
        let right: Vec<_> = (0..dimension)
            .map(|index| 0.5 + 0.125 * index as f64)
            .collect();
        let mut factor = TestCameraBlockFactor::default();
        assert!(factor.factor(&packed, dimension), "dimension {dimension}");
        let mut solution = vec![0.0; dimension];
        factor.solve(&right, &mut solution, dimension);
        let residual_squared: f64 = dense
            .iter()
            .zip(&right)
            .map(|(row, right)| {
                let product: f64 = row
                    .iter()
                    .zip(&solution)
                    .map(|(left, right)| left * right)
                    .sum();
                (product - right).powi(2)
            })
            .sum();
        let right_squared: f64 = right.iter().map(|value| value * value).sum();
        let relative_residual = (residual_squared / right_squared).sqrt();
        assert!(
            relative_residual <= 1e-12,
            "dimension {dimension}: relative residual {relative_residual:e}"
        );
    }
}

#[test]
fn anchored_six_point_coupling_matches_the_general_operator() {
    let fixture = fixture(12, 192, 7);
    let (model, mut workspace, retained) =
        retained_fixture_with_gauge(&fixture, 1e-4, ScaleGauge::Free);
    assert!(workspace.core.anchored_six_camera_layout);
    let input: Vec<_> = (0..workspace.core.layout.dimension())
        .map(|index| (0.017 * index as f64).sin())
        .collect();
    let mut specialized = vec![0.0; workspace.core.layout.dimension()];
    let mut general = vec![0.0; workspace.core.layout.dimension()];
    apply_point_coupling(
        model,
        &workspace.core,
        &retained.point_hessians,
        &input,
        &mut specialized,
    );
    workspace.core.anchored_six_camera_layout = false;
    apply_point_coupling(
        model,
        &workspace.core,
        &retained.point_hessians,
        &input,
        &mut general,
    );
    let scale = general.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let error = specialized
        .iter()
        .zip(&general)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    assert!(error <= 64.0 * f64::EPSILON * scale.max(1.0));
}

#[test]
fn unscaled_point_coupling_is_bitwise_identical_to_unit_scaling() {
    let fixture = fixture(12, 192, 7);
    let (_model, mut workspace, retained) =
        retained_fixture_with_gauge(&fixture, 1e-4, ScaleGauge::Free);
    let input: Vec<_> = (0..workspace.core.layout.dimension())
        .map(|index| (0.017 * index as f64).sin())
        .collect();
    for anchored_six_camera_layout in [true, false] {
        workspace.core.anchored_six_camera_layout = anchored_six_camera_layout;
        let mut scaled = vec![0.0; workspace.core.layout.dimension()];
        let mut unscaled = vec![0.0; workspace.core.layout.dimension()];
        accumulate_point_coupling_range::<false>(
            &workspace.core,
            &retained.point_hessians,
            &input,
            &mut scaled,
            0,
            workspace.core.signature.point_count(),
            1.0,
        );
        accumulate_point_coupling_range::<true>(
            &workspace.core,
            &retained.point_hessians,
            &input,
            &mut unscaled,
            0,
            workspace.core.signature.point_count(),
            1.0,
        );
        assert_eq!(scaled, unscaled);
    }
}

#[test]
fn point_major_operator_matches_fused_operator() {
    let fixture = fixture(12, 192, 7);
    let (model, workspace, retained) = retained_fixture(&fixture, 1e-4);
    #[cfg(feature = "parallel")]
    let mut retained = retained;
    let input: Vec<_> = (0..workspace.core.layout.dimension())
        .map(|index| (0.013 * index as f64).sin())
        .collect();
    let mut fused = vec![0.0; workspace.core.layout.dimension()];
    let mut point_major = vec![0.0; workspace.core.layout.dimension()];

    apply_reduced_system(
        model,
        &workspace.core,
        &retained.camera_blocks,
        &retained.point_hessians,
        &input,
        &mut fused,
    );
    let mut point_major_workspace = PointMajorParallelWorkspace::try_new(
        3,
        3,
        workspace.core.signature.camera_count(),
        workspace.core.layout.dimension(),
        model.offsets(),
    )
    .expect("point-major worker workspace is valid");
    point_major_workspace.apply(
        model,
        &workspace.core,
        Some(&retained.camera_blocks),
        &retained.point_hessians,
        &input,
        &mut point_major,
        -1.0,
    );
    let scale = fused.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let error = fused
        .iter()
        .zip(&point_major)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    assert!(
        error <= 64.0 * f64::EPSILON * scale.max(1.0),
        "point-major operator error {error:e} at scale {scale:e}"
    );

    #[cfg(feature = "parallel")]
    {
        assert!(
            retained.build_diagonal_point_coupling(
                SchurOperatorContext::retained(model),
                &workspace.core,
            )
        );
        let serial_blocks = retained.diagonal_point_coupling.clone();
        let mut parallel_blocks =
            vec![PackedCameraBlock::default(); workspace.core.signature.camera_count()];
        assert!(point_major_workspace.build_diagonal_point_coupling(
            &workspace.core,
            &retained.point_hessians,
            &mut parallel_blocks,
        ));
        let scale = serial_blocks
            .iter()
            .flat_map(|block| block.values)
            .map(f64::abs)
            .fold(0.0, f64::max);
        let error = serial_blocks
            .iter()
            .zip(&parallel_blocks)
            .flat_map(|(left, right)| left.values.iter().zip(right.values))
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        assert!(error <= 1e-11 * scale.max(1.0));
    }
}

#[test]
fn recomputed_operator_matches_retained_f64_operator_without_cross_cache() {
    let fixture = fixture(12, 192, 7);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let damping = 1e-4;
    let (model, retained_workspace, mut retained_state) =
        retained_fixture_with_options(&fixture, damping, gauge, IterativeSchurOptions::new());
    let (_, recomputed_workspace, mut recomputed_state) = retained_fixture_with_options(
        &fixture,
        damping,
        gauge,
        IterativeSchurOptions::new().with_matvec(RetainedSchurMatvec::Recomputed),
    );
    let (_, parallel_workspace, mut parallel_recomputed) = retained_fixture_with_options(
        &fixture,
        damping,
        gauge,
        IterativeSchurOptions::new()
            .with_matvec(RetainedSchurMatvec::RecomputedParallel { threads: 4 }),
    );
    assert!(retained_workspace.retains_cross_rows());
    assert!(!recomputed_workspace.retains_cross_rows());
    assert!(!parallel_workspace.retains_cross_rows());
    assert_eq!(recomputed_workspace.retained_cross_row_bytes(), 0);
    assert_eq!(parallel_workspace.retained_cross_row_bytes(), 0);
    assert_eq!(parallel_recomputed.parallel_threads(), Some(4));

    let input: Vec<_> = (0..retained_workspace.core.layout.dimension())
        .map(|index| (0.019 * index as f64).cos())
        .collect();
    let mut retained_output = vec![0.0; retained_workspace.core.layout.dimension()];
    let mut recomputed_output = vec![0.0; retained_workspace.core.layout.dimension()];
    let mut parallel_output = vec![0.0; retained_workspace.core.layout.dimension()];
    retained_state.matvec.apply_reduced_system(
        SchurOperatorContext::retained(model),
        &retained_workspace.core,
        &retained_state.camera_blocks,
        &retained_state.point_hessians,
        &input,
        &mut retained_output,
    );
    let scene_scale = super::super::optimizer::scene_scale(&fixture.poses, &fixture.points, gauge);
    recomputed_state.matvec.apply_reduced_system(
        SchurOperatorContext::recomputed(RecomputedSchurContext {
            model,
            poses: &fixture.poses,
            points: &fixture.points,
            gauge: GaugeState::new(gauge, &fixture.poses, scene_scale),
            loss: Loss::L2,
            noise: NoiseModel::Unit,
        }),
        &recomputed_workspace.core,
        &recomputed_state.camera_blocks,
        &recomputed_state.point_hessians,
        &input,
        &mut recomputed_output,
    );
    parallel_recomputed.matvec.apply_reduced_system(
        SchurOperatorContext::recomputed(RecomputedSchurContext {
            model,
            poses: &fixture.poses,
            points: &fixture.points,
            gauge: GaugeState::new(gauge, &fixture.poses, scene_scale),
            loss: Loss::L2,
            noise: NoiseModel::Unit,
        }),
        &parallel_workspace.core,
        &parallel_recomputed.camera_blocks,
        &parallel_recomputed.point_hessians,
        &input,
        &mut parallel_output,
    );
    let error = relative_norm_error(&recomputed_output, &retained_output);
    assert!(error <= 2e-12, "recomputed operator error {error:e}");
    let parallel_error = relative_norm_error(&parallel_output, &retained_output);
    assert!(
        parallel_error <= 2e-12,
        "parallel recomputed operator error {parallel_error:e}"
    );
}

#[test]
fn f32_retained_rows_match_across_fused_and_point_major_operators() {
    let fixture = fixture(12, 192, 7);
    let options = IterativeSchurOptions::new()
        .with_cross_precision(super::super::api::RetainedCrossPrecision::F32);
    let (model, workspace, retained) = retained_fixture_with_options(
        &fixture,
        1e-4,
        ScaleGauge::FixedBaseline { camera: 1 },
        options,
    );
    let mut point_major = PointMajorParallelWorkspace::try_new(
        3,
        3,
        workspace.core.signature.camera_count(),
        workspace.core.layout.dimension(),
        model.offsets(),
    )
    .expect("point-major worker workspace is valid");
    let input: Vec<_> = (0..workspace.core.layout.dimension())
        .map(|index| (0.017 * index as f64).sin())
        .collect();
    let mut fused = vec![0.0; workspace.core.layout.dimension()];
    let mut parallel = vec![0.0; workspace.core.layout.dimension()];
    apply_reduced_system(
        model,
        &workspace.core,
        &retained.camera_blocks,
        &retained.point_hessians,
        &input,
        &mut fused,
    );
    point_major.apply(
        model,
        &workspace.core,
        Some(&retained.camera_blocks),
        &retained.point_hessians,
        &input,
        &mut parallel,
        -1.0,
    );
    let scale = fused.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let error = fused
        .iter()
        .zip(&parallel)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    assert!(error <= 128.0 * f64::EPSILON * scale.max(1.0));
}

#[test]
fn schur_jacobi_split_reconstructs_the_retained_operator() {
    let fixture = fixture(12, 192, 7);
    let (model, workspace, mut retained) = retained_fixture(&fixture, 1e-4);
    assert!(retained.factor_camera_blocks(SchurOperatorContext::retained(model), &workspace.core));
    let input: Vec<_> = (0..workspace.core.layout.dimension())
        .map(|index| (0.031 * index as f64).cos())
        .collect();
    let mut exact = vec![0.0; workspace.core.layout.dimension()];
    let mut preconditioner = vec![0.0; workspace.core.layout.dimension()];
    let mut remainder = vec![0.0; workspace.core.layout.dimension()];

    apply_reduced_system(
        model,
        &workspace.core,
        &retained.camera_blocks,
        &retained.point_hessians,
        &input,
        &mut exact,
    );
    accumulate_camera_blocks(
        &retained.camera_blocks,
        &workspace.core,
        &input,
        &mut preconditioner,
        1.0,
    );
    accumulate_camera_blocks(
        &retained.diagonal_point_coupling,
        &workspace.core,
        &input,
        &mut preconditioner,
        -1.0,
    );
    apply_point_coupling(
        model,
        &workspace.core,
        &retained.point_hessians,
        &input,
        &mut remainder,
    );
    accumulate_camera_blocks(
        &retained.diagonal_point_coupling,
        &workspace.core,
        &input,
        &mut remainder,
        -1.0,
    );
    for (value, remainder) in preconditioner.iter_mut().zip(remainder) {
        *value -= remainder;
    }

    let scale = exact.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let error = exact
        .iter()
        .zip(&preconditioner)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    assert!(
        error <= 128.0 * f64::EPSILON * scale.max(1.0),
        "Schur-Jacobi split error {error:e} at scale {scale:e}"
    );
}

#[test]
fn damped_quadratic_prediction_includes_point_elimination_constant() {
    let fixture = fixture(12, 192, 7);
    let (model, mut workspace, mut retained) = retained_fixture(&fixture, 1e-2);
    workspace
        .ensure_direct_storage()
        .expect("direct oracle storage is available");
    assert!(retained.materialize_schur(model, &mut workspace.core));
    assert!(super::super::factor_camera_system(&mut workspace.core));
    assert!(super::super::solve_factored_camera_system(
        &mut workspace.core
    ));

    let point_reduction = workspace
        .core
        .point_cache_valid
        .iter()
        .zip(&workspace.core.point_inverse_gradients)
        .zip(&retained.point_hessians)
        .filter(|((valid, _), _)| **valid)
        .map(|((_, inverse_gradient), hessian)| {
            0.5 * hessian.bilinear(inverse_gradient, inverse_gradient)
        })
        .sum::<f64>();
    let expected = point_reduction - 0.5 * dot(&workspace.core.gradient, &workspace.core.step);
    let step = workspace.core.step.clone();
    let predicted = retained
        .predicted_reduction(model, &workspace.core, &step)
        .expect("the damped model predicts a positive reduction");
    let relative_error = (predicted - expected).abs() / expected.abs().max(f64::MIN_POSITIVE);
    assert!(
        relative_error <= 2e-12,
        "predicted={predicted:e} expected={expected:e} error={relative_error:e}"
    );

    let clipped: Vec<_> = step.iter().map(|value| 0.37 * value).collect();
    let retained = retained
        .predicted_reduction(model, &workspace.core, &clipped)
        .expect("the clipped step also predicts a reduction");
    let quadratic = super::super::direct::hessian_quadratic_form(&workspace.core, &clipped);
    let direct = crate::kernel::multi_view::optimizer::reduced_model_reduction(
        point_reduction,
        &workspace.core.gradient,
        &clipped,
        quadratic,
    );
    assert!((retained - direct).abs() <= 2e-12 * retained.abs().max(1.0));
}

#[cfg(feature = "parallel")]
#[test]
fn column_owned_fallback_materialization_matches_serial_for_f64_and_f32() {
    let fixture = fixture(24, 768, 9);
    for precision in [RetainedCrossPrecision::F64, RetainedCrossPrecision::F32] {
        let parallel_options = IterativeSchurOptions::new()
            .with_matvec(RetainedSchurMatvec::PointMajorParallel { threads: 4 })
            .with_cross_precision(precision);
        let (model, mut parallel_workspace, mut parallel_retained) = retained_fixture_with_options(
            &fixture,
            1e-2,
            ScaleGauge::FixedBaseline { camera: 1 },
            parallel_options,
        );
        parallel_workspace
            .ensure_direct_storage()
            .expect("parallel direct oracle storage is available");
        parallel_retained
            .ensure_fallback_materializer(&parallel_workspace.core)
            .expect("column-owned fallback storage is reservable");
        let reserved_bytes = parallel_retained.retained_bytes();
        assert!(
            parallel_retained
                .materialize_schur_serial_for_test(model, &mut parallel_workspace.core)
        );
        let expected = parallel_workspace.core.values.clone();

        assert!(parallel_retained.materialize_schur(model, &mut parallel_workspace.core));
        assert_materialization_close(&parallel_workspace.core.values, &expected, precision);
        let first_parallel = parallel_workspace.core.values.clone();

        parallel_workspace.core.values.fill(f64::NAN);
        assert!(parallel_retained.materialize_schur(model, &mut parallel_workspace.core));
        assert_materialization_close(&parallel_workspace.core.values, &expected, precision);
        assert_eq!(parallel_workspace.core.values, first_parallel);
        assert_eq!(parallel_retained.retained_bytes(), reserved_bytes);
    }
}

#[cfg(feature = "parallel")]
fn assert_materialization_close(
    actual: &[f64],
    expected: &[f64],
    precision: RetainedCrossPrecision,
) {
    let scale = expected.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    let tolerance = match precision {
        RetainedCrossPrecision::F64 => 256.0 * f64::EPSILON * scale.max(1.0),
        RetainedCrossPrecision::F32 => 2.0e-6 * scale.max(1.0),
    };
    assert!(
        error <= tolerance,
        "column-owned materialization error {error:e} exceeds {tolerance:e} at scale {scale:e}"
    );
}

#[cfg(feature = "parallel")]
#[test]
fn retained_direct_relinearization_matches_serial_for_f64_and_f32() {
    let fixture = fixture(24, 768, 9);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let model = ObservationModel::from(measurements);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let scene_scale = super::super::optimizer::scene_scale(&fixture.poses, &fixture.points, gauge);
    let gauge_state = GaugeState::new(gauge, &fixture.poses, scene_scale);

    for precision in [RetainedCrossPrecision::F64, RetainedCrossPrecision::F32] {
        let mut serial = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
            fixture.poses.len(),
            measurements,
            gauge,
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::Fused)
                .with_cross_precision(precision),
        )
        .expect("serial retained workspace is valid");
        serial
            .try_reserve_direct_storage()
            .expect("serial direct fallback storage is reservable");
        let serial_assembly = super::super::assemble_sparse(
            model,
            &fixture.poses,
            &fixture.points,
            gauge_state,
            Loss::Cauchy { scale: 2.0 },
            NoiseModel::Unit,
            1e-3,
            false,
            super::super::FixedVariableBlocks::NONE,
            super::super::BundleAdjustmentPriors::NONE,
            &mut serial,
        );
        assert!(serial_assembly.valid);

        let mut parallel = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
            fixture.poses.len(),
            measurements,
            gauge,
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::PointMajorParallel { threads: 4 })
                .with_cross_precision(precision),
        )
        .expect("parallel retained workspace is valid");
        parallel
            .try_reserve_direct_storage()
            .expect("parallel direct fallback storage is reservable");
        let parallel_assembly = super::super::assemble_sparse(
            model,
            &fixture.poses,
            &fixture.points,
            gauge_state,
            Loss::Cauchy { scale: 2.0 },
            NoiseModel::Unit,
            1e-3,
            false,
            super::super::FixedVariableBlocks::NONE,
            super::super::BundleAdjustmentPriors::NONE,
            &mut parallel,
        );
        assert!(parallel_assembly.valid);

        assert_materialization_close(&parallel.core.values, &serial.core.values, precision);
        assert_materialization_close(&parallel.core.gradient, &serial.core.gradient, precision);
        assert_materialization_close(
            &parallel.core.full_gradient,
            &serial.core.full_gradient,
            precision,
        );
        assert_materialization_close(&parallel.core.diagonal, &serial.core.diagonal, precision);
        let tolerance = match precision {
            RetainedCrossPrecision::F64 => 512.0 * f64::EPSILON,
            RetainedCrossPrecision::F32 => 2.0e-6,
        };
        for (actual, expected) in [
            (parallel_assembly.cost, serial_assembly.cost),
            (
                parallel_assembly.squared_residual,
                serial_assembly.squared_residual,
            ),
            (
                parallel_assembly.point_model_reduction,
                serial_assembly.point_model_reduction,
            ),
        ] {
            assert!((actual - expected).abs() <= tolerance * expected.abs().max(1.0));
        }
        assert_eq!(
            parallel_assembly.weak_point_count,
            serial_assembly.weak_point_count
        );
    }
}

#[test]
fn retained_pcg_meets_the_true_residual_guard() {
    let fixture = fixture(32, 512, 8);
    for damping in [1e-1, 1e-4] {
        let (model, workspace, mut retained) = retained_fixture(&fixture, damping);
        let initial_norm = norm(&workspace.core.gradient);
        let attempt = retained.solve(model, &workspace.core, None);
        let SchurAttempt::Accepted(evidence) = attempt else {
            panic!("retained PCG rejected a well-posed fixture at damping {damping:e}");
        };
        let oracle = retained.true_residual_ratio(
            SchurOperatorContext::retained(model),
            &workspace.core,
            initial_norm,
        );
        assert!(evidence.terms <= IterativeSchurOptions::DEFAULT_MAXIMUM_ITERATIONS);
        assert!(oracle <= IterativeSchurOptions::DEFAULT_RELATIVE_TOLERANCE);
        assert!((oracle - evidence.relative_residual).abs() <= 1e-12 * oracle.max(1.0));
    }
}

#[test]
fn auto_policy_uses_the_measured_observation_gate() {
    assert_eq!(auto_parallel_threads(32_767, 1), None);
    #[cfg(feature = "parallel")]
    {
        let expected = auto_available_threads().min(4);
        assert_eq!(auto_parallel_threads(32_768, 1), Some(expected.max(1)));
        let expected_cap = auto_available_threads().min(AUTO_MAXIMUM_THREADS);
        assert_eq!(
            auto_parallel_threads(usize::MAX, 1),
            Some(expected_cap.max(1))
        );
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread test pool is constructible");
        let three = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .expect("three-thread test pool is constructible");
        assert_eq!(
            one.install(|| auto_parallel_threads(usize::MAX, 1)),
            Some(1)
        );
        assert_eq!(
            three.install(|| auto_parallel_threads(usize::MAX, 1)),
            Some(3)
        );
        assert_eq!(
            three.install(|| {
                auto_parallel_threads(usize::MAX, AUTO_MAXIMUM_PARALLEL_AUXILIARY_BYTES / 2 + 1)
            }),
            None
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        assert_eq!(auto_parallel_threads(32_768, 1), None);
        assert_eq!(auto_parallel_threads(usize::MAX, 1), None);
    }

    let fixture = fixture(64, 4_096, 8);
    let (model, workspace, _retained) = retained_fixture(&fixture, 1e-4);
    let resolved = SchurMatvecWorkspace::try_new(
        RetainedSchurMatvec::Auto,
        None,
        model,
        workspace.core.signature.camera_count(),
        workspace.core.layout.dimension(),
    )
    .expect("automatic retained operator is constructible");
    let worker_bytes = calibrated_point_major_worker_bytes(
        workspace.core.signature.camera_count(),
        workspace.core.layout.dimension(),
    );
    if auto_parallel_threads(model.observation_count(), worker_bytes).is_some() {
        assert!(matches!(
            resolved,
            SchurMatvecWorkspace::PointMajorParallel(_)
        ));
        assert!(matches!(
            automatic_recomputed_matvec(
                workspace.core.signature.camera_count(),
                workspace.core.layout.dimension(),
                model.observation_count(),
            )
            .0,
            RetainedSchurMatvec::RecomputedParallel { .. }
        ));
    } else {
        assert!(matches!(resolved, SchurMatvecWorkspace::Fused));
        assert_eq!(
            automatic_recomputed_matvec(
                workspace.core.signature.camera_count(),
                workspace.core.layout.dimension(),
                model.observation_count(),
            ),
            (RetainedSchurMatvec::Recomputed, None)
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn calibrated_auto_matvec_is_bitwise_stable_across_executor_widths() {
    let fixture = fixture(64, 4_096, 8);
    let (model, workspace, retained) = retained_fixture(&fixture, 1e-4);
    let dimension = workspace.core.layout.dimension();
    let input: Vec<_> = (0..dimension)
        .map(|index| (0.013 * index as f64).sin())
        .collect();
    let worker_bytes =
        calibrated_point_major_worker_bytes(workspace.core.signature.camera_count(), dimension);
    let logical_shards = deterministic_point_major_shards(model.observation_count(), worker_bytes)
        .expect("eligible fixture has a topology-defined shard count");

    let apply = |threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("test pool is constructible");
        pool.install(|| {
            let mut operator = SchurMatvecWorkspace::try_new(
                RetainedSchurMatvec::Auto,
                None,
                model,
                workspace.core.signature.camera_count(),
                dimension,
            )
            .expect("automatic calibrated operator is constructible");
            assert_eq!(
                operator.parallel_threads(),
                Some(threads.min(logical_shards))
            );
            let SchurMatvecWorkspace::PointMajorParallel(point_major) = &operator else {
                panic!("eligible calibrated Auto must use point-major")
            };
            assert_eq!(point_major.point_ranges.len() - 1, logical_shards);

            let mut output = vec![0.0; dimension];
            operator.apply_reduced_system(
                SchurOperatorContext::retained(model),
                &workspace.core,
                &retained.camera_blocks,
                &retained.point_hessians,
                &input,
                &mut output,
            );
            output.into_iter().map(f64::to_bits).collect::<Vec<_>>()
        })
    };

    let expected = apply(1);
    for threads in [2, 4, 8] {
        assert_eq!(apply(threads), expected, "{threads}-thread output drifted");
    }
}

#[cfg(feature = "parallel")]
#[test]
fn calibrated_auto_solve_is_bitwise_stable_across_executor_widths() {
    let fixture = fixture(16, 4_096, 8);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };

    let solve = |threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("test pool is constructible");
        pool.install(|| {
            let mut poses = fixture.poses.clone();
            let mut points = fixture.points.clone();
            let iterative = IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::Auto)
                .with_direct_fallback(false);
            let mut workspace = SparseBundleAdjustmentWorkspace::try_new_with_options(
                poses.len(),
                measurements,
                gauge,
                SparseWorkspaceOptions::iterative_schur(iterative),
            )
            .expect("automatic calibrated workspace is constructible");
            let mut report = refine_ba_multi_view_sparse(
                &mut poses,
                &mut points,
                measurements,
                gauge,
                &mut workspace,
                Options::l2(3),
                NoiseModel::Unit,
            )
            .expect("calibrated fixture solves");
            assert!(report.execution_decision.retained_threads().is_some());
            // Executor width and its working-set cost are expected receipt
            // differences, not numerical results. Normalize only those fields
            // before comparing every remaining report value exactly.
            report.execution_decision.retained_threads = None;
            report.solver_working_set_bytes = 0;
            (poses, points, report)
        })
    };

    let expected = solve(1);
    for threads in [2, 4, 8] {
        let actual = solve(threads);
        assert_eq!(actual.0, expected.0, "{threads}-thread poses drifted");
        assert_eq!(actual.1, expected.1, "{threads}-thread points drifted");
        assert_eq!(actual.2, expected.2, "{threads}-thread report drifted");
    }
}

#[test]
fn pcg_iteration_limit_uses_dimension_then_finite_guard() {
    assert_eq!(pcg_iteration_limit(89, 200), 89);
    assert_eq!(pcg_iteration_limit(257, 200), 200);
    assert_eq!(pcg_iteration_limit(usize::MAX, 400), 400);
}
