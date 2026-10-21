// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Tests for deterministic direct-Schur indexing and assembly.
use crate::numerics::sparse::SymmetricPattern;

#[cfg(feature = "parallel")]
use crate::kernel::multi_view::{
    BundleAdjustmentPriors, FixedVariableBlocks, MultiViewMeasurements, PixelObservation,
    ScaleGauge,
};
#[cfg(feature = "parallel")]
use crate::kernel::{PinholeCamera, RelativePose};
#[cfg(feature = "parallel")]
use crate::numerics::refine::{Loss, NoiseModel};

#[cfg(feature = "parallel")]
use super::super::super::model::ObservationModel;
#[cfg(feature = "parallel")]
use crate::kernel::multi_view::schur::GaugeState;

use super::*;

#[allow(clippy::type_complexity)]
fn fixture() -> (
    SymmetricPattern,
    SparseScatter,
    Vec<usize>,
    Vec<u32>,
    Vec<usize>,
    Vec<PackedPointBlock>,
    Vec<[f64; 3]>,
) {
    let pattern = SymmetricPattern::new(3, vec![0, 3, 5, 6], vec![0, 1, 2, 1, 2, 2])
        .expect("test pattern is valid");
    let scatter =
        SparseScatter::try_new(&pattern, &[0, 1, 2], &[1, 1, 1]).expect("test scatter is valid");
    let point_offsets = vec![0, 2, 4, 6];
    let cameras = vec![0, 1, 0, 2, 1, 2];
    let cross_offsets = vec![0, 2, 4, 6];
    let hessians = vec![
        PackedPointBlock::from_matrix(&[[2.0, 0.1, 0.0], [0.1, 3.0, 0.2], [0.0, 0.2, 4.0]]),
        PackedPointBlock::from_matrix(&[[1.5, 0.0, 0.1], [0.0, 2.5, 0.0], [0.1, 0.0, 3.5]]),
        PackedPointBlock::from_matrix(&[[2.2, 0.2, 0.0], [0.2, 2.8, 0.1], [0.0, 0.1, 3.2]]),
    ];
    let crosses = vec![
        [0.2, -0.1, 0.3],
        [0.4, 0.5, -0.2],
        [-0.3, 0.2, 0.1],
        [0.6, -0.4, 0.2],
        [0.1, 0.7, -0.5],
        [-0.2, 0.3, 0.8],
    ];
    (
        pattern,
        scatter,
        point_offsets,
        cameras,
        cross_offsets,
        hessians,
        crosses,
    )
}

fn reference(
    scatter: &SparseScatter,
    hessians: &[PackedPointBlock],
    crosses: &[[f64; 3]],
    values: &mut [f64],
) {
    for (point, blocks) in [[(0, 0), (1, 1)], [(0, 2), (2, 3)], [(1, 4), (2, 5)]]
        .iter()
        .enumerate()
    {
        for (row_index, &(row, row_cross)) in blocks.iter().enumerate() {
            for &(column, column_cross) in &blocks[..=row_index] {
                let position = scatter.position(row, column).expect("edge exists");
                values[position] -=
                    hessians[point].bilinear(&crosses[row_cross], &crosses[column_cross]);
            }
        }
    }
}

#[test]
fn compact_index_preserves_point_order_and_skips_empty_cross_ranges() {
    let (pattern, scatter, _, _, _, _, _) = fixture();
    let assembly = DirectSchurAssemblyWorkspace::try_new(
        2,
        &[0, 2, 4, 6, 8],
        &[0, 1, 0, 2, 1, 2, 0, 1],
        &[1, 1, 1],
        &[0, 2, 4, 6, 6],
        pattern.dimension(),
        pattern.value_count(),
        &scatter,
    )
    .expect("direct assembly index is valid");

    let camera_zero =
        &assembly.block_points[assembly.block_point_offsets[0]..assembly.block_point_offsets[1]];
    assert_eq!(
        camera_zero
            .iter()
            .map(|reference| reference.point)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(assembly.point_block_offsets, vec![0, 2, 4, 6, 6]);
}

#[test]
fn materialization_matches_point_major_reference() {
    let (pattern, scatter, point_offsets, cameras, cross_offsets, hessians, crosses) = fixture();
    let mut assembly = DirectSchurAssemblyWorkspace::try_new(
        1,
        &point_offsets,
        &cameras,
        &[1, 1, 1],
        &cross_offsets,
        pattern.dimension(),
        pattern.value_count(),
        &scatter,
    )
    .expect("direct assembly index is valid");
    let base = vec![10.0, 1.0, 2.0, 11.0, 3.0, 12.0];
    let mut expected = base.clone();
    reference(&scatter, &hessians, &crosses, &mut expected);
    let mut actual = base;

    assert!(assembly.materialize(&scatter, &mut actual, &hessians, &crosses));
    assert_eq!(actual, expected);
}

#[test]
fn indexed_parameters_materialize_unique_shared_blocks() {
    let (pattern, scatter, _, _, cross_offsets, hessians, crosses) = fixture();
    // Each point lists unique active scalar parameters. This is the same
    // representation used by grouped radial tracks after a shared
    // intrinsic block has been deduplicated.
    let parameters = [0, 1, 0, 2, 1, 2];
    let mut assembly = DirectSchurAssemblyWorkspace::try_new_indexed(
        1,
        &[0, 2, 4, 6],
        &cross_offsets,
        &parameters,
        &[1, 1, 1],
        &[],
        pattern.dimension(),
        pattern.value_count(),
        &scatter,
    )
    .expect("indexed direct assembly is valid");
    let base = vec![10.0, 1.0, 2.0, 11.0, 3.0, 12.0];
    let mut expected = base.clone();
    reference(&scatter, &hessians, &crosses, &mut expected);
    let mut actual = base;

    assert!(assembly.materialize(&scatter, &mut actual, &hessians, &crosses));
    assert_eq!(actual, expected);
}

#[cfg(feature = "parallel")]
#[test]
fn materialization_is_bitwise_stable_across_thread_counts() {
    let (pattern, scatter, point_offsets, cameras, cross_offsets, hessians, crosses) = fixture();
    let mut one = DirectSchurAssemblyWorkspace::try_new(
        1,
        &point_offsets,
        &cameras,
        &[1, 1, 1],
        &cross_offsets,
        pattern.dimension(),
        pattern.value_count(),
        &scatter,
    )
    .expect("one-thread index is valid");
    let mut one_values = vec![10.0, 1.0, 2.0, 11.0, 3.0, 12.0];
    assert!(one.materialize(&scatter, &mut one_values, &hessians, &crosses));
    for threads in 2..=8 {
        let mut assembly = DirectSchurAssemblyWorkspace::try_new(
            threads,
            &point_offsets,
            &cameras,
            &[1, 1, 1],
            &cross_offsets,
            pattern.dimension(),
            pattern.value_count(),
            &scatter,
        )
        .expect("multi-thread index is valid");
        let mut values = vec![10.0, 1.0, 2.0, 11.0, 3.0, 12.0];
        assert!(assembly.materialize(&scatter, &mut values, &hessians, &crosses));
        assert_eq!(values, one_values, "worker count {threads}");
    }
}

#[cfg(feature = "parallel")]
struct FullPointFixture {
    pattern: SymmetricPattern,
    scatter: SparseScatter,
    poses: Vec<RelativePose>,
    points: Vec<[f64; 3]>,
    observations: Vec<PixelObservation>,
    point_offsets: Vec<usize>,
    observation_cameras: Vec<u32>,
    point_cross_offsets: Vec<usize>,
    cameras: Vec<PinholeCamera>,
}

#[cfg(feature = "parallel")]
fn full_point_fixture(point_count: usize) -> FullPointFixture {
    let dimension = 12;
    let mut col_ptr = Vec::with_capacity(dimension + 1);
    let mut row_idx = Vec::with_capacity(dimension * (dimension + 1) / 2);
    col_ptr.push(0);
    for column in 0..dimension {
        row_idx.extend(column..dimension);
        col_ptr.push(row_idx.len());
    }
    let pattern =
        SymmetricPattern::new(dimension, col_ptr, row_idx).expect("full point pattern is valid");
    let scatter =
        SparseScatter::try_new(&pattern, &[0, 6], &[6, 6]).expect("full point scatter is valid");
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let poses = vec![
        RelativePose::new(identity, [0.0; 3]),
        RelativePose::new(identity, [0.2, 0.0, 0.0]),
    ];
    let cameras = vec![PinholeCamera::new(500.0, 500.0, 0.0, 0.0); 2];
    let mut points = Vec::with_capacity(point_count);
    let mut observations = Vec::with_capacity(2 * point_count);
    let mut point_offsets = Vec::with_capacity(point_count + 1);
    let mut observation_cameras = Vec::with_capacity(2 * point_count);
    let mut point_cross_offsets = Vec::with_capacity(point_count + 1);
    point_offsets.push(0);
    point_cross_offsets.push(0);
    for point in 0..point_count {
        let value = [
            0.01 * (point % 31) as f64 - 0.15,
            0.01 * ((point / 31) % 29) as f64 - 0.14,
            4.0 + 0.001 * (point % 17) as f64,
        ];
        points.push(value);
        for camera in 0..2 {
            let camera_point = [value[0] + poses[camera].translation[0], value[1], value[2]];
            observations.push(PixelObservation::new(
                camera,
                [
                    cameras[camera].fx * camera_point[0] / camera_point[2],
                    cameras[camera].fy * camera_point[1] / camera_point[2],
                ],
            ));
            observation_cameras.push(camera as u32);
        }
        point_offsets.push(observations.len());
        point_cross_offsets.push(12 * (point + 1));
    }
    FullPointFixture {
        pattern,
        scatter,
        poses,
        points,
        observations,
        point_offsets,
        observation_cameras,
        point_cross_offsets,
        cameras,
    }
}

#[cfg(feature = "parallel")]
struct FullPointResult {
    assembly: schur::Assembly,
    valid: Vec<bool>,
    inverse_gradients: Vec<[f64; 3]>,
    damping_diagonals: Vec<[f64; 3]>,
    inverse_crosses: Vec<[f64; 3]>,
    gradient: Vec<f64>,
    full_gradient: Vec<f64>,
    diagonal: Vec<f64>,
    values: Vec<f64>,
}

#[cfg(feature = "parallel")]
fn assert_full_point_result_bitwise_eq(
    expected: &FullPointResult,
    actual: &FullPointResult,
    threads: usize,
) {
    fn floating_bits(result: &FullPointResult) -> impl Iterator<Item = u64> + '_ {
        let assembly = &result.assembly;
        [
            assembly.minimum_point_rcond,
            assembly.scaled_gradient_inf_norm,
            assembly.point_model_reduction,
            assembly.cost,
            assembly.squared_residual,
            assembly.minimum_reduced_diagonal,
            assembly.maximum_reduced_diagonal,
        ]
        .into_iter()
        .chain(result.inverse_gradients.iter().flatten().copied())
        .chain(result.damping_diagonals.iter().flatten().copied())
        .chain(result.inverse_crosses.iter().flatten().copied())
        .chain(result.gradient.iter().copied())
        .chain(result.full_gradient.iter().copied())
        .chain(result.diagonal.iter().copied())
        .chain(result.values.iter().copied())
        .map(f64::to_bits)
    }

    assert_eq!(actual.valid, expected.valid, "worker count {threads}");
    assert_eq!(
        (
            actual.assembly.valid,
            actual.assembly.weak_point_count,
            actual.assembly.damping_floor_clamp_count,
        ),
        (
            expected.assembly.valid,
            expected.assembly.weak_point_count,
            expected.assembly.damping_floor_clamp_count,
        ),
        "worker count {threads}"
    );
    assert!(
        floating_bits(actual).eq(floating_bits(expected)),
        "floating state differs for worker count {threads}"
    );
}

#[cfg(feature = "parallel")]
fn run_full_point_assembly(
    direct: &mut DirectSchurAssemblyWorkspace,
    fixture: &FullPointFixture,
) -> FullPointResult {
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.point_offsets,
        cameras: &fixture.cameras,
    };
    let model = ObservationModel::from(measurements);
    let point_count = fixture.points.len();
    let mut valid = vec![false; point_count];
    let mut inverse_gradients = vec![[0.0; 3]; point_count];
    let mut damping_diagonals = vec![[0.0; 3]; point_count];
    let mut inverse_crosses = vec![[0.0; 3]; 12 * point_count];
    let mut point_hessians = vec![PackedPointBlock::default(); point_count];
    let mut gradient = vec![0.0; 12];
    let mut full_gradient = vec![0.0; 12];
    let mut diagonal = vec![0.0; 12];
    let assembly = direct.assemble_points(
        schur::BaProblemView {
            model,
            poses: &fixture.poses,
            points: &fixture.points,
            fixed: FixedVariableBlocks::NONE,
            priors: BundleAdjustmentPriors::NONE,
        },
        schur::LinearizationPolicy {
            gauge: GaugeState::new(ScaleGauge::Free, &fixture.poses, 1.0),
            loss: Loss::L2,
            noise: NoiseModel::Unit,
            damping: 1e-3,
            prior_sqrt_scale: 1.0,
        },
        VariableLayout::calibrated(&[0, 6], &[6, 6]),
        &fixture.scatter,
        DirectPointCacheMut {
            point_steps: Some(PointStepCacheBuffers {
                valid: &mut valid,
                inverse_gradients: &mut inverse_gradients,
                damping_diagonals: &mut damping_diagonals,
            }),
            cross_offsets: &fixture.point_cross_offsets,
            inverse_crosses: PointCrossCacheMut::F64(&mut inverse_crosses),
            point_hessians: &mut point_hessians,
        },
        DirectReducedAssemblyMut {
            gradient: &mut gradient,
            full_gradient: &mut full_gradient,
            diagonal: &mut diagonal,
        },
    );
    assert!(assembly.valid);
    let mut values = vec![f64::NAN; fixture.pattern.value_count()];
    assert!(direct.seed_base_values(&fixture.scatter, &mut values));
    assert!(direct.materialize(
        &fixture.scatter,
        &mut values,
        &point_hessians,
        &inverse_crosses,
    ));
    FullPointResult {
        assembly,
        valid,
        inverse_gradients,
        damping_diagonals,
        inverse_crosses,
        gradient,
        full_gradient,
        diagonal,
        values,
    }
}

#[cfg(feature = "parallel")]
#[test]
fn full_point_assembly_is_bitwise_stable_across_worker_counts() {
    let fixture = full_point_fixture(4_097);
    let mut one = DirectSchurAssemblyWorkspace::try_new(
        1,
        &fixture.point_offsets,
        &fixture.observation_cameras,
        &[6, 6],
        &fixture.point_cross_offsets,
        fixture.pattern.dimension(),
        fixture.pattern.value_count(),
        &fixture.scatter,
    )
    .expect("one-thread full assembler is valid");
    assert!(one.logical_point_ranges.len() > 2);
    let one = run_full_point_assembly(&mut one, &fixture);
    for threads in 2..=8 {
        let mut assembly = DirectSchurAssemblyWorkspace::try_new(
            threads,
            &fixture.point_offsets,
            &fixture.observation_cameras,
            &[6, 6],
            &fixture.point_cross_offsets,
            fixture.pattern.dimension(),
            fixture.pattern.value_count(),
            &fixture.scatter,
        )
        .expect("multi-thread full assembler is valid");
        let actual = run_full_point_assembly(&mut assembly, &fixture);
        assert_full_point_result_bitwise_eq(&one, &actual, threads);
    }
}
