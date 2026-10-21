// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sparse-workspace topology and scatter regression tests.

use std::collections::BTreeSet;

use crate::numerics::sparse::SymmetricPattern;

use super::pattern::{
    radial_reduced_block_columns, radial_reduced_pattern, reduced_block_columns,
    reduced_block_columns_with_bitmap_limit, reduced_nonzero_count, reduced_pattern,
};
use super::topology::ReducedLayout;
use super::{SparseBundleAdjustmentWorkspace, SparseScatter, SparseSelfCalibrationWorkspace};
use crate::kernel::multi_view::BundleAdjustmentBlock;
use crate::kernel::multi_view::model::ObservationModel;
use crate::kernel::multi_view::self_calibration::SelfCalibratingMeasurements;
use crate::kernel::multi_view::sparse::api::{
    DirectSchurAssembly, SparseBundleAdjustmentError, SparsePointBackSubstitution,
    SparseWorkspaceOptions,
};
use crate::kernel::multi_view::{
    FixedVariableBlocks, MultiViewBundleAdjustmentError, MultiViewMeasurements, ScaleGauge,
};
use crate::kernel::{PinholeCamera, multi_view::PixelObservation};

#[test]
fn calibrated_signature_ignores_coordinates_and_rejects_topology_changes() {
    fn measurements<'a>(
        observations: &'a [PixelObservation],
        offsets: &'a [usize],
        cameras: &'a [PinholeCamera],
    ) -> MultiViewMeasurements<'a> {
        MultiViewMeasurements::Pixel {
            observations,
            point_offsets: offsets,
            cameras,
        }
    }

    let observations: [PixelObservation; 12] = std::array::from_fn(|index| {
        PixelObservation::new(index % 3, [index as f64, -(index as f64)])
    });
    let changed_coordinates =
        observations.map(|observation| PixelObservation::new(observation.camera, [37.0, -19.0]));
    let changed_camera_order = observations
        .map(|observation| PixelObservation::new((observation.camera + 1) % 3, observation.point));
    let point_offsets = [0, 3, 6, 9, 12];
    let changed_offsets = [0, 2, 6, 9, 12];
    let cameras = [PinholeCamera::new(500.0, 500.0, 0.0, 0.0); 3];
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let workspace = SparseBundleAdjustmentWorkspace::try_new(
        3,
        measurements(&observations, &point_offsets, &cameras),
        gauge,
    )
    .unwrap();

    assert!(workspace.matches_with_fixed(
        3,
        measurements(&changed_coordinates, &point_offsets, &cameras),
        gauge,
        FixedVariableBlocks::NONE,
    ));
    for (candidate, candidate_gauge, fixed) in [
        (
            measurements(&changed_coordinates, &changed_offsets, &cameras),
            gauge,
            FixedVariableBlocks::NONE,
        ),
        (
            measurements(&changed_camera_order, &point_offsets, &cameras),
            gauge,
            FixedVariableBlocks::NONE,
        ),
        (
            measurements(&changed_coordinates, &point_offsets, &cameras),
            ScaleGauge::Free,
            FixedVariableBlocks::NONE,
        ),
        (
            measurements(&changed_coordinates, &point_offsets, &cameras),
            gauge,
            FixedVariableBlocks {
                camera_poses: &[false, false, true],
                points: &[],
                intrinsics: &[],
                intrinsic_dimensions: &[],
            },
        ),
        (
            measurements(&changed_coordinates, &point_offsets, &cameras),
            gauge,
            FixedVariableBlocks {
                camera_poses: &[],
                points: &[true, false, false, false],
                intrinsics: &[],
                intrinsic_dimensions: &[],
            },
        ),
    ] {
        assert!(!workspace.matches_with_fixed(3, candidate, candidate_gauge, fixed));
    }
}

#[test]
fn direct_assembly_policy_rejects_invalid_fixed_worker_requests() {
    let observations: [PixelObservation; 12] = std::array::from_fn(|index| {
        PixelObservation::new(index % 3, [index as f64, -(index as f64)])
    });
    let point_offsets = [0, 3, 6, 9, 12];
    let cameras = [PinholeCamera::new(500.0, 500.0, 0.0, 0.0); 3];
    let measurements = MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &point_offsets,
        cameras: &cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };

    let zero = SparseBundleAdjustmentWorkspace::try_new_with_options(
        3,
        measurements,
        gauge,
        SparseWorkspaceOptions::direct_parallel(0),
    )
    .err()
    .expect("zero direct workers must be rejected");
    assert_eq!(
        zero,
        SparseBundleAdjustmentError::DirectSchurThreadsMustBePositive
    );

    let recomputed = SparseBundleAdjustmentWorkspace::try_new_with_options(
        3,
        measurements,
        gauge,
        SparseWorkspaceOptions::direct_with_assembly(
            SparsePointBackSubstitution::Recompute,
            DirectSchurAssembly::Parallel { threads: 2 },
        ),
    )
    .err()
    .expect("parallel direct assembly needs cached calibrated point rows");
    assert_eq!(
        recomputed,
        SparseBundleAdjustmentError::DirectSchurParallelRequiresCachedPoints
    );

    let radial = SparseSelfCalibrationWorkspace::try_new_with_direct_assembly(
        3,
        SelfCalibratingMeasurements::new(&observations, &point_offsets),
        gauge,
        FixedVariableBlocks::NONE,
        DirectSchurAssembly::Parallel { threads: 0 },
    )
    .err()
    .expect("zero radial direct workers must be rejected");
    assert_eq!(
        radial,
        SparseBundleAdjustmentError::DirectSchurThreadsMustBePositive
    );
}

#[test]
fn self_calibration_signature_rejects_group_and_fixed_changes() {
    let observations: [PixelObservation; 12] = std::array::from_fn(|index| {
        PixelObservation::new(index % 3, [index as f64, index as f64 + 0.5])
    });
    let changed_coordinates =
        observations.map(|observation| PixelObservation::new(observation.camera, [-7.0, 11.0]));
    let point_offsets = [0, 3, 6, 9, 12];
    let measurements = SelfCalibratingMeasurements::new(&observations, &point_offsets);
    let changed_measurements =
        SelfCalibratingMeasurements::new(&changed_coordinates, &point_offsets);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let camera_to_group = [0, 0, 1];
    let workspace = SparseSelfCalibrationWorkspace::try_new_grouped(
        3,
        2,
        &camera_to_group,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
    )
    .unwrap();

    assert!(workspace.matches(
        3,
        2,
        Some(&camera_to_group),
        changed_measurements,
        gauge,
        FixedVariableBlocks::NONE,
    ));
    assert!(!workspace.matches(
        3,
        2,
        Some(&[0, 1, 1]),
        changed_measurements,
        gauge,
        FixedVariableBlocks::NONE,
    ));
    assert!(!workspace.matches(
        3,
        2,
        Some(&camera_to_group),
        changed_measurements,
        gauge,
        FixedVariableBlocks {
            camera_poses: &[],
            points: &[],
            intrinsics: &[false, true],
            intrinsic_dimensions: &[],
        },
    ));
}

#[test]
fn self_calibration_workspace_reports_a_malformed_intrinsic_mask() {
    let observations: [PixelObservation; 12] = std::array::from_fn(|index| {
        PixelObservation::new(index % 3, [index as f64, index as f64 + 0.5])
    });
    let point_offsets = [0, 3, 6, 9, 12];
    let measurements = SelfCalibratingMeasurements::new(&observations, &point_offsets);
    let error = SparseSelfCalibrationWorkspace::try_new_grouped(
        3,
        2,
        &[0, 0, 1],
        measurements,
        ScaleGauge::FixedBaseline { camera: 1 },
        FixedVariableBlocks {
            camera_poses: &[],
            points: &[],
            intrinsics: &[true],
            intrinsic_dimensions: &[],
        },
    )
    .err()
    .expect("a short intrinsic mask must be rejected without panicking");

    assert_eq!(
        error,
        SparseBundleAdjustmentError::Problem(
            MultiViewBundleAdjustmentError::FixedBlockCountMismatch {
                block: BundleAdjustmentBlock::RadialIntrinsics,
                expected: 2,
                actual: 1,
            }
        )
    );
}

#[test]
fn block_reduced_pattern_matches_scalar_reference() {
    let observations = [
        PixelObservation::new(0, [0.0; 2]),
        PixelObservation::new(1, [0.0; 2]),
        PixelObservation::new(2, [0.0; 2]),
        PixelObservation::new(1, [0.0; 2]),
        PixelObservation::new(3, [0.0; 2]),
        PixelObservation::new(2, [0.0; 2]),
        PixelObservation::new(2, [0.0; 2]),
        PixelObservation::new(3, [0.0; 2]),
    ];
    let point_offsets = [0, 3, 5, 8];
    let cameras = [PinholeCamera::new(1.0, 1.0, 0.0, 0.0); 4];
    let model = ObservationModel::from(MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &point_offsets,
        cameras: &cameras,
    });
    let offsets = [0, 0, 5, 11];
    let dimensions = [0, 5, 6, 6];
    let dimension = 17;
    let observation_cameras: Vec<_> = observations
        .iter()
        .map(|observation| observation.camera as u32)
        .collect();
    let block_columns =
        reduced_block_columns(&point_offsets, &observation_cameras, &dimensions, &[]);
    let actual = reduced_pattern(&block_columns, dimension, &offsets, &dimensions);
    assert_eq!(
        reduced_nonzero_count(&block_columns, &dimensions),
        actual.1.len()
    );

    let mut columns: Vec<BTreeSet<usize>> = (0..dimension)
        .map(|column| BTreeSet::from([column]))
        .collect();
    let mut active = Vec::new();
    for point in 0..3 {
        active.clear();
        for observation in model.offsets()[point]..model.offsets()[point + 1] {
            let camera = model.camera(observation);
            active.extend(offsets[camera]..offsets[camera] + dimensions[camera]);
        }
        active.sort_unstable();
        active.dedup();
        for (right_index, &row) in active.iter().enumerate() {
            for &column in &active[..=right_index] {
                columns[column].insert(row);
            }
        }
    }
    let mut expected = (vec![0], Vec::new());
    for column in columns {
        expected.1.extend(column);
        expected.0.push(expected.1.len());
    }

    assert_eq!(actual, expected);
}

#[test]
fn calibrated_visibility_bitmap_and_tree_fallback_match() {
    let point_offsets = [0, 4, 7, 9];
    let observation_cameras = [0, 1, 2, 3, 1, 3, 4, 0, 4];
    let dimensions = [0, 6, 6, 5, 6];
    let bitmap = reduced_block_columns_with_bitmap_limit(
        &point_offsets,
        &observation_cameras,
        &dimensions,
        &[],
        usize::MAX,
    );
    let tree = reduced_block_columns_with_bitmap_limit(
        &point_offsets,
        &observation_cameras,
        &dimensions,
        &[],
        0,
    );
    assert_eq!(bitmap, tree);
}

#[test]
fn block_scatter_resolves_every_calibrated_and_grouped_scalar_entry() {
    let observations = [
        PixelObservation::new(0, [0.0; 2]),
        PixelObservation::new(1, [0.0; 2]),
        PixelObservation::new(2, [0.0; 2]),
        PixelObservation::new(3, [0.0; 2]),
        PixelObservation::new(1, [0.0; 2]),
        PixelObservation::new(3, [0.0; 2]),
    ];
    let point_offsets = [0, 4, 6];
    let calibrated_layout = ReducedLayout::try_calibrated(
        4,
        ScaleGauge::FixedBaseline { camera: 1 },
        crate::kernel::multi_view::CameraPoseVariables::RotationAndTranslation,
        &[],
    )
    .unwrap();
    let offsets = calibrated_layout.camera_offsets();
    let dimensions = calibrated_layout.camera_dimensions();
    let dimension = calibrated_layout.dimension();
    let observation_cameras: Vec<_> = observations
        .iter()
        .map(|observation| observation.camera as u32)
        .collect();
    let block_columns =
        reduced_block_columns(&point_offsets, &observation_cameras, dimensions, &[]);
    let (col_ptr, row_idx) = reduced_pattern(&block_columns, dimension, offsets, dimensions);
    let pattern = SymmetricPattern::new(dimension, col_ptr, row_idx).unwrap();
    let scatter = SparseScatter::try_new(&pattern, offsets, dimensions).unwrap();
    assert_scatter_matches_pattern(&pattern, &scatter);

    let camera_to_group = [0, 0, 1, 1];
    let layout = ReducedLayout::try_radial(
        4,
        2,
        ScaleGauge::FixedBaseline { camera: 1 },
        FixedVariableBlocks::NONE,
    )
    .unwrap();
    let radial_columns = radial_reduced_block_columns(
        &point_offsets,
        &observation_cameras,
        Some(&camera_to_group),
        layout.camera_dimensions(),
        layout.intrinsic_dimensions(),
        &[],
    );
    let mut block_offsets = layout.camera_offsets().to_vec();
    block_offsets.extend_from_slice(layout.intrinsic_offsets());
    let mut block_dimensions = layout.camera_dimensions().to_vec();
    block_dimensions.extend_from_slice(layout.intrinsic_dimensions());
    let (col_ptr, row_idx) = radial_reduced_pattern(
        &radial_columns,
        layout.dimension(),
        &block_offsets,
        &block_dimensions,
    );
    let pattern = SymmetricPattern::new(layout.dimension(), col_ptr, row_idx).unwrap();
    let scatter = SparseScatter::try_new(&pattern, &block_offsets, &block_dimensions).unwrap();
    assert_scatter_matches_pattern(&pattern, &scatter);
}

fn assert_scatter_matches_pattern(pattern: &SymmetricPattern, scatter: &SparseScatter) {
    let column_pointers = pattern.column_pointers();
    for column in 0..pattern.dimension() {
        let range = column_pointers[column]..column_pointers[column + 1];
        for (local, &row) in pattern.row_indices()[range.clone()].iter().enumerate() {
            assert_eq!(scatter.position(row, column), Some(range.start + local));
        }
    }
}
