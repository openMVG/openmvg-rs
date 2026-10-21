// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Selected sparse marginal covariance evidence.

use super::super::*;
use super::fixture::fixture;
use crate::kernel::multi_view::PosePrior;
use crate::kernel::multi_view::dense::{
    DenseBundleAdjustmentWorkspace, covariance_ba_multi_view_dense,
    covariance_ba_multi_view_dense_with_factors,
};
use crate::numerics::refine::UncertaintyModel;

#[test]
fn selected_sparse_camera_and_point_marginals_match_dense_covariance() {
    let fixture = fixture(4, 24, 4);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let loss = Loss::Cauchy { scale: 2.0 };
    let mut dense_workspace =
        DenseBundleAdjustmentWorkspace::try_new(fixture.poses.len(), fixture.points.len())
            .expect("dense fixture capacity is valid");
    let dense = covariance_ba_multi_view_dense(
        &fixture.poses,
        &fixture.points,
        measurements,
        gauge,
        &mut dense_workspace,
        loss,
        NoiseModel::Unit,
    )
    .expect("fixture has full-rank covariance");
    let mut sparse_workspace =
        SparseBundleAdjustmentWorkspace::try_new(fixture.poses.len(), measurements, gauge)
            .expect("sparse fixture topology is valid");

    let selected = selected_marginal_covariances_ba_multi_view_sparse(
        &fixture.poses,
        &fixture.points,
        measurements,
        gauge,
        &[3, 2],
        &[7, 0, 13],
        &mut sparse_workspace,
        loss,
        NoiseModel::Unit,
    )
    .expect("selected sparse covariance is full rank");

    assert_eq!(
        selected
            .cameras
            .iter()
            .map(|entry| entry.camera)
            .collect::<Vec<_>>(),
        vec![3, 2]
    );
    assert_eq!(
        selected
            .points
            .iter()
            .map(|entry| entry.point)
            .collect::<Vec<_>>(),
        vec![7, 0, 13]
    );
    assert_eq!(
        selected.uncertainty_model,
        UncertaintyModel::RobustFixedWeightLinearized
    );
    assert!(selected.observation_noise_whitened);
    assert_close(
        selected.normalization_scale(),
        dense.normalization_scale(),
        2e-14,
    );
    for marginal in &selected.cameras {
        let range = dense
            .camera_parameter_range(marginal.camera)
            .expect("selected camera exists");
        assert_eq!(range.len(), 6);
        for row in 0..6 {
            for column in 0..6 {
                let expected = dense.camera_covariance
                    [(range.start + row) * dense.camera_dimension + range.start + column];
                assert_close(marginal.matrix[row][column], expected, 1e-7);
            }
        }
    }
    for marginal in &selected.points {
        for row in 0..3 {
            for column in 0..3 {
                assert_close(
                    marginal.matrix[row][column],
                    dense.point_covariances[marginal.point][row][column],
                    1e-7,
                );
            }
        }
    }
}

#[test]
fn direct_and_iterative_workspaces_return_the_same_selected_marginals() {
    let fixture = fixture(5, 32, 4);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let mut direct =
        SparseBundleAdjustmentWorkspace::try_new(fixture.poses.len(), measurements, gauge)
            .expect("direct topology is valid");
    let mut iterative = SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
        fixture.poses.len(),
        measurements,
        gauge,
        IterativeSchurOptions::new().with_exact_camera_rank_diagnostics(false),
    )
    .expect("iterative topology is valid");
    assert_eq!(iterative.direct_numeric_bytes(), 0);
    let loss = Loss::L2;

    let direct_result = selected_marginal_covariances_ba_multi_view_sparse(
        &fixture.poses,
        &fixture.points,
        measurements,
        gauge,
        &[4, 2],
        &[1, 17],
        &mut direct,
        loss,
        NoiseModel::isotropic(1.7),
    )
    .expect("direct selected covariance is valid");
    let iterative_result = selected_marginal_covariances_ba_multi_view_sparse(
        &fixture.poses,
        &fixture.points,
        measurements,
        gauge,
        &[4, 2],
        &[1, 17],
        &mut iterative,
        loss,
        NoiseModel::isotropic(1.7),
    )
    .expect("iterative workspace uses the exact covariance query");

    assert!(iterative.direct_numeric_bytes() > 0);
    assert_eq!(
        direct_result.uncertainty_model,
        UncertaintyModel::ResidualLinearized
    );
    assert_eq!(direct_result, iterative_result);
}

#[test]
fn pose_prior_anchors_free_scale_and_matches_dense_covariance() {
    let fixture = fixture(4, 24, 4);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let sqrt_information = std::array::from_fn(|row| {
        std::array::from_fn(|column| if row == column { 1.0 } else { 0.0 })
    });
    let pose_priors = [PosePrior {
        camera: 1,
        mean: fixture.poses[1],
        sqrt_information,
    }];
    let priors = BundleAdjustmentPriors {
        poses: &pose_priors,
        points: &[],
        intrinsics: &[],
    };
    let mut dense_workspace =
        DenseBundleAdjustmentWorkspace::try_new(fixture.poses.len(), fixture.points.len())
            .expect("dense fixture capacity is valid");
    let dense = covariance_ba_multi_view_dense_with_factors(
        &fixture.poses,
        &fixture.points,
        measurements,
        ScaleGauge::Free,
        FixedVariableBlocks::NONE,
        priors,
        &mut dense_workspace,
        Loss::L2,
        NoiseModel::Unit,
    )
    .expect("the pose prior anchors free scale");
    let mut sparse_workspace = SparseBundleAdjustmentWorkspace::try_new(
        fixture.poses.len(),
        measurements,
        ScaleGauge::Free,
    )
    .expect("free-gauge sparse topology is valid");
    let selected = selected_marginal_covariances_ba_multi_view_sparse_with_factors(
        &fixture.poses,
        &fixture.points,
        measurements,
        ScaleGauge::Free,
        FixedVariableBlocks::NONE,
        priors,
        &[2],
        &[4],
        &mut sparse_workspace,
        Loss::L2,
        NoiseModel::Unit,
    )
    .expect("the exact sparse information recognizes the prior anchor");

    let range = dense.camera_parameter_range(2).unwrap();
    assert_eq!(range.len(), 6);
    for row in 0..6 {
        for column in 0..6 {
            let expected = dense.camera_covariance
                [(range.start + row) * dense.camera_dimension + range.start + column];
            assert_close(selected.cameras[0].matrix[row][column], expected, 1e-7);
        }
    }
    for row in 0..3 {
        for column in 0..3 {
            assert_close(
                selected.points[0].matrix[row][column],
                dense.point_covariances[4][row][column],
                1e-7,
            );
        }
    }
}

#[test]
fn selected_covariance_rejects_invalid_duplicate_fixed_and_gauge_blocks() {
    let fixture = fixture(4, 24, 4);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let mut workspace =
        SparseBundleAdjustmentWorkspace::try_new(fixture.poses.len(), measurements, gauge)
            .expect("fixture topology is valid");
    let mut query = |cameras: &[usize], points: &[usize]| {
        selected_marginal_covariances_ba_multi_view_sparse(
            &fixture.poses,
            &fixture.points,
            measurements,
            gauge,
            cameras,
            points,
            &mut workspace,
            Loss::L2,
            NoiseModel::Unit,
        )
    };

    assert!(matches!(
        query(&[2, 2], &[]),
        Err(SparseBundleAdjustmentCovarianceError::DuplicateSelectedCamera { camera: 2 })
    ));
    assert!(matches!(
        query(&[fixture.poses.len()], &[]),
        Err(
            SparseBundleAdjustmentCovarianceError::SelectedCameraOutOfBounds {
                selection: 0,
                camera: 4,
                camera_count: 4,
            }
        )
    ));
    assert!(matches!(
        query(&[0], &[]),
        Err(SparseBundleAdjustmentCovarianceError::SelectedCameraFixed { camera: 0 })
    ));
    assert!(matches!(
        query(&[1], &[]),
        Err(SparseBundleAdjustmentCovarianceError::SelectedCameraGaugeConstrained { camera: 1 })
    ));
    assert!(matches!(
        query(&[], &[3, 3]),
        Err(SparseBundleAdjustmentCovarianceError::DuplicateSelectedPoint { point: 3 })
    ));
    assert!(matches!(
        query(&[], &[fixture.points.len()]),
        Err(
            SparseBundleAdjustmentCovarianceError::SelectedPointOutOfBounds {
                selection: 0,
                point: 24,
                point_count: 24,
            }
        )
    ));

    let fixed_poses = [false, false, true, false];
    let mut fixed_points = vec![false; fixture.points.len()];
    fixed_points[3] = true;
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_poses,
        points: &fixed_points,
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    let mut fixed_workspace = SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks(
        fixture.poses.len(),
        measurements,
        gauge,
        fixed,
        SparseWorkspaceOptions::direct(),
    )
    .expect("fixed topology is valid");
    assert!(matches!(
        selected_marginal_covariances_ba_multi_view_sparse_with_factors(
            &fixture.poses,
            &fixture.points,
            measurements,
            gauge,
            fixed,
            BundleAdjustmentPriors::NONE,
            &[2],
            &[],
            &mut fixed_workspace,
            Loss::L2,
            NoiseModel::Unit,
        ),
        Err(SparseBundleAdjustmentCovarianceError::SelectedCameraFixed { camera: 2 })
    ));
    assert!(matches!(
        selected_marginal_covariances_ba_multi_view_sparse_with_factors(
            &fixture.poses,
            &fixture.points,
            measurements,
            gauge,
            fixed,
            BundleAdjustmentPriors::NONE,
            &[],
            &[3],
            &mut fixed_workspace,
            Loss::L2,
            NoiseModel::Unit,
        ),
        Err(SparseBundleAdjustmentCovarianceError::SelectedPointFixed { point: 3 })
    ));

    let mut free_workspace = SparseBundleAdjustmentWorkspace::try_new(
        fixture.poses.len(),
        measurements,
        ScaleGauge::Free,
    )
    .expect("free-gauge topology is valid");
    assert!(matches!(
        selected_marginal_covariances_ba_multi_view_sparse(
            &fixture.poses,
            &fixture.points,
            measurements,
            ScaleGauge::Free,
            &[2],
            &[3],
            &mut free_workspace,
            Loss::L2,
            NoiseModel::Unit,
        ),
        Err(SparseBundleAdjustmentCovarianceError::FreeScaleGauge)
    ));
}

fn assert_close(actual: f64, expected: f64, relative_tolerance: f64) {
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        (actual - expected).abs() <= relative_tolerance * scale,
        "actual={actual:e}, expected={expected:e}, relative error={:e}",
        (actual - expected).abs() / scale
    );
}
