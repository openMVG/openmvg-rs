// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "faer")]

mod dense_parity;
mod iterative_schur;
mod workspace;

#[cfg(feature = "parallel")]
use openmvg_ba::kernel::multi_view::sparse::DirectSchurAssembly;

use openmvg_ba::kernel::multi_view::sparse::{
    SparseExecutionBackend, SparseExecutionReason, SparseWorkspaceMemoryBudget,
};

use openmvg_ba::kernel::{
    RelativePose, multi_view::BundleAdjustmentPriors, multi_view::FixedVariableBlocks,
    multi_view::PixelObservation, multi_view::PointPrior, multi_view::PosePrior,
    multi_view::RadialIntrinsicPrior, multi_view::ScaleGauge,
    multi_view::self_calibration::RadialCamera,
    multi_view::self_calibration::SelfCalibratingMeasurements,
    multi_view::self_calibration::SelfCalibrationWorkspace,
    multi_view::self_calibration::refine_ba_multi_view_self_calibrating_dense,
    multi_view::self_calibration::refine_ba_multi_view_self_calibrating_dense_with_factors,
    multi_view::self_calibration::refine_ba_multi_view_self_calibrating_grouped_dense_with_factors,
    multi_view::sparse::IterativeSchurOptions, multi_view::sparse::RetainedCrossPrecision,
    multi_view::sparse::RetainedSchurMatvec, multi_view::sparse::SparseBundleAdjustmentError,
    multi_view::sparse::SparseCameraRankAssessment, multi_view::sparse::SparseCameraSolveQuality,
    multi_view::sparse::SparseCameraSolver, multi_view::sparse::SparseSelfCalibrationWorkspace,
    multi_view::sparse::refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors,
    multi_view::sparse::refine_ba_multi_view_self_calibrating_sparse,
};

use openmvg_ba::numerics::manifold;

use openmvg_ba::numerics::refine::{
    ConvergenceCriteria, ConvergenceCriterion, NoiseModel, Options, RefinementStatus,
};

const CAMERA_COUNT: usize = 4;

fn identity_pose() -> RelativePose {
    RelativePose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    )
}

fn scene() -> ([RelativePose; CAMERA_COUNT], Vec<[f64; 3]>) {
    let poses = [
        identity_pose(),
        RelativePose::new(
            manifold::exp_so3(&[0.018, 0.065, -0.012]),
            [0.72, 0.025, 0.08],
        ),
        RelativePose::new(
            manifold::exp_so3(&[-0.042, -0.052, 0.024]),
            [-0.61, 0.075, 0.13],
        ),
        RelativePose::new(
            manifold::exp_so3(&[0.052, -0.024, 0.032]),
            [0.19, -0.51, 0.11],
        ),
    ];
    let points = (0..36)
        .map(|index| {
            let column = (index % 6) as f64;
            let row = (index / 6) as f64;
            [
                -1.3 + 0.5 * column,
                -0.95 + 0.38 * row,
                3.25 + 0.28 * (index % 7) as f64 + 0.05 * row,
            ]
        })
        .collect();
    (poses, points)
}

fn observations(
    poses: &[RelativePose; CAMERA_COUNT],
    points: &[[f64; 3]],
    camera: impl Fn(usize) -> RadialCamera,
) -> (Vec<PixelObservation>, Vec<usize>) {
    let mut observations = Vec::with_capacity(CAMERA_COUNT * points.len());
    let mut offsets = Vec::with_capacity(points.len() + 1);
    offsets.push(0);
    for point in points {
        for (camera_index, pose) in poses.iter().enumerate() {
            observations.push(PixelObservation::new(
                camera_index,
                project(camera(camera_index), transform(*pose, *point)),
            ));
        }
        offsets.push(observations.len());
    }
    (observations, offsets)
}

fn perturbed_scene(
    truth_poses: &[RelativePose; CAMERA_COUNT],
    truth_points: &[[f64; 3]],
) -> ([RelativePose; CAMERA_COUNT], Vec<[f64; 3]>) {
    let mut poses = *truth_poses;
    for (camera, pose) in poses.iter_mut().enumerate().skip(1) {
        pose.rotation = manifold::mat_mul(
            &manifold::exp_so3(&[0.002 * camera as f64, -0.003, 0.0015 * camera as f64]),
            &pose.rotation,
        );
        pose.translation[0] += 0.006 * camera as f64;
        pose.translation[1] -= 0.004;
        pose.translation[2] += 0.003;
    }
    let truth_baseline = norm(&truth_poses[1].translation);
    let initial_baseline = norm(&poses[1].translation);
    poses[1].translation = poses[1]
        .translation
        .map(|value| value * truth_baseline / initial_baseline);
    let points = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let phase = index as f64;
            [
                point[0] + 0.006 * phase.sin(),
                point[1] - 0.005 * (0.7 * phase).cos(),
                point[2] + 0.008 * (0.4 * phase).sin(),
            ]
        })
        .collect();
    (poses, points)
}

fn diagonal<const N: usize>(value: f64) -> [[f64; N]; N] {
    std::array::from_fn(|row| std::array::from_fn(|column| if row == column { value } else { 0.0 }))
}

fn transform(pose: RelativePose, point: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        (0..3)
            .map(|column| pose.rotation[row][column] * point[column])
            .sum::<f64>()
            + pose.translation[row]
    })
}

fn project(camera: RadialCamera, point: [f64; 3]) -> [f64; 2] {
    let normalized = [point[0] / point[2], point[1] / point[2]];
    let radius_squared = normalized[0] * normalized[0] + normalized[1] * normalized[1];
    let distortion = 1.0 + camera.k1 * radius_squared + camera.k2 * radius_squared * radius_squared;
    normalized.map(|value| camera.focal * distortion * value)
}

fn norm(vector: &[f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn assert_objective_close(dense: f64, sparse: f64) {
    let tolerance = 5e-8 * (1.0 + dense.abs().max(sparse.abs()));
    assert!(
        (dense - sparse).abs() <= tolerance,
        "dense cost {dense:.16e}, sparse cost {sparse:.16e}, tolerance {tolerance:.3e}"
    );
}

fn assert_state_close(
    dense_poses: &[RelativePose],
    dense_points: &[[f64; 3]],
    dense_intrinsics: &[RadialCamera],
    sparse_poses: &[RelativePose],
    sparse_points: &[[f64; 3]],
    sparse_intrinsics: &[RadialCamera],
) {
    for (dense, sparse) in dense_poses.iter().zip(sparse_poses) {
        for (dense, sparse) in dense
            .rotation
            .iter()
            .flatten()
            .zip(sparse.rotation.iter().flatten())
        {
            assert!(
                (dense - sparse).abs() < 2e-5,
                "rotation: {dense} != {sparse}"
            );
        }
        for (dense, sparse) in dense.translation.iter().zip(sparse.translation) {
            assert!(
                (dense - sparse).abs() < 2e-5,
                "translation: {dense} != {sparse}"
            );
        }
    }
    for (dense, sparse) in dense_points
        .iter()
        .flatten()
        .zip(sparse_points.iter().flatten())
    {
        assert!((dense - sparse).abs() < 3e-5, "point: {dense} != {sparse}");
    }
    for (dense, sparse) in dense_intrinsics.iter().zip(sparse_intrinsics) {
        assert!(
            ((dense.focal - sparse.focal) / dense.focal).abs() < 2e-5,
            "focal: {} != {}",
            dense.focal,
            sparse.focal
        );
        assert!(
            (dense.k1 - sparse.k1).abs() < 2e-5,
            "k1: {} != {}",
            dense.k1,
            sparse.k1
        );
        assert!(
            (dense.k2 - sparse.k2).abs() < 2e-5,
            "k2: {} != {}",
            dense.k2,
            sparse.k2
        );
    }
}

#[cfg(feature = "parallel")]
fn exact_state_bits(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    intrinsics: &[RadialCamera],
) -> Vec<u64> {
    let mut bits = Vec::with_capacity(poses.len() * 12 + points.len() * 3 + intrinsics.len() * 3);
    for pose in poses {
        bits.extend(pose.rotation.iter().flatten().map(|value| value.to_bits()));
        bits.extend(pose.translation.iter().map(|value| value.to_bits()));
    }
    bits.extend(points.iter().flatten().map(|value| value.to_bits()));
    for camera in intrinsics {
        bits.extend([
            camera.focal.to_bits(),
            camera.k1.to_bits(),
            camera.k2.to_bits(),
        ]);
    }
    bits
}
