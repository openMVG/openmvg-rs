// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "faer")]

mod dense_parity;
mod gauge;
mod iterative_schur;
mod workspace;

use openmvg_ba::kernel::multi_view::sparse::{
    SparseExecutionBackend, SparseExecutionReason, SparseWorkspaceMemoryBudget,
};

use openmvg_ba::kernel::{
    PinholeCamera, RelativePose, multi_view::BundleAdjustmentPriors,
    multi_view::CalibratedBundleAdjustmentProblem, multi_view::FixedVariableBlocks,
    multi_view::MultiViewBundleAdjustmentError, multi_view::MultiViewMeasurements,
    multi_view::PixelObservation, multi_view::PointPrior, multi_view::PosePrior,
    multi_view::ScaleGauge, multi_view::TranslationComponent,
    multi_view::dense::DenseBundleAdjustmentWorkspace,
    multi_view::dense::refine_ba_multi_view_dense,
    multi_view::dense::refine_ba_multi_view_dense_with_factors,
    multi_view::sparse::IterativeSchurOptions, multi_view::sparse::RetainedCrossPrecision,
    multi_view::sparse::RetainedSchurMatvec, multi_view::sparse::SparseBundleAdjustmentError,
    multi_view::sparse::SparseBundleAdjustmentWorkspace,
    multi_view::sparse::SparseCameraRankAssessment, multi_view::sparse::SparseCameraSolveQuality,
    multi_view::sparse::SparseCameraSolver, multi_view::sparse::SparsePointBackSubstitution,
    multi_view::sparse::SparseWorkspaceOptions, multi_view::sparse::refine_ba_multi_view_sparse,
    multi_view::sparse::refine_ba_multi_view_sparse_with_factors,
};

use openmvg_ba::numerics::refine::{
    ConvergenceCriteria, ConvergenceCriterion, NoiseModel, Options, RefinementStatus,
};

fn identity_pose() -> RelativePose {
    RelativePose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    )
}

fn rotation_y(angle: f64) -> [[f64; 3]; 3] {
    let (sine, cosine) = angle.sin_cos();
    [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
}

fn transform(pose: &RelativePose, point: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        (0..3)
            .map(|column| pose.rotation[row][column] * point[column])
            .sum::<f64>()
            + pose.translation[row]
    })
}

struct Fixture {
    truth_poses: Vec<RelativePose>,
    truth_points: Vec<[f64; 3]>,
    observations: Vec<PixelObservation>,
    offsets: Vec<usize>,
    cameras: [PinholeCamera; 4],
}

fn fixture() -> Fixture {
    let truth_poses = vec![
        identity_pose(),
        RelativePose::new(rotation_y(0.04), [0.62, 0.02, 0.05]),
        RelativePose::new(rotation_y(-0.05), [-0.51, 0.04, 0.09]),
        RelativePose::new(rotation_y(0.07), [0.14, -0.37, 0.08]),
    ];
    let truth_points: Vec<_> = (0..24)
        .map(|index| {
            [
                -1.0 + 0.4 * (index % 6) as f64,
                -0.65 + 0.42 * (index / 6) as f64,
                4.0 + 0.16 * (index % 5) as f64,
            ]
        })
        .collect();
    let camera = PinholeCamera::new(720.0, 710.0, 320.0, 240.0);
    let cameras = [camera; 4];
    let mut observations = Vec::new();
    let mut offsets = vec![0];
    for (point_index, point) in truth_points.iter().enumerate() {
        for (camera_index, pose) in truth_poses.iter().enumerate() {
            if camera_index == 3 && point_index % 3 == 0 {
                continue;
            }
            let value = transform(pose, point);
            observations.push(PixelObservation::new(
                camera_index,
                [
                    camera.fx * value[0] / value[2] + camera.cx,
                    camera.fy * value[1] / value[2] + camera.cy,
                ],
            ));
        }
        offsets.push(observations.len());
    }
    Fixture {
        truth_poses,
        truth_points,
        observations,
        offsets,
        cameras,
    }
}

fn perturbed(
    truth_poses: &[RelativePose],
    truth_points: &[[f64; 3]],
) -> (Vec<RelativePose>, Vec<[f64; 3]>) {
    let mut poses = truth_poses.to_vec();
    for (camera, pose) in poses.iter_mut().enumerate().skip(1) {
        pose.translation[0] += 0.025 * camera as f64;
        pose.translation[1] -= 0.012 * camera as f64;
        pose.translation[2] += 0.018;
    }
    let points = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            [
                point[0] + 0.018 * (index as f64).sin(),
                point[1] - 0.015 * (0.7 * index as f64).cos(),
                point[2] + 0.025 * (0.4 * index as f64).sin(),
            ]
        })
        .collect();
    (poses, points)
}
