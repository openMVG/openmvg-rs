// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "faer")]

mod constraints;
mod direct;
mod iterative_schur;
mod workspace;

use openmvg_ba::kernel::multi_view::CameraPoseVariables;

use openmvg_ba::kernel::multi_view::sparse::{
    IterativeSchurOptions, RetainedSchurMatvec, SparseBundleAdjustmentCovarianceError,
    SparseCameraSolveQuality, SparseExecutionBackend, SparseWorkspaceMemoryBudget,
    selected_marginal_covariances_ba_multi_view_sparse,
};

use openmvg_ba::kernel::{
    PinholeCamera, RelativePose, multi_view::AngularObservation, multi_view::BundleAdjustmentBlock,
    multi_view::BundleAdjustmentPriors, multi_view::FixedVariableBlocks,
    multi_view::MultiViewBundleAdjustmentError, multi_view::MultiViewMeasurements,
    multi_view::PixelObservation, multi_view::PosePrior, multi_view::ScaleGauge,
    multi_view::TranslationComponent, multi_view::sparse::SparseBundleAdjustmentError,
    multi_view::sparse::SparseBundleAdjustmentWorkspace,
    multi_view::sparse::SparseCameraRankAssessment, multi_view::sparse::SparseWorkspaceOptions,
    multi_view::sparse::refine_ba_multi_view_sparse,
    multi_view::sparse::refine_ba_multi_view_sparse_with_factors,
};

use openmvg_ba::numerics::refine::{Loss, NoiseModel, Options};

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn rotation_y(angle: f64) -> [[f64; 3]; 3] {
    let (sine, cosine) = angle.sin_cos();
    [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
}

fn transform(pose: RelativePose, point: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        (0..3)
            .map(|column| pose.rotation[row][column] * point[column])
            .sum::<f64>()
            + pose.translation[row]
    })
}

fn norm(vector: [f64; 3]) -> f64 {
    vector
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn normalize(vector: [f64; 3]) -> [f64; 3] {
    let magnitude = norm(vector);
    vector.map(|value| value / magnitude)
}

struct Fixture {
    initial_poses: Vec<RelativePose>,
    initial_points: Vec<[f64; 3]>,
    pixels: Vec<PixelObservation>,
    bearings: Vec<AngularObservation>,
    offsets: Vec<usize>,
    cameras: Vec<PinholeCamera>,
}

struct ChainFixture {
    poses: Vec<RelativePose>,
    points: Vec<[f64; 3]>,
    pixels: Vec<PixelObservation>,
    offsets: Vec<usize>,
    cameras: Vec<PinholeCamera>,
}

fn fixture() -> Fixture {
    let truth_poses = vec![
        RelativePose::new(IDENTITY, [0.0; 3]),
        RelativePose::new(rotation_y(0.06), [0.72, 0.03, 0.06]),
        RelativePose::new(rotation_y(-0.08), [-0.57, 0.05, 0.11]),
        RelativePose::new(rotation_y(0.11), [0.18, -0.43, 0.09]),
    ];
    let truth_points = (0..30)
        .map(|index| {
            [
                -1.15 + 0.46 * (index % 6) as f64,
                -0.78 + 0.38 * (index / 6) as f64,
                4.2 + 0.21 * (index % 7) as f64,
            ]
        })
        .collect::<Vec<_>>();
    let camera = PinholeCamera::new(760.0, 735.0, 320.0, 240.0);
    let cameras = vec![camera; truth_poses.len()];
    let mut pixels = Vec::new();
    let mut bearings = Vec::new();
    let mut offsets = vec![0];
    for (point_index, &point) in truth_points.iter().enumerate() {
        for (camera_index, &pose) in truth_poses.iter().enumerate() {
            if camera_index == 3 && point_index % 4 == 0 {
                continue;
            }
            let camera_point = transform(pose, point);
            pixels.push(PixelObservation::new(
                camera_index,
                [
                    camera.fx * camera_point[0] / camera_point[2] + camera.cx,
                    camera.fy * camera_point[1] / camera_point[2] + camera.cy,
                ],
            ));
            bearings.push(AngularObservation::new(
                camera_index,
                normalize(camera_point),
            ));
        }
        offsets.push(pixels.len());
    }

    let mut initial_poses = truth_poses.clone();
    for (camera_index, pose) in initial_poses.iter_mut().enumerate().skip(1) {
        pose.translation[0] += 0.045 * camera_index as f64;
        pose.translation[1] -= 0.026 * camera_index as f64;
        pose.translation[2] += 0.032;
    }
    let initial_points = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            [
                point[0] + 0.035 * (0.7 * index as f64).sin(),
                point[1] - 0.028 * (0.4 * index as f64).cos(),
                point[2] + 0.042 * (0.3 * index as f64).sin(),
            ]
        })
        .collect();

    Fixture {
        initial_poses,
        initial_points,
        pixels,
        bearings,
        offsets,
        cameras,
    }
}

fn chain_fixture(camera_count: usize, points_per_triplet: usize) -> ChainFixture {
    assert!(camera_count >= 3);
    let spacing = 0.025;
    let camera = PinholeCamera::new(900.0, 900.0, 640.0, 480.0);
    let truth_poses = (0..camera_count)
        .map(|camera_index| RelativePose::new(IDENTITY, [-spacing * camera_index as f64, 0.0, 0.0]))
        .collect::<Vec<_>>();
    let mut poses = truth_poses.clone();
    for (camera_index, pose) in poses.iter_mut().enumerate().skip(1) {
        pose.translation[0] += 0.0008 * (0.13 * camera_index as f64).sin();
        pose.translation[1] += 0.0005 * (0.07 * camera_index as f64).cos();
        pose.translation[2] += 0.0004 * (0.11 * camera_index as f64).sin();
    }

    let mut points = Vec::with_capacity((camera_count - 2) * points_per_triplet);
    let mut pixels = Vec::with_capacity((camera_count - 2) * points_per_triplet * 3);
    let mut offsets = Vec::with_capacity((camera_count - 2) * points_per_triplet + 1);
    offsets.push(0);
    for first_camera in 0..camera_count - 2 {
        for sample in 0..points_per_triplet {
            let truth_point = [
                spacing * (first_camera as f64 + 1.0) + 0.012 * (sample as f64 - 0.5),
                0.08 * (sample as f64 - 0.5),
                4.5 + 0.1 * ((first_camera + sample) % 5) as f64,
            ];
            for (camera_index, pose) in truth_poses.iter().enumerate().skip(first_camera).take(3) {
                let camera_point = transform(*pose, truth_point);
                pixels.push(PixelObservation::new(
                    camera_index,
                    [
                        camera.fx * camera_point[0] / camera_point[2] + camera.cx,
                        camera.fy * camera_point[1] / camera_point[2] + camera.cy,
                    ],
                ));
            }
            let point_index = points.len();
            points.push([
                truth_point[0] + 0.001 * (0.17 * point_index as f64).sin(),
                truth_point[1] - 0.001 * (0.19 * point_index as f64).cos(),
                truth_point[2] + 0.002 * (0.05 * point_index as f64).sin(),
            ]);
            offsets.push(pixels.len());
        }
    }

    ChainFixture {
        poses,
        points,
        pixels,
        offsets,
        cameras: vec![camera; camera_count],
    }
}

fn pixel_measurements(fixture: &Fixture) -> MultiViewMeasurements<'_> {
    MultiViewMeasurements::Pixel {
        observations: &fixture.pixels,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    }
}

fn angular_measurements(fixture: &Fixture) -> MultiViewMeasurements<'_> {
    MultiViewMeasurements::Angular {
        observations: &fixture.bearings,
        point_offsets: &fixture.offsets,
    }
}

fn translation_only_workspace(
    camera_count: usize,
    measurements: MultiViewMeasurements<'_>,
    gauge: ScaleGauge,
) -> SparseBundleAdjustmentWorkspace {
    translation_only_workspace_with_options(
        camera_count,
        measurements,
        gauge,
        SparseWorkspaceOptions::direct(),
    )
}

fn translation_only_workspace_with_options(
    camera_count: usize,
    measurements: MultiViewMeasurements<'_>,
    gauge: ScaleGauge,
    options: SparseWorkspaceOptions,
) -> SparseBundleAdjustmentWorkspace {
    SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks_and_pose_variables(
        camera_count,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        CameraPoseVariables::TranslationOnly,
        options,
    )
    .expect("translation-only sparse workspace must be supported")
}

fn translation_only_workspace_with_fixed(
    camera_count: usize,
    measurements: MultiViewMeasurements<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
) -> SparseBundleAdjustmentWorkspace {
    SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks_and_pose_variables(
        camera_count,
        measurements,
        gauge,
        fixed,
        CameraPoseVariables::TranslationOnly,
        SparseWorkspaceOptions::direct(),
    )
    .expect("translation-only direct workspace with fixed blocks must be supported")
}

fn assert_close(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "{left:.16e} differs from {right:.16e} by more than {tolerance:.3e}"
    );
}
