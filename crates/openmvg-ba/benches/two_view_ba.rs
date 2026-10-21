// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Two-view bundle adjustment on a deterministic synthetic pair.
//!
//! Times `refine_ba_2view` with pixel and angular measurements at 32 to 8192 points, six
//! Levenberg-Marquardt iterations, and an L2 objective. Each configuration prints one
//! `BA_BENCH,...` line with the median and minimum wall time, the time per point iteration,
//! and the final RMSE.
//!
//! Run with: pixi run cargo bench -p openmvg-ba --bench two_view_ba --features unstable-internals

use std::hint::black_box;
use std::time::{Duration, Instant};

use openmvg_ba::kernel::{PinholeCamera, RelativePose, two_view::TwoViewMeasurements};
use openmvg_ba::numerics::manifold;
use openmvg_ba::numerics::refine::{ConvergenceCriteria, NoiseModel, Options};

#[derive(Clone, Copy)]
enum Mode {
    Pixel,
    Angular,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Pixel => "pixel",
            Self::Angular => "angular",
        }
    }
}

struct Fixture {
    initial_pose: RelativePose,
    initial_points: Vec<[f64; 3]>,
    pixels1: Vec<[f64; 2]>,
    pixels2: Vec<[f64; 2]>,
    bearings1: Vec<[f64; 3]>,
    bearings2: Vec<[f64; 3]>,
    cameras: [PinholeCamera; 2],
}

fn transform(pose: &RelativePose, point: &[f64; 3]) -> [f64; 3] {
    [
        pose.rotation[0][0] * point[0]
            + pose.rotation[0][1] * point[1]
            + pose.rotation[0][2] * point[2]
            + pose.translation[0],
        pose.rotation[1][0] * point[0]
            + pose.rotation[1][1] * point[1]
            + pose.rotation[1][2] * point[2]
            + pose.translation[1],
        pose.rotation[2][0] * point[0]
            + pose.rotation[2][1] * point[1]
            + pose.rotation[2][2] * point[2]
            + pose.translation[2],
    ]
}

fn normalize(vector: [f64; 3]) -> [f64; 3] {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    vector.map(|value| value / norm)
}

fn pixel(camera: PinholeCamera, point: &[f64; 3]) -> [f64; 2] {
    [
        (camera.fx * point[0] + camera.skew * point[1]) / point[2] + camera.cx,
        camera.fy * point[1] / point[2] + camera.cy,
    ]
}

fn fixture(point_count: usize) -> Fixture {
    let truth_pose = RelativePose::new(
        manifold::exp_so3(&[0.025, -0.055, 0.018]),
        [0.85, 0.1, 0.08],
    );
    let rotation_error = manifold::exp_so3(&[0.012, -0.009, 0.006]);
    let raw_translation = [0.845, 0.12, 0.065];
    let truth_baseline = truth_pose
        .translation
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let raw_norm = raw_translation
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let initial_pose = RelativePose::new(
        manifold::mat_mul(&rotation_error, &truth_pose.rotation),
        raw_translation.map(|value| truth_baseline * value / raw_norm),
    );
    let cameras = [
        PinholeCamera::new(500.0, 505.0, 320.0, 240.0),
        PinholeCamera::new(515.0, 510.0, 318.0, 242.0).with_skew(0.15),
    ];
    let truth_points: Vec<_> = (0..point_count)
        .map(|index| {
            let x = (index % 64) as f64 * 0.025 - 0.8;
            let y = ((index / 64) % 64) as f64 * 0.022 - 0.7;
            let z = 3.0 + 0.0007 * index as f64 + 0.08 * (index as f64 * 0.17).sin();
            [x, y, z]
        })
        .collect();
    let initial_points = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let phase = index as f64;
            [
                point[0] + 0.008 * phase.sin(),
                point[1] + 0.007 * (0.7 * phase).cos(),
                point[2] + 0.012 * (0.31 * phase).sin(),
            ]
        })
        .collect();

    let pixels1 = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let mut observation = pixel(cameras[0], point);
            observation[0] += 0.12 * (index as f64 * 0.37).sin();
            observation[1] += 0.12 * (index as f64 * 0.51).cos();
            observation
        })
        .collect();
    let pixels2 = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let mut observation = pixel(cameras[1], &transform(&truth_pose, point));
            observation[0] += 0.12 * (index as f64 * 0.43).cos();
            observation[1] += 0.12 * (index as f64 * 0.29).sin();
            observation
        })
        .collect();
    let bearings1 = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            normalize([
                point[0] + 2e-4 * (index as f64 * 0.37).sin(),
                point[1] + 2e-4 * (index as f64 * 0.51).cos(),
                point[2],
            ])
        })
        .collect();
    let bearings2 = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let mut transformed = transform(&truth_pose, point);
            transformed[0] += 2e-4 * (index as f64 * 0.43).cos();
            transformed[1] += 2e-4 * (index as f64 * 0.29).sin();
            normalize(transformed)
        })
        .collect();

    Fixture {
        initial_pose,
        initial_points,
        pixels1,
        pixels2,
        bearings1,
        bearings2,
        cameras,
    }
}

fn run(
    fixture: &Fixture,
    mode: Mode,
    points: &mut [[f64; 3]],
) -> openmvg_ba::kernel::two_view::BundleAdjustmentReport {
    let measurements = match mode {
        Mode::Pixel => TwoViewMeasurements::Pixel {
            observations1: &fixture.pixels1,
            observations2: &fixture.pixels2,
            cameras: fixture.cameras,
        },
        Mode::Angular => TwoViewMeasurements::Angular {
            bearings1: &fixture.bearings1,
            bearings2: &fixture.bearings2,
        },
    };
    let options = Options::l2(6).with_convergence(ConvergenceCriteria::new(1e-30, 1e-30, 1e-30));
    openmvg_ba::kernel::two_view::refine_ba_2view(
        black_box(&fixture.initial_pose),
        black_box(points),
        black_box(measurements),
        options,
        NoiseModel::Unit,
    )
    .expect("deterministic benchmark fixture must be valid")
}

fn benchmark(point_count: usize, mode: Mode) {
    let fixture = fixture(point_count);
    let mut points = fixture.initial_points.clone();
    for _ in 0..3 {
        points.copy_from_slice(&fixture.initial_points);
        let _ = black_box(run(&fixture, mode, &mut points));
    }

    let samples = if point_count <= 512 {
        31
    } else if point_count <= 2048 {
        17
    } else {
        9
    };
    let mut measurements = Vec::with_capacity(samples);
    for _ in 0..samples {
        points.copy_from_slice(&fixture.initial_points);
        let start = Instant::now();
        let report = black_box(run(&fixture, mode, &mut points));
        measurements.push((start.elapsed(), report.iteration_count, report.final_rmse));
    }
    measurements.sort_unstable_by_key(|sample| sample.0);
    let (median, iterations, rmse) = measurements[measurements.len() / 2];
    let minimum = measurements[0].0;
    let point_iterations = point_count * iterations.max(1);
    println!(
        "BA_BENCH,mode={},points={point_count},iterations={iterations},median_us={:.3},min_us={:.3},ns_per_point_iteration={:.3},final_rmse={rmse:.9}",
        mode.name(),
        micros(median),
        micros(minimum),
        median.as_nanos() as f64 / point_iterations as f64,
    );
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

fn main() {
    for mode in [Mode::Pixel, Mode::Angular] {
        for point_count in [32, 128, 512, 2048, 8192] {
            benchmark(point_count, mode);
        }
    }
}
