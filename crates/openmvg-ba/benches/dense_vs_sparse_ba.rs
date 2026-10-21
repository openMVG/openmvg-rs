// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dense-Schur against sparse multi-view bundle adjustment on small scenes.
//!
//! The native adjuster routes every multi-view solve to the sparse kernel. This benchmark asks
//! whether the dense-Schur kernel (`refine_ba_multi_view_dense`) would be faster for the small
//! problems where its dense camera system is cheap. Both kernels solve the same deterministic
//! synthetic problem (pinhole cameras on a ring, eight observations per point, L2 objective, at
//! most 30 iterations, fixed-baseline gauge) from the same start, and each configuration prints
//! one `DENSE_VS_SPARSE,...` line with the median wall time over seven solves, the iteration
//! count, and the final cost.
//!
//! Environment selectors: `OPENMVG_BA_DVS_CAMERAS=N` and `OPENMVG_BA_DVS_POINTS=N` run one size.
//!
//! Run with:
//! pixi run cargo bench -p openmvg-ba --bench dense_vs_sparse_ba --features unstable-internals,faer

use std::hint::black_box;
use std::time::Instant;

use openmvg_ba::kernel::{
    PinholeCamera, RelativePose,
    multi_view::{
        MultiViewMeasurements, PixelObservation, ScaleGauge,
        dense::{DenseBundleAdjustmentWorkspace, refine_ba_multi_view_dense},
        sparse::{
            SparseBundleAdjustmentWorkspace, SparsePointBackSubstitution, SparseWorkspaceOptions,
            refine_ba_multi_view_sparse,
        },
    },
};
use openmvg_ba::numerics::manifold;
use openmvg_ba::numerics::refine::{NoiseModel, Options};

struct Fixture {
    initial_poses: Vec<RelativePose>,
    initial_points: Vec<[f64; 3]>,
    observations: Vec<PixelObservation>,
    point_offsets: Vec<usize>,
    cameras: Vec<PinholeCamera>,
}

fn transform(pose: &RelativePose, point: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        pose.rotation[row][0] * point[0]
            + pose.rotation[row][1] * point[1]
            + pose.rotation[row][2] * point[2]
            + pose.translation[row]
    })
}

fn pixel(camera: PinholeCamera, point: &[f64; 3]) -> [f64; 2] {
    [
        (camera.fx * point[0] + camera.skew * point[1]) / point[2] + camera.cx,
        camera.fy * point[1] / point[2] + camera.cy,
    ]
}

fn fixture(camera_count: usize, point_count: usize, observations_per_point: usize) -> Fixture {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let truth_poses: Vec<_> = (0..camera_count)
        .map(|camera| {
            if camera == 0 {
                RelativePose::new(identity, [0.0; 3])
            } else {
                let angle = std::f64::consts::TAU * camera as f64 / camera_count as f64;
                RelativePose::new(
                    manifold::exp_so3(&[0.008 * angle, -0.025 * angle, 0.006 * angle]),
                    [0.7 * angle.cos(), 0.7 * angle.sin(), 0.04 * angle.sin()],
                )
            }
        })
        .collect();
    let initial_poses = truth_poses
        .iter()
        .enumerate()
        .map(|(camera, pose)| {
            if camera == 0 {
                *pose
            } else {
                let phase = 0.13 * camera as f64;
                let error = manifold::exp_so3(&[
                    0.006 * phase.sin(),
                    -0.005 * phase.cos(),
                    0.003 * (0.7 * phase).sin(),
                ]);
                RelativePose::new(
                    manifold::mat_mul(&error, &pose.rotation),
                    [
                        pose.translation[0] + 0.004,
                        pose.translation[1] - 0.003,
                        pose.translation[2] + 0.002,
                    ],
                )
            }
        })
        .collect();
    let truth_points: Vec<_> = (0..point_count)
        .map(|index| {
            let column = (index % 32) as f64;
            let row = ((index / 32) % 32) as f64;
            [
                0.045 * (column - 15.5),
                0.04 * (row - 15.5),
                3.5 + 0.15 * (0.13 * index as f64).sin(),
            ]
        })
        .collect();
    let initial_points = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            [
                point[0] + 0.006 * (0.31 * index as f64).sin(),
                point[1] + 0.006 * (0.27 * index as f64).cos(),
                point[2] + 0.009 * (0.19 * index as f64).sin(),
            ]
        })
        .collect();
    let cameras = vec![PinholeCamera::new(500.0, 505.0, 320.0, 240.0); camera_count];
    let track_length = observations_per_point.min(camera_count).max(2);
    let mut observations = Vec::with_capacity(track_length * point_count);
    let mut point_offsets = Vec::with_capacity(point_count + 1);
    point_offsets.push(0);
    for (point_index, point) in truth_points.iter().enumerate() {
        observations.push(PixelObservation::new(
            0,
            pixel(cameras[0], &transform(&truth_poses[0], point)),
        ));
        for track_camera in 1..track_length {
            let camera = 1 + (point_index + track_camera - 1) % (camera_count - 1);
            let mut measurement = pixel(cameras[camera], &transform(&truth_poses[camera], point));
            measurement[0] += 0.08 * (0.17 * (point_index + camera) as f64).sin();
            measurement[1] += 0.08 * (0.23 * (point_index + 2 * camera) as f64).cos();
            observations.push(PixelObservation::new(camera, measurement));
        }
        point_offsets.push(observations.len());
    }
    Fixture {
        initial_poses,
        initial_points,
        observations,
        point_offsets,
        cameras,
    }
}

const SAMPLES: usize = 7;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn selected(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn run(camera_count: usize, point_count: usize) {
    let fixture = fixture(camera_count, point_count, 8);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.point_offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let options = Options::l2(30);
    let observations = fixture.observations.len();
    let mut poses = fixture.initial_poses.clone();
    let mut points = fixture.initial_points.clone();

    let mut dense_us = f64::NAN;
    for (name, final_diagnostics) in [("dense", true), ("dense_no_final_diagnostics", false)] {
        let mut dense = DenseBundleAdjustmentWorkspace::try_new(camera_count, point_count)
            .expect("dense capacity fits")
            .with_final_diagnostics(final_diagnostics);
        let mut dense_run = || {
            poses.copy_from_slice(&fixture.initial_poses);
            points.copy_from_slice(&fixture.initial_points);
            let start = Instant::now();
            let report = refine_ba_multi_view_dense(
                black_box(&mut poses),
                black_box(&mut points),
                measurements,
                gauge,
                &mut dense,
                options,
                NoiseModel::Unit,
            )
            .expect("dense solve");
            (
                start.elapsed().as_secs_f64() * 1e6,
                report.iteration_count,
                report.final_cost,
            )
        };
        dense_run();
        let timings: Vec<_> = (0..SAMPLES).map(|_| dense_run()).collect();
        let (iterations, cost) = (timings[0].1, timings[0].2);
        let median_us = median(timings.iter().map(|t| t.0).collect());
        if final_diagnostics {
            dense_us = median_us;
        }
        println!(
            "DENSE_VS_SPARSE,kernel={name},cameras={camera_count},points={point_count},observations={observations},iterations={iterations},median_us={median_us:.1},final_cost={cost:.9e}"
        );
    }

    for (name, back_substitution) in [
        ("sparse_llt_cached", SparsePointBackSubstitution::Cached),
        (
            "sparse_llt_recompute",
            SparsePointBackSubstitution::Recompute,
        ),
    ] {
        let mut workspace = SparseBundleAdjustmentWorkspace::try_new_with_options(
            camera_count,
            measurements,
            gauge,
            SparseWorkspaceOptions::direct_with_assembly(
                back_substitution,
                openmvg_ba::kernel::multi_view::sparse::DirectSchurAssembly::Serial,
            ),
        )
        .expect("sparse topology");
        let mut sparse_run = || {
            poses.copy_from_slice(&fixture.initial_poses);
            points.copy_from_slice(&fixture.initial_points);
            let start = Instant::now();
            let report = refine_ba_multi_view_sparse(
                black_box(&mut poses),
                black_box(&mut points),
                measurements,
                gauge,
                &mut workspace,
                options,
                NoiseModel::Unit,
            )
            .expect("sparse solve");
            (
                start.elapsed().as_secs_f64() * 1e6,
                report.iteration_count,
                report.final_cost,
            )
        };
        sparse_run();
        let timings: Vec<_> = (0..SAMPLES).map(|_| sparse_run()).collect();
        let sparse_us = median(timings.iter().map(|t| t.0).collect());
        println!(
            "DENSE_VS_SPARSE,kernel={name},cameras={camera_count},points={point_count},observations={observations},iterations={},median_us={sparse_us:.1},final_cost={:.9e},speedup_vs_dense={:.2}",
            timings[0].1,
            timings[0].2,
            dense_us / sparse_us
        );
    }
}

fn main() {
    for camera_count in [8, 16, 32, 64, 96] {
        for point_count in [200, 800, 3000] {
            if selected("OPENMVG_BA_DVS_CAMERAS").is_some_and(|n| n != camera_count)
                || selected("OPENMVG_BA_DVS_POINTS").is_some_and(|n| n != point_count)
            {
                continue;
            }
            run(camera_count, point_count);
        }
    }
}
