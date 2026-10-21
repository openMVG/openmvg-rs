// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sparse multi-view bundle adjustment on a deterministic synthetic camera ring.
//!
//! Each point is observed by eight cameras; the solve uses an L2 objective, at most 30
//! iterations, and a fixed-baseline gauge. The benchmark times:
//!
//! - the direct LLT camera solver with recomputed and cached point back-substitution at 64,
//!   128, and 256 cameras;
//! - the direct LLT camera solver with cached back-substitution at 512 and 1024 cameras;
//! - the iterative Schur camera solver with cached back-substitution at 64 to 1024 cameras.
//!
//! Each configuration prints one `MULTI_BA_BENCH,...` line with the median wall time, the
//! workspace sizes in bytes, the final cost and RMSE, and the iterative solver counters.
//!
//! Environment selectors:
//!
//! - `OPENMVG_BA_BA_CAMERA_COUNT=N` runs only the configurations with `N` cameras.
//! - `OPENMVG_BA_BA_SOLVER=<name>` skips the iterative sweep unless `<name>` is `iterative`;
//!   the direct sweeps always run.
//! - `OPENMVG_BA_BA_POINT_COUNT=N` sets the point count (default 1024).
//! - `OPENMVG_BA_BA_MATVEC=fused|auto|parallel:N` sets the retained Schur matrix-vector policy
//!   (default `fused`).
//! - `OPENMVG_BA_BA_DIRECT_ASSEMBLY=serial|auto|parallel:N` sets the direct assembly policy for
//!   cached back-substitution (default `serial`).
//!
//! Run with: pixi run cargo bench -p openmvg-ba --bench multi_view_ba --features unstable-internals

use std::hint::black_box;
use std::time::{Duration, Instant};

use openmvg_ba::kernel::{
    PinholeCamera, RelativePose, multi_view::MultiViewMeasurements, multi_view::PixelObservation,
    multi_view::ScaleGauge,
};
#[cfg(feature = "faer")]
use openmvg_ba::kernel::{
    multi_view::sparse::DirectSchurAssembly, multi_view::sparse::RetainedSchurMatvec,
    multi_view::sparse::SparseBundleAdjustmentWorkspace, multi_view::sparse::SparseCameraSolver,
    multi_view::sparse::SparsePointBackSubstitution, multi_view::sparse::SparseWorkspaceOptions,
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

#[cfg(feature = "faer")]
fn benchmark_sparse(
    camera_count: usize,
    point_count: usize,
    observations_per_point: usize,
    point_back_substitution: SparsePointBackSubstitution,
    camera_solver: SparseCameraSolver,
) {
    let fixture = fixture(camera_count, point_count, observations_per_point);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &fixture.observations,
        point_offsets: &fixture.point_offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let retained_matvec = selected_matvec();
    let mut workspace = match camera_solver {
        SparseCameraSolver::DirectLlt => SparseBundleAdjustmentWorkspace::try_new_with_options(
            camera_count,
            measurements,
            gauge,
            SparseWorkspaceOptions::direct_with_assembly(
                point_back_substitution,
                if point_back_substitution == SparsePointBackSubstitution::Cached {
                    selected_direct_assembly()
                } else {
                    DirectSchurAssembly::Serial
                },
            ),
        ),
        SparseCameraSolver::IterativeSchur => {
            SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_matvec(
                camera_count,
                measurements,
                gauge,
                retained_matvec,
            )
        }
    }
    .expect("benchmark topology must be valid");
    let mut poses = fixture.initial_poses.clone();
    let mut points = fixture.initial_points.clone();
    let options = Options::l2(30);
    for _ in 0..2 {
        poses.copy_from_slice(&fixture.initial_poses);
        points.copy_from_slice(&fixture.initial_points);
        let report = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_sparse(
            black_box(&mut poses),
            black_box(&mut points),
            measurements,
            gauge,
            &mut workspace,
            options,
            NoiseModel::Unit,
        )
        .expect("deterministic benchmark fixture must be valid");
        let _ = black_box(report);
    }
    let samples = if camera_count <= 128 { 5 } else { 3 };
    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        poses.copy_from_slice(&fixture.initial_poses);
        points.copy_from_slice(&fixture.initial_points);
        let start = Instant::now();
        let report = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_sparse(
            black_box(&mut poses),
            black_box(&mut points),
            measurements,
            gauge,
            &mut workspace,
            options,
            NoiseModel::Unit,
        )
        .expect("deterministic benchmark fixture must be valid");
        timings.push((start.elapsed(), report));
    }
    timings.sort_unstable_by_key(|sample| sample.0);
    let (median, report) = timings[timings.len() / 2];
    println!(
        "MULTI_BA_BENCH,solver={camera_solver:?},direct_assembly_threads={:?},matvec={:?},matvec_threads={:?},gauge=fixed,point_back_substitution={:?},cameras={camera_count},points={point_count},observations_per_point={observations_per_point},iterations={},nonzeros={},point_cache_bytes={},retained_schur_bytes={},direct_numeric_bytes={},median_us={:.3},ns_per_observation_iteration={:.3},final_cost={:.9e},final_rmse={:.9},quality={:?},retained_steps={},fallback_steps={},inner_iterations={},max_retained_residual={:.6e}",
        workspace.execution_decision().direct_assembly_threads(),
        workspace.retained_schur_matvec(),
        workspace.retained_schur_parallel_threads(),
        workspace.point_back_substitution(),
        report.iteration_count,
        workspace.reduced_nonzero_count(),
        workspace.point_linearization_cache_bytes(),
        workspace.retained_schur_bytes(),
        workspace.direct_numeric_bytes(),
        micros(median),
        median.as_nanos() as f64
            / (fixture.observations.len() * report.iteration_count.max(1)) as f64,
        report.final_cost,
        report.final_rmse,
        report.camera_solve_quality,
        report.retained_camera_steps(),
        report.direct_fallback_steps,
        report.retained_inner_iterations(),
        report
            .maximum_retained_relative_residual()
            .unwrap_or(f64::NAN),
    );
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

#[cfg(feature = "faer")]
fn selected_camera_count(camera_count: usize) -> bool {
    std::env::var("OPENMVG_BA_BA_CAMERA_COUNT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .is_none_or(|selected| selected == camera_count)
}

#[cfg(feature = "faer")]
fn selected_solver(name: &str) -> bool {
    std::env::var("OPENMVG_BA_BA_SOLVER").map_or(true, |selected| selected == name)
}

#[cfg(feature = "faer")]
fn selected_point_count() -> usize {
    std::env::var("OPENMVG_BA_BA_POINT_COUNT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1024)
}

#[cfg(feature = "faer")]
fn selected_matvec() -> RetainedSchurMatvec {
    let Ok(selected) = std::env::var("OPENMVG_BA_BA_MATVEC") else {
        return RetainedSchurMatvec::Fused;
    };
    match selected.as_str() {
        "fused" => RetainedSchurMatvec::Fused,
        "auto" => RetainedSchurMatvec::Auto,
        _ => selected
            .strip_prefix("parallel:")
            .and_then(|threads| threads.parse::<usize>().ok())
            .map(|threads| RetainedSchurMatvec::PointMajorParallel { threads })
            .unwrap_or_else(|| {
                panic!(
                    "invalid OPENMVG_BA_BA_MATVEC={selected:?}; expected fused, auto, or parallel:N"
                )
            }),
    }
}

#[cfg(feature = "faer")]
fn selected_direct_assembly() -> DirectSchurAssembly {
    let Ok(selected) = std::env::var("OPENMVG_BA_BA_DIRECT_ASSEMBLY") else {
        return DirectSchurAssembly::Serial;
    };
    match selected.as_str() {
        "serial" => DirectSchurAssembly::Serial,
        "auto" => DirectSchurAssembly::Auto,
        _ => selected
            .strip_prefix("parallel:")
            .and_then(|threads| threads.parse::<usize>().ok())
            .map(|threads| DirectSchurAssembly::Parallel { threads })
            .unwrap_or_else(|| {
                panic!(
                    "invalid OPENMVG_BA_BA_DIRECT_ASSEMBLY={selected:?}; expected serial, auto, or parallel:N"
                )
            }),
    }
}

fn main() {
    #[cfg(feature = "faer")]
    let sparse_point_count = selected_point_count();
    #[cfg(feature = "faer")]
    for point_back_substitution in [
        SparsePointBackSubstitution::Recompute,
        SparsePointBackSubstitution::Cached,
    ] {
        for camera_count in [64, 128, 256]
            .into_iter()
            .filter(|camera_count| selected_camera_count(*camera_count))
        {
            benchmark_sparse(
                camera_count,
                sparse_point_count,
                8,
                point_back_substitution,
                SparseCameraSolver::DirectLlt,
            );
        }
    }
    #[cfg(feature = "faer")]
    for camera_count in [512, 1024]
        .into_iter()
        .filter(|camera_count| selected_camera_count(*camera_count))
    {
        benchmark_sparse(
            camera_count,
            sparse_point_count,
            8,
            SparsePointBackSubstitution::Cached,
            SparseCameraSolver::DirectLlt,
        );
    }
    #[cfg(feature = "faer")]
    if selected_solver("iterative") {
        for camera_count in [64, 128, 256, 512, 1024]
            .into_iter()
            .filter(|camera_count| selected_camera_count(*camera_count))
        {
            benchmark_sparse(
                camera_count,
                sparse_point_count,
                8,
                SparsePointBackSubstitution::Cached,
                SparseCameraSolver::IterativeSchur,
            );
        }
    }
}
