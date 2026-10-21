// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "faer")]

use openmvg_ba::kernel::{
    PinholeCamera, RelativePose, multi_view::MultiViewMeasurements, multi_view::PixelObservation,
    multi_view::ScaleGauge, multi_view::sparse::IterativeSchurOptions,
    multi_view::sparse::RetainedSchurMatvec, multi_view::sparse::SparseBundleAdjustmentReport,
    multi_view::sparse::SparseBundleAdjustmentWorkspace,
    multi_view::sparse::SparseCameraSolveQuality, multi_view::sparse::refine_ba_multi_view_sparse,
};
use openmvg_ba::numerics::refine::{NoiseModel, Options, RefinementStatus};

const CAMERA_COUNT: usize = 5;
const POINT_COUNT: usize = 40;
const SEED: u64 = 0x7a62_d419_c305_8ef1;

struct Scene {
    initial_poses: Vec<RelativePose>,
    initial_points: Vec<[f64; 3]>,
    observations: Vec<PixelObservation>,
    point_offsets: Vec<usize>,
    cameras: [PinholeCamera; CAMERA_COUNT],
}

#[derive(Clone, Copy)]
struct SeededRng(u64);

impl SeededRng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn range(&mut self, lower: f64, upper: f64) -> f64 {
        lower + (upper - lower) * self.unit()
    }
}

#[test]
fn seeded_sparse_backends_agree_across_scene_scales() {
    let scene = seeded_scene(SEED);

    for scene_scale in [1.0e-6, 1.0, 1.0e6] {
        compare_backends(&scene, scene_scale);
    }
}

fn compare_backends(scene: &Scene, scene_scale: f64) {
    let measurements = MultiViewMeasurements::Pixel {
        observations: &scene.observations,
        point_offsets: &scene.point_offsets,
        cameras: &scene.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let (initial_poses, initial_points) = scaled_initial_state(scene, scene_scale);
    let mut direct_poses = initial_poses.clone();
    let mut direct_points = initial_points.clone();
    let mut iterative_poses = initial_poses;
    let mut iterative_points = initial_points;
    let mut direct_workspace =
        SparseBundleAdjustmentWorkspace::try_new(CAMERA_COUNT, measurements, gauge)
            .expect("seeded direct topology must be valid");
    let iterative_options = IterativeSchurOptions::new()
        .with_maximum_iterations(400)
        .with_relative_tolerance(1.0e-10)
        .with_matvec(RetainedSchurMatvec::Fused)
        .with_exact_camera_rank_diagnostics(true);
    let mut iterative_workspace =
        SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
            CAMERA_COUNT,
            measurements,
            gauge,
            iterative_options,
        )
        .expect("seeded iterative topology must be valid");
    let options = Options::l2(35);

    let direct = refine_ba_multi_view_sparse(
        &mut direct_poses,
        &mut direct_points,
        measurements,
        gauge,
        &mut direct_workspace,
        options,
        NoiseModel::Unit,
    )
    .expect("direct sparse solve must remain valid across scene scales");
    let iterative = refine_ba_multi_view_sparse(
        &mut iterative_poses,
        &mut iterative_points,
        measurements,
        gauge,
        &mut iterative_workspace,
        options,
        NoiseModel::Unit,
    )
    .expect("retained iterative solve must remain valid across scene scales");

    assert_usable(&direct, scene_scale, "direct");
    assert_usable(&iterative, scene_scale, "iterative");
    assert_eq!(
        direct.camera_solve_quality,
        SparseCameraSolveQuality::Direct
    );
    assert_eq!(
        iterative.camera_solve_quality,
        SparseCameraSolveQuality::ResidualBoundedIterativeSchur,
        "scale={scene_scale:e}, {iterative:#?}"
    );
    assert!(iterative.iterative_camera_steps() > 0);
    assert_eq!(iterative.direct_fallback_steps, 0);
    assert_eq!(iterative.linear_guard_fallback_steps, 0);

    assert_scaled_close(
        direct.final_cost,
        iterative.final_cost,
        2.0e-9,
        2.0e-8,
        &format!("final objective at scene scale {scene_scale:e}"),
    );
    assert_scaled_close(
        direct.final_rmse,
        iterative.final_rmse,
        2.0e-10,
        2.0e-8,
        &format!("final RMSE at scene scale {scene_scale:e}"),
    );

    let coordinate_tolerance = 5.0e-7 * scene_scale;
    for (camera, (direct_pose, iterative_pose)) in
        direct_poses.iter().zip(&iterative_poses).enumerate()
    {
        for row in 0..3 {
            for column in 0..3 {
                assert_scaled_close(
                    direct_pose.rotation[row][column],
                    iterative_pose.rotation[row][column],
                    5.0e-8,
                    5.0e-8,
                    &format!("camera {camera} rotation at scene scale {scene_scale:e}"),
                );
            }
            assert!(
                (direct_pose.translation[row] - iterative_pose.translation[row]).abs()
                    <= coordinate_tolerance,
                "camera {camera} translation[{row}] differs at scale {scene_scale:e}: direct={:?}, iterative={:?}",
                direct_pose.translation,
                iterative_pose.translation
            );
        }
    }
    for (point, (direct_point, iterative_point)) in
        direct_points.iter().zip(&iterative_points).enumerate()
    {
        for coordinate in 0..3 {
            assert!(
                (direct_point[coordinate] - iterative_point[coordinate]).abs()
                    <= coordinate_tolerance,
                "point {point}[{coordinate}] differs at scale {scene_scale:e}: direct={direct_point:?}, iterative={iterative_point:?}"
            );
        }
    }

    compare_projections(
        scene,
        &direct_poses,
        &direct_points,
        &iterative_poses,
        &iterative_points,
        scene_scale,
    );
}

fn assert_usable(report: &SparseBundleAdjustmentReport, scene_scale: f64, backend: &str) {
    let usable_status = matches!(
        report.status,
        RefinementStatus::Converged(_) | RefinementStatus::MaxIterations
    );
    assert!(
        usable_status
            && report.accepted_step_count > 0
            && report.final_cost.is_finite()
            && report.final_cost <= report.initial_cost,
        "{backend} result is unusable at scale {scene_scale:e}: {report:#?}"
    );
}

fn compare_projections(
    scene: &Scene,
    direct_poses: &[RelativePose],
    direct_points: &[[f64; 3]],
    iterative_poses: &[RelativePose],
    iterative_points: &[[f64; 3]],
    scene_scale: f64,
) {
    for point in 0..POINT_COUNT {
        for observation in scene.point_offsets[point]..scene.point_offsets[point + 1] {
            let measurement = scene.observations[observation];
            let camera = measurement.camera;
            let direct = project(
                &direct_poses[camera],
                &direct_points[point],
                scene.cameras[camera],
            );
            let iterative = project(
                &iterative_poses[camera],
                &iterative_points[point],
                scene.cameras[camera],
            );
            for axis in 0..2 {
                assert_scaled_close(
                    direct[axis],
                    iterative[axis],
                    2.0e-7,
                    2.0e-9,
                    &format!(
                        "observation {observation} projection[{axis}] at scene scale {scene_scale:e}"
                    ),
                );
            }
        }
    }
}

fn assert_scaled_close(
    left: f64,
    right: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    context: &str,
) {
    let tolerance = absolute_tolerance + relative_tolerance * left.abs().max(right.abs());
    assert!(
        (left - right).abs() <= tolerance,
        "{context}: left={left:.16e}, right={right:.16e}, tolerance={tolerance:.3e}"
    );
}

fn scaled_initial_state(scene: &Scene, scene_scale: f64) -> (Vec<RelativePose>, Vec<[f64; 3]>) {
    let mut poses = scene.initial_poses.clone();
    for pose in poses.iter_mut().skip(1) {
        pose.translation = pose.translation.map(|value| scene_scale * value);
    }
    let points = scene
        .initial_points
        .iter()
        .map(|point| point.map(|value| scene_scale * value))
        .collect();
    (poses, points)
}

fn seeded_scene(seed: u64) -> Scene {
    let mut rng = SeededRng(seed);
    let angles = [0.0, 0.045, -0.055, 0.072, -0.032];
    let translations = [
        [0.0, 0.0, 0.0],
        [0.68, 0.02, 0.06],
        [-0.56, 0.11, 0.10],
        [0.19, -0.43, 0.13],
        [-0.24, 0.36, 0.07],
    ];
    let truth_poses: Vec<_> = angles
        .iter()
        .zip(translations)
        .map(|(&angle, translation)| RelativePose::new(rotation_y(angle), translation))
        .collect();
    let truth_points: Vec<_> = (0..POINT_COUNT)
        .map(|_| {
            [
                rng.range(-1.25, 1.25),
                rng.range(-0.85, 0.85),
                rng.range(3.8, 6.8),
            ]
        })
        .collect();
    let camera = PinholeCamera::new(760.0, 745.0, 320.0, 240.0);
    let cameras = [camera; CAMERA_COUNT];
    let mut observations = Vec::with_capacity(4 * POINT_COUNT);
    let mut point_offsets = Vec::with_capacity(POINT_COUNT + 1);
    point_offsets.push(0);
    for point in &truth_points {
        for (camera_index, pose) in truth_poses.iter().enumerate() {
            let visible = camera_index < 3 || rng.next_u64() & 1 == 0;
            if !visible {
                continue;
            }
            let mut pixel = project(pose, point, camera);
            pixel[0] += rng.range(-0.025, 0.025);
            pixel[1] += rng.range(-0.025, 0.025);
            observations.push(PixelObservation::new(camera_index, pixel));
        }
        point_offsets.push(observations.len());
    }

    let mut initial_poses = truth_poses;
    for (camera_index, pose) in initial_poses.iter_mut().enumerate().skip(1) {
        pose.rotation = rotation_y(angles[camera_index] + rng.range(-0.012, 0.012));
        for coordinate in &mut pose.translation {
            *coordinate += rng.range(-0.025, 0.025);
        }
    }
    let initial_points = truth_points
        .iter()
        .map(|point| point.map(|coordinate| coordinate + rng.range(-0.035, 0.035)))
        .collect();

    Scene {
        initial_poses,
        initial_points,
        observations,
        point_offsets,
        cameras,
    }
}

fn rotation_y(angle: f64) -> [[f64; 3]; 3] {
    let (sine, cosine) = angle.sin_cos();
    [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
}

fn project(pose: &RelativePose, point: &[f64; 3], camera: PinholeCamera) -> [f64; 2] {
    let camera_point: [f64; 3] = std::array::from_fn(|row| {
        pose.rotation[row][0] * point[0]
            + pose.rotation[row][1] * point[1]
            + pose.rotation[row][2] * point[2]
            + pose.translation[row]
    });
    [
        camera.fx * camera_point[0] / camera_point[2] + camera.cx,
        camera.fy * camera_point[1] / camera_point[2] + camera.cy,
    ]
}
