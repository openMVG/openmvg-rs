// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic sparse BA problem generator shared by the sparse solver tests.

use super::super::*;
use crate::kernel::{PinholeCamera, multi_view::PixelObservation};
use crate::numerics::manifold;

pub(crate) struct Fixture {
    pub(crate) poses: Vec<RelativePose>,
    pub(crate) points: Vec<[f64; 3]>,
    pub(crate) observations: Vec<PixelObservation>,
    pub(crate) offsets: Vec<usize>,
    pub(crate) cameras: Vec<PinholeCamera>,
}

pub(crate) fn fixture(camera_count: usize, point_count: usize, track_length: usize) -> Fixture {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let truth_poses: Vec<_> = (0..camera_count)
        .map(|camera| {
            if camera == 0 {
                RelativePose::new(identity, [0.0; 3])
            } else {
                let phase = 0.17 * camera as f64;
                RelativePose::new(
                    manifold::exp_so3(&[0.003 * phase, -0.012 * phase, 0.002 * phase]),
                    [
                        0.65 * phase.cos(),
                        0.65 * phase.sin(),
                        0.015 * camera as f64,
                    ],
                )
            }
        })
        .collect();
    let poses = truth_poses
        .iter()
        .enumerate()
        .map(|(camera, pose)| {
            if camera == 0 {
                *pose
            } else {
                RelativePose::new(
                    manifold::mat_mul(
                        &manifold::exp_so3(&[0.001, -0.0015, 0.0007]),
                        &pose.rotation,
                    ),
                    [
                        pose.translation[0] + 0.003,
                        pose.translation[1] - 0.002,
                        pose.translation[2] + 0.001,
                    ],
                )
            }
        })
        .collect();
    let truth_points: Vec<_> = (0..point_count)
        .map(|index| {
            [
                0.04 * ((index % 32) as f64 - 15.5),
                0.04 * (((index / 32) % 32) as f64 - 15.5),
                4.0 + 0.12 * (0.13 * index as f64).sin(),
            ]
        })
        .collect();
    let points = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            [
                point[0] + 0.004 * (0.31 * index as f64).sin(),
                point[1] + 0.004 * (0.27 * index as f64).cos(),
                point[2] + 0.006 * (0.19 * index as f64).sin(),
            ]
        })
        .collect();
    let cameras = vec![PinholeCamera::new(500.0, 505.0, 320.0, 240.0); camera_count];
    let track_length = track_length.clamp(2, camera_count);
    let mut observations = Vec::with_capacity(point_count * track_length);
    let mut offsets = Vec::with_capacity(point_count + 1);
    offsets.push(0);
    for (point_index, point) in truth_points.iter().enumerate() {
        observations.push(PixelObservation::new(0, project(cameras[0], point)));
        for track_camera in 1..track_length {
            let camera = 1 + (point_index + track_camera - 1) % (camera_count - 1);
            observations.push(PixelObservation::new(
                camera,
                project(cameras[camera], &transform(&truth_poses[camera], point)),
            ));
        }
        offsets.push(observations.len());
    }
    Fixture {
        poses,
        points,
        observations,
        offsets,
        cameras,
    }
}

fn transform(pose: &RelativePose, point: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        pose.rotation[row][0] * point[0]
            + pose.rotation[row][1] * point[1]
            + pose.rotation[row][2] * point[2]
            + pose.translation[row]
    })
}

fn project(camera: PinholeCamera, point: &[f64; 3]) -> [f64; 2] {
    [
        camera.fx * point[0] / point[2] + camera.cx,
        camera.fy * point[1] / point[2] + camera.cy,
    ]
}

pub(crate) fn relative_norm_error(value: &[f64], reference: &[f64]) -> f64 {
    let error = value
        .iter()
        .zip(reference)
        .map(|(value, reference)| (value - reference).powi(2))
        .sum::<f64>()
        .sqrt();
    error / optimizer::norm(reference).max(f64::MIN_POSITIVE)
}
