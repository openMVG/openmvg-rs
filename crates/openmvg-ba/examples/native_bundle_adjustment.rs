// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Refine perturbed cameras and points with the native bundle adjuster.
//!
//! You will learn: how to build a `BundleProblem` from poses, an intrinsic, points and
//! observations, how holding two exact poses fixes the gauge, and how to read the solve report.
//! The two held poses are exact here; with estimated poses, the "Choosing the gauge" section of
//! the crate documentation advises one fixed pose at most, or a `Gauge::FixedBaseline`.
//! Data: four cameras on a line see a 5 × 5 grid of points through one fixed pinhole camera; the
//! last two poses and every point start a few centimeters off, against exact observations.
//! Run: cargo run --release -p openmvg-ba --example native_bundle_adjustment
//! Expected: the solve converges in a few iterations, and the reprojection RMSE falls from
//! several pixels to below 1e-6 px.
//! Next: the `robust_loss_and_gauge` example adds wrong matches and a robust loss.

use nalgebra::{Point2, Point3, Rotation3};
use openmvg_ba::{
    BundleAdjuster, BundleIntrinsic, BundleObservation, BundlePoint, BundlePose, BundleProblem,
    IntrinsicVariables, NativeBundleAdjuster, PoseVariables, SolveConfig,
};
use openmvg_core::{
    camera::{Camera, CameraModel},
    model::{CameraIntrinsics, WorldToCameraPose},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let camera = Camera::try_pinhole(800.0, 800.0, 320.0, 240.0)?;
    let intrinsic = CameraIntrinsics::try_new(camera.clone(), 640, 480)?.with_fixed(true);
    let poses = (0..4)
        .map(|index| {
            let center = Point3::new(0.4 * f64::from(index), 0.0, 0.0);
            WorldToCameraPose::from_rotation_center(Rotation3::identity(), center)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let points = (0..25)
        .map(|index| {
            let (row, column) = (f64::from(index / 5), f64::from(index % 5));
            Point3::new(column - 1.4, row - 2.0, 6.0 + 0.3 * (row - column))
        })
        .collect::<Vec<_>>();

    let mut observations = Vec::new();
    for (camera_index, pose) in poses.iter().enumerate() {
        for (point_index, point) in points.iter().enumerate() {
            let pixel: Point2<f64> = camera.project(&pose.project_to_camera(point).coords)?;
            observations.push(BundleObservation::try_new(
                camera_index,
                0,
                point_index,
                pixel,
            )?);
        }
    }

    let pose_blocks = poses
        .iter()
        .enumerate()
        .map(|(index, pose)| {
            // The first two poses are exact, so holding both fixes the frame and the scale
            // without imposing any error.
            if index < 2 {
                return Ok(BundlePose::new(pose.clone(), PoseVariables::FIXED));
            }
            let center = pose.center() + nalgebra::Vector3::new(0.02, -0.01, 0.03);
            let perturbed = WorldToCameraPose::from_rotation_center(*pose.rotation(), center)?;
            Ok(BundlePose::new(perturbed, PoseVariables::FREE))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let initial_points = points
        .iter()
        .map(|point| BundlePoint::try_new(point + nalgebra::Vector3::new(0.05, 0.02, -0.04)))
        .collect::<Result<Vec<_>, _>>()?;
    let problem = BundleProblem::builder(
        pose_blocks,
        vec![BundleIntrinsic::try_new(
            intrinsic,
            IntrinsicVariables::FIXED,
        )?],
        initial_points,
        observations,
    )
    .build()?;

    let solution = NativeBundleAdjuster::new().solve(&problem, SolveConfig::default())?;
    let report = solution.report();
    assert!(report.initial_rmse() > 1.0, "the start is perturbed");
    assert!(report.final_rmse() < 1e-6, "exact observations are matched");
    println!(
        "status {:?} after {} iterations: rmse {:.3e} px -> {:.3e} px",
        report.status(),
        report.iteration_count(),
        report.initial_rmse(),
        report.final_rmse(),
    );
    Ok(())
}
