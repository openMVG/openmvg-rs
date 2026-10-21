// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Recover how a calibrated camera moved between two views, from matches that include wrong ones.
//!
//! You will learn: how to turn pixels into bearings with a calibrated camera, how
//! `estimate_essential` finds the essential matrix among wrong matches with a threshold in
//! normalized image units, and how the decomposition gives the rotation and the direction of
//! motion, but not its length.
//! Data: 100 points in front of two pinhole cameras one unit apart, with 0.5 px noise; 20% of
//! the matches point to random pixels. Seed 11, no files.
//! Run: cargo run --release -p openmvg-geometry --example relative_pose
//! Expected: rotation error below 1.5° and translation direction error below 3°, with at least
//! 90% of the good matches kept; bundle adjustment would refine both further.
//! Next: the `triangulate_and_resect` example continues to 3D points and a third camera; the
//! `openmvg-features` examples produce such matches from images.

use nalgebra::{Point2, Point3, Rotation3, Vector3};
use openmvg_core::camera::{Camera, CameraModel};
use openmvg_core::model::{WorldToCameraPose, rotation_angle_deg};
use openmvg_core::{AlgorithmRng, NormalizedImageThreshold, RunSeed};
use openmvg_geometry::correspondence::BearingCorrespondence;
use openmvg_geometry::essential::estimate_essential;
use openmvg_geometry::robust::RobustOptions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let focal = 800.0;
    let camera = Camera::try_pinhole(focal, focal, 320.0, 240.0)?;
    // The first camera sits at the world origin; the second moved one unit, mostly sideways, and
    // turned 8° toward the scene.
    let first = WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::origin())?;
    let rotation = Rotation3::from_axis_angle(&Vector3::y_axis(), (-8_f64).to_radians());
    let second = WorldToCameraPose::from_rotation_center(rotation, Point3::new(1.0, 0.1, 0.2))?;
    let mut rng = AlgorithmRng::new(RunSeed::from_u64(11));
    let noise = |rng: &mut AlgorithmRng| {
        let (x, y) = (rng.standard_normal_f32(), rng.standard_normal_f32());
        nalgebra::Vector2::new(f64::from(x), f64::from(y)) * 0.5
    };

    // Each match pairs the bearings of one point in both cameras. A calibrated camera turns a
    // pixel into a bearing, so the solver never sees pixels or the focal length.
    let (mut matches, mut is_good) = (Vec::new(), Vec::new());
    for i in 0..100 {
        let unit = |rng: &mut AlgorithmRng| rng.symmetric_unit_f64();
        let world = Point3::new(
            2.0 * unit(&mut rng),
            1.5 * unit(&mut rng),
            6.0 + 2.0 * unit(&mut rng),
        );
        let source = camera.project(&first.project_to_camera(&world).coords)? + noise(&mut rng);
        let good = i % 5 != 0;
        let target = if good {
            camera.project(&second.project_to_camera(&world).coords)? + noise(&mut rng)
        } else {
            Point2::new(640.0 * rng.unit_f64(), 480.0 * rng.unit_f64())
        };
        let (b1, b2) = (camera.bearing(&source)?, camera.bearing(&target)?);
        matches.push(BearingCorrespondence::try_new(
            b1.into_inner(),
            b2.into_inner(),
        )?);
        is_good.push(good);
    }

    // The residual is in normalized image units: pixels divided by the focal length.
    let threshold = NormalizedImageThreshold::try_new(2.0 / focal)?;
    let estimate = estimate_essential(&matches, threshold, &RobustOptions::default())?;
    let kept_good = estimate.inliers().iter().filter(|&&i| is_good[i]).count();
    let good_count = is_good.iter().filter(|&&good| good).count();

    // Four rotation and translation pairs fit the same essential matrix; only one places the
    // points in front of both cameras.
    let inliers: Vec<_> = estimate.inliers().iter().map(|&i| matches[i]).collect();
    let pose = estimate
        .model()
        .decompose_with_cheirality(&inliers)
        .ok_or("no pose passes the cheirality check")?;

    // The truth, from the first camera to the second. The translation is only a direction: two
    // views cannot tell a small motion near the scene from a large one far away.
    let rotation_error =
        rotation_angle_deg(&(pose.rotation_source_to_target().inverse() * rotation));
    let true_direction = (rotation * -Vector3::new(1.0, 0.1, 0.2)).normalize();
    let direction_error = pose
        .translation_source_to_target()
        .angle(&true_direction)
        .to_degrees();

    println!(
        "kept {kept_good} of {good_count} good matches and {} wrong ones",
        estimate.inliers().len() - kept_good
    );
    println!(
        "rotation error {rotation_error:.3}°, translation direction error {direction_error:.2}°"
    );
    assert!(
        kept_good * 10 >= good_count * 9,
        "the robust fit keeps the good matches"
    );
    // The estimate is the best five-point model, unless the linear refit on its inliers scores
    // better; a reconstruction refines it further with bundle adjustment.
    assert!(rotation_error < 1.5 && direction_error < 3.0);
    Ok(())
}
