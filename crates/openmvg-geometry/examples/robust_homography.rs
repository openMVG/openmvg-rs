// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Estimate a homography from matches that include wrong ones, and see why it must be robust.
//!
//! You will learn: how to build pixel correspondences, how `estimate_homography` separates good
//! matches from wrong ones, how to set its threshold from the noise and the residual it
//! measures, and what a plain least-squares fit does with the same data.
//! Data: 120 points on a plane seen by two pinhole cameras, with 0.5 px noise; 30% of the
//! matches point to random pixels. Seed 7, no files.
//! Run: cargo run --release -p openmvg-geometry --example robust_homography
//! Expected: at least 95% of the good matches kept and no wrong one; transfer error below 1 px
//! for the robust fit and above 10 px for the least-squares fit.
//! Next: the `relative_pose` example recovers how two calibrated cameras moved.

use nalgebra::{Matrix3, Point2, Point3, Rotation3, Vector2, Vector3};
use openmvg_core::camera::{Camera, CameraModel};
use openmvg_core::model::WorldToCameraPose;
use openmvg_core::{AlgorithmRng, PixelThreshold, RunSeed};
use openmvg_geometry::correspondence::PixelCorrespondence;
use openmvg_geometry::homography::{DltSolver, HomographyConfig, estimate_homography};
use openmvg_geometry::normalization::NormalizedCorrespondences;
use openmvg_geometry::robust::RobustOptions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let camera = Camera::try_pinhole(800.0, 800.0, 320.0, 240.0)?;
    let first = WorldToCameraPose::from_rotation_center(
        Rotation3::identity(),
        Point3::new(0.0, 0.0, -4.0),
    )?;
    let turn = Rotation3::from_axis_angle(&Vector3::y_axis(), 15_f64.to_radians())
        * Rotation3::from_axis_angle(&Vector3::x_axis(), 5_f64.to_radians());
    let second = WorldToCameraPose::from_rotation_center(turn, Point3::new(1.2, 0.2, -3.6))?;
    let mut rng = AlgorithmRng::new(RunSeed::from_u64(7));
    // Gaussian pixel noise with a 0.5 px standard deviation.
    let noise = |rng: &mut AlgorithmRng| {
        let (x, y) = (rng.standard_normal_f32(), rng.standard_normal_f32());
        Vector2::new(f64::from(x), f64::from(y)) * 0.5
    };

    // Points on the plane Z = 0, as both cameras see them. Three matches in every ten are
    // replaced by a random pixel: those are the wrong matches.
    let (mut matches, mut clean_targets, mut is_good) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..120 {
        let world = Point3::new(
            f64::from(i % 12) * 0.25 - 1.4,
            f64::from(i / 12) * 0.2 - 0.9,
            0.0,
        );
        let source = camera.project(&first.project_to_camera(&world).coords)?;
        let target = camera.project(&second.project_to_camera(&world).coords)?;
        let good = i % 10 >= 3;
        let observed = if good {
            target + noise(&mut rng)
        } else {
            Point2::new(640.0 * rng.unit_f64(), 480.0 * rng.unit_f64())
        };
        matches.push(PixelCorrespondence::new(source + noise(&mut rng), observed));
        clean_targets.push(target);
        is_good.push(good);
    }

    // Robust estimation: RANSAC draws 4-match samples and keeps the model most matches agree
    // with, then refits it on those inliers. The residual is the symmetric transfer error, which
    // adds the errors of both directions, so the threshold sits well above the 0.5 px noise.
    let threshold = PixelThreshold::try_new(3.0)?;
    let robust = estimate_homography(&matches, threshold, &RobustOptions::default())?;
    let kept_good = robust.inliers().iter().filter(|&&i| is_good[i]).count();
    let kept_wrong = robust.inliers().len() - kept_good;
    let good_count = is_good.iter().filter(|&&good| good).count();

    // A least-squares fit on every match lets the wrong ones pull the model away.
    let all = NormalizedCorrespondences::from_pixels(&matches);
    let naive = DltSolver::solve(&all, &HomographyConfig::default())?;
    let naive = naive
        .first()
        .ok_or("the least-squares fit found no homography")?;

    // Transfer error: how far each good point lands from where it really is in the second image.
    let transfer = |h: &Matrix3<f64>| {
        let errors = (0..matches.len()).filter(|&i| is_good[i]).map(|i| {
            let q = h * matches[i].point1.to_homogeneous();
            (Point2::new(q.x / q.z, q.y / q.z) - clean_targets[i]).norm()
        });
        errors.sum::<f64>() / good_count as f64
    };
    let (robust_error, naive_error) = (transfer(robust.model().matrix()), transfer(naive.matrix()));

    println!("kept {kept_good} of {good_count} good matches and {kept_wrong} wrong ones");
    println!("mean transfer error: robust {robust_error:.2} px, least squares {naive_error:.1} px");
    assert!(
        kept_good * 100 >= good_count * 95,
        "the robust fit keeps the good matches"
    );
    assert_eq!(kept_wrong, 0, "the robust fit rejects the wrong matches");
    assert!(robust_error < 1.0 && naive_error > 10.0);
    Ok(())
}
