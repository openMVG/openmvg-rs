// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bundle adjustment with wrong matches: a robust loss keeps them from pulling the solution, a
//! gauge fixes the frame and the scale, and the report says whether to trust the result.
//!
//! You will learn: how a Cauchy loss with a scale near the noise sets wrong matches aside where
//! plain least squares spreads their error over every camera, how `Gauge::FixedBaseline` fixes
//! the frame and the scale while every pose is free, and how to accept a result: the audit, the
//! status, the observability, the inlier count, and the residual of each observation.
//! Data: four cameras on a 60° arc see a block of 40 points through one fixed pinhole camera;
//! pixels carry 0.5 px of noise (seed 5), 4 observations are wrong matches 40 px off, and the
//! last three cameras and every point start a few centimeters off.
//! Run: cargo run --release -p openmvg-ba --example robust_loss_and_gauge
//! Expected: with the Cauchy loss exactly the 4 wrong matches stay beyond the loss scale and the
//! cameras end within 5 cm of the truth, at least five times closer than with plain least
//! squares.
//! Next: the "Accepting a result" section of the crate documentation; `openmvg-sfm` runs bundle
//! adjustment inside every reconstruction.

use nalgebra::{Point3, Rotation3, Vector2, Vector3};
use openmvg_ba::{
    BundleAdjuster, BundleIntrinsic, BundleObservation, BundlePoint, BundlePose, BundleProblem,
    Gauge, IntrinsicVariables, NativeBundleAdjuster, Observability, PoseVariables, ResidualScale,
    RobustLoss, Solution, SolveConfig, SolveStatus,
};
use openmvg_core::camera::{Camera, CameraModel};
use openmvg_core::model::{CameraIntrinsics, WorldToCameraPose};
use openmvg_core::{AlgorithmRng, PixelThreshold, RunSeed};

const WRONG: [usize; 4] = [3, 57, 101, 150];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let camera = Camera::try_pinhole(800.0, 800.0, 320.0, 240.0)?;
    let intrinsic = CameraIntrinsics::try_new(camera.clone(), 640, 480)?.with_fixed(true);
    // The cameras stand on an arc of radius 6, 60 degrees wide, and look at the middle of a
    // block of points 2 units deep.
    let target = Point3::new(0.0, 0.0, 6.0);
    let truth = (0..4)
        .map(|index| {
            let angle = 0.35 * (f64::from(index) - 1.5);
            let center = target + 6.0 * Vector3::new(angle.sin(), 0.1, -angle.cos());
            let look = Rotation3::face_towards(&(target - center), &Vector3::y());
            WorldToCameraPose::from_rotation_center(look.inverse(), center)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let points: Vec<_> = (0..40)
        .map(|index| {
            let (row, column) = (f64::from(index / 8), f64::from(index % 8));
            let depth = f64::from(index % 5) * 0.5 - 1.0;
            target + Vector3::new(0.35 * column - 1.2, 0.4 * row - 0.8, depth)
        })
        .collect();

    // Noisy pixels, and four wrong matches 40 px off.
    let mut rng = AlgorithmRng::new(RunSeed::from_u64(5));
    let mut observations = Vec::new();
    for (pose, camera_pose) in truth.iter().enumerate() {
        for (point, position) in points.iter().enumerate() {
            let pixel = camera.project(&camera_pose.project_to_camera(position).coords)?;
            let noise = Vector2::new(rng.standard_normal_f32(), rng.standard_normal_f32());
            let mut pixel = pixel + 0.5 * noise.cast::<f64>();
            // The epipolar lines run nearly horizontally here, so the wrong matches move
            // vertically; one moved along its line could fit another view, as the crate
            // documentation warns.
            if WRONG.contains(&observations.len()) {
                pixel.y += 40.0;
            }
            observations.push(BundleObservation::try_new(pose, 0, point, pixel)?);
        }
    }

    // Every pose is free and starts off, except pose 0. The gauge holds pose 0 and the distance
    // from it to pose 1, so the result has the frame and the scale of the truth; `Gauge::Free`
    // would keep those of the start.
    let mut poses = Vec::new();
    for (index, pose) in truth.iter().enumerate() {
        let shift = if index == 0 { 0.0 } else { 0.05 };
        let center = pose.center() + shift * Vector3::new(0.6, -0.4, 0.8);
        let start = WorldToCameraPose::from_rotation_center(*pose.rotation(), center)?;
        poses.push(BundlePose::new(start, PoseVariables::FREE));
    }
    let starts = points
        .iter()
        .map(|p| BundlePoint::try_new(p + Vector3::new(0.04, 0.03, -0.05)));
    let starts = starts.collect::<Result<Vec<_>, _>>()?;
    let lens = BundleIntrinsic::try_new(intrinsic, IntrinsicVariables::FIXED)?;
    let distance = (truth[1].center() - truth[0].center()).norm();
    let gauge = Gauge::FixedBaseline {
        first: 0,
        second: 1,
        distance,
    };
    let solve = |loss| -> Result<_, Box<dyn std::error::Error>> {
        let problem = BundleProblem::builder(
            poses.clone(),
            vec![lens.clone()],
            starts.clone(),
            observations.clone(),
        );
        let problem = problem.with_gauge(gauge).with_robust_loss(loss).build()?;
        let solution = NativeBundleAdjuster::new().solve(&problem, SolveConfig::default())?;
        Ok((problem, solution))
    };
    let worst_mm = |solution: &Solution| {
        let errors = solution.poses().iter().zip(&truth);
        let errors = errors.map(|(pose, truth)| (pose.center() - truth.center()).norm());
        1e3 * errors.fold(0.0, f64::max)
    };

    // Plain least squares gives every wrong match its full weight. A Cauchy loss with a 2 px
    // scale, four times the noise, sets them aside.
    let (_, plain) = solve(RobustLoss::Squared)?;
    let scale = 2.0;
    let loss = RobustLoss::Cauchy(ResidualScale::Pixels(PixelThreshold::try_new(scale)?));
    let (problem, robust) = solve(loss)?;

    // Accept the robust result only after its checks.
    robust.audit_for_commit(&problem, &SolveConfig::default())?;
    let report = robust.report();
    assert!(matches!(report.status(), SolveStatus::Converged(_)));
    let observability = report.diagnostics().observability();
    assert_eq!(observability, Observability::FullRank);
    let outliers = report.observation_count().zip(report.inlier_count());
    assert_eq!(
        outliers.map(|(all, inliers)| all - inliers),
        Some(WRONG.len())
    );
    let residuals = robust.observation_residuals(&problem)?;
    let beyond: Vec<_> = (0..residuals.len())
        .filter(|&index| residuals[index] > Some(scale))
        .collect();
    assert_eq!(
        beyond, WRONG,
        "exactly the wrong matches stay beyond the scale"
    );

    let (plain_mm, robust_mm) = (worst_mm(&plain), worst_mm(&robust));
    println!("worst camera error: least squares {plain_mm:.1} mm, Cauchy {robust_mm:.1} mm");
    println!("observations beyond the {scale} px scale: {beyond:?}, the wrong matches");
    assert!(robust_mm < 50.0, "the robust cameras land near the truth");
    assert!(plain_mm > 5.0 * robust_mm, "least squares is pulled off");
    Ok(())
}
