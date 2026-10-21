// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! From two calibrated views to 3D points, then to the pose of a third camera.
//!
//! You will learn: how to turn the relative pose of `estimate_essential` into the two
//! world-to-camera poses that `triangulate` takes, why the points come out in baselines, and how
//! `estimate_absolute_pose` places a third camera from 2D–3D matches that include wrong ones, with
//! a looser threshold because triangulated points carry their own error.
//! Data: 120 points in front of three pinhole cameras, with 0.5 px noise; one match in five
//! points to a random pixel, in the two views and in the third. Seed 7, no files.
//! Run: cargo run --release -p openmvg-geometry --example triangulate_and_resect
//! Expected: the median point lands within 5% of its distance, since the relative pose's own
//! error carries into every point, and the third camera within 0.5° and 0.03 baselines of the
//! truth.
//! Next: `openmvg-sfm` chains these steps over every image and refines the result with bundle
//! adjustment.

use std::collections::BTreeMap;

use nalgebra::{Point2, Point3, Rotation3, Vector2, Vector3};
use openmvg_core::RunSeed;
use openmvg_core::camera::{Camera, CameraModel};
use openmvg_core::model::{ViewId, WorldToCameraPose, rotation_angle_deg};
use openmvg_core::{AlgorithmRng, AngularThreshold, NormalizedImageThreshold, PixelThreshold};
use openmvg_geometry::correspondence::BearingCorrespondence;
use openmvg_geometry::essential::estimate_essential;
use openmvg_geometry::pose::estimate_absolute_pose;
use openmvg_geometry::pose::{AbsolutePoseOptions, BearingToPoint, PoseRefinement};
use openmvg_geometry::robust::RobustOptions;
use openmvg_geometry::triangulation::{TriangulationConfig, TriangulationMethod, triangulate};

type Error = Box<dyn std::error::Error>;

/// Where `pose` sees `point`, with 0.5 px of noise, or a random pixel for a wrong match.
fn observe(
    camera: &Camera,
    pose: &WorldToCameraPose,
    point: &Point3<f64>,
    good: bool,
    rng: &mut AlgorithmRng,
) -> Result<Point2<f64>, Error> {
    let noise = Vector2::new(rng.standard_normal_f32(), rng.standard_normal_f32()).cast::<f64>();
    let random = Point2::new(640.0 * rng.unit_f64(), 480.0 * rng.unit_f64());
    let seen = camera.project(&pose.project_to_camera(point).coords)?;
    Ok(if good { seen + 0.5 * noise } else { random })
}

fn main() -> Result<(), Error> {
    let focal = 800.0;
    let camera = Camera::try_pinhole(focal, focal, 320.0, 240.0)?;
    let turn = |degrees: f64| Rotation3::from_axis_angle(&Vector3::y_axis(), degrees.to_radians());
    let truth = [
        WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::origin())?,
        WorldToCameraPose::from_rotation_center(turn(-8.0), Point3::new(1.0, 0.1, 0.2))?,
        WorldToCameraPose::from_rotation_center(turn(6.0), Point3::new(-0.8, 0.2, 0.1))?,
    ];
    let mut rng = AlgorithmRng::new(RunSeed::from_u64(7));
    let points: Vec<_> = (0..120)
        .map(|_| [2.0, 1.5, 2.0].map(|size| size * rng.symmetric_unit_f64()))
        .map(|[x, y, z]| Point3::new(x, y, 6.0 + z))
        .collect();

    // 1. Relative pose of the first two views, from matched bearings.
    let (mut pixels, mut matches) = (Vec::new(), Vec::new());
    for (index, point) in points.iter().enumerate() {
        let a = observe(&camera, &truth[0], point, true, &mut rng)?;
        let b = observe(&camera, &truth[1], point, index % 5 != 0, &mut rng)?;
        let (from, to) = (
            camera.bearing(&a)?.into_inner(),
            camera.bearing(&b)?.into_inner(),
        );
        matches.push(BearingCorrespondence::try_new(from, to)?);
        pixels.push((a, b));
    }
    let threshold = NormalizedImageThreshold::try_new(2.0 / focal)?;
    let essential = estimate_essential(&matches, threshold, &RobustOptions::default())?;
    let inliers: Vec<_> = essential.inliers().iter().map(|&i| matches[i]).collect();
    let pose = essential.model().decompose_with_cheirality(&inliers);
    let pose = pose.ok_or("no pose puts the points in front of both cameras")?;

    // 2. The two poses: the first camera at the origin, the second one baseline away, at
    // C₂ = −Rᵀ t. Every length below is measured in baselines.
    let center = Point3::from(pose.baseline_direction_in_source().into_inner());
    let rotation = *pose.rotation_source_to_target();
    let second = WorldToCameraPose::from_rotation_center(rotation, center)?;
    let poses = BTreeMap::from([(ViewId::new(0), truth[0].clone()), (ViewId::new(1), second)]);
    let k = camera.intrinsic_matrix().ok_or("no matrix")?;
    let intrinsics = BTreeMap::from([(ViewId::new(0), k), (ViewId::new(1), k)]);

    // 3. Triangulate every inlier; with the true baseline, the points come back to scale.
    // Bundle adjustment would then remove the error that the relative pose carries into them.
    let (baseline, config) = (
        truth[1].center().coords.norm(),
        TriangulationConfig::default(),
    );
    let (mut triangulated, mut errors) = (Vec::new(), Vec::new());
    for &index in essential.inliers() {
        let seen = [
            (ViewId::new(0), pixels[index].0),
            (ViewId::new(1), pixels[index].1),
        ];
        let method = TriangulationMethod::Dlt;
        let Ok(estimate) = triangulate(method, &seen, &poses, &intrinsics, &config) else {
            continue;
        };
        let truth = points[index];
        if index % 5 != 0 {
            errors.push((estimate.point * baseline - truth).norm() / truth.coords.norm());
        }
        triangulated.push((index, estimate.point));
    }
    errors.sort_by(f64::total_cmp);
    let median = errors[errors.len() / 2];

    // 4. The third camera from 2D–3D matches, a fifth of them wrong. The points carry their
    // own error, so the threshold is looser than for exact points: 4 px instead of 2.
    let mut data = Vec::new();
    for (count, &(index, point)) in triangulated.iter().enumerate() {
        let seen = observe(&camera, &truth[2], &points[index], count % 5 != 0, &mut rng)?;
        data.push(BearingToPoint::try_new(point, camera.bearing(&seen)?)?);
    }
    let threshold = AngularThreshold::from_pixels(PixelThreshold::try_new(4.0)?, focal)?;
    let options = AbsolutePoseOptions::default().with_refinement(PoseRefinement::Perspective);
    let third = estimate_absolute_pose(&data, threshold, &options)?;
    let rotation_error =
        rotation_angle_deg(&(third.model().rotation() * truth[2].rotation().inverse()));
    let center_error = (third.model().center() - truth[2].center() / baseline).norm();

    let (count, kept) = (triangulated.len(), third.inliers().len());
    println!(
        "{count} points triangulated, median error {:.2}%",
        100.0 * median
    );
    println!("third camera: {kept} of {} matches kept", data.len());
    println!("rotation error {rotation_error:.3}°, center error {center_error:.4} baselines");
    assert!(median < 0.05, "the points land near the truth");
    assert!(rotation_error < 0.5, "the third camera turns as the truth");
    assert!(
        center_error < 0.03,
        "the third camera stands near the truth"
    );
    Ok(())
}
