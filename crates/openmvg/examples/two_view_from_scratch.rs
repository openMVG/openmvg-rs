// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! From two photographs to 3D points, one step at a time: what the front end and the engines do
//! inside, for a single image pair.
//!
//! You will learn: how the layers fit together. `openmvg-features` detects, describes and
//! matches keypoints; `openmvg-core` turns pixels into bearings; `openmvg-geometry` estimates the
//! essential matrix among wrong matches, decomposes it into a relative pose, and triangulates
//! the points. This shows the inside of the pipeline and is not a replacement for it: for real
//! reconstructions use `openmvg::sfm`, which also selects pairs across all images, builds
//! tracks, and refines everything with bundle adjustment.
//! Data: the first two Sceaux Castle images and the `K.txt` file of their camera intrinsics,
//! which `pixi run examples-real` fetches.
//! Run: cargo run --release -p openmvg --example two_view_from_scratch --features sfm-image-io --
//! <images>
//! Expected: on Sceaux Castle, more than 500 matches pass the essential matrix, and more than 90%
//! of them triangulate in front of both cameras, within 4 px of their observations.
//! Next: the crate documentation of `openmvg-sfm` explains the two engines that chain all of
//! this over every image.

use std::collections::BTreeMap;

use nalgebra::{Point2, Point3, Rotation3};
use openmvg::core::NormalizedImageThreshold;
use openmvg::core::camera::CameraModel;
use openmvg::core::model::{ViewId, WorldToCameraPose};
use openmvg::features::extraction::{AlpExtractor, AlpPreset, U8Descriptor};
use openmvg::features::matching::{HnswMatcher, LoweRatioFilter, SquaredL2DistanceU8};
use openmvg::geometry::correspondence::BearingCorrespondence;
use openmvg::geometry::essential::estimate_essential;
use openmvg::geometry::robust::RobustOptions;
use openmvg::geometry::triangulation::{TriangulationConfig, TriangulationMethod, triangulate};
use openmvg::sfm::image::{self, find_calibration_file, load_gray_image_max_dimension};

/// The descriptors of a feature set, as the slices a matcher reads.
fn slices(descriptors: &[U8Descriptor]) -> Vec<&[u8]> {
    descriptors
        .iter()
        .map(|d| d.as_array().as_slice())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let folder = std::env::args_os()
        .nth(1)
        .ok_or("usage: two_view_from_scratch <images>")?;
    let paths = image::discover_dataset_images(folder.as_ref())?;
    let [first_path, second_path, ..] = paths.as_slice() else {
        return Err("the folder needs at least two images".into());
    };
    let k_file = find_calibration_file(folder.as_ref(), first_path).ok_or("no K.txt found")?;
    let camera = image::load_pinhole_intrinsics_matrix(&k_file)?;

    // 1. Detect and describe as the default front end does: ALP at its `High` preset, on images
    // decoded with their longest edge at most 1600 px. Each keypoint maps back to source pixels,
    // where the intrinsics apply.
    let extractor = AlpExtractor::with_preset(AlpPreset::High);
    let mut views = Vec::new();
    for path in [first_path, second_path] {
        let (gray, geometry) = load_gray_image_max_dimension(path, Some(1600))?;
        let features = extractor.extract_u8(&gray)?;
        let to_source = |x, y| geometry.working_to_source(x, y);
        let pixels = features.keypoints().iter().map(|k| to_source(k.x, k.y));
        let pixels = pixels.map(|(x, y)| Point2::new(f64::from(x), f64::from(y)));
        views.push((pixels.collect::<Vec<_>>(), features));
    }
    let ((first, first_features), (second, second_features)) = (&views[0], &views[1]);

    // 2. Match with the approximate nearest-neighbor search of the default front end, and keep
    // the unambiguous matches with Lowe's ratio test on squared distances.
    let mut matcher: HnswMatcher<u8, SquaredL2DistanceU8> = HnswMatcher::new();
    matcher.build(&slices(second_features.descriptors()))?;
    let neighbors = matcher.query_batch(&slices(first_features.descriptors()), 2)?;
    let matches = LoweRatioFilter::new().apply_squared(&neighbors, 0.8);

    // 3. Turn each matched pixel into a bearing with the camera, and estimate the essential
    // matrix; the threshold is in normalized image units, pixels over the focal length.
    let focal = camera
        .focal_lengths()
        .ok_or("the camera has no focal length")?
        .0;
    let mut bearings = Vec::with_capacity(matches.len());
    for m in &matches {
        let from_first = camera.bearing(&first[m.query_index()])?.into_inner();
        let from_second = camera.bearing(&second[m.target_index()])?.into_inner();
        bearings.push(BearingCorrespondence::try_new(from_first, from_second)?);
    }
    let threshold = NormalizedImageThreshold::try_new(2.0 / focal)?;
    let essential = estimate_essential(&bearings, threshold, &RobustOptions::default())?;

    // 4. Decompose into the relative pose: the second camera's rotation and the direction of its
    // motion, at unit distance, since two views cannot measure the scale.
    let inliers: Vec<_> = essential.inliers().iter().map(|&i| bearings[i]).collect();
    let model = essential.model();
    let pose = model
        .decompose_with_cheirality(&inliers)
        .ok_or("no valid pose")?;
    let rotation = *pose.rotation_source_to_target();
    let center = -(rotation.inverse() * pose.translation_source_to_target().into_inner());
    let origin = WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::origin())?;
    let moved = WorldToCameraPose::from_rotation_center(rotation, Point3::from(center))?;
    let poses = BTreeMap::from([(ViewId::new(0), origin), (ViewId::new(1), moved)]);

    // 5. Triangulate every inlier match; a good point lies in front of both cameras and
    // reprojects within a few pixels of its observations.
    let k = camera
        .intrinsic_matrix()
        .ok_or("the camera has no intrinsic matrix")?;
    let intrinsics: BTreeMap<_, _> = poses.keys().map(|&view| (view, k)).collect();
    let (method, config) = (TriangulationMethod::Dlt, TriangulationConfig::default());
    let max_error = 4.0; // pixels
    let mut points = Vec::new();
    for &i in essential.inliers() {
        let m = &matches[i];
        let observations = [
            (ViewId::new(0), first[m.query_index()]),
            (ViewId::new(1), second[m.target_index()]),
        ];
        let Ok(point) = triangulate(method, &observations, &poses, &intrinsics, &config) else {
            continue;
        };
        let in_front = poses
            .values()
            .all(|pose| pose.project_to_camera(&point.point).z > 0.0);
        if in_front && point.reprojection_error.value() < max_error {
            points.push(point.point);
        }
    }

    let (keypoints, angle) = (first.len() + second.len(), rotation.angle().to_degrees());
    println!(
        "{keypoints} keypoints in the two images, {} matches after Lowe's ratio test",
        matches.len()
    );
    println!(
        "{} matches fit the essential matrix; the camera turned {angle:.1}°",
        inliers.len()
    );
    println!(
        "{} points in front of both cameras, within {max_error} px",
        points.len()
    );
    assert!(
        inliers.len() > 500,
        "the pair shares many consistent matches"
    );
    let fraction = points.len() as f64 / inliers.len() as f64;
    assert!(
        fraction > 0.9,
        "most inliers triangulate in front of both cameras"
    );
    Ok(())
}
