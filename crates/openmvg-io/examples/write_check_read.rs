// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Store a reconstruction in the OMVG format, check the files, read them back, and export the
//! reconstruction to COLMAP.
//!
//! You will learn: what a write and read round trip keeps (the cameras, poses and points,
//! exactly), how `check` reports whether a dataset conforms, why writing to an existing path
//! fails, and how views that share a camera model share one COLMAP camera.
//! Data: a synthetic scene built in code, four views of one pinhole camera on an arc around 60
//! points; no files.
//! Run: cargo run --release -p openmvg-io --example write_check_read
//! Expected: `check` finds no violation, the poses and points read back equal the originals
//! exactly, and the COLMAP model holds one camera and four images.
//! Next: the "What a round trip keeps" table in the crate documentation; the umbrella crate's
//! `sfm` example runs an engine on real images and exports its result.

use nalgebra::{Point3, Rotation3, Vector3};
use openmvg_core::camera::CameraModel;
use openmvg_io::{
    Error, ReadOptions, WriteOptions, check, read_reconstruction, write_colmap,
    write_reconstruction,
};
use openmvg_sfm::model::ImageMeta;
use openmvg_sfm::{
    Camera, Dataset, IntrinsicId, PoseId, Reconstruction, ReconstructionParts, SfmEngine, ViewId,
    WorldToCameraPose,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Four 640 x 480 views share one pinhole camera.
    let camera = Camera::try_pinhole(500.0, 500.0, 320.0, 240.0)?;
    let dataset = Dataset::builder()
        .image_meta((0..4).map(|view| ImageMeta::external_id(format!("view-{view}"), 640, 480)))
        .camera_models([camera.clone()])
        .shared_intrinsic(IntrinsicId::new(0))
        .build()?;

    // The cameras stand on an arc 6 units from the origin and look at it.
    let poses: Vec<_> = (0..4)
        .map(|view| {
            let angle = (f64::from(view) - 1.5) * 0.25;
            let center = Point3::new(6.0 * angle.sin(), 0.3, -6.0 * angle.cos());
            let look = Rotation3::face_towards(&(Point3::origin() - center), &Vector3::y());
            WorldToCameraPose::from_rotation_center(look.inverse(), center)
        })
        .collect::<Result<_, _>>()?;
    let mut parts = ReconstructionParts::new(
        7,
        SfmEngine::Incremental,
        dataset.camera_models().to_vec(),
        dataset.views().to_vec(),
    );
    for (view, pose) in poses.iter().enumerate() {
        parts = parts.with_pose(PoseId::new(view), pose.clone());
    }
    // A 10 x 6 grid of points near the origin, each seen by every view at its exact pixel.
    for index in 0..60 {
        let (x, y) = (f64::from(index % 10), f64::from(index / 10));
        let point = Point3::new(0.2 * x - 0.9, 0.2 * y - 0.5, 0.0);
        let mut seen = Vec::new();
        for (view, pose) in poses.iter().enumerate() {
            let pixel = camera.project(&pose.project_to_camera(&point).coords)?;
            seen.push((ViewId::new(view), pixel));
        }
        parts = parts.with_point(point, seen);
    }
    let original = Reconstruction::try_from_parts(parts)?;

    // Write the dataset, check it, and read it back. A writer never replaces a dataset: the
    // second write to the same path fails.
    let directory = std::env::temp_dir().join("openmvg-io-write-check-read.omvg");
    let colmap = directory.with_extension("colmap");
    let _ = std::fs::remove_dir_all(&directory);
    write_reconstruction(&original, &directory, &WriteOptions::new())?;
    let violations = check(&directory);
    assert!(violations.is_empty(), "{violations:?}");
    let again = write_reconstruction(&original, &directory, &WriteOptions::new());
    assert!(matches!(again, Err(Error::Exists(_))), "{again:?}");
    let read = read_reconstruction(&directory, &ReadOptions::new())?.reconstruction;

    // The geometry comes back bit for bit; feature indices and errors are renumbered and
    // recomputed, which this check does not look at.
    for (view, record) in original.registered_views() {
        assert_eq!(read.pose(view).map(|read| &read.pose), Some(&record.pose));
    }
    let points = |reconstruction: &Reconstruction| -> Vec<Point3<f64>> {
        let landmarks = reconstruction.landmarks().iter_active();
        landmarks.map(|landmark| landmark.point).collect()
    };
    assert_eq!(points(&read), points(&original));

    // The views share one camera model, so COLMAP gets one camera for four images.
    write_colmap(&read, &colmap)?;
    let cameras = std::fs::read_to_string(colmap.join("sparse/0/cameras.txt"))?;
    let lines: Vec<_> = cameras
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert_eq!(lines.len(), 1, "{lines:?}");

    let (views, count) = (read.registered_view_count(), read.point_count());
    println!("wrote, checked and read back {views} views and {count} points, unchanged");
    println!("a second write fails because the dataset exists");
    println!("COLMAP camera: {}", lines[0]);
    std::fs::remove_dir_all(&directory)?;
    std::fs::remove_dir_all(&colmap)?;
    Ok(())
}
