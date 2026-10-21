// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Project world points into a camera and recover the rays that see them.
//!
//! You will learn: the camera frame and the pixel convention of OpenMVG-rs, a world-to-camera
//! pose built from a rotation and a camera center, how lens distortion moves a pixel, and how a
//! pixel turns back into a bearing.
//! Data: one camera 5 units behind the world origin, turned 10° about its vertical axis, and
//! three world points; no files.
//! Run: cargo run --release -p openmvg-core --example camera_projection
//! Expected: every pixel returns to its ray within 1e-9; distortion moves the three pixels by
//! about 0.5 to 8 px and leaves the principal point in place.
//! Next: the `robust_homography` example of `openmvg-geometry` fits a model to matched pixels.

use nalgebra::{Point3, Rotation3, Vector3};
use openmvg_core::camera::{Camera, CameraModel, RadialDistortion, TangentialDistortion};
use openmvg_core::model::WorldToCameraPose;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A pinhole camera with an 800 px focal length and its principal point at pixel (320, 240).
    let pinhole = Camera::try_pinhole(800.0, 800.0, 320.0, 240.0)?;
    // The same camera seen through a lens with barrel distortion (negative k1).
    let radial = RadialDistortion::K1K2K3(-0.2, 0.05, 0.0);
    let lens = Camera::try_brown_conrady(
        800.0,
        800.0,
        320.0,
        240.0,
        radial,
        TangentialDistortion::P1P2(0.0, 0.0),
    )?;

    // A pose stores the world-to-camera rotation R and the camera center C, in world coordinates.
    let rotation = Rotation3::from_axis_angle(&Vector3::y_axis(), 10_f64.to_radians());
    let center = Point3::new(0.0, 0.0, -5.0);
    let pose = WorldToCameraPose::from_rotation_center(rotation, center)?;
    // The translation follows from them, t = −R C, so that X_camera = R X_world + t.
    assert!((pose.translation() + rotation * center.coords).norm() < 1e-12);

    let points = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, -0.5, 1.0),
        Point3::new(-1.5, 1.0, 2.0),
    ];
    for world in points {
        // The camera frame has X to the right, Y down and Z forward, into the scene.
        let in_camera = pose.project_to_camera(&world);
        assert!(in_camera.z > 0.0, "the point is in front of the camera");
        let pixel = pinhole.project(&in_camera.coords)?;
        let distorted = lens.project(&in_camera.coords)?;

        // A bearing is the unit ray toward the point, in the camera frame. Each camera inverts its
        // own model, distortion included, so both pixels lead back to the same ray.
        let ray = in_camera.coords.normalize();
        assert!((pinhole.bearing(&pixel)?.into_inner() - ray).norm() < 1e-9);
        assert!((lens.bearing(&distorted)?.into_inner() - ray).norm() < 1e-9);

        let shift = (distorted - pixel).norm();
        assert!(shift > 0.1, "distortion moves every off-axis pixel");
        println!(
            "world ({:4.1}, {:4.1}, {:4.1}) -> pixel ({:5.1}, {:5.1}); the lens moves it by {:.2} px",
            world.x, world.y, world.z, pixel.x, pixel.y, shift
        );
    }

    // A point on the optical axis projects onto the principal point, with or without distortion.
    let on_axis = center + rotation.inverse() * Vector3::new(0.0, 0.0, 3.0);
    let in_camera = pose.project_to_camera(&on_axis).coords;
    for camera in [&pinhole, &lens] {
        let pixel = camera.project(&in_camera)?;
        assert!((pixel.x - 320.0).abs() < 1e-9 && (pixel.y - 240.0).abs() < 1e-9);
    }
    println!("a point on the optical axis lands on the principal point (320, 240) in both cameras");
    Ok(())
}
