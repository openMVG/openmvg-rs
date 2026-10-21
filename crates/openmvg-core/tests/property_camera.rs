// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property-based tests for camera models.
//!
//! Tests the fundamental invariant: project(unproject(pixel)) ≈ pixel
//! and unproject(project(point3d)) is collinear with point3d.

use nalgebra::{Point2, Vector3};
use openmvg_core::{
    BrownConradyCamera, CameraModel, FisheyeCamera, FisheyeDistortion, PinholeCamera,
    RadialDistortion, SphericalCamera, TangentialDistortion,
};
use proptest::prelude::*;

// ─── Strategies ───────────────────────────────────────────────────────

fn valid_focal() -> impl Strategy<Value = f64> {
    100.0..5000.0_f64
}

fn valid_principal_x() -> impl Strategy<Value = f64> {
    100.0..2000.0_f64
}

fn valid_principal_y() -> impl Strategy<Value = f64> {
    100.0..1500.0_f64
}

/// A 3D point that is in front of the camera (positive Z) and not too far off-axis
fn point_in_front() -> impl Strategy<Value = Vector3<f64>> {
    (-2.0..2.0_f64, -2.0..2.0_f64, 1.0..20.0_f64).prop_map(|(x, y, z)| Vector3::new(x, y, z))
}

/// A point inside a moderate field of view where the generated Brown-Conrady
/// polynomials remain one-to-one.
fn point_in_injective_fov() -> impl Strategy<Value = Vector3<f64>> {
    (-0.8..0.8_f64, -0.8..0.8_f64, 1.0..20.0_f64)
        .prop_map(|(x, y, z)| Vector3::new(x * z, y * z, z))
}

/// Small radial distortion coefficients
fn small_radial() -> impl Strategy<Value = RadialDistortion> {
    (-0.05..0.05_f64, -0.01..0.01_f64, -0.001..0.001_f64)
        .prop_map(|(k1, k2, k3)| RadialDistortion::K1K2K3(k1, k2, k3))
}

/// Small tangential distortion coefficients
fn small_tangential() -> impl Strategy<Value = TangentialDistortion> {
    (-0.005..0.005_f64, -0.005..0.005_f64).prop_map(|(p1, p2)| TangentialDistortion::P1P2(p1, p2))
}

/// Small fisheye distortion coefficients
fn small_fisheye() -> impl Strategy<Value = FisheyeDistortion> {
    (
        -0.05..0.05_f64,
        -0.01..0.01_f64,
        -0.005..0.005_f64,
        -0.001..0.001_f64,
    )
        .prop_map(|(k1, k2, k3, k4)| FisheyeDistortion::K1K2K3K4(k1, k2, k3, k4))
}

// ─── Pinhole Camera ──────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// project(point3d) should produce finite pixel coordinates for points in front of camera
    #[test]
    fn pinhole_project_is_finite(
        f in valid_focal(),
        cx in valid_principal_x(),
        cy in valid_principal_y(),
        point in point_in_front(),
    ) {
        let cam = PinholeCamera::try_with_focal(f, cx, cy).unwrap();
        let pixel = cam.project(&point).unwrap();
        prop_assert!(pixel.x.is_finite(), "pixel.x not finite: {}", pixel.x);
        prop_assert!(pixel.y.is_finite(), "pixel.y not finite: {}", pixel.y);
    }

    /// unproject(project(point3d)) should be collinear with point3d
    /// i.e., the bearing direction should match
    #[test]
    fn pinhole_unproject_project_collinear(
        f in valid_focal(),
        cx in valid_principal_x(),
        cy in valid_principal_y(),
        point in point_in_front(),
    ) {
        let cam = PinholeCamera::try_with_focal(f, cx, cy).unwrap();
        let pixel = cam.project(&point).unwrap();
        let bearing = cam.unproject(&pixel).unwrap();

        // Normalize both to unit vectors and compare
        let expected_dir = point.normalize();
        let actual_dir = bearing.normalize();

        let dot = expected_dir.dot(&actual_dir);
        prop_assert!(
            (dot - 1.0).abs() < 1e-10,
            "Bearing not collinear: dot={}, expected_dir={:?}, actual_dir={:?}",
            dot, expected_dir, actual_dir
        );
    }

    /// project(unproject(pixel)) ≈ pixel (round-trip)
    #[test]
    fn pinhole_project_unproject_roundtrip(
        f in valid_focal(),
        cx in valid_principal_x(),
        cy in valid_principal_y(),
    ) {
        let cam = PinholeCamera::try_with_focal(f, cx, cy).unwrap();
        // Use a pixel near the center for stability
        let pixel = Point2::new(cx + 50.0, cy + 30.0);
        let bearing = cam.unproject(&pixel).unwrap();
        // Project the bearing back (at some depth)
        let point3d = bearing * 5.0; // arbitrary depth
        let reprojected = cam.project(&point3d).unwrap();

        let dx = (reprojected.x - pixel.x).abs();
        let dy = (reprojected.y - pixel.y).abs();
        prop_assert!(
            dx < 1e-10 && dy < 1e-10,
            "Round-trip error: dx={}, dy={}, pixel={:?}, reprojected={:?}",
            dx, dy, pixel, reprojected
        );
    }

    /// Double focal pinhole: fx != fy should still round-trip
    #[test]
    fn pinhole_double_focal_roundtrip(
        fx in valid_focal(),
        fy in valid_focal(),
        cx in valid_principal_x(),
        cy in valid_principal_y(),
        point in point_in_front(),
    ) {
        let cam = PinholeCamera::try_new(fx, fy, cx, cy).unwrap();
        let pixel = cam.project(&point).unwrap();
        let bearing = cam.unproject(&pixel).unwrap();

        let expected_dir = point.normalize();
        let actual_dir = bearing.normalize();
        let dot = expected_dir.dot(&actual_dir);

        prop_assert!(
            (dot - 1.0).abs() < 1e-10,
            "Double focal bearing not collinear: dot={}", dot
        );
    }
}

// ─── Brown-Conrady Camera ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Brown-Conrady project should produce finite results inside the model's one-to-one field of view
    #[test]
    fn brown_conrady_project_is_finite(
        f in valid_focal(),
        cx in valid_principal_x(),
        cy in valid_principal_y(),
        radial in small_radial(),
        tangential in small_tangential(),
        point in point_in_injective_fov(),
    ) {
        let cam = BrownConradyCamera::try_with_focal(f, cx, cy, radial, tangential).unwrap();
        let pixel = cam.project(&point).unwrap();
        prop_assert!(pixel.x.is_finite(), "pixel.x not finite");
        prop_assert!(pixel.y.is_finite(), "pixel.y not finite");
    }

    /// Brown-Conrady unproject(project(point)) should be approximately collinear
    #[test]
    fn brown_conrady_roundtrip_collinear(
        f in valid_focal(),
        cx in valid_principal_x(),
        cy in valid_principal_y(),
        radial in small_radial(),
        tangential in small_tangential(),
        point in point_in_injective_fov(),
    ) {
        let cam = BrownConradyCamera::try_with_focal(f, cx, cy, radial, tangential).unwrap();
        let pixel = cam.project(&point).unwrap();
        let bearing = cam.unproject(&pixel).unwrap();

        let expected_dir = point.normalize();
        let actual_dir = bearing.normalize();
        let dot = expected_dir.dot(&actual_dir);

        // Distortion models use iterative undistortion, so relax tolerance
        prop_assert!(
            (dot - 1.0).abs() < 1e-6,
            "Brown-Conrady round-trip error: dot={}, gap={}",
            dot, (dot - 1.0).abs()
        );
    }
}

// ─── Fisheye Camera ──────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Fisheye project should produce finite results for points in front
    #[test]
    fn fisheye_project_is_finite(
        f in valid_focal(),
        cx in valid_principal_x(),
        cy in valid_principal_y(),
        fisheye in small_fisheye(),
        point in point_in_front(),
    ) {
        let cam = FisheyeCamera::try_with_focal(f, cx, cy, fisheye).unwrap();
        let pixel = cam.project(&point).unwrap();
        prop_assert!(pixel.x.is_finite(), "pixel.x not finite");
        prop_assert!(pixel.y.is_finite(), "pixel.y not finite");
    }

    /// Fisheye unproject(project(point)) should be approximately collinear
    #[test]
    fn fisheye_roundtrip_collinear(
        f in valid_focal(),
        cx in valid_principal_x(),
        cy in valid_principal_y(),
        fisheye in small_fisheye(),
        point in point_in_front(),
    ) {
        let cam = FisheyeCamera::try_with_focal(f, cx, cy, fisheye).unwrap();
        let pixel = cam.project(&point).unwrap();
        let bearing = cam.unproject(&pixel).unwrap();

        let expected_dir = point.normalize();
        let actual_dir = bearing.normalize();
        let dot = expected_dir.dot(&actual_dir);

        prop_assert!(
            (dot - 1.0).abs() < 1e-6,
            "Fisheye round-trip error: dot={}, gap={}",
            dot, (dot - 1.0).abs()
        );
    }
}

// ─── Spherical Camera ────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Spherical camera: unproject(project(point)) should be collinear
    #[test]
    fn spherical_roundtrip_collinear(
        point in point_in_front(),
    ) {
        let cam = SphericalCamera::try_new(1024).unwrap();
        let pixel = cam.project(&point).unwrap();
        let bearing = cam.unproject(&pixel).unwrap();

        let expected_dir = point.normalize();
        let actual_dir = bearing.normalize();
        let dot = expected_dir.dot(&actual_dir);

        prop_assert!(
            (dot - 1.0).abs() < 1e-4,
            "Spherical round-trip error: dot={}, gap={}",
            dot, (dot - 1.0).abs()
        );
    }

    /// Spherical camera: project always produces finite coordinates
    #[test]
    fn spherical_project_is_finite(
        x in -10.0..10.0_f64,
        y in -10.0..10.0_f64,
        z in -10.0..10.0_f64,
    ) {
        // Skip zero vector
        if x.abs() < 1e-10 && y.abs() < 1e-10 && z.abs() < 1e-10 {
            return Ok(());
        }
        let cam = SphericalCamera::try_new(1024).unwrap();
        let point = Vector3::new(x, y, z);
        let pixel = cam.project(&point).unwrap();
        prop_assert!(pixel.x.is_finite());
        prop_assert!(pixel.y.is_finite());
    }
}

// ─── Error Conditions ────────────────────────────────────────────────

proptest! {
    /// Points behind the camera (negative Z) should fail projection for pinhole
    #[test]
    fn pinhole_rejects_points_behind_camera(
        f in valid_focal(),
        cx in valid_principal_x(),
        cy in valid_principal_y(),
        x in -5.0..5.0_f64,
        y in -5.0..5.0_f64,
        z in -20.0..-0.001_f64,
    ) {
        let cam = PinholeCamera::try_with_focal(f, cx, cy).unwrap();
        let point = Vector3::new(x, y, z);
        prop_assert!(cam.project(&point).is_err());
    }

    /// Invalid focal length (zero or negative) should fail construction
    #[test]
    fn pinhole_rejects_invalid_focal(
        f in -100.0..0.0_f64,
        cx in valid_principal_x(),
        cy in valid_principal_y(),
    ) {
        prop_assert!(PinholeCamera::try_with_focal(f, cx, cy).is_err());
    }
}
