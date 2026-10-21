// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Camera residual evaluation and per-camera residual policies.

use super::*;

#[test]
fn spherical_pixel_residual_wraps_across_the_longitude_seam() {
    let camera = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
    let residual = camera_pixel_residual(
        &camera,
        Point2::new(1023.75, 200.0),
        Point2::new(0.25, 200.0),
    );
    assert!((residual - 0.5).abs() < 1e-12);
}

#[test]
fn camera_bearing_error_matches_exact_projection() {
    let camera = Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap();
    let camera_point = Vector3::new(0.25, -0.1, 2.0);
    let observed = camera.project(&camera_point).unwrap();

    let residual = camera_bearing_error_rad(&camera, &camera_point, observed).unwrap();
    assert!(residual.abs() < 1e-12);
}

#[test]
fn residual_policy_uses_pixels_for_perspective_and_angles_for_spherical() {
    let perspective = Camera::try_pinhole(800.0, 600.0, 32.0, 24.0).unwrap();
    assert!(matches!(
        CameraResidualPolicy::from_pixel_limit(&perspective, 4.0).unwrap(),
        CameraResidualPolicy::PerspectivePixels { maximum } if maximum == 4.0
    ));
    let brown = Camera::try_brown_conrady(
        800.0,
        600.0,
        32.0,
        24.0,
        RadialDistortion::K1K2K3(-0.1, 0.01, 0.0),
        TangentialDistortion::P1P2(0.0, 0.0),
    )
    .unwrap();
    assert!(matches!(
        CameraResidualPolicy::from_pixel_limit(&brown, 4.0).unwrap(),
        CameraResidualPolicy::PerspectivePixels { maximum } if maximum == 4.0
    ));

    let fisheye = Camera::Fisheye(
        FisheyeCamera::try_with_focal(600.0, 32.0, 24.0, FisheyeDistortion::None).unwrap(),
    );
    assert!(matches!(
        CameraResidualPolicy::from_pixel_limit(&fisheye, 4.0).unwrap(),
        CameraResidualPolicy::Angular {
            maximum_radians,
            ..
        }
            if (maximum_radians - 4.0 / 600.0).abs() < 1e-12
    ));

    let spherical = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
    let CameraResidualPolicy::Angular {
        maximum_radians, ..
    } = CameraResidualPolicy::from_pixel_limit(&spherical, 4.0).unwrap()
    else {
        panic!("spherical cleanup must use an angular residual");
    };
    assert!((maximum_radians - 4.0 * std::f64::consts::TAU / 1024.0).abs() < 1e-12);
}
