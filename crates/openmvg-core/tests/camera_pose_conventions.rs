// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Matrix3, Point3, Rotation3, Vector3};
use openmvg_core::model::{RelativePose, WorldToCameraPose};

fn assert_close(a: Vector3<f64>, b: Vector3<f64>) {
    assert!(
        (a - b).norm() < 1e-10,
        "left={:?} right={:?} error={}",
        a,
        b,
        (a - b).norm()
    );
}

#[test]
fn camera_pose_translation_and_isometry_match_world_to_camera_convention() {
    let rotation = Rotation3::from_euler_angles(0.2, -0.3, 0.4);
    let center = Point3::new(1.5, -2.0, 3.0);
    let pose = WorldToCameraPose::from_rotation_center(rotation, center).unwrap();
    let world_point = Point3::new(4.0, -1.0, 8.0);

    let expected_translation = -(pose.rotation() * pose.center().coords);
    assert_close(pose.translation(), expected_translation);

    let expected_camera = Point3::from(pose.rotation() * (world_point - pose.center()));
    let camera_from_method = pose.project_to_camera(&world_point);
    let camera_from_isometry = pose.to_isometry().transform_point(&world_point);

    assert_close(camera_from_method.coords, expected_camera.coords);
    assert_close(camera_from_isometry.coords, expected_camera.coords);

    let recovered_world = pose.transform_to_world(&camera_from_method);
    let recovered_world_from_isometry = pose
        .to_isometry()
        .inverse_transform_point(&camera_from_method);

    assert_close(recovered_world.coords, world_point.coords);
    assert_close(recovered_world_from_isometry.coords, world_point.coords);
}

/// Relative-pose frame and translation sign: relative rotation and translation compose
/// source-camera points into the target camera frame; the deliberately opposite baseline-direction
/// convention is observably different.
#[test]
fn relative_pose_equations_fix_frames_and_translation_sign() {
    let pose_i = WorldToCameraPose::from_rotation_center(
        Rotation3::from_euler_angles(0.2, -0.1, 0.3),
        Point3::new(1.0, -2.0, 0.5),
    )
    .unwrap();
    let pose_j = WorldToCameraPose::from_rotation_center(
        Rotation3::from_euler_angles(-0.15, 0.25, -0.2),
        Point3::new(-0.5, 1.25, 2.0),
    )
    .unwrap();

    // x_j = R_ji x_i + t_ji, where t_ji is expressed in target camera j.
    let rotation_ji = pose_j.rotation() * pose_i.rotation().inverse();
    let translation_ji = pose_j.rotation() * (pose_i.center() - pose_j.center());
    let baseline = translation_ji.norm();
    let relative = RelativePose::from_projection_translation(
        rotation_ji,
        nalgebra::Unit::new_normalize(translation_ji),
    )
    .unwrap();
    for world in [
        Point3::new(0.5, -0.25, 4.0),
        Point3::new(-1.0, 2.0, 7.5),
        Point3::new(3.0, 0.75, -2.0),
    ] {
        let point_i = pose_i.project_to_camera(&world).coords;
        let point_j = pose_j.project_to_camera(&world).coords;
        let composed = relative.rotation_source_to_target() * point_i
            + baseline * relative.translation_source_to_target().as_ref();
        assert_close(composed, point_j);

        let bearing_i = point_i.normalize();
        let bearing_j = point_j.normalize();
        let t = relative.translation_source_to_target();
        let essential = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0)
            * relative.rotation_source_to_target();
        assert!(bearing_j.dot(&(essential * bearing_i)).abs() < 1e-12);
    }

    // The source-frame direction from camera i to camera j is not t_ji.
    // It is d_ij_in_i = -R_ji^T t_hat_ji.
    let baseline_direction_i = pose_i.rotation() * (pose_j.center() - pose_i.center()) / baseline;
    let converted_direction_i = relative.baseline_direction_in_source().into_inner();
    assert_close(baseline_direction_i, converted_direction_i);
    assert!((baseline_direction_i - relative.translation_source_to_target().as_ref()).norm() > 0.5);
}
