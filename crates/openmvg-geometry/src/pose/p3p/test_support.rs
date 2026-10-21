// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared assertions for the unit tests of non-default P3P configurations.
//!
//! The public dispatcher contract lives in `tests/p3p_properties.rs`; these
//! helpers mirror its checks for the crate-private solver options that the
//! dispatcher does not expose.

use nalgebra::{Isometry3, Point3, UnitQuaternion, UnitVector3};

use super::P3pSolutions;

pub(super) const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// Best rotation-angle plus scale-normalized center error over all candidates.
pub(super) fn relative_pose_error(
    candidates: &P3pSolutions,
    truth: &Isometry3<f64>,
    world: &[Point3<f64>; 3],
) -> f64 {
    let scene_scale = (world[0] - world[1])
        .norm()
        .max((world[0] - world[2]).norm())
        .max((world[1] - world[2]).norm());
    let truth_center = truth.inverse_transform_point(&Point3::origin());
    candidates
        .iter()
        .map(|candidate| {
            let rotation_error = UnitQuaternion::from_rotation_matrix(candidate.rotation())
                .angle_to(&truth.rotation);
            let center_error = (candidate.center() - truth_center).norm() / scene_scale;
            rotation_error + center_error
        })
        .fold(f64::INFINITY, f64::min)
}

/// Every candidate is a finite, cheiral pose that fits the minimal sample.
pub(super) fn assert_valid_candidates(
    name: &str,
    candidates: &P3pSolutions,
    world: &[Point3<f64>; 3],
    bearings: &[UnitVector3<f64>; 3],
) {
    for candidate in candidates {
        assert!(
            (UnitQuaternion::from_rotation_matrix(candidate.rotation())
                .quaternion()
                .norm_squared()
                - 1.0)
                .abs()
                <= 1e-10,
            "{name}: rotation quaternion is not unit length"
        );
        assert!(
            candidate
                .translation()
                .iter()
                .all(|value| value.is_finite()),
            "{name}: non-finite translation"
        );
        for (point, bearing) in world.iter().zip(bearings) {
            let camera = candidate.project_to_camera(point).coords;
            assert!(
                camera.iter().all(|value| value.is_finite()) && camera.dot(bearing) > 0.0,
                "{name}: returned a non-cheiral candidate"
            );
            let angular_residual = camera.cross(bearing).norm().atan2(camera.dot(bearing));
            assert!(
                angular_residual <= 2e-6,
                "{name}: candidate angular residual {angular_residual:e}"
            );
        }
    }
}

/// Valid candidates and a recovered truth pose for one sample.
pub(super) fn assert_recovers_truth(
    name: &str,
    candidates: &P3pSolutions,
    world: &[Point3<f64>; 3],
    bearings: &[UnitVector3<f64>; 3],
    truth: &Isometry3<f64>,
) {
    assert_valid_candidates(name, candidates, world, bearings);
    let error = relative_pose_error(candidates, truth, world);
    assert!(
        error < 1e-6,
        "{name}: relative pose error {error:e}, {} candidates",
        candidates.len()
    );
}

/// Recover the identity pose from every ordering of a camera-frame triangle.
pub(super) fn assert_recovers_all_permutations(
    name: &str,
    world: [Point3<f64>; 3],
    mut solve: impl FnMut([Point3<f64>; 3], [UnitVector3<f64>; 3]) -> P3pSolutions,
) {
    let bearings = world.map(|point| UnitVector3::new_normalize(point.coords));
    for permutation in PERMUTATIONS {
        let permuted_world = permutation.map(|index| world[index]);
        let permuted_bearings = permutation.map(|index| bearings[index]);
        let candidates = solve(permuted_world, permuted_bearings);
        assert_recovers_truth(
            name,
            &candidates,
            &permuted_world,
            &permuted_bearings,
            &Isometry3::identity(),
        );
    }
}
