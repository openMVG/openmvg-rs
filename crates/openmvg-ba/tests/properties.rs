// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::f64::consts::PI;

use openmvg_ba::kernel::RelativePose;
use openmvg_ba::numerics::manifold;
use proptest::prelude::*;

fn frobenius_error(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> f64 {
    left.iter()
        .flatten()
        .zip(right.iter().flatten())
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn frobenius_dot(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> f64 {
    left.iter()
        .flatten()
        .zip(right.iter().flatten())
        .map(|(left, right)| left * right)
        .sum()
}

fn frobenius_norm(matrix: &[[f64; 3]; 3]) -> f64 {
    frobenius_dot(matrix, matrix).sqrt()
}

fn dot(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn transpose(matrix: &manifold::Mat3) -> manifold::Mat3 {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn so3_exp_log_roundtrip_holds_across_the_principal_branch(
        raw_axis in prop::array::uniform3(-1.0f64..1.0),
        angle in 0.0f64..(PI - 1e-7),
    ) {
        let axis_norm = raw_axis.iter().map(|value| value * value).sum::<f64>().sqrt();
        prop_assume!(axis_norm > 1e-8);
        let tangent = raw_axis.map(|value| angle * value / axis_norm);
        let rotation = manifold::exp_so3(&tangent);
        let recovered = manifold::log_so3(&rotation);
        let reconstructed = manifold::exp_so3(&recovered);
        prop_assert!(frobenius_error(&rotation, &reconstructed) < 2e-8);
    }

    #[test]
    fn so3_exponential_is_a_proper_rotation(
        tangent in prop::array::uniform3(-4.0f64..4.0),
    ) {
        let rotation = manifold::exp_so3(&tangent);
        let matrix = manifold::to_na(&rotation);
        let orthogonality = matrix.transpose() * matrix - nalgebra::Matrix3::identity();
        prop_assert!(orthogonality.norm() < 3e-14);
        prop_assert!((matrix.determinant() - 1.0).abs() < 3e-14);
        prop_assert!(manifold::log_so3(&rotation).iter().all(|value| value.is_finite()));
    }

    #[test]
    fn symmetrized_so3_log_roundtrips_through_the_near_pi_branch(
        raw_axis in prop::array::uniform3(-1.0f64..1.0),
        distance_from_pi in 0.0f64..1e-4,
    ) {
        let axis_norm = raw_axis.iter().map(|value| value * value).sum::<f64>().sqrt();
        prop_assume!(axis_norm > 1e-8);
        let angle = PI - distance_from_pi;
        let tangent = raw_axis.map(|value| angle * value / axis_norm);
        let rotation = manifold::exp_so3(&tangent);
        let reconstructed = manifold::exp_so3(&manifold::log_so3(&rotation));
        prop_assert!(frobenius_error(&rotation, &reconstructed) < 2e-7);
    }

    #[test]
    fn so3_left_and_right_jacobians_match_finite_differences(
        raw_axis in prop::array::uniform3(-1.0f64..1.0),
        angle in 0.0f64..(PI - 1e-5),
    ) {
        let axis_norm = raw_axis.iter().map(|value| value * value).sum::<f64>().sqrt();
        prop_assume!(axis_norm > 1e-8);
        let tangent = raw_axis.map(|value| angle * value / axis_norm);
        let rotation = manifold::exp_so3(&tangent);
        let inverse = transpose(&rotation);
        let left = manifold::so3_left_jacobian(&tangent);
        let right = transpose(&manifold::so3_left_jacobian(&tangent));
        let epsilon = 2e-7;
        for column in 0..3 {
            let mut displaced = tangent;
            displaced[column] += epsilon;
            let perturbed = manifold::exp_so3(&displaced);
            let measured_left = manifold::log_so3(&manifold::mat_mul(&perturbed, &inverse));
            let measured_right = manifold::log_so3(&manifold::mat_mul(&inverse, &perturbed));
            for row in 0..3 {
                prop_assert!((measured_left[row] / epsilon - left[row][column]).abs() < 3e-7);
                prop_assert!((measured_right[row] / epsilon - right[row][column]).abs() < 3e-7);
            }
        }
    }

    #[test]
    fn so3_jacobians_and_inverses_cancel_across_the_principal_branch(
        raw_axis in prop::array::uniform3(-1.0f64..1.0),
        angle in 0.0f64..PI,
    ) {
        let axis_norm = raw_axis.iter().map(|value| value * value).sum::<f64>().sqrt();
        prop_assume!(axis_norm > 1e-8);
        let tangent = raw_axis.map(|value| angle * value / axis_norm);
        let left_identity = manifold::mat_mul(
            &manifold::so3_left_jacobian(&tangent),
            &manifold::so3_left_jacobian_inverse(&tangent),
        );
        let right_identity = manifold::mat_mul(
            &transpose(&manifold::so3_left_jacobian(&tangent)),
            &transpose(&manifold::so3_left_jacobian_inverse(&tangent)),
        );
        prop_assert!(frobenius_error(&left_identity, manifold::Rotation3::IDENTITY.as_matrix()) < 2e-14);
        prop_assert!(frobenius_error(&right_identity, manifold::Rotation3::IDENTITY.as_matrix()) < 2e-14);
    }

    #[test]
    fn checked_rotation_operations_preserve_so3(
        first in prop::array::uniform3(-2.0f64..2.0),
        second in prop::array::uniform3(-2.0f64..2.0),
        point in prop::array::uniform3(-10.0f64..10.0),
    ) {
        let first = manifold::Rotation3::try_from_rotation_vector(&first).unwrap();
        let second = manifold::Rotation3::try_from_rotation_vector(&second).unwrap();
        let composed = first.compose(second);
        prop_assert!(manifold::Rotation3::try_from_matrix(composed.into_matrix()).is_ok());
        let recovered = composed.inverse().act(&composed.act(&point));
        for coordinate in 0..3 {
            prop_assert!((recovered[coordinate] - point[coordinate]).abs() < 2e-13);
        }
    }

    #[test]
    fn projective_normalization_is_invariant_to_nonzero_scale(
        entries in prop::array::uniform9(-10.0f64..10.0),
        scale in prop_oneof![-100.0f64..-1e-6, 1e-6f64..100.0],
    ) {
        let matrix = std::array::from_fn(|row| {
            std::array::from_fn(|column| entries[3 * row + column])
        });
        prop_assume!(frobenius_norm(&matrix) > 1e-8);
        let scaled = matrix.map(|row| row.map(|value| scale * value));
        let left = manifold::normalize_projective(&matrix);
        let right = manifold::normalize_projective(&scaled);
        prop_assert!(frobenius_error(&left, &right) < 2e-13);
    }

    #[test]
    fn projective_normalization_handles_extreme_finite_scales(
        entries in prop::array::uniform9(-1.0f64..1.0),
        exponent in -250i32..250,
        negative in any::<bool>(),
    ) {
        let matrix = std::array::from_fn(|row| {
            std::array::from_fn(|column| entries[3 * row + column])
        });
        prop_assume!(matrix.iter().flatten().map(|value| value.abs()).fold(0.0, f64::max) > 1e-4);
        let scale = 10.0f64.powi(exponent) * if negative { -1.0 } else { 1.0 };
        let scaled = matrix.map(|row| row.map(|value| scale * value));
        let left = manifold::try_normalize_projective(&matrix).unwrap();
        let right = manifold::try_normalize_projective(&scaled).unwrap();
        prop_assert!(frobenius_error(&left, &right) < 5e-15);
    }

    #[test]
    fn s2_retraction_preserves_unit_norm(
        raw_direction in prop::array::uniform3(-1.0f64..1.0),
        step in prop::array::uniform2(-2.0f64..2.0),
    ) {
        let direction_norm = raw_direction.iter().map(|value| value * value).sum::<f64>().sqrt();
        prop_assume!(direction_norm > 1e-8);
        let direction = raw_direction.map(|value| value / direction_norm);
        let basis = manifold::s2_tangent_basis(&direction);
        let retracted = manifold::retract_s2(&direction, &basis, &step);
        let norm = retracted.iter().map(|value| value * value).sum::<f64>().sqrt();
        prop_assert!((norm - 1.0).abs() < 2e-14);
    }

    #[test]
    fn s2_basis_and_retraction_obey_the_exponential_map(
        raw_direction in prop::array::uniform3(-1.0f64..1.0),
        step in prop::array::uniform2(-1.0f64..1.0),
    ) {
        let direction_norm = raw_direction.iter().map(|value| value * value).sum::<f64>().sqrt();
        prop_assume!(direction_norm > 1e-8);
        let direction = raw_direction.map(|value| value / direction_norm);
        let basis = manifold::s2_tangent_basis(&direction);
        prop_assert!(dot(&direction, &basis[0]).abs() < 2e-15);
        prop_assert!(dot(&direction, &basis[1]).abs() < 2e-15);
        prop_assert!(dot(&basis[0], &basis[1]).abs() < 2e-15);
        prop_assert!((dot(&basis[0], &basis[0]) - 1.0).abs() < 2e-15);
        prop_assert!((dot(&basis[1], &basis[1]) - 1.0).abs() < 2e-15);
        let retracted = manifold::retract_s2(&direction, &basis, &step);
        let step_norm = step.iter().map(|value| value * value).sum::<f64>().sqrt();
        prop_assert!((dot(&direction, &retracted) - step_norm.cos()).abs() < 3e-15);
    }

    #[test]
    fn rank2_tangent_basis_is_orthonormal_and_tangent(
        entries in prop::array::uniform9(-1.0f64..1.0),
    ) {
        let ambient = std::array::from_fn(|row| {
            std::array::from_fn(|column| entries[3 * row + column])
        });
        let fundamental = manifold::try_project_rank2(&ambient).unwrap();
        let svd = nalgebra::SVD::new(manifold::to_na(&fundamental), false, false);
        prop_assume!(svd.singular_values[1] > 1e-5);
        // The determinant gradient is the exact ambient normal of the rank-2
        // variety. An independently recomputed SVD normal is unnecessarily
        // sensitive to last-bit null-vector changes.
        let normal = determinant_gradient(&fundamental);
        let normal_norm = frobenius_norm(&normal);
        let normal = normal.map(|row| row.map(|value| value / normal_norm));
        let basis = manifold::try_fundamental_tangent_basis(&fundamental).unwrap();
        for row in 0..7 {
            let scale_component = frobenius_dot(&basis[row], &fundamental).abs();
            prop_assert!(scale_component < 1e-10, "scale component {scale_component:e}");
            let normal_component = frobenius_dot(&basis[row], &normal).abs();
            prop_assert!(normal_component < 1e-10, "normal component {normal_component:e}");
            for column in 0..7 {
                let expected = if row == column { 1.0 } else { 0.0 };
                prop_assert!((frobenius_dot(&basis[row], &basis[column]) - expected).abs() < 1e-10);
            }
        }
    }

}

fn determinant_gradient(matrix: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [
            matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1],
            matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2],
            matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0],
        ],
        [
            matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2],
            matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0],
            matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1],
        ],
        [
            matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1],
            matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2],
            matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
        ],
    ]
}

#[test]
fn fallible_manifold_api_rejects_invalid_objects() {
    assert!(manifold::try_normalize_projective(&[[0.0; 3]; 3]).is_none());
    assert!(manifold::try_exp_so3(&[f64::NAN, 0.0, 0.0]).is_none());
}

#[test]
fn so3_inverse_jacobians_remain_stable_at_pi() {
    let axis_norm = (0.3f64 * 0.3 + 0.7 * 0.7 + 0.2 * 0.2).sqrt();
    let tangent = [0.3, -0.7, 0.2].map(|value| PI * value / axis_norm);
    for (jacobian, inverse) in [
        (
            manifold::so3_left_jacobian(&tangent),
            manifold::so3_left_jacobian_inverse(&tangent),
        ),
        (
            transpose(&manifold::so3_left_jacobian(&tangent)),
            transpose(&manifold::so3_left_jacobian_inverse(&tangent)),
        ),
    ] {
        assert!(inverse.iter().flatten().all(|value| value.is_finite()));
        assert!(
            frobenius_error(
                &manifold::mat_mul(&jacobian, &inverse),
                manifold::Rotation3::IDENTITY.as_matrix(),
            ) < 2e-14
        );
    }
}

#[test]
fn checked_rotation_and_pose_types_reject_invalid_state() {
    assert_eq!(
        manifold::Rotation3::try_from_matrix([[0.0; 3]; 3]),
        Err(manifold::Rotation3Error::NotOrthogonal)
    );
    assert_eq!(
        manifold::Rotation3::try_from_matrix([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],]),
        Err(manifold::Rotation3Error::ImproperDeterminant)
    );
    assert_eq!(
        manifold::Pose3::try_new(manifold::Rotation3::IDENTITY, [f64::NAN, 0.0, 0.0]),
        Err(manifold::Pose3Error::NonFiniteTranslation)
    );
}

#[test]
fn checked_pose_group_operations_and_relative_pose_conversion_agree() {
    let rotation = manifold::Rotation3::try_from_rotation_vector(&[0.2, -0.1, 0.05]).unwrap();
    let pose = manifold::Pose3::try_new(rotation, [1.0, -2.0, 0.5]).unwrap();
    let identity = pose.try_compose(pose.try_inverse().unwrap()).unwrap();
    assert!(
        frobenius_error(
            identity.rotation().as_matrix(),
            manifold::Rotation3::IDENTITY.as_matrix()
        ) < 2e-15
    );
    assert!(
        identity
            .translation()
            .iter()
            .all(|value| value.abs() < 2e-15)
    );

    let compatibility: RelativePose = pose.into();
    assert_eq!(compatibility.try_as_pose3().unwrap(), pose);
    assert_eq!(
        RelativePose::try_from_parts(compatibility.rotation, compatibility.translation).unwrap(),
        compatibility
    );
}
