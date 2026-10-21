// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Coupled `SE(3)` exponential.

use super::{Mat3, Pose3, Rotation3, so3_left_jacobian};

/// Fallible coupled `SE(3)` exponential of a tangent in rotation-then-translation
/// order `(omega, v)`.
///
/// The returned transform has `R = Exp(omega)` and
/// `t = J_l(omega) * v`.
pub fn try_exp_se3(tangent: &[f64; 6]) -> Option<Pose3> {
    if tangent.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let omega = [tangent[0], tangent[1], tangent[2]];
    let v = [tangent[3], tangent[4], tangent[5]];
    let rotation = Rotation3::try_from_rotation_vector(&omega).ok()?;
    let jacobian = so3_left_jacobian(&omega);
    let translation = mat_vec(&jacobian, &v);
    Pose3::try_new(rotation, translation).ok()
}

fn mat_vec(matrix: &Mat3, vector: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        matrix[row]
            .iter()
            .zip(vector)
            .map(|(coefficient, value)| coefficient * value)
            .sum()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(left: &[f64], right: &[f64], tolerance: f64) {
        for (left, right) in left.iter().zip(right) {
            assert!((left - right).abs() <= tolerance, "{left} != {right}");
        }
    }

    #[test]
    fn exp_couples_translation_through_the_left_jacobian() {
        let tangent = [0.2, -0.1, 0.3, 1.2, -0.7, 0.4];
        let pose = try_exp_se3(&tangent).unwrap();
        let omega = [0.2, -0.1, 0.3];
        let expected = mat_vec(&so3_left_jacobian(&omega), &[1.2, -0.7, 0.4]);
        near(&pose.translation(), &expected, 0.0);
        near(
            pose.rotation().as_matrix().as_flattened(),
            super::super::exp_so3(&omega).as_flattened(),
            0.0,
        );
        assert!(try_exp_se3(&[f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0]).is_none());
    }
}
