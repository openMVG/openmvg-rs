// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-size Lie-group and geometric-manifold operations used by refinement.

#![deny(missing_docs)]

use nalgebra::{Matrix3, SVD, Vector3};

mod se3;
mod so3;

pub use se3::try_exp_se3;

pub use so3::{
    Pose3, Pose3Error, Rotation3, Rotation3Error, exp_so3, log_so3, so3_left_jacobian,
    so3_left_jacobian_inverse, try_exp_so3,
};

/// Row-major fixed-size 3x3 matrix used at the public API boundary.
pub type Mat3 = [[f64; 3]; 3];

#[inline]
fn requested_svd_vectors(
    u: Option<Matrix3<f64>>,
    v_t: Option<Matrix3<f64>>,
) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
    Some((u?, v_t?))
}

#[inline]
/// Convert the crate's row-major array representation to `nalgebra`.
pub fn to_na(m: &Mat3) -> Matrix3<f64> {
    Matrix3::new(
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    )
}

#[inline]
/// Convert a fixed-size `nalgebra` matrix to the crate's array representation.
pub fn from_na(m: &Matrix3<f64>) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| m[(i, j)]))
}

#[inline]
/// Return the skew-symmetric matrix satisfying `skew(v) * x = v cross x`.
pub fn skew(v: &[f64; 3]) -> Mat3 {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

#[inline]
/// Multiply two 3x3 matrices without allocation.
pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0; 3]; 3];
    for (i, c_row) in c.iter_mut().enumerate() {
        for (j, value) in c_row.iter_mut().enumerate() {
            for (k, b_row) in b.iter().enumerate() {
                *value += a[i][k] * b_row[j];
            }
        }
    }
    c
}

#[inline]
fn frobenius_dot(a: &Mat3, b: &Mat3) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += a[i][j] * b[i][j];
        }
    }
    s
}

#[inline]
fn frobenius_norm(a: &Mat3) -> f64 {
    frobenius_dot(a, a).sqrt()
}

/// Canonical projective gauge: unit Frobenius norm and deterministic sign.
///
/// Invalid or zero input maps to the zero matrix. Use
/// [`try_normalize_projective`] when invalid input must be distinguished.
pub fn normalize_projective(m: &Mat3) -> Mat3 {
    try_normalize_projective(m).unwrap_or([[0.0; 3]; 3])
}

/// Normalize a finite, nonzero projective matrix without overflow or underflow.
pub fn try_normalize_projective(m: &Mat3) -> Option<Mat3> {
    let scale = m
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let scaled = std::array::from_fn(|row| std::array::from_fn(|column| m[row][column] / scale));
    let n = frobenius_norm(&scaled);
    if !(n.is_finite() && n > 0.0) {
        return None;
    }
    let mut r = scaled;
    for row in &mut r {
        for x in row {
            *x /= n;
        }
    }
    let mut pivot = r[0][0];
    for row in &r {
        for &x in row {
            if x.abs() > pivot.abs() {
                pivot = x;
            }
        }
    }
    if pivot < 0.0 {
        for row in &mut r {
            for x in row {
                *x = -*x;
            }
        }
    }
    Some(r)
}

/// An orthonormal basis of the tangent plane at t on S^2.
pub fn s2_tangent_basis(t: &[f64; 3]) -> [[f64; 3]; 2] {
    let tv = Vector3::new(t[0], t[1], t[2]).normalize();
    let reference = if tv[0].abs() <= tv[1].abs() && tv[0].abs() <= tv[2].abs() {
        Vector3::x()
    } else if tv[1].abs() <= tv[2].abs() {
        Vector3::y()
    } else {
        Vector3::z()
    };
    let b0 = tv.cross(&reference).normalize();
    let b1 = tv.cross(&b0).normalize();
    [[b0[0], b0[1], b0[2]], [b1[0], b1[1], b1[2]]]
}

/// Apply the S2 exponential-map retraction in a supplied tangent basis.
pub fn retract_s2(t: &[f64; 3], basis: &[[f64; 3]; 2], d: &[f64; 2]) -> [f64; 3] {
    let v = Vector3::new(
        basis[0][0] * d[0] + basis[1][0] * d[1],
        basis[0][1] * d[0] + basis[1][1] * d[1],
        basis[0][2] * d[0] + basis[1][2] * d[1],
    );
    let theta = v.norm();
    let tv = Vector3::new(t[0], t[1], t[2]).normalize();
    let out = if theta < 1e-10 {
        (tv + v).normalize()
    } else {
        tv * theta.cos() + v * (theta.sin() / theta)
    };
    [out[0], out[1], out[2]]
}

/// Construct an essential matrix `E = [t]_x R` from a relative pose.
pub fn essential_from_pose(r: &Mat3, t: &[f64; 3]) -> Mat3 {
    mat_mul(&skew(t), r)
}

/// Fallible rank-2 projection for a finite, nonzero input matrix.
pub fn try_project_rank2(f: &Mat3) -> Option<Mat3> {
    try_normalize_projective(f)?;
    let svd = SVD::new(to_na(f), true, true);
    let mut singular_values = svd.singular_values;
    let (u, v_t) = requested_svd_vectors(svd.u, svd.v_t)?;
    singular_values[2] = 0.0;
    let projected = from_na(&(u * Matrix3::from_diagonal(&singular_values) * v_t));
    try_normalize_projective(&projected)
}

/// Fallible rank-2 projective tangent construction.
pub fn try_fundamental_tangent_basis(f: &Mat3) -> Option<[Mat3; 7]> {
    let input = try_normalize_projective(f)?;
    let projected = try_project_rank2(&input)?;
    // Avoid a second SVD perturbation when the caller already supplied a
    // rank-2 state. This keeps the returned directions orthogonal to the exact
    // projective representative used by the retraction. Full-rank inputs are
    // replaced by their nearest rank-2 matrix.
    let projection_error = std::array::from_fn(|row| {
        std::array::from_fn(|column| input[row][column] - projected[row][column])
    });
    let f = if frobenius_norm(&projection_error) <= 1e-8 {
        input
    } else {
        projected
    };
    let svd = SVD::new(to_na(&f), true, true);
    let largest = svd.singular_values[0];
    if !(largest.is_finite()
        && largest > 0.0
        && svd.singular_values[1] > 64.0 * f64::EPSILON * largest)
    {
        return None;
    }
    let (u, v_t) = requested_svd_vectors(svd.u, svd.v_t)?;
    let v = v_t.transpose();
    let mut normal = determinant_gradient(&f);
    let normal_norm = frobenius_norm(&normal);
    if !(normal_norm.is_finite() && normal_norm > 64.0 * f64::EPSILON) {
        return None;
    }
    for row in &mut normal {
        for value in row {
            *value /= normal_norm;
        }
    }
    let pairs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (0, 2),
        (1, 2),
    ];
    let mut basis = [[[0.0; 3]; 3]; 7];
    let mut count = 0usize;
    for (ui, vj) in pairs {
        let outer = u.column(ui) * v.column(vj).transpose();
        let mut q = from_na(&outer);
        // Reorthogonalized modified Gram--Schmidt is cheap for seven 3x3
        // directions and prevents scale leakage for clustered singular values.
        for _ in 0..2 {
            let along_scale = frobenius_dot(&q, &f);
            for i in 0..3 {
                for j in 0..3 {
                    q[i][j] -= along_scale * f[i][j];
                }
            }
            let along_normal = frobenius_dot(&q, &normal);
            for i in 0..3 {
                for j in 0..3 {
                    q[i][j] -= along_normal * normal[i][j];
                }
            }
            for previous in basis.iter().take(count) {
                let coeff = frobenius_dot(&q, previous);
                for i in 0..3 {
                    for j in 0..3 {
                        q[i][j] -= coeff * previous[i][j];
                    }
                }
            }
        }
        let n = frobenius_norm(&q);
        if n > 1e-9 && count < 7 {
            for row in &mut q {
                for x in row {
                    *x /= n;
                }
            }
            basis[count] = q;
            count += 1;
        }
    }
    (count == 7).then_some(basis)
}

/// Gradient of the determinant with respect to the matrix entries.
fn determinant_gradient(matrix: &Mat3) -> Mat3 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallible_fundamental_tangent_rejects_degenerate_input() {
        assert!(try_fundamental_tangent_basis(&[[0.0; 3]; 3]).is_none());
        assert!(try_fundamental_tangent_basis(&[[f64::NAN; 3]; 3]).is_none());
    }

    fn matrix_error(a: &Mat3, b: &Mat3) -> f64 {
        let mut d = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                d[i][j] = a[i][j] - b[i][j];
            }
        }
        frobenius_norm(&d)
    }

    #[test]
    fn log_so3_roundtrip_near_pi() {
        let axis = Vector3::new(0.31, -0.72, 0.62).normalize();
        let angle = std::f64::consts::PI - 1e-7;
        let r = exp_so3(&[axis[0] * angle, axis[1] * angle, axis[2] * angle]);
        let recovered = exp_so3(&log_so3(&r));
        assert!(matrix_error(&r, &recovered) < 1e-6);
    }
}
