// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scaled Cholesky factorization for symmetric 3x3 point blocks.

#[cfg(any(feature = "faer", feature = "unstable-internals"))]
use crate::numerics::manifold::Mat3;

#[cfg(any(feature = "faer", feature = "unstable-internals"))]
pub(crate) const POINT_BLOCK_RCOND_THRESHOLD: f64 = 64.0 * f64::EPSILON * 3.0;

/// Spectral conditioning below which a point block is quarantined rather than eliminated.
///
/// Eliminating a block adds rounding of about `eps / rcond` of the camera system's scale, and the
/// camera-rank certificate accepts eigenvalues down to about `1e-11` of the largest. A block with
/// `rcond` above `1e-5` (a ray parallax of about 0.2 degrees) keeps the noise under that floor; a
/// lower threshold leaves indefinite noise of `1e-8` that fails a sound camera system, as measured
/// on Airavatesvara and Tartu. The solver's own weak-point test ([`POINT_BLOCK_RCOND_THRESHOLD`])
/// only catches blocks that cannot be factored at all.
pub(crate) const POINT_BLOCK_QUARANTINE_RCOND: f64 = 1.0e-5;

#[cfg(feature = "faer")]
const PACKED_DIMENSION: usize = 6;

/// Packed lower triangle of one symmetric 3x3 point block.
#[derive(Clone, Copy, Debug, Default)]
#[cfg(feature = "faer")]
pub(super) struct PackedPointBlock {
    values: [f64; PACKED_DIMENSION],
}

#[cfg(feature = "faer")]
impl PackedPointBlock {
    pub(super) fn from_matrix(matrix: &Mat3) -> Self {
        Self {
            values: [
                matrix[0][0],
                matrix[1][0],
                matrix[1][1],
                matrix[2][0],
                matrix[2][1],
                matrix[2][2],
            ],
        }
    }

    #[inline]
    pub(super) fn multiply(&self, vector: &[f64; 3]) -> [f64; 3] {
        let [a00, a10, a11, a20, a21, a22] = self.values;
        [
            a00 * vector[0] + a10 * vector[1] + a20 * vector[2],
            a10 * vector[0] + a11 * vector[1] + a21 * vector[2],
            a20 * vector[0] + a21 * vector[1] + a22 * vector[2],
        ]
    }

    #[inline]
    pub(super) fn bilinear(&self, left: &[f64; 3], right: &[f64; 3]) -> f64 {
        dot(left, &self.multiply(right))
    }

    pub(super) fn invert_in_place(&mut self) -> bool {
        let [a00, a10, a11, a20, a21, a22] = self.values;
        let matrix = [[a00, a10, a20], [a10, a11, a21], [a20, a21, a22]];
        let Some(factor) = PointBlockFactor::factor_damped(&matrix) else {
            return false;
        };
        // Both types pack the lower triangle in the same order.
        self.values = factor.inverse;
        true
    }
}

/// Inverse of a symmetric 3x3 point block, formed once from its Jacobi-scaled
/// Cholesky factor so that each of the many solves per factorization is a
/// division-free symmetric product.
#[cfg(any(feature = "faer", feature = "unstable-internals"))]
pub(super) struct PointBlockFactor {
    /// Packed lower triangle `[i00, i10, i11, i20, i21, i22]`.
    inverse: [f64; 6],
}

#[cfg(any(feature = "faer", feature = "unstable-internals"))]
impl PointBlockFactor {
    pub fn new(matrix: &Mat3, damping: f64) -> Option<Self> {
        let damped = Self::damped_matrix(matrix, damping)?;
        Self::factor_damped(&damped)
    }

    pub(super) fn damped_matrix(matrix: &Mat3, damping: f64) -> Option<Mat3> {
        let mut damped = *matrix;
        let diagonal = Self::damping_diagonal(matrix)?;
        for (coordinate, row) in damped.iter_mut().enumerate() {
            row[coordinate] += damping * diagonal[coordinate];
        }
        Some(damped)
    }

    pub(super) fn damping_diagonal(matrix: &Mat3) -> Option<[f64; 3]> {
        let diagonal_scale = (0..3)
            .map(|coordinate| matrix[coordinate][coordinate])
            .fold(0.0, f64::max);
        if !(diagonal_scale.is_finite() && diagonal_scale > 0.0) {
            return None;
        }
        let floor = 64.0 * f64::EPSILON * diagonal_scale;
        Some(std::array::from_fn(|coordinate| {
            matrix[coordinate][coordinate].max(floor)
        }))
    }

    fn factor_damped(damped: &Mat3) -> Option<Self> {
        let diagonal_scale = (0..3)
            .map(|coordinate| damped[coordinate][coordinate])
            .fold(0.0, f64::max);
        let floor = 64.0 * f64::EPSILON * diagonal_scale;
        let scale = std::array::from_fn(|index| damped[index][index].max(floor).sqrt());
        let mut lower = [[0.0; 3]; 3];
        for row in 0..3 {
            for column in 0..=row {
                let mut value = 0.5 * (damped[row][column] + damped[column][row])
                    / (scale[row] * scale[column]);
                for (&row_value, &column_value) in
                    lower[row][..column].iter().zip(&lower[column][..column])
                {
                    value -= row_value * column_value;
                }
                if row == column {
                    if !(value.is_finite() && value > 64.0 * f64::EPSILON) {
                        return None;
                    }
                    lower[row][column] = value.sqrt();
                } else {
                    lower[row][column] = value / lower[column][column];
                }
            }
        }
        // M = L^-1, lower triangular; the scaled inverse is M^T M and the
        // block inverse is D^-1 M^T M D^-1 with D the Jacobi scale.
        let m00 = lower[0][0].recip();
        let m11 = lower[1][1].recip();
        let m22 = lower[2][2].recip();
        let m10 = -lower[1][0] * m00 * m11;
        let m21 = -lower[2][1] * m11 * m22;
        let m20 = -(lower[2][0] * m00 + lower[2][1] * m10) * m22;
        let [d0, d1, d2] = scale.map(f64::recip);
        Some(Self {
            inverse: [
                (m00 * m00 + m10 * m10 + m20 * m20) * (d0 * d0),
                (m11 * m10 + m21 * m20) * (d1 * d0),
                (m11 * m11 + m21 * m21) * (d1 * d1),
                (m22 * m20) * (d2 * d0),
                (m22 * m21) * (d2 * d1),
                (m22 * m22) * (d2 * d2),
            ],
        })
    }

    #[inline]
    pub fn solve(&self, right_hand_side: &[f64; 3]) -> [f64; 3] {
        let [i00, i10, i11, i20, i21, i22] = self.inverse;
        let [x0, x1, x2] = *right_hand_side;
        [
            i00 * x0 + i10 * x1 + i20 * x2,
            i10 * x0 + i11 * x1 + i21 * x2,
            i20 * x0 + i21 * x1 + i22 * x2,
        ]
    }

    pub fn reciprocal_condition_number(&self, matrix: &Mat3) -> f64 {
        let inverse: Mat3 = std::array::from_fn(|column| {
            let mut basis = [0.0; 3];
            basis[column] = 1.0;
            self.solve(&basis)
        });
        let inverse = transpose(&inverse);
        let denominator = infinity_norm(matrix) * infinity_norm(&inverse);
        if denominator.is_finite() && denominator > 0.0 {
            denominator.recip()
        } else {
            0.0
        }
    }
}

#[cfg(any(feature = "faer", feature = "unstable-internals"))]
pub(super) fn reciprocal_condition_number(matrix: &Mat3) -> f64 {
    PointBlockFactor::new(matrix, 0.0)
        .map_or(0.0, |factor| factor.reciprocal_condition_number(matrix))
}

#[cfg(any(feature = "faer", feature = "unstable-internals"))]
fn transpose(matrix: &Mat3) -> Mat3 {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

#[cfg(any(feature = "faer", feature = "unstable-internals"))]
fn infinity_norm(matrix: &Mat3) -> f64 {
    matrix
        .iter()
        .map(|row| row.iter().map(|entry| entry.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

#[inline]
#[cfg(feature = "faer")]
pub(super) fn dot(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(all(test, any(feature = "faer", feature = "unstable-internals")))]
mod tests {
    use super::*;

    #[test]
    fn scaled_factor_accepts_well_defined_blocks_with_large_diagonal_ranges() {
        for matrix in [
            [[1.0, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]],
            [[1e12, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ] {
            let factor = PointBlockFactor::new(&matrix, 0.0)
                .expect("positive-definite diagonal block must factor");
            let right = [0.4, -0.3, 0.2];
            let solution = factor.solve(&right);
            for row in 0..3 {
                let reconstructed = (0..3)
                    .map(|column| matrix[row][column] * solution[column])
                    .sum::<f64>();
                assert!((reconstructed - right[row]).abs() < 1e-11);
            }
        }
    }

    #[test]
    fn inverse_is_accurate_for_correlated_anisotropic_blocks() {
        // Q is an exact orthogonal matrix, so H = Q D Q^T has the known
        // inverse Q D^-1 Q^T with condition number max(D) / min(D) = 1e9.
        let q =
            [[1.0, 2.0, 2.0], [2.0, 1.0, -2.0], [2.0, -2.0, 1.0]].map(|row| row.map(|v| v / 3.0));
        let compose = |diagonal: [f64; 3]| -> Mat3 {
            std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    (0..3).map(|k| q[row][k] * diagonal[k] * q[column][k]).sum()
                })
            })
        };
        let diagonal = [1e6, 1.0, 1e-3];
        let matrix = compose(diagonal);
        let reference = compose(diagonal.map(f64::recip));
        let factor = PointBlockFactor::new(&matrix, 0.0).expect("SPD block must factor");
        // Forward error is bounded by roughly kappa * eps = 2e-7 relative to
        // ||H^-1|| = 1e3; allow 1e-6 relative.
        let tolerance = 1e-6 * 1e3;
        for column in 0..3 {
            let mut basis = [0.0; 3];
            basis[column] = 1.0;
            let solution = factor.solve(&basis);
            for row in 0..3 {
                let expected = reference[row][column];
                assert!((solution[row] - expected).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn rank_deficient_block_is_rejected() {
        assert!(
            PointBlockFactor::new(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], 0.0,)
                .is_none()
        );
    }

    #[test]
    fn cached_damping_diagonal_exactly_reconstructs_point_damping() {
        let matrix = [[4.0, 0.2, 0.0], [0.2, 2.0, 0.1], [0.0, 0.1, 1.0]];
        let damping = 0.3;
        let diagonal = PointBlockFactor::damping_diagonal(&matrix).unwrap();
        let damped = PointBlockFactor::damped_matrix(&matrix, damping).unwrap();
        for coordinate in 0..3 {
            assert!(
                (damped[coordinate][coordinate]
                    - matrix[coordinate][coordinate]
                    - damping * diagonal[coordinate])
                    .abs()
                    < 1e-15
            );
        }
    }

    #[cfg(feature = "faer")]
    #[test]
    fn packed_inverse_reconstructs_the_identity() {
        let matrix = [[4.0, 0.2, -0.1], [0.2, 2.0, 0.1], [-0.1, 0.1, 1.0]];
        let mut inverse = PackedPointBlock::from_matrix(&matrix);
        assert!(inverse.invert_in_place());
        for column in 0..3 {
            let mut basis = [0.0; 3];
            basis[column] = 1.0;
            let solution = inverse.multiply(&basis);
            let reconstructed: [f64; 3] = std::array::from_fn(|row| {
                (0..3)
                    .map(|inner| matrix[row][inner] * solution[inner])
                    .sum()
            });
            for (row, &actual) in reconstructed.iter().enumerate() {
                let expected = usize::from(row == column) as f64;
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }
}
