// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fundamental matrix: the epipolar geometry of two uncalibrated cameras, from matched pixels.
//!
//! A point in one image constrains its match in the other image to a line, its epipolar line.
//! The fundamental matrix `F` encodes that constraint, `x₂ᵀ F x₁ = 0`, without knowing either
//! camera's intrinsics. Start with [`estimate_fundamental`]; with calibrated cameras, prefer the
//! essential matrix of [`crate::essential`].
//!
//! ## Available algorithms
//!
//! | Solver | Sample | Solutions | Status | Intended use |
//! |---|---:|---:|---|---|
//! | [`EightPointSolver`] | 8+ | 0–1 | Linear, overdetermined | Consensus refit |
//! | [`SevenPointSolver`] | 7 | 0–3 | Exact minimal | RANSAC hypotheses |
//!
//! ## Normalized input
//!
//! The 7- and 8-point solvers fit on Hartley-normalized coordinates, which keeps the linear
//! system well conditioned: their `solve` takes a
//! [`NormalizedCorrespondences`](crate::normalization::NormalizedCorrespondences) set built
//! from pixel correspondences, and returns pixel-space matrices (`F = T2ᵀ F_n T1`).
//! [`estimate_fundamental`] does this for you.
//!
//! ## Usage
//!
//! ```rust
//! use openmvg_geometry::SolverResult;
//! use openmvg_geometry::correspondence::PixelCorrespondence;
//! use openmvg_geometry::fundamental::{FundamentalMatrix, SevenPointSolver};
//! use openmvg_geometry::normalization::NormalizedCorrespondences;
//!
//! /// Up to three pixel-space fundamental matrices from seven pixel correspondences.
//! fn estimate(pixels: &[PixelCorrespondence; 7]) -> SolverResult<Vec<FundamentalMatrix>> {
//!     SevenPointSolver::solve(&NormalizedCorrespondences::from_pixels(pixels))
//! }
//! ```

mod eight_point;
mod seven_point;

use nalgebra::Matrix3;

use crate::{SolverError, SolverResult};

/// Conditioning policy for models fitted in Hartley-normalized coordinates.
/// Image-coordinate transforms can change this ratio, so it must not validate
/// a denormalized pixel-space representation.
pub(crate) const NORMALIZED_FUNDAMENTAL_CONDITIONING_RATIO: f64 = 1e-6;

/// Relative floating-point resolution of a 3x3 matrix's numerical rank.
///
/// Singular values move by at most the spectral norm of a matrix perturbation
/// (Weyl's inequality). The dimension-scaled f64 resolution, with a roundoff
/// margin of 64, is a representation policy rather than a scene-conditioning
/// threshold. Both nonzero singular directions must be resolved at this scale.
const FUNDAMENTAL_RANK_RESOLUTION: f64 = 64.0 * 3.0 * f64::EPSILON;

/// Fundamental matrix `F` of the epipolar constraint `x₂ᵀ F x₁ = 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FundamentalMatrix(Matrix3<f64>);

impl FundamentalMatrix {
    /// Validate and wrap a pixel-space fundamental matrix.
    ///
    /// Accepts exactly the matrices for which [`is_valid`](Self::is_valid)
    /// holds: a finite nonzero matrix of numerical rank 2 at f64 resolution
    /// (`σ2/σ1 > 192ε` and `σ3/σ1 ≤ 192ε`, where `ε = f64::EPSILON`).
    /// The matrix is stored as given, without rescaling or rank projection.
    /// Conditioning of the fitted scene is checked in normalized coordinates;
    /// this boundary only checks the supplied matrix representation.
    ///
    /// # Errors
    ///
    /// [`SolverError::InvalidInput`] when an entry is non-finite or the
    /// singular-value ratios are not those of a rank-2 matrix.
    pub fn try_new(matrix: Matrix3<f64>) -> SolverResult<Self> {
        if has_projective_rank_two(&matrix) {
            Ok(Self(matrix))
        } else {
            Err(SolverError::InvalidInput {
                reason: "fundamental matrix must be finite and have projective rank 2",
            })
        }
    }

    /// Wrap a matrix without validation (internal solver outputs).
    pub(crate) fn new(matrix: Matrix3<f64>) -> Self {
        Self(matrix)
    }

    /// Return the underlying 3x3 matrix.
    pub fn matrix(&self) -> &Matrix3<f64> {
        &self.0
    }

    /// Check whether the matrix satisfies the rank-2 constraint.
    ///
    /// The fundamental matrix must be finite and have rank 2. Because it is
    /// projective, a temporary copy is scaled before checking its numerical
    /// rank: `σ2/σ1 > 192ε` and `σ3/σ1 ≤ 192ε`, where `ε = f64::EPSILON`.
    /// This can refuse a representation whose second direction is unresolved
    /// in f64; it is not a coordinate-invariant geometric degeneracy test.
    pub fn is_valid(&self) -> bool {
        has_projective_rank_two(&self.0)
    }
}

impl AsRef<Matrix3<f64>> for FundamentalMatrix {
    fn as_ref(&self) -> &Matrix3<f64> {
        &self.0
    }
}

/// Direct singular values avoid squaring the condition number in a Gram
/// spectrum or amplifying determinant cancellation by a tiny second value.
fn projective_singular_values(matrix: &Matrix3<f64>) -> Option<nalgebra::Vector3<f64>> {
    if !matrix.iter().all(|value| value.is_finite()) {
        return None;
    }
    let scale = matrix
        .iter()
        .copied()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    // nalgebra's specialized Matrix3 SVD forms M^T M first. Appending an
    // exact zero row preserves all three singular values and selects its
    // direct bidiagonal SVD, with fixed-size stack storage throughout.
    let mut extended = nalgebra::Matrix4x3::zeros();
    extended.fixed_rows_mut::<3>(0).copy_from(&(matrix / scale));
    let values = extended
        .try_svd(false, false, f64::EPSILON, 64)?
        .singular_values;
    if values.iter().all(|value| value.is_finite()) && values[0] > 0.0 {
        Some(values)
    } else {
        None
    }
}

fn has_projective_rank_two(matrix: &Matrix3<f64>) -> bool {
    let Some(values) = projective_singular_values(matrix) else {
        return false;
    };
    let resolution = FUNDAMENTAL_RANK_RESOLUTION * values[0];
    values[1] > resolution && values[2] <= resolution
}

/// Candidate conditioning is evaluated before denormalization only.
fn normalized_model_is_conditioned(model: &FundamentalMatrix) -> bool {
    let Some(values) = projective_singular_values(model.matrix()) else {
        return false;
    };
    values[1] > NORMALIZED_FUNDAMENTAL_CONDITIONING_RATIO * values[0]
        && values[2] <= FUNDAMENTAL_RANK_RESOLUTION * values[0]
}

// Re-export the solver types.
mod robust;

pub use eight_point::EightPointSolver;
pub(crate) use eight_point::UnnormalizedFundamentalSolver;
pub use robust::{estimate_fundamental, estimate_fundamental_guided};
pub use seven_point::SevenPointSolver;

/// Transform a fundamental matrix from Hartley-normalized to input coordinates.
#[must_use]
pub(crate) fn denormalize(
    model: &FundamentalMatrix,
    normalization: &crate::normalization::PixelNormalization,
) -> FundamentalMatrix {
    FundamentalMatrix::new(normalization.denormalize_fundamental(*model.matrix()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Vector3};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn svd_validity_oracle(matrix: &Matrix3<f64>) -> bool {
        if !matrix.iter().all(|value| value.is_finite()) {
            return false;
        }
        let scale = matrix.iter().map(|value| value.abs()).fold(0.0, f64::max);
        if scale <= 0.0 {
            return false;
        }
        let mut extended = nalgebra::Matrix4x3::zeros();
        extended.fixed_rows_mut::<3>(0).copy_from(&(matrix / scale));
        let singular_values = extended.svd(false, false).singular_values;
        let scale = singular_values[0];
        if !scale.is_finite() || scale <= 0.0 {
            return false;
        }
        singular_values[1] / scale > FUNDAMENTAL_RANK_RESOLUTION
            && singular_values[2] / scale <= FUNDAMENTAL_RANK_RESOLUTION
    }

    fn matrix_with_singular_values(
        rng: &mut StdRng,
        second: f64,
        third: f64,
        scale: f64,
    ) -> Matrix3<f64> {
        let left = Rotation3::from_euler_angles(
            rng.gen_range(-3.0..3.0),
            rng.gen_range(-3.0..3.0),
            rng.gen_range(-3.0..3.0),
        );
        let right = Rotation3::from_euler_angles(
            rng.gen_range(-3.0..3.0),
            rng.gen_range(-3.0..3.0),
            rng.gen_range(-3.0..3.0),
        );
        left.matrix()
            * Matrix3::from_diagonal(&Vector3::new(scale, scale * second, scale * third))
            * right.matrix().transpose()
    }

    #[test]
    fn validity_is_projectively_scale_invariant() {
        let rank_two = Matrix3::new(1.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0);
        for scale in [1e-12, -1e-12, 1.0, -1.0, 1e12, -1e12] {
            assert!(FundamentalMatrix::new(rank_two * scale).is_valid());
        }
    }

    #[test]
    fn validity_rejects_degenerate_and_non_finite_matrices() {
        assert!(!FundamentalMatrix::new(Matrix3::<f64>::zeros()).is_valid());
        assert!(
            !FundamentalMatrix::new(Matrix3::from_diagonal(&nalgebra::Vector3::new(
                1.0, 0.0, 0.0,
            )))
            .is_valid()
        );
        assert!(!FundamentalMatrix::new(Matrix3::<f64>::identity()).is_valid());

        let mut non_finite = Matrix3::identity();
        non_finite[(0, 0)] = f64::NAN;
        assert!(!FundamentalMatrix::new(non_finite).is_valid());
    }

    #[test]
    fn try_new_accepts_exactly_the_valid_matrices() {
        let rank_two = Matrix3::new(1.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0);
        let model = FundamentalMatrix::try_new(rank_two).unwrap();
        assert_eq!(model.as_ref(), &rank_two);

        let mut non_finite = rank_two;
        non_finite[(2, 2)] = f64::INFINITY;
        for invalid in [Matrix3::identity(), Matrix3::zeros(), non_finite] {
            assert!(!FundamentalMatrix::new(invalid).is_valid());
            assert!(matches!(
                FundamentalMatrix::try_new(invalid),
                Err(SolverError::InvalidInput { .. })
            ));
        }
    }

    #[test]
    fn invariant_validity_matches_svd_around_rank_thresholds() {
        let mut rng = StdRng::seed_from_u64(0xF007_DA7A_51D0_2026);
        let tolerance = FUNDAMENTAL_RANK_RESOLUTION;
        let offsets = [0.5, 0.9, 0.99, 0.999_999, 1.000_001, 1.01, 1.1, 2.0];
        let mut checked = 0usize;
        for index in 0..512 {
            let scale = 10.0_f64.powf(rng.gen_range(-150.0..150.0));
            for &factor in &offsets {
                // Rank-1 boundary: sigma2 crosses the lower ratio threshold.
                let near_rank_one = matrix_with_singular_values(
                    &mut rng,
                    factor * tolerance,
                    0.1 * tolerance,
                    scale,
                );
                let expected = svd_validity_oracle(&near_rank_one);
                assert_eq!(
                    FundamentalMatrix::new(near_rank_one).is_valid(),
                    expected,
                    "rank-1 boundary mismatch at sample {index}, factor {factor}"
                );

                // Rank-3 boundary: sigma3 crosses the upper ratio threshold.
                let near_full_rank =
                    matrix_with_singular_values(&mut rng, 0.3, factor * tolerance, scale);
                let expected = svd_validity_oracle(&near_full_rank);
                assert_eq!(
                    FundamentalMatrix::new(near_full_rank).is_valid(),
                    expected,
                    "rank-3 boundary mismatch at sample {index}, factor {factor}"
                );
                checked += 2;
            }
        }
        assert_eq!(checked, 8192);
    }

    #[test]
    fn invariant_validity_matches_svd_on_random_adversarial_spectra() {
        let mut rng = StdRng::seed_from_u64(0xADDE_125A_5EED_0001);
        for index in 0..8192 {
            let second = 10.0_f64.powf(rng.gen_range(-9.0..0.0));
            let third = second * 10.0_f64.powf(rng.gen_range(-9.0..0.0));
            let scale = 10.0_f64.powf(rng.gen_range(-150.0..150.0));
            let matrix = matrix_with_singular_values(&mut rng, second, third, scale);
            assert_eq!(
                FundamentalMatrix::new(matrix).is_valid(),
                svd_validity_oracle(&matrix),
                "adversarial spectrum mismatch at sample {index}: [{second}, {third}]"
            );
        }
    }

    #[test]
    fn numerical_rank_and_normalized_conditioning_are_separate() {
        let resolved = Matrix3::from_diagonal(&Vector3::new(1.0, 1e-8, 0.0));
        let model = FundamentalMatrix::try_new(resolved).unwrap();
        assert_eq!(model.matrix().map(f64::to_bits), resolved.map(f64::to_bits));
        assert!(!normalized_model_is_conditioned(&model));
        assert!(
            FundamentalMatrix::try_new(Matrix3::from_diagonal(&Vector3::new(1.0, 1e-8, 1e-8,)))
                .is_err()
        );

        let tolerance = FUNDAMENTAL_RANK_RESOLUTION;
        for (second, third, expected) in [
            (0.5 * tolerance, 0.0, false),
            (2.0 * tolerance, 0.0, true),
            (0.3, 0.5 * tolerance, true),
            (0.3, 2.0 * tolerance, false),
        ] {
            let matrix = Matrix3::from_diagonal(&Vector3::new(1.0, second, third));
            assert_eq!(FundamentalMatrix::try_new(matrix).is_ok(), expected);
        }
        for scale in [1e-200, -1e-200, 1e200, -1e200] {
            let matrix = resolved * scale;
            let wrapped = FundamentalMatrix::try_new(matrix).unwrap();
            assert_eq!(wrapped.matrix().map(f64::to_bits), matrix.map(f64::to_bits));
        }
    }
}
