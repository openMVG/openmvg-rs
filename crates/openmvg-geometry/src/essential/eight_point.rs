// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 8-point algorithm for essential matrix estimation.
//!
//! Its teaching docs are on [`EightPointSolver`], where docs.rs shows them.

use crate::correspondence::BearingCorrespondence;
use crate::residuals::{sampson_error_essential, sampson_error_essential_squared};
use crate::shared::epipolar_constraint::{
    EpipolarConstraint, Vector9, solve_epipolar_ata, solve_epipolar_ata_from_rows,
};
use crate::solver_traits::{SingleSolutionSolver, Solver};
use crate::{SolverError, SolverResult};
use nalgebra::Matrix3;

use super::EssentialMatrix;

/// Essential matrix solver using the 8-point algorithm.
///
/// Estimates the essential matrix from at least 8 bearing correspondences.
/// The essential matrix relates calibrated camera views and has the structure `E = [t]_x * R`.
///
/// # How it works
///
/// Implements the 8-point algorithm for calibrated cameras (H. C. Longuet-Higgins,
/// "A computer algorithm for reconstructing a scene from two projections",
/// Nature, 1981).
///
/// # Algorithm
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │                    8-point essential matrix solver                          │
/// │                    (Longuet-Higgins algorithm)                              │
/// └─────────────────────────────────────────────────────────────────────────────┘
///
///     Input: 8+ bearing correspondences (b₁, b₂) from calibrated cameras
///            Each bearing is a unit vector: ||b|| = 1
///
///                    ┌───────────────────────────────────────┐
///                    │  Step 1: Build constraint matrix      │
///                    └───────────────────────────────────────┘
///                                      │
///                    For each correspondence (b₁ᵢ, b₂ᵢ):
///                    b₂ᵢᵀ · E · b₁ᵢ = 0
///
///                    Expand to linear equation in 9 unknowns of E:
///                    [b₂ₓb₁ₓ, b₂ₓb₁ᵧ, b₂ₓb₁ᵤ, b₂ᵧb₁ₓ, b₂ᵧb₁ᵧ, b₂ᵧb₁ᵤ, b₂ᵤb₁ₓ, b₂ᵤb₁ᵧ, b₂ᵤb₁ᵤ] · vec(E) = 0
///
///                    Build constraint matrix A (n×9) from n correspondences
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 2: Solve homogeneous system     │
///                    └───────────────────────────────────────┘
///                                      │
///                    Compute AᵀA (9×9 symmetric matrix)
///                    Eigenvalue decomposition: AᵀA = V·Λ·Vᵀ
///                    E = eigenvector with smallest eigenvalue
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 3: Enforce essential constraint │
///                    └───────────────────────────────────────┘
///                                      │
///                    SVD: E = U·Σ·Vᵀ
///                    Essential matrix must have singular values (σ, σ, 0)
///
///                    Enforce: Σ' = diag(σ̄, σ̄, 0), σ̄ = (σ₁ + σ₂) / 2
///                    E' = U·Σ'·Vᵀ
///                                      │
///                                      ▼
///     Output: 1 essential matrix E' with singular values (σ̄, σ̄, 0)
///
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │  Complexity: O(n) for constraint matrix, O(1) for eigenvalue problem        │
/// │  Memory: 9×9 normal matrix; no n×9 matrix is stored                         │
/// │  Role: consensus refit; the 5-point solver generates hypotheses             │
/// └─────────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Mathematical background
///
/// The essential matrix E relates calibrated camera views via:
/// - E = `[t]ₓ · R` (skew-symmetric translation × rotation)
/// - Singular values: (σ, σ, 0)
/// - det(E) = 0 (rank 2)
///
/// With 8+ points the system is overdetermined, which averages noise over all
/// points. As a RANSAC hypothesis generator it needs more iterations than the
/// minimal 5-point solver to draw an all-inlier sample.
///
/// See the usage example in [`crate::essential`].
///
/// # Algorithm
///
/// 1. Build the normal matrix AᵀA, where each row of A encodes x2^T * E * x1 = 0.
/// 2. Solve A * vec(E) = 0 via eigenvalue decomposition of AᵀA.
/// 3. Enforce the essential constraint (σ, σ, 0) on the singular values.
///
/// See the usage example in [`crate::essential`].
#[derive(Debug, Clone, Default)]
pub struct EightPointSolver;

// Crate-private: robust estimation drives it through the solver traits. The
// inherent surface exists only for the `unstable-test-support` evidence module.
#[cfg(feature = "unstable-test-support")]
crate::solver_traits::expose_solver_api!(EightPointSolver, BearingCorrespondence, EssentialMatrix);

impl Solver for EightPointSolver {
    type Datum = BearingCorrespondence;
    type Model = EssentialMatrix;
    type Config = ();
    type Unit = openmvg_core::Radians;

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_essential(&model.0, datum)
    }

    fn squared_residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_essential_squared(&model.0, datum)
    }

    fn min_sample_size() -> usize {
        8
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.is_valid()
    }
}

impl SingleSolutionSolver for EightPointSolver {
    fn solve_single(
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        if data.len() < 8 {
            return Err(SolverError::InsufficientData {
                required: 8,
                provided: data.len(),
            });
        }
        // Solve with essential constraint (two equal singular values + one zero)
        optional_essential(solve_epipolar_ata(data, EpipolarConstraint::Essential))
    }
}

/// Eight-point essential matrix refiner for normalized pinhole image rays.
///
/// Input bearings may be unit length or carry any other homogeneous scale.
/// They are dehomogenized to `(x / z, y / z, 1)` once while constructing the
/// overdetermined system. Use [`EightPointSolver`] for spherical cameras or rays
/// on the camera equator.
#[derive(Debug, Clone, Default)]
pub struct PinholeEssentialSolver;

impl Solver for PinholeEssentialSolver {
    type Datum = BearingCorrespondence;
    type Model = EssentialMatrix;
    type Config = ();
    type Unit = openmvg_core::NormalizedImage;

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        Self::squared_residual(model, datum).sqrt()
    }

    fn squared_residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        crate::residuals::sampson_error_essential_pinhole_squared(&model.0, datum)
    }

    fn min_sample_size() -> usize {
        8
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.is_valid()
    }
}

impl SingleSolutionSolver for PinholeEssentialSolver {
    fn solve_single(
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        if data.len() < Self::min_sample_size() {
            return Err(SolverError::InsufficientData {
                required: Self::min_sample_size(),
                provided: data.len(),
            });
        }

        optional_essential(solve_epipolar_ata_from_rows(
            data.iter().map(pinhole_constraint_row),
            EpipolarConstraint::Essential,
        ))
    }

    fn solve_single_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        if indices.len() < Self::min_sample_size() {
            return Err(SolverError::InsufficientData {
                required: Self::min_sample_size(),
                provided: indices.len(),
            });
        }
        let rows = indices.iter().map(|&index| {
            let correspondence = data.get(index).ok_or(SolverError::InvalidDataIndex {
                index,
                data_len: data.len(),
            })?;
            pinhole_constraint_row(correspondence)
        });
        optional_essential(solve_epipolar_ata_from_rows(
            rows,
            EpipolarConstraint::Essential,
        ))
    }
}

#[inline]
fn optional_essential(result: SolverResult<Matrix3<f64>>) -> SolverResult<Option<EssentialMatrix>> {
    match result {
        Ok(matrix) => Ok(Some(EssentialMatrix::new(matrix))),
        Err(SolverError::DegenerateConfiguration { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

#[inline]
fn pinhole_constraint_row(correspondence: &BearingCorrespondence) -> SolverResult<Vector9<f64>> {
    let b1 = *correspondence.bearing1();
    let b2 = *correspondence.bearing2();
    if b1.z.abs() < 1e-15 || b2.z.abs() < 1e-15 {
        return Err(SolverError::DegenerateConfiguration {
            reason: "pinhole essential-matrix refinement requires non-equatorial rays".to_owned(),
        });
    }
    let x1 = b1.x / b1.z;
    let y1 = b1.y / b1.z;
    let x2 = b2.x / b2.z;
    let y2 = b2.y / b2.z;
    Ok(Vector9::from([
        x2 * x1,
        y2 * x1,
        x1,
        x2 * y1,
        y2 * y1,
        y1,
        x2,
        y2,
        1.0,
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correspondence::PixelCorrespondence;
    use crate::solver_traits::MultiSolutionSolver;
    use nalgebra::{Rotation3, Vector3};

    #[test]
    fn test_essential_solver_basic() {
        let rotation = Rotation3::from_euler_angles(0.04, -0.12, 0.07);
        let translation = Vector3::new(0.45, -0.12, 0.08);
        let correspondences: Vec<_> = (0..15)
            .map(|i| {
                let t = i as f64;
                let point = Vector3::new(
                    (0.73 * t).sin() * 1.8,
                    (1.17 * t + 0.2).cos() * 1.5,
                    4.0 + (0.41 * t).sin().abs() * 3.0 + 0.09 * t,
                );
                let second = rotation * point + translation;
                BearingCorrespondence::try_new(point, second).unwrap()
            })
            .collect();

        let result = <EightPointSolver as MultiSolutionSolver>::solve(&correspondences, &());

        assert!(result.is_ok());
        let models = result.unwrap();
        assert_eq!(models.len(), 1);

        // Check essential matrix constraints
        let e = &models[0];
        let svd = e.0.svd(false, false);
        let s = svd.singular_values;

        // Third singular value should be ~0
        assert!(s[2].abs() < 1e-6);

        // First two should be approximately equal
        let ratio = (s[0] - s[1]).abs() / s[0];
        assert!(ratio < 1e-6);
    }

    #[test]
    fn repeated_bearings_return_no_model() {
        let correspondence = BearingCorrespondence::try_new(
            Vector3::new(0.1, -0.2, 1.0).normalize(),
            Vector3::new(0.2, -0.1, 1.0).normalize(),
        )
        .unwrap();
        let result =
            <EightPointSolver as MultiSolutionSolver>::solve(&[correspondence; 8], &()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn pinhole_refiner_returns_a_valid_low_residual_model() {
        let rotation = Rotation3::from_euler_angles(0.04, -0.12, 0.07);
        let translation = Vector3::new(0.45, -0.12, 0.08);
        let correspondences = (0..15)
            .map(|i| {
                let t = i as f64;
                let point = Vector3::new(
                    (0.73 * t).sin() * 1.8,
                    (1.17 * t + 0.2).cos() * 1.5,
                    4.0 + (0.41 * t).sin().abs() * 3.0 + 0.09 * t,
                );
                BearingCorrespondence::try_new(point, rotation * point + translation).unwrap()
            })
            .collect::<Vec<_>>();

        let model = PinholeEssentialSolver::solve(&correspondences, &())
            .expect("valid pinhole correspondences")
            .pop()
            .expect("one eight-point model");
        assert!(model.is_valid());
        let mean_squared_residual = correspondences
            .iter()
            .map(|datum| PinholeEssentialSolver::squared_residual(&model, datum))
            .sum::<f64>()
            / correspondences.len() as f64;
        assert!(mean_squared_residual < 1e-10, "{mean_squared_residual:e}");
    }

    #[test]
    fn pinhole_refiner_returns_no_model_for_equatorial_rays() {
        let correspondence = BearingCorrespondence::try_new(
            Vector3::new(0.1, -0.2, 1.0).normalize(),
            Vector3::new(0.2, -0.1, 1.0).normalize(),
        )
        .unwrap();
        let mut correspondences = vec![correspondence; 8];
        correspondences[0] =
            BearingCorrespondence::try_new(Vector3::x(), *correspondence.bearing2()).unwrap();
        let result = PinholeEssentialSolver::solve(&correspondences, &()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn pinhole_indexed_refiner_streams_the_same_normalized_system() {
        let rotation = Rotation3::from_euler_angles(0.03, -0.09, 0.05);
        let translation = Vector3::new(0.31, -0.17, 0.06);
        let correspondences = (0..14)
            .map(|i| {
                let u = i as f64;
                let point = Vector3::new((0.57 * u).sin(), (0.91 * u + 0.1).cos(), 3.5 + 0.2 * u);
                let second = rotation * point + translation;
                BearingCorrespondence::try_new(point * (1.2 + 0.03 * u), second * (0.8 + 0.02 * u))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let indices = (0..correspondences.len()).rev().collect::<Vec<_>>();
        let streamed =
            PinholeEssentialSolver::solve_single_from_indices(&indices, &correspondences, &())
                .unwrap()
                .unwrap();
        let normalized = indices
            .iter()
            .map(|&index| {
                let datum = &correspondences[index];
                PixelCorrespondence::from_coords(
                    datum.bearing1().x / datum.bearing1().z,
                    datum.bearing1().y / datum.bearing1().z,
                    datum.bearing2().x / datum.bearing2().z,
                    datum.bearing2().y / datum.bearing2().z,
                )
            })
            .collect::<Vec<_>>();
        let buffered = solve_epipolar_ata(&normalized, EpipolarConstraint::Essential).unwrap();
        let streamed_matrix = streamed.0 / streamed.0.norm();
        let buffered = buffered / buffered.norm();
        let projective_error = (streamed_matrix - buffered)
            .norm()
            .min((streamed_matrix + buffered).norm());
        assert!(projective_error < 1e-12, "{projective_error:e}");
    }

    #[test]
    fn pinhole_indexed_refiner_reports_an_invalid_index() {
        let data = vec![
            BearingCorrespondence::try_new(Vector3::new(0.1, -0.2, 1.0), Vector3::z())
                .unwrap();
            8
        ];
        assert!(matches!(
            PinholeEssentialSolver::solve_single_from_indices(
                &[0, 1, 2, 3, 4, 5, 6, 99],
                &data,
                &()
            ),
            Err(SolverError::InvalidDataIndex {
                index: 99,
                data_len: 8
            })
        ));
    }

    #[test]
    fn test_insufficient_points() {
        let correspondences: Vec<BearingCorrespondence> = (0..5)
            .map(|i| {
                let t = i as f64 * 0.1;
                BearingCorrespondence::try_new(
                    Vector3::new(t, 0.0, 1.0).normalize(),
                    Vector3::new(t + 0.1, 0.05, 1.0).normalize(),
                )
                .unwrap()
            })
            .collect();

        let result = <EightPointSolver as MultiSolutionSolver>::solve(&correspondences, &());

        assert!(result.is_err());
    }

    #[test]
    fn test_residual_computation() {
        // Create a simple essential matrix
        let e = EssentialMatrix::new(nalgebra::Matrix3::new(
            0.0, -0.1, 0.0, 0.1, 0.0, -0.1, 0.0, 0.1, 0.0,
        ));

        let corr = BearingCorrespondence::try_new(
            Vector3::new(0.0, 0.0, 1.0).normalize(),
            Vector3::new(0.0, 0.0, 1.0).normalize(),
        )
        .unwrap();

        let residual = EightPointSolver::residual(&e, &corr);
        assert!(residual.is_finite());
        assert!(residual >= 0.0);
    }
}
