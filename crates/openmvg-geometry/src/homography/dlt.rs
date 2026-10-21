// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct Linear Transform (DLT) algorithm for homography estimation.
//!
//! Its teaching docs are on [`DltSolver`], where docs.rs shows them.

use nalgebra::{DMatrix, Matrix3, SMatrix, SVector, SymmetricEigen};

use crate::correspondence::PixelCorrespondence;
use crate::normalization::{
    CorrespondenceCoordinates, NormalizedCorrespondence, NormalizedCorrespondences,
};
use crate::residuals::symmetric_transfer_distance_homography;
use crate::shared::svd_utils::{solve_homogeneous_system, vector_to_matrix3, vector9_to_matrix3};
use crate::solver_traits::{MultiSolutionSolver, SingleSolutionSolver, Solver};
use crate::{SolverError, SolverResult};

use super::{Homography, HomographyConfig, passes_cheirality_test};

/// Homography solver using the DLT algorithm.
///
/// Estimates the homography matrix from at least 4 pixel correspondences
/// using the Direct Linear Transform method.
///
/// # How it works
///
/// Implements the normalized DLT algorithm (R. Hartley and A. Zisserman,
/// "Multiple View Geometry in Computer Vision", Cambridge University Press,
/// 2nd edition, 2004, Algorithm 4.2).
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │                    4-point DLT homography solver                            │
/// │                (Direct Linear Transform algorithm)                          │
/// └─────────────────────────────────────────────────────────────────────────────┘
///
///     Input: 4+ pixel correspondences (p₁, p₂)
///            p₂ ~ H · p₁  (projective transformation)
///
///                    ┌───────────────────────────────────────┐
///                    │  Step 1: Formulate constraints        │
///                    └───────────────────────────────────────┘
///                                      │
///                    For each correspondence ((x₁,y₁), (x₂,y₂)):
///                    Cross-product constraint: p₂ × (H·p₁) = 0
///
///                    Expand to 2 linear equations per point:
///                    [-x₁, -y₁, -1,  0,   0,   0, x₂x₁, x₂y₁, x₂] · h = 0
///                    [ 0,   0,   0, -x₁, -y₁, -1, y₂x₁, y₂y₁, y₂] · h = 0
///
///                    where h = vec(H) = [h₁₁, h₁₂, h₁₃, h₂₁, h₂₂, h₂₃, h₃₁, h₃₂, h₃₃]ᵀ
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 2: Build constraint matrix      │
///                    └───────────────────────────────────────┘
///                                      │
///                    Stack equations into matrix A (2n × 9):
///                    ┌                                                         ┐
///                    │ -x₁¹, -y₁¹, -1, 0, 0, 0, x₂¹x₁¹, x₂¹y₁¹, x₂¹            │
///                    │  0, 0, 0, -x₁¹, -y₁¹, -1, y₂¹x₁¹, y₂¹y₁¹, y₂¹           │
///                    │                    ⋮                                    │
///                    │ -x₁ⁿ, -y₁ⁿ, -1, 0, 0, 0, x₂ⁿx₁ⁿ, x₂ⁿy₁ⁿ, x₂ⁿ            │
///                    │  0, 0, 0, -x₁ⁿ, -y₁ⁿ, -1, y₂ⁿx₁ⁿ, y₂ⁿy₁ⁿ, y₂ⁿ           │
///                    └                                                         ┘
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 3: Solve homogeneous system     │
///                    └───────────────────────────────────────┘
///                                      │
///                    For 4 points (8×9 matrix):
///                    ├── Compute AᵀA (9×9 symmetric)
///                    ├── Eigenvalue decomposition
///                    └── h = eigenvector with smallest eigenvalue
///
///                    For >4 points (overdetermined):
///                    ├── SVD: A = U·Σ·Vᵀ
///                    └── h = last column of V (smallest singular value)
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 4: Reshape and normalize        │
///                    └───────────────────────────────────────┘
///                                      │
///                    Reshape h (9-vector) to H (3×3 matrix)
///                    Normalize: H ← H / H[2,2]  (or H / ||H||_F)
///                                      │
///                                      ▼
///     Output: 1 homography matrix H
///
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │  Complexity: O(n) constraint matrix + O(1) eigenvalue problem               │
/// │  Memory: Stack-allocated for the 4-point case (RANSAC samples)              │
/// │  Note: Fits on normalized input; the public solve denormalizes              │
/// │  Variants: DLT and ACA (Affine-Core-Affine, exactly 4 points)               │
/// └─────────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Normalized input
///
/// [`DltSolver::solve`] takes a [`NormalizedCorrespondences`] set, so the Hartley-normalized
/// input is part of the type and the fit runs on well-conditioned coordinates. It returns at
/// most one homography in pixel coordinates, `H = T2⁻¹ H_n T1`, scaled so `H[2, 2] = 1` when
/// possible.
///
/// See the usage example in [`crate::homography`].
#[derive(Debug, Clone, Default)]
pub struct DltSolver;

impl DltSolver {
    /// Return the solver's minimum correspondence count.
    #[must_use]
    pub fn min_sample_size() -> usize {
        <Self as Solver>::min_sample_size()
    }

    /// Evaluate the symmetric transfer distance of a pixel-space model for one pixel correspondence.
    pub fn residual(model: &Homography, datum: &PixelCorrespondence) -> f64 {
        symmetric_transfer_distance_homography(&model.0, datum)
    }

    /// Validate the model's geometric invariants.
    #[must_use]
    pub fn is_valid(model: &Homography) -> bool {
        <Self as Solver>::is_valid(model)
    }

    /// Fit on normalized correspondences and return the pixel-space homography.
    pub fn solve(
        data: &NormalizedCorrespondences,
        config: &HomographyConfig,
    ) -> SolverResult<Vec<Homography>> {
        let models = <Self as MultiSolutionSolver>::solve(data.as_slice(), config)?;
        Ok(models
            .iter()
            .map(|model| {
                Homography::new(data.normalization().denormalize_homography(*model.matrix()))
                    .normalized()
            })
            .collect())
    }
}

/// Internal DLT core without normalization.
pub(crate) struct DltCore;

impl DltCore {
    /// Solve for homography in the coordinates of `correspondences`.
    /// Uses stack allocation for the 4-point minimal case.
    pub fn solve_unnormalized<D: CorrespondenceCoordinates>(
        correspondences: &[D],
    ) -> SolverResult<Homography> {
        if correspondences.len() < 4 {
            return Err(SolverError::InsufficientData {
                required: 4,
                provided: correspondences.len(),
            });
        }

        // Use stack-allocated version for minimal 4-point case (RANSAC hot path)
        if correspondences.len() == 4 {
            return Self::solve_4point(correspondences);
        }

        // Fallback to heap-allocated version for overdetermined case
        Self::solve_overdetermined(correspondences)
    }

    /// Stack-allocated solver for exactly 4 correspondences, used by RANSAC samples.
    fn solve_4point<D: CorrespondenceCoordinates>(
        correspondences: &[D],
    ) -> SolverResult<Homography> {
        debug_assert_eq!(correspondences.len(), 4);

        // Build constraint matrix (8x9) - stack allocated
        let mut a: SMatrix<f64, 8, 9> = SMatrix::zeros();

        for (i, corr) in correspondences.iter().map(D::coordinates).enumerate() {
            let x1 = corr.point1.x;
            let y1 = corr.point1.y;
            let x2 = corr.point2.x;
            let y2 = corr.point2.y;
            let one = 1.0;
            let zero = 0.0;

            // First row: -x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2
            let row1 = 2 * i;
            a[(row1, 0)] = -x1;
            a[(row1, 1)] = -y1;
            a[(row1, 2)] = -one;
            a[(row1, 3)] = zero;
            a[(row1, 4)] = zero;
            a[(row1, 5)] = zero;
            a[(row1, 6)] = x2 * x1;
            a[(row1, 7)] = x2 * y1;
            a[(row1, 8)] = x2;

            // Second row: 0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2
            let row2 = 2 * i + 1;
            a[(row2, 0)] = zero;
            a[(row2, 1)] = zero;
            a[(row2, 2)] = zero;
            a[(row2, 3)] = -x1;
            a[(row2, 4)] = -y1;
            a[(row2, 5)] = -one;
            a[(row2, 6)] = y2 * x1;
            a[(row2, 7)] = y2 * y1;
            a[(row2, 8)] = y2;
        }

        if !a.iter().all(|value| value.is_finite()) {
            return Err(SolverError::DegenerateConfiguration {
                reason: "homography constraints contain non-finite coordinates".to_string(),
            });
        }

        // Compute the null space using eigendecomposition of A^T * A (9x9 symmetric)
        Self::solve_normal_matrix(a.transpose() * a)
    }

    /// Allocation-free indexed DLT for normalized correspondences.
    fn solve_indexed<D: CorrespondenceCoordinates>(
        indices: &[usize],
        correspondences: &[D],
    ) -> SolverResult<Homography> {
        if indices.len() < 4 {
            return Err(SolverError::InsufficientData {
                required: 4,
                provided: indices.len(),
            });
        }

        let mut ata: SMatrix<f64, 9, 9> = SMatrix::zeros();
        for &index in indices {
            let correspondence = correspondences
                .get(index)
                .ok_or(SolverError::InvalidDataIndex {
                    index,
                    data_len: correspondences.len(),
                })?
                .coordinates();
            let x1 = correspondence.point1.x;
            let y1 = correspondence.point1.y;
            let x2 = correspondence.point2.x;
            let y2 = correspondence.point2.y;
            let one = 1.0;
            let zero = 0.0;
            let rows = [
                SVector::<f64, 9>::from_row_slice(&[
                    -x1,
                    -y1,
                    -one,
                    zero,
                    zero,
                    zero,
                    x2 * x1,
                    x2 * y1,
                    x2,
                ]),
                SVector::<f64, 9>::from_row_slice(&[
                    zero,
                    zero,
                    zero,
                    -x1,
                    -y1,
                    -one,
                    y2 * x1,
                    y2 * y1,
                    y2,
                ]),
            ];
            for row in rows {
                if !row.iter().all(|value| value.is_finite()) {
                    return Err(SolverError::DegenerateConfiguration {
                        reason: "homography constraints contain non-finite coordinates".to_string(),
                    });
                }
                ata += row * row.transpose();
            }
        }
        if !ata.iter().all(|value| value.is_finite()) {
            return Err(SolverError::DegenerateConfiguration {
                reason: "homography normal matrix overflowed".to_string(),
            });
        }
        Self::solve_normal_matrix(ata)
    }

    fn solve_normal_matrix(ata: SMatrix<f64, 9, 9>) -> SolverResult<Homography> {
        let eigen = SymmetricEigen::new(ata);

        let mut order = [0usize, 1, 2, 3, 4, 5, 6, 7, 8];
        order.sort_by(|&left, &right| {
            let left = eigen.eigenvalues[left].abs();
            let right = eigen.eigenvalues[right].abs();
            left.total_cmp(&right)
        });
        let spectrum: [f64; 9] = std::array::from_fn(|i| eigen.eigenvalues[order[i]].abs());
        if spectrum.iter().any(|value| !value.is_finite()) || spectrum[8] <= 0.0 {
            return Err(SolverError::DegenerateConfiguration {
                reason: "homography constraints have zero or non-finite numerical rank".to_string(),
            });
        }
        let tolerance = spectrum[8] * f64::EPSILON * 9.0 * 64.0;
        if spectrum[1] <= tolerance || spectrum[1] - spectrum[0] <= tolerance {
            return Err(SolverError::DegenerateConfiguration {
                reason: "homography constraints have an ambiguous null space".to_string(),
            });
        }

        let h_vec: SVector<f64, 9> = eigen.eigenvectors.column(order[0]).into();
        let h: Matrix3<f64> = vector9_to_matrix3(&h_vec);
        let model = Homography::new(h);
        if !model.is_valid() {
            return Err(SolverError::DegenerateConfiguration {
                reason: "homography null space does not define an invertible transform".to_string(),
            });
        }
        Ok(model)
    }

    /// Heap-allocated solver for overdetermined case (> 4 points).
    fn solve_overdetermined<D: CorrespondenceCoordinates>(
        correspondences: &[D],
    ) -> SolverResult<Homography> {
        let n = correspondences.len();

        // Build constraint matrix (2n x 9)
        let mut a = DMatrix::zeros(2 * n, 9);

        for (i, corr) in correspondences.iter().map(D::coordinates).enumerate() {
            let x1 = corr.point1.x;
            let y1 = corr.point1.y;
            let x2 = corr.point2.x;
            let y2 = corr.point2.y;
            let one = 1.0;
            let zero = 0.0;

            // First row: -x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2
            let row1 = 2 * i;
            a[(row1, 0)] = -x1;
            a[(row1, 1)] = -y1;
            a[(row1, 2)] = -one;
            a[(row1, 3)] = zero;
            a[(row1, 4)] = zero;
            a[(row1, 5)] = zero;
            a[(row1, 6)] = x2 * x1;
            a[(row1, 7)] = x2 * y1;
            a[(row1, 8)] = x2;

            // Second row: 0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2
            let row2 = 2 * i + 1;
            a[(row2, 0)] = zero;
            a[(row2, 1)] = zero;
            a[(row2, 2)] = zero;
            a[(row2, 3)] = -x1;
            a[(row2, 4)] = -y1;
            a[(row2, 5)] = -one;
            a[(row2, 6)] = y2 * x1;
            a[(row2, 7)] = y2 * y1;
            a[(row2, 8)] = y2;
        }

        // Solve Ah = 0
        let h_vec = solve_homogeneous_system(&a)?;
        let h = vector_to_matrix3(&h_vec)?;
        let model = Homography::new(h);
        if !model.is_valid() {
            return Err(SolverError::DegenerateConfiguration {
                reason: "homography null space does not define an invertible transform".to_string(),
            });
        }
        Ok(model)
    }
}

impl DltCore {
    fn solve_single<D: CorrespondenceCoordinates>(
        data: &[D],
        config: &HomographyConfig,
    ) -> SolverResult<Option<Homography>> {
        if data.len() < 4 {
            return Err(SolverError::InsufficientData {
                required: 4,
                provided: data.len(),
            });
        }
        if !data
            .iter()
            .all(|datum| correspondence_is_finite(datum.coordinates()))
        {
            return Err(SolverError::InvalidInput {
                reason: "homography coordinates must be finite",
            });
        }

        // Cheirality test: check that point configuration is valid
        // (quadrilateral orientation is preserved between images)
        // Only applied when enabled; disabled by default for RANSAC
        if config.check_cheirality && !passes_cheirality_test(data) {
            return Ok(None);
        }

        // Both DLT and ACA algorithms work in the coordinates of `data`
        let h = match config.method {
            super::HomographyMethod::Dlt => Self::solve_unnormalized(data),
            super::HomographyMethod::Aca => {
                // ACA is designed for exactly 4 points (minimal solver for RANSAC)
                if data.len() == 4 {
                    super::aca::AcaSolver::solve(data)
                } else {
                    // For overdetermined case (>4 points), fall back to DLT
                    Self::solve_unnormalized(data)
                }
            }
        };

        optional_homography(h)
    }

    fn solve_single_from_indices<D: CorrespondenceCoordinates>(
        indices: &[usize],
        data: &[D],
        config: &HomographyConfig,
    ) -> SolverResult<Option<Homography>> {
        if indices.len() < 4 {
            return Err(SolverError::InsufficientData {
                required: 4,
                provided: indices.len(),
            });
        }
        let get = |position: usize| {
            let index = indices[position];
            data.get(index)
                .copied()
                .ok_or(SolverError::InvalidDataIndex {
                    index,
                    data_len: data.len(),
                })
        };
        let sample = [get(0)?, get(1)?, get(2)?, get(3)?];
        if !sample
            .iter()
            .all(|datum| correspondence_is_finite(datum.coordinates()))
        {
            return Err(SolverError::InvalidInput {
                reason: "homography coordinates must be finite",
            });
        }
        if config.check_cheirality && !passes_cheirality_test(&sample) {
            return Ok(None);
        }

        let homography = if indices.len() == 4 {
            match config.method {
                super::HomographyMethod::Dlt => Self::solve_4point(&sample),
                super::HomographyMethod::Aca => super::aca::AcaSolver::solve(&sample),
            }
        } else {
            Self::solve_indexed(indices, data)
        };
        optional_homography(homography)
    }
}

impl Solver for DltSolver {
    type Datum = NormalizedCorrespondence;
    type Model = Homography;
    type Config = HomographyConfig;
    type Unit = openmvg_core::Unitless;

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        symmetric_transfer_distance_homography(&model.0, datum.coordinates())
    }

    fn min_sample_size() -> usize {
        4
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.is_valid()
    }
}

fn correspondence_is_finite(correspondence: &PixelCorrespondence) -> bool {
    correspondence
        .point1
        .coords
        .iter()
        .chain(correspondence.point2.coords.iter())
        .all(|value| value.is_finite())
}

fn optional_homography(result: SolverResult<Homography>) -> SolverResult<Option<Homography>> {
    match result {
        Ok(model) => Ok(Some(model.normalized())),
        Err(SolverError::DegenerateConfiguration { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Allocation-free single-solution implementation used by robust refinement.
impl SingleSolutionSolver for DltSolver {
    fn solve_single(
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        DltCore::solve_single(data, config)
    }

    fn solve_single_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        DltCore::solve_single_from_indices(indices, data, config)
    }
}

/// The DLT/ACA kernel driven directly on raw pixel coordinates, without
/// Hartley normalization, as used by the compact classic robust loop
/// (`ClassicConfig::estimate_homography`).
pub(crate) struct UnnormalizedHomographySolver;

impl Solver for UnnormalizedHomographySolver {
    type Datum = PixelCorrespondence;
    type Model = Homography;
    type Config = HomographyConfig;
    type Unit = openmvg_core::Pixels;

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        symmetric_transfer_distance_homography(&model.0, datum)
    }

    fn min_sample_size() -> usize {
        4
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.is_valid()
    }
}

impl SingleSolutionSolver for UnnormalizedHomographySolver {
    fn solve_single(
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        DltCore::solve_single(data, config)
    }

    fn solve_single_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        DltCore::solve_single_from_indices(indices, data, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Point2, Vector3};

    fn create_correspondences_from_homography(
        h: &Matrix3<f64>,
        points: &[Point2<f64>],
    ) -> Vec<PixelCorrespondence> {
        points
            .iter()
            .map(|p1| {
                let x1 = Vector3::new(p1.x, p1.y, 1.0);
                let x2 = h * x1;
                let p2 = Point2::new(x2.x / x2.z, x2.y / x2.z);
                PixelCorrespondence::new(*p1, p2)
            })
            .collect()
    }

    #[test]
    fn test_dlt_identity() {
        // Identity homography; use more points to avoid degeneracy
        let h_true = Matrix3::identity();

        let points = vec![
            Point2::new(100.0, 100.0),
            Point2::new(200.0, 100.0),
            Point2::new(200.0, 200.0),
            Point2::new(100.0, 200.0),
            Point2::new(150.0, 150.0),
            Point2::new(120.0, 180.0),
        ];

        let correspondences = create_correspondences_from_homography(&h_true, &points);

        // Normalize data (as RANSAC would do)
        let normalized_data = NormalizedCorrespondences::from_pixels(&correspondences);

        // The public solve returns the homography in pixel coordinates
        let config = HomographyConfig::default();
        let result = DltSolver::solve(&normalized_data, &config);

        assert!(result.is_ok());
        let models = result.unwrap();
        assert_eq!(models.len(), 1);
        let h_estimated = models[0];

        // Check residuals are small
        let mut total_error = 0.0;
        for corr in &correspondences {
            let error = DltSolver::residual(&h_estimated, corr);
            total_error += error;
        }
        let avg_error = total_error / correspondences.len() as f64;
        assert!(avg_error < 1e-6, "Average error too large: {}", avg_error);
    }

    #[test]
    fn test_dlt_translation() {
        // Translation homography: x' = x + 50, y' = y + 30
        let h_true = Matrix3::new(1.0, 0.0, 50.0, 0.0, 1.0, 30.0, 0.0, 0.0, 1.0);

        let points = vec![
            Point2::new(100.0, 100.0),
            Point2::new(200.0, 100.0),
            Point2::new(200.0, 200.0),
            Point2::new(100.0, 200.0),
            Point2::new(150.0, 150.0),
        ];

        let correspondences = create_correspondences_from_homography(&h_true, &points);

        // Normalize data (as RANSAC would do)
        let normalized_data = NormalizedCorrespondences::from_pixels(&correspondences);

        // The public solve returns the homography in pixel coordinates
        let config = HomographyConfig::default();
        let result = DltSolver::solve(&normalized_data, &config);

        assert!(result.is_ok());
        let models = result.unwrap();
        let h_estimated = models[0];

        // Check residuals are small
        let mut total_error = 0.0;
        for corr in &correspondences {
            let error = DltSolver::residual(&h_estimated, corr);
            total_error += error;
        }
        let avg_error = total_error / correspondences.len() as f64;
        assert!(avg_error < 1e-6, "Average error too large: {}", avg_error);
    }

    #[test]
    fn test_dlt_rejects_collinear_minimal_sample() {
        let correspondences: Vec<_> = (0..4)
            .map(|i| {
                let x = i as f64;
                PixelCorrespondence::from_coords(x, 2.0 * x, x + 0.3, 2.0 * x - 0.2)
            })
            .collect();

        let result = DltSolver::solve(
            &NormalizedCorrespondences::from_pixels(&correspondences),
            &HomographyConfig::default(),
        );
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_dlt_accepts_noisy_overdetermined_data() {
        let h_true = Matrix3::new(1.05, 0.03, 20.0, -0.02, 0.97, -15.0, 2e-4, -1e-4, 1.0);
        let points = vec![
            Point2::new(-120.0, -80.0),
            Point2::new(30.0, -100.0),
            Point2::new(160.0, -40.0),
            Point2::new(-150.0, 20.0),
            Point2::new(10.0, 35.0),
            Point2::new(140.0, 70.0),
            Point2::new(-60.0, 130.0),
            Point2::new(90.0, 150.0),
        ];
        let mut correspondences = create_correspondences_from_homography(&h_true, &points);
        for (i, correspondence) in correspondences.iter_mut().enumerate() {
            let noise = 1e-3 * (i as f64 + 0.4).sin();
            correspondence.point2.x += noise;
            correspondence.point2.y -= 0.6 * noise;
        }
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

        let models = DltSolver::solve(&normalized, &HomographyConfig::default()).unwrap();
        let estimated = models[0];
        let average = correspondences
            .iter()
            .map(|corr| DltSolver::residual(&estimated, corr))
            .sum::<f64>()
            / correspondences.len() as f64;
        assert!(average < 0.01, "average symmetric error: {average}");
    }

    #[test]
    fn test_indexed_dlt_matches_slice_solver() {
        let h_true = Matrix3::new(1.05, 0.03, 20.0, -0.02, 0.97, -15.0, 2e-4, -1e-4, 1.0);
        let points = vec![
            Point2::new(-120.0, -80.0),
            Point2::new(30.0, -100.0),
            Point2::new(160.0, -40.0),
            Point2::new(-150.0, 20.0),
            Point2::new(10.0, 35.0),
            Point2::new(140.0, 70.0),
            Point2::new(-60.0, 130.0),
            Point2::new(90.0, 150.0),
        ];
        let correspondences = create_correspondences_from_homography(&h_true, &points);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);
        let indices: Vec<_> = (0..normalized.len()).rev().collect();
        let config = HomographyConfig::default();

        let slice = DltSolver::solve_single(normalized.as_slice(), &config)
            .unwrap()
            .unwrap();
        let indexed =
            DltSolver::solve_single_from_indices(&indices, normalized.as_slice(), &config)
                .unwrap()
                .unwrap();
        let normalized_slice = slice.0 / slice.0.norm();
        let normalized_indexed = indexed.0 / indexed.0.norm();
        let distance = (normalized_slice - normalized_indexed)
            .norm()
            .min((normalized_slice + normalized_indexed).norm());

        assert!(distance < 1e-8, "projective matrix distance: {distance}");
    }

    #[test]
    fn test_insufficient_points() {
        let correspondences = vec![
            PixelCorrespondence::from_coords(100.0, 100.0, 150.0, 150.0),
            PixelCorrespondence::from_coords(200.0, 100.0, 250.0, 150.0),
            PixelCorrespondence::from_coords(200.0, 200.0, 250.0, 250.0),
        ];

        let config = HomographyConfig::default();
        let result = DltSolver::solve(
            &NormalizedCorrespondences::from_pixels(&correspondences),
            &config,
        );

        assert!(result.is_err());
        if let Err(SolverError::InsufficientData { required, provided }) = result {
            assert_eq!(required, 4);
            assert_eq!(provided, 3);
        }
    }

    #[test]
    fn test_homography_validity() {
        // Valid homography (non-singular)
        let h_valid: Homography = Homography::new(Matrix3::identity());
        assert!(h_valid.is_valid());

        // Invalid homography (singular)
        let h_invalid: Homography = Homography::new(Matrix3::zeros());
        assert!(!h_invalid.is_valid());

        for scale in [1e-12, -1e-12, 1e12, -1e12] {
            assert!(Homography::new(Matrix3::identity() * scale).is_valid());
        }

        let rank_deficient = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 0.0));
        assert!(!Homography::new(rank_deficient).is_valid());

        let mut non_finite = Matrix3::identity();
        non_finite[(1, 1)] = f64::INFINITY;
        assert!(!Homography::new(non_finite).is_valid());

        let accepted = Homography::try_new(Matrix3::identity() * 2.0).unwrap();
        assert_eq!(accepted.as_ref(), &(Matrix3::identity() * 2.0));
        for invalid in [Matrix3::zeros(), rank_deficient, non_finite] {
            assert!(matches!(
                Homography::try_new(invalid),
                Err(SolverError::InvalidInput { .. })
            ));
        }
    }

    #[test]
    fn test_solver_residual_is_unsquared_symmetric_distance() {
        let h = Homography::new(Matrix3::identity());
        let corr = PixelCorrespondence::from_coords(0.0, 0.0, 3.0, 4.0);

        // Both transfer directions have a five-pixel error, so their combined
        // Euclidean distance is sqrt(5^2 + 5^2), not the squared value 50.
        let residual = DltSolver::residual(&h, &corr);
        assert!((residual - 50.0_f64.sqrt()).abs() < 1e-12);
    }
}
