// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 8-point algorithm for fundamental matrix estimation.
//!
//! Its teaching docs are on [`EightPointSolver`], where docs.rs shows them.

use crate::correspondence::PixelCorrespondence;
use crate::normalization::{
    CorrespondenceCoordinates, NormalizedCorrespondence, NormalizedCorrespondences,
};
use crate::residuals::{sampson_error_fundamental, sampson_error_fundamental_squared};
use crate::shared::epipolar_constraint::{
    EpipolarConstraint, EpipolarCorrespondence, solve_epipolar_ata, solve_epipolar_ata_from_rows,
};
use crate::solver_traits::{MultiSolutionSolver, SingleSolutionSolver, Solver};
use crate::{SolverError, SolverResult};

use super::{FundamentalMatrix, denormalize, normalized_model_is_conditioned};

/// Fundamental matrix solver using the 8-point algorithm.
///
/// Implements the normalized 8-point algorithm for estimating the fundamental
/// matrix from at least 8 pixel correspondences.
///
/// # How it works
///
/// Implements the normalized 8-point algorithm (R. Hartley, "In Defense of the
/// Eight-Point Algorithm", IEEE TPAMI, 1997).
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │                    8-point fundamental matrix solver                        │
/// │                  (Hartley's normalized algorithm, 1997)                     │
/// └─────────────────────────────────────────────────────────────────────────────┘
///
///     Input: 8+ pixel correspondences (p₁, p₂)
///            (Hartley-normalized: `NormalizedCorrespondences`)
///
///                    ┌───────────────────────────────────────┐
///                    │  Step 1: Build constraint matrix      │
///                    └───────────────────────────────────────┘
///                                      │
///                    For each correspondence (p₁ᵢ, p₂ᵢ) = ((x₁,y₁), (x₂,y₂)):
///                    [x₂x₁, x₂y₁, x₂, y₂x₁, y₂y₁, y₂, x₁, y₁, 1] · vec(F) = 0
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
///
///                    F = eigenvector with smallest eigenvalue
///                    Reshape 9-vector to 3×3 matrix F
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 3: Enforce rank-2 constraint    │
///                    └───────────────────────────────────────┘
///                                      │
///                    SVD: F = U·Σ·Vᵀ  where Σ = diag(σ₁, σ₂, σ₃)
///
///                    Fundamental matrix must have rank 2:
///                    Set σ₃ = 0:  Σ' = diag(σ₁, σ₂, 0)
///
///                    F' = U·Σ'·Vᵀ
///                                      │
///                                      ▼
///     Output: 1 fundamental matrix F' with enforced rank-2 constraint
///
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │  Complexity: O(n) constraint matrix + O(1) eigenvalue/SVD                   │
/// │  Memory: 9×9 normal matrix; no n×9 matrix is stored                         │
/// │  Note: Fits on normalized input; the public solve denormalizes              │
/// │  Comparison: Uses all n ≥ 8 points; the 7-point solver is minimal           │
/// └─────────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Normalized input
///
/// The fit runs on Hartley-normalized coordinates (Hartley 1997).
/// [`EightPointSolver::solve`] takes a [`NormalizedCorrespondences`] set, which carries the
/// `T1`/`T2` transforms, and returns at most one matrix in pixel coordinates,
/// `F = T2ᵀ F_n T1`. Robust estimation normalizes the whole dataset once rather than at every
/// draw.
///
/// See the usage example in [`crate::fundamental`].
#[derive(Debug, Clone, Default)]
pub struct EightPointSolver;

impl EightPointSolver {
    /// Return the solver's minimum correspondence count.
    #[must_use]
    pub fn min_sample_size() -> usize {
        <Self as Solver>::min_sample_size()
    }

    /// Evaluate the Sampson distance of a pixel-space model for one pixel correspondence.
    pub fn residual(model: &FundamentalMatrix, datum: &PixelCorrespondence) -> f64 {
        sampson_error_fundamental(&model.0, datum)
    }

    /// Validate the model's geometric invariants.
    #[must_use]
    pub fn is_valid(model: &FundamentalMatrix) -> bool {
        <Self as Solver>::is_valid(model)
    }

    /// Fit on normalized correspondences and return the pixel-space model.
    ///
    /// Conditioning is checked before denormalization. A pixel representation
    /// whose rank is unresolved in f64 is omitted from the returned models.
    pub fn solve(data: &NormalizedCorrespondences) -> SolverResult<Vec<FundamentalMatrix>> {
        let models = <Self as MultiSolutionSolver>::solve(data.as_slice(), &())?;
        Ok(models
            .iter()
            .map(|model| denormalize(model, data.normalization()))
            .filter(FundamentalMatrix::is_valid)
            .collect())
    }
}

/// Internal 8-point core shared by the normalized kernel and the raw adapter.
pub(crate) struct EightPointCore;

impl EightPointCore {
    /// Solve for the fundamental matrix in the coordinates of `correspondences`.
    ///
    /// This is the core algorithm; it neither normalizes nor denormalizes.
    pub fn solve_unnormalized<C: EpipolarCorrespondence<f64>>(
        correspondences: &[C],
    ) -> SolverResult<FundamentalMatrix> {
        if correspondences.len() < 8 {
            return Err(SolverError::InsufficientData {
                required: 8,
                provided: correspondences.len(),
            });
        }

        // Solve using A^T*A formulation with rank-2 constraint
        let f = solve_epipolar_ata(correspondences, EpipolarConstraint::Rank2)?;

        Ok(FundamentalMatrix::new(f))
    }

    fn solve_single<D>(data: &[D]) -> SolverResult<Option<FundamentalMatrix>>
    where
        D: CorrespondenceCoordinates + EpipolarCorrespondence<f64>,
    {
        if data.len() < 8 {
            return Err(SolverError::InsufficientData {
                required: 8,
                provided: data.len(),
            });
        }
        if !data
            .iter()
            .all(|datum| correspondence_is_finite(datum.coordinates()))
        {
            return Err(SolverError::InvalidInput {
                reason: "fundamental-matrix coordinates must be finite",
            });
        }

        optional_fundamental(Self::solve_unnormalized(data))
    }

    fn solve_single_from_indices<D: EpipolarCorrespondence<f64>>(
        indices: &[usize],
        data: &[D],
    ) -> SolverResult<Option<FundamentalMatrix>> {
        if indices.len() < 8 {
            return Err(SolverError::InsufficientData {
                required: 8,
                provided: indices.len(),
            });
        }
        let rows = indices.iter().map(|&index| {
            let correspondence = data.get(index).ok_or(SolverError::InvalidDataIndex {
                index,
                data_len: data.len(),
            })?;
            Ok(correspondence.to_constraint_row())
        });
        optional_fundamental(
            solve_epipolar_ata_from_rows(rows, EpipolarConstraint::Rank2)
                .map(FundamentalMatrix::new),
        )
    }
}

impl Solver for EightPointSolver {
    type Datum = NormalizedCorrespondence;
    type Model = FundamentalMatrix;
    type Config = ();
    type Unit = openmvg_core::Unitless;

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_fundamental(&model.0, datum.coordinates())
    }

    fn squared_residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_fundamental_squared(&model.0, datum.coordinates())
    }

    fn min_sample_size() -> usize {
        8
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.is_valid()
    }
}

/// Allocation-free single-solution implementation used by robust refinement.
impl SingleSolutionSolver for EightPointSolver {
    fn solve_single(
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        Ok(EightPointCore::solve_single(data)?.filter(normalized_model_is_conditioned))
    }

    fn solve_single_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        Ok(EightPointCore::solve_single_from_indices(indices, data)?
            .filter(normalized_model_is_conditioned))
    }
}

/// The 8-point kernel driven directly on raw pixel coordinates, without
/// Hartley normalization, as used by the compact classic robust loop
/// (`ClassicConfig::estimate_fundamental`).
pub(crate) struct UnnormalizedFundamentalSolver;

impl Solver for UnnormalizedFundamentalSolver {
    type Datum = PixelCorrespondence;
    type Model = FundamentalMatrix;
    type Config = ();
    type Unit = openmvg_core::Pixels;

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_fundamental(&model.0, datum)
    }

    fn squared_residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_fundamental_squared(&model.0, datum)
    }

    fn min_sample_size() -> usize {
        8
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.is_valid()
    }
}

impl SingleSolutionSolver for UnnormalizedFundamentalSolver {
    fn solve_single(
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        EightPointCore::solve_single(data)
    }

    fn solve_single_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        EightPointCore::solve_single_from_indices(indices, data)
    }
}

#[inline]
fn correspondence_is_finite(correspondence: &PixelCorrespondence) -> bool {
    correspondence
        .point1
        .coords
        .iter()
        .chain(correspondence.point2.coords.iter())
        .all(|value| value.is_finite())
}

fn optional_fundamental(
    result: SolverResult<FundamentalMatrix>,
) -> SolverResult<Option<FundamentalMatrix>> {
    match result {
        Ok(model) => Ok(Some(model)),
        Err(SolverError::DegenerateConfiguration { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Point2, Rotation3, Unit, Vector3};

    /// Create a ground-truth fundamental matrix from a known camera geometry.
    ///
    /// Camera 1: P₁ = K [I | 0]
    /// Camera 2: P₂ = K [R | t]
    /// F = K⁻ᵀ · [t]× · R · K⁻¹
    fn make_ground_truth_fundamental(
        focal: f64,
        cx: f64,
        cy: f64,
        rotation: &Rotation3<f64>,
        translation: &Vector3<f64>,
    ) -> Matrix3<f64> {
        let k = Matrix3::new(focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0);
        let k_inv = k.try_inverse().unwrap();
        let t = translation;
        let t_cross = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
        let r = rotation.matrix();
        k_inv.transpose() * t_cross * r * k_inv
    }

    /// Generate pixel correspondences by projecting 3D points through a camera pair.
    ///
    /// Camera 1: P₁ = K [I | 0]  (origin)
    /// Camera 2: P₂ = K [R | t]
    fn generate_correspondences_from_cameras(
        focal: f64,
        cx: f64,
        cy: f64,
        rotation: &Rotation3<f64>,
        translation: &Vector3<f64>,
        num_points: usize,
    ) -> Vec<PixelCorrespondence> {
        let k = Matrix3::new(focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0);
        let r = rotation.matrix();

        let mut correspondences = Vec::with_capacity(num_points);
        for i in 0..num_points {
            // 3D points in front of both cameras (Z > 0, varied positions)
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (num_points as f64);
            let radius = 0.5 + (i as f64) * 0.1;
            let world_pt = Vector3::new(
                radius * angle.cos(),
                radius * angle.sin(),
                4.0 + (i as f64) * 0.3,
            );

            // Project to camera 1: x₁ = K · X
            let proj1 = k * world_pt;
            let p1 = Point2::new(proj1.x / proj1.z, proj1.y / proj1.z);

            // Project to camera 2: x₂ = K · (R · X + t)
            let cam2_pt = r * world_pt + translation;
            let proj2 = k * cam2_pt;
            let p2 = Point2::new(proj2.x / proj2.z, proj2.y / proj2.z);

            correspondences.push(PixelCorrespondence::new(p1, p2));
        }

        correspondences
    }

    /// Compare two 3x3 matrices up to sign and scale.
    ///
    /// Returns the Frobenius distance after normalizing both matrices to unit norm
    /// and aligning signs (since F is defined only up to scale).
    fn matrix_distance_up_to_scale(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
        let a_norm = a / a.norm();
        let b_norm = b / b.norm();

        // F is defined up to scale, so compare both sign conventions
        let dist_pos = (a_norm - b_norm).norm();
        let dist_neg = (a_norm + b_norm).norm();
        dist_pos.min(dist_neg)
    }

    #[test]
    fn test_indexed_eight_point_matches_slice_solver() {
        let rotation = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::y()), 0.25);
        let translation = Vector3::new(1.0, 0.2, 0.0);
        let correspondences =
            generate_correspondences_from_cameras(500.0, 320.0, 240.0, &rotation, &translation, 20);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);
        let indices: Vec<_> = (0..normalized.len()).rev().collect();

        let slice = EightPointSolver::solve_single(normalized.as_slice(), &())
            .unwrap()
            .unwrap();
        let indexed =
            EightPointSolver::solve_single_from_indices(&indices, normalized.as_slice(), &())
                .unwrap()
                .unwrap();

        assert!(matrix_distance_up_to_scale(&slice.0, &indexed.0) < 1e-8);
    }

    /// Regression test: verify the solver returns F, NOT Fᵀ.
    ///
    /// This test uses an asymmetric camera geometry (rotation + translation)
    /// that produces F ≢ Fᵀ (not proportional). It then directly compares the
    /// estimated matrix against ground truth to ensure the correct orientation.
    ///
    /// Background: The epipolar constraint x₂ᵀ·F·x₁ = 0 uses a Kronecker-product
    /// constraint row whose layout must match the matrix reshaping convention.
    /// A row-major constraint row with column-major reshape produces Fᵀ instead
    /// of F. This test catches that class of bugs because:
    ///   - The ground truth F is asymmetric (F ≠ αFᵀ for any scalar α)
    ///   - We compare the estimated matrix directly against F_true vs F_trueᵀ
    ///   - The algebraic error x₂ᵀ·F_est·x₁ is NOT zero when F_est = Fᵀ and
    ///     F is asymmetric (unlike the skew-symmetric case where F = -Fᵀ ∝ Fᵀ)
    #[test]
    fn test_eight_point_recovers_correct_f_not_transpose() {
        // Camera geometry with a non-trivial rotation producing an asymmetric F
        let focal = 500.0;
        let cx = 320.0;
        let cy = 240.0;

        // 15° rotation around Y axis + non-trivial translation
        let rotation =
            Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::y()), 15.0_f64.to_radians());
        let translation = Vector3::new(1.0, 0.2, 0.0);

        let f_true = make_ground_truth_fundamental(focal, cx, cy, &rotation, &translation);
        let f_true_t = f_true.transpose();

        // Verify our ground truth is genuinely asymmetric: F ≢ Fᵀ
        let asymmetry = matrix_distance_up_to_scale(&f_true, &f_true_t);
        assert!(
            asymmetry > 0.1,
            "Test requires asymmetric F, but F ≈ Fᵀ (distance = {:.6}). \
             Use a camera geometry that produces an asymmetric fundamental matrix.",
            asymmetry
        );

        // Generate noiseless correspondences from the known camera pair
        let correspondences =
            generate_correspondences_from_cameras(focal, cx, cy, &rotation, &translation, 20);

        // The public solve normalizes, fits and denormalizes — as RANSAC does
        let normalized_data = NormalizedCorrespondences::from_pixels(&correspondences);

        let models = EightPointSolver::solve(&normalized_data)
            .expect("Solver should succeed on noiseless data");
        assert_eq!(models.len(), 1);

        let f_estimated = &models[0];

        // 1) F_est should match F_true (not its transpose)
        let dist_to_f = matrix_distance_up_to_scale(&f_estimated.0, &f_true);
        let dist_to_ft = matrix_distance_up_to_scale(&f_estimated.0, &f_true_t);

        assert!(
            dist_to_f < 0.05,
            "Estimated F does not match ground truth F.\n\
             Distance to F_true:  {:.6}\n\
             Distance to F_trueᵀ: {:.6}\n\
             If dist_to_ft < dist_to_f, the solver is returning Fᵀ instead of F \
             (row-major/column-major reshape mismatch).",
            dist_to_f,
            dist_to_ft,
        );

        assert!(
            dist_to_f < dist_to_ft,
            "Estimated F is closer to F_trueᵀ than to F_true!\n\
             Distance to F_true:  {:.6}\n\
             Distance to F_trueᵀ: {:.6}\n\
             This indicates the solver is returning the TRANSPOSE of the \
             fundamental matrix (column-major vs row-major vectorization bug).",
            dist_to_f,
            dist_to_ft,
        );

        // 2) Algebraic error should be near zero
        for (i, corr) in correspondences.iter().enumerate() {
            let x1 = Vector3::new(corr.point1.x, corr.point1.y, 1.0);
            let x2 = Vector3::new(corr.point2.x, corr.point2.y, 1.0);
            let error = (x2.transpose() * f_estimated.0 * x1)[(0, 0)].abs();
            assert!(
                error < 1e-6,
                "Algebraic error at correspondence {}: {:.2e} (expected < 1e-6)",
                i,
                error
            );
        }
    }

    #[test]
    fn test_insufficient_points() {
        let correspondences: Vec<PixelCorrespondence> = (0..5)
            .map(|i| {
                PixelCorrespondence::from_coords(
                    100.0 + i as f64 * 10.0,
                    200.0,
                    150.0 + i as f64 * 10.0,
                    250.0,
                )
            })
            .collect();

        let result =
            EightPointSolver::solve(&NormalizedCorrespondences::from_pixels(&correspondences));

        assert!(result.is_err());
        if let Err(SolverError::InsufficientData { required, provided }) = result {
            assert_eq!(required, 8);
            assert_eq!(provided, 5);
        } else {
            panic!("Expected InsufficientData error");
        }
    }
}
