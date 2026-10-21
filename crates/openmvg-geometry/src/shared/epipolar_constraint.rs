// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared epipolar constraint system for essential and fundamental matrix estimation.
//!
//! This module provides the A^T*A formulation for solving epipolar geometry problems.
//! It stores only a 9x9 normal matrix, independent of the number of
//! correspondences, which suits RANSAC loops.

use nalgebra::{Matrix3, RealField, SMatrix, SVector};
use num_traits::ToPrimitive;

use crate::{SolverError, SolverResult};

/// 9x9 matrix type used for the epipolar normal matrix.
pub type Matrix9x9<T> = SMatrix<T, 9, 9>;
pub type Vector9<T> = SVector<T, 9>;

/// Constraint enforcement strategy for epipolar matrices.
///
/// Determines how to project the least-squares solution onto the valid constraint manifold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EpipolarConstraint {
    /// Enforce the rank-2 constraint by setting the smallest singular value to zero.
    ///
    /// Use for the fundamental matrix. The result has singular values
    /// [sigma1, sigma2, 0] with sigma1 >= sigma2 > 0.
    #[default]
    Rank2,

    /// Enforce the essential matrix constraints: two equal singular values and one zero.
    ///
    /// Use for the essential matrix. The result has singular values
    /// [sigma, sigma, 0] with sigma > 0, the mean of the two largest input values.
    Essential,
}

/// Trait for types that can provide epipolar constraint rows.
///
/// Provides one interface for both essential and fundamental matrix estimation.
pub trait EpipolarCorrespondence<T: RealField> {
    /// Convert the correspondence to a constraint row for the epipolar equation.
    ///
    /// Returns a 9-element vector representing the vectorized constraint.
    /// Encodes the constraint x2^T * E/F * x1 = 0 as a linear system.
    fn to_constraint_row(&self) -> Vector9<T>;
}

impl EpipolarCorrespondence<f64> for crate::correspondence::BearingCorrespondence {
    fn to_constraint_row(&self) -> Vector9<f64> {
        // Cache the bearing components in local variables so the compiler can
        // keep them in registers and vectorize the products. Unlike
        // PixelCorrespondence, where z = 1 allows skipping multiplications,
        // bearings have arbitrary z values, so all 9 products are needed.
        let x1_x = self.bearing1().x;
        let x1_y = self.bearing1().y;
        let x1_z = self.bearing1().z;
        let x2_x = self.bearing2().x;
        let x2_y = self.bearing2().y;
        let x2_z = self.bearing2().z;

        // Kronecker product: x1 ⊗ x2 (column-major vectorization of E)
        // For epipolar constraint b₂ᵀ · E · b₁ = 0, with E vectorized column-major:
        // Row 1: x1_x * [x2_x, x2_y, x2_z]
        // Row 2: x1_y * [x2_x, x2_y, x2_z]
        // Row 3: x1_z * [x2_x, x2_y, x2_z]
        Vector9::from([
            x1_x * x2_x,
            x1_x * x2_y,
            x1_x * x2_z,
            x1_y * x2_x,
            x1_y * x2_y,
            x1_y * x2_z,
            x1_z * x2_x,
            x1_z * x2_y,
            x1_z * x2_z,
        ])
    }
}

impl EpipolarCorrespondence<f64> for &crate::correspondence::BearingCorrespondence {
    fn to_constraint_row(&self) -> Vector9<f64> {
        (*self).to_constraint_row()
    }
}

impl EpipolarCorrespondence<f64> for crate::correspondence::PixelCorrespondence {
    fn to_constraint_row(&self) -> Vector9<f64> {
        // Homogeneous z = 1 for pixel coordinates, so the multiplications by
        // 1.0 are skipped: 4 multiplications instead of 9.
        let x1_x = self.point1.x;
        let x1_y = self.point1.y;
        let x2_x = self.point2.x;
        let x2_y = self.point2.y;

        // Column-major vectorization of F (consistent with nalgebra's from_iterator):
        // For epipolar constraint x₂ᵀ · F · x₁ = 0, with F vectorized column-major,
        // the constraint row a satisfies aᵀ · vec(F) = x₂ᵀ · F · x₁
        // where a[j*3 + i] = x₂[i] · x₁[j]
        //
        // Col 0 of F (j=0): x₁[0] * [x₂[0], x₂[1], x₂[2]]
        // Col 1 of F (j=1): x₁[1] * [x₂[0], x₂[1], x₂[2]]
        // Col 2 of F (j=2): x₁[2] * [x₂[0], x₂[1], x₂[2]]  (x₁[2]=1 for pixels)
        Vector9::from([
            x2_x * x1_x, // F[0,0]: x2_x * x1_x
            x2_y * x1_x, // F[1,0]: x2_y * x1_x
            x1_x,        // F[2,0]: 1 * x1_x (no multiply)
            x2_x * x1_y, // F[0,1]: x2_x * x1_y
            x2_y * x1_y, // F[1,1]: x2_y * x1_y
            x1_y,        // F[2,1]: 1 * x1_y (no multiply)
            x2_x,        // F[0,2]: x2_x * 1 (no multiply)
            x2_y,        // F[1,2]: x2_y * 1 (no multiply)
            1.0,         // F[2,2]: 1 * 1 (constant)
        ])
    }
}

impl EpipolarCorrespondence<f64> for &crate::correspondence::PixelCorrespondence {
    fn to_constraint_row(&self) -> Vector9<f64> {
        (*self).to_constraint_row()
    }
}

/// Build the A^T*A matrix from correspondences.
///
/// Computes the 9x9 symmetric positive semi-definite normal matrix directly
/// from at least 8 correspondences, without building the n×9 matrix A.
pub fn build_ata<T: RealField + Copy, C: EpipolarCorrespondence<T>>(
    correspondences: &[C],
) -> SolverResult<Matrix9x9<T>> {
    build_ata_from_rows(
        correspondences
            .iter()
            .map(|correspondence| Ok(correspondence.to_constraint_row())),
    )
}

/// Build `A^T A` from constraint rows supplied lazily.
///
/// This internal streaming form lets camera-specific solvers normalize each
/// correspondence directly into the accumulator without materializing a
/// second correspondence buffer.
pub fn build_ata_from_rows<T, I>(rows: I) -> SolverResult<Matrix9x9<T>>
where
    T: RealField + Copy,
    I: IntoIterator<Item = SolverResult<Vector9<T>>>,
{
    // A flat array lets LLVM auto-vectorize the accumulation.
    let mut ata_flat = [T::zero(); 81]; // 9x9 flattened, row-major

    // Accumulate outer products: A^T*A = sum(a_i * a_i^T)
    let mut count = 0usize;
    for row in rows {
        let a = row?;
        count += 1;
        if !a.iter().all(|value| value.is_finite()) {
            return Err(SolverError::DegenerateConfiguration {
                reason: "epipolar constraints contain non-finite coordinates".to_string(),
            });
        }

        // Upper triangle only: 45 of the 81 products. The inner loop still
        // walks contiguous memory for auto-vectorization.
        for i in 0..9 {
            let a_i = a[i];
            let row_start = i * 9;
            for j in i..9 {
                ata_flat[row_start + j] += a_i * a[j];
            }
        }
    }
    // Mirror the lower triangle. `a_i * a_j == a_j * a_i` exactly and each
    // entry sums the rows in the same order, so this is bit-identical to
    // accumulating the full outer product.
    for i in 1..9 {
        for j in 0..i {
            ata_flat[i * 9 + j] = ata_flat[j * 9 + i];
        }
    }

    if count < 8 {
        return Err(SolverError::InsufficientData {
            required: 8,
            provided: count,
        });
    }

    // Convert flat array to Matrix9x9 (row-major order matches our layout)
    let ata = Matrix9x9::from_row_slice(&ata_flat);
    if !ata.iter().all(|value| value.is_finite()) {
        return Err(SolverError::DegenerateConfiguration {
            reason: "epipolar normal matrix overflowed".to_string(),
        });
    }
    Ok(ata)
}

/// Solve for epipolar matrix using the A^T*A formulation.
///
/// Finds the eigenvector of A^T*A corresponding to the smallest eigenvalue,
/// then applies `constraint` (`Rank2` for F, `Essential` for E) and returns the
/// constrained 3x3 matrix. Requires at least 8 correspondences.
pub fn solve_epipolar_ata<T: RealField + Copy + ToPrimitive, C: EpipolarCorrespondence<T>>(
    correspondences: &[C],
    constraint: EpipolarConstraint,
) -> SolverResult<Matrix3<T>> {
    let ata = build_ata(correspondences)?;
    solve_epipolar_normal_matrix(ata, constraint)
}

/// Solve an epipolar system from lazily generated constraint rows.
pub fn solve_epipolar_ata_from_rows<T, I>(
    rows: I,
    constraint: EpipolarConstraint,
) -> SolverResult<Matrix3<T>>
where
    T: RealField + Copy + ToPrimitive,
    I: IntoIterator<Item = SolverResult<Vector9<T>>>,
{
    let ata = build_ata_from_rows(rows)?;
    solve_epipolar_normal_matrix(ata, constraint)
}

fn solve_epipolar_normal_matrix<T: RealField + Copy + ToPrimitive>(
    ata: Matrix9x9<T>,
    constraint: EpipolarConstraint,
) -> SolverResult<Matrix3<T>> {
    // Solve eigenvalue problem and verify that the least-squares direction is
    // unique.  Eight correspondences require rank at least eight; repeated or
    // critical configurations leave two or more indistinguishable directions.
    let eig = ata.symmetric_eigen();
    let mut order = [0usize, 1, 2, 3, 4, 5, 6, 7, 8];
    order.sort_by(|&a, &b| {
        let a_abs = eig.eigenvalues[a].to_f64().unwrap_or(f64::NAN).abs();
        let b_abs = eig.eigenvalues[b].to_f64().unwrap_or(f64::NAN).abs();
        a_abs.total_cmp(&b_abs)
    });
    let eigenvalues: [f64; 9] =
        std::array::from_fn(|i| eig.eigenvalues[order[i]].to_f64().unwrap_or(f64::NAN).abs());
    if eigenvalues.iter().any(|value| !value.is_finite()) || eigenvalues[8] <= 0.0 {
        return Err(SolverError::DegenerateConfiguration {
            reason: "epipolar eigendecomposition produced a non-finite or zero spectrum"
                .to_string(),
        });
    }
    let tolerance = eigenvalues[8] * f64::EPSILON * 9.0 * 64.0;
    if eigenvalues[1] <= tolerance || eigenvalues[1] - eigenvalues[0] <= tolerance {
        return Err(SolverError::DegenerateConfiguration {
            reason: "epipolar constraints have an ambiguous null space".to_string(),
        });
    }

    let f_vec = eig.eigenvectors.column(order[0]);
    let f_est = Matrix3::from_iterator(f_vec.iter().copied());
    if !f_est.iter().all(|value| value.is_finite()) {
        return Err(SolverError::DegenerateConfiguration {
            reason: "epipolar null-space vector is non-finite".to_string(),
        });
    }

    // Apply constraint
    let constrained = match constraint {
        EpipolarConstraint::Rank2 => {
            // Enforce rank-2: set smallest singular value to zero
            let svd = f_est.svd(true, true);
            let u = svd.u.ok_or(SolverError::SvdFailed)?;
            let vt = svd.v_t.ok_or(SolverError::SvdFailed)?;
            let mut s = svd.singular_values;

            s[2] = T::zero();

            Ok(u * Matrix3::from_diagonal(&s) * vt)
        }

        EpipolarConstraint::Essential => {
            // Enforce essential: two equal singular values + one zero
            let svd = f_est.svd(true, true);
            let u = svd.u.ok_or(SolverError::SvdFailed)?;
            let vt = svd.v_t.ok_or(SolverError::SvdFailed)?;
            let s = svd.singular_values;

            // Average the two largest singular values
            let two = T::one() + T::one();
            let sigma_avg = (s[0] + s[1]) / two;
            let s_essential = nalgebra::Vector3::new(sigma_avg, sigma_avg, T::zero());

            Ok(u * Matrix3::from_diagonal(&s_essential) * vt)
        }
    }?;

    if !constrained.iter().all(|value| value.is_finite()) {
        return Err(SolverError::SvdFailed);
    }
    Ok(constrained)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correspondence::{BearingCorrespondence, PixelCorrespondence};
    use nalgebra::{Point2, Rotation3, Vector3};

    fn two_view_correspondences() -> (Vec<PixelCorrespondence>, Vec<BearingCorrespondence>) {
        let rotation = Rotation3::from_euler_angles(0.04, -0.12, 0.07);
        let translation = Vector3::new(0.45, -0.12, 0.08);
        let points = [
            Vector3::new(-1.2, -0.7, 4.0),
            Vector3::new(0.3, -1.1, 5.2),
            Vector3::new(1.4, -0.2, 6.1),
            Vector3::new(-0.8, 1.3, 4.7),
            Vector3::new(0.6, 0.9, 7.0),
            Vector3::new(1.7, 1.1, 5.5),
            Vector3::new(-1.5, 0.4, 6.6),
            Vector3::new(0.1, 1.8, 5.9),
            Vector3::new(1.0, -1.4, 7.4),
            Vector3::new(-0.4, -1.7, 5.7),
            Vector3::new(1.9, 0.2, 8.0),
            Vector3::new(-1.8, 1.5, 7.2),
        ];

        let bearings: Vec<_> = points
            .iter()
            .map(|point| {
                let second = rotation * point + translation;
                BearingCorrespondence::try_new(*point, second).unwrap()
            })
            .collect();
        let pixels = points
            .iter()
            .enumerate()
            .map(|(i, point)| {
                let second = rotation * point + translation;
                // Add realistic, deterministic measurement noise to the second view.
                let noise = 1e-5 * (i as f64 + 0.3).sin();
                PixelCorrespondence::from_coords(
                    point.x / point.z,
                    point.y / point.z,
                    second.x / second.z + noise,
                    second.y / second.z - 0.7 * noise,
                )
            })
            .collect();
        (pixels, bearings)
    }

    #[test]
    fn test_pixel_constraint_row() {
        let corr = PixelCorrespondence::new(Point2::new(100.0, 200.0), Point2::new(150.0, 250.0));
        let row = corr.to_constraint_row();

        assert_eq!(row.len(), 9);
        // Last element should be 1*1 = 1 (homogeneous coords)
        assert_eq!(row[8], 1.0);
        // Column-major order: a[j*3+i] = x2[i] * x1[j]
        // row[0] = x2_x * x1_x = 150 * 100 = 15000
        assert_eq!(row[0], 15000.0);
        // row[1] = x2_y * x1_x = 250 * 100 = 25000
        assert_eq!(row[1], 25000.0);
        // row[2] = 1 * x1_x = 100
        assert_eq!(row[2], 100.0);
        // row[3] = x2_x * x1_y = 150 * 200 = 30000
        assert_eq!(row[3], 30000.0);
        // row[6] = x2_x * 1 = 150
        assert_eq!(row[6], 150.0);
    }

    #[test]
    fn test_bearing_constraint_row() {
        let corr: BearingCorrespondence = BearingCorrespondence::try_new(
            Vector3::new(1.0, 0.0, 1.0).normalize(),
            Vector3::new(0.9, 0.1, 1.0).normalize(),
        )
        .unwrap();
        let row = corr.to_constraint_row();

        assert_eq!(row.len(), 9);
        for val in row.iter() {
            assert!(val.is_finite());
        }
    }

    fn full_outer_product_reference(rows: &[Vector9<f64>]) -> [f64; 81] {
        let mut ata = [0.0; 81];
        for a in rows {
            for i in 0..9 {
                for j in 0..9 {
                    ata[i * 9 + j] += a[i] * a[j];
                }
            }
        }
        ata
    }

    #[test]
    fn triangular_accumulation_is_bit_identical_to_the_full_outer_product() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
        };
        let rows: Vec<Vector9<f64>> = (0..257)
            .map(|_| Vector9::from_fn(|_, _| next() * 1e3))
            .collect();
        let ata = build_ata_from_rows(rows.iter().copied().map(Ok)).unwrap();
        let reference = full_outer_product_reference(&rows);
        for i in 0..9 {
            for j in 0..9 {
                assert_eq!(ata[(i, j)].to_bits(), reference[i * 9 + j].to_bits());
            }
        }
    }

    #[test]
    fn test_build_ata_symmetric() {
        let correspondences: Vec<BearingCorrespondence> = (0..10)
            .map(|i| {
                let t = i as f64 * 0.1;
                BearingCorrespondence::try_new(
                    Vector3::new(t, t * 0.5, 1.0).normalize(),
                    Vector3::new(t + 0.1, t * 0.5 + 0.05, 1.0).normalize(),
                )
                .unwrap()
            })
            .collect();

        let ata = build_ata(&correspondences).unwrap();

        // Check symmetry
        for i in 0..9 {
            for j in 0..9 {
                assert!((ata[(i, j)] - ata[(j, i)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_solve_fundamental() {
        let (correspondences, _) = two_view_correspondences();

        let f = solve_epipolar_ata(&correspondences, EpipolarConstraint::Rank2).unwrap();

        // Check rank-2 (det should be ~0)
        assert!(f.determinant().abs() < 1e-10);
    }

    #[test]
    fn test_solve_essential() {
        let (_, correspondences) = two_view_correspondences();

        let e = solve_epipolar_ata(&correspondences, EpipolarConstraint::Essential).unwrap();

        // Check essential constraint: two equal singular values + one zero
        let svd = e.svd(false, false);
        let s = svd.singular_values;

        assert!(s[2].abs() < 1e-10);
        assert!((s[0] - s[1]).abs() / s[0] < 1e-6);
    }

    #[test]
    fn test_repeated_correspondences_have_ambiguous_nullspace() {
        let correspondence = PixelCorrespondence::from_coords(0.1, -0.2, 0.15, -0.18);
        let result = solve_epipolar_ata(&vec![correspondence; 8], EpipolarConstraint::Rank2);
        assert!(matches!(
            result,
            Err(SolverError::DegenerateConfiguration { .. })
        ));
    }

    #[test]
    fn test_insufficient_correspondences() {
        let correspondences = vec![
            BearingCorrespondence::try_new(
                Vector3::new(1.0, 0.0, 1.0),
                Vector3::new(0.9, 0.1, 1.0),
            )
            .unwrap(),
        ];

        let result = solve_epipolar_ata(&correspondences, EpipolarConstraint::Rank2);
        assert!(result.is_err());

        if let Err(SolverError::InsufficientData { required, provided }) = result {
            assert_eq!(required, 8);
            assert_eq!(provided, 1);
        } else {
            panic!("Expected InsufficientData error");
        }
    }
}
