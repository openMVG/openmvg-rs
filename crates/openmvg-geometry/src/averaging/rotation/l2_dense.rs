// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # L2 dense rotation averaging
//!
//! Its teaching docs are on [`average_rotations_dense`], where docs.rs shows them.

use nalgebra::{DMatrix, Matrix3, Rotation3};
use openmvg_core::model::{EstimationError, EstimationResult, RelativeRotation, ViewId};
use std::collections::BTreeMap;

use crate::averaging::{RotationConfig, RotationEstimate, RotationSolver};

/// L2 dense rotation averaging solver using eigendecomposition.
#[derive(Debug, Clone, Default)]
pub struct L2DenseRotationSolver;

impl L2DenseRotationSolver {
    pub fn new() -> Self {
        Self
    }

    /// Project a 3x3 matrix to the closest rotation matrix using SVD.
    ///
    /// Public for testing.
    pub fn closest_rotation(m: &Matrix3<f64>) -> EstimationResult<Rotation3<f64>> {
        let norm = m.norm();
        if norm < 1e-10 || !norm.is_finite() {
            return Err(EstimationError::Numerical(format!(
                "degenerate matrix with norm {}",
                norm
            )));
        }

        let svd = m.svd(true, true);
        let u = svd
            .u
            .ok_or_else(|| EstimationError::Numerical("SVD failed".to_string()))?;
        let v_t = svd
            .v_t
            .ok_or_else(|| EstimationError::Numerical("SVD failed".to_string()))?;

        let uv_t = u * v_t;
        let mut orientation = Matrix3::identity();
        if uv_t.determinant() < 0.0 {
            // Kabsch/orthogonal-Procrustes correction: flip only the singular
            // direction with the smallest singular value. Multiplying the
            // whole matrix by -1 produces a rotation, but not the closest one.
            orientation[(2, 2)] = -1.0;
        }
        let r = u * orientation * v_t;

        if !r.iter().all(|x| x.is_finite()) {
            return Err(EstimationError::Numerical(
                "rotation contains non-finite values".to_string(),
            ));
        }

        Ok(Rotation3::from_matrix_unchecked(r))
    }

    /// Build the view index mapping (view_id -> contiguous 0-based index).
    fn build_view_indices(
        relative_rotations: &[RelativeRotation],
    ) -> (BTreeMap<ViewId, usize>, Vec<ViewId>) {
        let mut view_set = std::collections::BTreeSet::new();
        for rel in relative_rotations {
            view_set.insert(rel.view_i());
            view_set.insert(rel.view_j());
        }
        let views: Vec<ViewId> = view_set.into_iter().collect();
        let view_to_idx: BTreeMap<ViewId, usize> =
            views.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        (view_to_idx, views)
    }

    /// Build the normal matrix `A^T * A` directly for the system `A*x = 0`.
    ///
    /// Constraint: sqrt(w) * (R_j - R_ij * R_i) = 0
    ///
    /// Each edge contributes four 3x3 blocks to the normal matrix:
    /// - `(i, i) += w * R_ij^T * R_ij`
    /// - `(i, j) -= w * R_ij^T`
    /// - `(j, i) -= w * R_ij`
    /// - `(j, j) += w * I`
    ///
    /// Direct assembly avoids the `3m x 3n` constraint matrix and its dense
    /// matrix multiplication. Keeping `R_ij^T * R_ij` explicit, rather than
    /// replacing it with identity, also preserves the exact algebra for input
    /// matrices whose orthogonality is only approximate.
    fn build_normal_matrix(
        relative_rotations: &[RelativeRotation],
        view_to_idx: &BTreeMap<ViewId, usize>,
        num_cameras: usize,
    ) -> DMatrix<f64> {
        let dimension = num_cameras * 3;
        let mut normal = DMatrix::<f64>::zeros(dimension, dimension);

        for rel in relative_rotations {
            let i = view_to_idx[&rel.view_i()];
            let j = view_to_idx[&rel.view_j()];
            let rotation = rel.rotation();
            let r_ij = rotation.matrix();
            let weight = rel.weight();
            let rtr = r_ij.transpose() * r_ij;

            for row in 0..3 {
                for col in 0..3 {
                    normal[(3 * i + row, 3 * i + col)] += weight * rtr[(row, col)];
                    normal[(3 * i + row, 3 * j + col)] -= weight * r_ij[(col, row)];
                    normal[(3 * j + row, 3 * i + col)] -= weight * r_ij[(row, col)];
                }
                normal[(3 * j + row, 3 * j + row)] += weight;
            }
        }

        normal
    }

    /// Compute the average angular error.
    fn compute_residual(
        rotations: &BTreeMap<ViewId, Rotation3<f64>>,
        relative_rotations: &[RelativeRotation],
    ) -> f64 {
        super::mean_angular_residual(rotations, relative_rotations)
    }
}

impl RotationSolver for L2DenseRotationSolver {
    type Workspace = ();

    fn solve(
        &self,
        relative_rotations: &[RelativeRotation],
        _config: &RotationConfig,
    ) -> EstimationResult<RotationEstimate> {
        if relative_rotations.is_empty() {
            return Err(EstimationError::InsufficientData {
                required: 1,
                provided: 0,
            });
        }
        for relative_rotation in relative_rotations {
            relative_rotation.validate()?;
        }

        let (view_to_idx, views) = Self::build_view_indices(relative_rotations);
        let num_cameras = views.len();

        if num_cameras < 2 {
            return Err(EstimationError::InsufficientData {
                required: 2,
                provided: num_cameras,
            });
        }

        // Assemble A^T * A directly in O(m) block operations. Constructing A
        // first would make this step a dense 3n-by-3m matrix multiplication.
        let ata = Self::build_normal_matrix(relative_rotations, &view_to_idx, num_cameras);

        // Eigendecomposition of A^T * A
        let eigen = ata.symmetric_eigen();
        let eigenvalues = eigen.eigenvalues.as_slice();

        // Sort eigenvalue indices by absolute value (smallest first)
        let mut eig_pairs: Vec<(usize, f64)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        eig_pairs.sort_by(|a, b| a.1.abs().total_cmp(&b.1.abs()));

        // Extract the 3 smallest eigenvector indices
        let idx0 = eig_pairs[0].0;
        let idx1 = eig_pairs[1].0;
        let idx2 = eig_pairs[2].0;

        let nullspace_vec0 = eigen.eigenvectors.column(idx0);
        let nullspace_vec1 = eigen.eigenvectors.column(idx1);
        let nullspace_vec2 = eigen.eigenvectors.column(idx2);
        // Camera `c`'s 3×3 block: rows `3c..3c + 3` of the three null vectors.
        let nullspace = [&nullspace_vec0, &nullspace_vec1, &nullspace_vec2];
        let nullspace_block = |base: usize| Matrix3::from_fn(|row, col| nullspace[col][base + row]);

        // The 3 eigenvectors form an arbitrary orthonormal basis for the nullspace.
        // The camera rotations are a specific 3D rotation of this basis.
        // We need to find the 3×3 transformation matrix T such that:
        //   [v0_block, v1_block, v2_block] * T ≈ rotation matrix for each camera.
        //
        // Strategy: Find the camera with the best-conditioned 3×3 block, use it
        // to compute T, then apply to all cameras. This is much more robust than
        // always using camera 0.
        //
        // The eigenvector sign freedom needs no explicit search: for any sign
        // matrix S = diag(±1, ±1, ±1), the anchor pseudo-inverse below gives
        // T = pinv(V_anchor·S) = S·pinv(V_anchor), so the reconstruction
        // R_c = (V_c·S)·T = V_c·S·S·pinv(V_anchor) = V_c·pinv(V_anchor) is
        // independent of S (S² = I). The null-space columns are used as-is.

        // Extract the anchor camera's 3×3 block for basis transformation.
        // Use camera 0 (will be gauge-fixed to identity). If degenerate,
        // try subsequent cameras until one works.
        let mut anchor_cam = 0;
        let mut v_anchor_block;
        let mut svd;
        loop {
            v_anchor_block = nullspace_block(3 * anchor_cam);
            svd = v_anchor_block.svd(true, true);
            let s = &svd.singular_values;
            if s[0] > 1e-10 && s[2] / s[0] > 1e-6 {
                break; // Good condition — use this camera
            }
            anchor_cam += 1;
            if anchor_cam >= num_cameras {
                return Err(EstimationError::Numerical(format!(
                    "all {} cameras have degenerate null-space blocks",
                    num_cameras
                )));
            }
        }

        // Compute T via SVD pseudo-inverse
        let u = svd
            .u
            .ok_or_else(|| EstimationError::Numerical("SVD of anchor block failed".into()))?;
        let v_t = svd
            .v_t
            .ok_or_else(|| EstimationError::Numerical("SVD of anchor block failed".into()))?;
        let s = &svd.singular_values;

        // T = V * S^{-1} * U^T (pseudo-inverse of V0_block)
        let s_inv =
            Matrix3::from_diagonal(&nalgebra::Vector3::new(1.0 / s[0], 1.0 / s[1], 1.0 / s[2]));
        let t_mat = v_t.transpose() * s_inv * u.transpose();

        // Reconstruct rotation matrices: R_i = [v0_block_i, v1_block_i, v2_block_i] * T
        let mut global_rotations = BTreeMap::new();
        for (cam_idx, &view_id) in views.iter().enumerate() {
            let vi_block = nullspace_block(3 * cam_idx);

            let rotation_mat = vi_block * t_mat;

            // Project to closest SO(3) rotation
            match Self::closest_rotation(&rotation_mat) {
                Ok(rotation) => {
                    global_rotations.insert(view_id, rotation);
                }
                Err(_) => {
                    // Degenerate eigenvector block for this camera — use identity as fallback
                    global_rotations.insert(view_id, Rotation3::identity());
                }
            }
        }

        // Force the anchor camera to be Identity by applying its inverse to all rotations
        let r_anchor = global_rotations[&views[anchor_cam]];
        let r_anchor_inv = r_anchor.transpose();

        // Verify r_anchor_inv is valid
        if !r_anchor_inv.matrix().iter().all(|x| x.is_finite()) {
            return Err(EstimationError::Numerical(
                "anchor rotation inverse contains NaN".to_string(),
            ));
        }

        for rotation in global_rotations.values_mut() {
            *rotation *= r_anchor_inv;
        }

        // Validate all rotations are proper SO(3) matrices
        for (&view_id, rotation) in &global_rotations {
            let m = rotation.matrix();
            let det = m.determinant();
            let rtr = m.transpose() * m;
            let ortho_error = (rtr - Matrix3::identity()).norm();

            if (det - 1.0).abs() > 0.01 || ortho_error > 0.01 {
                return Err(EstimationError::Numerical(format!(
                    "invalid rotation for view {}: det={}, ortho_error={}",
                    view_id, det, ortho_error
                )));
            }
        }

        let residual = Self::compute_residual(&global_rotations, relative_rotations);

        Ok(RotationEstimate {
            rotations: global_rotations,
            iteration_count: 1,
            residual: crate::averaging::checked_angular_residual(residual)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_normal_matrix_matches_explicit_constraint_product() {
        let relative_rotations = vec![
            RelativeRotation::try_new(
                ViewId::new(2),
                ViewId::new(7),
                Rotation3::from_euler_angles(0.2, -0.1, 0.3),
                0.4,
            )
            .unwrap(),
            RelativeRotation::try_new(
                ViewId::new(7),
                ViewId::new(11),
                Rotation3::from_euler_angles(-0.3, 0.5, 0.1),
                1.5,
            )
            .unwrap(),
            RelativeRotation::try_new(
                ViewId::new(2),
                ViewId::new(11),
                Rotation3::from_euler_angles(0.6, 0.2, -0.4),
                2.0,
            )
            .unwrap(),
        ];
        let (view_to_idx, views) = L2DenseRotationSolver::build_view_indices(&relative_rotations);
        let dimension = 3 * views.len();
        let mut constraints = DMatrix::<f64>::zeros(3 * relative_rotations.len(), dimension);

        for (edge, rel) in relative_rotations.iter().enumerate() {
            let i = view_to_idx[&rel.view_i()];
            let j = view_to_idx[&rel.view_j()];
            let rotation = rel.rotation();
            let sqrt_weight = rel.weight().sqrt();
            for row in 0..3 {
                for col in 0..3 {
                    constraints[(3 * edge + row, 3 * i + col)] =
                        -sqrt_weight * rotation.matrix()[(row, col)];
                }
                constraints[(3 * edge + row, 3 * j + row)] = sqrt_weight;
            }
        }

        let expected = constraints.transpose() * constraints;
        let actual = L2DenseRotationSolver::build_normal_matrix(
            &relative_rotations,
            &view_to_idx,
            views.len(),
        );

        assert!(
            (&actual - &expected).norm() < 1e-12,
            "direct assembly differs from A^T A by {}",
            (&actual - expected).norm()
        );
        assert!((&actual - actual.transpose()).norm() < 1e-14);
    }

    #[test]
    fn test_project_to_so3_identity() {
        let mat = Matrix3::identity();
        let rot = L2DenseRotationSolver::closest_rotation(&mat).unwrap();
        assert!((rot.matrix() - Matrix3::identity()).norm() < 1e-10);
    }

    #[test]
    fn test_project_to_so3_noisy() {
        let rot = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
        let noisy =
            rot.matrix() + Matrix3::new(0.01, -0.01, 0.02, 0.01, -0.02, 0.01, -0.01, 0.02, -0.01);
        let projected = L2DenseRotationSolver::closest_rotation(&noisy).unwrap();

        let det = projected.matrix().determinant();
        assert!((det - 1.0).abs() < 1e-10, "Determinant should be 1");

        let rtr = projected.matrix().transpose() * projected.matrix();
        assert!(
            (rtr - Matrix3::identity()).norm() < 1e-10,
            "R^T R should be I"
        );
    }

    #[test]
    fn test_project_to_so3_reflection_is_nearest_rotation() {
        // Distinct singular values make the Procrustes optimum unique.  The
        // nearest rotation is I; negating U*V^T would instead flip two axes.
        let reflected = Matrix3::new(2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.1);
        let projected = L2DenseRotationSolver::closest_rotation(&reflected).unwrap();

        assert!((projected.matrix() - Matrix3::identity()).norm() < 1e-12);
        assert!((projected.matrix().determinant() - 1.0).abs() < 1e-12);

        let old_whole_matrix_flip = Matrix3::new(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0);
        assert!(
            (reflected - projected.matrix()).norm() < (reflected - old_whole_matrix_flip).norm()
        );
    }

    #[test]
    fn test_sphere_cameras() {
        // Test with cameras arranged on a sphere looking toward origin
        // This mimics the synthetic scene Sphere camera arrangement
        use std::f64::consts::PI;

        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let num_cams = 8;
        let radius = 5.0;

        // Generate camera rotations using look_at (similar to synthetic.rs)
        let mut rotations = Vec::new();
        for i in 0..num_cams {
            let theta = 2.0 * PI * i as f64 / golden_ratio;
            let phi = (1.0 - 2.0 * (i as f64 + 0.5) / num_cams as f64).acos();

            // Camera position on sphere
            let x = radius * phi.sin() * theta.cos();
            let y = radius * phi.sin() * theta.sin();
            let z = radius * phi.cos();

            // Look direction toward origin
            let look_dir = nalgebra::Vector3::new(-x, -y, -z).normalize();
            let up = nalgebra::Vector3::new(0.0, 0.0, 1.0);

            // Build rotation matrix (camera to world)
            let forward = look_dir;
            let right = forward.cross(&up);
            let right = if right.norm() < 1e-6 {
                nalgebra::Vector3::new(1.0, 0.0, 0.0)
            } else {
                right.normalize()
            };
            let true_up = right.cross(&forward);

            let mat = nalgebra::Matrix3::from_columns(&[right, true_up, forward]);
            rotations.push(Rotation3::from_matrix_unchecked(mat));
        }

        // Generate all pairwise relative rotations: R_ij = R_j * R_i^{-1}
        let mut relative_rotations = Vec::new();
        for i in 0..num_cams {
            for j in (i + 1)..num_cams {
                let r_ij = rotations[j] * rotations[i].inverse();
                relative_rotations.push(
                    RelativeRotation::try_new(ViewId::new(i), ViewId::new(j), r_ij, 1.0).unwrap(),
                );
            }
        }

        let solver = L2DenseRotationSolver::new();
        let config = RotationConfig::default();
        let result = solver.solve(&relative_rotations, &config).unwrap();

        assert_eq!(result.rotations.len(), num_cams);

        // Verify relative rotations are recovered correctly
        let mut max_error = 0.0f64;
        for i in 0..num_cams {
            for j in (i + 1)..num_cams {
                let ri = result.rotations.get(&ViewId::new(i)).unwrap();
                let rj = result.rotations.get(&ViewId::new(j)).unwrap();
                let recovered_rij = *rj * ri.inverse();
                let expected_rij = rotations[j] * rotations[i].inverse();
                let error = (recovered_rij.matrix() - expected_rij.matrix()).norm();
                max_error = max_error.max(error);
            }
        }

        assert!(
            max_error < 0.1,
            "Sphere cameras: max relative rotation error (Frobenius) = {}",
            max_error
        );
    }
}
