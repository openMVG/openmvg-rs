// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Rotation averaging solvers
//!
//! Its teaching docs are on [`average_rotations`], where docs.rs shows them.

// The solver links target `faer`-gated items; full-feature doc builds check them.
#![cfg_attr(not(feature = "faer"), allow(rustdoc::broken_intra_doc_links))]

mod l2_dense;
#[cfg(feature = "faer")]
mod l2_sparse;
mod robust;

pub(crate) use l2_dense::L2DenseRotationSolver;
#[cfg(feature = "faer")]
pub(crate) use l2_sparse::L2SparseRotationSolver;
#[cfg(any(feature = "faer", test))]
pub(crate) use robust::RobustKernel;
pub(crate) use robust::{RobustConfig, RobustRotationSolver};

#[cfg(feature = "faer")]
use openmvg_ba::RobustLoss;
use openmvg_core::graph::component_partition;
use openmvg_core::model::{EstimationError, EstimationResult, RelativeRotation};

use crate::averaging::{RotationConfig, RotationEstimate, RotationSolver};

/// Mean unweighted edge residual in radians, matching
/// [`crate::averaging::RotationEstimate::residual`].
pub(crate) fn mean_angular_residual(
    rotations: &std::collections::BTreeMap<openmvg_core::model::ViewId, nalgebra::Rotation3<f64>>,
    relative_rotations: &[openmvg_core::model::RelativeRotation],
) -> f64 {
    if relative_rotations.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;
    for relative_rotation in relative_rotations {
        let (Some(view_i), Some(view_j)) = (
            rotations.get(&relative_rotation.view_i()),
            rotations.get(&relative_rotation.view_j()),
        ) else {
            return f64::INFINITY;
        };
        let error = (relative_rotation.rotation() * view_i).inverse() * view_j;
        let cosine = ((error.matrix().trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        total += cosine.acos();
    }
    total / relative_rotations.len() as f64
}

/// Reject a relative-rotation graph in several connected parts: the solve would relate nothing
/// across them, yet return rotations for every view.
pub(crate) fn ensure_connected(relative_rotations: &[RelativeRotation]) -> EstimationResult<()> {
    let mut nodes = std::collections::BTreeMap::new();
    let mut node = |view| {
        let next = nodes.len();
        *nodes.entry(view).or_insert(next)
    };
    let endpoints: Vec<_> = relative_rotations
        .iter()
        .map(|relative| (node(relative.view_i()), node(relative.view_j())))
        .collect();
    let partition = component_partition(nodes.len(), &endpoints)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    if partition.component_count() > 1 {
        return Err(EstimationError::InvalidInput(format!(
            "the relative-rotation graph has {} connected parts; keep the largest one first",
            partition.component_count()
        )));
    }
    Ok(())
}

/// Default robust sparse rotation averager.
#[cfg(feature = "faer")]
pub(crate) type DefaultRotationSolver = RobustRotationSolver<L2SparseRotationSolver>;

/// Create the default rotation averager.
#[cfg(feature = "faer")]
pub(crate) fn default_rotation_solver() -> DefaultRotationSolver {
    RobustRotationSolver::new(L2SparseRotationSolver::new(), RobustConfig::default())
}

/// Estimate absolute rotations with the default robust sparse algorithm.
///
/// # How it works
///
/// Rotation averaging recovers absolute camera rotations from pairwise
/// relative rotations: given measurements R_ij such that R_j ≈ R_ij * R_i,
/// it solves for the absolute rotations R_i.
///
/// ```text
/// relative rotations -> input checks -> sparse L2 solve
///                     -> SO(3) projection -> robust reweighting
///                     -> gauge-fixed absolute rotations
/// ```
///
/// The view graph must be connected: nothing relates the rotations of two separate parts, so a
/// graph in several parts is rejected. Keep its largest connected part first, for example with
/// [`component_partition`](openmvg_core::graph::component_partition).
///
/// # Errors
///
/// [`EstimationError::InvalidInput`] for an invalid configuration or relative rotation, or a
/// graph in several connected parts; [`EstimationError::InsufficientData`] without any relative
/// rotation or with fewer than two views; [`EstimationError::Numerical`] when the solve does not
/// produce proper rotations. [`average_rotations_with_robust_loss`] and
/// [`average_rotations_dense`] take the same input and return the same errors.
///
/// ## Default solver
///
/// With the `faer` feature, [`average_rotations`] runs the default robust sparse
/// solver. [`average_rotations_dense`] runs the same robust loop over a dense
/// eigendecomposition and needs no optional dependency.
///
/// ## Implementations
///
/// - [`average_rotations`]: robust IRLS over sparse L2 (requires `faer`).
/// - [`average_rotations_dense`]: robust IRLS over dense L2, `O(n³)`, always available.
///
/// ## References
///
/// - A. Chatterjee, V. M. Govindu, "Efficient and Robust Large-Scale Rotation Averaging",
///   ICCV 2013.
/// - D. Martinec, T. Pajdla, "Robust Rotation and Translation Estimation in Multiview
///   Reconstruction", CVPR 2007.
///
/// # The sparse L2 solver
///
/// Implements L2 (least-squares) rotation averaging with sparse matrices. The rotations are
/// determined only up to one global rotation, the gauge, so the solver fixes the best-connected
/// camera's rotation to I and solves the remaining, non-singular normal equations
/// `H_rr·X_r = −H_r0` with one deterministic sparse Cholesky factorization.
///
/// # Algorithm
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │                    Sparse L2 rotation averaging                      │
/// └──────────────────────────────────────────────────────────────────────┘
/// Input: weighted relative rotations Rᵢⱼ ≈ RⱼRᵢᵀ
///                                  │
///                                  ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 1: Assemble 3×3-block weighted rotation Laplacian H = AᵀA       │
/// └──────────────────────────────────────────────────────────────────────┘
/// Hᵢᵢ = (Σw)I; Hᵢⱼ = −wRᵢⱼᵀ.
///                                  │
///                                  ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 2: Ground the best-connected camera                             │
/// └──────────────────────────────────────────────────────────────────────┘
/// Fix its rotation to I and form the nonsingular system HᵣᵣXᵣ = −Hᵣ₀.
///                                  │
///                                  ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 3: Sparse Cholesky solve; dense fallback on failure             │
/// └──────────────────────────────────────────────────────────────────────┘
///                                  │
///                                  ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 4: Project each 3×3 block to SO(3) and normalize the gauge      │
/// └──────────────────────────────────────────────────────────────────────┘
/// Output: n absolute rotations Rᵢ ∈ SO(3)
///
///   Complexity: one sparse Cholesky factorization + triangular solve; no dense O(n³)
///               eigendecomposition. Repeated solves over one edge list (IRLS) build
///               the block graph and the symbolic factorization once.
/// ```
///
/// # References
///
/// - D. Martinec, T. Pajdla, "Robust Rotation and Translation Estimation in
///   Multiview Reconstruction", CVPR 2007.
/// - D. Martinec, "Robust Multiview Reconstruction", PhD thesis, Czech Technical
///   University in Prague, 2008 (formula 6.62).
#[cfg(feature = "faer")]
pub fn average_rotations(
    relative_rotations: &[RelativeRotation],
    config: &RotationConfig,
) -> EstimationResult<RotationEstimate> {
    default_rotation_solver().solve(relative_rotations, config)
}

/// Estimate absolute rotations with an explicitly selected robust loss.
///
/// The loss is applied in the same IRLS loop as [`average_rotations`] (sparse L2
/// inner solve, default iteration limits). A Huber or Cauchy scale must be
/// angular (`ResidualScale::Angular(θ)`); it is converted exactly to the chordal
/// residual `‖R_j R_iᵀ − R̄_ij‖_F = 2√2·sin(θ/2)` used by the reweighting. This
/// preserves the angular transition of the loss; the chordal objective differs
/// from BA's angular objective. A pixel scale is rejected with
/// [`EstimationError::InvalidInput`].
///
/// [`EstimationError::InvalidInput`]: openmvg_core::model::EstimationError::InvalidInput
///
/// # How it works
///
/// An IRLS loop wraps an inner L2 rotation solver, sparse here and dense in
/// [`average_rotations_dense`], and downweights outlier measurements with a robust
/// kernel. At each iteration:
///
/// 1. Solve the weighted rotation averaging problem using the inner solver
/// 2. Compute the Frobenius-norm residual for each edge
/// 3. Update weights w_ij ← kernel(r_ij) using the chosen robust kernel
/// 4. Apply weights to κ_ij: κ'_ij = κ_original × w_ij
/// 5. Repeat until weights converge
///
/// # Algorithm
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │           Robust rotation averaging (IRLS wrapper)                      │
/// │           around an inner L2 solver, sparse or dense                    │
/// └─────────────────────────────────────────────────────────────────────────┘
///
///     Input: m relative rotations R̄_ij with weights κ_ij,
///            inner L2 solver S, robust kernel ρ
///
///                    ┌───────────────────────────────────────┐
///                    │  Step 1: Initialize                   │
///                    │  w_ij ← 1.0 for all edges             │
///                    │  Store original κ_ij                  │
///                    └───────────────────────────────────────┘
///                                      │
///                    ┌─────────────────▼─────────────────────┐
///                    │  Step 2: Solve weighted problem       │
///                    │  κ'_ij ← κ_ij × w_ij                  │
///                    │  result ← S.solve(reweighted edges)   │
///                    └───────────────────────────────────────┘
///                                      │
///                    ┌─────────────────▼─────────────────────┐
///                    │  Step 3: Compute edge residuals       │
///                    │  r_ij = ||R̂_j R̂_i^T − R̄_ij||_F        │
///                    └───────────────────────────────────────┘
///                                      │
///                    ┌─────────────────▼─────────────────────┐
///                    │  Step 4: Update weights               │
///                    │  w_ij ← ρ(r_ij) (robust kernel)       │
///                    │  max_change ← max |w_new − w_old|     │
///                    └───────────────────────────────────────┘
///                                      │
///                              max_change < tol?
///                              ┌── yes ──┴── no ──┐
///                              ▼                   ▼
///                         Converged            Go to Step 2
///
///     Output: n absolute rotations R̂_i (outlier-robust)
/// ```
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "faer")]
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use nalgebra::{Rotation3, Vector3};
/// use openmvg_ba::{ResidualScale, RobustLoss};
/// use openmvg_core::AngularThreshold;
/// use openmvg_core::model::{RelativeRotation, ViewId};
/// use openmvg_geometry::averaging::{RotationConfig, average_rotations_with_robust_loss};
///
/// // Four cameras turned 10° apart about the vertical axis; every pair is measured once.
/// let turn = |degrees: f64| Rotation3::from_axis_angle(&Vector3::y_axis(), degrees.to_radians());
/// let truth: Vec<_> = (0..4).map(|i| turn(10.0 * f64::from(i))).collect();
/// // One pair is wrong, 60° off.
/// let error = Rotation3::from_axis_angle(&Vector3::x_axis(), 60_f64.to_radians());
/// let mut pairs = Vec::new();
/// for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
///     let mut measured = truth[j] * truth[i].inverse();
///     if (i, j) == (0, 3) {
///         measured = error * measured;
///     }
///     pairs.push(RelativeRotation::try_new(ViewId::new(i), ViewId::new(j), measured, 1.0)?);
/// }
///
/// // A Cauchy loss with a 2° scale sets the wrong pair aside: the 0-3 turn comes back within a
/// // fraction of a degree, though its measurement was 60° off.
/// let scale = AngularThreshold::try_new(2_f64.to_radians())?;
/// let loss = RobustLoss::Cauchy(ResidualScale::Angular(scale));
/// let estimate = average_rotations_with_robust_loss(&pairs, &RotationConfig::default(), loss)?;
/// let rotation = |view: usize| estimate.rotations[&ViewId::new(view)];
/// let off = (rotation(3) * rotation(0).inverse()).angle_to(&(truth[3] * truth[0].inverse()));
/// assert!(off.to_degrees() < 0.2, "the 0-3 turn is off by {off} rad");
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "faer"))]
/// # fn main() {}
/// ```
///
/// # Design
///
/// - Working-copy measurements: weights change on a working copy of the
///   `RelativeRotation` slice (only the `weight` field changes).
/// - Pre-allocated residual buffer, reused across IRLS iterations.
/// - Reused inner-solver state: the inner solver's workspace lives across IRLS
///   iterations, so the sparse kernel builds its graph and symbolic factorization once.
/// - Fused weight update: new weights and the maximum change come from a single pass.
/// - Generic over the inner solver: supports the default sparse kernel and
///   private reference implementations.
///
/// # References
///
/// - A. Chatterjee, V. M. Govindu, "Efficient and Robust Large-Scale Rotation Averaging",
///   ICCV 2013.
/// - A. Chatterjee, V. M. Govindu, "Robust Relative Rotation Averaging", IEEE TPAMI 40(4),
///   2018.
///
/// Each inner solver documents its own references.
#[cfg(feature = "faer")]
pub fn average_rotations_with_robust_loss(
    relative_rotations: &[RelativeRotation],
    config: &RotationConfig,
    robust: RobustLoss,
) -> EstimationResult<RotationEstimate> {
    let kernel = RobustKernel::try_from_loss(robust)?;
    RobustRotationSolver::new(
        L2SparseRotationSolver::new(),
        RobustConfig::with_kernel(kernel),
    )
    .solve(relative_rotations, config)
}

/// Estimate absolute rotations with the robust loop of [`average_rotations`] over a dense
/// eigendecomposition.
///
/// The inner solve is the Martinec–Pajdla dense L2 relaxation, `O(n³)` in the view count, so
/// it suits small graphs and builds without the `faer` feature. It needs no optional
/// dependency.
///
/// # How it works
///
/// Implements L2 (least-squares) rotation averaging using dense eigendecomposition.
///
/// # Algorithm
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │                 L2 dense rotation averaging solver                          │
/// │                   (Martinec–Pajdla, CVPR 2007)                              │
/// └─────────────────────────────────────────────────────────────────────────────┘
///
///     Input: m pairwise relative rotations R_ij with weights w_ij
///            Relative constraint: R_ij ≈ R_j · R_i⁻¹
///
///                    ┌───────────────────────────────────────┐
///                    │  Step 1: Build normal matrix          │
///                    └───────────────────────────────────────┘
///                                      │
///                    For each relative rotation (i, j):
///                    sqrt(w_ij) · (R_j − R_ij · R_i) = 0
///
///                    Accumulate L = AᵀA directly (3n × 3n):
///                    ┌                                                   ┐
///                    │  L_ii += w R_ijᵀR_ij   L_ij -= w R_ijᵀ             │
///                    │  L_jj += w I            L_ji -= w R_ij              │
///                    └                                                   ┘
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 2: Eigendecomposition           │
///                    └───────────────────────────────────────┘
///                                      │
///                    Eigenvalue decomposition: AᵀA = V·Λ·Vᵀ
///                    Extract 3 eigenvectors with smallest eigenvalues
///                    (nullspace of AᵀA, dimension = 3 from rotation gauge)
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 3: Nullspace basis alignment    │
///                    └───────────────────────────────────────┘
///                                      │
///                    The 3 eigenvectors are an arbitrary basis for
///                    the nullspace — not the rotation columns directly.
///
///                    Fix: align via camera-0 gauge fixing:
///                      V₀ = 3×3 block of camera 0 from eigenvectors
///                      T  = V₀⁻¹  (SVD pseudo-inverse)
///                      Rᵢ = Vᵢ · T  for each camera i
///
///                    This resolves sign/basis ambiguity that causes
///                    180° flips when eigenvalues are degenerate.
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 4: Project to SO(3)             │
///                    └───────────────────────────────────────┘
///                                      │
///                    For each camera i:
///                      SVD: Rᵢ = U·Σ·Vᵀ
///                      D = diag(1, 1, sign(det(U·Vᵀ)))
///                      Rᵢ ← U·D·Vᵀ  (closest proper rotation)
///
///                    Gauge-fix: Rᵢ ← Rᵢ · R₀ᵀ  (camera 0 = identity)
///                                      │
///                                      ▼
///     Output: n absolute rotations Rᵢ ∈ SO(3), gauge-fixed to R₀ = I
///
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │  Complexity: O(n³) for eigendecomposition of AᵀA                            │
/// │  Memory:     O(n²) for the 3n × 3n normal matrix                            │
/// │  Practical:  small graphs; use the sparse L2 solver for large scenes        │
/// └─────────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # References
///
/// - D. Martinec, T. Pajdla, "Robust Rotation and Translation Estimation in
///   Multiview Reconstruction", CVPR 2007.
/// - D. Martinec, "Robust Multiview Reconstruction", PhD thesis, Czech Technical
///   University in Prague, 2008 (formula 6.62).
pub fn average_rotations_dense(
    relative_rotations: &[RelativeRotation],
    config: &RotationConfig,
) -> EstimationResult<RotationEstimate> {
    RobustRotationSolver::new(L2DenseRotationSolver::new(), RobustConfig::default())
        .solve(relative_rotations, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::averaging::{RotationConfig, RotationSolver};
    use nalgebra::Matrix3;
    use nalgebra::Rotation3;
    use openmvg_core::model::{RelativeRotation, ViewId};
    use std::f64::consts::PI;

    // =========================================================================
    // Test Helpers
    // =========================================================================

    fn rot_x(angle: f64) -> Rotation3<f64> {
        Rotation3::from_euler_angles(angle, 0.0, 0.0)
    }

    fn rot_y(angle: f64) -> Rotation3<f64> {
        Rotation3::from_euler_angles(0.0, angle, 0.0)
    }

    fn rot_z(angle: f64) -> Rotation3<f64> {
        Rotation3::from_euler_angles(0.0, 0.0, angle)
    }

    fn rotation_error(r1: &Rotation3<f64>, r2: &Rotation3<f64>) -> f64 {
        (r1.matrix() - r2.matrix()).norm()
    }

    // =========================================================================
    // Shared Test Cases - Run for ALL rotation solver implementations
    // =========================================================================

    /// Test a rotation solver with a triangle configuration (3 cameras).
    fn run_test_triangle<S: RotationSolver>(solver: &S) {
        let r0 = Rotation3::identity();
        let r1 = rot_z(0.3);
        let r2 = rot_z(0.7);

        let r_01 = r1 * r0.inverse();
        let r_12 = r2 * r1.inverse();
        let r_02 = r2 * r0.inverse();

        let relative_rotations = vec![
            RelativeRotation::try_new(ViewId::new(0), ViewId::new(1), r_01, 1.0).unwrap(),
            RelativeRotation::try_new(ViewId::new(1), ViewId::new(2), r_12, 1.0).unwrap(),
            RelativeRotation::try_new(ViewId::new(0), ViewId::new(2), r_02, 1.0).unwrap(),
        ];

        let config = RotationConfig::default();
        let result = solver.solve(&relative_rotations, &config).unwrap();

        assert_eq!(result.rotations.len(), 3);

        let recovered_r0 = result.rotations.get(&ViewId::new(0)).unwrap();
        assert!(
            (recovered_r0.matrix() - Matrix3::identity()).norm() < 0.01,
            "Camera 0 should be identity"
        );

        let recovered_r1 = result.rotations.get(&ViewId::new(1)).unwrap();
        let recovered_r01 = *recovered_r1 * recovered_r0.inverse();
        let error = rotation_error(&recovered_r01, &r_01);
        assert!(error < 0.01, "Relative rotation error: {}", error);
    }

    /// Test a rotation solver with a chain configuration (5 cameras).
    fn run_test_chain_rotations<S: RotationSolver>(solver: &S) {
        let angles = [0.0, 0.2, 0.4, 0.6, 0.8];
        let rotations: Vec<Rotation3<f64>> = angles.iter().map(|&a| rot_z(a)).collect();

        let mut relative_rotations = Vec::new();
        for i in 0..4 {
            let r_ij = rotations[i + 1] * rotations[i].inverse();
            relative_rotations.push(
                RelativeRotation::try_new(ViewId::new(i), ViewId::new(i + 1), r_ij, 1.0).unwrap(),
            );
        }
        relative_rotations.push(
            RelativeRotation::try_new(
                ViewId::new(0),
                ViewId::new(4),
                rotations[4] * rotations[0].inverse(),
                1.0,
            )
            .unwrap(),
        );
        relative_rotations.push(
            RelativeRotation::try_new(
                ViewId::new(0),
                ViewId::new(2),
                rotations[2] * rotations[0].inverse(),
                1.0,
            )
            .unwrap(),
        );

        let config = RotationConfig::default();
        let result = solver.solve(&relative_rotations, &config).unwrap();

        assert_eq!(result.rotations.len(), 5);

        let r0 = result.rotations.get(&ViewId::new(0)).unwrap();
        assert!(
            (r0.matrix() - Matrix3::identity()).norm() < 0.01,
            "First rotation should be identity"
        );

        for i in 0..5 {
            let recovered = result.rotations.get(&ViewId::new(i)).unwrap();
            let expected = rotations[0].inverse() * rotations[i];
            let error = rotation_error(recovered, &expected);
            assert!(error < 0.01, "Camera {} error: {}", i, error);
        }
    }

    /// Test a rotation solver with mixed axis rotations (4 cameras).
    fn run_test_mixed_axis_rotations<S: RotationSolver>(solver: &S) {
        let r0 = Rotation3::identity();
        let r1 = rot_x(0.2);
        let r2 = rot_y(0.3);
        let r3 = rot_z(0.15);
        let rotations = [r0, r1, r2, r3];

        let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let relative_rotations: Vec<_> = pairs
            .iter()
            .map(|&(i, j)| {
                RelativeRotation::try_new(
                    ViewId::new(i),
                    ViewId::new(j),
                    rotations[j] * rotations[i].inverse(),
                    1.0,
                )
                .unwrap()
            })
            .collect();

        let config = RotationConfig::default();
        let result = solver.solve(&relative_rotations, &config).unwrap();

        assert_eq!(result.rotations.len(), 4);

        for &(i, j) in &pairs {
            let ri = result.rotations.get(&ViewId::new(i)).unwrap();
            let rj = result.rotations.get(&ViewId::new(j)).unwrap();
            let recovered_rij = *rj * ri.inverse();
            let expected_rij = rotations[j] * rotations[i].inverse();
            let error = rotation_error(&recovered_rij, &expected_rij);
            assert!(error < 0.01, "Edge {}->{} error: {}", i, j, error);
        }
    }

    /// Test a rotation solver with square formation (90° rotations).
    fn run_test_square_formation<S: RotationSolver>(solver: &S) {
        let r0 = Rotation3::identity();
        let r1 = rot_z(PI / 2.0);
        let r2 = rot_z(PI);
        let r3 = rot_z(-PI / 2.0);
        let rotations = [r0, r1, r2, r3];

        let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let relative_rotations: Vec<_> = pairs
            .iter()
            .map(|&(i, j)| {
                RelativeRotation::try_new(
                    ViewId::new(i),
                    ViewId::new(j),
                    rotations[j] * rotations[i].inverse(),
                    1.0,
                )
                .unwrap()
            })
            .collect();

        let config = RotationConfig::default();
        let result = solver.solve(&relative_rotations, &config).unwrap();

        assert_eq!(result.rotations.len(), 4);

        for &(i, j) in &pairs {
            let ri = result.rotations.get(&ViewId::new(i)).unwrap();
            let rj = result.rotations.get(&ViewId::new(j)).unwrap();
            let recovered_rij = *rj * ri.inverse();
            let expected_rij = rotations[j] * rotations[i].inverse();
            let error = rotation_error(&recovered_rij, &expected_rij);
            assert!(error < 0.01, "Edge {}->{} error: {}", i, j, error);
        }
    }

    /// Test a rotation solver with only two views.
    fn run_test_two_view_rotation<S: RotationSolver>(solver: &S) {
        let r_12 = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
        let relative_rotations =
            vec![RelativeRotation::try_new(ViewId::new(0), ViewId::new(1), r_12, 1.0).unwrap()];

        let config = RotationConfig::default();
        let result = solver.solve(&relative_rotations, &config).unwrap();

        assert_eq!(result.rotations.len(), 2);

        let r0 = result.rotations.get(&ViewId::new(0)).unwrap();
        assert!(
            (r0.matrix() - Matrix3::identity()).norm() < 0.1,
            "First rotation should be near identity"
        );

        let r1 = result.rotations.get(&ViewId::new(1)).unwrap();
        let recovered_rel = *r1 * r0.inverse();
        let error = rotation_error(&recovered_rel, &r_12);
        assert!(error < 0.1, "Rotation error: {}", error);
    }

    /// Test a rotation solver with empty input.
    fn run_test_insufficient_data<S: RotationSolver>(solver: &S) {
        let config = RotationConfig::default();
        let result = solver.solve(&[], &config);
        assert!(result.is_err());
    }

    // =========================================================================
    // L2 Dense Rotation Solver Tests
    // =========================================================================

    #[test]
    fn test_dense_triangle() {
        run_test_triangle(&L2DenseRotationSolver::new());
    }

    #[test]
    fn test_dense_chain_rotations() {
        run_test_chain_rotations(&L2DenseRotationSolver::new());
    }

    #[test]
    fn test_dense_mixed_axis_rotations() {
        run_test_mixed_axis_rotations(&L2DenseRotationSolver::new());
    }

    #[test]
    fn test_dense_square_formation() {
        run_test_square_formation(&L2DenseRotationSolver::new());
    }

    #[test]
    fn test_dense_two_view_rotation() {
        run_test_two_view_rotation(&L2DenseRotationSolver::new());
    }

    #[test]
    fn test_dense_insufficient_data() {
        run_test_insufficient_data(&L2DenseRotationSolver::new());
    }

    /// Two triangles that share no view: nothing relates their rotations.
    fn two_separate_triangles() -> Vec<RelativeRotation> {
        [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
            .into_iter()
            .map(|(i, j)| {
                RelativeRotation::try_new(ViewId::new(i), ViewId::new(j), rot_y(0.1), 1.0).unwrap()
            })
            .collect()
    }

    #[test]
    fn dense_averaging_rejects_a_graph_in_two_parts() {
        let result = average_rotations_dense(&two_separate_triangles(), &RotationConfig::default());
        assert!(
            matches!(result, Err(openmvg_core::model::EstimationError::InvalidInput(ref reason)) if reason.contains("2 connected parts")),
            "{result:?}"
        );
    }

    #[cfg(feature = "faer")]
    #[test]
    fn sparse_averaging_rejects_a_graph_in_two_parts() {
        let config = RotationConfig::default();
        let edges = two_separate_triangles();
        assert!(matches!(
            average_rotations(&edges, &config),
            Err(openmvg_core::model::EstimationError::InvalidInput(_))
        ));
        let loss = RobustLoss::Cauchy(openmvg_ba::ResidualScale::Angular(
            openmvg_core::AngularThreshold::try_new(0.05).unwrap(),
        ));
        assert!(matches!(
            average_rotations_with_robust_loss(&edges, &config, loss),
            Err(openmvg_core::model::EstimationError::InvalidInput(_))
        ));
    }

    // =========================================================================
    // L2 Sparse Rotation Solver Tests
    // =========================================================================

    #[cfg(feature = "faer")]
    #[test]
    fn test_sparse_triangle() {
        run_test_triangle(&L2SparseRotationSolver::new());
    }

    #[cfg(feature = "faer")]
    #[test]
    fn test_sparse_chain_rotations() {
        run_test_chain_rotations(&L2SparseRotationSolver::new());
    }

    #[cfg(feature = "faer")]
    #[test]
    fn test_sparse_mixed_axis_rotations() {
        run_test_mixed_axis_rotations(&L2SparseRotationSolver::new());
    }

    #[cfg(feature = "faer")]
    #[test]
    fn test_sparse_square_formation() {
        run_test_square_formation(&L2SparseRotationSolver::new());
    }

    #[cfg(feature = "faer")]
    #[test]
    fn test_sparse_two_view_rotation() {
        run_test_two_view_rotation(&L2SparseRotationSolver::new());
    }

    #[cfg(feature = "faer")]
    #[test]
    fn test_sparse_insufficient_data() {
        run_test_insufficient_data(&L2SparseRotationSolver::new());
    }

    /// Large connected scene (200 cameras, 600 variables) that exercises the sparse
    /// grounded direct-solve path, not the dense fallback. With noise-free input, the
    /// recovered relative rotations must match ground truth and agree with the dense solver.
    #[cfg(feature = "faer")]
    #[test]
    fn test_sparse_large_scene_grounded_solve() {
        let num = 200usize;
        let gt: Vec<Rotation3<f64>> = (0..num)
            .map(|i| {
                let a = i as f64;
                rot_z(0.03 * a) * rot_y(0.017 * a) * rot_x(0.011 * a)
            })
            .collect();
        // Connected graph: chain + skip-5 edges (extra constraints, still sparse).
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..num - 1 {
            edges.push((i, i + 1));
        }
        for i in 0..num - 5 {
            edges.push((i, i + 5));
        }
        let rels: Vec<_> = edges
            .iter()
            .map(|&(i, j)| {
                RelativeRotation::try_new(
                    ViewId::new(i),
                    ViewId::new(j),
                    gt[j] * gt[i].inverse(),
                    1.0,
                )
                .unwrap()
            })
            .collect();

        let config = RotationConfig::default();
        let dense = L2DenseRotationSolver::new().solve(&rels, &config).unwrap();
        // Drive the grounded direct-solve DIRECTLY. The public `solve()` would mask a broken
        // grounded path behind its dense fallback, so this explicit call is what actually
        // guards the sparse path against regression.
        let grounded = l2_sparse::L2SparseWorkspace::default()
            .solve_grounded(&rels)
            .expect("grounded direct-solve must succeed on a connected graph");
        assert_eq!(grounded.len(), num);
        let sparse = L2SparseRotationSolver::new().solve(&rels, &config).unwrap();
        assert_eq!(sparse.rotations.len(), num);

        // All gauge-free; compare recovered RELATIVE rotations to GT per edge. `grounded` is the
        // direct sparse path; `dense`/`sparse` cross-check the eigen reference and public API.
        for &(i, j) in &edges {
            let expected = gt[j] * gt[i].inverse();
            let gi = grounded.get(&ViewId::new(i)).unwrap();
            let gj = grounded.get(&ViewId::new(j)).unwrap();
            let err_g = rotation_error(&(*gj * gi.inverse()), &expected);
            assert!(
                err_g < 1e-3,
                "grounded edge {i}->{j} rotation error {err_g}"
            );
            for (name, res) in [("dense", &dense), ("sparse", &sparse)] {
                let ri = res.rotations.get(&ViewId::new(i)).unwrap();
                let rj = res.rotations.get(&ViewId::new(j)).unwrap();
                let recovered = *rj * ri.inverse();
                let err = rotation_error(&recovered, &expected);
                assert!(err < 1e-3, "{name} edge {i}->{j} rotation error {err}");
            }
        }
    }

    // =========================================================================
    // Dense vs Sparse Equivalence Test
    // =========================================================================

    #[cfg(feature = "faer")]
    #[test]
    fn test_dense_vs_sparse_equivalence() {
        let r0 = Rotation3::identity();
        let r1 = rot_z(0.3);
        let r2 = rot_z(0.5);

        let relative_rotations = vec![
            RelativeRotation::try_new(ViewId::new(0), ViewId::new(1), r1 * r0.inverse(), 1.0)
                .unwrap(),
            RelativeRotation::try_new(ViewId::new(0), ViewId::new(2), r2 * r0.inverse(), 1.0)
                .unwrap(),
            RelativeRotation::try_new(ViewId::new(1), ViewId::new(2), r2 * r1.inverse(), 1.0)
                .unwrap(),
        ];

        let dense_solver = L2DenseRotationSolver::new();
        let sparse_solver = L2SparseRotationSolver::new();
        let config = RotationConfig::default();

        let dense_result = dense_solver.solve(&relative_rotations, &config).unwrap();
        let sparse_result = sparse_solver.solve(&relative_rotations, &config).unwrap();

        for i in 0..3 {
            let dense_r = dense_result.rotations.get(&ViewId::new(i)).unwrap();
            let sparse_r = sparse_result.rotations.get(&ViewId::new(i)).unwrap();
            let diff = rotation_error(dense_r, sparse_r);
            assert!(
                diff < 0.01,
                "Dense vs Sparse mismatch for camera {}: {}",
                i,
                diff
            );
        }
    }

    // =========================================================================
    // Robust IRLS Wrapper Tests
    // =========================================================================

    /// Robust rotation averaging: the retained robust rotation policy recovers a mixed-axis pose
    /// graph; world-to-camera rotation composition remains fixed.
    #[test]
    fn test_robust_mixed_axis_rotations() {
        let solver = RobustRotationSolver::new(
            L2DenseRotationSolver::new(),
            RobustConfig::with_kernel(RobustKernel::Cauchy { c: 1.0 }),
        );
        run_test_mixed_axis_rotations(&solver);
    }

    #[test]
    fn test_robust_insufficient_data() {
        let solver =
            RobustRotationSolver::new(L2DenseRotationSolver::new(), RobustConfig::default());
        run_test_insufficient_data(&solver);
    }

    #[cfg(feature = "faer")]
    #[test]
    fn test_robust_loss_rejects_pixel_scale() {
        let relative =
            [
                RelativeRotation::try_new(
                    ViewId::new(0),
                    ViewId::new(1),
                    Rotation3::identity(),
                    1.0,
                )
                .unwrap(),
            ];
        let loss = RobustLoss::Cauchy(openmvg_ba::ResidualScale::Pixels(
            openmvg_core::PixelThreshold::ONE_PIXEL,
        ));
        let result =
            average_rotations_with_robust_loss(&relative, &RotationConfig::default(), loss);
        assert!(matches!(
            result,
            Err(openmvg_core::model::EstimationError::InvalidInput(_))
        ));
    }
}
