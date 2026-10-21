// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Translation averaging solvers
//!
//! Its teaching docs are on [`average_translations`], where docs.rs shows them.
mod l1_irls;

pub(crate) use l1_irls::L1IrlsTranslationSolver;

use std::collections::BTreeMap;

use nalgebra::Rotation3;
use openmvg_core::model::{EstimationResult, RelativeTranslation, ViewId};

use crate::averaging::{TranslationConfig, TranslationEstimate, TranslationSolver};

/// Estimate absolute camera centers with the default L1-IRLS algorithm.
///
/// # How it works
///
/// Translation averaging recovers absolute camera positions from pairwise
/// relative translation directions and known absolute rotations.
///
/// ```text
/// directions + rotations -> direction residuals -> L1-IRLS
///                         -> gauge/convergence checks -> camera centers
/// ```
///
/// The solver runs Iteratively Reweighted Least Squares (IRLS) with a Huber loss whose width
/// shrinks over the iterations, so a large residual is weighted roughly as in an L1 fit. The
/// reweighting follows Chatterjee and Govindu's L1-IRLS for rotation averaging, applied here to
/// translation directions.
///
/// The views are the ones the edges name. The first edge's `view_i`
/// is held at the origin to fix the gauge. The view graph must be connected.
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │                  L1-IRLS translation averaging solver                       │
/// │             (Huber IRLS, after Chatterjee & Govindu, ICCV 2013)             │
/// └─────────────────────────────────────────────────────────────────────────────┘
///
///     Input: m pairwise translation directions d_ij (unit vectors, camera frame)
///            n absolute w2c rotations R_i ∈ SO(3)  (same as WorldToCameraPose::rotation)
///            Convention: d_ij = R_w2c · (t_j − t_i).normalize()
///
///                    ┌───────────────────────────────────────┐
///                    │  Step 1: Transform to world frame     │
///                    └───────────────────────────────────────┘
///                                      │
///                    For each edge (i, j):
///                      d_ij^world = R_iᵀ · d_ij   (w2c → c2w, rotate to world)
///                    First edge's view_i fixed at origin (gauge)
///                    → 3·(n−1) unknowns (reduced system)
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 2: IRLS iteration               │
///                    └───────────────────────────────────────┘
///                                      │
///                    Repeat max_iterations times:
///                      ┌─────────────────────────────────────┐
///                      │ a) Update scales                    │
///                      │    s_k = ||Δt||² / (Δt·d)           │
///                      │    (optimal per-edge baseline)      │
///                      ├─────────────────────────────────────┤
///                      │ b) Update Huber weights             │
///                      │    r_k = ||Δt/s − d||               │
///                      │    w_k = 1           if r_k ≤ τ     │
///                      │    w_k = τ / r_k     otherwise      │
///                      ├─────────────────────────────────────┤
///                      │ c) Solve normal equations           │
///                      │    (AᵀWA + λI)t = AᵀWb              │
///                      └─────────────────────────────────────┘
///                    Dense Cholesky up to 101 views, Jacobi-
///                    preconditioned conjugate gradient above
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 3: Convergence check            │
///                    └───────────────────────────────────────┘
///                                      │
///                    Mean angular residual:
///                      θ = (1/m) Σ acos(Δt̂ · d_k)
///                    Stop if |θ_prev − θ| < tolerance
///                                      │
///                                      ▼
///     Output: n camera centers t_i ∈ ℝ³, the first edge's view_i at 0
///
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │  Complexity: O(n³) per dense Cholesky solve × max_iterations                │
/// │  Memory:     O(n²) for the dense 3(n−1) × 3(n−1) normal matrix;             │
/// │              the conjugate gradient path above 101 views is matrix-free     │
/// │  Stops early when the mean angular residual settles                         │
/// └─────────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Errors
///
/// [`EstimationError::InvalidInput`] when the configuration or an edge is invalid, an edge's view
/// has no finite absolute rotation, or the edges do not form a connected graph;
/// [`EstimationError::InsufficientData`] when there are no edges;
/// [`EstimationError::Numerical`] when a linear solve fails; and
/// [`EstimationError::DegenerateConfiguration`] when every edge's estimated baseline vanishes.
///
/// [`EstimationError::InvalidInput`]: openmvg_core::model::EstimationError::InvalidInput
/// [`EstimationError::InsufficientData`]: openmvg_core::model::EstimationError::InsufficientData
/// [`EstimationError::Numerical`]: openmvg_core::model::EstimationError::Numerical
/// [`EstimationError::DegenerateConfiguration`]: openmvg_core::model::EstimationError::DegenerateConfiguration
///
/// # References
///
/// - A. Chatterjee, V. M. Govindu, "Efficient and Robust Large-Scale Rotation Averaging",
///   ICCV 2013. The L1-IRLS scheme there is for rotations; this function applies its
///   reweighting to translation directions.
pub fn average_translations(
    relative_translations: &[RelativeTranslation],
    absolute_rotations: &BTreeMap<ViewId, Rotation3<f64>>,
    config: &TranslationConfig,
) -> EstimationResult<TranslationEstimate> {
    L1IrlsTranslationSolver::new().solve(relative_translations, absolute_rotations, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    /// The supported translation averaging signature.
    type TranslationFn = fn(
        &[RelativeTranslation],
        &BTreeMap<ViewId, Rotation3<f64>>,
        &TranslationConfig,
    ) -> EstimationResult<TranslationEstimate>;

    #[test]
    fn the_default_entry_point_preserves_the_gauge_and_directions() {
        // A clean triangle fixes the gauge and recovers the edge directions.
        let centers = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(0.7, 1.5, 0.4),
        ];
        let relative: Vec<RelativeTranslation> = [(0, 1), (0, 2), (1, 2)]
            .iter()
            .map(|&(i, j)| {
                RelativeTranslation::try_new(
                    ViewId::new(i),
                    ViewId::new(j),
                    (centers[j] - centers[i]).normalize(),
                    1.0,
                )
                .unwrap()
            })
            .collect();
        let rotations: BTreeMap<_, _> = (0..3)
            .map(|v| (ViewId::new(v), Rotation3::identity()))
            .collect();
        let config = TranslationConfig::default();
        let entry: TranslationFn = average_translations;
        let estimate = entry(&relative, &rotations, &config).unwrap();
        assert!(estimate.translations[&ViewId::new(0)].norm() < 1e-12);
        let edge = estimate.translations[&ViewId::new(2)] - estimate.translations[&ViewId::new(0)];
        let truth = (centers[2] - centers[0]).normalize();
        assert!(edge.normalize().dot(&truth) > 0.9999);
    }
}
