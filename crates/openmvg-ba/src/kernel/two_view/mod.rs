// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-size two-view calibrated bundle adjustment.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Two-view bundle adjustment                                           │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: pose (R, t) of camera 2, n ≥ 5 points X in the camera 1 frame, n
//!        observations per view (pixels or bearings), loss, noise σ
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Validate and fix the scale                                   │
//! └──────────────────────────────────────────────────────────────────────┘
//! Reject non-finite data, R ∉ SO(3), ‖t‖ ≤ 64 ε and unprojectable points.
//! Keep b = ‖t‖ for the whole run: it is the scale gauge.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Linearize every point                                        │
//! └──────────────────────────────────────────────────────────────────────┘
//! Residual r ∈ ℝ² per view (pixels, or radians in the tangent plane of the
//! observed bearing), whitened by 1/σ and weighted by the loss. Unknowns:
//! δ ∈ ℝ⁵ (3 rotation, 2 direction of t on S²) and 3 per point.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Eliminate the points (Schur complement)                      │
//! └──────────────────────────────────────────────────────────────────────┘
//! Per point, a damped 3×3 Cholesky of Hpp. Summing over points gives
//! S = Hcc − Hcp Hpp⁻¹ Hpc and g = gc − Hcp Hpp⁻¹ gp: a 5×5 system.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 4: Solve the 5×5 system                                         │
//! └──────────────────────────────────────────────────────────────────────┘
//! Cholesky solve of S δ = −g with diagonal damping λ. Scale the step so that
//! max(‖rotation step‖, ‖direction step‖) ≤ 0.5.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 5: Back-substitute, test and accept                             │
//! └──────────────────────────────────────────────────────────────────────┘
//! Point steps from the same blocks, candidate pose by retraction. With the
//! gain ratio ρ = actual / predicted reduction, ρ > 0 commits pose and points
//! and lowers λ; otherwise the state is kept and λ rises. Repeat from step 2.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 6: Stop and report                                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! Gradient, step and cost tests on the last loss stage, or the iteration
//! budget. The undamped 5×5 system at the end gives the rank evidence.
//!   │
//!   ▼
//! Output: pose with ‖t‖ = b, points updated in place, BundleAdjustmentReport
//! ```
//!
//! # Problem and method
//!
//! Camera 1 is the world frame. Camera 2 has the world-to-camera pose `X₂ = R X + t`. The
//! unknowns are `R ∈ SO(3)`, the direction of `t`, and `n` points. The length `b = ‖t‖` stays at
//! its initial value, so the scale gauge is fixed and the camera system has no null direction.
//! The cost is `Σₖ Σᵥ ρ(‖rₖᵥ‖ / σₖ)` over points `k` and views `v`, with `ρ` from [`Loss`] and
//! `σₖ` from [`NoiseModel`]. [`TwoViewMeasurements::Pixel`] residuals are the projection minus
//! the observation, in pixels, with fixed pinhole intrinsics. [`TwoViewMeasurements::Angular`]
//! residuals are the exact `S²` logarithm of the predicted bearing in the tangent plane of the
//! observed bearing, in radians.
//!
//! The rotation update is `R ← exp(ω) R` and the direction of `t` moves by a retraction on `S²`,
//! so every step stays on the manifold. A point moves by `b · Δ`, which keeps point steps
//! dimensionless. Damping is `λ` times the diagonal of each block, adapted by the Nielsen rule
//! from an initial value of `1e-4`. The loss schedule is the one every native solve shares: a
//! Cauchy loss starts with an L2 warm-up of at most five iterations.
//!
//! # Failure modes
//!
//! [`refine_ba_2view`] returns a [`BundleAdjustmentError`] for mismatched or non-finite input,
//! fewer than five points, invalid intrinsics, a rotation outside `SO(3)`, a zero baseline, a
//! point outside the projection domain, or invalid options or noise. During the loop, a
//! non-finite or unprojectable linearization or a failed factorization rejects the step and
//! raises `λ`. The report carries a [`RefinementStatus`]: `Stalled` when no step is accepted
//! in the last stage, and `Degenerate` when the final undamped system is not well constrained
//! ([`BundleAdjustmentDiagnostics::is_well_constrained`]). That covers a camera rank below five,
//! or a 3×3 point block that cannot be factored or has a reciprocal condition estimate of at
//! most `64 · 3 · ε`.
//!
//! # Alternatives
//!
//! Use [`multi_view`](crate::kernel::multi_view) for three or more views or for
//! self-calibration, or when only centers and points move. This kernel keeps the intrinsics
//! fixed and models no lens distortion.
//! `NativeBundleAdjuster` selects it for a problem with two poses, one fixed anchor, one
//! free full pose and complete two-view tracks.
//!
//! [`Loss`]: crate::numerics::refine::Loss
//! [`NoiseModel`]: crate::numerics::refine::NoiseModel
//! [`RefinementStatus`]: crate::numerics::refine::RefinementStatus

mod api;
#[cfg(feature = "unstable-internals")]
mod covariance;
mod optimizer;
mod residual;
mod schur;

pub use crate::kernel::camera::{PinholeCamera, RelativePose};
pub use api::{
    BundleAdjustmentDiagnostics, BundleAdjustmentError, BundleAdjustmentReport, TwoViewMeasurements,
};
#[cfg(feature = "unstable-internals")]
pub use covariance::{
    TwoViewBundleAdjustmentCovariance, TwoViewBundleAdjustmentCovarianceError, covariance_ba_2view,
};
pub use optimizer::refine_ba_2view;
