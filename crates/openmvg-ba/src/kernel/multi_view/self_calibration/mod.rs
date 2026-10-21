// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-view bundle adjustment with BAL-style radial self-calibration.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │            Self-calibrating multi-view bundle adjustment             │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: poses (camera 0 fixed), points Xᵢ ∈ ℝ³, radial intrinsics (f, k1, k2)
//! per camera or per group (camera_to_group), pixel tracks relative to the
//! principal point, loss, noise σ, scale gauge, optional fixed blocks, priors
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Check the input                                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! As in the calibrated kernel, plus f > 0 with finite k1, k2, a valid group
//! map (every group used), and every point inside each camera's radial branch.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Linearize with the radial camera                             │
//! └──────────────────────────────────────────────────────────────────────┘
//! n = (X/Z, Y/Z), r² = ‖n‖², u = f (1 + k1 r² + k2 r⁴) n; residual is
//! u − measured (px). Jacobians: J_p (2×3), J_pose (2×6), and the 2×3 block
//! J_int = [u, f r² n, f r⁴ n] for (log f, k1, k2).
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Eliminate the points (Schur complement)                      │
//! └──────────────────────────────────────────────────────────────────────┘
//! Same elimination as the calibrated kernel. Each camera block holds the pose
//! (6; 5 for a baseline camera; 0 for camera 0) and its 3 intrinsics. With
//! groups: all pose blocks, then one 3-parameter block per group.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 4: Solve S δc = −ĝ                                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! Dense Cholesky, sparse LLT (AMD), or retained block-Jacobi PCG. PCG keeps
//! pose blocks, intrinsic blocks and their edges; point couplings are applied
//! matrix-free.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 5: Recover the points and retract                               │
//! └──────────────────────────────────────────────────────────────────────┘
//! δpᵢ and the pose retraction as in the calibrated kernel. Intrinsics:
//! f ← f·exp(δ log f), k1 ← k1 + δk1, k2 ← k2 + δk2; parameters beyond the free
//! leading 1 to 3 keep their input values.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 6: Accept or reject (Nielsen gain ratio)                        │
//! └──────────────────────────────────────────────────────────────────────┘
//! Same LM loop and stop tests as the calibrated kernel; the camera step is
//! not capped at norm 0.5 here.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 7: Certify the final information                                │
//! └──────────────────────────────────────────────────────────────────────┘
//! Undamped reassembly with the intrinsic blocks: point conditioning, rank vs
//! the gauge nullity. Output: poses, points and intrinsics updated in place,
//! plus the calibrated kernel's report.
//! ```
//!
//! # Problem and method
//!
//! `RadialCamera` is the centered two-coefficient model `u = f (1 + k1 r² + k2
//! r⁴) (X/Z, Y/Z)` with `r² = (X² + Y²)/Z²`. It follows the BAL
//! parameterization after converting BAL's negative-depth, negated-image
//! convention. Measurements are pixel coordinates relative to the principal
//! point, which is not a variable, and each intrinsic block has one focal
//! length. The variables of a block are `(log f, k1, k2)`;
//! `FixedVariableBlocks::intrinsic_dimensions` frees only the leading 1, 2, or
//! 3 of them (a pinhole camera refines `f` alone), and `intrinsics` fixes whole
//! blocks. Camera 0 keeps a fixed pose and free intrinsics. A block belongs to
//! one camera, or is shared through `camera_to_group`: a group is one
//! 3-parameter block however many cameras use it. Loss, noise model, gauge,
//! priors, and the LM loop are those of the calibrated kernel; only the
//! Jacobian blocks, the retraction, and the projection domain differ.
//!
//! Projection is rejected for `Z ≤ 0`, for `r²` at or beyond the first positive
//! root of `1 + 3 k1 r² + 5 k2 r⁴` (the end of the increasing radial branch),
//! and where the radial Jacobian determinant is at most 1e-12.
//! `RadialCamera::bearing` inverts a measurement on that branch by Newton steps
//! safeguarded by bisection. A measurement outside the branch still has a
//! forward residual; `noninvertible_measurement_count` reports such
//! measurements so that a caller can drop them before bearing-based
//! triangulation. `SelfCalibratingMeasurements::new_signed_depth`
//! (`unstable-internals`) accepts either depth sign, for BAL data.
//!
//! # Failure modes
//!
//! Input errors are `InvalidIntrinsics`, `IntrinsicGroupMapCountMismatch`,
//! `InvalidIntrinsicGroupCount`, `InvalidIntrinsicGroup`,
//! `UnusedIntrinsicGroup`, and `PointNotProjectable`; the calibrated failure
//! modes (rejected iterations, `Stalled`, `Degenerate`) apply otherwise. A
//! full-rank final system does not certify the intrinsics: near-critical motion
//! leaves it numerically full rank while the intrinsics drift along a weak
//! direction. The native adjuster therefore takes the marginal covariance of
//! each free group from the final information at unit noise, and holds a group
//! at its input values, re-solving from the input state, when the standard
//! deviation of `log f`, `k1 r²`, or `k2 r⁴` (r the normalized radius of the
//! farthest image corner) exceeds 0.01 or is NaN, or when the information
//! cannot be factored. Blocks with `IntrinsicCertification::Waived` are exempt
//! from the bound but not from NaN.
//!
//! # Alternatives
//!
//! Use `multi_view` when the intrinsics are known: its camera blocks have 6
//! unknowns instead of 9, and it holds Brown-Conrady distortion fixed. This
//! module does not refine the principal
//! point, aspect ratio, skew, tangential distortion, or `k3`; the native
//! adjuster sends a problem with a free intrinsic block here only for pixel
//! residuals, perspective cameras with equal `fx` and `fy`, zero `k3` and
//! tangential terms, a fixed principal point, and full-pose camera blocks, and
//! returns `SolveError::Unsupported` otherwise. It uses the sparse grouped
//! solver, where `LinearSolver::Auto` picks LLT while it fits the memory budget
//! and cached retained PCG otherwise; `RecomputedParallel` is rejected for
//! self-calibration. The dense solvers (`unstable-internals`) are the
//! reference.

mod api;
#[cfg(feature = "unstable-internals")]
mod covariance;
pub(super) mod model;
mod optimizer;

#[cfg(feature = "unstable-internals")]
pub use super::api::{
    AngularObservation, MultiViewBundleAdjustmentDiagnostics, MultiViewBundleAdjustmentError,
    MultiViewBundleAdjustmentReport, PixelObservation, ScaleGauge, TranslationComponent,
};
#[cfg(feature = "unstable-internals")]
pub use super::factors::{
    BundleAdjustmentPriors, FixedVariableBlocks, PointPrior, PosePrior, RadialIntrinsicPrior,
};
#[cfg(all(feature = "faer", feature = "unstable-internals"))]
pub use super::sparse::{
    SparseSelfCalibrationWorkspace,
    refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors,
    refine_ba_multi_view_self_calibrating_sparse,
};
pub use api::{RadialCamera, SelfCalibratingMeasurements};
#[cfg(feature = "unstable-internals")]
pub use api::{SelfCalibratingBundleAdjustmentProblem, SelfCalibrationWorkspace};
#[cfg(feature = "unstable-internals")]
pub use covariance::{
    covariance_ba_multi_view_self_calibrating_dense,
    covariance_ba_multi_view_self_calibrating_grouped_dense,
};
pub(super) use optimizer::radial_model;
#[cfg(feature = "faer")]
pub(super) use optimizer::{exact_model_reduction, retract_cameras};
#[cfg(feature = "unstable-internals")]
pub use optimizer::{
    refine_ba_multi_view_self_calibrating_dense,
    refine_ba_multi_view_self_calibrating_dense_with_factors,
    refine_ba_multi_view_self_calibrating_grouped_dense,
    refine_ba_multi_view_self_calibrating_grouped_dense_with_factors,
};
