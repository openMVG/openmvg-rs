// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-view bundle adjustment for two or more views.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                 Multi-view bundle adjustment kernel                  │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: poses (R, t) per camera with X_camera = R X + t, camera 0 fixed;
//! points Xᵢ ∈ ℝ³; point-major tracks; one 2-D residual per observation
//! (rad or px); loss; noise σ; scale gauge; optional fixed blocks and priors
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Check the input                                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! Finite poses with R ∈ SO(3), identity anchor, 2 or more observations per
//! free point, projectable points. Failure: a typed error before any update.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Linearize every track                                        │
//! └──────────────────────────────────────────────────────────────────────┘
//! Residual rᵢₖ ∈ ℝ², Jacobians J_p (2×3) and J_c (2×6 pose, +3 intrinsics),
//! √w = √(IRLS weight of ‖r‖/σ) / σ. Accumulate Vᵢ (3×3), gᵢ, Wᵢ, U, g_c.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Eliminate the points (Schur complement)                      │
//! └──────────────────────────────────────────────────────────────────────┘
//! V̂ᵢ = Vᵢ + λ diag(Vᵢ),  S = U + λ diag(U) − Σᵢ Wᵢ V̂ᵢ⁻¹ Wᵢᵀ,
//! ĝ = g_c − Σᵢ Wᵢ V̂ᵢ⁻¹ gᵢ. S is D×D: D = 6(V−1) for V cameras with a free
//! scale gauge, 5 + 6(V−2) with a baseline gauge (no fixed blocks).
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 4: Solve S δc = −ĝ                                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! Dense: Cholesky of S. Sparse (faer): exact LLT with AMD ordering, or
//! block-Jacobi PCG on S without forming it, with an optional LLT fallback.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 5: Recover the points and retract                               │
//! └──────────────────────────────────────────────────────────────────────┘
//! δpᵢ = −V̂ᵢ⁻¹ (gᵢ + Wᵢᵀ δc). Pose blocks are capped at norm 0.5;
//! R ← exp(ω) R, t ← t + s δt (S² chart for a baseline camera), X ← X + s δp;
//! s is the scene scale: RMS point norm, or the baseline norm if fixed.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 6: Accept or reject (Nielsen gain ratio)                        │
//! └──────────────────────────────────────────────────────────────────────┘
//! ρ = actual / predicted cost reduction. ρ > 0 accepts and lowers λ, else λ
//! grows. Stops on gradient, step, relative cost, or the iteration budget.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 7: Certify the final information                                │
//! └──────────────────────────────────────────────────────────────────────┘
//! Undamped reassembly: point-block conditioning, camera rank vs the gauge
//! nullity. Output: poses and points updated in place, plus a report with
//! status (Converged, MaxIterations, Stalled, Degenerate), costs, diagnostics.
//! ```
//!
//! # Problem and method
//!
//! The kernel minimizes `Σ ℓ(‖rᵢₖ‖/σᵢₖ)` over observations k of points i, plus
//! Gaussian prior terms `½‖L r‖²`, by Levenberg-Marquardt (LM) over poses and
//! points. `ℓ` is L2, Huber, or Cauchy; Cauchy starts with
//! `min(max_iterations/4, 5)` L2 iterations. `MultiViewMeasurements` picks the
//! residual: `Angular` (S² logarithm, radians), `Pixel` (pinhole per pose),
//! `FixedPerspective` and `FixedPerspectiveIndexed` (fixed Core pinhole or
//! Brown-Conrady models, pixels). `ScaleGauge::Free` leaves one null direction
//! in `S`, which the final check expects unless a fixed pose, a fixed point, or
//! a prior constrains scale; `FixedBaseline` puts one camera on SO(3) × S², and
//! `FixedTranslationComponent` holds one translation component.
//!
//! The damping `λ` scales the pre-elimination diagonals of `U` and `Vᵢ`,
//! floored per block at `64 ε` times the block maximum. The predicted reduction
//! is that of the undamped model, including the constant that point elimination
//! removes. A step with `ρ > 0` is accepted and sets `λ ← λ·max(1/3,
//! 1−(2ρ−1)³)`; a rejection multiplies `λ` by `ν`, which doubles per
//! consecutive rejection. `λ` starts at 1e-4 (1e-1 for retained PCG). The stop
//! tests are scaled gradient ≤ `gradient·(1+cost)`, step norm ≤ `step`, and
//! relative cost reduction ≤ `relative_cost` (1e-10, 1e-10, 1e-12 in
//! `ConvergenceCriteria::DEFAULT`).
//!
//! The retained solver applies `S x = U x − Σᵢ Wᵢ V̂ᵢ⁻¹ Wᵢᵀ x` point by point,
//! from cached `V̂ᵢ⁻¹ Wᵢᵀ` rows (`f64` or `f32`) or rows recomputed per
//! application, and preconditions with the block diagonal of `S`. A solve is
//! accepted when the recomputed `‖b−S x‖/‖b‖` is at most the tolerance (default
//! 1e-2); ordinary steps run at most `min(maximum_iterations, D)` iterations
//! (default 600). A rejected solve falls back to LLT when `direct_fallback` is
//! set (default). A step or cost stop with a residual above
//! `min(√relative_cost, 1e-6)` is confirmed by a tighter re-solve first.
//! Covariance (`unstable-internals`) inverts the undamped information and sets
//! `Cov(Xᵢ) = Vᵢ⁻¹ + Vᵢ⁻¹ Wᵢᵀ S⁻¹ Wᵢ Vᵢ⁻¹`.
//!
//! # Failure modes
//!
//! Invalid input returns `MultiViewBundleAdjustmentError` before any update.
//! Inside the loop, a non-finite assembly, an unfactorable point block, a
//! Cholesky pivot at most `64 ε` times the largest diagonal, or a non-positive
//! PCG curvature rejects the iteration and raises `λ`; so does a candidate that
//! puts a point on a camera center, without parallax, or outside a projection
//! domain. No accepted step in the final loss stage gives `Stalled`.
//! `Degenerate` means the final undamped system has a point block with
//! reciprocal condition number at most `3·64 ε`, or a camera rank other than
//! `D` minus the expected gauge nullity. With exact rank diagnostics off the
//! camera rank is unassessed (`NotAssessed` in the sparse report) and
//! `is_well_constrained` does not fail on it. Covariance returns
//! `FreeScaleGauge`, `CameraRankDeficient`, or `PointRankDeficient` instead of
//! a matrix.
//!
//! # Alternatives
//!
//! `kernel::two_view` handles exactly two cameras with a fixed-size 5×5 camera
//! Schur system; the native adjuster routes to it through `can_use_two_view`.
//! Translation-only problems, distorted cameras included, use this kernel with
//! `CameraPoseVariables::TranslationOnly`, and `kernel::fixed_rotation` probes
//! their observability before the solve. `self_calibration` adds
//! focal length and radial coefficients as variables. The `dense` path
//! (`unstable-internals`) is the auditable reference. The native adjuster uses
//! the sparse path, where `LinearSolver::Auto` picks LLT when its memory
//! estimate fits four fifths of the budget (1 GiB default), cached retained PCG
//! first for 250 000 or more observations with D ≥ 768 when it fits, and
//! retained PCG when LLT does not fit. `Direct` forces LLT; `Iterative` forces
//! PCG with tolerance 1e-1 and no LLT fallback.

mod api;
mod camera_kernel;
#[cfg(any(test, feature = "unstable-internals"))]
mod covariance;
#[cfg(any(test, feature = "unstable-internals"))]
pub mod dense;
#[cfg(feature = "unstable-internals")]
mod evaluation;
mod factors;
mod model;
mod optimizer;
mod schur;
pub mod self_calibration;
#[cfg(feature = "faer")]
pub mod sparse;
#[cfg(feature = "unstable-internals")]
mod triangulation;
mod validation;

pub use api::{
    AngularObservation, BundleAdjustmentBlock, CameraPoseVariables, MultiViewBundleAdjustmentError,
    MultiViewMeasurements, PixelObservation, ScaleGauge,
};
#[cfg(feature = "unstable-internals")]
pub use api::{
    CalibratedBundleAdjustmentProblem, MultiViewBundleAdjustmentEvaluation,
    MultiViewBundleAdjustmentReport, MultiViewObservationResidual, MultiViewTrackClassification,
    MultiViewTrackDiagnostics, MultiViewTrackLifecycleEvaluation, MultiViewTrackLifecycleSummary,
    MultiViewTriangulationReport,
};
#[cfg(any(test, feature = "unstable-internals"))]
pub use api::{MultiViewBundleAdjustmentDiagnostics, TranslationComponent};
#[cfg(any(test, feature = "unstable-internals"))]
pub use factors::PosePrior;
#[cfg(feature = "unstable-internals")]
pub use factors::RadialIntrinsicPrior;
pub use factors::{BundleAdjustmentPriors, FixedVariableBlocks, PointPrior};
#[cfg(feature = "unstable-internals")]
pub use self_calibration::SelfCalibratingBundleAdjustmentProblem;
