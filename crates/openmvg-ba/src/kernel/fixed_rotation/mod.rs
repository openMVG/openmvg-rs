// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Observability probe of translation-and-structure problems with fixed rotations.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Fixed-rotation observability probe                                   │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: fixed rotations Rᵢ and intrinsics, centers Cᵢ, points Xₖ, tracks of
//!        (camera, pixel) with at least two observations, gauge, loss, limits
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Validate and fix the gauge                                   │
//! └──────────────────────────────────────────────────────────────────────┘
//! Reject inconsistent sizes, non-finite values, short tracks, a bad gauge
//! or limits. Fix held cameras, the anchor center, and the largest baseline
//! coordinate of the scale camera.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Select a bounded subset of tracks                            │
//! └──────────────────────────────────────────────────────────────────────┘
//! Shortest tracks first, so that every free camera is covered, within the
//! track, observation, and per-track caps.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Eliminate each point in square-root form                     │
//! └──────────────────────────────────────────────────────────────────────┘
//! Per track, the weighted point Jacobians projected off their own column
//! space (thin QR) give a positive semidefinite camera contribution. A point
//! a camera cannot project, or whose block is singular, counts as weak.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 4: Equilibrate and factor the reduced camera system             │
//! └──────────────────────────────────────────────────────────────────────┘
//! Every free coordinate must keep a fraction of its information before
//! elimination. Scale to unit diagonal, shift by the minimum normalized
//! eigenvalue, and run a simplicial sparse Cholesky.
//!   │
//!   ▼
//! Output: FullRankCertified, or NotCertified with the reason (coverage, weak
//!         points, factorization, resource limit) and the counts behind it
//! ```
//!
//! # Problem and method
//!
//! With the rotations and intrinsics held, a camera `i` projects `x = π(Rᵢ (Xₖ − Cᵢ))`, where `π`
//! is the pinhole model with radial `k1, k2, k3` and tangential `p1, p2` distortion. The unknowns
//! are the centers of the cameras that are not held and every point. With rotations fixed, the
//! remaining symmetry is a similarity of the scene: one held camera center and one coordinate of
//! a second center remove it, under both gauges of [`FixedRotationGauge`].
//!
//! A success certifies the whole problem, because the omitted tracks only add positive
//! semidefinite terms. `NotCertified` does not show rank deficiency: the caller defers or runs a
//! complete assessment. Cameras linked only by two-view tracks keep a free baseline length, which
//! the information test of step 4 catches. `NativeBundleAdjuster::preflight` runs this probe on
//! every translation-only problem with one camera model per pose; the solve itself is the sparse
//! multi-view kernel's.
//!
//! # Failure modes
//!
//! [`FixedRotationError`] reports `InvalidProblem` and `InvalidOptions`; a point that cannot be
//! projected makes its track weak instead of failing the probe.

mod api;
mod observability;
#[cfg(test)]
mod tests;
mod validation;

use nalgebra::SMatrix;

#[cfg(feature = "unstable-internals")]
pub use api::FixedRotationError;
pub use api::{
    FixedRotationGauge, FixedRotationIntrinsics, FixedRotationObservabilityProbe,
    FixedRotationObservabilityProbeOptions, FixedRotationObservation, FixedRotationProbeReason,
    FixedRotationProbeStatus, FixedRotationProblem,
};
pub use observability::probe_fixed_rotation_observability;

type Matrix2x3 = SMatrix<f64, 2, 3>;
