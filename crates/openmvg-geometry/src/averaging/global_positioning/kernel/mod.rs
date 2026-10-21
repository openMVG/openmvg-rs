// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-reusing BATA global-positioning kernel with fixed rotations.
//!
//! This module solves for camera centers and optional 3D points from world-frame
//! camera-to-point bearings, oriented camera-to-camera directions, or both. It
//! implements the bounded, depth-independent BATA objective with hidden positive
//! scales. Points are eliminated exactly, and the reduced scalar camera system is
//! factored with the sequential sparse Cholesky of `direct`, analyzed once per plan,
//! with the coordinates solved as three right-hand sides. When that factorization
//! would exceed its memory budget, or without the `faer` feature, the system is
//! solved by fused three-right-hand-side PCG, with a bounded dense Cholesky for
//! small problems and after a failed PCG solve. PCG scans compact camera-pair
//! couplings assembled once per outer iteration; high-degree tracks switch to a
//! matrix-free point-Schur apply when retaining every pair contribution would cost
//! more than four observation slots per active constraint. The projected
//! Gauss-Newton polish that follows factors its 3x3-block camera system with the
//! same sparse Cholesky and runs until its relative cost settles.
//!
//! The checked [`GlobalPositioningPlan`] owns normalized measurements and compact
//! flat topology. [`GlobalPositioningWorkspace`] owns every solve-time buffer, so
//! repeated warmed solves do not allocate. [`GlobalPositioningPlan::solve_into`]
//! is transactional: caller outputs change only after a successful solve.
//!
//! Every declared camera must participate in an active constraint. Frontends that
//! carry extra rotations should compact active cameras before preparation and map
//! the dense solution back afterward. A camera-point pair may appear only once;
//! this is both the checked dense-API contract and the condition under which the
//! matrix-free point-Schur operator exactly matches explicit point elimination.
//!
//! With the optional `parallel` feature, only independent index-aligned passes
//! switch to Rayon, and only above 250,000 active constraints. Inner sums and
//! PCG reductions keep their serial order, so results are bit-identical across
//! thread counts. Smaller problems remain serial to avoid scheduling overhead.

mod api;
mod dense;
#[cfg(feature = "faer")]
mod direct;
mod pcg;
mod plan;
mod polish;
mod robust;
mod solve;
#[cfg(test)]
mod tests;
mod workspace;

pub(super) const DENSE_DIRECT_MAX_CAMERAS: usize = 256;
pub(super) const MATERIALIZED_PAIR_INCIDENCE_FACTOR: usize = 4;
/// Ridge added to every non-anchor diagonal entry of the reduced camera systems.
const RIDGE: f64 = 1e-6;

pub(crate) use api::GlobalPositioningReport;
#[cfg(feature = "unstable-test-support")]
pub use api::ReducedSystemBackend;
pub use api::{
    BataOptions, GlobalPositioningError, GlobalPositioningProblem, PositioningObservation,
};
pub use plan::GlobalPositioningPlan;
#[cfg(feature = "unstable-test-support")]
pub use workspace::GlobalPositioningWorkspace;
