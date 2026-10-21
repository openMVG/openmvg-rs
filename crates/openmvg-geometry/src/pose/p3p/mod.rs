// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Calibrated perspective-three-point pose estimation.
//!
//! Every solver consumes exactly three world points and corresponding unit
//! bearings, returning at most four poses satisfying `b_i ~ R * X_i + t`.
//! Bearings are directed rays: every returned pose has strictly positive depth
//! for all three observations. Invalid input, degenerate geometry, numerical
//! root failure, and absence of a physical solution all return an empty
//! [`P3pSolutions`] rather than an error.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                         P3P family                                   │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: exactly 3 world points Xᵢ and 3 unit bearings bᵢ
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Solver-specific algebraic roots                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! Ding, Ke, Kneip, LambdaTwist, Nakano, and Wu reduce the geometry to
//! conditioned cubic/quartic root problems.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Pose recovery and validation                                 │
//! └──────────────────────────────────────────────────────────────────────┘
//! Recover depths, enforce R ∈ SO(3), and require positive cheirality. Only
//! the `Fused` fallback drops duplicates, so near a multiple root the other
//! methods can return two nearly equal poses.
//!   │
//!   ▼
//! Output: at most 4 physical pose candidates per minimal sample
//!
//! Complexity: fixed-size arithmetic.
//! ```
//!
//! The public surface is the minimal dispatcher [`solve`]; the robust
//! [`estimate_absolute_pose`](super::estimate_absolute_pose) draws its hypotheses from it. [`P3pMethod`]
//! selects the algebraic formulation. [`P3pMethod::Fused`] (the default)
//! adds a depth polish, a Newton-Schulz rotation step, and a Ke/LambdaTwist
//! fallback near a multiple cubic root to the Ding pencil. Keep the default; the other
//! formulations are there to compare against, for example to check a result or to study the
//! methods. Robust scoring uses the
//! unsquared angular residual in radians, infinite behind the camera (see
//! [`ResectionSolver::residual`]).
//!
//! | [`P3pMethod`] | Algebraic path | Intended use |
//! |---|---|---|
//! | [`Fused`](P3pMethod::Fused) | Ding pencil + short polish + Newton-Schulz | default; Ke/LambdaTwist fallback near multiple roots |
//! | [`LambdaTwist`](P3pMethod::LambdaTwist) | cubic + known-zero eigensystem | independent baseline |
//! | [`Ding`](P3pMethod::Ding) | conic-cubic pencil | reference solver; primary path of `Fused` |
//! | [`KeOrtho`](P3pMethod::KeOrtho) | conic quartic | independent fallback/reference |
//! | [`Kneip`](P3pMethod::Kneip) | camera/world frames + quartic | independent frame reference |
//! | [`Wu`](P3pMethod::Wu) | canonical conic quartic | reference solver |
//! | [`Nakano`](P3pMethod::Nakano) | rational quartic | independent reference solver |
//!
//! Every method shares the same [`P3pSolutions`] result. Each method has its
//! own page with its derivation and failure modes: [`fused`], [`lambdatwist`], [`ding`],
//! [`ke`], [`kneip`], [`wu`] and [`nakano`].
//!
//! [`ResectionSolver::residual`]: crate::pose::ResectionSolver::residual

mod common;
pub mod ding;
pub mod fused;
pub mod ke;
pub mod kneip;
pub mod lambdatwist;
pub mod nakano;
pub(super) mod robust;
#[cfg(test)]
mod test_support;
pub mod wu;

pub use common::P3pSolutions;

use ding::DingP3pSolver;
use fused::{FusedP3pConfig, FusedP3pSolver};
use ke::{KeP3pConfig, KeP3pSolver};
use kneip::KneipP3pSolver;
use lambdatwist::LambdaTwistP3pSolver;
use nakano::NakanoP3pSolver;
use wu::WuP3pSolver;

use nalgebra::{Point3, UnitVector3};

/// Algebraic method used for a calibrated P3P minimal solve.
///
/// Keep the default, [`P3pMethod::Fused`]; the other formulations are there to compare against.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum P3pMethod {
    /// Ding pencil with depth polish and a Ke/LambdaTwist fallback (the default), described in
    /// [`fused`].
    #[default]
    Fused,
    /// LambdaTwist cubic formulation, described in [`lambdatwist`].
    LambdaTwist,
    /// Ding conic-cubic formulation, described in [`ding`].
    Ding,
    /// Ke's conic-quartic formulation with orthonormalized rotation recovery, described in
    /// [`ke`].
    KeOrtho,
    /// Kneip camera/world-frame quartic formulation, described in [`kneip`].
    Kneip,
    /// Nakano rational-quartic formulation, described in [`nakano`].
    Nakano,
    /// Wu canonical conic-quartic formulation, described in [`wu`].
    Wu,
}

/// Solve one calibrated P3P sample with the selected algebraic method.
///
/// Returns zero to four cheiral world-to-camera poses in deterministic solver
/// order, without heap allocation. Non-finite input, degenerate geometry and
/// the absence of a physical solution all yield an empty result.
pub fn solve(
    world: [Point3<f64>; 3],
    bearings: [UnitVector3<f64>; 3],
    method: P3pMethod,
) -> P3pSolutions {
    match method {
        P3pMethod::Fused => {
            FusedP3pSolver::solve_minimal(world, bearings, FusedP3pConfig::default())
        }
        P3pMethod::LambdaTwist => LambdaTwistP3pSolver::solve_minimal(world, bearings),
        P3pMethod::Ding => DingP3pSolver::solve_minimal(world, bearings),
        P3pMethod::KeOrtho => KeP3pSolver::solve_minimal(world, bearings, KeP3pConfig),
        P3pMethod::Kneip => KneipP3pSolver::solve_minimal(world, bearings),
        P3pMethod::Nakano => NakanoP3pSolver::solve_minimal(world, bearings),
        P3pMethod::Wu => WuP3pSolver::solve_minimal(world, bearings),
    }
}
