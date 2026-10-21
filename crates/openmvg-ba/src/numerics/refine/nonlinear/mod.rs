// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared nonlinear policy objects, separate from residual linearization.

mod continuation;
mod controller;
mod levenberg_marquardt;
mod progress;
#[cfg(any(feature = "faer", feature = "unstable-internals"))]
mod run_state;

pub(crate) use continuation::{final_stage_index, stages};
#[cfg(any(feature = "faer", feature = "unstable-internals"))]
pub use controller::DEFAULT_INITIAL_DAMPING;
pub use controller::NielsenController;
pub use levenberg_marquardt::{LmProblem, LmSolution, minimize};
pub use progress::{RefinementProgress, StepRejection, StepRejections};
#[cfg(any(feature = "faer", feature = "unstable-internals"))]
pub use run_state::LmRunState;
