// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Initialized pose-refinement kernel behind `refine_bearing_pose` and `refine_perspective_pose`.
//!
//! Its teaching docs are on [`refine_bearing_pose`], where docs.rs shows them.

mod api;
mod optimizer;
mod projection;
mod state;

pub use api::{PnpError, PnpReport, PnpResidualUnit};
pub use optimizer::{optimize_bearing_pnp, optimize_pixel_pnp};

pub use state::RefinementPose;
