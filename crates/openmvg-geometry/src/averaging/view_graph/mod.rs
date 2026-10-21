// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # View graph conditioning
//!
//! Edge-outlier handling for pairwise relative-rotation graphs, applied before the global
//! averaging solves.
//!
//! ## Pipeline position
//!
//! ```text
//! Pairwise R_ij (verified pairs)
//!     │
//!     ▼
//! ┌────────────────────────────┐
//! │ condition_edges_adaptive   │  cycle reliability + graduated robust solve
//! └───────────┬────────────────┘
//!             ▼  per-edge confidence, rotations, residuals
//! ┌────────────────────────────┐
//! │ cut_edges_protecting_      │  drop low-score edges, never a bridge
//! │ bridges                    │
//! └───────────┬────────────────┘
//!             ▼
//!   rotation averaging, translation averaging, positioning
//! ```
//!
//! | Function | Method |
//! |----------|--------|
//! | [`condition_edges_adaptive`] | Cycle reliability plus a graduated robust solve, scales from the data, no cut |
//! | [`cut_edges_protecting_bridges`] | Hard cut of low-score edges that keeps the graph connected |
//! | [`keep_strongest_edges_per_view`] | Keeps each view's best verified pairs by inliers, dropping the weak pairs that chain unrelated parts of a scene |

mod adaptive;
mod conditioning;
mod strongest_edges;

pub use adaptive::{
    AdaptiveConditioning, AdaptiveConditioningError, AdaptiveEdge, condition_edges_adaptive,
};
pub use conditioning::{EdgeCut, EdgeCutError, cut_edges_protecting_bridges};
pub use strongest_edges::{StrengthEdge, StrongestEdgesError, keep_strongest_edges_per_view};
