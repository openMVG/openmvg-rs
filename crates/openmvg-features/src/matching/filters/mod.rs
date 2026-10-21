// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature match filtering implementations.
//!
//! - [`LoweRatioFilter`]: Lowe ratio filtering with explicit squared/unsquared entry points
//!
//! Reciprocal matching is selected with [`crate::matching::MatchAcceptance`].

pub(crate) mod distance_ratio;

pub use distance_ratio::LoweRatioFilter;
