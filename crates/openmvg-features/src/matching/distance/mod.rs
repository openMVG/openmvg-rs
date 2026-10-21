// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Distance metrics for feature matching.
//!
//! The public items are re-exported from [`crate::matching`], which documents
//! their use.

mod common;
mod hamming;
mod l2;
mod l2_u8;
mod squared_l2;

// Primary API - standalone functions (recommended)
pub use common::{
    // Main functions (automatic SIMD dispatch where available)
    l2_f32,
    l2_f64,
    squared_l2_f32,
    squared_l2_f64,
};

// Hamming distance for binary descriptors
pub use hamming::{HammingDistance, hamming_u8};

// L2 distance for u8 quantized descriptors
#[cfg(test)]
pub(crate) use l2_u8::MAX_SQUARED_L2_U8_DIMENSION;
pub use l2_u8::{L2DistanceU8, SquaredL2DistanceU8, l2_u8, squared_l2_u8};

// Trait-based API (for generic code)
pub use l2::L2Distance;
pub use squared_l2::SquaredL2Distance;
