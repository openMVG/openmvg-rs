// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature matcher implementations.
//!
//! This module is the one public path for the concrete matchers and their
//! configuration types; the implementation submodules are private.

mod base;
mod brute_force;
mod hnsw;

pub use brute_force::BruteForceMatcher;
pub use hnsw::{HnswConfig, HnswMatcher};

/// Borrow owned test descriptors as the slice batch the matcher APIs take.
#[cfg(test)]
pub(crate) fn as_slices<T>(features: &[Vec<T>]) -> Vec<&[T]> {
    features.iter().map(Vec::as_slice).collect()
}
