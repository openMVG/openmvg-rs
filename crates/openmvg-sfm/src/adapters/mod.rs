// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Concrete backends for the stage contracts in `backends`: SIFT feature extraction, pair
//! matching, geometric verification, and track triangulation.

#[cfg(feature = "image-io")]
pub mod features;
#[cfg_attr(
    not(any(feature = "image-io", feature = "unstable-internals")),
    allow(dead_code)
)]
pub mod matching;
pub mod triangulation;
#[cfg_attr(
    not(any(feature = "image-io", feature = "unstable-internals")),
    allow(dead_code)
)]
pub mod verification;
