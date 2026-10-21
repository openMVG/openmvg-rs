// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Incremental and Global engines: when to run which operation of `ops`, and which
//! results to keep.
//!
//! Branch modules are crate-private. The crate root exposes the supported API, keeping internal
//! file organization free to evolve without creating deep public import paths.

pub(crate) mod global;
pub(crate) mod incremental;
