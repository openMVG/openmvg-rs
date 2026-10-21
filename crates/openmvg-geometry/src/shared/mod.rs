// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared utilities for geometric solvers.
//!
//! Contains common algorithms and helpers used across multiple solvers.

pub(crate) mod epipolar_constraint;
pub(crate) mod polynomial;
pub(crate) mod svd_utils;
