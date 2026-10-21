// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible allocation shared by the graph primitives.

use super::GraphError;

/// Allocate `len` copies of `value`, reporting allocation failure.
pub(super) fn allocate_filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>, GraphError> {
    let mut values = reserved_vec(len)?;
    values.resize(len, value);
    Ok(values)
}

/// Allocate an empty vector with exactly `capacity` reserved slots.
pub(super) fn reserved_vec<T>(capacity: usize) -> Result<Vec<T>, GraphError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| GraphError::AllocationFailed)?;
    Ok(values)
}
