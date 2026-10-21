// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared retained-Schur solve evidence.

#[derive(Clone, Copy, Debug)]
pub(crate) struct SchurSolve {
    pub(crate) terms: usize,
    pub(crate) relative_residual: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RejectedSchurSolve {
    pub(crate) terms: usize,
    pub(crate) relative_residual: Option<f64>,
}

pub(crate) enum SchurAttempt {
    Accepted(SchurSolve),
    Rejected(RejectedSchurSolve),
}

pub(super) fn dot3(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

pub(super) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
pub(super) fn norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}
