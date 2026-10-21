// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic SIFT (Scale-Invariant Feature Transform) detector and descriptor.
//!
//! Its teaching docs are on [`SiftExtractor`], where docs.rs shows them.
pub(crate) mod canonical_descriptor;
pub(crate) mod common;
mod detector;
mod localization;
mod scale_space;

pub use detector::SiftExtractor;

// ============================================================================
// Errors
// ============================================================================

/// Errors reported by [`SiftExtractor::validate`] and its detection entry points.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SiftError {
    /// One detector configuration field is outside its supported domain.
    InvalidConfig {
        /// Name of the offending field.
        field: &'static str,
        /// Human-readable description of the accepted domain.
        expected: &'static str,
    },
}

impl core::fmt::Display for SiftError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig { field, expected } => {
                write!(formatter, "SIFT {field} must be {expected}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SiftError {}

// ============================================================================
// SIFT Presets
// ============================================================================

/// Predefined SIFT thresholds, the presets of OpenMVG's C++ SIFT Anatomy describer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SiftPreset {
    /// Standard contrast threshold at native resolution: `peak_threshold = 0.04`,
    /// `first_octave = 0`. The preset of [`SiftExtractor::default`].
    Normal,
    /// Lower contrast threshold that also keeps weaker keypoints: `peak_threshold = 0.01`,
    /// `first_octave = 0`.
    High,
    /// The `High` threshold on an image doubled first (`first_octave = -1`), to find smaller
    /// blobs at a higher cost.
    Ultra,
}

#[cfg(test)]
mod tests;
