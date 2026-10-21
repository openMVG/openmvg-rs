// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Detector-independent keypoint geometry.

use super::FeatureDataError;

/// Detector-independent keypoint geometry in source-image coordinates.
///
/// Deserialization applies the [`Self::try_new`] checks.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "KeypointWire")]
pub struct Keypoint {
    x: f32,
    y: f32,
    scale: f32,
    orientation: f32,
}

validated_wire!(
    Keypoint,
    #[serde(rename = "Keypoint")]
    struct KeypointWire {
        x: f32,
        y: f32,
        scale: f32,
        orientation: f32,
    }
);

impl Keypoint {
    /// Create a keypoint; see the getters for the meaning and units of each
    /// argument.
    ///
    /// # Errors
    ///
    /// [`FeatureDataError::NonFiniteKeypoint`] if any value is NaN or
    /// infinite, [`FeatureDataError::NegativeKeypointScale`] if `scale < 0`.
    pub fn try_new(x: f32, y: f32, scale: f32, orientation: f32) -> Result<Self, FeatureDataError> {
        let keypoint = Self {
            x,
            y,
            scale,
            orientation,
        };
        keypoint.validate()?;
        Ok(keypoint)
    }

    fn validate(self) -> Result<(), FeatureDataError> {
        if !self.is_finite() {
            return Err(FeatureDataError::NonFiniteKeypoint);
        }
        if self.scale < 0.0 {
            return Err(FeatureDataError::NegativeKeypointScale);
        }
        Ok(())
    }

    /// Whether all four values are finite (always true for a keypoint built
    /// with [`Self::try_new`]).
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.scale.is_finite()
            && self.orientation.is_finite()
    }

    /// Column coordinate in source-image pixels.
    #[must_use]
    pub const fn x(self) -> f32 {
        self.x
    }

    /// Row coordinate in source-image pixels.
    #[must_use]
    pub const fn y(self) -> f32 {
        self.y
    }

    /// Detector scale in source-image pixels (non-negative), for example the
    /// Gaussian sigma of a SIFT detection.
    #[must_use]
    pub const fn scale(self) -> f32 {
        self.scale
    }

    /// Orientation in radians.
    #[must_use]
    pub const fn orientation(self) -> f32 {
        self.orientation
    }
}
