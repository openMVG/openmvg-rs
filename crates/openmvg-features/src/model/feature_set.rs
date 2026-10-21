// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-image feature containers: extraction output and persistent feature geometry.

use nalgebra::Point2;
use openmvg_core::model::FeatureId;

use super::{DescriptorSet, FeatureDataError, Keypoint};

/// The keypoints of one image paired with their descriptors, descriptor `i` describing
/// keypoint `i`: what matching an image needs.
///
/// Deserialization applies [`Self::validate`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "ExtractedFeaturesWire")]
pub struct ExtractedFeatures {
    keypoints: Vec<Keypoint>,
    descriptors: DescriptorSet,
}

validated_wire!(
    ExtractedFeatures,
    #[serde(rename = "ExtractedFeatures")]
    struct ExtractedFeaturesWire {
        keypoints: Vec<Keypoint>,
        descriptors: DescriptorSet,
    }
);

impl ExtractedFeatures {
    /// Pair keypoints with their descriptors; descriptor `i` describes
    /// keypoint `i`.
    ///
    /// # Errors
    ///
    /// Any error of [`Self::validate`].
    pub fn try_new(
        keypoints: Vec<Keypoint>,
        descriptors: DescriptorSet,
    ) -> Result<Self, FeatureDataError> {
        let features = Self {
            keypoints,
            descriptors,
        };
        features.validate()?;
        Ok(features)
    }

    /// Check keypoint validity, descriptor validity and that both have the
    /// same count.
    ///
    /// # Errors
    ///
    /// [`FeatureDataError::NonFiniteKeypoint`],
    /// [`FeatureDataError::NegativeKeypointScale`], any
    /// [`DescriptorSet::validate`] error, or
    /// [`FeatureDataError::FeatureDescriptorCountMismatch`].
    pub fn validate(&self) -> Result<(), FeatureDataError> {
        if self.keypoints.iter().any(|keypoint| !keypoint.is_finite()) {
            return Err(FeatureDataError::NonFiniteKeypoint);
        }
        if self.keypoints.iter().any(|keypoint| keypoint.scale() < 0.0) {
            return Err(FeatureDataError::NegativeKeypointScale);
        }
        self.descriptors.validate()?;
        if self.keypoints.len() != self.descriptors.len() {
            return Err(FeatureDataError::FeatureDescriptorCountMismatch {
                keypoints: self.keypoints.len(),
                descriptors: self.descriptors.len(),
            });
        }
        Ok(())
    }

    /// Number of features.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    /// Whether there are no features.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }

    /// Keypoints; index `i` is feature `i`.
    #[must_use]
    pub fn keypoints(&self) -> &[Keypoint] {
        &self.keypoints
    }

    /// Descriptors, in keypoint order.
    #[must_use]
    pub const fn descriptors(&self) -> &DescriptorSet {
        &self.descriptors
    }

    /// Clone the compact pixel geometry retained after matching data is released.
    #[must_use]
    pub fn feature_set(&self) -> FeatureSet {
        FeatureSet {
            pixels: self
                .keypoints
                .iter()
                .map(|keypoint| Point2::new(f64::from(keypoint.x()), f64::from(keypoint.y())))
                .collect(),
        }
    }
}

/// Persistent per-view feature geometry addressed by [`FeatureId`].
///
/// Deserialization applies [`Self::validate`].
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "FeatureSetWire")]
pub struct FeatureSet {
    pixels: Vec<Point2<f64>>,
}

validated_wire!(
    FeatureSet,
    #[serde(rename = "FeatureSet")]
    struct FeatureSetWire {
        pixels: Vec<Point2<f64>>,
    }
);

impl FeatureSet {
    /// Create a feature set from pixel positions `(x = column, y = row)` in
    /// source-image pixels; index `i` is [`FeatureId`] `i`.
    ///
    /// # Errors
    ///
    /// [`FeatureDataError::NonFinitePixel`] if a coordinate is NaN or infinite.
    pub fn try_new(pixels: Vec<Point2<f64>>) -> Result<Self, FeatureDataError> {
        let set = Self { pixels };
        set.validate()?;
        Ok(set)
    }

    /// Check that every coordinate is finite.
    ///
    /// # Errors
    ///
    /// [`FeatureDataError::NonFinitePixel`].
    pub fn validate(&self) -> Result<(), FeatureDataError> {
        if self
            .pixels
            .iter()
            .any(|pixel| !pixel.coords.iter().all(|value| value.is_finite()))
        {
            return Err(FeatureDataError::NonFinitePixel);
        }
        Ok(())
    }

    /// Number of features.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Whether there are no features.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Pixel position of `feature`, in source-image pixels.
    ///
    /// # Errors
    ///
    /// [`FeatureDataError::FeatureOutOfRange`] if `feature` is not in the set.
    pub fn pixel(&self, feature: FeatureId) -> Result<Point2<f64>, FeatureDataError> {
        self.pixels
            .get(feature.index())
            .copied()
            .ok_or(FeatureDataError::FeatureOutOfRange {
                feature,
                count: self.pixels.len(),
            })
    }

    /// All pixel positions, indexed by [`FeatureId`].
    #[must_use]
    pub fn pixels(&self) -> &[Point2<f64>] {
        &self.pixels
    }
}
