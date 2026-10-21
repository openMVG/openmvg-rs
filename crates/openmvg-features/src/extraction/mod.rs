// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature extraction: detect keypoints in a [`GrayImage`] and describe them.
//!
//! Every detector returns [`ScaleSpaceFeatures`]: keypoints with their position, scale and
//! orientation in input-image pixels, and one 128-value descriptor per keypoint. `extract_u8`
//! stores the descriptors as [`U8Descriptor`] (128 bytes each) and `extract_f32` as
//! [`F32Descriptor`] (512 bytes). Both use the one scale that [`U8Descriptor`] describes, and
//! the bytes are the `f32` values rounded. `detect_keypoints` returns the keypoints alone.
//!
//! # Choosing a detector
//!
//! | Detector | Finds | Choose it |
//! |---|---|---|
//! | [`AlpExtractor`] | Laplacian blobs, at a continuous scale | by default; `openmvg-sfm` uses it |
//! | [`SiftExtractor`] | difference-of-Gaussian blobs, on sampled scales | for classic SIFT, a scale that stays more consistent under scale and rotation changes, several orientations per keypoint, or small blobs with [`SiftPreset::Ultra`] |
//!
//! ALP and SIFT share the RootSIFT descriptor, and their `Normal` and `High` presets use
//! equivalent contrast thresholds, so either detector can replace the other and both can be
//! compared on the same images; the `detect_and_compare` example runs both on synthetic blobs.
//! Measure on your own images before you switch.
//!
//! # Limiting the number of features
//!
//! Each detector takes a [`FeatureBudgetPolicy`] through `with_feature_budget`, applied before
//! description, so a cap also saves description time. When you need a cap, use
//! [`FeatureBudgetPolicy::strongest`]: the strongest responses are the keypoints most likely to
//! be found again in other views. A spatial grid spreads keypoints out, but can drop those
//! repeatable ones.

pub(crate) mod alp;
mod convolution;
pub(crate) mod feature_budget;
pub(crate) mod sift;
pub(crate) mod storage;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Geometry and scale-space response shared by SIFT-like detectors.
///
/// Response semantics and scale-level counts are detector-specific. SIFT, for
/// example, stores a DoG response while ALP stores a sigma-normalized LoG
/// response.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct ScaleSpaceKeypoint {
    /// Horizontal position in the caller's image coordinate system.
    /// Public detector outputs use original input-image coordinates.
    pub x: f32,
    /// Vertical position in the caller's image coordinate system.
    /// Public detector outputs use original input-image coordinates.
    pub y: f32,
    /// Detector-specific scale in the caller's image coordinate system.
    /// Public detector outputs use original input-image pixels; scale conventions
    /// depend on the extractor and do not imply a universal Gaussian width.
    pub sigma: f32,
    /// Dominant orientation in radians.
    pub orientation: f32,
    /// Signed response produced by the detector's scale-space operator; the feature budget
    /// keeps the keypoints with the strongest one.
    pub peak: f32,
    /// Detector octave index; zero denotes the first processed octave.
    pub octave: usize,
    /// Detector-specific scale level within the octave.
    pub scale_index: usize,
    /// Whether the scale-space response is a local maximum rather than minimum.
    pub is_max: bool,
}

impl ScaleSpaceKeypoint {
    /// Create a keypoint at `(x, y)` with scale `sigma` and `orientation` in radians, in
    /// input-image pixels. Its response and scale-space indices are zero; a detector of your
    /// own sets them, since the feature budget ranks keypoints by [`Self::peak`].
    #[must_use]
    pub const fn new(x: f32, y: f32, sigma: f32, orientation: f32) -> Self {
        Self {
            x,
            y,
            sigma,
            orientation,
            peak: 0.0,
            octave: 0,
            scale_index: 0,
            is_max: false,
        }
    }
}

/// Scale-space keypoint geometry and typed descriptors with one-to-one alignment.
///
/// The ALP and SIFT detectors return this type. The two sequences always
/// have the same length: descriptor `i` describes keypoint `i`. Only the
/// detectors construct a non-empty value.
#[derive(Clone, Debug)]
pub struct ScaleSpaceFeatures<D> {
    keypoints: Vec<ScaleSpaceKeypoint>,
    descriptors: Vec<D>,
}

impl<D> ScaleSpaceFeatures<D> {
    /// Detected keypoints with position, scale, and orientation, in input-image
    /// coordinates.
    #[must_use]
    pub fn keypoints(&self) -> &[ScaleSpaceKeypoint] {
        &self.keypoints
    }

    /// Descriptors aligned one-to-one with [`Self::keypoints`].
    #[must_use]
    pub fn descriptors(&self) -> &[D] {
        &self.descriptors
    }

    /// Number of detected features (keypoint/descriptor pairs).
    #[must_use]
    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    /// Whether the detector returned no features.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }

    /// Return the aligned keypoints and descriptors.
    #[must_use]
    pub fn into_parts(self) -> (Vec<ScaleSpaceKeypoint>, Vec<D>) {
        (self.keypoints, self.descriptors)
    }
}

impl<D> Default for ScaleSpaceFeatures<D> {
    fn default() -> Self {
        Self {
            keypoints: Vec::new(),
            descriptors: Vec::new(),
        }
    }
}

// The single public path for each descriptor implementation module.
pub use alp::{AlpError, AlpExtractor, AlpPreset, GrayImage};
pub use feature_budget::{
    FeatureBudgetKeypoint, FeatureBudgetPolicy, FeatureCandidate, SpatialGridBudget,
    StrongestResponseBudget, apply_feature_budget,
};
pub use sift::{SiftError, SiftExtractor, SiftPreset};
pub use storage::{
    DescriptorError, DescriptorStorage, F32Descriptor, SIFT_DESCRIPTOR_DIM, U8Descriptor,
};

impl FeatureBudgetKeypoint for ScaleSpaceKeypoint {
    fn feature_candidate(&self) -> FeatureCandidate {
        FeatureCandidate::new(self.x, self.y, self.peak.abs())
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn public_descriptor_errors_implement_std_error() {
        fn assert_error<T: std::error::Error>() {}

        assert_error::<DescriptorError>();
    }
}
