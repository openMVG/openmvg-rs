// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature-extraction contract: [`FeatureExtractor`] and its per-image context.

use openmvg_core::RunSeed;

use crate::model::ImageMeta;
use crate::{Result, ViewId};
use openmvg_features::model::ExtractedFeatures;

/// Inputs passed to a [`FeatureExtractor`] for one image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FeatureContext {
    /// View whose image is being extracted.
    pub view: ViewId,
    /// Per-image stream seed derived from the run seed; the only randomness source an
    /// extractor may use.
    pub seed: RunSeed,
}

/// Detects keypoints and descriptors in one image.
pub trait FeatureExtractor: Send + Sync {
    /// Extract the features of the image described by `meta`.
    fn extract(&self, ctx: &FeatureContext, meta: &ImageMeta) -> Result<ExtractedFeatures>;

    /// Working memory one extraction may need, in GiB, at the largest image size it supports.
    ///
    /// The front end divides the available memory by this value to bound how many images it
    /// extracts at once. The default suits a detector that builds a full scale-space pyramid
    /// of a full-resolution image.
    // Only the parallel front end bounds its workers.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    fn worker_memory_gib(&self) -> f64 {
        4.0
    }
}
