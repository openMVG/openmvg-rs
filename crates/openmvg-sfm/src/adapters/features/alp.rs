// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ALP feature extraction for the SfM pipeline.
//!
//! ALP emits 128-D canonical-grid RootSIFT descriptors for the
//! configured [`crate::backends::PairMatcher`] policy.
//!
//! ```text
//! Image path -> grayscale image -> ALP anchors at continuous sigma
//!            -> canonical-grid RootSIFT-u8 features
//! Pipeline: configured pair matching
//! ```

use openmvg_features::extraction::{AlpExtractor, AlpPreset, FeatureBudgetPolicy};

use super::encode_scale_space_u8;
use super::sift::SiftFrontendPreset;
use crate::backends::{FeatureContext, FeatureExtractor};
use crate::image::decode::load_gray_image_max_dimension;
use crate::model::{ImageMeta, ImageSource};
use crate::{Result, SfmError};
use openmvg_features::model::ExtractedFeatures;

/// Path-backed ALP feature extractor.
///
/// Its default uses [`AlpPreset::High`] without a feature-count cap:
/// one dominant orientation per keypoint and canonical-grid descriptors.
#[derive(Debug, Clone)]
pub struct AlpFeatureExtractor {
    preset: SiftFrontendPreset,
    alp_preset: AlpPreset,
    feature_budget: FeatureBudgetPolicy,
    max_dimension: Option<u32>,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl AlpFeatureExtractor {
    /// Extracts with the descriptor encoding of `preset` at native resolution.
    pub fn new(preset: SiftFrontendPreset) -> Self {
        Self {
            preset,
            alp_preset: AlpPreset::High,
            feature_budget: FeatureBudgetPolicy::Unlimited,
            max_dimension: None,
        }
    }

    /// Set the ALP response-threshold preset.
    pub const fn with_alp_preset(mut self, preset: AlpPreset) -> Self {
        self.alp_preset = preset;
        self
    }

    /// Select anchors with this budget before orientation and description.
    pub const fn with_feature_budget(mut self, budget: FeatureBudgetPolicy) -> Self {
        self.feature_budget = budget;
        self
    }

    /// Downsamples images whose longest edge exceeds `max_dimension` pixels before extraction.
    pub fn with_max_dimension(mut self, max_dimension: u32) -> Self {
        self.max_dimension = Some(max_dimension);
        self
    }

    /// Set or clear the extraction resolution cap.
    pub fn with_optional_max_dimension(mut self, max_dimension: Option<u32>) -> Self {
        self.max_dimension = max_dimension;
        self
    }

    /// Return the algorithm and descriptor version identifier.
    ///
    /// Preset, budget and resolution are configured separately.
    pub const fn policy_id(&self) -> &'static str {
        match self.preset {
            SiftFrontendPreset::Quality => "alp_rootsift_u8_v5",
        }
    }
}

impl FeatureExtractor for AlpFeatureExtractor {
    /// ALP streams its scale space: one extraction peaked at about 0.25 GiB on a 25.6 MP
    /// frame (6200 × 4130), the decoded image included. The budget leaves room for larger
    /// sensors.
    fn worker_memory_gib(&self) -> f64 {
        0.75
    }

    fn extract(&self, _ctx: &FeatureContext, meta: &ImageMeta) -> Result<ExtractedFeatures> {
        let ImageSource::Path(path) = &meta.source else {
            return Err(SfmError::InvalidInput(format!(
                "ALP extraction requires path-backed image metadata for `{}`",
                meta.name
            )));
        };
        let (image, geometry) = load_gray_image_max_dimension(path, self.max_dimension)?;
        super::validate_decoded_geometry(meta, geometry)?;
        let detector =
            AlpExtractor::with_preset(self.alp_preset).with_feature_budget(self.feature_budget);
        match self.preset {
            SiftFrontendPreset::Quality => {
                let (keypoints, descriptors) = detector
                    .extract_u8(&image)
                    .map_err(|error| SfmError::InvalidConfig(error.to_string()))?
                    .into_parts();
                encode_scale_space_u8("ALP", keypoints, descriptors, geometry)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};
    use tempfile::TempDir;

    #[test]
    fn policy_id_records_the_detector_and_descriptor_policy() {
        let quality = AlpFeatureExtractor::new(SiftFrontendPreset::Quality);
        assert_eq!(quality.policy_id(), "alp_rootsift_u8_v5");
    }

    #[test]
    fn extracts_blobs_in_source_coordinates_at_both_resolutions() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("blobs.png");
        let image = GrayImage::from_fn(192, 128, |x, y| {
            let blob = |cx: f32, cy: f32| {
                let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                (-d2 / (2.0 * 16.0)).exp()
            };
            let value = 40.0 + 180.0 * (blob(50.0, 40.0) + blob(130.0, 90.0));
            Luma([value.min(255.0) as u8])
        });
        image.save(&path).unwrap();
        let meta = ImageMeta::path(&path, 192, 128);
        let ctx = FeatureContext {
            view: crate::ViewId::new(0),
            seed: openmvg_core::RunSeed::from_u64(0),
        };

        let full = AlpFeatureExtractor::new(SiftFrontendPreset::Quality)
            .extract(&ctx, &meta)
            .unwrap();
        assert!(full.keypoints().iter().any(|keypoint| {
            (keypoint.x() - 50.0).abs() < 2.0 && (keypoint.y() - 40.0).abs() < 2.0
        }));
        assert!(full.descriptors().as_u8().is_some());

        let halved = AlpFeatureExtractor::new(SiftFrontendPreset::Quality)
            .with_max_dimension(96)
            .extract(&ctx, &meta)
            .unwrap();
        assert!(halved.keypoints().iter().any(|keypoint| {
            (keypoint.x() - 130.0).abs() < 3.0 && (keypoint.y() - 90.0).abs() < 3.0
        }));
        assert!(halved.descriptors().as_u8().is_some());
    }
}
