// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SIFT feature extraction for the SfM pipeline.
//!
//! The frontend uses RootSIFT-u8 at the selected detector density. Its preset
//! supplies the corresponding exact [`PairMatcher`]; approximate matching lives
//! in [`crate::adapters::matching`].

use openmvg_features::extraction::U8Descriptor;
use openmvg_features::extraction::{FeatureBudgetPolicy, ScaleSpaceFeatures, SiftExtractor};

use super::{DetectorDensity, encode_scale_space_u8};
use crate::adapters::matching::brute_force::{ROOT_SIFT_U8_RATIO, RootSiftU8Matcher};
use crate::backends::{FeatureContext, FeatureExtractor, PairContext, PairMatcher};
use crate::image::decode::{ImageGeometry, load_gray_image_max_dimension};
use crate::model::ImageSource;
use crate::two_view::PairMatches;
use crate::{Result, SfmError};
use openmvg_features::model::ExtractedFeatures;

/// RootSIFT-u8 extraction and exact brute-force matching policy.
///
/// The retained `quality` tag is part of frontend configuration serialization.
/// Detector density is selected independently by [`DetectorDensity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SiftFrontendPreset {
    /// 128-byte RootSIFT-u8 with exact squared-L2 matching at Lowe ratio 0.8.
    #[default]
    Quality,
}

/// Fully resolved SIFT extraction and descriptor-matching policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SiftFrontendConfig {
    preset: SiftFrontendPreset,
    #[serde(default)]
    density: DetectorDensity,
    max_dimension: Option<u32>,
}

config_accessors!(SiftFrontendConfig {
    /// Descriptor encoding and exact-matching preset.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy preset, with_preset: SiftFrontendPreset;
    /// Detector density. This does not change descriptor encoding or matching.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy density, with_density: DetectorDensity;
});

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl SiftFrontendConfig {
    /// Uses `preset` at normal density and native resolution.
    pub const fn new(preset: SiftFrontendPreset) -> Self {
        Self {
            preset,
            density: DetectorDensity::Normal,
            max_dimension: None,
        }
    }

    /// Longest image edge used during extraction. `None` keeps native resolution.
    #[must_use]
    pub const fn max_dimension(&self) -> Option<u32> {
        self.max_dimension
    }

    /// Downsamples images whose longest edge exceeds `max_dimension` pixels before extraction.
    pub const fn with_max_dimension(mut self, max_dimension: u32) -> Self {
        self.max_dimension = Some(max_dimension);
        self
    }

    /// Set or clear the extraction resolution cap.
    pub(crate) const fn with_optional_max_dimension(mut self, max_dimension: Option<u32>) -> Self {
        self.max_dimension = max_dimension;
        self
    }

    /// Builds the extractor for this preset, density, and resolution cap.
    pub fn extractor(self) -> SiftFeatureExtractor {
        let extractor = self.preset.extractor().with_density(self.density);
        match self.max_dimension {
            Some(max_dimension) => extractor.with_max_dimension(max_dimension),
            None => extractor,
        }
    }
}

impl Default for SiftFrontendConfig {
    fn default() -> Self {
        Self::new(SiftFrontendPreset::Quality)
    }
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl SiftFrontendPreset {
    /// Stable identifier of the extraction policy, such as `sift_rootsift_u8_v1`.
    pub const fn policy_id(self) -> &'static str {
        match self {
            Self::Quality => "sift_rootsift_u8_v1",
        }
    }

    /// Lowe ratio threshold of the preset's exact matcher.
    pub const fn ratio(self) -> f32 {
        match self {
            Self::Quality => ROOT_SIFT_U8_RATIO,
        }
    }

    /// Builds an extractor for this preset at normal density and native resolution.
    pub fn extractor(self) -> SiftFeatureExtractor {
        SiftFeatureExtractor::new(self)
    }

    /// Report name of the preset's exact matcher.
    pub const fn matcher_name(self) -> &'static str {
        match self {
            Self::Quality => "rootsift_u8_exact_l2_ratio_0.8",
        }
    }
}

/// The preset's exact brute-force matcher at its default Lowe ratio.
impl PairMatcher for SiftFrontendPreset {
    fn match_features(
        &self,
        ctx: &PairContext,
        fi: &ExtractedFeatures,
        fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        match self {
            Self::Quality => RootSiftU8Matcher::default().match_features(ctx, fi, fj),
        }
    }
}

/// Path-backed SIFT feature extractor.
#[derive(Debug, Clone)]
pub struct SiftFeatureExtractor {
    preset: SiftFrontendPreset,
    density: DetectorDensity,
    feature_budget: FeatureBudgetPolicy,
    max_dimension: Option<u32>,
}

impl SiftFeatureExtractor {
    /// Extracts with `preset` at normal density and native resolution.
    pub fn new(preset: SiftFrontendPreset) -> Self {
        Self {
            preset,
            density: DetectorDensity::Normal,
            feature_budget: FeatureBudgetPolicy::Unlimited,
            max_dimension: None,
        }
    }

    /// Sets the detector density.
    pub const fn with_density(mut self, density: DetectorDensity) -> Self {
        self.density = density;
        self
    }

    /// Select oriented candidates with this budget before description.
    pub const fn with_feature_budget(mut self, budget: FeatureBudgetPolicy) -> Self {
        self.feature_budget = budget;
        self
    }

    /// Downsamples images whose longest edge exceeds `max_dimension` pixels before extraction.
    pub fn with_max_dimension(mut self, max_dimension: u32) -> Self {
        self.max_dimension = Some(max_dimension);
        self
    }
}

impl FeatureExtractor for SiftFeatureExtractor {
    /// SIFT builds full Gaussian and difference-of-Gaussian pyramids: one extraction peaked at
    /// about 1.6 GiB on a 25.6 MP frame (6200 × 4130).
    fn worker_memory_gib(&self) -> f64 {
        4.0
    }

    fn extract(
        &self,
        _ctx: &FeatureContext,
        meta: &crate::model::ImageMeta,
    ) -> Result<ExtractedFeatures> {
        let ImageSource::Path(path) = &meta.source else {
            return Err(SfmError::InvalidInput(format!(
                "SIFT extraction requires path-backed image metadata for `{}`",
                meta.name
            )));
        };
        let (image, geometry) = load_gray_image_max_dimension(path, self.max_dimension)?;
        super::validate_decoded_geometry(meta, geometry)?;
        let detector = SiftExtractor::with_preset(self.density.sift_preset())
            .with_feature_budget(self.feature_budget);
        match self.preset {
            SiftFrontendPreset::Quality => {
                let features = detector
                    .extract_u8(&image)
                    .map_err(|error| SfmError::InvalidConfig(error.to_string()))?;
                encode_sift_u8(features, geometry)
            }
        }
    }
}

fn encode_sift_u8(
    features: ScaleSpaceFeatures<U8Descriptor>,
    geometry: ImageGeometry,
) -> Result<ExtractedFeatures> {
    let (keypoints, descriptors) = features.into_parts();
    encode_scale_space_u8("SIFT", keypoints, descriptors, geometry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmvg_features::extraction::ScaleSpaceKeypoint;

    #[test]
    fn presets_bind_storage_ratio_and_matcher() {
        assert_eq!(SiftFrontendPreset::default(), SiftFrontendPreset::Quality);
        let quality = SiftFrontendPreset::Quality;
        assert_eq!(quality.policy_id(), "sift_rootsift_u8_v1");
        assert_eq!(quality.extractor().preset, quality);
        assert_eq!(quality.ratio(), ROOT_SIFT_U8_RATIO);
        assert_eq!(quality.matcher_name(), "rootsift_u8_exact_l2_ratio_0.8");
    }

    #[test]
    fn density_is_explicit_and_partial_config_defaults_to_normal() {
        let high = SiftFrontendConfig::new(SiftFrontendPreset::Quality)
            .with_density(DetectorDensity::High)
            .extractor();
        assert_eq!(high.density, DetectorDensity::High);

        let partial: SiftFrontendConfig =
            serde_json::from_str(r#"{"preset":"quality","max_dimension":1368}"#).unwrap();
        assert_eq!(partial.density, DetectorDensity::Normal);
    }

    #[test]
    fn rootsift_encoding_preserves_source_keypoint_geometry() {
        let keypoint = ScaleSpaceKeypoint::new(3.0, 4.0, 2.0, 0.5);
        let descriptor: [f32; 128] = core::array::from_fn(|index| index as f32);
        let geometry = ImageGeometry::try_new(20, 20, 10, 10).unwrap();
        let root = encode_scale_space_u8(
            "SIFT",
            vec![keypoint],
            vec![U8Descriptor::try_from_finalized(&descriptor).unwrap()],
            geometry,
        )
        .unwrap();
        assert_eq!(root.keypoints()[0].x(), 6.5);
        assert_eq!(root.keypoints()[0].scale(), 4.0);
    }
}
