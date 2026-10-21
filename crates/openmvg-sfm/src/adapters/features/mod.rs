// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature extraction adapters and their shared descriptor encodings.

use openmvg_features::extraction::ScaleSpaceKeypoint;
use openmvg_features::extraction::{AlpPreset, DescriptorStorage, SiftPreset, U8Descriptor};

use crate::image::decode::ImageGeometry;
use crate::{Result, SfmError};
use openmvg_features::model::{DescriptorSet, ExtractedFeatures, Keypoint};

pub(crate) mod alp;
pub(crate) mod sift;

#[cfg(feature = "unstable-internals")]
pub use alp::AlpFeatureExtractor;
#[cfg(feature = "unstable-internals")]
pub use sift::SiftFeatureExtractor;
pub use sift::{SiftFrontendConfig, SiftFrontendPreset};

/// Detector density shared by SIFT and ALP, independent of descriptor storage and matching.
///
/// Each level selects the detector preset of the same name, whose contrast thresholds are
/// aligned between the two detectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum DetectorDensity {
    /// Standard contrast threshold: SIFT and ALP `Normal`.
    #[default]
    Normal,
    /// Lower contrast threshold for more coverage: SIFT and ALP `High`.
    High,
    /// SIFT `Ultra`: the `High` threshold with a doubled first octave. ALP has no upsampled
    /// octave and rejects it; raise the maximum image dimension for finer ALP scales.
    Ultra,
}

impl DetectorDensity {
    const fn sift_preset(self) -> SiftPreset {
        match self {
            Self::Normal => SiftPreset::Normal,
            Self::High => SiftPreset::High,
            Self::Ultra => SiftPreset::Ultra,
        }
    }

    /// The ALP preset, or `None` for [`Self::Ultra`].
    pub(crate) const fn alp_preset(self) -> Option<AlpPreset> {
        match self {
            Self::Normal => Some(AlpPreset::Normal),
            Self::High => Some(AlpPreset::High),
            Self::Ultra => None,
        }
    }
}

/// Bind decoded source pixels to the grid declared by the dataset's metadata.
fn validate_decoded_geometry(
    meta: &crate::model::ImageMeta,
    geometry: ImageGeometry,
) -> Result<()> {
    if geometry.source_width() != meta.width as usize
        || geometry.source_height() != meta.height as usize
    {
        return Err(SfmError::InvalidInput(format!(
            "image {} decodes to {}x{}, but metadata declares {}x{}",
            meta.name,
            geometry.source_width(),
            geometry.source_height(),
            meta.width,
            meta.height
        )));
    }
    Ok(())
}

fn source_keypoint(
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    geometry: ImageGeometry,
) -> Result<Keypoint> {
    let (x, y) = geometry.working_to_source(x, y);
    Ok(Keypoint::try_new(
        x,
        y,
        geometry.working_length_to_source(sigma),
        orientation,
    )?)
}

fn source_scale_space_keypoint(
    keypoint: ScaleSpaceKeypoint,
    geometry: ImageGeometry,
) -> Result<Keypoint> {
    source_keypoint(
        keypoint.x,
        keypoint.y,
        keypoint.sigma,
        keypoint.orientation,
        geometry,
    )
}

fn ensure_scale_space_alignment(
    detector: &str,
    keypoint_count: usize,
    descriptor_count: usize,
) -> Result<()> {
    if keypoint_count != descriptor_count {
        return Err(SfmError::InvalidInput(format!(
            "{detector} returned {keypoint_count} keypoints but {descriptor_count} descriptors"
        )));
    }
    Ok(())
}

pub(super) fn encode_scale_space_u8(
    detector: &str,
    keypoints: Vec<ScaleSpaceKeypoint>,
    descriptors: Vec<U8Descriptor>,
    geometry: ImageGeometry,
) -> Result<ExtractedFeatures> {
    ensure_scale_space_alignment(detector, keypoints.len(), descriptors.len())?;
    let keypoints: Result<Vec<_>> = keypoints
        .into_iter()
        .map(|keypoint| source_scale_space_keypoint(keypoint, geometry))
        .collect();
    let data = descriptors
        .iter()
        .flat_map(|descriptor| descriptor.as_slice().iter().copied())
        .collect();
    Ok(ExtractedFeatures::try_new(
        keypoints?,
        DescriptorSet::try_u8(data, 128)?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::{encode_scale_space_u8, source_keypoint};
    use crate::image::decode::{ImageGeometry, load_gray_image_downsampled};
    use image::{GrayImage, Luma};
    use openmvg_features::extraction::{DescriptorStorage, ScaleSpaceKeypoint, U8Descriptor};
    use tempfile::TempDir;

    #[test]
    fn built_in_extractors_reject_a_different_decoded_source_grid() {
        use crate::backends::{FeatureContext, FeatureExtractor};
        use crate::model::ImageMeta;
        let root = TempDir::new().unwrap();
        let path = root.path().join("source.png");
        GrayImage::new(64, 48).save(&path).unwrap();
        let meta = ImageMeta::path(path, 32, 24);
        let context = FeatureContext {
            view: crate::ViewId::new(0),
            seed: crate::RunSeed::from_u64(7),
        };
        let sift = super::sift::SiftFeatureExtractor::new(super::SiftFrontendPreset::Quality);
        let alp = super::alp::AlpFeatureExtractor::new(super::SiftFrontendPreset::Quality);
        for extractor in [
            &sift as &dyn FeatureExtractor,
            &alp as &dyn FeatureExtractor,
        ] {
            let error = extractor.extract(&context, &meta).unwrap_err();
            assert!(matches!(&error, crate::SfmError::InvalidInput(_)));
            assert!(error.to_string().contains("metadata declares"));
        }
    }

    #[test]
    fn shared_scale_space_encoding_is_aligned_on_the_source_grid() {
        let geometry = ImageGeometry::try_new(20, 20, 10, 10).unwrap();
        let keypoint = ScaleSpaceKeypoint::new(3.0, 4.0, 2.0, 0.5);
        let descriptor: [f32; 128] = core::array::from_fn(|index| (index * 17 % 251) as f32);
        let encoded = encode_scale_space_u8(
            "test",
            vec![keypoint],
            vec![U8Descriptor::try_from_finalized(&descriptor).unwrap()],
            geometry,
        )
        .unwrap();
        assert_eq!(encoded.keypoints()[0].x(), 6.5);
        assert_eq!(encoded.keypoints()[0].scale(), 4.0);
        let Some((data, dimension)) = encoded.descriptors().as_u8() else {
            panic!("expected RootSIFT byte storage")
        };
        assert_eq!(dimension, 128);
        assert_eq!(
            data,
            U8Descriptor::try_from_finalized(&descriptor)
                .unwrap()
                .as_slice()
        );

        assert!(encode_scale_space_u8("test", vec![keypoint], Vec::new(), geometry).is_err());
    }

    #[test]
    fn reduced_feature_geometry_returns_to_source_coordinates() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("odd.png");
        GrayImage::from_pixel(7, 5, Luma([0])).save(&path).unwrap();
        let (_, geometry) = load_gray_image_downsampled(&path, 2).unwrap();
        let keypoint = source_keypoint(1.5, 1.0, 2.0, 0.0, geometry).unwrap();

        assert!((keypoint.x() - (2.0 * 7.0 / 3.0 - 0.5)).abs() < 1e-6);
        assert!((keypoint.y() - 3.25).abs() < 1e-6);
        let expected_scale = 2.0_f32 * ((7.0_f32 / 3.0) * (5.0 / 2.0)).sqrt();
        assert!((keypoint.scale() - expected_scale).abs() < 1e-6);
    }
}
