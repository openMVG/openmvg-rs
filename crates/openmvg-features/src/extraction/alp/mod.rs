// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ALP (Analytic Laplacian Polynomial) feature detector and descriptor.
//!
//! Its teaching docs are on [`AlpExtractor`], where docs.rs shows them.

mod blur;
mod geometry_tables;
mod octave_dedup;
#[cfg(test)]
mod reference;
mod scale_fit;
mod stream;
mod tables;
#[cfg(test)]
mod tests;

use super::feature_budget::{FeatureBudget, FeatureBudgetPolicy};
use super::{ScaleSpaceFeatures, ScaleSpaceKeypoint};
use nalgebra as na;
use scale_fit::{SCALE_WINDOW, get_nearest_scale_index, octave_count, octave_scale};
use tables::SIGMAS;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::math_compat::log2f;

/// Grayscale image that every detector reads, with intensities in `[0, 255]`.
///
/// It is a column-major `f32` matrix with one matrix row per image row: `nrows()` is the
/// height, `ncols()` the width, and pixel `(x, y)` is `image[(y, x)]`. Each image column is
/// contiguous in memory, the layout the ALP detector reads fastest. Decoders usually return
/// row-major bytes; convert them in one pass:
///
/// ```
/// use openmvg_features::extraction::GrayImage;
///
/// let (width, height) = (3, 2);
/// let bytes: [u8; 6] = [0, 1, 2, 3, 4, 5]; // row-major, as an image decoder returns them
/// let image = GrayImage::from_fn(height, width, |y, x| f32::from(bytes[y * width + x]));
/// assert_eq!(image[(1, 2)], 5.0); // x = 2, y = 1
/// ```
pub type GrayImage = na::DMatrix<f32>;

/// Errors reported by [`AlpExtractor::validate`] and its detection entry points.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AlpError {
    /// One detector configuration field is outside its supported domain.
    InvalidConfig {
        /// Name of the offending field.
        field: &'static str,
        /// Human-readable description of the accepted domain.
        expected: &'static str,
    },
}

impl core::fmt::Display for AlpError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig { field, expected } => {
                write!(formatter, "ALP {field} must be {expected}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AlpError {}

/// Predefined ALP response thresholds.
///
/// The name is the keypoint density, not the threshold: `High` lowers the response threshold,
/// so it keeps more keypoints, the weaker ones too. `Normal`, the extractor default, keeps
/// fewer, stronger keypoints. Choose `High` for reconstruction and tracks, where more matches
/// help; `openmvg-sfm` uses it. Presets do not resize the input or
/// cap the number of features; use [`AlpExtractor::with_feature_budget`] to limit the output.
///
/// A SIFT contrast threshold `c` (three scales per octave, `[0, 1]` input)
/// corresponds to about `255 · c / ln 2 ≈ 368 · c` here, so each preset pairs
/// with the SIFT preset of the same name: [`SiftPreset::High`](super::SiftPreset::High)
/// (`c = 0.01`, about 3.7) and [`SiftPreset::Normal`](super::SiftPreset::Normal)
/// (`c = 0.04`, about 14.7). Other thresholds go through [`AlpExtractor::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "std", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum AlpPreset {
    /// Low response threshold that also keeps weaker keypoints, for more coverage; the
    /// counterpart of SIFT `High`, and the preset the `openmvg-sfm` front end uses.
    High,
    /// Standard response threshold, the counterpart of SIFT `Normal`, and the preset of
    /// [`AlpExtractor::default`].
    Normal,
}

/// Analytic Laplacian Polynomial (ALP) feature detector with a RootSIFT descriptor.
///
/// ALP finds blob-like keypoints at a continuous scale and describes each one with the
/// 128-value RootSIFT descriptor that [`SiftExtractor`](super::SiftExtractor) also uses, so ALP
/// features go through the same matchers and ratio test as SIFT ones. It is the detector of the
/// `openmvg-sfm` front end. The input is a [`GrayImage`] with intensities in `[0, 255]`; decoding
/// and resizing are the caller's. [`Default`] selects [`AlpPreset::Normal`]. An image smaller
/// than 10 pixels in either dimension yields no features.
///
/// # How it works
///
/// ALP approximates the scale response with a cubic from four σ²-normalized
/// Laplacian-of-Gaussian levels per octave (σ ∈ {1.6, 2.26, 3.2, 4.53}) and takes the extrema of
/// that continuous response in closed form, from the roots of its derivative, so the scale of a
/// keypoint is not quantized to the levels. An octave owns an extremum when an interior level
/// carries its strongest sample, and a candidate must beat its neighbors' cubics within one level
/// step (half an octave) of its scale, the continuous counterpart of SIFT's 26-neighbor test.
/// Curvature checks reject wrong scale polarity, spatial saddles, singular fits, and poorly
/// localized edge responses. Detection streams over the image one column at a time and never
/// holds a full Gaussian or Laplacian plane.
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────────────────┐
/// │             ALP streaming detection and description pipeline             │
/// └──────────────────────────────────────────────────────────────────────────┘
///
///  Input: image (f32, column-major), octave count from its size
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  Pass 1: detection, per octave, one column at a time                │
///  │                                                                     │
///  │  1. Streaming Gaussian levels                                       │
///  │     Separable blur of the octave base → 4 Gaussian columns,         │
///  │     σ ∈ {1.6, 2.26, 3.2, 4.53}, held in 3-column rings              │
///  │     Every second column of σ = 3.2, 2× decimated → next octave base │
///  └─────────────────────────────────────────────────────────────────────┘
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  2. Laplacian and scale response                                    │
///  │     Five-point Laplacian × σ² on the Gaussian ring                  │
///  │     Per pixel: cubic P(σ) fitted to the 4 Laplacian levels; the     │
///  │     roots of P'(σ) give the scale extrema in closed form            │
///  └─────────────────────────────────────────────────────────────────────┘
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  3. Candidate scan                                                  │
///  │     |extremum| > peak_threshold, σ in the window [1.90, 4.15]       │
///  │     Octave owner: an interior level holds the strongest sample      │
///  │     Scale curvature P''(σ) negative at a maximum, positive at a     │
///  │     minimum, and |P''(σ)| > sigma_threshold                         │
///  │     Center beats the eight neighbor cubics over σ/√2 … σ·√2         │
///  └─────────────────────────────────────────────────────────────────────┘
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  4. Refinement                                                      │
///  │     Spatial Hessian H: positive determinant (not a saddle), trace   │
///  │     negative at a maximum and positive at a minimum                 │
///  │     Edge test tr(H)²/det(H) ≤ curv_ratio_threshold                  │
///  │     Sub-pixel quadratic fit on the 3×3 cubics, |offset| ≤ 0.6       │
///  └─────────────────────────────────────────────────────────────────────┘
///        │   repeat for each octave
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  5. Selection                                                       │
///  │     Adjacent-octave duplicates → keep the strongest response        │
///  │     Feature budget on the keypoints, before any description         │
///  └─────────────────────────────────────────────────────────────────────┘
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  Pass 2: description, per octave, kept keypoints only               │
///  │                                                                     │
///  │  6. Re-streamed supports                                            │
///  │     Re-blur only the columns the descriptors cover, 64 at a time;   │
///  │     supports at σ ≥ 3.2 are read from the next octave               │
///  │     Columns and keypoints run in parallel; the output is the same   │
///  │     for every thread count                                          │
///  └─────────────────────────────────────────────────────────────────────┘
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  7. Orientation                                                     │
///  │     Gaussian-weighted 36-bin gradient histogram                     │
///  │     One orientation per keypoint, its highest peak; none → dropped  │
///  └─────────────────────────────────────────────────────────────────────┘
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  8. Descriptor on a grid aligned with the keypoint                  │
///  │     Level blended at the keypoint σ, resampled on 24 × 24 samples   │
///  │     0.625 σ apart in the keypoint frame                             │
///  │     Central differences give the gradients in that frame            │
///  │     4 × 4 cells × 8 orientation bins = 128-value RootSIFT           │
///  └─────────────────────────────────────────────────────────────────────┘
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  9. Reported scale                                                  │
///  │     σ corrected with a Gaussian blob model from the Laplacian       │
///  │     responses, so that a Gaussian blob reports its standard         │
///  │     deviation; the descriptor keeps the detected σ                  │
///  └─────────────────────────────────────────────────────────────────────┘
///
///  Output: keypoints (x, y, σ, θ) with 128-value descriptors, at most one
///          per kept keypoint
///
/// ┌──────────────────────────────────────────────────────────────────────────┐
/// │  Complexity (per image):                                                 │
/// │  Pass 1:  O(W · H)          — all octaves sum to 4/3 of the base pixels  │
/// │  Pass 2:  O(C · H + K)      — C covered columns, K keypoints; the cost   │
/// │                               per keypoint does not grow with σ          │
/// └──────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Memory
///
/// Besides the input, ALP keeps the octave bases, about a third of the image area, and column
/// rings proportional to the image height. Keypoint and descriptor storage grows with the number
/// of features. A feature budget limits the output, not the candidates that detection finds.
///
/// # References
///
/// The continuous-scale detector belongs to the polynomial scale-space family:
/// Koutaki and Uchimura represent Gaussian and scale-normalized LoG scale space
/// with polynomials and build SIFT detectors on it, and Okutani, Sugimoto and
/// Kamata detect and describe keypoints from a polynomial regression of scale
/// space. The orientation and descriptor follow SIFT, with RootSIFT
/// normalization.
///
/// - G. Koutaki and K. Uchimura, "Scale-Space Processing Using Polynomial
///   Representations," CVPR 2014, pp. 2744–2751.
/// - R. Okutani, K. Sugimoto and S. Kamata, "Efficient Keypoint Detection and
///   Description via Polynomial Regression of Scale Space," ICASSP 2016,
///   pp. 1357–1361.
/// - G. Koutaki and K. Uchimura, "XY-Separable Scale-Space Filtering by
///   Polynomial Representations and Its Applications," IEICE Trans. Inf. &
///   Syst., vol. E100.D, no. 4, pp. 645–654, 2017.
/// - D. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints,"
///   IJCV 2004, vol. 60, no. 2, pp. 91–110.
/// - R. Arandjelović and A. Zisserman, "Three things everyone should know
///   to improve object retrieval," CVPR 2012.
#[derive(Debug)]
pub struct AlpExtractor {
    peak_threshold: f32,
    sigma_threshold: f32,
    curv_ratio_threshold: f32,
    feature_budget: FeatureBudgetPolicy,
}

impl AlpExtractor {
    /// Construct the detector for one of the preset operating points.
    pub fn with_preset(preset: AlpPreset) -> Self {
        let peak_threshold = match preset {
            AlpPreset::High => 3.5,
            AlpPreset::Normal => 15.0,
        };
        Self::new(peak_threshold, 0.0, 12.1)
    }

    /// Create an ALP detector with the given thresholds.
    ///
    /// Input intensities must use the public `[0, 255]` detector convention.
    /// Both response thresholds scale linearly with the input intensity range.
    ///
    /// - `peak_threshold`: minimum sigma-squared-normalized LoG response
    ///   magnitude. Higher values detect fewer, stronger features. The presets
    ///   use 3.5 (`High`) and 15.0 (`Normal`).
    /// - `sigma_threshold`: minimum scale-space curvature (second derivative
    ///   of the polynomial) at the scale extremum. It rejects poorly localized
    ///   scale extrema rather than controlling feature density. The presets
    ///   use 0.0: an absolute curvature floor removes low-contrast blobs that
    ///   still match across wide baselines.
    /// - `curv_ratio_threshold`: edge rejection threshold on the Harris-like
    ///   trace²/det ratio, SIFT's `(r+1)²/r` where `r` is the eigenvalue
    ///   ratio. 12.1 corresponds to r=10 (the SIFT standard) and is the preset
    ///   value. It rejects edge responses rather than controlling feature
    ///   density.
    ///
    /// A low `peak_threshold` with an unlimited budget returns all accepted
    /// localized, oriented frames; extraction also requires a valid descriptor.
    /// Their signed response is in [`ScaleSpaceKeypoint::peak`].
    pub fn new(peak_threshold: f32, sigma_threshold: f32, curv_ratio_threshold: f32) -> Self {
        Self {
            peak_threshold,
            sigma_threshold,
            curv_ratio_threshold,
            feature_budget: FeatureBudgetPolicy::Unlimited,
        }
    }

    /// Apply a built-in feature-count policy to the detected keypoints, before
    /// orientation assignment and descriptor construction.
    ///
    /// Each kept keypoint yields at most one feature, at its dominant
    /// orientation, so the output never exceeds the limit. A keypoint without
    /// an orientation peak or a valid descriptor is dropped, not replaced.
    pub fn with_feature_budget(mut self, feature_budget: FeatureBudgetPolicy) -> Self {
        self.feature_budget = feature_budget;
        self
    }

    /// Return the feature-count policy set by [`Self::with_feature_budget`]
    /// ([`FeatureBudgetPolicy::Unlimited`] unless changed).
    pub const fn feature_budget(&self) -> FeatureBudgetPolicy {
        self.feature_budget
    }

    /// Check that every configuration field is inside its supported domain.
    ///
    /// # Errors
    ///
    /// Returns [`AlpError::InvalidConfig`] naming the first invalid field:
    /// `peak_threshold` and `sigma_threshold` must be finite and non-negative,
    /// `curv_ratio_threshold` finite and positive.
    pub fn validate(&self) -> Result<(), AlpError> {
        let checks = [
            (
                "peak_threshold",
                self.peak_threshold.is_finite() && self.peak_threshold >= 0.0,
                "finite and non-negative",
            ),
            (
                "sigma_threshold",
                self.sigma_threshold.is_finite() && self.sigma_threshold >= 0.0,
                "finite and non-negative",
            ),
            (
                "curv_ratio_threshold",
                self.curv_ratio_threshold.is_finite() && self.curv_ratio_threshold > 0.0,
                "finite and positive",
            ),
        ];
        for (field, valid, expected) in checks {
            if !valid {
                return Err(AlpError::InvalidConfig { field, expected });
            }
        }
        Ok(())
    }

    /// Pass-1 parameters derived from the configuration.
    fn detection_params(&self) -> stream::DetectionParams {
        stream::DetectionParams {
            peak_threshold: self.peak_threshold,
            sigma_threshold: self.sigma_threshold,
            curv_ratio_threshold: self.curv_ratio_threshold,
            sigma_window: SCALE_WINDOW,
        }
    }

    /// Run the streaming pipeline under an explicit budget; the configuration must be valid.
    fn run_stream<B>(
        &self,
        image: &GrayImage,
        feature_budget: &B,
        describe: bool,
    ) -> Vec<stream::StreamFeature>
    where
        B: FeatureBudget + ?Sized,
    {
        stream::run(image, &self.detection_params(), feature_budget, describe)
    }

    /// Detect oriented keypoint geometry without constructing descriptors.
    /// Retained results contain no descriptor payload.
    ///
    /// An image smaller than 10 pixels in either dimension yields no features.
    ///
    /// # Errors
    ///
    /// Returns [`AlpError::InvalidConfig`] when [`Self::validate`] rejects the configuration.
    pub fn detect_keypoints(&self, image: &GrayImage) -> Result<Vec<ScaleSpaceKeypoint>, AlpError> {
        self.validate()?;
        Ok(stream::run_stored(
            image,
            &self.detection_params(),
            &self.feature_budget,
            false,
            |_| Some(()),
        )
        .into_iter()
        .map(|feature| Self::public_geometry(&feature))
        .collect())
    }

    /// Public geometry: corrected σ, original-image coordinates.
    #[inline]
    fn public_geometry<D>(feature: &stream::StreamFeature<D>) -> ScaleSpaceKeypoint {
        let scale = octave_scale(feature.anchor.octave);
        let mut geometry = feature.anchor;
        geometry.x *= scale;
        geometry.y *= scale;
        geometry.sigma = feature.public_sigma * scale;
        geometry
    }

    fn typed_result<D: super::storage::DescriptorStorage>(
        stream: &[stream::StreamFeature],
    ) -> ScaleSpaceFeatures<D> {
        let mut features = ScaleSpaceFeatures::default();
        features.keypoints.reserve(stream.len());
        features.descriptors.reserve(stream.len());
        for feature in stream {
            let Some(descriptor) = feature.descriptor.as_ref() else {
                continue;
            };
            let Ok(descriptor) = D::try_from_finalized(descriptor) else {
                continue;
            };
            features.keypoints.push(Self::public_geometry(feature));
            features.descriptors.push(descriptor);
        }
        features
    }

    /// Describe keypoints found elsewhere with the ALP descriptor.
    ///
    /// `positions` holds `(x, y, sigma, orientation)` in input-image pixels and radians. `sigma`
    /// sets the descriptor's sampling scale directly; it is not the corrected scale that
    /// detection reports in [`ScaleSpaceKeypoint::sigma`](super::ScaleSpaceKeypoint::sigma), and
    /// it is clamped to the range the image's octaves cover. A position with a non-finite value,
    /// a non-positive scale, or no valid descriptor is dropped. The result is grouped by scale,
    /// not in input order: read each feature's position and clamped scale from its keypoint.
    pub fn describe_at_support_scale(
        &self,
        image: &GrayImage,
        positions: &[(f32, f32, f32, f32)],
    ) -> ScaleSpaceFeatures<super::storage::F32Descriptor> {
        let max_octaves = octave_count(image);
        if max_octaves == 0 || positions.is_empty() {
            return ScaleSpaceFeatures::default();
        }
        // Group positions by octave: sigma ≈ SIGMAS[s] * 2^oct. Use the highest
        // completed octave below the requested scale; rounding would move
        // half-octave values up and then clamp their local sigma upward.
        let mut by_octave: Vec<Vec<stream::DescribeRequest>> =
            (0..max_octaves).map(|_| Vec::new()).collect();
        for &(x, y, sigma, orientation) in positions {
            if !x.is_finite()
                || !y.is_finite()
                || !sigma.is_finite()
                || !orientation.is_finite()
                || sigma <= 0.0
            {
                continue;
            }
            let sigma_clamped = sigma.max(SIGMAS[0]);
            let oct = (log2f(sigma_clamped / SIGMAS[0]).floor() as usize).min(max_octaves - 1);
            let scale = octave_scale(oct);
            by_octave[oct].push(stream::DescribeRequest {
                x: x / scale,
                y: y / scale,
                sigma: (sigma_clamped / scale).clamp(SIGMAS[0], SIGMAS[3]),
                orientation: Some(orientation),
            });
        }
        let Some(last_used) = by_octave.iter().rposition(|requests| !requests.is_empty()) else {
            return ScaleSpaceFeatures::default();
        };
        // Group indices were assigned against the full sampled octave range.
        by_octave.truncate(last_used + 1);
        let described = stream::describe_requests(image, &by_octave);
        let mut features = ScaleSpaceFeatures::default();
        for (oct, (requests, described)) in by_octave.iter().zip(described).enumerate() {
            let scale = octave_scale(oct);
            for item in described {
                let Some(descriptor) = item.descriptor else {
                    continue;
                };
                let request = requests[item.request];
                features.keypoints.push(ScaleSpaceKeypoint {
                    x: request.x * scale,
                    y: request.y * scale,
                    sigma: request.sigma * scale,
                    orientation: item.orientation,
                    peak: 0.0,
                    octave: oct,
                    scale_index: get_nearest_scale_index(request.sigma),
                    is_max: true,
                });
                features
                    .descriptors
                    .push(super::storage::F32Descriptor::new(descriptor));
            }
        }
        features
    }

    /// Detect keypoints and return typed descriptors.
    ///
    /// The storage type `D` selects the descriptor format: [`F32Descriptor`] (512 bytes per
    /// descriptor) or [`U8Descriptor`] (128 bytes). Either way, each descriptor is a RootSIFT
    /// vector scaled to an L2 norm of 512 with every value capped at 255, the scale SIFT
    /// implementations use; `U8Descriptor` rounds those values to bytes. All descriptors share
    /// that one scale, so a matcher can compare raw bytes directly.
    ///
    /// [`F32Descriptor`]: super::F32Descriptor
    /// [`U8Descriptor`]: super::U8Descriptor
    ///
    /// # Examples
    ///
    /// ```
    /// use openmvg_features::extraction::{AlpExtractor, AlpPreset, GrayImage};
    /// use openmvg_features::extraction::{F32Descriptor, U8Descriptor};
    ///
    /// let image = GrayImage::zeros(32, 32);
    /// let detector = AlpExtractor::with_preset(AlpPreset::Normal);
    ///
    /// let result_f32 = detector.extract_typed::<F32Descriptor>(&image).expect("valid preset");
    /// let result_u8 = detector.extract_typed::<U8Descriptor>(&image).expect("valid preset");
    /// assert_eq!(result_f32.keypoints().len(), result_f32.descriptors().len());
    /// assert_eq!(result_u8.keypoints().len(), result_u8.descriptors().len());
    /// ```
    ///
    /// An image smaller than 10 pixels in either dimension yields no features.
    ///
    /// # Errors
    ///
    /// Returns [`AlpError::InvalidConfig`] when [`Self::validate`] rejects the configuration.
    pub fn extract_typed<D: super::storage::DescriptorStorage>(
        &self,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<D>, AlpError> {
        self.extract_typed_with_budget(image, &self.feature_budget)
    }

    /// Detect typed descriptors with an explicit budget policy.
    ///
    /// An image smaller than 10 pixels in either dimension yields no features.
    ///
    /// # Errors
    ///
    /// Returns [`AlpError::InvalidConfig`] when [`Self::validate`] rejects the configuration.
    fn extract_typed_with_budget<D: super::storage::DescriptorStorage>(
        &self,
        image: &GrayImage,
        feature_budget: &FeatureBudgetPolicy,
    ) -> Result<ScaleSpaceFeatures<D>, AlpError> {
        self.validate()?;
        Ok(Self::typed_result(&self.run_stream(
            image,
            feature_budget,
            true,
        )))
    }

    /// Shorthand for [`extract_typed::<F32Descriptor>`](Self::extract_typed).
    ///
    /// An image smaller than 10 pixels in either dimension yields no features.
    ///
    /// # Errors
    ///
    /// Returns [`AlpError::InvalidConfig`] when [`Self::validate`] rejects the configuration.
    #[inline]
    pub fn extract_f32(
        &self,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<super::storage::F32Descriptor>, AlpError> {
        self.extract_typed(image)
    }

    /// Shorthand for [`extract_typed::<U8Descriptor>`](Self::extract_typed): 128 bytes per
    /// descriptor, on the shared scale that raw-byte matchers compare byte for byte. `openmvg-sfm`
    /// stores and matches these.
    ///
    /// An image smaller than 10 pixels in either dimension yields no features.
    ///
    /// # Errors
    ///
    /// Returns [`AlpError::InvalidConfig`] when [`Self::validate`] rejects the configuration.
    #[inline]
    pub fn extract_u8(
        &self,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<super::storage::U8Descriptor>, AlpError> {
        self.validate()?;
        let stream = stream::run_stored(
            image,
            &self.detection_params(),
            &self.feature_budget,
            true,
            |descriptor| super::storage::U8Descriptor::try_from_finalized(&descriptor).ok(),
        );
        let mut features = ScaleSpaceFeatures::default();
        features.keypoints.reserve(stream.len());
        features.descriptors.reserve(stream.len());
        for feature in stream {
            let geometry = Self::public_geometry(&feature);
            if let Some(descriptor) = feature.descriptor {
                features.keypoints.push(geometry);
                features.descriptors.push(descriptor);
            }
        }
        Ok(features)
    }
}

impl Default for AlpExtractor {
    fn default() -> Self {
        Self::with_preset(AlpPreset::Normal)
    }
}
