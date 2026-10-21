// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The public SIFT detector: configuration, the octave-by-octave detection
//! loop with the feature budget, and descriptor construction.

use super::super::alp::GrayImage;
use super::super::feature_budget::{
    FeatureBudget, FeatureBudgetKeypoint, FeatureBudgetPolicy, selection_mask,
};
use super::super::storage::DescriptorStorage;
use super::super::{ScaleSpaceFeatures, ScaleSpaceKeypoint};
use super::common::{
    SIFT_DESCR_SIZE, SiftDescriptorScratch, compute_sift_descriptor_from_gradients_with_scratch,
};
use super::localization::{
    SiftFeature, assign_orientations, find_3d_discrete_extrema, refine_keypoints,
};
use super::scale_space::{
    GradientOctave, HierarchicalGaussianScaleSpace, compute_dogs, compute_gradients,
};
use super::{SiftError, SiftPreset};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

const MIN_INPUT_DIMENSION: usize = 8;
const BASE_SIGMA: f32 = 1.6;
const ASSUMED_INPUT_SIGMA: f32 = 0.5;

// ============================================================================
// SIFT Descriptor Computation
// ============================================================================

/// Compute a SIFT descriptor from a precomputed gradient octave.
///
/// Coordinates are converted to octave space before being passed to the
/// generic builder (same convention as `assign_orientations`).
fn compute_descriptor(
    grads: &GradientOctave,
    keypoint: &SiftFeature,
    root_sift: bool,
    scratch: &mut SiftDescriptorScratch,
) -> Option<[f32; SIFT_DESCR_SIZE]> {
    let gradient = grads.slices[keypoint.scale_index].as_ref()?;

    let delta = grads.delta;
    let x = keypoint.x / delta;
    let y = keypoint.y / delta;
    let sigma = keypoint.sigma / delta;

    compute_sift_descriptor_from_gradients_with_scratch(
        x,
        y,
        sigma,
        keypoint.orientation,
        root_sift,
        &gradient.x,
        &gradient.y,
        gradient.rows,
        gradient.cols,
        scratch,
    )
}

/// Compute valid descriptors and visit them in keypoint order.
fn for_each_descriptor(
    keypoints: Vec<SiftFeature>,
    grads: &GradientOctave,
    root_sift: bool,
    mut visit: impl FnMut(SiftFeature, [f32; SIFT_DESCR_SIZE]),
) {
    #[cfg(feature = "parallel")]
    {
        let descriptors: Vec<_> = keypoints
            .par_iter()
            .map_init(SiftDescriptorScratch::default, |scratch, keypoint| {
                compute_descriptor(grads, keypoint, root_sift, scratch)
            })
            .collect();
        for (keypoint, descriptor) in keypoints.into_iter().zip(descriptors) {
            if let Some(descriptor) = descriptor {
                visit(keypoint, descriptor);
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = SiftDescriptorScratch::default();
        for keypoint in keypoints {
            if let Some(descriptor) = compute_descriptor(grads, &keypoint, root_sift, &mut scratch)
            {
                visit(keypoint, descriptor);
            }
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Classic SIFT (Scale-Invariant Feature Transform) feature extractor.
///
/// SIFT finds difference-of-Gaussian blobs on a sampled scale space and describes each one with
/// a 128-value descriptor, RootSIFT by default. [`Default`] is [`SiftPreset::Normal`]; start from a
/// preset and adjust individual parameters with the `with_*` setters. SIFT can emit several
/// orientations, and so several features, for one keypoint, and [`SiftPreset::Ultra`] doubles
/// the image first to find smaller blobs.
///
/// # How it works
///
/// The pipeline follows the "Anatomy of the SIFT Method" (Rey Otero & Delbracio, IPOL 2013), the
/// method of OpenMVG's C++ SIFT Anatomy describer.
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────────────────┐
/// │                   SIFT detection and description pipeline                │
/// └──────────────────────────────────────────────────────────────────────────┘
///
///  Input: image (f32)
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  1. Gaussian scale space                                            │
///  │     num_scales + 3 Gaussian blurs per octave                        │
///  │     2× decimation between octaves                                   │
///  │     Gradient images (gx, gy) precomputed per octave                 │
///  └─────────────────────────────────────────────────────────────────────┘
///        │   repeat for each octave
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  2. Difference of Gaussians                                         │
///  │     num_scales + 2 DoG slices per octave                            │
///  │     `DoG[i] = Gaussian[i+1] - Gaussian[i]`                          │
///  └─────────────────────────────────────────────────────────────────────┘
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  3. Discrete extrema                                                │
///  │     26-neighbor check across (x, y, scale)                          │
///  │     Contrast: |DoG| > 0.8 · peak_threshold / num_scales             │
///  └─────────────────────────────────────────────────────────────────────┘
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  4. Sub-pixel refinement                                            │
///  │     3D Taylor expansion → Hessian inversion → sub-pixel offset      │
///  │     Shift the sample until every |offset| < 0.6  (≤ 5 steps)        │
///  │     Contrast: |interpolated DoG| > peak_threshold / num_scales      │
///  └─────────────────────────────────────────────────────────────────────┘
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  5. Edge rejection                                                  │
///  │     Hessian ratio (Hxx + Hyy)² / det(H)                             │
///  │     det ≤ 0 (saddle point) → reject                                 │
///  │     ratio > (r + 1)² / r with r = edge_threshold → reject           │
///  └─────────────────────────────────────────────────────────────────────┘
///        │
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  6. Orientation assignment                                          │
///  │     36-bin gradient histogram, Gaussian-weighted by σ window        │
///  │     6-pass [¼ ½ ¼] circular smoothing                               │
///  │     Every peak ≥ 80% of the highest gives an orientation            │
///  │     Parabolic sub-bin interpolation → continuous angle              │
///  └─────────────────────────────────────────────────────────────────────┘
///        │   one keypoint may emit multiple orientations
///        ▼
///  ┌─────────────────────────────────────────────────────────────────────┐
///  │  7. Descriptor                                                      │
///  │     4×4 spatial grid × 8 orientation bins → 128-value vector        │
///  │     Trilinear accumulation of the gradients                         │
///  │     Classic: L2 → clip 0.2 → L2; RootSIFT: L1 → square root         │
///  └─────────────────────────────────────────────────────────────────────┘
///        │
///        ▼
///  Output: keypoints (x, y, σ, θ) with 128-value descriptors
///
/// ┌──────────────────────────────────────────────────────────────────────────┐
/// │  Complexity (per image):                                                 │
/// │  Scale space: O(N_oct · S · W · H)  — N_oct octaves, S scales/octave     │
/// │  Extrema:     O(N_oct · S · W · H)  — 26-neighbor scan                   │
/// │  Descriptor:  O(K · R²)             — K keypoints, R ≈ 10.6σ half-width  │
/// └──────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Examples
///
/// ```
/// use openmvg_features::extraction::GrayImage;
/// use openmvg_features::extraction::{SiftExtractor, SiftPreset};
///
/// let image = GrayImage::zeros(32, 32);
/// let extractor = SiftExtractor::with_preset(SiftPreset::High).with_num_octaves(4);
/// let features = extractor.extract_f32(&image).expect("valid preset");
/// assert_eq!(features.keypoints().len(), features.descriptors().len());
/// assert!(features.keypoints().iter().all(|keypoint| keypoint.x.is_finite()));
/// ```
///
/// # References
///
/// - D. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints,"
///   IJCV 2004, vol. 60, no. 2, pp. 91–110.
/// - I. Rey Otero & M. Delbracio, "Anatomy of the SIFT Method," IPOL 2013.
/// - R. Arandjelović & A. Zisserman, "Three things everyone should know
///   to improve object retrieval," CVPR 2012.
#[derive(Clone, Debug)]
pub struct SiftExtractor {
    pub(super) first_octave: i32,
    pub(super) num_octaves: usize,
    pub(super) num_scales: usize,
    pub(super) peak_threshold: f32,
    pub(super) edge_threshold: f32,
    pub(super) root_sift: bool,
    pub(super) feature_budget: FeatureBudgetPolicy,
}

config_accessors!(SiftExtractor {
    /// First octave: 0 = native resolution, -1 = 2× upscale.
    copy first_octave, with_first_octave: i32;
    /// Maximum number of octaves.
    copy num_octaves, with_num_octaves: usize;
    /// Number of scales per octave (default: 3).
    copy num_scales, with_num_scales: usize;
    /// DoG contrast threshold (default: 0.04), expressed on a `[0, 1]` intensity scale. The
    /// image itself stays in `[0, 255]`, as every detector expects; SIFT rescales it inside.
    ///
    /// Detection compares DoG values with `peak_threshold / num_scales`, and prefilters the
    /// discrete extrema at 80% of that.
    copy peak_threshold, with_peak_threshold: f32;
    /// Hessian eigenvalue ratio threshold (default: 10.0).
    copy edge_threshold, with_edge_threshold: f32;
    /// Apply RootSIFT normalization (default: true).
    copy root_sift, with_root_sift: bool;
    /// Feature-count policy applied after orientation assignment and before
    /// descriptor construction (default: [`FeatureBudgetPolicy::Unlimited`]).
    ///
    /// The budget is global over all octaves, and only the selected features are described.
    /// Each orientation is one candidate, ranked by its absolute interpolated DoG response;
    /// selected features keep detector order. The count is an upper bound, because a keypoint
    /// without a valid descriptor is dropped.
    copy feature_budget, with_feature_budget: FeatureBudgetPolicy;
});

impl SiftExtractor {
    /// Create a SIFT detector from a preset.
    ///
    /// Every preset uses 6 octaves, 3 scales per octave, edge threshold 10 and
    /// RootSIFT; the presets differ only in `peak_threshold` and `first_octave`
    /// (see [`SiftPreset`]). Adjust individual parameters with the `with_*`
    /// setters.
    pub const fn with_preset(preset: SiftPreset) -> Self {
        let (first_octave, peak_threshold) = match preset {
            SiftPreset::Normal => (0, 0.04),
            SiftPreset::High => (0, 0.01),
            SiftPreset::Ultra => (-1, 0.01),
        };
        Self {
            first_octave,
            num_octaves: 6,
            num_scales: 3,
            peak_threshold,
            edge_threshold: 10.0,
            root_sift: true,
            feature_budget: FeatureBudgetPolicy::Unlimited,
        }
    }

    /// Check that every configuration field is inside its supported domain.
    ///
    /// # Errors
    ///
    /// Returns [`SiftError::InvalidConfig`] naming the first invalid field:
    /// `num_octaves` and `num_scales` must be positive, the scale-level count must fit its
    /// metadata buffers, `first_octave` must be
    /// `-1` or `0`, `peak_threshold` finite and non-negative, and
    /// `edge_threshold` finite and positive.
    pub fn validate(&self) -> Result<(), SiftError> {
        let scale_count_is_representable = self
            .num_scales
            .checked_add(super::scale_space::SUPPLEMENTARY_LEVELS)
            .is_some_and(|count| {
                core::alloc::Layout::array::<GrayImage>(count).is_ok()
                    && core::alloc::Layout::array::<Option<super::scale_space::GradientSlice>>(
                        count,
                    )
                    .is_ok()
            });
        let checks = [
            ("num_octaves", self.num_octaves > 0, "positive"),
            (
                "num_scales",
                self.num_scales > 0 && scale_count_is_representable,
                "positive with a representable scale-level count",
            ),
            (
                "first_octave",
                matches!(self.first_octave, -1 | 0),
                "-1 or 0",
            ),
            (
                "peak_threshold",
                self.peak_threshold.is_finite() && self.peak_threshold >= 0.0,
                "finite and non-negative",
            ),
            (
                "edge_threshold",
                self.edge_threshold.is_finite() && self.edge_threshold > 0.0,
                "finite and positive",
            ),
        ];
        for (field, valid, expected) in checks {
            if !valid {
                return Err(SiftError::InvalidConfig { field, expected });
            }
        }
        Ok(())
    }

    /// Detect oriented keypoint geometry without constructing descriptors.
    ///
    /// This follows the same scale-space, extrema, refinement, gradient, and
    /// orientation path as [`Self::extract_typed`].
    ///
    /// An image smaller than 8 pixels in either dimension yields no features.
    ///
    /// # Errors
    ///
    /// Returns [`SiftError::InvalidConfig`] when [`Self::validate`] rejects the configuration.
    pub fn detect_keypoints(
        &self,
        image: &GrayImage,
    ) -> Result<Vec<ScaleSpaceKeypoint>, SiftError> {
        self.validate()?;
        let mut geometries = Vec::new();
        self.for_each_selected_oriented_octave(
            image,
            &self.feature_budget,
            |keypoints, _gradients| {
                geometries.reserve(keypoints.len());
                geometries.extend(keypoints.into_iter().map(SiftFeature::geometry));
            },
        );
        // Results usually outlive octave scratch; release spare geometric growth capacity.
        geometries.shrink_to_fit();
        Ok(geometries)
    }

    fn for_each_oriented_octave<V>(&self, image: &GrayImage, mut visit: V)
    where
        V: FnMut(Vec<SiftFeature>, GradientOctave),
    {
        // Public entry points validate the configuration before visiting octaves.
        if image.nrows() < MIN_INPUT_DIMENSION || image.ncols() < MIN_INPUT_DIMENSION {
            return;
        }

        // Convert to [0, 1] range (SiftAnatomy convention)
        let img_f = image * (1.0 / 255.0);

        let (sigma_min, delta_min) = if self.first_octave == -1 {
            (BASE_SIGMA / 2.0, 0.5f32)
        } else {
            (BASE_SIGMA, 1.0f32)
        };

        let mut scale_space = HierarchicalGaussianScaleSpace::new(
            self.num_octaves,
            self.num_scales,
            sigma_min,
            delta_min,
            ASSUMED_INPUT_SIGMA,
        );
        scale_space.set_image(&img_f);
        // The scale-space owns its blurred base. Release the normalized source before
        // octave processing so concurrent image extraction does not retain one extra
        // full-resolution float image per worker.
        drop(img_f);

        let per_octave_threshold = self.peak_threshold / self.num_scales as f32;

        while let Some(octave) = scale_space.next_octave() {
            // 1. DoG
            let dogs = compute_dogs(&octave);

            // 2. Find 3D extrema
            let candidates = find_3d_discrete_extrema(&dogs, per_octave_threshold);

            // 3. Refine positions
            let refined =
                refine_keypoints(&dogs, candidates, per_octave_threshold, self.edge_threshold);
            // DoG slices are no longer needed after sub-pixel refinement. Dropping them
            // before gradient construction avoids retaining both full octave pyramids.
            drop(dogs);

            // 4. Compute gradients
            let grads = compute_gradients(&octave);
            // Gradients own the samples required by orientation and descriptors; the
            // Gaussian octave can be released before those comparatively long stages.
            drop(octave);

            // 5. Assign orientations (may produce multiple keypoints)
            let oriented = assign_orientations(refined, &grads);

            visit(oriented, grads);
            // Gradients are dropped here before the next octave is materialized.
        }
    }

    fn for_each_budgeted_oriented_octave<B, V>(
        &self,
        image: &GrayImage,
        feature_budget: &B,
        mut visit: V,
    ) where
        B: FeatureBudget + ?Sized,
        V: FnMut(Vec<SiftFeature>, GradientOctave),
    {
        debug_assert!(!feature_budget.is_unlimited());

        let mut octaves = Vec::new();
        self.for_each_oriented_octave(image, |keypoints, gradients| {
            if !keypoints.is_empty() {
                octaves.push((keypoints, gradients));
            }
        });

        let candidate_count = octaves.iter().map(|(keypoints, _)| keypoints.len()).sum();
        let mut candidates = Vec::with_capacity(candidate_count);
        for (keypoints, _) in &octaves {
            candidates.extend(
                keypoints
                    .iter()
                    .map(FeatureBudgetKeypoint::feature_candidate),
            );
        }

        let mut keep =
            selection_mask(&candidates, image.ncols(), image.nrows(), feature_budget).into_iter();
        for (keypoints, _) in &mut octaves {
            keypoints.retain(|_| keep.next().unwrap_or(false));
        }

        for (keypoints, gradients) in octaves {
            if !keypoints.is_empty() {
                visit(keypoints, gradients);
            }
        }
    }

    fn for_each_selected_oriented_octave<B, V>(
        &self,
        image: &GrayImage,
        feature_budget: &B,
        visit: V,
    ) where
        B: FeatureBudget + ?Sized,
        V: FnMut(Vec<SiftFeature>, GradientOctave),
    {
        if feature_budget.is_unlimited() {
            self.for_each_oriented_octave(image, visit);
        } else {
            self.for_each_budgeted_oriented_octave(image, feature_budget, visit);
        }
    }

    /// Detect keypoints and return typed descriptors.
    ///
    /// The storage type `D` selects [`F32Descriptor`](crate::extraction::F32Descriptor) or
    /// [`U8Descriptor`](crate::extraction::U8Descriptor); both hold values on the shared
    /// SIFT scale that [`U8Descriptor`](crate::extraction::U8Descriptor) describes.
    ///
    /// An image smaller than 8 pixels in either dimension yields no features.
    ///
    /// # Errors
    ///
    /// Returns [`SiftError::InvalidConfig`] when [`Self::validate`] rejects the configuration.
    pub fn extract_typed<D: DescriptorStorage>(
        &self,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<D>, SiftError> {
        self.validate()?;
        let mut features = ScaleSpaceFeatures::default();
        self.for_each_selected_oriented_octave(
            image,
            &self.feature_budget,
            |keypoints, gradients| {
                features.keypoints.reserve(keypoints.len());
                features.descriptors.reserve(keypoints.len());
                for_each_descriptor(
                    keypoints,
                    &gradients,
                    self.root_sift,
                    |keypoint, descriptor| {
                        if let Ok(descriptor) = D::try_from_finalized(&descriptor) {
                            features.keypoints.push(keypoint.geometry());
                            features.descriptors.push(descriptor);
                        }
                    },
                );
            },
        );
        // Results usually outlive octave scratch; release spare per-octave growth capacity.
        features.keypoints.shrink_to_fit();
        features.descriptors.shrink_to_fit();
        Ok(features)
    }

    /// Shorthand for [`extract_typed::<F32Descriptor>`](Self::extract_typed).
    ///
    /// An image smaller than 8 pixels in either dimension yields no features.
    ///
    /// # Errors
    ///
    /// Returns [`SiftError::InvalidConfig`] when [`Self::validate`] rejects the configuration.
    #[inline]
    pub fn extract_f32(
        &self,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<super::super::storage::F32Descriptor>, SiftError> {
        self.extract_typed(image)
    }

    /// Shorthand for [`extract_typed::<U8Descriptor>`](Self::extract_typed): 128 bytes per
    /// descriptor, on the shared scale that raw-byte matchers compare byte for byte.
    ///
    /// An image smaller than 8 pixels in either dimension yields no features.
    ///
    /// # Errors
    ///
    /// Returns [`SiftError::InvalidConfig`] when [`Self::validate`] rejects the configuration.
    #[inline]
    pub fn extract_u8(
        &self,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<super::super::storage::U8Descriptor>, SiftError> {
        self.extract_typed(image)
    }
}

impl Default for SiftExtractor {
    fn default() -> Self {
        Self::with_preset(SiftPreset::Normal)
    }
}
