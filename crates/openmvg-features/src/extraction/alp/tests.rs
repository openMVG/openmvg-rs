// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::geometry_tables::{
    ALP_GEOMETRY_BLEND_BETA, ALP_GEOMETRY_BLEND_BOOST, ALP_GEOMETRY_COMPATIBILITY_LOWER,
    ALP_GEOMETRY_COMPATIBILITY_UPPER, ALP_GEOMETRY_FINAL_MOVE_CAP_LOG2, ALP_GEOMETRY_PROFILE_CASES,
    ALP_GEOMETRY_REFERENCE_CASES,
};
use super::octave_dedup::{AdjacentOctavePreference, remove_adjacent_octave_duplicates};
use super::scale_fit::*;
use super::tables::ALP_SCALE_REFERENCE_CASES;
use super::tables::SIGMAS;
use super::*;
use crate::extraction::convolution::GaussianFilter;
use crate::extraction::feature_budget::{FeatureBudget, FeatureBudgetKeypoint, FeatureCandidate};
use crate::extraction::{DescriptorStorage, U8Descriptor};
use core::f32::consts::PI;

/// Internal combined value retained only while assembling legacy test fixtures.
#[derive(Clone, Debug)]
pub(crate) struct DescribedKeypoint {
    /// Horizontal position in image coordinates (sub-pixel).
    pub x: f32,
    /// Vertical position in image coordinates (sub-pixel).
    pub y: f32,
    /// Characteristic scale (Gaussian σ in original image pixels).
    pub sigma: f32,
    /// Dominant orientation in radians.
    pub orientation: f32,
    /// Sigma-squared-normalized LoG response at the keypoint (contrast measure).
    pub peak: f32,
    /// Octave index (0 = original resolution, 1 = half, …).
    pub octave: usize,
    /// Detector-specific scale metadata.
    ///
    /// Detected ALP points store the spatial Q-fit bucket (1 or 2),
    /// not the nearest Gaussian level. [`AlpExtractor::describe_at_support_scale`]
    /// stores the nearest Gaussian level for externally supplied positions.
    pub scale_index: usize,
    /// `true` if the keypoint is an LoG maximum, `false` if a minimum.
    pub is_max: bool,
    /// 128-D SIFT descriptor (RootSIFT-normalized after `detect()`).
    /// `None` if the descriptor has not been computed yet.
    pub descriptor: Option<[f32; 128]>,
}

impl FeatureBudgetKeypoint for DescribedKeypoint {
    fn feature_candidate(&self) -> FeatureCandidate {
        FeatureCandidate::new(self.x, self.y, self.peak.abs())
    }
}

impl From<&DescribedKeypoint> for ScaleSpaceKeypoint {
    fn from(kp: &DescribedKeypoint) -> Self {
        ScaleSpaceKeypoint {
            x: kp.x,
            y: kp.y,
            sigma: kp.sigma,
            orientation: kp.orientation,
            peak: kp.peak,
            octave: kp.octave,
            scale_index: kp.scale_index,
            is_max: kp.is_max,
        }
    }
}

impl From<ScaleSpaceKeypoint> for DescribedKeypoint {
    fn from(g: ScaleSpaceKeypoint) -> Self {
        DescribedKeypoint {
            x: g.x,
            y: g.y,
            sigma: g.sigma,
            orientation: g.orientation,
            peak: g.peak,
            octave: g.octave,
            scale_index: g.scale_index,
            is_max: g.is_max,
            descriptor: None,
        }
    }
}

impl AlpExtractor {
    /// Detect keypoints and compute descriptors for the given grayscale image.
    fn detect(&self, image: &GrayImage) -> Vec<DescribedKeypoint> {
        self.detect_with_budget(image, &self.feature_budget)
    }

    /// Detect with a caller-provided budget policy.
    fn detect_with_budget<B: FeatureBudget + ?Sized>(
        &self,
        image: &GrayImage,
        feature_budget: &B,
    ) -> Vec<DescribedKeypoint> {
        self.run_stream(image, feature_budget, true)
            .iter()
            .map(|feature| {
                let mut keypoint = DescribedKeypoint::from(Self::public_geometry(feature));
                keypoint.descriptor = feature.descriptor;
                keypoint
            })
            .collect()
    }
}

// Helper function to create a synthetic Gaussian blob image
fn create_gaussian_blob_image(
    size: usize,
    center: (f32, f32),
    sigma: f32,
    amplitude: f32,
) -> GrayImage {
    let mut img = GrayImage::zeros(size, size);
    let (cx, cy) = center;

    for i in 0..size {
        for j in 0..size {
            let dx = j as f32 - cx;
            let dy = i as f32 - cy;
            let dist_sq = dx * dx + dy * dy;
            let val = amplitude * (-dist_sq / (2.0 * sigma * sigma)).exp();
            img[(i, j)] = val;
        }
    }
    img
}

fn create_constant_image(size: usize, value: f32) -> GrayImage {
    GrayImage::from_element(size, size, value)
}

fn create_textured_image(size: usize) -> GrayImage {
    GrayImage::from_fn(size, size, |row, col| {
        ((row * 37 + col * 17 + (row * col) % 251) % 256) as f32
    })
}

// ========================================================================
// AlpExtractor Tests
// ========================================================================

#[test]
fn invalid_input_yields_empty_results() {
    let detector = AlpExtractor::default();
    assert!(
        detector
            .detect_keypoints(&GrayImage::zeros(9, 64))
            .unwrap()
            .is_empty()
    );
    let image = create_textured_image(128);
    assert!(
        !AlpExtractor::new(5.0, 2.0, 12.1)
            .extract_f32(&image)
            .unwrap()
            .is_empty()
    );
    for (detector, field) in [
        (AlpExtractor::new(f32::NAN, 4.8, 12.1), "peak_threshold"),
        (AlpExtractor::new(-1.0, 4.8, 12.1), "peak_threshold"),
        (
            AlpExtractor::new(30.0, f32::INFINITY, 12.1),
            "sigma_threshold",
        ),
        (AlpExtractor::new(30.0, 4.8, 0.0), "curv_ratio_threshold"),
    ] {
        assert!(matches!(
            detector.validate(),
            Err(AlpError::InvalidConfig { field: reported, .. }) if reported == field
        ));
        // Every detection entry point reports the same configuration error.
        let error = detector.validate().unwrap_err();
        assert_eq!(detector.detect_keypoints(&image).unwrap_err(), error);
        assert_eq!(detector.extract_f32(&image).unwrap_err(), error);
        assert_eq!(detector.extract_u8(&image).unwrap_err(), error);
    }
    for preset in [AlpPreset::High, AlpPreset::Normal] {
        assert_eq!(AlpExtractor::with_preset(preset).validate(), Ok(()));
    }
}

#[test]
fn scale_fit_preserves_constant_and_zero_responses() {
    for value in [0.0, -0.0, 0.25, 100.0, -100.0, f32::MAX, -f32::MAX] {
        let [a, b, c, d] = compute_poly_coeffs(value, value, value, value);
        assert_eq!([a, b, c], [0.0; 3]);
        assert_eq!(d, value);
        let (minimum, maximum, minimum_sigma, maximum_sigma) =
            super::stream::scale_envelope([value; 4]);
        assert_eq!([minimum, maximum], [value; 2]);
        assert_eq!([minimum_sigma, maximum_sigma], [0.0; 2]);
    }
}

#[test]
fn scale_fit_reverses_coefficient_polarity() {
    let samples = [3.25, -7.5, 2.125, 11.0];
    let positive = compute_poly_coeffs(samples[0], samples[1], samples[2], samples[3]);
    let negative = compute_poly_coeffs(-samples[0], -samples[1], -samples[2], -samples[3]);
    assert_eq!(
        negative.map(f32::to_bits),
        positive.map(|value| (-value).to_bits())
    );
}

#[test]
fn julia_reference_vectors_cover_base_scale_coefficients() {
    for case in ALP_SCALE_REFERENCE_CASES {
        let [first, second, third, reference] = case.sample_bits.map(f32::from_bits);
        let actual = compute_poly_coeffs(first, second, third, reference);
        assert_eq!(
            actual.map(f32::to_bits),
            case.coefficient_bits,
            "Julia/Rust scale coefficients differ for {}",
            case.name
        );
    }
}

#[test]
fn test_alp_presets_use_common_stability_guards() {
    for (preset, peak) in [(AlpPreset::High, 3.5), (AlpPreset::Normal, 15.0)] {
        let detector = AlpExtractor::with_preset(preset);
        assert_eq!(detector.peak_threshold, peak);
        assert_eq!(detector.sigma_threshold, 0.0);
        assert_eq!(detector.curv_ratio_threshold, 12.1);
        assert_eq!(detector.feature_budget(), FeatureBudgetPolicy::Unlimited);
    }
}

#[test]
fn spatial_fit_recovers_asymmetric_xy_offset() {
    let expected_x = 0.35;
    let expected_y = -0.20;
    let mut samples = [0.0; 9];
    for row in 0..3 {
        for col in 0..3 {
            let x = col as f32 - 1.0;
            let y = row as f32 - 1.0;
            samples[row * 3 + col] =
                10.0 - 2.0 * (x - expected_x).powi(2) - (y - expected_y).powi(2);
        }
    }

    let [w1, w2, w3, w4, w5] = fit_spatial_quadratic(0, &samples);
    let denominator = 4.0 * w1 * w2 - w3 * w3;
    let delta_y = (w3 * w4 - 2.0 * w1 * w5) / denominator;
    let delta_x = (-w4 - w3 * delta_y) / (2.0 * w1);

    assert!((delta_x - expected_x).abs() < 0.04, "x={delta_x}");
    assert!((delta_y - expected_y).abs() < 0.04, "y={delta_y}");
}

#[test]
fn test_response_budget_caps_typed_output_before_description() {
    let image = create_textured_image(128);
    let unlimited = AlpExtractor::new(5.0, 2.0, 12.1)
        .extract_u8(&image)
        .unwrap();
    assert!(unlimited.keypoints.len() > 8);

    let detector =
        AlpExtractor::new(5.0, 2.0, 12.1).with_feature_budget(FeatureBudgetPolicy::strongest(8));
    let budgeted = detector.extract_u8(&image).unwrap();
    assert!(budgeted.keypoints.len() <= 8);
    assert_eq!(budgeted.keypoints.len(), budgeted.descriptors.len());
}

#[test]
fn test_zero_budget_skips_all_descriptors() {
    let image = create_textured_image(96);
    let detector = AlpExtractor::with_preset(AlpPreset::Normal)
        .with_feature_budget(FeatureBudgetPolicy::spatial_four_by_four(0));
    let budgeted = detector.extract_u8(&image).unwrap();
    assert!(budgeted.keypoints.is_empty());
    assert!(budgeted.descriptors.is_empty());
}

#[cfg(feature = "std")]
#[test]
fn public_correction_runs_once_per_emitted_candidate() {
    let image = create_textured_image(128);
    let extractor = AlpExtractor::new(5.0, 2.0, 12.1);
    for describe in [false, true] {
        for budget in [
            FeatureBudgetPolicy::Unlimited,
            FeatureBudgetPolicy::strongest(8),
            FeatureBudgetPolicy::spatial_four_by_four(0),
        ] {
            reset_public_correction_calls();
            let features = extractor.run_stream(&image, &budget, describe);
            assert_eq!(public_correction_calls(), features.len());
            if budget.is_unlimited() {
                assert!(features.len() > 8);
            }
            if describe {
                assert!(features.iter().all(|feature| feature.descriptor.is_some()));
            }
        }
    }
}

#[test]
fn external_description_drops_invalid_support_geometry() {
    let image = create_textured_image(128);
    let extractor = AlpExtractor::default();
    assert_eq!(
        extractor
            .describe_at_support_scale(&image, &[(64.0, 64.0, 2.4, 0.0)])
            .len(),
        1
    );
    let mut positions = vec![(64.0, 64.0, 0.0, 0.0), (64.0, 64.0, -1.0, 0.0)];
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        positions.extend([
            (value, 64.0, 2.4, 0.0),
            (64.0, value, 2.4, 0.0),
            (64.0, 64.0, value, 0.0),
            (64.0, 64.0, 2.4, value),
        ]);
    }
    assert!(
        extractor
            .describe_at_support_scale(&image, &positions)
            .is_empty()
    );
}

#[test]
fn describe_at_support_scale_preserves_representable_input_scales() {
    let image = create_textured_image(128);
    let requested = [2.4, 3.0, 3.2, 4.0];
    let positions: Vec<_> = requested
        .iter()
        .enumerate()
        .map(|(index, &sigma)| (56.0 + index as f32 * 4.0, 64.0, sigma, 0.0))
        .collect();

    let described = AlpExtractor::default().describe_at_support_scale(&image, &positions);
    assert_eq!(described.keypoints.len(), requested.len());
    for expected in requested {
        assert!(
            described
                .keypoints
                .iter()
                .any(|keypoint| (keypoint.sigma - expected).abs() <= f32::EPSILON),
            "missing requested sigma {expected}; got {:?}",
            described
                .keypoints
                .iter()
                .map(|keypoint| keypoint.sigma)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_detect_empty_image_returns_empty() {
    let detector = AlpExtractor::new(10.0, 5.0, 20.0);
    let img = GrayImage::zeros(32, 32);
    let keypoints = detector.detect(&img);
    assert_eq!(
        keypoints.len(),
        0,
        "Empty image should produce no keypoints"
    );
}

#[test]
fn test_detect_constant_image_returns_empty() {
    let detector = AlpExtractor::new(10.0, 5.0, 20.0);
    let img = create_constant_image(64, 128.0);
    let keypoints = detector.detect(&img);
    assert_eq!(
        keypoints.len(),
        0,
        "Constant image should produce no keypoints"
    );
}

#[test]
fn test_detect_single_blob() {
    let detector = AlpExtractor::new(5.0, 2.0, 20.0);
    let img = create_gaussian_blob_image(100, (50.0, 50.0), 5.0, 200.0);
    let keypoints = detector.detect(&img);

    // Should detect at least one keypoint near the blob center
    assert!(
        !keypoints.is_empty(),
        "Should detect keypoints in blob image"
    );

    // Check that at least one keypoint is near the blob center
    let has_near_center = keypoints.iter().any(|kp| {
        let dx = kp.x - 50.0;
        let dy = kp.y - 50.0;
        (dx * dx + dy * dy).sqrt() < 20.0
    });
    assert!(has_near_center, "Should detect keypoint near blob center");
}

#[test]
fn test_detect_multiple_blobs() {
    let mut img = GrayImage::zeros(150, 150);

    let blob_centers = [(40.0, 40.0), (110.0, 40.0), (75.0, 110.0)];
    for &(cx, cy) in &blob_centers {
        for i in 0..150 {
            for j in 0..150 {
                let dx = j as f32 - cx;
                let dy = i as f32 - cy;
                let val = 150.0 * (-0.5 * (dx * dx + dy * dy) / 25.0).exp();
                img[(i, j)] += val;
            }
        }
    }

    let detector = AlpExtractor::new(5.0, 2.0, 20.0);
    let keypoints = detector.detect(&img);

    assert!(
        keypoints.len() >= 2,
        "Should detect multiple keypoints for multiple blobs"
    );
}

#[test]
fn test_keypoints_within_image_bounds() {
    let detector = AlpExtractor::new(5.0, 2.0, 20.0);
    let img = create_gaussian_blob_image(100, (50.0, 50.0), 8.0, 200.0);
    let keypoints = detector.detect(&img);

    for kp in &keypoints {
        assert!(kp.x >= 0.0, "Keypoint x should be >= 0");
        assert!(kp.y >= 0.0, "Keypoint y should be >= 0");
        assert!(kp.x < 100.0, "Keypoint x should be < image width");
        assert!(kp.y < 100.0, "Keypoint y should be < image height");
    }
}

#[test]
fn test_keypoint_sigma_in_valid_range() {
    let detector = AlpExtractor::new(5.0, 2.0, 20.0);
    let img = create_gaussian_blob_image(100, (50.0, 50.0), 5.0, 200.0);
    let keypoints = detector.detect(&img);

    for kp in &keypoints {
        assert!(kp.sigma > 0.0, "Keypoint sigma should be positive");
        assert!(kp.sigma < 100.0, "Keypoint sigma should be reasonable");
    }
}

#[test]
fn test_threshold_affects_keypoint_count() {
    let img = create_gaussian_blob_image(100, (50.0, 50.0), 5.0, 200.0);

    let detector_low = AlpExtractor::new(1.0, 2.0, 20.0);
    let detector_high = AlpExtractor::new(50.0, 2.0, 20.0);

    let kps_low = detector_low.detect(&img);
    let kps_high = detector_high.detect(&img);

    assert!(
        kps_low.len() >= kps_high.len(),
        "Lower threshold should produce more keypoints: low={}, high={}",
        kps_low.len(),
        kps_high.len()
    );
}

// ========================================================================
// ScaleSpace Tests
// ========================================================================

// ========================================================================
// Gradient Computation Tests
// ========================================================================

fn test_geometry(x: f32, y: f32, sigma: f32, peak: f32, is_max: bool) -> ScaleSpaceKeypoint {
    ScaleSpaceKeypoint {
        x,
        y,
        sigma,
        orientation: 0.0,
        peak,
        octave: 0,
        scale_index: get_nearest_scale_index(sigma),
        is_max,
    }
}

#[test]
fn test_adjacent_octave_dedup_keeps_strongest_signed_peak() {
    let mut previous = vec![
        test_geometry(10.0, 10.0, 3.2, 20.0, true),
        test_geometry(30.0, 30.0, 3.2, -40.0, false),
    ];
    let mut current = vec![
        test_geometry(5.1, 5.1, 1.6, 30.0, true),
        test_geometry(15.1, 15.1, 1.6, -25.0, false),
    ];

    remove_adjacent_octave_duplicates(
        &mut previous,
        &mut current,
        AdjacentOctavePreference::StrongestSignedResponse,
    );

    assert_eq!(previous.len(), 1);
    assert_eq!(previous[0].peak, -40.0);
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].peak, 30.0);
}

#[test]
fn test_adjacent_octave_dedup_preserves_opposite_polarity() {
    let mut previous = vec![test_geometry(10.0, 10.0, 3.2, 20.0, true)];
    let mut current = vec![test_geometry(5.0, 5.0, 1.6, -30.0, false)];

    remove_adjacent_octave_duplicates(
        &mut previous,
        &mut current,
        AdjacentOctavePreference::StrongestSignedResponse,
    );

    assert_eq!(previous.len(), 1);
    assert_eq!(current.len(), 1);
}

fn remove_adjacent_octave_duplicates_reference(
    previous: &mut Vec<ScaleSpaceKeypoint>,
    current: &mut Vec<ScaleSpaceKeypoint>,
) {
    let mut keep_previous = vec![true; previous.len()];
    let mut keep_current = vec![true; current.len()];
    for (previous_index, previous_keypoint) in previous.iter().enumerate() {
        if !keep_previous[previous_index] {
            continue;
        }
        for (current_index, current_keypoint) in current.iter().enumerate() {
            if !keep_current[current_index] || previous_keypoint.is_max != current_keypoint.is_max {
                continue;
            }
            let spatial_distance = (2.0 * current_keypoint.x - previous_keypoint.x).abs()
                + (2.0 * current_keypoint.y - previous_keypoint.y).abs();
            let scale_distance = (2.0 * current_keypoint.sigma - previous_keypoint.sigma).abs();
            if spatial_distance >= 2.0 || scale_distance >= 0.4 {
                continue;
            }
            let previous_is_weaker = if previous_keypoint.is_max {
                previous_keypoint.peak < current_keypoint.peak
            } else {
                previous_keypoint.peak > current_keypoint.peak
            };
            if previous_is_weaker {
                keep_previous[previous_index] = false;
            } else {
                keep_current[current_index] = false;
            }
            break;
        }
    }
    let mut previous_keep = keep_previous.into_iter();
    previous.retain(|_| previous_keep.next().unwrap_or(false));
    let mut current_keep = keep_current.into_iter();
    current.retain(|_| current_keep.next().unwrap_or(false));
}

#[test]
fn test_indexed_adjacent_octave_dedup_matches_reference_order() {
    let mut previous = Vec::new();
    let mut current = Vec::new();
    for index in 0..80 {
        let x = 4.0 + (index % 10) as f32 * 7.1;
        let y = 4.0 + (index / 10) as f32 * 6.7;
        let sigma = 3.2 + (index % 5) as f32 * 0.15;
        let is_max = index % 3 != 0;
        let peak = if is_max {
            10.0 + index as f32
        } else {
            -10.0 - index as f32
        };
        previous.push(test_geometry(x, y, sigma, peak, is_max));
        if index % 2 == 0 {
            current.push(test_geometry(
                0.5 * x + 0.1,
                0.5 * y - 0.1,
                0.5 * sigma + 0.05,
                peak * if index % 4 == 0 { 1.2 } else { 0.8 },
                is_max,
            ));
        }
    }
    for index in 0..60 {
        let is_max = index % 2 == 0;
        current.push(test_geometry(
            1.3 + (index % 12) as f32 * 5.3,
            2.1 + (index / 12) as f32 * 7.7,
            1.6 + (index % 4) as f32 * 0.13,
            if is_max { 20.0 } else { -20.0 },
            is_max,
        ));
    }

    let mut expected_previous = previous.clone();
    let mut expected_current = current.clone();
    remove_adjacent_octave_duplicates_reference(&mut expected_previous, &mut expected_current);
    remove_adjacent_octave_duplicates(
        &mut previous,
        &mut current,
        AdjacentOctavePreference::StrongestSignedResponse,
    );

    assert_eq!(previous, expected_previous);
    assert_eq!(current, expected_current);
}

// ========================================================================
// GaussianFilter Tests
// ========================================================================

#[test]
fn test_gaussian_filter_kernel_normalized() {
    let filter = GaussianFilter::new(2.0);
    // Full kernel = centre + 2 × (all remaining taps).
    let kernel = filter.kernel_half();
    let sum: f32 = kernel[0] + 2.0 * kernel[1..].iter().sum::<f32>();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Full kernel should sum to 1.0, got {}",
        sum
    );
}

#[test]
fn test_gaussian_filter_kernel_symmetric() {
    let filter = GaussianFilter::new(1.5);
    // The half-kernel is the right half of a symmetric Gaussian, so it
    // must be monotonically non-increasing (values decrease away from centre).
    let kernel = filter.kernel_half();
    for i in 1..kernel.len() {
        assert!(
            kernel[i] <= kernel[i - 1] + 1e-6,
            "Half-kernel should be monotonically non-increasing at index {}",
            i
        );
    }
}

#[test]
fn test_gaussian_filter_preserves_constant() {
    let filter = GaussianFilter::new(2.0);
    let img = create_constant_image(32, 100.0);
    let result = filter.convolve_separable(&img);

    // Interior pixels should be unchanged
    for i in 5..27 {
        for j in 5..27 {
            let diff = (result[(i, j)] - 100.0).abs();
            assert!(
                diff < 0.1,
                "Constant image should be preserved after convolution, diff={}",
                diff
            );
        }
    }
}

#[test]
fn test_gaussian_filter_large_parallel_path_preserves_constant() {
    let filter = GaussianFilter::new(2.0);
    let img = create_constant_image(192, 37.0);
    let result = filter.convolve_separable(&img);

    for &value in result.iter() {
        assert!((value - 37.0).abs() < 1e-4);
    }
}

// ========================================================================
// get_nearest_scale_index Tests
// ========================================================================

#[test]
fn test_get_nearest_scale_index_exact_values() {
    // SIGMAS = [1.6, 2.262742, 3.2, 4.525483]
    assert_eq!(get_nearest_scale_index(1.6), 0);
    assert_eq!(get_nearest_scale_index(2.262742), 1);
    assert_eq!(get_nearest_scale_index(3.2), 2);
    assert_eq!(get_nearest_scale_index(4.525483), 3);
}

#[test]
fn test_get_nearest_scale_index_intermediate_values() {
    // SIGMAS = [1.6, 2.262742, 3.2, 4.525483]
    // Midpoints: (1.6+2.26)/2=1.93, (2.26+3.2)/2=2.73, (3.2+4.52)/2=3.86
    // Test values between scales
    assert_eq!(get_nearest_scale_index(1.9), 0); // 1.9 < 1.93, closer to 1.6
    assert_eq!(get_nearest_scale_index(2.0), 1); // 2.0 > 1.93, closer to 2.26
    assert_eq!(get_nearest_scale_index(2.7), 1); // 2.7 < 2.73, closer to 2.26
    assert_eq!(get_nearest_scale_index(4.0), 3); // 4.0 > 3.86, closer to 4.52
}

#[test]
fn q_fit_uses_only_reference_central_levels() {
    assert_eq!(get_q_scale_index(SIGMAS[0]), 1);
    assert_eq!(get_q_scale_index(SIGMAS[1]), 1);
    assert_eq!(get_q_scale_index(SIGMAS[1] + 0.001), 2);
    assert_eq!(get_q_scale_index(SIGMAS[3]), 2);
}

#[test]
fn test_get_nearest_scale_index_extreme_values() {
    // Very small sigma should return 0
    assert_eq!(get_nearest_scale_index(0.5), 0);

    // Very large sigma should return 3
    assert_eq!(get_nearest_scale_index(10.0), 3);
}

// ========================================================================
// LoGColumnRollingBuffer Tests
// ========================================================================

// ========================================================================
// Keypoint Merge/Filter Tests
// ========================================================================

#[test]
fn test_keypoints_have_descriptors_after_detection() {
    let detector = AlpExtractor::new(5.0, 2.0, 20.0);
    let img = create_gaussian_blob_image(100, (50.0, 50.0), 5.0, 200.0);
    let keypoints = detector.detect(&img);

    for kp in &keypoints {
        if let Some(ref desc) = kp.descriptor {
            assert_eq!(desc.len(), 128, "SIFT descriptor should have 128 elements");
        }
    }
}

#[test]
fn test_keypoint_orientation_in_valid_range() {
    let detector = AlpExtractor::new(5.0, 2.0, 20.0);
    let img = create_gaussian_blob_image(100, (50.0, 50.0), 5.0, 200.0);
    let keypoints = detector.detect(&img);

    for kp in &keypoints {
        assert!(
            kp.orientation >= -PI && kp.orientation <= PI,
            "Orientation should be in [-PI, PI], got {}",
            kp.orientation
        );
    }
}

// ========================================================================
// Downsample Tests
// ========================================================================

#[test]
fn test_extract_typed_f32() {
    use super::super::storage::DescriptorStorage;

    let img = create_gaussian_blob_image(100, (50.0, 50.0), 5.0, 200.0);
    let detector = AlpExtractor::new(5.0, 2.0, 20.0);

    let result = detector.extract_f32(&img).unwrap();

    assert!(
        !result.keypoints.is_empty(),
        "Should detect keypoints with f32"
    );
    assert_eq!(result.keypoints.len(), result.descriptors.len());

    // Check descriptor dimension
    for desc in &result.descriptors {
        assert_eq!(desc.as_slice().len(), 128);
    }
}

#[test]
fn detection_only_returns_oriented_geometry() {
    let img = create_gaussian_blob_image(100, (50.0, 50.0), 5.0, 200.0);
    let detector = AlpExtractor::new(5.0, 2.0, 20.0);
    let detected = detector.detect_keypoints(&img).unwrap();
    let described = detector.extract_f32(&img).unwrap();

    assert!(!detected.is_empty());
    assert!(detected.len() >= described.keypoints.len());
    assert!(detected.iter().all(|keypoint| {
        keypoint.x.is_finite()
            && keypoint.y.is_finite()
            && keypoint.sigma.is_finite()
            && keypoint.orientation.is_finite()
    }));
}

#[cfg(all(feature = "parallel", feature = "std"))]
#[test]
fn typed_output_is_thread_count_invariant() {
    let image = GrayImage::from_fn(192, 256, |row, col| {
        ((row * 17 + col * 29 + row * col * 3) % 251) as f32
    });
    let detector =
        AlpExtractor::new(5.0, 2.0, 20.0).with_feature_budget(FeatureBudgetPolicy::strongest(512));
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build test thread pool")
            .install(|| detector.extract_f32(&image).unwrap())
    };

    let sequential = run(1);
    let parallel = run(4);
    assert!(
        sequential.keypoints.len() >= 64,
        "{}",
        sequential.keypoints.len()
    );
    assert_eq!(parallel.keypoints, sequential.keypoints);
    assert_eq!(parallel.descriptors, sequential.descriptors);
}

#[test]
fn test_extract_typed_u8() {
    use super::super::storage::DescriptorStorage;

    let img = create_gaussian_blob_image(100, (50.0, 50.0), 5.0, 200.0);
    let detector = AlpExtractor::new(5.0, 2.0, 20.0);

    let result = detector.extract_u8(&img).unwrap();

    assert!(
        !result.keypoints.is_empty(),
        "Should detect keypoints with u8"
    );
    assert_eq!(result.keypoints.len(), result.descriptors.len());

    let result_f32 = detector.extract_f32(&img).unwrap();
    assert_eq!(result_f32.descriptors.len(), result.descriptors.len());

    // Shared-grid u8 storage is a direct round+clamp of the finalized ALP
    // descriptor. It must not stretch every descriptor to a private max of 255.
    for (float_desc, desc) in result_f32.descriptors.iter().zip(&result.descriptors) {
        assert_eq!(desc.as_slice().len(), 128);
        for (&float_value, &byte) in float_desc.as_slice().iter().zip(desc.as_slice()) {
            assert_eq!(byte, float_value.round().clamp(0.0, 255.0) as u8);
        }
    }
}

#[test]
fn stored_u8_and_geometry_paths_preserve_float_pipeline_bits() {
    let image = GrayImage::from_fn(192, 256, |row, col| {
        ((row * 17 + col * 29 + row * col * 3) % 251) as f32
    });
    let bits = |keypoint: &ScaleSpaceKeypoint| {
        [
            keypoint.x.to_bits(),
            keypoint.y.to_bits(),
            keypoint.sigma.to_bits(),
            keypoint.orientation.to_bits(),
            keypoint.peak.to_bits(),
            keypoint.octave as u32,
            keypoint.scale_index as u32,
            u32::from(keypoint.is_max),
        ]
    };
    for budget in [
        FeatureBudgetPolicy::Unlimited,
        FeatureBudgetPolicy::strongest(128),
        FeatureBudgetPolicy::spatial_four_by_four(64),
        FeatureBudgetPolicy::strongest(0),
    ] {
        let detector = AlpExtractor::with_preset(AlpPreset::High).with_feature_budget(budget);
        let expected = detector.extract_typed::<U8Descriptor>(&image).unwrap();
        let actual = detector.extract_u8(&image).unwrap();
        assert_eq!(
            actual.keypoints.iter().map(bits).collect::<Vec<_>>(),
            expected.keypoints.iter().map(bits).collect::<Vec<_>>()
        );
        assert_eq!(actual.descriptors.len(), expected.descriptors.len());
        for (actual, expected) in actual.descriptors.iter().zip(&expected.descriptors) {
            assert_eq!(actual.as_slice(), expected.as_slice());
            assert_eq!(actual.scale().to_bits(), expected.scale().to_bits());
            assert_eq!(actual.scale(), 255.0);
        }
        let expected_geometry: Vec<_> = detector
            .run_stream(&image, &budget, false)
            .iter()
            .map(AlpExtractor::public_geometry)
            .collect();
        assert_eq!(
            detector
                .detect_keypoints(&image)
                .unwrap()
                .iter()
                .map(bits)
                .collect::<Vec<_>>(),
            expected_geometry.iter().map(bits).collect::<Vec<_>>()
        );
    }
}

#[test]
fn geometry_and_u8_storage_do_not_reserve_float_descriptor_payloads() {
    use core::mem::size_of;
    assert!(
        size_of::<stream::StreamFeature<()>>() + 500
            < size_of::<stream::StreamFeature<[f32; 128]>>()
    );
    assert!(
        size_of::<stream::StreamFeature<U8Descriptor>>() + 350
            < size_of::<stream::StreamFeature<[f32; 128]>>()
    );
}

#[test]
fn custom_non_send_descriptor_storage_still_converts_in_output_order() {
    #[cfg(not(feature = "std"))]
    use alloc::rc::Rc;
    #[cfg(feature = "std")]
    use std::rc::Rc;

    #[derive(Clone)]
    struct LocalDescriptor(Rc<[f32; 128]>);
    impl DescriptorStorage for LocalDescriptor {
        const DIM: usize = 128;
        type Element = f32;
        fn from_root_sift(values: &[f32]) -> Result<Self, super::super::DescriptorError> {
            let values = values.try_into().map_err(|_| {
                super::super::DescriptorError::DimensionMismatch {
                    expected: 128,
                    actual: values.len(),
                }
            })?;
            Ok(Self(Rc::new(values)))
        }
        fn as_slice(&self) -> &[f32] {
            self.0.as_ref()
        }
    }
    let image = create_gaussian_blob_image(100, (50.0, 50.0), 5.0, 200.0);
    let detector = AlpExtractor::new(5.0, 2.0, 20.0);
    let expected = detector.extract_f32(&image).unwrap();
    let actual = detector.extract_typed::<LocalDescriptor>(&image).unwrap();
    assert!(!actual.keypoints.is_empty());
    assert_eq!(actual.keypoints, expected.keypoints);
    for (actual, expected) in actual.descriptors.iter().zip(&expected.descriptors) {
        assert_eq!(actual.as_slice(), expected.as_slice());
    }
}

#[cfg(all(feature = "parallel", feature = "std"))]
#[test]
fn stored_u8_and_geometry_paths_are_thread_count_invariant() {
    let image = GrayImage::from_fn(192, 256, |row, col| {
        ((row * 17 + col * 29 + row * col * 3) % 251) as f32
    });
    let detector = AlpExtractor::with_preset(AlpPreset::High)
        .with_feature_budget(FeatureBudgetPolicy::strongest(256));
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build test thread pool")
            .install(|| {
                let output = detector.extract_u8(&image).unwrap();
                (
                    output.keypoints,
                    output
                        .descriptors
                        .into_iter()
                        .map(|descriptor| (*descriptor.as_array(), descriptor.scale().to_bits()))
                        .collect::<Vec<_>>(),
                    detector.detect_keypoints(&image).unwrap(),
                )
            })
    };
    let single = run(1);
    assert!(!single.0.is_empty());
    assert_eq!(single, run(4));
}

#[test]
fn test_extract_typed_f32_vs_u8_consistency() {
    let img = create_gaussian_blob_image(100, (50.0, 50.0), 5.0, 200.0);
    let detector = AlpExtractor::new(5.0, 2.0, 20.0);
    let result_f32 = detector.extract_f32(&img).unwrap();
    let result_u8 = detector.extract_u8(&img).unwrap();

    assert_eq!(result_f32.keypoints, result_u8.keypoints);
    assert_eq!(result_f32.descriptors.len(), result_u8.descriptors.len());
}

#[test]
fn direct_and_typed_paths_match() {
    use super::super::storage::DescriptorStorage;

    let img = create_gaussian_blob_image(100, (50.0, 50.0), 5.0, 200.0);
    let detector = AlpExtractor::new(5.0, 2.0, 20.0);
    let direct = detector.detect(&img);
    let typed = detector.extract_f32(&img).unwrap();

    assert_eq!(direct.len(), typed.keypoints.len());
    for ((keypoint, geometry), descriptor) in
        direct.iter().zip(&typed.keypoints).zip(&typed.descriptors)
    {
        assert_eq!(ScaleSpaceKeypoint::from(keypoint), *geometry);
        assert_eq!(
            keypoint.descriptor.as_ref().expect("descriptor"),
            descriptor.as_slice()
        );
    }
}

#[test]
fn test_extract_typed_l2_distance_correlation() {
    use super::super::storage::DescriptorStorage;

    // Create two different blob images at different locations
    let img1 = create_gaussian_blob_image(100, (50.0, 50.0), 5.0, 200.0);
    let img2 = create_gaussian_blob_image(100, (45.0, 45.0), 6.0, 180.0);

    let detector = AlpExtractor::new(5.0, 2.0, 20.0);

    let result1_f32 = detector.extract_f32(&img1).unwrap();
    let result2_f32 = detector.extract_f32(&img2).unwrap();
    let result1_u8 = detector.extract_u8(&img1).unwrap();
    let result2_u8 = detector.extract_u8(&img2).unwrap();

    // Skip if no keypoints in either image
    if result1_f32.descriptors.is_empty() || result2_f32.descriptors.is_empty() {
        return;
    }

    // Compare first descriptor from each
    let f32_dist: f32 = result1_f32.descriptors[0]
        .as_slice()
        .iter()
        .zip(result2_f32.descriptors[0].as_slice())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let u8_dist: u32 = result1_u8.descriptors[0]
        .as_slice()
        .iter()
        .zip(result2_u8.descriptors[0].as_slice())
        .map(|(&a, &b)| u32::from(a.abs_diff(b)).pow(2))
        .sum();

    assert!(
        f32_dist > 0.0,
        "f32 distance should be non-zero for different images"
    );
    assert!(
        u8_dist > 0,
        "u8 distance should be non-zero for different images"
    );
}

// --- Size-guard and small-resolution crash tests ---
// These exist to pin the behavior introduced by the max_octaves fix.
// None of them assert a keypoint count; the sole contract is "no panic".

#[test]
fn test_detect_below_size_guard_returns_empty() {
    // h < 10 guard: must return empty without panicking.
    let det = AlpExtractor::new(10.0, 5.0, 20.0);
    for sz in [1_usize, 5, 9] {
        let img = GrayImage::zeros(sz, sz);
        assert!(
            det.detect(&img).is_empty(),
            "{}×{} is below the h<10 guard — expected empty",
            sz,
            sz
        );
    }
}

#[test]
fn test_detect_does_not_crash_small_resolution() {
    // Sizes in [10, 128]: previously max_octaves could be 0 due to
    // integer division, silently returning empty instead of processing.
    // Must not panic for any of these.
    let det = AlpExtractor::new(10.0, 5.0, 20.0);
    for sz in [10_usize, 15, 32, 50, 63, 64, 65, 128] {
        let img = GrayImage::zeros(sz, sz);
        let _ = det.detect(&img);
    }
}

#[test]
fn test_detect_does_not_crash_non_square_small() {
    // min(h, w) drives the octave formula; ensure rectangular images
    // with a small short side are handled without panic.
    let det = AlpExtractor::new(10.0, 5.0, 20.0);
    let _ = det.detect(&GrayImage::zeros(10, 200));
    let _ = det.detect(&GrayImage::zeros(200, 10));
    let _ = det.detect(&GrayImage::zeros(63, 500));
    let _ = det.detect(&GrayImage::zeros(500, 63));
}

#[test]
fn alp_extraction_matches_fixed_fixture_bits() {
    use super::super::storage::DescriptorStorage;

    fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        hash
    }

    let image = GrayImage::from_fn(192, 256, |row, col| {
        ((row * 17 + col * 29 + row * col * 3) % 251) as f32
    });
    let result = AlpExtractor::new(5.0, 2.0, 20.0)
        .with_feature_budget(FeatureBudgetPolicy::strongest(256))
        .extract_f32(&image)
        .unwrap();
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for keypoint in &result.keypoints {
        for bits in [
            keypoint.x.to_bits(),
            keypoint.y.to_bits(),
            keypoint.sigma.to_bits(),
            keypoint.orientation.to_bits(),
            keypoint.peak.to_bits(),
            keypoint.octave as u32,
            keypoint.scale_index as u32,
            u32::from(keypoint.is_max),
        ] {
            hash = hash_bytes(hash, &bits.to_le_bytes());
        }
    }
    for descriptor in &result.descriptors {
        for &value in descriptor.as_slice() {
            hash = hash_bytes(hash, &value.to_bits().to_le_bytes());
        }
    }
    assert_eq!(result.keypoints.len(), 93);
    // This fixture covers anchor geometry, public scale, and descriptors for
    // the scale response operators, scale ownership and spatial competition
    // within one level of the anchor sigma.
    // Transcendental implementations differ between std and libm, so these
    // complete fingerprints are x86-64 regression locks for both profiles.
    // Generated coefficient bits are covered independently on every target.
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    assert_eq!(hash, 0xd8d0_b617_f47d_58b4);
    #[cfg(all(target_arch = "x86_64", not(feature = "std")))]
    assert_eq!(hash, 0xab38_f111_a08d_0466);
}

#[test]
fn public_scale_head_rejects_flat_responses_exactly() {
    for value in [
        0.0,
        -0.0,
        1e-5,
        1.0,
        100.0,
        255.0,
        -100.0,
        f32::MAX,
        -f32::MAX,
    ] {
        for anchor in [1.6, 2.4, 4.3] {
            for maximum in [false, true] {
                assert!(corrected_sigma_from_anchor(&[value; 4], anchor, maximum).is_none());
            }
        }
    }
}

#[test]
fn julia_profile_vectors_cover_statistic_and_damping() {
    for case in ALP_GEOMETRY_PROFILE_CASES {
        let samples = case.sample_bits.map(f32::from_bits);
        assert_eq!(
            profile_compatibility_statistic(&samples).map(f32::to_bits),
            case.statistic_bits,
            "Julia/Rust profile statistic differs for {}",
            case.name
        );
        assert_eq!(
            profile_blend_beta(&samples).to_bits(),
            case.beta_bits,
            "Julia/Rust profile damping differs for {}",
            case.name
        );
    }
}

#[test]
fn profile_damping_is_bounded_and_continuous_at_envelope_boundaries() {
    let lower = ALP_GEOMETRY_COMPATIBILITY_LOWER;
    let upper = ALP_GEOMETRY_COMPATIBILITY_UPPER;
    assert!(lower.is_finite() && lower > 0.0);
    assert_eq!(upper, 4.0 * lower);
    assert_eq!(profile_beta_from_statistic(None), ALP_GEOMETRY_BLEND_BETA);
    assert_eq!(
        profile_beta_from_statistic(Some(lower)),
        ALP_GEOMETRY_BLEND_BOOST
    );
    assert_eq!(
        profile_beta_from_statistic(Some(upper)),
        ALP_GEOMETRY_BLEND_BETA
    );
    let middle = lower + (upper - lower) * 0.5;
    assert_eq!(
        profile_beta_from_statistic(Some(middle)).to_bits(),
        0.6f32.to_bits()
    );
    let scores = [
        0.0,
        lower,
        f32::from_bits(lower.to_bits() + 1),
        middle,
        f32::from_bits(upper.to_bits() - 1),
        upper,
        f32::MAX,
    ];
    let mut previous = ALP_GEOMETRY_BLEND_BOOST;
    for score in scores {
        let beta = profile_beta_from_statistic(Some(score));
        assert!((ALP_GEOMETRY_BLEND_BETA..=ALP_GEOMETRY_BLEND_BOOST).contains(&beta));
        assert!(beta <= previous);
        previous = beta;
    }
}

#[test]
fn profile_move_preserves_cap_and_infinite_ratio_behavior() {
    let unsupported = profile_blend_beta(&[0.0; 4]);
    let cap = ALP_GEOMETRY_FINAL_MOVE_CAP_LOG2;
    assert_eq!(cap.to_bits(), (1.0f32 / 40.0).to_bits());
    assert_eq!(profile_bounded_move(unsupported, f32::INFINITY), cap);
    assert_eq!(profile_bounded_move(unsupported, f32::NEG_INFINITY), -cap);
    assert!(profile_bounded_move(unsupported, f32::NAN).is_nan());
    for raw_step in [-f32::MAX, -0.1, -0.02, -0.0, 0.0, 0.02, 0.1, f32::MAX] {
        let move_log2 = profile_bounded_move(unsupported, raw_step);
        assert!(move_log2.is_finite() && move_log2.abs() <= cap);
    }
    let boundary = 0.0625f32;
    let below = f32::from_bits(boundary.to_bits() - 1);
    let above = f32::from_bits(boundary.to_bits() + 1);
    for profile in ALP_GEOMETRY_PROFILE_CASES {
        if ![
            "gaussian_line",
            "margin_profile",
            "outside_envelope",
            "zero_response",
        ]
        .contains(&profile.name)
        {
            continue;
        }
        let samples = profile.sample_bits.map(f32::from_bits);
        let beta = profile_blend_beta(&samples);
        for raw_step in [
            -f32::MAX,
            f32::NEG_INFINITY,
            -above,
            -boundary,
            -below,
            -0.0,
            0.0,
            below,
            boundary,
            above,
            f32::INFINITY,
            f32::MAX,
            f32::NAN,
        ] {
            let complete_policy = (beta * raw_step).clamp(-cap, cap);
            let optimized = profile_bounded_move(beta, raw_step);
            if complete_policy.is_nan() {
                assert!(optimized.is_nan());
            } else {
                assert_eq!(
                    optimized.to_bits(),
                    complete_policy.to_bits(),
                    "saturation shortcut differs for {} at {raw_step:?}",
                    profile.name
                );
            }
        }
    }
}

#[test]
fn julia_reference_vectors_cover_anchor_to_public_sigma() {
    for case in ALP_GEOMETRY_REFERENCE_CASES {
        let samples = case.sample_bits.map(f32::from_bits);
        let anchor_sigma = f32::from_bits(case.anchor_sigma_bits);
        let correction = corrected_sigma_from_anchor(&samples, anchor_sigma, case.is_maximum);
        assert_eq!(
            correction.is_some(),
            case.correction_applied,
            "Julia/Rust correction branch differs for {}",
            case.name
        );

        let actual_public_sigma = correction.unwrap_or(anchor_sigma);
        if !case.correction_applied {
            assert_eq!(
                actual_public_sigma.to_bits(),
                case.anchor_sigma_bits,
                "fallback changed the anchor sigma for {}",
                case.name
            );
            assert_eq!(
                case.expected_public_sigma_bits, case.anchor_sigma_bits,
                "Julia fallback fixture is not anchor-exact for {}",
                case.name
            );
            continue;
        }

        let ulp_difference = actual_public_sigma
            .to_bits()
            .abs_diff(case.expected_public_sigma_bits);
        assert!(
            ulp_difference <= 2,
            "Julia/Rust public sigma differs by {ulp_difference} ULP for {}: actual={actual_public_sigma:?}, expected={:?}",
            case.name,
            f32::from_bits(case.expected_public_sigma_bits)
        );
    }
}

#[test]
fn stationary_scale_candidates_have_their_declared_curvature_type() {
    let witness = [0x4247fff9, 0x4247fffe, 0x42480000, 0x42480000].map(f32::from_bits);
    for samples in [
        witness,
        witness.map(|v| -v),
        [-0.13, -0.42, -0.37, -0.11],
        [0.13, 0.42, 0.37, 0.11],
        [0.0; 4],
        [100.0; 4],
    ] {
        let [a, b, _, _] = compute_poly_coeffs(samples[0], samples[1], samples[2], samples[3]);
        let (_, _, minimum, maximum) = stream::scale_envelope(samples);
        if minimum != 0.0 {
            assert!(6.0 * a * minimum + 2.0 * b > 0.0);
        }
        if maximum != 0.0 {
            assert!(6.0 * a * maximum + 2.0 * b < 0.0);
        }
    }
    let (_, _, _, maximum) = stream::scale_envelope(witness);
    assert!(maximum > 0.0);
    assert!(
        maximum < 4.0,
        "tied minimum must not replace the maximum root"
    );
}

#[test]
fn spatial_refinement_requires_finite_matching_definiteness() {
    for maximum in [false, true] {
        let sign = if maximum { -1.0 } else { 1.0 };
        let center = if maximum { 50.0 } else { -50.0 };
        let cache: [CubicCoefficients; 9] = core::array::from_fn(|index| {
            let x = (index % 3) as f32 - 1.0;
            let y = (index / 3) as f32 - 1.0;
            // A stationary scale term leaves the spatial samples unchanged.
            [
                0.0,
                sign,
                -2.0 * sign * 3.2,
                center + sign * (x * x + y * y + 3.2 * 3.2),
            ]
        });
        let mut correct = test_geometry(4.0, 4.0, 3.2, center, maximum);
        assert!(refine_keypoint_2d(&mut correct, &cache, 12.1));
        let mut wrong = test_geometry(4.0, 4.0, 3.2, center, !maximum);
        assert!(!refine_keypoint_2d(&mut wrong, &cache, 12.1));
        let mut nonfinite = cache;
        nonfinite[0][3] = f32::NAN;
        let mut geometry = test_geometry(4.0, 4.0, 3.2, center, maximum);
        assert!(!refine_keypoint_2d(&mut geometry, &nonfinite, 12.1));
    }
}

#[cfg(feature = "std")]
#[test]
fn external_description_rebuilds_only_used_prefix_and_ignores_unused_thresholds() {
    use super::super::storage::DescriptorStorage;
    use super::blur::{column_blur_calls, reset_column_blur_calls};
    let image = create_textured_image(256);
    let extractor = AlpExtractor::default();
    let irrelevant_thresholds = AlpExtractor::new(f32::NAN, f32::NAN, f32::NAN);
    for (sigma, octave) in [(2.4, 0), (4.8, 1), (6.4, 2)] {
        let positions = [(128.0, 128.0, sigma, 0.2)];
        stream::reset_octave_base_rebuilds();
        let result = extractor.describe_at_support_scale(&image, &positions);
        assert_eq!(stream::octave_base_rebuilds(), octave);
        assert_eq!(result.len(), 1);
        assert_eq!(result.keypoints[0].octave, octave);
        assert_eq!(result.keypoints[0].sigma.to_bits(), sigma.to_bits());
        let alternate = irrelevant_thresholds.describe_at_support_scale(&image, &positions);
        assert_eq!(result.keypoints, alternate.keypoints);
        assert_eq!(
            result.descriptors[0]
                .as_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            alternate.descriptors[0]
                .as_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
    stream::reset_octave_base_rebuilds();
    reset_column_blur_calls();
    let invalid = [
        (f32::NAN, 128.0, 2.4, 0.0),
        (128.0, 128.0, 0.0, 0.0),
        (128.0, 128.0, 2.4, f32::INFINITY),
    ];
    assert!(
        extractor
            .describe_at_support_scale(&image, &invalid)
            .is_empty()
    );
    assert_eq!(stream::octave_base_rebuilds(), 0);
    assert_eq!(column_blur_calls(), 0);
}

#[test]
fn band_extreme_matches_dense_sampling_of_the_band() {
    let cubics: [CubicCoefficients; 6] = [
        [0.3, -2.0, 3.0, 1.0],
        [-0.25, 1.5, -1.0, -4.0],
        [0.0, 4.0, -20.0, 14.5],
        [0.0, -1.0, 5.0, 2.0],
        [1.0 / 3.0, -1.25, -3.5, 2.270_833],
        [0.0, 0.0, 0.0, 7.0],
    ];
    for coeffs in cubics {
        for sigma in [1.95, 2.4, 3.1, 4.1] {
            let (low, high) = neighbor_scale_band(sigma);
            for is_max in [false, true] {
                let sampled = (0..=4000)
                    .map(|step| horner(coeffs, low + (high - low) * step as f32 / 4000.0))
                    .fold(if is_max { f32::MIN } else { f32::MAX }, |best, value| {
                        if is_max {
                            best.max(value)
                        } else {
                            best.min(value)
                        }
                    });
                let analytic = band_extreme(coeffs, low, high, is_max);
                assert!(
                    (analytic - sampled).abs() <= 1e-4 * (1.0 + sampled.abs()),
                    "coeffs={coeffs:?} sigma={sigma} is_max={is_max}: {analytic} vs {sampled}"
                );
            }
        }
    }
    assert!(band_extreme([f32::NAN, 0.0, 0.0, 0.0], 2.0, 2.5, true).is_nan());
}

/// A bright Gaussian blob of width `s` has the centered σ²-normalized LoG profile
/// `−2σ²s²/(σ²+s²)²`, symmetric in log σ. Exactly one of two adjacent octaves
/// owns its extremum, and octave `o` owns `s ∈ (√(σ₀σ₁), √(σ₂σ₃))`.
#[test]
fn scale_ownership_assigns_each_blob_scale_to_one_octave() {
    let profile = |s: f32, octave_scale: f32| -> [f32; 4] {
        core::array::from_fn(|level| {
            let sigma = SIGMAS[level] * octave_scale;
            -2.0 * sigma * sigma * s * s / (sigma * sigma + s * s).powi(2)
        })
    };
    let low = (SIGMAS[0] * SIGMAS[1]).sqrt();
    let high = (SIGMAS[2] * SIGMAS[3]).sqrt();
    for step in 0..=200 {
        let s = 1.7 + 2.6 * step as f32 / 200.0;
        let own = owns_scale_extremum(&profile(s, 1.0), false);
        let next = owns_scale_extremum(&profile(s, 2.0), false);
        if (s - low).abs() > 1e-3 && (s - high).abs() > 1e-3 {
            assert_eq!(own, s > low && s < high, "s={s}");
            assert!(!(own && next), "s={s} is owned by two octaves");
        }
        if s > high + 1e-3 {
            assert!(next, "s={s} belongs to the next octave");
        }
        // A dark blob mirrors the bright minimum into a maximum.
        let bright = profile(s, 1.0).map(|value| -value);
        assert_eq!(owns_scale_extremum(&bright, true), own, "s={s}");
    }
}
