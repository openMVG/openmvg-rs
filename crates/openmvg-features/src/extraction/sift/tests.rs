// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::common::SIFT_DESCR_SIZE;
use super::localization::{SiftCandidate, SiftFeature, is_local_min_max, refine_keypoints};
use super::scale_space::{
    DogOctave, HierarchicalGaussianScaleSpace, compute_dogs, decimate_2x, upsample_2x,
};
use super::*;
use crate::extraction::feature_budget::{
    FeatureBudget, FeatureBudgetKeypoint, FeatureBudgetPolicy,
};
use crate::extraction::storage::DescriptorStorage;
use crate::extraction::{F32Descriptor, ScaleSpaceFeatures, U8Descriptor};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use nalgebra as na;

type TestImage = na::DMatrix<f32>;

/// Create a synthetic image with a Gaussian blob at (cx, cy) with given sigma.
fn make_blob(size: usize, cx: f32, cy: f32, sigma: f32, amplitude: f32) -> TestImage {
    let mut img = TestImage::zeros(size, size);
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    for i in 0..size {
        for j in 0..size {
            let dx = j as f32 - cx;
            let dy = i as f32 - cy;
            img[(i, j)] = amplitude * (-(dx * dx + dy * dy) * inv_2s2).exp();
        }
    }
    // Scale to [0, 255] range
    let max_val = img.max();
    if max_val > 0.0 {
        img *= 255.0 / max_val;
    }
    img
}

/// Create a gradient image (horizontal ramp).
fn make_gradient(size: usize) -> TestImage {
    let mut img = TestImage::zeros(size, size);
    for i in 0..size {
        for j in 0..size {
            img[(i, j)] = (j as f32 / size as f32) * 255.0;
        }
    }
    img
}

fn make_texture(size: usize) -> TestImage {
    TestImage::from_fn(size, size, |row, column| {
        ((row * 37 + column * 61 + row * column * 13) % 256) as f32
    })
}

#[test]
fn test_sift_detector_creation() {
    let det = SiftExtractor::default();
    assert_eq!(det.first_octave, 0);
    assert_eq!(det.num_scales, 3);
    assert!((det.peak_threshold - 0.04).abs() < 1e-6);
    assert!((det.edge_threshold - 10.0).abs() < 1e-6);
    assert!(det.root_sift);
}

#[test]
fn invalid_input_yields_empty_results() {
    let detector = SiftExtractor::default();
    assert!(
        detector
            .detect_keypoints(&TestImage::zeros(64, 7))
            .unwrap()
            .is_empty()
    );
    let valid = make_blob(64, 32.0, 32.0, 5.0, 1.0);
    for (detector, field) in [
        (SiftExtractor::default().with_num_octaves(0), "num_octaves"),
        (SiftExtractor::default().with_num_scales(0), "num_scales"),
        (
            SiftExtractor::default().with_first_octave(1),
            "first_octave",
        ),
        (
            SiftExtractor::default().with_peak_threshold(f32::NAN),
            "peak_threshold",
        ),
        (
            SiftExtractor::default().with_edge_threshold(0.0),
            "edge_threshold",
        ),
    ] {
        assert!(matches!(
            detector.validate(),
            Err(SiftError::InvalidConfig { field: reported, .. }) if reported == field
        ));
        // Every detection entry point reports the same configuration error.
        let error = detector.validate().unwrap_err();
        assert_eq!(detector.detect_keypoints(&valid).unwrap_err(), error);
        assert_eq!(detector.extract_f32(&valid).unwrap_err(), error);
        assert_eq!(detector.extract_u8(&valid).unwrap_err(), error);
    }
    for preset in [SiftPreset::Normal, SiftPreset::High, SiftPreset::Ultra] {
        assert_eq!(SiftExtractor::with_preset(preset).validate(), Ok(()));
    }
}

#[test]
fn result_accessors_expose_aligned_parts() {
    let image = make_blob(96, 48.0, 48.0, 7.0, 1.0);
    let result = SiftExtractor::default().extract_f32(&image).unwrap();
    assert!(!result.is_empty());
    assert_eq!(result.len(), result.keypoints().len());
    assert_eq!(result.len(), result.descriptors().len());
    let (keypoints, descriptors) = result.clone().into_parts();
    assert_eq!(keypoints.as_slice(), result.keypoints());
    assert_eq!(descriptors.as_slice(), result.descriptors());
    assert!(ScaleSpaceFeatures::<F32Descriptor>::default().is_empty());
}

#[test]
fn internal_detection_records_do_not_embed_descriptor_storage() {
    assert!(core::mem::size_of::<SiftCandidate>() <= 24);
    assert!(core::mem::size_of::<SiftFeature>() <= 48);
    assert!(
        core::mem::size_of::<SiftFeature>() * 8 < core::mem::size_of::<[f32; SIFT_DESCR_SIZE]>()
    );
}

#[test]
fn detection_only_skips_descriptor_construction() {
    let detector = SiftExtractor::with_preset(SiftPreset::Normal);
    let image = make_blob(96, 48.0, 48.0, 7.0, 1.0);
    let detected = detector.detect_keypoints(&image).unwrap();
    let described = detector.extract_f32(&image).unwrap().keypoints;

    assert!(detected.len() >= described.len());
    assert!(detected.iter().all(|keypoint| {
        keypoint.x.is_finite()
            && keypoint.y.is_finite()
            && keypoint.sigma.is_finite()
            && keypoint.orientation.is_finite()
    }));
}

#[test]
fn test_sift_presets() {
    let normal = SiftExtractor::with_preset(SiftPreset::Normal);
    let high = SiftExtractor::with_preset(SiftPreset::High);
    let ultra = SiftExtractor::with_preset(SiftPreset::Ultra);

    assert_eq!(normal.first_octave, 0);
    assert!((normal.peak_threshold - 0.04).abs() < 1e-6);

    assert_eq!(high.first_octave, 0);
    assert!((high.peak_threshold - 0.01).abs() < 1e-6);

    assert_eq!(ultra.first_octave, -1);
    assert!((ultra.peak_threshold - 0.01).abs() < 1e-6);

    for detector in [normal, high, ultra] {
        assert_eq!(detector.num_octaves, 6);
        assert_eq!(detector.num_scales, 3);
        assert_eq!(detector.edge_threshold, 10.0);
        assert!(detector.root_sift);
    }
}

#[test]
fn test_sift_default_is_normal_preset() {
    let default = SiftExtractor::default();
    let normal = SiftExtractor::with_preset(SiftPreset::Normal);
    assert_eq!(
        (
            default.first_octave(),
            default.num_octaves(),
            default.num_scales(),
            default.peak_threshold().to_bits(),
            default.edge_threshold().to_bits(),
            default.root_sift(),
        ),
        (
            normal.first_octave(),
            normal.num_octaves(),
            normal.num_scales(),
            normal.peak_threshold().to_bits(),
            normal.edge_threshold().to_bits(),
            normal.root_sift(),
        )
    );
}

#[test]
fn test_sift_empty_image() {
    let det = SiftExtractor::with_preset(SiftPreset::Normal);
    let img = TestImage::zeros(4, 4); // Too small
    let kps = det.extract_f32(&img).unwrap().keypoints;
    assert!(kps.is_empty());
}

#[test]
fn test_sift_constant_image() {
    let det = SiftExtractor::with_preset(SiftPreset::Normal);
    let img = TestImage::from_element(128, 128, 128.0);
    let kps = det.extract_f32(&img).unwrap().keypoints;
    assert!(kps.is_empty(), "Constant image should produce no keypoints");
}

#[test]
fn test_sift_detect_blob() {
    let det = SiftExtractor::default().with_peak_threshold(0.01);
    let img = make_blob(200, 100.0, 100.0, 8.0, 255.0);
    let kps = det.extract_f32(&img).unwrap().keypoints;
    assert!(
        !kps.is_empty(),
        "Should detect keypoint(s) on a Gaussian blob"
    );
    // Check that at least one keypoint is near the center
    let near_center = kps
        .iter()
        .any(|kp| (kp.x - 100.0).abs() < 30.0 && (kp.y - 100.0).abs() < 30.0);
    assert!(
        near_center,
        "At least one keypoint should be near the blob center"
    );
}

#[test]
fn test_sift_descriptor_dimension() {
    let det = SiftExtractor::default().with_peak_threshold(0.01);
    let img = make_blob(200, 100.0, 100.0, 8.0, 255.0);
    let features = det.extract_f32(&img).unwrap();
    assert_eq!(features.keypoints.len(), features.descriptors.len());
    for desc in &features.descriptors {
        assert_eq!(desc.as_slice().len(), 128);
    }
}

#[test]
fn test_sift_descriptor_range() {
    let det = SiftExtractor::default().with_peak_threshold(0.01);
    let img = make_blob(200, 100.0, 100.0, 8.0, 255.0);
    let features = det.extract_f32(&img).unwrap();
    for desc in &features.descriptors {
        for &v in desc.as_slice() {
            assert!(v >= 0.0, "Descriptor values should be non-negative");
            assert!(v <= 255.0, "Descriptor values should be <= 255");
        }
    }
}

#[test]
fn test_sift_keypoints_within_bounds() {
    let det = SiftExtractor::default().with_peak_threshold(0.01);
    let (h, w) = (300, 400);
    let img = make_gradient(h.max(w));
    let sub = img.view((0, 0), (h, w)).clone_owned();
    let kps = det.extract_f32(&sub).unwrap().keypoints;
    for kp in &kps {
        assert!(kp.x >= 0.0 && kp.x <= w as f32, "x={} out of bounds", kp.x);
        assert!(kp.y >= 0.0 && kp.y <= h as f32, "y={} out of bounds", kp.y);
    }
}

#[test]
fn test_sift_threshold_affects_count() {
    let img = make_blob(200, 100.0, 100.0, 8.0, 255.0);

    let det_low = SiftExtractor::default().with_peak_threshold(0.005);
    let det_high = SiftExtractor::default().with_peak_threshold(0.05);

    let kps_low = det_low.extract_f32(&img).unwrap().keypoints;
    let kps_high = det_high.extract_f32(&img).unwrap().keypoints;

    assert!(
        kps_low.len() >= kps_high.len(),
        "Lower threshold should detect more or equal keypoints: {} vs {}",
        kps_low.len(),
        kps_high.len()
    );
}

#[test]
fn test_dog_computation() {
    let size = 64;
    let img = make_blob(size, 32.0, 32.0, 5.0, 255.0);
    let img_f = &img * (1.0 / 255.0);

    let mut ss = HierarchicalGaussianScaleSpace::new(1, 3, 1.6, 1.0, 0.5);
    ss.set_image(&img_f);
    let octave = ss.next_octave().unwrap();

    assert_eq!(octave.slices.len(), 6); // 3 + 3 supplementary
    let dogs = compute_dogs(&octave);
    assert_eq!(dogs.slices.len(), 5); // n-1 DoGs

    // Verify DoG = Gaussian[s+1] - Gaussian[s]
    let (h, w) = (dogs.slices[0].nrows(), dogs.slices[0].ncols());
    for i in 0..h {
        for j in 0..w {
            let expected = octave.slices[1][(i, j)] - octave.slices[0][(i, j)];
            let actual = dogs.slices[0][(i, j)];
            assert!(
                (expected - actual).abs() < 1e-6,
                "DoG mismatch at ({}, {}): {} vs {}",
                i,
                j,
                expected,
                actual
            );
        }
    }
}

#[test]
fn test_extrema_use_signed_comparison() {
    let mut slices = vec![TestImage::zeros(3, 3); 3];
    slices[1][(1, 1)] = 5.0;
    slices[0][(1, 1)] = -6.0;
    assert!(is_local_min_max(&slices, 1, 1, 1));

    slices[2][(1, 1)] = 6.0;
    assert!(!is_local_min_max(&slices, 1, 1, 1));

    slices[1][(1, 1)] = -5.0;
    slices[0][(1, 1)] = 6.0;
    slices[2][(1, 1)] = -4.0;
    assert!(is_local_min_max(&slices, 1, 1, 1));
}

#[test]
fn candidates_refining_to_one_sample_yield_one_keypoint() {
    // Quadratic DoG peak on sample (row 8, column 8, scale 1). The candidate
    // one column to the right shifts onto that sample while refining.
    let peak = |row: usize, col: usize, scale: usize| {
        let (dr, dc, ds) = (row as f32 - 8.0, col as f32 - 8.0, scale as f32 - 1.0);
        1.0 - 0.01 * (dr * dr + dc * dc + ds * ds)
    };
    let dogs = DogOctave {
        octave_level: 0,
        delta: 1.0,
        sigmas: vec![1.6, 2.0, 2.5],
        slices: (0..3)
            .map(|scale| TestImage::from_fn(17, 17, |row, col| peak(row, col, scale)))
            .collect(),
    };
    let candidate = |col: usize| SiftCandidate {
        scale_index: 1,
        i: col as i32,
        j: 8,
        val: peak(8, col, 1),
    };

    let refined = refine_keypoints(&dogs, vec![candidate(8), candidate(9)], 0.01, 10.0);

    assert_eq!(refined.len(), 1);
    assert_eq!(
        (refined[0].scale_index, refined[0].j, refined[0].i),
        (1, 8, 8)
    );
}

#[test]
fn test_upsample_2x() {
    let img = TestImage::from_fn(4, 4, |i, j| (i * 4 + j) as f32);
    let up = upsample_2x(&img);
    assert_eq!(up.nrows(), 8);
    assert_eq!(up.ncols(), 8);
    // Corner should be preserved
    assert!((up[(0, 0)] - img[(0, 0)]).abs() < 0.01);
}

#[test]
fn test_decimate_2x() {
    let img = TestImage::from_element(8, 8, 42.0);
    let dec = decimate_2x(&img);
    assert_eq!(dec.nrows(), 4);
    assert_eq!(dec.ncols(), 4);
    // Constant image should be preserved
    for i in 0..4 {
        for j in 0..4 {
            assert!((dec[(i, j)] - 42.0).abs() < 1e-6);
        }
    }
}

#[test]
fn test_smooth_circular_histogram() {
    use crate::extraction::sift::common::{ORI_HIST_BINS, smooth_orientation_histogram};
    let mut hist = [0.0f32; ORI_HIST_BINS];
    hist[0] = 1.0;
    smooth_orientation_histogram(&mut hist, 1);
    // After one pass: h[0] = h[35]*0.25 + 1.0*0.5 + h[1]*0.25 = 0 + 0.5 + 0 = 0.5
    assert!((hist[0] - 0.5).abs() < 1e-6);
    // Neighbors should get some weight
    assert!(hist[1] > 0.0);
    assert!(hist[ORI_HIST_BINS - 1] > 0.0);
}

#[test]
fn test_sift_typed_f32_u8_consistency() {
    let det = SiftExtractor::default().with_peak_threshold(0.01);
    let img = make_blob(200, 100.0, 100.0, 8.0, 255.0);

    let result_f32 = det.extract_f32(&img).unwrap();
    let result_u8 = det.extract_u8(&img).unwrap();

    // Same number of keypoints
    assert_eq!(result_f32.keypoints.len(), result_u8.keypoints.len());

    // Geometry should be identical
    for (kf, ku) in result_f32.keypoints.iter().zip(result_u8.keypoints.iter()) {
        assert!((kf.x - ku.x).abs() < 1e-6);
        assert!((kf.y - ku.y).abs() < 1e-6);
        assert!((kf.sigma - ku.sigma).abs() < 1e-6);
    }

    for (float, quantized) in result_f32.descriptors.iter().zip(&result_u8.descriptors) {
        for (&value, &byte) in float.as_slice().iter().zip(quantized.as_slice()) {
            assert_eq!(byte, value.round().clamp(0.0, 255.0) as u8);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn sift_output_fingerprint_is_stable() {
    fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        hash
    }

    let result = SiftExtractor::with_preset(SiftPreset::Normal)
        .extract_f32(&make_texture(128))
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

    assert_eq!(result.keypoints.len(), 21);
    // Standard-library and libm transcendental implementations differ in
    // their final bits, so lock both supported x86-64 arithmetic profiles.
    #[cfg(feature = "std")]
    assert_eq!(hash, 0x58b3_fef6_ecfa_040f);
    #[cfg(not(feature = "std"))]
    assert_eq!(hash, 0x76f4_4c37_906d_6aa7);
}

#[test]
fn typed_u8_matches_quantized_f32_output_exactly() {
    let detector = SiftExtractor::default()
        .with_num_octaves(4)
        .with_peak_threshold(0.01);
    let image = make_blob(160, 80.0, 80.0, 7.0, 255.0);

    let f32_output = detector.extract_f32(&image).unwrap();
    let u8_output = detector.extract_u8(&image).unwrap();

    assert!(!f32_output.keypoints.is_empty());
    assert_eq!(f32_output.keypoints, u8_output.keypoints);
    assert_eq!(f32_output.descriptors.len(), u8_output.descriptors.len());
    for (float, byte) in f32_output.descriptors.iter().zip(&u8_output.descriptors) {
        let expected = U8Descriptor::from_finalized_array(float.as_array());
        assert_eq!(expected.as_array(), byte.as_array());
        assert_eq!(expected.scale().to_bits(), byte.scale().to_bits());
    }
}

#[test]
fn non_binding_budget_preserves_all_output_forms_exactly() {
    let detector = SiftExtractor::default()
        .with_num_octaves(4)
        .with_peak_threshold(0.01);
    let image = make_texture(128);
    let budgeted_detector = detector
        .clone()
        .with_feature_budget(FeatureBudgetPolicy::strongest(usize::MAX));

    assert_eq!(
        detector.detect_keypoints(&image).unwrap(),
        budgeted_detector.detect_keypoints(&image).unwrap()
    );

    let direct_typed = detector.extract_f32(&image).unwrap();
    let budgeted_typed = budgeted_detector.extract_f32(&image).unwrap();
    assert_eq!(direct_typed.keypoints, budgeted_typed.keypoints);
    assert_eq!(direct_typed.descriptors, budgeted_typed.descriptors);

    let direct_u8 = detector.extract_u8(&image).unwrap();
    let budgeted_u8 = budgeted_detector.extract_u8(&image).unwrap();
    assert_eq!(direct_u8.keypoints, budgeted_u8.keypoints);
    assert_eq!(direct_u8.descriptors.len(), budgeted_u8.descriptors.len());
    for (direct, budgeted) in direct_u8.descriptors.iter().zip(&budgeted_u8.descriptors) {
        assert_eq!(direct.as_slice(), budgeted.as_slice());
        assert_eq!(direct.scale().to_bits(), budgeted.scale().to_bits());
    }
}

#[test]
fn strongest_budget_selects_before_description_and_preserves_source_order() {
    let detector = SiftExtractor::default()
        .with_num_octaves(4)
        .with_peak_threshold(0.005);
    let image = make_texture(128);
    let budget = FeatureBudgetPolicy::strongest(8);
    let full = detector.extract_f32(&image).unwrap();
    assert!(full.keypoints.len() > 8);

    let candidates: Vec<_> = full
        .keypoints
        .iter()
        .map(FeatureBudgetKeypoint::feature_candidate)
        .collect();
    let expected_indices = budget.select(&candidates, image.ncols(), image.nrows());
    let budgeted_detector = detector.clone().with_feature_budget(budget);
    let budgeted = budgeted_detector.extract_f32(&image).unwrap();
    assert_eq!(
        budgeted_detector.detect_keypoints(&image).unwrap(),
        budgeted.keypoints
    );

    assert_eq!(budgeted.keypoints.len(), expected_indices.len());
    assert_eq!(budgeted.descriptors.len(), expected_indices.len());
    for ((keypoint, descriptor), &source_index) in budgeted
        .keypoints
        .iter()
        .zip(&budgeted.descriptors)
        .zip(&expected_indices)
    {
        assert_eq!(*keypoint, full.keypoints[source_index]);
        assert_eq!(*descriptor, full.descriptors[source_index]);
    }
}

#[test]
fn zero_budget_skips_all_sift_descriptors() {
    let detector = SiftExtractor::default()
        .with_num_octaves(4)
        .with_peak_threshold(0.005);
    let image = make_texture(96);
    let detector = detector.with_feature_budget(FeatureBudgetPolicy::strongest(0));

    let f32_result = detector.extract_f32(&image).unwrap();
    let u8_result = detector.extract_u8(&image).unwrap();

    assert!(f32_result.keypoints.is_empty());
    assert!(f32_result.descriptors.is_empty());
    assert!(u8_result.keypoints.is_empty());
    assert!(u8_result.descriptors.is_empty());
}

#[test]
fn test_sift_multiple_blobs() {
    let size = 300;
    let mut img = TestImage::zeros(size, size);
    // Add blobs at different positions and scales
    let blobs = [(75.0, 75.0, 5.0), (200.0, 100.0, 10.0), (150.0, 220.0, 7.0)];
    for &(cx, cy, sigma) in &blobs {
        let blob = make_blob(size, cx, cy, sigma, 200.0);
        img += &blob;
    }
    img = img.map(|v| v.min(255.0));

    let det = SiftExtractor::default().with_peak_threshold(0.005);
    let kps = det.extract_f32(&img).unwrap().keypoints;
    assert!(
        kps.len() >= 2,
        "Should detect at least 2 keypoints from 3 blobs, got {}",
        kps.len()
    );
}

#[test]
fn test_gaussian_blob_scale_is_consistent() {
    let detector = SiftExtractor::default()
        .with_num_octaves(8)
        .with_peak_threshold(0.001);
    let mut ratios = Vec::new();
    for truth in [3.0f32, 6.0, 12.0, 16.0] {
        let image = make_blob(257, 128.0, 128.0, truth, 255.0);
        let keypoint = detector
            .extract_f32(&image)
            .unwrap()
            .keypoints
            .into_iter()
            .filter(|kp| (kp.x - 128.0).abs() < 3.0 && (kp.y - 128.0).abs() < 3.0)
            .max_by(|a, b| a.peak.abs().total_cmp(&b.peak.abs()))
            .unwrap_or_else(|| panic!("no center keypoint for sigma={truth}"));
        ratios.push(keypoint.sigma / truth);
    }
    let min_ratio = ratios.iter().copied().fold(f32::INFINITY, f32::min);
    let max_ratio = ratios.iter().copied().fold(0.0f32, f32::max);
    assert!(min_ratio > 0.8 && max_ratio < 1.0, "ratios={ratios:?}");
    assert!(max_ratio - min_ratio < 0.05, "ratios={ratios:?}");
}

#[test]
fn coarse_octave_blob_centres_are_unbiased() {
    let detector = SiftExtractor::default()
        .with_num_octaves(8)
        .with_peak_threshold(0.001);
    let (cx, cy) = (200.3f32, 199.6f32);
    for (truth, octave) in [(5.0f32, 1), (10.0, 2), (20.0, 3)] {
        let keypoint = detector
            .extract_f32(&make_blob(401, cx, cy, truth, 255.0))
            .unwrap()
            .keypoints
            .into_iter()
            .filter(|kp| (kp.x - cx).abs() < 6.0 && (kp.y - cy).abs() < 6.0)
            .max_by(|a, b| a.peak.abs().total_cmp(&b.peak.abs()))
            .unwrap_or_else(|| panic!("no center keypoint for sigma={truth}"));
        let (dx, dy) = (keypoint.x - cx, keypoint.y - cy);
        assert_eq!(keypoint.octave, octave, "sigma={truth}");
        assert!(
            dx.abs() < 0.1 && dy.abs() < 0.1,
            "sigma={truth}: ({dx}, {dy})"
        );
    }
}

#[test]
fn test_sift_first_octave_minus1() {
    let img = make_blob(100, 50.0, 50.0, 6.0, 255.0);

    let det_normal = SiftExtractor::default().with_peak_threshold(0.01);
    let det_ultra = SiftExtractor::default()
        .with_first_octave(-1)
        .with_peak_threshold(0.01);

    let kps_normal = det_normal.extract_f32(&img).unwrap().keypoints;
    let kps_ultra = det_ultra.extract_f32(&img).unwrap().keypoints;

    // Ultra mode (upscaled) should generally detect more keypoints
    // but both should find something
    assert!(
        !kps_normal.is_empty() || !kps_ultra.is_empty(),
        "At least one mode should detect keypoints"
    );

    let normalized = &img * (1.0 / 255.0);
    let mut scale_space = HierarchicalGaussianScaleSpace::new(1, 3, 0.8, 0.5, 0.5);
    scale_space.set_image(&normalized);
    let first = scale_space.next_octave().expect("Ultra first octave");
    assert_eq!(first.octave_level, 0);
}

#[test]
fn test_edge_rejection() {
    // Create a strong edge (horizontal line)
    let size = 128;
    let mut img = TestImage::from_element(size, size, 128.0);
    for j in 0..size {
        img[(size / 2, j)] = 255.0;
        img[(size / 2 + 1, j)] = 255.0;
    }

    // With low edge threshold: many points rejected
    let det_strict = SiftExtractor::default()
        .with_peak_threshold(0.005)
        .with_edge_threshold(5.0);
    // With high edge threshold: more points pass
    let det_lenient = SiftExtractor::default()
        .with_peak_threshold(0.005)
        .with_edge_threshold(100.0);

    let kps_strict = det_strict.extract_f32(&img).unwrap().keypoints;
    let kps_lenient = det_lenient.extract_f32(&img).unwrap().keypoints;

    assert!(
        kps_strict.len() <= kps_lenient.len(),
        "Stricter edge threshold should produce fewer or equal keypoints"
    );
}

#[test]
fn test_hierarchical_scale_space_octave_count() {
    let img = TestImage::from_element(256, 256, 128.0) * (1.0 / 255.0);
    let mut ss = HierarchicalGaussianScaleSpace::new(10, 3, 1.6, 1.0, 0.5);
    ss.set_image(&img);

    let mut count = 0;
    while ss.next_octave().is_some() {
        count += 1;
    }
    // 256 / 32 = 8, log2(8) = 3, so max ~3 octaves
    assert!(
        (1..=5).contains(&count),
        "Octave count {} seems wrong",
        count
    );
}

// --- Size-guard and small-resolution crash tests ---
// These exist to pin the behavior of the set_image() octave clamp.
// None of them assert a keypoint count; the sole contract is "no panic".

#[test]
fn test_sift_below_size_guard_returns_empty() {
    // image.nrows() < 8 guard: must return empty without panicking.
    let det = SiftExtractor::with_preset(SiftPreset::Normal);
    for sz in [1_usize, 4, 7] {
        let img = TestImage::zeros(sz, sz);
        assert!(
            det.extract_f32(&img).unwrap().keypoints.is_empty(),
            "{}×{} is below the h<8 guard — expected empty",
            sz,
            sz
        );
    }
}

#[test]
fn test_sift_does_not_crash_small_resolution() {
    // Sizes from the guard boundary up to the first full-octave threshold.
    // Must not panic for any of these.
    let det = SiftExtractor::with_preset(SiftPreset::Normal);
    for sz in [8_usize, 9, 10, 15, 32, 50, 63, 64, 65, 128] {
        let img = TestImage::zeros(sz, sz);
        let _ = det.extract_f32(&img).unwrap();
    }
}

#[test]
fn test_sift_does_not_crash_non_square_small() {
    // min(h, w) drives the octave clamp; ensure rectangular images
    // with a small short side are handled without panic.
    let det = SiftExtractor::with_preset(SiftPreset::Normal);
    let _ = det.extract_f32(&TestImage::zeros(8, 200)).unwrap();
    let _ = det.extract_f32(&TestImage::zeros(200, 8)).unwrap();
    let _ = det.extract_f32(&TestImage::zeros(63, 500)).unwrap();
    let _ = det.extract_f32(&TestImage::zeros(500, 63)).unwrap();
}

#[test]
fn scale_level_metadata_must_fit_without_allocating_it() {
    for count in [usize::MAX, usize::MAX - 1, usize::MAX - 2, usize::MAX - 3] {
        let detector = SiftExtractor::default().with_num_scales(count);
        assert!(matches!(
            detector.validate(),
            Err(SiftError::InvalidConfig {
                field: "num_scales",
                ..
            })
        ));
    }
    assert!(SiftExtractor::default().validate().is_ok());
}
