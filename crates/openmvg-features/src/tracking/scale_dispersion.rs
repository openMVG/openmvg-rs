// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scale-dispersion filter (track-level rejection).
//!
//! Rejects entire tracks whose SIFT scale ratios are inconsistent, which
//! indicates a likely track merge. It requires no geometric information.
//!
//! For a track with L observations, we compute `1.4826 * MAD(log(scale))`.
//! Merged tracks have dispersed or multimodal log-scales and therefore a high
//! score. Computing the statistic directly is order-invariant and linear in L,
//! apart from median selection.
//!
//! # Complexity
//! O(L) expected time and O(L) temporary memory per track.
//!
//! # Usage
//!
//! [`evaluate_scales`] scores the already-collected scales of one track, and
//! [`filter_indices`] returns the indices of the tracks that pass, given a
//! `(view, feature) -> scale` lookup.

use super::statistics::fast_median;
use super::types::TrackGroup as Track;
use openmvg_core::model::{FeatureId, ViewId};
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
use rayon::prelude::*;

/// MAD to sigma scaling factor (1 / Phi^-1(3/4) ≈ 1.4826).
const MAD_SCALE: f64 = 1.4826;

/// Configuration for the scale dispersion filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleDispersionConfig {
    threshold: f64,
}

config_accessors!(ScaleDispersionConfig {
    /// Rejection threshold on the MAD-based score. Default: 0.35.
    copy threshold, with_threshold: f64;
});

impl Default for ScaleDispersionConfig {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ScaleDispersionConfig {
    /// Create the default configuration: a threshold of 0.35.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { threshold: 0.35 }
    }
}

/// Scale-dispersion score of one track and its verdict.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ScaleDispersionReport {
    /// Median absolute deviation of the track's `ln(scale)` values, times 1.4826 so that it
    /// estimates a standard deviation. Higher means less consistent scales.
    pub score: f64,
    /// Whether the track should be rejected.
    pub reject: bool,
}

impl ScaleDispersionReport {
    /// Report for a track that always passes (for example, too short).
    #[inline]
    const fn pass() -> Self {
        Self {
            score: 0.0,
            reject: false,
        }
    }
}

/// Evaluate a single track for scale consistency.
///
/// Returns the score and whether the track should be rejected.
/// Tracks with fewer than 3 observations are always kept (score = 0).
#[inline]
#[must_use]
fn evaluate<F>(
    track: &Track,
    scale_lookup: F,
    config: &ScaleDispersionConfig,
) -> ScaleDispersionReport
where
    F: Fn(ViewId, FeatureId) -> f64,
{
    if track.len() < 3 {
        return ScaleDispersionReport::pass();
    }
    let scales: Vec<_> = track
        .iter()
        .map(|observation| scale_lookup(observation.view, observation.feature))
        .collect();
    evaluate_scales(&scales, config)
}

/// Evaluate the already-collected scales of one track.
///
/// Non-finite and non-positive values are ignored. Fewer than three valid
/// scales pass because a robust dispersion estimate is not identifiable.
#[inline]
#[must_use]
pub fn evaluate_scales(scales: &[f64], config: &ScaleDispersionConfig) -> ScaleDispersionReport {
    if scales.len() < 3 {
        return ScaleDispersionReport::pass();
    }
    let mut log_scales: Vec<_> = scales
        .iter()
        .copied()
        .filter(|scale| scale.is_finite() && *scale > 0.0)
        .map(f64::ln)
        .collect();
    if log_scales.len() < 3 {
        return ScaleDispersionReport::pass();
    }

    let median_log_scale = fast_median(&mut log_scales);
    for log_scale in &mut log_scales {
        *log_scale = (*log_scale - median_log_scale).abs();
    }
    let mad = fast_median(&mut log_scales);
    let score = mad * MAD_SCALE;

    ScaleDispersionReport {
        score,
        reject: score > config.threshold,
    }
}

/// Filter tracks, returning the indices into `tracks` of those that pass.
///
/// `scale_lookup` maps `(view, feature)` to the feature's detector scale.
/// Use the returned indices to access surviving tracks from the original
/// slice; no track is copied. Tracks with fewer than 3 observations always
/// pass.
#[must_use]
pub fn filter_indices<F>(
    tracks: &[Track],
    scale_lookup: F,
    config: &ScaleDispersionConfig,
) -> Vec<usize>
where
    F: Fn(ViewId, FeatureId) -> f64 + Sync,
{
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    let result = tracks
        .par_iter()
        .enumerate()
        .filter_map(|(i, track)| {
            let report = evaluate(track, &scale_lookup, config);
            if report.reject { None } else { Some(i) }
        })
        .collect();
    #[cfg(any(not(feature = "parallel"), target_arch = "wasm32"))]
    let result = tracks
        .iter()
        .enumerate()
        .filter_map(|(i, track)| {
            let report = evaluate(track, &scale_lookup, config);
            if report.reject { None } else { Some(i) }
        })
        .collect();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::TrackObservation;

    fn observation(view: usize, feature: usize) -> TrackObservation {
        TrackObservation::new(ViewId::new(view), FeatureId::new(feature))
    }

    /// Helper: Create a track and a scale lookup function from a list of scales.
    /// Each scale is associated with a unique (image_id=index, feature_id=0).
    fn make_track_and_lookup(
        scales: &[f64],
    ) -> (Track, impl Fn(ViewId, FeatureId) -> f64 + Sync + '_) {
        let track: Track = scales
            .iter()
            .enumerate()
            .map(|(index, _)| observation(index, 0))
            .collect();
        let lookup = move |view: ViewId, _feature: FeatureId| -> f64 { scales[view.index()] };
        (track, lookup)
    }

    #[test]
    fn consistent_scales_pass() {
        let (track, lookup) = make_track_and_lookup(&[2.0, 2.1, 1.9, 2.05, 2.0]);
        let config = ScaleDispersionConfig::default();
        let report = evaluate(&track, &lookup, &config);
        assert!(!report.reject, "Consistent scales should not be rejected");
        assert!(report.score < 0.1);
    }

    #[test]
    fn bimodal_scales_rejected() {
        let (track, lookup) = make_track_and_lookup(&[
            2.0, 2.1, 1.9, 2.05, 2.0, // Group A: scale ~2
            50.0, 51.0, 49.0, 50.5, 50.0, // Group B: scale ~50
        ]);
        let config = ScaleDispersionConfig::new().with_threshold(0.12);
        let report = evaluate(&track, &lookup, &config);
        assert!(
            report.reject,
            "Bimodal scales should be rejected with tight threshold, score={}",
            report.score
        );

        // Also verify that consistent scales still pass with the tight threshold.
        let (track2, lookup2) =
            make_track_and_lookup(&[2.0, 2.1, 1.9, 2.05, 2.0, 2.02, 1.98, 2.03, 1.97, 2.01]);
        let report2 = evaluate(&track2, &lookup2, &config);
        assert!(
            !report2.reject,
            "Consistent scales should pass even with tight threshold, score={}",
            report2.score
        );
    }

    #[test]
    fn short_track_always_passes() {
        let (track, lookup) = make_track_and_lookup(&[1.0, 100.0]);
        let config = ScaleDispersionConfig::default();
        let report = evaluate(&track, &lookup, &config);
        assert!(!report.reject);
        assert_eq!(report.score, 0.0);
    }

    #[test]
    fn score_increases_with_dispersion() {
        let (t1, l1) = make_track_and_lookup(&[2.0, 2.1, 1.9, 2.05, 2.0]);
        let (t2, l2) = make_track_and_lookup(&[2.0, 4.0, 1.0, 8.0, 0.5]);
        let config = ScaleDispersionConfig::default();
        let r1 = evaluate(&t1, &l1, &config);
        let r2 = evaluate(&t2, &l2, &config);
        assert!(
            r2.score > r1.score,
            "More dispersed track should have higher score: {} vs {}",
            r2.score,
            r1.score
        );
    }

    #[test]
    fn filter_indices_works() {
        let scales_good = [2.0, 2.1, 1.9, 2.05, 2.0];
        let scales_bad = [2.0, 2.1, 1.9, 50.0, 51.0, 49.0];

        let track_good: Track = scales_good
            .iter()
            .enumerate()
            .map(|(index, _)| observation(index, 0))
            .collect();
        let track_bad: Track = scales_bad
            .iter()
            .enumerate()
            .map(|(index, _)| observation(index + 100, 0))
            .collect();

        let tracks = vec![track_good.clone(), track_bad];

        let lookup = |view: ViewId, _feature: FeatureId| -> f64 {
            if view.index() < 100 {
                scales_good[view.index()]
            } else {
                scales_bad[view.index() - 100]
            }
        };

        let config = ScaleDispersionConfig::new().with_threshold(0.12);
        let indices = filter_indices(&tracks, lookup, &config);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0); // First track (good) survives
    }

    #[test]
    fn long_track_uses_same_deterministic_statistic() {
        let scales: Vec<f64> = (0..33).map(|i| 2.0 + (i as f64) * 0.01).collect();
        let (track, lookup) = make_track_and_lookup(&scales);

        let config = ScaleDispersionConfig::default();
        let report = evaluate(&track, &lookup, &config);

        // Should pass since scales are consistent
        assert!(!report.reject);
        assert!(report.score < 0.1);
    }

    #[test]
    fn score_is_order_invariant() {
        let config = ScaleDispersionConfig::default();
        let blocked = [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0];
        let interleaved = [1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0];
        assert_eq!(
            evaluate_scales(&blocked, &config),
            evaluate_scales(&interleaved, &config)
        );
        assert!(evaluate_scales(&blocked, &config).reject);
    }

    #[test]
    fn invalid_scales_are_ignored() {
        let config = ScaleDispersionConfig::default();
        let base = evaluate_scales(&[7.0, 7.1, 6.9, 7.05], &config);
        let contaminated = evaluate_scales(&[7.0, 7.1, 6.9, 7.05, 0.0, -1.0, f64::NAN], &config);
        assert_eq!(base, contaminated);
    }
}
