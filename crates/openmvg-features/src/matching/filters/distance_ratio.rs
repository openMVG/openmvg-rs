// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lowe's distance ratio test ([`LoweRatioFilter`]).

use crate::matching::types::{FilteredMatch, Match, is_valid_lowe_ratio};

/// Distance ratio filter for feature matches (Lowe's ratio test).
///
/// The distance ratio test drops ambiguous matches by comparing the distance from a query
/// descriptor to its nearest target with the distance to its second-nearest target. A match is
/// kept if the ratio is below a threshold (typically 0.7 to 0.8).
///
/// This removes matches where several targets are about as close, which often means a wrong
/// match on a repetitive pattern or noise. David Lowe introduced the test with SIFT, but it
/// applies to any distance-based matching.
///
/// The filter is one-way: several queries may keep the same target. To also require the target
/// to choose the query back, as `openmvg-sfm` does, use
/// [`MatchAcceptance::Reciprocal`](crate::matching::MatchAcceptance::Reciprocal).
///
/// # Algorithm
///
/// The caller supplies, for each query descriptor, its two nearest target descriptors, nearest
/// first, from any matcher. The filter keeps the nearest target only if:
///
/// ```text
/// distance(best_match) / distance(second_best_match) < threshold
/// ```
///
/// # Threshold selection
///
/// The filter has no built-in threshold: each call takes the ratio. Lowe's
/// original recommendation for SIFT is 0.8. Lower values keep fewer, more
/// distinctive matches; higher values keep more matches and more outliers.
///
/// | Threshold | Precision | Recall | Use case |
/// |-----------|-----------|--------|----------|
/// | 0.6 | Very high | Low | Strict matching, repetitive textures |
/// | 0.7 | High | Medium | Balanced |
/// | 0.8 | Medium | High | Lowe's original value |
/// | 0.9 | Low | Very high | Permissive, low-texture scenes |
///
/// # Descriptor type considerations
///
/// - Floating-point descriptors (SIFT, RootSIFT) use L2 distance; 0.7 to 0.8
///   is the usual range.
/// - Binary descriptors use Hamming distance. Their distances take few
///   distinct values, which makes the ratio less discriminative, so they may
///   need a lower threshold.
///
/// # Examples
///
/// ```rust
/// use openmvg_features::matching::{LoweRatioFilter, Match, MatcherError};
///
/// let nearest = vec![vec![Match::try_new(4, 0.2)?, Match::try_new(7, 0.5)?]];
/// let filtered = LoweRatioFilter::new().apply_unsquared(&nearest, 0.7);
/// assert_eq!(filtered[0].target_index(), 4);
/// # Ok::<(), MatcherError>(())
/// ```
///
/// # References
///
/// - Lowe, D.G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints."
///   International Journal of Computer Vision, 60(2), 91-110.
///   DOI: 10.1023/B:VISI.0000029664.99615.94
#[derive(Debug, Clone, Copy, Default)]
pub struct LoweRatioFilter;

impl LoweRatioFilter {
    /// Create a distance ratio filter.
    pub fn new() -> Self {
        Self
    }

    /// Shared implementation after selecting the distance domain.
    fn apply(matches: &[Vec<Match>], ratio: f32, squared_distances: bool) -> Vec<FilteredMatch> {
        if !is_valid_lowe_ratio(ratio) {
            return Vec::new();
        }

        matches
            .iter()
            .enumerate()
            .filter_map(|(query_index, query_matches)| {
                let [best, second, ..] = query_matches.as_slice() else {
                    return None;
                };
                let reported_ratio = lowe_ratio(
                    f64::from(best.distance),
                    f64::from(second.distance),
                    ratio,
                    squared_distances,
                )?;
                Some(FilteredMatch::new_unchecked(
                    query_index,
                    best.index,
                    best.distance,
                    reported_ratio,
                ))
            })
            .collect()
    }

    /// Apply Lowe's ratio test to ordinary (unsquared) distances.
    ///
    /// `ratio` must be finite and lie strictly inside `(0, 1)`; any other value rejects every
    /// match. Each query's candidates must be ordered nearest first. A query with fewer than
    /// two candidates, which an approximate matcher can return, or with a second distance of
    /// zero, gives no match.
    pub fn apply_unsquared(&self, matches: &[Vec<Match>], ratio: f32) -> Vec<FilteredMatch> {
        Self::apply(matches, ratio, false)
    }

    /// Apply Lowe's ratio expressed in ordinary L2 units to squared-L2
    /// distances. Squaring the threshold preserves the intended inequality:
    /// `sqrt(d1) / sqrt(d2) < ratio` iff `d1 / d2 < ratio^2`.
    ///
    /// The inputs follow the rules of [`Self::apply_unsquared`].
    pub fn apply_squared(&self, matches: &[Vec<Match>], ratio: f32) -> Vec<FilteredMatch> {
        Self::apply(matches, ratio, true)
    }
}

/// Lowe's test on one query's two nearest distances, for a ratio already
/// checked by [`is_valid_lowe_ratio`].
///
/// Returns the ratio in ordinary distance units when `best / second` is below
/// the threshold. Squared distances are compared against `ratio²`. Distances
/// must be finite with `best >= 0` and `second > 0`.
#[inline]
pub(crate) fn lowe_ratio(best: f64, second: f64, ratio: f32, squared: bool) -> Option<f32> {
    if !best.is_finite() || !second.is_finite() || best < 0.0 || second <= 0.0 {
        return None;
    }
    let raw_ratio = best / second;
    let threshold = if squared {
        f64::from(ratio) * f64::from(ratio)
    } else {
        f64::from(ratio)
    };
    (raw_ratio < threshold).then(|| {
        if squared {
            raw_ratio.sqrt() as f32
        } else {
            raw_ratio as f32
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_ratio_filter_good_matches() {
        let filter = LoweRatioFilter::new();

        // Create matches where first match is much better than second
        let matches = vec![
            vec![
                Match::new_unchecked(0, 0.1), // Good match
                Match::new_unchecked(1, 1.0), // Much worse match
            ],
            vec![
                Match::new_unchecked(2, 0.2), // Good match
                Match::new_unchecked(3, 1.5), // Much worse match
            ],
        ];

        let filtered = filter.apply_unsquared(&matches, 0.8);

        assert_eq!(filtered.len(), 2);

        // Check first filtered match
        assert_eq!(filtered[0].query_index, 0);
        assert_eq!(filtered[0].target_index, 0);
        assert_eq!(filtered[0].distance, 0.1);
        assert!((filtered[0].ratio - 0.1).abs() < 1e-6); // 0.1 / 1.0 = 0.1

        // Check second filtered match
        assert_eq!(filtered[1].query_index, 1);
        assert_eq!(filtered[1].target_index, 2);
        assert_eq!(filtered[1].distance, 0.2);
        assert!((filtered[1].ratio - 0.133333).abs() < 1e-4); // 0.2 / 1.5 ≈ 0.133
    }

    #[test]
    fn test_distance_ratio_filter_bad_matches() {
        let filter = LoweRatioFilter::new();

        // Create matches where both matches are similarly good (ambiguous)
        let matches = vec![vec![
            Match::new_unchecked(0, 0.5), // Best match
            Match::new_unchecked(1, 0.6), // Similar match (ratio = 0.5/0.6 ≈ 0.83 > 0.8)
        ]];

        let filtered = filter.apply_unsquared(&matches, 0.8);

        // Should be filtered out due to high ratio
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_squared_distance_ratio_squares_threshold() {
        let filter = LoweRatioFilter::new();
        let matches = vec![vec![
            Match::new_unchecked(0, 0.7),
            Match::new_unchecked(1, 1.0),
        ]];

        assert_eq!(filter.apply_unsquared(&matches, 0.8).len(), 1);
        assert!(filter.apply_squared(&matches, 0.8).is_empty());

        let accepted = vec![vec![
            Match::new_unchecked(0, 0.36),
            Match::new_unchecked(1, 1.0),
        ]];
        let filtered = filter.apply_squared(&accepted, 0.8);
        assert!((filtered[0].ratio - 0.6).abs() < 1e-6);
    }

    #[test]
    fn invalid_ratio_or_distance_is_rejected() {
        let filter = LoweRatioFilter::new();
        let valid = vec![vec![
            Match::new_unchecked(0, 0.1),
            Match::new_unchecked(1, 1.0),
        ]];
        for invalid in [f32::NAN, f32::NEG_INFINITY, -0.8, 0.0, 1.0, f32::INFINITY] {
            assert!(filter.apply_unsquared(&valid, invalid).is_empty());
            assert!(filter.apply_squared(&valid, invalid).is_empty());
        }

        let invalid = vec![vec![
            Match::new_unchecked(0, f32::NAN),
            Match::new_unchecked(1, 1.0),
        ]];
        assert!(filter.apply_unsquared(&invalid, 0.8).is_empty());
    }

    #[test]
    fn positive_sub_epsilon_second_distance_is_valid() {
        let filter = LoweRatioFilter::new();
        let matches = vec![vec![
            Match::new_unchecked(0, 0.0),
            Match::new_unchecked(1, 1.0e-8),
        ]];

        assert_eq!(filter.apply_unsquared(&matches, 0.8).len(), 1);
        assert_eq!(filter.apply_squared(&matches, 0.8).len(), 1);
    }

    #[test]
    fn tiny_positive_ratio_does_not_underflow_when_squared() {
        let filter = LoweRatioFilter::new();
        let matches = vec![vec![
            Match::new_unchecked(0, 0.0),
            Match::new_unchecked(1, 1.0),
        ]];

        assert_eq!(filter.apply_squared(&matches, f32::MIN_POSITIVE).len(), 1);
    }

    #[test]
    fn squared_and_unsquared_domains_make_the_same_decision() {
        let filter = LoweRatioFilter::new();
        for (best, second) in [(0.0_f32, 1.0e-8), (0.1, 0.3), (0.6, 1.0), (0.95, 1.0)] {
            let ordinary = vec![vec![
                Match::new_unchecked(0, best),
                Match::new_unchecked(1, second),
            ]];
            let squared = vec![vec![
                Match::new_unchecked(0, best * best),
                Match::new_unchecked(1, second * second),
            ]];
            for ratio in [0.5, 0.8, 0.9] {
                assert_eq!(
                    filter.apply_unsquared(&ordinary, ratio).len(),
                    filter.apply_squared(&squared, ratio).len()
                );
            }
        }
    }

    #[test]
    fn test_distance_ratio_filter_single_match() {
        let filter = LoweRatioFilter::new();

        let matches = vec![
            vec![Match::new_unchecked(0, 0.1)], // Only one match
        ];

        // Default behavior requires two matches
        let filtered = filter.apply_unsquared(&matches, 0.8);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_distance_ratio_filter_no_matches() {
        let filter = LoweRatioFilter::new();

        let matches = vec![
            vec![], // No matches
        ];

        let filtered = filter.apply_unsquared(&matches, 0.8);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_distance_ratio_filter_zero_distance() {
        let filter = LoweRatioFilter::new();

        // Second match has zero distance (should be filtered out to avoid division by zero)
        let matches = vec![vec![
            Match::new_unchecked(0, 0.1),
            Match::new_unchecked(1, 0.0), // Zero distance
        ]];

        let filtered = filter.apply_unsquared(&matches, 0.8);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_distance_ratio_filter_boundary_case() {
        let filter = LoweRatioFilter::new();

        // Ratio exactly at threshold
        let matches = vec![vec![
            Match::new_unchecked(0, 0.8),
            Match::new_unchecked(1, 1.0), // Ratio = 0.8
        ]];

        let filtered = filter.apply_unsquared(&matches, 0.8);
        assert_eq!(filtered.len(), 0); // Should be filtered out (>= threshold)

        // Just below threshold
        let matches = vec![vec![
            Match::new_unchecked(0, 0.79),
            Match::new_unchecked(1, 1.0), // Ratio = 0.79
        ]];

        let filtered = filter.apply_unsquared(&matches, 0.8);
        assert_eq!(filtered.len(), 1); // Should pass
    }
}
