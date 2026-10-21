// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Brute-force feature matcher.
//!
//! Its teaching docs are on [`BruteForceMatcher`], where docs.rs shows them.

use crate::matching::{
    acceptance::{MatchAcceptance, NearestTwo},
    distance::SquaredL2DistanceU8,
    matchers::base::{
        BaseMatcherValidation, DirectMatchingUtils, MatcherAlgorithms, MatcherCommon,
    },
    traits::{DistanceMetric, FeatureMatcher},
    types::{FilteredMatch, Match, MatcherError},
};

// ============================================================================
// BruteForceMatcher
// ============================================================================

/// Brute-force feature matcher with exhaustive search.
///
/// Compares each query against all database features to find the exact
/// nearest neighbors. `Send + Sync`, so it composes with image-level
/// parallelism.
///
/// # How it works
///
/// Exhaustive nearest neighbor search by computing distances to all database features.
/// Guarantees exact results but scales linearly with database size.
///
/// # Algorithm overview
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │                        Brute-force feature matcher                          │
/// │                   Exact nearest neighbor via exhaustive search              │
/// └─────────────────────────────────────────────────────────────────────────────┘
///
///     Input: Query descriptor q ∈ ℝᵈ, Database D = {d₁, d₂, ..., dₙ}
///
/// ┌───────────────────────────────────────────────────────────────────────────┐
/// │                        Build phase (optional)                             │
/// └───────────────────────────────────────────────────────────────────────────┘
///                                    │
///                                    ▼
///     Store database features D (no index structure needed)
///     Validate: consistent dimensions, non-empty
///
/// ┌───────────────────────────────────────────────────────────────────────────┐
/// │                        Query phase (per query)                            │
/// └───────────────────────────────────────────────────────────────────────────┘
///                                    │
///                                    ▼
///     ┌─────────────────────────────────────────────────────────────────────┐
///     │  1. Exhaustive distance computation                                 │
///     └─────────────────────────────────────────────────────────────────────┘
///                                    │
///         For each database descriptor dᵢ:
///           dist[i] = distance(q, dᵢ)   (e.g., L2, squared L2)
///
///         Complexity: O(n × d) where n = database size, d = dimension
///                                    │
///                                    ▼
///     ┌─────────────────────────────────────────────────────────────────────┐
///     │  2. Sort and select top-k                                           │
///     └─────────────────────────────────────────────────────────────────────┘
///                                    │
///         Select the k smallest distances, ties broken by index
///         Return them sorted ascending, with indices
///
///         Complexity: O(n + k log k) for k < n/4, otherwise O(n log n)
///
///     Output: k nearest neighbors with exact distances
/// ```
///
/// # Complexity
///
/// | Operation | Time Complexity | Space Complexity |
/// |-----------|-----------------|------------------|
/// | Build     | O(n × d)        | O(n × d)         |
/// | Query     | O(n × d)        | O(n)             |
///
/// Where n = database size, d = descriptor dimension.
///
/// # Performance characteristics
///
/// - Accuracy: exact nearest neighbors.
/// - Each query compares every database descriptor, so matching two images costs the product
///   of their descriptor counts; [`HnswMatcher`](crate::matching::HnswMatcher) searches an
///   index instead.
/// - Direct matching skips the build step for one-time matching.
///
/// # Examples
///
/// ```rust
/// use openmvg_features::matching::BruteForceMatcher;
/// use openmvg_features::matching::{FeatureMatcher, MatcherError, SquaredL2Distance};
///
/// # fn run() -> Result<(), MatcherError> {
/// let database: [&[f32]; 2] = [&[0.0, 0.0], &[1.0, 1.0]];
/// let queries: [&[f32]; 1] = [&[0.1, 0.1]];
///
/// // Build + query for repeated searches.
/// let mut matcher = BruteForceMatcher::new(SquaredL2Distance::new());
/// matcher.build(&database)?;
/// let matches = matcher.query_batch(&queries, 2)?;
/// assert_eq!(matches[0][0].index(), 0);
///
/// // Direct matching avoids storing a second copy for one-time searches.
/// let matches = BruteForceMatcher::match_direct_slices(
///     SquaredL2Distance::new(),
///     &queries,
///     &database,
///     2
/// )?;
/// assert_eq!(matches[0][0].index(), 0);
/// # Ok(())
/// # }
/// # run().unwrap();
/// ```
#[derive(Debug)]
pub struct BruteForceMatcher<T, D>
where
    T: Clone + Send + Sync,
    D: DistanceMetric<T> + Send + Sync,
{
    common: MatcherCommon<T, D>,
}

impl<T, D> BruteForceMatcher<T, D>
where
    T: Clone + Send + Sync,
    D: DistanceMetric<T> + Send + Sync,
{
    /// Create a brute-force matcher with the given distance metric.
    pub fn new(distance_metric: D) -> Self {
        Self {
            common: MatcherCommon::new(distance_metric),
        }
    }
}

impl<T, D> BaseMatcherValidation<T> for BruteForceMatcher<T, D>
where
    T: Clone + Send + Sync,
    D: DistanceMetric<T> + Send + Sync,
{
    fn get_features(&self) -> Option<&[Vec<T>]> {
        self.common.get_features()
    }

    fn get_feature_dimension(&self) -> Option<usize> {
        self.common.get_feature_dimension()
    }
}

impl<T, D> FeatureMatcher<T> for BruteForceMatcher<T, D>
where
    T: Clone + Send + Sync,
    D: DistanceMetric<T> + Send + Sync,
{
    fn build(&mut self, features: &[&[T]]) -> Result<(), MatcherError> {
        let dimension = self.validate_build_features(features)?;
        for (index, &descriptor) in features.iter().enumerate() {
            if !self
                .common
                .distance_metric()
                .is_valid_descriptor(descriptor)
            {
                return Err(MatcherError::InvalidDescriptor { index });
            }
        }
        self.common
            .store_features(features.iter().map(|f| f.to_vec()).collect(), dimension);
        Ok(())
    }

    fn query_single(&self, query: &[T], k: usize) -> Result<Vec<Match>, MatcherError> {
        let features = self.get_features().ok_or(MatcherError::NotBuilt)?;
        self.validate_query(query)?;
        if !self.common.distance_metric().is_valid_descriptor(query) {
            return Err(MatcherError::InvalidQuery);
        }
        self.validate_k(k)?;

        if k == 2 {
            return Ok(MatcherAlgorithms::find_2nn(
                query,
                features.iter().map(Vec::as_slice),
                self.common.distance_metric(),
            ));
        }

        let distances = MatcherAlgorithms::compute_all_distances(
            query,
            features.iter().map(Vec::as_slice),
            self.common.distance_metric(),
        );
        let matches = MatcherAlgorithms::sort_and_truncate_matches(distances, k);

        Ok(matches)
    }

    fn query_batch(&self, queries: &[&[T]], k: usize) -> Result<Vec<Vec<Match>>, MatcherError> {
        self.get_features().ok_or(MatcherError::NotBuilt)?;
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        self.validate_k(k)?;

        for &query in queries {
            self.validate_query(query)?;
            if !self.common.distance_metric().is_valid_descriptor(query) {
                return Err(MatcherError::InvalidQuery);
            }
        }

        let features = self.get_features().ok_or(MatcherError::NotBuilt)?;
        Ok(queries
            .iter()
            .map(|query| {
                DirectMatchingUtils::process_direct_query(
                    query,
                    features.iter().map(Vec::as_slice),
                    k,
                    self.common.distance_metric(),
                )
            })
            .collect())
    }
}

// ============================================================================
// Direct matching (no stored database copy)
// ============================================================================

impl<T, D> BruteForceMatcher<T, D>
where
    T: Clone + Send + Sync,
    D: DistanceMetric<T> + Send + Sync,
{
    /// Direct matching with borrowed descriptor slices.
    ///
    /// Both queries and database are `&[&[T]]`, so callers holding fixed-size
    /// arrays can pass slices without copying descriptor elements.
    pub fn match_direct_slices(
        distance_metric: D,
        queries: &[&[T]],
        database: &[&[T]],
        k: usize,
    ) -> Result<Vec<Vec<Match>>, MatcherError> {
        if queries.is_empty() || database.is_empty() {
            return Ok(vec![Vec::new(); queries.len()]);
        }

        if k == 0 || k > database.len() {
            return Err(MatcherError::InvalidK { k });
        }

        // Validate dimensions
        if let (Some(&q), Some(&db)) = (queries.first(), database.first()) {
            if q.is_empty() {
                return Err(MatcherError::EmptyQuery);
            }
            let expected_dim = db.len();
            if expected_dim == 0 {
                return Err(MatcherError::EmptyFeatures);
            }
            for &feature in database {
                if feature.len() != expected_dim {
                    return Err(MatcherError::DimensionMismatch {
                        expected: expected_dim,
                        actual: feature.len(),
                    });
                }
            }
            for &query in queries {
                if query.len() != expected_dim {
                    return Err(MatcherError::DimensionMismatch {
                        expected: expected_dim,
                        actual: query.len(),
                    });
                }
            }
        }

        DirectMatchingUtils::validate_descriptor_values(
            queries.iter().copied(),
            database.iter().copied(),
            &distance_metric,
        )?;

        let results: Vec<Vec<Match>> = queries
            .iter()
            .map(|query| {
                DirectMatchingUtils::process_direct_query(
                    query,
                    database.iter().copied(),
                    k,
                    &distance_metric,
                )
            })
            .collect();

        Ok(results)
    }
}

impl BruteForceMatcher<u8, SquaredL2DistanceU8> {
    /// Exact matching of 128-byte descriptors with Lowe's ratio on squared L2
    /// distances; `ratio` is in ordinary L2 units.
    ///
    /// Each descriptor pair is compared once, updating the two nearest
    /// neighbors in both directions, and the SIMD kernel is selected once per
    /// query. Returns no matches for an invalid ratio or when
    /// [`MatchAcceptance::can_match`] rejects the descriptor counts.
    pub fn match_exact_u8(
        query: &[[u8; 128]],
        database: &[[u8; 128]],
        ratio: f32,
        acceptance: MatchAcceptance,
    ) -> Vec<FilteredMatch> {
        if !acceptance.can_match(query.len(), database.len()) {
            return Vec::new();
        }
        let metric = SquaredL2DistanceU8::new();
        let mut backward = acceptance.backward_scan(database.len());
        let mut forward = Vec::with_capacity(query.len());
        for (query_index, descriptor) in query.iter().enumerate() {
            forward.push(metric.fold_distances(
                descriptor,
                database.iter().map(|target| target.as_slice()),
                NearestTwo::EMPTY,
                |mut nearest, target_index, distance| {
                    // Byte squared-L2 distances are integers below 2^24, exact in f32.
                    let distance = distance as u32;
                    nearest.consider(distance, target_index);
                    if let Some(reverse) = backward.get_mut(target_index) {
                        reverse.consider(distance, query_index);
                    }
                    nearest
                },
            ));
        }
        acceptance.select_scans(&forward, &backward, ratio, true)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::distance::L2Distance;
    use crate::matching::matchers::as_slices;

    #[test]
    fn test_brute_force_matcher_build() {
        let mut matcher = BruteForceMatcher::new(L2Distance::new());
        let features = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0f32, 5.0, 6.0],
            vec![7.0f32, 8.0, 9.0],
        ];

        assert!(matcher.build(&as_slices(&features)).is_ok());
        assert_eq!(matcher.indexed_descriptor_count(), 3);
        assert_eq!(matcher.feature_dimension(), Some(3));
        assert!(matcher.get_features().is_some());
    }

    #[test]
    fn test_brute_force_matcher_empty_features() {
        let mut matcher = BruteForceMatcher::<f32, _>::new(L2Distance::new());
        assert!(matches!(
            matcher.build(&[]),
            Err(MatcherError::EmptyFeatures)
        ));
    }

    #[test]
    fn test_brute_force_matcher_query_single() {
        let mut matcher = BruteForceMatcher::new(L2Distance::new());
        let features = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0f32, 5.0, 6.0],
            vec![1.1f32, 2.1, 3.1], // Close to first
        ];

        matcher.build(&as_slices(&features)).unwrap();

        let query = vec![1.0f32, 2.0, 3.0];
        let matches = matcher.query_single(&query, 2).unwrap();

        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].index, 0); // Exact match
        assert!(matches[0].distance < 0.001);
        assert_eq!(matches[1].index, 2); // Close match
    }

    #[test]
    fn test_direct_matching() {
        let database = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0f32, 5.0, 6.0],
            vec![1.1f32, 2.1, 3.1],
        ];

        let queries = vec![vec![1.0f32, 2.0, 3.0], vec![4.0f32, 5.0, 6.0]];

        let results = BruteForceMatcher::match_direct_slices(
            L2Distance::new(),
            &as_slices(&queries),
            &as_slices(&database),
            2,
        )
        .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0].index, 0);
        assert_eq!(results[1][0].index, 1);
    }

    #[test]
    fn invalid_float_descriptors_are_rejected() {
        let mut matcher = BruteForceMatcher::new(L2Distance::new());
        assert!(matches!(
            matcher.build(&[&[f32::NAN, 0.0]]),
            Err(MatcherError::InvalidDescriptor { index: 0 })
        ));

        matcher.build(&[&[0.0, 0.0], &[1.0, 1.0]]).unwrap();
        assert!(matches!(
            matcher.query_single(&[f32::INFINITY, 0.0], 1),
            Err(MatcherError::InvalidQuery)
        ));
        assert!(matches!(
            matcher.query_single(&[f32::MAX, 0.0], 1),
            Err(MatcherError::InvalidQuery)
        ));
        assert!(matches!(
            BruteForceMatcher::match_direct_slices(
                L2Distance::new(),
                &[&[0.0, f32::NAN]],
                &[&[0.0, 0.0]],
                1,
            ),
            Err(MatcherError::InvalidQuery)
        ));
    }

    #[test]
    fn empty_batch_has_consistent_semantics() {
        let mut matcher = BruteForceMatcher::new(L2Distance::new());
        matcher.build(&[&[0.0f32]]).unwrap();
        assert_eq!(
            matcher.query_batch(&[], 0).unwrap(),
            Vec::<Vec<Match>>::new()
        );
        assert_eq!(
            BruteForceMatcher::match_direct_slices(L2Distance::new(), &[], &[&[0.0f32]], 0,)
                .unwrap(),
            Vec::<Vec<Match>>::new()
        );
    }

    #[test]
    fn exact_u8_matching_agrees_with_generic_search_and_acceptance() {
        let mut state = 7_u64;
        let mut next_byte = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        };
        let database: Vec<[u8; 128]> = (0..31)
            .map(|_| std::array::from_fn(|_| next_byte()))
            .collect();
        let mut query: Vec<[u8; 128]> = database[..12]
            .iter()
            .map(|descriptor| descriptor.map(|value| value.saturating_add(next_byte() % 4)))
            .collect();
        // A closer copy of target 5: one-way keeps both queries, reciprocal
        // only the closer one.
        query[6] = database[5].map(|value| value.saturating_add(1));
        query.push(std::array::from_fn(|_| next_byte()));

        let slices_q: Vec<&[u8]> = query.iter().map(|d| d.as_slice()).collect();
        let slices_db: Vec<&[u8]> = database.iter().map(|d| d.as_slice()).collect();
        let forward = BruteForceMatcher::match_direct_slices(
            SquaredL2DistanceU8::new(),
            &slices_q,
            &slices_db,
            2,
        )
        .unwrap();
        for (acceptance, expected_len) in [
            (MatchAcceptance::Reciprocal, 11),
            (MatchAcceptance::OneWay, 12),
        ] {
            let expected = acceptance
                .filter_squared(&forward, 0.8, |targets| {
                    let rows: Vec<&[u8]> = targets.iter().map(|&t| slices_db[t]).collect();
                    BruteForceMatcher::match_direct_slices(
                        SquaredL2DistanceU8::new(),
                        &rows,
                        &slices_q,
                        2,
                    )
                })
                .unwrap();
            assert_eq!(expected.len(), expected_len, "{acceptance:?}");
            assert_eq!(
                BruteForceMatcher::match_exact_u8(&query, &database, 0.8, acceptance),
                expected
            );
        }
        assert!(
            BruteForceMatcher::match_exact_u8(&query, &database, 1.0, MatchAcceptance::OneWay)
                .is_empty()
        );
    }

    #[test]
    fn failed_rebuild_preserves_the_previous_database() {
        let mut matcher = BruteForceMatcher::new(L2Distance::new());
        matcher.build(&[&[0.0f32], &[2.0]]).unwrap();
        assert!(matcher.build(&[&[f32::NAN]]).is_err());

        let matches = matcher.query_single(&[0.1], 1).unwrap();
        assert_eq!(matches[0].index, 0);
    }
}
