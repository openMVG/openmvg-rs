// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared matcher validation and selection helpers.

use crate::matching::{
    traits::DistanceMetric,
    types::{Match, MatcherError},
};

/// Common validation and utility methods for all matcher implementations.
pub(crate) trait BaseMatcherValidation<T> {
    /// Get the stored features (if any).
    fn get_features(&self) -> Option<&[Vec<T>]>;

    /// Get the feature dimension.
    fn get_feature_dimension(&self) -> Option<usize>;

    /// Get the number of features in the database.
    fn indexed_descriptor_count(&self) -> usize {
        self.get_features().map_or(0, |f| f.len())
    }

    /// Get the feature dimension.
    fn feature_dimension(&self) -> Option<usize> {
        self.get_feature_dimension()
    }

    /// Validate query dimensions against database.
    fn validate_query(&self, query: &[T]) -> Result<(), MatcherError> {
        if query.is_empty() {
            return Err(MatcherError::EmptyQuery);
        }

        if let Some(expected_dim) = self.feature_dimension()
            && query.len() != expected_dim
        {
            return Err(MatcherError::DimensionMismatch {
                expected: expected_dim,
                actual: query.len(),
            });
        }

        Ok(())
    }

    /// Validate k parameter.
    fn validate_k(&self, k: usize) -> Result<(), MatcherError> {
        if k == 0 {
            return Err(MatcherError::InvalidK { k });
        }

        let db_size = self.indexed_descriptor_count();
        if k > db_size {
            return Err(MatcherError::InvalidK { k });
        }

        Ok(())
    }

    /// Validate build input features.
    fn validate_build_features(&self, features: &[&[T]]) -> Result<usize, MatcherError> {
        if features.is_empty() {
            return Err(MatcherError::EmptyFeatures);
        }

        // Validate that all features have the same dimension
        let first_dim = features[0].len();
        if first_dim == 0 {
            return Err(MatcherError::EmptyFeatures);
        }

        for feature in features.iter() {
            if feature.len() != first_dim {
                return Err(MatcherError::DimensionMismatch {
                    expected: first_dim,
                    actual: feature.len(),
                });
            }
        }

        Ok(first_dim)
    }
}

/// Common matcher functionality that can be shared across implementations.
#[derive(Debug)]
pub(crate) struct MatcherCommon<T, D>
where
    T: Clone + Send + Sync,
    D: DistanceMetric<T> + Send + Sync,
{
    pub(crate) distance_metric: D,
    pub(crate) features: Option<Vec<Vec<T>>>,
    pub(crate) feature_dimension: Option<usize>,
}

impl<T, D> MatcherCommon<T, D>
where
    T: Clone + Send + Sync,
    D: DistanceMetric<T> + Send + Sync,
{
    /// Create new common matcher functionality.
    pub fn new(distance_metric: D) -> Self {
        Self {
            distance_metric,
            features: None,
            feature_dimension: None,
        }
    }

    /// Store features after validation (takes ownership for efficiency).
    pub fn store_features(&mut self, features: Vec<Vec<T>>, dimension: usize) {
        self.features = Some(features);
        self.feature_dimension = Some(dimension);
    }

    /// Get a reference to the distance metric.
    pub fn distance_metric(&self) -> &D {
        &self.distance_metric
    }
}

impl<T, D> BaseMatcherValidation<T> for MatcherCommon<T, D>
where
    T: Clone + Send + Sync,
    D: DistanceMetric<T> + Send + Sync,
{
    fn get_features(&self) -> Option<&[Vec<T>]> {
        self.features.as_deref()
    }

    fn get_feature_dimension(&self) -> Option<usize> {
        self.feature_dimension
    }
}

/// A database index and its distance to the query.
type Candidate = (usize, f32);

/// Common algorithms for matching operations.
pub(crate) struct MatcherAlgorithms;

impl MatcherAlgorithms {
    #[inline]
    fn is_valid_distance(distance: f32) -> bool {
        distance.is_finite() && distance >= 0.0
    }

    #[inline]
    fn compare_distance_index(left: &(usize, f32), right: &(usize, f32)) -> std::cmp::Ordering {
        let distance_order = if left.1 == right.1 {
            std::cmp::Ordering::Equal
        } else {
            left.1.total_cmp(&right.1)
        };
        distance_order.then_with(|| left.0.cmp(&right.0))
    }

    /// Keep the best `k` distances in deterministic `(distance, index)` order.
    pub(crate) fn retain_top_k_distances(distances: &mut Vec<(usize, f32)>, k: usize) {
        let k = k.min(distances.len());
        if k == 0 {
            distances.clear();
        } else if k < distances.len() / 4 {
            distances.select_nth_unstable_by(k - 1, Self::compare_distance_index);
            distances.truncate(k);
            distances.sort_unstable_by(Self::compare_distance_index);
        } else {
            distances.sort_unstable_by(Self::compare_distance_index);
            distances.truncate(k);
        }
    }

    /// Sort and truncate matches by distance.
    ///
    /// Uses partial selection when k is small relative to the total count.
    /// For k < n/4, uses select_nth_unstable + sort which is O(n + k*log(k)).
    /// For larger k, uses full sort which is O(n*log(n)).
    pub fn sort_and_truncate_matches(mut distances: Vec<(usize, f32)>, k: usize) -> Vec<Match> {
        if distances.is_empty() {
            return Vec::new();
        }

        Self::retain_top_k_distances(&mut distances, k);

        distances
            .into_iter()
            .map(|(idx, distance)| Match::new_unchecked(idx, distance))
            .collect()
    }

    /// Compute distances to all features (brute force).
    pub fn compute_all_distances<'a, T, D>(
        query: &[T],
        features: impl ExactSizeIterator<Item = &'a [T]>,
        distance_metric: &D,
    ) -> Vec<(usize, f32)>
    where
        T: Clone + Send + Sync + 'a,
        D: DistanceMetric<T> + Send + Sync,
    {
        let capacity = features.len();
        distance_metric.fold_distances(
            query,
            features,
            Vec::with_capacity(capacity),
            |mut distances, index, distance| {
                if Self::is_valid_distance(distance) {
                    distances.push((index, distance));
                }
                distances
            },
        )
    }

    /// Find the 2 nearest neighbors with zero per-query allocation.
    ///
    /// Tracks best and second-best inline as it scans all features.
    /// Returns a `Vec<Match>` of length 2 (or 1 if only 1 feature, or 0 if empty),
    /// sorted by distance ascending (best first).
    pub fn find_2nn<'a, T, D>(
        query: &[T],
        features: impl IntoIterator<Item = &'a [T]>,
        distance_metric: &D,
    ) -> Vec<Match>
    where
        T: Clone + Send + Sync + 'a,
        D: DistanceMetric<T> + Send + Sync,
    {
        let (best, second) = distance_metric.fold_distances(
            query,
            features,
            (None, None),
            |(mut best, mut second): (Option<Candidate>, Option<Candidate>), index, distance| {
                if Self::is_valid_distance(distance) {
                    let candidate = (index, distance);
                    if best.as_ref().is_none_or(|current| {
                        Self::compare_distance_index(&candidate, current).is_lt()
                    }) {
                        second = best;
                        best = Some(candidate);
                    } else if second.as_ref().is_none_or(|current| {
                        Self::compare_distance_index(&candidate, current).is_lt()
                    }) {
                        second = Some(candidate);
                    }
                }
                (best, second)
            },
        );

        [best, second]
            .into_iter()
            .flatten()
            .map(|(index, distance)| Match::new_unchecked(index, distance))
            .collect()
    }
}

/// Direct matching utilities used by the brute-force matcher.
pub(crate) struct DirectMatchingUtils;

impl DirectMatchingUtils {
    /// Validate descriptors according to the configured metric.
    pub fn validate_descriptor_values<'a, T: 'a, D>(
        queries: impl IntoIterator<Item = &'a [T]>,
        database: impl IntoIterator<Item = &'a [T]>,
        distance_metric: &D,
    ) -> Result<(), MatcherError>
    where
        D: DistanceMetric<T>,
    {
        for (index, descriptor) in database.into_iter().enumerate() {
            if !distance_metric.is_valid_descriptor(descriptor) {
                return Err(MatcherError::InvalidDescriptor { index });
            }
        }
        if queries
            .into_iter()
            .any(|query| !distance_metric.is_valid_descriptor(query))
        {
            return Err(MatcherError::InvalidQuery);
        }
        Ok(())
    }

    /// Process single query against database directly.
    pub fn process_direct_query<'a, T, D>(
        query: &[T],
        database: impl ExactSizeIterator<Item = &'a [T]>,
        k: usize,
        distance_metric: &D,
    ) -> Vec<Match>
    where
        T: Clone + Send + Sync + 'a,
        D: DistanceMetric<T> + Send + Sync,
    {
        if k == 2 {
            return MatcherAlgorithms::find_2nn(query, database, distance_metric);
        }
        let distances = MatcherAlgorithms::compute_all_distances(query, database, distance_metric);
        MatcherAlgorithms::sort_and_truncate_matches(distances, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::distance::L2Distance;

    #[test]
    fn test_matcher_common() {
        let mut common = MatcherCommon::new(L2Distance::new());

        assert!(common.get_features().is_none());
        assert_eq!(common.indexed_descriptor_count(), 0);
        assert_eq!(common.feature_dimension(), None);

        let features = vec![vec![1.0f32, 2.0, 3.0], vec![4.0f32, 5.0, 6.0]];

        let dimension = common
            .validate_build_features(&crate::matching::matchers::as_slices(&features))
            .unwrap();
        common.store_features(features, dimension);

        assert!(common.get_features().is_some());
        assert_eq!(common.indexed_descriptor_count(), 2);
        assert_eq!(common.feature_dimension(), Some(3));
    }

    #[test]
    fn test_validation_methods() {
        let mut common = MatcherCommon::new(L2Distance::new());
        let features = vec![vec![1.0f32, 2.0, 3.0]];
        let dimension = common
            .validate_build_features(&crate::matching::matchers::as_slices(&features))
            .unwrap();
        common.store_features(features, dimension);

        // Valid query
        let query = vec![1.0f32, 2.0, 3.0];
        assert!(common.validate_query(&query).is_ok());

        // Invalid query dimensions
        let bad_query = vec![1.0f32, 2.0];
        assert!(matches!(
            common.validate_query(&bad_query),
            Err(MatcherError::DimensionMismatch { .. })
        ));

        // Valid k
        assert!(common.validate_k(1).is_ok());

        // Invalid k
        assert!(matches!(
            common.validate_k(0),
            Err(MatcherError::InvalidK { .. })
        ));
        assert!(matches!(
            common.validate_k(2),
            Err(MatcherError::InvalidK { .. })
        ));
    }

    #[test]
    fn test_matcher_algorithms() {
        let features = [
            vec![1.0f32, 2.0, 3.0],
            vec![4.0f32, 5.0, 6.0],
            vec![7.0f32, 8.0, 9.0],
        ];
        let query = vec![1.0f32, 2.0, 3.0];
        let distance_metric = L2Distance::new();

        // Test compute_all_distances
        let distances = MatcherAlgorithms::compute_all_distances(
            &query,
            features.iter().map(Vec::as_slice),
            &distance_metric,
        );
        assert_eq!(distances.len(), 3);
        assert_eq!(distances[0].0, 0); // Index of first feature
        assert!(distances[0].1 < 0.001); // Should be very close to 0 (exact match)

        // Test sort_and_truncate_matches
        let matches = MatcherAlgorithms::sort_and_truncate_matches(distances, 2);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].index, 0); // Best match should be index 0
        assert!(matches[0].distance < 0.001);
    }

    #[test]
    fn top_k_ordering_and_ties_are_deterministic() {
        let mut distances = vec![(3, 1.0), (1, 1.0), (2, 0.5), (0, 1.0)];
        MatcherAlgorithms::retain_top_k_distances(&mut distances, 3);

        assert_eq!(distances, vec![(2, 0.5), (0, 1.0), (1, 1.0)]);

        let mut zeros = vec![(2, -0.0), (1, 0.0), (0, 0.0)];
        MatcherAlgorithms::retain_top_k_distances(&mut zeros, 2);
        assert_eq!(
            zeros.iter().map(|&(index, _)| index).collect::<Vec<_>>(),
            [0, 1]
        );
    }

    #[test]
    fn non_finite_distances_are_never_ranked() {
        let features = [vec![f32::NAN], vec![2.0], vec![1.0]];
        let query = vec![0.0];
        let metric = L2Distance::new();

        let all = MatcherAlgorithms::compute_all_distances(
            &query,
            features.iter().map(Vec::as_slice),
            &metric,
        );
        let matches = MatcherAlgorithms::sort_and_truncate_matches(all, 3);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].index, 2);
        assert_eq!(matches[1].index, 1);

        let direct = DirectMatchingUtils::process_direct_query(
            &query,
            features.iter().map(Vec::as_slice),
            3,
            &metric,
        );
        assert_eq!(direct, matches);
    }
}
