// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HNSW (Hierarchical Navigable Small World) feature matcher.
//!
//! Its teaching docs are on [`HnswMatcher`], where docs.rs shows them.

mod graph;

use crate::matching::distance::SquaredL2Distance;
use crate::matching::traits::DistanceMetric;
use crate::matching::types::{Match, MatcherError};
use graph::{HnswGraph, MIN_EF_CONSTRUCTION, SearchScratch};
use openmvg_core::RunSeed;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const DEFAULT_HNSW_SEED: RunSeed = RunSeed::from_u64(0x4f50_454e_4853_4e57);

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the HNSW matcher.
///
/// Graph connectivity is fixed: 12 links per descriptor on the upper layers and 24 on layer 0.
/// The two beam widths below trade build and query time for recall.
///
/// | Parameter         | Default | [`high_recall`](Self::high_recall) |
/// |-------------------|---------|------------------------------------|
/// | `ef_construction` | 24      | 48                                 |
/// | `ef_search`       | 24      | 36                                 |
///
/// [`balanced`](Self::balanced) returns the default.
///
/// ```
/// use openmvg_features::matching::{HnswConfig, RunSeed};
///
/// let config = HnswConfig::default().with_seed(RunSeed::from_u64(7));
/// assert_eq!(config.seed(), RunSeed::from_u64(7));
/// ```
#[derive(Debug, Clone)]
pub struct HnswConfig {
    ef_construction: usize,
    ef_search: usize,
    seed: RunSeed,
}

config_accessors!(HnswConfig {
    /// Number of candidates explored during graph construction.
    copy ef_construction, with_ef_construction: usize;
    /// Search beam width: candidates explored during queries.
    /// It is raised to `k` for calls requesting more neighbors.
    copy ef_search, with_ef_search: usize;
    /// [`RunSeed`] that draws the random layer of each descriptor, so one descriptor set and
    /// one seed always give the same graph.
    copy seed, with_seed: RunSeed;
});

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            ef_construction: 24,
            ef_search: 24,
            seed: DEFAULT_HNSW_SEED,
        }
    }
}

impl HnswConfig {
    /// Balanced default construction/query policy.
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Higher-recall policy for cases where build time is secondary.
    pub fn high_recall() -> Self {
        Self {
            ef_construction: 48,
            ef_search: 36,
            ..Self::default()
        }
    }
}

// ============================================================================
// HnswMatcher
// ============================================================================

/// Approximate nearest neighbor matcher over a Hierarchical Navigable Small World (HNSW) graph,
/// the default matcher of `openmvg-sfm`.
///
/// Build one index per image with [`build`](Self::build), then find each query descriptor's
/// two nearest neighbors with [`query_batch`](Self::query_batch). Turning those candidates into
/// matches is [`MatchAcceptance`](crate::matching::MatchAcceptance)'s job; the
/// [Examples section of the `matching` module](crate::matching#examples) does both.
///
/// # How it works
///
/// Reference: Y. A. Malkov, D. A. Yashunin, "Efficient and robust approximate nearest neighbor
/// search using Hierarchical Navigable Small World graphs", IEEE TPAMI, 2020,
/// doi:10.1109/TPAMI.2018.2889473.
///
/// ```text
/// ┌────────────────────────────────────────────────────────────────────────────┐
/// │                          HNSW matcher (approximate)                        │
/// └────────────────────────────────────────────────────────────────────────────┘
/// Input: database of n descriptors of dimension d (f32 or u8), query descriptors,
///        k, ef_search; distances are the metric's units (squared L2 by default)
///   │
///   ▼
/// ┌────────────────────────────────────────────────────────────────────────────┐
/// │ Build: for descriptor i = 0, 1, ..., n-1, in input order                   │
/// └────────────────────────────────────────────────────────────────────────────┘
///   level(i) = ⌊−ln(u) / ln 12⌋ capped at 32, u ∈ (0,1) from the seeded RNG
///   greedy descent (ef = 1) from the entry point down to layer level(i)+1
///   on layers level(i)..0: beam search with ef_construction candidates, keep
///     12 links chosen by the diversity heuristic, add reverse links, and
///     prune a full list (24 links on layer 0, 12 above) with the same heuristic
///   a node whose level exceeds the current maximum becomes the entry point
///   │
///   ▼
/// ┌────────────────────────────────────────────────────────────────────────────┐
/// │ Query: for each query descriptor q                                         │
/// └────────────────────────────────────────────────────────────────────────────┘
///   greedy descent (ef = 1) from the entry point through layers max..1
///   layer 0: beam search with ef = min(max(ef_search, k), n)
///   sort candidates by (distance, index) and keep the first k
///   │
///   ▼
/// Output: per query, k `Match` values (index, distance), nearest first
/// ```
///
/// # Algorithm
///
/// The index is a stack of proximity graphs over the same descriptors. Every descriptor lives on
/// layer 0. A descriptor also lives on the layers up to its random level, so each higher layer
/// holds about a twelfth of the descriptors below it. A search enters at the top, moves greedily
/// to the closest node of each sparse layer, and spends its effort on layer 0, where a best-first
/// beam of `ef` candidates expands neighbors until the closest unexpanded candidate is farther
/// than the worst of the `ef` best found. `ef` is the beam width: `ef_construction` while
/// building, `ef_search` while querying. A wider beam finds more true neighbors and costs more.
///
/// The matcher returns the `k` nearest descriptors of each query. The ratio test and the
/// reciprocal check that turn those candidates into correspondences are
/// [`MatchAcceptance`](crate::matching::MatchAcceptance), applied by the caller.
///
/// Determinism follows from the code, not from a tolerance:
///
/// - Insertion is sequential in input order and the levels come from [`RunSeed`], so one
///   descriptor set and one seed give one graph.
/// - Candidates are ordered by distance and then by descriptor index, so equal distances
///   never depend on heap or hash order.
/// - `query_batch` is sequential. `query_all_stored_with_ef` runs the queries in parallel
///   with the `parallel` feature, and each result is independent of the others.
///
/// # Failure modes
///
/// The search is approximate: the result can miss a true neighbor, and a larger `ef_search`
/// lowers that risk at a higher query cost. `build` returns [`MatcherError::EmptyFeatures`] for an
/// empty database or zero dimension, [`MatcherError::DimensionMismatch`] for ragged rows,
/// [`MatcherError::InvalidDescriptor`] for a descriptor the metric rejects (non-finite or
/// overflowing float values, or a `u8` descriptor too long for the metric), and
/// [`MatcherError::IndexTooLarge`] at `u32::MAX` descriptors. A failed build keeps the previous
/// index. A query returns [`MatcherError::NotBuilt`] before a build, and [`MatcherError::InvalidK`]
/// when `k` is zero or exceeds the descriptor count.
///
/// # Alternatives
///
/// [`BruteForceMatcher`](crate::matching::BruteForceMatcher) returns exact neighbors with a cost
/// linear in the database size. HNSW stores a graph with up to 24 links per descriptor on layer 0
/// besides the descriptors, and needs a build step per database before the first query.
///
/// # Memory
///
/// Besides a copy of the descriptors, each descriptor costs 96 bytes of layer-0 links, plus a
/// few for the sparse upper layers: a `u8` index takes about twice the memory of its
/// descriptors, and an `f32` one a little more than its descriptors.
/// [`index_memory_bytes`](Self::index_memory_bytes) reports the actual figure.
///
/// # Usage
///
/// The matcher is generic over:
/// - `T` — Element type (`f32` or `u8`)
/// - `Met` — Slice distance metric implementing [`DistanceMetric<T>`](crate::matching::DistanceMetric)
///   (default: [`SquaredL2Distance`](crate::matching::SquaredL2Distance))
///
/// ```rust
/// use openmvg_features::matching::{HnswConfig, HnswMatcher};
/// use openmvg_features::matching::{MatcherError, SquaredL2DistanceU8};
///
/// let config = HnswConfig::default();
/// let mut matcher = HnswMatcher::<u8, SquaredL2DistanceU8>::with_config(config);
/// let database = [[0_u8; 128], [100; 128]];
/// let database: Vec<&[u8]> = database.iter().map(|d| d.as_slice()).collect();
/// matcher.build(&database)?;
/// let neighbors = matcher.query_batch(&[&[10_u8; 128]], 2)?;
/// assert_eq!(neighbors[0][0].index(), 0); // nearest first
/// # Ok::<(), MatcherError>(())
/// ```
///
/// # Type parameters
///
/// - `T` — Descriptor element type (e.g., `f32` or `u8`). Descriptors are passed as `&[T]`
///   slices.
/// - `Met` — Distance metric implementing [`DistanceMetric<T>`]. Returned [`Match`] distances
///   are in the metric's units.
pub struct HnswMatcher<T, Met = SquaredL2Distance>
where
    T: Copy + Send + Sync + 'static,
    Met: DistanceMetric<T> + Default + Send + Sync,
{
    config: HnswConfig,
    feature_dimension: Option<usize>,
    descriptor_count: usize,
    index: Option<HnswGraph<T, Met>>,
}

impl<T, Met> std::fmt::Debug for HnswMatcher<T, Met>
where
    T: Copy + Send + Sync + 'static,
    Met: DistanceMetric<T> + Default + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswMatcher")
            .field("config", &self.config)
            .field("feature_dimension", &self.feature_dimension)
            .field("descriptor_count", &self.descriptor_count)
            .field("index_built", &self.index.is_some())
            .finish()
    }
}

impl<T, Met> HnswMatcher<T, Met>
where
    T: Copy + Send + Sync + 'static,
    Met: DistanceMetric<T> + Default + Send + Sync,
{
    /// Create a new HNSW matcher with default configuration.
    pub fn new() -> Self {
        Self::with_config(HnswConfig::default())
    }

    /// Create a new HNSW matcher with custom configuration.
    pub fn with_config(mut config: HnswConfig) -> Self {
        config.ef_construction = config.ef_construction.max(MIN_EF_CONSTRUCTION);
        config.ef_search = config.ef_search.max(1);
        Self {
            config,
            feature_dimension: None,
            descriptor_count: 0,
            index: None,
        }
    }

    /// Return the normalized index configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Number of descriptors in the current database.
    pub fn indexed_descriptor_count(&self) -> usize {
        self.descriptor_count
    }

    /// Descriptor dimension, or `None` before a successful build.
    pub fn feature_dimension(&self) -> Option<usize> {
        self.feature_dimension
    }

    /// Return whether a searchable index has been built.
    pub fn is_built(&self) -> bool {
        self.index.is_some()
    }

    /// Capacity-based bytes used by descriptors, nodes, and adjacency blocks.
    pub fn index_memory_bytes(&self) -> usize {
        self.index.as_ref().map_or(0, HnswGraph::memory_bytes)
    }

    /// Build the index from borrowed descriptor slices, all of one length.
    ///
    /// The descriptors are copied into the index, so the slices need not outlive it.
    pub fn build(&mut self, features: &[&[T]]) -> Result<(), MatcherError> {
        let (n, dim) = Self::validate_features(features.iter().copied())?;
        let mut index = self.empty_index(dim, n);
        let mut scratch = SearchScratch::default();
        for &slice in features {
            index.insert(slice, &mut scratch);
        }
        self.finish_index(index, n, dim);
        Ok(())
    }

    /// Query k nearest neighbors for a single feature.
    pub fn query_single(&self, query: &[T], k: usize) -> Result<Vec<Match>, MatcherError> {
        self.query_single_with_ef(query, k, self.config.ef_search)
    }

    /// Query one descriptor with an explicit search beam.
    pub fn query_single_with_ef(
        &self,
        query: &[T],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Match>, MatcherError> {
        let index = self.index.as_ref().ok_or(MatcherError::NotBuilt)?;
        self.validate_query(query)?;
        self.validate_k(k)?;

        let ef_search = ef_search.max(k).min(self.descriptor_count);

        let mut scratch = SearchScratch::default();
        let matches = index
            .search(query, k, ef_search, &mut scratch)
            .iter()
            .map(|candidate| {
                Match::new_unchecked(candidate.index, f32::from_bits(candidate.distance))
            })
            .collect();

        Ok(matches)
    }

    /// Query using one descriptor already stored in the index.
    ///
    /// It saves keeping a second copy of the descriptors to query an image against itself.
    pub fn query_stored_with_ef(
        &self,
        descriptor_index: usize,
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Match>, MatcherError> {
        let mut scratch = SearchScratch::default();
        self.query_stored_with_scratch(descriptor_index, k, ef_search, &mut scratch)
    }

    /// Query every stored descriptor while reusing one search workspace per worker.
    pub fn query_all_stored_with_ef(
        &self,
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Vec<Match>>, MatcherError> {
        self.index.as_ref().ok_or(MatcherError::NotBuilt)?;
        self.validate_k(k)?;
        #[cfg(feature = "parallel")]
        {
            (0..self.descriptor_count)
                .into_par_iter()
                .map_init(SearchScratch::default, |scratch, descriptor_index| {
                    self.query_stored_with_scratch(descriptor_index, k, ef_search, scratch)
                })
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut scratch = SearchScratch::default();
            (0..self.descriptor_count)
                .map(|descriptor_index| {
                    self.query_stored_with_scratch(descriptor_index, k, ef_search, &mut scratch)
                })
                .collect()
        }
    }

    fn query_stored_with_scratch(
        &self,
        descriptor_index: usize,
        k: usize,
        ef_search: usize,
        scratch: &mut SearchScratch,
    ) -> Result<Vec<Match>, MatcherError> {
        let index = self.index.as_ref().ok_or(MatcherError::NotBuilt)?;
        if descriptor_index >= self.descriptor_count {
            return Err(MatcherError::DescriptorIndexOutOfRange {
                index: descriptor_index,
                count: self.descriptor_count,
            });
        }
        self.validate_k(k)?;
        let ef_search = ef_search.max(k).min(self.descriptor_count);
        Ok(index
            .search(index.descriptor(descriptor_index), k, ef_search, scratch)
            .iter()
            .map(|candidate| {
                Match::new_unchecked(candidate.index, f32::from_bits(candidate.distance))
            })
            .collect())
    }

    /// Query k nearest neighbors for multiple feature slices.
    ///
    /// The queries run one after another, so several images can be matched in parallel.
    pub fn query_batch(&self, queries: &[&[T]], k: usize) -> Result<Vec<Vec<Match>>, MatcherError> {
        self.query_batch_with_ef(queries, k, self.config.ef_search)
    }

    /// Query borrowed descriptors with an explicit search beam.
    pub fn query_batch_with_ef(
        &self,
        queries: &[&[T]],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Vec<Match>>, MatcherError> {
        let index = self.index.as_ref().ok_or(MatcherError::NotBuilt)?;

        if queries.is_empty() {
            return Ok(Vec::new());
        }
        self.validate_k(k)?;

        for &query in queries {
            self.validate_query(query)?;
        }

        let ef_search = ef_search.max(k).min(self.descriptor_count);

        let mut scratch = SearchScratch::default();
        let results = queries
            .iter()
            .map(|query| {
                index
                    .search(query, k, ef_search, &mut scratch)
                    .iter()
                    .map(|candidate| {
                        Match::new_unchecked(candidate.index, f32::from_bits(candidate.distance))
                    })
                    .collect()
            })
            .collect();

        Ok(results)
    }

    // ── Validation helpers ──

    fn validate_features<'a>(
        features: impl IntoIterator<Item = &'a [T]>,
    ) -> Result<(usize, usize), MatcherError>
    where
        T: 'a,
    {
        let metric = Met::default();
        let mut features = features.into_iter();
        let Some(first) = features.next() else {
            return Err(MatcherError::EmptyFeatures);
        };
        let dim = first.len();
        if dim == 0 {
            return Err(MatcherError::EmptyFeatures);
        }
        if !metric.is_valid_descriptor(first) {
            return Err(MatcherError::InvalidDescriptor { index: 0 });
        }
        let mut count = 1;
        for f in features {
            if f.len() != dim {
                return Err(MatcherError::DimensionMismatch {
                    expected: dim,
                    actual: f.len(),
                });
            }
            if !metric.is_valid_descriptor(f) {
                return Err(MatcherError::InvalidDescriptor { index: count });
            }
            count += 1;
        }
        if count >= u32::MAX as usize {
            return Err(MatcherError::IndexTooLarge { count });
        }
        Ok((count, dim))
    }

    fn empty_index(&self, dimension: usize, features: usize) -> HnswGraph<T, Met> {
        let mut index = HnswGraph::new(
            dimension,
            self.config.ef_construction,
            self.config.seed,
            Met::default(),
        );
        index.reserve(features);
        index
    }

    fn finish_index(&mut self, mut index: HnswGraph<T, Met>, count: usize, dimension: usize) {
        index.finish_build();
        self.feature_dimension = Some(dimension);
        self.descriptor_count = count;
        self.index = Some(index);
    }

    fn validate_query(&self, query: &[T]) -> Result<(), MatcherError> {
        if query.is_empty() {
            return Err(MatcherError::EmptyQuery);
        }
        if !Met::default().is_valid_descriptor(query) {
            return Err(MatcherError::InvalidQuery);
        }
        if let Some(dim) = self.feature_dimension
            && query.len() != dim
        {
            return Err(MatcherError::DimensionMismatch {
                expected: dim,
                actual: query.len(),
            });
        }
        Ok(())
    }

    fn validate_k(&self, k: usize) -> Result<(), MatcherError> {
        if k == 0 || k > self.descriptor_count {
            return Err(MatcherError::InvalidK { k });
        }
        Ok(())
    }
}

impl<T, Met> Default for HnswMatcher<T, Met>
where
    T: Copy + Send + Sync + 'static,
    Met: DistanceMetric<T> + Default + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// FeatureMatcher trait implementation
// ============================================================================

impl<T, Met> crate::matching::traits::FeatureMatcher<T> for HnswMatcher<T, Met>
where
    T: Copy + Send + Sync + 'static,
    Met: DistanceMetric<T> + Default + Send + Sync,
{
    fn build(&mut self, features: &[&[T]]) -> Result<(), MatcherError> {
        HnswMatcher::build(self, features)
    }

    fn query_single(&self, query: &[T], k: usize) -> Result<Vec<Match>, MatcherError> {
        HnswMatcher::query_single(self, query, k)
    }

    fn query_batch(&self, queries: &[&[T]], k: usize) -> Result<Vec<Vec<Match>>, MatcherError> {
        HnswMatcher::query_batch(self, queries, k)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::distance::SquaredL2DistanceU8;
    use crate::matching::matchers::as_slices;

    #[test]
    fn test_hnsw_matcher_build() {
        let mut matcher = HnswMatcher::<f32, SquaredL2Distance>::new();
        let features: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..128).map(|j| (i * 128 + j) as f32).collect())
            .collect();

        assert!(matcher.build(&as_slices(&features)).is_ok());
        assert_eq!(matcher.indexed_descriptor_count(), 100);
        assert_eq!(matcher.feature_dimension(), Some(128));
        assert!(matcher.is_built());
    }

    #[test]
    fn test_hnsw_matcher_empty_features() {
        let mut matcher = HnswMatcher::<f32, SquaredL2Distance>::new();
        assert!(matches!(
            matcher.build(&[]),
            Err(MatcherError::EmptyFeatures)
        ));
    }

    #[test]
    fn oversized_u8_descriptors_are_rejected_before_distance_evaluation() {
        let oversized = vec![0u8; crate::matching::distance::MAX_SQUARED_L2_U8_DIMENSION + 1];
        let mut matcher = HnswMatcher::<u8, SquaredL2DistanceU8>::new();

        assert!(matches!(
            matcher.build(&[oversized.as_slice()]),
            Err(MatcherError::InvalidDescriptor { index: 0 })
        ));
    }

    #[test]
    fn overflowing_float_descriptors_are_rejected_before_indexing() {
        let mut matcher = HnswMatcher::<f32, SquaredL2Distance>::new();

        assert!(matches!(
            matcher.build(&[&[f32::MAX]]),
            Err(MatcherError::InvalidDescriptor { index: 0 })
        ));
    }

    #[test]
    fn test_hnsw_matcher_dimension_mismatch() {
        let mut matcher = HnswMatcher::<f32, SquaredL2Distance>::new();
        let features = vec![vec![1.0f32, 2.0, 3.0], vec![4.0f32, 5.0]];
        assert!(matches!(
            matcher.build(&as_slices(&features)),
            Err(MatcherError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_hnsw_matcher_query_single_exact() {
        let config = HnswConfig {
            ef_construction: 200,
            ef_search: 32,
            ..HnswConfig::default()
        };
        let mut matcher = HnswMatcher::<f32, SquaredL2Distance>::with_config(config);

        let features: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 * 100.0; 8]).collect();
        matcher.build(&as_slices(&features)).unwrap();

        let query = &features[0];
        let results = matcher.query_single(query, 1).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].index, 0);
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn test_hnsw_matcher_query_batch() {
        let config = HnswConfig {
            ef_construction: 200,
            ef_search: 32,
            ..HnswConfig::default()
        };
        let mut matcher = HnswMatcher::<f32, SquaredL2Distance>::with_config(config);

        let features: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 * 100.0; 8]).collect();
        matcher.build(&as_slices(&features)).unwrap();

        let queries = [
            features[0].as_slice(),
            features[10].as_slice(),
            features[25].as_slice(),
        ];
        let results = matcher.query_batch(&queries, 2).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0][0].index, 0);
        assert_eq!(results[1][0].index, 10);
        assert_eq!(results[2][0].index, 25);
    }

    #[test]
    fn test_hnsw_matcher_not_built_error() {
        let matcher = HnswMatcher::<f32, SquaredL2Distance>::new();
        let query = vec![1.0f32, 2.0, 3.0];
        assert!(matches!(
            matcher.query_single(&query, 1),
            Err(MatcherError::NotBuilt)
        ));
    }

    #[test]
    fn test_hnsw_matcher_invalid_k() {
        let mut matcher = HnswMatcher::<f32, SquaredL2Distance>::new();
        matcher.build(&[&[1.0f32, 2.0, 3.0]]).unwrap();
        assert!(matches!(
            matcher.query_single(&[1.0f32, 2.0, 3.0], 0),
            Err(MatcherError::InvalidK { .. })
        ));
        assert!(matches!(
            matcher.query_single(&[1.0f32, 2.0, 3.0], 2),
            Err(MatcherError::InvalidK { .. })
        ));
    }

    #[test]
    fn test_hnsw_u8_matcher() {
        let config = HnswConfig {
            ef_construction: 200,
            ef_search: 32,
            ..HnswConfig::default()
        };
        let mut matcher = HnswMatcher::<u8, SquaredL2DistanceU8>::with_config(config);

        let features: Vec<Vec<u8>> = (0..100).map(|i| vec![(i * 2) as u8; 128]).collect();
        matcher.build(&as_slices(&features)).unwrap();

        let query = &features[0];
        let results = matcher.query_single(query, 1).unwrap();
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_hnsw_build_and_query_slices() {
        let config = HnswConfig {
            ef_construction: 200,
            ef_search: 32,
            ..HnswConfig::default()
        };
        let mut matcher = HnswMatcher::<u8, SquaredL2DistanceU8>::with_config(config);

        let raw_data: Vec<Vec<u8>> = (0..100).map(|i| vec![(i * 2) as u8; 128]).collect();
        let slices: Vec<&[u8]> = raw_data.iter().map(|v| v.as_slice()).collect();

        matcher.build(&slices).unwrap();
        assert_eq!(matcher.indexed_descriptor_count(), 100);

        let query_slices: Vec<&[u8]> = vec![raw_data[0].as_slice(), raw_data[25].as_slice()];
        let results = matcher.query_batch(&query_slices, 1).unwrap();
        assert_eq!(results[0][0].index, 0);
        assert_eq!(results[1][0].index, 25);
    }

    #[test]
    fn test_hnsw_default_config() {
        let config = HnswConfig::default();
        assert_eq!(config.ef_construction, 24);
        assert_eq!(config.ef_search, 24);
        assert_eq!(config.seed, DEFAULT_HNSW_SEED);
    }

    #[test]
    fn undersized_beams_are_normalized() {
        let matcher = HnswMatcher::<u8, SquaredL2DistanceU8>::with_config(HnswConfig {
            ef_construction: 0,
            ef_search: 0,
            ..HnswConfig::default()
        });
        assert_eq!(matcher.config().ef_construction, MIN_EF_CONSTRUCTION);
        assert_eq!(matcher.config().ef_search, 1);
    }

    #[test]
    fn test_hnsw_large_dataset_recall() {
        let n = 1000;
        let dim = 32;
        let k = 5;

        let config = HnswConfig {
            ef_construction: 200,
            ef_search: 100,
            ..HnswConfig::default()
        };
        let mut matcher = HnswMatcher::<f32, SquaredL2Distance>::with_config(config);

        let features: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32; dim]).collect();
        matcher.build(&as_slices(&features)).unwrap();

        let query = features[500].clone();
        let results = matcher.query_single(&query, k).unwrap();

        assert_eq!(results.len(), k);
        assert_eq!(results[0].index, 500);
        assert!(results[0].distance < 1e-6);

        for m in &results {
            assert!((m.index as isize - 500).unsigned_abs() <= k);
        }
    }

    #[test]
    fn test_hnsw_rebuild() {
        let mut matcher = HnswMatcher::<f32, SquaredL2Distance>::new();
        matcher
            .build(&[&[1.0f32, 2.0, 3.0], &[4.0, 5.0, 6.0]])
            .unwrap();
        assert_eq!(matcher.indexed_descriptor_count(), 2);

        let features2: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 3]).collect();
        matcher.build(&as_slices(&features2)).unwrap();
        assert_eq!(matcher.indexed_descriptor_count(), 10);
    }

    #[test]
    fn failed_rebuild_preserves_the_previous_index() {
        let mut matcher = HnswMatcher::<f32, SquaredL2Distance>::new();
        matcher.build(&[&[0.0f32, 0.0], &[2.0, 2.0]]).unwrap();
        assert!(matches!(
            matcher.build(&[&[f32::NAN, 0.0]]),
            Err(MatcherError::InvalidDescriptor { index: 0 })
        ));

        let matches = matcher.query_single(&[0.1, 0.1], 1).unwrap();
        assert_eq!(matches[0].index, 0);
    }

    #[test]
    fn test_hnsw_build_is_deterministic() {
        let features: Vec<Vec<u8>> = (0..256)
            .map(|i| (0..64).map(|j| ((i * 17 + j * 29) & 0xff) as u8).collect())
            .collect();
        let queries = &as_slices(&features[40..80]);
        let mut first = HnswMatcher::<u8, SquaredL2DistanceU8>::new();
        let mut second = HnswMatcher::<u8, SquaredL2DistanceU8>::new();
        first.build(&as_slices(&features)).unwrap();
        second.build(&as_slices(&features)).unwrap();

        assert_eq!(
            first.query_batch(queries, 5).unwrap(),
            second.query_batch(queries, 5).unwrap()
        );
        assert_eq!(first.index_memory_bytes(), second.index_memory_bytes());
        assert!(first.index.as_ref().unwrap().validate());
        assert!(second.index.as_ref().unwrap().validate());
    }

    #[test]
    fn test_hnsw_u8_and_f32_share_graph_results() {
        let u8_features: Vec<Vec<u8>> = (0..200)
            .map(|i| (0..32).map(|j| ((i * 7 + j * 11) & 0xff) as u8).collect())
            .collect();
        let f32_features: Vec<Vec<f32>> = u8_features
            .iter()
            .map(|feature| feature.iter().map(|&value| f32::from(value)).collect())
            .collect();
        let mut u8_matcher = HnswMatcher::<u8, SquaredL2DistanceU8>::new();
        let mut f32_matcher = HnswMatcher::<f32, SquaredL2Distance>::new();
        u8_matcher.build(&as_slices(&u8_features)).unwrap();
        f32_matcher.build(&as_slices(&f32_features)).unwrap();

        for query_index in [0, 17, 83, 199] {
            let u8_results = u8_matcher
                .query_single(&u8_features[query_index], 5)
                .unwrap();
            let f32_results = f32_matcher
                .query_single(&f32_features[query_index], 5)
                .unwrap();
            assert_eq!(
                u8_results
                    .iter()
                    .map(|result| result.index)
                    .collect::<Vec<_>>(),
                f32_results
                    .iter()
                    .map(|result| result.index)
                    .collect::<Vec<_>>()
            );
        }
    }
}
