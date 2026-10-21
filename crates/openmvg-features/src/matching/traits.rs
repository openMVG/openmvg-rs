// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core traits for the feature matching system.

use crate::matching::{Match, MatcherError};

/// Trait for computing distances between feature vectors.
///
/// Implementations must be deterministic and return a finite, non-negative
/// value for any equal-length descriptors accepted by
/// [`Self::is_valid_descriptor`].
pub trait DistanceMetric<T> {
    /// Compute the distance between two equal-length feature vectors.
    ///
    /// Lower is more similar.
    fn distance(&self, a: &[T], b: &[T]) -> f32;

    /// Return whether a descriptor is valid for this metric.
    ///
    /// Matchers call this before distance evaluation so malformed descriptors
    /// produce [`MatcherError`] instead of a panic or silently disappearing
    /// from a result. Metrics whose element type has no invalid values can use
    /// the default implementation.
    fn is_valid_descriptor(&self, _descriptor: &[T]) -> bool {
        true
    }

    /// Fold one query's distances without prescribing result storage.
    ///
    /// The default calls [`Self::distance`] once per candidate. Built-in
    /// metrics may override this operation-scoped hook to amortize safe
    /// runtime SIMD dispatch across a complete database scan. Matchers use it
    /// internally; ordinary metric implementations need only provide
    /// [`Self::distance`].
    #[doc(hidden)]
    fn fold_distances<'a, I, R, F>(&self, query: &[T], candidates: I, initial: R, mut fold: F) -> R
    where
        Self: Sized,
        T: 'a,
        I: IntoIterator<Item = &'a [T]>,
        F: FnMut(R, usize, f32) -> R,
    {
        candidates
            .into_iter()
            .enumerate()
            .fold(initial, |state, (index, candidate)| {
                fold(state, index, self.distance(query, candidate))
            })
    }
}

/// Trait for feature matching implementations.
pub trait FeatureMatcher<T> {
    /// Build the matcher from borrowed descriptor slices.
    ///
    /// Owned descriptors are passed as `iter().map(Vec::as_slice)`; indexes
    /// copy the values once into their own storage.
    fn build(&mut self, features: &[&[T]]) -> Result<(), MatcherError>;

    /// Query up to `k` database descriptors, nearest first.
    ///
    /// Exact matchers return exactly `k` candidates. Approximate or filtered
    /// indexes may return fewer. Distances are expressed in the selected
    /// metric's units and must be finite and non-negative.
    fn query_single(&self, query: &[T], k: usize) -> Result<Vec<Match>, MatcherError>;

    /// Query a batch of descriptor slices, returning up to `k` candidates per query.
    ///
    /// A built matcher returns an empty result for an empty batch, independent
    /// of `k`. Non-empty batches require `k` between 1 and the number of indexed descriptors.
    fn query_batch(&self, queries: &[&[T]], k: usize) -> Result<Vec<Vec<Match>>, MatcherError>;
}
