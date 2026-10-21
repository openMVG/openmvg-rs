// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core data structures and error types for feature matching.

/// A nearest-neighbor candidate for one query descriptor.
///
/// Matchers produce these values; [`Match::try_new`] builds one from
/// caller-supplied data and checks the distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match {
    pub(crate) index: usize,
    pub(crate) distance: f32,
}

impl Match {
    /// Create a candidate after checking that `distance` is finite and
    /// non-negative.
    ///
    /// # Errors
    ///
    /// Returns [`MatcherError::InvalidMatchDistance`] when `distance` is NaN,
    /// infinite, or negative.
    pub fn try_new(index: usize, distance: f32) -> Result<Self, MatcherError> {
        validate_match_distance(distance)?;
        Ok(Self { index, distance })
    }

    /// Unchecked constructor for the matchers' hot path, whose distances come
    /// from the metric itself.
    #[inline]
    pub(crate) const fn new_unchecked(index: usize, distance: f32) -> Self {
        Self { index, distance }
    }

    /// Index of the matched feature in the database.
    #[inline]
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Distance in the supplied metric's units (lower is better); finite and
    /// non-negative.
    #[inline]
    #[must_use]
    pub const fn distance(&self) -> f32 {
        self.distance
    }
}

/// A pair retained by a descriptor-matching filter.
///
/// Matchers and filters produce these values; [`FilteredMatch::try_new`] builds
/// one from caller-supplied data and checks the distance and ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilteredMatch {
    pub(crate) query_index: usize,
    pub(crate) target_index: usize,
    pub(crate) distance: f32,
    pub(crate) ratio: f32,
}

impl FilteredMatch {
    /// Create a filtered match after checking that `distance` is finite and
    /// non-negative and that `ratio` is finite and in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns [`MatcherError::InvalidMatchDistance`] for an invalid distance
    /// and [`MatcherError::InvalidMatchRatio`] for an invalid ratio.
    pub fn try_new(
        query_index: usize,
        target_index: usize,
        distance: f32,
        ratio: f32,
    ) -> Result<Self, MatcherError> {
        validate_match_distance(distance)?;
        if !(ratio.is_finite() && (0.0..=1.0).contains(&ratio)) {
            return Err(MatcherError::InvalidMatchRatio { value: ratio });
        }
        Ok(Self {
            query_index,
            target_index,
            distance,
            ratio,
        })
    }

    /// Unchecked constructor for the matchers' and filters' hot path, whose
    /// distances and ratios come from the metric and the ratio test.
    #[inline]
    pub(crate) const fn new_unchecked(
        query_index: usize,
        target_index: usize,
        distance: f32,
        ratio: f32,
    ) -> Self {
        Self {
            query_index,
            target_index,
            distance,
            ratio,
        }
    }

    /// Index of the query feature.
    #[inline]
    #[must_use]
    pub const fn query_index(&self) -> usize {
        self.query_index
    }

    /// Index of the matched database feature.
    #[inline]
    #[must_use]
    pub const fn target_index(&self) -> usize {
        self.target_index
    }

    /// Best-match distance in the matcher metric's units; finite and
    /// non-negative.
    #[inline]
    #[must_use]
    pub const fn distance(&self) -> f32 {
        self.distance
    }

    /// Ratio of best to second-best distance in ordinary distance units, in
    /// `[0, 1]`; the ratio test keeps it below its threshold.
    #[inline]
    #[must_use]
    pub const fn ratio(&self) -> f32 {
        self.ratio
    }
}

#[inline]
fn validate_match_distance(distance: f32) -> Result<(), MatcherError> {
    if distance.is_finite() && distance >= 0.0 {
        Ok(())
    } else {
        Err(MatcherError::InvalidMatchDistance { value: distance })
    }
}

/// Error types for feature matching operations.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum MatcherError {
    /// A query was attempted before building the index.
    #[error("matcher not built: call build() before querying")]
    NotBuilt,

    /// No database descriptors were supplied, or they have zero dimensions.
    #[error("empty feature set provided")]
    EmptyFeatures,

    /// Descriptor dimensions disagree.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Required descriptor dimension.
        expected: usize,
        /// Observed descriptor dimension.
        actual: usize,
    },

    /// The requested neighbor count is zero or exceeds the database size.
    #[error("invalid k value {k}: must be > 0 and <= database size")]
    InvalidK {
        /// Requested neighbor count.
        k: usize,
    },

    /// A query descriptor has zero dimensions.
    #[error("query feature is empty")]
    EmptyQuery,

    /// A database descriptor is invalid for the selected metric: a non-finite or overflowing
    /// float value, or a `u8` descriptor too long for its distance to fit in a `u32`.
    #[error("descriptor {index} contains invalid values")]
    InvalidDescriptor {
        /// Index of the invalid descriptor.
        index: usize,
    },

    /// A stored-descriptor index is outside the index.
    #[error("descriptor index {index} is outside database size {count}")]
    DescriptorIndexOutOfRange {
        /// Requested descriptor index.
        index: usize,
        /// Number of stored descriptors.
        count: usize,
    },

    /// A query descriptor is invalid for the selected metric.
    #[error("query descriptor contains invalid values")]
    InvalidQuery,

    /// The matcher's compact index type cannot represent this many entries.
    #[error("matcher index cannot contain {count} descriptors")]
    IndexTooLarge {
        /// Requested descriptor count.
        count: usize,
    },

    /// A match distance passed to [`Match::try_new`] or
    /// [`FilteredMatch::try_new`] is NaN, infinite, or negative.
    #[error("invalid match distance {value}: expected a finite non-negative value")]
    InvalidMatchDistance {
        /// Invalid distance value.
        value: f32,
    },

    /// A ratio passed to [`FilteredMatch::try_new`] is not a finite value in
    /// `[0, 1]`.
    #[error("invalid match ratio {value}: expected a finite value in [0, 1]")]
    InvalidMatchRatio {
        /// Invalid ratio value.
        value: f32,
    },
}

/// Return whether `value` is a valid Lowe ratio threshold.
#[inline]
pub(crate) fn is_valid_lowe_ratio(value: f32) -> bool {
    value.is_finite() && value > 0.0 && value < 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_try_new_checks_distance() {
        let matched = Match::try_new(3, 0.25).unwrap();
        assert_eq!((matched.index(), matched.distance()), (3, 0.25));
        assert_eq!(matched, Match::new_unchecked(3, 0.25));
        for distance in [-0.1, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                Match::try_new(0, distance),
                Err(MatcherError::InvalidMatchDistance { .. })
            ));
        }
    }

    #[test]
    fn filtered_match_try_new_checks_distance_and_ratio() {
        let matched = FilteredMatch::try_new(1, 2, 0.5, 0.75).unwrap();
        assert_eq!(
            (
                matched.query_index(),
                matched.target_index(),
                matched.distance(),
                matched.ratio()
            ),
            (1, 2, 0.5, 0.75)
        );
        assert_eq!(matched, FilteredMatch::new_unchecked(1, 2, 0.5, 0.75));
        assert!(FilteredMatch::try_new(0, 0, 0.0, 0.0).is_ok());
        assert!(FilteredMatch::try_new(0, 0, 0.0, 1.0).is_ok());
        assert!(matches!(
            FilteredMatch::try_new(0, 0, f32::NAN, 0.5),
            Err(MatcherError::InvalidMatchDistance { .. })
        ));
        for ratio in [-0.1, 1.1, f32::NAN] {
            assert!(matches!(
                FilteredMatch::try_new(0, 0, 0.5, ratio),
                Err(MatcherError::InvalidMatchRatio { .. })
            ));
        }
    }
}
