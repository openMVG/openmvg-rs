// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Descriptor matches between two ordered image feature sets.

use openmvg_core::model::FeatureId;

use super::FeatureDataError;

/// Descriptor match between two ordered image feature sets.
///
/// Deserialization applies the [`Self::try_new`] checks.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "FeatureMatchWire")]
pub struct FeatureMatch {
    first: FeatureId,
    second: FeatureId,
    distance: Option<f32>,
}

validated_wire!(
    FeatureMatch,
    #[serde(rename = "FeatureMatch")]
    struct FeatureMatchWire {
        first: FeatureId,
        second: FeatureId,
        distance: Option<f32>,
    }
);

impl FeatureMatch {
    /// Match feature `first` of the first image with feature `second` of the
    /// second image. `distance` is the descriptor distance, if known.
    ///
    /// # Errors
    ///
    /// [`FeatureDataError::InvalidMatchDistance`] if `distance` is NaN,
    /// infinite or negative.
    pub fn try_new(
        first: FeatureId,
        second: FeatureId,
        distance: Option<f32>,
    ) -> Result<Self, FeatureDataError> {
        let matched = Self {
            first,
            second,
            distance,
        };
        matched.validate()?;
        Ok(matched)
    }

    fn validate(self) -> Result<(), FeatureDataError> {
        if self
            .distance
            .is_some_and(|distance| !distance.is_finite() || distance < 0.0)
        {
            return Err(FeatureDataError::InvalidMatchDistance);
        }
        Ok(())
    }

    /// Feature in the first image of the pair.
    #[must_use]
    pub const fn first(self) -> FeatureId {
        self.first
    }

    /// Feature in the second image of the pair.
    #[must_use]
    pub const fn second(self) -> FeatureId {
        self.second
    }

    /// Non-negative descriptor distance in the matcher's metric (lower is
    /// better), or `None` when the match carries no distance.
    #[must_use]
    pub const fn distance(self) -> Option<f32> {
        self.distance
    }
}

/// Sorted, duplicate-free matches for one ordered image pair.
///
/// Deserialization applies [`Self::validate`]; unlike [`Self::try_new`] it
/// does not sort, so out-of-order input is rejected.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "MatchSetWire")]
pub struct MatchSet {
    matches: Vec<FeatureMatch>,
}

validated_wire!(
    MatchSet,
    #[serde(rename = "MatchSet")]
    struct MatchSetWire {
        matches: Vec<FeatureMatch>,
    }
);

impl MatchSet {
    /// Sort matches by `(first, second)` and validate them.
    ///
    /// # Errors
    ///
    /// [`FeatureDataError::NonCanonicalMatchOrder`] if two matches share the
    /// same feature pair; see also [`Self::validate`].
    pub fn try_new(mut matches: Vec<FeatureMatch>) -> Result<Self, FeatureDataError> {
        matches.sort_by_key(|matched| (matched.first, matched.second));
        let set = Self { matches };
        set.validate()?;
        Ok(set)
    }

    /// Check that every distance is finite and non-negative and that matches
    /// are strictly increasing by `(first, second)`.
    ///
    /// # Errors
    ///
    /// [`FeatureDataError::InvalidMatchDistance`] or
    /// [`FeatureDataError::NonCanonicalMatchOrder`].
    pub fn validate(&self) -> Result<(), FeatureDataError> {
        if self.matches.iter().any(|matched| {
            matched
                .distance
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        }) {
            return Err(FeatureDataError::InvalidMatchDistance);
        }
        if self.matches.windows(2).any(|window| {
            (window[0].first, window[0].second) >= (window[1].first, window[1].second)
        }) {
            return Err(FeatureDataError::NonCanonicalMatchOrder);
        }
        Ok(())
    }

    /// Matches sorted by `(first, second)`.
    #[must_use]
    pub fn as_slice(&self) -> &[FeatureMatch] {
        &self.matches
    }

    /// Number of matches.
    #[must_use]
    pub fn len(&self) -> usize {
        self.matches.len()
    }

    /// Whether the set holds no matches.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }
}
