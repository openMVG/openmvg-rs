// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Value types for deterministic feature-track construction.

use std::collections::BTreeMap;

use openmvg_core::model::{FeatureId, TrackId, ViewId};

/// One feature identity in one view.
///
/// Pixel coordinates do not live here. Resolve `feature` through the owning
/// view's feature set when geometry is required.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
// Closed set: a track observation is exactly a (view, feature) identity pair.
pub struct TrackObservation {
    /// View that contains the feature.
    pub view: ViewId,
    /// Index of the feature in that view's feature set.
    pub feature: FeatureId,
}

impl TrackObservation {
    /// Pair a view with one of its features.
    #[must_use]
    pub const fn new(view: ViewId, feature: FeatureId) -> Self {
        Self { view, feature }
    }
}

/// One pairwise link supplied to track construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// Closed set: a pairwise link is exactly two observations.
pub struct TrackMatch {
    /// One endpoint of the link.
    pub first: TrackObservation,
    /// The other endpoint; links are undirected, so the order carries no meaning.
    pub second: TrackObservation,
}

impl TrackMatch {
    /// Link two observations.
    #[must_use]
    pub const fn new(first: TrackObservation, second: TrackObservation) -> Self {
        Self { first, second }
    }
}

/// A correspondence with a finite quality score; larger scores are preferred.
///
/// Scores only affect ambiguous merges. They need not be probabilities, but
/// must be comparable across the matches supplied to one track build.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredMatch {
    /// The two matched observations.
    matched: TrackMatch,
    /// Finite match quality; larger values are processed first.
    score: f32,
}

impl ScoredMatch {
    /// Construct a scored link after validating its ordering score.
    pub fn try_new(
        first: TrackObservation,
        second: TrackObservation,
        score: f32,
    ) -> Result<Self, TrackError> {
        if !score.is_finite() {
            return Err(TrackError::NonFiniteMatchScore);
        }
        Ok(Self {
            matched: TrackMatch::new(first, second),
            score,
        })
    }

    /// The two matched observations.
    #[must_use]
    pub const fn matched(self) -> TrackMatch {
        self.matched
    }

    /// Finite match quality; larger values are processed first.
    #[must_use]
    pub const fn score(self) -> f32 {
        self.score
    }

    /// Convert a non-negative descriptor distance into an ordering score.
    ///
    /// Pass the [`FilteredMatch::distance`](crate::matching::FilteredMatch::distance) of the
    /// match. The track builders only compare scores, so squared distances, as the squared-L2
    /// matchers report them, work as well as plain ones: the score is the negated distance, so
    /// closer matches are admitted first.
    pub fn try_from_distance(
        first: TrackObservation,
        second: TrackObservation,
        distance: f32,
    ) -> Result<Self, TrackError> {
        if !distance.is_finite() || distance < 0.0 {
            return Err(TrackError::InvalidMatchDistance);
        }
        Self::try_new(first, second, -distance)
    }
}

/// A group of observations that the track builders link together, before any check.
///
/// Either builder can return a group of a single observation, and [`build_tracks`] can
/// return one with two features of one view. Keep the groups you trust, such as those with two
/// or more observations, and store them as [`Track`]s in a [`TrackSet`], which check both.
///
/// [`build_tracks`]: crate::tracking::build_tracks
pub type TrackGroup = Vec<TrackObservation>;

/// Transient observation groups produced by tracking algorithms.
pub type TrackGroups = Vec<TrackGroup>;

/// One validated, persistently identified feature track.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Track {
    id: TrackId,
    observations: Vec<TrackObservation>,
}

impl<'de> serde::Deserialize<'de> for Track {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct SerializedTrack {
            id: TrackId,
            observations: Vec<TrackObservation>,
        }

        let serialized = SerializedTrack::deserialize(deserializer)?;
        Self::try_new(serialized.id, serialized.observations).map_err(serde::de::Error::custom)
    }
}

impl Track {
    /// Construct a track with at least two observations and at most one observation per view.
    pub fn try_new(
        id: TrackId,
        mut observations: Vec<TrackObservation>,
    ) -> Result<Self, TrackSetError> {
        if observations.len() > u32::MAX as usize {
            return Err(TrackSetError::TooManyObservations { track: id });
        }
        observations.sort_unstable();
        if observations.len() < 2 {
            return Err(TrackSetError::TooShort { track: id });
        }
        if observations
            .windows(2)
            .any(|window| window[0].view == window[1].view)
        {
            return Err(TrackSetError::DuplicateView { track: id });
        }
        Ok(Self { id, observations })
    }

    /// The ID given to [`Self::try_new`]. A [`TrackSet`] requires the IDs of its tracks to
    /// run from 0 without gaps.
    #[must_use]
    pub const fn id(&self) -> TrackId {
        self.id
    }

    /// Canonically sorted observations.
    #[must_use]
    pub fn observations(&self) -> &[TrackObservation] {
        &self.observations
    }
}

/// Canonically ordered feature tracks with a deterministic per-view index.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct TrackSet {
    tracks: Vec<Track>,
    #[serde(skip)]
    tracks_by_view: BTreeMap<ViewId, Vec<TrackId>>,
}

impl TrackSet {
    /// Validate and index a dense, zero-based track-ID domain.
    pub fn try_new(tracks: Vec<Track>) -> Result<Self, TrackSetError> {
        let mut tracks = tracks
            .into_iter()
            .map(|track| Track::try_new(track.id, track.observations))
            .collect::<Result<Vec<_>, _>>()?;
        tracks.sort_by_key(Track::id);
        let mut tracks_by_view = BTreeMap::<_, Vec<_>>::new();
        for (index, track) in tracks.iter().enumerate() {
            if track.id().index() != index {
                return Err(TrackSetError::InvalidId {
                    track: track.id(),
                    expected: index,
                });
            }
            for observation in track.observations() {
                tracks_by_view
                    .entry(observation.view)
                    .or_default()
                    .push(track.id());
            }
        }
        Ok(Self {
            tracks,
            tracks_by_view,
        })
    }

    /// Tracks in stable [`TrackId`] order.
    #[must_use]
    pub fn as_slice(&self) -> &[Track] {
        &self.tracks
    }

    /// Look up a track by persistent identity.
    #[must_use]
    pub fn get(&self, track_id: TrackId) -> Option<&Track> {
        self.tracks.get(track_id.index())
    }

    /// Track identities containing an observation from `view`.
    #[must_use]
    pub fn observing_view(&self, view: ViewId) -> &[TrackId] {
        self.tracks_by_view.get(&view).map_or(&[], Vec::as_slice)
    }

    /// Number of tracks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Whether the set holds no tracks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }
}

impl<'de> serde::Deserialize<'de> for TrackSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct SerializedTrackSet {
            tracks: Vec<Track>,
        }

        let serialized = SerializedTrackSet::deserialize(deserializer)?;
        Self::try_new(serialized.tracks).map_err(serde::de::Error::custom)
    }
}

/// Invalid canonical feature-track state.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TrackSetError {
    /// The track has more than `u32::MAX` observations.
    #[error("track {track} exceeds compact observation-index capacity")]
    TooManyObservations {
        /// Offending track.
        track: TrackId,
    },
    /// The track has fewer than two observations.
    #[error("track {track} must contain at least two observations")]
    TooShort {
        /// Offending track.
        track: TrackId,
    },
    /// The track has more than one observation in the same view.
    #[error("track {track} contains multiple observations from one view")]
    DuplicateView {
        /// Offending track.
        track: TrackId,
    },
    /// Track IDs, once sorted, are not exactly `0..len`.
    #[error("track ID {track} is not the expected dense ID {expected}")]
    InvalidId {
        /// Track found at sorted position `expected`.
        track: TrackId,
        /// Dense ID required at that position.
        expected: usize,
    },
}

/// Result and accounting from deterministic conflict-aware track construction.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TrackBuildReport {
    /// Stable lexicographically ordered tracks, including isolated nodes.
    pub tracks: TrackGroups,
    /// Edges that joined two separate components with disjoint view sets.
    pub accepted_match_count: usize,
    /// Edges rejected because merging would create two observations in one view.
    pub rejected_conflict_count: usize,
    /// Duplicate or cycle edges whose endpoints were already connected.
    pub redundant_match_count: usize,
}

/// Errors from building a [`ScoredMatch`].
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum TrackError {
    /// [`ScoredMatch::try_new`] received a NaN or infinite score.
    #[error("match score must be finite")]
    NonFiniteMatchScore,
    /// [`ScoredMatch::try_from_distance`] received a NaN, infinite or negative
    /// distance.
    #[error("match distance must be finite and non-negative")]
    InvalidMatchDistance,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(view: usize, feature: usize) -> TrackObservation {
        TrackObservation::new(ViewId::new(view), FeatureId::new(feature))
    }

    #[test]
    fn scored_match_from_distance_preserves_preference() {
        let close =
            ScoredMatch::try_from_distance(observation(0, 0), observation(1, 0), 2.0).unwrap();
        let far =
            ScoredMatch::try_from_distance(observation(0, 1), observation(1, 0), 5.0).unwrap();
        assert!(close.score() > far.score());
        assert!(
            ScoredMatch::try_from_distance(observation(0, 0), observation(1, 0), f32::NAN).is_err()
        );
        assert!(
            ScoredMatch::try_from_distance(observation(0, 0), observation(1, 0), -1.0).is_err()
        );
    }

    #[test]
    fn track_index_is_stable_and_rebuilt_after_deserialization() {
        let tracks = TrackSet::try_new(vec![
            Track::try_new(TrackId::new(1), vec![observation(1, 1), observation(2, 1)]).unwrap(),
            Track::try_new(TrackId::new(0), vec![observation(0, 0), observation(1, 0)]).unwrap(),
        ])
        .unwrap();
        assert_eq!(
            tracks.observing_view(ViewId::new(1)),
            &[TrackId::new(0), TrackId::new(1)]
        );

        let json = serde_json::to_string(&tracks).unwrap();
        let restored: TrackSet = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, tracks);
        assert_eq!(restored.observing_view(ViewId::new(2)), &[TrackId::new(1)]);
    }

    #[test]
    fn track_set_rejects_sparse_and_duplicate_ids() {
        let sparse = vec![
            Track::try_new(TrackId::new(1), vec![observation(0, 0), observation(1, 0)]).unwrap(),
        ];
        assert!(matches!(
            TrackSet::try_new(sparse),
            Err(TrackSetError::InvalidId { .. })
        ));

        let duplicate = vec![
            Track::try_new(TrackId::new(0), vec![observation(0, 0), observation(1, 0)]).unwrap(),
            Track::try_new(TrackId::new(0), vec![observation(0, 1), observation(1, 1)]).unwrap(),
        ];
        assert!(matches!(
            TrackSet::try_new(duplicate),
            Err(TrackSetError::InvalidId { .. })
        ));
    }

    #[test]
    fn track_set_deserialization_validates_ids() {
        let tracks = TrackSet::try_new(vec![
            Track::try_new(TrackId::new(0), vec![observation(0, 0), observation(1, 0)]).unwrap(),
        ])
        .unwrap();
        let mut value = serde_json::to_value(tracks).unwrap();
        value["tracks"][0]["id"] = serde_json::json!(usize::MAX);
        assert!(serde_json::from_value::<TrackSet>(value).is_err());
    }

    #[test]
    fn track_deserialization_preserves_validation_boundary() {
        let too_short = serde_json::json!({
            "id": 0,
            "observations": [{"view": 0, "feature": 0}],
        });
        assert!(serde_json::from_value::<Track>(too_short).is_err());

        let duplicate_view = serde_json::json!({
            "id": 0,
            "observations": [
                {"view": 0, "feature": 0},
                {"view": 0, "feature": 1},
            ],
        });
        assert!(serde_json::from_value::<Track>(duplicate_view).is_err());
    }
}
