// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature tracking: building feature tracks from pairwise matches.
//!
//! This module builds feature tracks from pairwise correspondences for Structure
//! from Motion (SfM), grouping corresponding features across views with a union-find
//! structure. It re-exports [`ViewId`] and [`FeatureId`] from `openmvg-core`, so track
//! observations can be built without another dependency.
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │                        Track construction                                  │
//! └────────────────────────────────────────────────────────────────────────────┘
//! Input: matches (view, feature) ↔ (view, feature), optionally with a finite score
//!   │
//!   ▼
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Index the observations                                             │
//! └────────────────────────────────────────────────────────────────────────────┘
//! Collect the distinct (view, feature) pairs, sort them, and give each a dense
//! index. One disjoint set per observation.
//!   │
//!   ▼
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Union the matches                                                  │
//! └────────────────────────────────────────────────────────────────────────────┘
//! build_tracks:         union every match.
//! build_tracks_scored:  sort matches by score (descending), then by feature pair;
//!                       for each match:
//!                         same set already            → redundant, skip
//!                         view sets intersect         → conflict, reject
//!                         otherwise                   → union, merge view sets
//!   │
//!   ▼
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Extract the components                                             │
//! └────────────────────────────────────────────────────────────────────────────┘
//! Group observations by set, sort each group, then sort the groups.
//!   │
//!   ▼
//! Output: sorted groups of observations (build_tracks), or the groups plus
//!         accepted, conflict and redundant match counts (build_tracks_scored)
//! ```
//!
//! # Algorithm
//!
//! A track is a connected component of the match graph whose nodes are `(view, feature)`
//! observations. [`build_tracks`] takes the transitive closure of the matches: if view 0 matches
//! view 1 and view 1 matches view 2 on the shared feature, all three observations form one group,
//! whether or not views 0 and 2 matched directly. The cost is `O(m log m)` time and `O(m)`
//! memory for `m` matches.
//!
//! A single wrong match can merge two scene points into one component, and the closure then holds
//! two features from one view. [`build_tracks_scored`] limits this. It admits matches from the
//! best score to the worst and joins two components only when they have no view in common, so each
//! group has at most one observation per view. The rule is greedy and is not claimed to solve the
//! global weighted set-packing problem.
//!
//! # Determinism
//!
//! The observations are sorted before indexing and the output is sorted, so the result does not
//! depend on the input order of the matches or on which node represents a set. The scored
//! variant orders ties by score and then by the sorted feature pair, so equal scores are
//! resolved by identity, not by input position.
//!
//! # Failure modes
//!
//! [`ScoredMatch::try_new`] returns [`TrackError::NonFiniteMatchScore`] for a NaN or infinite
//! score, so the builders never see one. Empty input returns an empty result. Both builders return groups of any length, including single observations left
//! by rejected matches. [`Track::try_new`] is the check that a persistent track has at least two
//! observations and at most one per view, and returns [`TrackSetError`] otherwise.
//! [`filter_by_scale_dispersion`] drops groups whose log feature scales are spread widely
//! (a likely merge of two scene points); groups of fewer than three valid scales always pass.
//!
//! # Alternatives
//!
//! Matches that come from descriptor matching can always be wrong, so build their tracks with
//! [`build_tracks_scored`] (or [`build_tracks_scored_owned`], which sorts the supplied buffer in
//! place instead of copying it), as `openmvg-sfm` does. Use [`build_tracks`] only when the
//! matches are known to be consistent and a plain connected-component grouping is enough.
//!
//! # Core concepts
//!
//! - Observation: a typed `(view, feature)` identity without copied pixel data
//! - Match: a correspondence between two features in different views
//! - Track: the set of feature observations of one scene point across views
//!
//! # Contents
//!
//! - [`build_tracks`] groups matches into connected components.
//! - [`build_tracks_scored`] and [`build_tracks_scored_owned`] build conflict-aware tracks
//!   (at most one observation per view) from quality-ranked matches.
//! - [`evaluate_scales`] and [`filter_by_scale_dispersion`] reject tracks whose
//!   feature scales are inconsistent.
//! - [`TrackObservation`], [`TrackMatch`], [`ScoredMatch`], [`Track`], and [`TrackSet`] are the
//!   typed observation, match, and track values.
//!
//! # Quick start
//!
//! ## Basic usage
//! ```rust
//! use openmvg_features::tracking::{FeatureId, TrackMatch, TrackObservation, ViewId, build_tracks};
//!
//! let observation = |view, feature| {
//!     TrackObservation::new(ViewId::new(view), FeatureId::new(feature))
//! };
//! let matches = vec![
//!     TrackMatch::new(observation(0, 10), observation(1, 20)),
//!     TrackMatch::new(observation(1, 20), observation(2, 30)),
//! ];
//!
//! let tracks = build_tracks(&matches);
//! assert_eq!(tracks, vec![vec![observation(0, 10), observation(1, 20), observation(2, 30)]]);
//! ```
//!
//! ## Matches that can be wrong
//!
//! Score each match by its descriptor distance and let [`build_tracks_scored`] reject the
//! matches that would put two features of one view in the same track. With real matches, the
//! distance is the [`FilteredMatch::distance`](crate::matching::FilteredMatch::distance) of
//! each match that [`MatchAcceptance`](crate::matching::MatchAcceptance) kept, its query and
//! target indices are the feature indices in the two views, and squared distances work as
//! well as plain ones, since only their order counts. Here the distances are written out:
//!
//! ```rust
//! use openmvg_features::tracking::{
//!     FeatureId, ScoredMatch, TrackError, TrackObservation, ViewId, build_tracks_scored,
//! };
//!
//! let observation = |view, feature| {
//!     TrackObservation::new(ViewId::new(view), FeatureId::new(feature))
//! };
//! // The last match is wrong: it sends a second feature of view 0 to the same scene point.
//! let matches = [
//!     ScoredMatch::try_from_distance(observation(0, 10), observation(1, 20), 0.1)?,
//!     ScoredMatch::try_from_distance(observation(1, 20), observation(2, 30), 0.2)?,
//!     ScoredMatch::try_from_distance(observation(0, 11), observation(2, 30), 0.9)?,
//! ];
//!
//! let report = build_tracks_scored(&matches);
//! assert_eq!(report.rejected_conflict_count, 1);
//! // Feature 11 is left alone; keep the groups of two or more observations as tracks.
//! let tracks: Vec<_> = report.tracks.into_iter().filter(|group| group.len() >= 2).collect();
//! assert_eq!(tracks, vec![vec![observation(0, 10), observation(1, 20), observation(2, 30)]]);
//! # Ok::<(), TrackError>(())
//! ```

mod algorithms;
mod scale_dispersion;
mod statistics;
mod types;

pub use algorithms::{build_tracks, build_tracks_scored, build_tracks_scored_owned};
pub use openmvg_core::model::{FeatureId, ViewId};
pub use types::{
    ScoredMatch, Track, TrackBuildReport, TrackError, TrackGroup, TrackGroups, TrackMatch,
    TrackObservation, TrackSet, TrackSetError,
};

pub use scale_dispersion::{
    ScaleDispersionConfig, ScaleDispersionReport, evaluate_scales,
    filter_indices as filter_by_scale_dispersion,
};
