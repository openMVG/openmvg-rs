// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core algorithms for feature track building.
//!
//! This module groups pairwise correspondences into feature tracks with the
//! dense disjoint sets from [`openmvg_core::graph`].

use super::types::{
    ScoredMatch, TrackBuildReport, TrackGroups as TrackSet, TrackMatch, TrackObservation,
};
use openmvg_core::graph::DenseDisjointSets;
use std::collections::{BTreeMap, BTreeSet};

/// Disjoint sets over the dense indices of the sorted track nodes.
///
/// Every index comes from `node_indices`, so it is below the node count and
/// the range checks of [`DenseDisjointSets`] cannot fail.
struct NodeSets(DenseDisjointSets);

impl NodeSets {
    fn new(len: usize) -> Self {
        match DenseDisjointSets::try_new(len) {
            Ok(sets) => Self(sets),
            Err(_) => std::alloc::handle_alloc_error(
                std::alloc::Layout::array::<usize>(len)
                    .unwrap_or_else(|_| std::alloc::Layout::new::<usize>()),
            ),
        }
    }

    fn find(&mut self, node: usize) -> usize {
        let root = self.0.find(node);
        debug_assert!(root.is_ok(), "track node {node} is in range");
        root.unwrap_or(node)
    }

    fn union(&mut self, first: usize, second: usize) {
        let merged = self.0.union(first, second);
        debug_assert!(
            merged.is_ok(),
            "track nodes {first} and {second} are in range"
        );
    }
}

/// Build feature tracks from pairwise matches: each track is the set of observations that the
/// matches connect, directly or through other observations.
///
/// Use it only when the matches are known to be consistent; [`build_tracks_scored`] keeps one
/// observation per view when some matches can be wrong, and the
/// [`tracking`](crate::tracking) module docs explain both. Sorting the match endpoints costs
/// `O(m log m)` time and `O(m)` memory for `m` matches.
///
/// Empty input gives no tracks, a feature appears in at most one track, the tracks and their
/// observations come in lexicographic order, and a feature matched to itself gives a group of
/// one observation.
///
/// # Examples
///
/// ```rust
/// use openmvg_features::tracking::{FeatureId, TrackMatch, TrackObservation, ViewId, build_tracks};
///
/// let observation = |view, feature| {
///     TrackObservation::new(ViewId::new(view), FeatureId::new(feature))
/// };
/// let matches = vec![
///     TrackMatch::new(observation(0, 10), observation(1, 20)),
///     TrackMatch::new(observation(1, 20), observation(2, 30)),
/// ];
/// let tracks = build_tracks(&matches);
/// assert_eq!(tracks.len(), 1);
/// assert!(tracks[0].contains(&observation(2, 30)));
/// ```
pub fn build_tracks(matches: &[TrackMatch]) -> TrackSet {
    if matches.is_empty() {
        return Vec::new();
    }

    let nodes = sorted_nodes(matches.iter().copied());
    let indices = node_indices(&nodes);
    let mut node_sets = NodeSets::new(nodes.len());
    for &TrackMatch { first, second } in matches {
        node_sets.union(indices[&first], indices[&second]);
    }
    extract_tracks(&nodes, &mut node_sets)
}

/// Build deterministic, conflict-aware tracks from quality-ranked matches.
///
/// Edges are processed from highest score to lowest, with a canonical
/// feature-pair tie break. An edge joins two components only when their view
/// sets are disjoint. This greedy rule is simple and deterministic; it is not
/// claimed to solve the global weighted set-packing problem.
///
/// Every returned track is sorted and contains at most one feature per view. Sorting the `m`
/// matches costs `O(m log m)`; each match then checks the views of the two components it would
/// join, at a cost up to the size of the smaller one, so very long tracks make this slower.
/// Every score is finite: the [`ScoredMatch`] constructors reject the others.
#[must_use]
pub fn build_tracks_scored(matches: &[ScoredMatch]) -> TrackBuildReport {
    build_tracks_scored_owned(matches.to_vec())
}

/// Build deterministic, conflict-aware tracks while consuming the scored-match buffer.
///
/// It gives the same tracks as [`build_tracks_scored`] but sorts the matches you hand over in
/// place, instead of copying them, which saves memory when there are many. Use
/// [`build_tracks_scored`] when you need to keep your matches.
#[must_use]
pub fn build_tracks_scored_owned(mut matches: Vec<ScoredMatch>) -> TrackBuildReport {
    if matches.is_empty() {
        return TrackBuildReport {
            tracks: Vec::new(),
            accepted_match_count: 0,
            rejected_conflict_count: 0,
            redundant_match_count: 0,
        };
    }

    let mut nodes = sorted_nodes(matches.iter().map(|matched| matched.matched()));
    // `sorted_nodes` starts with two entries per match. Release the usually large excess
    // capacity before allocating the feature index and conflict-tracking state.
    nodes.shrink_to_fit();
    let indices = node_indices(&nodes);
    let compare_matches = |left: &ScoredMatch, right: &ScoredMatch| {
        let (left_first, left_second) = canonical_pair(left.matched());
        let (right_first, right_second) = canonical_pair(right.matched());
        let left_score = if left.score() == 0.0 {
            0.0
        } else {
            left.score()
        };
        let right_score = if right.score() == 0.0 {
            0.0
        } else {
            right.score()
        };
        right_score
            .total_cmp(&left_score)
            .then_with(|| left_first.cmp(&right_first))
            .then_with(|| left_second.cmp(&right_second))
    };
    matches.sort_unstable_by(compare_matches);

    let mut node_sets = NodeSets::new(nodes.len());
    let mut component_images: Vec<ComponentImages> = nodes
        .iter()
        .map(|observation| ComponentImages::singleton(observation.view.index()))
        .collect();
    let mut accepted_match_count = 0;
    let mut rejected_conflict_count = 0;
    let mut redundant_match_count = 0;

    for matched in matches {
        let (first, second) = canonical_pair(matched.matched());
        let first_root = node_sets.find(indices[&first]);
        let second_root = node_sets.find(indices[&second]);
        if first_root == second_root {
            redundant_match_count += 1;
            continue;
        }
        if component_images[first_root].intersects(&component_images[second_root]) {
            rejected_conflict_count += 1;
            continue;
        }

        let first_images = std::mem::take(&mut component_images[first_root]);
        let second_images = std::mem::take(&mut component_images[second_root]);
        let merged_images = first_images.merge(second_images);
        node_sets.union(first_root, second_root);
        let merged_root = node_sets.find(first_root);
        component_images[merged_root] = merged_images;
        accepted_match_count += 1;
    }

    TrackBuildReport {
        tracks: extract_tracks(&nodes, &mut node_sets),
        accepted_match_count,
        rejected_conflict_count,
        redundant_match_count,
    }
}

fn canonical_pair(matched: TrackMatch) -> (TrackObservation, TrackObservation) {
    let TrackMatch { first, second } = matched;
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

fn sorted_nodes(matches: impl Iterator<Item = TrackMatch>) -> Vec<TrackObservation> {
    let mut nodes: Vec<_> = matches
        .flat_map(|matched| [matched.first, matched.second])
        .collect();
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

fn node_indices(nodes: &[TrackObservation]) -> BTreeMap<TrackObservation, usize> {
    nodes
        .iter()
        .copied()
        .enumerate()
        .map(|(index, feature)| (feature, index))
        .collect()
}

fn extract_tracks(nodes: &[TrackObservation], node_sets: &mut NodeSets) -> TrackSet {
    let mut groups: BTreeMap<usize, Vec<TrackObservation>> = BTreeMap::new();
    for (index, &feature) in nodes.iter().enumerate() {
        groups
            .entry(node_sets.find(index))
            .or_default()
            .push(feature);
    }
    let mut tracks: TrackSet = groups.into_values().collect();
    for track in &mut tracks {
        track.sort_unstable();
    }
    tracks.sort_unstable();
    tracks
}

fn sorted_intersects(first: &[usize], second: &[usize]) -> bool {
    let (mut left, mut right) = (0, 0);
    while left < first.len() && right < second.len() {
        match first[left].cmp(&second[right]) {
            std::cmp::Ordering::Less => left += 1,
            std::cmp::Ordering::Greater => right += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

const HASHED_COMPONENT_THRESHOLD: usize = 32;

/// Small tracks use sorted vectors; long tracks switch once to a hash set so
/// chain-shaped inputs do not repeatedly copy the entire component.
#[derive(Default)]
enum ComponentImages {
    #[default]
    Empty,
    Singleton(usize),
    Small(Vec<usize>),
    Large(BTreeSet<usize>),
}

impl ComponentImages {
    fn singleton(image: usize) -> Self {
        Self::Singleton(image)
    }

    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Singleton(_) => 1,
            Self::Small(images) => images.len(),
            Self::Large(images) => images.len(),
        }
    }

    fn intersects(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Empty, _) | (_, Self::Empty) => false,
            (Self::Singleton(first), Self::Singleton(second)) => first == second,
            (Self::Singleton(image), Self::Small(images))
            | (Self::Small(images), Self::Singleton(image)) => images.binary_search(image).is_ok(),
            (Self::Singleton(image), Self::Large(images))
            | (Self::Large(images), Self::Singleton(image)) => images.contains(image),
            (Self::Small(first), Self::Small(second)) => sorted_intersects(first, second),
            (Self::Small(small), Self::Large(large)) | (Self::Large(large), Self::Small(small)) => {
                small.iter().any(|image| large.contains(image))
            }
            (Self::Large(first), Self::Large(second)) => {
                let (small, large) = if first.len() <= second.len() {
                    (first, second)
                } else {
                    (second, first)
                };
                small.iter().any(|image| large.contains(image))
            }
        }
    }

    fn merge(self, other: Self) -> Self {
        let total = self.len() + other.len();
        match (self, other) {
            (Self::Empty, other) | (other, Self::Empty) => other,
            (Self::Singleton(first), Self::Singleton(second)) => Self::Small(if first < second {
                vec![first, second]
            } else {
                vec![second, first]
            }),
            (Self::Singleton(image), Self::Small(mut images))
            | (Self::Small(mut images), Self::Singleton(image))
                if total <= HASHED_COMPONENT_THRESHOLD =>
            {
                let index = images.binary_search(&image).unwrap_or_else(|index| index);
                images.insert(index, image);
                Self::Small(images)
            }
            (Self::Singleton(image), Self::Large(mut images))
            | (Self::Large(mut images), Self::Singleton(image)) => {
                images.insert(image);
                Self::Large(images)
            }
            (Self::Singleton(image), Self::Small(images))
            | (Self::Small(images), Self::Singleton(image)) => {
                let mut merged = BTreeSet::new();
                merged.insert(image);
                merged.extend(images);
                Self::Large(merged)
            }
            (Self::Small(first), Self::Small(second)) if total <= HASHED_COMPONENT_THRESHOLD => {
                Self::Small(merge_sorted(&first, &second))
            }
            (Self::Large(mut first), Self::Large(mut second)) => {
                if first.len() < second.len() {
                    std::mem::swap(&mut first, &mut second);
                }
                first.extend(second);
                Self::Large(first)
            }
            (Self::Large(mut large), Self::Small(small))
            | (Self::Small(small), Self::Large(mut large)) => {
                large.extend(small);
                Self::Large(large)
            }
            (Self::Small(first), Self::Small(second)) => {
                let mut images = BTreeSet::new();
                images.extend(first);
                images.extend(second);
                Self::Large(images)
            }
        }
    }
}

fn merge_sorted(first: &[usize], second: &[usize]) -> Vec<usize> {
    let mut merged = Vec::with_capacity(first.len() + second.len());
    let (mut left, mut right) = (0, 0);
    while left < first.len() && right < second.len() {
        if first[left] < second[right] {
            merged.push(first[left]);
            left += 1;
        } else {
            merged.push(second[right]);
            right += 1;
        }
    }
    merged.extend_from_slice(&first[left..]);
    merged.extend_from_slice(&second[right..]);
    merged
}

#[cfg(test)]
mod tests {
    use super::super::types::TrackError;
    use super::*;
    use openmvg_core::model::{FeatureId, ViewId};

    fn observation(view: usize, feature: usize) -> TrackObservation {
        TrackObservation::new(ViewId::new(view), FeatureId::new(feature))
    }

    fn matched(
        first_view: usize,
        first_feature: usize,
        second_view: usize,
        second_feature: usize,
    ) -> TrackMatch {
        TrackMatch::new(
            observation(first_view, first_feature),
            observation(second_view, second_feature),
        )
    }

    fn scored(first: TrackObservation, second: TrackObservation, score: f32) -> ScoredMatch {
        ScoredMatch::try_new(first, second, score).expect("finite test score")
    }

    #[test]
    fn test_build_tracks_basic() {
        let matches = vec![
            matched(0, 1, 1, 2),
            matched(1, 2, 2, 3),
            matched(3, 4, 4, 5),
        ];

        let mut tracks = build_tracks(&matches);
        for t in tracks.iter_mut() {
            t.sort();
        }
        tracks.sort();

        assert_eq!(tracks.len(), 2);
        assert!(tracks.contains(&vec![
            observation(0, 1),
            observation(1, 2),
            observation(2, 3)
        ]));
        assert!(tracks.contains(&vec![observation(3, 4), observation(4, 5)]));
    }

    #[test]
    fn test_build_tracks_empty_input() {
        let matches = vec![];
        let tracks = build_tracks(&matches);
        assert!(tracks.is_empty());
    }

    #[test]
    fn test_build_tracks_single_match() {
        let matches = vec![matched(0, 10, 1, 20)];
        let tracks = build_tracks(&matches);

        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].len(), 2);
        assert!(tracks[0].contains(&observation(0, 10)));
        assert!(tracks[0].contains(&observation(1, 20)));
    }

    #[test]
    fn test_build_tracks_self_match() {
        let matches = vec![matched(0, 10, 0, 10)];
        let tracks = build_tracks(&matches);

        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].len(), 1);
        assert!(tracks[0].contains(&observation(0, 10)));
    }

    #[test]
    fn test_build_tracks_complex_chain() {
        let matches = vec![
            matched(0, 100, 1, 200),
            matched(1, 200, 2, 300),
            matched(2, 300, 3, 400),
            matched(1, 200, 4, 500), // Branch from observation(1, 200)
        ];

        let tracks = build_tracks(&matches);
        assert_eq!(tracks.len(), 1);

        // All features should be in the same track
        let track = &tracks[0];
        assert_eq!(track.len(), 5);
        assert!(track.contains(&observation(0, 100)));
        assert!(track.contains(&observation(1, 200)));
        assert!(track.contains(&observation(2, 300)));
        assert!(track.contains(&observation(3, 400)));
        assert!(track.contains(&observation(4, 500)));
    }

    #[test]
    fn unscored_connected_components_are_permutation_invariant() {
        let matches = vec![
            matched(2, 30, 1, 20),
            matched(4, 50, 3, 40),
            matched(0, 10, 1, 20),
            matched(3, 40, 2, 30),
        ];
        let expected = build_tracks(&matches);
        for offset in 0..matches.len() {
            let mut permuted = matches.clone();
            permuted.rotate_left(offset);
            if offset % 2 == 1 {
                permuted.reverse();
            }
            assert_eq!(build_tracks(&permuted), expected);
        }
    }

    #[test]
    fn scored_builder_keeps_strong_consistent_branches() {
        let matches = vec![
            scored(observation(0, 0), observation(1, 0), 10.0),
            scored(observation(1, 0), observation(2, 0), 9.0),
            scored(observation(0, 1), observation(2, 0), 8.0),
            scored(observation(0, 1), observation(3, 1), 7.0),
        ];
        let outcome = build_tracks_scored(&matches);
        assert_eq!(
            outcome.tracks,
            vec![
                vec![observation(0, 0), observation(1, 0), observation(2, 0)],
                vec![observation(0, 1), observation(3, 1)]
            ]
        );
        assert_eq!(outcome.accepted_match_count, 3);
        assert_eq!(outcome.rejected_conflict_count, 1);
        assert_eq!(outcome.redundant_match_count, 0);
    }

    #[test]
    fn scored_builder_is_permutation_invariant() {
        let matches = vec![
            scored(observation(0, 0), observation(1, 0), 4.0),
            scored(observation(1, 0), observation(2, 0), 3.0),
            scored(observation(0, 1), observation(2, 0), 2.0),
            scored(observation(0, 1), observation(3, 1), 1.0),
            scored(observation(3, 1), observation(0, 1), 1.0),
        ];
        let expected = build_tracks_scored(&matches);
        for offset in 0..matches.len() {
            let mut permuted = matches.clone();
            permuted.rotate_left(offset);
            if offset % 2 == 1 {
                permuted.reverse();
            }
            assert_eq!(build_tracks_scored(&permuted), expected);
        }
    }

    #[test]
    fn owned_scored_builder_matches_borrowed_contract() {
        let matches = vec![
            scored(observation(0, 0), observation(1, 0), 4.0),
            scored(observation(1, 0), observation(2, 0), 3.0),
            scored(observation(0, 1), observation(2, 0), 2.0),
            scored(observation(3, 1), observation(0, 1), -0.0),
            scored(observation(0, 1), observation(3, 1), 0.0),
        ];

        assert_eq!(
            build_tracks_scored_owned(matches.clone()),
            build_tracks_scored(&matches)
        );
    }

    #[test]
    fn equal_scores_use_canonical_feature_tie_break() {
        let outcome = build_tracks_scored(&[
            scored(observation(0, 1), observation(1, 0), 1.0),
            scored(observation(0, 0), observation(1, 0), 1.0),
        ]);
        assert_eq!(
            outcome.tracks,
            vec![
                vec![observation(0, 0), observation(1, 0)],
                vec![observation(0, 1)]
            ]
        );
        assert_eq!(outcome.rejected_conflict_count, 1);
    }

    #[test]
    fn reversed_duplicate_edges_are_redundant() {
        let outcome = build_tracks_scored(&[
            scored(observation(0, 0), observation(1, 0), 2.0),
            scored(observation(1, 0), observation(0, 0), 2.0),
        ]);
        assert_eq!(
            outcome.tracks,
            vec![vec![observation(0, 0), observation(1, 0)]]
        );
        assert_eq!(outcome.accepted_match_count, 1);
        assert_eq!(outcome.redundant_match_count, 1);
    }

    #[test]
    fn direct_same_image_edges_are_conflicts() {
        let outcome = build_tracks_scored(&[scored(observation(0, 0), observation(0, 1), 0.0)]);
        assert_eq!(
            outcome.tracks,
            vec![vec![observation(0, 0)], vec![observation(0, 1)]]
        );
        assert_eq!(outcome.rejected_conflict_count, 1);
    }

    #[test]
    fn non_finite_scores_are_rejected() {
        let result = ScoredMatch::try_new(observation(0, 0), observation(1, 0), f32::NAN);
        assert!(matches!(result, Err(TrackError::NonFiniteMatchScore)));
    }

    #[test]
    fn long_components_preserve_the_one_image_invariant() {
        let mut matches = Vec::new();
        for image in 0..63 {
            matches.push(scored(
                observation(image, 0),
                observation(image + 1, 0),
                1.0,
            ));
        }
        matches.push(scored(observation(0, 1), observation(63, 0), 0.0));

        let outcome = build_tracks_scored(&matches);
        assert_eq!(outcome.rejected_conflict_count, 1);
        assert_eq!(outcome.tracks.len(), 2);
        assert_eq!(outcome.tracks[0].len(), 64);
        assert_eq!(outcome.tracks[1], vec![observation(0, 1)]);
    }
}
