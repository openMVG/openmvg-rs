// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Correspondence acceptance shared by every descriptor matcher.
//!
//! Each matcher finds the two nearest targets of every query descriptor and
//! applies Lowe's ratio test. [`MatchAcceptance`] decides whether that is
//! enough or whether the target must also choose the query back.

use crate::matching::filters::LoweRatioFilter;
use crate::matching::filters::distance_ratio::lowe_ratio;
use crate::matching::types::{FilteredMatch, Match, is_valid_lowe_ratio};

/// Rule that turns two-nearest-neighbor candidates into correspondences.
///
/// Both rules start from Lowe's ratio test on the query's two nearest targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum MatchAcceptance {
    /// Keep a pair only when each descriptor is the other's nearest neighbor
    /// and passes the ratio test in its own direction.
    ///
    /// Matches are one-to-one and do not depend on which image is the query.
    /// Both images need at least two descriptors.
    #[default]
    Reciprocal,
    /// Keep every query whose nearest target passes the ratio test.
    ///
    /// Several queries may share a target.
    OneWay,
}

impl MatchAcceptance {
    /// Return whether a pair with these descriptor counts can produce any match.
    ///
    /// The ratio test needs two targets. [`Self::Reciprocal`] also runs it
    /// from the target side, so it needs two queries as well.
    pub const fn can_match(self, queries: usize, targets: usize) -> bool {
        let min_queries = match self {
            Self::Reciprocal => 2,
            Self::OneWay => 1,
        };
        queries >= min_queries && targets >= 2
    }

    /// Accept squared-L2 candidates, where `forward[query]` lists the query's
    /// nearest targets, nearest first. `ratio` is in ordinary L2 units.
    ///
    /// For [`Self::Reciprocal`], `reverse_query` receives the sorted, unique
    /// targets that passed the forward test and returns each one's nearest
    /// queries, in the same order. [`Self::OneWay`] never calls it. The
    /// [Examples section of the `matching` module](crate::matching#examples) writes this closure
    /// for two HNSW indexes.
    pub fn filter_squared<E>(
        self,
        forward: &[Vec<Match>],
        ratio: f32,
        reverse_query: impl FnOnce(&[usize]) -> Result<Vec<Vec<Match>>, E>,
    ) -> Result<Vec<FilteredMatch>, E> {
        let filter = LoweRatioFilter::new();
        let forward = filter.apply_squared(forward, ratio);
        if self == Self::OneWay || forward.is_empty() {
            return Ok(forward);
        }
        let mut targets: Vec<_> = forward.iter().map(|matched| matched.target_index).collect();
        targets.sort_unstable();
        targets.dedup();
        let mut query_by_slot = vec![None; targets.len()];
        for matched in filter.apply_squared(&reverse_query(&targets)?, ratio) {
            if let Some(query) = query_by_slot.get_mut(matched.query_index) {
                *query = Some(matched.target_index);
            }
        }
        Ok(forward
            .into_iter()
            .filter(|matched| {
                targets
                    .binary_search(&matched.target_index)
                    .ok()
                    .and_then(|slot| query_by_slot[slot])
                    == Some(matched.query_index)
            })
            .collect())
    }

    /// Accept exhaustive integer-distance scans.
    ///
    /// `backward[target]` holds the target's two nearest queries and is read
    /// only for [`Self::Reciprocal`]. `squared` selects the squared-L2 ratio
    /// convention of [`LoweRatioFilter::apply_squared`].
    pub(crate) fn select_scans(
        self,
        forward: &[NearestTwo],
        backward: &[NearestTwo],
        ratio: f32,
        squared: bool,
    ) -> Vec<FilteredMatch> {
        if !is_valid_lowe_ratio(ratio) {
            return Vec::new();
        }
        forward
            .iter()
            .enumerate()
            .filter_map(|(query, nearest)| {
                let reported_ratio = nearest.lowe_ratio(ratio, squared)?;
                let (distance, target) = nearest.best;
                if self == Self::Reciprocal {
                    let reverse = &backward[target];
                    if reverse.best.1 != query || reverse.lowe_ratio(ratio, squared).is_none() {
                        return None;
                    }
                }
                Some(FilteredMatch::new_unchecked(
                    query,
                    target,
                    distance as f32,
                    reported_ratio,
                ))
            })
            .collect()
    }

    /// Per-target nearest queries for an exhaustive scan: one entry per
    /// target for [`Self::Reciprocal`], none for [`Self::OneWay`].
    pub(crate) fn backward_scan(self, targets: usize) -> Vec<NearestTwo> {
        match self {
            Self::Reciprocal => vec![NearestTwo::EMPTY; targets],
            Self::OneWay => Vec::new(),
        }
    }
}

/// Two nearest candidates of an exhaustive scan, ordered by
/// `(distance, index)` so ties resolve to the lowest index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NearestTwo {
    pub(crate) best: (u32, usize),
    pub(crate) second: (u32, usize),
}

impl NearestTwo {
    pub(crate) const EMPTY: Self = Self {
        best: (u32::MAX, usize::MAX),
        second: (u32::MAX, usize::MAX),
    };

    #[inline(always)]
    pub(crate) fn consider(&mut self, distance: u32, index: usize) {
        let candidate = (distance, index);
        // `best <= second`, so most candidates are rejected by one comparison.
        if candidate < self.second {
            if candidate < self.best {
                self.second = self.best;
                self.best = candidate;
            } else {
                self.second = candidate;
            }
        }
    }

    fn lowe_ratio(&self, ratio: f32, squared: bool) -> Option<f32> {
        if self.second == Self::EMPTY.second {
            return None;
        }
        lowe_ratio(
            f64::from(self.best.0),
            f64::from(self.second.0),
            ratio,
            squared,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(distances: &[&[u32]], acceptance: MatchAcceptance) -> Vec<(usize, usize)> {
        let targets = distances[0].len();
        let mut backward = acceptance.backward_scan(targets);
        let forward: Vec<_> = distances
            .iter()
            .enumerate()
            .map(|(query, row)| {
                let mut nearest = NearestTwo::EMPTY;
                for (target, &distance) in row.iter().enumerate() {
                    nearest.consider(distance, target);
                    if let Some(reverse) = backward.get_mut(target) {
                        reverse.consider(distance, query);
                    }
                }
                nearest
            })
            .collect();
        acceptance
            .select_scans(&forward, &backward, 0.8, false)
            .iter()
            .map(|matched| (matched.query_index, matched.target_index))
            .collect()
    }

    #[test]
    fn reciprocal_is_the_default() {
        assert_eq!(MatchAcceptance::default(), MatchAcceptance::Reciprocal);
    }

    #[test]
    fn nearest_two_breaks_ties_by_index() {
        let mut nearest = NearestTwo::EMPTY;
        nearest.consider(7, 2);
        nearest.consider(7, 1);
        nearest.consider(9, 0);
        assert_eq!(nearest.best, (7, 1));
        assert_eq!(nearest.second, (7, 2));
    }

    #[test]
    fn descriptor_counts_gate_each_rule() {
        assert!(!MatchAcceptance::OneWay.can_match(1, 1));
        assert!(MatchAcceptance::OneWay.can_match(1, 2));
        assert!(!MatchAcceptance::Reciprocal.can_match(1, 2));
        assert!(MatchAcceptance::Reciprocal.can_match(2, 2));
    }

    #[test]
    fn reciprocal_requires_the_ratio_test_from_the_target() {
        // Query 0 picks target 0 unambiguously (10 vs 100), but target 0 sees
        // queries 0 and 1 at 10 and 11.
        let distances: [&[u32]; 2] = [&[10, 100], &[11, 5]];
        assert_eq!(
            scan(&distances, MatchAcceptance::OneWay),
            vec![(0, 0), (1, 1)]
        );
        assert_eq!(scan(&distances, MatchAcceptance::Reciprocal), vec![(1, 1)]);

        let distances: [&[u32]; 2] = [&[1, 100], &[50, 5]];
        assert_eq!(
            scan(&distances, MatchAcceptance::Reciprocal),
            vec![(0, 0), (1, 1)]
        );
    }

    #[test]
    fn one_way_keeps_repeated_targets() {
        let distances: [&[u32]; 2] = [&[1, 100], &[2, 100]];
        assert_eq!(
            scan(&distances, MatchAcceptance::OneWay),
            vec![(0, 0), (1, 0)]
        );
        assert_eq!(scan(&distances, MatchAcceptance::Reciprocal), vec![(0, 0)]);
    }

    #[test]
    fn filter_squared_matches_the_exact_scan_rule() {
        let forward = vec![
            vec![Match::new_unchecked(0, 1.0), Match::new_unchecked(1, 100.0)],
            vec![Match::new_unchecked(0, 1.2), Match::new_unchecked(1, 100.0)],
            vec![Match::new_unchecked(1, 4.0), Match::new_unchecked(0, 100.0)],
        ];
        let reverse = [
            vec![Match::new_unchecked(0, 1.0), Match::new_unchecked(1, 1.2)],
            vec![Match::new_unchecked(2, 4.0), Match::new_unchecked(0, 100.0)],
        ];
        let mut requested = Vec::new();
        let reciprocal = MatchAcceptance::Reciprocal
            .filter_squared(&forward, 0.8, |targets| {
                requested = targets.to_vec();
                Ok::<_, ()>(targets.iter().map(|&t| reverse[t].clone()).collect())
            })
            .unwrap();
        assert_eq!(requested, vec![0, 1]);
        let pairs: Vec<_> = reciprocal
            .iter()
            .map(|matched| (matched.query_index, matched.target_index))
            .collect();
        // Target 0 sees queries 0 and 1 at 1 and 1.2, above the 0.8² threshold.
        assert_eq!(pairs, vec![(2, 1)]);

        let one_way = MatchAcceptance::OneWay
            .filter_squared(&forward, 0.8, |_| -> Result<Vec<Vec<Match>>, ()> {
                panic!("one-way acceptance must not run a reverse query")
            })
            .unwrap();
        assert_eq!(one_way.len(), 3);
    }

    #[test]
    fn single_candidate_scans_never_pass() {
        let mut nearest = NearestTwo::EMPTY;
        nearest.consider(0, 0);
        assert_eq!(nearest.lowe_ratio(0.8, true), None);
    }
}
