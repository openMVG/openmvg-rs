// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic undirected graph of verified view pairs, with per-pair activity and confidence.

use crate::{PairId, Result, SfmError, ViewId, ViewPair};

/// One adjacency entry of a [`PairGraph`] view: a view it shares a pair with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Neighbor {
    /// The other endpoint of the pair.
    pub view: ViewId,
    pub(crate) pair_id: PairId,
}

/// Deterministic undirected image-pair graph.
///
/// Construction treats candidate pairs as a set: endpoint orientation and
/// repeated endpoint-only candidates are normalized away. Measurement-bearing
/// graph filters use a stricter policy because duplicate payloads are ambiguous.
///
/// ```
/// # #[cfg(feature = "unstable-internals")]
/// # fn main() -> Result<(), openmvg_sfm::SfmError> {
/// use openmvg_sfm::unstable::PairGraph;
/// use openmvg_sfm::ViewId;
///
/// let graph = PairGraph::exhaustive(3)?;
/// assert_eq!(graph.len(), 3);
/// assert_eq!(graph.neighbors(ViewId::new(0))?.len(), 2);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "unstable-internals"))]
/// # fn main() {}
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PairGraph {
    view_count: usize,
    pairs: Vec<ViewPair>,
    active: Vec<bool>,
    confidences: Vec<f64>,
    adjacency_offsets: Vec<usize>,
    adjacency: Vec<Neighbor>,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl PairGraph {
    /// Build a graph over `view_count` views from candidate pairs, all active with unit confidence.
    ///
    /// Duplicates are removed and pairs are stored in canonical sorted order. Fails when a pair
    /// has equal endpoints or an endpoint outside `0..view_count`.
    pub fn try_new(view_count: usize, pairs: impl IntoIterator<Item = ViewPair>) -> Result<Self> {
        let mut canonical = Vec::new();
        for pair in pairs {
            let (view_i, view_j) = pair.endpoints();
            if view_i == view_j {
                return Err(SfmError::InvalidInput(format!(
                    "pair endpoints must differ: {view_i}"
                )));
            }
            if view_i.index() >= view_count {
                return Err(SfmError::InvalidId {
                    kind: ViewId::kind(),
                    index: view_i.index(),
                });
            }
            if view_j.index() >= view_count {
                return Err(SfmError::InvalidId {
                    kind: ViewId::kind(),
                    index: view_j.index(),
                });
            }
            canonical.push(pair);
        }

        canonical.sort();
        canonical.dedup();
        let active = vec![true; canonical.len()];
        let confidences = vec![1.0; canonical.len()];
        let (adjacency_offsets, adjacency) = Self::build_adjacency(view_count, &canonical);

        Ok(Self {
            view_count,
            pairs: canonical,
            active,
            confidences,
            adjacency_offsets,
            adjacency,
        })
    }

    /// Build the graph holding every unordered pair of `view_count` views.
    pub fn exhaustive(view_count: usize) -> Result<Self> {
        Self::try_new(
            view_count,
            (0..view_count).flat_map(|i| {
                ((i + 1)..view_count).map(move |j| ViewPair::new(ViewId::new(i), ViewId::new(j)))
            }),
        )
    }

    /// Build a graph over `view_count` views with no pairs.
    pub fn empty(view_count: usize) -> Self {
        Self {
            view_count,
            pairs: Vec::new(),
            active: Vec::new(),
            confidences: Vec::new(),
            adjacency_offsets: vec![0; view_count + 1],
            adjacency: Vec::new(),
        }
    }

    /// Number of views (graph vertices), including views without any pair.
    pub fn view_count(&self) -> usize {
        self.view_count
    }

    /// Number of pairs, active or deactivated.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether the graph has no pairs.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Every pair in canonical sorted order, active or deactivated.
    pub fn pairs(&self) -> &[ViewPair] {
        &self.pairs
    }

    /// Returns whether this graph contains the canonical view pair.
    #[must_use]
    pub fn contains_pair(&self, pair: ViewPair) -> bool {
        self.pair_id(pair.first(), pair.second()).is_some()
    }

    pub(crate) fn pair(&self, pair_id: PairId) -> Option<ViewPair> {
        self.pairs.get(pair_id.index()).copied()
    }

    pub(crate) fn pair_id(&self, view_i: ViewId, view_j: ViewId) -> Option<PairId> {
        if view_i == view_j {
            return None;
        }
        let pair = ViewPair::new(view_i, view_j);
        self.pairs.binary_search(&pair).ok().map(PairId::new)
    }

    pub(crate) fn is_active(&self, pair_id: PairId) -> bool {
        self.active.get(pair_id.index()).copied().unwrap_or(false)
    }

    /// Returns whether the canonical view pair is active in this graph.
    #[must_use]
    pub fn is_active_pair(&self, pair: ViewPair) -> bool {
        self.pair_id(pair.first(), pair.second())
            .is_some_and(|pair_id| self.is_active(pair_id))
    }

    /// Deactivates an existing canonical view pair.
    pub fn deactivate_pair(&mut self, pair: ViewPair) -> Result<()> {
        let pair_id = self.pair_id(pair.first(), pair.second()).ok_or_else(|| {
            SfmError::InvalidInput(format!("pair {pair} is not present in this graph"))
        })?;
        self.deactivate_pair_and(pair_id)
    }

    /// Reactivates an existing canonical view pair.
    pub fn activate_pair(&mut self, pair: ViewPair) -> Result<()> {
        let pair_id = self.pair_id(pair.first(), pair.second()).ok_or_else(|| {
            SfmError::InvalidInput(format!("pair {pair} is not present in this graph"))
        })?;
        self.activate_pair_and(pair_id)
    }

    /// Number of active pairs.
    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|is_active| **is_active).count()
    }

    /// Graph-level confidence multiplier shared by every downstream solver.
    /// Newly created edges have unit confidence.
    pub(crate) fn confidence(&self, pair_id: PairId) -> Option<f64> {
        self.confidences.get(pair_id.index()).copied()
    }

    /// Returns the graph-level confidence multiplier for a canonical view pair.
    #[must_use]
    pub fn confidence_for_pair(&self, pair: ViewPair) -> Option<f64> {
        let pair_id = self.pair_id(pair.first(), pair.second())?;
        self.confidence(pair_id)
    }

    /// Set a finite confidence in `[0, 1]` for an edge.
    pub(crate) fn set_confidence(&mut self, pair_id: PairId, confidence: f64) -> Result<()> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(SfmError::InvalidInput(format!(
                "pair confidence must be finite and in [0, 1], got {confidence}"
            )));
        }
        let slot = self
            .confidences
            .get_mut(pair_id.index())
            .ok_or(SfmError::InvalidId {
                kind: PairId::kind(),
                index: pair_id.index(),
            })?;
        *slot = confidence;
        Ok(())
    }

    /// Returns the length of the active mask (should equal `self.len()`).
    pub fn active_mask_len(&self) -> usize {
        self.active.len()
    }

    pub(crate) fn active_pair_ids(&self) -> impl Iterator<Item = PairId> + '_ {
        self.active
            .iter()
            .enumerate()
            .filter_map(|(index, is_active)| is_active.then_some(PairId::new(index)))
    }

    pub(crate) fn deactivate_pair_and(&mut self, pair_id: PairId) -> Result<()> {
        let slot = self
            .active
            .get_mut(pair_id.index())
            .ok_or(SfmError::InvalidId {
                kind: PairId::kind(),
                index: pair_id.index(),
            })?;
        *slot = false;
        Ok(())
    }

    /// Reactivate an existing pair without changing its stored matches/model.
    ///
    /// This is used when a conditioned pose-initialization subgraph is temporary:
    /// the full verified graph can then be restored for track reachability and P3P growth.
    pub(crate) fn activate_pair_and(&mut self, pair_id: PairId) -> Result<()> {
        let slot = self
            .active
            .get_mut(pair_id.index())
            .ok_or(SfmError::InvalidId {
                kind: PairId::kind(),
                index: pair_id.index(),
            })?;
        *slot = true;
        Ok(())
    }

    /// Views sharing a pair with `view_id`, active or deactivated; fails for an unknown view.
    pub fn neighbors(&self, view_id: ViewId) -> Result<&[Neighbor]> {
        if view_id.index() >= self.view_count {
            return Err(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: view_id.index(),
            });
        }
        let start = self.adjacency_offsets[view_id.index()];
        let end = self.adjacency_offsets[view_id.index() + 1];
        Ok(&self.adjacency[start..end])
    }

    /// Validate the graph's canonical storage and derived adjacency index.
    pub fn validate(&self) -> Result<()> {
        if self.active.len() != self.pairs.len() {
            return Err(SfmError::InvalidInput(format!(
                "pair active-mask length {} does not match pair count {}",
                self.active.len(),
                self.pairs.len()
            )));
        }
        if self.confidences.len() != self.pairs.len() {
            return Err(SfmError::InvalidInput(format!(
                "pair confidence length {} does not match pair count {}",
                self.confidences.len(),
                self.pairs.len()
            )));
        }
        for (index, pair) in self.pairs.iter().enumerate() {
            if pair.first() >= pair.second() {
                return Err(SfmError::InvalidInput(format!(
                    "pair {index} is not canonical"
                )));
            }
            if pair.second().index() >= self.view_count {
                return Err(SfmError::InvalidId {
                    kind: ViewId::kind(),
                    index: pair.second().index(),
                });
            }
            if index > 0 && self.pairs[index - 1] >= *pair {
                return Err(SfmError::InvalidInput(
                    "pair graph entries must be strictly sorted and unique".to_owned(),
                ));
            }
            let confidence = self.confidences[index];
            if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                return Err(SfmError::InvalidInput(format!(
                    "pair {index} has invalid graph confidence {confidence}"
                )));
            }
        }

        let (expected_offsets, expected_adjacency) =
            Self::build_adjacency(self.view_count, &self.pairs);
        if self.adjacency_offsets != expected_offsets || self.adjacency != expected_adjacency {
            return Err(SfmError::InvalidInput(
                "pair graph adjacency index is inconsistent with its pairs".to_owned(),
            ));
        }
        Ok(())
    }

    fn build_adjacency(view_count: usize, pairs: &[ViewPair]) -> (Vec<usize>, Vec<Neighbor>) {
        let mut buckets: Vec<Vec<Neighbor>> = vec![Vec::new(); view_count];
        for (pair_index, pair) in pairs.iter().enumerate() {
            let pair_id = PairId::new(pair_index);
            buckets[pair.first().index()].push(Neighbor {
                view: pair.second(),
                pair_id,
            });
            buckets[pair.second().index()].push(Neighbor {
                view: pair.first(),
                pair_id,
            });
        }

        let mut offsets = Vec::with_capacity(view_count + 1);
        let mut adjacency = Vec::with_capacity(pairs.len().saturating_mul(2));
        offsets.push(0);
        for mut bucket in buckets {
            bucket.sort_unstable_by_key(|neighbor| (neighbor.view, neighbor.pair_id));
            adjacency.extend(bucket);
            offsets.push(adjacency.len());
        }

        (offsets, adjacency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_is_bounded_and_defaults_to_one() {
        let mut graph = PairGraph::exhaustive(2).unwrap();
        let pair = PairId::new(0);
        assert_eq!(graph.confidence(pair), Some(1.0));
        graph.set_confidence(pair, 0.125).unwrap();
        assert_eq!(graph.confidence(pair), Some(0.125));
        assert!(graph.set_confidence(pair, -0.1).is_err());
        assert!(graph.set_confidence(pair, f64::NAN).is_err());
        assert!(graph.set_confidence(PairId::new(1), 0.5).is_err());
    }

    #[test]
    fn validation_detects_corrupt_derived_state() {
        let mut graph = PairGraph::exhaustive(3).unwrap();
        graph.adjacency.pop();
        assert!(graph.validate().is_err());
    }

    #[test]
    fn duplicate_pair_is_normalized() {
        let graph = PairGraph::try_new(
            3,
            [
                ViewPair::new(ViewId::new(0), ViewId::new(2)),
                ViewPair::new(ViewId::new(0), ViewId::new(2)),
            ],
        )
        .unwrap();

        assert_eq!(
            graph.pairs(),
            &[ViewPair::new(ViewId::new(0), ViewId::new(2))]
        );
    }

    #[test]
    fn reverse_duplicate_pair_is_normalized() {
        let graph = PairGraph::try_new(
            3,
            [
                ViewPair::new(ViewId::new(0), ViewId::new(2)),
                ViewPair::new(ViewId::new(2), ViewId::new(0)),
            ],
        )
        .unwrap();

        assert_eq!(
            graph.pairs(),
            &[ViewPair::new(ViewId::new(0), ViewId::new(2))]
        );
    }
}
