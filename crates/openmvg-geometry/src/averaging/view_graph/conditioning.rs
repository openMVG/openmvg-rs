// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Connectivity-preserving hard cut of low-consistency view-graph edges.
//!
//! Its teaching docs are on [`cut_edges_protecting_bridges`], where docs.rs shows them.

use openmvg_core::graph::{DenseNodeMap, GraphError, complete_connectivity};
use openmvg_core::model::ViewId;
use thiserror::Error;

/// Invalid input to [`cut_edges_protecting_bridges`].
///
/// Parallel edges and self-loops are accepted and keep their input identity; numeric inputs are
/// checked strictly so invalid scores cannot silently turn into an arbitrary ordering.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum EdgeCutError {
    /// Scores and endpoints must describe the same edge set.
    #[error("score count {scores} does not match endpoint count {endpoints}")]
    LengthMismatch {
        /// Number of supplied consistency scores.
        scores: usize,
        /// Number of supplied graph edges.
        endpoints: usize,
    },
    /// Every consistency score must be finite and lie in `[0, 1]`.
    #[error("score at edge {edge} must be finite and in [0, 1], got {value}")]
    InvalidScore {
        /// Index of the invalid edge score.
        edge: usize,
        /// Rejected score value.
        value: f64,
    },
    /// The cut threshold must be finite and lie in `[0, 1]`.
    #[error("cut threshold must be finite and in [0, 1], got {0}")]
    InvalidThreshold(f64),
    /// Dense endpoint indexing or connectivity-preserving completion failed.
    #[error(transparent)]
    Graph(#[from] GraphError),
}

/// Result of [`cut_edges_protecting_bridges`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EdgeCut {
    /// Per-edge keep mask; `false` means cut as a redundant low-score edge.
    pub keep: Vec<bool>,
    /// Number of edges cut.
    pub hard_removed_count: usize,
    /// Number of low-score candidates spared because cutting them would have disconnected the
    /// graph.
    pub protected_bridge_count: usize,
}

impl EdgeCut {
    /// Number of edges kept.
    #[inline]
    pub fn kept_count(&self) -> usize {
        self.keep.iter().filter(|&&k| k).count()
    }
}

/// Cut every edge scoring below `cut_below` that is not needed to keep the graph connected.
///
/// - `scores[i]` in `[0, 1]`: per-edge consistency score (1 = fully consistent).
/// - `endpoints[i]`: the `(view_i, view_j)` of edge `i` (arbitrary [`ViewId`]s).
///
/// The kept subgraph never loses connectivity relative to the input.
///
/// # How it works
///
/// A hard cut is only safe where the edge is redundant: cutting a bridge orphans views and
/// silently shrinks coverage. [`cut_edges_protecting_bridges`] removes every edge whose
/// consistency score is below a threshold except those the graph needs to stay connected.
///
/// # Algorithm
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │                      Bridge-protecting edge cut                      │
/// └──────────────────────────────────────────────────────────────────────┘
/// Input: scores s_e in [0, 1], endpoints (i, j)
///                                   │
///                                   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 1: Candidates = edges with s_e < cut_below, worst first         │
/// └──────────────────────────────────────────────────────────────────────┘
///                                   │
///                                   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 2: Disjoint-set forest seeded with every non-candidate edge     │
/// └──────────────────────────────────────────────────────────────────────┘
///                                   │
///                                   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 3: Visit candidates best first (reverse of worst-first order):  │
/// │   joins two components -> keep (a bridge: removal would disconnect)  │
/// │   otherwise            -> cut (redundant, sits inside a cycle)       │
/// └──────────────────────────────────────────────────────────────────────┘
///                                   │
///                                   ▼
/// Output: keep mask, hard_removed_count, protected_bridge_count
/// ```
///
/// This equals deleting candidates worst first while protecting any edge whose removal would break
/// the connectivity of the edges kept so far, in `O(E log E)` for the ordering plus a near-linear
/// disjoint-set pass.
///
/// # Errors
///
/// [`EdgeCutError::LengthMismatch`] when `scores` and `endpoints` differ in length,
/// [`EdgeCutError::InvalidScore`] for a score outside `[0, 1]`,
/// [`EdgeCutError::InvalidThreshold`] for a threshold outside `[0, 1]`, and
/// [`EdgeCutError::Graph`] if the endpoints cannot be indexed.
pub fn cut_edges_protecting_bridges(
    scores: &[f64],
    endpoints: &[(ViewId, ViewId)],
    cut_below: f64,
) -> Result<EdgeCut, EdgeCutError> {
    let n_edges = scores.len();
    if n_edges != endpoints.len() {
        return Err(EdgeCutError::LengthMismatch {
            scores: n_edges,
            endpoints: endpoints.len(),
        });
    }
    if !cut_below.is_finite() || !(0.0..=1.0).contains(&cut_below) {
        return Err(EdgeCutError::InvalidThreshold(cut_below));
    }
    if let Some((edge, &value)) = scores
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || !(0.0..=1.0).contains(*value))
    {
        return Err(EdgeCutError::InvalidScore { edge, value });
    }
    let mut keep = vec![true; n_edges];
    // Candidates: low-score edges, worst first.
    let mut candidates: Vec<usize> = (0..n_edges).filter(|&i| scores[i] < cut_below).collect();
    if candidates.is_empty() {
        return Ok(EdgeCut {
            keep,
            hard_removed_count: 0,
            protected_bridge_count: 0,
        });
    }
    candidates.sort_by(|&a, &b| {
        let canonical = |edge: usize| {
            let (first, second) = endpoints[edge];
            (first.min(second), first.max(second))
        };
        scores[a]
            .total_cmp(&scores[b])
            .then_with(|| canonical(a).cmp(&canonical(b)))
            // Parallel edges retain distinct input identity by contract.
            .then(a.cmp(&b))
    });

    let endpoint_indices: Vec<_> = endpoints
        .iter()
        .map(|&(first, second)| (first.index(), second.index()))
        .collect();
    let nodes = DenseNodeMap::try_from_endpoints(&endpoint_indices)?;
    let dense_endpoints = nodes.try_remap_endpoints(&endpoint_indices)?;
    for &candidate in &candidates {
        keep[candidate] = false;
    }
    // Reverse of the worst-first deletion order: the best-scoring candidate that is necessary to
    // restore connectivity is kept; redundant lower-score candidates stay removed.
    candidates.reverse();
    let completion =
        complete_connectivity(nodes.node_count(), &dense_endpoints, &keep, &candidates)?;
    let protected_bridge_count = completion.added_edges().len();
    Ok(EdgeCut {
        hard_removed_count: candidates.len() - protected_bridge_count,
        protected_bridge_count,
        keep: completion.into_keep(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(first: usize, second: usize) -> (ViewId, ViewId) {
        (ViewId::new(first), ViewId::new(second))
    }

    fn reference_reverse_delete(
        scores: &[f64],
        endpoints: &[(ViewId, ViewId)],
        threshold: f64,
    ) -> Vec<bool> {
        let mut keep = vec![true; endpoints.len()];
        let mut candidates: Vec<_> = (0..endpoints.len())
            .filter(|&index| scores[index] < threshold)
            .collect();
        candidates.sort_by(|&first, &second| {
            let canonical = |edge: usize| {
                let (first, second) = endpoints[edge];
                (first.min(second), first.max(second))
            };
            scores[first]
                .total_cmp(&scores[second])
                .then_with(|| canonical(first).cmp(&canonical(second)))
                .then(first.cmp(&second))
        });

        for candidate in candidates {
            keep[candidate] = false;
            let (source, target) = endpoints[candidate];
            let mut stack = vec![source];
            let mut visited = std::collections::BTreeSet::from([source]);
            while let Some(node) = stack.pop() {
                for (edge, &(first, second)) in endpoints.iter().enumerate() {
                    if !keep[edge] {
                        continue;
                    }
                    let neighbor = if first == node {
                        second
                    } else if second == node {
                        first
                    } else {
                        continue;
                    };
                    if visited.insert(neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
            if !visited.contains(&target) {
                keep[candidate] = true;
            }
        }
        keep
    }

    /// Two 5-node cliques (K5, 10 edges each) joined by a single bridge, each side thickened by a
    /// hub node. Returns the endpoints and the index of the sole A-B bridge `(4, 5)`.
    fn two_cliques_with_bridge() -> (Vec<(ViewId, ViewId)>, usize) {
        let mut endpoints: Vec<(ViewId, ViewId)> = Vec::new();
        for i in 0..5usize {
            for j in (i + 1)..5 {
                endpoints.push(pair(i, j));
            }
        }
        for i in 5..10usize {
            for j in (i + 1)..10 {
                endpoints.push(pair(i, j));
            }
        }
        endpoints.push(pair(4, 5));
        let bridge = endpoints.len() - 1;
        // Hubs join each clique without adding a second A-B link.
        for a in 0..5usize {
            endpoints.push(pair(10, a));
        }
        for b in 5..10usize {
            endpoints.push(pair(11, b));
        }
        (endpoints, bridge)
    }

    /// A low-score bridge and a low-score in-cycle edge: the cycle edge is cut, the bridge kept.
    #[test]
    fn a_low_score_cycle_edge_is_cut_and_a_low_score_bridge_is_protected() {
        let (endpoints, bridge) = two_cliques_with_bridge();
        let mut scores = vec![0.9; endpoints.len()];
        let cycle_edge = endpoints.iter().position(|&e| e == pair(1, 2)).unwrap();
        scores[cycle_edge] = 0.01;
        assert_eq!(endpoints[bridge], pair(4, 5));
        scores[bridge] = 0.02;

        let cut = cut_edges_protecting_bridges(&scores, &endpoints, 0.05).unwrap();

        assert!(
            !cut.keep[cycle_edge],
            "low-score in-cycle edge should be cut"
        );
        assert!(cut.keep[bridge], "low-score bridge must be protected");
        assert_eq!(cut.hard_removed_count, 1);
        assert_eq!(cut.protected_bridge_count, 1);
        assert_eq!(cut.kept_count(), endpoints.len() - 1);
    }

    /// A sparse path has only bridges: nothing can be cut however low the scores.
    #[test]
    fn a_path_graph_loses_nothing() {
        let endpoints = vec![pair(0, 1), pair(1, 2), pair(2, 3), pair(3, 4), pair(4, 5)];
        let mut scores = vec![0.9; endpoints.len()];
        scores[2] = 0.001;
        let cut = cut_edges_protecting_bridges(&scores, &endpoints, 0.05).unwrap();
        assert!(cut.keep.iter().all(|&k| k));
        assert_eq!(cut.hard_removed_count, 0);
        assert_eq!(cut.protected_bridge_count, 1);
    }

    /// A clean graph keeps every edge.
    #[test]
    fn a_clean_graph_keeps_everything() {
        let endpoints = vec![
            pair(0, 1),
            pair(0, 2),
            pair(0, 3),
            pair(1, 2),
            pair(1, 3),
            pair(2, 3),
        ];
        let cut = cut_edges_protecting_bridges(&[0.95; 6], &endpoints, 0.05).unwrap();
        assert!(cut.keep.iter().all(|&k| k));
        assert_eq!(cut.hard_removed_count, 0);
        assert_eq!(cut.protected_bridge_count, 0);
    }

    /// Two low-score edges: worst-first with the current kept set removes the first (redundant) and
    /// protects the second, which becomes a bridge after the first removal.
    #[test]
    fn the_second_edge_of_a_two_edge_cut_is_protected() {
        let mut endpoints: Vec<(ViewId, ViewId)> = Vec::new();
        for i in 0..5usize {
            for j in (i + 1)..5 {
                endpoints.push(pair(i, j));
            }
        }
        for i in 6..11usize {
            for j in (i + 1)..11 {
                endpoints.push(pair(i, j));
            }
        }
        endpoints.push(pair(3, 5));
        let e35 = endpoints.len() - 1;
        endpoints.push(pair(2, 5));
        endpoints.push(pair(5, 6));
        let e56 = endpoints.len() - 1;
        for a in 0..5usize {
            endpoints.push(pair(11, a));
        }
        for b in 6..11usize {
            endpoints.push(pair(12, b));
        }
        let mut scores = vec![0.9; endpoints.len()];
        scores[e35] = 0.005;
        scores[e56] = 0.01;

        let cut = cut_edges_protecting_bridges(&scores, &endpoints, 0.05).unwrap();

        assert!(!cut.keep[e35], "redundant low-score edge removed");
        assert!(cut.keep[e56], "sole B-connector must be protected");
        assert!(cut.protected_bridge_count >= 1);
    }

    #[test]
    fn reverse_kruskal_matches_worst_first_deletion_exhaustively() {
        let endpoints = vec![
            pair(0, 1),
            pair(0, 2),
            pair(0, 3),
            pair(1, 2),
            pair(1, 3),
            pair(2, 3),
            pair(2, 3), // Parallel measurements must remain distinct edges.
            pair(3, 3), // A self-loop is always redundant.
        ];
        let threshold = 0.5;
        for candidate_mask in 0usize..(1usize << endpoints.len()) {
            let scores: Vec<_> = (0..endpoints.len())
                .map(|edge| {
                    if candidate_mask & (1 << edge) != 0 {
                        0.01 + edge as f64 * 0.001
                    } else {
                        0.9
                    }
                })
                .collect();
            let expected = reference_reverse_delete(&scores, &endpoints, threshold);
            let actual = cut_edges_protecting_bridges(&scores, &endpoints, threshold).unwrap();
            assert_eq!(actual.keep, expected, "candidate mask {candidate_mask:#x}");
        }
    }

    #[test]
    fn empty_input_cuts_nothing() {
        let cut = cut_edges_protecting_bridges(&[], &[], 0.05).unwrap();
        assert!(cut.keep.is_empty());
        assert_eq!(cut.hard_removed_count, 0);
        assert_eq!(cut.protected_bridge_count, 0);
    }

    #[test]
    fn invalid_input_is_rejected() {
        assert!(matches!(
            cut_edges_protecting_bridges(&[0.5], &[], 0.05),
            Err(EdgeCutError::LengthMismatch {
                scores: 1,
                endpoints: 0,
            })
        ));
        assert!(matches!(
            cut_edges_protecting_bridges(&[f64::NAN], &[pair(0, 1)], 0.05),
            Err(EdgeCutError::InvalidScore { edge: 0, value }) if value.is_nan()
        ));
        assert!(matches!(
            cut_edges_protecting_bridges(&[0.5], &[pair(0, 1)], 1.5),
            Err(EdgeCutError::InvalidThreshold(_))
        ));
    }
}
