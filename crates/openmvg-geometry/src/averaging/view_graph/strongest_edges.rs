// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Strongest-edges-per-view filter: keep, for every view, its best verified pairs by inliers.
//!
//! Its teaching docs are on [`keep_strongest_edges_per_view`], where docs.rs shows them.

use std::collections::BTreeMap;

use openmvg_core::model::ViewId;
use thiserror::Error;

/// One verified pair as seen by the filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrengthEdge {
    /// First view of the pair.
    pub view_i: ViewId,
    /// Second view of the pair; must differ from `view_i`.
    pub view_j: ViewId,
    /// Geometric inliers of the verified pair.
    pub inliers: usize,
}

/// Failure of [`keep_strongest_edges_per_view`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum StrongestEdgesError {
    /// An input edge joins a view to itself.
    #[error("edge {0} joins a view to itself")]
    SelfLoop(usize),
    /// The per-view budget must be positive.
    #[error("the per-view edge budget must be positive")]
    ZeroBudget,
}

/// Mark the edges that are among the `max_edges_per_view` strongest at either endpoint.
///
/// Edges are ranked at each view by descending inliers, then by ascending `(view_i, view_j)`,
/// so the result is deterministic and independent of input order. The returned mask is in input
/// order.
///
/// # How it works
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │                    Strongest edges per view filter                   │
/// └──────────────────────────────────────────────────────────────────────┘
/// Input: verified edges (view_i, view_j, inlier count), budget K per view
///                                   │
///                                   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 1: Rank the edges at each view by inliers, ties by view ids     │
/// └──────────────────────────────────────────────────────────────────────┘
///                                   │
///                                   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 2: An edge stays when it is in the top K at either endpoint     │
/// └──────────────────────────────────────────────────────────────────────┘
///                                   │
///                                   ▼
/// Output: keep mask (a view of degree <= K keeps all its edges)
/// ```
///
/// Dense graphs of overlapping photographs carry many weak verified pairs, a large share of them
/// inconsistent with the scene. The pairs a view trusts most are its strongest, so keeping
/// each view's `K` best removes the weak pairs that chain unrelated parts of a scene together
/// while every view keeps its best neighbors. A view with at most `K` edges is untouched, so
/// small or sparse graphs are unchanged; the filter never needs a threshold on the absolute
/// inlier count.
///
/// The union rule does not keep the graph connected. The engines already work on the largest
/// component of what remains.
///
/// # Errors
///
/// [`StrongestEdgesError::SelfLoop`] for an edge from a view to itself and
/// [`StrongestEdgesError::ZeroBudget`] for a zero budget.
pub fn keep_strongest_edges_per_view(
    edges: &[StrengthEdge],
    max_edges_per_view: usize,
) -> Result<Vec<bool>, StrongestEdgesError> {
    if max_edges_per_view == 0 {
        return Err(StrongestEdgesError::ZeroBudget);
    }
    let mut incident: BTreeMap<ViewId, Vec<usize>> = BTreeMap::new();
    for (index, edge) in edges.iter().enumerate() {
        if edge.view_i == edge.view_j {
            return Err(StrongestEdgesError::SelfLoop(index));
        }
        incident.entry(edge.view_i).or_default().push(index);
        incident.entry(edge.view_j).or_default().push(index);
    }
    let mut keep = vec![false; edges.len()];
    for list in incident.values_mut() {
        list.sort_by_key(|&index| {
            let edge = &edges[index];
            (
                std::cmp::Reverse(edge.inliers),
                edge.view_i.min(edge.view_j),
                edge.view_i.max(edge.view_j),
            )
        });
        for &index in list.iter().take(max_edges_per_view) {
            keep[index] = true;
        }
    }
    Ok(keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(i: usize, j: usize, inliers: usize) -> StrengthEdge {
        StrengthEdge {
            view_i: ViewId::new(i),
            view_j: ViewId::new(j),
            inliers,
        }
    }

    #[test]
    fn a_view_with_at_most_k_edges_keeps_all_of_them() {
        let edges = [edge(0, 1, 5), edge(0, 2, 9), edge(1, 2, 1)];
        assert_eq!(keep_strongest_edges_per_view(&edges, 2).unwrap(), [true; 3]);
        assert_eq!(
            keep_strongest_edges_per_view(&edges, 10).unwrap(),
            [true; 3]
        );
    }

    #[test]
    fn the_weakest_edge_of_a_crowded_view_is_dropped_unless_its_other_end_keeps_it() {
        // View 0 has three edges; with K = 2 its weakest (0,3) is dropped, but view 3 only
        // has that edge, so the union rule keeps it. (0,1) and (0,2) are the strong ones.
        let edges = [edge(0, 1, 50), edge(0, 2, 40), edge(0, 3, 3)];
        assert_eq!(
            keep_strongest_edges_per_view(&edges, 2).unwrap(),
            [true, true, true]
        );
        // Give view 3 two stronger edges: now (0,3) is the weakest at both ends.
        let crowded = [
            edge(0, 1, 50),
            edge(0, 2, 40),
            edge(0, 3, 3),
            edge(3, 1, 60),
            edge(3, 2, 70),
        ];
        let keep = keep_strongest_edges_per_view(&crowded, 2).unwrap();
        assert_eq!(keep, [true, true, false, true, true]);
    }

    #[test]
    fn ties_break_on_the_view_ids_and_input_order_does_not_matter() {
        let edges = [
            edge(0, 1, 10),
            edge(0, 2, 10),
            edge(0, 3, 10),
            edge(1, 2, 90),
            edge(1, 3, 80),
            edge(2, 3, 70),
        ];
        let keep = keep_strongest_edges_per_view(&edges, 1).unwrap();
        let mut shuffled: Vec<StrengthEdge> = edges.iter().rev().copied().collect();
        for e in &mut shuffled {
            std::mem::swap(&mut e.view_i, &mut e.view_j);
        }
        let again = keep_strongest_edges_per_view(&shuffled, 1).unwrap();
        for (k, kept) in keep.iter().enumerate() {
            assert_eq!(*kept, again[edges.len() - 1 - k]);
        }
        // View 0's three ties keep the smallest ids: (0,1); (0,2) and (0,3) survive only
        // through the strong edges of their other endpoints (2,3 keep (1,2) and (1,3)).
        assert!(keep[0]);
    }

    #[test]
    fn every_view_keeps_min_k_and_its_degree_edges() {
        let mut edges = Vec::new();
        for i in 0..12 {
            for j in (i + 1)..12 {
                edges.push(edge(i, j, 1 + (i * 7 + j * 13) % 50));
            }
        }
        let keep = keep_strongest_edges_per_view(&edges, 4).unwrap();
        for view in 0..12 {
            let kept = edges
                .iter()
                .zip(&keep)
                .filter(|(e, k)| **k && (e.view_i.index() == view || e.view_j.index() == view))
                .count();
            assert!(kept >= 4, "view {view} keeps {kept}");
        }
    }

    #[test]
    fn invalid_input_is_rejected_and_empty_input_is_handled() {
        assert_eq!(
            keep_strongest_edges_per_view(&[edge(2, 2, 5)], 3),
            Err(StrongestEdgesError::SelfLoop(0))
        );
        assert_eq!(
            keep_strongest_edges_per_view(&[edge(0, 1, 5)], 0),
            Err(StrongestEdgesError::ZeroBudget)
        );
        assert!(keep_strongest_edges_per_view(&[], 5).unwrap().is_empty());
    }
}
