// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::alloc::{allocate_filled, reserved_vec};
use super::{DenseDisjointSets, GraphError};

/// Connectivity-preserving completion of an initially retained edge set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectivityCompletion {
    keep: Vec<bool>,
    added_edges: Vec<usize>,
    component_count: usize,
}

impl ConnectivityCompletion {
    /// Final keep mask in input edge order.
    #[inline]
    #[must_use]
    pub fn keep(&self) -> &[bool] {
        &self.keep
    }

    /// Candidate edges re-added to restore input connectivity, in selection order.
    #[inline]
    #[must_use]
    pub fn added_edges(&self) -> &[usize] {
        &self.added_edges
    }

    /// Number of connected components in the full input graph, including isolates.
    #[inline]
    #[must_use]
    pub const fn component_count(&self) -> usize {
        self.component_count
    }

    /// Consume the result and return its final keep mask.
    #[must_use]
    pub fn into_keep(self) -> Vec<bool> {
        self.keep
    }
}

/// Re-add the minimum ordered subset of dropped edges needed to match the full
/// input graph's connected components.
///
/// `candidate_order` must contain every initially dropped edge exactly once,
/// ordered from most to least desirable. Parallel edges and self-loops retain
/// their input identity; self-loops are never needed for connectivity.
///
/// # Errors
///
/// Returns [`GraphError::CountMismatch`] when `initially_kept` does not have one entry per
/// edge, [`GraphError::EndpointOutOfBounds`] for an endpoint outside `0..node_count`,
/// [`GraphError::CandidateOutOfBounds`], [`GraphError::CandidateAlreadyKept`],
/// [`GraphError::DuplicateCandidate`], or [`GraphError::MissingCandidate`] when
/// `candidate_order` is not exactly the dropped edges, and [`GraphError::AllocationFailed`]
/// when the workspace cannot be allocated.
pub fn complete_connectivity(
    node_count: usize,
    endpoints: &[(usize, usize)],
    initially_kept: &[bool],
    candidate_order: &[usize],
) -> Result<ConnectivityCompletion, GraphError> {
    if endpoints.len() != initially_kept.len() {
        return Err(GraphError::CountMismatch {
            what: "keep flags",
            expected: endpoints.len(),
            found: initially_kept.len(),
        });
    }
    for (edge, &(first, second)) in endpoints.iter().enumerate() {
        for endpoint in [first, second] {
            if endpoint >= node_count {
                return Err(GraphError::EndpointOutOfBounds {
                    edge,
                    endpoint,
                    node_count,
                });
            }
        }
    }

    let mut seen = allocate_filled(endpoints.len(), false)?;
    for (candidate, &edge) in candidate_order.iter().enumerate() {
        if edge >= endpoints.len() {
            return Err(GraphError::CandidateOutOfBounds { candidate, edge });
        }
        if initially_kept[edge] {
            return Err(GraphError::CandidateAlreadyKept { edge });
        }
        if std::mem::replace(&mut seen[edge], true) {
            return Err(GraphError::DuplicateCandidate { edge });
        }
    }
    if let Some(edge) = initially_kept
        .iter()
        .zip(&seen)
        .position(|(&keep, &seen)| !keep && !seen)
    {
        return Err(GraphError::MissingCandidate { edge });
    }
    drop(seen);
    if endpoints.is_empty() {
        return Ok(ConnectivityCompletion {
            keep: Vec::new(),
            added_edges: Vec::new(),
            component_count: node_count,
        });
    }

    let mut sets = DenseDisjointSets::try_new(node_count)?;
    let mut keep = reserved_vec(initially_kept.len())?;
    keep.extend_from_slice(initially_kept);
    for (&(first, second), &is_kept) in endpoints.iter().zip(initially_kept) {
        if is_kept && first != second {
            sets.union_valid(first, second);
        }
    }
    let mut added_edges = reserved_vec(candidate_order.len().min(node_count.saturating_sub(1)))?;
    for &edge in candidate_order {
        let (first, second) = endpoints[edge];
        if first != second && sets.union_valid(first, second) {
            keep[edge] = true;
            added_edges.push(edge);
        }
    }
    Ok(ConnectivityCompletion {
        keep,
        added_edges,
        component_count: sets.component_count(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restores_each_input_component_in_preference_order() {
        let endpoints = [(0, 1), (1, 2), (0, 2), (3, 4)];
        let result =
            complete_connectivity(5, &endpoints, &[true, false, false, false], &[2, 1, 3]).unwrap();
        assert_eq!(result.keep(), &[true, false, true, true]);
        assert_eq!(result.added_edges(), &[2, 3]);
        assert_eq!(result.component_count(), 2);
    }

    #[test]
    fn validates_candidate_permutation() {
        let endpoints = [(0, 1), (1, 2)];
        assert_eq!(
            complete_connectivity(3, &endpoints, &[true, false], &[]),
            Err(GraphError::MissingCandidate { edge: 1 })
        );
        assert_eq!(
            complete_connectivity(3, &endpoints, &[true, false], &[1, 1]),
            Err(GraphError::DuplicateCandidate { edge: 1 })
        );
    }
}
