// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::GraphError;
use super::alloc::reserved_vec;

/// Sorted unique external node identifiers and their dense indexing.
///
/// Construction accepts parallel pairs and self-loops because remapping itself
/// has no graph-multiplicity policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseNodeMap {
    original_ids: Vec<usize>,
}

impl DenseNodeMap {
    /// Build a deterministic map from all identifiers present in `endpoints`.
    ///
    /// Dense identifiers follow increasing external identifier order and are
    /// therefore independent of edge order and endpoint orientation.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::CapacityExceeded`] when twice the edge count overflows `usize`
    /// and [`GraphError::AllocationFailed`] when the identifiers cannot be stored.
    pub fn try_from_endpoints(endpoints: &[(usize, usize)]) -> Result<Self, GraphError> {
        let identifier_count = checked_endpoint_count(endpoints.len())?;
        let mut ids = reserved_vec(identifier_count)?;
        for &(first, second) in endpoints {
            ids.push(first);
            ids.push(second);
        }
        ids.sort_unstable();
        ids.dedup();
        if ids.len() < identifier_count {
            let mut compact_ids = reserved_vec(ids.len())?;
            compact_ids.extend(ids);
            ids = compact_ids;
        }
        Ok(Self { original_ids: ids })
    }

    /// Number of distinct indexed nodes.
    #[inline]
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.original_ids.len()
    }

    /// Whether no node identifiers were supplied.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.original_ids.is_empty()
    }

    /// Sorted external identifiers, indexed by dense node ID.
    #[inline]
    #[must_use]
    pub fn original_ids(&self) -> &[usize] {
        &self.original_ids
    }

    /// Dense identifier for an external identifier, if present.
    #[inline]
    #[must_use]
    pub fn dense_id(&self, original_id: usize) -> Option<usize> {
        self.original_ids.binary_search(&original_id).ok()
    }

    /// External identifier for a dense identifier, if it is in range.
    #[inline]
    #[must_use]
    pub fn original_id(&self, dense_id: usize) -> Option<usize> {
        self.original_ids.get(dense_id).copied()
    }

    /// Remap endpoint identifiers through this node map, preserving edge order
    /// and input orientation.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::NodeNotIndexed`] for an identifier absent from the map and
    /// [`GraphError::AllocationFailed`] when the result cannot be allocated.
    pub fn try_remap_endpoints(
        &self,
        endpoints: &[(usize, usize)],
    ) -> Result<Vec<(usize, usize)>, GraphError> {
        let mut dense = reserved_vec(endpoints.len())?;
        for (edge, &(first, second)) in endpoints.iter().enumerate() {
            let first = self
                .dense_id(first)
                .ok_or(GraphError::NodeNotIndexed { edge, node: first })?;
            let second = self
                .dense_id(second)
                .ok_or(GraphError::NodeNotIndexed { edge, node: second })?;
            dense.push((first, second));
        }
        Ok(dense)
    }
}

fn checked_endpoint_count(num_edges: usize) -> Result<usize, GraphError> {
    num_edges
        .checked_mul(2)
        .ok_or(GraphError::CapacityExceeded {
            node_count: 0,
            edge_count: num_edges,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_map_is_independent_of_order_orientation_and_multiplicity() {
        let first = DenseNodeMap::try_from_endpoints(&[(8, 2), (5, 8), (8, 2)]).unwrap();
        let second = DenseNodeMap::try_from_endpoints(&[(8, 5), (2, 8)]).unwrap();
        assert_eq!(first, second);
        for (dense, &original) in first.original_ids().iter().enumerate() {
            assert_eq!(first.dense_id(original), Some(dense));
            assert_eq!(first.original_id(dense), Some(original));
        }
        assert_eq!(first.try_remap_endpoints(&[(8, 2)]), Ok(vec![(2, 0)]));
        assert_eq!(
            first.try_remap_endpoints(&[(8, 9)]),
            Err(GraphError::NodeNotIndexed { edge: 0, node: 9 })
        );
    }

    #[test]
    fn endpoint_arithmetic_overflow_is_rejected_before_allocation() {
        assert_eq!(
            checked_endpoint_count(usize::MAX),
            Err(GraphError::CapacityExceeded {
                node_count: 0,
                edge_count: usize::MAX,
            })
        );
    }
}
