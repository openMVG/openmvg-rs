// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::alloc::{allocate_filled, reserved_vec};
use super::{DenseDisjointSets, GraphError};

/// Stable connected-component labels for a dense undirected graph.
///
/// Labels are dense in `0..component_count` and ordered by each component's
/// smallest node, so they are invariant to endpoint order and orientation.
/// Isolated nodes are represented as singleton components. Self-loops and
/// parallel edges are accepted because neither changes connectivity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentPartition {
    labels: Vec<usize>,
    sizes: Vec<usize>,
}

impl ComponentPartition {
    /// Number of represented nodes.
    #[inline]
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.labels.len()
    }

    /// Number of connected components, including isolates.
    #[inline]
    #[must_use]
    pub const fn component_count(&self) -> usize {
        self.sizes.len()
    }

    /// Stable component label for every dense node.
    #[inline]
    #[must_use]
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Stable component label for one node, if it is in range.
    #[inline]
    #[must_use]
    pub fn label(&self, node: usize) -> Option<usize> {
        self.labels.get(node).copied()
    }

    /// Component sizes indexed by stable label.
    #[inline]
    #[must_use]
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Size of one component label, if it is in range.
    #[inline]
    #[must_use]
    pub fn size(&self, component: usize) -> Option<usize> {
        self.sizes.get(component).copied()
    }

    /// Largest component label, breaking size ties by smallest member node.
    #[must_use]
    pub fn largest_component(&self) -> Option<usize> {
        self.sizes
            .iter()
            .enumerate()
            .max_by(|(left_label, left_size), (right_label, right_size)| {
                left_size
                    .cmp(right_size)
                    .then_with(|| right_label.cmp(left_label))
            })
            .map(|(label, _)| label)
    }
}

/// Compute stable connected-component labels for dense undirected endpoints.
///
/// # Errors
///
/// Returns [`GraphError::EndpointOutOfBounds`] for an endpoint outside `0..node_count` and
/// [`GraphError::AllocationFailed`] when the workspace cannot be allocated.
pub fn component_partition(
    node_count: usize,
    endpoints: &[(usize, usize)],
) -> Result<ComponentPartition, GraphError> {
    if endpoints.is_empty() {
        let mut labels = reserved_vec(node_count)?;
        labels.extend(0..node_count);
        return Ok(ComponentPartition {
            labels,
            sizes: allocate_filled(node_count, 1usize)?,
        });
    }
    let mut sets = union_endpoints(node_count, endpoints)?;
    let mut label_by_root = allocate_filled(node_count, usize::MAX)?;
    let mut labels = reserved_vec(node_count)?;
    let mut sizes = reserved_vec(sets.component_count())?;
    for node in 0..node_count {
        let root = sets.find_valid(node);
        let label = if label_by_root[root] == usize::MAX {
            let label = sizes.len();
            label_by_root[root] = label;
            sizes.push(0);
            label
        } else {
            label_by_root[root]
        };
        labels.push(label);
        sizes[label] += 1;
    }
    Ok(ComponentPartition { labels, sizes })
}

fn union_endpoints(
    node_count: usize,
    endpoints: &[(usize, usize)],
) -> Result<DenseDisjointSets, GraphError> {
    let mut sets = DenseDisjointSets::try_new(node_count)?;
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
        sets.union_valid(first, second);
    }
    Ok(sets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_stable_and_include_isolates() {
        let partition = component_partition(6, &[(4, 2), (1, 0), (2, 3)]).unwrap();
        assert_eq!(partition.labels(), &[0, 0, 1, 1, 1, 2]);
        assert_eq!(partition.sizes(), &[2, 3, 1]);
        assert_eq!(partition.largest_component(), Some(1));
    }

    #[test]
    fn largest_component_ties_use_smallest_member() {
        let partition = component_partition(4, &[(2, 3), (0, 1)]).unwrap();
        assert_eq!(partition.largest_component(), Some(0));
    }

    #[test]
    fn self_loops_and_parallel_edges_do_not_change_connectivity() {
        let partition = component_partition(3, &[(0, 0), (0, 1), (1, 0)]).unwrap();
        assert_eq!(partition.labels(), &[0, 0, 1]);
        assert_eq!(partition.component_count(), 2);
    }
}
