// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::GraphError;
use super::alloc::reserved_vec;

/// Dense disjoint-set union with path halving and union by size.
///
/// Nodes are the integers in `0..len`. The structure tracks component count and component
/// sizes, making it suitable for Kruskal's spanning-tree algorithm and incremental graph
/// connectivity without embedding application payloads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseDisjointSets {
    parents: Vec<usize>,
    sizes: Vec<usize>,
    component_count: usize,
}

impl DenseDisjointSets {
    /// Construct one singleton set per dense node.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::AllocationFailed`] when the sets cannot be allocated.
    pub fn try_new(len: usize) -> Result<Self, GraphError> {
        let mut parents = reserved_vec(len)?;
        parents.extend(0..len);
        let mut sizes = reserved_vec(len)?;
        sizes.resize(len, 1);
        Ok(Self {
            parents,
            sizes,
            component_count: len,
        })
    }

    /// Number of represented nodes.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.parents.len()
    }

    /// Whether no nodes are represented.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }

    /// Current number of disjoint components.
    #[inline]
    #[must_use]
    pub const fn component_count(&self) -> usize {
        self.component_count
    }

    /// Representative of the set containing `node`, with path halving.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::NodeOutOfBounds`] when `node` is not below [`Self::len`].
    pub fn find(&mut self, node: usize) -> Result<usize, GraphError> {
        self.validate_node(node)?;
        Ok(self.find_valid(node))
    }

    /// Merge two sets, returning `true` when they were distinct before the call.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::NodeOutOfBounds`] when either node is not below [`Self::len`].
    pub fn union(&mut self, first: usize, second: usize) -> Result<bool, GraphError> {
        self.validate_node(first)?;
        self.validate_node(second)?;
        Ok(self.union_valid(first, second))
    }

    pub(super) fn union_valid(&mut self, first: usize, second: usize) -> bool {
        let mut first_root = self.find_valid(first);
        let mut second_root = self.find_valid(second);
        if first_root == second_root {
            return false;
        }
        if self.sizes[first_root] < self.sizes[second_root] {
            std::mem::swap(&mut first_root, &mut second_root);
        }
        self.parents[second_root] = first_root;
        self.sizes[first_root] += self.sizes[second_root];
        self.component_count -= 1;
        true
    }

    fn validate_node(&self, node: usize) -> Result<(), GraphError> {
        if node < self.len() {
            Ok(())
        } else {
            Err(GraphError::NodeOutOfBounds {
                node,
                node_count: self.len(),
            })
        }
    }

    pub(super) fn find_valid(&mut self, mut node: usize) -> usize {
        while self.parents[node] != node {
            self.parents[node] = self.parents[self.parents[node]];
            node = self.parents[node];
        }
        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_tracks_sizes_and_component_count() {
        let mut sets = DenseDisjointSets::try_new(5).unwrap();
        assert_eq!(sets.component_count(), 5);
        assert!(sets.union(0, 1).unwrap());
        assert!(sets.union(1, 2).unwrap());
        assert!(!sets.union(2, 0).unwrap());
        assert_eq!(sets.component_count(), 3);
        assert_eq!(sets.find(0).unwrap(), sets.find(2).unwrap());
    }

    #[test]
    fn invalid_nodes_are_reported() {
        let mut sets = DenseDisjointSets::try_new(2).unwrap();
        assert_eq!(
            sets.union(0, 2),
            Err(GraphError::NodeOutOfBounds {
                node: 2,
                node_count: 2,
            })
        );
    }
}
