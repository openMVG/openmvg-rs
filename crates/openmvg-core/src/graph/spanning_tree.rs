// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::alloc::{allocate_filled, reserved_vec};
use super::{DenseDisjointSets, GraphError};

/// One root-oriented edge of a spanning tree.
///
/// Entries returned by [`maximum_spanning_tree`] are ordered so that every
/// parent precedes its children. `edge_index` refers to the corresponding
/// entry in the input endpoint and weight slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanningTreeEdge {
    edge_index: usize,
    parent: usize,
    child: usize,
}

impl SpanningTreeEdge {
    /// Index of the selected input edge.
    #[inline]
    #[must_use]
    pub const fn edge_index(self) -> usize {
        self.edge_index
    }

    /// Tree parent, starting from the requested root.
    #[inline]
    #[must_use]
    pub const fn parent(self) -> usize {
        self.parent
    }

    /// Tree child reached through the selected edge.
    #[inline]
    #[must_use]
    pub const fn child(self) -> usize {
        self.child
    }
}

#[derive(Clone, Copy)]
struct TreeHalfEdge {
    neighbor: usize,
    edge_index: usize,
}

/// Build a deterministic maximum-weight spanning tree.
///
/// Nodes are dense identifiers in `0..node_count`. Parallel edges are allowed;
/// self-loops, invalid endpoints, and non-finite weights are rejected. Ties are
/// resolved by canonical endpoints, then input edge index. Kruskal's algorithm,
/// which takes the heaviest edges first and skips any that closes a cycle, selects the
/// tree; a compact `O(node_count)` adjacency then orients only its selected edges
/// from `root`. The returned traversal can therefore be consumed directly for
/// payload propagation, with every parent preceding its children.
///
/// # Errors
///
/// Returns [`GraphError::EmptyGraph`] when `node_count` is zero,
/// [`GraphError::NodeOutOfBounds`] when `root` is not a node,
/// [`GraphError::CountMismatch`] when `weights` does not have one entry per edge,
/// [`GraphError::EndpointOutOfBounds`], [`GraphError::SelfLoop`], or
/// [`GraphError::NonFiniteWeight`] for an invalid edge, [`GraphError::Disconnected`] when the
/// tree from `root` does not reach every node, and [`GraphError::AllocationFailed`] when the
/// workspace cannot be allocated.
pub fn maximum_spanning_tree(
    node_count: usize,
    endpoints: &[(usize, usize)],
    weights: &[f64],
    root: usize,
) -> Result<Vec<SpanningTreeEdge>, GraphError> {
    if node_count == 0 {
        return Err(GraphError::EmptyGraph);
    }
    if root >= node_count {
        return Err(GraphError::NodeOutOfBounds {
            node: root,
            node_count,
        });
    }
    validate_weighted_edges(node_count, endpoints, weights)?;
    if node_count == 1 {
        return Ok(Vec::new());
    }
    let selected_edges = select_maximum_spanning_forest(node_count, endpoints, weights)?;
    let tree = orient_forest_component(node_count, endpoints, &selected_edges, root)?;
    if tree.len() != node_count - 1 {
        return Err(GraphError::Disconnected {
            reached: tree.len() + 1,
            node_count,
        });
    }
    Ok(tree)
}

fn validate_weighted_edges(
    node_count: usize,
    endpoints: &[(usize, usize)],
    weights: &[f64],
) -> Result<(), GraphError> {
    if endpoints.len() != weights.len() {
        return Err(GraphError::CountMismatch {
            what: "spanning-tree weights",
            expected: endpoints.len(),
            found: weights.len(),
        });
    }
    validate_endpoints(node_count, endpoints)?;
    for (edge, &weight) in weights.iter().enumerate() {
        if !weight.is_finite() {
            return Err(GraphError::NonFiniteWeight { edge });
        }
    }
    Ok(())
}

fn validate_endpoints(node_count: usize, endpoints: &[(usize, usize)]) -> Result<(), GraphError> {
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
        if first == second {
            return Err(GraphError::SelfLoop { edge, node: first });
        }
    }
    Ok(())
}

/// Select the maximum-weight spanning forest of validated input with Kruskal.
///
/// Weight ties are resolved by canonical endpoints and then input edge index.
/// The selected edge indices are returned in descending-preference order.
fn select_maximum_spanning_forest(
    node_count: usize,
    endpoints: &[(usize, usize)],
    weights: &[f64],
) -> Result<Vec<usize>, GraphError> {
    if endpoints.is_empty() {
        return Ok(Vec::new());
    }
    let mut order = reserved_vec(endpoints.len())?;
    order.extend(0..endpoints.len());
    order.sort_unstable_by(|&left, &right| {
        let canonical = |edge: usize| {
            let (first, second) = endpoints[edge];
            (first.min(second), first.max(second))
        };
        weights[right]
            .total_cmp(&weights[left])
            .then_with(|| canonical(left).cmp(&canonical(right)))
            .then_with(|| left.cmp(&right))
    });

    let mut sets = DenseDisjointSets::try_new(node_count)?;
    let mut selected = reserved_vec(endpoints.len().min(node_count.saturating_sub(1)))?;
    for edge in order {
        let (first, second) = endpoints[edge];
        if sets.union_valid(first, second) {
            selected.push(edge);
        }
    }
    Ok(selected)
}

fn orient_forest_component(
    node_count: usize,
    endpoints: &[(usize, usize)],
    selected_edges: &[usize],
    root: usize,
) -> Result<Vec<SpanningTreeEdge>, GraphError> {
    let mut degrees = allocate_filled(node_count, 0usize)?;
    for &edge in selected_edges {
        let (first, second) = endpoints[edge];
        degrees[first] += 1;
        degrees[second] += 1;
    }
    let offset_count = node_count
        .checked_add(1)
        .ok_or(GraphError::AllocationFailed)?;
    let mut offsets = allocate_filled(offset_count, 0usize)?;
    for node in 0..node_count {
        offsets[node + 1] = offsets[node] + degrees[node];
    }
    let half_edge_count = selected_edges
        .len()
        .checked_mul(2)
        .ok_or(GraphError::AllocationFailed)?;
    let mut adjacency = allocate_filled(
        half_edge_count,
        TreeHalfEdge {
            neighbor: 0,
            edge_index: usize::MAX,
        },
    )?;
    let mut cursors = degrees;
    cursors.copy_from_slice(&offsets[..node_count]);
    for &edge_index in selected_edges {
        let (first, second) = endpoints[edge_index];
        for (node, neighbor) in [(first, second), (second, first)] {
            adjacency[cursors[node]] = TreeHalfEdge {
                neighbor,
                edge_index,
            };
            cursors[node] += 1;
        }
    }

    let mut visited = allocate_filled(node_count, false)?;
    let mut queue = reserved_vec(node_count)?;
    let mut tree = reserved_vec(node_count.saturating_sub(1))?;
    visited[root] = true;
    queue.push(root);
    let mut next = 0usize;
    while next < queue.len() {
        let parent = queue[next];
        next += 1;
        for branch in &adjacency[offsets[parent]..offsets[parent + 1]] {
            if visited[branch.neighbor] {
                continue;
            }
            visited[branch.neighbor] = true;
            queue.push(branch.neighbor);
            tree.push(SpanningTreeEdge {
                edge_index: branch.edge_index,
                parent,
                child: branch.neighbor,
            });
        }
    }
    Ok(tree)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_heaviest_edges_with_root_oriented_traversal() {
        let endpoints = [(0, 2), (2, 1), (1, 0)];
        let tree = maximum_spanning_tree(3, &endpoints, &[1.0, 90.0, 100.0], 0).unwrap();
        assert_eq!(
            tree,
            [
                SpanningTreeEdge {
                    edge_index: 2,
                    parent: 0,
                    child: 1,
                },
                SpanningTreeEdge {
                    edge_index: 1,
                    parent: 1,
                    child: 2,
                },
            ]
        );
    }

    #[test]
    fn deterministic_ties_prefer_canonical_endpoints() {
        let endpoints = [(1, 2), (0, 2), (0, 1)];
        let tree = maximum_spanning_tree(3, &endpoints, &[1.0; 3], 0).unwrap();
        assert_eq!((tree[0].parent(), tree[0].child()), (0, 1));
        assert_eq!((tree[1].parent(), tree[1].child()), (0, 2));
    }

    #[test]
    fn parallel_edges_are_allowed_and_the_heavier_one_wins() {
        let endpoints = [(0, 1), (1, 0)];
        let tree = maximum_spanning_tree(2, &endpoints, &[2.0, 3.0], 0).unwrap();
        assert_eq!(tree[0].edge_index(), 1);
    }

    #[test]
    fn disconnected_graph_reports_reached_nodes() {
        assert_eq!(
            maximum_spanning_tree(4, &[(0, 1), (2, 3)], &[1.0, 1.0], 0),
            Err(GraphError::Disconnected {
                reached: 2,
                node_count: 4,
            })
        );
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert_eq!(
            maximum_spanning_tree(0, &[], &[], 0),
            Err(GraphError::EmptyGraph)
        );
        assert_eq!(
            maximum_spanning_tree(2, &[(0, 2)], &[1.0], 0),
            Err(GraphError::EndpointOutOfBounds {
                edge: 0,
                endpoint: 2,
                node_count: 2,
            })
        );
        assert_eq!(
            maximum_spanning_tree(2, &[(0, 1)], &[f64::NAN], 0),
            Err(GraphError::NonFiniteWeight { edge: 0 })
        );
    }

    #[test]
    fn forest_spans_every_component_and_preserves_isolates() {
        let endpoints = [(0, 1), (1, 2), (0, 2), (3, 4)];
        let forest = select_maximum_spanning_forest(6, &endpoints, &[5.0, 4.0, 1.0, 3.0]).unwrap();
        assert_eq!(forest, [0, 1, 3]);
    }

    #[test]
    fn tree_rejects_self_loops() {
        assert_eq!(
            maximum_spanning_tree(2, &[(0, 0), (0, 1)], &[10.0, 1.0], 0),
            Err(GraphError::SelfLoop { edge: 0, node: 0 })
        );
    }
}
