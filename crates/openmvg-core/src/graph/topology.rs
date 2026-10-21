// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::GraphError;
use super::alloc::{allocate_filled, reserved_vec};

/// Payload-free topology of a dense undirected multigraph.
///
/// Connected-component labels are stable: components are ordered by their
/// smallest node. Biconnected block identifiers are opaque: they depend on
/// the depth-first search and input order. Parallel edges retain distinct identities;
/// isolated nodes are represented explicitly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UndirectedTopology {
    component_count: usize,
    component_by_node: Vec<usize>,
    biconnected_block_count: usize,
    biconnected_block_by_edge: Vec<usize>,
    articulation_nodes: Vec<usize>,
    bridge_edges: Vec<usize>,
    isolated_nodes: Vec<usize>,
}

#[derive(Clone, Copy)]
struct DfsParent {
    node: usize,
    edge: usize,
}

const NO_PARENT: DfsParent = DfsParent {
    node: usize::MAX,
    edge: usize::MAX,
};

impl UndirectedTopology {
    /// Number of declared nodes, including isolates.
    #[inline]
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.component_by_node.len()
    }

    /// Number of input edges.
    #[inline]
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.biconnected_block_by_edge.len()
    }

    /// Number of connected components, including isolated nodes.
    #[inline]
    #[must_use]
    pub const fn component_count(&self) -> usize {
        self.component_count
    }

    /// Stable connected-component label for every dense node.
    #[inline]
    #[must_use]
    pub fn component_by_node(&self) -> &[usize] {
        &self.component_by_node
    }

    /// Number of edge blocks in the block-cut decomposition.
    #[inline]
    #[must_use]
    pub const fn biconnected_block_count(&self) -> usize {
        self.biconnected_block_count
    }

    /// Opaque biconnected-block identifier for every input edge.
    #[inline]
    #[must_use]
    pub fn biconnected_block_by_edge(&self) -> &[usize] {
        &self.biconnected_block_by_edge
    }

    /// Sorted dense node IDs whose removal increases component count.
    #[inline]
    #[must_use]
    pub fn articulation_nodes(&self) -> &[usize] {
        &self.articulation_nodes
    }

    /// Sorted input edge indices whose removal increases component count.
    #[inline]
    #[must_use]
    pub fn bridge_edges(&self) -> &[usize] {
        &self.bridge_edges
    }

    /// Sorted dense node IDs with degree zero.
    #[inline]
    #[must_use]
    pub fn isolated_nodes(&self) -> &[usize] {
        &self.isolated_nodes
    }
}

/// Analyze connected components and the block-cut structure of a dense
/// undirected multigraph in one iterative depth-first pass (Tarjan's algorithm).
///
/// The block-cut structure splits the edges into biconnected blocks, which stay
/// connected when any one node is removed, joined at articulation nodes.
///
/// Time and memory are both `O(node_count + endpoints.len())`. Edge identity
/// is preserved, so parallel edges are handled correctly. Self-loops are
/// rejected: a loop is never a bridge and joins no two blocks.
///
/// # Errors
///
/// Returns [`GraphError::EndpointOutOfBounds`] or [`GraphError::SelfLoop`] for an invalid
/// edge, [`GraphError::CapacityExceeded`] when the adjacency size overflows `usize`, and
/// [`GraphError::AllocationFailed`] when the workspace cannot be allocated.
pub fn analyze_undirected_topology(
    node_count: usize,
    endpoints: &[(usize, usize)],
) -> Result<UndirectedTopology, GraphError> {
    let edge_count = endpoints.len();
    let too_large = || GraphError::CapacityExceeded {
        node_count,
        edge_count,
    };
    let offset_count = node_count.checked_add(1).ok_or_else(too_large)?;
    let adjacency_count = edge_count.checked_mul(2).ok_or_else(too_large)?;
    if endpoints.is_empty() {
        let mut component_by_node = reserved_vec(node_count)?;
        component_by_node.extend(0..node_count);
        let mut isolated_nodes = reserved_vec(node_count)?;
        isolated_nodes.extend(0..node_count);
        return Ok(UndirectedTopology {
            component_count: node_count,
            component_by_node,
            biconnected_block_count: 0,
            biconnected_block_by_edge: Vec::new(),
            articulation_nodes: Vec::new(),
            bridge_edges: Vec::new(),
            isolated_nodes,
        });
    }

    let mut degrees = allocate_filled(node_count, 0usize)?;
    for (edge, &(first, second)) in endpoints.iter().enumerate() {
        for node in [first, second] {
            if node >= node_count {
                return Err(GraphError::EndpointOutOfBounds {
                    edge,
                    endpoint: node,
                    node_count,
                });
            }
        }
        if first == second {
            return Err(GraphError::SelfLoop { edge, node: first });
        }
        degrees[first] = degrees[first].checked_add(1).ok_or_else(too_large)?;
        degrees[second] = degrees[second].checked_add(1).ok_or_else(too_large)?;
    }

    // Dense CSR adjacency preserves edge identity, so parallel edges remain
    // distinguishable without one allocation per node.
    let mut offsets = allocate_filled(offset_count, 0usize)?;
    for node in 0..node_count {
        offsets[node + 1] = offsets[node]
            .checked_add(degrees[node])
            .ok_or_else(too_large)?;
    }
    debug_assert_eq!(offsets[node_count], adjacency_count);
    // Keep the two hot fields in separate arrays rather than interleaving them
    // in an array of structs. This costs one small allocation and keeps each
    // field's accesses contiguous on large sparse graphs.
    let mut neighbors = allocate_filled(adjacency_count, 0usize)?;
    let mut edge_ids = allocate_filled(adjacency_count, usize::MAX)?;
    // Reuse the degree allocation as CSR write cursors, then DFS read cursors.
    degrees.copy_from_slice(&offsets[..node_count]);
    for (edge, &(first, second)) in endpoints.iter().enumerate() {
        let at_first = degrees[first];
        neighbors[at_first] = second;
        edge_ids[at_first] = edge;
        degrees[first] += 1;
        let at_second = degrees[second];
        neighbors[at_second] = first;
        edge_ids[at_second] = edge;
        degrees[second] += 1;
    }
    degrees.copy_from_slice(&offsets[..node_count]);

    let mut discovery = allocate_filled(node_count, usize::MAX)?;
    let mut low_link = allocate_filled(node_count, 0usize)?;
    let mut next_adjacency = degrees;
    let mut parent = allocate_filled(node_count, NO_PARENT)?;
    let mut component_by_node = allocate_filled(node_count, usize::MAX)?;
    let mut articulation = allocate_filled(node_count, false)?;
    let mut seen_edge = allocate_filled(edge_count, false)?;
    let mut bridge_mask = allocate_filled(edge_count, false)?;
    let mut biconnected_block_by_edge = allocate_filled(edge_count, usize::MAX)?;
    let mut edge_stack = reserved_vec(edge_count)?;
    let mut dfs_stack = reserved_vec(node_count)?;
    let mut component_count = 0usize;
    let mut biconnected_block_count = 0usize;
    let mut time = 0usize;

    for root in 0..node_count {
        if discovery[root] != usize::MAX {
            continue;
        }
        let component = component_count;
        component_count += 1;
        let mut root_child_count = 0usize;
        discovery[root] = time;
        low_link[root] = time;
        component_by_node[root] = component;
        time += 1;
        dfs_stack.push(root);

        while let Some(&node) = dfs_stack.last() {
            if next_adjacency[node] < offsets[node + 1] {
                let adjacency_index = next_adjacency[node];
                next_adjacency[node] += 1;
                let edge = edge_ids[adjacency_index];
                if seen_edge[edge] {
                    continue;
                }
                seen_edge[edge] = true;
                edge_stack.push(edge);
                let neighbor = neighbors[adjacency_index];
                if discovery[neighbor] == usize::MAX {
                    if node == root {
                        root_child_count += 1;
                    }
                    parent[neighbor] = DfsParent { node, edge };
                    discovery[neighbor] = time;
                    low_link[neighbor] = time;
                    component_by_node[neighbor] = component;
                    time += 1;
                    dfs_stack.push(neighbor);
                } else {
                    low_link[node] = low_link[node].min(discovery[neighbor]);
                }
                continue;
            }

            dfs_stack.pop();
            let node_parent = parent[node];
            if node_parent.node != usize::MAX {
                if low_link[node] >= discovery[node_parent.node] {
                    if parent[node_parent.node].node != usize::MAX {
                        articulation[node_parent.node] = true;
                    }
                    pop_biconnected_block(
                        &mut edge_stack,
                        Some(node_parent.edge),
                        biconnected_block_count,
                        &mut biconnected_block_by_edge,
                    );
                    biconnected_block_count += 1;
                }
                if low_link[node] > discovery[node_parent.node] {
                    bridge_mask[node_parent.edge] = true;
                }
                low_link[node_parent.node] = low_link[node_parent.node].min(low_link[node]);
            } else {
                articulation[node] = root_child_count > 1;
                if !edge_stack.is_empty() {
                    pop_biconnected_block(
                        &mut edge_stack,
                        None,
                        biconnected_block_count,
                        &mut biconnected_block_by_edge,
                    );
                    biconnected_block_count += 1;
                }
            }
        }
    }

    debug_assert!(
        biconnected_block_by_edge
            .iter()
            .all(|&block| block < biconnected_block_count)
    );
    drop(neighbors);
    drop(edge_ids);
    drop(discovery);
    drop(low_link);
    drop(next_adjacency);
    drop(parent);
    drop(seen_edge);
    drop(edge_stack);
    drop(dfs_stack);
    let articulation_nodes = collect_marked(&articulation)?;
    let bridge_edges = collect_marked(&bridge_mask)?;
    let isolated_count = offsets
        .windows(2)
        .filter(|adjacency| adjacency[0] == adjacency[1])
        .count();
    let mut isolated_nodes = reserved_vec(isolated_count)?;
    for node in 0..node_count {
        if offsets[node] == offsets[node + 1] {
            isolated_nodes.push(node);
        }
    }

    Ok(UndirectedTopology {
        component_count,
        component_by_node,
        biconnected_block_count,
        biconnected_block_by_edge,
        articulation_nodes,
        bridge_edges,
        isolated_nodes,
    })
}

fn pop_biconnected_block(
    edge_stack: &mut Vec<usize>,
    boundary: Option<usize>,
    block: usize,
    biconnected_block_by_edge: &mut [usize],
) {
    let mut reached_boundary = boundary.is_none();
    while let Some(edge) = edge_stack.pop() {
        debug_assert_eq!(biconnected_block_by_edge[edge], usize::MAX);
        biconnected_block_by_edge[edge] = block;
        if boundary == Some(edge) {
            reached_boundary = true;
            break;
        }
    }
    debug_assert!(reached_boundary);
}

fn collect_marked(mask: &[bool]) -> Result<Vec<usize>, GraphError> {
    let marked_count = mask.iter().filter(|&&marked| marked).count();
    let mut indices = reserved_vec(marked_count)?;
    for (index, &marked) in mask.iter().enumerate() {
        if marked {
            indices.push(index);
        }
    }
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_triangle_bridge_and_isolate() {
        let topology = analyze_undirected_topology(5, &[(0, 1), (1, 2), (2, 0), (2, 3)]).unwrap();

        assert_eq!(topology.component_count(), 2);
        assert_eq!(topology.component_by_node(), &[0, 0, 0, 0, 1]);
        assert_eq!(topology.biconnected_block_count(), 2);
        let blocks = topology.biconnected_block_by_edge();
        assert_eq!(blocks[0], blocks[1]);
        assert_eq!(blocks[1], blocks[2]);
        assert_ne!(blocks[2], blocks[3]);
        assert_eq!(topology.articulation_nodes(), &[2]);
        assert_eq!(topology.bridge_edges(), &[3]);
        assert_eq!(topology.isolated_nodes(), &[4]);
    }

    #[test]
    fn parallel_edges_share_a_block_and_are_not_bridges() {
        let topology = analyze_undirected_topology(2, &[(0, 1), (0, 1)]).unwrap();

        assert_eq!(topology.biconnected_block_count(), 1);
        assert_eq!(topology.biconnected_block_by_edge(), &[0, 0]);
        assert!(topology.articulation_nodes().is_empty());
        assert!(topology.bridge_edges().is_empty());
    }

    #[test]
    fn rejects_invalid_endpoints_and_self_loops() {
        assert_eq!(
            analyze_undirected_topology(2, &[(0, 2)]),
            Err(GraphError::EndpointOutOfBounds {
                edge: 0,
                endpoint: 2,
                node_count: 2,
            })
        );
        assert_eq!(
            analyze_undirected_topology(2, &[(1, 1)]),
            Err(GraphError::SelfLoop { edge: 0, node: 1 })
        );
    }
}
