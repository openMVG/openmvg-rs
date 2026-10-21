// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Topology audit of the view graph.
//!
//! Its teaching docs are on [`audit_graph_topology`], where docs.rs shows them.

use std::fmt;

use openmvg_core::graph::{GraphError, UndirectedTopology, analyze_undirected_topology};
#[cfg(test)]
use openmvg_core::{graph::DenseNodeMap, model::RelativeRotation};

/// Invalid dense graph input supplied to [`audit_graph_topology`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphTopologyError {
    /// The declared graph cannot be represented by the platform's contiguous
    /// allocation/indexing limits.
    GraphTooLarge {
        /// Declared number of dense views.
        view_count: usize,
        /// Number of input edges.
        edge_count: usize,
    },
    /// Temporary audit storage could not be allocated.
    AllocationFailed {
        /// Declared number of dense views.
        view_count: usize,
        /// Number of input edges.
        edge_count: usize,
    },
    /// An edge endpoint is outside the declared dense view range.
    ViewOutOfRange {
        /// Index of the offending edge in the input slice.
        edge: usize,
        /// Out-of-range endpoint index.
        view: usize,
        /// Declared number of dense views; valid endpoints are `0..view_count`.
        view_count: usize,
    },
    /// Pose/view graphs do not admit self-loop constraints.
    SelfLoop {
        /// Index of the offending edge in the input slice.
        edge: usize,
        /// View index used as both endpoints.
        view: usize,
    },
}

impl fmt::Display for GraphTopologyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GraphTooLarge {
                view_count,
                edge_count,
            } => write!(
                formatter,
                "graph with {view_count} views and {edge_count} edges exceeds platform limits"
            ),
            Self::AllocationFailed {
                view_count,
                edge_count,
            } => write!(
                formatter,
                "cannot allocate topology audit for {view_count} views and {edge_count} edges"
            ),
            Self::ViewOutOfRange {
                edge,
                view,
                view_count,
            } => write!(
                formatter,
                "edge {edge} references view {view}, outside 0..{view_count}"
            ),
            Self::SelfLoop { edge, view } => {
                write!(formatter, "edge {edge} is a self-loop on view {view}")
            }
        }
    }
}

impl std::error::Error for GraphTopologyError {}

/// Non-mutating topology preflight for a dense-indexed view graph.
///
/// `candidate_core_*` denotes the largest vertex-biconnected block, selected
/// deterministically by view coverage, then edge count, then lexicographically
/// smallest view set. It is only a *candidate* translation core: numerical
/// parallel rigidity must be verified separately before it can be called rigid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphTopologyAudit {
    /// Shared raw connected and block-cut topology.
    topology: UndirectedTopology,
    /// Opaque ID of the selected candidate block, or `None` when no edge block
    /// exists.
    candidate_core_block: Option<usize>,
    /// Number of views in the selected candidate block.
    candidate_core_num_views: usize,
    /// Number of edges in the selected candidate block.
    candidate_core_num_edges: usize,
    /// Dense-view membership mask for the selected candidate block.
    candidate_core_view_mask: Vec<bool>,
    /// Input-edge membership mask for the selected candidate block.
    candidate_core_edge_mask: Vec<bool>,
    /// Non-core views in the same connected component as the candidate core.
    candidate_tail_views: Vec<usize>,
    /// Non-isolated views in other connected components.
    disconnected_views: Vec<usize>,
}

impl GraphTopologyAudit {
    /// Number of declared views, including isolates.
    #[must_use]
    pub const fn view_count(&self) -> usize {
        self.topology.node_count()
    }

    /// Number of input edges.
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.topology.edge_count()
    }

    /// Number of connected components, including isolated views.
    #[must_use]
    pub const fn connected_component_count(&self) -> usize {
        self.topology.component_count()
    }

    /// Dense connected-component label for each view.
    #[must_use]
    pub fn connected_component_by_view(&self) -> &[usize] {
        self.topology.component_by_node()
    }

    /// Number of blocks in the block-cut decomposition.
    #[must_use]
    pub const fn biconnected_block_count(&self) -> usize {
        self.topology.biconnected_block_count()
    }

    /// Opaque block ID for each input edge.
    #[must_use]
    pub fn biconnected_block_by_edge(&self) -> &[usize] {
        self.topology.biconnected_block_by_edge()
    }

    /// Opaque ID of the selected candidate block, if one exists.
    #[must_use]
    pub const fn candidate_core_block(&self) -> Option<usize> {
        self.candidate_core_block
    }

    /// Number of views in the selected candidate block.
    #[must_use]
    pub const fn candidate_core_view_count(&self) -> usize {
        self.candidate_core_num_views
    }

    /// Number of edges in the selected candidate block.
    #[must_use]
    pub const fn candidate_core_edge_count(&self) -> usize {
        self.candidate_core_num_edges
    }

    /// Dense-view membership mask for the selected candidate block.
    #[must_use]
    pub fn candidate_core_view_mask(&self) -> &[bool] {
        &self.candidate_core_view_mask
    }

    /// Input-edge membership mask for the selected candidate block.
    #[must_use]
    pub fn candidate_core_edge_mask(&self) -> &[bool] {
        &self.candidate_core_edge_mask
    }

    /// Sorted dense IDs of articulation views.
    #[must_use]
    pub fn articulation_views(&self) -> &[usize] {
        self.topology.articulation_nodes()
    }

    /// Sorted input indices of bridge edges.
    #[must_use]
    pub fn bridge_edges(&self) -> &[usize] {
        self.topology.bridge_edges()
    }

    /// Sorted dense IDs of isolated views.
    #[must_use]
    pub fn isolated_views(&self) -> &[usize] {
        self.topology.isolated_nodes()
    }

    /// Non-core views in the candidate's connected component.
    #[must_use]
    pub fn candidate_tail_views(&self) -> &[usize] {
        &self.candidate_tail_views
    }

    /// Non-isolated views in connected components outside the candidate.
    #[must_use]
    pub fn disconnected_views(&self) -> &[usize] {
        &self.disconnected_views
    }

    /// Whether a non-empty candidate block contains every declared view.
    #[must_use]
    pub fn candidate_covers_all_views(&self) -> bool {
        self.candidate_core_block.is_some()
            && self.candidate_core_num_views == self.topology.node_count()
    }
}

#[derive(Clone, Copy, Debug)]
struct BlockCandidate {
    block: usize,
    view_count: usize,
    edge_count: usize,
    min_views: (usize, usize),
}

impl BlockCandidate {
    fn outranks(self, other: Self) -> bool {
        // Distinct vertex blocks share at most one view. Consequently, after
        // equal view/edge counts, their two smallest IDs produce exactly the
        // same ordering as comparing their complete sorted view sets—without
        // sorting or retaining those sets.
        self.view_count > other.view_count
            || (self.view_count == other.view_count && self.edge_count > other.edge_count)
            || (self.view_count == other.view_count
                && self.edge_count == other.edge_count
                && self.min_views < other.min_views)
    }
}

/// Audit connected components, bridges, articulations, and vertex-biconnected
/// blocks through the shared payload-free graph topology kernel.
///
/// Time and memory are both `O(num_views + edges.len())`. Edge identity is
/// preserved, so parallel edges are handled correctly. Explicit `num_views`
/// makes isolated cameras observable instead of silently dropping them.
///
/// # How it works
///
/// Reports connected components, bridges, articulations and the largest
/// biconnected core for pre-averaging graph cleaning. The dense audit operates
/// on endpoint pairs.
///
/// Topology ignores weights, residuals, and directions, so callers must first
/// decide which measurements are credible: a cross-edge can remove
/// a real articulation topologically even when that measurement is an outlier.
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │                        Topology audit                                │
/// └──────────────────────────────────────────────────────────────────────┘
/// Input: num_views, edges [(i, j)] with dense indices in 0..num_views
///   │
///   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 1: Tarjan block-cut pass                                        │
/// └──────────────────────────────────────────────────────────────────────┘
/// `analyze_undirected_topology`: iterative, O(num_views + edges). Gives
/// components, bridge edges, articulation views, isolated views, and one
/// biconnected block ID per edge. Rejects out-of-range ends and self-loops.
///   │
///   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 2: Rank the blocks                                              │
/// └──────────────────────────────────────────────────────────────────────┘
/// Count the views and edges of each block. Largest view count wins, then
/// largest edge count, then the smaller pair of smallest view IDs.
///   │
///   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 3: Masks and views outside the core                             │
/// └──────────────────────────────────────────────────────────────────────┘
/// Mark the winner's edges and views. Split the other non-isolated views
/// into the tail (same component as the core) and the disconnected views.
/// Output: `GraphTopologyAudit`; no edge or view is removed or reordered
/// ```
///
/// # Problem and method
///
/// Translation directions fix the positions of the views only up to one scale
/// for each part of the graph that is joined to the rest through a single view.
/// A view whose removal splits the graph is an articulation view, and an edge
/// whose removal splits it is a bridge. The vertex-biconnected blocks are the
/// maximal sets of edges that no single view separates. Two blocks share at most
/// one view, which is why comparing the two smallest view IDs of equal blocks
/// gives the same order as comparing their full sorted view sets.
///
/// The audit selects one block as the candidate core and leaves every
/// decision to the caller: it returns the core as view and edge masks in the
/// caller's index spaces, the non-core views in the core's component
/// (`candidate_tail_views`), the non-isolated views in other components
/// (`disconnected_views`), and the isolated views. The core of a graph with
/// no cycle is a single edge, and a graph without edges has no core
/// (`candidate_core_block` is `None`).
///
/// The audit applies no threshold. Parallel edges keep separate identities: two
/// edges between the same views form one block and are not bridges. A candidate
/// block is a topological candidate only.
/// [`audit_parallel_rigidity`](super::audit_parallel_rigidity) tests whether it is parallel
/// rigid.
///
/// # Failure modes
///
/// [`audit_graph_topology`] returns a [`GraphTopologyError`]:
///
/// - `ViewOutOfRange` when an edge endpoint is not below `num_views`,
/// - `SelfLoop` for an edge `(v, v)`,
/// - `GraphTooLarge` when the declared sizes cannot be indexed on the platform,
/// - `AllocationFailed` when the temporary storage cannot be reserved.
///
/// # Alternatives
///
/// For a plain connected-component filter, `GraphTopologyAudit` already reports
/// `connected_component_by_view`. The rank audit in
/// [`audit_parallel_rigidity`](super::audit_parallel_rigidity) starts from this module's candidate
/// core.
pub fn audit_graph_topology(
    num_views: usize,
    edges: &[(usize, usize)],
) -> Result<GraphTopologyAudit, GraphTopologyError> {
    let topology = analyze_undirected_topology(num_views, edges)
        .map_err(|error| map_graph_error(error, num_views, edges.len()))?;
    let biconnected_block_count = topology.biconnected_block_count();
    let biconnected_block_by_edge = topology.biconnected_block_by_edge();
    if biconnected_block_count == 0 {
        debug_assert!(edges.is_empty());
        return Ok(GraphTopologyAudit {
            topology,
            candidate_core_block: None,
            candidate_core_num_views: 0,
            candidate_core_num_edges: 0,
            candidate_core_view_mask: topology_buffer(num_views, false, num_views, edges.len())?,
            candidate_core_edge_mask: Vec::new(),
            candidate_tail_views: Vec::new(),
            disconnected_views: Vec::new(),
        });
    }

    // Bucket edge identities by raw block ID. This keeps candidate selection
    // linear even when input edges from different blocks are interleaved.
    let mut block_offsets = topology_buffer(
        biconnected_block_count
            .checked_add(1)
            .ok_or(GraphTopologyError::GraphTooLarge {
                view_count: num_views,
                edge_count: edges.len(),
            })?,
        0usize,
        num_views,
        edges.len(),
    )?;
    for &block in biconnected_block_by_edge {
        block_offsets[block + 1] += 1;
    }
    for block in 0..biconnected_block_count {
        block_offsets[block + 1] += block_offsets[block];
    }
    let mut block_cursors = topology_vec(block_offsets.len(), num_views, edges.len())?;
    block_cursors.extend_from_slice(&block_offsets);
    let mut edges_by_block = topology_buffer(edges.len(), usize::MAX, num_views, edges.len())?;
    for (edge, &block) in biconnected_block_by_edge.iter().enumerate() {
        edges_by_block[block_cursors[block]] = edge;
        block_cursors[block] += 1;
    }

    let mut view_block_marker = topology_buffer(num_views, usize::MAX, num_views, edges.len())?;
    let mut best_candidate = None::<BlockCandidate>;
    for block in 0..biconnected_block_count {
        let block_edges = &edges_by_block[block_offsets[block]..block_offsets[block + 1]];
        let mut view_count = 0usize;
        let mut min_views = (usize::MAX, usize::MAX);
        for &edge in block_edges {
            for view in [edges[edge].0, edges[edge].1] {
                if view_block_marker[view] == block {
                    continue;
                }
                view_block_marker[view] = block;
                view_count += 1;
                if view < min_views.0 {
                    min_views.1 = min_views.0;
                    min_views.0 = view;
                } else if view < min_views.1 {
                    min_views.1 = view;
                }
            }
        }
        let candidate = BlockCandidate {
            block,
            view_count,
            edge_count: block_edges.len(),
            min_views,
        };
        if best_candidate.is_none_or(|best| candidate.outranks(best)) {
            best_candidate = Some(candidate);
        }
    }
    drop(block_offsets);
    drop(block_cursors);
    drop(edges_by_block);
    drop(view_block_marker);

    let candidate_core_block = best_candidate.map(|candidate| candidate.block);
    let candidate_core_num_views = best_candidate.map_or(0, |candidate| candidate.view_count);
    let candidate_core_num_edges = best_candidate.map_or(0, |candidate| candidate.edge_count);
    let mut candidate_core_edge_mask = topology_buffer(edges.len(), false, num_views, edges.len())?;
    for (keep, &block) in candidate_core_edge_mask
        .iter_mut()
        .zip(biconnected_block_by_edge)
    {
        *keep = Some(block) == candidate_core_block;
    }
    let mut candidate_core_view_mask = topology_buffer(num_views, false, num_views, edges.len())?;
    for (&(view_i, view_j), &keep) in edges.iter().zip(&candidate_core_edge_mask) {
        if keep {
            candidate_core_view_mask[view_i] = true;
            candidate_core_view_mask[view_j] = true;
        }
    }
    let core_component = candidate_core_view_mask
        .iter()
        .position(|&keep| keep)
        .map(|view| topology.component_by_node()[view]);
    let core_component_view_count = core_component.map_or(0, |component| {
        topology
            .component_by_node()
            .iter()
            .filter(|&&label| label == component)
            .count()
    });
    let mut candidate_tail_views = topology_vec(
        core_component_view_count.saturating_sub(candidate_core_num_views),
        num_views,
        edges.len(),
    )?;
    let mut disconnected_views = topology_vec(
        num_views
            .saturating_sub(topology.isolated_nodes().len())
            .saturating_sub(core_component_view_count),
        num_views,
        edges.len(),
    )?;
    let isolated_views = topology.isolated_nodes();
    let mut next_isolated = 0usize;
    for (view, &is_core) in candidate_core_view_mask.iter().enumerate() {
        let is_isolated = isolated_views.get(next_isolated) == Some(&view);
        if is_isolated {
            next_isolated += 1;
        }
        if is_core || is_isolated {
            continue;
        }
        if Some(topology.component_by_node()[view]) == core_component {
            candidate_tail_views.push(view);
        } else {
            disconnected_views.push(view);
        }
    }

    Ok(GraphTopologyAudit {
        topology,
        candidate_core_block,
        candidate_core_num_views,
        candidate_core_num_edges,
        candidate_core_view_mask,
        candidate_core_edge_mask,
        candidate_tail_views,
        disconnected_views,
    })
}

fn map_graph_error(error: GraphError, view_count: usize, edge_count: usize) -> GraphTopologyError {
    match error {
        GraphError::EndpointOutOfBounds {
            edge,
            endpoint,
            node_count,
        } => GraphTopologyError::ViewOutOfRange {
            edge,
            view: endpoint,
            view_count: node_count,
        },
        GraphError::SelfLoop { edge, node: view } => GraphTopologyError::SelfLoop { edge, view },
        GraphError::AllocationFailed => GraphTopologyError::AllocationFailed {
            view_count,
            edge_count,
        },
        _ => GraphTopologyError::GraphTooLarge {
            view_count,
            edge_count,
        },
    }
}

fn topology_buffer<T: Clone>(
    len: usize,
    value: T,
    view_count: usize,
    edge_count: usize,
) -> Result<Vec<T>, GraphTopologyError> {
    let mut values = topology_vec(len, view_count, edge_count)?;
    values.resize(len, value);
    Ok(values)
}

fn topology_vec<T>(
    capacity: usize,
    view_count: usize,
    edge_count: usize,
) -> Result<Vec<T>, GraphTopologyError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| GraphTopologyError::AllocationFailed {
            view_count,
            edge_count,
        })?;
    Ok(values)
}

// ============================================================================
// Bi-Connected Components — iterative Tarjan
// ============================================================================

/// Keep only edges in the largest bi-connected component.
///
/// A bi-connected component has no articulation points — removing any single
/// view still leaves the component connected. For translation-direction
/// positioning this is a necessary topology guard: two otherwise rigid lobes
/// sharing only one view have independent scales around that articulation.
/// Parallel rigidity in 3D still requires a separate numerical audit.
/// The retained block maximizes view count, then edge count, then chooses the
/// lexicographically smallest view-ID set.
///
/// Sparse view IDs are first sorted/remapped in `O(E log E)` time. The dense
/// audit then uses an iterative, non-recursive `O(V + E)` Tarjan pass.
#[cfg(test)]
fn filter_largest_biconnected_component(edges: &[RelativeRotation]) -> BlockSelection {
    if edges.is_empty() {
        return empty_result();
    }

    let endpoints: Vec<_> = edges
        .iter()
        .map(|edge| (edge.view_i().index(), edge.view_j().index()))
        .collect();
    let nmap = DenseNodeMap::try_from_endpoints(&endpoints)
        .expect("connectivity wrapper requires addressable endpoint storage");
    // In this compatibility wrapper, self-loops never belong to a biconnected
    // block, while the strict public audit
    // rejects them as invalid pose-graph input.
    let mut dense_edges = Vec::with_capacity(edges.len());
    for edge in edges {
        if edge.view_i() == edge.view_j() {
            continue;
        }
        dense_edges.push((
            nmap.dense_id(edge.view_i().index())
                .expect("endpoint is indexed"),
            nmap.dense_id(edge.view_j().index())
                .expect("endpoint is indexed"),
        ));
    }
    let audit = audit_graph_topology(nmap.node_count(), &dense_edges)
        .expect("dense remapping produces valid non-loop endpoints");
    let mut dense_mask = audit.candidate_core_edge_mask.iter().copied();
    let in_largest: Vec<_> = edges
        .iter()
        .map(|edge| {
            (edge.view_i() != edge.view_j())
                && dense_mask
                    .next()
                    .expect("every non-loop edge has a dense audit result")
        })
        .collect();
    debug_assert!(dense_mask.next().is_none());
    let largest_size = audit.candidate_core_view_count();
    let removed_edges = in_largest.iter().filter(|&&keep| !keep).count();

    BlockSelection {
        inlier_mask: in_largest,
        component_count: audit.biconnected_block_count(),
        largest_component_size: largest_size,
        removed_edges,
    }
}

#[cfg(test)]
fn empty_result() -> BlockSelection {
    BlockSelection {
        inlier_mask: vec![],
        component_count: 0,
        largest_component_size: 0,
        removed_edges: 0,
    }
}

/// Edges of the block kept by [`filter_largest_biconnected_component`].
#[cfg(test)]
struct BlockSelection {
    inlier_mask: Vec<bool>,
    component_count: usize,
    largest_component_size: usize,
    removed_edges: usize,
}

#[cfg(test)]
impl BlockSelection {
    fn kept_count(&self) -> usize {
        self.inlier_mask.len() - self.removed_edges
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use openmvg_core::model::ViewId;

    fn edge(i: usize, j: usize) -> RelativeRotation {
        RelativeRotation::try_new(ViewId::new(i), ViewId::new(j), Rotation3::identity(), 1.0)
            .unwrap()
    }

    #[test]
    fn test_bicc_triangle() {
        let edges = vec![edge(0, 1), edge(1, 2), edge(0, 2)];
        let result = filter_largest_biconnected_component(&edges);
        assert_eq!(result.largest_component_size, 3);
        assert_eq!(result.removed_edges, 0);
        assert!(result.inlier_mask.iter().all(|&b| b));
    }

    #[test]
    fn test_bicc_separates_cycles_sharing_an_articulation_view() {
        // The two lobes have no bridge edge, but can scale independently about
        // their shared view 2. A bridge-only test would incorrectly keep both.
        let edges = vec![
            edge(0, 1),
            edge(1, 2),
            edge(2, 0),
            edge(2, 3),
            edge(3, 4),
            edge(4, 2),
        ];
        let result = filter_largest_biconnected_component(&edges);
        assert_eq!(result.component_count, 2);
        assert_eq!(result.largest_component_size, 3);
        assert_eq!(result.kept_count(), 3);
    }

    #[test]
    fn test_bicc_bridge() {
        // Two triangles connected by a bridge at node 2
        let edges = vec![
            edge(0, 1),
            edge(1, 2),
            edge(0, 2), // triangle 1
            edge(2, 3), // bridge
            edge(3, 4),
            edge(4, 5),
            edge(3, 5), // triangle 2
        ];
        let result = filter_largest_biconnected_component(&edges);
        assert!(result.component_count >= 3); // two triangles + bridge
        assert!(result.largest_component_size >= 3);
    }

    #[test]
    fn test_bicc_chain() {
        // Chain: 0-1-2-3 — every edge is a bridge
        let edges = vec![edge(0, 1), edge(1, 2), edge(2, 3)];
        let result = filter_largest_biconnected_component(&edges);
        assert_eq!(result.component_count, 3);
        assert_eq!(result.largest_component_size, 2);
    }

    #[test]
    fn test_bicc_fully_connected() {
        // K4: single bi-connected component
        let edges = vec![
            edge(0, 1),
            edge(0, 2),
            edge(0, 3),
            edge(1, 2),
            edge(1, 3),
            edge(2, 3),
        ];
        let result = filter_largest_biconnected_component(&edges);
        assert_eq!(result.component_count, 1);
        assert_eq!(result.largest_component_size, 4);
        assert_eq!(result.removed_edges, 0);
    }

    #[test]
    fn test_bicc_empty() {
        let result = filter_largest_biconnected_component(&[]);
        assert_eq!(result.component_count, 0);
    }

    #[test]
    fn relative_rotation_rejects_self_loops_before_filtering() {
        assert!(
            RelativeRotation::try_new(ViewId::new(0), ViewId::new(0), Rotation3::identity(), 1.0,)
                .is_err()
        );
    }

    #[test]
    fn test_bicc_single_edge() {
        let edges = vec![edge(0, 1)];
        let result = filter_largest_biconnected_component(&edges);
        assert_eq!(result.component_count, 1);
        assert_eq!(result.largest_component_size, 2);
    }

    #[test]
    fn test_bicc_large_cycle() {
        // 10-node cycle: should be a single bi-connected component
        let edges: Vec<_> = (0..10).map(|i| edge(i, (i + 1) % 10)).collect();
        let result = filter_largest_biconnected_component(&edges);
        assert_eq!(result.component_count, 1);
        assert_eq!(result.largest_component_size, 10);
        assert_eq!(result.removed_edges, 0);
    }

    #[test]
    fn topology_audit_reports_isolates_bridges_and_articulations() {
        let audit = audit_graph_topology(6, &[(0, 1), (1, 2), (2, 3)]).unwrap();

        assert_eq!(audit.connected_component_count(), 3);
        assert_eq!(audit.biconnected_block_count(), 3);
        assert_eq!(audit.bridge_edges(), &[0, 1, 2]);
        assert_eq!(audit.articulation_views(), &[1, 2]);
        assert_eq!(audit.isolated_views(), &[4, 5]);
        assert_eq!(audit.candidate_core_view_count(), 2);
        assert_eq!(audit.candidate_core_edge_count(), 1);
        // Equal one-edge blocks select the lexicographically first view set.
        assert_eq!(
            audit.candidate_core_view_mask,
            vec![true, true, false, false, false, false]
        );
        assert_eq!(audit.candidate_tail_views, vec![2, 3]);
        assert!(audit.disconnected_views.is_empty());
    }

    #[test]
    fn topology_audit_finds_articulation_without_a_bridge() {
        let edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)];
        let audit = audit_graph_topology(5, &edges).unwrap();

        assert!(audit.bridge_edges().is_empty());
        assert_eq!(audit.articulation_views(), &[2]);
        assert_eq!(audit.biconnected_block_count(), 2);
        assert_eq!(
            audit.candidate_core_view_mask,
            vec![true, true, true, false, false]
        );
        assert_eq!(audit.candidate_tail_views, vec![3, 4]);

        // Both equal-score blocks share their minimum view (the articulation),
        // so the second-smallest view is the exact lexicographic tie-break.
        let permuted = [(4, 2), (3, 4), (2, 3), (2, 0), (1, 2), (0, 1)];
        let permuted_audit = audit_graph_topology(5, &permuted).unwrap();
        assert_eq!(
            audit.candidate_core_view_mask,
            permuted_audit.candidate_core_view_mask
        );
    }

    #[test]
    fn topology_audit_handles_parallel_edges_by_identity() {
        let audit = audit_graph_topology(3, &[(0, 1), (0, 1), (1, 2)]).unwrap();

        assert_eq!(audit.bridge_edges(), &[2]);
        assert_eq!(audit.articulation_views(), &[1]);
        assert_eq!(audit.candidate_core_edge_mask, vec![true, true, false]);
    }

    #[test]
    fn topology_audit_groups_three_parallel_edges_without_a_bridge() {
        let audit = audit_graph_topology(2, &[(0, 1), (1, 0), (0, 1)]).unwrap();

        assert!(audit.bridge_edges().is_empty());
        assert!(audit.articulation_views().is_empty());
        assert_eq!(audit.biconnected_block_count(), 1);
        assert_eq!(audit.biconnected_block_by_edge(), &[0, 0, 0]);
        assert!(audit.candidate_covers_all_views());
    }

    #[test]
    fn topology_audit_root_articulation_uses_dfs_child_count() {
        let fork = audit_graph_topology(3, &[(0, 1), (0, 2)]).unwrap();
        assert_eq!(fork.articulation_views(), &[0]);

        let root_cycle = audit_graph_topology(4, &[(0, 1), (1, 2), (2, 0), (2, 3)]).unwrap();
        assert_eq!(root_cycle.articulation_views(), &[2]);
        assert!(!root_cycle.articulation_views().contains(&0));
    }

    #[test]
    fn topology_audit_reports_component_ownership_and_empty_semantics() {
        let empty = audit_graph_topology(0, &[]).unwrap();
        assert_eq!(empty.connected_component_count(), 0);
        assert!(empty.connected_component_by_view().is_empty());
        assert_eq!(empty.candidate_core_block, None);
        assert!(!empty.candidate_covers_all_views());

        let all_isolates = audit_graph_topology(4, &[]).unwrap();
        assert_eq!(all_isolates.connected_component_count(), 4);
        assert_eq!(all_isolates.connected_component_by_view(), &[0, 1, 2, 3]);
        assert_eq!(all_isolates.isolated_views(), &[0, 1, 2, 3]);
        assert_eq!(all_isolates.biconnected_block_count(), 0);
        assert!(!all_isolates.candidate_covers_all_views());

        let audit = audit_graph_topology(6, &[(0, 2), (3, 4)]).unwrap();
        assert_eq!(audit.connected_component_count(), 4);
        assert_eq!(audit.connected_component_by_view(), &[0, 1, 0, 2, 2, 3]);
        assert_eq!(audit.isolated_views(), &[1, 5]);
    }

    #[test]
    fn topology_audit_rejects_unrepresentable_graph_sizes_before_allocating() {
        assert_eq!(
            audit_graph_topology(usize::MAX, &[]),
            Err(GraphTopologyError::GraphTooLarge {
                view_count: usize::MAX,
                edge_count: 0,
            })
        );
    }

    #[test]
    fn topology_audit_is_non_recursive_on_a_long_chain() {
        const NUM_VIEWS: usize = 50_000;
        let edges: Vec<_> = (1..NUM_VIEWS).map(|view| (view - 1, view)).collect();
        let audit = audit_graph_topology(NUM_VIEWS, &edges).unwrap();

        assert_eq!(audit.bridge_edges().len(), edges.len());
        assert_eq!(audit.articulation_views().len(), NUM_VIEWS - 2);
        assert_eq!(audit.biconnected_block_count(), edges.len());
    }

    #[test]
    fn topology_audit_distinguishes_tails_disconnected_views_and_isolates() {
        let edges = [(0, 1), (1, 2), (2, 0), (2, 3), (4, 5), (5, 6), (6, 4)];
        let audit = audit_graph_topology(8, &edges).unwrap();

        assert_eq!(
            audit.candidate_core_view_mask,
            vec![true, true, true, false, false, false, false, false]
        );
        assert_eq!(audit.candidate_tail_views, vec![3]);
        assert_eq!(audit.disconnected_views, vec![4, 5, 6]);
        assert_eq!(audit.isolated_views(), &[7]);
    }

    #[test]
    fn topology_audit_rejects_invalid_pose_graph_edges() {
        assert_eq!(
            audit_graph_topology(2, &[(0, 2)]),
            Err(GraphTopologyError::ViewOutOfRange {
                edge: 0,
                view: 2,
                view_count: 2,
            })
        );
        assert_eq!(
            audit_graph_topology(2, &[(1, 1)]),
            Err(GraphTopologyError::SelfLoop { edge: 0, view: 1 })
        );
    }
}
