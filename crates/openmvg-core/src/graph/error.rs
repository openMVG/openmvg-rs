// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

/// Failure of a checked graph primitive.
///
/// One error type serves every algorithm in the [`graph`](crate::graph) module; each function
/// documents which variants it can return.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum GraphError {
    /// A rooted tree cannot be constructed without nodes.
    #[error("maximum spanning tree requires at least one node")]
    EmptyGraph,
    /// A dense node identifier (disjoint-set node or spanning-tree root) is
    /// outside the node range.
    #[error("node {node} is outside {node_count} nodes")]
    NodeOutOfBounds {
        /// Dense node identifier supplied by the caller.
        node: usize,
        /// Number of nodes in the graph.
        node_count: usize,
    },
    /// An input endpoint is outside the dense node range.
    #[error("edge {edge} endpoint {endpoint} is outside {node_count} nodes")]
    EndpointOutOfBounds {
        /// Index of the invalid input edge.
        edge: usize,
        /// Dense node identifier found on that edge.
        endpoint: usize,
        /// Number of nodes declared by the caller.
        node_count: usize,
    },
    /// The algorithm does not admit self-loops.
    #[error("edge {edge} is a self-loop on node {node}")]
    SelfLoop {
        /// Index of the self-loop in the input edge list.
        edge: usize,
        /// Node connected to itself.
        node: usize,
    },
    /// An endpoint remapping references an identifier absent from the node map.
    #[error("edge {edge} references node {node}, which is absent from the dense node map")]
    NodeNotIndexed {
        /// Index of the input edge containing the unknown node.
        edge: usize,
        /// External node identifier absent from the node map.
        node: usize,
    },
    /// A per-edge input slice (weights, keep flags) does not have one entry per edge.
    #[error("expected {expected} {what}, got {found}")]
    CountMismatch {
        /// Name of the per-edge input.
        what: &'static str,
        /// Number of input edges requiring an entry.
        expected: usize,
        /// Number of entries supplied by the caller.
        found: usize,
    },
    /// A candidate edge index is outside the edge range.
    #[error("connectivity candidate {candidate} references missing edge {edge}")]
    CandidateOutOfBounds {
        /// Position of the invalid entry in the candidate order.
        candidate: usize,
        /// Out-of-range input edge index stored at that position.
        edge: usize,
    },
    /// A dropped edge must appear exactly once in the completion order.
    #[error("edge {edge} occurs more than once in the connectivity candidate order")]
    DuplicateCandidate {
        /// Input edge index that occurs more than once.
        edge: usize,
    },
    /// Kept edges are already mandatory and must not occur in the candidate order.
    #[error("edge {edge} is already kept and cannot be a connectivity candidate")]
    CandidateAlreadyKept {
        /// Kept input edge incorrectly included as a candidate.
        edge: usize,
    },
    /// Every dropped edge must appear in the candidate order.
    #[error("dropped edge {edge} is missing from the connectivity candidate order")]
    MissingCandidate {
        /// Dropped input edge omitted from the candidate order.
        edge: usize,
    },
    /// Edge ordering requires finite weights.
    #[error("edge {edge} has a non-finite spanning-tree weight")]
    NonFiniteWeight {
        /// Index of the edge carrying the invalid weight.
        edge: usize,
    },
    /// The requested nodes do not form one connected component.
    #[error("spanning tree reached {reached} of {node_count} nodes")]
    Disconnected {
        /// Number of nodes reachable from the requested root.
        reached: usize,
        /// Total number of nodes in the graph.
        node_count: usize,
    },
    /// Graph index arithmetic exceeds platform or compact-index limits.
    #[error("graph with {node_count} nodes and {edge_count} edges exceeds index limits")]
    CapacityExceeded {
        /// Number of nodes (0 when the overflow is detected before nodes are counted).
        node_count: usize,
        /// Number of input edges.
        edge_count: usize,
    },
    /// Temporary or retained graph storage could not be allocated.
    #[error("cannot allocate graph workspace")]
    AllocationFailed,
}
