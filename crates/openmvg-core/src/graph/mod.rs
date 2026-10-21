// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked, payload-free graph indexing primitives.
//!
//! The algorithms here see only nodes and edges, numbered densely from zero: you keep your
//! own edge data, such as rotations or scores, and index it with the edge numbers they return.
//! The module provides dense indexing, disjoint sets, stable component partitions,
//! connectivity completion, spanning trees, and undirected topology.
//!
//! ## Input policies
//!
//! The algorithms expose only the graph semantics needed by their callers:
//!
//! | API | Parallel edges | Self-loops | Explicit isolated nodes |
//! | --- | --- | --- | --- |
//! | [`component_partition`] | Accepted | Accepted; no effect | Included through `node_count` |
//! | [`complete_connectivity`] | Accepted | Accepted; never needed | Included through `node_count` |
//! | [`maximum_spanning_tree`] | Accepted | Rejected | Included through `node_count`; disconnected input is rejected |
//! | [`analyze_undirected_topology`] | Accepted with distinct edge identity | Rejected | Included through `node_count` |
//!
//! Connectivity algorithms can ignore loops, while block-cut topology (the biconnected blocks
//! of a graph and the articulation nodes that join them) rejects inputs whose meaning would
//! otherwise be ambiguous.

mod alloc;
mod components;
mod connectivity;
mod disjoint_sets;
mod error;
mod node_map;
mod spanning_tree;
mod topology;

pub use components::{ComponentPartition, component_partition};
pub use connectivity::{ConnectivityCompletion, complete_connectivity};
pub use disjoint_sets::DenseDisjointSets;
pub use error::GraphError;
pub use node_map::DenseNodeMap;
pub use spanning_tree::{SpanningTreeEdge, maximum_spanning_tree};
pub use topology::{UndirectedTopology, analyze_undirected_topology};
