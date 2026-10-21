// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! View-graph audits and filters around averaging.
//!
//! Topology and parallel-rigidity audits for pre-averaging graph cleaning.

mod graph_connectivity;
mod parallel_rigidity;

pub use graph_connectivity::{GraphTopologyAudit, GraphTopologyError, audit_graph_topology};

pub use parallel_rigidity::{
    DirectionConstraint, ParallelRigidityAudit, ParallelRigidityConfig, ParallelRigidityError,
    ParallelRigidityMethod, StructuralRigidityStatus, audit_parallel_rigidity,
};
