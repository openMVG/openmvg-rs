// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Track-level triangulation with explicit consensus and acceptance policies.
//!
//! Its teaching docs are on [`TrackTriangulator`], where docs.rs shows them.

mod config;
mod observation;
mod result;
mod triangulator;

pub use config::{
    AcceptanceConfig, ConsensusConfig, ConsensusThreshold, PairHypothesisBudget,
    TrackTriangulationConfig, TriangulationPolicy,
};
pub use observation::{ResolvedTrackObservation, TriangulationContext, TriangulationObservation};
pub use result::{
    TrackTriangulationRejection, TriangulatedTrack, TriangulationError, TriangulationQuality,
};
pub use triangulator::TrackTriangulator;

#[cfg(test)]
mod tests;
