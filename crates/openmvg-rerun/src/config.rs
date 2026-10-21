// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::num::NonZeroU64;
use std::time::Duration;

/// Which SfM events carry visual snapshots.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryCapture {
    /// Record only cheap progress metrics and event messages. Geometry-derived quality metrics are
    /// sampled with visual snapshots and are therefore disabled too.
    Disabled,
    /// Record lightweight camera and graph growth at registration events. Landmark clouds and
    /// residual summaries are sampled every `stride` events and at global, rollback, selection,
    /// and terminal boundaries.
    Every(NonZeroU64),
}

impl Default for GeometryCapture {
    fn default() -> Self {
        Self::Every(NonZeroU64::MIN)
    }
}

/// Graph projections included in a geometry frame.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphLayers {
    /// Initial verified view graph, deterministically sampled to the configured edge limit.
    pub verified: bool,
    /// Verified edges retained by the current reconstruction branch.
    pub active: bool,
    /// Active edges whose endpoint cameras are both registered.
    pub registered_induced: bool,
    /// Active edges with exactly one registered endpoint.
    pub frontier: bool,
}

impl Default for GraphLayers {
    fn default() -> Self {
        Self {
            verified: true,
            active: false,
            registered_induced: true,
            frontier: false,
        }
    }
}

/// Bounded, deterministic conversion and recording policy.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct RerunRecorderConfig {
    /// Landmark and residual-detail cadence. Camera and graph growth is retained at every
    /// registration checkpoint. The default is full detail at every checkpoint.
    pub geometry: GeometryCapture,
    /// View-graph layers recorded at each geometry snapshot.
    pub graph_layers: GraphLayers,
    /// Maximum landmarks copied into one 3D snapshot. Zero disables landmark rendering; quality
    /// metrics still use the full active observation support.
    pub max_landmarks: usize,
    /// Maximum edges copied into each graph layer and the 3D registered-edge snapshot. Zero
    /// retains graph nodes but disables edge rendering.
    pub max_edges_per_layer: usize,
    /// Camera-axis length in reconstruction world units.
    pub camera_axis_length: f64,
    /// Rerun SDK byte budget before its internal lossless queue applies backpressure.
    pub max_bytes_in_flight: u64,
    /// Maximum duration used when explicitly flushing the recorder.
    pub flush_timeout: Duration,
    /// Whether tentative optimization states receive geometry snapshots.
    pub include_tentative_geometry: bool,
}

impl Default for RerunRecorderConfig {
    fn default() -> Self {
        Self {
            geometry: GeometryCapture::default(),
            graph_layers: GraphLayers::default(),
            max_landmarks: 100_000,
            max_edges_per_layer: 100_000,
            camera_axis_length: 0.1,
            max_bytes_in_flight: 64 * 1024 * 1024,
            flush_timeout: Duration::from_secs(10),
            include_tentative_geometry: true,
        }
    }
}

impl RerunRecorderConfig {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if !self.camera_axis_length.is_finite() || self.camera_axis_length <= 0.0 {
            return Err("camera_axis_length must be finite and positive");
        }
        if self.max_bytes_in_flight == 0 {
            return Err("max_bytes_in_flight must be positive");
        }
        if self.flush_timeout.is_zero() {
            return Err("flush_timeout must be positive");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_graph_layers_match_the_two_graph_blueprint() {
        let layers = GraphLayers::default();
        assert!(layers.verified);
        assert!(!layers.active);
        assert!(layers.registered_induced);
        assert!(!layers.frontier);
    }

    #[test]
    fn default_capture_keeps_full_detail() {
        assert_eq!(
            GeometryCapture::default(),
            GeometryCapture::Every(NonZeroU64::MIN)
        );
    }

    #[test]
    fn rejects_invalid_numerical_configuration() {
        let invalid_axis = RerunRecorderConfig {
            camera_axis_length: f64::NAN,
            ..RerunRecorderConfig::default()
        };
        assert!(invalid_axis.validate().is_err());

        let no_buffer = RerunRecorderConfig {
            max_bytes_in_flight: 0,
            ..RerunRecorderConfig::default()
        };
        assert!(no_buffer.validate().is_err());

        let no_flush_timeout = RerunRecorderConfig {
            flush_timeout: Duration::ZERO,
            ..RerunRecorderConfig::default()
        };
        assert!(no_flush_timeout.validate().is_err());
    }
}
