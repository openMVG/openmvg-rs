// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Accepted tracks, their quality evidence, and triangulation failures.

use nalgebra::Point3;
use openmvg_core::camera::ProjectionError;
use openmvg_core::model::{EstimationError, ViewId};
use openmvg_core::{AngularResidual, PixelResidual};

use super::TriangulationObservation;

/// Residual and conditioning evidence for an accepted point.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TriangulationQuality {
    /// Largest supporting pair angle, in radians.
    pub max_parallax: AngularResidual,
    /// Mean supporting angular residual, in radians.
    pub mean_angular_residual: AngularResidual,
    /// Maximum supporting angular residual, in radians.
    pub max_angular_residual: AngularResidual,
    /// Mean supporting camera-model reprojection residual, in pixels.
    pub mean_pixel_residual: PixelResidual,
    /// Maximum supporting camera-model reprojection residual, in pixels.
    pub max_pixel_residual: PixelResidual,
    /// Number of observations supplied to triangulation.
    pub input_observation_count: usize,
    /// Number of observations supporting the final point.
    pub inlier_observation_count: usize,
    /// Number of two-view hypotheses scored; zero for all-observations fitting.
    pub tested_hypothesis_count: usize,
}

/// Accepted point and the exact observations that support it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TriangulatedTrack {
    /// Estimated world point.
    pub point: Point3<f64>,
    /// Indices into the caller's observation slice.
    pub inlier_indices: Vec<usize>,
    /// Final geometric evidence.
    pub quality: TriangulationQuality,
}

impl TriangulatedTrack {
    /// Iterate the exact input observations supporting the returned point.
    ///
    /// The supplied slice must be the one, or an index-compatible copy of the
    /// one, passed to [`TrackTriangulator::triangulate`](super::TrackTriangulator::triangulate).
    pub fn supporting_observations<'a>(
        &'a self,
        observations: &'a [TriangulationObservation],
    ) -> Result<impl Iterator<Item = &'a TriangulationObservation> + 'a, TriangulationError> {
        if let Some(&index) = self
            .inlier_indices
            .iter()
            .find(|&&index| index >= observations.len())
        {
            return Err(TriangulationError::ObservationIndexOutOfBounds {
                index,
                observations: observations.len(),
            });
        }
        Ok(self
            .inlier_indices
            .iter()
            .map(move |&index| &observations[index]))
    }
}

/// Named final rejection from an otherwise valid track-estimation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrackTriangulationRejection {
    /// Consensus/refinement alternation did not reach a fixed support set.
    UnstableConsensus,
    /// No supporting pair reached the configured final parallax.
    InsufficientParallax,
    /// At least one supporting ray placed the point behind its camera.
    Cheirality,
    /// A supporting angular residual exceeded the final threshold.
    AngularResidual,
    /// The point could not be projected through every supporting camera model.
    ProjectionDomain,
    /// A supporting pixel residual exceeded the final threshold.
    PixelResidual,
}

impl std::fmt::Display for TrackTriangulationRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::UnstableConsensus => "consensus support did not stabilize",
            Self::InsufficientParallax => "insufficient parallax",
            Self::Cheirality => "non-positive oriented ray distance",
            Self::AngularResidual => "angular residual exceeds acceptance threshold",
            Self::ProjectionDomain => "point lies outside a supporting camera projection domain",
            Self::PixelResidual => "pixel residual exceeds acceptance threshold",
        })
    }
}

/// Failure to prepare, estimate, or validate a track.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum TriangulationError {
    /// Invalid configuration.
    #[error("invalid triangulation configuration: {0}")]
    InvalidConfig(&'static str),
    /// Not enough observations were supplied.
    #[error("triangulation needs at least {required} observations, got {provided}")]
    InsufficientObservations {
        /// Minimum observation count required by the selected policy.
        required: usize,
        /// Observation count supplied by the caller.
        provided: usize,
    },
    /// A result was resolved against an incompatible observation slice.
    #[error("observation index {index} is outside a slice of length {observations}")]
    ObservationIndexOutOfBounds {
        /// Invalid observation index.
        index: usize,
        /// Number of available observations.
        observations: usize,
    },
    /// A view has no pose.
    #[error("view {0} has no pose")]
    MissingPose(ViewId),
    /// A view has no camera model.
    #[error("view {0} has no camera model")]
    MissingCamera(ViewId),
    /// The same view occurs more than once in one track.
    #[error("view {0} occurs more than once in the track")]
    DuplicateView(ViewId),
    /// A camera model could not unproject an observation.
    #[error("cannot unproject observation {index} in view {view_id}: {source}")]
    Unprojection {
        /// Index of the observation that failed.
        index: usize,
        /// View containing the observation.
        view_id: ViewId,
        /// Camera-model error.
        #[source]
        source: ProjectionError,
    },
    /// No geometrically valid point hypothesis was found.
    #[error("no valid triangulation hypothesis")]
    NoHypothesis,
    /// Consensus support was insufficient.
    #[error("triangulation consensus has {inliers}/{observations} inliers")]
    InsufficientConsensus {
        /// Number of observations supporting the final point.
        inliers: usize,
        /// Total number of supplied observations.
        observations: usize,
    },
    /// The final point violated a named acceptance rule.
    #[error("triangulation rejected: {0}")]
    Rejected(TrackTriangulationRejection),
    /// A numerical primitive failed.
    #[error("triangulation primitive failed: {0}")]
    Numerical(#[source] EstimationError),
}
