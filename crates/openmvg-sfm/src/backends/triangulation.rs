// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Triangulation contract: [`Triangulator`], its posed observations, and its outcomes.

use nalgebra::{Point2, Point3};
use openmvg_core::{PixelResidual, camera::Camera, model::WorldToCameraPose};

use crate::{Result, SfmError, ViewId};

/// One measured pixel with its already-resolved camera context.
///
/// Unlike `openmvg_geometry::triangulation::TriangulationObservation` (a view and a pixel), it
/// borrows the pose and camera of the view, so a [`Triangulator`] needs no workspace lookup.
#[derive(Debug, Clone, Copy)]
pub struct PosedPixelObservation<'a> {
    /// View that measured the pixel.
    pub view: ViewId,
    /// Measured image position, in pixels.
    pub xy: Point2<f64>,
    /// World-to-camera pose of that view.
    pub pose: &'a WorldToCameraPose,
    /// Camera model of that view.
    pub camera: &'a Camera,
}

/// Landmark accepted by a [`Triangulator`], with the observations that support it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TriangulatedLandmark {
    /// Landmark position in world coordinates.
    pub point: Point3<f64>,
    /// Mean supporting reprojection residual in original-image pixels.
    pub reprojection_error: PixelResidual,
    /// Strictly increasing, unique indices into the observation slice supplied
    /// to [`Triangulator::triangulate`]. At least two observations are required.
    pub supporting_observation_indices: Box<[usize]>,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl TriangulatedLandmark {
    /// Number of supporting observations.
    pub fn supporting_observation_count(&self) -> usize {
        self.supporting_observation_indices.len()
    }

    /// Validate the result returned by a [`Triangulator`] implementation.
    pub fn validate(&self, observation_count: usize) -> Result<()> {
        if !self
            .point
            .coords
            .iter()
            .all(|coordinate| coordinate.is_finite())
        {
            return Err(SfmError::InvalidInput(
                "triangulator returned a non-finite point".to_owned(),
            ));
        }
        let support = &self.supporting_observation_indices;
        if support.len() < 2 {
            return Err(SfmError::InvalidInput(
                "triangulator returned fewer than two supporting observations".to_owned(),
            ));
        }
        if support.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(SfmError::InvalidInput(
                "triangulator support indices must be strictly increasing".to_owned(),
            ));
        }
        if support
            .last()
            .is_some_and(|&index| index >= observation_count)
        {
            return Err(SfmError::InvalidInput(format!(
                "triangulator support index exceeds {observation_count} input observations"
            )));
        }
        Ok(())
    }
}

/// Stable reason an otherwise valid track did not produce a landmark.
#[non_exhaustive]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TriangulationRejectReason {
    /// Fewer than the required source-track observations currently have poses.
    InsufficientPosedObservations,
    /// The triangulator received fewer observations than its geometric minimum.
    InsufficientObservations,
    /// No sampled observation subset produced valid geometry.
    NoValidHypothesis,
    /// The best hypothesis did not satisfy the configured support requirement.
    InsufficientConsensus,
    /// Consensus/refinement alternation did not reach a fixed support set.
    UnstableConsensus,
    /// Supporting rays did not reach the final conditioning angle.
    InsufficientParallax,
    /// The point was not in front of every supporting camera.
    Cheirality,
    /// A supporting angular residual exceeded the final threshold.
    AngularResidual,
    /// A supporting pixel residual exceeded the triangulator threshold.
    PixelResidual,
    /// A supporting camera model could not project or unproject the geometry.
    ProjectionDomain,
    /// The pipeline's post-triangulation reprojection limit was exceeded.
    ReprojectionLimit,
    /// The observation geometry was singular or otherwise degenerate.
    DegenerateGeometry,
    /// A numerical primitive failed on otherwise valid input.
    NumericalFailure,
    /// A custom triangulator rejected the track for an implementation-specific reason.
    RejectedByImplementation,
}

impl TriangulationRejectReason {
    /// Stable snake-case identifier for logs and external reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InsufficientPosedObservations => "insufficient_posed_observations",
            Self::InsufficientObservations => "insufficient_observations",
            Self::NoValidHypothesis => "no_valid_hypothesis",
            Self::InsufficientConsensus => "insufficient_consensus",
            Self::UnstableConsensus => "unstable_consensus",
            Self::InsufficientParallax => "insufficient_parallax",
            Self::Cheirality => "cheirality",
            Self::AngularResidual => "angular_residual",
            Self::PixelResidual => "pixel_residual",
            Self::ProjectionDomain => "projection_domain",
            Self::ReprojectionLimit => "reprojection_limit",
            Self::DegenerateGeometry => "degenerate_geometry",
            Self::NumericalFailure => "numerical_failure",
            Self::RejectedByImplementation => "rejected_by_implementation",
        }
    }
}

impl std::fmt::Display for TriangulationRejectReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Result of a valid triangulation request.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriangulationVerdict {
    /// The track produced a landmark.
    Accepted(TriangulatedLandmark),
    /// The track produced no landmark, for the given reason.
    Rejected(TriangulationRejectReason),
}

impl From<TriangulatedLandmark> for TriangulationVerdict {
    fn from(point: TriangulatedLandmark) -> Self {
        Self::Accepted(point)
    }
}

/// Triangulates one landmark from the posed observations of a track.
pub trait Triangulator: Send + Sync {
    /// Estimate one point and identify the exact input observations supporting it.
    ///
    /// Expected geometric failure is [`TriangulationVerdict::Rejected`]. `Err`
    /// is reserved for invalid configuration/state or an implementation failure.
    /// Accepted malformed support is a contract violation detected by the pipeline.
    fn triangulate(
        &self,
        observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict>;
}
