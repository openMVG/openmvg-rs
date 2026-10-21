// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Track triangulation adapters: robust camera-model-aware consensus and a clean-data DLT path.

use std::collections::BTreeMap;

use nalgebra::Matrix3;
use openmvg_core::{
    AngularThreshold,
    camera::{Camera, CameraModel, PinholeCamera},
    model::{EstimationError, ViewId, WorldToCameraPose},
};
use openmvg_geometry::triangulation::{
    ConsensusConfig, ConsensusThreshold, PairHypothesisBudget, ResolvedTrackObservation,
    TrackTriangulationConfig, TrackTriangulationRejection, TrackTriangulator, TriangulationConfig,
    TriangulationError, TriangulationMethod, TriangulationPolicy, triangulate as triangulate_point,
};

use crate::backends::{
    PosedPixelObservation, TriangulatedLandmark, TriangulationRejectReason, TriangulationVerdict,
    Triangulator,
};
use crate::{Result, SfmError};

/// DLT triangulation adapter that fits every supplied observation.
///
/// It cannot remove contaminated track members, so it suits clean data. Prefer
/// [`TrackTriangulatorAdapter::default`] for camera-model-aware robust consensus.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub struct DltTriangulator {
    config: TriangulationConfig,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl DltTriangulator {
    /// Creates a DLT triangulator with the given solver configuration.
    pub fn new(config: TriangulationConfig) -> Self {
        Self { config }
    }
}

impl Default for DltTriangulator {
    fn default() -> Self {
        Self::new(TriangulationConfig::default())
    }
}

impl Triangulator for DltTriangulator {
    fn triangulate(
        &self,
        observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        // DLT is a perspective-only compatibility path. Brown pixels are
        // unprojected once and reprojected through their underlying pinhole model; fisheye
        // and spherical tracks must use `TrackTriangulatorAdapter`.
        let solver_observations: Vec<_> = observations
            .iter()
            .map(|observation| {
                let (pinhole, distorted) = perspective_pinhole(observation.camera)?;
                let xy =
                    if distorted {
                        let bearing = observation.camera.unproject(&observation.xy).map_err(
                            |error| {
                                SfmError::InvalidInput(format!(
                                    "DLT triangulation could not unproject a Brown pixel: {error}"
                                ))
                            },
                        )?;
                        pinhole.project(&bearing).map_err(|error| {
                            SfmError::InvalidInput(format!(
                                "DLT triangulation could not form a virtual pinhole pixel: {error}"
                            ))
                        })?
                    } else {
                        observation.xy
                    };
                Ok((observation.view, xy))
            })
            .collect::<Result<_>>()?;
        let poses: BTreeMap<ViewId, WorldToCameraPose> = observations
            .iter()
            .map(|observation| (observation.view, observation.pose.clone()))
            .collect();
        let intrinsics: BTreeMap<ViewId, Matrix3<f64>> = observations
            .iter()
            .map(|observation| {
                perspective_pinhole(observation.camera)
                    .map(|(pinhole, _)| {
                        pinhole
                            .intrinsic_matrix()
                            .map(|matrix| (observation.view, matrix))
                            .ok_or_else(|| {
                                SfmError::InvalidInput(
                                    "perspective camera did not expose an intrinsic matrix"
                                        .to_owned(),
                                )
                            })
                    })
                    .and_then(|result| result)
            })
            .collect::<Result<_>>()?;

        let result = match triangulate_point(
            TriangulationMethod::Dlt,
            &solver_observations,
            &poses,
            &intrinsics,
            &self.config,
        ) {
            Ok(result) => result,
            Err(error) => return map_operator_error(error),
        };

        let mut residual_sum = 0.0;
        for observation in observations {
            let camera_point = observation.pose.project_to_camera(&result.point);
            let projected = match observation.camera.project(&camera_point.coords) {
                Ok(projected) => projected,
                Err(_) => {
                    return Ok(TriangulationVerdict::Rejected(
                        TriangulationRejectReason::ProjectionDomain,
                    ));
                }
            };
            residual_sum += crate::ops::reconstruction::camera_pixel_residual(
                observation.camera,
                projected,
                observation.xy,
            );
        }

        Ok(TriangulatedLandmark {
            point: result.point,
            reprojection_error: openmvg_core::PixelResidual::try_new(
                residual_sum / observations.len() as f64,
            )
            .map_err(|_| {
                SfmError::InvalidInput("triangulation produced an invalid pixel residual".into())
            })?,
            supporting_observation_indices: (0..solver_observations.len())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
        .into())
    }
}

/// Camera-model-aware track triangulation for pipeline tracks.
#[derive(Debug, Clone)]
pub struct TrackTriangulatorAdapter {
    inner: TrackTriangulator,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl TrackTriangulatorAdapter {
    /// Wraps a track triangulator built from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`SfmError::InvalidInput`] when `config` is rejected by the geometry crate.
    pub fn try_new(config: TrackTriangulationConfig) -> Result<Self> {
        let inner = TrackTriangulator::try_new(config)
            .map_err(|error| SfmError::InvalidInput(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Recommended robust consensus preset, independent of the camera model (the default).
    pub fn robust() -> Self {
        Self {
            inner: TrackTriangulator::robust(),
        }
    }

    /// Clean-data preset that fits every observation of a track without outlier removal.
    pub fn all_observations() -> Self {
        Self {
            inner: TrackTriangulator::all_observations(),
        }
    }

    /// Robust consensus for the first triangulation after global rotation/translation averaging.
    ///
    /// Averaged poses are degree-accurate, not pixel-accurate: a one-degree rotation error is
    /// already tens of pixels at full resolution, so a pixel gate sized for refined poses rejects
    /// almost every track and starves the fixed-rotation BA. This keeps the robust preset's pair
    /// consensus and parallax rules with a three-degree angular tolerance; the robust
    /// retriangulation after bundle adjustment restores pixel precision.
    pub fn for_averaged_poses() -> Result<Self> {
        let tolerance = AngularThreshold::from_degrees(3.0)
            .map_err(|error| SfmError::InvalidInput(error.to_string()))?;
        let robust = TrackTriangulationConfig::default();
        Self::try_new(
            robust
                .with_policy(TriangulationPolicy::Consensus(
                    ConsensusConfig::default()
                        .with_threshold(ConsensusThreshold::Angular(tolerance))
                        .with_min_inliers(2)
                        .with_min_inlier_ratio(0.5)
                        .with_pair_budget(PairHypothesisBudget::CONSERVATIVE),
                ))
                .with_acceptance(robust.acceptance().with_max_angular_residual(tolerance)),
        )
    }
}

impl Default for TrackTriangulatorAdapter {
    fn default() -> Self {
        Self::robust()
    }
}

impl Triangulator for TrackTriangulatorAdapter {
    fn triangulate(
        &self,
        observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        let resolved = observations.iter().map(|observation| {
            ResolvedTrackObservation::new(
                observation.view,
                observation.xy,
                observation.pose,
                observation.camera,
            )
        });
        let result = match self.inner.triangulate_resolved_iter(resolved) {
            Ok(result) => result,
            Err(error) => return map_track_error(error),
        };
        Ok(TriangulatedLandmark {
            point: result.point,
            reprojection_error: result.quality.mean_pixel_residual,
            supporting_observation_indices: result.inlier_indices.into_boxed_slice(),
        }
        .into())
    }
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
fn perspective_pinhole(camera: &Camera) -> Result<(&PinholeCamera, bool)> {
    match camera {
        Camera::Pinhole(pinhole) => Ok((pinhole, false)),
        Camera::BrownConrady(brown) => Ok((brown.pinhole(), true)),
        Camera::Fisheye(_) => Err(SfmError::UnsupportedCameraModel {
            context: "DLT triangulation",
            camera_model: "fisheye",
        }),
        Camera::Spherical(_) => Err(SfmError::UnsupportedCameraModel {
            context: "DLT triangulation",
            camera_model: "spherical",
        }),
    }
}

fn map_operator_error(error: EstimationError) -> Result<TriangulationVerdict> {
    let reason = match error {
        EstimationError::InsufficientData { .. } => {
            TriangulationRejectReason::InsufficientObservations
        }
        EstimationError::DegenerateConfiguration { .. } => {
            TriangulationRejectReason::DegenerateGeometry
        }
        EstimationError::Numerical(_) => TriangulationRejectReason::NumericalFailure,
        EstimationError::ViewNotFound(_) | EstimationError::InvalidInput(_) => {
            return Err(SfmError::InvalidInput(format!(
                "triangulation contract failed: {error}"
            )));
        }
        _ => {
            return Err(SfmError::InvalidInput(format!(
                "triangulation contract failed: {error}"
            )));
        }
    };
    Ok(TriangulationVerdict::Rejected(reason))
}

fn map_track_error(error: TriangulationError) -> Result<TriangulationVerdict> {
    let reason = match error {
        TriangulationError::InsufficientObservations { .. } => {
            TriangulationRejectReason::InsufficientObservations
        }
        TriangulationError::Unprojection { .. } => TriangulationRejectReason::ProjectionDomain,
        TriangulationError::NoHypothesis => TriangulationRejectReason::NoValidHypothesis,
        TriangulationError::InsufficientConsensus { .. } => {
            TriangulationRejectReason::InsufficientConsensus
        }
        TriangulationError::Rejected(rejection) => match rejection {
            TrackTriangulationRejection::UnstableConsensus => {
                TriangulationRejectReason::UnstableConsensus
            }
            TrackTriangulationRejection::InsufficientParallax => {
                TriangulationRejectReason::InsufficientParallax
            }
            TrackTriangulationRejection::Cheirality => TriangulationRejectReason::Cheirality,
            TrackTriangulationRejection::AngularResidual => {
                TriangulationRejectReason::AngularResidual
            }
            TrackTriangulationRejection::ProjectionDomain => {
                TriangulationRejectReason::ProjectionDomain
            }
            TrackTriangulationRejection::PixelResidual => TriangulationRejectReason::PixelResidual,
            _ => {
                return Err(SfmError::InvalidInput(format!(
                    "unsupported triangulation rejection: {rejection}"
                )));
            }
        },
        TriangulationError::Numerical(error) => return map_operator_error(error),
        TriangulationError::InvalidConfig(_)
        | TriangulationError::ObservationIndexOutOfBounds { .. }
        | TriangulationError::MissingPose(_)
        | TriangulationError::MissingCamera(_)
        | TriangulationError::DuplicateView(_) => {
            return Err(SfmError::InvalidInput(format!(
                "triangulation contract failed: {error}"
            )));
        }
        other => {
            return Err(SfmError::InvalidInput(format!(
                "unsupported triangulation failure: {other}"
            )));
        }
    };
    Ok(TriangulationVerdict::Rejected(reason))
}
