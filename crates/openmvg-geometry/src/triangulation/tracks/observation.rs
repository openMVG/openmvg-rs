// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Track observations and their resolution into world rays.

use std::collections::BTreeMap;

use crate::triangulation::WorldRay;
use nalgebra::Point2;
use openmvg_core::camera::{Camera, CameraModel};
use openmvg_core::model::{ViewId, WorldToCameraPose};

use super::TriangulationError;

/// Pixel observation belonging to one track.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TriangulationObservation {
    /// View containing this observation.
    pub view_id: ViewId,
    /// Measured pixel coordinate.
    pub pixel: Point2<f64>,
}

impl TriangulationObservation {
    /// Pair a view with its measured pixel.
    #[must_use]
    pub const fn new(view_id: ViewId, pixel: Point2<f64>) -> Self {
        Self { view_id, pixel }
    }
}

/// Poses and camera models required to calibrate and score a track.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TriangulationContext<'a> {
    /// World-to-camera poses by view.
    pub poses: &'a BTreeMap<ViewId, WorldToCameraPose>,
    /// Camera models by view.
    pub cameras: &'a BTreeMap<ViewId, Camera>,
}

impl<'a> TriangulationContext<'a> {
    /// Borrow the per-view poses and camera models.
    #[must_use]
    pub const fn new(
        poses: &'a BTreeMap<ViewId, WorldToCameraPose>,
        cameras: &'a BTreeMap<ViewId, Camera>,
    ) -> Self {
        Self { poses, cameras }
    }
}

/// One observation with its pose and camera model already resolved.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ResolvedTrackObservation<'a> {
    /// View that owns the observation.
    pub view_id: ViewId,
    /// Observed image coordinate.
    pub pixel: Point2<f64>,
    /// World-to-camera pose for the observation.
    pub pose: &'a WorldToCameraPose,
    /// Camera model used to interpret the pixel.
    pub camera: &'a Camera,
}

impl<'a> ResolvedTrackObservation<'a> {
    /// Pair an observation with its already-resolved pose and camera model.
    #[must_use]
    pub const fn new(
        view_id: ViewId,
        pixel: Point2<f64>,
        pose: &'a WorldToCameraPose,
        camera: &'a Camera,
    ) -> Self {
        Self {
            view_id,
            pixel,
            pose,
            camera,
        }
    }
}

/// Observation with its camera ray resolved in the world frame.
#[derive(Debug, Clone, Copy)]
pub(super) struct PreparedObservation<'a> {
    pub(super) input_index: usize,
    pub(super) pixel: Point2<f64>,
    pub(super) ray: WorldRay,
    pub(super) pose: &'a WorldToCameraPose,
    pub(super) camera: &'a Camera,
}

pub(super) fn prepare_observations<'a>(
    observations: &[TriangulationObservation],
    context: TriangulationContext<'a>,
) -> Result<Vec<PreparedObservation<'a>>, TriangulationError> {
    if observations.len() < 2 {
        return Err(TriangulationError::InsufficientObservations {
            required: 2,
            provided: observations.len(),
        });
    }
    let prepared: Vec<_> = observations
        .iter()
        .enumerate()
        .map(|(input_index, observation)| {
            let pose = context
                .poses
                .get(&observation.view_id)
                .ok_or(TriangulationError::MissingPose(observation.view_id))?;
            let camera = context
                .cameras
                .get(&observation.view_id)
                .ok_or(TriangulationError::MissingCamera(observation.view_id))?;
            let bearing = camera.unproject(&observation.pixel).map_err(|source| {
                TriangulationError::Unprojection {
                    index: input_index,
                    view_id: observation.view_id,
                    source,
                }
            })?;
            let ray = WorldRay::try_from_camera(observation.view_id, pose, bearing)
                .map_err(TriangulationError::Numerical)?;
            Ok(PreparedObservation {
                input_index,
                pixel: observation.pixel,
                ray,
                pose,
                camera,
            })
        })
        .collect::<Result<_, _>>()?;
    canonicalize_prepared(prepared)
}

pub(super) fn prepare_resolved_observations<'a, I>(
    observations: I,
) -> Result<Vec<PreparedObservation<'a>>, TriangulationError>
where
    I: IntoIterator<Item = ResolvedTrackObservation<'a>>,
    I::IntoIter: ExactSizeIterator,
{
    let observations = observations.into_iter();
    let observation_count = observations.len();
    if observation_count < 2 {
        return Err(TriangulationError::InsufficientObservations {
            required: 2,
            provided: observation_count,
        });
    }
    let prepared: Vec<_> = observations
        .enumerate()
        .map(|(input_index, observation)| {
            let bearing = observation
                .camera
                .unproject(&observation.pixel)
                .map_err(|source| TriangulationError::Unprojection {
                    index: input_index,
                    view_id: observation.view_id,
                    source,
                })?;
            let ray = WorldRay::try_from_camera(observation.view_id, observation.pose, bearing)
                .map_err(TriangulationError::Numerical)?;
            Ok(PreparedObservation {
                input_index,
                pixel: observation.pixel,
                ray,
                pose: observation.pose,
                camera: observation.camera,
            })
        })
        .collect::<Result<_, _>>()?;
    canonicalize_prepared(prepared)
}

fn canonicalize_prepared(
    mut prepared: Vec<PreparedObservation<'_>>,
) -> Result<Vec<PreparedObservation<'_>>, TriangulationError> {
    prepared.sort_by_key(|observation| observation.ray.view_id());
    if let Some(duplicate) = prepared
        .windows(2)
        .find(|pair| pair[0].ray.view_id() == pair[1].ray.view_id())
    {
        return Err(TriangulationError::DuplicateView(
            duplicate[0].ray.view_id(),
        ));
    }
    Ok(prepared)
}
