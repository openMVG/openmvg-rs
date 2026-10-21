// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Assembly of a [`Reconstruction`] from stored parts.
//!
//! ```text
//! ReconstructionParts (cameras, views, poses, points with their observations)
//!        |  check view IDs, pose blocks, observed views
//!        v
//! feature geometry: one feature per observation, numbered per view in point order
//!        |
//!        v
//! one track and one active landmark per point  ->  Reconstruction (validated)
//! ```

use nalgebra::{Point2, Point3};
use openmvg_core::{CameraModel, PixelResidual, RunSeed};
use openmvg_features::tracking::TrackSetError;

use super::Reconstruction;
use crate::model::{Landmark, Landmarks, ObservationSupport, PoseRecord, ViewRecord};
use crate::workflow::run::SfmEngine;
use crate::workspace::builder::SfmWorkspaceBuilder;
use crate::{
    Camera, FeatureId, FeatureSet, LandmarkId, PoseId, Result, SfmError, SfmWorkspace, Track,
    TrackId, TrackObservation, TrackSet, ViewId, WorldToCameraPose,
};

/// The stored content of a [`Reconstruction`], from which [`Reconstruction::try_from_parts`]
/// rebuilds it.
///
/// It holds what a scene file records: camera models, views, the pose of every registered pose
/// block, and the points with the pixels that observe them. Feature geometry, tracks, and
/// reprojection errors are derived on assembly.
#[derive(Debug, Clone)]
#[must_use = "the parts have no effect until Reconstruction::try_from_parts is called"]
pub struct ReconstructionParts {
    root_seed: RunSeed,
    engine: SfmEngine,
    cameras: Vec<Camera>,
    views: Vec<ViewRecord>,
    poses: Vec<(PoseId, WorldToCameraPose)>,
    points: Vec<PartPoint>,
    calibrated_intrinsics: bool,
}

#[derive(Debug, Clone)]
struct PartPoint {
    position: Point3<f64>,
    observations: Vec<(ViewId, Point2<f64>)>,
}

impl ReconstructionParts {
    /// Parts with no poses or points yet: the engine that produced the reconstruction, camera
    /// models indexed by [`IntrinsicId`](crate::IntrinsicId), and views indexed by [`ViewId`].
    pub fn new(
        root_seed: impl Into<RunSeed>,
        engine: SfmEngine,
        cameras: Vec<Camera>,
        views: Vec<ViewRecord>,
    ) -> Self {
        Self {
            root_seed: root_seed.into(),
            engine,
            cameras,
            views,
            poses: Vec::new(),
            points: Vec::new(),
            calibrated_intrinsics: false,
        }
    }

    /// Register pose block `pose`, and with it every view that uses it.
    pub fn with_pose(mut self, pose: PoseId, world_to_camera: WorldToCameraPose) -> Self {
        self.poses.push((pose, world_to_camera));
        self
    }

    /// Add one point, which gets the next [`LandmarkId`], with its measured pixel in each
    /// registered view that observes it.
    pub fn with_point(
        mut self,
        position: Point3<f64>,
        observations: Vec<(ViewId, Point2<f64>)>,
    ) -> Self {
        self.points.push(PartPoint {
            position,
            observations,
        });
        self
    }

    /// Mark the camera models as a known calibration that the engine kept fixed, rather than
    /// estimates it refined. Off by default.
    pub const fn with_calibrated_intrinsics(mut self, calibrated: bool) -> Self {
        self.calibrated_intrinsics = calibrated;
        self
    }
}

impl Reconstruction {
    /// Rebuild a reconstruction from its stored parts.
    ///
    /// Each observation becomes a feature of its view, numbered per view in the order the points
    /// were added, and each point becomes the track of its observations and an active landmark
    /// supported by all of them. A landmark's reprojection error is the mean pixel distance over
    /// the observations that project. The reconstruction has no components, since the parts do
    /// not record them.
    ///
    /// # Errors
    ///
    /// Fails like [`crate::DatasetBuilder::build`] if the camera models or their assignment do
    /// not match the views. Otherwise fails with [`SfmError::InvalidInput`] if a view ID differs
    /// from its position, a pose block is unknown or given twice, or, naming the point, if a
    /// point has fewer than two observations, observes a view twice, an unknown view, or an
    /// unregistered one, or has a non-finite position or pixel.
    pub fn try_from_parts(parts: ReconstructionParts) -> Result<Self> {
        let ReconstructionParts {
            root_seed,
            engine,
            cameras,
            views,
            poses,
            points,
            calibrated_intrinsics,
        } = parts;
        if let Some((index, view)) = views
            .iter()
            .enumerate()
            .find(|(index, view)| view.id.index() != *index)
        {
            return Err(SfmError::InvalidInput(format!(
                "view {} is stored at position {index}; view IDs must be dense and in order",
                view.id
            )));
        }

        let mut pixels = vec![Vec::new(); views.len()];
        let mut tracks = Vec::with_capacity(points.len());
        for (index, point) in points.iter().enumerate() {
            if !point.position.coords.iter().all(|value| value.is_finite()) {
                return Err(point_error(index, "has a non-finite position"));
            }
            let mut observations = Vec::with_capacity(point.observations.len());
            for &(view, pixel) in &point.observations {
                let features = pixels.get_mut(view.index()).ok_or_else(|| {
                    point_error(index, &format!("observes the unknown view {view}"))
                })?;
                if !pixel.coords.iter().all(|value| value.is_finite()) {
                    return Err(point_error(
                        index,
                        &format!("has a non-finite pixel in view {view}"),
                    ));
                }
                observations.push(TrackObservation::new(view, FeatureId::new(features.len())));
                features.push(pixel);
            }
            let track = Track::try_new(TrackId::new(index), observations).map_err(|error| {
                point_error(
                    index,
                    &match error {
                        TrackSetError::TooShort { .. } => "has fewer than two observations".into(),
                        TrackSetError::DuplicateView { .. } => "observes a view twice".into(),
                        other => other.to_string(),
                    },
                )
            })?;
            tracks.push(track);
        }
        let feature_sets = pixels
            .into_iter()
            .map(FeatureSet::try_new)
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut workspace = SfmWorkspaceBuilder::new()
            .root_seed(root_seed)
            .image_meta(views.iter().map(|view| view.meta.clone()))
            .camera_models(cameras)
            .view_intrinsics(views.iter().map(|view| view.intrinsic_id))
            .view_poses(views.iter().map(|view| view.pose_id))
            .tracks(TrackSet::try_new(tracks)?, feature_sets)
            .build()?;
        for (pose, world_to_camera) in poses {
            let previous = workspace
                .poses
                .set(pose.index(), PoseRecord::new(pose, world_to_camera, None))
                .map_err(|_| SfmError::InvalidInput(format!("pose block {pose} does not exist")))?;
            if previous.is_some() {
                return Err(SfmError::InvalidInput(format!(
                    "pose block {pose} is given twice"
                )));
            }
        }

        let landmarks = points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let error = mean_reprojection_error(&workspace, point);
                let error = PixelResidual::try_new(error).map_err(|_| {
                    point_error(index, "has a reprojection error that is not finite")
                })?;
                Ok(Landmark::active(
                    LandmarkId::new(index),
                    TrackId::new(index),
                    point.position,
                    error,
                    ObservationSupport::default(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        workspace.landmarks = Landmarks::try_new(landmarks)?;
        workspace.validate()?;
        Ok(Self::new(workspace, engine, calibrated_intrinsics))
    }
}

/// The mean pixel distance between `point` projected into each observing view and its measured
/// pixel, over the observations that project; zero if none does.
fn mean_reprojection_error(workspace: &SfmWorkspace, point: &PartPoint) -> f64 {
    let distances: Vec<f64> = point
        .observations
        .iter()
        .filter_map(|&(view, pixel)| {
            let record = workspace.view(view)?;
            let camera = workspace.camera_model(record.intrinsic_id)?;
            let pose = &workspace.pose(view)?.pose;
            let projected = camera
                .project(&pose.project_to_camera(&point.position).coords)
                .ok()?;
            Some((projected - pixel).norm())
        })
        .collect();
    if distances.is_empty() {
        0.0
    } else {
        distances.iter().sum::<f64>() / distances.len() as f64
    }
}

fn point_error(point: usize, message: &str) -> SfmError {
    SfmError::InvalidInput(format!("point {point} {message}"))
}

#[cfg(test)]
mod tests;
