// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pose-registration (resection) contract: [`PoseRegistrar`], its 2D-3D correspondences, and
//! the pose estimate it returns.

use nalgebra::{Point2, Point3};
use openmvg_core::{AngularResidual, RunSeed, camera::Camera, model::WorldToCameraPose};

use crate::{LandmarkId, Result, ViewId};

/// One 2D-3D correspondence between an active landmark and its measurement in the view
/// being registered.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PoseRegistrationObservation {
    /// Landmark providing the 3D point.
    pub landmark: LandmarkId,
    /// Landmark position in world coordinates.
    pub point: Point3<f64>,
    /// Measured image position in the view being registered, in pixels.
    pub xy: Point2<f64>,
}

/// Per-view inputs passed to a [`PoseRegistrar`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PoseRegistrationContext {
    /// View being registered.
    pub view: ViewId,
    /// Camera model of that view, used to turn pixels into bearings.
    pub camera: Camera,
    /// Whether the registrar may estimate the focal length of `camera`.
    ///
    /// True when the view's intrinsic group is an
    /// [`IntrinsicPrior::Guess`](crate::model::IntrinsicPrior::Guess) and no other registered view
    /// shares it, so the focal in `camera` is a placeholder only this view's measurements can
    /// correct. When false, the focal is trusted or already fixed by a registered view, and an
    /// estimate carrying [`PoseRegistrationEstimate::focal_px`] is rejected as invalid input.
    pub focal_estimable: bool,
    /// Per-view stream seed derived from the run seed; the only randomness source a
    /// registrar may use.
    pub seed: RunSeed,
}

/// Pose returned by a [`PoseRegistrar`], before the pipeline's inlier and pose-edge gates.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PoseRegistrationEstimate {
    /// Estimated world-to-camera pose.
    pub pose: WorldToCameraPose,
    /// Inliers among the supplied observations; must not exceed their number.
    pub inliers: usize,
    /// Median unsquared angular bearing residual, in radians, evaluated with
    /// [`Self::focal_px`] when the estimate carries one.
    pub residual_median: AngularResidual,
    /// Non-empty name of the estimation method, recorded in the registration report.
    pub method: String,
    /// Focal length, in pixels, estimated together with the pose; `None` keeps the focal of
    /// [`PoseRegistrationContext::camera`].
    ///
    /// The pipeline sets the focal of the view's intrinsic group to this value, scaling both
    /// focal lengths of the camera by the same factor so the aspect ratio, principal point, and
    /// distortion are unchanged. It must be finite and positive, and the registrar may return
    /// it only when [`PoseRegistrationContext::focal_estimable`] is true.
    pub focal_px: Option<f64>,
}

/// Estimates the pose of a new view from 2D-3D correspondences (resection).
pub trait PoseRegistrar: Send + Sync {
    /// Estimate the world-to-camera pose of `ctx.view`.
    ///
    /// `Ok(None)` means no pose could be estimated, and the pipeline rejects the view; `Err`
    /// is reserved for invalid input or an implementation failure.
    fn estimate_pose(
        &self,
        ctx: &PoseRegistrationContext,
        observations: &[PoseRegistrationObservation],
    ) -> Result<Option<PoseRegistrationEstimate>>;
}
