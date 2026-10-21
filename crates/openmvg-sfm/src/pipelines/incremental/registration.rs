// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Registration of one view: resect it against the active landmarks with a
//! [`PoseRegistrar`], gate the estimate, and commit the outcome.

use nalgebra::{Point2, Point3, Rotation3};
#[cfg(test)]
use nalgebra::{UnitVector3, Vector3};
use openmvg_core::AngularThreshold;
use openmvg_core::camera::Camera;
#[cfg(test)]
use openmvg_core::camera::CameraModel;
use openmvg_geometry::pose::p4pf::estimate_absolute_pose_with_focal;
use openmvg_geometry::pose::{
    AbsolutePoseEstimate, AbsolutePoseOptions, BearingToPoint, PixelToPoint, PoseRefinement,
    estimate_absolute_pose,
};
use openmvg_geometry::robust::RobustOptions;
use openmvg_geometry::robust::threshold::PixelThreshold;

use crate::backends::{
    PoseRegistrar, PoseRegistrationContext, PoseRegistrationEstimate, PoseRegistrationObservation,
};
use crate::model::{IntrinsicPrior, LandmarkStatus};
use crate::{ComponentId, IntrinsicId, Result, SfmError, SfmWorkspace, ViewId};

use super::registration_index::RegistrationObservationIndex;

/// Why an incremental registration attempt for one view was rejected.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum RegistrationFailureReason {
    /// The view already has a pose; nothing was attempted.
    AlreadyRegistered,
    /// Fewer active 2D-3D observations than the configured minimum; nothing was attempted.
    NotEnoughVisibleLandmarks,
    /// The robust pose estimation returned no pose.
    PoseEstimationFailed,
    /// The estimate kept fewer inliers than
    /// [`registration_min_inliers`](crate::config::IncrementalSfmConfig::registration_min_inliers),
    /// or a smaller share of the attempted observations than
    /// [`registration_min_inlier_ratio`](crate::config::IncrementalSfmConfig::registration_min_inlier_ratio).
    NotEnoughInliers,
    /// The resected rotation disagrees with the view's relative-rotation edges to
    /// already-posed neighbors by more than the allowed threshold: the pose is likely a mirror
    /// solution or otherwise wrong.
    RotationInconsistent,
    /// The resected camera center disagrees with the view's relative-translation edges
    /// to already-posed neighbors by more than the allowed threshold: the view would sit in the
    /// wrong place even though its rotation is consistent.
    TranslationInconsistent,
    /// Robust retriangulation kept fewer supporting points than registration requires.
    /// The tentative pose is rolled back so later map growth may retry it with stronger support.
    PostRefitInsufficientSupport,
    /// A previously registered pose outside the initial pair retained too few
    /// reprojection-valid landmark observations after bundle adjustment. The pose is removed,
    /// but its tracks remain, so later growth may register the view again with more support.
    PostBundleInsufficientSupport,
}

impl RegistrationFailureReason {
    /// Reason name, the same spelling as its JSON form, such as `not_enough_inliers`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AlreadyRegistered => "already_registered",
            Self::NotEnoughVisibleLandmarks => "not_enough_visible_landmarks",
            Self::PoseEstimationFailed => "pose_estimation_failed",
            Self::NotEnoughInliers => "not_enough_inliers",
            Self::RotationInconsistent => "rotation_inconsistent",
            Self::TranslationInconsistent => "translation_inconsistent",
            Self::PostRefitInsufficientSupport => "post_refit_insufficient_support",
            Self::PostBundleInsufficientSupport => "post_bundle_insufficient_support",
        }
    }
}

/// Writes [`RegistrationFailureReason::as_str`].
impl std::fmt::Display for RegistrationFailureReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A registration attempt that did not produce a pose.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RegistrationFailure {
    /// The view that was not registered.
    pub view: ViewId,
    /// Why it was not registered.
    pub reason: RegistrationFailureReason,
}

/// An accepted resection of one view.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RegistrationReport {
    /// The registered view.
    pub view: ViewId,
    /// Non-empty name of the pose estimator that produced the pose.
    pub method: String,
    /// 2D-3D observations passed to the registrar.
    pub attempted_observations: usize,
    /// Registrar inliers among `attempted_observations`.
    pub inliers: usize,
    /// Focal length, in pixels, the registrar estimated with the pose and the pipeline applied
    /// to the view's intrinsic group; `None` when the focal was left unchanged.
    pub focal_px: Option<f64>,
    /// Median unsquared angular bearing residual, in radians.
    pub residual_median: openmvg_core::AngularResidual,
    /// The accepted world-to-camera pose.
    pub pose: openmvg_core::model::WorldToCameraPose,
}

/// Result of one registration attempt.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistrationVerdict {
    /// The pose passed every gate and was accepted.
    Registered(RegistrationReport),
    /// Estimation ran but returned no pose, or its pose failed an acceptance gate.
    Rejected(RegistrationFailure),
    /// No estimation ran: the view is already posed or lacks enough 2D-3D observations.
    Deferred(RegistrationFailure),
}

/// Acceptance gates for resecting one view against the current landmarks.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RegistrationConfig {
    min_observations: usize,
    min_inliers: usize,
    min_inlier_ratio: f64,
    max_rotation_edge_error: Option<AngularThreshold>,
    max_translation_edge_error: Option<AngularThreshold>,
}

config_accessors!(RegistrationConfig {
    /// Minimum 2D-3D observations required to attempt pose estimation.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy min_observations, with_min_observations: usize;
    /// Minimum RANSAC inliers required to accept a registered pose.
    copy min_inliers, with_min_inliers: usize;
    /// Minimum P3P inlier *ratio* (inliers / attempted observations) to accept a
    /// pose. A best-connected camera can register against many landmarks yet fit
    /// only a handful (a wrong pose); the absolute `min_inliers` alone lets those
    /// through and they poison re-triangulation. Requiring a ratio defers such a
    /// camera until more structure supports it. `0.0` disables the ratio gate.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy min_inlier_ratio, with_min_inlier_ratio: f64;
    /// Reject a resected pose whose rotation disagrees (median over posed neighbors)
    /// with its relative-rotation graph edges by more than this threshold — a
    /// fold guard that enforces the global rotation consistency the averaging engine
    /// has and P3P chaining lacks. `None` disables the check.
    copy max_rotation_edge_error, with_max_rotation_edge_error: Option<AngularThreshold>;
    /// Optionally reject a resected pose whose **camera center** disagrees (median over
    /// posed neighbors) with its relative-**translation** graph edges by more than this
    /// this threshold. This is intended for a conditioned graph: raw pairwise directions
    /// can be less reliable than a P3P pose supported by many 2D-3D observations.
    /// `None` = no check (the default).
    copy max_translation_edge_error, with_max_translation_edge_error: Option<AngularThreshold>;
});

impl Default for RegistrationConfig {
    fn default() -> Self {
        Self {
            min_observations: 6,
            min_inliers: 6,
            max_rotation_edge_error: None,
            max_translation_edge_error: None,
            min_inlier_ratio: 0.25,
        }
    }
}

/// Immutable registration input prepared from one reconstruction snapshot.
///
/// Keeping preparation separate from estimation makes the proposal boundary explicit and lets
/// registration remain read-only until the driver opens its commit transaction.
#[derive(Debug, Clone)]
pub(crate) struct PreparedRegistration {
    view: ViewId,
    context: PoseRegistrationContext,
    observations: Vec<PoseRegistrationObservation>,
}

#[derive(Debug, Clone)]
pub(crate) enum RegistrationPreparation {
    Ready(PreparedRegistration),
    Finished(RegistrationVerdict),
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub(crate) fn prepare_view_registration(
    workspace: &SfmWorkspace,
    view: ViewId,
    config: &RegistrationConfig,
) -> Result<RegistrationPreparation> {
    prepare_registration_with_min_observations(workspace, view, config.min_observations, None)
}

pub(crate) fn prepare_view_registration_indexed(
    workspace: &SfmWorkspace,
    index: &RegistrationObservationIndex,
    view: ViewId,
    config: &RegistrationConfig,
) -> Result<RegistrationPreparation> {
    prepare_registration_with_min_observations(
        workspace,
        view,
        config.min_observations,
        Some(index),
    )
}

fn prepare_registration_with_min_observations(
    workspace: &SfmWorkspace,
    view: ViewId,
    min_observations: usize,
    index: Option<&RegistrationObservationIndex>,
) -> Result<RegistrationPreparation> {
    if workspace.has_pose(view) {
        return Ok(RegistrationPreparation::Finished(
            RegistrationVerdict::Deferred(RegistrationFailure {
                view,
                reason: RegistrationFailureReason::AlreadyRegistered,
            }),
        ));
    }

    let observations = index.map_or_else(
        || pose_registration_observations(workspace, view),
        |index| pose_registration_observations_indexed(workspace, index, view),
    );
    if observations.len() < min_observations {
        return Ok(RegistrationPreparation::Finished(
            RegistrationVerdict::Deferred(RegistrationFailure {
                view,
                reason: RegistrationFailureReason::NotEnoughVisibleLandmarks,
            }),
        ));
    }

    let image_record = workspace.view(view).ok_or(SfmError::InvalidId {
        kind: ViewId::kind(),
        index: view.index(),
    })?;
    let camera = workspace
        .camera_model(image_record.intrinsic_id)
        .ok_or(SfmError::InvalidId {
            kind: IntrinsicId::kind(),
            index: image_record.intrinsic_id.index(),
        })?
        .clone();
    let focal_estimable = workspace.intrinsic_prior(image_record.intrinsic_id)
        == Some(IntrinsicPrior::Guess)
        && !workspace.registered_views().any(|(other, _)| {
            other != view
                && workspace
                    .view(other)
                    .is_some_and(|record| record.intrinsic_id == image_record.intrinsic_id)
        });
    Ok(RegistrationPreparation::Ready(PreparedRegistration {
        view,
        context: PoseRegistrationContext {
            view,
            camera,
            focal_estimable,
            seed: workspace.derive_seed(
                crate::determinism::POSE_REGISTRATION_SEED_DOMAIN,
                &[view.value()],
            ),
        },
        observations,
    }))
}

pub(crate) fn evaluate_prepared_registration(
    workspace: &SfmWorkspace,
    prepared: &PreparedRegistration,
    registrar: &dyn PoseRegistrar,
    config: &RegistrationConfig,
) -> Result<RegistrationVerdict> {
    let view = prepared.view;
    let Some(estimate) = registrar.estimate_pose(&prepared.context, &prepared.observations)? else {
        return Ok(RegistrationVerdict::Rejected(RegistrationFailure {
            view,
            reason: RegistrationFailureReason::PoseEstimationFailed,
        }));
    };
    if estimate.inliers > prepared.observations.len() {
        return Err(SfmError::InvalidInput(format!(
            "pose registrar reported {} inliers for {} observations",
            estimate.inliers,
            prepared.observations.len()
        )));
    }
    if estimate.method.trim().is_empty() {
        return Err(SfmError::InvalidInput(
            "pose registrar returned an empty method name".to_owned(),
        ));
    }
    if let Some(focal) = estimate.focal_px {
        if !prepared.context.focal_estimable {
            return Err(SfmError::InvalidInput(format!(
                "pose registrar estimated a focal for view {view}, whose intrinsic group is shared \
                 with a registered view"
            )));
        }
        if !(focal.is_finite() && focal > 0.0) {
            return Err(SfmError::InvalidInput(format!(
                "pose registrar returned focal {focal} px; it must be finite and positive"
            )));
        }
    }
    let ratio_floor =
        (config.min_inlier_ratio * prepared.observations.len() as f64).ceil() as usize;
    if estimate.inliers < config.min_inliers.max(ratio_floor) {
        return Ok(RegistrationVerdict::Rejected(RegistrationFailure {
            view,
            reason: RegistrationFailureReason::NotEnoughInliers,
        }));
    }

    if let Some(reason) = registration_pose_edge_failure(workspace, view, &estimate.pose, config) {
        return Ok(RegistrationVerdict::Rejected(RegistrationFailure {
            view,
            reason,
        }));
    }

    Ok(RegistrationVerdict::Registered(RegistrationReport {
        view,
        method: estimate.method,
        attempted_observations: prepared.observations.len(),
        inliers: estimate.inliers,
        focal_px: estimate.focal_px,
        residual_median: estimate.residual_median,
        pose: estimate.pose,
    }))
}

fn registration_pose_edge_failure(
    workspace: &SfmWorkspace,
    view: ViewId,
    pose: &openmvg_core::model::WorldToCameraPose,
    config: &RegistrationConfig,
) -> Option<RegistrationFailureReason> {
    if config.max_rotation_edge_error.is_some_and(|threshold| {
        let max_deg = threshold.value().to_degrees();
        rotation_edge_consistency_deg(workspace, view, pose.rotation())
            .is_some_and(|median_err| median_err > max_deg)
    }) {
        return Some(RegistrationFailureReason::RotationInconsistent);
    }
    if config.max_translation_edge_error.is_some_and(|threshold| {
        let max_deg = threshold.value().to_degrees();
        translation_edge_consistency_deg(workspace, view, pose.center(), pose.rotation())
            .is_some_and(|median_err| median_err > max_deg)
    }) {
        return Some(RegistrationFailureReason::TranslationInconsistent);
    }
    None
}

/// Commit a registration proposal without rebuilding reconstruction components.
///
/// The incremental growth driver immediately follows an accepted pose with its scoped track
/// update, which owns the single component refresh for that transaction. Public one-shot
/// registration remains self-contained through [`register_view`].
pub(crate) fn commit_registration_outcome(
    workspace: &mut SfmWorkspace,
    outcome: RegistrationVerdict,
) -> Result<RegistrationVerdict> {
    match &outcome {
        RegistrationVerdict::Registered(report) => {
            if let Some(focal) = report.focal_px {
                apply_estimated_focal(workspace, report.view, focal)?;
            }
            workspace.set_pose_for_view(
                report.view,
                report.pose.clone(),
                Some(ComponentId::new(0)),
            )?;
        }
        RegistrationVerdict::Rejected(failure) | RegistrationVerdict::Deferred(failure) => {
            workspace.record_registration_failure(failure.clone());
        }
    }
    Ok(outcome)
}

/// Set the focal length of `view`'s intrinsic group to `focal_px`, keeping the camera's aspect
/// ratio, principal point, and distortion.
///
/// Landmarks triangulated before the change keep their positions; the refinement that follows
/// a registration brings them back in line with the new focal.
fn apply_estimated_focal(workspace: &mut SfmWorkspace, view: ViewId, focal_px: f64) -> Result<()> {
    let intrinsic_id = workspace
        .view(view)
        .ok_or(SfmError::InvalidId {
            kind: ViewId::kind(),
            index: view.index(),
        })?
        .intrinsic_id;
    let camera = workspace
        .camera_model(intrinsic_id)
        .ok_or(SfmError::InvalidId {
            kind: IntrinsicId::kind(),
            index: intrinsic_id.index(),
        })?;
    let rescaled = with_focal_px(camera, focal_px)?;
    workspace.replace_camera_model(intrinsic_id, rescaled)
}

/// `camera` with both focal lengths scaled so the horizontal one equals `focal_px`.
fn with_focal_px(camera: &Camera, focal_px: f64) -> Result<Camera> {
    let unsupported = |camera_model: &'static str| SfmError::UnsupportedCameraModel {
        context: "focal estimation during registration",
        camera_model,
    };
    let invalid = |error: openmvg_core::camera::CameraError| {
        SfmError::InvalidInput(format!("estimated focal {focal_px} px: {error}"))
    };
    match camera {
        Camera::Pinhole(pinhole) => {
            let scale = focal_px / pinhole.focal_x();
            Camera::try_pinhole(
                focal_px,
                pinhole.focal_y() * scale,
                pinhole.principal_x(),
                pinhole.principal_y(),
            )
            .map_err(invalid)
        }
        Camera::BrownConrady(brown) => {
            let scale = focal_px / brown.focal_x();
            Camera::try_brown_conrady(
                focal_px,
                brown.focal_y() * scale,
                brown.principal_x(),
                brown.principal_y(),
                *brown.radial_distortion(),
                *brown.tangential_distortion(),
            )
            .map_err(invalid)
        }
        Camera::Fisheye(_) => Err(unsupported("Fisheye")),
        Camera::Spherical(_) => Err(unsupported("Spherical")),
    }
}

/// Resect `image` against the active landmarks with `registrar` and commit the outcome.
///
/// An accepted pose is written to the workspace in component 0; a rejected or deferred attempt
/// is recorded as a registration failure. Gate failures are outcomes, not errors.
///
/// # Errors
///
/// [`SfmError::InvalidId`] if the view or its intrinsic is unknown, and
/// [`SfmError::InvalidInput`] if the registrar reports more inliers than observations or an
/// empty method name, plus any error the registrar or the workspace update returns.
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub fn register_view(
    workspace: &mut SfmWorkspace,
    view: ViewId,
    registrar: &dyn PoseRegistrar,
    config: &RegistrationConfig,
) -> Result<RegistrationVerdict> {
    let outcome = match prepare_view_registration(workspace, view, config)? {
        RegistrationPreparation::Ready(prepared) => {
            evaluate_prepared_registration(workspace, &prepared, registrar, config)?
        }
        RegistrationPreparation::Finished(outcome) => outcome,
    };
    let registered = matches!(outcome, RegistrationVerdict::Registered(_));
    let outcome = commit_registration_outcome(workspace, outcome)?;
    if registered {
        crate::ops::reconstruction::refresh_primary_component(workspace);
    }
    Ok(outcome)
}

pub(crate) fn pose_registration_observations(
    workspace: &SfmWorkspace,
    view: ViewId,
) -> Vec<PoseRegistrationObservation> {
    let mut observations = Vec::new();
    for landmark in workspace.landmarks().as_slice() {
        if landmark.status != LandmarkStatus::Active {
            continue;
        }
        let Some(track) = workspace.tracks().get(landmark.source_track) else {
            continue;
        };
        let Some(track_observation) = track
            .observations()
            .iter()
            .find(|observation| observation.view == view)
        else {
            continue;
        };
        let Ok(xy) = workspace.track_observation_pixel(track_observation) else {
            continue;
        };
        observations.push(PoseRegistrationObservation {
            landmark: landmark.id,
            point: landmark.point,
            xy,
        });
    }
    observations.sort_by_key(|observation| observation.landmark);
    observations
}

pub(crate) fn pose_registration_observations_indexed(
    workspace: &SfmWorkspace,
    index: &RegistrationObservationIndex,
    view: ViewId,
) -> Vec<PoseRegistrationObservation> {
    let mut observations = Vec::with_capacity(index.visible_landmarks(view));
    for entry in index.observations(workspace, view) {
        let Some(landmark) = workspace.landmarks().get(entry.landmark) else {
            continue;
        };
        if landmark.status != LandmarkStatus::Active {
            continue;
        }
        let Some(track) = workspace.tracks().get(landmark.source_track) else {
            continue;
        };
        let Some(track_observation) = track.observations().get(entry.track_observation.index())
        else {
            continue;
        };
        if track_observation.view != view {
            continue;
        }
        let Ok(xy) = workspace.track_observation_pixel(track_observation) else {
            continue;
        };
        observations.push(PoseRegistrationObservation {
            landmark: landmark.id,
            point: landmark.point,
            xy,
        });
    }
    observations
}

/// GenRANSAC registrar using FusedP3P hypotheses, each with positive-depth, residual, and
/// distinctness checks.
///
/// Hypotheses use an angular gate and model-compatible all-inlier refinement:
/// perspective cameras retain their pixel-chart objective, while generalized
/// cameras use the full-sphere angular objective.
#[derive(Debug, Clone)]
pub struct GenRansacP3pRegistrar {
    max_draws: usize,
    confidence: f64,
    inlier_angle_rad: f64,
}

config_accessors!(GenRansacP3pRegistrar {
    /// Hard cap on minimal samples across all GenRANSAC rounds.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy max_draws, with_max_draws: usize;
    /// Target probability of drawing an all-inlier minimal sample.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy confidence, with_confidence: f64;
    /// Inlier gate as an unsquared angular reprojection threshold in radians; spherical views
    /// use twice this value.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy inlier_angle_rad, with_inlier_angle_rad: f64;
});

/// Factor applied to the inlier gate of a spherical view. A pixel of an equirectangular image
/// subtends `2π / width`, several times the angle of a pinhole pixel, so the angular gate that
/// admits a pinhole keypoint error of a few pixels admits under two pixels on a sphere. At twice
/// the gate, the 113-frame Prague sequence registers 107 views against 55 at the base gate.
const SPHERICAL_GATE_SCALE: f64 = 2.0;

impl Default for GenRansacP3pRegistrar {
    fn default() -> Self {
        Self {
            max_draws: 1000,
            confidence: 0.9999,
            inlier_angle_rad: 0.005,
        }
    }
}

impl PoseRegistrar for GenRansacP3pRegistrar {
    fn estimate_pose(
        &self,
        ctx: &PoseRegistrationContext,
        observations: &[PoseRegistrationObservation],
    ) -> Result<Option<PoseRegistrationEstimate>> {
        if observations.len() < 3 {
            return Ok(None);
        }
        let gate = match ctx.camera {
            Camera::Spherical(_) => self.inlier_angle_rad * SPHERICAL_GATE_SCALE,
            _ => self.inlier_angle_rad,
        };
        let threshold = match AngularThreshold::try_new(gate) {
            Ok(threshold) => threshold,
            Err(_) => return Ok(None),
        };
        let corrs = observations
            .iter()
            .map(|obs| bearing_pose_correspondence(obs.point, obs.xy, &ctx.camera))
            .collect::<Result<Vec<_>>>()?;
        let refinement = match &ctx.camera {
            Camera::Pinhole(_) | Camera::BrownConrady(_) => PoseRefinement::Perspective,
            Camera::Fisheye(_) | Camera::Spherical(_) => PoseRefinement::Bearing,
        };
        let options = AbsolutePoseOptions::default()
            .with_robust_options(
                RobustOptions::default()
                    .with_confidence(self.confidence)
                    .with_max_draws(self.max_draws)
                    .with_seed(ctx.seed.derive("sfm/pose-registration/genransac/v1", &[])),
            )
            .with_refinement(refinement);
        let result = match estimate_absolute_pose(&corrs, threshold, &options) {
            Ok(result) => result,
            Err(_) => return Ok(None),
        };
        Ok(finalize_p3p_pose(
            result,
            observations,
            "genransac-fused-p3p-resection",
        ))
    }
}

/// GenRANSAC registrar that estimates the focal length together with the pose.
///
/// For a view whose [`PoseRegistrationContext::focal_estimable`] is true it runs four-point
/// hypotheses with an unknown focal length (P4Pf), refits the pose and focal on the consensus,
/// and returns the focal in [`PoseRegistrationEstimate::focal_px`]. Every other view, and a
/// camera with lens distortion, is registered by the [`GenRansacP3pRegistrar`] it wraps, which
/// also supplies the search controls.
///
/// The search threshold is an image-plane distance of `inlier_angle_rad` times 0.6 of the longest
/// image side, because the focal length is the unknown; the reported inliers and residual are
/// counted at the estimated focal with the same angle.
#[derive(Debug, Clone, Default)]
pub struct GenRansacP4pfRegistrar {
    p3p: GenRansacP3pRegistrar,
}

/// Fraction of the longest image side used as the focal length that turns the angular inlier
/// gate into the pixel threshold of the focal search.
const SEARCH_FOCAL_FRACTION: f64 = 0.6;
/// Minimum correspondences of one four-point hypothesis.
const P4PF_MIN_OBSERVATIONS: usize = 4;

impl GenRansacP4pfRegistrar {
    fn estimate_with_focal(
        &self,
        ctx: &PoseRegistrationContext,
        observations: &[PoseRegistrationObservation],
        principal: Point2<f64>,
    ) -> Option<PoseRegistrationEstimate> {
        let data = observations
            .iter()
            .map(|obs| PixelToPoint::try_new(obs.point, obs.xy))
            .collect::<std::result::Result<Vec<_>, _>>()
            .ok()?;
        let longest_side = 2.0 * principal.x.max(principal.y);
        let threshold = PixelThreshold::try_new(
            self.p3p.inlier_angle_rad * SEARCH_FOCAL_FRACTION * longest_side,
        )
        .ok()?;
        let options = RobustOptions::default()
            .with_confidence(self.p3p.confidence)
            .with_max_draws(self.p3p.max_draws)
            .with_seed(
                ctx.seed
                    .derive("sfm/pose-registration/genransac-p4pf/v1", &[]),
            );
        let estimate =
            estimate_absolute_pose_with_focal(&data, principal, threshold, &options).ok()?;
        let focal = estimate.model().focal_px();
        let angles: Vec<f64> = estimate
            .residuals()
            .iter()
            .map(|residual| residual.atan2(focal))
            .collect();
        let inliers = angles
            .iter()
            .filter(|angle| **angle <= self.p3p.inlier_angle_rad)
            .count();
        Some(PoseRegistrationEstimate {
            pose: estimate.model().pose().ok()?,
            inliers,
            residual_median: openmvg_core::AngularResidual::try_new(median(angles)).ok()?,
            method: "genransac-p4pf-resection".to_owned(),
            focal_px: Some(focal),
        })
    }
}

/// Whether every distortion coefficient of `camera` is zero, as in the declared-but-unestimated
/// radial model a view without focal metadata starts from.
fn is_distortion_free(camera: &openmvg_core::camera::BrownConradyCamera) -> bool {
    let radial = camera.radial_distortion();
    let tangential = camera.tangential_distortion();
    [
        radial.k1(),
        radial.k2(),
        radial.k3(),
        tangential.p1(),
        tangential.p2(),
    ]
    .iter()
    .all(|coefficient| *coefficient == 0.0)
}

impl PoseRegistrar for GenRansacP4pfRegistrar {
    fn estimate_pose(
        &self,
        ctx: &PoseRegistrationContext,
        observations: &[PoseRegistrationObservation],
    ) -> Result<Option<PoseRegistrationEstimate>> {
        let principal = match &ctx.camera {
            Camera::Pinhole(camera) => {
                Some(Point2::new(camera.principal_x(), camera.principal_y()))
            }
            Camera::BrownConrady(camera) if is_distortion_free(camera) => {
                Some(Point2::new(camera.principal_x(), camera.principal_y()))
            }
            _ => None,
        };
        match principal {
            Some(principal)
                if ctx.focal_estimable && observations.len() >= P4PF_MIN_OBSERVATIONS =>
            {
                Ok(self.estimate_with_focal(ctx, observations, principal))
            }
            _ => self.p3p.estimate_pose(ctx, observations),
        }
    }
}

fn finalize_p3p_pose(
    result: AbsolutePoseEstimate,
    observations: &[PoseRegistrationObservation],
    method: &str,
) -> Option<PoseRegistrationEstimate> {
    if result.residuals().len() != observations.len() {
        return None;
    }
    Some(PoseRegistrationEstimate {
        pose: result.model().clone(),
        inliers: result.inlier_count(),
        residual_median: openmvg_core::AngularResidual::try_new(median(
            result.residuals().to_vec(),
        ))
        .ok()?,
        method: method.to_owned(),
        focal_px: None,
    })
}

/// Whether the relative pose of the pair `(a, b)` was decomposed with a placeholder focal length.
///
/// A relative rotation and baseline direction come from an essential matrix fitted under both
/// cameras' focal lengths. When either focal is an
/// [`IntrinsicPrior::Guess`], the pose is biased by up to tens of degrees, and a correctly
/// registered view would disagree with it. Such a pair is not evidence for or against a pose.
fn pair_rests_on_a_guessed_focal(workspace: &SfmWorkspace, a: ViewId, b: ViewId) -> bool {
    [a, b].into_iter().any(|view| {
        workspace
            .view(view)
            .and_then(|record| workspace.intrinsic_prior(record.intrinsic_id))
            == Some(IntrinsicPrior::Guess)
    })
}

/// Median angular disagreement (deg) between a candidate rotation `r_est` for `image`
/// and the rotation predicted by each relative-rotation edge to an already-posed
/// neighbor. Convention (matches the averaging engine): for a canonical pair (a,b),
/// `R_b = R_rel · R_a`. `None` if the view has no posed neighbors with a relative pose.
pub(crate) fn rotation_edge_consistency_deg(
    workspace: &SfmWorkspace,
    view: ViewId,
    r_est: &Rotation3<f64>,
) -> Option<f64> {
    let neighbors = workspace.pair_graph().neighbors(view).ok()?;
    let mut errs: Vec<f64> = Vec::new();
    for nb in neighbors {
        if !workspace.pair_graph().is_active(nb.pair_id)
            || pair_rests_on_a_guessed_focal(workspace, view, nb.view)
        {
            continue;
        }
        let Some(nb_rec) = workspace.pose(nb.view) else {
            continue;
        };
        let Some(model) = workspace.geometric_model_store().get(nb.pair_id.index()) else {
            continue;
        };
        let Some(rp) = model.relative_pose() else {
            continue;
        };
        let Some(pair) = workspace.pair_graph().pair(nb.pair_id) else {
            continue;
        };
        let r_nb = nb_rec.pose.rotation();
        // R_b = R_rel · R_a. If `image` is a: R_a = R_relᵀ · R_b; if `image` is b: R_b = R_rel · R_a.
        let predicted = if pair.first() == view {
            rp.rotation_source_to_target().transpose() * r_nb
        } else {
            rp.rotation_source_to_target() * r_nb
        };
        let delta = r_est.matrix() * predicted.matrix().transpose();
        let cosine = ((delta.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        errs.push(cosine.acos().to_degrees());
    }
    if errs.is_empty() {
        return None;
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(errs[errs.len() / 2])
}

/// Median angular disagreement (deg) between the resected camera center `c_est` for
/// `image` and the direction predicted by each relative-translation edge to an
/// already-posed neighbor. Convention (matches the averaging engine): for a canonical
/// pair `(a, b)`, the source-frame baseline direction is the unit direction of `C_b` in
/// camera `a`, so its world-frame direction is `R_a^T * d_ab`. `None`
/// if the view has no posed neighbor with a usable relative translation.
pub(crate) fn translation_edge_consistency_deg(
    workspace: &SfmWorkspace,
    view: ViewId,
    c_est: &Point3<f64>,
    r_est: &Rotation3<f64>,
) -> Option<f64> {
    let neighbors = workspace.pair_graph().neighbors(view).ok()?;
    let mut errs: Vec<f64> = Vec::new();
    for nb in neighbors {
        if !workspace.pair_graph().is_active(nb.pair_id)
            || pair_rests_on_a_guessed_focal(workspace, view, nb.view)
        {
            continue;
        }
        let Some(nb_rec) = workspace.pose(nb.view) else {
            continue;
        };
        let Some(model) = workspace.geometric_model_store().get(nb.pair_id.index()) else {
            continue;
        };
        let Some(rp) = model.relative_pose() else {
            continue;
        };
        let Some(pair) = workspace.pair_graph().pair(nb.pair_id) else {
            continue;
        };
        let c_nb = nb_rec.pose.center().coords;
        let t_rel = rp.baseline_direction_in_source().into_inner();
        // World-frame direction of `C_b − C_a` from the edge = R_aᵀ · t_rel, and the
        // observed direction of `C_b − C_a` from the current centers.
        let (predicted, observed) = if pair.first() == view {
            // image is a: R_a = r_est, observed = C_nb − C_est.
            (r_est.transpose() * t_rel, c_nb - c_est.coords)
        } else {
            // image is b: R_a = R_nb, observed = C_est − C_nb.
            (
                nb_rec.pose.rotation().transpose() * t_rel,
                c_est.coords - c_nb,
            )
        };
        let (pn, on) = (predicted.norm(), observed.norm());
        if pn < 1e-9 || on < 1e-9 {
            continue;
        }
        let cos = (predicted.dot(&observed) / (pn * on)).clamp(-1.0, 1.0);
        errs.push(cos.acos().to_degrees());
    }
    if errs.is_empty() {
        return None;
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(errs[errs.len() / 2])
}

fn bearing_pose_correspondence(
    point: Point3<f64>,
    xy: Point2<f64>,
    camera: &openmvg_core::camera::Camera,
) -> Result<BearingToPoint> {
    let bearing = camera.bearing(&xy).map_err(|error| {
        SfmError::InvalidInput(format!(
            "pose registration could not compute a bearing: {error}"
        ))
    })?;
    BearingToPoint::try_new(point, bearing)
        .map_err(|error| SfmError::InvalidInput(error.to_string()))
}

fn median(mut values: Vec<f64>) -> f64 {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return f64::INFINITY;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p3p_registration_residual_is_an_angle_in_radians() {
        let angle: f64 = 0.2;
        let correspondence = BearingToPoint::try_new(
            Point3::new(0.0, 0.0, 5.0),
            UnitVector3::new_normalize(Vector3::new(angle.sin(), 0.0, angle.cos())),
        )
        .unwrap();
        let residual = openmvg_geometry::pose::ResectionSolver::residual(
            &openmvg_core::model::WorldToCameraPose::identity(),
            &correspondence,
        );
        assert!((residual - angle).abs() < 1e-12);
    }

    #[test]
    fn spherical_p3p_registers_and_refines_rear_hemisphere_points() {
        use openmvg_core::camera::{Camera, SphericalCamera};

        let camera = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
        let points = [
            Point3::new(-1.2, -0.4, -4.0),
            Point3::new(0.8, -0.7, -5.0),
            Point3::new(1.5, 0.6, -6.0),
            Point3::new(-0.9, 1.1, -5.5),
            Point3::new(0.2, -1.4, -7.0),
            Point3::new(-1.7, 0.3, -6.5),
            Point3::new(1.1, 1.5, -8.0),
            Point3::new(-0.3, -0.2, -3.5),
        ];
        let observations = points
            .iter()
            .enumerate()
            .map(|(index, point)| PoseRegistrationObservation {
                landmark: crate::LandmarkId::new(index),
                point: *point,
                xy: camera.project(&point.coords).unwrap(),
            })
            .collect::<Vec<_>>();

        let estimate = GenRansacP3pRegistrar {
            max_draws: 512,
            inlier_angle_rad: 1e-5,
            ..GenRansacP3pRegistrar::default()
        }
        .estimate_pose(
            &PoseRegistrationContext {
                view: ViewId::new(0),
                camera,
                focal_estimable: false,
                seed: 29.into(),
            },
            &observations,
        )
        .unwrap()
        .expect("rear-hemisphere spherical bearings should register");

        assert_eq!(estimate.inliers, observations.len());
        assert!(estimate.pose.center().coords.norm() < 1e-5);
        assert!(estimate.pose.rotation().angle() < 1e-5);
    }

    #[test]
    fn a_spherical_view_gets_twice_the_configured_angular_gate() {
        use openmvg_core::camera::{Camera, SphericalCamera};

        let camera = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
        let points = [
            Point3::new(-1.2, -0.4, 4.0),
            Point3::new(0.8, -0.7, 5.0),
            Point3::new(1.5, 0.6, 6.0),
            Point3::new(-0.9, 1.1, 5.5),
            Point3::new(0.2, -1.4, 7.0),
            Point3::new(-1.7, 0.3, 6.5),
            Point3::new(1.1, 1.5, 8.0),
            Point3::new(-0.3, -0.2, 3.5),
        ];
        // One row of pixels subtends π / 512 ≈ 0.0061 rad, alternating in sign so no pose
        // absorbs it: over the configured gate of 0.005 rad, under the spherical gate of 0.010.
        let observations = points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let mut xy = camera.project(&point.coords).unwrap();
                xy.y += if index % 2 == 0 { 1.0 } else { -1.0 };
                PoseRegistrationObservation {
                    landmark: crate::LandmarkId::new(index),
                    point: *point,
                    xy,
                }
            })
            .collect::<Vec<_>>();
        let context = PoseRegistrationContext {
            view: ViewId::new(0),
            camera,
            focal_estimable: false,
            seed: 31.into(),
        };

        let registered = GenRansacP3pRegistrar::default()
            .estimate_pose(&context, &observations)
            .unwrap()
            .expect("a one-row error is inside the spherical gate");
        // A configured gate of 0.0025 rad gives the spherical view 0.005 rad, the gate before
        // the factor existed.
        let unscaled = GenRansacP3pRegistrar::default()
            .with_inlier_angle_rad(0.0025)
            .estimate_pose(&context, &observations)
            .unwrap();

        assert!(registered.inliers >= 6, "{} inliers", registered.inliers);
        assert!(unscaled.is_none_or(|estimate| estimate.inliers < registered.inliers));
    }

    #[test]
    fn genransac_fused_p3p_recovers_pose_with_outliers() {
        let expected = openmvg_core::model::WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(0.08, -0.13, 0.05),
            Point3::new(0.31, -0.18, 0.24),
        )
        .unwrap();
        let intrinsics = crate::Camera::try_pinhole(700.0, 700.0, 400.0, 300.0).unwrap();
        let points = (0..32)
            .map(|index| {
                let u = index as f64;
                Point3::new(
                    1.8 * (0.71 * u + 0.2).sin(),
                    1.5 * (1.17 * u - 0.3).cos(),
                    4.0 + 0.12 * u + 0.5 * (0.43 * u).sin().abs(),
                )
            })
            .collect::<Vec<_>>();
        let image_points = points
            .iter()
            .map(|point| {
                let camera = expected.project_to_camera(point);
                Point2::new(
                    700.0 * camera.x / camera.z + 400.0,
                    700.0 * camera.y / camera.z + 300.0,
                )
            })
            .collect::<Vec<_>>();
        let mut observations = points
            .iter()
            .zip(&image_points)
            .enumerate()
            .map(|(index, (&point, &xy))| PoseRegistrationObservation {
                landmark: crate::LandmarkId::new(index),
                point,
                xy,
            })
            .collect::<Vec<_>>();
        for index in 0..8 {
            observations.push(PoseRegistrationObservation {
                landmark: crate::LandmarkId::new(points.len() + index),
                point: points[index],
                xy: image_points[(index + 13) % image_points.len()],
            });
        }

        let estimate = GenRansacP3pRegistrar {
            max_draws: 256,
            ..GenRansacP3pRegistrar::default()
        }
        .estimate_pose(
            &PoseRegistrationContext {
                view: ViewId::new(0),
                camera: intrinsics,
                focal_estimable: false,
                seed: 73.into(),
            },
            &observations,
        )
        .unwrap()
        .expect("GenRANSAC should recover the dominant pose");

        estimate
            .pose
            .validate()
            .expect("registrar must not return non-finite geometry");
        let delta = estimate.pose.rotation().matrix() * expected.rotation().matrix().transpose();
        let rotation_error = ((delta.trace() - 1.0) * 0.5).clamp(-1.0, 1.0).acos();
        let center_error = (estimate.pose.center() - expected.center()).norm();
        assert!(rotation_error < 2.0e-3, "rotation error: {rotation_error}");
        assert!(center_error < 1.0e-2, "center error: {center_error}");
        assert!(estimate.inliers >= points.len() - 1);
        assert_eq!(estimate.method, "genransac-fused-p3p-resection");
    }

    #[test]
    fn rescaling_a_focal_keeps_the_aspect_ratio_principal_point_and_distortion() {
        use openmvg_core::{RadialDistortion, TangentialDistortion};
        let brown = Camera::try_brown_conrady(
            800.0,
            840.0,
            320.0,
            240.0,
            RadialDistortion::K1K2K3(-0.1, 0.01, 0.0),
            TangentialDistortion::P1P2(0.001, -0.001),
        )
        .unwrap();

        let Camera::BrownConrady(rescaled) = with_focal_px(&brown, 400.0).unwrap() else {
            panic!("the model must stay Brown-Conrady");
        };

        assert_eq!((rescaled.focal_x(), rescaled.focal_y()), (400.0, 420.0));
        assert_eq!(
            (rescaled.principal_x(), rescaled.principal_y()),
            (320.0, 240.0)
        );
        let Camera::BrownConrady(original) = &brown else {
            unreachable!();
        };
        assert_eq!(rescaled.radial_distortion(), original.radial_distortion());
        assert_eq!(
            rescaled.tangential_distortion(),
            original.tangential_distortion()
        );

        let pinhole = Camera::try_pinhole(500.0, 500.0, 400.0, 300.0).unwrap();
        assert_eq!(
            with_focal_px(&pinhole, 250.0).unwrap().focal_lengths(),
            Some((250.0, 250.0))
        );
    }
}
