// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental scene initialization: turn a selected initial pair into the initial posed views
//! and triangulated structure that the growth loop then extends.
//!
//! Two bootstrap paths:
//! - **Pair** — place two views from a verified relative pose (unit baseline), then
//!   triangulate their shared tracks.
//! - **Oracle** — place the initial-pair views from ground-truth poses (testing only).
//!
//! Shared PoseIds are supported by stored reconstructions and backend bundle adjustment, but
//! this initializer currently requires one pose per view and rejects aliases before any work.
//!
//! Every public entry point is transactional: on any failure the workspace is rolled back to
//! its state at entry so a failed initial pair never leaves partial poses behind.

use nalgebra::{Point2, Point3, Rotation3, Vector3};
use openmvg_ba::RobustLoss;
use openmvg_core::camera::CameraModel;
use openmvg_core::model::WorldToCameraPose;

#[cfg(feature = "unstable-test-support")]
use crate::LandmarkId;
use crate::ba::backend::BundleAdjustmentBackend;
use crate::ba::config::{BundleAdjustConfig, OptimizationScope};
use crate::backends::Triangulator;
use crate::config::{GaugeScalePolicy, IncrementalSfmConfig, LengthUnit};
#[cfg(feature = "unstable-test-support")]
use crate::evaluation::ground_truth::GroundTruthReconstruction;
use crate::model::ReconstructionComponent;
#[cfg(feature = "unstable-test-support")]
use crate::model::{Landmark, Landmarks};
use crate::ops::reconstruction::{
    BundleReport, TriangulationRejectionSummary, triangulate_incremental,
};
use crate::ops::rollback::GeometryRollback;
use crate::{ComponentId, Result, SfmError, SfmWorkspace, ViewId};

use super::initial_pair::{InitialPairHypothesis, select_initial_pair_for_incremental};

/// Outcome of placing the initial pair and triangulating its initial structure.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IncrementalInitializationReport {
    /// The initial pair that was initialized.
    pub seed: InitialPairHypothesis,
    /// Views posed by the initialization.
    pub initialized_views: Vec<crate::ViewId>,
    /// Landmarks triangulated from the initial-pair views.
    pub initialized_landmarks: usize,
    /// Why initial-pair tracks failed to triangulate.
    pub triangulation_rejections: TriangulationRejectionSummary,
    /// Constrained two-view adjustment of the initial pair, when enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_bundle_adjustment: Option<BundleReport>,
}

/// Backend-aware initial-pair initializer used by the incremental driver.
///
/// An explicitly requested initial-pair adjustment fails before mutating any state when no
/// backend was supplied.
pub(crate) fn initialize_incremental_seed_with_bundle_adjustment(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    triangulator: &dyn Triangulator,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
) -> Result<IncrementalInitializationReport> {
    let mut poses = std::collections::BTreeSet::new();
    for view in workspace.views() {
        if !poses.insert(view.pose_id) {
            return Err(SfmError::UnsupportedSharedPoseInitialization { pose: view.pose_id });
        }
    }
    if config.bundle_adjust_initial_pair() && ba_backend.is_none() {
        return Err(SfmError::InvalidConfig(
            "bundle_adjust_initial_pair requires a bundle-adjustment backend".to_owned(),
        ));
    }
    let transaction = GeometryRollback::capture(workspace);
    match initialize_incremental_seed_inner(workspace, config, triangulator, ba_backend) {
        Ok(report) => Ok(report),
        Err(err) => {
            transaction.rollback(workspace);
            Err(err)
        }
    }
}

fn initialize_incremental_seed_inner(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    triangulator: &dyn Triangulator,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
) -> Result<IncrementalInitializationReport> {
    if workspace.tracks().is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "tracks",
            method: "build_tracks",
        });
    }

    let seed = select_initial_pair_for_incremental(workspace, config)?;
    bootstrap_pair_seed(workspace, config, triangulator, seed, ba_backend)
}

/// Two-view bootstrap: place `view_a` at the origin (identity), place `view_b` from the
/// verified pair's relative pose (unit baseline), assign both to component 0, then triangulate
/// all shared tracks visible from the initial pair.
fn bootstrap_pair_seed(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    triangulator: &dyn Triangulator,
    seed: InitialPairHypothesis,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
) -> Result<IncrementalInitializationReport> {
    if seed.views.len() < 2 || seed.pairs.is_empty() {
        return Err(SfmError::NoInitialPairCandidate);
    }
    let view_a = seed.views[0];
    let view_b = seed.views[1];
    let seed_pair = seed.pairs[0];
    let pair_id = workspace
        .pair_graph()
        .pair_id(seed_pair.first(), seed_pair.second())
        .ok_or(SfmError::NoInitialPairCandidate)?;

    let model = workspace
        .geometric_model_store()
        .get(pair_id.index())
        .ok_or(SfmError::MissingPrerequisite {
            prerequisite: "verified geometric model for the initial pair",
            method: "verify_pairs",
        })?;
    let relative_pose = model.relative_pose().ok_or_else(|| {
        SfmError::InvalidInput(format!("initial pair {pair_id} has no relative pose"))
    })?;
    // The relative pose stores projection translation in camera B. Convert it through the
    // named source-frame baseline operation before choosing the unit-baseline initial gauge.
    let rotation_b = *relative_pose.rotation_source_to_target();
    let center_b = Point3::from(relative_pose.baseline_direction_in_source().into_inner());

    let component = Some(ComponentId::new(0));
    workspace.set_pose_for_view(
        view_a,
        WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::origin())
            .map_err(|error| SfmError::InvalidInput(error.to_string()))?,
        component,
    )?;

    workspace.set_pose_for_view(
        view_b,
        WorldToCameraPose::from_rotation_center(rotation_b, center_b)
            .map_err(|error| SfmError::InvalidInput(error.to_string()))?,
        component,
    )?;

    // The opt-in initial-pair adjustment fixes the first pose as the world frame and the
    // unit initial baseline as scale. Persist that explicit gauge before triangulation:
    // component refreshes during growth retain it, so later BA stages cannot drift
    // the monocular scale. The default path keeps a free gauge.
    let gauge_scale_policy = if config.bundle_adjust_initial_pair() {
        GaugeScalePolicy::FixedBaseline {
            first: view_a,
            second: view_b,
            distance: 1.0,
            unit: LengthUnit::Arbitrary,
        }
    } else {
        GaugeScalePolicy::Free
    };
    workspace.components = vec![ReconstructionComponent::new(
        ComponentId::new(0),
        vec![view_a, view_b],
        Vec::new(),
        gauge_scale_policy,
    )];

    // Initial-pair triangulation is permissive (no reprojection cull): the two views
    // define the initial structure; culling happens later during growth. This also
    // refreshes component 0 from the two posed images while preserving the selected
    // gauge policy above.
    let tri = triangulate_incremental(workspace, triangulator, 2, f64::INFINITY)?;
    if tri.new_landmarks == 0 {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "triangulatable initial-pair tracks",
            method: "build_tracks",
        });
    }

    let seed_bundle_adjustment = if config.bundle_adjust_initial_pair() {
        let ba_backend = ba_backend.ok_or_else(|| {
            SfmError::InvariantViolation("initial-pair BA was enabled without a backend".to_owned())
        })?;
        Some(refine_two_view_seed(workspace, view_a, view_b, ba_backend)?)
    } else {
        None
    };
    if let Some(ba_backend) = ba_backend
        && config.self_calibrate_intrinsics()
    {
        calibrate_guessed_seed_focals(workspace, view_a, view_b, ba_backend);
    }

    Ok(IncrementalInitializationReport {
        initialized_views: vec![view_a, view_b],
        initialized_landmarks: tri.new_landmarks,
        triangulation_rejections: tri.rejections,
        seed_bundle_adjustment,
        seed,
    })
}

/// Rounds of focal calibration of a guessed initial pair.
const SEED_CALIBRATION_ROUNDS: usize = 3;
/// Largest relative focal change of a round after which the calibration is considered converged.
const SEED_FOCAL_CONVERGED: f64 = 0.02;
/// A calibrated focal outside this multiple range of its guess is not believed.
const SEED_FOCAL_RANGE: (f64, f64) = (0.25, 4.0);

/// Calibrate the focal of an initial pair whose intrinsic group is an
/// [`IntrinsicPrior::Guess`](crate::model::IntrinsicPrior::Guess).
///
/// A focal guessed from the image size can be twice the truth. The structure and the poses built
/// on it then make every later registration fail, and no later stage corrects it. The two seed
/// views are adjusted with their guessed intrinsics free, up to a few rounds, until the focal
/// stops moving. Trusted groups are left alone, so a pair with no guessed group does nothing.
///
/// The step is an improvement, not a requirement: a failed adjustment, or a focal outside
/// [`SEED_FOCAL_RANGE`] of its guess, leaves the seed as it was.
fn calibrate_guessed_seed_focals(
    workspace: &mut SfmWorkspace,
    view_a: ViewId,
    view_b: ViewId,
    ba_backend: &dyn BundleAdjustmentBackend,
) {
    let mut groups = Vec::new();
    for view in [view_a, view_b] {
        let Some(record) = workspace.view(view) else {
            return;
        };
        let group = record.intrinsic_id;
        if workspace.intrinsic_prior(group) == Some(crate::model::IntrinsicPrior::Guess)
            && !groups.contains(&group)
        {
            groups.push(group);
        }
    }
    if groups.is_empty() {
        return;
    }
    let focals = |workspace: &SfmWorkspace| -> Option<Vec<f64>> {
        groups
            .iter()
            .map(|group| {
                workspace
                    .camera_model(*group)
                    .and_then(|camera| camera.focal_lengths())
                    .map(|(focal_x, _)| focal_x)
            })
            .collect()
    };
    let Some(guessed) = focals(workspace) else {
        return;
    };
    for _ in 0..SEED_CALIBRATION_ROUNDS {
        let Some(before) = focals(workspace) else {
            return;
        };
        let snapshot = super::transaction::BundleAdjustmentRollback::capture(workspace);
        let config = BundleAdjustConfig::default()
            .with_scope(OptimizationScope::SelfCalibration)
            .with_robust_loss(RobustLoss::Squared)
            .with_fixed_views([view_a].into_iter().collect());
        if workspace.bundle_adjust(ba_backend, &config).is_err() {
            return;
        }
        let Some(after) = focals(workspace) else {
            return;
        };
        let believable = after.iter().zip(&guessed).all(|(focal, guess)| {
            focal.is_finite()
                && (SEED_FOCAL_RANGE.0..=SEED_FOCAL_RANGE.1).contains(&(focal / guess))
        });
        if !believable {
            let _ = snapshot.rollback(workspace);
            return;
        }
        let change = after
            .iter()
            .zip(&before)
            .map(|(after, before)| (after - before).abs() / before)
            .fold(0.0_f64, f64::max);
        if change < SEED_FOCAL_CONVERGED {
            return;
        }
    }
}

const MIN_SEED_BA_LANDMARKS: usize = 8;

pub(crate) fn refine_two_view_seed(
    workspace: &mut SfmWorkspace,
    view_a: ViewId,
    view_b: ViewId,
    ba_backend: &dyn BundleAdjustmentBackend,
) -> Result<BundleReport> {
    let active_before = workspace.landmarks().active_count();
    if active_before < MIN_SEED_BA_LANDMARKS {
        return Err(SfmError::InvalidInput(format!(
            "initial-pair BA requires at least {MIN_SEED_BA_LANDMARKS} active landmarks, got {active_before}"
        )));
    }

    let ba_config = BundleAdjustConfig::default()
        .with_scope(OptimizationScope::PosesAndPoints)
        // RANSAC verification and robust triangulation have already removed gross
        // correspondence outliers. Least squares optimizes the initial pair's actual
        // pixel reprojection RMSE instead of trading it away behind robust weights.
        .with_robust_loss(RobustLoss::Squared)
        .with_fixed_views([view_a].into_iter().collect());
    let report = workspace.bundle_adjust(ba_backend, &ba_config)?;
    let rmse_tolerance = 64.0
        * f64::EPSILON
        * report
            .ba
            .initial_rmse
            .abs()
            .max(report.ba.final_rmse.abs())
            .max(1.0);
    if report.ba.final_rmse > report.ba.initial_rmse + rmse_tolerance {
        return Err(SfmError::InvalidInput(format!(
            "initial-pair BA increased reprojection RMSE from {} to {}",
            report.ba.initial_rmse, report.ba.final_rmse
        )));
    }
    if report.rejected_landmarks != 0 {
        return Err(SfmError::InvalidInput(format!(
            "initial-pair BA rejected {} initial-pair landmarks",
            report.rejected_landmarks
        )));
    }
    if workspace.landmarks().active_count() != active_before {
        return Err(SfmError::InvalidInput(
            "initial-pair BA changed the active initial-pair landmark count".to_owned(),
        ));
    }
    validate_refined_seed_geometry(
        workspace,
        view_a,
        view_b,
        &report.ba,
        ba_config.max_iterations(),
    )?;
    workspace.validate()?;
    Ok(report)
}

fn validate_refined_seed_geometry(
    workspace: &SfmWorkspace,
    view_a: ViewId,
    view_b: ViewId,
    report: &crate::ba::backend::BundleAdjustmentReport,
    max_iterations: usize,
) -> Result<()> {
    let pose = |view: ViewId| {
        workspace
            .pose(view)
            .map(|record| &record.pose)
            .ok_or(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: view.index(),
            })
    };
    let baseline = (pose(view_b)?.center() - pose(view_a)?.center()).norm();
    let expected_baseline = workspace
        .components()
        .iter()
        .find_map(|component| match component.gauge_scale_policy {
            GaugeScalePolicy::FixedBaseline {
                first: fixed_a,
                second: fixed_b,
                distance,
                ..
            } if (fixed_a == view_a && fixed_b == view_b)
                || (fixed_a == view_b && fixed_b == view_a) =>
            {
                Some(distance)
            }
            _ => None,
        })
        .unwrap_or(1.0);
    let tolerance = crate::ops::reconstruction::fixed_baseline_roundoff_tolerance(
        baseline,
        expected_baseline,
        report.iteration_count,
        report.diagnostics.accepted_step_count,
        max_iterations,
    );
    if !baseline.is_finite() || (baseline - expected_baseline).abs() > tolerance {
        return Err(SfmError::InvalidInput(format!(
            "initial-pair BA violated the fixed baseline: expected {expected_baseline}, got {baseline}"
        )));
    }

    for landmark in workspace.landmarks().iter_active() {
        let track = workspace
            .tracks()
            .get(landmark.source_track)
            .ok_or(SfmError::InvalidId {
                kind: "landmark source track",
                index: landmark.source_track.index(),
            })?;
        let mut posed_support = 0usize;
        for (_, observation) in landmark.supporting_observations(track) {
            let Some(record) = workspace.pose(observation.view) else {
                continue;
            };
            let camera_point = record.pose.project_to_camera(&landmark.point);
            let view = workspace
                .view(observation.view)
                .ok_or(SfmError::InvalidId {
                    kind: ViewId::kind(),
                    index: observation.view.index(),
                })?;
            let camera = workspace
                .camera_model(view.intrinsic_id)
                .ok_or(SfmError::InvalidId {
                    kind: crate::IntrinsicId::kind(),
                    index: view.intrinsic_id.index(),
                })?;
            let pixel = workspace.track_observation_pixel(observation)?;
            if !seed_observation_has_valid_depth(camera, &camera_point.coords, pixel) {
                return Err(SfmError::InvalidInput(format!(
                    "initial-pair BA left landmark {} outside the projection domain or at non-positive bearing depth in {}",
                    landmark.id, observation.view
                )));
            }
            posed_support += 1;
        }
        if posed_support < 2 {
            return Err(SfmError::InvalidInput(format!(
                "initial-pair landmark {} has fewer than two retained posed observations",
                landmark.id
            )));
        }
    }
    Ok(())
}

/// Test cheirality along the measured camera ray, including supported rear-hemisphere rays.
fn seed_observation_has_valid_depth(
    camera: &crate::Camera,
    point: &Vector3<f64>,
    pixel: Point2<f64>,
) -> bool {
    if !point.iter().all(|value| value.is_finite()) {
        return false;
    }
    let Ok(projected) = camera.project(point) else {
        return false;
    };
    if !projected.coords.iter().all(|value| value.is_finite()) {
        return false;
    }
    let magnitude = point.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if magnitude == 0.0 {
        return false;
    }
    let scaled_point = point / magnitude;
    camera
        .bearing(&pixel)
        .is_ok_and(|bearing| scaled_point.dot(bearing.as_ref()) > 0.0)
}

/// Place the initial-pair views from ground-truth poses and triangulate their shared tracks.
///
/// Testing only: lets tests exercise growth, registration, and BA from a known-good initial pair.
#[cfg(feature = "unstable-test-support")]
pub fn initialize_incremental_seed_from_oracle(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    reconstruction: &GroundTruthReconstruction,
) -> Result<IncrementalInitializationReport> {
    reconstruction.validate()?;
    let transaction = GeometryRollback::capture(workspace);
    match initialize_incremental_seed_from_oracle_inner(workspace, config, reconstruction) {
        Ok(report) => Ok(report),
        Err(err) => {
            transaction.rollback(workspace);
            Err(err)
        }
    }
}

#[cfg(feature = "unstable-test-support")]
fn initialize_incremental_seed_from_oracle_inner(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    reconstruction: &GroundTruthReconstruction,
) -> Result<IncrementalInitializationReport> {
    if workspace.tracks().is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "tracks",
            method: "build_tracks",
        });
    }

    let seed = select_initial_pair_for_incremental(workspace, config)?;
    let component = ComponentId::new(0);
    for view in &seed.views {
        let pose = reconstruction
            .poses
            .iter()
            .find(|(id, _)| id == view)
            .ok_or_else(|| SfmError::InvalidInput(format!("missing oracle pose for {view}")))?
            .1
            .clone();
        workspace.set_pose_for_view(*view, pose, Some(component))?;
    }

    let landmarks = seed_landmarks_from_oracle_tracks(workspace, &seed.views)?;
    let initialized_landmarks = landmarks.len();
    let landmark_ids = landmarks.iter().map(|landmark| landmark.id).collect();
    workspace.landmarks = Landmarks::try_new(landmarks)?;
    let gauge_scale_policy = if config.bundle_adjust_initial_pair() {
        let view_a = *seed.views.first().ok_or(SfmError::NoInitialPairCandidate)?;
        let view_b = *seed.views.get(1).ok_or(SfmError::NoInitialPairCandidate)?;
        let pose_a = workspace.pose(view_a).ok_or(SfmError::InvalidId {
            kind: ViewId::kind(),
            index: view_a.index(),
        })?;
        let pose_b = workspace.pose(view_b).ok_or(SfmError::InvalidId {
            kind: ViewId::kind(),
            index: view_b.index(),
        })?;
        let distance = (pose_b.pose.center() - pose_a.pose.center()).norm();
        if !distance.is_finite() || distance <= 0.0 {
            return Err(SfmError::InvalidInput(
                "oracle initial pair requires a positive finite baseline".to_owned(),
            ));
        }
        GaugeScalePolicy::FixedBaseline {
            first: view_a,
            second: view_b,
            distance,
            unit: LengthUnit::Arbitrary,
        }
    } else {
        GaugeScalePolicy::Free
    };
    workspace.components = vec![ReconstructionComponent::new(
        component,
        seed.views.clone(),
        landmark_ids,
        gauge_scale_policy,
    )];

    Ok(IncrementalInitializationReport {
        initialized_views: seed.views.clone(),
        initialized_landmarks,
        triangulation_rejections: TriangulationRejectionSummary::default(),
        seed_bundle_adjustment: None,
        seed,
    })
}

#[cfg(feature = "unstable-test-support")]
fn seed_landmarks_from_oracle_tracks(
    workspace: &SfmWorkspace,
    seed_views: &[ViewId],
) -> Result<Vec<Landmark>> {
    let mut landmarks = Vec::new();
    let triangulator = crate::adapters::triangulation::TrackTriangulatorAdapter::default();

    for track in workspace.tracks().as_slice() {
        let prepared_observations = track
            .observations()
            .iter()
            .enumerate()
            .filter(|(_, observation)| seed_views.contains(&observation.view))
            .map(|(source_index, observation)| {
                let view = workspace
                    .view(observation.view)
                    .ok_or(SfmError::InvalidId {
                        kind: ViewId::kind(),
                        index: observation.view.index(),
                    })?;
                let pose =
                    workspace
                        .pose(observation.view)
                        .ok_or(SfmError::MissingPrerequisite {
                            prerequisite: "initial-pair camera pose",
                            method: "initialize_incremental_seed_from_oracle",
                        })?;
                let camera =
                    workspace
                        .camera_model(view.intrinsic_id)
                        .ok_or(SfmError::InvalidId {
                            kind: crate::IntrinsicId::kind(),
                            index: view.intrinsic_id.index(),
                        })?;
                Ok((
                    source_index,
                    crate::backends::PosedPixelObservation {
                        view: observation.view,
                        xy: workspace.track_observation_pixel(observation)?,
                        pose: &pose.pose,
                        camera,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let (source_indices, observations): (Vec<_>, Vec<_>) =
            prepared_observations.into_iter().unzip();
        if observations.len() < 2 {
            continue;
        }
        let crate::backends::TriangulationVerdict::Accepted(triangulated) =
            triangulator.triangulate(&observations)?
        else {
            continue;
        };
        landmarks.push(Landmark::active(
            LandmarkId::new(landmarks.len()),
            track.id(),
            triangulated.point,
            triangulated.reprojection_error,
            crate::model::ObservationSupport::try_from_indices(
                source_indices,
                track.observations().len(),
            )?,
        ));
    }
    if landmarks.is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "triangulatable initial-pair tracks",
            method: "build_tracks",
        });
    }
    Ok(landmarks)
}

#[cfg(test)]
mod depth_tests {
    use super::*;
    use openmvg_core::camera::{FisheyeCamera, FisheyeDistortion, PinholeCamera, SphericalCamera};

    #[test]
    fn initial_seed_depth_follows_the_observed_camera_ray() {
        let spherical = crate::Camera::Spherical(SphericalCamera::try_new(640).unwrap());
        let rear = Vector3::new(0.0, 0.0, -5.0);
        let measured = spherical.project(&rear).unwrap();
        assert!(seed_observation_has_valid_depth(
            &spherical, &rear, measured
        ));
        let opposite = spherical.project(&Vector3::z()).unwrap();
        assert!(!seed_observation_has_valid_depth(
            &spherical, &rear, opposite
        ));

        let pinhole = PinholeCamera::try_new(500.0, 500.0, 320.0, 160.0).unwrap();
        let fisheye = crate::Camera::Fisheye(
            FisheyeCamera::try_new(pinhole.clone(), FisheyeDistortion::None).unwrap(),
        );
        let wide = Vector3::new(1.0, 0.0, -0.2);
        let measured = fisheye.project(&wide).unwrap();
        assert!(seed_observation_has_valid_depth(&fisheye, &wide, measured));
        let perspective = crate::Camera::Pinhole(pinhole);
        assert!(!seed_observation_has_valid_depth(
            &perspective,
            &rear,
            Point2::new(320.0, 160.0)
        ));
        let front = Vector3::new(0.0, 0.0, 5.0);
        assert!(seed_observation_has_valid_depth(
            &perspective,
            &front,
            Point2::new(320.0, 160.0)
        ));
    }
}
