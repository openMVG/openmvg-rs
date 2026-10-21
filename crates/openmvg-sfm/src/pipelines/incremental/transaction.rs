// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rollback state for a registration or a bundle adjustment the Incremental engine may reject.

use crate::model::{Landmark, PoseRecord, ReconstructionComponent};
use crate::workspace::store::SparseStore;
use crate::{LandmarkId, Result, SfmError, ViewId};
use nalgebra::Point3;
use openmvg_core::camera::Camera;

#[derive(Debug)]
struct ActiveLandmarkRollback {
    id: LandmarkId,
    point: Point3<f64>,
    reprojection_error: openmvg_core::PixelResidual,
    observation_support: crate::model::ObservationSupport,
}

/// Compact rollback state for a whole-scene bundle-adjustment transaction.
///
/// BA may change every pose, camera model, active landmark geometry/support, and derived component,
/// but it never changes immutable tracks or the landmark-to-track index. Cleanup can reject an
/// active landmark, so the snapshot records its original active state; landmarks already rejected
/// at capture are outside the mutation domain. Any landmarks appended by a subsequent recovery
/// step are removed on rollback.
#[derive(Debug)]
pub(crate) struct BundleAdjustmentRollback {
    poses: SparseStore<PoseRecord>,
    active_landmarks: Vec<ActiveLandmarkRollback>,
    landmark_len: usize,
    components: Vec<ReconstructionComponent>,
    registration_failure_len: usize,
    camera_models: Vec<Camera>,
    payload_bytes: u128,
}

impl BundleAdjustmentRollback {
    pub fn capture(workspace: &crate::SfmWorkspace) -> Self {
        let mut selected_support_bytes = 0usize;
        let active_landmarks = workspace
            .landmarks()
            .iter_active()
            .map(|landmark| {
                if !landmark.observation_support.is_all()
                    && let Some(track) = workspace.tracks().get(landmark.source_track)
                {
                    selected_support_bytes +=
                        landmark.observation_support.len(track.observations().len())
                            * std::mem::size_of::<crate::TrackObservationIndex>();
                }
                ActiveLandmarkRollback {
                    id: landmark.id,
                    point: landmark.point,
                    reprojection_error: landmark.reprojection_error,
                    observation_support: landmark.observation_support.clone(),
                }
            })
            .collect::<Vec<_>>();
        let component_payload_bytes = workspace
            .components()
            .iter()
            .map(|component| {
                component.views.len() * std::mem::size_of::<ViewId>()
                    + component.landmarks.len() * std::mem::size_of::<LandmarkId>()
            })
            .sum::<usize>();
        let payload_bytes = workspace.pose_store().slot_count()
            * std::mem::size_of::<Option<PoseRecord>>()
            + active_landmarks.len() * std::mem::size_of::<ActiveLandmarkRollback>()
            + selected_support_bytes
            + std::mem::size_of_val(workspace.components())
            + component_payload_bytes
            + std::mem::size_of_val(workspace.camera_models());
        Self {
            poses: workspace.pose_store().clone(),
            landmark_len: workspace.landmarks().len(),
            active_landmarks,
            components: workspace.components().to_vec(),
            registration_failure_len: workspace.registration_failures().len(),
            camera_models: workspace.camera_models().to_vec(),
            payload_bytes: payload_bytes as u128,
        }
    }

    pub const fn payload_bytes(&self) -> u128 {
        self.payload_bytes
    }

    pub fn rollback(self, workspace: &mut crate::SfmWorkspace) -> Result<()> {
        workspace.landmarks.truncate(self.landmark_len)?;
        for state in self.active_landmarks {
            let landmark =
                workspace
                    .landmarks
                    .get_mut(state.id)
                    .ok_or(SfmError::InvariantViolation(format!(
                        "BA rollback landmark {} disappeared after capture",
                        state.id
                    )))?;
            landmark.point = state.point;
            landmark.reprojection_error = state.reprojection_error;
            landmark.observation_support = state.observation_support;
            landmark.status = crate::model::LandmarkStatus::Active;
        }
        workspace.poses = self.poses;
        workspace.components = self.components;
        workspace
            .registration_failures
            .truncate(self.registration_failure_len);
        workspace.restore_camera_models(self.camera_models);
        Ok(())
    }
}

/// Rollback state for one camera registration and its scoped track update.
///
/// Registration changes only the proposed pose slot, failure-log entries, landmarks whose
/// immutable tracks observe that image, and a dense suffix of newly triangulated landmarks. Capturing that
/// explicit mutation domain avoids cloning the complete reconstruction for every registered view.
#[derive(Debug)]
pub(crate) struct GrowthCommitTransaction {
    pose_id: crate::PoseId,
    pose: Option<PoseRecord>,
    landmarks: Vec<Landmark>,
    landmark_len: usize,
    registration_failure_len: usize,
}

impl GrowthCommitTransaction {
    pub fn capture(workspace: &crate::SfmWorkspace, image: ViewId) -> Result<Self> {
        debug_assert_eq!(
            workspace.components().len(),
            1,
            "incremental growth requires exactly one primary component"
        );
        let pose_id = workspace.pose_id(image).ok_or(SfmError::InvalidId {
            kind: ViewId::kind(),
            index: image.index(),
        })?;
        let pose = workspace.pose_by_id(pose_id).cloned();

        let landmarks = workspace
            .tracks()
            .observing_view(image)
            .iter()
            .copied()
            .filter_map(|track| workspace.landmarks().for_track(track))
            .map(|landmark| {
                workspace
                    .landmarks()
                    .get(landmark)
                    .cloned()
                    .ok_or(SfmError::InvalidId {
                        kind: LandmarkId::kind(),
                        index: landmark.index(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            pose_id,
            pose,
            landmarks,
            landmark_len: workspace.landmarks().len(),
            registration_failure_len: workspace.registration_failures().len(),
        })
    }

    pub fn rollback(self, workspace: &mut crate::SfmWorkspace) -> Result<()> {
        workspace.landmarks.truncate(self.landmark_len)?;
        for record in self.landmarks {
            let landmark = record.id;
            *workspace
                .landmarks
                .get_mut(landmark)
                .ok_or(SfmError::InvariantViolation(format!(
                    "growth rollback landmark {landmark} disappeared after capture"
                )))? = record;
        }
        match self.pose {
            Some(record) => {
                workspace.poses.set(self.pose_id.index(), record)?;
            }
            None => {
                workspace.poses.clear(self.pose_id.index())?;
            }
        }
        workspace
            .registration_failures
            .truncate(self.registration_failure_len);
        crate::ops::reconstruction::refresh_primary_component(workspace);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point2, Point3, Rotation3};

    use crate::model::{
        ImageMeta, Landmark, LandmarkStatus, Landmarks, ObservationSupport, PoseRecord,
    };
    use crate::pipelines::incremental::{RegistrationFailure, RegistrationFailureReason};
    use crate::workspace::reports::StageReport;
    use crate::{
        Camera, ComponentId, FeatureId, FeatureSet, IntrinsicId, LandmarkId, SfmWorkspace, Track,
        TrackId, TrackObservation, TrackSet, WorldToCameraPose,
    };

    use super::{BundleAdjustmentRollback, GrowthCommitTransaction};
    use crate::ops::rollback::GeometryRollback;

    #[test]
    fn rollback_restores_workspace_reports_with_geometry() {
        let mut workspace = SfmWorkspace::builder()
            .image_meta(vec![ImageMeta::external_id("image", 64, 48)])
            .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        let snapshot = GeometryRollback::capture(&workspace);
        workspace.reports.stages.push(StageReport {
            stage: "mutated".to_owned(),
            input_count: 1,
            attempted_count: 1,
            success_count: 1,
            rejected_count: 0,
            skipped_count: 0,
            seed: None,
            config_digest: String::new(),
            warnings: Vec::new(),
            rejection_reasons: Vec::new(),
        });

        snapshot.rollback(&mut workspace);

        assert!(workspace.reports().stages.is_empty());
    }

    #[test]
    fn growth_rollback_restores_only_the_touched_geometry() {
        let observations = |feature, images: &[usize]| {
            images
                .iter()
                .map(|&image| TrackObservation {
                    view: crate::ViewId::new(image),
                    feature: FeatureId::new(feature),
                })
                .collect()
        };
        let tracks = TrackSet::try_new(vec![
            Track::try_new(TrackId::new(0), observations(0, &[0, 1, 2])).unwrap(),
            Track::try_new(TrackId::new(1), observations(1, &[0, 1])).unwrap(),
        ])
        .unwrap();
        let mut workspace = SfmWorkspace::builder()
            .image_meta(vec![
                ImageMeta::external_id("image-0", 64, 48),
                ImageMeta::external_id("image-1", 64, 48),
                ImageMeta::external_id("image-2", 64, 48),
            ])
            .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .tracks(
                tracks,
                (0..3).map(|image| {
                    FeatureSet::try_new(
                        (0..2)
                            .map(|feature| Point2::new(feature as f64 + image as f64, 0.0))
                            .collect(),
                    )
                    .unwrap()
                }),
            )
            .build()
            .unwrap();
        workspace
            .set_pose(PoseRecord {
                id: crate::PoseId::new(0),
                pose: WorldToCameraPose::from_rotation_center(
                    Rotation3::identity(),
                    Point3::origin(),
                )
                .unwrap(),
                component: Some(ComponentId::new(0)),
            })
            .unwrap();
        workspace.landmarks = Landmarks::try_new(vec![Landmark::active(
            LandmarkId::new(0),
            TrackId::new(0),
            Point3::new(0.0, 0.0, 5.0),
            openmvg_core::PixelResidual::try_new(0.1).unwrap(),
            ObservationSupport::try_from_indices([0, 2], 3).unwrap(),
        )])
        .unwrap();
        crate::ops::reconstruction::refresh_primary_component(&mut workspace);

        let original_landmark = workspace
            .landmarks()
            .get(LandmarkId::new(0))
            .unwrap()
            .clone();
        let original_components = workspace.components().to_vec();
        let transaction =
            GrowthCommitTransaction::capture(&workspace, crate::ViewId::new(1)).unwrap();
        assert_eq!(transaction.pose_id, crate::PoseId::new(1));
        assert_eq!(transaction.landmarks.len(), 1);

        workspace
            .set_pose(PoseRecord {
                id: crate::PoseId::new(1),
                pose: WorldToCameraPose::from_rotation_center(
                    Rotation3::identity(),
                    Point3::new(1.0, 0.0, 0.0),
                )
                .unwrap(),
                component: Some(ComponentId::new(0)),
            })
            .unwrap();
        let changed = workspace.landmarks.get_mut(LandmarkId::new(0)).unwrap();
        changed.point = Point3::new(9.0, 9.0, 9.0);
        changed.reprojection_error = openmvg_core::PixelResidual::try_new(9.0).unwrap();
        changed.observation_support = ObservationSupport::default();
        changed.status = LandmarkStatus::Rejected;
        workspace
            .landmarks
            .push(Landmark::active(
                LandmarkId::new(1),
                TrackId::new(1),
                Point3::new(1.0, 0.0, 5.0),
                openmvg_core::PixelResidual::try_new(0.2).unwrap(),
                ObservationSupport::default(),
            ))
            .unwrap();
        workspace.record_registration_failure(RegistrationFailure {
            view: crate::ViewId::new(1),
            reason: RegistrationFailureReason::PoseEstimationFailed,
        });
        crate::ops::reconstruction::refresh_primary_component(&mut workspace);

        transaction.rollback(&mut workspace).unwrap();

        assert!(!workspace.pose_store().has(1));
        assert_eq!(workspace.landmarks().len(), 1);
        assert_eq!(
            workspace.landmarks().get(LandmarkId::new(0)),
            Some(&original_landmark)
        );
        assert_eq!(workspace.landmarks().for_track(TrackId::new(1)), None);
        assert!(workspace.registration_failures().is_empty());
        assert_eq!(workspace.components(), original_components);

        let original_pose = PoseRecord {
            id: crate::PoseId::new(1),
            pose: WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(1.0, 2.0, 3.0),
            )
            .unwrap(),
            component: Some(ComponentId::new(0)),
        };
        workspace.set_pose(original_pose.clone()).unwrap();
        crate::ops::reconstruction::refresh_primary_component(&mut workspace);
        let transaction =
            GrowthCommitTransaction::capture(&workspace, crate::ViewId::new(1)).unwrap();
        workspace
            .set_pose(PoseRecord {
                id: crate::PoseId::new(1),
                pose: WorldToCameraPose::from_rotation_center(
                    Rotation3::identity(),
                    Point3::new(8.0, 8.0, 8.0),
                )
                .unwrap(),
                component: Some(ComponentId::new(0)),
            })
            .unwrap();

        transaction.rollback(&mut workspace).unwrap();

        assert_eq!(workspace.pose_store().get(1), Some(&original_pose));
    }

    #[test]
    fn bundle_adjustment_rollback_restores_mutable_geometry_without_cloning_track_index() {
        let tracks = TrackSet::try_new(vec![
            Track::try_new(
                TrackId::new(0),
                vec![
                    TrackObservation {
                        view: crate::ViewId::new(0),
                        feature: FeatureId::new(0),
                    },
                    TrackObservation {
                        view: crate::ViewId::new(1),
                        feature: FeatureId::new(0),
                    },
                ],
            )
            .unwrap(),
            Track::try_new(
                TrackId::new(1),
                vec![
                    TrackObservation {
                        view: crate::ViewId::new(0),
                        feature: FeatureId::new(1),
                    },
                    TrackObservation {
                        view: crate::ViewId::new(1),
                        feature: FeatureId::new(1),
                    },
                ],
            )
            .unwrap(),
            Track::try_new(
                TrackId::new(2),
                vec![
                    TrackObservation {
                        view: crate::ViewId::new(0),
                        feature: FeatureId::new(2),
                    },
                    TrackObservation {
                        view: crate::ViewId::new(1),
                        feature: FeatureId::new(2),
                    },
                ],
            )
            .unwrap(),
        ])
        .unwrap();
        let mut workspace = SfmWorkspace::builder()
            .image_meta(vec![
                ImageMeta::external_id("image-0", 64, 48),
                ImageMeta::external_id("image-1", 64, 48),
            ])
            .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .tracks(
                tracks,
                [
                    FeatureSet::try_new(vec![
                        Point2::new(10.0, 10.0),
                        Point2::new(20.0, 20.0),
                        Point2::new(30.0, 30.0),
                    ])
                    .unwrap(),
                    FeatureSet::try_new(vec![
                        Point2::new(11.0, 10.0),
                        Point2::new(21.0, 20.0),
                        Point2::new(31.0, 30.0),
                    ])
                    .unwrap(),
                ],
            )
            .build()
            .unwrap();
        workspace
            .set_pose(PoseRecord {
                id: crate::PoseId::new(0),
                pose: WorldToCameraPose::identity(),
                component: Some(ComponentId::new(0)),
            })
            .unwrap();
        let mut previously_rejected = Landmark::active(
            LandmarkId::new(1),
            TrackId::new(1),
            Point3::new(1.0, 0.0, 5.0),
            openmvg_core::PixelResidual::try_new(0.2).unwrap(),
            ObservationSupport::default(),
        );
        previously_rejected.status = LandmarkStatus::Rejected;
        workspace.landmarks = Landmarks::try_new(vec![
            Landmark::active(
                LandmarkId::new(0),
                TrackId::new(0),
                Point3::new(0.0, 0.0, 5.0),
                openmvg_core::PixelResidual::try_new(0.1).unwrap(),
                ObservationSupport::default(),
            ),
            previously_rejected,
        ])
        .unwrap();
        crate::ops::reconstruction::refresh_primary_component(&mut workspace);
        let original_pose = workspace.pose_store().clone();
        let original_landmark = workspace
            .landmarks()
            .get(LandmarkId::new(0))
            .unwrap()
            .clone();
        let original_rejected = workspace
            .landmarks()
            .get(LandmarkId::new(1))
            .unwrap()
            .clone();
        let original_components = workspace.components().to_vec();
        let rollback = BundleAdjustmentRollback::capture(&workspace);
        assert!(rollback.payload_bytes() > 0);
        assert_eq!(rollback.landmark_len, 2);
        assert_eq!(rollback.active_landmarks.len(), 1);
        assert_eq!(rollback.active_landmarks[0].id, LandmarkId::new(0));

        workspace.poses.clear(0).unwrap();
        let landmark = workspace.landmarks.get_mut(LandmarkId::new(0)).unwrap();
        landmark.point = Point3::new(9.0, 9.0, 9.0);
        landmark.status = LandmarkStatus::Rejected;
        workspace
            .landmarks
            .push(Landmark::active(
                LandmarkId::new(2),
                TrackId::new(2),
                Point3::new(2.0, 0.0, 5.0),
                openmvg_core::PixelResidual::try_new(0.3).unwrap(),
                ObservationSupport::default(),
            ))
            .unwrap();
        workspace.record_registration_failure(RegistrationFailure {
            view: crate::ViewId::new(1),
            reason: RegistrationFailureReason::PoseEstimationFailed,
        });
        crate::ops::reconstruction::refresh_primary_component(&mut workspace);

        rollback.rollback(&mut workspace).unwrap();

        assert_eq!(workspace.pose_store(), &original_pose);
        assert_eq!(workspace.landmarks().len(), 2);
        assert_eq!(
            workspace.landmarks().get(LandmarkId::new(0)),
            Some(&original_landmark)
        );
        assert_eq!(
            workspace.landmarks().get(LandmarkId::new(1)),
            Some(&original_rejected)
        );
        assert_eq!(workspace.landmarks().for_track(TrackId::new(2)), None);
        assert!(workspace.registration_failures().is_empty());
        assert_eq!(workspace.components(), original_components);
    }
}
