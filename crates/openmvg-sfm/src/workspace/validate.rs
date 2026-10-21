// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whole-workspace consistency checks across views, matches, models, poses, landmarks, and
//! components. Retained `f64` pixels must round to the corresponding `f32` keypoint
//! coordinates when both representations are present.

use std::collections::BTreeSet;

use crate::two_view::GeometricModel;
use crate::{ComponentId, LandmarkId, PairId, Result, SfmError, SfmWorkspace, TrackId, ViewId};

pub fn validate_workspace(workspace: &SfmWorkspace) -> Result<()> {
    let view_count = workspace.views().len();
    let mut names = BTreeSet::new();
    let mut assigned_poses = BTreeSet::new();

    for (index, image) in workspace.views().iter().enumerate() {
        if image.id.index() != index {
            return Err(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: image.id.index(),
            });
        }
        if image.meta.name.is_empty() {
            return Err(SfmError::InvalidInput(format!(
                "image {} has an empty name",
                image.id
            )));
        }
        if image.meta.width == 0 || image.meta.height == 0 {
            return Err(SfmError::InvalidInput(format!(
                "image {} has invalid dimensions {}x{}",
                image.id, image.meta.width, image.meta.height
            )));
        }
        if !names.insert(image.meta.name.as_str()) {
            return Err(SfmError::InvalidInput(format!(
                "duplicate image name `{}`",
                image.meta.name
            )));
        }
        if image.intrinsic_id.index() >= workspace.camera_models().len() {
            return Err(SfmError::InvalidId {
                kind: crate::IntrinsicId::kind(),
                index: image.intrinsic_id.index(),
            });
        }
        assigned_poses.insert(image.pose_id);
    }
    crate::workspace::validate_camera_models_for_views(
        workspace.views(),
        workspace.camera_models(),
    )?;

    if workspace.pair_graph().view_count() != view_count {
        return Err(SfmError::InvalidInput(format!(
            "pair graph has {} images but workspace has {}",
            workspace.pair_graph().view_count(),
            view_count
        )));
    }
    workspace.pair_graph().validate()?;
    if workspace.feature_store().slot_count() != view_count {
        return Err(SfmError::InvalidInput(format!(
            "feature store has {} slots but workspace has {} images",
            workspace.feature_store().slot_count(),
            view_count
        )));
    }
    if workspace.keypoint_store().slot_count() != view_count {
        return Err(SfmError::InvalidInput(format!(
            "keypoint store has {} slots but workspace has {view_count} views",
            workspace.keypoint_store().slot_count()
        )));
    }
    if workspace.feature_set_store().slot_count() != view_count {
        return Err(SfmError::InvalidInput(format!(
            "feature-set store has {} slots but workspace has {view_count} views",
            workspace.feature_set_store().slot_count()
        )));
    }
    for (index, feature_set) in workspace.feature_set_store().iter_populated() {
        feature_set.validate().map_err(|error| {
            SfmError::InvalidInput(format!("feature set for view {index} is invalid: {error}"))
        })?;
    }
    for (index, features) in workspace.feature_store().iter_populated() {
        features.validate().map_err(|error| {
            SfmError::InvalidInput(format!(
                "features at image slot {index} are invalid: {error}"
            ))
        })?;
        if workspace.keypoint_store().get(index).map(Vec::as_slice) != Some(features.keypoints()) {
            return Err(SfmError::InvalidInput(format!(
                "features at image slot {index} disagree with the stored keypoints"
            )));
        }
    }
    for (index, keypoints) in workspace.keypoint_store().iter_populated() {
        let feature_set = workspace.feature_set_store().get(index).ok_or_else(|| {
            SfmError::InvalidInput(format!(
                "keypoints at image slot {index} have no retained feature geometry"
            ))
        })?;
        if feature_set.len() != keypoints.len() {
            return Err(SfmError::InvalidInput(format!(
                "keypoints at image slot {index} disagree with retained feature geometry count"
            )));
        }
        if feature_set
            .pixels()
            .iter()
            .zip(keypoints)
            .any(|(pixel, keypoint)| {
                pixel.x as f32 != keypoint.x() || pixel.y as f32 != keypoint.y()
            })
        {
            return Err(SfmError::InvalidInput(format!(
                "keypoints at image slot {index} disagree with retained feature geometry after f32 rounding"
            )));
        }
    }

    let pair_count = workspace.pair_graph().len();
    if workspace.match_store().slot_count() != pair_count {
        return Err(SfmError::InvalidInput(format!(
            "match store has {} slots but pair graph has {} pairs",
            workspace.match_store().slot_count(),
            pair_count
        )));
    }
    if workspace.geometric_model_store().slot_count() != pair_count {
        return Err(SfmError::InvalidInput(format!(
            "geometric model store has {} slots but pair graph has {} pairs",
            workspace.geometric_model_store().slot_count(),
            pair_count
        )));
    }
    if workspace.is_matching_data_released() {
        if workspace.tracks().is_empty() {
            return Err(SfmError::InvalidInput(
                "matching data cannot be released before tracks with retained feature geometry exist"
                    .to_owned(),
            ));
        }
        if workspace.feature_store().populated_count() != 0
            || workspace.keypoint_store().populated_count() != 0
            || workspace.match_store().populated_count() != 0
        {
            return Err(SfmError::InvalidInput(
                "matching data is marked released but feature, keypoint, or match records remain"
                    .to_owned(),
            ));
        }
    }
    let pose_count = assigned_poses.len();
    if (0..pose_count).any(|index| !assigned_poses.contains(&crate::PoseId::new(index))) {
        return Err(SfmError::InvalidInput(
            "view pose assignments must form a dense pose-ID domain".to_owned(),
        ));
    }
    if workspace.pose_store().slot_count() != pose_count {
        return Err(SfmError::InvalidInput(format!(
            "pose store has {} slots but workspace has {pose_count} pose groups",
            workspace.pose_store().slot_count(),
        )));
    }

    for (index, matches) in workspace.match_store().iter_populated() {
        let pair_id = PairId::new(index);
        let pair = workspace
            .pair_graph()
            .pair(pair_id)
            .ok_or(SfmError::InvalidId {
                kind: PairId::kind(),
                index,
            })?;
        if matches.pair() != pair {
            return Err(SfmError::InvalidInput(format!(
                "match record at slot {index} is labeled {}",
                matches.pair()
            )));
        }
        let keypoints_a = workspace.keypoint_store().get(pair.first().index()).ok_or(
            SfmError::MissingPrerequisite {
                prerequisite: "keypoints for stored matches",
                method: "extract_features",
            },
        )?;
        let keypoints_b = workspace
            .keypoint_store()
            .get(pair.second().index())
            .ok_or(SfmError::MissingPrerequisite {
                prerequisite: "keypoints for stored matches",
                method: "extract_features",
            })?;
        let mut previous = None;
        for matched in matches.as_slice() {
            let key = (matched.first(), matched.second());
            if previous.is_some_and(|previous| previous >= key) {
                return Err(SfmError::InvalidInput(format!(
                    "matches at slot {index} are not strictly sorted and unique"
                )));
            }
            if matched
                .distance()
                .is_some_and(|distance| !distance.is_finite() || distance < 0.0)
            {
                return Err(SfmError::InvalidInput(format!(
                    "match at slot {index} has an invalid distance"
                )));
            }
            if matched.first().index() >= keypoints_a.len()
                || matched.second().index() >= keypoints_b.len()
            {
                return Err(SfmError::InvalidInput(format!(
                    "match at slot {index} references an unknown feature"
                )));
            }
            previous = Some(key);
        }
    }

    for (index, model) in workspace.geometric_model_store().iter_populated() {
        model.evidence().validate().map_err(|error| {
            SfmError::InvalidInput(format!(
                "geometric model at slot {index} has invalid evidence: {error}"
            ))
        })?;
        if !model_geometry_is_finite(model) {
            return Err(SfmError::InvalidInput(format!(
                "geometric model at slot {index} is not finite"
            )));
        }
        if let Some(matches) = workspace.match_store().get(index) {
            model.validate_for_matches(matches)?;
        }
    }

    // Validate pose identity and finite rigid geometry.
    for (index, pose_record) in workspace.pose_store().iter_populated() {
        if pose_record.id.index() != index {
            return Err(SfmError::InvalidInput(format!(
                "pose at slot {index} is labeled {}",
                pose_record.id
            )));
        }
        let r = pose_record.pose.rotation().matrix();
        let rtr = r.transpose() * r;
        let identity = nalgebra::Matrix3::<f64>::identity();
        let deviation = (rtr - identity).norm();
        if !r.iter().all(|value| value.is_finite())
            || !pose_record
                .pose
                .center()
                .coords
                .iter()
                .all(|value| value.is_finite())
            || !deviation.is_finite()
            || deviation > 1e-6
            || (r.determinant() - 1.0).abs() > 1e-6
        {
            return Err(SfmError::InvalidInput(format!(
                "pose at slot {index} contains invalid rigid geometry"
            )));
        }
    }

    validate_global_estimates(workspace, view_count)?;

    // Validate every relative rotation, including rotation-only evidence.
    for (index, model) in workspace.geometric_model_store().iter_populated() {
        if let Some(rotation) = model.relative_rotation() {
            let matrix = rotation.matrix();
            let rotation_error =
                (matrix.transpose() * matrix - nalgebra::Matrix3::identity()).norm();
            if !matrix.iter().all(|value| value.is_finite())
                || !rotation_error.is_finite()
                || rotation_error > 1e-6
                || (matrix.determinant() - 1.0).abs() > 1e-6
            {
                return Err(SfmError::InvalidInput(format!(
                    "geometric model at slot {index} has an invalid relative rotation"
                )));
            }
        }
        if let Some(relative_pose) = model.relative_pose() {
            let t_norm = relative_pose.translation_source_to_target().norm();
            if !t_norm.is_finite() || (t_norm - 1.0).abs() > 1e-6 {
                return Err(SfmError::InvalidInput(format!(
                    "geometric model at slot {index} has an invalid relative pose"
                )));
            }
        }
    }

    for (index, track) in workspace.tracks().as_slice().iter().enumerate() {
        if track.id().index() != index {
            return Err(SfmError::InvalidId {
                kind: TrackId::kind(),
                index: track.id().index(),
            });
        }
        for observation in track.observations() {
            if observation.view.index() >= view_count
                || workspace.track_observation_pixel(observation).is_err()
            {
                return Err(SfmError::InvalidInput(format!(
                    "track {} contains an invalid observation",
                    track.id()
                )));
            }
        }
        if track.observations().len() < 2 || track.observations().len() > u32::MAX as usize {
            return Err(SfmError::InvalidInput(format!(
                "track {} has invalid observation count {}",
                track.id(),
                track.observations().len()
            )));
        }
        if track.observations().windows(2).any(|window| {
            (window[0].view, window[0].feature) >= (window[1].view, window[1].feature)
                || window[0].view == window[1].view
        }) {
            return Err(SfmError::InvalidInput(format!(
                "track {} observations are not canonical",
                track.id()
            )));
        }
    }

    for (index, landmark) in workspace.landmarks().as_slice().iter().enumerate() {
        if landmark.id.index() != index {
            return Err(SfmError::InvalidId {
                kind: LandmarkId::kind(),
                index: landmark.id.index(),
            });
        }
        let track = workspace
            .tracks()
            .get(landmark.source_track)
            .ok_or(SfmError::InvalidId {
                kind: TrackId::kind(),
                index: landmark.source_track.index(),
            })?;
        landmark
            .observation_support
            .validate(track.observations().len())?;
        if !landmark.point.coords.iter().all(|value| value.is_finite()) {
            return Err(SfmError::InvalidInput(format!(
                "landmark {} contains invalid geometry or reprojection error",
                landmark.id
            )));
        }
        if landmark.status == crate::model::LandmarkStatus::Active
            && let Some((_, observation)) = landmark
                .supporting_observations(track)
                .find(|(_, observation)| !workspace.has_pose(observation.view))
        {
            return Err(SfmError::InvalidInput(format!(
                "active landmark {} is supported by unregistered view {}",
                landmark.id, observation.view
            )));
        }
    }

    validate_components(workspace)?;

    for failure in workspace.registration_failures() {
        if failure.view.index() >= view_count {
            return Err(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: failure.view.index(),
            });
        }
    }

    Ok(())
}

fn validate_global_estimates(workspace: &SfmWorkspace, view_count: usize) -> Result<()> {
    let (rotations, centers) = match (workspace.rotation_estimates(), workspace.camera_centers()) {
        (None, None) => return Ok(()),
        (Some(rotations), Some(centers)) => (rotations, centers),
        _ => {
            return Err(SfmError::InvalidInput(
                "global rotation and camera-center estimates must be stored together".to_owned(),
            ));
        }
    };
    let mut rotation_ids = BTreeSet::new();
    for (image, rotation) in rotations.rotations() {
        let matrix = rotation.matrix();
        let error = (matrix.transpose() * matrix - nalgebra::Matrix3::identity()).norm();
        if image.index() >= view_count
            || !rotation_ids.insert(*image)
            || !matrix.iter().all(|value| value.is_finite())
            || !error.is_finite()
            || error > 1e-6
            || (matrix.determinant() - 1.0).abs() > 1e-6
        {
            return Err(SfmError::InvalidInput(
                "global rotation estimates contain invalid entries".to_owned(),
            ));
        }
    }

    let mut center_ids = BTreeSet::new();
    for (image, center) in centers.centers() {
        if image.index() >= view_count
            || !center_ids.insert(*image)
            || !rotation_ids.contains(image)
            || !center.iter().all(|value| value.is_finite())
        {
            return Err(SfmError::InvalidInput(
                "global camera-center estimates contain invalid entries".to_owned(),
            ));
        }
    }
    Ok(())
}

fn model_geometry_is_finite(model: &GeometricModel) -> bool {
    match model {
        GeometricModel::Essential { matrix, .. }
        | GeometricModel::Fundamental { matrix, .. }
        | GeometricModel::Homography { matrix, .. } => matrix.iter().all(|value| value.is_finite()),
        GeometricModel::RotationOnly { rotation, .. } => {
            rotation.matrix().iter().all(|value| value.is_finite())
        }
    }
}

fn validate_components(workspace: &SfmWorkspace) -> Result<()> {
    let mut component_ids = BTreeSet::new();
    let mut assigned_views = BTreeSet::new();
    let mut assigned_landmarks = BTreeSet::new();

    for component in workspace.components() {
        if !component_ids.insert(component.id) {
            return Err(SfmError::InvalidInput(format!(
                "duplicate reconstruction component {}",
                component.id
            )));
        }
        if component.views.windows(2).any(|pair| pair[0] >= pair[1])
            || component
                .landmarks
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(SfmError::InvalidInput(format!(
                "reconstruction component {} members are not strictly sorted and unique",
                component.id
            )));
        }
        for &image in &component.views {
            let pose = workspace.pose(image).ok_or(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: image.index(),
            })?;
            if pose.component != Some(component.id) || !assigned_views.insert(image) {
                return Err(SfmError::InvalidInput(format!(
                    "view {image} has inconsistent reconstruction-component membership"
                )));
            }
        }
        for &landmark in &component.landmarks {
            let landmark_record =
                workspace
                    .landmarks()
                    .get(landmark)
                    .ok_or(SfmError::InvalidId {
                        kind: LandmarkId::kind(),
                        index: landmark.index(),
                    })?;
            if landmark_record.status != crate::model::LandmarkStatus::Active {
                return Err(SfmError::InvalidInput(format!(
                    "rejected landmark {landmark} remains in reconstruction component {}",
                    component.id
                )));
            }
            if !assigned_landmarks.insert(landmark) {
                return Err(SfmError::InvalidInput(format!(
                    "landmark {landmark} belongs to multiple reconstruction components"
                )));
            }
        }
    }

    for (_, pose) in workspace.pose_store().iter_populated() {
        if let Some(component) = pose.component {
            let assignments_are_consistent = workspace
                .views()
                .iter()
                .filter(|view| view.pose_id == pose.id)
                .all(|view| assigned_views.contains(&view.id));
            if !component_ids.contains(&component) || !assignments_are_consistent {
                return Err(SfmError::InvalidId {
                    kind: ComponentId::kind(),
                    index: component.index(),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImageMeta, PoseRecord};
    use crate::{Camera, IntrinsicId};
    use nalgebra::{Point3, Rotation3};
    use openmvg_core::model::WorldToCameraPose;
    use openmvg_features::model::{DescriptorSet, ExtractedFeatures, Keypoint};

    #[test]
    fn pose_slot_must_match_embedded_pose_id() {
        let mut workspace = SfmWorkspace::builder()
            .image_meta([
                ImageMeta::external_id("a", 64, 48),
                ImageMeta::external_id("b", 64, 48),
            ])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        workspace
            .poses
            .set(
                0,
                PoseRecord {
                    id: crate::PoseId::new(1),
                    pose: WorldToCameraPose::from_rotation_center(
                        Rotation3::identity(),
                        Point3::origin(),
                    )
                    .unwrap(),
                    component: None,
                },
            )
            .unwrap();

        assert!(workspace.validate().is_err());
    }

    fn featured_workspace() -> SfmWorkspace {
        let mut workspace = SfmWorkspace::builder()
            .image_meta([ImageMeta::external_id("a", 64, 48)])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        let features = ExtractedFeatures::try_new(
            vec![Keypoint::try_new(3.0, 4.0, 1.0, 0.0).unwrap()],
            DescriptorSet::try_u8(vec![1], 1).unwrap(),
        )
        .unwrap();
        workspace
            .feature_sets
            .set(0, features.feature_set())
            .unwrap();
        workspace.set_features(0, features).unwrap();
        workspace.validate().unwrap();
        workspace
    }

    #[test]
    fn extracted_features_must_agree_with_stored_keypoints() {
        let mut workspace = featured_workspace();
        workspace
            .keypoints
            .set(0, vec![Keypoint::try_new(3.0, 5.0, 1.0, 0.0).unwrap()])
            .unwrap();

        assert!(workspace.validate().is_err());
    }

    #[test]
    fn retained_pixels_must_round_to_the_stored_keypoint() {
        let mut workspace = featured_workspace();
        workspace.release_descriptors().unwrap();
        let precise = nalgebra::Point2::new(3.0 + 1e-8, 4.0 - 1e-8);
        workspace
            .feature_sets
            .set(0, crate::FeatureSet::try_new(vec![precise]).unwrap())
            .unwrap();
        workspace.validate().unwrap();
        assert_eq!(workspace.feature_sets.get(0).unwrap().pixels(), &[precise]);

        workspace
            .feature_sets
            .set(
                0,
                crate::FeatureSet::try_new(vec![nalgebra::Point2::new(3.125, 4.0)]).unwrap(),
            )
            .unwrap();
        let error = workspace.validate().unwrap_err();
        assert!(error.to_string().contains("after f32 rounding"), "{error}");
    }

    #[test]
    fn keypoints_require_retained_feature_geometry() {
        let mut workspace = featured_workspace();
        workspace.release_descriptors().unwrap();
        workspace.validate().unwrap();
        workspace.feature_sets.clear(0).unwrap();

        assert!(workspace.validate().is_err());
    }

    #[test]
    fn active_landmark_support_must_be_registered() {
        let observations = (0..2)
            .map(|view| crate::TrackObservation::new(ViewId::new(view), crate::FeatureId::new(0)))
            .collect();
        let tracks = crate::TrackSet::try_new(vec![
            crate::Track::try_new(TrackId::new(0), observations).unwrap(),
        ])
        .unwrap();
        let feature_sets = (0..2)
            .map(|view| {
                crate::FeatureSet::try_new(vec![nalgebra::Point2::new(view as f64, 0.0)]).unwrap()
            })
            .collect::<Vec<_>>();
        let mut workspace = SfmWorkspace::builder()
            .image_meta([
                ImageMeta::external_id("a", 64, 48),
                ImageMeta::external_id("b", 64, 48),
            ])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .tracks(tracks, feature_sets)
            .build()
            .unwrap();
        let register = |workspace: &mut SfmWorkspace, view: usize| {
            let pose_id = workspace.view(ViewId::new(view)).unwrap().pose_id;
            let pose = WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(view as f64, 0.0, 0.0),
            )
            .unwrap();
            workspace
                .poses
                .set(pose_id.index(), PoseRecord::new(pose_id, pose, None))
                .unwrap();
        };
        register(&mut workspace, 0);
        workspace
            .landmarks
            .push(crate::model::Landmark::active(
                LandmarkId::new(0),
                TrackId::new(0),
                Point3::new(0.5, 0.0, 4.0),
                openmvg_core::PixelResidual::ZERO,
                crate::model::ObservationSupport::default(),
            ))
            .unwrap();

        let error = workspace.validate().unwrap_err().to_string();
        assert!(error.contains("unregistered view"), "{error}");
        register(&mut workspace, 1);
        workspace.validate().unwrap();
    }

    #[test]
    fn negative_landmark_error_is_unrepresentable() {
        assert!(openmvg_core::PixelResidual::try_new(-1.0).is_err());
    }
}
