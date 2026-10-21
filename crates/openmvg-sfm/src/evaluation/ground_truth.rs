// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validated ground-truth scenes and reconstructions in the form compared by the evaluation
//! metrics.

use std::collections::BTreeSet;

use nalgebra::Point3;
use openmvg_core::camera::Camera;
use openmvg_core::model::WorldToCameraPose;
use serde::{Deserialize, Serialize};

use crate::model::ImageMeta;
use crate::{IntrinsicId, LandmarkId, Result, SfmError, ViewId, ViewPair};

/// Reference scene for evaluation: image metadata, cameras, poses, points, and the view pairs
/// scored by the relative metrics.
///
/// Every list is sorted by id and checked on construction and on deserialization.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroundTruthScene {
    /// Image metadata by view; names are unique and dimensions non-zero.
    pub image_meta: Vec<(ViewId, ImageMeta)>,
    /// Camera models by intrinsic id.
    pub camera_models: Vec<(IntrinsicId, Camera)>,
    /// World-to-camera poses by view, with rotations orthonormal to `1e-6`.
    pub poses: Vec<(ViewId, WorldToCameraPose)>,
    /// Finite world points by landmark id.
    pub points: Vec<(LandmarkId, Point3<f64>)>,
    /// Distinct pairs of posed views, sorted, used for relative rotation/translation errors.
    pub pairs: Vec<ViewPair>,
}

impl GroundTruthScene {
    /// Sorts every list by id (and the pairs), then validates the scene.
    ///
    /// # Errors
    ///
    /// Returns [`SfmError::InvalidInput`] for a pair with equal endpoints or any failure of
    /// [`Self::validate`].
    pub fn try_new(
        mut image_meta: Vec<(ViewId, ImageMeta)>,
        mut camera_models: Vec<(IntrinsicId, Camera)>,
        mut poses: Vec<(ViewId, WorldToCameraPose)>,
        mut points: Vec<(LandmarkId, Point3<f64>)>,
        pairs: Vec<ViewPair>,
    ) -> Result<Self> {
        image_meta.sort_by_key(|(id, _)| *id);
        camera_models.sort_by_key(|(id, _)| *id);
        poses.sort_by_key(|(id, _)| *id);
        points.sort_by_key(|(id, _)| *id);
        let mut pairs = pairs
            .into_iter()
            .map(|pair| {
                if pair.first() == pair.second() {
                    Err(SfmError::InvalidInput(format!(
                        "pair endpoints must differ: {}",
                        pair.first()
                    )))
                } else {
                    Ok(pair)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        pairs.sort_unstable();

        let scene = Self {
            image_meta,
            camera_models,
            poses,
            points,
            pairs,
        };
        scene.validate()?;
        Ok(scene)
    }

    /// Validate identifiers, numerical values, and pair references.
    pub fn validate(&self) -> Result<()> {
        unique_indices(
            self.image_meta.iter().map(|(id, _)| id.index()),
            ViewId::kind(),
        )?;
        unique_indices(
            self.camera_models.iter().map(|(id, _)| id.index()),
            IntrinsicId::kind(),
        )?;
        unique_indices(
            self.poses.iter().map(|(id, _)| id.index()),
            "ground-truth pose",
        )?;
        unique_indices(
            self.points.iter().map(|(id, _)| id.index()),
            "ground-truth point",
        )?;

        let mut image_names = BTreeSet::new();
        for (id, meta) in &self.image_meta {
            if meta.name.is_empty() || meta.width == 0 || meta.height == 0 {
                return Err(SfmError::InvalidInput(format!(
                    "ground-truth metadata for {id} must have a name and non-zero dimensions"
                )));
            }
            if !image_names.insert(meta.name.as_str()) {
                return Err(SfmError::InvalidInput(format!(
                    "duplicate ground-truth image name `{}`",
                    meta.name
                )));
            }
        }
        validate_poses(&self.poses)?;
        validate_points(&self.points)?;

        let pose_ids: BTreeSet<_> = self.poses.iter().map(|(id, _)| *id).collect();
        let mut unique_pairs = BTreeSet::new();
        for &pair in &self.pairs {
            if pair.first() == pair.second() {
                return Err(SfmError::InvalidInput(format!(
                    "pair endpoints must differ: {}",
                    pair.first()
                )));
            }
            if !pose_ids.contains(&pair.first()) || !pose_ids.contains(&pair.second()) {
                return Err(SfmError::InvalidInput(format!(
                    "ground-truth pair {pair} references a view without a pose"
                )));
            }
            if !unique_pairs.insert(pair) {
                return Err(SfmError::InvalidInput(format!(
                    "duplicate ground-truth pair ({}, {})",
                    pair.first(),
                    pair.second()
                )));
            }
        }
        Ok(())
    }

    /// Copies the poses and points into a [`GroundTruthReconstruction`].
    pub fn reconstruction(&self) -> GroundTruthReconstruction {
        GroundTruthReconstruction {
            poses: self.poses.clone(),
            points: self.points.clone(),
        }
    }
}

impl<'de> Deserialize<'de> for GroundTruthScene {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SerializedScene {
            image_meta: Vec<(ViewId, ImageMeta)>,
            camera_models: Vec<(IntrinsicId, Camera)>,
            poses: Vec<(ViewId, WorldToCameraPose)>,
            points: Vec<(LandmarkId, Point3<f64>)>,
            pairs: Vec<ViewPair>,
        }

        let scene = SerializedScene::deserialize(deserializer)?;
        Self::try_new(
            scene.image_meta,
            scene.camera_models,
            scene.poses,
            scene.points,
            scene.pairs,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Poses and points of a reconstruction, in the form compared against a [`GroundTruthScene`].
///
/// Both lists are sorted by id and checked on construction and on deserialization.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroundTruthReconstruction {
    /// World-to-camera poses by view, with rotations orthonormal to `1e-6`.
    pub poses: Vec<(ViewId, WorldToCameraPose)>,
    /// Finite world points by landmark id.
    pub points: Vec<(LandmarkId, Point3<f64>)>,
}

impl GroundTruthReconstruction {
    /// Sorts both lists by id, then validates them.
    ///
    /// # Errors
    ///
    /// Returns [`SfmError::InvalidInput`] for duplicate ids, an invalid rotation, or non-finite
    /// geometry.
    pub fn try_new(
        mut poses: Vec<(ViewId, WorldToCameraPose)>,
        mut points: Vec<(LandmarkId, Point3<f64>)>,
    ) -> Result<Self> {
        poses.sort_by_key(|(id, _)| *id);
        points.sort_by_key(|(id, _)| *id);
        let reconstruction = Self { poses, points };
        reconstruction.validate()?;
        Ok(reconstruction)
    }

    /// Checks for duplicate pose or point ids, non-orthonormal rotations, and non-finite values.
    ///
    /// # Errors
    ///
    /// Returns [`SfmError::InvalidInput`] describing the first violation.
    pub fn validate(&self) -> Result<()> {
        unique_indices(
            self.poses.iter().map(|(id, _)| id.index()),
            "reconstruction pose",
        )?;
        unique_indices(
            self.points.iter().map(|(id, _)| id.index()),
            "reconstruction point",
        )?;
        validate_poses(&self.poses)?;
        validate_points(&self.points)
    }
}

impl<'de> Deserialize<'de> for GroundTruthReconstruction {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SerializedReconstruction {
            poses: Vec<(ViewId, WorldToCameraPose)>,
            points: Vec<(LandmarkId, Point3<f64>)>,
        }

        let reconstruction = SerializedReconstruction::deserialize(deserializer)?;
        Self::try_new(reconstruction.poses, reconstruction.points).map_err(serde::de::Error::custom)
    }
}

fn unique_indices(indices: impl IntoIterator<Item = usize>, kind: &'static str) -> Result<()> {
    let mut unique = BTreeSet::new();
    for index in indices {
        if !unique.insert(index) {
            return Err(SfmError::InvalidInput(format!(
                "duplicate {kind} id {index}"
            )));
        }
    }
    Ok(())
}

fn validate_poses(poses: &[(ViewId, WorldToCameraPose)]) -> Result<()> {
    for (id, pose) in poses {
        let rotation = pose.rotation().matrix();
        let orthogonality_error =
            (rotation.transpose() * rotation - nalgebra::Matrix3::identity()).norm();
        if !pose.center().coords.iter().all(|value| value.is_finite())
            || !rotation.iter().all(|value| value.is_finite())
            || orthogonality_error > 1e-6
            || (rotation.determinant() - 1.0).abs() > 1e-6
        {
            return Err(SfmError::InvalidInput(format!(
                "pose {id} contains invalid geometry"
            )));
        }
    }
    Ok(())
}

fn validate_points(points: &[(LandmarkId, Point3<f64>)]) -> Result<()> {
    for (id, point) in points {
        if !point.coords.iter().all(|value| value.is_finite()) {
            return Err(SfmError::InvalidInput(format!(
                "point {id} contains non-finite geometry"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    fn pose(x: f64) -> WorldToCameraPose {
        WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::new(x, 0.0, 0.0))
            .unwrap()
    }

    #[test]
    fn constructors_canonicalize_order_and_pair_direction() {
        let scene = GroundTruthScene::try_new(
            vec![
                (ViewId::new(1), ImageMeta::external_id("b", 64, 48)),
                (ViewId::new(0), ImageMeta::external_id("a", 64, 48)),
            ],
            vec![(
                IntrinsicId::new(0),
                Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap(),
            )],
            vec![(ViewId::new(1), pose(1.0)), (ViewId::new(0), pose(0.0))],
            Vec::new(),
            vec![ViewPair::new(ViewId::new(1), ViewId::new(0))],
        )
        .unwrap();

        assert_eq!(scene.poses[0].0, ViewId::new(0));
        assert_eq!(scene.pairs, [ViewPair::new(ViewId::new(0), ViewId::new(1))]);
    }

    #[test]
    fn duplicate_ids_and_undirected_pairs_are_rejected() {
        let duplicate_poses = GroundTruthReconstruction {
            poses: vec![(ViewId::new(0), pose(0.0)), (ViewId::new(0), pose(1.0))],
            points: Vec::new(),
        };
        assert!(duplicate_poses.validate().is_err());

        let result = GroundTruthScene::try_new(
            vec![
                (ViewId::new(0), ImageMeta::external_id("a", 64, 48)),
                (ViewId::new(1), ImageMeta::external_id("b", 64, 48)),
            ],
            Vec::new(),
            vec![(ViewId::new(0), pose(0.0)), (ViewId::new(1), pose(1.0))],
            Vec::new(),
            vec![
                ViewPair::new(ViewId::new(0), ViewId::new(1)),
                ViewPair::new(ViewId::new(1), ViewId::new(0)),
            ],
        );
        assert!(result.is_err());
    }

    #[test]
    fn deserialization_cannot_bypass_validation() {
        let invalid = serde_json::json!({
            "poses": [[0, pose(0.0)], [0, pose(1.0)]],
            "points": [],
        });
        assert!(serde_json::from_value::<GroundTruthReconstruction>(invalid).is_err());
    }
}
