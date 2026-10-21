// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gauge, fixed-block, and intrinsic-group layout policy.

use super::super::super::BundleAdjustmentBlock;
use crate::kernel::multi_view::{
    CameraPoseVariables, FixedVariableBlocks, MultiViewBundleAdjustmentError, ScaleGauge,
};

/// Physical-pose incidence for elimination caches, independent of the
/// exposure CSR used for residual values. Stored only when rows repeat poses.
struct PointPoseIncidence {
    offsets: Vec<usize>,
    poses: Vec<u32>,
}

pub(in crate::kernel::multi_view::sparse) struct SparseProblemSignature {
    camera_count: usize,
    point_count: usize,
    observation_cameras: Vec<u32>,
    point_offsets: Vec<usize>,
    point_pose_incidence: Option<PointPoseIncidence>,
    observation_intrinsics: Option<Vec<usize>>,
    gauge: ScaleGauge,
    pose_variables: CameraPoseVariables,
    fixed_camera_poses: Vec<bool>,
    pose_masks: Vec<crate::PoseVariables>,
    explicit_pose_masks: bool,
    model_identity: Option<crate::kernel::multi_view::model::ObservationModelIdentity>,
    fixed_points: Vec<bool>,
}

impl SparseProblemSignature {
    pub(in crate::kernel::multi_view::sparse) fn try_new(
        camera_count: usize,
        point_offsets: &[usize],
        observation_cameras: impl Iterator<Item = usize>,
        gauge: ScaleGauge,
        pose_variables: CameraPoseVariables,
        fixed: FixedVariableBlocks<'_>,
    ) -> Result<Self, MultiViewBundleAdjustmentError> {
        let observation_cameras: Vec<u32> = observation_cameras
            .map(|camera| {
                u32::try_from(camera)
                    .map_err(|_| MultiViewBundleAdjustmentError::CameraCapacityOverflow)
            })
            .collect::<Result<_, _>>()?;
        let point_count = point_offsets.len().saturating_sub(1);
        let point_pose_incidence =
            unique_point_poses(camera_count, point_offsets, &observation_cameras)?;
        Ok(Self {
            camera_count,
            point_count,
            observation_cameras,
            point_offsets: point_offsets.to_vec(),
            point_pose_incidence,
            observation_intrinsics: None,
            gauge,
            pose_variables,
            fixed_camera_poses: canonical_pose_mask(fixed.camera_poses, camera_count),
            fixed_points: canonical_mask(fixed.points, point_count),
            pose_masks: legacy_pose_masks(camera_count, pose_variables, fixed.camera_poses),
            explicit_pose_masks: false,
            model_identity: None,
        })
    }

    pub(in crate::kernel::multi_view::sparse) fn with_pose_masks(
        mut self,
        pose_masks: &[crate::PoseVariables],
        fixed: FixedVariableBlocks<'_>,
    ) -> Self {
        self.pose_masks.clear();
        self.pose_masks.extend_from_slice(pose_masks);
        for (camera, mask) in self.pose_masks.iter_mut().enumerate() {
            if !fixed.camera_poses.is_empty() && fixed.camera_poses[camera] {
                *mask = crate::PoseVariables::FIXED;
            }
        }
        self.fixed_camera_poses = canonical_mask(fixed.camera_poses, self.camera_count);
        self.explicit_pose_masks = true;
        self
    }

    pub(in crate::kernel::multi_view::sparse) fn with_observation_intrinsics(
        mut self,
        selectors: Option<&[usize]>,
    ) -> Self {
        self.observation_intrinsics = selectors.map(ToOwned::to_owned);
        self
    }

    pub(in crate::kernel::multi_view::sparse) fn observation_intrinsics_match(
        &self,
        selectors: Option<&[usize]>,
    ) -> bool {
        self.observation_intrinsics.as_deref() == selectors
    }

    pub(in crate::kernel::multi_view::sparse) fn point_pose_offsets(&self) -> &[usize] {
        self.point_pose_incidence
            .as_ref()
            .map_or(&self.point_offsets, |incidence| &incidence.offsets)
    }

    pub(in crate::kernel::multi_view::sparse) fn point_poses(&self) -> &[u32] {
        self.point_pose_incidence
            .as_ref()
            .map_or(&self.observation_cameras, |incidence| &incidence.poses)
    }

    pub(in crate::kernel::multi_view::sparse) fn with_model_identity(
        mut self,
        identity: crate::kernel::multi_view::model::ObservationModelIdentity,
    ) -> Self {
        self.model_identity = Some(identity);
        self
    }

    pub(in crate::kernel::multi_view::sparse) fn model_identity_matches(
        &self,
        identity: crate::kernel::multi_view::model::ObservationModelIdentity,
    ) -> bool {
        self.model_identity == Some(identity)
    }

    pub(in crate::kernel::multi_view::sparse) fn pose_masks(&self) -> &[crate::PoseVariables] {
        &self.pose_masks
    }

    pub(in crate::kernel::multi_view::sparse) const fn explicit_pose_masks(&self) -> bool {
        self.explicit_pose_masks
    }

    pub(in crate::kernel::multi_view::sparse) fn matches(
        &self,
        camera_count: usize,
        point_offsets: &[usize],
        observation_cameras: impl Iterator<Item = usize>,
        gauge: ScaleGauge,
        pose_variables: CameraPoseVariables,
        fixed: FixedVariableBlocks<'_>,
    ) -> bool {
        camera_count == self.camera_count
            && gauge == self.gauge
            && pose_variables == self.pose_variables
            && point_offsets == self.point_offsets()
            && observation_cameras.eq(self
                .observation_cameras()
                .iter()
                .map(|&camera| camera as usize))
            && if self.explicit_pose_masks {
                mask_matches(fixed.camera_poses, &self.fixed_camera_poses)
            } else {
                pose_mask_matches(fixed.camera_poses, &self.fixed_camera_poses)
            }
            && mask_matches(fixed.points, &self.fixed_points)
    }

    pub(in crate::kernel::multi_view::sparse) const fn camera_count(&self) -> usize {
        self.camera_count
    }

    pub(in crate::kernel::multi_view::sparse) const fn point_count(&self) -> usize {
        self.point_count
    }

    pub(in crate::kernel::multi_view::sparse) fn observation_cameras(&self) -> &[u32] {
        &self.observation_cameras
    }

    pub(in crate::kernel::multi_view::sparse) fn point_offsets(&self) -> &[usize] {
        &self.point_offsets
    }

    pub(in crate::kernel::multi_view::sparse) fn fixed_points(&self) -> &[bool] {
        &self.fixed_points
    }

    pub(in crate::kernel::multi_view::sparse) const fn pose_variables(
        &self,
    ) -> CameraPoseVariables {
        self.pose_variables
    }
}

pub(in crate::kernel::multi_view::sparse) struct SelfCalibrationSignature {
    intrinsic_group_count: usize,
    camera_to_group: Option<Vec<usize>>,
    intrinsic_dimensions: Vec<usize>,
}

impl SelfCalibrationSignature {
    pub(in crate::kernel::multi_view::sparse) fn try_new(
        camera_count: usize,
        intrinsic_group_count: usize,
        camera_to_group: Option<&[usize]>,
        fixed: FixedVariableBlocks<'_>,
    ) -> Result<Self, MultiViewBundleAdjustmentError> {
        let fixed_intrinsics = fixed.intrinsics;
        if let Some(camera_to_group) = camera_to_group {
            validate_group_topology(camera_count, intrinsic_group_count, camera_to_group)?;
        }
        if !fixed_intrinsics.is_empty() && fixed_intrinsics.len() != intrinsic_group_count {
            return Err(MultiViewBundleAdjustmentError::FixedBlockCountMismatch {
                block: BundleAdjustmentBlock::RadialIntrinsics,
                expected: intrinsic_group_count,
                actual: fixed_intrinsics.len(),
            });
        }
        if !fixed.intrinsic_dimensions.is_empty()
            && fixed.intrinsic_dimensions.len() != intrinsic_group_count
        {
            return Err(MultiViewBundleAdjustmentError::FixedBlockCountMismatch {
                block: BundleAdjustmentBlock::RadialIntrinsics,
                expected: intrinsic_group_count,
                actual: fixed.intrinsic_dimensions.len(),
            });
        }
        Ok(Self {
            intrinsic_group_count,
            camera_to_group: camera_to_group.map(ToOwned::to_owned),
            intrinsic_dimensions: intrinsic_dimensions(fixed, intrinsic_group_count),
        })
    }

    pub(in crate::kernel::multi_view::sparse) fn matches(
        &self,
        intrinsic_group_count: usize,
        camera_to_group: Option<&[usize]>,
        fixed: FixedVariableBlocks<'_>,
    ) -> bool {
        self.intrinsic_group_count == intrinsic_group_count
            && self.camera_to_group.as_deref() == camera_to_group
            && (fixed.intrinsics.is_empty() || fixed.intrinsics.len() == intrinsic_group_count)
            && (fixed.intrinsic_dimensions.is_empty()
                || fixed.intrinsic_dimensions.len() == intrinsic_group_count)
            && (0..intrinsic_group_count).all(|group| {
                fixed.intrinsic_dimension(group)
                    == self.intrinsic_dimensions.get(group).copied().unwrap_or(3)
            })
    }

    #[cfg(feature = "unstable-internals")]
    pub(in crate::kernel::multi_view::sparse) const fn intrinsic_group_count(&self) -> usize {
        self.intrinsic_group_count
    }

    pub(in crate::kernel::multi_view::sparse) fn camera_to_group(&self) -> Option<&[usize]> {
        self.camera_to_group.as_deref()
    }
}

pub(in crate::kernel::multi_view::sparse) struct ReducedLayout {
    camera_offsets: Vec<usize>,
    camera_dimensions: Vec<usize>,
    intrinsic_offsets: Vec<usize>,
    intrinsic_dimensions: Vec<usize>,
    dimension: usize,
}

impl ReducedLayout {
    pub(in crate::kernel::multi_view::sparse) fn try_calibrated(
        camera_count: usize,
        gauge: ScaleGauge,
        pose_variables: CameraPoseVariables,
        fixed_camera_poses: &[bool],
    ) -> Result<Self, MultiViewBundleAdjustmentError> {
        let mut offsets = vec![0; camera_count];
        let mut dimensions = vec![0; camera_count];
        let mut dimension = 0usize;
        for camera in 0..camera_count {
            let local = if camera == 0
                || (!fixed_camera_poses.is_empty() && fixed_camera_poses[camera])
            {
                0
            } else if matches!(gauge, ScaleGauge::FixedBaseline { camera: fixed } | ScaleGauge::FixedTranslationComponent { camera: fixed, .. } if fixed == camera)
            {
                match pose_variables {
                    CameraPoseVariables::RotationAndTranslation => 5,
                    CameraPoseVariables::TranslationOnly => 2,
                }
            } else {
                match pose_variables {
                    CameraPoseVariables::RotationAndTranslation => 6,
                    CameraPoseVariables::TranslationOnly => 3,
                }
            };
            offsets[camera] = dimension;
            dimensions[camera] = local;
            dimension = dimension
                .checked_add(local)
                .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
        }
        Ok(Self {
            camera_offsets: offsets,
            camera_dimensions: dimensions,
            intrinsic_offsets: Vec::new(),
            intrinsic_dimensions: Vec::new(),
            dimension,
        })
    }

    pub(in crate::kernel::multi_view::sparse) fn try_calibrated_with_pose_masks(
        camera_count: usize,
        gauge: ScaleGauge,
        masks: &[crate::PoseVariables],
        fixed_camera_poses: &[bool],
    ) -> Result<Self, MultiViewBundleAdjustmentError> {
        if masks.len() != camera_count {
            return Err(MultiViewBundleAdjustmentError::FixedBlockCountMismatch {
                block: BundleAdjustmentBlock::WorldToCameraPose,
                expected: camera_count,
                actual: masks.len(),
            });
        }
        let mut layout = Self {
            camera_offsets: vec![0; camera_count],
            camera_dimensions: vec![0; camera_count],
            intrinsic_offsets: Vec::new(),
            intrinsic_dimensions: Vec::new(),
            dimension: 0,
        };
        for (camera, &mask) in masks.iter().enumerate() {
            let fixed = !fixed_camera_poses.is_empty() && fixed_camera_poses[camera];
            let constrained_translation = matches!(gauge,
                ScaleGauge::FixedBaseline { camera: selected }
                | ScaleGauge::FixedTranslationComponent { camera: selected, .. }
                if camera == selected);
            let dimension = if fixed || mask.is_fixed() {
                0
            } else {
                3 * usize::from(mask.rotation())
                    + if mask.translation() {
                        if constrained_translation { 2 } else { 3 }
                    } else {
                        0
                    }
            };
            layout.camera_offsets[camera] = layout.dimension;
            layout.camera_dimensions[camera] = dimension;
            layout.dimension = layout
                .dimension
                .checked_add(dimension)
                .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
        }
        Ok(layout)
    }

    pub(in crate::kernel::multi_view::sparse) fn try_radial(
        camera_count: usize,
        intrinsic_group_count: usize,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
    ) -> Result<Self, MultiViewBundleAdjustmentError> {
        let gauge_camera = match gauge {
            ScaleGauge::Free => None,
            ScaleGauge::FixedBaseline { camera }
            | ScaleGauge::FixedTranslationComponent { camera, .. } => Some(camera),
        };
        let mut dimension = 0usize;
        let mut camera_offsets = vec![0; camera_count];
        let mut camera_dimensions = vec![0; camera_count];
        for camera in 0..camera_count {
            let local =
                if camera == 0 || (!fixed.camera_poses.is_empty() && fixed.camera_poses[camera]) {
                    0
                } else if gauge_camera == Some(camera) {
                    5
                } else {
                    6
                };
            camera_offsets[camera] = dimension;
            camera_dimensions[camera] = local;
            dimension = dimension
                .checked_add(local)
                .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
        }
        let mut intrinsic_offsets = vec![0; intrinsic_group_count];
        let mut intrinsic_dimensions = vec![0; intrinsic_group_count];
        for group in 0..intrinsic_group_count {
            let local = fixed.intrinsic_dimension(group);
            intrinsic_offsets[group] = dimension;
            intrinsic_dimensions[group] = local;
            dimension = dimension
                .checked_add(local)
                .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
        }
        Ok(Self {
            camera_offsets,
            camera_dimensions,
            intrinsic_offsets,
            intrinsic_dimensions,
            dimension,
        })
    }

    pub(in crate::kernel::multi_view::sparse) const fn dimension(&self) -> usize {
        self.dimension
    }

    pub(in crate::kernel::multi_view::sparse) fn camera_offsets(&self) -> &[usize] {
        &self.camera_offsets
    }

    pub(in crate::kernel::multi_view::sparse) fn camera_dimensions(&self) -> &[usize] {
        &self.camera_dimensions
    }

    pub(in crate::kernel::multi_view::sparse) fn intrinsic_offsets(&self) -> &[usize] {
        &self.intrinsic_offsets
    }

    pub(in crate::kernel::multi_view::sparse) fn intrinsic_dimensions(&self) -> &[usize] {
        &self.intrinsic_dimensions
    }
}

/// Free parameters per radial intrinsic block, empty when every block frees all three.
fn intrinsic_dimensions(fixed: FixedVariableBlocks<'_>, count: usize) -> Vec<usize> {
    let dimensions = (0..count)
        .map(|group| fixed.intrinsic_dimension(group))
        .collect::<Vec<_>>();
    if dimensions.iter().all(|&dimension| dimension == 3) {
        Vec::new()
    } else {
        dimensions
    }
}

fn canonical_mask(mask: &[bool], count: usize) -> Vec<bool> {
    if mask.is_empty() || mask.iter().all(|&value| !value) {
        Vec::new()
    } else {
        debug_assert_eq!(mask.len(), count);
        mask.to_vec()
    }
}

fn canonical_pose_mask(mask: &[bool], count: usize) -> Vec<bool> {
    if mask.is_empty() || mask.iter().skip(1).all(|&value| !value) {
        return Vec::new();
    }
    debug_assert_eq!(mask.len(), count);
    let mut canonical = mask.to_vec();
    canonical[0] = false;
    canonical
}

fn mask_matches(mask: &[bool], expected: &[bool]) -> bool {
    mask == expected || (mask.iter().all(|&value| !value) && expected.iter().all(|&value| !value))
}

fn pose_mask_matches(mask: &[bool], expected: &[bool]) -> bool {
    if expected.is_empty() {
        mask.is_empty() || mask.iter().skip(1).all(|&value| !value)
    } else {
        mask.len() == expected.len()
            && mask
                .iter()
                .enumerate()
                .all(|(index, &value)| expected[index] == (index != 0 && value))
    }
}

fn validate_group_topology(
    camera_count: usize,
    intrinsic_group_count: usize,
    camera_to_group: &[usize],
) -> Result<(), MultiViewBundleAdjustmentError> {
    if camera_to_group.len() != camera_count {
        return Err(
            MultiViewBundleAdjustmentError::IntrinsicGroupMapCountMismatch {
                expected: camera_count,
                actual: camera_to_group.len(),
            },
        );
    }
    if intrinsic_group_count == 0 || intrinsic_group_count > camera_count {
        return Err(MultiViewBundleAdjustmentError::InvalidIntrinsicGroupCount {
            cameras: camera_count,
            groups: intrinsic_group_count,
        });
    }
    let mut used = vec![false; intrinsic_group_count];
    for (camera, &group) in camera_to_group.iter().enumerate() {
        let Some(slot) = used.get_mut(group) else {
            return Err(MultiViewBundleAdjustmentError::InvalidIntrinsicGroup {
                camera,
                group,
                groups: intrinsic_group_count,
            });
        };
        *slot = true;
    }
    if let Some(group) = used.iter().position(|&is_used| !is_used) {
        return Err(MultiViewBundleAdjustmentError::UnusedIntrinsicGroup { group });
    }
    Ok(())
}

fn legacy_pose_masks(
    camera_count: usize,
    variables: CameraPoseVariables,
    fixed: &[bool],
) -> Vec<crate::PoseVariables> {
    let free = match variables {
        CameraPoseVariables::RotationAndTranslation => crate::PoseVariables::FREE,
        CameraPoseVariables::TranslationOnly => crate::PoseVariables::FREE.with_rotation(false),
    };
    (0..camera_count)
        .map(|camera| {
            if camera == 0 || (!fixed.is_empty() && fixed[camera]) {
                crate::PoseVariables::FIXED
            } else {
                free
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::mask_matches;

    #[test]
    fn empty_and_explicit_all_free_masks_are_equivalent() {
        assert!(mask_matches(&[], &[]));
        assert!(mask_matches(&[false, false], &[]));
        assert!(mask_matches(&[], &[false, false]));
        assert!(!mask_matches(&[false, true], &[]));
        assert!(!mask_matches(&[false, true], &[false, false]));
    }
}

fn unique_point_poses(
    camera_count: usize,
    offsets: &[usize],
    cameras: &[u32],
) -> Result<Option<PointPoseIncidence>, MultiViewBundleAdjustmentError> {
    let mut marked = vec![false; camera_count];
    let mut touched = Vec::new();
    let mut repeated = false;
    for track in offsets.windows(2) {
        touched.clear();
        let observations = cameras
            .get(track[0]..track[1])
            .ok_or(MultiViewBundleAdjustmentError::InvalidTrackOffsets)?;
        for &camera in observations {
            let camera = camera as usize;
            let Some(mark) = marked.get_mut(camera) else {
                return Err(MultiViewBundleAdjustmentError::InvalidObservationCamera {
                    observation: track[0],
                    camera,
                });
            };
            if *mark {
                repeated = true;
            } else {
                *mark = true;
                touched.push(camera);
            }
        }
        for &camera in &touched {
            marked[camera] = false;
        }
        if repeated {
            break;
        }
    }
    if !repeated {
        return Ok(None);
    }
    let mut incidence = PointPoseIncidence {
        offsets: vec![0],
        poses: Vec::new(),
    };
    for track in offsets.windows(2) {
        touched.clear();
        let observations = cameras
            .get(track[0]..track[1])
            .ok_or(MultiViewBundleAdjustmentError::InvalidTrackOffsets)?;
        for &pose in observations {
            let camera = pose as usize;
            let Some(mark) = marked.get_mut(camera) else {
                return Err(MultiViewBundleAdjustmentError::InvalidObservationCamera {
                    observation: track[0],
                    camera,
                });
            };
            if !*mark {
                *mark = true;
                touched.push(camera);
                // First encounter order matches ActiveParameterTracker's rows.
                incidence.poses.push(pose);
            }
        }
        for &camera in &touched {
            marked[camera] = false;
        }
        incidence.offsets.push(incidence.poses.len());
    }
    Ok(Some(incidence))
}

#[cfg(test)]
mod shared_incidence_tests {
    use super::*;

    #[test]
    fn repeated_rows_keep_exposure_identity_and_unique_pose_first_order() {
        let signature = SparseProblemSignature::try_new(
            3,
            &[0, 4, 6],
            [2, 0, 2, 1, 1, 1].into_iter(),
            ScaleGauge::Free,
            CameraPoseVariables::RotationAndTranslation,
            FixedVariableBlocks::NONE,
        )
        .unwrap();
        assert_eq!(signature.point_offsets(), &[0, 4, 6]);
        assert_eq!(signature.observation_cameras(), &[2, 0, 2, 1, 1, 1]);
        assert_eq!(signature.point_pose_offsets(), &[0, 3, 4]);
        assert_eq!(signature.point_poses(), &[2, 0, 1, 1]);
        let signature = signature.with_observation_intrinsics(Some(&[0, 1, 2, 3, 4, 5]));
        assert!(signature.observation_intrinsics_match(Some(&[0, 1, 2, 3, 4, 5])));
        assert!(!signature.observation_intrinsics_match(Some(&[2, 1, 0, 3, 4, 5])));
    }
}
