// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Variable layout, gauge state, and active-parameter tracking.

use super::{POINT_DIMENSION, norm};
use crate::kernel::RelativePose;
#[cfg(feature = "faer")]
use crate::kernel::multi_view::CameraPoseVariables;
#[cfg(any(test, feature = "unstable-internals"))]
use crate::kernel::multi_view::FixedVariableBlocks;
use crate::kernel::multi_view::ScaleGauge;
#[cfg(any(test, feature = "unstable-internals"))]
use crate::kernel::multi_view::dense::DenseBundleAdjustmentWorkspace;
use crate::numerics::manifold;

#[derive(Clone, Copy)]
pub(in crate::kernel::multi_view) struct VariableLayout<'a> {
    pub offsets: &'a [usize],
    pub pose_dimensions: &'a [usize],
    pub intrinsic_offsets: &'a [usize],
    pub intrinsic_dimensions: &'a [usize],
    pub pose_masks: &'a [crate::PoseVariables],
    #[cfg(feature = "faer")]
    pub pose_variables: CameraPoseVariables,
}

impl<'a> VariableLayout<'a> {
    #[cfg(any(feature = "faer", test))]
    pub const fn calibrated(offsets: &'a [usize], dimensions: &'a [usize]) -> Self {
        Self {
            offsets,
            pose_dimensions: dimensions,
            intrinsic_offsets: &[],
            intrinsic_dimensions: &[],
            pose_masks: &[],
            #[cfg(feature = "faer")]
            pose_variables: CameraPoseVariables::RotationAndTranslation,
        }
    }

    #[cfg(feature = "faer")]
    pub const fn calibrated_translation_only(
        offsets: &'a [usize],
        dimensions: &'a [usize],
    ) -> Self {
        Self {
            offsets,
            pose_dimensions: dimensions,
            intrinsic_offsets: &[],
            intrinsic_dimensions: &[],
            pose_masks: &[],
            pose_variables: CameraPoseVariables::TranslationOnly,
        }
    }

    #[cfg(any(feature = "faer", test))]
    pub const fn with_pose_masks(mut self, masks: &'a [crate::PoseVariables]) -> Self {
        self.pose_masks = masks;
        self
    }

    #[inline]
    pub(super) fn pose_mask(self, camera: usize) -> crate::PoseVariables {
        self.pose_masks
            .get(camera)
            .copied()
            .unwrap_or(crate::PoseVariables::FREE)
    }

    #[inline]
    pub(super) fn intrinsic(self, group: usize) -> Option<(usize, usize)> {
        let dimension = self.intrinsic_dimensions.get(group).copied().unwrap_or(0);
        (dimension > 0).then(|| (self.intrinsic_offsets[group], dimension))
    }
}

#[derive(Clone, Copy)]
pub(in crate::kernel::multi_view) struct GaugeState {
    pub fixed_camera: Option<usize>,
    pub fixed_component: Option<usize>,
    pub fixed_component_value: f64,
    pub baseline: f64,
    pub scene_scale: f64,
    pub direction: [f64; 3],
    pub basis: [[f64; 3]; 2],
}

impl GaugeState {
    pub fn new(gauge: ScaleGauge, poses: &[RelativePose], scene_scale: f64) -> Self {
        match gauge {
            ScaleGauge::Free => Self {
                fixed_camera: None,
                fixed_component: None,
                fixed_component_value: 0.0,
                baseline: 0.0,
                scene_scale,
                direction: [0.0; 3],
                basis: [[0.0; 3]; 2],
            },
            ScaleGauge::FixedBaseline { camera } => {
                let baseline = norm(&poses[camera].translation);
                let direction = poses[camera].translation.map(|value| value / baseline);
                Self {
                    fixed_camera: Some(camera),
                    fixed_component: None,
                    fixed_component_value: 0.0,
                    baseline,
                    scene_scale,
                    direction,
                    basis: manifold::s2_tangent_basis(&direction),
                }
            }
            ScaleGauge::FixedTranslationComponent { camera, component } => Self {
                fixed_camera: Some(camera),
                fixed_component: Some(component.index()),
                fixed_component_value: poses[camera].translation[component.index()],
                baseline: 0.0,
                scene_scale,
                direction: [0.0; 3],
                basis: [[0.0; 3]; 2],
            },
        }
    }
}

#[cfg(test)]
pub(in crate::kernel::multi_view) fn configure_layout(
    workspace: &mut DenseBundleAdjustmentWorkspace,
    camera_count: usize,
    gauge: ScaleGauge,
    intrinsic_dimension: usize,
) -> usize {
    configure_layout_with_fixed(
        workspace,
        camera_count,
        gauge,
        intrinsic_dimension,
        FixedVariableBlocks::NONE,
    )
}

#[cfg(any(test, feature = "unstable-internals"))]
pub(in crate::kernel::multi_view) fn configure_layout_with_fixed(
    workspace: &mut DenseBundleAdjustmentWorkspace,
    camera_count: usize,
    gauge: ScaleGauge,
    intrinsic_dimension: usize,
    fixed: FixedVariableBlocks<'_>,
) -> usize {
    let mut offset = 0;
    for camera in 0..camera_count {
        let pose_dimension = if camera == 0 || is_fixed(fixed.camera_poses, camera) {
            0
        } else if gauge_camera(gauge) == Some(camera) {
            5
        } else {
            6
        };
        let free_intrinsic_dimension = if intrinsic_dimension == 0 {
            0
        } else {
            fixed.intrinsic_dimension(camera)
        };
        let dimension = pose_dimension + free_intrinsic_dimension;
        workspace.camera_offsets[camera] = offset;
        workspace.camera_dimensions[camera] = dimension;
        workspace.camera_pose_dimensions[camera] = pose_dimension;
        workspace.intrinsic_offsets[camera] = offset + pose_dimension;
        workspace.intrinsic_dimensions[camera] = free_intrinsic_dimension;
        offset += dimension;
    }
    offset
}

#[cfg(feature = "unstable-internals")]
pub(in crate::kernel::multi_view) fn configure_grouped_layout_with_fixed(
    workspace: &mut DenseBundleAdjustmentWorkspace,
    camera_count: usize,
    gauge: ScaleGauge,
    intrinsic_group_count: usize,
    fixed: FixedVariableBlocks<'_>,
) -> usize {
    let mut offset = 0;
    for camera in 0..camera_count {
        let dimension = if camera == 0 || is_fixed(fixed.camera_poses, camera) {
            0
        } else if gauge_camera(gauge) == Some(camera) {
            5
        } else {
            6
        };
        workspace.camera_offsets[camera] = offset;
        workspace.camera_dimensions[camera] = dimension;
        workspace.camera_pose_dimensions[camera] = dimension;
        offset += dimension;
    }
    for group in 0..intrinsic_group_count {
        let dimension = fixed.intrinsic_dimension(group);
        workspace.intrinsic_offsets[group] = offset;
        workspace.intrinsic_dimensions[group] = dimension;
        offset += dimension;
    }
    offset
}

#[cfg(any(test, feature = "unstable-internals"))]
const fn gauge_camera(gauge: ScaleGauge) -> Option<usize> {
    match gauge {
        ScaleGauge::Free => None,
        ScaleGauge::FixedBaseline { camera }
        | ScaleGauge::FixedTranslationComponent { camera, .. } => Some(camera),
    }
}

#[inline]
pub(super) const fn is_fixed(mask: &[bool], index: usize) -> bool {
    !mask.is_empty() && mask[index]
}

pub(super) struct ActiveParameterTracker<'a> {
    active_parameters: &'a mut [usize],
    parameter_marks: &'a mut [bool],
    pub(super) count: usize,
}

impl<'a> ActiveParameterTracker<'a> {
    pub(super) fn new(active_parameters: &'a mut [usize], parameter_marks: &'a mut [bool]) -> Self {
        Self {
            active_parameters,
            parameter_marks,
            count: 0,
        }
    }

    /// Register the parameter runs of an observation, each a whole variable
    /// block, and zero the cross rows of the new ones.
    pub(super) fn register(&mut self, runs: [std::ops::Range<usize>; 2], cross: &mut [f64]) {
        let (cross_rows, _) = cross.as_chunks_mut::<POINT_DIMENSION>();
        for run in runs.into_iter().filter(|run| !run.is_empty()) {
            // A block is new exactly when its first parameter is, so only that
            // parameter is marked.
            if std::mem::replace(&mut self.parameter_marks[run.start], true) {
                continue;
            }
            let count = self.count + run.len();
            for (slot, parameter) in self.active_parameters[self.count..count]
                .iter_mut()
                .zip(run.clone())
            {
                *slot = parameter;
            }
            self.count = count;
            cross_rows[run].fill([0.0; POINT_DIMENSION]);
        }
    }
}

impl Drop for ActiveParameterTracker<'_> {
    fn drop(&mut self) {
        for &parameter in &self.active_parameters[..self.count] {
            self.parameter_marks[parameter] = false;
        }
    }
}
