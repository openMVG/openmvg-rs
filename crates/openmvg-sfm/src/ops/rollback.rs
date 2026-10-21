// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Internal rollback snapshots shared by pipeline branches.

use openmvg_core::camera::Camera;

use crate::model::{Landmarks, PoseRecord, ReconstructionComponent};
use crate::pipelines::incremental::RegistrationFailure;
use crate::workspace::reports::WorkspaceReports;
use crate::workspace::store::SparseStore;

/// Rollback state for mutations to reconstruction geometry and its derived reports.
///
/// This snapshot covers poses, landmarks, components, registration failures,
/// camera models, and workspace reports. It does not capture front-end features, matches,
/// geometric models, tracks, rotation estimates, or camera-center estimates.
#[derive(Debug)]
pub(crate) struct GeometryRollback {
    poses: SparseStore<PoseRecord>,
    landmarks: Landmarks,
    components: Vec<ReconstructionComponent>,
    registration_failures: Vec<RegistrationFailure>,
    camera_models: Vec<Camera>,
    reports: WorkspaceReports,
}

impl GeometryRollback {
    pub(crate) fn capture(workspace: &crate::SfmWorkspace) -> Self {
        Self {
            poses: workspace.pose_store().clone(),
            landmarks: workspace.landmarks().clone(),
            components: workspace.components().to_vec(),
            registration_failures: workspace.registration_failures().to_vec(),
            camera_models: workspace.camera_models().to_vec(),
            reports: workspace.reports.clone(),
        }
    }

    pub(crate) fn rollback(self, workspace: &mut crate::SfmWorkspace) {
        workspace.poses = self.poses;
        workspace.landmarks = self.landmarks;
        workspace.components = self.components;
        workspace.registration_failures = self.registration_failures;
        workspace.reports = self.reports;
        workspace.restore_camera_models(self.camera_models);
    }
}
