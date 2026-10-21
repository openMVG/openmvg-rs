// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Global rotation and translation averaging over the active verified view graph, with
//! consistency and post-translation validation diagnostics.

use crate::backends::{
    CameraCenters, RotationAveragingFn, RotationEstimates, TranslationAveragingFn,
};
use crate::model::PoseRecord;
use crate::two_view::GeometricModel;
use crate::{ComponentId, Result, SfmError, SfmWorkspace, ViewId, ViewPair};
use nalgebra::Point3;
use openmvg_core::camera::CameraModel;
use openmvg_core::graph::complete_connectivity;
use openmvg_core::model::{RelativeRotation, RelativeTranslation, WorldToCameraPose};
use openmvg_geometry::averaging::{RotationConfig, TranslationConfig};

/// One verified view pair entering global averaging, with its view-graph confidence.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PairEdge {
    pub(crate) pair: ViewPair,
    pub(crate) view_i: ViewId,
    pub(crate) view_j: ViewId,
    /// View-graph consistency multiplier shared by rotation and translation.
    pub(crate) confidence: f64,
}

/// Translation strategy that produced the camera centers of a global averaging run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobalAveragingMethod {
    /// Raw-track global positioning: camera centers and track points solved jointly from bearings.
    GlobalPositioning,
    /// Rotation averaging followed by translation averaging of pairwise baseline directions.
    #[default]
    SeparateRotationTranslation,
}

/// Outcome of one global rotation and translation averaging pass.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlobalAveragingReport {
    /// Translation strategy; older JSON without the key reads as `SeparateRotationTranslation`.
    #[serde(default)]
    pub method: GlobalAveragingMethod,
    /// Active verified pair edges given to the averaging solvers.
    pub active_pair_count: usize,
    /// Cameras covered by the rotation graph, including rotation-only evidence.
    pub rotation_count: usize,
    /// Cameras in the solved metric translation domain. A smaller value means the
    /// remaining rotation-only cameras were explicitly deferred for later growth.
    pub center_count: usize,
    /// Iterations reported by the rotation solver.
    pub rotation_iterations: usize,
    /// Iterations reported by the translation or positioning solver; `0` for a rotation-only pass.
    pub translation_iterations: usize,
    /// Final rotation-solver residual, in radians.
    pub rotation_residual: f64,
    /// Final translation- or positioning-solver residual, in radians; `0.0` for a rotation-only
    /// pass.
    pub translation_residual: f64,
    /// Raw stable-track subset consumed by a track-positioning replacement, absent for
    /// direction-only averaging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positioning_track_count: Option<usize>,
    /// Track observations consumed by global positioning; absent for direction-only averaging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positioning_observation_count: Option<usize>,
}

mod diagnostics;
pub(crate) use diagnostics::angular_distribution;
mod dispatch;
mod positioning_input;
mod rigid_core;

#[cfg(all(test, not(feature = "unstable-internals")))]
pub(crate) use diagnostics::RotationAveragingCameraDiagnostic;
pub use diagnostics::{
    AngularResidualDistribution, RotationAveragingConsistencyReport, VerifiedPairSelection,
    VerifiedPoseGraphConsistencyReport, verified_pose_graph_consistency,
};
#[cfg(feature = "unstable-internals")]
pub use diagnostics::{RotationAveragingCameraDiagnostic, VerifiedPoseGraphCameraDiagnostic};
#[cfg(feature = "unstable-internals")]
pub use dispatch::run_global_averaging_scoped;
pub(crate) use dispatch::{
    GlobalRotationPreparation, ValidatedGlobalRotations, active_geometric_models,
    condition_rotations_adaptive, pair_median_parallax_deg, populate_pose_records,
    prepare_global_rotation_solution, rotation_only_averaging_report,
    run_global_translation_from_validated_rotations,
};
pub(crate) use positioning_input::retain_global_positioning_input;

pub(crate) fn rotation_averaging_consistency_from_preparation(
    workspace: &SfmWorkspace,
    preparation: &GlobalRotationPreparation,
    selection: VerifiedPairSelection,
) -> Result<RotationAveragingConsistencyReport> {
    diagnostics::rotation_averaging_consistency_from_estimates(
        workspace,
        preparation.validate_for(workspace)?.estimates(),
        selection,
    )
}

struct DenseImageMap {
    originals: Vec<ViewId>,
}

impl DenseImageMap {
    fn from_edges(edges: &[PairEdge]) -> Result<Self> {
        let mut originals: Vec<_> = edges
            .iter()
            .flat_map(|edge| [edge.view_i, edge.view_j])
            .collect();
        originals.sort_unstable();
        originals.dedup();
        if originals.is_empty() {
            return Err(SfmError::InvalidInput(
                "cannot average an empty active view graph".to_owned(),
            ));
        }
        Ok(Self { originals })
    }

    fn len(&self) -> usize {
        self.originals.len()
    }

    fn compact(&self, original: ViewId) -> Result<ViewId> {
        self.originals
            .binary_search(&original)
            .map(ViewId::new)
            .map_err(|_| {
                SfmError::InvalidInput(format!("active edge references unmapped view {original}"))
            })
    }

    fn restore(&self, compact: ViewId) -> Result<ViewId> {
        self.originals.get(compact.index()).copied().ok_or_else(|| {
            SfmError::InvalidInput(format!(
                "averaging solver returned out-of-range compact view {compact}"
            ))
        })
    }

    fn remap_edges(&self, edges: &[PairEdge]) -> Result<Vec<PairEdge>> {
        edges
            .iter()
            .map(|edge| {
                Ok(PairEdge {
                    pair: edge.pair,
                    view_i: self.compact(edge.view_i)?,
                    view_j: self.compact(edge.view_j)?,
                    confidence: edge.confidence,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
