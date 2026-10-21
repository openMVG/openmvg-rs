// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Landmark triangulation, bundle adjustment, and reconstruction cleanup over a workspace.

use std::collections::BTreeMap;

use crate::ba::backend::{BaBackendStatus, BundleAdjustmentBackend, BundleAdjustmentReport};
use crate::ba::config::{BundleAdjustConfig, OptimizationScope};
use crate::backends::{
    PosedPixelObservation, TriangulatedLandmark, TriangulationRejectReason, TriangulationVerdict,
    Triangulator,
};
use crate::config::GaugeScalePolicy;
use crate::model::{
    Landmark, LandmarkStatus, Landmarks, ObservationSupport, ReconstructionComponent,
};
use crate::{
    ComponentId, IntrinsicId, LandmarkId, Result, SfmError, SfmWorkspace, TrackId, ViewId,
};
use nalgebra::Point3;

struct PosedObservation<'a> {
    view: ViewId,
    xy: nalgebra::Point2<f64>,
    pose: &'a openmvg_core::model::WorldToCameraPose,
    camera: &'a openmvg_core::camera::Camera,
}

struct PosedTrack<'a> {
    observations: Vec<PosedObservation<'a>>,
    source_indices: Vec<usize>,
}

struct AcceptedTriangulation {
    point: Point3<f64>,
    reprojection_error: openmvg_core::PixelResidual,
    support: ObservationSupport,
}

type RefitOutcome = (LandmarkId, PosedTriangulationOutcome);

enum PosedTriangulationOutcome {
    Accepted(AcceptedTriangulation),
    Rejected(TriangulationRejectReason),
}

#[cfg(feature = "parallel")]
const MIN_PARALLEL_JOBS: usize = 128;

#[cfg(feature = "parallel")]
fn map_ordered<T, U, F>(items: &[T], map: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    if items.len() < MIN_PARALLEL_JOBS {
        items.iter().map(&map).collect()
    } else {
        use rayon::prelude::*;
        items.par_iter().map(map).collect()
    }
}

#[cfg(not(feature = "parallel"))]
fn map_ordered<T, U, F>(items: &[T], map: F) -> Vec<U>
where
    F: Fn(&T) -> U,
{
    items.iter().map(map).collect()
}

fn take_first_error<T>(results: &mut Vec<Result<T>>) -> Option<SfmError> {
    let index = results.iter().position(Result::is_err)?;
    results.swap_remove(index).err()
}

/// Per-reason counts of rejected triangulation attempts.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TriangulationRejectionSummary {
    counts: BTreeMap<TriangulationRejectReason, usize>,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl TriangulationRejectionSummary {
    /// Number of triangulation attempts rejected for one reason.
    pub fn count(&self, reason: TriangulationRejectReason) -> usize {
        self.counts.get(&reason).copied().unwrap_or(0)
    }

    /// Total rejected attempts represented by this summary.
    pub fn total(&self) -> usize {
        self.counts.values().sum()
    }

    /// Whether no rejection has been recorded.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Iterate non-zero counts in stable reason order.
    pub fn iter(&self) -> impl Iterator<Item = (TriangulationRejectReason, usize)> + '_ {
        self.counts.iter().map(|(&reason, &count)| (reason, count))
    }

    fn record(&mut self, reason: TriangulationRejectReason) {
        *self.counts.entry(reason).or_default() += 1;
    }

    pub(crate) fn merge(&mut self, other: Self) {
        for (reason, count) in other.counts {
            *self.counts.entry(reason).or_default() += count;
        }
    }
}

impl PosedTrack<'_> {
    fn len(&self) -> usize {
        self.observations.len()
    }
}

/// Outcome of one full triangulation pass over every track.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TriangulationReport {
    /// Tracks considered: every track in the workspace.
    pub attempted_tracks: usize,
    /// Landmarks created; the pass replaces all previous landmarks.
    pub triangulated_landmarks: usize,
    /// Rejected tracks, by reason.
    pub rejections: TriangulationRejectionSummary,
    /// Scale gauge recorded on the rebuilt reconstruction component.
    pub gauge_scale_policy: GaugeScalePolicy,
    /// Minimum posed observations required by this bootstrap pass.
    #[serde(default = "default_triangulation_min_observations")]
    pub min_observations: usize,
}

const fn default_triangulation_min_observations() -> usize {
    2
}

/// Outcome of one bundle-adjustment solve committed to the workspace.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BundleReport {
    /// Camera poses in the solved problem.
    pub adjusted_cameras: usize,
    /// Landmarks in the solved problem.
    pub adjusted_landmarks: usize,
    /// Landmarks rejected because fewer than two finite projected observations
    /// survived the BA commit.
    pub rejected_landmarks: usize,
    /// Invalid projected observations pruned from landmarks that remained active.
    pub removed_observations: usize,
    /// Backend report for the solve.
    pub ba: BundleAdjustmentReport,
}

/// One robust bundle-adjustment and cleanup pass.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BundleAdjustAndCleanRoundReport {
    /// The bundle-adjustment solve of this round.
    pub bundle_adjustment: BundleReport,
    /// Landmarks rejected because fewer than two valid observations remain or
    /// because the surviving rays have insufficient triangulation angle.
    pub removed_after_adjustment: usize,
    /// Individual high-residual or invalid observations pruned from landmarks
    /// that retain at least two valid observations.
    pub removed_observations_after_adjustment: usize,
}

/// Typed receipt for the bounded robust BA/cleanup schedule.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BundleAdjustAndCleanReport {
    /// Version of this report layout; currently `1`.
    pub schema_version: u32,
    /// Executed BA-and-cleanup rounds, in order (at least one).
    pub rounds: Vec<BundleAdjustAndCleanRoundReport>,
    /// Final polish executed only when every configured cleanup round removed structure.
    pub final_bundle_adjustment: Option<BundleReport>,
    /// Why the schedule stopped.
    pub termination: BundleAdjustAndCleanTermination,
    /// Landmarks removed across all rounds and the final polish.
    pub total_removed_landmarks: usize,
    /// Observations removed across all rounds and the final polish.
    pub total_removed_observations: usize,
}

/// Why a bounded BA-and-cleanup schedule stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleAdjustAndCleanTermination {
    /// A round removed nothing, or no active landmark was left.
    Stable,
    /// A round's removals fell below the configured negligible fraction, so the final polish was
    /// skipped.
    NegligibleRemoval,
    /// Every round still removed structure, so a final bundle adjustment ran on the cleaned scene.
    FinalPolishCompleted,
}

#[derive(Debug, Clone, Copy)]
enum FinalPolishPolicy {
    Required,
    SkipAfterNegligibleRemoval {
        minimum_rounds: usize,
        maximum_removed_fraction: f64,
    },
}

fn cleanup_removal_is_negligible(
    removed_landmarks: usize,
    active_landmarks_before: usize,
    maximum_removed_fraction: f64,
) -> bool {
    removed_landmarks as f64 <= maximum_removed_fraction * active_landmarks_before as f64
}

pub(crate) mod bundle;
mod cleanup;
mod residual;
mod track_growth;
mod triangulation;

#[cfg(test)]
use bundle::bundle_adjust_and_clean;
pub(crate) use bundle::bundle_adjustment_problem;
pub(crate) use bundle::{
    angular_regression_limit, bundle_adjust,
    bundle_adjust_and_clean_report_skipping_negligible_verification,
    fixed_baseline_roundoff_tolerance, pipeline_ba_config_for_workspace,
};
pub use cleanup::PoseSupportCleanupReport;
pub(crate) use cleanup::release_matching_data;
pub(crate) use cleanup::{LandmarkSupportCleanupReport, clean_outlier_landmark_support};
pub(crate) use cleanup::{
    refresh_primary_component, remove_underdetermined_poses, remove_unsupported_poses,
    retain_posed_landmark_support, retain_primary_bundle_component,
};
pub(crate) use residual::{CameraResidualPolicy, camera_bearing_error_rad, camera_pixel_residual};
pub use track_growth::RetriangulationReport;
pub(crate) use track_growth::retriangulate_incremental;
pub(crate) use track_growth::{
    recover_missing_tracks, refit_incremental_for_views, retriangulate_incremental_for_views,
};
#[cfg(feature = "unstable-internals")]
pub use triangulation::IncrementalTriangulationReport;
pub(crate) use triangulation::triangulate_with_min_observations;
pub use triangulation::{triangulate, triangulate_incremental};
pub(crate) use triangulation::{triangulate_global_translation, triangulate_incremental_for_views};

#[cfg(test)]
mod tests;
