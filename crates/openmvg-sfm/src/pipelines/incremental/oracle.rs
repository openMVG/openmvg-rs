// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental growth from a ground-truth initial pair, for oracle experiments.

use crate::ba::backend::BundleAdjustmentBackend;
use crate::backends::{PoseRegistrar, Triangulator};
use crate::config::IncrementalSfmConfig;
use crate::events::{EventEmitter, SfmEventBranch};
use crate::{Result, SfmError, SfmWorkspace};
use crate::{
    evaluation::ground_truth::{GroundTruthReconstruction, GroundTruthScene},
    evaluation::{AlignmentMethod, EvaluationConfig, evaluate_reconstruction_against_gt},
};

use super::{
    driver::{
        IncrementalEvents, incremental_sfm_from_initialized_inner, validate_incremental_ba_policy,
    },
    next_view::registration_candidates,
    report::{IncrementalOracleSfmReport, IncrementalSfmReport},
    scene_initializer::{IncrementalInitializationReport, initialize_incremental_seed_from_oracle},
};

/// Continue the default incremental pipeline from an already initialized initial pair.
///
/// This is the narrow seam used by oracle initial-pair experiments. It shares the exact
/// growth, retriangulation, and BA tail used by [`incremental_sfm_with`], so an experiment
/// cannot exercise a second, benchmark-only SfM implementation.
pub(crate) fn incremental_sfm_from_initialized_with(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    initialization: IncrementalInitializationReport,
    registrar: &dyn PoseRegistrar,
    triangulator: &dyn Triangulator,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
) -> Result<IncrementalSfmReport> {
    let mut emitter = EventEmitter::disabled();
    let mut events = IncrementalEvents::new(SfmEventBranch::Main, &mut emitter);
    incremental_sfm_from_initialized_inner(
        workspace,
        config,
        initialization,
        registrar,
        triangulator,
        ba_backend,
        &mut events,
    )
}

/// Run incremental growth from a ground-truth initial pair with the default GenRANSAC-P3P and
/// robust track-triangulation policies, without bundle adjustment.
pub fn incremental_sfm_oracle(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    ground_truth: &GroundTruthScene,
) -> Result<IncrementalOracleSfmReport> {
    incremental_sfm_oracle_with(
        workspace,
        config,
        ground_truth,
        &crate::pipelines::incremental::GenRansacP3pRegistrar::default(),
        &crate::adapters::triangulation::TrackTriangulatorAdapter::default(),
        None,
    )
}

/// Run the real incremental growth tail from a ground-truth initial pair.
///
/// Ground truth is used only to place the two initial cameras and to evaluate the
/// result. The initial landmarks are triangulated from track observations, and every later
/// camera is estimated by `registrar`. This isolates P3P, robust track triangulation, and an
/// optional BA backend without bypassing the default incremental driver.
#[allow(clippy::too_many_arguments)]
pub fn incremental_sfm_oracle_with(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    ground_truth: &GroundTruthScene,
    registrar: &dyn PoseRegistrar,
    triangulator: &dyn Triangulator,
    bundle_adjuster: Option<&mut dyn crate::ba::BundleAdjuster>,
) -> Result<IncrementalOracleSfmReport> {
    let adjuster = bundle_adjuster.map(crate::ba::native::DirectBundleAdjustmentBackend::new);
    let ba_backend = adjuster
        .as_ref()
        .map(|adjuster| adjuster as &dyn BundleAdjustmentBackend);
    ground_truth.validate()?;
    validate_incremental_ba_policy(config, ba_backend)?;
    let initialization =
        initialize_incremental_seed_from_oracle(workspace, config, &ground_truth.reconstruction())?;
    let reconstruction = incremental_sfm_from_initialized_with(
        workspace,
        config,
        initialization,
        registrar,
        triangulator,
        ba_backend,
    )?;
    let growth_candidates = registration_candidates(workspace, &config.growth_config());
    let evaluation_config = EvaluationConfig::default().with_alignment(AlignmentMethod::None);
    let evaluation = evaluate_reconstruction_against_gt(
        &workspace_ground_truth_reconstruction(workspace),
        ground_truth,
        &evaluation_config,
    )
    .map_err(|err| SfmError::InvalidInput(format!("oracle evaluation failed: {err}")))?;
    Ok(IncrementalOracleSfmReport {
        reconstruction,
        growth_candidates,
        evaluation,
    })
}

fn workspace_ground_truth_reconstruction(workspace: &SfmWorkspace) -> GroundTruthReconstruction {
    GroundTruthReconstruction {
        poses: workspace
            .registered_views()
            .map(|(view, pose)| (view, pose.pose.clone()))
            .collect(),
        points: workspace
            .landmarks()
            .as_slice()
            .iter()
            .map(|landmark| (landmark.id, landmark.point))
            .collect(),
    }
}
