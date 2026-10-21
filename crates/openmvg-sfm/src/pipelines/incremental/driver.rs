// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental engine orchestration: initial pair, view-by-view registration, periodic and
//! final bundle adjustment, and the stopping rules.

use crate::ba::backend::BundleAdjustmentBackend;
use crate::backends::{PoseRegistrar, Triangulator};
use crate::config::{GaugeScalePolicy, IncrementalSfmConfig, LengthUnit};
use crate::events::{
    EventEmitter, SfmEventBranch, SfmEventPhase, SfmEventSink, SfmEventTransition,
};
use crate::ops::reconstruction::TriangulationRejectionSummary;
use crate::ops::rollback::GeometryRollback;
use crate::progress::{SfmProgressSink, SfmProgressStage, cancellation_point};
use crate::{Result, SfmError, SfmWorkspace, ViewId};

use super::{
    growth::{GrowthLimits, GrowthLoopReport, GrowthPass},
    refinement::{FinalRefinementOutcome, run_final_refinement},
    report::IncrementalSfmReport,
    scene_initializer::{
        IncrementalInitializationReport, initialize_incremental_seed_with_bundle_adjustment,
        refine_two_view_seed,
    },
    transaction::BundleAdjustmentRollback,
};

pub(super) struct IncrementalEvents<'emitter, 'sink> {
    branch: SfmEventBranch,
    emitter: &'emitter mut EventEmitter<'sink>,
    next_registration: usize,
    next_global_refinement: usize,
    /// Views growth has not registered yet, counted down by [`SfmProgressStage::Registration`].
    unregistered: Vec<ViewId>,
}

impl<'emitter, 'sink> IncrementalEvents<'emitter, 'sink> {
    pub(super) fn new(branch: SfmEventBranch, emitter: &'emitter mut EventEmitter<'sink>) -> Self {
        Self {
            branch,
            emitter,
            next_registration: 0,
            next_global_refinement: 0,
            unregistered: Vec::new(),
        }
    }

    pub(super) fn progress(&self) -> &'sink dyn SfmProgressSink {
        self.emitter.progress()
    }

    pub(super) fn begin_registration(&mut self, workspace: &SfmWorkspace) {
        self.unregistered = workspace
            .views()
            .iter()
            .map(|view| view.id)
            .filter(|&view| !workspace.has_pose(view))
            .collect();
        self.progress()
            .begin(SfmProgressStage::Registration, self.unregistered.len());
    }

    /// Completes the views growth gave up on.
    pub(super) fn finish_registration(&mut self) {
        self.progress()
            .advance(SfmProgressStage::Registration, self.unregistered.len());
        self.unregistered.clear();
        self.progress().finish(SfmProgressStage::Registration);
    }

    pub(super) fn emit(
        &mut self,
        phase: SfmEventPhase,
        transition: SfmEventTransition,
        workspace: &SfmWorkspace,
    ) {
        self.emitter.emit(self.branch, phase, transition, workspace);
    }

    pub(super) fn emit_registration(&mut self, workspace: &SfmWorkspace) {
        let index = self.next_registration;
        self.next_registration = self.next_registration.saturating_add(1);
        self.emit(
            SfmEventPhase::IncrementalRegistration { index },
            SfmEventTransition::Committed,
            workspace,
        );
        // A view counts once, on its first registration, even if a later cleanup removes it.
        let pending = self.unregistered.len();
        self.unregistered.retain(|&view| !workspace.has_pose(view));
        self.progress().advance(
            SfmProgressStage::Registration,
            pending - self.unregistered.len(),
        );
    }

    pub(super) fn next_global_refinement(&mut self) -> usize {
        let round = self.next_global_refinement;
        self.next_global_refinement = self.next_global_refinement.saturating_add(1);
        round
    }
}

/// Run incremental SfM from an already verified workspace with stable precomputed tracks.
///
/// Executes the initial-pair, P3P growth, track updates, global BA, rollback, and
/// finalization path; feature extraction, matching, verification, and track construction are
/// the caller's job. Existing private dense pair indices and [`crate::TrackId`] values are
/// preserved. When `sink` is given it receives immutable, behavior-neutral checkpoints.
/// If any stage fails, all reconstruction state is restored to its state at entry.
#[cfg_attr(not(feature = "unstable-test-support"), allow(dead_code))]
pub fn incremental_sfm_with(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    registrar: &dyn PoseRegistrar,
    triangulator: &dyn Triangulator,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
    sink: Option<&mut dyn SfmEventSink>,
) -> Result<IncrementalSfmReport> {
    let mut events = match sink {
        Some(sink) => EventEmitter::observing(sink),
        None => EventEmitter::disabled(),
    };
    incremental_sfm_pretracked_with_events(
        workspace,
        config,
        registrar,
        triangulator,
        ba_backend,
        &mut events,
    )
}

pub(crate) fn incremental_sfm_pretracked_with_events(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    registrar: &dyn PoseRegistrar,
    triangulator: &dyn Triangulator,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
    emitter: &mut EventEmitter<'_>,
) -> Result<IncrementalSfmReport> {
    validate_incremental_ba_policy(config, ba_backend)?;
    if workspace.tracks().is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "precomputed tracks",
            method: "build_tracks",
        });
    }
    if workspace.geometric_model_store().populated_count() == 0 {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "verified geometric models",
            method: "verify_pairs",
        });
    }
    let transaction = GeometryRollback::capture(workspace);
    let result = incremental_sfm_pretracked_with_events_inner(
        workspace,
        config,
        registrar,
        triangulator,
        ba_backend,
        emitter,
    );
    if result.is_err() {
        transaction.rollback(workspace);
        emitter.emit(
            SfmEventBranch::Main,
            SfmEventPhase::IncrementalFinal,
            SfmEventTransition::RolledBack,
            workspace,
        );
    }
    result
}

fn incremental_sfm_pretracked_with_events_inner(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    registrar: &dyn PoseRegistrar,
    triangulator: &dyn Triangulator,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
    emitter: &mut EventEmitter<'_>,
) -> Result<IncrementalSfmReport> {
    let progress = emitter.progress();
    cancellation_point(progress)?;
    progress.begin(SfmProgressStage::InitialPair, 1);
    let initialization = initialize_incremental_seed_with_bundle_adjustment(
        workspace,
        config,
        triangulator,
        ba_backend,
    )?;
    progress.advance(SfmProgressStage::InitialPair, 1);
    progress.finish(SfmProgressStage::InitialPair);
    let mut events = IncrementalEvents::new(SfmEventBranch::Main, emitter);
    let report = incremental_sfm_from_initialized_inner(
        workspace,
        config,
        initialization,
        registrar,
        triangulator,
        ba_backend,
        &mut events,
    )?;
    // Stages that roll back on any error may have absorbed a cancellation; checking inside the
    // caller's transaction restores the workspace.
    cancellation_point(progress)?;
    Ok(report)
}

/// Reject configs whose stages need bundle adjustment when no adjuster was supplied.
pub(super) fn validate_incremental_ba_policy(
    config: &IncrementalSfmConfig,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
) -> Result<()> {
    config.validate()?;
    if config.bundle_adjust_initial_pair() && ba_backend.is_none() {
        return Err(SfmError::InvalidConfig(
            "bundle_adjust_initial_pair requires a bundle-adjustment backend".to_owned(),
        ));
    }
    if config.self_calibrate_intrinsics() && ba_backend.is_none() {
        return Err(SfmError::InvalidConfig(
            "self_calibrate_intrinsics requires a bundle-adjustment backend".to_owned(),
        ));
    }
    if config.bundle_adjust_growth_ratio() > 0.0 && ba_backend.is_none() {
        return Err(SfmError::InvalidConfig(
            "bundle_adjust_growth_ratio requires a bundle-adjustment backend".to_owned(),
        ));
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(super) fn incremental_sfm_from_initialized_inner(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    mut initialization: IncrementalInitializationReport,
    registrar: &dyn PoseRegistrar,
    triangulator: &dyn Triangulator,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
    events: &mut IncrementalEvents<'_, '_>,
) -> Result<IncrementalSfmReport> {
    validate_incremental_ba_policy(config, ba_backend)?;
    if workspace.tracks().is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "tracks",
            method: "build_tracks",
        });
    }
    if initialization.initialized_views.is_empty()
        || initialization
            .initialized_views
            .iter()
            .any(|view| !workspace.has_pose(*view))
    {
        return Err(SfmError::InvalidInput(
            "initialized incremental growth requires every reported initial-pair view to have a pose"
                .to_owned(),
        ));
    }
    if workspace.components().len() != 1 {
        return Err(SfmError::InvalidInput(
            "initialized incremental growth requires exactly one reconstruction component"
                .to_owned(),
        ));
    }
    if config.bundle_adjust_initial_pair() && initialization.seed_bundle_adjustment.is_none() {
        let view_a = *initialization.initialized_views.first().ok_or_else(|| {
            SfmError::InvalidInput("initial-pair BA requires two initialized views".to_owned())
        })?;
        let view_b = *initialization.initialized_views.get(1).ok_or_else(|| {
            SfmError::InvalidInput("initial-pair BA requires two initialized views".to_owned())
        })?;
        let snapshot = BundleAdjustmentRollback::capture(workspace);
        let center_a = workspace
            .pose(view_a)
            .ok_or_else(|| {
                SfmError::InvalidInput(format!("initial-pair pose {view_a} is missing"))
            })?
            .pose
            .center();
        let center_b = workspace
            .pose(view_b)
            .ok_or_else(|| {
                SfmError::InvalidInput(format!("initial-pair pose {view_b} is missing"))
            })?
            .pose
            .center();
        let baseline = (center_b - center_a).norm();
        if !baseline.is_finite() || baseline <= 0.0 {
            return Err(SfmError::InvalidInput(
                "initial-pair BA requires a positive finite baseline".to_owned(),
            ));
        }
        let component = workspace
            .components
            .iter_mut()
            .find(|component| {
                component.views.contains(&view_a) && component.views.contains(&view_b)
            })
            .ok_or_else(|| {
                SfmError::InvalidInput(
                    "initial-pair views must share a reconstruction component".to_owned(),
                )
            })?;
        component.gauge_scale_policy = GaugeScalePolicy::FixedBaseline {
            first: view_a,
            second: view_b,
            distance: baseline,
            unit: LengthUnit::Arbitrary,
        };
        let seed_ba_backend = ba_backend.ok_or_else(|| {
            SfmError::InvalidConfig(
                "initial-pair bundle adjustment requires a bundle-adjustment backend".to_owned(),
            )
        })?;
        match refine_two_view_seed(workspace, view_a, view_b, seed_ba_backend) {
            Ok(report) => initialization.seed_bundle_adjustment = Some(report),
            Err(error) => {
                snapshot.rollback(workspace)?;
                return Err(error);
            }
        }
    }
    events.emit(
        SfmEventPhase::IncrementalInitialPair,
        SfmEventTransition::Committed,
        workspace,
    );
    let tail = run_incremental_growth_tail(
        workspace,
        config,
        registrar,
        triangulator,
        ba_backend,
        initialization.triangulation_rejections.clone(),
        events,
    )?;
    let GrowthTailOutcome {
        growth,
        final_refinement,
    } = tail;
    let hit_max_iterations = growth.hit_max_iterations;
    let grown_view_count = workspace
        .registered_view_count()
        .saturating_sub(initialization.initialized_views.len());
    let growth = growth.into_public(grown_view_count);
    let mut report = IncrementalSfmReport::from_workspace(
        workspace,
        initialization,
        growth,
        final_refinement.cleaned_landmarks,
        hit_max_iterations,
    );
    report.final_ba_rejected_count = final_refinement.rejected_count;
    report.final_retriangulation = final_refinement.retriangulation;
    report.final_refinement = final_refinement.report;
    workspace.validate()?;
    events.emit(
        SfmEventPhase::IncrementalFinal,
        SfmEventTransition::Completed,
        workspace,
    );
    Ok(report)
}

struct GrowthTailOutcome {
    growth: GrowthLoopReport,
    final_refinement: FinalRefinementOutcome,
}

fn run_incremental_growth_tail(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    registrar: &dyn PoseRegistrar,
    triangulator: &dyn Triangulator,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
    mut triangulation_rejections: TriangulationRejectionSummary,
    events: &mut IncrementalEvents<'_, '_>,
) -> Result<GrowthTailOutcome> {
    // Poses registered before the tail define the incoming gauge and are outside its ownership.
    // Only views added by this tail may be peeled.
    let protected_views = workspace
        .registered_views()
        .map(|(view, _)| view)
        .collect::<Vec<_>>();
    let registration_config = config.registration_config();
    let growth_config = config.growth_config();
    let n = workspace.views().len();
    let max_iterations = if config.max_iterations() == 0 {
        n.saturating_mul(n).saturating_add(1)
    } else {
        config.max_iterations()
    };
    let growth_limits = GrowthLimits { max_iterations };

    cancellation_point(events.progress())?;
    events.begin_registration(workspace);
    let mut growth = GrowthPass {
        config,
        registrar,
        registration: &registration_config,
        growth: &growth_config,
        triangulator,
        ba_backend,
        protected_views: &protected_views,
        limits: growth_limits,
    }
    .run(workspace, events)?;
    events.finish_registration();

    let final_refinement = run_final_refinement(
        workspace,
        config,
        &registration_config,
        triangulator,
        ba_backend,
        &protected_views,
        events,
    )?;
    if let Some(report) = &final_refinement.retriangulation {
        triangulation_rejections.merge(report.refit_rejections.clone());
        triangulation_rejections.merge(report.growth_rejections.clone());
    }
    triangulation_rejections.merge(growth.triangulation_rejections);
    growth.triangulation_rejections = triangulation_rejections;

    Ok(GrowthTailOutcome {
        growth,
        final_refinement,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::incremental::refinement::tests::workspace_with_filtered_outlier;
    use crate::{Camera, IntrinsicId};

    #[test]
    fn incremental_transaction_rolls_back_calibration() {
        let mut workspace = workspace_with_filtered_outlier();
        let original = workspace.camera_models()[0].clone();
        let snapshot = GeometryRollback::capture(&workspace);
        let Camera::Pinhole(pinhole) = &original else {
            panic!("test workspace must use a pinhole camera");
        };
        let modified = Camera::try_pinhole(
            pinhole.focal_x() * 1.25,
            pinhole.focal_y() * 1.25,
            pinhole.principal_x(),
            pinhole.principal_y(),
        )
        .unwrap();
        workspace
            .replace_camera_model(IntrinsicId::new(0), modified)
            .unwrap();
        snapshot.rollback(&mut workspace);
        assert_eq!(workspace.camera_models()[0], original);
    }
}
