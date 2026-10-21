// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Engine execution: backend resolution, transactions, and report assembly.

use openmvg_geometry::averaging::{average_rotations, average_translations};

use super::config::{GlobalBackends, IncrementalBackends, SfmConfig, SfmEngine};
use super::report::{GlobalOutcome, RunReport, SfmOutcome};
use crate::adapters::triangulation::TrackTriangulatorAdapter;
use crate::ba::NativeBundleAdjuster;
use crate::ba::backend::BundleAdjustmentBackend;
use crate::ba::native::DirectBundleAdjustmentBackend;
use crate::config::IncrementalSfmConfig;
use crate::events::{EventEmitter, SfmEventBranch, SfmEventPhase, SfmEventTransition};
use crate::pipelines::global::{GlobalCompletionReport, GlobalSfmConfig};
use crate::pipelines::global::{
    GlobalRuntime, PreparedGlobal, apply_pair_conditioning_state,
    global_core_from_owned_prepared_rotations_with_events, prepare_global,
};
use crate::pipelines::incremental::{
    GenRansacP4pfRegistrar, incremental_sfm_pretracked_with_events,
};
use crate::progress::{SfmProgressStage, cancellation_point};
use crate::{Result, Runner, SfmError, SfmWorkspace};

/// The configured engine with its resolved algorithms.
enum EngineRun<'r> {
    Incremental(&'r IncrementalSfmConfig, IncrementalBackends<'r>),
    Global(&'r GlobalSfmConfig, GlobalBackends<'r>),
}

// Engine execution over a workspace. Each engine uses its standard registration, averaging, and
// triangulation algorithms and native bundle adjustment unless a setter replaces them. Backends
// supplied for an engine other than the configured one are rejected when the run starts.
impl Runner<'_> {
    /// Reconstruct an owned workspace with the runner's engine policy as given.
    ///
    /// Incremental releases matching data before reconstruction because it consumes only tracks.
    /// An observed run retains one pristine snapshot so an error can emit a terminal rollback
    /// event.
    #[cfg(feature = "unstable-internals")]
    pub fn run_workspace(self, prepared: SfmWorkspace) -> Result<(SfmWorkspace, RunReport)> {
        self.run_owned(prepared)
    }

    /// [`Self::run_workspace`], also the stable path behind [`Self::run_prepared_owned`].
    pub(crate) fn run_owned(self, prepared: SfmWorkspace) -> Result<(SfmWorkspace, RunReport)> {
        let observed = self.observer.is_some();
        self.dispatch(|engine, bundle_adjustment, events| {
            let rollback = observed.then(|| prepared.clone());
            match engine {
                EngineRun::Incremental(config, backends) => run_incremental_from_owned_with_events(
                    prepared,
                    config,
                    backends,
                    bundle_adjustment,
                    events,
                    rollback.as_ref(),
                ),
                EngineRun::Global(config, backends) => run_global_from_owned_with_events(
                    prepared,
                    config,
                    backends,
                    bundle_adjustment,
                    events,
                    rollback.as_ref(),
                ),
            }
        })
    }

    /// Reconstruct a workspace in place, retaining its frontend matching data.
    ///
    /// A failed global run leaves the prepared workspace unchanged.
    #[cfg(feature = "unstable-internals")]
    pub fn run_in_place(self, workspace: &mut SfmWorkspace) -> Result<RunReport> {
        self.dispatch(|engine, bundle_adjustment, events| match engine {
            EngineRun::Incremental(config, backends) => run_incremental_in_place_with_events(
                workspace,
                config,
                backends,
                bundle_adjustment,
                events,
            ),
            EngineRun::Global(config, backends) => run_global_in_place_with_events(
                workspace,
                config,
                backends,
                bundle_adjustment,
                events,
            ),
        })
    }

    /// Validate the configured engine, resolve its default algorithms, and run it.
    fn dispatch<T>(
        self,
        run: impl FnOnce(
            EngineRun<'_>,
            &dyn BundleAdjustmentBackend,
            &mut EventEmitter<'_>,
        ) -> Result<T>,
    ) -> Result<T> {
        let Self {
            config,
            bundle_adjuster,
            #[cfg(feature = "unstable-test-support")]
            bundle_adjustment_backend,
            incremental_backends,
            global_backends,
            observer,
            progress,
        } = self;
        let mut native = NativeBundleAdjuster::new();
        let adjuster = DirectBundleAdjustmentBackend::new(match bundle_adjuster {
            Some(adjuster) => adjuster,
            None => &mut native,
        });
        #[cfg(feature = "unstable-test-support")]
        let bundle_adjustment = bundle_adjustment_backend.unwrap_or(&adjuster);
        #[cfg(not(feature = "unstable-test-support"))]
        let bundle_adjustment: &dyn BundleAdjustmentBackend = &adjuster;
        let mut events = observer
            .map_or_else(EventEmitter::disabled, EventEmitter::observing)
            .with_progress(progress);
        match config.engine() {
            SfmEngine::Incremental => {
                if global_backends.is_some() {
                    return Err(SfmError::InvalidConfig(
                        "global backends cannot run the incremental engine".to_owned(),
                    ));
                }
                config.incremental().validate()?;
                let defaults;
                let backends = match incremental_backends {
                    Some(backends) => backends,
                    None => {
                        defaults = (
                            GenRansacP4pfRegistrar::default(),
                            TrackTriangulatorAdapter::default(),
                        );
                        IncrementalBackends::new(&defaults.0, &defaults.1)
                    }
                };
                run(
                    EngineRun::Incremental(config.incremental(), backends),
                    bundle_adjustment,
                    &mut events,
                )
            }
            SfmEngine::Global => {
                if incremental_backends.is_some() {
                    return Err(SfmError::InvalidConfig(
                        "incremental backends cannot run the global engine".to_owned(),
                    ));
                }
                config.global().validate()?;
                let defaults;
                let backends = match global_backends {
                    Some(backends) => backends,
                    None => {
                        defaults = (
                            TrackTriangulatorAdapter::for_averaged_poses()?,
                            TrackTriangulatorAdapter::default(),
                        );
                        GlobalBackends::new(
                            average_rotations,
                            average_translations,
                            &defaults.0,
                            &defaults.1,
                        )
                    }
                };
                run(
                    EngineRun::Global(config.global(), backends),
                    bundle_adjustment,
                    &mut events,
                )
            }
        }
    }
}

fn incremental_report_config(config: &IncrementalSfmConfig) -> SfmConfig {
    let mut report_config = SfmConfig::for_engine(SfmEngine::Incremental);
    report_config.set_incremental(config.clone());
    report_config
}

fn global_report_config(config: &GlobalSfmConfig) -> SfmConfig {
    let mut report_config = SfmConfig::for_engine(SfmEngine::Global);
    report_config.set_global(config.clone());
    report_config
}

fn run_incremental_from_owned_with_events(
    mut prepared: SfmWorkspace,
    config: &IncrementalSfmConfig,
    backends: IncrementalBackends<'_>,
    bundle_adjustment: &dyn BundleAdjustmentBackend,
    events: &mut EventEmitter<'_>,
    rollback: Option<&SfmWorkspace>,
) -> Result<(SfmWorkspace, RunReport)> {
    let engine = SfmEngine::Incremental;
    events.emit(
        SfmEventBranch::Main,
        SfmEventPhase::Run { engine },
        SfmEventTransition::Started,
        &prepared,
    );
    if let Err(error) = prepared.release_matching_data() {
        let event_workspace = rollback.unwrap_or(&prepared);
        events.emit(
            SfmEventBranch::Main,
            SfmEventPhase::Run { engine },
            SfmEventTransition::RolledBack,
            event_workspace,
        );
        return Err(error);
    }
    let report = match incremental_sfm_pretracked_with_events(
        &mut prepared,
        config,
        backends.registrar,
        backends.triangulator,
        Some(bundle_adjustment),
        events,
    ) {
        Ok(report) => report,
        Err(error) => {
            let event_workspace = rollback.unwrap_or(&prepared);
            events.emit(
                SfmEventBranch::Main,
                SfmEventPhase::Run { engine },
                SfmEventTransition::RolledBack,
                event_workspace,
            );
            return Err(error);
        }
    };
    let report = RunReport::try_new(
        &prepared,
        incremental_report_config(config),
        SfmOutcome::Incremental {
            report: Box::new(report),
        },
    )?;
    events.emit(
        SfmEventBranch::Main,
        SfmEventPhase::Run { engine },
        SfmEventTransition::Completed,
        &prepared,
    );
    Ok((prepared, report))
}

#[cfg(feature = "unstable-internals")]
fn run_incremental_in_place_with_events(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    backends: IncrementalBackends<'_>,
    bundle_adjustment: &dyn BundleAdjustmentBackend,
    events: &mut EventEmitter<'_>,
) -> Result<RunReport> {
    let engine = SfmEngine::Incremental;
    events.emit(
        SfmEventBranch::Main,
        SfmEventPhase::Run { engine },
        SfmEventTransition::Started,
        workspace,
    );
    let report = match incremental_sfm_pretracked_with_events(
        workspace,
        config,
        backends.registrar,
        backends.triangulator,
        Some(bundle_adjustment),
        events,
    ) {
        Ok(report) => report,
        Err(error) => {
            events.emit(
                SfmEventBranch::Main,
                SfmEventPhase::Run { engine },
                SfmEventTransition::RolledBack,
                workspace,
            );
            return Err(error);
        }
    };
    let report = RunReport::try_new(
        workspace,
        incremental_report_config(config),
        SfmOutcome::Incremental {
            report: Box::new(report),
        },
    )?;
    events.emit(
        SfmEventBranch::Main,
        SfmEventPhase::Run { engine },
        SfmEventTransition::Completed,
        workspace,
    );
    Ok(report)
}

fn run_global_from_owned_with_events(
    prepared: SfmWorkspace,
    config: &GlobalSfmConfig,
    backends: GlobalBackends<'_>,
    bundle_adjustment: &dyn BundleAdjustmentBackend,
    events: &mut EventEmitter<'_>,
    rollback: Option<&SfmWorkspace>,
) -> Result<(SfmWorkspace, RunReport)> {
    let engine = SfmEngine::Global;
    events.emit(
        SfmEventBranch::Main,
        SfmEventPhase::Run { engine },
        SfmEventTransition::Started,
        &prepared,
    );
    match run_global_branch(prepared, config, backends, bundle_adjustment, events) {
        Ok(result) => {
            events.emit(
                SfmEventBranch::Main,
                SfmEventPhase::Run { engine },
                SfmEventTransition::Completed,
                &result.0,
            );
            Ok(result)
        }
        Err(error) => {
            if let Some(rollback) = rollback {
                events.emit(
                    SfmEventBranch::Main,
                    SfmEventPhase::Run { engine },
                    SfmEventTransition::RolledBack,
                    rollback,
                );
            }
            Err(error)
        }
    }
}

#[cfg(feature = "unstable-internals")]
fn run_global_in_place_with_events(
    workspace: &mut SfmWorkspace,
    config: &GlobalSfmConfig,
    backends: GlobalBackends<'_>,
    bundle_adjustment: &dyn BundleAdjustmentBackend,
    events: &mut EventEmitter<'_>,
) -> Result<RunReport> {
    let engine = SfmEngine::Global;
    events.emit(
        SfmEventBranch::Main,
        SfmEventPhase::Run { engine },
        SfmEventTransition::Started,
        workspace,
    );
    match run_global_branch(
        workspace.clone(),
        config,
        backends,
        bundle_adjustment,
        events,
    ) {
        Ok((completed, report)) => {
            *workspace = completed;
            events.emit(
                SfmEventBranch::Main,
                SfmEventPhase::Run { engine },
                SfmEventTransition::Completed,
                workspace,
            );
            Ok(report)
        }
        Err(error) => {
            events.emit(
                SfmEventBranch::Main,
                SfmEventPhase::Run { engine },
                SfmEventTransition::RolledBack,
                workspace,
            );
            Err(error)
        }
    }
}

fn run_global_branch(
    prepared: SfmWorkspace,
    config: &GlobalSfmConfig,
    backends: GlobalBackends<'_>,
    bundle_adjustment: &dyn BundleAdjustmentBackend,
    events: &mut EventEmitter<'_>,
) -> Result<(SfmWorkspace, RunReport)> {
    let mut global_prepared = prepared.clone();
    let progress = events.progress();
    cancellation_point(progress)?;
    progress.begin(SfmProgressStage::RotationAveraging, 1);
    let PreparedGlobal {
        report: preparation,
        pair_state,
        rotations: rotation_preparation,
    } = prepare_global(&mut global_prepared, backends.rotation, config)?;
    progress.advance(SfmProgressStage::RotationAveraging, 1);
    progress.finish(SfmProgressStage::RotationAveraging);
    events.emit(
        SfmEventBranch::GlobalPreparation,
        SfmEventPhase::GlobalPreparation,
        SfmEventTransition::Checkpoint,
        &global_prepared,
    );
    let rebuild = |fallback: &SfmWorkspace| {
        let mut rebuilt = fallback.clone();
        apply_pair_conditioning_state(&mut rebuilt, &pair_state)?;
        Ok(rebuilt)
    };
    let adaptive_runtime = GlobalRuntime {
        direction_translation: backends.translation,
        blind_triangulator: backends.bootstrap_triangulator,
        robust_triangulator: backends.robust_triangulator,
        bundle_adjustment,
    };
    let (completed, adaptive_report) = global_core_from_owned_prepared_rotations_with_events(
        prepared,
        global_prepared,
        config,
        &rotation_preparation,
        adaptive_runtime,
        rebuild,
        events,
    )?;
    completed.validate()?;
    let result = match adaptive_report.completion() {
        GlobalCompletionReport::Reconstructed { .. } => GlobalOutcome::Reconstructed {
            report: Box::new(adaptive_report),
        },
        GlobalCompletionReport::NoReconstruction { .. } => GlobalOutcome::NoReconstruction {
            report: Box::new(adaptive_report),
        },
    };
    let outcome = SfmOutcome::Global {
        preparation,
        result,
    };
    let report = RunReport::try_new(&completed, global_report_config(config), outcome)?;
    // Candidate failures are recorded rather than raised, so a canceled candidate surfaces here.
    cancellation_point(progress)?;
    Ok((completed, report))
}
