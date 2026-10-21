// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reconstruction events for live visualization; observing a run does not change its result.
//!
//! At fixed points of a run, such as after a view registers, an event lends the sink a
//! read-only view of the reconstruction. Events own no copy of the state, expose nothing
//! mutable, and cannot change what the engine does next.
//!
//! Front-end and reconstruction progress is reported separately through the thread-safe
//! [`SfmProgressSink`].

use crate::model::ViewRecord;
pub use crate::pipelines::global::GlobalCandidate;
use crate::progress::SfmProgressSink;
use crate::workflow::run::SfmEngine;
use crate::{
    Camera, IntrinsicId, Result, SfmWorkspace, Track, TrackId, TrackObservation,
    TrackObservationIndex, TrackSet, ViewId, ViewPair, WorldToCameraPose,
};
use nalgebra::{Point2, Point3, Rotation3};

/// Coarse event category used by sinks to disable work before an event is constructed.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfmEventKey {
    /// Engine run start, completion, and rollback ([`SfmEventPhase::Run`]).
    Run,
    /// Global view-graph preparation (rotation averaging).
    Graph,
    /// Global positioning-candidate stages: translation, structure preparation, and candidate
    /// admission.
    GlobalCandidate,
    /// Global bundle-adjustment stages: fixed-rotation BA, cleanup, full BA, robust rounds, and
    /// self-calibration.
    BundleAdjustment,
    /// Every incremental-engine stage.
    Incremental,
}

/// Reconstruction branch represented by an event's snapshot.
///
/// Candidate branches are separate copies of the reconstruction until a later event selects
/// one as the run result.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfmEventBranch {
    /// The reconstruction that becomes the run result.
    Main,
    /// The Global engine's preparation copy, before any positioning candidate is tried.
    GlobalPreparation,
    /// One isolated global positioning attempt.
    GlobalCandidate {
        /// Zero-based attempt number within the run.
        attempt: usize,
        /// Positioning method tried by this attempt.
        candidate: GlobalCandidate,
    },
}

/// Deterministic orchestration checkpoint.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfmEventPhase {
    /// The whole engine run.
    Run {
        /// Engine executing the run.
        engine: SfmEngine,
    },
    /// Global rotation averaging over the verified view graph has finished.
    GlobalPreparation,
    /// A candidate's camera translations (and its initial landmarks) have been estimated.
    GlobalTranslationAveraging,
    /// A candidate's landmark structure for fixed-rotation bundle adjustment has been prepared.
    GlobalTranslationCore,
    /// Bundle adjustment with camera rotations held fixed.
    GlobalFixedRotation,
    /// Landmark and pose-support cleanup after fixed-rotation bundle adjustment.
    GlobalPostFixedRotationCleanup,
    /// Bundle adjustment over every pose and landmark.
    GlobalBundleAdjustment,
    /// One robust retriangulation, full bundle adjustment, and outlier-cleanup round.
    GlobalRobustRound {
        /// Zero-based round number.
        round: usize,
    },
    /// Global intrinsic self-calibration.
    GlobalSelfCalibration,
    /// The decision to keep or discard a positioning candidate.
    GlobalCandidateAdmission,
    /// The Incremental engine's initial pair has been initialized.
    IncrementalInitialPair,
    /// One view has been registered into the Incremental reconstruction.
    IncrementalRegistration {
        /// Zero-based count of registrations committed earlier in the run.
        index: usize,
    },
    /// One periodic full bundle adjustment during incremental growth.
    IncrementalBundleAdjustment {
        /// Zero-based count of periodic bundle adjustments attempted earlier in the run.
        round: usize,
    },
    /// The final incremental bundle adjustment.
    IncrementalFinalBundleAdjustment,
    /// The finished Incremental reconstruction.
    IncrementalFinal,
}

impl SfmEventPhase {
    /// Category a sink can filter this phase by through [`SfmEventSink::enabled`].
    #[must_use]
    pub const fn key(self) -> SfmEventKey {
        match self {
            Self::Run { .. } => SfmEventKey::Run,
            Self::GlobalPreparation => SfmEventKey::Graph,
            Self::GlobalTranslationAveraging
            | Self::GlobalTranslationCore
            | Self::GlobalCandidateAdmission => SfmEventKey::GlobalCandidate,
            Self::GlobalFixedRotation
            | Self::GlobalPostFixedRotationCleanup
            | Self::GlobalBundleAdjustment
            | Self::GlobalRobustRound { .. }
            | Self::GlobalSelfCalibration => SfmEventKey::BundleAdjustment,
            Self::IncrementalInitialPair
            | Self::IncrementalRegistration { .. }
            | Self::IncrementalBundleAdjustment { .. }
            | Self::IncrementalFinalBundleAdjustment
            | Self::IncrementalFinal => SfmEventKey::Incremental,
        }
    }
}

/// Meaning of the snapshot supplied with an event.
///
/// `Committed` is relative to the named branch. A committed global-candidate state remains
/// isolated from [`SfmEventBranch::Main`] until it is selected or discarded.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfmEventTransition {
    /// The phase begins; the snapshot is its input state.
    Started,
    /// The phase reached an intermediate state that carries no accept/reject decision.
    Checkpoint,
    /// The phase produced a result that has not yet passed its acceptance checks.
    Tentative,
    /// The phase's result was accepted into its branch.
    Committed,
    /// The phase's result was rejected or it failed; the snapshot is the restored prior state.
    RolledBack,
    /// A global candidate branch was abandoned.
    Discarded,
    /// A global candidate branch was chosen as the run result.
    Selected,
    /// The phase finished successfully; the snapshot is its final state.
    Completed,
}

/// Read-only, behavior-neutral view of live reconstruction state.
///
/// The snapshot borrows the engine's live state but exposes only the stable identities and
/// observation queries that a visualization needs.
#[derive(Clone, Copy)]
pub struct SfmSnapshot<'a> {
    #[cfg_attr(not(feature = "observer"), allow(dead_code))]
    workspace: &'a SfmWorkspace,
}

impl std::fmt::Debug for SfmSnapshot<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The borrowed workspace is engine-internal; keep its contents out of debug output.
        f.debug_struct("SfmSnapshot").finish_non_exhaustive()
    }
}

#[cfg_attr(not(feature = "observer"), allow(dead_code))]
impl<'a> SfmSnapshot<'a> {
    pub(crate) const fn new(workspace: &'a SfmWorkspace) -> Self {
        Self { workspace }
    }

    /// Builds a snapshot for in-tree unstable tooling and tests.
    #[cfg(feature = "unstable-internals")]
    #[must_use]
    pub const fn from_workspace(workspace: &'a SfmWorkspace) -> Self {
        Self::new(workspace)
    }

    /// Every view record, in view-ID order.
    #[must_use]
    pub fn views(self) -> &'a [ViewRecord] {
        self.workspace.views()
    }

    /// Record of `view`, or `None` if the ID is unknown.
    #[must_use]
    pub fn view(self, view: ViewId) -> Option<&'a ViewRecord> {
        self.workspace.view(view)
    }

    /// Current camera model of `intrinsic`, or `None` if the ID is unknown.
    #[must_use]
    pub fn camera_model(self, intrinsic: IntrinsicId) -> Option<&'a Camera> {
        self.workspace.camera_model(intrinsic)
    }

    /// Every candidate view pair, active or not.
    #[must_use]
    pub fn pairs(self) -> &'a [ViewPair] {
        self.workspace.pair_graph().pairs()
    }

    /// Whether `pair` is a known pair that has not been deactivated.
    #[must_use]
    pub fn is_pair_active(self, pair: ViewPair) -> bool {
        self.workspace.pair_graph().is_active_pair(pair)
    }

    /// Whether `pair` has an accepted two-view geometric model.
    #[must_use]
    pub fn is_pair_verified(self, pair: ViewPair) -> bool {
        self.workspace.geometric_models_view().get(pair).is_some()
    }

    /// Number of pairs with an accepted two-view geometric model.
    #[must_use]
    pub fn verified_pair_count(self) -> usize {
        self.workspace.geometric_models_view().populated_count()
    }

    /// Number of views that hold extracted keypoints (not the number of keypoints).
    #[must_use]
    pub fn extracted_view_count(self) -> usize {
        self.workspace.keypoint_store().populated_count()
    }

    /// Number of registered pose blocks; views sharing a pose count once.
    #[must_use]
    pub fn registered_pose_count(self) -> usize {
        self.workspace.pose_store().populated_count()
    }

    /// Whether descriptors and raw matches have been released ahead of reconstruction.
    #[must_use]
    pub const fn is_matching_data_released(self) -> bool {
        self.workspace.is_matching_data_released()
    }

    /// Relative rotation of `pair`'s verified model, including rotation-only models; `None` if
    /// the pair is unverified or its model carries no rotation.
    #[must_use]
    pub fn pair_relative_rotation(self, pair: ViewPair) -> Option<&'a Rotation3<f64>> {
        self.workspace
            .geometric_models_view()
            .get(pair)?
            .relative_rotation()
    }

    /// Relative pose (rotation and unit-scale translation) of `pair`'s verified model; `None` if
    /// the pair is unverified or its model has no translation (rotation-only).
    #[must_use]
    pub fn pair_relative_pose(self, pair: ViewPair) -> Option<&'a crate::RelativePose> {
        self.workspace
            .geometric_models_view()
            .get(pair)?
            .relative_pose()
    }

    /// Every multi-view feature track.
    #[must_use]
    pub fn tracks(self) -> &'a TrackSet {
        self.workspace.tracks()
    }

    /// World-to-camera pose of `view`, or `None` if the view is unknown or not registered.
    #[must_use]
    pub fn pose(self, view: ViewId) -> Option<&'a WorldToCameraPose> {
        self.workspace.pose(view).map(|record| &record.pose)
    }

    /// Whether `view`'s assigned pose has been registered.
    #[must_use]
    pub fn has_pose(self, view: ViewId) -> bool {
        self.workspace.has_pose(view)
    }

    /// Number of views whose assigned pose has been registered.
    #[must_use]
    pub fn registered_view_count(self) -> usize {
        self.workspace.registered_view_count()
    }

    /// Registered views and their world-to-camera poses, in view-ID order.
    pub fn registered_views(self) -> impl Iterator<Item = (ViewId, &'a WorldToCameraPose)> + 'a {
        self.workspace
            .registered_views()
            .map(|(view, record)| (view, &record.pose))
    }

    /// Number of landmarks currently active in the reconstruction.
    #[must_use]
    pub fn active_landmark_count(self) -> usize {
        self.workspace.landmarks().active_count()
    }

    /// Every active landmark.
    pub fn active_landmarks(self) -> impl Iterator<Item = SfmLandmarkSnapshot<'a>> + 'a {
        self.workspace
            .landmarks()
            .iter_active()
            .map(|landmark| SfmLandmarkSnapshot { landmark })
    }

    /// Pixel position of a track observation's feature.
    ///
    /// Fails if the observation's view has no feature geometry or its feature ID is out of range.
    pub fn track_observation_pixel(self, observation: &TrackObservation) -> Result<Point2<f64>> {
        self.workspace.track_observation_pixel(observation)
    }

    /// Exposes the underlying workspace only to explicitly unstable in-tree tooling.
    #[cfg(feature = "unstable-internals")]
    #[must_use]
    pub const fn unstable_workspace(self) -> &'a SfmWorkspace {
        self.workspace
    }
}

/// Read-only landmark data exposed to observation sinks.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "observer"), allow(dead_code))]
pub struct SfmLandmarkSnapshot<'a> {
    landmark: &'a crate::model::Landmark,
}

#[cfg_attr(not(feature = "observer"), allow(dead_code))]
impl<'a> SfmLandmarkSnapshot<'a> {
    /// Track the landmark was triangulated from.
    #[must_use]
    pub const fn source_track(self) -> TrackId {
        self.landmark.source_track
    }

    /// Landmark position in world coordinates.
    #[must_use]
    pub const fn point(self) -> &'a Point3<f64> {
        &self.landmark.point
    }

    /// Observations of `track` that currently support the landmark, with their index in the
    /// track. `track` must be the landmark's [`Self::source_track`].
    pub fn supporting_observations(
        self,
        track: &'a Track,
    ) -> impl Iterator<Item = (TrackObservationIndex, &'a TrackObservation)> + 'a {
        self.landmark.supporting_observations(track)
    }
}

/// One small, borrowed observation of live SfM state.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SfmEvent<'a> {
    /// Run-local emission counter, starting at 0 and increasing by one per emitted event
    /// (events a sink disabled still consume a number).
    #[cfg_attr(not(feature = "observer"), allow(dead_code))]
    pub sequence: u64,
    /// Reconstruction branch the snapshot belongs to.
    #[cfg_attr(not(feature = "observer"), allow(dead_code))]
    pub branch: SfmEventBranch,
    /// Orchestration stage that emitted the event.
    #[cfg_attr(not(feature = "observer"), allow(dead_code))]
    pub phase: SfmEventPhase,
    /// What happened to the stage's state.
    #[cfg_attr(not(feature = "observer"), allow(dead_code))]
    pub transition: SfmEventTransition,
    /// Borrowed view of the branch state at the event; valid only during the callback.
    #[cfg_attr(not(feature = "observer"), allow(dead_code))]
    pub snapshot: SfmSnapshot<'a>,
}

impl<'a> SfmEvent<'a> {
    /// Builds an event for in-tree unstable tooling and tests.
    #[cfg(feature = "unstable-internals")]
    #[must_use]
    pub const fn new(
        sequence: u64,
        branch: SfmEventBranch,
        phase: SfmEventPhase,
        transition: SfmEventTransition,
        snapshot: SfmSnapshot<'a>,
    ) -> Self {
        Self {
            sequence,
            branch,
            phase,
            transition,
            snapshot,
        }
    }
}

/// Synchronous sink for behavior-neutral SfM telemetry.
///
/// Implementations must not retain `event.snapshot` beyond [`Self::record`]. Expensive geometry
/// extraction, buffering, and I/O belong in the sink and can be skipped through [`Self::enabled`].
/// The callback has no control-flow return; implementations must not panic if observation should
/// remain behavior-neutral.
pub trait SfmEventSink {
    /// Whether events of category `key` should be delivered to [`Self::record`]; defaults to
    /// every category.
    fn enabled(&self, _key: SfmEventKey) -> bool {
        true
    }

    /// Receive one event.
    fn record(&mut self, event: SfmEvent<'_>);
}

impl<F> SfmEventSink for F
where
    F: for<'event> FnMut(SfmEvent<'event>),
{
    fn record(&mut self, event: SfmEvent<'_>) {
        self(event);
    }
}

/// Run-local sequencing and optional dispatch, carrying the run's progress sink to the stage
/// boundaries.
pub(crate) struct EventEmitter<'sink> {
    sink: Option<&'sink mut (dyn SfmEventSink + 'sink)>,
    progress: &'sink dyn SfmProgressSink,
    next_sequence: u64,
}

impl<'sink> EventEmitter<'sink> {
    pub(crate) const fn disabled() -> Self {
        Self {
            sink: None,
            progress: &(),
            next_sequence: 0,
        }
    }

    pub(crate) fn observing(sink: &'sink mut (dyn SfmEventSink + 'sink)) -> Self {
        Self {
            sink: Some(sink),
            progress: &(),
            next_sequence: 0,
        }
    }

    pub(crate) const fn with_progress(mut self, progress: &'sink dyn SfmProgressSink) -> Self {
        self.progress = progress;
        self
    }

    pub(crate) const fn progress(&self) -> &'sink dyn SfmProgressSink {
        self.progress
    }

    pub(crate) fn emit(
        &mut self,
        branch: SfmEventBranch,
        phase: SfmEventPhase,
        transition: SfmEventTransition,
        workspace: &SfmWorkspace,
    ) {
        let sequence = self.next_sequence;
        // Saturation keeps observation non-panicking even after the practically unreachable
        // limit of `u64::MAX` emitted events. The terminal value remains monotonic.
        self.next_sequence = self.next_sequence.saturating_add(1);
        let key = phase.key();
        let Some(sink) = self.sink.as_deref_mut() else {
            return;
        };
        if sink.enabled(key) {
            sink.record(SfmEvent {
                sequence,
                branch,
                phase,
                transition,
                snapshot: SfmSnapshot::new(workspace),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImageMeta, PoseRecord};
    use crate::{Camera, IntrinsicId, WorldToCameraPose};

    #[test]
    fn emitter_sequences_enabled_events_without_allocating_snapshots() {
        struct Sink(Vec<(u64, SfmEventPhase)>);

        impl SfmEventSink for Sink {
            fn enabled(&self, key: SfmEventKey) -> bool {
                key != SfmEventKey::Graph
            }

            fn record(&mut self, event: SfmEvent<'_>) {
                self.0.push((event.sequence, event.phase));
            }
        }

        let workspace = SfmWorkspace::builder().build().unwrap();
        let mut sink = Sink(Vec::new());
        {
            let mut events = EventEmitter::observing(&mut sink);
            events.emit(
                SfmEventBranch::Main,
                SfmEventPhase::Run {
                    engine: SfmEngine::Global,
                },
                SfmEventTransition::Started,
                &workspace,
            );
            events.emit(
                SfmEventBranch::GlobalPreparation,
                SfmEventPhase::GlobalPreparation,
                SfmEventTransition::Checkpoint,
                &workspace,
            );
            events.emit(
                SfmEventBranch::GlobalCandidate {
                    attempt: 0,
                    candidate: GlobalCandidate::TranslationAveraging,
                },
                SfmEventPhase::GlobalTranslationAveraging,
                SfmEventTransition::Checkpoint,
                &workspace,
            );
        }

        assert_eq!(
            sink.0,
            vec![
                (
                    0,
                    SfmEventPhase::Run {
                        engine: SfmEngine::Global
                    }
                ),
                (2, SfmEventPhase::GlobalTranslationAveraging),
            ]
        );
    }

    #[test]
    fn events_distinguish_tentative_candidate_from_restored_state() {
        let mut workspace = SfmWorkspace::builder()
            .image_meta([ImageMeta::external_id("image", 64, 48)])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        let snapshot = crate::ops::rollback::GeometryRollback::capture(&workspace);
        let branch = SfmEventBranch::GlobalCandidate {
            attempt: 1,
            candidate: GlobalCandidate::GlobalPositioning,
        };
        let mut observed = Vec::new();
        let mut sink = |event: SfmEvent<'_>| {
            observed.push((
                event.branch,
                event.transition,
                event.snapshot.registered_view_count(),
            ));
        };
        {
            let mut events = EventEmitter::observing(&mut sink);

            workspace
                .set_pose(PoseRecord {
                    id: crate::PoseId::new(0),
                    pose: WorldToCameraPose::identity(),
                    component: None,
                })
                .unwrap();
            events.emit(
                branch,
                SfmEventPhase::GlobalBundleAdjustment,
                SfmEventTransition::Tentative,
                &workspace,
            );
            snapshot.rollback(&mut workspace);
            events.emit(
                branch,
                SfmEventPhase::GlobalBundleAdjustment,
                SfmEventTransition::RolledBack,
                &workspace,
            );
        }

        assert_eq!(
            observed,
            vec![
                (branch, SfmEventTransition::Tentative, 1),
                (branch, SfmEventTransition::RolledBack, 0),
            ]
        );
    }
}
