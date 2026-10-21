// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progress reporting and cancellation for the front end and the engines.
//!
//! Implement [`SfmProgressSink`], or, with the `console-progress` feature, use
//! `ConsoleProgress` for terminal progress bars, and pass it to `Frontend::build_with_report`
//! and [`Runner::progress`](crate::Runner::progress).
//!
//! Progress is write-only apart from cancellation: the library reads nothing back but
//! [`SfmProgressSink::is_canceled`], which can only end a run with
//! [`SfmError::Canceled`](crate::SfmError::Canceled), so every run that completes produces the
//! same reconstruction with or without a sink. Unlike `SfmEventSink`, which observes curated
//! reconstruction state through `&mut self`, progress is also reported from inside the parallel
//! front-end loops, so sinks take `&self` and must be [`Sync`].
//!
//! The library keeps no wall clock; a sink that needs stage durations measures them between
//! [`SfmProgressSink::begin`] and [`SfmProgressSink::finish`].

#[cfg(feature = "console-progress")]
mod console;

#[cfg(feature = "console-progress")]
pub use console::ConsoleProgress;

/// Front-end or reconstruction stage whose progress is being reported.
///
/// The Incremental engine reports [`Self::InitialPair`], [`Self::Registration`], and
/// [`Self::BundleAdjustment`]. The Global engine reports [`Self::RotationAveraging`] once, then
/// [`Self::TranslationAveraging`], [`Self::Triangulation`], and [`Self::BundleAdjustment`] for
/// every translation candidate it tries, and [`Self::SelfCalibration`] for the admitted one.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SfmProgressStage {
    /// Feature detection and description, counted in images.
    FeatureExtraction,
    /// Per-image matcher state, such as approximate nearest-neighbor indexes, built once before
    /// pair matching; counted in images. Only matchers that prepare per-image state report it,
    /// after the matching operation starts and before [`Self::Matching`] begins.
    MatcherPreparation,
    /// Descriptor matching, counted in image pairs.
    Matching,
    /// Robust two-view geometric verification, counted in image pairs.
    Verification,
    /// Radial distortion estimation for camera groups whose distortion is unknown, counted in
    /// camera groups; it re-verifies the pairs incident to every group it re-models.
    RadialCalibration,
    /// Merging the verified pair matches into multi-view tracks, counted in image pairs.
    TrackBuilding,
    /// Two-view initialization of an Incremental reconstruction, counted as one initial pair.
    InitialPair,
    /// Incremental growth, counted in the views still unregistered when it starts. Each view
    /// counts when it is first registered, before any bundle adjustment that follows, and the
    /// views growth gives up on are completed when it ends.
    Registration,
    /// Global rotation averaging, including the removal of pairs whose rotation disagrees,
    /// counted as one solve.
    RotationAveraging,
    /// Global translation averaging or positioning for one candidate, counted as one solve.
    TranslationAveraging,
    /// Global landmark triangulation, counted in tracks.
    Triangulation,
    /// Final bundle adjustment, counted in solves. The Global engine plans several solves (first
    /// with the rotations held, then with everything free, then robust rounds) and counts a
    /// round it rolls back or skips as done; the Incremental engine counts its whole final
    /// refinement, with its several solves and cleanups, as one item.
    BundleAdjustment,
    /// Global camera self-calibration of the admitted candidate, counted as one solve.
    SelfCalibration,
}

/// Receiver for front-end and reconstruction progress.
///
/// For each stage, [`Self::begin`] happens-before every [`Self::advance`] of that stage, which
/// happen-before [`Self::finish`]. Stages never overlap, but a stage repeats when its operation is
/// called again. Each successful operation emits one `begin`/`finish` pair for its stage, even when
/// it selects no items, and the `advance` deltas between them sum to the `begin` total. A failed
/// operation abandons its stage without calling `finish`.
///
/// Any method may run on any thread, including Rayon workers. Implementations must be cheap,
/// non-blocking, and must not panic. Every method defaults to a no-op that never cancels, and `()`
/// is the sink that ignores everything.
pub trait SfmProgressSink: Sync {
    /// A stage starts with `total` items to process.
    fn begin(&self, stage: SfmProgressStage, total: usize) {
        let _ = (stage, total);
    }

    /// `completed` more items of `stage` finished successfully.
    fn advance(&self, stage: SfmProgressStage, completed: usize) {
        let _ = (stage, completed);
    }

    /// Every item of `stage` finished and its results were committed.
    fn finish(&self, stage: SfmProgressStage) {
        let _ = stage;
    }

    /// Whether the run should stop. Once it returns `true` it must keep returning `true`.
    ///
    /// The library polls it before every image and pair of the front end, before each round of
    /// radial calibration, before every registration, Global translation candidate, and bundle
    /// adjustment, and once more before an operation returns. A canceled operation fails with
    /// [`SfmError::Canceled`](crate::SfmError::Canceled) and returns no partial result; a solve
    /// already running, such as one bundle adjustment, finishes first.
    fn is_canceled(&self) -> bool {
        false
    }
}

impl SfmProgressSink for () {}

/// Fail with [`SfmError::Canceled`](crate::SfmError::Canceled) once `progress` asks the run to
/// stop.
pub(crate) fn cancellation_point(progress: &dyn SfmProgressSink) -> crate::Result<()> {
    if progress.is_canceled() {
        Err(crate::SfmError::Canceled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use super::{SfmProgressSink, SfmProgressStage};
    use std::sync::Mutex;

    /// Progress notification captured by [`RecordingProgress`]; `Advance` deltas are merged per
    /// stage because their thread interleaving is not deterministic.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum ProgressRecord {
        Begin(SfmProgressStage, usize),
        Advanced(SfmProgressStage, usize),
        Finish(SfmProgressStage),
    }

    #[derive(Debug, Default)]
    pub(crate) struct RecordingProgress(Mutex<Vec<ProgressRecord>>);

    impl RecordingProgress {
        /// Recorded notifications, with each run of consecutive `advance` calls for one stage
        /// collapsed into a single `Advanced` total.
        pub(crate) fn records(&self) -> Vec<ProgressRecord> {
            let calls = self.0.lock().unwrap();
            let mut records: Vec<ProgressRecord> = Vec::new();
            for &call in calls.iter() {
                match (records.last_mut(), call) {
                    (
                        Some(ProgressRecord::Advanced(stage, total)),
                        ProgressRecord::Advanced(next_stage, completed),
                    ) if *stage == next_stage => *total += completed,
                    _ => records.push(call),
                }
            }
            records
        }

        /// Number of notifications received, before any merging.
        #[cfg(feature = "image-io")]
        pub(crate) fn notification_count(&self) -> usize {
            self.0.lock().unwrap().len()
        }
    }

    /// [`RecordingProgress`] that asks the run to stop once it has received `limit`
    /// notifications.
    #[cfg(feature = "image-io")]
    #[derive(Debug, Default)]
    pub(crate) struct CancelAfter {
        limit: usize,
        progress: RecordingProgress,
    }

    #[cfg(feature = "image-io")]
    impl CancelAfter {
        pub(crate) fn new(limit: usize) -> Self {
            Self {
                limit,
                progress: RecordingProgress::default(),
            }
        }

        /// Whether any stage began after the run was asked to stop.
        pub(crate) fn began_after_cancellation(&self) -> bool {
            let calls = self.progress.0.lock().unwrap();
            calls[self.limit.min(calls.len())..]
                .iter()
                .any(|call| matches!(call, ProgressRecord::Begin(..)))
        }
    }

    #[cfg(feature = "image-io")]
    impl SfmProgressSink for CancelAfter {
        fn begin(&self, stage: SfmProgressStage, total: usize) {
            self.progress.begin(stage, total);
        }

        fn advance(&self, stage: SfmProgressStage, completed: usize) {
            self.progress.advance(stage, completed);
        }

        fn finish(&self, stage: SfmProgressStage) {
            self.progress.finish(stage);
        }

        fn is_canceled(&self) -> bool {
            self.progress.0.lock().unwrap().len() >= self.limit
        }
    }

    impl SfmProgressSink for RecordingProgress {
        fn begin(&self, stage: SfmProgressStage, total: usize) {
            self.0
                .lock()
                .unwrap()
                .push(ProgressRecord::Begin(stage, total));
        }

        fn advance(&self, stage: SfmProgressStage, completed: usize) {
            self.0
                .lock()
                .unwrap()
                .push(ProgressRecord::Advanced(stage, completed));
        }

        fn finish(&self, stage: SfmProgressStage) {
            self.0.lock().unwrap().push(ProgressRecord::Finish(stage));
        }
    }
}
