// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Follow the front end and an engine run stage by stage, and cancel a run cleanly.
//!
//! You will learn: how to implement `SfmProgressSink`, pass it to the front end and to an engine
//! through its `Runner`, and stop a run from the sink: the run fails with `SfmError::Canceled`
//! and returns nothing partial, and the prepared view graph stays usable for the next run.
//! Data: the Sceaux Castle images (11 photographs and their `K.txt` calibration), which
//! `pixi run examples-real` fetches; any folder of overlapping photographs works.
//! Run: cargo run --release -p openmvg-sfm --example progress_and_cancellation --features
//! image-io -- <images>
//! Expected: the sink sees every front-end stage begin and finish; on Sceaux Castle an
//! Incremental run canceled after its first registration fails with `Canceled`, and the next run
//! on the same graph registers all 11 views.
//! Next: with the `console-progress` feature, `ConsoleProgress` draws these stages as terminal
//! progress bars; the `engines_and_reports` example saves the graph to a file.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use openmvg_sfm::progress::{SfmProgressSink, SfmProgressStage};
use openmvg_sfm::{Dataset, Frontend, Incremental, SfmError};

/// A sink that logs each stage, counts registrations, and can ask the run to stop.
struct Log {
    stages: Mutex<Vec<(SfmProgressStage, usize, bool)>>,
    registered: AtomicUsize,
    /// Cancel once this many views are registered; `None` never cancels.
    stop_after: Option<usize>,
}

impl Log {
    fn new(stop_after: Option<usize>) -> Self {
        let (stages, registered) = (Mutex::new(Vec::new()), AtomicUsize::new(0));
        Self {
            stages,
            registered,
            stop_after,
        }
    }

    fn stages(&self) -> Vec<(SfmProgressStage, usize, bool)> {
        self.stages
            .lock()
            .map(|stages| stages.clone())
            .unwrap_or_default()
    }
}

// Progress is reported from worker threads too, hence `&self` and the atomic counter.
impl SfmProgressSink for Log {
    fn begin(&self, stage: SfmProgressStage, total: usize) {
        if let Ok(mut stages) = self.stages.lock() {
            stages.push((stage, total, false));
        }
    }

    fn advance(&self, stage: SfmProgressStage, completed: usize) {
        if stage == SfmProgressStage::Registration {
            self.registered.fetch_add(completed, Ordering::SeqCst);
        }
    }

    fn finish(&self, stage: SfmProgressStage) {
        if let Ok(mut stages) = self.stages.lock()
            && let Some(entry) = stages.iter_mut().rev().find(|entry| entry.0 == stage)
        {
            entry.2 = true;
        }
    }

    fn is_canceled(&self) -> bool {
        let registered = self.registered.load(Ordering::SeqCst);
        self.stop_after.is_some_and(|limit| registered >= limit)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let images = std::env::args_os()
        .nth(1)
        .ok_or("usage: progress_and_cancellation <image-directory>")?;
    let dataset = Dataset::from_dir(images)?;

    // The front end reports each stage with its item count: images, pairs, or camera groups.
    let log = Log::new(None);
    let (graph, _report) = Frontend::default().build_with_report(&dataset, &log)?;
    for (stage, total, finished) in log.stages() {
        println!("front end: {stage:?} over {total} items, finished: {finished}");
        assert!(finished, "every stage that begins also finishes");
    }

    // Cancel after the first registered view: the run stops at its next check and returns
    // the error, with no partial reconstruction.
    let cancel = Log::new(Some(1));
    let canceled = Incremental::default()
        .runner()
        .progress(&cancel)
        .run_prepared(&graph);
    assert!(matches!(canceled, Err(SfmError::Canceled)), "{canceled:?}");
    println!(
        "canceled after {} registration",
        cancel.registered.load(Ordering::SeqCst)
    );

    // The graph is immutable, so the next run starts from it unchanged and goes to the end.
    let log = Log::new(None);
    let output = Incremental::default()
        .runner()
        .progress(&log)
        .run_prepared(&graph)?;
    let registered = output.reconstruction().registered_view_count();
    println!(
        "the next run registered {registered} of {} views",
        dataset.views().len()
    );
    assert!(
        registered >= 2,
        "a reconstruction has at least its initial pair"
    );
    Ok(())
}
