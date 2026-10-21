// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Terminal progress bars for [`SfmProgressSink`], one per stage.

// Terminal progress bars for the SfM front end and reconstruction. Bars start hidden and are
// attached to a fresh draw target only when a stage begins with work to do, so stages that never
// run (matcher preparation for exact matchers) or select nothing leave no output, and a repeated
// stage, such as every stage of a retried global candidate, draws a new line instead of redrawing
// an earlier one.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressFinish, ProgressStyle};

use super::{SfmProgressSink, SfmProgressStage};

// The bar takes whatever width the label, counts, and timings leave, so a line never wraps.
const RUNNING_TEMPLATE: &str = "{prefix:>20.cyan.bold} [{wide_bar:.green/dim}] {pos:>7}/{len:<7} {msg} {elapsed:>4} (ETA {eta})";
const FINISHED_TEMPLATE: &str =
    "{prefix:>20.green.bold} [{wide_bar:.green/dim}] {pos:>7}/{len:<7} {msg} {elapsed:>4}";
const PROGRESS_CHARS: &str = "█▓▒░ ";
/// Redraw cadence, so elapsed time and ETA stay live while slow items (full-resolution feature
/// extraction) are in flight.
const STEADY_TICK: Duration = Duration::from_millis(250);

/// Display name and item unit of each stage, in pipeline order.
const STAGES: [(SfmProgressStage, &str, &str); 13] = [
    (
        SfmProgressStage::FeatureExtraction,
        "Extracting features",
        "images",
    ),
    (
        SfmProgressStage::MatcherPreparation,
        "Indexing descriptors",
        "images",
    ),
    (SfmProgressStage::Matching, "Matching pairs", "pairs"),
    (SfmProgressStage::Verification, "Verifying pairs", "pairs"),
    (
        SfmProgressStage::RadialCalibration,
        "Calibrating lenses",
        "groups",
    ),
    (SfmProgressStage::TrackBuilding, "Building tracks", "pairs"),
    (SfmProgressStage::InitialPair, "Initial pair", "pairs"),
    (SfmProgressStage::Registration, "Registering views", "views"),
    (
        SfmProgressStage::RotationAveraging,
        "Rotation averaging",
        "solves",
    ),
    (
        SfmProgressStage::TranslationAveraging,
        "Translation avg.",
        "solves",
    ),
    (SfmProgressStage::Triangulation, "Triangulating", "tracks"),
    (
        SfmProgressStage::BundleAdjustment,
        "Bundle adjustment",
        "solves",
    ),
    (
        SfmProgressStage::SelfCalibration,
        "Self-calibration",
        "solves",
    ),
];

/// Terminal progress bars on stderr, one per front-end and reconstruction stage.
///
/// Nothing is drawn unless stderr is a terminal, which keeps redirected logs clean. Stages that
/// never run or have no work leave no line, and a repeated stage, such as every stage of a
/// retried Global translation candidate, draws a new line. A stage abandoned by a failed or
/// canceled operation keeps its last position instead of showing complete.
///
/// Reporting from parallel workers costs little, and the bars redraw at a limited rate. The
/// sink never cancels.
///
/// ```no_run
/// use openmvg_sfm::progress::ConsoleProgress;
///
/// let progress = ConsoleProgress::stderr();
/// // Pass `&progress` to `Frontend::build_with_report` and `Runner::progress`.
/// # let _ = progress;
/// ```
#[derive(Debug)]
pub struct ConsoleProgress {
    target: fn() -> ProgressDrawTarget,
    bars: [ProgressBar; STAGES.len()],
    /// Whether each bar drew a line that no newline has ended yet. A drawn bar leaves the cursor
    /// at the end of its line, so the next output would overwrite or extend it.
    open_lines: [AtomicBool; STAGES.len()],
}

impl ConsoleProgress {
    /// Draw on stderr when it is a terminal.
    #[must_use]
    pub fn stderr() -> Self {
        Self::with_target(ProgressDrawTarget::stderr)
    }

    /// Draw each begun stage on a fresh target from `target`.
    #[must_use]
    fn with_target(target: fn() -> ProgressDrawTarget) -> Self {
        Self {
            target,
            bars: std::array::from_fn(|_| {
                ProgressBar::hidden().with_finish(ProgressFinish::Abandon)
            }),
            open_lines: std::array::from_fn(|_| AtomicBool::new(false)),
        }
    }

    fn stage(&self, stage: SfmProgressStage) -> Option<(usize, &'static str, &'static str)> {
        STAGES
            .iter()
            .position(|(known, _, _)| *known == stage)
            .map(|index| (index, STAGES[index].1, STAGES[index].2))
    }

    /// The bar of `stage`.
    #[cfg(test)]
    fn bar(&self, stage: SfmProgressStage) -> Option<&ProgressBar> {
        self.stage(stage).map(|(index, _, _)| &self.bars[index])
    }

    /// End the line of bar `index` if it drew one, so the next output starts on a fresh line.
    fn end_line(&self, index: usize) {
        if self.open_lines[index].swap(false, Ordering::Relaxed) {
            let _ = writeln!(std::io::stderr());
        }
    }
}

impl SfmProgressSink for ConsoleProgress {
    fn begin(&self, stage: SfmProgressStage, total: usize) {
        let Some((index, label, unit)) = self.stage(stage) else {
            return;
        };
        let bar = &self.bars[index];
        // Detach any earlier run first: reconfiguring draws, and must not redraw its old line.
        // Stages never overlap, so every other unfinished bar was abandoned by a failed operation;
        // detach those too, so that neither a later tick nor the final drop redraws them, and end
        // their lines so that this bar does not draw over them.
        for (other_index, other) in self.bars.iter().enumerate() {
            if !other.is_finished() {
                other.disable_steady_tick();
                other.set_draw_target(ProgressDrawTarget::hidden());
                self.end_line(other_index);
            }
        }
        bar.disable_steady_tick();
        bar.set_draw_target(ProgressDrawTarget::hidden());
        if total == 0 {
            return;
        }
        bar.set_style(style(RUNNING_TEMPLATE));
        bar.set_prefix(label);
        bar.set_message(unit);
        bar.set_length(total as u64);
        bar.reset();
        let target = (self.target)();
        let visible = !target.is_hidden();
        bar.set_draw_target(target);
        if visible {
            bar.tick();
            bar.enable_steady_tick(STEADY_TICK);
            self.open_lines[index].store(true, Ordering::Relaxed);
        }
    }

    fn advance(&self, stage: SfmProgressStage, completed: usize) {
        if let Some((index, _, _)) = self.stage(stage) {
            self.bars[index].inc(completed as u64);
        }
    }

    fn finish(&self, stage: SfmProgressStage) {
        let Some((index, _, _)) = self.stage(stage) else {
            return;
        };
        let bar = &self.bars[index];
        bar.disable_steady_tick();
        bar.set_style(style(FINISHED_TEMPLATE));
        bar.finish();
        // The finished line stays on screen: end it, so the next stage or the caller's output
        // does not overwrite it.
        self.end_line(index);
    }
}

fn style(template: &str) -> ProgressStyle {
    ProgressStyle::with_template(template)
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars(PROGRESS_CHARS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_parse() {
        for template in [RUNNING_TEMPLATE, FINISHED_TEMPLATE] {
            assert!(ProgressStyle::with_template(template).is_ok(), "{template}");
        }
    }

    #[test]
    fn every_stage_has_its_own_bar() {
        let progress = ConsoleProgress::with_target(ProgressDrawTarget::hidden);
        for (total, (stage, _, _)) in (2..).zip(STAGES) {
            progress.begin(stage, total);
            for _ in 0..total {
                progress.advance(stage, 1);
            }
            let Some(bar) = progress.bar(stage) else {
                panic!("{stage:?} has no bar");
            };
            assert_eq!(bar.length(), Some(total as u64));
            assert_eq!(bar.position(), total as u64);
        }
    }

    #[test]
    fn repeated_stage_restarts_its_bar() {
        let progress = ConsoleProgress::with_target(ProgressDrawTarget::hidden);
        let stage = SfmProgressStage::Matching;
        progress.begin(stage, 4);
        progress.advance(stage, 4);
        progress.finish(stage);
        progress.begin(stage, 2);
        progress.advance(stage, 1);

        let Some(bar) = progress.bar(stage) else {
            panic!("matching has no bar");
        };
        assert_eq!(bar.length(), Some(2));
        assert_eq!(bar.position(), 1);
    }

    #[test]
    fn abandoned_stage_keeps_its_position() {
        let progress = ConsoleProgress::with_target(ProgressDrawTarget::hidden);
        let stage = SfmProgressStage::Verification;
        progress.begin(stage, 5);
        progress.advance(stage, 2);

        let Some(bar) = progress.bar(stage) else {
            panic!("verification has no bar");
        };
        // Dropping an unfinished bar finishes it with its configured style.
        bar.finish_using_style();
        assert_eq!(bar.position(), 2);
    }

    /// A visible terminal that discards its output.
    #[derive(Debug)]
    struct DiscardedTerm;

    impl indicatif::TermLike for DiscardedTerm {
        fn width(&self) -> u16 {
            80
        }
        fn move_cursor_up(&self, _: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn move_cursor_down(&self, _: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn move_cursor_right(&self, _: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn move_cursor_left(&self, _: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn write_line(&self, _: &str) -> std::io::Result<()> {
            Ok(())
        }
        fn write_str(&self, _: &str) -> std::io::Result<()> {
            Ok(())
        }
        fn clear_line(&self) -> std::io::Result<()> {
            Ok(())
        }
        fn flush(&self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn next_stage_detaches_an_abandoned_bar() {
        let progress =
            ConsoleProgress::with_target(|| ProgressDrawTarget::term_like(Box::new(DiscardedTerm)));
        let abandoned = SfmProgressStage::BundleAdjustment;
        progress.begin(abandoned, 3);
        progress.advance(abandoned, 1);
        let Some(bar) = progress.bar(abandoned) else {
            panic!("bundle adjustment has no bar");
        };
        assert!(!bar.is_hidden());

        // A failed global candidate leaves its bundle adjustment unfinished before the next
        // candidate begins.
        progress.begin(SfmProgressStage::TranslationAveraging, 1);
        assert!(bar.is_hidden());
        assert_eq!(bar.position(), 1);
    }

    #[test]
    fn empty_stage_is_never_attached() {
        let progress = ConsoleProgress::with_target(ProgressDrawTarget::stderr);
        let stage = SfmProgressStage::MatcherPreparation;
        progress.begin(stage, 0);
        progress.finish(stage);

        let Some(bar) = progress.bar(stage) else {
            panic!("matcher preparation has no bar");
        };
        assert_eq!(bar.length(), None);
        assert!(bar.is_hidden());
    }

    #[test]
    fn empty_rerun_detaches_the_previous_bar() {
        let progress = ConsoleProgress::with_target(ProgressDrawTarget::stderr);
        let stage = SfmProgressStage::Verification;
        progress.begin(stage, 2);
        progress.advance(stage, 2);
        progress.finish(stage);
        progress.begin(stage, 0);
        progress.finish(stage);

        let Some(bar) = progress.bar(stage) else {
            panic!("verification has no bar");
        };
        assert!(bar.is_hidden());
    }

    #[test]
    fn finished_and_abandoned_lines_are_ended() {
        let progress =
            ConsoleProgress::with_target(|| ProgressDrawTarget::term_like(Box::new(DiscardedTerm)));
        let (first, second) = (SfmProgressStage::Matching, SfmProgressStage::Verification);
        let open = |stage| {
            let (index, _, _) = progress.stage(stage).unwrap();
            progress.open_lines[index].load(Ordering::Relaxed)
        };
        progress.begin(first, 3);
        assert!(open(first), "a drawn bar leaves its line open");
        progress.finish(first);
        assert!(!open(first), "finishing ends the line");
        progress.begin(second, 3);
        progress.begin(first, 2);
        assert!(
            !open(second),
            "an abandoned bar's line is ended before the next stage draws"
        );
    }
}
