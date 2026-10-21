// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reconstruct an image directory with incremental SfM and record it as a Rerun `.rrd` file.
//!
//! ```text
//! pixi run cargo run --release --manifest-path crates/openmvg-rerun/Cargo.toml \
//!   --example record_sfm -- /path/to/images reconstruction.rrd
//! rerun reconstruction.rrd
//! ```

use std::error::Error;

use openmvg_rerun::{RerunRecorder, RerunRecorderConfig};
use openmvg_sfm::{Dataset, Frontend, Incremental, ViewGraph};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let (Some(images), Some(output), None) = (arguments.next(), arguments.next(), arguments.next())
    else {
        return Err("usage: record_sfm <image-dir> <output.rrd>".into());
    };

    let dataset = Dataset::from_dir(&images)?;
    let graph = ViewGraph::build(&dataset, &Frontend::default())?;
    let mut recorder = RerunRecorder::save(&output, RerunRecorderConfig::default())?;
    // A visualization failure disables the recorder; it never changes the reconstruction.
    let result = Incremental::default()
        .runner()
        .observer(&mut recorder)
        .run_prepared_owned(graph);
    let report = recorder.finish();
    let output_summary = result?;

    println!(
        "registered {} views; wrote {} events ({} geometry frames) to {output}",
        output_summary.reconstruction().registered_view_count(),
        report.events_written,
        report.geometry_frames_written,
    );
    if let Some(error) = report.first_error {
        return Err(format!("recording stopped early: {error}").into());
    }
    Ok(())
}
