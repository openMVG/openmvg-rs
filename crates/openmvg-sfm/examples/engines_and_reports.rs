// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepare a dataset once, run both engines on it, and read what each report says.
//!
//! You will learn: how to build the view graph once and save it with `checkpoint`, run the
//! Incremental and Global engines on the same graph, read each run's route and per-view
//! verdicts, and check that a run repeats exactly.
//! Data: the Sceaux Castle images (11 photographs and their `K.txt` calibration), which
//! `pixi run examples-real` fetches; any folder of overlapping photographs works.
//! Run: cargo run --release -p openmvg-sfm --example engines_and_reports --features image-io --
//! <images>
//! Expected: on Sceaux Castle, both engines register all 11 views, and a second Incremental run
//! gives the same reconstruction, byte for byte.
//! Next: the umbrella crate's `two_view_from_scratch` example opens up the front end and the
//! engines for one image pair.

use openmvg_sfm::reports::ViewVerdictStatus;
use openmvg_sfm::{Dataset, Frontend, Global, Incremental, SfmOutput, ViewGraph, checkpoint};

/// Print the route, the registered and omitted views, and the point count of one run.
fn summarize(engine: &str, output: &SfmOutput) {
    let (report, reconstruction) = (output.report(), output.reconstruction());
    let omitted = report.view_verdicts().iter();
    let omitted = omitted
        .filter(|verdict| matches!(verdict.status(), ViewVerdictStatus::Unregistered { .. }));
    println!(
        "{engine}: route {}, {} views registered, {} left out, {} points",
        report.final_route().as_str(),
        reconstruction.registered_view_count(),
        omitted.count(),
        reconstruction.point_count()
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let images = std::env::args_os()
        .nth(1)
        .ok_or("usage: engines_and_reports <image-directory>")?;
    // The images and their camera: from the K.txt calibration, or from the EXIF focal length.
    let dataset = Dataset::from_dir(images)?;

    // The front end detects, matches and verifies every pair once; it is the expensive part.
    let graph = ViewGraph::build(&dataset, &Frontend::default())?;
    println!(
        "{} views, {} verified pairs",
        dataset.views().len(),
        graph.verified_pair_count()
    );

    // Save the prepared graph and read it back: later runs can start from the file.
    let path = std::env::temp_dir().join("openmvg-engines-and-reports.omvgrs");
    checkpoint::write_view_graph(&graph, &path)?;
    let graph = checkpoint::read_view_graph(&path)?;
    std::fs::remove_file(&path)?;

    let incremental = Incremental::default().run_prepared(&graph)?;
    summarize("Incremental", &incremental);
    let global = Global::default().run_prepared(&graph)?;
    summarize("Global", &global);
    // A Global run may keep only part of the scene, or none of it; its report says why.
    for verdict in global.report().view_verdicts() {
        if let ViewVerdictStatus::Unregistered { reason } = verdict.status() {
            println!("  Global left a view out: {reason}");
        }
    }

    // The same graph, configuration and seed give the same reconstruction at any thread count.
    let again = Incremental::default().run_prepared(&graph)?;
    let (first, second) = (
        incremental.reconstruction().to_bytes()?,
        again.reconstruction().to_bytes()?,
    );
    assert_eq!(first, second, "a run repeats exactly");
    println!("a second Incremental run gives the same reconstruction, byte for byte");
    Ok(())
}
