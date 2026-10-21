// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reconstruct a folder of images: the shortest complete Structure from Motion (SfM) program.
//!
//! You will learn: the three calls of an image-to-reconstruction program (read the dataset, run
//! an engine, export the result), how to report each failure without panicking, and how to tell
//! a reconstruction that left out most views.
//! Data: a folder of overlapping photographs, such as the Sceaux Castle images that
//! `pixi run examples-real` fetches; a `K.txt` beside them gives the camera intrinsics.
//! Run: cargo run --release -p openmvg --example sfm --features sfm-image-io,io -- <images>
//! [<colmap-directory>]
//! Expected: on Sceaux Castle, 10 or 11 of the 11 views registered via incremental.
//! Next: the `two_view_from_scratch` example opens up what happens inside, step by step.

use std::process::ExitCode;

use openmvg::io::write_colmap;
use openmvg::sfm::{Dataset, Incremental};

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let (Some(image_dir), colmap_dir, None) = (args.next(), args.next(), args.next()) else {
        eprintln!("usage: sfm <image-directory> [<colmap-output-directory>]");
        return ExitCode::from(2);
    };
    // The camera comes from a K.txt file, else the EXIF focal length, else the image size.
    let dataset = match Dataset::from_dir(image_dir) {
        Ok(dataset) => dataset,
        Err(error) => {
            eprintln!("cannot load dataset: {error}");
            return ExitCode::FAILURE;
        }
    };
    // `run` builds the view graph with the default front end, then reconstructs it.
    match Incremental::default().run(&dataset) {
        Ok(result) => {
            let registered = result.reconstruction().registered_view_count();
            let views = dataset.views().len();
            // The route says which engine produced the reconstruction.
            println!(
                "{registered} of {views} views registered via {}",
                result.report().final_route().as_str()
            );
            // A view that overlaps too little with the others is left out. When most are left
            // out, the result describes only a corner of the scene: report it as a failure.
            if 2 * registered < views {
                eprintln!("most views were left out: check that the images overlap");
                return ExitCode::FAILURE;
            }
            if let Some(colmap_dir) = colmap_dir
                && let Err(error) = write_colmap(result.reconstruction(), colmap_dir)
            {
                eprintln!("cannot write the COLMAP model: {error}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("reconstruction failed: {error}");
            ExitCode::FAILURE
        }
    }
}
