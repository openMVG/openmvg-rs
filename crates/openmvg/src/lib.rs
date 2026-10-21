// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multiple View Geometry and Structure from Motion (SfM) in Rust, behind one dependency.
//!
//! OpenMVG-rs recovers cameras and 3D structure from overlapping photographs. This umbrella
//! crate re-exports each layer under its own module, [`core`], [`features`], [`geometry`],
//! [`sfm`] and [`io`], so you can start with one dependency and reach any layer. Each module is
//! the full documentation of its crate.
//!
//! ## Start here
//!
//! Pick the module that matches your goal:
//!
//! | Goal | Module | Umbrella feature |
//! |---|---|---|
//! | Reconstruct a folder of images | [`sfm`] | `sfm-image-io` |
//! | Store or exchange datasets and reconstructions, export to COLMAP | [`io`] | `io` |
//! | Detect, describe and match features | [`features`] | default |
//! | Solve geometry: two-view models, pose, triangulation, averaging | [`geometry`] | always (`faer`, a default, adds sparse averaging) |
//! | Cameras, poses and the conventions every crate shares | [`core`] | always |
//!
//! To reconstruct images without writing code, install the `openmvg-sfm` binary, from the
//! `openmvg-cli` crate. The [`sfm`] and [`io`] pages write paths as `openmvg_sfm::…` and
//! `openmvg_io::…`; through this crate they are `openmvg::sfm::…` and `openmvg::io::…`. A
//! feature badge on those pages names the sub-crate's feature: `image-io` there is
//! `sfm-image-io` here, `observer` is `sfm-observer`, and the `omvg` feature of `openmvg-io`
//! comes with `io`. `console-progress` has no umbrella feature: depend on `openmvg-sfm` with it
//! for terminal progress bars. [Terms](#terms), near the end of this page, defines the words the
//! pages use.
//!
//! ```toml
//! [dependencies]
//! openmvg = { version = "0.1", features = ["sfm-image-io", "io"] }
//! ```
//!
//! ## Example
//!
//! Reconstruct a folder of images, then save the points and cameras as a PLY point cloud and
//! as a COLMAP model:
//!
//! ```no_run
//! # #[cfg(all(feature = "sfm-image-io", feature = "io"))]
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     use openmvg::io::write_colmap;
//!     use openmvg::sfm::{Dataset, Incremental};
//!
//!     // The camera comes from a `K.txt` file, else the EXIF focal length, else the image size.
//!     let dataset = Dataset::from_dir("images")?;
//!     // `run` builds the view graph with the default front end, then reconstructs it.
//!     let output = Incremental::default().run(&dataset)?;
//!     let reconstruction = output.reconstruction();
//!     println!("{} views registered", reconstruction.registered_view_count());
//!
//!     // A point cloud for any 3D viewer, and a text model for the tools that read COLMAP.
//!     reconstruction.write_ply("scene.ply")?;
//!     write_colmap(reconstruction, "colmap")?;
//!     Ok(())
//! }
//! # #[cfg(not(all(feature = "sfm-image-io", feature = "io")))]
//! # fn main() {}
//! ```
//!
//! The PLY file holds the points in white and the camera centers in green, so it has a few more
//! vertices than points; any point-cloud viewer opens it. [`io::write_colmap`] writes
//! `cameras.txt`, `images.txt` and `points3D.txt` into `colmap/sparse/0`; COLMAP's graphical
//! interface opens that folder with *File › Import model*. A reconstruction from photographs
//! alone has an arbitrary scale: its units are not meters. Both writers replace earlier output.
//! To keep the result with its observations, read it back later or share it with other tools,
//! write an OMVG dataset with [`io::write_reconstruction`], which publishes a new directory and
//! fails if the path exists.
//!
//! Every layer also works on its own. With the default features, find the keypoint of a bright
//! blob:
//!
//! ```
//! use openmvg::features::extraction::{AlpExtractor, AlpPreset, GrayImage};
//!
//! // A 96 × 96 image with one Gaussian blob at its center; intensities run from 0 to 255.
//! let image = GrayImage::from_fn(96, 96, |y, x| {
//!     let squared_distance = (x as f32 - 48.0).powi(2) + (y as f32 - 48.0).powi(2);
//!     200.0 * (-squared_distance / 32.0).exp()
//! });
//! let features = AlpExtractor::with_preset(AlpPreset::Normal).extract_u8(&image)?;
//! let keypoint = &features.keypoints()[0];
//! assert!((keypoint.x - 48.0).abs() < 0.5 && (keypoint.y - 48.0).abs() < 0.5);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Choosing the front end and the engine
//!
//! `Incremental::default().run(&dataset)` is shorthand for two steps: [`sfm::ViewGraph::build`]
//! with [`sfm::Frontend::default`], which detects, matches and verifies every image pair, then
//! [`sfm::Incremental::run_prepared`] on the resulting view graph. Take the two steps yourself
//! to choose the detector, the matcher or the working image size with the [`sfm::Frontend`]
//! builder, or to run both engines on one graph:
//!
//! ```no_run
//! # #[cfg(feature = "sfm-image-io")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use openmvg::sfm::{Dataset, ExtractionMethod, Frontend, Incremental, MatchingMethod, ViewGraph};
//!
//! let dataset = Dataset::from_dir("images")?;
//! let frontend = Frontend::default()
//!     .with_extraction_method(ExtractionMethod::Sift)
//!     .with_matching_method(MatchingMethod::BruteForce);
//! let graph = ViewGraph::build(&dataset, &frontend)?;
//! let output = Incremental::default().run_prepared(&graph)?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "sfm-image-io"))]
//! # fn main() {}
//! ```
//!
//! The [`sfm`] page explains how to choose an engine and what matters for a good
//! reconstruction.
//!
//! ## How the crates fit together
//!
//! The figure reads at three levels. The top row is what an application calls: a folder of
//! images becomes a [`Dataset`](sfm::Dataset); [`Frontend`](sfm::Frontend) turns it into a
//! [`ViewGraph`](sfm::ViewGraph); the Incremental or Global engine turns that into a
//! [`Reconstruction`](sfm::Reconstruction), saved as PLY by [`sfm`] or as COLMAP and OMVG files
//! by [`io`]. The middle row is what happens inside for one image pair: [`features`] detects,
//! describes and matches keypoints with Lowe's ratio test; [`geometry`] estimates a robust
//! essential matrix and the relative pose, which the front end keeps as a verified pair, and later
//! triangulates the points inside the engines; bundle adjustment, from the `openmvg-ba` crate,
//! refines poses and points. [`core`] holds the cameras, poses, identifiers and units that every
//! step shares. [Terms](#terms) defines the words this page uses.
//!
#![doc = include_str!("../figures/pipeline_map.svg")]
//!
//! ## Learning path
//!
//! 1. [`core`]: the camera model, the pose convention and the pixel convention.
//! 2. [`features`]: keypoints, descriptors, matches and tracks.
//! 3. [`geometry`]: from matches to relative poses, absolute poses and 3D points.
//! 4. [`sfm`]: the front end and the two engines that chain all of it into a reconstruction,
//!    and [`sfm::ba`], the bundle adjustment of `openmvg-ba` that every engine run includes.
//! 5. [`io`]: storing and sharing the results.
//!
//! Each crate's page starts with what it is for, an example, and its own learning path.
//!
//! From a clone of the repository, two examples run on a folder of images. `sfm` is the
//! shortest complete program, and `two_view_from_scratch` opens up the pipeline for one image
//! pair, from keypoints to triangulated points, with a comment on every step:
//! `cargo run --release -p openmvg --example <name> --features sfm-image-io,io -- <images>`.
//! `pixi run examples-real` fetches the Sceaux Castle images and runs both on them.
//!
//! ## Terms
//!
//! | Term | Meaning |
//! |---|---|
//! | View | One image of the dataset, with the camera that took it ([`sfm::Dataset::views`]) |
//! | Camera, intrinsics | The model that maps camera-frame rays to pixels: focal length, principal point and lens distortion ([`core::camera::Camera`], [`core::model::CameraIntrinsics`]) |
//! | Pose | Where a view looks from: the world-to-camera rotation `R` and the camera center `C`; the translation is `t = −R C` ([`core::model::WorldToCameraPose`]) |
//! | Bearing | The unit ray from the camera center through a pixel, which the geometry solvers take instead of pixels ([`core::camera::Camera::bearing`]) |
//! | Relative pose, absolute pose | The rotation and translation direction between two views; the pose of one view from its 2D–3D matches ([`core::model::RelativePose`], [`geometry::pose`]) |
//! | RANSAC, robust estimation | Fitting a model to matches that include wrong ones, by keeping the model that most matches agree with ([`geometry::robust`]) |
//! | Triangulation | Computing a 3D point from its observations in two or more posed views ([`geometry::triangulation`]) |
//! | Rotation averaging, global positioning | Estimating every view's rotation at once from the pairwise relative rotations, then every position, together with the points, from the bearings of the tracks, as the Global engine does ([`geometry::averaging`]) |
//! | Observation | One view's pixel of a track or of a 3D point |
//! | EXIF | The metadata a camera writes into an image file, including the focal length that [`sfm::Dataset::from_dir`] reads when there is no `K.txt` |
//! | Keypoint, descriptor | A distinctive image location with its scale and orientation, and the vector that describes its neighborhood for matching ([`features::extraction`]) |
//! | ALP | Analytic Laplacian Polynomial, the default detector, which finds blobs at a continuous scale ([`features::extraction::AlpExtractor`]) |
//! | RootSIFT | The 128-value SIFT descriptor with each value square-rooted after normalization, which makes distances between descriptors compare their gradient histograms better; ALP always produces it and SIFT by default ([`features::extraction::SiftExtractor::with_root_sift`]) |
//! | Calibration | Finding a camera's intrinsics: in advance, written to a `K.txt` file, or during the reconstruction, when the engines refine them from a first guess |
//! | Engine | One of the two reconstruction strategies, [`sfm::Incremental`] or [`sfm::Global`], that turn a view graph into a reconstruction |
//! | HNSW | Hierarchical Navigable Small World, the approximate nearest-neighbor search that matches descriptors by default ([`features::matching::HnswMatcher`]) |
//! | Lowe's ratio test | Keep a match only when the nearest descriptor is clearly closer than the second nearest ([`features::matching::LoweRatioFilter`]) |
//! | Track | The observations of one scene point across views, linked through pairwise matches ([`features::tracking::Track`]) |
//! | Essential matrix | The 3 × 3 matrix that relates matching rays of two calibrated views, and gives their relative pose ([`geometry::essential::estimate_essential`]) |
//! | Front end | Detection, matching and verification of every image pair, the first stage of a reconstruction ([`sfm::Frontend`]) |
//! | Verified pair | An image pair whose matches fit one two-view geometry; its inlier matches and relative pose are kept ([`sfm::ViewGraph::verified_pair_count`]) |
//! | View graph | The views, their keypoints, matches and tracks, and the verified pairs that link them: the input of both engines ([`sfm::ViewGraph`]) |
//! | Registration | Giving a view a pose in the reconstruction; a view that is not registered is left out ([`sfm::Reconstruction::registered_view_count`]) |
//! | Landmark | A track triangulated into a 3D point of the reconstruction ([`sfm::Reconstruction::landmarks`]) |
//! | Bundle adjustment (BA) | The joint refinement of poses, 3D points and, optionally, camera intrinsics that best fits the observed pixels; every engine run includes it |
//! | OMVG | The open dataset format of [`io`]: a JSON manifest and Parquet tables, which other tools and languages can read |
//!
//! ## Features
//!
//! | Feature | Default | Effect |
//! |---|---|---|
//! | `features` | yes | [`features`]: detection, description, matching and tracking |
//! | `faer` | yes | Sparse rotation averaging and global positioning in [`geometry`]; [`sfm`] always uses it |
//! | `parallel` | yes | Native multi-threading in every enabled crate, with results independent of the thread count |
//! | `sfm` | no | [`sfm`]: the Incremental and Global engines with native bundle adjustment |
//! | `sfm-image-io` | no | `sfm` plus image discovery, decoding and EXIF metadata |
//! | `sfm-observer` | no | SfM reconstruction events for live visualization, which never change the result ([`sfm::events`]) |
//! | `io` | no | [`io`]: the OMVG community format and COLMAP export; the only feature that adds the Arrow and Parquet dependencies |
//! | `full` | no | All of the above |
//!
//! The umbrella never enables an `unstable-*` feature of the crates it re-exports. A
//! `wasm32-unknown-unknown` build runs on one thread: use `--no-default-features` plus the
//! modules it needs.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// The root docs link the optional `features`, `sfm`, `sfm::events` and `io` re-exports;
// full-feature doc builds check them.
#![cfg_attr(
    not(all(feature = "features", feature = "sfm-observer", feature = "io")),
    allow(rustdoc::broken_intra_doc_links)
)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented
    )
)]

pub use openmvg_core as core;
pub use openmvg_geometry as geometry;

#[cfg(feature = "features")]
pub use openmvg_features as features;

#[cfg(feature = "sfm")]
pub use openmvg_sfm as sfm;

#[cfg(feature = "io")]
pub use openmvg_io as io;

// Compile the examples from the README included in this crate's package.
#[cfg(all(doctest, feature = "full"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
