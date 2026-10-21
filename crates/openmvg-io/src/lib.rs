// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Share multi-view datasets and reconstructions in the OMVG community format, check that they
//! conform, and export a reconstruction to COLMAP.
//!
//! An OMVG dataset is a directory holding a JSON manifest and one Parquet file per table, with
//! content hashes that any implementation can verify. This crate reads and writes it
//! for the `openmvg-sfm` types, so a prepared view graph or a finished reconstruction can be
//! stored, shared with other tools, and read back with its geometry intact. The normative text
//! is the OMVG format specification, version `0.1-draft`, at `docs/format/OMVG_FORMAT_SPEC.md`
//! in the [repository](https://github.com/openMVG/openmvg-rs). The format is a draft and may
//! change before version 1.0.
//!
//! A dataset declares one or more *profiles*, and each profile names the tables it must hold
//! besides the cameras and views: `dataset` holds keypoints, and optionally matches and pairs;
//! `reconstruction` holds poses, points and the pixels that observe them; and `benchmark` holds
//! ground-truth poses. On these pages a
//! *dataset* is such an OMVG directory; `openmvg_sfm::Dataset`, the input of a reconstruction,
//! is a different thing.
//!
//! ```text
//! scene.omvg/manifest.json + <table>.parquet
//!        |  decode each table, verify the hashes, and check every rule
//!        v
//!   ViewGraph (dataset profile)  ---->  Incremental / Global  ---->  Reconstruction
//!        ^                                                               |
//!        `------------------ write (dataset, reconstruction) <-----------'
//! ```
//!
//! ## Start here
//!
//! - [`write_reconstruction`] and [`read_reconstruction`] store a finished `Reconstruction`.
//! - [`check`] verifies a dataset against every rule of the specification and returns each
//!   [`Violation`] with its stable [`Rule`] identifier, such as `hash.table`; an empty list means
//!   the dataset conforms. The `omvg validate <path>` command runs the same check and exits 0
//!   if the dataset conforms, 1 if it does not, and 2 on a usage error; `cargo install
//!   openmvg-io` installs it.
//! - [`write_colmap`] writes a reconstruction as a COLMAP text model, for the tools that read
//!   one.
//!
//! ## Example
//!
//! Store a two-view reconstruction, check the files, and read it back:
//!
//! ```rust
//! # #[cfg(feature = "omvg")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use nalgebra::{Point2, Point3, Rotation3};
//! use openmvg_io::{ReadOptions, WriteOptions, check, read_reconstruction, write_reconstruction};
//! use openmvg_sfm::model::ImageMeta;
//! use openmvg_sfm::{
//!     Camera, Dataset, IntrinsicId, PoseId, Reconstruction, ReconstructionParts, SfmEngine,
//!     ViewId, WorldToCameraPose,
//! };
//!
//! // Two views share one pinhole camera; each view has its own pose.
//! let dataset = Dataset::builder()
//!     .image_meta(["left", "right"].map(|name| ImageMeta::external_id(name, 640, 480)))
//!     .camera_models([Camera::try_pinhole(500.0, 500.0, 320.0, 240.0)?])
//!     .shared_intrinsic(IntrinsicId::new(0))
//!     .build()?;
//! let pose = |x| {
//!     WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::new(x, 0.0, 0.0))
//! };
//! let (cameras, views) = (dataset.camera_models().to_vec(), dataset.views().to_vec());
//! // One point, seen in both views at the pixels it projects to.
//! let parts = ReconstructionParts::new(7, SfmEngine::Incremental, cameras, views)
//!     .with_pose(PoseId::new(0), pose(0.0)?)
//!     .with_pose(PoseId::new(1), pose(1.0)?)
//!     .with_point(Point3::new(0.5, 0.0, 5.0), vec![
//!         (ViewId::new(0), Point2::new(370.0, 240.0)),
//!         (ViewId::new(1), Point2::new(270.0, 240.0)),
//!     ]);
//! let reconstruction = Reconstruction::try_from_parts(parts)?;
//!
//! let directory = std::env::temp_dir().join("openmvg-io-crate-example.omvg");
//! let _ = std::fs::remove_dir_all(&directory);
//! write_reconstruction(&reconstruction, &directory, &WriteOptions::new())?;
//! assert!(check(&directory).is_empty());
//! let stored = read_reconstruction(&directory, &ReadOptions::new())?;
//! assert_eq!(stored.reconstruction.registered_view_count(), 2);
//! std::fs::remove_dir_all(&directory)?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "omvg"))]
//! # fn main() {}
//! ```
//!
//! The example builds its reconstruction by hand to stay self-contained. In an application the
//! reconstruction comes from an engine of `openmvg-sfm` run on a folder of images, which needs
//! that crate's `image-io` feature, and is stored the same way:
//!
//! ```no_run
//! # #[cfg(feature = "omvg")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use openmvg_io::{WriteOptions, check, write_reconstruction};
//! use openmvg_sfm::{Dataset, Incremental};
//!
//! let dataset = Dataset::from_dir("castle/images")?;
//! let output = Incremental::default().run(&dataset)?;
//! let options = WriteOptions::new().with_image_root("castle/images");
//! write_reconstruction(output.reconstruction(), "castle.omvg", &options)?;
//! assert!(check("castle.omvg").is_empty());
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "omvg"))]
//! # fn main() {}
//! ```
//!
//! ## What each function carries
//!
//! The typed API maps the format's profiles onto the Structure from Motion (SfM) types of
//! `openmvg-sfm`. [`write_view_graph`] and
//! [`read_view_graph`] carry a `ViewGraph` through the `dataset` profile,
//! [`write_reconstruction`] and [`read_reconstruction`] carry a `Reconstruction` through the
//! `reconstruction` profile, and [`read_ground_truth`] reads the poses of a `benchmark`.
//! [`write_correspondences`] and [`read_correspondences`] carry putative matches, which the
//! receiving tool verifies itself, and optional ground truth through the `dataset` profile.
//!
//! [`read_reconstruction`], [`read_correspondences`] and [`read_ground_truth`] return the
//! dataset's [`LengthUnit`] with their result, and [`WriteOptions::with_length_unit`] relabels
//! the written lengths without rescaling them.
//!
//! ## What a round trip keeps
//!
//! A value written and read back keeps its geometry exactly. What the format does not record
//! comes back renumbered, recomputed, or from [`ReadOptions`]:
//!
//! | Value | Kept | Not kept |
//! |---|---|---|
//! | `Reconstruction` | Cameras, views, poses, the active points and the pixels that observe them | Feature indices, renumbered per view; reprojection errors, recomputed; rejected points; components |
//! | `ViewGraph` | Cameras, views, keypoints, matches, and the inlier flags and two-view model of each verified pair | Pair weights, which the front end sets to one and the reader sets back to one; the extraction method |
//! | Both | Image files, when written and read with the same image root | The root seed and a reconstruction's engine, which [`ReadOptions`] gives; an image source that is not a file, which reads back as a file path |
//!
//! ## What the tables hold
//!
//! Each table is one Parquet file, which any Parquet reader opens. Pixels follow this library's
//! convention: integer coordinates are pixel centers, and `(0, 0)` is the center of the top-left
//! pixel. Poses are world to camera: a rotation `R`, stored row-major, and the camera center `C`.
//!
//! | Table | One row per | Main columns |
//! |---|---|---|
//! | `cameras` | set of intrinsics, shared by the views that use it | `model`, `width`, `height`, `fx`, `fy`, `cx`, `cy`, distortion |
//! | `views` | view | `camera_id`, `pose_id`, `image`, the path below the image root |
//! | `poses` | registered pose, which several views can share | `rotation`, `center` |
//! | `points` | 3D point | `xyz`, `error_px` |
//! | `observations` | pixel of a point in a view | `point_id`, `view_id`, `x`, `y` |
//! | `keypoints` | keypoint of a view | `x`, `y`, `scale`, `orientation` |
//! | `matches` | putative match of an image pair | the two features, `distance`, `inlier` |
//! | `pairs` | examined image pair, accepted or rejected | `model`, `matrix`, relative `rotation` and unit `translation` |
//! | `gt_poses` | ground-truth pose of a view | `rotation`, `center` |
//!
//! The specification gives every column, its type and its rules.
//!
//! ## Image paths
//!
//! Images are not part of a dataset: each view stores the path of its image relative to an
//! *image root*. By default the writer stores each image's file name, and the reader resolves
//! it against the dataset directory, which usually does not hold the images. Give the folder of
//! the images to both sides, and every view read back points to its file again:
//!
//! ```rust,no_run
//! # #[cfg(feature = "omvg")]
//! # fn store(reconstruction: &openmvg_sfm::Reconstruction) -> openmvg_io::Result<()> {
//! use openmvg_io::{ReadOptions, WriteOptions, read_reconstruction, write_reconstruction};
//!
//! let images = "castle/images";
//! let write = WriteOptions::new().with_image_root(images);
//! write_reconstruction(reconstruction, "castle.omvg", &write)?;
//! let read = ReadOptions::new().with_image_root(images);
//! let stored = read_reconstruction("castle.omvg", &read)?;
//! # Ok(())
//! # }
//! ```
//!
//! With an image root, each image is stored as its path below the root, subfolders included,
//! and each view reads back named by that path.
//!
//! ## OMVG or a checkpoint
//!
//! `openmvg-sfm` also saves a `ViewGraph` or a `Reconstruction` in a binary file of its own,
//! with [`openmvg_sfm::checkpoint`]. Choose by who reads the file back:
//!
//! - A checkpoint keeps the whole value, with its seed, engine, components and rejected points,
//!   and a new write replaces the file. A build reads only the checkpoint version it writes:
//!   use checkpoints to save and resume your own runs.
//! - An OMVG dataset is an open format that other tools and languages read and that [`check`]
//!   verifies. It keeps the geometry, as the table above lists, and each write publishes a new
//!   directory. Use it to share data or to inspect it with any Parquet reader.
//!
//! ## Common mistakes
//!
//! - Replaying a stored view graph with a different seed or configuration. A graph replays the
//!   same reconstruction only with the same engine configuration and root seed. The format does
//!   not store the seed: a graph built with the default seed replays with
//!   [`ReadOptions::new`]; for any other seed, send the graph's `root_seed()` with the dataset,
//!   as the 64 hexadecimal digits that its `Display` writes and `parse` reads back, and give it
//!   to [`ReadOptions::with_root_seed`].
//! - Reading a dataset back without its image root. The views then point into the dataset
//!   directory; give the reader the root the writer had.
//! - Writing to an existing path. The writers publish a new directory and fail with
//!   [`Error::Exists`] if the path exists; remove the old dataset first, or choose a new path.
//! - Shifting COLMAP coordinates by hand. COLMAP puts the center of the top-left pixel at
//!   `(0.5, 0.5)`; [`write_colmap`] adds that shift.
//!
//! ## Learning path
//!
//! 1. [`write_reconstruction`], [`read_reconstruction`] and [`check`]: store, verify and
//!    reload a result.
//! 2. [`write_view_graph`] and [`read_view_graph`]: share a prepared front end, so that another
//!    person or tool can run the engines on it. To save your own runs, a checkpoint is simpler,
//!    as [OMVG or a checkpoint](#omvg-or-a-checkpoint) explains.
//! 3. [`write_correspondences`] and [`read_correspondences`]: exchange putative matches with
//!    another tool.
//! 4. [`read_ground_truth`]: compare a reconstruction with the poses of a benchmark.
//! 5. [`write_colmap`]: hand a reconstruction to the COLMAP ecosystem.
//!
//! From a clone of the repository, the `write_check_read` example builds a small reconstruction,
//! writes it, checks the files, reads it back unchanged, and exports it to COLMAP:
//! `cargo run --release -p openmvg-io --example write_check_read`.
//!
//! ## Features
//!
//! | Feature | Default | Effect |
//! |---|---|---|
//! | `omvg` | yes | The OMVG format and the `omvg` command; the only feature of any supported crate that brings Arrow and Parquet |
//!
//! With `default-features = false` the crate is the COLMAP export alone. The internal binary
//! files and the PLY export stay in `openmvg-sfm`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// The root docs link the OMVG API; documentation builds use the default `omvg` profile.
#![cfg_attr(not(feature = "omvg"), allow(rustdoc::broken_intra_doc_links))]
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

mod colmap;
mod error;
#[cfg(test)]
mod fixtures;
#[cfg(feature = "omvg")]
mod format;
#[cfg(feature = "omvg")]
mod mapping;
#[cfg(feature = "omvg")]
mod options;

pub use colmap::write_colmap;
pub use error::{Error, Result};
#[cfg(feature = "omvg")]
pub use format::{FORMAT_VERSION, Rule, Violation, check};
#[cfg(feature = "omvg")]
pub use mapping::{
    Correspondences, GroundTruth, GroundTruthView, StoredReconstruction, read_correspondences,
    read_ground_truth, read_reconstruction, read_view_graph, write_correspondences,
    write_reconstruction, write_view_graph,
};
#[cfg(feature = "omvg")]
pub use options::{LengthUnit, ReadOptions, WriteOptions};

// Compile the README example.
#[cfg(all(doctest, feature = "omvg"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
