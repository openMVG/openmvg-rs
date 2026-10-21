# Changelog

This file lists the notable changes to the OpenMVG-rs crates. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The eight publishable crates
(`openmvg-core`, `openmvg-geometry`, `openmvg-features`, `openmvg-ba`, `openmvg-sfm`, `openmvg-io`,
the `openmvg` umbrella and `openmvg-cli`) are released together and share one version. Before 1.0, a minor release can
break the API. [MIGRATION.md](MIGRATION.md) states the compatibility rules.

## [0.1.0] - 2026-10-24

The first published release: a pure-Rust library for multiple view geometry and Structure from
Motion (SfM), from camera models and minimal solvers to complete Incremental and Global
reconstruction, with Python bindings. Fixed inputs and seeds give the same output at every
supported thread count ([reproducibility contract](docs/REPRODUCIBILITY.md)).

### Added

#### `openmvg-core`

- Camera models with projection, unprojection and checked constructors: pinhole (one or two
  focal lengths), pinhole with one to three radial coefficients, Brown-Conrady, fisheye, and
  spherical.
- Thresholds and residuals typed by their unit: pixels, radians, normalized image coordinates
  and world distance.
- Typed identifiers, world-to-camera poses, relative rotation and translation measurements, and
  graph algorithms (connected components, spanning trees).
- A versioned RNG protocol (`determinism`). Every randomized API takes a seed from the caller.

#### `openmvg-geometry`

- Minimal and linear solvers: fundamental matrix (7-point, 8-point), essential matrix (5-point,
  8-point) with pose decomposition, homography (DLT, ACA), and calibrated absolute pose (seven
  P3P formulations).
- Robust estimation: an `estimate_<model>` entry point per model, plain and guided, built on
  GenRANSAC. There is also a classic RANSAC/MSAC that accepts external solvers through the
  public `robust::Solver` traits.
- Triangulation: DLT, midpoint, L1 and L∞ angular, and inverse-depth-weighted midpoint.
- Rotation averaging (robust IRLS over a sparse or dense solve), L1-IRLS translation averaging,
  global positioning, and view-graph consistency filters.
- Closed-form rigid and similarity alignment (Kabsch, Umeyama).

#### `openmvg-features`

- SIFT and ALP detectors with SIFT descriptors stored as `f32` or `u8`, and RootSIFT. Their
  extraction methods return an error for an invalid configuration and an empty result for an
  image below the detector minimum.
- Brute-force and HNSW matchers over L2 and Hamming distances, with Lowe's ratio
  test and reciprocal matching. The HNSW index is deterministic for a given seed.
- Track building with union-find.

#### `openmvg-ba`

- A checked `BundleProblem` builder with robust losses, gauge choice, and per-pose and
  per-intrinsic variable masks.
- `NativeBundleAdjuster`, a Levenberg-Marquardt solver with a Schur complement over the points,
  which returns a `Solution` the caller can inspect before applying it.

#### `openmvg-sfm`

- `Dataset::from_dir` loads a folder of images. A `K.txt` file gives fixed intrinsics; otherwise
  they are initialized from EXIF and refined during the run. Images are grouped by camera, lens,
  and focal length in millimetres, so a zoom lens at two settings is never one camera. An image
  whose EXIF focal length does not convert to pixels on its own uses the pixels per millimetre of
  the same camera's other images, or else a guess that keeps the zoom ratios. A 35 mm-equivalent
  focal length is converted against the full-frame diagonal for the camera's native 3:2 or 4:3
  format, so 4:3 and cropped images get the right focal; when present, it also overrules a
  focal-plane resolution or known sensor width it contradicts. A focal-plane resolution stays
  valid on pixels rotated after capture, and one that implies no real sensor size is ignored.
- With `image-io`, an `ImageProject` assigns the images of a folder to camera groups, each with a
  camera model and focal length, and excludes the images it does not name. It is a draft JSON file
  (schema version 1); `PreparedImageDataset::to_project` writes the automatic grouping as one,
  and `prepare_image_project` turns one back into the same dataset. `image_size` reads the pixel
  size of an image from the start of its file.
- The `Frontend` detects features, matches pairs, and verifies their geometry, producing an
  immutable `ViewGraph`. It extracts streaming ALP features by default;
  `Frontend::with_extraction_method` selects SIFT. The `ViewGraph` and the front-end
  report record the extraction method.
- The Incremental and Global engines, whose `default()` is the recommended configuration and
  includes bundle adjustment. A `Runner` adds progress reporting, cancellation and observers.
- `Reconstruction` and `RunReport` outputs. `ViewGraph` and `Reconstruction` files are binary,
  versioned and checksummed, with the conventional extension `.omvgrs`. PLY export supports ASCII
  and binary through `PlyFormat`, defaults to binary, and includes camera centers as green points.

#### `openmvg-io`

- Reader, writer and validator for the OMVG community format, `0.1-draft`: a manifest and
  Parquet tables with content hashes and stable validation rule identifiers.
- Read and write `ViewGraph`, correspondences and `Reconstruction`; read ground-truth poses
  with their length unit. The specification, independent Python reference and conformance
  vectors live in [docs/format](docs/format/OMVG_FORMAT_SPEC.md).
- COLMAP text export and the `omvg validate` command.

#### `openmvg`

- The umbrella crate re-exports `core` and `geometry`, plus `features`, `sfm` and `io` through
  their corresponding features. Its `parallel` feature turns on multi-threading in every
  enabled crate, and the `sfm` example reconstructs a folder of images.

#### `openmvg-cli`

- `openmvg-sfm` reconstructs a folder of images with the Incremental or Global engine through
  the stable SfM facade: EXIF camera grouping, a `K.txt` calibration or a project file, SIFT or
  ALP features, HNSW or brute-force matching, self-calibration, and PLY, OMVG and COLMAP output
  with a versioned JSON report. Images of the folder that cannot be read, such as empty or
  truncated files, are skipped with a warning and listed in the report.
- `openmvg-sfm-configure`, behind the `tui` feature, is a terminal app that groups
  the images of a folder into cameras and writes the project file `openmvg-sfm --project` reads.

#### Python bindings

- The `openmvg-rs` package (`import openmvg_rs`, not yet on PyPI), in `bindings/python`.
  `reconstruct(image_dir, engine=...)` runs either engine and returns a `Reconstruction`, which
  writes COLMAP and PLY, saves and loads the binary format, and pickles. Its `points` array and
  its `views` (image, `Pose` and `Camera` of each registered view) give the results to NumPy.
- `estimate_fundamental`, `estimate_homography`, `estimate_essential` and
  `estimate_absolute_pose` take NumPy arrays and return the model and an inlier mask; the pose
  estimators return the same `Pose` type as `Reconstruction.views`.

#### Platforms

- Rust 1.97 or later. The library stack supports serial `wasm32-unknown-unknown` builds with
  native default features disabled. `openmvg-features` also supports `no_std + alloc` through
  `libm`; [the developer guide](DEVELOPER.md#webassembly-profiles) lists the configurations.
