// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Local features: detect keypoints in an image, describe them, match them between images,
//! and link the matches into tracks.
//!
//! This is the front end of Structure from Motion (SfM). Its matches feed the solvers of
//! `openmvg-geometry`, and `openmvg-sfm` runs it on every image of a dataset. The crate decodes
//! no image files: it takes a [`GrayImage`](extraction::GrayImage) that you fill from a decoder
//! of your choice, or from the `image-io` loaders of `openmvg-sfm`.
//!
//! ## Start here
//!
//! - [`AlpExtractor`](extraction::AlpExtractor), the default detector, finds keypoints and
//!   describes them with 128-value RootSIFT descriptors.
//! - [`HnswMatcher`](matching::HnswMatcher) finds each descriptor's nearest neighbors in another
//!   image, and [`MatchAcceptance`](matching::MatchAcceptance) keeps the unambiguous matches.
//! - [`build_tracks_scored`](tracking::build_tracks_scored) links pairwise matches into
//!   multi-view tracks with at most one feature per view.
//!
//! ## Example
//!
//! Detect and describe the keypoints of an image with four bright blobs:
//!
//! ```rust
//! use openmvg_features::extraction::{AlpExtractor, AlpPreset, GrayImage};
//!
//! // A 128 × 128 image: rows are image rows, intensities run from 0 to 255.
//! let blobs = [(32.0, 32.0, 3.0), (96.0, 40.0, 5.0), (40.0, 96.0, 4.0), (90.0, 92.0, 6.0)];
//! let image = GrayImage::from_fn(128, 128, |y, x| {
//!     let intensity = |&(cx, cy, size): &(f32, f32, f32)| {
//!         let squared_distance = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
//!         200.0 * (-squared_distance / (2.0 * size * size)).exp()
//!     };
//!     blobs.iter().map(intensity).sum::<f32>()
//! });
//!
//! let features = AlpExtractor::with_preset(AlpPreset::Normal).extract_u8(&image)?;
//! assert!(features.keypoints().len() >= blobs.len());
//! assert_eq!(features.keypoints().len(), features.descriptors().len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Getting good results
//!
//! - Use ALP. [`AlpPreset::Normal`](extraction::AlpPreset::Normal), the extractor default,
//!   keeps fewer, stronger keypoints. [`AlpPreset::High`](extraction::AlpPreset::High) means
//!   high density: it lowers the response threshold and also keeps weaker keypoints. Choose
//!   `High` for reconstruction and tracks, where more matches help, and for scenes with little
//!   texture; `openmvg-sfm` uses it. The [`extraction`] page compares ALP
//!   with [`SiftExtractor`](extraction::SiftExtractor), the classic SIFT pipeline.
//! - Store descriptors as `u8` ([`extract_u8`](extraction::AlpExtractor::extract_u8)) unless
//!   you need full precision: its values take 128 bytes instead of 512 (plus a 4-byte scale),
//!   and `openmvg-sfm` uses them.
//! - Match with [`HnswMatcher`](matching::HnswMatcher), the approximate graph search that
//!   `openmvg-sfm` uses by default. [`BruteForceMatcher`](matching::BruteForceMatcher) is exact
//!   and suits small sets or checking a result.
//! - Ask the matcher for two neighbors per query and keep a match only when the nearest is
//!   clearly closer than the second, with a ratio of 0.8 on plain distances, the value
//!   `openmvg-sfm` uses; that ratio test removes most wrong matches before geometry sees
//!   them. [`MatchAcceptance::Reciprocal`](matching::MatchAcceptance::Reciprocal),
//!   the default, runs it both ways and keeps one-to-one matches.
//!
//! ## Common mistakes
//!
//! - Indexing [`GrayImage`](extraction::GrayImage) as `(x, y)`. It is a column-major matrix
//!   with one matrix row per image row, so pixel `(x, y)` is `image[(y, x)]`.
//! - Applying the ratio test to squared distances with an unsquared ratio. After a
//!   squared-distance metric such as [`SquaredL2DistanceU8`](matching::SquaredL2DistanceU8),
//!   use [`MatchAcceptance::filter_squared`](matching::MatchAcceptance::filter_squared) or
//!   [`LoweRatioFilter::apply_squared`](matching::LoweRatioFilter::apply_squared), which take
//!   the ratio on plain distances.
//! - Comparing descriptors from different extractors or storage types. Match `u8` with `u8`,
//!   and ALP with ALP.
//!
//! ## Learning path
//!
//! 1. [`extraction`]: choosing between the ALP and SIFT detectors, how each works, and the
//!    descriptor storage.
//! 2. [`matching`]: choosing a matcher, the ratio test, and the matching path of `openmvg-sfm`.
//! 3. [`model`]: the checked keypoint, feature and match values that `openmvg-sfm` stores.
//! 4. [`tracking`]: tracks built from pairwise matches, the rule that keeps a wrong match from
//!    joining two scene points, and the scale check that drops inconsistent tracks.
//!
//! From a clone of the repository, two examples build synthetic images with a known truth and
//! check the results: `detect_and_compare` runs ALP and SIFT on the same blobs, measures how
//! precisely each locates their centers, and checks the scale ALP reports; `match_and_filter`
//! matches two views of a scene with HNSW and shows what the ratio test removes. Run them with
//! `cargo run --release -p openmvg-features --example <name>`.
//!
//! ## Features
//!
//! | Feature | Default | Effect |
//! |---|---|---|
//! | `std` | yes | Matching, tracking, the checked `model` values, and their standard-library dependencies |
//! | `parallel` | yes | Rayon in the detectors, in `HnswMatcher::query_all_stored_with_ef`, and in the track scale check; results do not depend on the thread count |
//! | `libm` | no | `no_std + alloc` math through `libm`, with `default-features = false`; extraction only |
//!
//! For a `no_std + alloc` build:
//!
//! ```toml
//! [dependencies]
//! openmvg-features = { version = "0.1", default-features = false, features = ["libm"] }
//! ```
//!
//! Randomized APIs take explicit seeds; the crate never reads host entropy. The binary Hamming
//! distance uses hardware popcount, native on aarch64 and WASM. On x86-64, build with
//! `-C target-cpu=x86-64-v2` so the compiler emits `POPCNT` instead of a software sequence.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// The root docs link the `std` modules; documentation builds use the default `std` profile.
#![cfg_attr(not(feature = "std"), allow(rustdoc::broken_intra_doc_links))]
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

#[cfg(all(not(feature = "std"), not(feature = "libm")))]
compile_error!("openmvg-features without `std` requires the `libm` feature; enable it explicitly");

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;

#[macro_use]
mod macros;

pub(crate) mod math_compat;

pub mod extraction;

#[cfg(feature = "std")]
pub mod matching;

#[cfg(feature = "std")]
pub mod model;

#[cfg(feature = "std")]
pub mod tracking;

// Compile the README code examples.
#[cfg(all(doctest, feature = "std"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
