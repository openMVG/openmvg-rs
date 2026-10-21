// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature matching: find each descriptor's nearest neighbors in another image, then keep the
//! unambiguous matches.
//!
//! Matching two images takes two steps:
//!
//! 1. A matcher returns, for each query descriptor of the first image, its two nearest targets
//!    in the second.
//! 2. [`MatchAcceptance`] keeps a match when the nearest target is clearly closer than the
//!    second, Lowe's ratio test. Its default, [`MatchAcceptance::Reciprocal`], also runs the
//!    test from the target side and requires the target to choose the query back, so the
//!    matches are one-to-one and do not depend on which image is the query; `openmvg-sfm` does
//!    this. [`MatchAcceptance::OneWay`], like [`LoweRatioFilter`], is the ratio test alone.
//!
//! # Choosing a matcher
//!
//! | Matcher | Search | Choose it |
//! |---|---|---|
//! | [`HnswMatcher`] | approximate, over a graph built once per image | by default; `openmvg-sfm` uses it |
//! | [`BruteForceMatcher`] | exact, every pair | to check another matcher, or when the cost of comparing every pair is acceptable |
//!
//! The cost of exact matching grows with the product of the two images' descriptor counts,
//! while [`HnswMatcher`] searches an index instead. For `u8` descriptors,
//! [`BruteForceMatcher::match_exact_u8`] does the whole exact path in one call: the nearest
//! neighbors, the ratio test, and the acceptance rule. The reciprocal check works with any
//! matcher through [`MatchAcceptance::filter_squared`]: its closure asks the index of the first
//! image for the nearest neighbors of the targets, as the Example below does with HNSW.
//!
//! # Examples
//!
//! The path `openmvg-sfm` takes: `u8` descriptors, one HNSW index per image, and reciprocal
//! acceptance with a ratio of 0.8.
//!
//! ```rust
//! use openmvg_features::matching::{
//!     HnswMatcher, MatchAcceptance, MatcherError, SquaredL2DistanceU8,
//! };
//!
//! // The descriptors of two images, as byte slices. From `extract_u8` they come as
//! // `features.descriptors().iter().map(|d| d.as_array().as_slice())`, with no trait import.
//! let a = [[0_u8, 0, 0, 0], [50, 50, 50, 50], [100, 0, 100, 0]];
//! let b = [[1_u8, 0, 0, 0], [52, 50, 49, 50], [100, 1, 100, 0]];
//! let a: Vec<&[u8]> = a.iter().map(|d| d.as_slice()).collect();
//! let b: Vec<&[u8]> = b.iter().map(|d| d.as_slice()).collect();
//!
//! // One index per image: the forward queries search `b`, the reverse ones `a`.
//! let mut index_a: HnswMatcher<u8, SquaredL2DistanceU8> = HnswMatcher::new();
//! let mut index_b: HnswMatcher<u8, SquaredL2DistanceU8> = HnswMatcher::new();
//! index_a.build(&a)?;
//! index_b.build(&b)?;
//!
//! let forward = index_b.query_batch(&a, 2)?;
//! // The ratio is stated on plain distances; `filter_squared` squares it to compare the
//! // squared distances of the metric.
//! let matches = MatchAcceptance::Reciprocal.filter_squared(&forward, 0.8, |targets| {
//!     // The targets that passed the forward test, searched for in `a`.
//!     let rows: Vec<&[u8]> = targets.iter().map(|&target| b[target]).collect();
//!     index_a.query_batch(&rows, 2)
//! })?;
//! assert_eq!(matches.len(), 3);
//! assert!(matches.iter().all(|m| m.query_index() == m.target_index()));
//! # Ok::<(), MatcherError>(())
//! ```
//!
//! # Distance metrics
//!
//! The distance metrics compute similarity between feature descriptors.
//! Callers use the portable APIs below; 128-dimensional `f32` and `u8`
//! squared-L2 calls select the AVX2 kernel automatically on supported x86-64
//! hosts and otherwise use the scalar fallback.
//!
//! For most use cases, use the standalone functions:
//!
//! ```rust
//! use openmvg_features::matching::{squared_l2_f32, l2_f32};
//!
//! let a = vec![1.0f32, 2.0, 3.0];
//! let b = vec![4.0f32, 5.0, 6.0];
//!
//! // Default nearest-neighbor metric: same ordering as L2, without the sqrt.
//! let sq_dist = squared_l2_f32(&a, &b);
//!
//! // For actual distance measurement
//! let dist = l2_f32(&a, &b);
//! ```
//!
//! ## Binary descriptors
//!
//! For binary descriptors stored as bytes, use the Hamming distance:
//!
//! ```rust
//! use openmvg_features::matching::hamming_u8;
//!
//! let first = vec![0b1111_0000_u8; 32]; // a 256-bit binary descriptor
//! let second = vec![0b1111_1111_u8; 32];
//!
//! assert_eq!(hamming_u8(&first, &second), 128);
//! ```
//!
//! ## Quantized descriptors (u8)
//!
//! The `u8` descriptors of the detectors share one scale, so their bytes compare directly
//! with the `u8` functions:
//!
//! ```rust
//! use openmvg_features::matching::{squared_l2_u8, l2_u8, L2DistanceU8};
//!
//! // Quantized SIFT descriptors (128 dimensions, u8 values)
//! let desc1 = vec![128u8; 128];
//! let desc2 = vec![129u8; 128];
//!
//! // Integer squared distance (for NN search ordering)
//! let sq_dist = squared_l2_u8(&desc1, &desc2);
//!
//! // L2 distance (sqrt of integer distance)
//! let dist = l2_u8(&desc1, &desc2);
//! ```
//!
//! ## Trait-based API
//!
//! For generic code, use the [`DistanceMetric`] trait implementations:
//!
//! ```rust
//! use openmvg_features::matching::{L2Distance, SquaredL2Distance, DistanceMetric};
//!
//! let metric = SquaredL2Distance::new();
//! let dist = metric.distance(&[1.0f32, 2.0], &[3.0, 4.0]);
//! ```
//!
//! ## Performance
//!
//! | Function | Cost | Use case |
//! |----------|------|----------|
//! | `squared_l2_*` | No square root | Float descriptor NN search |
//! | `l2_*` | One extra square root per call | When the actual distance is needed |
//! | `squared_l2_u8` | Integer arithmetic | Quantized descriptor NN search |
//! | `hamming_u8` | XOR and popcount | Binary descriptor matching |

mod acceptance;
pub(crate) mod distance;
mod filters;
mod matchers;
mod traits;
mod types;

pub use acceptance::MatchAcceptance;
pub use distance::{
    HammingDistance, L2Distance, L2DistanceU8, SquaredL2Distance, SquaredL2DistanceU8, hamming_u8,
    l2_f32, l2_f64, l2_u8, squared_l2_f32, squared_l2_f64, squared_l2_u8,
};
pub use filters::LoweRatioFilter;
pub use matchers::{BruteForceMatcher, HnswConfig, HnswMatcher};
// The seed `HnswConfig::with_seed` takes, so that a features-only user can build one. Shown as a
// link to openmvg-core, whose page documents it.
#[doc(no_inline)]
pub use openmvg_core::RunSeed;
pub use traits::{DistanceMetric, FeatureMatcher};
pub use types::{FilteredMatch, Match, MatcherError};
