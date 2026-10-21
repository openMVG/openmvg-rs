// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compact RANSAC/MSAC reference implementation.
//!
//! GenRANSAC is the default. This implementation stays small
//! and readable for teaching and controlled family-level comparisons.
//!
//! References: O. Chum, J. Matas, "Matching with PROSAC - Progressive Sample
//! Consensus", CVPR 2005 (sampler); J. L. Schönberger, V. Larsson, M. Pollefeys,
//! "Fixing the RANSAC Stopping Criterion", arXiv:2503.07829, 2025 (stopping rule).
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                    Classic RANSAC / MSAC loop                        │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: data (n items), solver S (sample size m), threshold τ in the
//!        solver's unit, confidence, max_iterations, optional seed
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Validate                                                     │
//! └──────────────────────────────────────────────────────────────────────┘
//! Check the configuration, m > 0, min_inliers ≥ m, and n ≥ min_inliers.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Draw a minimal sample                                        │
//! └──────────────────────────────────────────────────────────────────────┘
//! m distinct indices, uniform from a seeded generator, or PROSAC prefixes
//! of data sorted best-first.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Solve and score each candidate                               │
//! └──────────────────────────────────────────────────────────────────────┘
//! Every model the solver returns for the sample is scored against all n
//! items. RANSAC score: inlier count. MSAC score: −Σ min(r², τ²). A model
//! replaces the best only if its score is strictly higher.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 4: Terminate                                                    │
//! └──────────────────────────────────────────────────────────────────────┘
//! Uniform sampling: stop at the iteration count the confidence implies for
//! the best inlier ratio (at least 10, at most max_iterations).
//! PROSAC: run all max_iterations.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 5: Check consensus                                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! Output: `RobustEstimate` with the best model, its inlier indices
//! (residual ≤ τ), and the statistics; no refit
//! ```
//!
//! # Problem and method
//!
//! A solver turns `m` data items into zero or more models. Outliers make a
//! least-squares fit on all data wrong, so the loop fits minimal samples and
//! keeps the model that the most data agree with. An item is an inlier when
//! its unsquared residual is at most `τ`. The residual is the solver's
//! `Solver::residual`; the threshold carries the same unit (`Solver::Unit`), and
//! [`ClassicConfig::estimate`] accepts only a threshold in that unit. The image-plane
//! entry points `estimate_homography` and `estimate_fundamental` accept `Pixels` or
//! `NormalizedImage` thresholds, matching the frame of the input points.
//!
//! RANSAC counts inliers. MSAC scores each item by `min(r², τ²)` and keeps the
//! lowest total, so inliers with small residuals count more than inliers near
//! `τ`. Both use the same inlier set (`r ≤ τ`) for the output. The scorers stop
//! scoring a candidate as soon as it cannot beat the best score.
//!
//! With uniform sampling, the number of draws needed after finding a model
//! with `k` inliers among `N` items is `⌈ln(1 − c) / ln(1 − p)⌉` with
//! `p = Π_{i<m} (k − i) / (N − i)`, the probability that a sample drawn without
//! replacement is all inliers. This replaces the `(k/N)^m` approximation. The
//! loop never stops before 10 iterations. PROSAC (`RansacBuilder::prosac`) starts
//! from the best `m` items and grows the pool on a fixed schedule over the
//! whole `max_iterations` budget, so it ignores the confidence bound.
//!
//! Defaults: confidence 0.99, `max_iterations` 1000. Without a seed, the sampler
//! uses a fixed library seed, so a run is repeatable.
//!
//! # Failure modes
//!
//! - `RobustError::InvalidConfig`: `min_inliers` of zero, confidence outside
//!   `(0, 1)`, `max_iterations` of zero, or `min_inliers` below the solver's sample
//!   size.
//! - `RobustError::InsufficientData`: fewer items than the minimum consensus.
//! - `RobustError::AllSamplesDegenerate`: every drawn sample failed or gave
//!   no valid model. `RobustError::NoValidModel`: otherwise no model was kept.
//! - `RobustError::InsufficientConsensus`: the best model has fewer inliers than
//!   `min_inliers` (default: the sample size).
//! - Solver errors for bad input (`InsufficientData`, `InvalidSampleSize`,
//!   `InvalidDataIndex`, `InvalidInput`) abort the run; other solver errors
//!   count as a degenerate sample.
//!
//! # Alternatives
//!
//! Typed facades such as `estimate_homography` use GenRANSAC, which also runs
//! local optimization and a final refit on the consensus set; this loop returns
//! the best minimal-sample model as is. Use this module to read the algorithm, to
//! compare families under the same loop, or to plug an external solver into
//! [`ClassicConfig::estimate`].

mod config;
mod estimator;
mod sampler;
mod scoring;
mod termination;

pub use config::{ClassicConfig, ImagePlaneUnit, RansacBuilder};
