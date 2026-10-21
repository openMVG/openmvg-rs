// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Two-view verification contract: [`GeometricVerifier`] and its per-pair context.

use openmvg_core::{RunSeed, camera::Camera};

use crate::two_view::{PairMatches, VerificationVerdict};
use crate::{Result, ViewId, ViewPair};
use openmvg_features::model::ExtractedFeatures;

/// Inputs passed to a [`GeometricVerifier`] for one matched pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VerificationContext {
    /// Canonical pair being verified.
    pub pair: ViewPair,
    /// First endpoint (`pair.first()`, the smaller view ID).
    pub view_i: ViewId,
    /// Second endpoint (`pair.second()`).
    pub view_j: ViewId,
    /// Pair-local stream seed for robust estimation, derived from the run seed.
    pub seed: RunSeed,
}

/// Estimates the two-view geometry of a matched pair and classifies its matches.
pub trait GeometricVerifier: Send + Sync {
    /// Verify `matches` between `fi` and `fj` with the cameras of the two views.
    ///
    /// Both acceptance and rejection are `Ok`; `Err` is reserved for invalid input or an
    /// implementation failure.
    fn verify_pair(
        &self,
        ctx: &VerificationContext,
        fi: &ExtractedFeatures,
        fj: &ExtractedFeatures,
        matches: &PairMatches,
        camera_i: &Camera,
        camera_j: &Camera,
    ) -> Result<VerificationVerdict>;
}
