// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Contracts of the swappable pipeline stages: the extractor, pair proposer, matcher, verifier,
//! triangulator, and pose-registrar traits, the per-call contexts passed to them, and the
//! validated rotation and camera-center estimates returned by averaging. The bundle-adjustment
//! contract lives in `ba::backend`.

mod averaging;
// Front-end stages run only through `image-io` preparation and the unstable API.
#[cfg_attr(
    not(any(feature = "image-io", feature = "unstable-internals")),
    allow(dead_code)
)]
mod features;
#[cfg_attr(
    not(any(feature = "image-io", feature = "unstable-internals")),
    allow(dead_code)
)]
mod matching;
mod registration;
mod triangulation;
#[cfg_attr(
    not(any(feature = "image-io", feature = "unstable-internals")),
    allow(dead_code)
)]
mod verification;

pub use averaging::{
    CameraCenters, RotationAveragingFn, RotationEstimates, TranslationAveragingFn,
};
pub use features::{FeatureContext, FeatureExtractor};
pub use matching::{
    PairContext, PairMatcher, PairMatcherPreparation, PairProposer, PreparedPairMatcher,
};
pub use registration::{
    PoseRegistrar, PoseRegistrationContext, PoseRegistrationEstimate, PoseRegistrationObservation,
};
pub use triangulation::{
    PosedPixelObservation, TriangulatedLandmark, TriangulationRejectReason, TriangulationVerdict,
    Triangulator,
};
pub use verification::{GeometricVerifier, VerificationContext};
