// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structure from Motion (SfM): turn a set of overlapping photographs into camera poses,
//! camera models and a sparse cloud of 3D points.
//!
//! This crate is the top of the stack. It runs the front end of `openmvg-features` on every
//! image, verifies image pairs with the solvers of `openmvg-geometry`, and reconstructs the
//! scene with one of two engines, both refined by the bundle adjustment of `openmvg-ba`.
//! `openmvg-io` stores the results in the OMVG format or exports them to COLMAP.
//!
//! ```text
//! Dataset::from_dir(...) or Dataset::builder()
//!          │
//!          ▼  ViewGraph::build(&dataset, &Frontend::default())  (image-io),
//!          │  or checkpoint::read_view_graph(...)
//!       ViewGraph
//!          │
//!          ├── Incremental::run_prepared(...)
//!          └── Global::run_prepared(...)
//!          ▼
//!     SfmOutput: Reconstruction + RunReport
//! ```
//!
//! ## Start here
//!
//! - [`Dataset`] is the input: the images and their camera models.
//!   `Dataset::from_dir` reads a folder (feature `image-io`). It takes the camera from a
//!   `K.txt` calibration when there is one; otherwise it groups the images by their EXIF camera,
//!   lens, and focal length and starts from the EXIF focal length, converted to pixels with the
//!   camera's other images when its own EXIF cannot, or from a guess based on the image size when
//!   EXIF gives none. An image without EXIF camera make and model gets a camera of its own.
//! - [`ViewGraph`] is the prepared front end: keypoints, verified image pairs and tracks.
//!   `Frontend` builds it from a dataset, and it is the expensive part, so save it once with
//!   [`checkpoint`] and run the engines on it as often as you like.
//! - [`Incremental`] and [`Global`] are the two engines. Each returns an [`SfmOutput`] holding
//!   the [`Reconstruction`] and a [`RunReport`](reports::RunReport) that explains it.
//!
//! ## Example
//!
//! Reconstruct a folder of images and save the points:
//!
//! ```no_run
//! # #[cfg(feature = "image-io")]
//! # fn main() -> openmvg_sfm::Result<()> {
//! use openmvg_sfm::{Dataset, Frontend, Incremental, ViewGraph};
//!
//! let dataset = Dataset::from_dir("images")?;
//! let graph = ViewGraph::build(&dataset, &Frontend::default())?;
//! let output = Incremental::default().run_prepared(&graph)?;
//!
//! let registered = output.reconstruction().registered_view_count();
//! println!("{registered} of {} views registered", dataset.views().len());
//! output.reconstruction().write_ply("sparse.ply")?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "image-io"))]
//! # fn main() {}
//! ```
//!
//! A dataset can also be described in code, without image files. Each view references one
//! shared camera model ([`IntrinsicId`]) and one pose ([`PoseId`]):
//!
//! ```
//! use openmvg_sfm::model::ImageMeta;
//! use openmvg_sfm::{Camera, Dataset, IntrinsicId};
//!
//! let dataset = Dataset::builder()
//!     .image_meta([ImageMeta::external_id("view-0", 640, 480)])
//!     .camera_models([Camera::try_pinhole(500.0, 500.0, 320.0, 240.0)?])
//!     .shared_intrinsic(IntrinsicId::new(0))
//!     .build()?;
//! assert_eq!(dataset.views().len(), 1);
//! # Ok::<(), openmvg_sfm::SfmError>(())
//! ```
//!
//! ## Choosing an engine
//!
//! [`Incremental`] starts from two views and adds the others one at a time; it tries to register
//! every view and says why it left any out. [`Global`] estimates all rotations, then all
//! positions, from the whole view graph at once, and returns a reconstruction only when one
//! passes its consistency checks. Start with Incremental. Global bundle-adjusts a few times
//! rather than again after each added view, so its run time grows more gently with the number
//! of views. Each engine's page explains how it works and when to prefer it.
//!
//! ## Reading the result
//!
//! [`RunReport::final_route`](reports::RunReport::final_route) says which engine produced the
//! reconstruction, or that Global found none, and
//! [`RunReport::global_empty_reason`](reports::RunReport::global_empty_reason) says why.
//! [`RunReport::view_verdicts`](reports::RunReport::view_verdicts) gives every input view's
//! outcome: registered, with its pose in [`Reconstruction::pose`], or left out, with the
//! reason. Routes and reasons print their names with `Display`, such as
//! `incremental: not_enough_inliers`. The route, the omission reasons and the registration
//! failures print the same names as their JSON; [`GlobalEmptyReason`](reports::GlobalEmptyReason)
//! has names of its own, because the JSON records the engine's more detailed reason instead.
//! The report implements `serde::Serialize`, so `serde_json` writes it as JSON, versioned by
//! its `schema_version` field.
//!
//! ## Saving and reusing work
//!
//! [`checkpoint`] writes and reads the prepared [`ViewGraph`] and the [`Reconstruction`] in a
//! versioned binary file; [`Reconstruction::write_ply`] writes the points and camera centers for
//! a viewer. The OMVG community format and the COLMAP export live in `openmvg-io`.
//!
//! ## Getting good results
//!
//! - Give a `K.txt` calibration when you know the camera: a text file with the nine values of
//!   the 3 × 3 matrix `K`, row by row. Calibrated cameras stay fixed; estimated ones are refined
//!   by every engine.
//! - Start from `Frontend::default()`, the recommended front end, and read its
//!   [`FrontendReport`](reports::FrontendReport) when few pairs verify.
//! - Keep the root seed fixed to repeat a run exactly: the same inputs, configuration and seed
//!   give the same result at every thread count (set it with `RAYON_NUM_THREADS`), and
//!   [`Reconstruction::to_bytes`] lets you compare two results byte for byte. The default is [`Dataset::DEFAULT_SEED`], 42,
//!   which a [`RunSeed`] prints as 64 hexadecimal digits starting `2a00`. The seed belongs to the
//!   [`Dataset`] and drives
//!   the front end as well as the engines, so to see how stable a reconstruction is, build the
//!   dataset with another seed (`DatasetConfig::with_seed`, or [`DatasetBuilder::root_seed`])
//!   and run the front end again.
//!
//! ## Common mistakes
//!
//! - Running the front end again for every engine run instead of saving the [`ViewGraph`].
//! - Comparing two runs that used different seeds or configurations.
//! - Treating a Global run without a reconstruction as an error: it is a valid outcome, with
//!   the reason in the report. Run [`Incremental`] on the same graph instead.
//! - Giving two views the same [`PoseId`] before reconstruction. Stored reconstructions can
//!   share poses, but both engines need one pose per view to start.
//!
//! ## Learning path
//!
//! 1. [`Dataset`] and [`model`]: the views, their camera models and their poses.
//! 2. `Frontend` and [`ViewGraph`]: detection, matching, pair verification and tracks.
//! 3. [`Incremental`] and [`Global`]: the two engines, and [`config`] for their policies.
//! 4. [`SfmOutput`], [`Reconstruction`] and [`reports`]: the result and its explanation.
//! 5. [`checkpoint`]: saving the prepared graph and the result.
//! 6. [`Runner`], [`progress`] and, with the `observer` feature, `events`: progress bars,
//!    cancellation and live visualization.
//! 7. [`ba`]: plugging in another bundle adjuster.
//!
//! From a clone of the repository, two examples check their own results on a folder of
//! images: `engines_and_reports` builds the view graph once, saves it, runs both engines on it,
//! reads their reports and checks that a run repeats exactly, and `progress_and_cancellation`
//! follows a run stage by stage through an [`SfmProgressSink`](progress::SfmProgressSink) and
//! cancels one cleanly. Run them with `cargo run --release -p openmvg-sfm --example <name>
//! --features image-io -- <images>`; `pixi run examples-real` runs both on the Sceaux Castle
//! images.
//!
//! Poses and pixels follow the conventions of `openmvg-core`: world-to-camera poses, and pixel
//! centers on integer coordinates.
//!
//! ## Features
//!
//! | Feature | Default | Effect |
//! |---|---|---|
//! | `parallel` | yes | Rayon in every stage; results do not depend on the thread count |
//! | `image-io` | no | Image discovery, decoding and EXIF metadata, `Dataset::from_dir` and `Frontend` |
//! | `observer` | no | Reconstruction events for live visualization; observing does not change the result |
//! | `console-progress` | no | Terminal progress bars (`progress::ConsoleProgress`); native only |
//! | `unstable-internals` | no | The mutable orchestration layer under `unstable::`; not for applications, outside semver |
//! | `unstable-test-support` | no | Test fixtures and fault-injection seams; implies `unstable-internals`; outside semver |
//!
//! The two engines and everything reachable without `unstable-*` features are the stable v0.1
//! API. Report JSON is versioned by its `schema_version` field, and an engine configuration
//! by the report that embeds it; `ViewGraph` and reconstruction files are versioned by the
//! schema version in their preamble. None of these follows the crate's semver. A serial
//! `wasm32-unknown-unknown` build is supported with `--no-default-features`; randomized APIs
//! take caller-supplied seeds and never read browser entropy.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]
#![deny(missing_docs)]
// The stable façade must not expose types it cannot name. `unstable-internals` opts out, and so
// does the crate's own unit-test build, whose `testing` helpers are not part of the façade.
#![cfg_attr(not(any(test, feature = "unstable-internals")), warn(unnameable_types))]
#![cfg_attr(docsrs, feature(doc_cfg))]
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

#[macro_use]
mod macros;

mod adapters;
mod atomic;
/// Bundle adjustment (BA): the contract an adjuster implements, and the native one.
///
/// Most callers never touch it: [`Incremental::run_prepared`] and [`Global::run_prepared`] use
/// the native adjuster. To plug in another one, implement the index-based contract here and pass
/// it to [`Runner::bundle_adjuster`]; the engines translate their IDs to indices for it.
pub mod ba;
mod backends;
pub mod checkpoint;
/// The policies of the Incremental and Global engines: thresholds, iteration limits and stage
/// switches. Each engine's `Default` is the recommended policy.
pub mod config;
mod determinism;
mod error;
#[cfg(feature = "unstable-internals")]
mod evaluation;
#[cfg(feature = "observer")]
pub mod events;
#[cfg(not(feature = "observer"))]
mod events;
mod ids;
/// Image discovery, decoding, EXIF inspection, and image projects.
#[cfg(feature = "image-io")]
pub mod image;
pub mod model;
mod ops;
mod pipelines;
pub mod progress;
/// What a run and the front end report: the route that produced the reconstruction, every
/// view's outcome, and the front end's per-stage counts. Start with
/// [`RunReport`](reports::RunReport).
pub mod reports;
#[cfg(any(test, feature = "unstable-test-support"))]
pub mod testing;
mod two_view;
mod workflow;
mod workspace;

// Shown as links to openmvg-core, whose pages and examples document them.
#[doc(no_inline)]
pub use openmvg_core::{
    RunSeed,
    camera::Camera,
    model::{RelativePose, WorldToCameraPose},
};

#[cfg(feature = "image-io")]
pub use adapters::features::DetectorDensity;
pub use error::{BundleAdjustmentFailure, BundleAdjustmentFailureKind, Result, SfmError};
pub use ids::{
    ComponentId, FeatureId, IntrinsicId, LandmarkId, PoseId, TrackId, TrackObservationIndex, ViewId,
};
pub use openmvg_core::ViewPair;
#[cfg(feature = "image-io")]
pub use openmvg_features::extraction::FeatureBudgetPolicy;
pub use openmvg_features::{
    model::{FeatureMatch, FeatureSet, Keypoint},
    tracking::{Track, TrackObservation, TrackSet},
};
pub use workflow::run::SfmEngine;
pub use workflow::{
    Dataset, DatasetBuilder, ExtractionMethod, Global, Incremental, PlyFormat, Reconstruction,
    ReconstructionParts, Runner, SfmOutput, ViewGraph, ViewGraphParts,
};
#[cfg(feature = "image-io")]
pub use workflow::{DatasetConfig, Frontend, MatchingMethod};

// The stable façade's boundary, checked as doc tests only: these names must stay unreachable
// without `unstable-internals`.
/// The supported API does not expose the mutable orchestration layer:
///
/// ```compile_fail
/// use openmvg_sfm::SfmWorkspace;
/// ```
///
/// ```compile_fail
/// use openmvg_sfm::SfmWorkspaceBuilder;
/// ```
///
/// ```compile_fail
/// use openmvg_sfm::PairGraph;
/// ```
///
/// ```compile_fail
/// use openmvg_sfm::unstable;
/// ```
///
/// ```compile_fail
/// use openmvg_sfm::adapters;
/// ```
///
/// ```compile_fail
/// use openmvg_sfm::backends;
/// ```
///
/// Stable configuration and reports live in named modules rather than the crate root:
///
/// ```compile_fail
/// use openmvg_sfm::IncrementalSfmConfig;
/// ```
///
/// ```compile_fail
/// use openmvg_sfm::SfmConfig;
/// ```
///
/// ```compile_fail
/// use openmvg_sfm::BundleAdjustConfig;
/// ```
#[cfg(all(doctest, not(feature = "unstable-internals")))]
#[derive(Debug)]
pub struct StableFacadeBoundary;

// Internal code imports the public names above, plus these two, from the crate root; every
// other item is imported from its module.
pub(crate) use ids::PairId;
pub(crate) use workspace::SfmWorkspace;

/// Items public only under `unstable-internals`, outside semver.
///
/// Applications should normally prefer [`crate::Incremental::run_prepared`] and
/// [`crate::Global::run_prepared`]. Engine execution over a prepared workspace lives on
/// [`crate::Runner`] ([`crate::Runner::run_workspace`]); this module groups the mutable workspace,
/// engine policies, reports, backends, and low-level operations by purpose.
///
/// ```
/// use openmvg_sfm::reports::RunReport;
/// use openmvg_sfm::unstable::global::GlobalBaIterationLimits;
/// use openmvg_sfm::unstable::incremental::GenRansacP3pRegistrar;
/// use openmvg_sfm::unstable::{SfmConfig, SfmWorkspace};
/// use openmvg_sfm::{Result, Runner, SfmEngine};
///
/// // The engine's standard algorithms and native bundle adjustment, unless a setter replaces them.
/// fn reconstruct(prepared: SfmWorkspace) -> Result<(SfmWorkspace, RunReport)> {
///     Runner::for_config(SfmConfig::for_engine(SfmEngine::Global)).run_workspace(prepared)
/// }
///
/// let _ = GenRansacP3pRegistrar::default();
/// let _ = GlobalBaIterationLimits::default();
/// let _ = reconstruct;
/// ```
#[cfg(feature = "unstable-internals")]
pub mod unstable {
    pub use crate::ba::config::{BundleAdjustConfig, OptimizationScope};
    pub use crate::pipelines::global::GlobalPreparationReport;
    pub use crate::workflow::run::{GlobalOutcome, SfmConfig, SfmOutcome};
    pub use crate::workspace::SfmWorkspace;
    pub use crate::workspace::builder::SfmWorkspaceBuilder;
    pub use crate::workspace::pair_graph::{Neighbor, PairGraph};
    pub use crate::workspace::selector::{PairSelector, ViewSelector};
    pub use crate::workspace::store::SparseStore;
    pub use crate::workspace::views::{
        FeaturesView, LandmarkSet, MatchesView, PairModelsView, PoseSet,
    };
    // Descriptors enter only through the unstable workspace and backend APIs.
    pub use openmvg_features::model::{DescriptorSet, ExtractedFeatures};

    /// Global SfM policies, reports, and candidate diagnostics.
    pub mod global {
        pub use crate::pipelines::global::{
            CandidateAdmission, CandidateAdmissionEvidence, CandidateRejectionReason,
            FixedRotationObservabilityEvidence, GlobalBaIterationLimits,
            GlobalCandidateAttemptReport, GlobalCandidateDeferral,
            GlobalCandidatePostAdmissionFailure, GlobalCandidateRoute, GlobalCandidateStage,
            GlobalCandidateStatus, GlobalCompletionReport, GlobalFinishReport,
            GlobalNoReconstructionReason, GlobalPolicy, GlobalPoseTrustReport,
            GlobalPostFixedRotationCleanupConfig, GlobalPostFixedRotationCleanupNotAttempted,
            GlobalPostFixedRotationCleanupRejection, GlobalPostFixedRotationCleanupReport,
            GlobalPostFixedRotationCleanupStatus, GlobalRetriangulationRoundReport,
            GlobalSelfCalibrationRejection, GlobalSelfCalibrationReport,
            GlobalSelfCalibrationStatus, GlobalSfmReport, GlobalTrackValidationConfig,
            GlobalTrackValidationReport, GlobalTranslationReport, GlobalTrappedViewDiagnostic,
            MetricCoreCertificationEvidence, MetricCoreCertificationPolicy, PostFitTrustEvidence,
            PostFitTrustPolicy, PretrackedGlobalPositioningConfig, RotationConsensusEvidence,
            RotationConsensusPolicy,
        };
    }

    /// Incremental SfM policies, reports, and custom-engine building blocks.
    pub mod incremental {
        pub use crate::pipelines::incremental::{
            GenRansacP3pRegistrar, GenRansacP4pfRegistrar, GrowthConfig,
            GrowthRegistrationDiagnostic, IncrementalFinalRefinementReport,
            IncrementalFinalRefinementStatus, IncrementalGrowthReport,
            IncrementalInitializationReport, IncrementalSfmReport, IncrementalTermination,
            InitialPairHypothesis, InitialPairReasonCode, InitialPairScoreBreakdown,
            PostRegistrationSupportDiagnostic, RegistrationCandidate, RegistrationConfig,
            RegistrationFailure, RegistrationReport, RegistrationVerdict, register_view,
            registration_candidates, select_initial_pair,
        };
    }

    /// Reusable low-level operations for custom orchestration.
    pub mod ops {
        pub use crate::ops::{
            averaging, frontend, probes, reconstruction, track_scale, view_graph,
        };
    }

    /// Concrete feature, matching, verification, and triangulation backends.
    pub mod adapters {
        pub use crate::adapters::*;
    }

    /// Backend contracts used by custom SfM compositions.
    pub mod backends {
        pub use crate::ba::backend::{
            BaBackendFailure, BaCommitRejection, BaConvergenceReason, BaDiagnostics, BaFailureKind,
            BaNotApplicable, BaObservabilityDiagnostics, BaPreflightDecision,
            BaPreflightDeferralReason, BaPreflightReport, BaRankAssessment, BaSolveStatus,
            BaUnsupportedCapability, BundleAdjustmentReport,
        };
        pub use crate::backends::{
            CameraCenters, FeatureContext, FeatureExtractor, GeometricVerifier, PairContext,
            PairMatcher, PairMatcherPreparation, PairProposer, PoseRegistrar,
            PoseRegistrationContext, PoseRegistrationEstimate, PoseRegistrationObservation,
            PosedPixelObservation, PreparedPairMatcher, RotationEstimates, TriangulatedLandmark,
            TriangulationRejectReason, TriangulationVerdict, Triangulator, VerificationContext,
        };
        pub use crate::ops::reconstruction::bundle::bundle_adjustment_report;
    }

    /// Unstable workspace-only configuration.
    pub mod config {
        pub use crate::config::internal::WorkspaceBuildConfig;
    }

    /// Reusable reconstruction evaluation metrics.
    ///
    /// Pose and trajectory metrics shared by reconstruction tests and custom applications.
    pub mod evaluation {
        pub use crate::evaluation::*;
    }

    /// Workspace-level quality, stage, and manifest reports, produced only by the
    /// unstable [`SfmWorkspace`] API.
    pub mod reports {
        pub use crate::workspace::quality::{
            BearingQualityReport, ErrorStats, QualityReport, ThresholdRate,
        };
        pub use crate::workspace::reports::{
            RejectionSummary, RunManifest, StageReport, WorkspaceReports,
        };
    }

    /// Core SfM data-transfer and reconstruction types.
    pub mod types {
        pub use crate::model::observation_support::ObservationSupport;
        pub use crate::ops::pairs::PairProposalReport;
        pub use crate::ops::tracks::TrackBuildReport;
        pub use crate::two_view::*;
    }
}

// Compile the README example.
#[cfg(all(doctest, feature = "image-io"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
