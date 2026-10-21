// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental SfM pipeline.
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │              Incremental engine (Incremental::run_prepared)                │
//! └────────────────────────────────────────────────────────────────────────────┘
//! Input: ViewGraph workspace with verified pairs and tracks, no poses, no landmarks
//! Entry: incremental_sfm_pretracked_with_events, inside a GeometryRollback
//!   │
//!   ▼
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Initial pair                                                       │
//! └────────────────────────────────────────────────────────────────────────────┘
//! select_initial_pair_for_incremental scores every verified pair (parallax band,
//! rotation-cycle closure, component size, frontier, support) and takes the best.
//! bootstrap_pair_seed: view A = identity, view B = relative pose at unit baseline;
//! triangulate_incremental (2 observations, no reprojection cull);
//! refine_two_view_seed (BA, view A fixed); calibrate_guessed_seed_focals.
//!   │
//!   ▼
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Growth loop (GrowthPass::run), one view per iteration              │
//! └────────────────────────────────────────────────────────────────────────────┘
//! select_registration_candidate_indexed → prepare_view_registration_indexed →
//! evaluate_prepared_registration (PoseRegistrar: robust 2D-3D resection, then the
//! inlier, inlier-ratio and rotation-edge gates) → commit_registration_outcome →
//! retriangulate_incremental_for_views → post-refit support check.
//! A view that fails a gate goes to the `failed` set and is retried only after the
//! frontier is exhausted with net growth, or after a committed BA.
//!   │
//!   ▼
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Periodic BA (try_growth_global_ba), by cadence or relative growth  │
//! └────────────────────────────────────────────────────────────────────────────┘
//! bundle_adjust, then remove_unsupported_poses until stable. A drop in landmarks,
//! a higher bearing RMSE, or more invalid observations or edge violations rolls the
//! whole step back.
//!   │
//!   ▼
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │ Step 4: Final refinement (run_final_refinement)                            │
//! └────────────────────────────────────────────────────────────────────────────┘
//! optional self-calibration BA, then up to 4 rounds of BA plus outlier cleanup
//! (4 px, 2°), pose-support peeling, recover_missing_tracks; rolled back on a
//! quality regression.
//! Output: IncrementalSfmReport with an `IncrementalTermination` and per-view diagnostics
//! ```
//!
//! # Algorithm
//!
//! The engine starts from two views and adds one view at a time, so every pose is anchored to
//! landmarks that earlier views already triangulated. The initial pair fixes the gauge: view A
//! has the identity pose and view B sits at unit baseline from the verified relative pose. Pairs
//! with at least 8 verified inliers (by default) are scored. A pair enters the preferred band when
//! its median triangulation angle over up to 300 shared tracks is at least 3° and its
//! triangle-closure error is at most 5° (a pair in no triangle stays eligible). Within a band the
//! score adds component size, registration frontier, and inlier support, in that priority.
//!
//! The frontier is every unposed view that observes enough active landmarks (12 by default,
//! `min_registration_support`) and has at least one verified pair to a posed view. When the
//! frontier is empty, a second scan accepts views with as many landmarks as the inlier minimum
//! (8 by default). The loop keeps the candidates whose landmark count is at least 76% of the
//! best one, prefers those whose observations cover four or more cells of a 4x4 image grid, and
//! breaks ties by landmark count, verified posed neighbors, track observations, fewer past
//! failures, and `ViewId`. A candidate is resected with a
//! [`PoseRegistrar`](crate::backends::PoseRegistrar). The default [`GenRansacP4pfRegistrar`]
//! estimates the focal length together with the pose when the view's intrinsic group is a guess
//! that no posed view shares and the camera is a pinhole or distortion-free Brown-Conrady
//! model. Every other view goes through [`registration::GenRansacP3pRegistrar`]. The pose is accepted when
//! its inlier count reaches `max(min_inliers, ⌈min_inlier_ratio × observations⌉)` (8 and 0.25 by
//! default) and its rotation agrees with the view's relative-rotation edges within 20°.
//!
//! Bundle adjustment (BA) runs every `bundle_adjust_every` registrations (8 when
//! self-calibration is on and the cadence is 0) and whenever the posed views or active landmarks
//! grew by `bundle_adjust_growth_ratio` (0.1 by default) since the last run.
//!
//! # Determinism
//!
//! Candidate order has a total tie-break on `ViewId`, the failed set is ordered, and each
//! registration draws its RANSAC seed from the workspace seed and the view index.
//!
//! # Failure modes
//!
//! A view that fails a gate is a recorded outcome, not an error: [`RegistrationFailureReason`]
//! names it (`NotEnoughVisibleLandmarks`, `PoseEstimationFailed`, `NotEnoughInliers`,
//! `RotationInconsistent`, `TranslationInconsistent`, `PostRefitInsufficientSupport`,
//! `PostBundleInsufficientSupport`). The run stops with `IncrementalTermination::Complete`,
//! `FrontierExhausted`, `AllCandidatesRejected` or `MaxIterations`, and the report lists each
//! view left unregistered in [`UnregisteredViewDiagnostic`]. The run returns an error for missing
//! tracks or verified models, an invalid configuration,
//! [`SfmError::NoInitialPairCandidate`](crate::SfmError::NoInitialPairCandidate), an initial pair
//! with no triangulatable track, or two views that share one `PoseId`. Any error restores the
//! workspace to its state at entry. A periodic or final BA step whose result regresses the
//! reconstruction (fewer than half the landmarks, a higher bearing RMSE, more invalid
//! observations, or more pose-edge violations) or fails recoverably is rolled back, and the run
//! continues from the state before that step.
//!
//! # Alternatives
//!
//! The Global engine solves all rotations and translations from the view graph in one pass
//! and has no initial pair or growth loop, but it reconstructs only a core of views that passes
//! its consistency checks. The Incremental engine registers each view from landmarks that are
//! already triangulated, and a view it cannot place stays unregistered with a recorded reason.
//! It pays one resection per attempted view and a BA over the growing scene at each cadence step.
//!
//! # Module layout
//!
//! The driver sequences the stages and owns the entry points; `growth` runs the registration
//! loop with periodic bundle adjustment, `refinement` owns the shared bundle-adjustment policy,
//! quality gates, and final refinement, and the neighboring modules own the small policies and
//! receipts used along the way. Reusable cross-pipeline operations live under [`crate::ops`].

mod driver;
mod growth;
pub(crate) mod initial_pair;
pub(crate) mod next_view;
#[cfg(feature = "unstable-test-support")]
mod oracle;
mod refinement;
pub(crate) mod registration;
pub(crate) mod registration_index;
mod report;
pub(crate) mod scene_initializer;
pub(crate) mod transaction;

pub(crate) use driver::incremental_sfm_pretracked_with_events;
#[cfg(feature = "unstable-test-support")]
pub use driver::incremental_sfm_with;
#[cfg(feature = "unstable-internals")]
pub use initial_pair::{
    InitialPairHypothesis, InitialPairReasonCode, InitialPairScoreBreakdown, select_initial_pair,
};
pub use next_view::GrowthConfig;
#[cfg(feature = "unstable-internals")]
pub use next_view::{RegistrationCandidate, registration_candidates};
#[cfg(feature = "unstable-test-support")]
pub use oracle::{incremental_sfm_oracle, incremental_sfm_oracle_with};
#[cfg(feature = "unstable-internals")]
pub use registration::{GenRansacP3pRegistrar, RegistrationReport, register_view};
pub use registration::{
    GenRansacP4pfRegistrar, RegistrationConfig, RegistrationFailure, RegistrationFailureReason,
    RegistrationVerdict,
};
#[cfg(feature = "unstable-test-support")]
pub use report::IncrementalOracleSfmReport;
#[cfg(feature = "unstable-internals")]
pub use report::{
    GrowthRegistrationDiagnostic, IncrementalFinalRefinementReport,
    IncrementalFinalRefinementStatus, IncrementalGrowthReport, IncrementalTermination,
    PostRegistrationSupportDiagnostic,
};
pub use report::{IncrementalSfmReport, UnregisteredViewDiagnostic};
#[cfg(feature = "unstable-internals")]
pub use scene_initializer::IncrementalInitializationReport;
#[cfg(feature = "unstable-test-support")]
pub use scene_initializer::initialize_incremental_seed_from_oracle;
