// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bundle adjustment (BA): refine camera poses, optional intrinsics and 3D points together,
//! so that every point reprojects as close as possible to the pixels where it was observed.
//!
//! `openmvg-sfm` runs bundle adjustment inside every reconstruction. Use this crate directly to
//! refine a reconstruction of your own, or to give `openmvg-sfm` another solver through its
//! `Runner`. Poses, cameras and pixels follow the conventions of `openmvg-core`.
//!
//! ## Start here
//!
//! - [`BundleProblem::builder`] takes the initial poses, intrinsics, points and observations,
//!   and the objective: the [`ResidualSpace`], the [`RobustLoss`] and the [`Gauge`].
//! - [`NativeBundleAdjuster`] solves it with Levenberg-Marquardt on the reduced camera system
//!   that [`LinearSolver`] describes.
//! - The returned [`Solution`] holds the refined state and a [`SolveReport`]. The problem stays
//!   unchanged, so you can check the report before you use the result.
//!
//! ## Example
//!
//! Two fixed cameras see nine points; refine the points from a start 5 cm off on each axis:
//!
//! ```rust
//! # #[cfg(feature = "faer")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use nalgebra::{Point3, Rotation3, Vector3};
//! use openmvg_ba::{
//!     BundleAdjuster, BundleIntrinsic, BundleObservation, BundlePoint, BundlePose, BundleProblem,
//!     IntrinsicVariables, NativeBundleAdjuster, PoseVariables, SolveConfig,
//! };
//! use openmvg_core::camera::{Camera, CameraModel};
//! use openmvg_core::model::{CameraIntrinsics, WorldToCameraPose};
//!
//! let camera = Camera::try_pinhole(800.0, 800.0, 320.0, 240.0)?;
//! let intrinsic = CameraIntrinsics::try_new(camera.clone(), 640, 480)?.with_fixed(true);
//! let mut poses = Vec::new();
//! for x in [0.0, 1.0] {
//!     let center = Point3::new(x, 0.0, 0.0);
//!     poses.push(WorldToCameraPose::from_rotation_center(Rotation3::identity(), center)?);
//! }
//! let points: Vec<_> =
//!     (0..9).map(|i| Point3::new(f64::from(i % 3) - 0.5, f64::from(i / 3) - 1.0, 5.0)).collect();
//!
//! // One observation per camera and point: (pose index, intrinsic index, point index, pixel).
//! let mut observations = Vec::new();
//! for (pose_index, pose) in poses.iter().enumerate() {
//!     for (point_index, point) in points.iter().enumerate() {
//!         let pixel = camera.project(&pose.project_to_camera(point).coords)?;
//!         observations.push(BundleObservation::try_new(pose_index, 0, point_index, pixel)?);
//!     }
//! }
//!
//! let offset = Vector3::new(0.05, -0.05, 0.05);
//! let start = points.iter().map(|point| BundlePoint::try_new(point + offset));
//! let problem = BundleProblem::builder(
//!     poses.into_iter().map(|pose| BundlePose::new(pose, PoseVariables::FIXED)).collect(),
//!     vec![BundleIntrinsic::try_new(intrinsic, IntrinsicVariables::FIXED)?],
//!     start.collect::<Result<_, _>>()?,
//!     observations,
//! )
//! .build()?;
//! let solution = NativeBundleAdjuster::new().solve(&problem, SolveConfig::default())?;
//! assert!(solution.report().final_rmse() < 1e-6);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "faer"))]
//! # fn main() {}
//! ```
//!
//! ## Getting good results
//!
//! - Start close. Bundle adjustment polishes a reconstruction; it converges to the nearest
//!   minimum, so poses and points must already come from a good estimate.
//! - Decide the frame and the scale, as [Choosing the gauge](#choosing-the-gauge) explains.
//! - Use a robust loss when observations can still contain wrong matches, with a scale of two
//!   to four times the per-axis pixel noise of good observations. For pixel residuals and a 2 px
//!   scale, pass `RobustLoss::Cauchy(ResidualScale::Pixels(PixelThreshold::try_new(2.0)?))` to
//!   [`BundleProblemBuilder::with_robust_loss`], with `PixelThreshold` from `openmvg_core`.
//!   [`RobustLoss`] says when to prefer Cauchy or Huber, and what scale to choose.
//! - Check the result before you use it, as [Accepting a result](#accepting-a-result) lists.
//!
//! ## Choosing the gauge
//!
//! Poses and points can move, rotate and scale together without changing any residual, so the
//! observations leave seven directions open: three of position, three of orientation, one of
//! scale. Something must fix them:
//!
//! | Situation | Choice | The result has |
//! |---|---|---|
//! | Polishing an estimate, with a known distance between two cameras | [`Gauge::FixedBaseline`], every pose free | the frame of pose `first` and the known scale |
//! | Polishing an estimate, scale unknown | [`Gauge::Free`], every pose free or one pose fixed | the frame and the scale of the start |
//! | Known world coordinates of some points | those [`BundlePoint`]s fixed with [`BundlePoint::with_fixed`], [`Gauge::Free`]; with free intrinsics, also one fixed pose | the frame and the scale of the control points |
//!
//! Do not fix two estimated poses to set the scale. Two held poses fix twelve directions, five
//! more than the gauge needs, and impose their errors on every other camera and point. The
//! `native_bundle_adjustment` example holds two poses only because its poses are exact.
//!
//! ## Accepting a result
//!
//! A solve returns a [`Solution`] even when the result is poor, and the problem stays unchanged,
//! so check the solution before you use it:
//!
//! 1. [`Solution::audit_for_commit`] passes. It rejects a result that is unsafe to use: an
//!    objective that rose, a stalled or degenerate solve, a camera or point that the observations
//!    do not determine, a changed gauge.
//! 2. [`SolveReport::status`] is [`SolveStatus::Converged`]. [`SolveStatus::MaxIterations`]
//!    passes the audit when the objective dropped: the result is usable but not settled, so raise
//!    the iteration limit or start closer. A re-solve after a quarantine (see [Weak and
//!    quarantined points](#weak-and-quarantined-points)) runs at most eight iterations whatever
//!    the limit, so there only a closer start helps. [`SolveStatus::Skipped`] means the input
//!    state came back unchanged.
//! 3. The observability is [`Observability::FullRank`], or [`Observability::OnlyExpectedGauge`]
//!    under [`Gauge::Free`]. [`Observability::NotAssessed`] fails the audit; it happens only
//!    under an explicit workspace budget too small for the check, as [`LinearSolver`] explains.
//! 4. Under a robust loss, [`SolveReport::inlier_count`] is close to the number of observations
//!    you expect to be good, out of [`SolveReport::observation_count`]. Far fewer inliers mean
//!    the start is too far off or the scale too tight; all of them, when you know some matches
//!    are wrong, mean the wrong ones fit within the scale. Under [`RobustLoss::Squared`] every
//!    observation counts as an inlier, so this check says nothing.
//! 5. Then compare [`SolveReport::final_inlier_rmse`] with the per-axis pixel noise you expect: a
//!    good fit leaves it about √2 times that noise, since a residual norm adds two axes. Under a
//!    robust loss it cannot exceed the scale, by construction, so it looks good on its own even
//!    when the fit is not. The plain [`SolveReport::final_rmse`] counts every observation, so a
//!    few wrong matches dominate it.
//!
//! To see which observations the robust loss set aside, compare
//! [`Solution::observation_residuals`] with the loss scale.
//!
//! These checks cannot see every failure. A wrong match that happens to lie along its epipolar
//! line can fit the cameras and pass all of them, leaving its point misplaced; remove wrong
//! matches before bundle adjustment where you can.
//!
//! ## Weak and quarantined points
//!
//! A *weak* point is one the observations cannot place: it is seen once or never, it sits on an
//! observing camera center, it is seen along rays without parallax, it is seen by cameras that
//! cannot project it, or its observations constrain it too poorly to eliminate it from the
//! camera system. [`Solution::weak_points`] lists the weak points of any solution.
//!
//! The native adjuster *quarantines* weak points: it removes those it finds before the solve,
//! solves again without any that appear after it, and leaves them all at their input position.
//! [`Solution::quarantined_points`] lists them, the report's error figures leave out their
//! observations, and [`Solution::audit_for_commit`] accepts them; their landmarks must not be
//! trusted. A weak point that remains in the solved system, counted by
//! [`SolveDiagnostics::weak_point_count`], fails the audit.
//!
//! When every point is weak before the solve, nothing is left to adjust and the native adjuster
//! returns [`SolveError::Unsupported`].
//!
//! ## Common mistakes
//!
//! - Expecting a known scale without giving one. With [`Gauge::Free`] and nothing fixed, the
//!   adjuster holds an anchor pose and a camera distance at their starting values, so the result
//!   has the frame and scale of the start.
//! - Reading [`Solution::audit_for_commit`] as a quality check. It confirms that the result is
//!   safe to use; the status and the inlier error say whether it is good.
//! - Giving pixels whose origin is the top-left pixel corner. Pixel centers sit on integer
//!   coordinates, as in `openmvg-core`.
//! - Expecting bundle adjustment to remove wrong matches by itself. Without a robust loss, each
//!   wrong match pulls the solution with the full least-squares weight.
//!
//! ## Learning path
//!
//! 1. [`BundleProblem`]: the variables (poses, intrinsics, points), the observations, and the
//!    objective ([`ResidualSpace`], [`RobustLoss`], [`Gauge`]).
//! 2. [`NativeBundleAdjuster`]: how the problem is solved, and the [`LinearSolver`] choices for
//!    the camera system.
//! 3. [`Solution`] and [`SolveReport`]: the refined state, the convergence status, and the
//!    rank and stationarity checks.
//! 4. [`BundleAdjuster`]: the trait another solver implements to replace the native one.
//! 5. [`numerics`]: forward-mode jets, rotation and pose manifolds, and the Levenberg-Marquardt
//!    loop for refiners of your own.
//!
//! From a clone of the repository, two examples check their own results:
//! `native_bundle_adjustment` refines two perturbed cameras and every point, and
//! `robust_loss_and_gauge` adds wrong matches, compares plain least squares with a Cauchy loss,
//! fixes the gauge with a baseline, and runs the checks of [Accepting a
//! result](#accepting-a-result). Run them with
//! `cargo run --release -p openmvg-ba --example <name>`.
//!
//! ## Features
//!
//! | Feature | Default | Effect |
//! |---|---|---|
//! | `faer` | yes | The native adjuster and its sparse factorization |
//! | `parallel` | no | Rayon in the native adjuster; results do not depend on the thread count |
//! | `unstable-internals` | no | Low-level kernels; not for applications, outside semver |
//!
//! Without `faer` the crate keeps the problem, solution and report types and the
//! [`BundleAdjuster`] trait, but has no native adjuster.
//!
// Without `faer` the native adjuster does not exist: its links on this page point to the
// feature table instead. The blank line above ends the paragraph, so this is a link definition.
#![cfg_attr(
    not(feature = "faer"),
    doc = "[`NativeBundleAdjuster`]: crate#features"
)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
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

#[cfg(feature = "unstable-internals")]
pub mod kernel;
#[cfg(not(feature = "unstable-internals"))]
mod kernel;
#[cfg(feature = "faer")]
mod native;
pub mod numerics;
mod problem;
mod quarantine;
mod solver;

#[cfg(feature = "faer")]
pub use native::NativeBundleAdjuster;
// The pre-solve certificate that openmvg-sfm reads; not part of the documented API.
#[cfg(feature = "faer")]
#[doc(hidden)]
pub use native::{NativePreflight, NativePreflightStatus};
pub use numerics::refine::{StepRejection, StepRejections};
pub use problem::{
    BundleIntrinsic, BundleObservation, BundlePoint, BundlePose, BundleProblem,
    BundleProblemBuilder, Gauge, IntrinsicCertification, IntrinsicVariables, PoseVariables,
    ProblemError, ResidualScale, ResidualSpace, RobustLoss,
};
pub use solver::{
    AuditError, BundleAdjuster, ConvergenceReason, LinearSolver, Observability,
    ObservabilityDiagnostics, Solution, SolutionError, SolveConfig, SolveDiagnostics, SolveError,
    SolveReport, SolveStatus, StationarityAudit,
};

// Compile the README example.
#[cfg(all(doctest, feature = "faer"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
