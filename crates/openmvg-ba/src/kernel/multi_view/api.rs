// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public data model for dense-Schur multi-view bundle adjustment.

#![deny(missing_docs)]

use std::error::Error;
use std::fmt;

use crate::numerics::refine::RefinementError;
#[cfg(feature = "unstable-internals")]
use crate::numerics::refine::StepRejections;
#[cfg(feature = "unstable-internals")]
use crate::numerics::refine::nonlinear::DEFAULT_INITIAL_DAMPING;
#[cfg(feature = "unstable-internals")]
use crate::numerics::refine::{Loss, NoiseModel, RefinementStatus};

use super::super::PinholeCamera;
#[cfg(feature = "unstable-internals")]
use super::super::RelativePose;

/// Uniform Jacobian dispatch chart for camera poses.
///
/// Uniform workspace constructors explicitly anchor camera zero and select
/// one specialized Jacobian/retraction kernel for the complete solve. The
/// per-camera-mask constructor can enable camera zero and restrict individual
/// rotation/center coordinates within this chart; its exact masks are available
/// through the workspace's `camera_pose_masks` getter. Poses selected through
/// [`FixedVariableBlocks`](super::FixedVariableBlocks) remain fixed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum CameraPoseVariables {
    /// Optimize the ordinary six-dimensional `SE(3)` pose blocks.
    #[default]
    RotationAndTranslation,
    /// Keep every rotation bit-exact and optimize only camera translations.
    TranslationOnly,
}

/// Treatment of the monocular scale gauge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleGauge {
    /// Do not constrain scale. The final camera Hessian is expected to have
    /// exactly one null direction unless fixed blocks or finite priors constrain
    /// scale. Ordinary covariance requires the resulting information to be full
    /// rank.
    Free,
    /// Preserve the translation norm of one non-anchor camera. That camera is
    /// refined on `SO(3) x S2`; every other variable camera uses `SO(3) x R3`.
    FixedBaseline {
        /// Non-anchor camera whose translation norm is preserved.
        camera: usize,
    },
    /// Preserve one Cartesian translation component of a non-anchor camera.
    ///
    /// This is the subset-parameterization gauge commonly used by generic BA
    /// frameworks. Unlike [`FixedBaseline`](Self::FixedBaseline), it leaves the
    /// other two translation components in a Euclidean chart.
    #[cfg_attr(
        not(any(test, feature = "unstable-internals")),
        allow(dead_code, reason = "constructed by the oracle gauge tests")
    )]
    FixedTranslationComponent {
        /// Non-anchor camera containing the fixed component.
        camera: usize,
        /// Translation component kept at its input value.
        component: TranslationComponent,
    },
}

/// A Cartesian component of a camera translation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    not(any(test, feature = "unstable-internals")),
    allow(dead_code, reason = "constructed by the oracle gauge tests")
)]
pub enum TranslationComponent {
    /// Horizontal component.
    X,
    /// Vertical component.
    Y,
    /// Depth component.
    Z,
}

impl TranslationComponent {
    pub(super) const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// Kind of bundle-adjustment variable block referenced by a typed error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BundleAdjustmentBlock {
    /// Camera rotation and translation.
    WorldToCameraPose,
    /// Euclidean 3D point.
    Point,
    /// Radial `(log focal, k1, k2)` parameters.
    RadialIntrinsics,
}

impl fmt::Display for BundleAdjustmentBlock {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::WorldToCameraPose => "camera-pose",
            Self::Point => "point",
            Self::RadialIntrinsics => "radial-intrinsic",
        })
    }
}

/// One pixel observation in a point-major track array.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelObservation {
    /// Zero-based camera index.
    pub camera: usize,
    /// Measured pixel coordinates.
    pub point: [f64; 2],
}

impl PixelObservation {
    /// Construct a camera-indexed pixel observation.
    pub const fn new(camera: usize, point: [f64; 2]) -> Self {
        Self { camera, point }
    }
}

/// One calibrated bearing observation in a point-major track array.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularObservation {
    /// Zero-based camera index.
    pub camera: usize,
    /// Finite nonzero observed bearing.
    pub bearing: [f64; 3],
}

impl AngularObservation {
    /// Construct a camera-indexed bearing observation.
    pub const fn new(camera: usize, bearing: [f64; 3]) -> Self {
        Self { camera, bearing }
    }
}

/// Sparse visibility stored in compressed point-row form.
///
/// `point_offsets` has `points.len() + 1` entries. Observations for point `i`
/// occupy `point_offsets[i]..point_offsets[i + 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MultiViewMeasurements<'a> {
    /// Exact `S2` logarithm residuals in radians.
    Angular {
        /// Point-major bearing observations.
        observations: &'a [AngularObservation],
        /// Compressed point-row offsets into `observations`.
        point_offsets: &'a [usize],
    },
    /// Standard fixed-intrinsics pinhole residuals in pixels.
    Pixel {
        /// Point-major pixel observations.
        observations: &'a [PixelObservation],
        /// Compressed point-row offsets into `observations`.
        point_offsets: &'a [usize],
        /// One fixed pinhole model per camera pose.
        cameras: &'a [PinholeCamera],
    },
    /// Exact fixed Core pinhole or Brown-Conrady residuals in pixels.
    FixedPerspective {
        /// Point-major pixel observations.
        observations: &'a [PixelObservation],
        /// Compressed point-row offsets into `observations`.
        point_offsets: &'a [usize],
        /// One fixed Core model per camera pose; only observed models must be perspective.
        cameras: &'a [openmvg_core::camera::Camera],
    },
    /// Fixed Core perspective models selected independently for every residual.
    /// Physical pose indices remain in `PixelObservation.camera`; repeated
    /// pose/point rows are independent measurements and are all retained.
    FixedPerspectiveIndexed {
        /// Point-major pixel observations, indexed by physical pose.
        observations: &'a [PixelObservation],
        /// Compressed point-row offsets into `observations`.
        point_offsets: &'a [usize],
        /// Fixed camera models; their count is independent of the pose count.
        cameras: &'a [openmvg_core::camera::Camera],
        /// One index into `cameras` per observation, in the same point-major order.
        observation_intrinsics: &'a [usize],
    },
}

impl<'a> MultiViewMeasurements<'a> {
    /// Number of camera observations across all tracks.
    pub const fn observation_count(self) -> usize {
        match self {
            Self::Angular { observations, .. } => observations.len(),
            Self::Pixel { observations, .. }
            | Self::FixedPerspective { observations, .. }
            | Self::FixedPerspectiveIndexed { observations, .. } => observations.len(),
        }
    }

    /// Number of point tracks represented by the CSR offsets.
    ///
    /// This is structural only: malformed offsets are reported when a
    /// workspace is constructed or a solve begins.
    #[cfg(feature = "unstable-internals")]
    pub const fn point_count(self) -> usize {
        self.point_offsets().len().saturating_sub(1)
    }

    pub(super) const fn point_offsets(self) -> &'a [usize] {
        match self {
            Self::Angular { point_offsets, .. }
            | Self::Pixel { point_offsets, .. }
            | Self::FixedPerspective { point_offsets, .. }
            | Self::FixedPerspectiveIndexed { point_offsets, .. } => point_offsets,
        }
    }

    pub(super) const fn observation_intrinsics(self) -> Option<&'a [usize]> {
        match self {
            Self::FixedPerspectiveIndexed {
                observation_intrinsics,
                ..
            } => Some(observation_intrinsics),
            _ => None,
        }
    }

    #[cfg(any(feature = "faer", test))]
    pub(super) const fn observation_camera(self, observation: usize) -> usize {
        match self {
            Self::Angular { observations, .. } => observations[observation].camera,
            Self::Pixel { observations, .. }
            | Self::FixedPerspective { observations, .. }
            | Self::FixedPerspectiveIndexed { observations, .. } => {
                observations[observation].camera
            }
        }
    }
}

/// Zero-copy calibrated multi-view problem description.
///
/// The view borrows point-major measurements and optional per-observation
/// noise. Poses, points, solver scratch, and LM controls remain explicit at
/// the solve call, so constructing or copying this value never allocates.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(feature = "unstable-internals")]
pub struct CalibratedBundleAdjustmentProblem<'a> {
    measurements: MultiViewMeasurements<'a>,
    gauge: ScaleGauge,
    noise: NoiseModel<'a>,
}

#[cfg(feature = "unstable-internals")]
impl<'a> CalibratedBundleAdjustmentProblem<'a> {
    /// Borrow a calibrated problem with unit residual noise.
    #[cfg(feature = "unstable-internals")]
    pub const fn new(measurements: MultiViewMeasurements<'a>, gauge: ScaleGauge) -> Self {
        Self {
            measurements,
            gauge,
            noise: NoiseModel::Unit,
        }
    }

    /// Attach shared or per-observation residual standard deviations.
    #[cfg(feature = "unstable-internals")]
    pub const fn with_noise(mut self, noise: NoiseModel<'a>) -> Self {
        self.noise = noise;
        self
    }

    /// Borrowed point-major measurements.
    #[cfg(feature = "unstable-internals")]
    pub const fn measurements(self) -> MultiViewMeasurements<'a> {
        self.measurements
    }

    /// Selected monocular scale treatment.
    #[cfg(feature = "unstable-internals")]
    pub const fn gauge(self) -> ScaleGauge {
        self.gauge
    }

    /// Residual-noise model applied during whitening.
    #[cfg(feature = "unstable-internals")]
    pub const fn noise(self) -> NoiseModel<'a> {
        self.noise
    }

    /// Number of observations in all point tracks.
    #[cfg(feature = "unstable-internals")]
    pub const fn observation_count(self) -> usize {
        self.measurements.observation_count()
    }

    /// Number of point tracks described by the CSR offsets.
    #[cfg(feature = "unstable-internals")]
    pub const fn point_count(self) -> usize {
        self.measurements.point_count()
    }
}

/// Residual evidence for one point-major observation.
///
/// The residual is always in the measurement model's native tangent units:
/// pixels for pinhole measurements and radians for angular measurements.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg(feature = "unstable-internals")]
pub struct MultiViewObservationResidual {
    /// Prediction minus measurement in the model's two residual coordinates.
    pub residual: [f64; 2],
    /// Euclidean residual norm divided by this observation's noise sigma.
    pub standardized_norm: f64,
}

/// Observation-only quality evidence for one 3D point track.
///
/// Conditioning uses the same analytic projection Jacobian, whitening, and
/// robust IRLS weight as bundle adjustment. Priors are excluded
/// so this describes geometric track support rather than external constraints.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(feature = "unstable-internals")]
pub struct MultiViewTrackDiagnostics {
    /// Number of observations represented by these diagnostics.
    ///
    /// Ordinary problem evaluation uses the complete track. Lifecycle
    /// evaluation uses only observations retained by its residual gate.
    pub observation_count: usize,
    /// Root-mean-square of the track's raw residual coordinates.
    pub rmse: f64,
    /// Largest noise-standardized two-dimensional residual norm in the track.
    pub maximum_standardized_residual_norm: f64,
    /// Reciprocal infinity-norm condition estimate of the 3x3 point Hessian.
    pub point_hessian_reciprocal_condition_number: f64,
    /// Whether the point Hessian is singular at the solver's numerical threshold.
    pub weak: bool,
}

#[cfg(feature = "unstable-internals")]
impl Default for MultiViewTrackDiagnostics {
    fn default() -> Self {
        Self {
            observation_count: 0,
            rmse: 0.0,
            maximum_standardized_residual_norm: 0.0,
            point_hessian_reciprocal_condition_number: 0.0,
            // Empty or uninitialized evidence must never look well constrained.
            weak: true,
        }
    }
}

/// Aggregate observation and track quality at a fixed BA state.
#[must_use = "inspect residual and conditioning evidence before accepting tracks"]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg(feature = "unstable-internals")]
pub struct MultiViewBundleAdjustmentEvaluation {
    /// Robust observation objective after noise whitening.
    pub robust_cost: f64,
    /// Root-mean-square of all raw residual coordinates.
    pub rmse: f64,
    /// Largest noise-standardized two-dimensional residual norm.
    pub maximum_standardized_residual_norm: f64,
    /// Number of numerically singular observation-only point Hessians.
    pub weak_point_count: usize,
    /// Worst reciprocal condition estimate among point Hessians.
    pub minimum_point_hessian_reciprocal_condition_number: f64,
}

/// Recommended lifecycle state for one point track after residual filtering.
///
/// This is evidence for an SfM controller; bundle adjustment never applies the
/// recommendation or changes the visibility graph itself.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg(feature = "unstable-internals")]
pub enum MultiViewTrackClassification {
    /// Every observation passed the residual gate and the point block is well
    /// conditioned.
    Stable,
    /// At least two observations remain after rejection, so the point should
    /// be triangulated again.
    Retriangulate,
    /// The current views do not geometrically constrain a stable triangulation.
    WeakGeometry,
    /// Fewer than two observations remain after residual filtering.
    #[default]
    InsufficientSupport,
}

/// Aggregate result of classifying observations and point-track lifecycle.
#[must_use = "inspect rejected observations and track recommendations"]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg(feature = "unstable-internals")]
pub struct MultiViewTrackLifecycleSummary {
    /// Observations at or below the standardized-residual threshold.
    pub retained_observation_count: usize,
    /// Observations above the threshold or with non-finite diagnostics.
    pub rejected_observation_count: usize,
    /// Well-conditioned tracks whose observations all passed the gate.
    pub stable_track_count: usize,
    /// Supported tracks recommended for triangulation from retained
    /// observations.
    pub retriangulation_track_count: usize,
    /// Tracks whose point Hessian exposes numerically weak viewing geometry.
    pub weak_geometry_track_count: usize,
    /// Tracks with fewer than two retained observations.
    pub insufficient_support_track_count: usize,
}

/// Result and numerical evidence for one independently triangulated track.
///
/// The point is returned as a candidate; triangulation never mutates the
/// caller's map. This lets an SfM front end apply lifecycle, cheirality, and
/// application-specific acceptance policy outside the optimizer core.
#[must_use = "inspect status and conditioning before inserting the point"]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(feature = "unstable-internals")]
pub struct MultiViewTriangulationReport {
    /// Candidate point in the world frame.
    pub point: [f64; 3],
    /// Explicit convergence or degeneracy status.
    pub status: RefinementStatus,
    /// Number of retained observations used by the track.
    pub observation_count: usize,
    /// LM iterations attempted after the linear ray-intersection initializer.
    pub iteration_count: usize,
    /// Accepted nonlinear steps.
    pub accepted_step_count: usize,
    /// Rejected nonlinear steps.
    pub rejected_step_count: usize,
    /// Robust objective at the linear initializer.
    pub initial_cost: f64,
    /// Robust objective at the returned point.
    pub final_cost: f64,
    /// Final point-gradient norm scaled by diagonal curvature.
    pub gradient_inf_norm: f64,
    /// Final accepted step norm relative to the point magnitude.
    pub step_norm: f64,
    /// Reciprocal infinity-norm condition estimate of the final point Hessian.
    pub point_hessian_reciprocal_condition_number: f64,
}

#[cfg(feature = "unstable-internals")]
impl MultiViewTriangulationReport {
    /// Return whether this candidate has enough finite, conditioned,
    /// non-increasing numerical evidence for an SfM owner to consider it.
    ///
    /// This remains a numerical test rather than an insertion:
    /// cheirality, track identity, and map mutation stay caller-owned.
    #[cfg(feature = "unstable-internals")]
    pub fn is_usable_proposal(&self) -> bool {
        let status_usable = match self.status {
            RefinementStatus::Converged(_) => true,
            RefinementStatus::MaxIterations => {
                self.iteration_count > 0 && self.accepted_step_count > 0
            }
            RefinementStatus::Stalled | RefinementStatus::Degenerate => false,
        };
        let cost_tolerance =
            64.0 * f64::EPSILON * self.initial_cost.abs().max(self.final_cost.abs()).max(1.0);
        status_usable
            && self.observation_count >= 2
            && self.point.iter().all(|value| value.is_finite())
            && self.initial_cost.is_finite()
            && self.final_cost.is_finite()
            && self.gradient_inf_norm.is_finite()
            && self.step_norm.is_finite()
            && self.point_hessian_reciprocal_condition_number > 0.0
            && self.final_cost <= self.initial_cost + cost_tolerance
    }
}

#[cfg(feature = "unstable-internals")]
impl Default for MultiViewTriangulationReport {
    fn default() -> Self {
        Self {
            point: [0.0; 3],
            status: RefinementStatus::Degenerate,
            observation_count: 0,
            iteration_count: 0,
            accepted_step_count: 0,
            rejected_step_count: 0,
            initial_cost: f64::INFINITY,
            final_cost: f64::INFINITY,
            gradient_inf_norm: f64::INFINITY,
            step_norm: 0.0,
            point_hessian_reciprocal_condition_number: 0.0,
        }
    }
}

/// Residual evidence and retained-track lifecycle evidence from one pass.
#[must_use = "inspect both observation quality and track recommendations"]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg(feature = "unstable-internals")]
pub struct MultiViewTrackLifecycleEvaluation {
    /// Quality of the complete, unfiltered observation set at the supplied
    /// state.
    pub evaluation: MultiViewBundleAdjustmentEvaluation,
    /// Classification after applying the standardized-residual gate.
    pub lifecycle: MultiViewTrackLifecycleSummary,
}

/// Rank and conditioning evidence for the final undamped, diagonally
/// equilibrated Schur system.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(any(test, feature = "unstable-internals"))]
pub struct MultiViewBundleAdjustmentDiagnostics {
    /// Dimension of the reduced camera system.
    pub pose_dimension: usize,
    /// Whether the final undamped reduced camera Hessian rank was assessed.
    pub pose_rank_assessed: bool,
    /// Numerical rank of the final undamped reduced camera Hessian.
    ///
    /// This is zero when [`Self::pose_rank_assessed`] is false.
    pub pose_rank: usize,
    /// Null directions implied by the selected gauge, fixed blocks, and prior
    /// information at the final state.
    pub expected_gauge_nullity: usize,
    /// Smallest observable/largest eigenvalue ratio after diagonal
    /// equilibration and exclusion of expected gauge nulls.
    pub observable_camera_hessian_reciprocal_condition_number: f64,
    /// Whether final eliminated point-block ranks were assessed.
    pub point_rank_assessed: bool,
    /// Number of eliminated 3x3 point blocks that were numerically singular.
    ///
    /// This is zero when [`Self::point_rank_assessed`] is false.
    pub weak_point_count: usize,
    /// Worst reciprocal condition estimate among the point blocks.
    pub minimum_point_hessian_reciprocal_condition_number: f64,
}

#[cfg(any(test, feature = "unstable-internals"))]
impl MultiViewBundleAdjustmentDiagnostics {
    /// Nullity beyond the direction implied by the selected gauge.
    ///
    /// This returns zero when camera rank was not assessed. Inspect
    /// [`Self::pose_rank_assessed`] when affirmative rank evidence is needed.
    #[cfg(feature = "unstable-internals")]
    pub const fn additional_camera_nullity(self) -> usize {
        if self.pose_rank_assessed {
            self.pose_dimension
                .saturating_sub(self.pose_rank + self.expected_gauge_nullity)
        } else {
            0
        }
    }

    /// Whether the camera Hessian contains only the expected gauge nullity.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn camera_has_only_expected_nullity(self) -> bool {
        self.pose_rank_assessed
            && self.pose_rank
                == self
                    .pose_dimension
                    .saturating_sub(self.expected_gauge_nullity)
    }

    /// Whether no assessed camera or point degeneracy was detected.
    ///
    /// As with sparse BA, explicitly unassessed rank evidence does not turn a
    /// numerically successful optimization into a degenerate result.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn is_well_constrained(self) -> bool {
        (!self.pose_rank_assessed || self.camera_has_only_expected_nullity())
            && (!self.point_rank_assessed || self.weak_point_count == 0)
    }
}

/// Outcome of an in-place multi-view solve.
#[must_use = "inspect status and diagnostics before accepting the BA solution"]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(feature = "unstable-internals")]
pub struct MultiViewBundleAdjustmentReport {
    /// Loss used by the solve.
    pub loss: Loss,
    /// Explicit numerical solution status.
    pub status: RefinementStatus,
    /// Scale-gauge treatment used by the solve.
    pub scale_gauge: ScaleGauge,
    /// Finite positive world-coordinate scale used by the local parameterization.
    pub normalization_scale: f64,
    /// Preserved baseline norm, or `None` for another gauge choice.
    pub fixed_baseline: Option<f64>,
    /// Preserved translation component, or `None` for other gauge choices.
    pub fixed_translation_component: Option<f64>,
    /// Number of camera poses.
    pub pose_count: usize,
    /// Number of refined 3D points.
    pub point_count: usize,
    /// Number of camera observations.
    pub observation_count: usize,
    /// Final measurements outside the camera model's unambiguous inverse
    /// domain.
    ///
    /// A nonzero count does not invalidate the forward BA objective: robust
    /// outliers may legitimately remain. Downstream triangulation should use
    /// lifecycle decisions or otherwise exclude these measurements.
    pub noninvertible_measurement_count: usize,
    /// LM iterations attempted.
    pub iteration_count: usize,
    /// Trial steps accepted.
    pub accepted_step_count: usize,
    /// Trial steps rejected.
    pub rejected_step_count: usize,
    /// The rejected steps by reason; their total is `rejected_step_count`.
    pub step_rejections: StepRejections,
    /// Initial robust objective.
    pub initial_cost: f64,
    /// Final robust objective.
    pub final_cost: f64,
    /// Initial unweighted residual root-mean-square.
    pub initial_rmse: f64,
    /// Final unweighted residual root-mean-square.
    pub final_rmse: f64,
    /// Maximum full camera-or-point gradient divided by the square root of its
    /// corresponding diagonal Gauss--Newton curvature.
    ///
    /// This is NaN when final diagnostics were explicitly disabled.
    pub gradient_inf_norm: f64,
    /// Root-sum-square of camera and dimensionless point-step RMS values.
    pub step_norm: f64,
    /// Initial dimensionless Nielsen damping value used by this solve.
    pub initial_damping: f64,
    /// Final dimensionless Nielsen damping value.
    pub final_damping: f64,
    /// Gain ratio from the last evaluated finite trial step, if available.
    pub last_gain_ratio: Option<f64>,
    /// Final undamped rank and conditioning evidence.
    pub diagnostics: MultiViewBundleAdjustmentDiagnostics,
}

/// Invalid input or workspace for multi-view bundle adjustment.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiViewBundleAdjustmentError {
    /// Workspace camera dimensions overflowed `usize`.
    CameraCapacityOverflow,
    /// Initial LM damping is not finite and strictly positive.
    #[cfg(feature = "unstable-internals")]
    InitialDampingInvalid,
    /// Too few cameras were supplied.
    InsufficientCameras {
        /// Minimum accepted camera count.
        required: usize,
        /// Supplied camera count.
        actual: usize,
    },
    /// Too few 3D points were supplied.
    InsufficientPoints {
        /// Minimum accepted point count.
        required: usize,
        /// Supplied point count.
        actual: usize,
    },
    /// A camera rotation is not a finite member of `SO(3)`.
    InvalidRotation {
        /// Invalid camera index.
        camera: usize,
    },
    /// A camera translation contains NaN or infinity.
    NonFiniteTranslation {
        /// Invalid camera index.
        camera: usize,
    },
    /// Camera zero is not the required identity world frame.
    AnchorCameraNotIdentity,
    /// A 3D point component contains NaN or infinity.
    NonFinitePoint {
        /// Invalid point index.
        point: usize,
        /// Invalid coordinate index.
        coordinate: usize,
    },
    /// CSR offsets are not monotone, bounded, or of the required length.
    InvalidTrackOffsets,
    /// A free point has fewer than two measurement rows.
    InsufficientTrackObservations {
        /// Track's point index.
        point: usize,
        /// Number of observations in the track.
        actual: usize,
    },
    /// An observation refers to a camera outside the pose slice.
    InvalidObservationCamera {
        /// Invalid observation index.
        observation: usize,
        /// Out-of-range camera index.
        camera: usize,
    },
    /// An observation component contains NaN or infinity.
    NonFiniteMeasurement {
        /// Invalid observation index.
        observation: usize,
    },
    /// An angular observation is zero length or otherwise invalid.
    InvalidBearing {
        /// Invalid observation index.
        observation: usize,
    },
    /// Fixed or per-view intrinsic count does not match camera count.
    IntrinsicsCountMismatch {
        /// Required number of camera models.
        expected: usize,
        /// Supplied number of camera models.
        actual: usize,
    },
    /// Per-observation intrinsic selectors do not match the residual row count.
    ObservationIntrinsicCountMismatch {
        /// Number of residual rows requiring a selector.
        expected: usize,
        /// Number of supplied selectors.
        actual: usize,
    },
    /// A residual selects a fixed camera model outside its model table.
    InvalidObservationIntrinsic {
        /// Residual row index.
        observation: usize,
        /// Invalid fixed-camera model index.
        intrinsic: usize,
        /// Number of fixed-camera models.
        intrinsic_count: usize,
    },
    /// Grouped camera-to-intrinsic map does not match camera count.
    IntrinsicGroupMapCountMismatch {
        /// Required number of map entries.
        expected: usize,
        /// Supplied number of map entries.
        actual: usize,
    },
    /// Grouped self-calibration has zero or more groups than cameras.
    InvalidIntrinsicGroupCount {
        /// Number of camera poses.
        cameras: usize,
        /// Supplied number of intrinsic groups.
        groups: usize,
    },
    /// A camera refers to an intrinsic group outside the supplied slice.
    InvalidIntrinsicGroup {
        /// Camera containing the invalid map entry.
        camera: usize,
        /// Invalid group index.
        group: usize,
        /// Number of supplied groups.
        groups: usize,
    },
    /// An intrinsic group is not referenced by any camera.
    UnusedIntrinsicGroup {
        /// Unused group index.
        group: usize,
    },
    /// A pinhole or radial camera contains invalid intrinsic parameters.
    InvalidIntrinsics {
        /// Camera index, or intrinsic-group index for grouped calibration.
        camera: usize,
    },
    /// The selected gauge camera is the anchor or outside the pose slice.
    InvalidGaugeCamera {
        /// Invalid gauge camera index.
        camera: usize,
    },
    /// The selected fixed baseline is zero length or non-finite.
    InvalidGaugeBaseline {
        /// Invalid gauge camera index.
        camera: usize,
    },
    /// The selected fixed translation component is zero or non-finite.
    InvalidGaugeTranslationComponent {
        /// Invalid gauge camera index.
        camera: usize,
        /// Invalid component.
        component: TranslationComponent,
    },
    /// A fixed-block mask is neither empty nor the required length.
    FixedBlockCountMismatch {
        /// Kind of fixed block.
        block: BundleAdjustmentBlock,
        /// Required number of entries.
        expected: usize,
        /// Supplied number of entries.
        actual: usize,
    },
    /// A radial intrinsic block frees a number of parameters outside `1..=3`.
    InvalidIntrinsicDimension {
        /// Invalid intrinsic block index.
        block: usize,
        /// Requested number of free `(log focal, k1, k2)` parameters.
        dimension: usize,
    },
    /// A gauge constraint targets an already fixed camera pose.
    GaugeCameraIsFixed {
        /// Redundantly constrained camera.
        camera: usize,
    },
    /// A prior refers to a missing or exactly fixed block.
    InvalidPriorTarget {
        /// Kind of prior block.
        block: BundleAdjustmentBlock,
        /// Invalid block index.
        index: usize,
    },
    /// A prior mean or square-root information matrix is invalid.
    InvalidPrior {
        /// Kind of prior block.
        block: BundleAdjustmentBlock,
        /// Invalid block index.
        index: usize,
    },
    /// Prior blocks are duplicated or not strictly increasing by index.
    PriorOrder {
        /// Kind of prior block.
        block: BundleAdjustmentBlock,
        /// Previous block index.
        previous: usize,
        /// Current block index, which must be greater than `previous`.
        current: usize,
    },
    /// A 3D point lies outside a camera's projection domain.
    PointNotProjectable {
        /// Point index.
        point: usize,
        /// Camera index.
        camera: usize,
    },
    /// An observation-residual slice does not match the observation count.
    #[cfg(feature = "unstable-internals")]
    ObservationResidualCountMismatch {
        /// Required output entries.
        expected: usize,
        /// Supplied output entries.
        actual: usize,
    },
    /// A track-diagnostic slice does not match the point count.
    #[cfg(feature = "unstable-internals")]
    TrackDiagnosticCountMismatch {
        /// Required output entries.
        expected: usize,
        /// Supplied output entries.
        actual: usize,
    },
    /// A triangulation-report slice does not match the point count.
    #[cfg(feature = "unstable-internals")]
    TriangulationReportCountMismatch {
        /// Required output entries.
        expected: usize,
        /// Supplied output entries.
        actual: usize,
    },
    /// Observation-inlier output length does not match the observation count.
    #[cfg(feature = "unstable-internals")]
    ObservationInlierMaskCountMismatch {
        /// Required output entries.
        expected: usize,
        /// Supplied output entries.
        actual: usize,
    },
    /// Track-classification output length does not match the point count.
    #[cfg(feature = "unstable-internals")]
    TrackClassificationCountMismatch {
        /// Required output entries.
        expected: usize,
        /// Supplied output entries.
        actual: usize,
    },
    /// Track-selection input length does not match the point count.
    #[cfg(feature = "unstable-internals")]
    TrackSelectionCountMismatch {
        /// Required selection entries.
        expected: usize,
        /// Supplied selection entries.
        actual: usize,
    },
    /// The workspace cannot hold all camera poses.
    WorkspaceCameraCapacity {
        /// Required camera capacity.
        required: usize,
        /// Allocated camera capacity.
        capacity: usize,
    },
    /// The workspace cannot hold all 3D points.
    WorkspacePointCapacity {
        /// Required point capacity.
        required: usize,
        /// Allocated point capacity.
        capacity: usize,
    },
    /// Shared refinement options or noise configuration are invalid.
    Configuration(RefinementError),
}

impl fmt::Display for MultiViewBundleAdjustmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CameraCapacityOverflow => formatter.write_str("camera capacity is too large"),
            #[cfg(feature = "unstable-internals")]
            Self::InitialDampingInvalid => {
                formatter.write_str("initial damping must be finite and positive")
            }
            Self::InsufficientCameras { required, actual } => write!(
                formatter,
                "multi-view bundle adjustment requires at least {required} cameras, received {actual}"
            ),
            Self::InsufficientPoints { required, actual } => write!(
                formatter,
                "multi-view bundle adjustment requires at least {required} points, received {actual}"
            ),
            Self::InvalidRotation { camera } => {
                write!(formatter, "camera {camera} rotation is not in SO(3)")
            }
            Self::NonFiniteTranslation { camera } => {
                write!(formatter, "camera {camera} translation is not finite")
            }
            Self::AnchorCameraNotIdentity => formatter.write_str(
                "camera 0 is the fixed world frame and must have identity rotation and zero translation",
            ),
            Self::NonFinitePoint { point, coordinate } => write!(
                formatter,
                "3D point {point}, coordinate {coordinate} is not finite"
            ),
            Self::InvalidTrackOffsets => formatter.write_str(
                "point offsets must start at zero, be monotone, and end at the observation count",
            ),
            Self::InsufficientTrackObservations { point, actual } => write!(
                formatter,
                "point {point} requires at least two observations, received {actual}"
            ),
            Self::InvalidObservationCamera {
                observation,
                camera,
            } => write!(
                formatter,
                "observation {observation} refers to nonexistent camera {camera}"
            ),
            Self::NonFiniteMeasurement { observation } => {
                write!(formatter, "observation {observation} is not finite")
            }
            Self::InvalidBearing { observation } => write!(
                formatter,
                "observation {observation} bearing must be finite and nonzero"
            ),
            Self::ObservationIntrinsicCountMismatch { expected, actual } => write!(
                formatter, "expected {expected} observation intrinsic selectors, got {actual}",
            ),
            Self::InvalidObservationIntrinsic { observation, intrinsic, intrinsic_count } => write!(
                formatter, "observation {observation} selects intrinsic {intrinsic} outside {intrinsic_count} fixed models",
            ),
            Self::IntrinsicsCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} camera intrinsics, received {actual}"
            ),
            Self::IntrinsicGroupMapCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} camera-to-intrinsic-group entries, received {actual}"
            ),
            Self::InvalidIntrinsicGroupCount { cameras, groups } => write!(
                formatter,
                "{cameras} cameras require between one and {cameras} intrinsic groups, received {groups}"
            ),
            Self::InvalidIntrinsicGroup {
                camera,
                group,
                groups,
            } => write!(
                formatter,
                "camera {camera} refers to intrinsic group {group}, but only {groups} groups exist"
            ),
            Self::UnusedIntrinsicGroup { group } => {
                write!(formatter, "intrinsic group {group} is not assigned to any camera")
            }
            Self::InvalidIntrinsics { camera } => write!(
                formatter,
                "camera {camera} intrinsics must be finite with positive focal lengths"
            ),
            Self::InvalidGaugeCamera { camera } => write!(
                formatter,
                "fixed-baseline camera {camera} must be a non-anchor camera in the pose slice"
            ),
            Self::InvalidGaugeBaseline { camera } => write!(
                formatter,
                "fixed-baseline camera {camera} must have a finite nonzero translation"
            ),
            Self::InvalidGaugeTranslationComponent { camera, component } => write!(
                formatter,
                "fixed translation component {component:?} of camera {camera} must be finite and nonzero"
            ),
            Self::FixedBlockCountMismatch {
                block,
                expected,
                actual,
            } => write!(
                formatter,
                "expected either zero or {expected} {block} fixed-block flags, received {actual}"
            ),
            Self::InvalidIntrinsicDimension { block, dimension } => write!(
                formatter,
                "radial intrinsic block {block} frees {dimension} parameters; expected 1, 2, or 3"
            ),
            Self::GaugeCameraIsFixed { camera } => write!(
                formatter,
                "scale-gauge camera {camera} cannot also be an exactly fixed pose"
            ),
            Self::InvalidPriorTarget { block, index } => write!(
                formatter,
                "{block} prior target {index} is outside the problem or exactly fixed"
            ),
            Self::InvalidPrior { block, index } => {
                write!(formatter, "{block} prior {index} contains an invalid mean or square-root information matrix")
            }
            Self::PriorOrder {
                block,
                previous,
                current,
            } => write!(
                formatter,
                "{block} priors must be strictly ordered; index {current} follows {previous}"
            ),
            Self::PointNotProjectable { point, camera } => write!(
                formatter,
                "point {point} is outside the projection domain of camera {camera}"
            ),
            #[cfg(feature = "unstable-internals")]
            Self::ObservationResidualCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} observation-residual outputs, received {actual}"
            ),
            #[cfg(feature = "unstable-internals")]
            Self::TrackDiagnosticCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} track-diagnostic outputs, received {actual}"
            ),
            #[cfg(feature = "unstable-internals")]
            Self::TriangulationReportCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} triangulation-report outputs, received {actual}"
            ),
            #[cfg(feature = "unstable-internals")]
            Self::ObservationInlierMaskCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} observation-inlier outputs, received {actual}"
            ),
            #[cfg(feature = "unstable-internals")]
            Self::TrackClassificationCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} track-classification outputs, received {actual}"
            ),
            #[cfg(feature = "unstable-internals")]
            Self::TrackSelectionCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} track-selection entries, received {actual}"
            ),
            Self::WorkspaceCameraCapacity { required, capacity } => write!(
                formatter,
                "workspace camera capacity is {capacity}; solve requires {required}"
            ),
            Self::WorkspacePointCapacity { required, capacity } => write!(
                formatter,
                "workspace point capacity is {capacity}; solve requires {required}"
            ),
            Self::Configuration(error) => error.fmt(formatter),
        }
    }
}

impl Error for MultiViewBundleAdjustmentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Configuration(error) => Some(error),
            _ => None,
        }
    }
}

impl From<RefinementError> for MultiViewBundleAdjustmentError {
    fn from(error: RefinementError) -> Self {
        Self::Configuration(error)
    }
}

/// Reusable storage for the dynamic dense camera system.
///
/// [`try_new`](Self::try_new) allocates once. Refinement calls using a
/// sufficiently large workspace do not allocate.
#[cfg(any(test, feature = "unstable-internals"))]
pub struct DenseBundleAdjustmentWorkspace {
    pub(super) camera_capacity: usize,
    pub(super) point_capacity: usize,
    pub(super) camera_offsets: Vec<usize>,
    pub(super) camera_dimensions: Vec<usize>,
    pub(super) camera_pose_dimensions: Vec<usize>,
    pub(super) intrinsic_offsets: Vec<usize>,
    pub(super) intrinsic_dimensions: Vec<usize>,
    #[cfg(feature = "unstable-internals")]
    pub(super) candidate_poses: Vec<RelativePose>,
    #[cfg(feature = "unstable-internals")]
    pub(super) candidate_points: Vec<[f64; 3]>,
    pub(super) hessian: Vec<f64>,
    pub(super) factor: Vec<f64>,
    pub(super) gradient: Vec<f64>,
    pub(super) full_gradient: Vec<f64>,
    pub(super) step: Vec<f64>,
    pub(super) diagonal: Vec<f64>,
    pub(super) cross: Vec<f64>,
    pub(super) active_parameters: Vec<usize>,
    pub(super) parameter_marks: Vec<bool>,
    #[cfg(feature = "unstable-internals")]
    pub(super) initial_damping: f64,
    pub(super) exact_camera_rank_diagnostics: bool,
    pub(super) final_diagnostics: bool,
}

#[cfg(any(test, feature = "unstable-internals"))]
impl DenseBundleAdjustmentWorkspace {
    /// Allocate scratch for calibrated BA up to the supplied capacities.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn try_new(
        camera_capacity: usize,
        point_capacity: usize,
    ) -> Result<Self, MultiViewBundleAdjustmentError> {
        let camera_dimension = camera_capacity
            .saturating_sub(1)
            .checked_mul(6)
            .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
        Self::try_new_with_dimension(camera_capacity, point_capacity, camera_dimension)
    }

    #[cfg(any(test, feature = "unstable-internals"))]
    pub(super) fn try_new_with_dimension(
        camera_capacity: usize,
        point_capacity: usize,
        camera_dimension: usize,
    ) -> Result<Self, MultiViewBundleAdjustmentError> {
        let matrix_size = camera_dimension
            .checked_mul(camera_dimension)
            .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
        #[cfg(feature = "unstable-internals")]
        let placeholder = RelativePose::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
        Ok(Self {
            camera_capacity,
            point_capacity,
            camera_offsets: vec![0; camera_capacity],
            camera_dimensions: vec![0; camera_capacity],
            camera_pose_dimensions: vec![0; camera_capacity],
            intrinsic_offsets: vec![0; camera_capacity],
            intrinsic_dimensions: vec![0; camera_capacity],
            #[cfg(feature = "unstable-internals")]
            candidate_poses: vec![placeholder; camera_capacity],
            #[cfg(feature = "unstable-internals")]
            candidate_points: vec![[0.0; 3]; point_capacity],
            hessian: vec![0.0; matrix_size],
            factor: vec![0.0; matrix_size],
            gradient: vec![0.0; camera_dimension],
            full_gradient: vec![0.0; camera_dimension],
            step: vec![0.0; camera_dimension],
            diagonal: vec![0.0; camera_dimension],
            cross: vec![0.0; camera_dimension * 3],
            active_parameters: vec![0; camera_dimension],
            parameter_marks: vec![false; camera_dimension],
            #[cfg(feature = "unstable-internals")]
            initial_damping: DEFAULT_INITIAL_DAMPING,
            exact_camera_rank_diagnostics: true,
            final_diagnostics: true,
        })
    }

    /// Select exact final camera-rank and condition-number diagnostics.
    ///
    /// The default is enabled. Disabling this preserves point-block checks and
    /// nonlinear convergence while avoiding the cubic spectral decomposition.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn with_exact_camera_rank_diagnostics(mut self, enabled: bool) -> Self {
        self.exact_camera_rank_diagnostics = enabled;
        self
    }

    /// Whether exact final camera-rank diagnostics are enabled.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn exact_camera_rank_diagnostics(&self) -> bool {
        self.exact_camera_rank_diagnostics
    }

    /// Select final undamped camera and point-rank diagnostics.
    ///
    /// The default is enabled. Disabling this avoids rebuilding the final
    /// undamped Schur system. Nonlinear convergence and final objective metrics
    /// are still evaluated, but rank fields are explicitly marked unassessed.
    /// This setting takes precedence over exact camera-rank diagnostics.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn with_final_diagnostics(mut self, enabled: bool) -> Self {
        self.final_diagnostics = enabled;
        self
    }

    /// Whether final undamped rank diagnostics are enabled.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn final_diagnostics(&self) -> bool {
        self.final_diagnostics
    }

    /// Set the positive finite initial dimensionless LM damping value.
    /// Invalid values are rejected when refinement begins.
    #[cfg(feature = "unstable-internals")]
    pub const fn with_initial_damping(mut self, initial_damping: f64) -> Self {
        self.initial_damping = initial_damping;
        self
    }

    /// Configured initial dimensionless LM damping value.
    #[cfg(feature = "unstable-internals")]
    pub const fn initial_damping(&self) -> f64 {
        self.initial_damping
    }

    /// Maximum number of cameras accepted by this workspace.
    #[cfg(feature = "unstable-internals")]
    pub const fn camera_capacity(&self) -> usize {
        self.camera_capacity
    }

    /// Maximum number of points accepted by this workspace.
    #[cfg(feature = "unstable-internals")]
    pub const fn point_capacity(&self) -> usize {
        self.point_capacity
    }
}
