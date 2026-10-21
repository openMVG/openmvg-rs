// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked measurements, policy, errors, and diagnostics.

use std::error::Error;
use std::fmt;

use nalgebra::Vector3;
use openmvg_core::RunSeed;

/// One checked-input measurement for fixed-rotation global positioning.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum PositioningObservation {
    /// World-frame ray from a camera toward a point.
    Bearing {
        /// Dense camera index in `0..camera_count`.
        camera: usize,
        /// Dense point index in `0..point_count`.
        point: usize,
        /// Finite, nonzero world-frame ray.
        world_bearing: Vector3<f64>,
        /// Finite positive prior weight.
        weight: f64,
    },
    /// Oriented world-frame camera-to-camera direction.
    RelativeDirection {
        /// Source camera index.
        source: usize,
        /// Target camera index.
        target: usize,
        /// Finite, nonzero direction from `source` to `target`.
        source_to_target: Vector3<f64>,
        /// Finite positive prior weight before family-mass balancing.
        weight: f64,
    },
}

impl PositioningObservation {
    /// Construct a camera-to-point bearing measurement.
    #[must_use]
    pub const fn bearing(
        camera: usize,
        point: usize,
        world_bearing: Vector3<f64>,
        weight: f64,
    ) -> Self {
        Self::Bearing {
            camera,
            point,
            world_bearing,
            weight,
        }
    }

    /// Construct an oriented camera-to-camera direction measurement.
    #[must_use]
    pub const fn relative_direction(
        source: usize,
        target: usize,
        source_to_target: Vector3<f64>,
        weight: f64,
    ) -> Self {
        Self::RelativeDirection {
            source,
            target,
            source_to_target,
            weight,
        }
    }
}

/// Borrowed measurements used to prepare one immutable positioning plan.
#[derive(Clone, Copy, Debug)]
pub struct GlobalPositioningProblem<'a> {
    /// Number of dense camera variables.
    pub camera_count: usize,
    /// Number of dense point variables.
    pub point_count: usize,
    /// Camera-to-point bearings and optional oriented camera-to-camera directions.
    pub observations: &'a [PositioningObservation],
}

/// BATA continuation and robust-estimation policy.
///
/// Defaults use 20 outer iterations, two unweighted warmup iterations,
/// angular Huber widths graduated from `0.5` to `0.05` radians, and a
/// 10-degree hard cut after iteration 15. Alternation derives weights from
/// chord length; polish derives weights from its forward sine residual.
/// Widths preserve the angular transition location in these different charts;
/// their loss curves and the reported angular RMS are distinct quantities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BataOptions {
    /// Maximum robust outer iterations.
    pub max_outer_iterations: usize,
    /// Initial iterations with unit robust weights.
    pub warmup_iterations: usize,
    /// Initial Huber weight transition angle in radians, mapped to chord length
    /// `2 sin(angle/2)` during alternation.
    pub initial_huber_delta_radians: f64,
    /// Final Huber transition angle in radians. Polish maps it to the forward
    /// projected residual width `sin(angle)`. Widths beyond a chart's angular
    /// domain make its weights quadratic throughout that domain.
    pub final_huber_delta_radians: f64,
    /// Enable the usual hard angular/cheirality cut from this zero-based iteration on.
    /// With both measurement families, relative directions use one quarter
    /// of `hard_cut_deg` after warmup, even before this iteration, while cuts
    /// are enabled by the outer-iteration budget.
    ///
    /// The graduated Huber width reaches its final value at this iteration, and convergence is
    /// only accepted from it on, so the final robust loss has always run. A value of
    /// `max_outer_iterations` or more disables the hard cut; the graduation then ends at the last
    /// iteration. This also disables the angular cut in polish; its finite,
    /// forward, nonzero-baseline projection domain still applies. A final
    /// connectivity fallback keeps the angular cut disabled during polish.
    pub hard_cut_start_iteration: usize,
    /// Hard angular cut in degrees.
    pub hard_cut_deg: f64,
    /// Deterministic seed for the fallback initialization.
    ///
    /// The public adapter forwards [`super::super::GlobalPositioningConfig::seed`].
    pub seed: RunSeed,
    /// Aggregate relative-direction mass relative to bearing-observation mass.
    /// Zero disables all relative-direction constraints.
    pub direction_reweight: f64,
    /// Most projected Gauss-Newton steps run after the alternation, which only has to initialize
    /// them; the polish usually stops earlier on its relative-cost tolerance. Zero disables it.
    pub polish_steps: usize,
}

impl Default for BataOptions {
    fn default() -> Self {
        Self {
            max_outer_iterations: 20,
            warmup_iterations: 2,
            initial_huber_delta_radians: 0.5,
            final_huber_delta_radians: 0.05,
            hard_cut_start_iteration: 15,
            hard_cut_deg: 10.0,
            seed: RunSeed::from_u64(42),
            direction_reweight: 1.0,
            polish_steps: 50,
        }
    }
}

impl BataOptions {
    pub(super) const fn hard_cut_enabled(self) -> bool {
        self.hard_cut_start_iteration < self.max_outer_iterations
    }

    /// Validate the numerical policy and return it unchanged for fluent setup.
    pub fn validate(self) -> Result<Self, GlobalPositioningError> {
        if self.max_outer_iterations == 0
            || !self.initial_huber_delta_radians.is_finite()
            || self.initial_huber_delta_radians <= 0.0
            || !self.final_huber_delta_radians.is_finite()
            || self.final_huber_delta_radians <= 0.0
            || self.final_huber_delta_radians > self.initial_huber_delta_radians
            || !self.hard_cut_deg.is_finite()
            || !(0.0..=180.0).contains(&self.hard_cut_deg)
            || !self.direction_reweight.is_finite()
            || self.direction_reweight < 0.0
        {
            return Err(GlobalPositioningError::InvalidOptions);
        }
        Ok(self)
    }
}

/// Reduced camera-system backend used by the last successful linear solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReducedSystemBackend {
    /// Allocation-reusing dense in-place Cholesky.
    DenseCholesky,
    /// Matrix-free Jacobi-preconditioned conjugate gradients with three fused
    /// coordinate right-hand sides.
    MatrixFreePcg,
    /// Compact materialized scalar couplings with fused three-right-hand-side PCG.
    SparseCouplingPcg,
    /// PCG did not certify its residual and the bounded dense safety path was used.
    DenseFallbackAfterPcg,
    /// Deterministic sparse Cholesky of the scalar camera system, analyzed once per workspace and
    /// solving the three coordinates as one factorization's right-hand sides.
    #[cfg_attr(not(feature = "faer"), allow(dead_code))]
    SparseCholesky,
}

/// Successful BATA solve diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalPositioningReport {
    /// Robust outer iterations completed.
    pub iteration_count: usize,
    /// Projected Gauss-Newton steps accepted after the alternation.
    pub polish_step_count: usize,
    /// Iterations whose hard cuts would have disconnected the active graph and were replaced by
    /// Huber weights for that iteration.
    pub connectivity_fallback_count: usize,
    /// Deterministically selected translation-gauge anchor in the final iteration.
    pub anchor_camera: usize,
    /// Backend used by the final reduced solve.
    pub reduced_backend: ReducedSystemBackend,
    /// Total PCG iterations across reduced solves, excluding direction initialization.
    pub pcg_iteration_count: usize,
    /// Whether deterministic index-parallel passes were used for this solve.
    pub parallel_execution: bool,
    /// Combined RMS angular residual in radians. An undefined zero/nonfinite
    /// baseline receives a right-angle diagnostic penalty. Projected polish
    /// descent does not guarantee that this angular RMS decreases.
    pub rms_angular_error: f64,
    /// Camera-to-point RMS angular residual, absent in direction-only problems.
    /// Undefined baselines receive the same right-angle diagnostic penalty.
    pub bearing_rms_angular_error: Option<f64>,
    /// Camera-to-camera RMS angular residual, absent in bearing-only problems.
    /// Undefined baselines receive the same right-angle diagnostic penalty.
    pub direction_rms_angular_error: Option<f64>,
}

/// Checked-plan and solve failures.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalPositioningError {
    /// Fewer than two cameras were supplied.
    TooFewCameras {
        /// Declared camera count.
        provided: usize,
    },
    /// No active bearing or direction constraint was supplied.
    EmptyProblem,
    /// A configuration field is non-finite, non-positive, or inconsistent.
    InvalidOptions,
    /// A bearing references a camera outside the declared range.
    BearingCameraOutOfBounds {
        /// Index of the invalid observation.
        observation: usize,
    },
    /// A bearing references a point outside the declared range.
    BearingPointOutOfBounds {
        /// Index of the invalid observation.
        observation: usize,
    },
    /// A bearing weight is not finite and positive.
    InvalidBearingWeight {
        /// Index of the invalid observation.
        observation: usize,
    },
    /// A bearing vector is non-finite or numerically zero.
    InvalidBearingVector {
        /// Index of the invalid observation.
        observation: usize,
    },
    /// A camera observes the same point more than once.
    DuplicateBearing {
        /// Index of the second observation for the same camera-point pair.
        observation: usize,
    },
    /// A relative direction references a camera outside the declared range.
    DirectionCameraOutOfBounds {
        /// Index of the invalid direction.
        direction: usize,
    },
    /// A relative direction is a self-loop.
    DirectionSelfLoop {
        /// Index of the invalid direction.
        direction: usize,
    },
    /// A direction weight is not finite and positive.
    InvalidDirectionWeight {
        /// Index of the invalid direction.
        direction: usize,
    },
    /// A direction vector is non-finite or numerically zero.
    InvalidDirectionVector {
        /// Index of the invalid direction.
        direction: usize,
    },
    /// More than one direction was supplied for the same unordered camera pair.
    DuplicateDirection {
        /// Index of the second constraint for the same unordered camera pair.
        direction: usize,
    },
    /// A camera has no active constraint after direction-family reweighting.
    UnconstrainedCamera {
        /// Dense index of the unconstrained camera.
        camera: usize,
    },
    /// A point has fewer than two bearing observations.
    UnderconstrainedPoint {
        /// Dense index of the underconstrained point.
        point: usize,
    },
    /// Constraint counting proves that the problem is underdetermined.
    NecessarilyUnderdetermined,
    /// The joint camera-point graph is disconnected.
    Disconnected,
    /// Caller camera output length does not match the plan.
    CameraOutputCountMismatch {
        /// Required output length.
        expected: usize,
        /// Supplied output length.
        actual: usize,
    },
    /// Caller point output length does not match the plan.
    PointOutputCountMismatch {
        /// Required output length.
        expected: usize,
        /// Supplied output length.
        actual: usize,
    },
    /// The workspace buffer shape does not match the plan.
    WorkspaceMismatch,
    /// Reusable numerical storage could not be reserved.
    WorkspaceAllocationFailed,
    /// A small checked graph is not parallel-rigid under its measured directions.
    NotParallelRigid,
    /// The robust direction-only initialization failed.
    DirectionInitializationFailed,
    /// The reduced camera solve failed its numerical checks.
    LinearSolveFailed,
    /// The solution collapsed to coincident cameras.
    DegenerateSolution,
    /// A final camera or point value is NaN or infinite.
    NonFiniteSolution,
}

impl fmt::Display for GlobalPositioningError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::TooFewCameras { provided } => write!(
                formatter,
                "global positioning needs at least two cameras, got {provided}"
            ),
            Self::EmptyProblem => {
                formatter.write_str("global positioning has no active constraints")
            }
            Self::InvalidOptions => formatter.write_str("invalid BATA options"),
            Self::BearingCameraOutOfBounds { observation } => write!(
                formatter,
                "bearing {observation} has an out-of-range camera"
            ),
            Self::BearingPointOutOfBounds { observation } => {
                write!(formatter, "bearing {observation} has an out-of-range point")
            }
            Self::InvalidBearingWeight { observation } => {
                write!(formatter, "bearing {observation} has an invalid weight")
            }
            Self::InvalidBearingVector { observation } => {
                write!(formatter, "bearing {observation} has an invalid vector")
            }
            Self::DuplicateBearing { observation } => write!(
                formatter,
                "bearing {observation} duplicates a camera-point observation"
            ),
            Self::DirectionCameraOutOfBounds { direction } => write!(
                formatter,
                "direction {direction} has an out-of-range camera"
            ),
            Self::DirectionSelfLoop { direction } => {
                write!(formatter, "direction {direction} is a self-loop")
            }
            Self::InvalidDirectionWeight { direction } => {
                write!(formatter, "direction {direction} has an invalid weight")
            }
            Self::InvalidDirectionVector { direction } => {
                write!(formatter, "direction {direction} has an invalid vector")
            }
            Self::DuplicateDirection { direction } => {
                write!(formatter, "direction {direction} duplicates a camera pair")
            }
            Self::UnconstrainedCamera { camera } => {
                write!(formatter, "camera {camera} has no active constraint")
            }
            Self::UnderconstrainedPoint { point } => {
                write!(formatter, "point {point} has fewer than two observations")
            }
            Self::NecessarilyUnderdetermined => {
                formatter.write_str("global positioning is necessarily underdetermined")
            }
            Self::Disconnected => formatter.write_str("global positioning graph is disconnected"),
            Self::CameraOutputCountMismatch { expected, actual } => write!(
                formatter,
                "camera output count mismatch: expected {expected}, got {actual}"
            ),
            Self::PointOutputCountMismatch { expected, actual } => write!(
                formatter,
                "point output count mismatch: expected {expected}, got {actual}"
            ),
            Self::WorkspaceMismatch => {
                formatter.write_str("global positioning workspace shape mismatch")
            }
            Self::WorkspaceAllocationFailed => {
                formatter.write_str("global positioning workspace allocation failed")
            }
            Self::NotParallelRigid => {
                formatter.write_str("global positioning constraints are not parallel-rigid")
            }
            Self::DirectionInitializationFailed => {
                formatter.write_str("BATA direction initialization failed")
            }
            Self::LinearSolveFailed => formatter.write_str("BATA reduced camera solve failed"),
            Self::DegenerateSolution => {
                formatter.write_str("BATA solution collapsed to coincident cameras")
            }
            Self::NonFiniteSolution => formatter.write_str("BATA produced a non-finite solution"),
        }
    }
}

impl Error for GlobalPositioningError {}
