// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Problem, option, result, and error types of the fixed-rotation observability probe.

use std::error::Error;
use std::fmt;

/// Fixed camera calibration of a fixed-rotation problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedRotationIntrinsics {
    /// Horizontal focal length.
    pub focal_x: f64,
    /// Vertical focal length.
    pub focal_y: f64,
    /// Horizontal principal point.
    pub principal_x: f64,
    /// Vertical principal point.
    pub principal_y: f64,
    /// First radial-distortion coefficient.
    pub k1: f64,
    /// Second radial-distortion coefficient.
    pub k2: f64,
    /// Third radial-distortion coefficient.
    pub k3: f64,
    /// First tangential-distortion coefficient.
    pub p1: f64,
    /// Second tangential-distortion coefficient.
    pub p2: f64,
}

/// One observation in a point-major fixed-rotation track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedRotationObservation {
    /// Camera index.
    pub camera: usize,
    /// Observed pixel coordinates.
    pub xy: [f64; 2],
}

/// Borrowed fixed-rotation problem: held rotations and intrinsics, camera centers, points, and tracks.
#[derive(Clone, Copy, Debug)]
pub struct FixedRotationProblem<'a> {
    /// Fixed world-to-camera rotations.
    pub rotations: &'a [[[f64; 3]; 3]],
    /// Initial camera centers.
    pub centers: &'a [[f64; 3]],
    /// Initial Euclidean points.
    pub points: &'a [[f64; 3]],
    /// Fixed camera intrinsics, one block per camera.
    pub intrinsics: &'a [FixedRotationIntrinsics],
    /// Point-major observation tracks.
    pub observations: &'a [Vec<FixedRotationObservation>],
    /// Whole camera-center blocks fixed by the caller.
    pub fixed_cameras: &'a [bool],
    /// Translation gauge.
    pub gauge: FixedRotationGauge,
}

/// Gauge constraint of a fixed-rotation problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FixedRotationGauge {
    /// Fix one camera center and one coordinate of another camera.
    Free {
        /// Fully fixed camera-center index.
        anchor: usize,
        /// Camera whose strongest baseline coordinate fixes scale.
        scale_camera: usize,
    },
    /// Preserve an explicit camera-center baseline.
    FixedBaseline {
        /// Fully fixed camera-center index.
        anchor: usize,
        /// Second baseline endpoint.
        scale_camera: usize,
        /// Baseline distance; it must be positive and finite, and the probe fixes the same
        /// coordinates as [`FixedRotationGauge::Free`].
        distance: f64,
    },
}

/// Bounded work limits for the sparse fixed-rotation observability probe.
///
/// The probe selects a deterministic camera-covering subset of tracks. A
/// positive result is a valid full-problem certificate because every omitted
/// track contributes another positive-semidefinite Schur term. A negative
/// result is only "not certified", because a capped subset cannot prove
/// that the complete problem is rank deficient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedRotationObservabilityProbeOptions {
    /// Maximum selected point tracks.
    pub max_tracks: usize,
    /// Maximum selected image observations.
    pub max_observations: usize,
    /// Skip individual tracks larger than this bound.
    pub max_observations_per_track: usize,
    /// Larger per-track bound used only when a free camera has no eligible
    /// short track. This bounded coverage exception can certify an otherwise
    /// well-constrained core without admitting arbitrary mega-tracks.
    pub max_coverage_observations_per_track: usize,
    /// Maximum scalar nonzeros in the assembled lower Schur triangle.
    pub max_reduced_nonzeros: usize,
    /// Maximum scalar nonzeros in the symbolic Cholesky factor.
    pub max_factor_nonzeros: usize,
    /// Maximum dense-triangle fill bound admitted before symbolic analysis.
    pub max_symbolic_dense_nonzeros: usize,
    /// Positive normalized eigenvalue margin required by sparse Cholesky.
    pub minimum_normalized_eigenvalue: f64,
}

impl Default for FixedRotationObservabilityProbeOptions {
    fn default() -> Self {
        Self {
            max_tracks: 25_000,
            max_observations: 100_000,
            max_observations_per_track: 32,
            max_coverage_observations_per_track: 256,
            max_reduced_nonzeros: 1_000_000,
            max_factor_nonzeros: 4_000_000,
            max_symbolic_dense_nonzeros: 32_000_000,
            minimum_normalized_eigenvalue: 1e-10,
        }
    }
}

/// Why a capped sparse probe could not certify fixed-rotation observability.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedRotationProbeReason {
    /// The caps could not cover every camera with a free translation coordinate.
    InsufficientCameraCoverage,
    /// The requested sparse pattern or factor exceeded a configured work cap.
    ResourceLimit,
    /// Selected point blocks did not provide usable finite Schur contributions.
    WeakPointGeometry,
    /// The shifted normalized sparse camera system was not positive definite.
    SparseFactorizationFailed,
}

/// Fail-closed outcome of the fixed-rotation observability probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedRotationProbeStatus {
    /// The capped subset alone has a positive-definite gauge-reduced camera Schur system.
    FullRankCertified,
    /// The capped subset did not provide a positive certificate.
    NotCertified(FixedRotationProbeReason),
}

/// Typed diagnostics from the bounded fixed-rotation observability probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedRotationObservabilityProbe {
    /// Fail-closed certification status.
    pub status: FixedRotationProbeStatus,
    /// Gauge-reduced camera dimension tested by the probe.
    pub pose_dimension: usize,
    /// Selected point-track count.
    pub selected_tracks: usize,
    /// Selected image-observation count.
    pub selected_observations: usize,
    /// Cameras with free translation coordinates covered by usable selected tracks.
    pub covered_free_cameras: usize,
    /// Selected point blocks that could not be eliminated.
    pub weak_point_count: usize,
    /// Scalar nonzeros in the assembled lower Schur triangle.
    pub reduced_nonzeros: usize,
    /// Scalar nonzeros required by the symbolic sparse Cholesky factor.
    pub factor_nonzeros: usize,
}

/// Failure to validate a fixed-rotation problem or probe its observability.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedRotationError {
    /// Input dimensions, tracks, or gauge are invalid.
    InvalidProblem(&'static str),
    /// Loss or convergence options are invalid.
    InvalidOptions(&'static str),
    /// The probe met a value it cannot evaluate.
    Numerical(&'static str),
}

impl fmt::Display for FixedRotationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProblem(message)
            | Self::InvalidOptions(message)
            | Self::Numerical(message) => formatter.write_str(message),
        }
    }
}

impl Error for FixedRotationError {}
