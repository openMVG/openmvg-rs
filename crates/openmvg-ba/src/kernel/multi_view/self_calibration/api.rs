// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public state and workspace for opt-in camera self-calibration.

#![deny(missing_docs)]

#[cfg(any(test, feature = "unstable-internals"))]
use super::super::MultiViewBundleAdjustmentError;
use super::super::PixelObservation;
#[cfg(feature = "unstable-internals")]
use super::super::ScaleGauge;
#[cfg(any(test, feature = "unstable-internals"))]
use super::super::dense::DenseBundleAdjustmentWorkspace;
#[cfg(feature = "unstable-internals")]
use crate::numerics::refine::NoiseModel;

/// Centered two-coefficient radial camera model.
///
/// A positive-depth camera point `(X, Y, Z)` maps to
/// `focal * (1 + k1*r² + k2*r⁴) * (X/Z, Y/Z)`.
/// This uses BAL's intrinsic parameterization after converting BAL's
/// negative-depth, negated-image-coordinate convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialCamera {
    /// Positive focal length in pixels.
    pub focal: f64,
    /// Quadratic radial-distortion coefficient.
    pub k1: f64,
    /// Quartic radial-distortion coefficient.
    pub k2: f64,
}

impl RadialCamera {
    /// Construct a centered radial camera.
    pub const fn new(focal: f64, k1: f64, k2: f64) -> Self {
        Self { focal, k1, k2 }
    }

    /// Invert a centered radial measurement into a unit camera bearing.
    ///
    /// Returns `None` for invalid camera state, non-finite measurements, or
    /// measurements outside the central orientation-preserving branch.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn bearing(self, measurement: [f64; 2]) -> Option<[f64; 3]> {
        super::model::radial_bearing(self, &measurement)
    }

    pub(crate) fn is_valid(self) -> bool {
        self.focal.is_finite() && self.focal > 0.0 && self.k1.is_finite() && self.k2.is_finite()
    }
}

/// Sparse point-major pixel tracks for camera self-calibration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfCalibratingMeasurements<'a> {
    pub(super) observations: &'a [PixelObservation],
    pub(super) point_offsets: &'a [usize],
    pub(super) signed_depth: bool,
}

impl<'a> SelfCalibratingMeasurements<'a> {
    /// Construct measurements from point-major CSR tracks.
    ///
    /// `point_offsets` must have one more entry than the point slice supplied
    /// to the solver. Each track must observe at least two distinct cameras.
    pub const fn new(observations: &'a [PixelObservation], point_offsets: &'a [usize]) -> Self {
        Self {
            observations,
            point_offsets,
            signed_depth: false,
        }
    }

    /// Construct algebraic radial measurements that permit either depth sign.
    ///
    /// This is intended for benchmark formats such as BAL whose projection is
    /// defined for every finite nonzero depth. SfM pipelines should normally
    /// use [`Self::new`] so cheirality remains part of the model contract.
    #[cfg(feature = "unstable-internals")]
    pub const fn new_signed_depth(
        observations: &'a [PixelObservation],
        point_offsets: &'a [usize],
    ) -> Self {
        Self {
            observations,
            point_offsets,
            signed_depth: true,
        }
    }

    /// Number of pixel observations.
    pub const fn observation_count(self) -> usize {
        self.observations.len()
    }

    /// Number of point tracks represented by the CSR offsets.
    #[cfg(feature = "unstable-internals")]
    pub const fn point_count(self) -> usize {
        self.point_offsets.len().saturating_sub(1)
    }

    #[cfg(feature = "faer")]
    pub(in crate::kernel::multi_view) const fn observations(self) -> &'a [PixelObservation] {
        self.observations
    }

    #[cfg(feature = "faer")]
    pub(in crate::kernel::multi_view) const fn point_offsets(self) -> &'a [usize] {
        self.point_offsets
    }

    pub(in crate::kernel::multi_view) const fn signed_depth(self) -> bool {
        self.signed_depth
    }
}

/// Zero-copy radial self-calibration problem description.
///
/// Intrinsics may be per-camera or shared through `camera_to_group`. Poses,
/// points, solver scratch, and nonlinear controls remain explicit at each
/// operation. The same view drives sparse optimization, lifecycle evaluation,
/// and selected-track triangulation without allocating.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(feature = "unstable-internals")]
pub struct SelfCalibratingBundleAdjustmentProblem<'a> {
    measurements: SelfCalibratingMeasurements<'a>,
    intrinsic_groups: &'a [RadialCamera],
    camera_to_group: Option<&'a [usize]>,
    gauge: ScaleGauge,
    noise: NoiseModel<'a>,
}

#[cfg(feature = "unstable-internals")]
impl<'a> SelfCalibratingBundleAdjustmentProblem<'a> {
    /// Borrow a per-camera radial problem with unit residual noise.
    #[cfg(feature = "unstable-internals")]
    pub const fn new(
        measurements: SelfCalibratingMeasurements<'a>,
        cameras: &'a [RadialCamera],
        gauge: ScaleGauge,
    ) -> Self {
        Self {
            measurements,
            intrinsic_groups: cameras,
            camera_to_group: None,
            gauge,
            noise: NoiseModel::Unit,
        }
    }

    /// Borrow a grouped radial problem with unit residual noise.
    #[cfg(feature = "unstable-internals")]
    pub const fn new_grouped(
        measurements: SelfCalibratingMeasurements<'a>,
        intrinsic_groups: &'a [RadialCamera],
        camera_to_group: &'a [usize],
        gauge: ScaleGauge,
    ) -> Self {
        Self {
            measurements,
            intrinsic_groups,
            camera_to_group: Some(camera_to_group),
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

    /// Borrowed point-major radial measurements.
    #[cfg(feature = "unstable-internals")]
    pub const fn measurements(self) -> SelfCalibratingMeasurements<'a> {
        self.measurements
    }

    /// Borrowed radial intrinsic groups.
    #[cfg(feature = "unstable-internals")]
    pub const fn intrinsic_groups(self) -> &'a [RadialCamera] {
        self.intrinsic_groups
    }

    /// Camera-to-group assignment, or `None` for per-camera intrinsics.
    #[cfg(feature = "unstable-internals")]
    pub const fn camera_to_group(self) -> Option<&'a [usize]> {
        self.camera_to_group
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

    /// Number of point tracks represented by the CSR offsets.
    #[cfg(feature = "unstable-internals")]
    pub const fn point_count(self) -> usize {
        self.measurements.point_count()
    }
}

/// Reusable allocation for dense self-calibrating bundle adjustment.
///
/// For a nonzero camera capacity, construction allocates for the maximum
/// `3 + 9 * (camera_capacity - 1)` reduced dimension: three anchor-camera
/// intrinsics and nine parameters for every other camera. Solves within both
/// capacities allocate nothing.
#[cfg(any(test, feature = "unstable-internals"))]
pub struct SelfCalibrationWorkspace {
    pub(super) dense: DenseBundleAdjustmentWorkspace,
    #[cfg(feature = "unstable-internals")]
    pub(super) candidate_cameras: Vec<RadialCamera>,
}

#[cfg(any(test, feature = "unstable-internals"))]
impl SelfCalibrationWorkspace {
    /// Allocate a reusable workspace.
    pub fn try_new(
        camera_capacity: usize,
        point_capacity: usize,
    ) -> Result<Self, MultiViewBundleAdjustmentError> {
        let variable_camera_dimension = camera_capacity
            .saturating_sub(1)
            .checked_mul(9)
            .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
        let anchor_dimension = if camera_capacity == 0 { 0 } else { 3 };
        let camera_dimension = variable_camera_dimension
            .checked_add(anchor_dimension)
            .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
        Ok(Self {
            dense: DenseBundleAdjustmentWorkspace::try_new_with_dimension(
                camera_capacity,
                point_capacity,
                camera_dimension,
            )?,
            #[cfg(feature = "unstable-internals")]
            candidate_cameras: vec![RadialCamera::new(1.0, 0.0, 0.0); camera_capacity],
        })
    }

    /// Select exact final camera-rank and condition-number diagnostics.
    ///
    /// The default is enabled. Disabling this preserves point-block checks and
    /// nonlinear convergence while avoiding the cubic spectral decomposition.
    pub fn with_exact_camera_rank_diagnostics(mut self, enabled: bool) -> Self {
        self.dense = self.dense.with_exact_camera_rank_diagnostics(enabled);
        self
    }

    /// Whether exact final camera-rank diagnostics are enabled.
    pub const fn exact_camera_rank_diagnostics(&self) -> bool {
        self.dense.exact_camera_rank_diagnostics()
    }

    /// Select final undamped camera and point-rank diagnostics.
    ///
    /// The default is enabled. Disabling this avoids rebuilding the final
    /// undamped Schur system while retaining final objective evaluation and
    /// takes precedence over exact camera-rank diagnostics.
    pub fn with_final_diagnostics(mut self, enabled: bool) -> Self {
        self.dense = self.dense.with_final_diagnostics(enabled);
        self
    }

    /// Whether final undamped rank diagnostics are enabled.
    pub const fn final_diagnostics(&self) -> bool {
        self.dense.final_diagnostics()
    }

    /// Set the positive finite initial dimensionless LM damping value.
    /// Invalid values are rejected when refinement begins.
    #[cfg(feature = "unstable-internals")]
    pub fn with_initial_damping(mut self, initial_damping: f64) -> Self {
        self.dense = self.dense.with_initial_damping(initial_damping);
        self
    }

    /// Configured initial dimensionless LM damping value.
    #[cfg(feature = "unstable-internals")]
    pub const fn initial_damping(&self) -> f64 {
        self.dense.initial_damping()
    }

    /// Maximum number of cameras accepted by the workspace.
    #[cfg(feature = "unstable-internals")]
    pub const fn camera_capacity(&self) -> usize {
        self.dense.camera_capacity()
    }

    /// Maximum number of points accepted by the workspace.
    #[cfg(feature = "unstable-internals")]
    pub const fn point_capacity(&self) -> usize {
        self.dense.point_capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_allocates_the_exact_free_gauge_maximum() {
        for (cameras, expected_dimension) in [(0, 0), (1, 3), (4, 30)] {
            let workspace = SelfCalibrationWorkspace::try_new(cameras, 4)
                .expect("small workspace capacity must fit");
            assert_eq!(workspace.dense.step.len(), expected_dimension);
            assert_eq!(
                workspace.dense.hessian.len(),
                expected_dimension * expected_dimension
            );
        }
    }

    #[test]
    fn exact_rank_diagnostics_are_enabled_by_default_and_can_be_disabled() {
        let exact = SelfCalibrationWorkspace::try_new(4, 8).expect("small workspace");
        assert!(exact.exact_camera_rank_diagnostics());
        assert!(exact.final_diagnostics());

        let unassessed = exact
            .with_exact_camera_rank_diagnostics(false)
            .with_final_diagnostics(false);
        assert!(!unassessed.exact_camera_rank_diagnostics());
        assert!(!unassessed.final_diagnostics());
    }
}
