// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared values and crate-private implementation contract for triangulation.

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Point2, Point3};
use openmvg_core::{
    AngularResidual, PixelResidual, PixelThreshold, WorldDistanceResidual,
    model::{EstimationError, EstimationResult, ViewId, WorldToCameraPose},
};

/// Configuration shared by multi-view triangulation algorithms.
///
/// The defaults accept a point when two of its rays meet at an angle of at least 2°, its mean
/// reprojection error over the views is at most 4 px, and it lies in front of every camera
/// (`min_depth` is 0, so any positive depth passes).
#[derive(Debug, Clone)]
pub struct TriangulationConfig {
    min_parallax: AngularResidual,
    max_pixel_residual: PixelThreshold,
    min_depth: WorldDistanceResidual,
}

config_accessors!(TriangulationConfig {
    /// Minimum angle, in radians, between two viewing rays taken as unoriented lines; default
    /// 2°. Rays closer to parallel do not fix the point's depth.
    copy min_parallax, with_min_parallax: AngularResidual;
    /// Maximum mean reprojection error over the contributing views, in pixels; default 4 px.
    copy max_pixel_residual, with_max_pixel_residual: PixelThreshold;
    /// Minimum camera-space depth of the point in every view, in world units; default 0. The
    /// point must also lie strictly in front of every camera.
    copy min_depth, with_min_depth: WorldDistanceResidual;
});

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            min_parallax: AngularResidual::TWO_DEGREES,
            max_pixel_residual: PixelThreshold::FOUR_PIXELS,
            min_depth: WorldDistanceResidual::ZERO,
        }
    }
}

impl TriangulationConfig {
    /// Validate angular, reprojection, and cheirality thresholds.
    pub fn validate(&self) -> EstimationResult<()> {
        if self.min_parallax.value() > std::f64::consts::FRAC_PI_2 {
            return Err(EstimationError::InvalidInput(
                "min_parallax must not exceed pi/2".to_owned(),
            ));
        }
        Ok(())
    }
}

/// One triangulated point with its mean reprojection error.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TriangulatedPoint {
    /// Triangulated world point.
    pub point: Point3<f64>,
    /// Mean reprojection error across the contributing views.
    pub reprojection_error: PixelResidual,
    /// Number of views used.
    pub view_count: usize,
}

/// Internal common shape implemented by the triangulation algorithms.
pub(crate) trait TriangulationSolver: Send + Sync {
    fn triangulate(
        &self,
        observations: &[(ViewId, Point2<f64>)],
        poses: &BTreeMap<ViewId, WorldToCameraPose>,
        intrinsics: &BTreeMap<ViewId, Matrix3<f64>>,
        config: &TriangulationConfig,
    ) -> EstimationResult<TriangulatedPoint>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangulation_config_rejects_pi_parallax() {
        let config = TriangulationConfig::default()
            .with_min_parallax(AngularResidual::try_new(std::f64::consts::PI).unwrap());
        assert!(config.validate().is_err());
    }
}
