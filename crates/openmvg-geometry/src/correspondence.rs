// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Matched points, the input of every solver.
//!
//! The `f64` correspondence types that every solver consumes:
//! - [`PixelCorrespondence`]: pairs of 2D pixel coordinates.
//! - [`BearingCorrespondence`]: pairs of unit bearings.
//! - [`Point3Correspondence`]: source-to-target 3D point pairs.

use nalgebra::{Point2, Point3, Vector2, Vector3};

use crate::{SolverError, SolverResult};

/// Correspondence between two views using pixel coordinates.
///
/// Input to the uncalibrated solvers (fundamental matrix, homography).
///
/// # Examples
///
/// ```rust
/// use nalgebra::Point2;
/// use openmvg_geometry::correspondence::PixelCorrespondence;
///
/// let corr = PixelCorrespondence::new(
///     Point2::new(100.0, 200.0),  // Point in image 1
///     Point2::new(150.0, 210.0),  // Corresponding point in image 2
/// );
/// ```
// Closed set: a two-view correspondence is exactly one matched point pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelCorrespondence {
    /// Point in the first image, (x, y) in pixels.
    pub point1: Point2<f64>,
    /// Point in the second image, (x, y) in pixels.
    pub point2: Point2<f64>,
}

impl PixelCorrespondence {
    /// Create a new pixel correspondence.
    pub fn new(point1: Point2<f64>, point2: Point2<f64>) -> Self {
        Self { point1, point2 }
    }

    /// Create from raw coordinates.
    pub fn from_coords(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self {
            point1: Point2::new(x1, y1),
            point2: Point2::new(x2, y2),
        }
    }
}

/// Correspondence between two views as a pair of unit bearings.
///
/// Input to the calibrated solvers (essential matrix). A bearing is a
/// unit-norm direction in the camera frame.
///
/// # Examples
///
/// ```rust
/// use nalgebra::Vector3;
/// use openmvg_geometry::correspondence::BearingCorrespondence;
///
/// let corr = BearingCorrespondence::try_new(
///     Vector3::new(0.0, 0.0, 1.0),  // Looking straight ahead
///     Vector3::new(0.1, 0.0, 1.0),  // Slightly to the right
/// ).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BearingCorrespondence {
    /// Unit bearing in the first camera frame.
    bearing1: Vector3<f64>,
    /// Unit bearing in the second camera frame.
    bearing2: Vector3<f64>,
}

impl BearingCorrespondence {
    /// Create a correspondence by scale-safely normalizing two directions.
    pub fn try_new(bearing1: Vector3<f64>, bearing2: Vector3<f64>) -> SolverResult<Self> {
        Ok(Self {
            bearing1: normalize_bearing(bearing1)?,
            bearing2: normalize_bearing(bearing2)?,
        })
    }

    /// Create from normalized 2D image coordinates.
    ///
    /// Maps normalized image coordinates (x, y) to the bearing `(x, y, 1) / ‖(x, y, 1)‖`.
    pub fn try_from_normalized_coords(p1: Vector2<f64>, p2: Vector2<f64>) -> SolverResult<Self> {
        Self::try_new(Vector3::new(p1.x, p1.y, 1.0), Vector3::new(p2.x, p2.y, 1.0))
    }

    /// Unit bearing in the first camera frame.
    #[inline]
    #[must_use]
    pub const fn bearing1(&self) -> &Vector3<f64> {
        &self.bearing1
    }

    /// Unit bearing in the second camera frame.
    #[inline]
    #[must_use]
    pub const fn bearing2(&self) -> &Vector3<f64> {
        &self.bearing2
    }

    /// Check that both bearings are finite and have unit norm (tolerance `1e-6`).
    pub fn validate(&self) -> SolverResult<()> {
        let tolerance = 1e-6;

        if self
            .bearing1
            .iter()
            .chain(self.bearing2.iter())
            .any(|component| !component.is_finite())
        {
            return Err(SolverError::Numerical {
                reason: "bearing components must be finite".to_owned(),
            });
        }

        let norm1 = self.bearing1.norm();
        let norm2 = self.bearing2.norm();

        if (norm1 - 1.0).abs() > tolerance {
            return Err(SolverError::Numerical {
                reason: format!("bearing1 not normalized: norm = {norm1:.6}"),
            });
        }

        if (norm2 - 1.0).abs() > tolerance {
            return Err(SolverError::Numerical {
                reason: format!("bearing2 not normalized: norm = {norm2:.6}"),
            });
        }

        Ok(())
    }

    /// Return the normalized image coordinates (x/z, y/z) of both bearings.
    pub fn try_to_normalized_coords(self) -> SolverResult<(Vector2<f64>, Vector2<f64>)> {
        let p1 = Vector2::new(
            self.bearing1.x / self.bearing1.z,
            self.bearing1.y / self.bearing1.z,
        );
        let p2 = Vector2::new(
            self.bearing2.x / self.bearing2.z,
            self.bearing2.y / self.bearing2.z,
        );
        if p1.iter().chain(p2.iter()).all(|value| value.is_finite()) {
            Ok((p1, p2))
        } else {
            Err(SolverError::InvalidInput {
                reason: "bearing cannot be represented by finite normalized coordinates",
            })
        }
    }
}

fn normalize_bearing(bearing: Vector3<f64>) -> SolverResult<Vector3<f64>> {
    let mut scale = 0.0;
    for value in bearing.iter() {
        if !value.is_finite() {
            return Err(SolverError::InvalidInput {
                reason: "bearing components must be finite",
            });
        }
        let magnitude = value.abs();
        if magnitude > scale {
            scale = magnitude;
        }
    }
    if scale == 0.0 {
        return Err(SolverError::InvalidInput {
            reason: "bearing direction must be non-zero",
        });
    }

    let scaled = bearing / scale;
    let norm = scaled.norm();
    if !norm.is_finite() || norm == 0.0 {
        return Err(SolverError::Numerical {
            reason: "bearing normalization failed".to_owned(),
        });
    }
    let normalized = scaled / norm;
    if normalized.iter().all(|value| value.is_finite()) {
        Ok(normalized)
    } else {
        Err(SolverError::Numerical {
            reason: "bearing normalization produced non-finite values".to_owned(),
        })
    }
}

/// Correspondence between a 3D source point and its 3D target point.
///
/// Rigid-alignment models map `source` into the coordinate frame of
/// `target`: `target = rotation * source + translation`.
// Closed set: a 3D-3D correspondence is exactly one source-target point pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3Correspondence {
    /// Point in the source coordinate frame.
    pub source: Point3<f64>,
    /// Matching point in the target coordinate frame.
    pub target: Point3<f64>,
}

impl Point3Correspondence {
    /// Create a source-to-target 3D correspondence.
    pub fn new(source: Point3<f64>, target: Point3<f64>) -> Self {
        Self { source, target }
    }

    /// Create a correspondence from raw coordinates.
    pub fn from_coords(sx: f64, sy: f64, sz: f64, tx: f64, ty: f64, tz: f64) -> Self {
        Self {
            source: Point3::new(sx, sy, sz),
            target: Point3::new(tx, ty, tz),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Vector2, Vector3};

    #[test]
    fn test_bearing_from_normalized_coords() {
        let corr = BearingCorrespondence::try_from_normalized_coords(
            Vector2::new(0.0, 0.0),
            Vector2::new(0.1, 0.05),
        )
        .unwrap();

        assert!(corr.validate().is_ok());

        // (0, 0, 1) normalized should be (0, 0, 1)
        assert!((corr.bearing1().z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn bearing_constructor_rejects_invalid_directions() {
        assert!(BearingCorrespondence::try_new(Vector3::zeros(), Vector3::z()).is_err());
        assert!(
            BearingCorrespondence::try_new(Vector3::new(f64::NAN, 0.0, 1.0), Vector3::z()).is_err()
        );
        assert!(
            BearingCorrespondence::try_new(Vector3::z(), Vector3::new(0.0, f64::INFINITY, 1.0))
                .is_err()
        );
    }

    /// Scale-safe bearing normalization: finite non-unit directions are normalized before entering
    /// geometry solvers; extreme input scales cannot leak magnitude into the unit-bearing contract.
    #[test]
    fn bearing_normalization_is_scale_safe() {
        let large = BearingCorrespondence::try_new(
            Vector3::new(f64::MAX, f64::MAX / 2.0, -f64::MAX / 4.0),
            Vector3::new(-f64::MAX / 3.0, f64::MAX, f64::MAX / 5.0),
        )
        .unwrap();
        assert!((large.bearing1().norm() - 1.0).abs() < 1e-14);
        assert!((large.bearing2().norm() - 1.0).abs() < 1e-14);

        let tiny = f64::from_bits(1);
        let small = BearingCorrespondence::try_new(
            Vector3::new(tiny, tiny * 2.0, tiny * 3.0),
            Vector3::new(-tiny * 3.0, tiny, tiny * 2.0),
        )
        .unwrap();
        assert!((small.bearing1().norm() - 1.0).abs() < 1e-14);
        assert!((small.bearing2().norm() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn normalized_coordinates_reject_equatorial_bearings() {
        let correspondence = BearingCorrespondence::try_new(Vector3::x(), Vector3::z()).unwrap();
        assert!(correspondence.try_to_normalized_coords().is_err());
    }

    #[test]
    fn bearing_correspondence_layout_is_unchanged() {
        assert_eq!(
            std::mem::size_of::<BearingCorrespondence>(),
            6 * std::mem::size_of::<f64>()
        );
    }
}
