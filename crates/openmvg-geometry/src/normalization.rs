// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hartley normalization for numerical stability.
//!
//! Normalization transforms point coordinates to improve the conditioning of
//! linear systems in geometric estimation. The gain is largest for points with
//! large pixel coordinates.
//!
//! ## Normalization strategy
//!
//! The standard Hartley normalization:
//! 1. Translates points so centroid is at origin
//! 2. Scales points so average distance from origin is sqrt(2)
//!
//! On raw pixel coordinates of magnitude `s`, the entries of a linear constraint
//! matrix range from 1 to about `s²`, so its condition number grows with `s`.
//! After normalization every coordinate is of order 1 (R. Hartley, "In Defense of
//! the Eight-Point Algorithm", IEEE TPAMI, 1997).
//!
//! ## Where normalization applies
//!
//! The Fundamental-matrix and homography kernels fit on Hartley-normalized
//! coordinates, and the type system says so: their public `solve` takes a
//! [`NormalizedCorrespondences`] set (built from pixel correspondences, holding
//! both transforms) and returns pixel-space models denormalized with that
//! set's transforms. Each element is a [`NormalizedCorrespondence`], which only this crate
//! creates, so a solver never receives coordinates that were not normalized.
//!
//! Through the [`Solver`](crate::robust::Solver) and
//! [`MultiSolutionSolver`](crate::robust::MultiSolutionSolver) traits, the same solvers take a
//! slice of [`NormalizedCorrespondence`] and return their models in the normalized frame, not
//! in pixels; the residuals are then measured in that frame too.
//!
//! ## Scalar type
//!
//! All normalization types operate on `f64` coordinates, matching the concrete
//! `f64` solver inputs.

use nalgebra::{Matrix3, Matrix4, Point2, Point3, Vector2, Vector3};

use crate::correspondence::PixelCorrespondence;
use crate::shared::epipolar_constraint::{EpipolarCorrespondence, Vector9};

/// Normalization transform for a set of 2D points.
///
/// Stores the 3x3 transformation matrix that normalizes points.
#[derive(Debug, Clone)]
pub(crate) struct PointNormalization {
    /// 3x3 transformation matrix, applied as `T * [x, y, 1]^T`.
    transform: Matrix3<f64>,
}

impl PointNormalization {
    /// Create identity normalization (no transformation).
    pub(crate) fn identity() -> Self {
        Self {
            transform: Matrix3::identity(),
        }
    }

    /// Compute normalization transform from a set of points.
    ///
    /// Centers points at origin and scales so average distance is sqrt(2).
    pub(crate) fn from_points(points: &[Point2<f64>]) -> Self {
        if points.is_empty() {
            return Self::identity();
        }

        let n = points.len() as f64;

        // Compute centroid
        let centroid = points
            .iter()
            .fold(Vector2::zeros(), |acc, p| acc + Vector2::new(p.x, p.y))
            / n;

        // Hartley's isotropic normalization uses the mean Euclidean distance
        // from the centroid (not the root-mean-square distance).
        let mean_dist = points
            .iter()
            .map(|p| {
                let dx = p.x - centroid.x;
                let dy = p.y - centroid.y;
                (dx * dx + dy * dy).sqrt()
            })
            .fold(0.0, |acc, d| acc + d)
            / n;

        // Handle degenerate case (all points identical)
        let tolerance = 1e-10;
        if mean_dist < tolerance {
            return Self::identity();
        }

        let sqrt_2 = std::f64::consts::SQRT_2;
        let scale = sqrt_2 / mean_dist;

        // Build normalization matrix
        let transform = Matrix3::new(
            scale,
            0.0,
            -scale * centroid.x,
            0.0,
            scale,
            -scale * centroid.y,
            0.0,
            0.0,
            1.0,
        );

        Self { transform }
    }

    /// Return the 3x3 transform, applied as `T * [x, y, 1]^T`.
    #[inline]
    pub(crate) fn transform(&self) -> &Matrix3<f64> {
        &self.transform
    }

    /// Apply normalization to a single point.
    pub(crate) fn normalize_point(&self, point: &Point2<f64>) -> Point2<f64> {
        let h = Vector3::new(point.x, point.y, 1.0);
        let nh = self.transform * h;
        Point2::new(nh.x / nh.z, nh.y / nh.z)
    }
}

/// Hartley normalization of 3D world points: centroid at the origin, mean
/// distance sqrt(3).
#[derive(Debug, Clone, Copy)]
pub(crate) struct WorldNormalization {
    centroid: Vector3<f64>,
    scale: f64,
}

impl WorldNormalization {
    /// Compute the normalization of `points`, or `None` when they have no
    /// spatial extent.
    pub(crate) fn from_points<I>(points: I) -> Option<Self>
    where
        I: IntoIterator<Item = Point3<f64>>,
        I::IntoIter: Clone + ExactSizeIterator,
    {
        let points = points.into_iter();
        let count = points.len() as f64;
        let centroid = points
            .clone()
            .fold(Vector3::zeros(), |sum, point| sum + point.coords)
            / count;
        let mean_distance = points
            .map(|point| (point.coords - centroid).norm())
            .sum::<f64>()
            / count;
        if !mean_distance.is_finite() || mean_distance <= f64::EPSILON {
            return None;
        }
        Some(Self {
            centroid,
            scale: 3.0_f64.sqrt() / mean_distance,
        })
    }

    /// Return the 4x4 transform, applied as `T * [X, Y, Z, 1]^T`.
    pub(crate) fn transform(&self) -> Matrix4<f64> {
        let (scale, centroid) = (self.scale, self.centroid);
        Matrix4::new(
            scale,
            0.0,
            0.0,
            -scale * centroid.x,
            0.0,
            scale,
            0.0,
            -scale * centroid.y,
            0.0,
            0.0,
            scale,
            -scale * centroid.z,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Map one point into the normalized frame.
    #[inline]
    pub(crate) fn normalize_point(&self, point: &Point3<f64>) -> Point3<f64> {
        Point3::from((point.coords - self.centroid) * self.scale)
    }
}

/// Normalization for pixel correspondences (stores transforms for both views).
///
/// Robust estimation normalizes the full dataset once with it, then
/// denormalizes the fitted models.
#[derive(Debug, Clone)]
pub struct PixelNormalization {
    /// Transform for points in image 1.
    transform1: Matrix3<f64>,
    /// Transform for points in image 2.
    transform2: Matrix3<f64>,
    /// Cached inverse of `transform2`, used to denormalize homographies.
    transform2_inv: Matrix3<f64>,
}

impl PixelNormalization {
    /// Compute normalization from correspondences.
    ///
    /// Computes separate normalization transforms for each image's points.
    /// Also caches the inverse of `transform2` for homography denormalization.
    pub fn from_correspondences(correspondences: &[PixelCorrespondence]) -> Self {
        let points1: Vec<Point2<f64>> = correspondences.iter().map(|c| c.point1).collect();
        let points2: Vec<Point2<f64>> = correspondences.iter().map(|c| c.point2).collect();

        let norm1 = PointNormalization::from_points(&points1);
        let norm2 = PointNormalization::from_points(&points2);

        // Cache the inverse of transform2 for homography denormalization
        let transform2_inv = norm2
            .transform
            .try_inverse()
            .unwrap_or_else(Matrix3::identity);

        Self {
            transform1: norm1.transform,
            transform2: norm2.transform,
            transform2_inv,
        }
    }

    /// Apply normalization to correspondences.
    #[cfg(test)]
    pub(crate) fn normalize(
        &self,
        correspondences: &[PixelCorrespondence],
    ) -> Vec<PixelCorrespondence> {
        self.normalize_with(correspondences, |correspondence| correspondence)
    }

    /// Normalize every correspondence and wrap it with `wrap`.
    fn normalize_with<T>(
        &self,
        correspondences: &[PixelCorrespondence],
        wrap: impl Fn(PixelCorrespondence) -> T,
    ) -> Vec<T> {
        let norm1 = PointNormalization {
            transform: self.transform1,
        };
        let norm2 = PointNormalization {
            transform: self.transform2,
        };

        correspondences
            .iter()
            .map(|c| {
                wrap(PixelCorrespondence::new(
                    norm1.normalize_point(&c.point1),
                    norm2.normalize_point(&c.point2),
                ))
            })
            .collect()
    }

    /// Denormalize a fundamental matrix.
    ///
    /// Given F computed from normalized points, returns F for original coordinates.
    /// Formula: F_denorm = T2^T * F_norm * T1
    pub fn denormalize_fundamental(&self, f_normalized: Matrix3<f64>) -> Matrix3<f64> {
        self.transform2.transpose() * f_normalized * self.transform1
    }

    /// Denormalize a homography matrix.
    ///
    /// Given H computed from normalized points, returns H for original coordinates.
    /// Formula: H_denorm = T2^-1 * H_norm * T1
    ///
    /// Uses the cached inverse of `T2`.
    pub fn denormalize_homography(&self, h_normalized: Matrix3<f64>) -> Matrix3<f64> {
        self.transform2_inv * h_normalized * self.transform1
    }
}

/// One Hartley-normalized correspondence.
///
/// Only [`NormalizedCorrespondences`] creates these, so a value always lives in
/// the coordinate frame of its set's `T1`/`T2` transforms and can never be a
/// raw pixel correspondence by accident.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedCorrespondence(PixelCorrespondence);

impl NormalizedCorrespondence {
    /// Return the normalized point in image 1.
    #[inline]
    pub fn point1(&self) -> &Point2<f64> {
        &self.0.point1
    }

    /// Return the normalized point in image 2.
    #[inline]
    pub fn point2(&self) -> &Point2<f64> {
        &self.0.point2
    }
}

/// Hartley-normalized correspondences together with their `T1`/`T2` transforms.
///
/// [`NormalizedCorrespondences::from_pixels`] applies
/// [`PixelNormalization::from_correspondences`] and maps every
/// correspondence through both transforms. The Fundamental-matrix and homography
/// kernels (`EightPointSolver`, `SevenPointSolver`, `DltSolver`) fit
/// on these coordinates and denormalize every model with
/// [`Self::normalization`], so they return pixel-space models.
#[derive(Debug, Clone)]
pub struct NormalizedCorrespondences {
    normalization: PixelNormalization,
    correspondences: Vec<NormalizedCorrespondence>,
}

impl NormalizedCorrespondences {
    /// Hartley-normalize pixel correspondences, keeping both transforms.
    pub fn from_pixels(correspondences: &[PixelCorrespondence]) -> Self {
        let normalization = PixelNormalization::from_correspondences(correspondences);
        let correspondences =
            normalization.normalize_with(correspondences, NormalizedCorrespondence);
        Self {
            normalization,
            correspondences,
        }
    }

    /// Return the transforms used to normalize, and to denormalize models.
    #[inline]
    pub fn normalization(&self) -> &PixelNormalization {
        &self.normalization
    }

    /// Return the normalized correspondences in input order.
    #[inline]
    pub fn as_slice(&self) -> &[NormalizedCorrespondence] {
        &self.correspondences
    }

    /// Return the number of correspondences.
    #[inline]
    pub fn len(&self) -> usize {
        self.correspondences.len()
    }

    /// Return whether the set holds no correspondence.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.correspondences.is_empty()
    }
}

/// Coordinates shared by the pixel and the normalized datum types, so one
/// kernel implementation serves both with identical arithmetic.
pub(crate) trait CorrespondenceCoordinates: Copy {
    /// Return the two points (pixel or normalized, depending on the type).
    fn coordinates(&self) -> &PixelCorrespondence;
}

impl CorrespondenceCoordinates for PixelCorrespondence {
    #[inline]
    fn coordinates(&self) -> &PixelCorrespondence {
        self
    }
}

impl CorrespondenceCoordinates for NormalizedCorrespondence {
    #[inline]
    fn coordinates(&self) -> &PixelCorrespondence {
        &self.0
    }
}

impl EpipolarCorrespondence<f64> for NormalizedCorrespondence {
    #[inline]
    fn to_constraint_row(&self) -> Vector9<f64> {
        self.0.to_constraint_row()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point2;

    /// Hartley-normalize `points`, returning them with the transform.
    fn normalize_points(points: &[Point2<f64>]) -> (Vec<Point2<f64>>, Matrix3<f64>) {
        let norm = PointNormalization::from_points(points);
        let normalized = points.iter().map(|p| norm.normalize_point(p)).collect();
        (normalized, norm.transform)
    }

    #[test]
    fn test_point_normalization() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(100.0, 0.0),
            Point2::new(100.0, 100.0),
            Point2::new(0.0, 100.0),
        ];

        let (normalized, transform) = normalize_points(&points);

        // Check that transform is not identity
        assert!((transform - Matrix3::identity()).norm() > 1e-6);

        // Check that normalized points have centroid at origin
        let centroid: Vector2<f64> = normalized
            .iter()
            .fold(Vector2::zeros(), |acc, p| acc + Vector2::new(p.x, p.y))
            / 4.0;
        assert!(centroid.norm() < 1e-10);

        // Check that average distance is sqrt(2)
        let avg_dist: f64 = normalized
            .iter()
            .map(|p| (p.x * p.x + p.y * p.y).sqrt())
            .sum::<f64>()
            / 4.0;
        assert!((avg_dist - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_hartley_uses_mean_distance_for_unequal_radii() {
        let points = vec![
            Point2::new(-9.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(3.0, 0.0),
            Point2::new(5.0, 0.0),
        ];

        let (normalized, _) = normalize_points(&points);
        let mean_dist =
            normalized.iter().map(|p| p.coords.norm()).sum::<f64>() / normalized.len() as f64;

        assert!((mean_dist - std::f64::consts::SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn test_degenerate_points() {
        // All points identical
        let points = vec![
            Point2::new(50.0, 50.0),
            Point2::new(50.0, 50.0),
            Point2::new(50.0, 50.0),
        ];

        let (_, transform) = normalize_points(&points);

        // Should return identity transform for degenerate case
        assert!((transform - Matrix3::identity()).norm() < 1e-10);
    }

    #[test]
    fn test_empty_points() {
        let points: Vec<Point2<f64>> = vec![];
        let (normalized, transform) = normalize_points(&points);

        assert!(normalized.is_empty());
        assert!((transform - Matrix3::identity()).norm() < 1e-10);
    }

    #[test]
    fn test_correspondence_normalization() {
        let correspondences = vec![
            PixelCorrespondence::from_coords(100.0, 200.0, 110.0, 210.0),
            PixelCorrespondence::from_coords(200.0, 300.0, 220.0, 320.0),
            PixelCorrespondence::from_coords(300.0, 400.0, 330.0, 440.0),
            PixelCorrespondence::from_coords(400.0, 500.0, 440.0, 560.0),
        ];

        let norm = PixelNormalization::from_correspondences(&correspondences);
        let normalized = norm.normalize(&correspondences);

        // Both sets should have centered points
        let centroid1: Vector2<f64> = normalized.iter().fold(Vector2::zeros(), |acc, c| {
            acc + Vector2::new(c.point1.x, c.point1.y)
        }) / 4.0;
        let centroid2: Vector2<f64> = normalized.iter().fold(Vector2::zeros(), |acc, c| {
            acc + Vector2::new(c.point2.x, c.point2.y)
        }) / 4.0;

        assert!(centroid1.norm() < 1e-10);
        assert!(centroid2.norm() < 1e-10);
    }

    #[test]
    fn test_fundamental_denormalization() {
        let correspondences = vec![
            PixelCorrespondence::from_coords(100.0, 200.0, 110.0, 210.0),
            PixelCorrespondence::from_coords(200.0, 300.0, 220.0, 320.0),
        ];

        let norm = PixelNormalization::from_correspondences(&correspondences);

        // Create a simple "normalized" fundamental matrix
        let f_norm = Matrix3::new(0.0, 0.0, 0.1, 0.0, 0.0, -0.1, -0.1, 0.1, 0.0);

        let f_denorm = norm.denormalize_fundamental(f_norm);

        // Check that the result is different (was actually transformed)
        assert!((f_denorm - f_norm).norm() > 1e-6);
    }

    #[test]
    fn test_normalized_set_matches_pixel_normalization_bit_for_bit() {
        let correspondences: Vec<_> = (0..9)
            .map(|i| {
                let t = f64::from(i);
                PixelCorrespondence::from_coords(
                    120.0 + 37.5 * t,
                    80.0 + 11.0 * (0.7 * t).sin(),
                    95.0 + 35.25 * t,
                    260.0 - 13.0 * (0.4 * t).cos(),
                )
            })
            .collect();
        let reference = PixelNormalization::from_correspondences(&correspondences);
        let expected = reference.normalize(&correspondences);

        let set = NormalizedCorrespondences::from_pixels(&correspondences);

        assert_eq!(set.len(), correspondences.len());
        assert_eq!(set.normalization().transform1, reference.transform1);
        assert_eq!(set.normalization().transform2, reference.transform2);
        assert_eq!(set.normalization().transform2_inv, reference.transform2_inv);
        for (normalized, expected) in set.as_slice().iter().zip(&expected) {
            assert_eq!(normalized.point1(), &expected.point1);
            assert_eq!(normalized.point2(), &expected.point2);
        }
    }
}
