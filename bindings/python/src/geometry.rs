// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Robust two-view and absolute-pose estimators behind `openmvg_rs.estimate_*`.
//!
//! The Python layer validates the arguments and hands over `float64` arrays of the right shape.
//! Each function copies its arrays once, while it holds the GIL, into the correspondence types
//! of `openmvg-geometry`, then estimates with the GIL released. Strided and Fortran-ordered
//! arrays are read element by element, so their layout does not matter.

use nalgebra::{Matrix3, Point2, Point3, Vector3};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use openmvg_core::camera::Camera;
use openmvg_core::{AngularThreshold, NormalizedImageThreshold, PixelThreshold};
use openmvg_geometry::correspondence::{BearingCorrespondence, PixelCorrespondence};
use openmvg_geometry::essential::estimate_essential;
use openmvg_geometry::fundamental::estimate_fundamental;
use openmvg_geometry::homography::estimate_homography;
use openmvg_geometry::pose::{
    AbsolutePoseOptions, BearingToPoint, PoseRefinement, estimate_absolute_pose,
};
use openmvg_geometry::robust::{RobustError, RobustOptions};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::EstimationError;

/// A 3x3 matrix and the inlier mask.
type MatrixEstimate<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<bool>>);

/// The essential matrix, the relative rotation and translation, and the inlier mask.
type RelativePoseEstimate<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<bool>>,
);

/// The rotation, the translation, and the inlier mask.
type AbsolutePoseEstimate<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<bool>>,
);

/// Copy the rows of an `(n, D)` array.
fn rows<const D: usize>(array: &PyReadonlyArray2<'_, f64>) -> Vec<[f64; D]> {
    array
        .as_array()
        .rows()
        .into_iter()
        .map(|row| std::array::from_fn(|column| row[column]))
        .collect()
}

/// Copy a 3x3 camera matrix into a pinhole camera; the Python layer checks its structure.
///
/// Also returns the smaller focal length, which turns a pixel threshold into a normalized or
/// angular one.
fn camera(matrix: &PyReadonlyArray2<'_, f64>) -> PyResult<(Camera, f64)> {
    let k = matrix.as_array();
    let (fx, fy) = (k[(0, 0)], k[(1, 1)]);
    let camera = Camera::try_pinhole(fx, fy, k[(0, 2)], k[(1, 2)])
        .map_err(|error| PyValueError::new_err(format!("invalid camera matrix: {error}")))?;
    Ok((camera, fx.min(fy)))
}

/// The library's default robust search, seeded by the caller.
fn options(seed: u64) -> RobustOptions {
    RobustOptions::default().with_seed(seed)
}

fn invalid(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn estimation_error(error: &RobustError) -> PyErr {
    EstimationError::new_err(error.to_string())
}

fn matrix_array<'py>(py: Python<'py>, matrix: &Matrix3<f64>) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((3, 3), |(row, column)| matrix[(row, column)]).into_pyarray(py)
}

fn vector_array<'py>(py: Python<'py>, vector: &Vector3<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, vector.as_slice())
}

fn inlier_mask<'py>(
    py: Python<'py>,
    count: usize,
    inliers: &[usize],
) -> Bound<'py, PyArray1<bool>> {
    let mut mask = vec![false; count];
    for &index in inliers {
        mask[index] = true;
    }
    mask.into_pyarray(py)
}

fn pixel_correspondences(
    points1: &PyReadonlyArray2<'_, f64>,
    points2: &PyReadonlyArray2<'_, f64>,
) -> Vec<PixelCorrespondence> {
    rows::<2>(points1)
        .into_iter()
        .zip(rows::<2>(points2))
        .map(|([x1, y1], [x2, y2])| PixelCorrespondence::from_coords(x1, y1, x2, y2))
        .collect()
}

/// The estimator behind `openmvg_rs.estimate_fundamental`; see its Python docstring.
#[pyfunction(name = "estimate_fundamental")]
#[pyo3(signature = (points1, points2, threshold, seed))]
pub fn fundamental<'py>(
    py: Python<'py>,
    points1: PyReadonlyArray2<'py, f64>,
    points2: PyReadonlyArray2<'py, f64>,
    threshold: f64,
    seed: u64,
) -> PyResult<MatrixEstimate<'py>> {
    let data = pixel_correspondences(&points1, &points2);
    let threshold = PixelThreshold::try_new(threshold).map_err(invalid)?;
    let options = options(seed);
    let result = py
        .detach(|| estimate_fundamental(&data, threshold, &options))
        .map_err(|error| estimation_error(&error))?;
    Ok((
        matrix_array(py, result.model().matrix()),
        inlier_mask(py, data.len(), result.inliers()),
    ))
}

/// The estimator behind `openmvg_rs.estimate_homography`; see its Python docstring.
#[pyfunction(name = "estimate_homography")]
#[pyo3(signature = (points1, points2, threshold, seed))]
pub fn homography<'py>(
    py: Python<'py>,
    points1: PyReadonlyArray2<'py, f64>,
    points2: PyReadonlyArray2<'py, f64>,
    threshold: f64,
    seed: u64,
) -> PyResult<MatrixEstimate<'py>> {
    let data = pixel_correspondences(&points1, &points2);
    let threshold = PixelThreshold::try_new(threshold).map_err(invalid)?;
    let options = options(seed);
    let result = py
        .detach(|| estimate_homography(&data, threshold, &options))
        .map_err(|error| estimation_error(&error))?;
    Ok((
        matrix_array(py, result.model().matrix()),
        inlier_mask(py, data.len(), result.inliers()),
    ))
}

/// The estimator behind `openmvg_rs.estimate_essential`; see its Python docstring.
///
/// The pixel threshold becomes a normalized-image one through the smallest focal length of the
/// two cameras, as in the reconstruction pipelines.
#[pyfunction(name = "estimate_essential")]
#[pyo3(signature = (points1, points2, k1, k2, threshold, seed))]
pub fn essential<'py>(
    py: Python<'py>,
    points1: PyReadonlyArray2<'py, f64>,
    points2: PyReadonlyArray2<'py, f64>,
    k1: PyReadonlyArray2<'py, f64>,
    k2: PyReadonlyArray2<'py, f64>,
    threshold: f64,
    seed: u64,
) -> PyResult<RelativePoseEstimate<'py>> {
    let ((camera1, focal1), (camera2, focal2)) = (camera(&k1)?, camera(&k2)?);
    let data = rows::<2>(&points1)
        .into_iter()
        .zip(rows::<2>(&points2))
        .map(|([x1, y1], [x2, y2])| {
            let bearing1 = camera1.bearing(&Point2::new(x1, y1)).map_err(invalid)?;
            let bearing2 = camera2.bearing(&Point2::new(x2, y2)).map_err(invalid)?;
            BearingCorrespondence::try_new(bearing1.into_inner(), bearing2.into_inner())
                .map_err(invalid)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let pixels = PixelThreshold::try_new(threshold).map_err(invalid)?;
    let threshold =
        NormalizedImageThreshold::from_pixels(pixels, focal1.min(focal2)).map_err(invalid)?;
    let options = options(seed);
    let (result, pose) = py
        .detach(|| {
            let result = estimate_essential(&data, threshold, &options)?;
            let inliers = result
                .inliers()
                .iter()
                .map(|&index| data[index])
                .collect::<Vec<_>>();
            let pose = result.model().decompose_with_cheirality(&inliers);
            Ok((result, pose))
        })
        .map_err(|error| estimation_error(&error))?;
    let pose = pose.ok_or_else(|| {
        EstimationError::new_err("no relative pose puts the inliers in front of both cameras")
    })?;
    Ok((
        matrix_array(py, result.model().matrix()),
        matrix_array(py, pose.rotation_source_to_target().matrix()),
        vector_array(py, pose.translation_source_to_target()),
        inlier_mask(py, data.len(), result.inliers()),
    ))
}

/// The estimator behind `openmvg_rs.estimate_absolute_pose`; see its Python docstring.
///
/// P3P generates the hypotheses. The pixel threshold becomes an angular one through the smaller
/// focal length, and the final pose is refined on the perspective reprojection error, as in the
/// Incremental engine.
#[pyfunction(name = "estimate_absolute_pose")]
#[pyo3(signature = (points3d, points2d, k, threshold, seed))]
pub fn absolute_pose<'py>(
    py: Python<'py>,
    points3d: PyReadonlyArray2<'py, f64>,
    points2d: PyReadonlyArray2<'py, f64>,
    k: PyReadonlyArray2<'py, f64>,
    threshold: f64,
    seed: u64,
) -> PyResult<AbsolutePoseEstimate<'py>> {
    let (camera, focal) = camera(&k)?;
    let data = rows::<3>(&points3d)
        .into_iter()
        .zip(rows::<2>(&points2d))
        .map(|([x, y, z], [u, v])| {
            let bearing = camera.bearing(&Point2::new(u, v)).map_err(invalid)?;
            BearingToPoint::try_new(Point3::new(x, y, z), bearing).map_err(invalid)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let pixels = PixelThreshold::try_new(threshold).map_err(invalid)?;
    let threshold = AngularThreshold::from_pixels(pixels, focal).map_err(invalid)?;
    let options = AbsolutePoseOptions::default()
        .with_robust_options(options(seed))
        .with_refinement(PoseRefinement::Perspective);
    let result = py
        .detach(|| estimate_absolute_pose(&data, threshold, &options))
        .map_err(|error| estimation_error(&error))?;
    let pose = result.model();
    Ok((
        matrix_array(py, pose.rotation_matrix()),
        vector_array(py, &pose.translation()),
        inlier_mask(py, data.len(), result.inliers()),
    ))
}
