// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Anchor-frame conversion and requested physical scene rescaling.

use crate::SolveError;
use crate::kernel::RelativePose;
use crate::numerics::manifold::{Pose3 as NativePose3, Rotation3 as NativeRotation3};
use nalgebra::{Matrix3, Point3, Rotation3, Vector3};
use openmvg_core::model::WorldToCameraPose;

pub(super) struct WorldRebase {
    anchor_rotation: Rotation3<f64>,
    anchor_center: Point3<f64>,
    scale: f64,
}

impl WorldRebase {
    pub(super) fn try_new(anchor: &WorldToCameraPose, scale: f64) -> Result<Self, SolveError> {
        anchor
            .validate()
            .map_err(|error| SolveError::Internal(error.to_string()))?;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(SolveError::Unsupported(
                "world scale must be finite and positive".to_owned(),
            ));
        }
        Ok(Self {
            anchor_rotation: *anchor.rotation(),
            anchor_center: *anchor.center(),
            scale,
        })
    }

    pub(super) fn pose_to_native(
        &self,
        pose: &WorldToCameraPose,
    ) -> Result<RelativePose, SolveError> {
        if *pose.rotation() == self.anchor_rotation && *pose.center() == self.anchor_center {
            return Ok(RelativePose::from(NativePose3::IDENTITY));
        }
        let rotation = *pose.rotation() * self.anchor_rotation.inverse();
        let translation = *pose.rotation() * (self.anchor_center - pose.center()) * self.scale;
        NativePose3::try_new(
            NativeRotation3::try_from_matrix(rows_from_matrix(rotation.matrix()))
                .map_err(|error| SolveError::Internal(error.to_string()))?,
            [translation.x, translation.y, translation.z],
        )
        .map(RelativePose::from)
        .map_err(|error| SolveError::Internal(error.to_string()))
    }

    pub(super) fn point_to_native(&self, point: &Point3<f64>) -> Result<[f64; 3], SolveError> {
        if !point.coords.iter().all(|coordinate| coordinate.is_finite()) {
            return Err(SolveError::Internal("non-finite world point".to_owned()));
        }
        let transformed = self.anchor_rotation * (point - self.anchor_center) * self.scale;
        Ok([transformed.x, transformed.y, transformed.z])
    }

    pub(super) fn pose_from_native(
        &self,
        pose: RelativePose,
    ) -> Result<WorldToCameraPose, SolveError> {
        let pose =
            NativePose3::try_from(pose).map_err(|error| SolveError::Internal(error.to_string()))?;
        let relative_rotation = matrix_from_rows(pose.rotation().as_matrix());
        let rotation =
            Rotation3::from_matrix_unchecked(relative_rotation * self.anchor_rotation.matrix());
        let translation = Vector3::from(pose.translation());
        // The input scale sets the physical baseline requested by the problem.
        // Raising restores the anchor frame while retaining that scene scale.
        let relative_center = -(relative_rotation.transpose() * translation);
        let center = self.anchor_center + self.anchor_rotation.inverse() * relative_center;
        WorldToCameraPose::from_rotation_center(rotation, center)
            .map_err(|error| SolveError::Internal(error.to_string()))
    }

    pub(super) fn point_from_native(&self, point: [f64; 3]) -> Result<Point3<f64>, SolveError> {
        if !point.iter().all(|coordinate| coordinate.is_finite()) {
            return Err(SolveError::Numerical(
                "native BA returned a non-finite point".to_owned(),
            ));
        }
        Ok(self.anchor_center + self.anchor_rotation.inverse() * Vector3::from(point))
    }
}

pub(super) fn rows_from_matrix(matrix: &Matrix3<f64>) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[(row, column)]))
}

fn matrix_from_rows(matrix: &[[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::from_fn(|row, column| matrix[row][column])
}
