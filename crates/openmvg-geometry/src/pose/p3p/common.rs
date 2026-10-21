// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{SolverError, SolverResult};
use arrayvec::ArrayVec;
use nalgebra::{Isometry3, Matrix3, Point3, Rotation3, UnitQuaternion, UnitVector3, Vector3};

use crate::pose::{
    BearingToPoint, SolverPose, WorldToCameraPose, bearing_pose_angular_residual,
    camera_pose_from_world_to_camera,
};

pub(crate) const MAX_P3P_SOLUTIONS: usize = 4;
pub(crate) const NEAR_ROTATION_TOLERANCE: f64 = 1e-10;
pub(crate) const PROJECTED_RAY_TOLERANCE: f64 = 9.536_743_164_062_5e-7;
pub(crate) type MinimalSample = ([Point3<f64>; 3], [UnitVector3<f64>; 3]);

/// Allocation-free container for the zero to four poses admitted by P3P.
#[derive(Debug, Clone, Default)]
#[must_use]
pub struct P3pSolutions {
    poses: ArrayVec<WorldToCameraPose, MAX_P3P_SOLUTIONS>,
}

impl P3pSolutions {
    /// Number of valid candidates.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.poses.len()
    }

    /// Whether the solver found no valid candidate.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    /// Candidate poses in deterministic solver order.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[WorldToCameraPose] {
        self.poses.as_slice()
    }

    /// Iterate over candidate poses without allocating.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, WorldToCameraPose> {
        self.as_slice().iter()
    }

    /// Select the finite candidate with the lowest caller-provided score.
    #[must_use]
    pub fn best_by(
        &self,
        mut score: impl FnMut(&WorldToCameraPose) -> f64,
    ) -> Option<&WorldToCameraPose> {
        self.iter()
            .filter_map(|pose| {
                let value = score(pose);
                value.is_finite().then_some((pose, value))
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(pose, _)| pose)
    }

    /// Convert to an owned vector for callers that need dynamic storage.
    #[must_use]
    pub fn into_vec(self) -> Vec<WorldToCameraPose> {
        self.poses.into_iter().collect()
    }

    pub(crate) fn from_solver(solutions: SolverP3pSolutions) -> Self {
        let mut poses = ArrayVec::new();
        for solver_pose in solutions {
            if let Ok(pose) = camera_pose_from_world_to_camera(&solver_pose) {
                let _ = poses.try_push(pose);
            }
        }
        Self { poses }
    }
}

impl std::ops::Index<usize> for P3pSolutions {
    type Output = WorldToCameraPose;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<'a> IntoIterator for &'a P3pSolutions {
    type Item = &'a WorldToCameraPose;
    type IntoIter = std::slice::Iter<'a, WorldToCameraPose>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for P3pSolutions {
    type Item = WorldToCameraPose;
    type IntoIter = arrayvec::IntoIter<WorldToCameraPose, MAX_P3P_SOLUTIONS>;

    fn into_iter(self) -> Self::IntoIter {
        self.poses.into_iter()
    }
}

/// Private allocation-free P3P candidates in the solver's exact `(q, t)` chart.
#[derive(Debug, Clone, Default)]
pub(crate) struct SolverP3pSolutions {
    poses: ArrayVec<SolverPose<f64>, MAX_P3P_SOLUTIONS>,
}

impl SolverP3pSolutions {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.poses.len()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    #[inline]
    pub(crate) fn push(&mut self, pose: SolverPose<f64>) {
        let _ = self.poses.try_push(pose);
    }

    #[inline]
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, SolverPose<f64>> {
        self.poses.iter()
    }

    #[inline]
    pub(crate) fn into_vec(self) -> Vec<SolverPose<f64>> {
        self.poses.into_iter().collect()
    }
}

impl std::ops::Index<usize> for SolverP3pSolutions {
    type Output = SolverPose<f64>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.poses[index]
    }
}

impl<'a> IntoIterator for &'a SolverP3pSolutions {
    type Item = &'a SolverPose<f64>;
    type IntoIter = std::slice::Iter<'a, SolverPose<f64>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for SolverP3pSolutions {
    type Item = SolverPose<f64>;
    type IntoIter = arrayvec::IntoIter<SolverPose<f64>, MAX_P3P_SOLUTIONS>;

    fn into_iter(self) -> Self::IntoIter {
        self.poses.into_iter()
    }
}

/// Validated fixed-size P3P input.
pub(crate) struct P3pInput {
    pub(crate) world: [Vector3<f64>; 3],
    pub(crate) bearings: [Vector3<f64>; 3],
}

impl P3pInput {
    #[inline]
    pub(crate) fn new(world: [Point3<f64>; 3], bearings: [UnitVector3<f64>; 3]) -> Option<Self> {
        let world_vectors = world.map(|point| point.coords);
        let bearing_vectors = bearings.map(UnitVector3::into_inner);
        if world_vectors
            .iter()
            .chain(&bearing_vectors)
            .any(|vector| vector.iter().any(|value| !value.is_finite()))
        {
            return None;
        }

        Some(Self {
            world: world_vectors,
            bearings: bearing_vectors,
        })
    }

    /// Append an edge-basis pose recovered from three finite, positive depths.
    ///
    /// [`rotation_from_edges`] preserves orientation because both bases use
    /// their first-two-column cross product. The near-rotation path is also
    /// cheiral; projection can disturb that invariant and retains the check.
    #[inline]
    pub(crate) fn push_edge_basis_pose_from_positive_depths(
        &self,
        output: &mut SolverP3pSolutions,
        raw_rotation: Matrix3<f64>,
        translation: Vector3<f64>,
    ) {
        self.push_pose_with_cheirality(output, raw_rotation, translation, true);
    }

    /// Append an edge-basis pose already certified against the common
    /// near-rotation tolerance by the solver that constructed it.
    ///
    /// When the solver cannot also prove the minimal-sample angular bound,
    /// validate all three rays before accepting the pose.
    #[inline]
    pub(crate) fn push_certified_edge_basis_pose_from_positive_depths(
        &self,
        output: &mut SolverP3pSolutions,
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
        minimal_sample_fit_proven: bool,
    ) {
        if !minimal_sample_fit_proven
            && !self.projected_pose_fits_minimal_sample(&rotation, &translation)
        {
            return;
        }
        self.push_valid_rotation(output, rotation, translation, true);
    }

    #[inline]
    fn push_pose_with_cheirality(
        &self,
        output: &mut SolverP3pSolutions,
        raw_rotation: Matrix3<f64>,
        translation: Vector3<f64>,
        edge_basis_with_positive_depths: bool,
    ) {
        let is_near_rotation = if edge_basis_with_positive_depths {
            has_near_orthonormal_columns(&raw_rotation)
        } else {
            is_near_rotation(&raw_rotation)
        };
        if is_near_rotation {
            self.push_valid_rotation(
                output,
                raw_rotation,
                translation,
                edge_basis_with_positive_depths,
            );
            return;
        }
        let Some(rotation) = project_to_rotation(raw_rotation) else {
            return;
        };
        // Projection changes the rotation, so preserve the camera coordinate
        // of one observation before validating the projected hypothesis.
        let camera0 = raw_rotation * self.world[0] + translation;
        let translation = camera0 - rotation * self.world[0];
        if !self.projected_pose_fits_minimal_sample(&rotation, &translation) {
            return;
        }
        self.push_valid_rotation(output, rotation, translation, true);
    }

    /// Append a pose whose rotation is already orthonormal by construction.
    #[inline]
    pub(crate) fn push_orthonormal_pose(
        &self,
        output: &mut SolverP3pSolutions,
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
    ) {
        if rotation.iter().any(|value| !value.is_finite()) {
            return;
        }
        self.push_valid_rotation(output, rotation, translation, false);
    }

    /// Append an orthonormal pose whose third observation has proven positive depth.
    #[inline]
    pub(crate) fn push_orthonormal_pose_with_positive_third_depth(
        &self,
        output: &mut SolverP3pSolutions,
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
    ) {
        if rotation.iter().any(|value| !value.is_finite())
            || !self.first_two_are_cheiral(&rotation, &translation)
        {
            return;
        }
        self.push_valid_rotation(output, rotation, translation, true);
    }

    #[inline]
    fn push_valid_rotation(
        &self,
        output: &mut SolverP3pSolutions,
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
        cheirality_proven: bool,
    ) {
        if !cheirality_proven && !self.is_cheiral(&rotation, &translation) {
            return;
        }
        if !translation.iter().all(|value| value.is_finite()) {
            return;
        }
        let quaternion =
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation));
        let pose = Isometry3::from_parts(translation.into(), quaternion);
        output.push(pose);
    }

    #[inline]
    fn is_cheiral(&self, rotation: &Matrix3<f64>, translation: &Vector3<f64>) -> bool {
        (rotation * self.world[0] + translation).dot(&self.bearings[0]) > 0.0
            && (rotation * self.world[1] + translation).dot(&self.bearings[1]) > 0.0
            && (rotation * self.world[2] + translation).dot(&self.bearings[2]) > 0.0
    }

    #[inline]
    fn first_two_are_cheiral(&self, rotation: &Matrix3<f64>, translation: &Vector3<f64>) -> bool {
        (rotation * self.world[0] + translation).dot(&self.bearings[0]) > 0.0
            && (rotation * self.world[1] + translation).dot(&self.bearings[1]) > 0.0
    }

    #[inline]
    fn projected_pose_fits_minimal_sample(
        &self,
        rotation: &Matrix3<f64>,
        translation: &Vector3<f64>,
    ) -> bool {
        self.world
            .iter()
            .zip(&self.bearings)
            .all(|(world, bearing)| {
                let camera = rotation * world + translation;
                ray_fits(&camera, bearing)
            })
    }

    pub(crate) fn fitting_candidates(&self, candidates: &SolverP3pSolutions) -> SolverP3pSolutions {
        let mut fitting = SolverP3pSolutions::default();
        for pose in candidates {
            let valid = self
                .world
                .iter()
                .zip(&self.bearings)
                .all(|(world, bearing)| {
                    let camera = pose.rotation * world + pose.translation.vector;
                    ray_fits(&camera, bearing)
                });
            if valid {
                fitting.push(*pose);
            }
        }
        fitting
    }
}

#[inline]
fn ray_fits(camera: &Vector3<f64>, bearing: &Vector3<f64>) -> bool {
    let ray_depth = camera.dot(bearing);
    ray_depth.is_finite()
        && ray_depth > 0.0
        && camera.cross(bearing).norm_squared()
            <= PROJECTED_RAY_TOLERANCE * PROJECTED_RAY_TOLERANCE * ray_depth * ray_depth
}

#[inline]
fn is_near_rotation(matrix: &Matrix3<f64>) -> bool {
    has_near_orthonormal_columns(matrix)
        && matrix
            .column(0)
            .cross(&matrix.column(1))
            .dot(&matrix.column(2))
            > 0.0
}

#[inline]
fn has_near_orthonormal_columns(matrix: &Matrix3<f64>) -> bool {
    let first = matrix.column(0);
    let second = matrix.column(1);
    let third = matrix.column(2);
    let error = (first.norm_squared() - 1.0).abs()
        + (second.norm_squared() - 1.0).abs()
        + (third.norm_squared() - 1.0).abs()
        + first.dot(&second).abs()
        + first.dot(&third).abs()
        + second.dot(&third).abs();
    error <= NEAR_ROTATION_TOLERANCE
}

/// Convert an exact minimal correspondence slice to fixed arrays.
pub(crate) fn exact_sample(data: &[BearingToPoint]) -> SolverResult<MinimalSample> {
    match data.len() {
        0..=2 => Err(SolverError::InsufficientData {
            required: 3,
            provided: data.len(),
        }),
        3 => Ok((
            std::array::from_fn(|index| data[index].world_point),
            std::array::from_fn(|index| data[index].bearing),
        )),
        provided => Err(SolverError::InvalidSampleSize {
            expected: 3,
            provided,
        }),
    }
}

/// Read one exact minimal indexed sample without cloning or allocating.
pub(crate) fn exact_indexed_sample(
    indices: &[usize],
    data: &[BearingToPoint],
) -> SolverResult<MinimalSample> {
    match indices.len() {
        0..=2 => Err(SolverError::InsufficientData {
            required: 3,
            provided: indices.len(),
        }),
        3 => {
            if let Some(&index) = indices.iter().find(|&&index| index >= data.len()) {
                return Err(SolverError::InvalidDataIndex {
                    index,
                    data_len: data.len(),
                });
            }
            Ok((
                std::array::from_fn(|index| data[indices[index]].world_point),
                std::array::from_fn(|index| data[indices[index]].bearing),
            ))
        }
        provided => Err(SolverError::InvalidSampleSize {
            expected: 3,
            provided,
        }),
    }
}

/// Unsquared angular residual in radians, with positive-ray cheirality.
pub(crate) fn angular_residual(pose: &SolverPose<f64>, correspondence: &BearingToPoint) -> f64 {
    bearing_pose_angular_residual(pose, correspondence)
}

/// Modified Gram-Schmidt projection to a proper rotation.
#[inline]
fn project_to_rotation(matrix: Matrix3<f64>) -> Option<Matrix3<f64>> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let first = matrix.column(0).into_owned();
    let first_norm = first.norm();
    if first_norm <= f64::EPSILON.sqrt() {
        return None;
    }
    let first = first / first_norm;
    let second = matrix.column(1).into_owned() - first * first.dot(&matrix.column(1));
    let second_norm = second.norm();
    if second_norm <= f64::EPSILON.sqrt() {
        return None;
    }
    let second = second / second_norm;
    let third = first.cross(&second);
    Some(Matrix3::from_columns(&[first, second, third]))
}

#[inline]
pub(crate) fn world_basis_inverse(
    first_edge: Vector3<f64>,
    second_edge: Vector3<f64>,
) -> Option<Matrix3<f64>> {
    let normal = first_edge.cross(&second_edge);
    let determinant = normal.norm_squared();
    let geometry_scale_squared = first_edge.norm_squared().max(second_edge.norm_squared());
    if !determinant.is_finite()
        || determinant <= 64.0 * f64::EPSILON * geometry_scale_squared * geometry_scale_squared
    {
        return None;
    }
    let inverse_determinant = determinant.recip();
    let dual_first = second_edge.cross(&normal) * inverse_determinant;
    let dual_second = normal.cross(&first_edge) * inverse_determinant;
    let dual_normal = normal * inverse_determinant;
    Some(Matrix3::from_rows(&[
        dual_first.transpose(),
        dual_second.transpose(),
        dual_normal.transpose(),
    ]))
}

#[inline]
pub(crate) fn rotation_from_edges(
    first_camera_edge: Vector3<f64>,
    second_camera_edge: Vector3<f64>,
    world_basis_inverse: &Matrix3<f64>,
) -> Matrix3<f64> {
    Matrix3::from_columns(&[
        first_camera_edge,
        second_camera_edge,
        first_camera_edge.cross(&second_camera_edge),
    ]) * world_basis_inverse
}

#[inline]
pub(crate) fn refine_depths(
    depths: &mut Vector3<f64>,
    squared_edges: [f64; 3],
    bearing_cosines: [f64; 3],
    iterations: usize,
) {
    let [a01, a02, a12] = squared_edges;
    let [m01, m02, m12] = bearing_cosines;
    let residual_tolerance = 1e-10 * a01.max(a02).max(a12);
    for _ in 0..iterations {
        let [d0, d1, d2] = [depths[0], depths[1], depths[2]];
        let residual = Vector3::new(
            d0 * d0 - 2.0 * d0 * d1 * m01 + d1 * d1 - a01,
            d0 * d0 - 2.0 * d0 * d2 * m02 + d2 * d2 - a02,
            d1 * d1 - 2.0 * d1 * d2 * m12 + d2 * d2 - a12,
        );
        let residual_norm = residual.abs().sum();
        if residual_norm < residual_tolerance {
            break;
        }
        let x11 = d0 - d1 * m01;
        let x12 = d1 - d0 * m01;
        let x21 = d0 - d2 * m02;
        let x23 = d2 - d0 * m02;
        let x32 = d1 - d2 * m12;
        let x33 = d2 - d1 * m12;
        let denominator = x11 * x23 * x32 + x12 * x21 * x33;
        if !denominator.is_finite() || denominator == 0.0 {
            break;
        }
        let inverse = 0.5 / denominator;
        let candidate = Vector3::new(
            d0 + (-x23 * x32 * residual[0] - x12 * x33 * residual[1] + x12 * x23 * residual[2])
                * inverse,
            d1 + (-x21 * x33 * residual[0] + x11 * x33 * residual[1] - x11 * x23 * residual[2])
                * inverse,
            d2 + (x21 * x32 * residual[0] - x11 * x32 * residual[1] - x12 * x21 * residual[2])
                * inverse,
        );
        if candidate.iter().any(|value| !value.is_finite()) {
            break;
        }
        *depths = candidate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_world_basis_inverse_matches_identity() {
        let first = Vector3::new(2.0, -1.0, 0.5);
        let second = Vector3::new(-0.3, 1.7, 2.1);
        let basis = Matrix3::from_columns(&[first, second, first.cross(&second)]);
        let inverse = world_basis_inverse(first, second).unwrap();
        assert!((inverse * basis - Matrix3::identity()).norm() < 1e-14);
    }

    #[test]
    fn projected_pose_reanchors_translation_and_validates_rays() {
        let offset = Vector3::new(1.0e7, -2.0e7, 3.0e7);
        let world = [
            Point3::from(offset + Vector3::new(-1.0, -0.5, 4.0)),
            Point3::from(offset + Vector3::new(1.0, -0.5, 5.0)),
            Point3::from(offset + Vector3::new(0.2, 1.0, 6.0)),
        ];
        let raw_rotation = Matrix3::new(1.0 + 1e-9, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let camera0 = Vector3::new(-1.0, -0.5, 4.0);
        let raw_translation = camera0 - raw_rotation * world[0].coords;
        let bearings = world
            .map(|point| UnitVector3::new_normalize(raw_rotation * point.coords + raw_translation));
        let input = P3pInput::new(world, bearings).unwrap();
        let mut output = SolverP3pSolutions::default();
        input.push_pose_with_cheirality(&mut output, raw_rotation, raw_translation, false);

        assert_eq!(output.len(), 1);
        for (point, bearing) in world.iter().zip(&bearings) {
            let camera = output[0].transform_point(point).coords;
            let angle = camera.cross(bearing).norm().atan2(camera.dot(bearing));
            assert!(angle < 1e-8, "projected angular residual was {angle:e}");
        }
    }
}
