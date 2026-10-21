// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The bounded sparse observability probe and the projection it linearizes.

use super::Matrix2x3;
use super::api::{
    FixedRotationError, FixedRotationGauge, FixedRotationIntrinsics,
    FixedRotationObservabilityProbe, FixedRotationObservabilityProbeOptions,
    FixedRotationObservation, FixedRotationProbeReason, FixedRotationProbeStatus,
    FixedRotationProblem,
};
use super::validation::{validate_probe_options, validate_problem};
use crate::kernel::point_block::{POINT_BLOCK_RCOND_THRESHOLD, reciprocal_condition_number};
use crate::numerics::refine::Loss;
use crate::numerics::sparse::{SparseCholesky, SymmetricPattern};
use nalgebra::{DMatrix, Matrix3, Rotation3, Vector2, Vector3};
use openmvg_core::camera::{
    BrownConradyCamera, CameraModel, PinholeCamera, RadialDistortion, TangentialDistortion,
};
use std::collections::BTreeMap;

/// Test a deterministic capped subset of fixed-rotation tracks with a sparse
/// gauge-reduced Schur factorization.
///
/// [`FixedRotationProbeStatus::FullRankCertified`] is a one-way admission
/// certificate: adding the omitted positive-semidefinite track contributions
/// cannot destroy full rank. [`FixedRotationProbeStatus::NotCertified`] is not
/// evidence that the full problem is deficient and should trigger deferment or
/// a more complete assessment.
pub fn probe_fixed_rotation_observability(
    problem: FixedRotationProblem<'_>,
    loss: Loss,
    options: FixedRotationObservabilityProbeOptions,
) -> Result<FixedRotationObservabilityProbe, FixedRotationError> {
    validate_problem(problem)?;
    loss.try_validate()
        .map_err(|_| FixedRotationError::InvalidOptions("invalid robust loss"))?;
    validate_probe_options(options)?;

    let rotations = problem
        .rotations
        .iter()
        .map(|rotation| {
            Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(&[
                rotation[0][0],
                rotation[0][1],
                rotation[0][2],
                rotation[1][0],
                rotation[1][1],
                rotation[1][2],
                rotation[2][0],
                rotation[2][1],
                rotation[2][2],
            ]))
        })
        .collect::<Vec<_>>();
    let centers = problem
        .centers
        .iter()
        .copied()
        .map(Vector3::from)
        .collect::<Vec<_>>();
    let points = problem
        .points
        .iter()
        .copied()
        .map(Vector3::from)
        .collect::<Vec<_>>();
    let constraints = gauge_constraints(problem.gauge, problem.fixed_cameras, &centers)?;
    let (free_coordinates, camera_dimension) = free_coordinate_map(&constraints);
    if camera_dimension == 0 {
        return Ok(FixedRotationObservabilityProbe {
            status: FixedRotationProbeStatus::FullRankCertified,
            pose_dimension: camera_dimension,
            selected_tracks: 0,
            selected_observations: 0,
            covered_free_cameras: 0,
            weak_point_count: 0,
            reduced_nonzeros: 0,
            factor_nonzeros: 0,
        });
    }
    let free_cameras = constraints
        .fixed
        .iter()
        .map(|components| components.iter().any(|fixed| !fixed))
        .collect::<Vec<_>>();
    let required_free_cameras = free_cameras.iter().filter(|free| **free).count();
    let selected = select_probe_tracks(problem.observations, &free_cameras, options);
    let selected_observations = selected
        .iter()
        .map(|&point| problem.observations[point].len())
        .sum();
    let mut report = FixedRotationObservabilityProbe {
        status: FixedRotationProbeStatus::NotCertified(
            FixedRotationProbeReason::InsufficientCameraCoverage,
        ),
        pose_dimension: camera_dimension,
        selected_tracks: selected.len(),
        selected_observations,
        covered_free_cameras: 0,
        weak_point_count: 0,
        reduced_nonzeros: 0,
        factor_nonzeros: 0,
    };
    if selected.len() > options.max_tracks
        || selected_observations > options.max_observations
        || camera_dimension > options.max_reduced_nonzeros
    {
        report.status =
            FixedRotationProbeStatus::NotCertified(FixedRotationProbeReason::ResourceLimit);
        return Ok(report);
    }

    let mut entries = BTreeMap::<(usize, usize), f64>::new();
    for diagonal in 0..camera_dimension {
        entries.insert((diagonal, diagonal), 0.0);
    }
    let mut covered = vec![false; free_cameras.len()];
    // Information each free coordinate receives before the points are eliminated.
    let mut raw_information = vec![0.0f64; camera_dimension];
    'tracks: for point_index in selected {
        let track = &problem.observations[point_index];
        let mut blocks = Vec::with_capacity(track.len());
        for observation in track {
            // A point that one of its cameras cannot project has no usable block: it is weak.
            let Ok(linearized) = linearize(
                rotations[observation.camera],
                centers[observation.camera],
                points[point_index],
                problem.intrinsics[observation.camera],
                Vector2::from(observation.xy),
                loss,
            ) else {
                report.weak_point_count += 1;
                continue 'tracks;
            };
            blocks.push((
                observation.camera,
                linearized.weight.sqrt() * linearized.point_jacobian,
            ));
        }
        // The square-root form is positive semidefinite by construction; the explicit
        // `N_a - N_a C^-1 N_b` loses the camera system's diagonal to rounding when a point
        // block is stiff.
        let jacobians = blocks.iter().map(|block| block.1).collect::<Vec<_>>();
        let Some(schur) = rank_point_schur(&jacobians) else {
            report.weak_point_count += 1;
            continue;
        };
        for (camera, _) in &blocks {
            if free_cameras[*camera] {
                covered[*camera] = true;
            }
        }
        for (camera, jacobian) in &blocks {
            for local in 0..3 {
                if let Some(coordinate) = free_coordinates[3 * camera + local] {
                    raw_information[coordinate] += jacobian.column(local).norm_squared();
                }
            }
        }
        let free_scalar_count = blocks
            .iter()
            .map(|(camera, _)| {
                free_coordinates[3 * camera..3 * camera + 3]
                    .iter()
                    .filter(|coordinate| coordinate.is_some())
                    .count()
            })
            .sum::<usize>();
        let contribution_bound = free_scalar_count
            .checked_mul(free_scalar_count + 1)
            .map(|value| value / 2)
            .unwrap_or(usize::MAX);
        if entries.len().saturating_add(contribution_bound) > options.max_reduced_nonzeros {
            report.status =
                FixedRotationProbeStatus::NotCertified(FixedRotationProbeReason::ResourceLimit);
            report.covered_free_cameras = covered.iter().filter(|value| **value).count();
            report.reduced_nonzeros = entries.len();
            return Ok(report);
        }
        for (index_a, (camera_a, _)) in blocks.iter().enumerate() {
            for (index_b, (camera_b, _)) in blocks.iter().enumerate() {
                let block = schur
                    .fixed_view::<3, 3>(3 * index_a, 3 * index_b)
                    .into_owned();
                add_sparse_free_block(&mut entries, *camera_a, *camera_b, block, &free_coordinates);
            }
        }
    }
    report.covered_free_cameras = covered.iter().filter(|value| **value).count();
    report.reduced_nonzeros = entries.len();
    if report.covered_free_cameras != required_free_cameras {
        report.status = FixedRotationProbeStatus::NotCertified(
            FixedRotationProbeReason::InsufficientCameraCoverage,
        );
        return Ok(report);
    }
    if report.weak_point_count == report.selected_tracks {
        report.status =
            FixedRotationProbeStatus::NotCertified(FixedRotationProbeReason::WeakPointGeometry);
        return Ok(report);
    }

    let diagonal = (0..camera_dimension)
        .map(|index| entries.get(&(index, index)).copied().unwrap_or(0.0))
        .collect::<Vec<_>>();
    // Equilibrating by the diagonal turns any positive entry into a unit pivot, so a coordinate
    // whose information the points absorb entirely (a baseline length that only two-view tracks
    // see) would pass on rounding noise alone. It must keep a fraction of what it started with.
    if diagonal.iter().zip(&raw_information).any(|(value, raw)| {
        !(value.is_finite() && *value > f64::MIN_POSITIVE)
            || *value <= options.minimum_normalized_eigenvalue * raw
    }) {
        report.status =
            FixedRotationProbeStatus::NotCertified(FixedRotationProbeReason::WeakPointGeometry);
        return Ok(report);
    }
    for (&(column, row), value) in &mut entries {
        *value /= (diagonal[row] * diagonal[column]).sqrt();
        if row == column {
            *value -= options.minimum_normalized_eigenvalue;
        }
    }
    if entries.values().any(|value| !value.is_finite()) {
        report.status =
            FixedRotationProbeStatus::NotCertified(FixedRotationProbeReason::WeakPointGeometry);
        return Ok(report);
    }

    let (pattern, values) = sparse_lower_triangle(camera_dimension, entries);
    let worst_case_factor_nonzeros = camera_dimension
        .checked_add(1)
        .and_then(|next| camera_dimension.checked_mul(next))
        .map(|value| value / 2)
        .unwrap_or(usize::MAX);
    if worst_case_factor_nonzeros > options.max_symbolic_dense_nonzeros {
        report.status =
            FixedRotationProbeStatus::NotCertified(FixedRotationProbeReason::ResourceLimit);
        return Ok(report);
    }
    // This bounded probe needs only a positive-definiteness certificate. A supernodal
    // factorization invokes Faer's large thread-local GEMM packing cache, which can dominate
    // process RSS while providing no stronger certificate.
    let Some(mut cholesky) = pattern.and_then(SparseCholesky::analyze_simplicial) else {
        report.status = FixedRotationProbeStatus::NotCertified(
            FixedRotationProbeReason::SparseFactorizationFailed,
        );
        return Ok(report);
    };
    report.factor_nonzeros = cholesky.factor_value_count();
    if report.factor_nonzeros > options.max_factor_nonzeros {
        report.status =
            FixedRotationProbeStatus::NotCertified(FixedRotationProbeReason::ResourceLimit);
        return Ok(report);
    }
    if !cholesky.reserve(0) {
        report.status =
            FixedRotationProbeStatus::NotCertified(FixedRotationProbeReason::ResourceLimit);
        return Ok(report);
    }
    report.status = if cholesky.factor(&values) {
        FixedRotationProbeStatus::FullRankCertified
    } else {
        FixedRotationProbeStatus::NotCertified(FixedRotationProbeReason::SparseFactorizationFailed)
    };
    Ok(report)
}

fn free_coordinate_map(constraints: &GaugeConstraints) -> (Vec<Option<usize>>, usize) {
    let mut next = 0usize;
    let coordinates = constraints
        .fixed
        .iter()
        .flat_map(|components| components.iter())
        .map(|fixed| {
            if *fixed {
                None
            } else {
                let coordinate = next;
                next += 1;
                Some(coordinate)
            }
        })
        .collect();
    (coordinates, next)
}

fn select_probe_tracks(
    observations: &[Vec<FixedRotationObservation>],
    free_cameras: &[bool],
    options: FixedRotationObservabilityProbeOptions,
) -> Vec<usize> {
    let mut candidates = observations
        .iter()
        .enumerate()
        .filter(|(_, track)| {
            track.len() <= options.max_coverage_observations_per_track
                && track
                    .iter()
                    .any(|observation| free_cameras[observation.camera])
        })
        .map(|(point, track)| (track.len(), point))
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    let mut selected = vec![false; observations.len()];
    let mut covered = vec![false; free_cameras.len()];
    let mut selected_tracks = 0usize;
    let mut selected_observations = 0usize;
    for &(_, point) in &candidates {
        let track = &observations[point];
        if track.len() > options.max_observations_per_track {
            continue;
        }
        let adds_coverage = track
            .iter()
            .any(|observation| free_cameras[observation.camera] && !covered[observation.camera]);
        if !adds_coverage
            || selected_tracks == options.max_tracks
            || selected_observations.saturating_add(track.len()) > options.max_observations
        {
            continue;
        }
        selected[point] = true;
        selected_tracks += 1;
        selected_observations += track.len();
        for observation in track {
            if free_cameras[observation.camera] {
                covered[observation.camera] = true;
            }
        }
    }
    if covered
        .iter()
        .zip(free_cameras)
        .any(|(covered, free)| *free && !covered)
    {
        for &(_, point) in &candidates {
            let track = &observations[point];
            if track.len() <= options.max_observations_per_track || selected[point] {
                continue;
            }
            let adds_coverage = track.iter().any(|observation| {
                free_cameras[observation.camera] && !covered[observation.camera]
            });
            if !adds_coverage
                || selected_tracks == options.max_tracks
                || selected_observations.saturating_add(track.len()) > options.max_observations
            {
                continue;
            }
            selected[point] = true;
            selected_tracks += 1;
            selected_observations += track.len();
            for observation in track {
                if free_cameras[observation.camera] {
                    covered[observation.camera] = true;
                }
            }
        }
    }
    if covered
        .iter()
        .zip(free_cameras)
        .any(|(covered, free)| *free && !covered)
    {
        return selected
            .iter()
            .enumerate()
            .filter_map(|(point, selected)| selected.then_some(point))
            .collect();
    }
    for &(_, point) in &candidates {
        if selected[point] || observations[point].len() > options.max_observations_per_track {
            continue;
        }
        let count = observations[point].len();
        if selected_tracks == options.max_tracks
            || selected_observations.saturating_add(count) > options.max_observations
        {
            continue;
        }
        selected[point] = true;
        selected_tracks += 1;
        selected_observations += count;
    }
    selected
        .iter()
        .enumerate()
        .filter_map(|(point, selected)| selected.then_some(point))
        .collect()
}

fn add_sparse_free_block(
    entries: &mut BTreeMap<(usize, usize), f64>,
    camera_a: usize,
    camera_b: usize,
    block: Matrix3<f64>,
    free_coordinates: &[Option<usize>],
) {
    for local_a in 0..3 {
        let Some(row) = free_coordinates[3 * camera_a + local_a] else {
            continue;
        };
        for local_b in 0..3 {
            let Some(column) = free_coordinates[3 * camera_b + local_b] else {
                continue;
            };
            if row >= column {
                *entries.entry((column, row)).or_default() += block[(local_a, local_b)];
            }
        }
    }
}

fn sparse_lower_triangle(
    dimension: usize,
    entries: BTreeMap<(usize, usize), f64>,
) -> (Option<SymmetricPattern>, Vec<f64>) {
    let mut col_ptr = vec![0usize; dimension + 1];
    let mut row_idx = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    let mut current_column = 0usize;
    for ((column, row), value) in entries {
        while current_column < column {
            current_column += 1;
            col_ptr[current_column] = row_idx.len();
        }
        row_idx.push(row);
        values.push(value);
    }
    while current_column < dimension {
        current_column += 1;
        col_ptr[current_column] = row_idx.len();
    }
    (SymmetricPattern::new(dimension, col_ptr, row_idx), values)
}

/// Whether a point block is weak for rank assessment: it fails the shared point-block weak
/// criterion of the calibrated kernels, or it has no inverse. `try_inverse` fails exactly on a
/// zero determinant, which a block passing the criterion reaches only through underflow.
fn is_weak_point_block(hessian: &Matrix3<f64>) -> bool {
    let block = std::array::from_fn(|row| std::array::from_fn(|column| hessian[(row, column)]));
    reciprocal_condition_number(&block) <= POINT_BLOCK_RCOND_THRESHOLD
        || hessian.determinant() == 0.0
}

/// Marginal camera information of one point for rank assessment, or `None` if the point is weak.
///
/// Eliminating the point from the normal equations, `N_a - N_a C^-1 N_b`, loses about
/// `eps ||N_a||^2 / lambda_min(C)`: a point that a solve has pulled onto one camera center has a
/// block many orders stiffer than the rest of that camera's evidence, and its rounding alone
/// becomes a large negative eigenvalue of the camera system. The square-root form projects the
/// whitened Jacobians `J_a` off the point's column space before squaring. With the thin QR
/// factor `Q` of the stacked `J`, `Z_a = (E_a - Q Q_a^T) J_a` and the blocks are `Z_a^T Z_b`:
/// positive semidefinite by construction, with rounding bounded by `eps ||J||`.
pub(super) fn rank_point_schur(jacobians: &[Matrix2x3]) -> Option<DMatrix<f64>> {
    if is_weak_point_block(
        &jacobians
            .iter()
            .map(|jacobian| jacobian.transpose() * jacobian)
            .sum(),
    ) {
        return None;
    }
    let rows = 2 * jacobians.len();
    let mut stacked = DMatrix::zeros(rows, 3);
    for (index, jacobian) in jacobians.iter().enumerate() {
        stacked
            .fixed_view_mut::<2, 3>(2 * index, 0)
            .copy_from(jacobian);
    }
    let basis = stacked.qr().q();
    let mut projected = DMatrix::zeros(rows, 3 * jacobians.len());
    for (index, jacobian) in jacobians.iter().enumerate() {
        let mut columns = projected.fixed_columns_mut::<3>(3 * index);
        columns -= &basis * (basis.fixed_rows::<2>(2 * index).transpose() * jacobian);
        let mut rows = columns.fixed_rows_mut::<2>(2 * index);
        rows += jacobian;
    }
    Some(projected.transpose() * projected)
}

#[derive(Clone, Copy)]
pub(super) struct LinearizedObservation {
    pub(super) point_jacobian: Matrix2x3,
    pub(super) weight: f64,
}

pub(super) struct GaugeConstraints {
    pub(super) fixed: Vec<[bool; 3]>,
}

pub(super) fn gauge_constraints(
    gauge: FixedRotationGauge,
    fixed_cameras: &[bool],
    centers: &[Vector3<f64>],
) -> Result<GaugeConstraints, FixedRotationError> {
    let mut fixed: Vec<_> = fixed_cameras.iter().map(|fixed| [*fixed; 3]).collect();
    let (anchor, scale_camera) = match gauge {
        FixedRotationGauge::Free {
            anchor,
            scale_camera,
        }
        | FixedRotationGauge::FixedBaseline {
            anchor,
            scale_camera,
            ..
        } => (anchor, scale_camera),
    };
    fixed[anchor] = [true; 3];
    if !fixed[scale_camera].iter().all(|value| *value) {
        let baseline = centers[scale_camera] - centers[anchor];
        let component = [baseline.x.abs(), baseline.y.abs(), baseline.z.abs()]
            .into_iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .filter(|(_, magnitude)| magnitude.is_finite() && *magnitude > 0.0)
            .map(|(component, _)| component)
            .ok_or(FixedRotationError::InvalidProblem(
                "fixed-rotation BA cannot fix scale from coincident camera centers",
            ))?;
        fixed[scale_camera][component] = true;
    }
    Ok(GaugeConstraints { fixed })
}

pub(super) fn linearize(
    rotation: Rotation3<f64>,
    center: Vector3<f64>,
    point: Vector3<f64>,
    intrinsics: FixedRotationIntrinsics,
    observed: Vector2<f64>,
    loss: Loss,
) -> Result<LinearizedObservation, FixedRotationError> {
    let camera_point = rotation * (point - center);
    if !camera_point.iter().all(|value| value.is_finite()) || camera_point.z <= 0.0 {
        return Err(FixedRotationError::Numerical(
            "fixed-rotation BA encountered a point behind a camera",
        ));
    }
    let camera = PinholeCamera::try_new(
        intrinsics.focal_x,
        intrinsics.focal_y,
        intrinsics.principal_x,
        intrinsics.principal_y,
    )
    .and_then(|pinhole| {
        BrownConradyCamera::try_new(
            pinhole,
            RadialDistortion::K1K2K3(intrinsics.k1, intrinsics.k2, intrinsics.k3),
            TangentialDistortion::P1P2(intrinsics.p1, intrinsics.p2),
        )
    })
    .map_err(|_| FixedRotationError::Numerical("fixed-rotation BA intrinsics are invalid"))?;
    let (pixel, projection_jacobian) = camera
        .project_with_jacobian(&camera_point)
        .map_err(|_| FixedRotationError::Numerical("fixed-rotation BA projection is not finite"))?;
    let residual = pixel.coords - observed;
    let point_jacobian = projection_jacobian * rotation.matrix();
    let sqrt_weight = loss.relative_sqrt_weight(residual.norm());
    Ok(LinearizedObservation {
        point_jacobian,
        weight: sqrt_weight * sqrt_weight,
    })
}
