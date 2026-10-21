// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free analytic PnP linearization and LM control.

use super::projection::{
    Projection, ReprojectionLinearization, angular_projection, pixel_projection,
};
use super::{PnpError, PnpReport, PnpResidualUnit, RefinementPose};
use openmvg_ba::numerics::manifold;
use openmvg_ba::numerics::refine::{
    LmProblem, LmSolution, Loss, NoiseModel, NormalEquations, Options, minimize,
    normal_equation_diagnostics,
};
use openmvg_core::camera::PinholeCamera;

pub(super) const POSE_DIMENSION: usize = 6;
const MINIMUM_POINTS: usize = 3;

struct Linearization {
    equations: NormalEquations<POSE_DIMENSION>,
    cost: f64,
    squared_residual: f64,
}

#[derive(Clone, Copy)]
enum Measurements<'a> {
    Pixels {
        observations: &'a [[f64; 2]],
        camera: &'a PinholeCamera,
    },
    Bearings(&'a [[f64; 3]]),
}

impl Measurements<'_> {
    fn len(self) -> usize {
        match self {
            Self::Pixels { observations, .. } => observations.len(),
            Self::Bearings(observations) => observations.len(),
        }
    }

    fn residual_units(self) -> PnpResidualUnit {
        match self {
            Self::Pixels { .. } => PnpResidualUnit::Pixels,
            Self::Bearings(_) => PnpResidualUnit::Radians,
        }
    }

    fn project(self, index: usize, camera_point: &[f64; 3]) -> Option<Projection> {
        match self {
            Self::Pixels {
                observations,
                camera,
            } => pixel_projection(camera, &observations[index], camera_point),
            Self::Bearings(observations) => {
                directed_angular_projection(&observations[index], camera_point)
            }
        }
    }
}

#[derive(Clone, Copy)]
enum PoseUpdate {
    Product,
    CoupledSe3,
}

/// Refine a calibrated world-to-camera pose from fixed 3D points and pixels.
///
/// Translation uses an additive product-manifold update normalized by the
/// largest initial camera-frame coordinate. The function performs no heap
/// allocation. `noise` is expressed in pixels and contains either one shared
/// standard deviation or one value per correspondence.
pub fn optimize_pixel_pnp(
    initial_pose: &RefinementPose,
    world_points: &[[f64; 3]],
    observations: &[[f64; 2]],
    camera: &PinholeCamera,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<PnpReport, PnpError> {
    options.try_validate()?;
    validate_problem(initial_pose, world_points, observations, camera, noise)?;
    optimize(
        initial_pose,
        world_points,
        Measurements::Pixels {
            observations,
            camera,
        },
        PoseUpdate::Product,
        options,
        noise,
    )
}

/// Refine a calibrated world-to-camera pose from fixed 3D points and bearings.
///
/// Residuals are exact two-dimensional `S2` logarithms in radians. Pose steps
/// use a left-multiplicative coupled `SE(3)` retraction, with translation
/// coordinates normalized by scene scale. Bearings may point anywhere on the
/// viewing sphere; validity is determined by their directed-ray depth rather
/// than by camera-space `z`. The function performs no heap allocation.
/// `noise` is expressed in radians and contains either one shared standard
/// deviation or one value per correspondence.
pub fn optimize_bearing_pnp(
    initial_pose: &RefinementPose,
    world_points: &[[f64; 3]],
    observations: &[[f64; 3]],
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<PnpReport, PnpError> {
    options.try_validate()?;
    validate_bearing_problem(initial_pose, world_points, observations, noise)?;
    optimize(
        initial_pose,
        world_points,
        Measurements::Bearings(observations),
        PoseUpdate::CoupledSe3,
        options,
        noise,
    )
}

/// Pose refinement against fixed world points.
struct PoseProblem<'a> {
    world_points: &'a [[f64; 3]],
    measurements: Measurements<'a>,
    pose_update: PoseUpdate,
    noise: NoiseModel<'a>,
    translation_scale: f64,
}

impl LmProblem<POSE_DIMENSION> for PoseProblem<'_> {
    type State = RefinementPose;

    fn linearize(
        &self,
        pose: &RefinementPose,
        loss: Loss,
    ) -> Option<(NormalEquations<POSE_DIMENSION>, f64)> {
        linearize(
            pose,
            self.world_points,
            self.measurements,
            self.pose_update,
            loss,
            self.noise,
            self.translation_scale,
        )
        .map(|linearization| (linearization.equations, linearization.cost))
    }

    fn gradient_inf_norm(&self, equations: &NormalEquations<POSE_DIMENSION>) -> f64 {
        scaled_gradient_inf_norm(equations)
    }

    fn retract(
        &self,
        pose: &RefinementPose,
        step: &[f64; POSE_DIMENSION],
    ) -> Option<RefinementPose> {
        retract_pose(pose, step, self.translation_scale, self.pose_update)
    }

    fn cost(&self, pose: &RefinementPose, loss: Loss) -> Option<f64> {
        metrics(pose, self.world_points, self.measurements, loss, self.noise)
            .ok()
            .map(|(cost, _)| cost)
    }
}

fn optimize(
    initial_pose: &RefinementPose,
    world_points: &[[f64; 3]],
    measurements: Measurements<'_>,
    pose_update: PoseUpdate,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<PnpReport, PnpError> {
    let translation_scale = translation_scale(initial_pose, world_points)?;
    let initial = metrics(
        initial_pose,
        world_points,
        measurements,
        options.loss,
        noise,
    )
    .map_err(|point| PnpError::PointNotProjectable { point })?;
    let problem = PoseProblem {
        world_points,
        measurements,
        pose_update,
        noise,
        translation_scale,
    };
    let LmSolution {
        state: pose,
        progress,
        damping,
    } = minimize(&problem, *initial_pose, options);

    let final_linearization = linearize(
        &pose,
        world_points,
        measurements,
        pose_update,
        options.loss,
        noise,
        translation_scale,
    );
    let diagnostics = final_linearization.as_ref().map_or_else(
        || {
            normal_equation_diagnostics(
                &NormalEquations::<POSE_DIMENSION>::new(),
                measurements.len() * 2,
            )
        },
        |linearization| {
            normal_equation_diagnostics(&linearization.equations, measurements.len() * 2)
        },
    );
    let (final_objective, final_squared_residual, gradient_inf_norm) = final_linearization.map_or(
        (f64::INFINITY, f64::INFINITY, f64::INFINITY),
        |linearization| {
            (
                linearization.cost,
                linearization.squared_residual,
                scaled_gradient_inf_norm(&linearization.equations),
            )
        },
    );
    Ok(PnpReport {
        pose,
        loss: options.loss,
        residual_units: measurements.residual_units(),
        status: progress.status(diagnostics),
        point_count: world_points.len(),
        translation_scale,
        iteration_count: progress.iteration_count,
        accepted_step_count: progress.accepted_step_count,
        rejected_step_count: progress.rejected_step_count,
        initial_objective: initial.0,
        final_objective,
        initial_rmse: (initial.1 / (2 * world_points.len()) as f64).sqrt(),
        final_rmse: (final_squared_residual / (2 * world_points.len()) as f64).sqrt(),
        gradient_inf_norm,
        step_norm: progress.step_norm,
        final_damping: damping.value(),
        last_gain_ratio: damping.last_gain_ratio(),
        diagnostics,
    })
}

pub(super) fn validate_problem(
    initial_pose: &RefinementPose,
    world_points: &[[f64; 3]],
    observations: &[[f64; 2]],
    camera: &PinholeCamera,
    noise: NoiseModel<'_>,
) -> Result<(), PnpError> {
    validate_common(initial_pose, world_points, observations.len(), noise)?;
    for (observation, value) in observations.iter().enumerate() {
        if let Some(coordinate) = value.iter().position(|value| !value.is_finite()) {
            return Err(PnpError::NonFiniteObservation {
                observation,
                coordinate,
            });
        }
    }
    for (point, world_point) in world_points.iter().enumerate() {
        let camera_point = transform(initial_pose, world_point);
        if pixel_projection(camera, &observations[point], &camera_point).is_none() {
            return Err(PnpError::PointNotProjectable { point });
        }
    }
    Ok(())
}

fn validate_bearing_problem(
    initial_pose: &RefinementPose,
    world_points: &[[f64; 3]],
    observations: &[[f64; 3]],
    noise: NoiseModel<'_>,
) -> Result<(), PnpError> {
    validate_common(initial_pose, world_points, observations.len(), noise)?;
    for (observation, bearing) in observations.iter().enumerate() {
        if let Some(coordinate) = bearing.iter().position(|value| !value.is_finite()) {
            return Err(PnpError::NonFiniteObservation {
                observation,
                coordinate,
            });
        }
        let norm_squared = squared_norm(bearing);
        if !(norm_squared.is_finite() && norm_squared > f64::MIN_POSITIVE) {
            return Err(PnpError::InvalidBearing { observation });
        }
    }
    for (point, world_point) in world_points.iter().enumerate() {
        let camera_point = transform(initial_pose, world_point);
        if directed_angular_projection(&observations[point], &camera_point).is_none() {
            return Err(PnpError::PointNotProjectable { point });
        }
    }
    Ok(())
}

fn validate_common(
    initial_pose: &RefinementPose,
    world_points: &[[f64; 3]],
    observation_count: usize,
    noise: NoiseModel<'_>,
) -> Result<(), PnpError> {
    if world_points.len() != observation_count {
        return Err(PnpError::MeasurementCountMismatch {
            points: world_points.len(),
            observations: observation_count,
        });
    }
    if world_points.len() < MINIMUM_POINTS {
        return Err(PnpError::InsufficientPoints {
            required: MINIMUM_POINTS,
            actual: world_points.len(),
        });
    }
    for (point, value) in world_points.iter().enumerate() {
        if let Some(coordinate) = value.iter().position(|value| !value.is_finite()) {
            return Err(PnpError::NonFinitePoint { point, coordinate });
        }
    }
    if initial_pose.try_as_pose3().is_err() {
        return Err(PnpError::InvalidPose);
    }
    noise.try_validate(world_points.len())?;
    Ok(())
}

fn linearize(
    pose: &RefinementPose,
    world_points: &[[f64; 3]],
    measurements: Measurements<'_>,
    pose_update: PoseUpdate,
    loss: Loss,
    noise: NoiseModel<'_>,
    translation_scale: f64,
) -> Option<Linearization> {
    let mut equations = NormalEquations::new();
    let mut cost = 0.0;
    let mut squared_residual = 0.0;
    for (index, world_point) in world_points.iter().enumerate() {
        let factor = observation_factor(
            pose,
            world_point,
            measurements,
            index,
            translation_scale,
            pose_update,
        )?;
        let residual_norm = norm(&factor.residual);
        let inverse_sigma = noise.try_inverse_standard_deviation(index).ok()?;
        let sqrt_weight = inverse_sigma * loss.sqrt_weight(residual_norm * inverse_sigma);
        for residual in 0..2 {
            equations.accumulate_scalar(
                factor.residual[residual],
                &factor.camera_jacobian[residual],
                sqrt_weight,
            );
        }
        cost += loss.cost(residual_norm * inverse_sigma);
        squared_residual += residual_norm * residual_norm;
    }
    Some(Linearization {
        equations,
        cost,
        squared_residual,
    })
}

#[cfg(test)]
pub(super) fn reprojection_factor(
    pose: &RefinementPose,
    world_point: &[f64; 3],
    observation: &[f64; 2],
    camera: &PinholeCamera,
    translation_scale: f64,
) -> Option<ReprojectionLinearization<POSE_DIMENSION>> {
    observation_factor(
        pose,
        world_point,
        Measurements::Pixels {
            observations: std::slice::from_ref(observation),
            camera,
        },
        0,
        translation_scale,
        PoseUpdate::Product,
    )
}

#[cfg(test)]
fn bearing_factor(
    pose: &RefinementPose,
    world_point: &[f64; 3],
    observation: &[f64; 3],
    translation_scale: f64,
) -> Option<ReprojectionLinearization<POSE_DIMENSION>> {
    observation_factor(
        pose,
        world_point,
        Measurements::Bearings(std::slice::from_ref(observation)),
        0,
        translation_scale,
        PoseUpdate::CoupledSe3,
    )
}

fn observation_factor(
    pose: &RefinementPose,
    world_point: &[f64; 3],
    measurements: Measurements<'_>,
    observation: usize,
    translation_scale: f64,
    pose_update: PoseUpdate,
) -> Option<ReprojectionLinearization<POSE_DIMENSION>> {
    let rotated = rotate(&pose.rotation, world_point);
    let camera_point =
        std::array::from_fn(|coordinate| rotated[coordinate] + pose.translation[coordinate]);
    let projection = measurements.project(observation, &camera_point)?;
    let motion = match pose_update {
        PoseUpdate::Product => product_pose_motion(&rotated, translation_scale),
        PoseUpdate::CoupledSe3 => coupled_pose_motion(&camera_point, translation_scale),
    };
    let jacobian = multiply_2x3_3x6(&projection.jacobian, &motion);
    Some(ReprojectionLinearization::new(projection, jacobian))
}

fn product_pose_motion(rotated: &[f64; 3], translation_scale: f64) -> [[f64; POSE_DIMENSION]; 3] {
    [
        [0.0, rotated[2], -rotated[1], translation_scale, 0.0, 0.0],
        [-rotated[2], 0.0, rotated[0], 0.0, translation_scale, 0.0],
        [rotated[1], -rotated[0], 0.0, 0.0, 0.0, translation_scale],
    ]
}

fn coupled_pose_motion(
    camera_point: &[f64; 3],
    translation_scale: f64,
) -> [[f64; POSE_DIMENSION]; 3] {
    [
        [
            0.0,
            camera_point[2],
            -camera_point[1],
            translation_scale,
            0.0,
            0.0,
        ],
        [
            -camera_point[2],
            0.0,
            camera_point[0],
            0.0,
            translation_scale,
            0.0,
        ],
        [
            camera_point[1],
            -camera_point[0],
            0.0,
            0.0,
            0.0,
            translation_scale,
        ],
    ]
}

fn multiply_2x3_3x6(
    left: &[[f64; 3]; 2],
    right: &[[f64; POSE_DIMENSION]; 3],
) -> [[f64; POSE_DIMENSION]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum()
        })
    })
}

fn metrics(
    pose: &RefinementPose,
    world_points: &[[f64; 3]],
    measurements: Measurements<'_>,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Result<(f64, f64), usize> {
    let mut cost = 0.0;
    let mut squared_residual = 0.0;
    for (index, world_point) in world_points.iter().enumerate() {
        let camera_point = transform(pose, world_point);
        let projection = measurements.project(index, &camera_point).ok_or(index)?;
        let residual_norm = norm(&projection.residual);
        let inverse_sigma = noise
            .try_inverse_standard_deviation(index)
            .map_err(|_| index)?;
        cost += loss.cost(residual_norm * inverse_sigma);
        squared_residual += residual_norm * residual_norm;
    }
    Ok((cost, squared_residual))
}

pub(super) fn translation_scale(
    pose: &RefinementPose,
    world_points: &[[f64; 3]],
) -> Result<f64, PnpError> {
    let scale = world_points
        .iter()
        .flat_map(|point| transform(pose, point))
        .map(f64::abs)
        .fold(0.0, f64::max);
    if scale.is_finite() && scale > f64::MIN_POSITIVE {
        Ok(scale)
    } else {
        Err(PnpError::InvalidPose)
    }
}

fn retract_product_pose(
    pose: &RefinementPose,
    step: &[f64; POSE_DIMENSION],
    translation_scale: f64,
) -> RefinementPose {
    let update = manifold::exp_so3(&[step[0], step[1], step[2]]);
    RefinementPose {
        rotation: manifold::mat_mul(&update, &pose.rotation),
        translation: std::array::from_fn(|coordinate| {
            pose.translation[coordinate] + translation_scale * step[3 + coordinate]
        }),
    }
}

fn retract_pose(
    pose: &RefinementPose,
    step: &[f64; POSE_DIMENSION],
    translation_scale: f64,
    pose_update: PoseUpdate,
) -> Option<RefinementPose> {
    match pose_update {
        PoseUpdate::Product => Some(retract_product_pose(pose, step, translation_scale)),
        PoseUpdate::CoupledSe3 => {
            let tangent = [
                step[0],
                step[1],
                step[2],
                translation_scale * step[3],
                translation_scale * step[4],
                translation_scale * step[5],
            ];
            let update = manifold::try_exp_se3(&tangent)?;
            let pose = pose.try_as_pose3().ok()?;
            update.try_compose(pose).ok().map(Into::into)
        }
    }
}

fn directed_angular_projection(observation: &[f64; 3], point: &[f64; 3]) -> Option<Projection> {
    let observation_norm_squared = squared_norm(observation);
    let point_norm_squared = squared_norm(point);
    if !(observation_norm_squared.is_finite()
        && observation_norm_squared > f64::MIN_POSITIVE
        && point_norm_squared.is_finite()
        && point_norm_squared > f64::MIN_POSITIVE)
    {
        return None;
    }
    let depth = dot(observation, point);
    let depth_tolerance =
        64.0 * f64::EPSILON * (observation_norm_squared * point_norm_squared).sqrt();
    if !(depth.is_finite() && depth > depth_tolerance) {
        return None;
    }
    angular_projection(observation, point)
}

fn rotate(rotation: &[[f64; 3]; 3], point: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        (0..3)
            .map(|column| rotation[row][column] * point[column])
            .sum()
    })
}

fn transform(pose: &RefinementPose, point: &[f64; 3]) -> [f64; 3] {
    let rotated = rotate(&pose.rotation, point);
    std::array::from_fn(|coordinate| rotated[coordinate] + pose.translation[coordinate])
}

fn scaled_gradient_inf_norm(equations: &NormalEquations<POSE_DIMENSION>) -> f64 {
    let scale = equations
        .hessian()
        .iter()
        .enumerate()
        .map(|(index, row)| row[index])
        .fold(0.0, f64::max);
    if !(scale.is_finite() && scale > 0.0) {
        return f64::INFINITY;
    }
    let floor = 64.0 * f64::EPSILON * scale;
    equations
        .gradient()
        .iter()
        .enumerate()
        .map(|(index, gradient)| {
            gradient.abs() / equations.hessian()[index][index].max(floor).sqrt()
        })
        .fold(0.0, f64::max)
}

fn norm<const N: usize>(vector: &[f64; N]) -> f64 {
    squared_norm(vector).sqrt()
}

fn squared_norm<const N: usize>(vector: &[f64; N]) -> f64 {
    vector.iter().map(|value| value * value).sum()
}

fn dot<const N: usize>(left: &[f64; N], right: &[f64; N]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PnP pose Jacobian finite-difference parity: the non-identity pose Jacobian matches an
    /// independent central-difference oracle; all six pose-chart directions are checked in pixel
    /// residual space.
    #[test]
    fn analytic_pose_jacobian_matches_central_differences() {
        let pose = RefinementPose::new(manifold::exp_so3(&[0.12, -0.07, 0.04]), [0.2, -0.1, 0.35]);
        let point = [0.7, -0.4, 4.2];
        let camera = PinholeCamera::try_new(730.0, 690.0, 321.0, 239.0).unwrap();
        let observation = [418.0, 171.0];
        let scale = 4.8;
        let factor = reprojection_factor(&pose, &point, &observation, &camera, scale).unwrap();
        let epsilon = 1e-7;

        for parameter in 0..POSE_DIMENSION {
            let mut plus = [0.0; POSE_DIMENSION];
            let mut minus = [0.0; POSE_DIMENSION];
            plus[parameter] = epsilon;
            minus[parameter] = -epsilon;
            let plus_pose = retract_product_pose(&pose, &plus, scale);
            let minus_pose = retract_product_pose(&pose, &minus, scale);
            let plus_factor =
                reprojection_factor(&plus_pose, &point, &observation, &camera, scale).unwrap();
            let minus_factor =
                reprojection_factor(&minus_pose, &point, &observation, &camera, scale).unwrap();
            for residual in 0..2 {
                let numerical = (plus_factor.residual[residual] - minus_factor.residual[residual])
                    / (2.0 * epsilon);
                let analytic = factor.camera_jacobian[residual][parameter];
                let tolerance = 2e-6 * analytic.abs().max(1.0);
                assert!(
                    (analytic - numerical).abs() <= tolerance,
                    "residual={residual}, parameter={parameter}, analytic={analytic}, numerical={numerical}"
                );
            }
        }
    }

    #[test]
    fn angular_se3_jacobian_matches_central_differences_for_rear_bearing() {
        let pose = RefinementPose::new(manifold::exp_so3(&[0.08, -0.05, 0.03]), [0.2, -0.1, 0.15]);
        let point = [-1.1, 0.4, -4.2];
        let camera_point = transform(&pose, &point);
        assert!(camera_point[2] < 0.0);
        let mut observation = [
            camera_point[0] + 0.03,
            camera_point[1] - 0.02,
            camera_point[2] + 0.01,
        ];
        let inverse_norm = norm(&observation).recip();
        observation = observation.map(|value| value * inverse_norm);
        let scale = 4.8;
        let factor = bearing_factor(&pose, &point, &observation, scale).unwrap();
        let epsilon = 1e-7;

        for parameter in 0..POSE_DIMENSION {
            let mut plus = [0.0; POSE_DIMENSION];
            let mut minus = [0.0; POSE_DIMENSION];
            plus[parameter] = epsilon;
            minus[parameter] = -epsilon;
            let plus_pose = retract_pose(&pose, &plus, scale, PoseUpdate::CoupledSe3).unwrap();
            let minus_pose = retract_pose(&pose, &minus, scale, PoseUpdate::CoupledSe3).unwrap();
            let plus_factor = bearing_factor(&plus_pose, &point, &observation, scale).unwrap();
            let minus_factor = bearing_factor(&minus_pose, &point, &observation, scale).unwrap();
            for residual in 0..2 {
                let numerical = (plus_factor.residual[residual] - minus_factor.residual[residual])
                    / (2.0 * epsilon);
                let analytic = factor.camera_jacobian[residual][parameter];
                let tolerance = 3e-6 * analytic.abs().max(1.0);
                assert!(
                    (analytic - numerical).abs() <= tolerance,
                    "residual={residual}, parameter={parameter}, analytic={analytic}, numerical={numerical}"
                );
            }
        }
    }
}
