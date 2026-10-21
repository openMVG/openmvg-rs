// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Absolute pose and focal length from 3D-2D correspondences (P4Pf), for a camera whose focal
//! length is unknown.
//!
//! ```text
//! pixel + 3D point pairs -> center at the principal point -> P4Pf hypotheses (4 points, pose + f)
//!                        -> pixel reprojection score -> Levenberg-Marquardt refit on the consensus
//!                        -> pose, focal length, inliers
//! ```
//!
//! The camera is a pinhole with square pixels and a known principal point; the focal length is the
//! unknown. A four-point minimal solver makes the hypotheses, and
//! [`estimate_absolute_pose_with_focal`] scores them by pixel reprojection error and refits the
//! pose and focal length on each consensus.
//!
//! Poses are world-to-camera: `X_camera = R X_world + t`, and a pixel projects to
//! `f (X_camera.x, X_camera.y) / X_camera.z + principal_point`.

mod kernel;

use nalgebra::{Matrix3, Point2, SMatrix, SVector, Vector3};
use openmvg_core::Pixels;
use openmvg_core::model::WorldToCameraPose;

use super::{PixelToPoint, camera_pose_from_world_to_camera, nearest_rotation};
use crate::robust::error::RobustError;
use crate::robust::genransac::config::{GenRansacConfig, RobustOptions};
use crate::robust::genransac::estimator::{GenRansacEstimator, RobustGuidance};
use crate::robust::genransac::refiner::GenRansacRefiner;
use crate::robust::result::EstimateStats;
use crate::robust::threshold::PixelThreshold;
use crate::solver_traits::{MultiSolutionSolver, Solver};
use crate::{SolverError, SolverResult};

#[cfg(feature = "unstable-test-support")]
#[doc(hidden)]
pub mod test_support {
    //! Raw access to the four-point kernel for the reference fixtures.
    pub use super::kernel::testing::house_nullspace;
    pub use super::kernel::{
        FRONT_END, LANE, LANE_DECLINED, MAX_POSES, Pose as P4pfPose, p4pf as solve_p4pf_kernel,
    };
}

/// Minimum correspondences of one P4Pf hypothesis.
const MINIMAL_SAMPLE: usize = 4;
/// Levenberg-Marquardt iterations of the consensus refit.
const REFIT_ITERATIONS: usize = 20;
/// A refit needs more equations than its seven unknowns: 6 points give 12.
const MIN_REFIT_POINTS: usize = 6;

/// A camera pose together with its focal length and the principal point it assumes.
#[derive(Debug, Clone, PartialEq)]
pub struct FocalPoseModel {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    focal_px: f64,
    principal_point: Point2<f64>,
}

impl FocalPoseModel {
    /// World-to-camera rotation, row-major in the matrix.
    #[must_use]
    pub const fn rotation(&self) -> &Matrix3<f64> {
        &self.rotation
    }

    /// Translation of `X_camera = R X_world + t`.
    #[must_use]
    pub const fn translation(&self) -> &Vector3<f64> {
        &self.translation
    }

    /// Focal length in pixels.
    #[must_use]
    pub const fn focal_px(&self) -> f64 {
        self.focal_px
    }

    /// Principal point in pixels, used to center the observations.
    #[must_use]
    pub const fn principal_point(&self) -> &Point2<f64> {
        &self.principal_point
    }

    /// The pose as the kernel's rotation-plus-center value.
    ///
    /// # Errors
    ///
    /// [`SolverError::InvalidInput`] when the rotation or translation is not finite.
    pub fn pose(&self) -> SolverResult<WorldToCameraPose> {
        let isometry = nalgebra::Isometry3::from_parts(
            self.translation.into(),
            nearest_rotation(&self.rotation),
        );
        camera_pose_from_world_to_camera(&isometry).map_err(|_| SolverError::InvalidInput {
            reason: "focal pose must be finite",
        })
    }

    /// Pixel distance between a correspondence's observation and its projection, infinite for a
    /// point on or behind the camera plane.
    fn reprojection_error(&self, datum: &PixelToPoint) -> f64 {
        let camera_point = self.rotation * datum.world_point.coords + self.translation;
        if !(camera_point.z.is_finite() && camera_point.z > 0.0) {
            return f64::INFINITY;
        }
        let projected_x = self.focal_px * camera_point.x / camera_point.z + self.principal_point.x;
        let projected_y = self.focal_px * camera_point.y / camera_point.z + self.principal_point.y;
        (projected_x - datum.pixel.x).hypot(projected_y - datum.pixel.y)
    }
}

/// Per-call configuration of [`P4pfSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct P4pfConfig {
    /// Principal point in pixels; observations are centered on it before the solve.
    principal_point: Point2<f64>,
}

/// Minimal four-point solver for the pose and focal length, with up to nine candidate models.
#[derive(Debug, Clone, Copy)]
pub(super) struct P4pfSolver;

impl P4pfSolver {
    /// Candidate models of exactly four correspondences.
    ///
    /// # Errors
    ///
    /// [`SolverError::InvalidSampleSize`] unless `data` holds exactly four correspondences.
    fn solve_sample(
        data: &[PixelToPoint],
        config: &P4pfConfig,
    ) -> SolverResult<Vec<FocalPoseModel>> {
        let [first, second, third, fourth] = data else {
            return Err(SolverError::InvalidSampleSize {
                expected: MINIMAL_SAMPLE,
                provided: data.len(),
            });
        };
        let sample = [first, second, third, fourth];
        let mut models = Vec::new();
        Self::emit_models(sample, config, |model| models.push(model));
        Ok(models)
    }

    fn emit_models<F: FnMut(FocalPoseModel)>(
        sample: [&PixelToPoint; MINIMAL_SAMPLE],
        config: &P4pfConfig,
        mut emit: F,
    ) -> usize {
        let world: [[f64; 3]; MINIMAL_SAMPLE] = std::array::from_fn(|index| {
            let point = &sample[index].world_point;
            [point.x, point.y, point.z]
        });
        let image: [[f64; 2]; MINIMAL_SAMPLE] = std::array::from_fn(|index| {
            let pixel = &sample[index].pixel;
            [
                pixel.x - config.principal_point.x,
                pixel.y - config.principal_point.y,
            ]
        });
        let mut poses = [kernel::Pose::default(); kernel::MAX_POSES];
        let mut status = kernel::FRONT_END;
        let count = kernel::p4pf(&world, &image, &mut poses, &mut status);
        let mut emitted = 0;
        for pose in &poses[..count] {
            let model = FocalPoseModel {
                rotation: Matrix3::from_row_slice(&pose.r),
                translation: Vector3::from(pose.t),
                focal_px: pose.f,
                principal_point: config.principal_point,
            };
            if Self::is_valid(&model) {
                emit(model);
                emitted += 1;
            }
        }
        emitted
    }
}

impl Solver for P4pfSolver {
    type Datum = PixelToPoint;
    type Model = FocalPoseModel;
    type Config = P4pfConfig;
    type Unit = Pixels;

    fn min_sample_size() -> usize {
        MINIMAL_SAMPLE
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        model.reprojection_error(datum)
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.focal_px.is_finite()
            && model.focal_px > 0.0
            && model.translation.iter().all(|value| value.is_finite())
            && model.rotation.iter().all(|value| value.is_finite())
    }
}

impl MultiSolutionSolver for P4pfSolver {
    fn solve(data: &[Self::Datum], config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        Self::solve_sample(data, config)
    }

    fn for_each_solution_from_indices<F>(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
        emit: F,
    ) -> SolverResult<usize>
    where
        F: FnMut(Self::Model),
    {
        let &[first, second, third, fourth] = indices else {
            return Err(SolverError::InvalidSampleSize {
                expected: MINIMAL_SAMPLE,
                provided: indices.len(),
            });
        };
        let fetch = |index: usize| {
            data.get(index).ok_or(SolverError::InvalidDataIndex {
                index,
                data_len: data.len(),
            })
        };
        let sample = [fetch(first)?, fetch(second)?, fetch(third)?, fetch(fourth)?];
        Ok(Self::emit_models(sample, config, emit))
    }
}

/// Levenberg-Marquardt refit of the pose and focal length on `points`, minimizing the squared
/// pixel reprojection error. Returns `None` when the data cannot constrain the model.
fn refit_focal_pose(seed: &FocalPoseModel, points: &[&PixelToPoint]) -> Option<FocalPoseModel> {
    type Normal = SMatrix<f64, 7, 7>;
    type Gradient = SVector<f64, 7>;

    if points.len() < MIN_REFIT_POINTS {
        return None;
    }
    let cost = |model: &FocalPoseModel| -> f64 {
        points
            .iter()
            .map(|datum| {
                let error = model.reprojection_error(datum);
                error * error
            })
            .sum()
    };
    let mut model = seed.clone();
    let mut model_cost = cost(&model);
    if !model_cost.is_finite() {
        return None;
    }
    let mut damping = 1e-3;
    for _ in 0..REFIT_ITERATIONS {
        let mut normal = Normal::zeros();
        let mut gradient = Gradient::zeros();
        for datum in points {
            let rotated = model.rotation * datum.world_point.coords;
            let camera_point = rotated + model.translation;
            if !(camera_point.z.is_finite() && camera_point.z > 0.0) {
                return None;
            }
            let inverse_depth = 1.0 / camera_point.z;
            let normalized_x = camera_point.x * inverse_depth;
            let normalized_y = camera_point.y * inverse_depth;
            let residual = [
                model.focal_px * normalized_x + model.principal_point.x - datum.pixel.x,
                model.focal_px * normalized_y + model.principal_point.y - datum.pixel.y,
            ];
            // d(pixel)/d(camera point) for x and y, then the chain through the 7 unknowns:
            // a left rotation increment w (d(camera point) = -[R X]x w), the translation, and f.
            let skew = Matrix3::new(
                0.0, -rotated.z, rotated.y, rotated.z, 0.0, -rotated.x, -rotated.y, rotated.x, 0.0,
            );
            let d_point = [
                Vector3::new(
                    model.focal_px * inverse_depth,
                    0.0,
                    -model.focal_px * normalized_x * inverse_depth,
                ),
                Vector3::new(
                    0.0,
                    model.focal_px * inverse_depth,
                    -model.focal_px * normalized_y * inverse_depth,
                ),
            ];
            for (row, residual) in residual.iter().enumerate() {
                let d_rotation = -(skew.transpose() * d_point[row]);
                let mut jacobian = Gradient::zeros();
                jacobian.fixed_rows_mut::<3>(0).copy_from(&d_rotation);
                jacobian.fixed_rows_mut::<3>(3).copy_from(&d_point[row]);
                jacobian[6] = if row == 0 { normalized_x } else { normalized_y };
                normal += jacobian * jacobian.transpose();
                gradient += jacobian * *residual;
            }
        }
        let mut accepted = false;
        for _ in 0..8 {
            let mut damped = normal;
            for index in 0..7 {
                damped[(index, index)] += damping * normal[(index, index)].max(1e-12);
            }
            let Some(step) = damped.cholesky().map(|factor| factor.solve(&(-gradient))) else {
                damping *= 10.0;
                continue;
            };
            let rotation_step =
                nalgebra::Rotation3::from_scaled_axis(Vector3::new(step[0], step[1], step[2]));
            let candidate = FocalPoseModel {
                rotation: *rotation_step.matrix() * model.rotation,
                translation: model.translation + Vector3::new(step[3], step[4], step[5]),
                focal_px: model.focal_px + step[6],
                principal_point: model.principal_point,
            };
            let candidate_cost = cost(&candidate);
            if candidate_cost.is_finite() && candidate_cost < model_cost && candidate.focal_px > 0.0
            {
                let converged = (model_cost - candidate_cost) <= 1e-12 * model_cost.max(1e-30);
                model = candidate;
                model_cost = candidate_cost;
                damping = (damping * 0.3).max(1e-9);
                accepted = true;
                if converged {
                    return Some(model);
                }
                break;
            }
            damping *= 10.0;
        }
        if !accepted {
            break;
        }
    }
    P4pfSolver::is_valid(&model).then_some(model)
}

/// Refits the pose and focal length on the current consensus.
struct FocalPoseRefiner;

impl GenRansacRefiner<P4pfSolver> for FocalPoseRefiner {
    fn refit(
        &self,
        seed: &FocalPoseModel,
        inliers: &[usize],
        data: &[PixelToPoint],
        _threshold: f64,
    ) -> Result<Option<FocalPoseModel>, RobustError> {
        let points: Vec<&PixelToPoint> = inliers.iter().filter_map(|&i| data.get(i)).collect();
        Ok(refit_focal_pose(seed, &points))
    }
}

/// Robust estimate of a camera pose and focal length.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FocalPoseEstimate {
    model: FocalPoseModel,
    inliers: Vec<usize>,
    residuals: Vec<f64>,
    stats: EstimateStats,
}

impl FocalPoseEstimate {
    /// The pose, focal length, and principal point.
    #[must_use]
    pub const fn model(&self) -> &FocalPoseModel {
        &self.model
    }

    /// Indices of the final inliers.
    #[must_use]
    pub fn inliers(&self) -> &[usize] {
        &self.inliers
    }

    /// Number of final inliers.
    #[must_use]
    pub fn inlier_count(&self) -> usize {
        self.inliers.len()
    }

    /// Pixel reprojection error of every input correspondence against the final model.
    #[must_use]
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// Robust-search statistics, with the inlier ratio updated for the final model.
    #[must_use]
    pub const fn stats(&self) -> &EstimateStats {
        &self.stats
    }
}

/// Estimate a camera pose and its focal length from 3D-to-pixel correspondences.
///
/// `principal_point` is the known principal point in the pixel coordinates of `data`; the pixel
/// size is taken as square. `threshold` is a reprojection error in pixels, and inliers are
/// counted against the final model, which is refitted on its consensus with every inlier.
///
/// # Errors
///
/// Returns [`RobustError::InsufficientData`] with fewer than four correspondences, and another
/// [`RobustError`] when the search configuration is invalid or no hypothesis reaches a consensus.
pub fn estimate_absolute_pose_with_focal(
    data: &[PixelToPoint],
    principal_point: Point2<f64>,
    threshold: PixelThreshold,
    options: &RobustOptions,
) -> Result<FocalPoseEstimate, RobustError> {
    let search = GenRansacConfig::from_options(GenRansacConfig::builder(), options);
    let result = GenRansacEstimator::estimate_guided::<P4pfSolver, _>(
        data,
        threshold.into(),
        &search,
        &P4pfConfig { principal_point },
        RobustGuidance::default(),
        &FocalPoseRefiner,
    )?;
    let model = result.model;
    let residuals: Vec<f64> = data
        .iter()
        .map(|datum| model.reprojection_error(datum))
        .collect();
    let inliers: Vec<usize> = residuals
        .iter()
        .enumerate()
        .filter_map(|(index, &residual)| (residual <= threshold.value()).then_some(index))
        .collect();
    let mut stats = result.stats;
    stats.inlier_ratio = inliers.len() as f64 / data.len().max(1) as f64;
    Ok(FocalPoseEstimate {
        model,
        inliers,
        residuals,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Rotation3};

    const FOCAL: f64 = 1_400.0;
    const PRINCIPAL: [f64; 2] = [640.0, 480.0];

    fn scene(count: usize) -> (FocalPoseModel, Vec<PixelToPoint>) {
        let rotation = Rotation3::from_euler_angles(0.12, -0.2, 0.07).into_inner();
        let translation = Vector3::new(0.3, -0.2, 6.0);
        let truth = FocalPoseModel {
            rotation,
            translation,
            focal_px: FOCAL,
            principal_point: Point2::new(PRINCIPAL[0], PRINCIPAL[1]),
        };
        let data = (0..count)
            .map(|index| {
                let u = index as f64;
                let world = Point3::new(
                    1.6 * (0.9 * u + 0.2).sin(),
                    1.3 * (1.3 * u - 0.4).cos(),
                    0.7 * (0.5 * u).sin() + 0.1 * u,
                );
                let camera = rotation * world.coords + translation;
                let pixel = Point2::new(
                    FOCAL * camera.x / camera.z + PRINCIPAL[0],
                    FOCAL * camera.y / camera.z + PRINCIPAL[1],
                );
                PixelToPoint::new(world, pixel)
            })
            .collect();
        (truth, data)
    }

    fn rotation_error(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
        let delta = a * b.transpose();
        ((delta.trace() - 1.0) * 0.5).clamp(-1.0, 1.0).acos()
    }

    #[test]
    fn a_minimal_sample_returns_the_true_pose_and_focal_among_its_candidates() {
        let (truth, data) = scene(4);
        let config = P4pfConfig {
            principal_point: truth.principal_point,
        };

        let models = P4pfSolver::solve_sample(&data, &config).unwrap();

        assert!(models.iter().any(|model| {
            (model.focal_px - FOCAL).abs() < 1e-6 * FOCAL
                && rotation_error(&model.rotation, &truth.rotation) < 1e-8
                && (model.translation - truth.translation).norm() < 1e-7
        }));
    }

    #[test]
    fn a_minimal_solve_rejects_a_sample_of_the_wrong_size() {
        let (truth, data) = scene(5);
        let config = P4pfConfig {
            principal_point: truth.principal_point,
        };

        let error = P4pfSolver::solve_sample(&data, &config).unwrap_err();

        assert_eq!(
            error,
            SolverError::InvalidSampleSize {
                expected: 4,
                provided: 5
            }
        );
    }

    #[test]
    fn a_sample_with_an_index_outside_the_data_is_an_error_not_a_panic() {
        let config = P4pfConfig {
            principal_point: Point2::new(0.0, 0.0),
        };
        let error = P4pfSolver::for_each_solution_from_indices(&[0, 1, 2, 3], &[], &config, |_| {})
            .unwrap_err();
        assert_eq!(
            error,
            SolverError::InvalidDataIndex {
                index: 0,
                data_len: 0
            }
        );
    }

    #[test]
    fn residual_is_the_pixel_reprojection_error_and_infinite_behind_the_camera() {
        let (truth, data) = scene(1);
        assert!(P4pfSolver::residual(&truth, &data[0]) < 1e-9);
        let shifted = PixelToPoint::new(
            data[0].world_point,
            Point2::new(data[0].pixel.x + 3.0, data[0].pixel.y + 4.0),
        );
        assert!((P4pfSolver::residual(&truth, &shifted) - 5.0).abs() < 1e-9);
        let behind = PixelToPoint::new(Point3::new(0.0, 0.0, -100.0), data[0].pixel);
        assert_eq!(P4pfSolver::residual(&truth, &behind), f64::INFINITY);
    }

    #[test]
    fn the_robust_facade_recovers_pose_and_focal_through_outliers() {
        let (truth, mut data) = scene(40);
        let inliers = data.len();
        for index in 0..10 {
            let target = (index + 7) % inliers;
            data.push(PixelToPoint::new(
                data[index].world_point,
                Point2::new(
                    data[target].pixel.x + 55.0,
                    data[target].pixel.y - 31.0 + index as f64,
                ),
            ));
        }
        let options = RobustOptions::default().with_seed(11);
        let threshold = PixelThreshold::try_new(1.5).unwrap();

        let estimate =
            estimate_absolute_pose_with_focal(&data, truth.principal_point, threshold, &options)
                .unwrap();

        let model = estimate.model();
        assert!(
            (model.focal_px() - FOCAL).abs() < 1e-3 * FOCAL,
            "{}",
            model.focal_px()
        );
        assert!(rotation_error(model.rotation(), &truth.rotation) < 1e-4);
        assert!(estimate.inlier_count() >= inliers);
        assert_eq!(estimate.residuals().len(), data.len());
        assert!(model.pose().is_ok());
    }

    #[test]
    fn noisy_observations_leave_the_focal_within_a_few_percent() {
        let (truth, mut data) = scene(60);
        for (index, datum) in data.iter_mut().enumerate() {
            let u = index as f64;
            datum.pixel.x += 0.6 * (2.7 * u).sin();
            datum.pixel.y += 0.6 * (1.9 * u).cos();
        }
        let estimate = estimate_absolute_pose_with_focal(
            &data,
            truth.principal_point,
            PixelThreshold::try_new(2.0).unwrap(),
            &RobustOptions::default().with_seed(5),
        )
        .unwrap();

        let relative = (estimate.model().focal_px() - FOCAL).abs() / FOCAL;
        assert!(relative < 0.03, "relative focal error {relative}");
        assert!(estimate.inlier_count() >= 55);
    }

    #[test]
    fn the_estimate_is_deterministic_for_a_seed() {
        let (truth, data) = scene(30);
        let run = || {
            estimate_absolute_pose_with_focal(
                &data,
                truth.principal_point,
                PixelThreshold::try_new(1.0).unwrap(),
                &RobustOptions::default().with_seed(3),
            )
            .unwrap()
        };
        let (first, second) = (run(), run());
        assert_eq!(first.model(), second.model());
        assert_eq!(first.inliers(), second.inliers());
    }
}
