// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Robust fundamental matrix estimation in pixel space.
//!
//! Its teaching docs are on [`estimate_fundamental`], where docs.rs shows them.

use std::sync::Arc;

use crate::correspondence::PixelCorrespondence;
use crate::fundamental::{EightPointSolver, FundamentalMatrix, SevenPointSolver, denormalize};
use crate::normalization::NormalizedCorrespondences;
use crate::residuals::sampson_error_fundamental_squared;
use crate::{SolverError, SolverResult};

use crate::robust::error::RobustError;
use crate::robust::result::RobustEstimate;
use crate::robust::threshold::PixelThreshold;
use crate::solver_traits::{MultiSolutionSolver, SingleSolutionSolver, Solver};

use crate::robust::genransac::config::{GenRansacConfig, RobustOptions};
use crate::robust::genransac::estimator::{GenRansacEstimator, RobustGuidance};
use crate::robust::genransac::refiner::SolverRefiner;

#[derive(Debug, Clone)]
struct PreparedFundamentalConfig(Arc<NormalizedCorrespondences>);

impl PreparedFundamentalConfig {
    fn new(data: &[PixelCorrespondence]) -> Self {
        Self(Arc::new(NormalizedCorrespondences::from_pixels(data)))
    }
}

#[derive(Debug, Clone)]
struct PixelSevenPointSolver;

impl Solver for PixelSevenPointSolver {
    type Datum = PixelCorrespondence;
    type Model = FundamentalMatrix;
    type Config = PreparedFundamentalConfig;
    type Unit = openmvg_core::Pixels;

    fn min_sample_size() -> usize {
        7
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        SevenPointSolver::residual(model, datum)
    }

    fn squared_residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_fundamental_squared(model.matrix(), datum)
    }

    fn is_valid(model: &Self::Model) -> bool {
        SevenPointSolver::is_valid(model)
    }
}

impl MultiSolutionSolver for PixelSevenPointSolver {
    fn solve(data: &[Self::Datum], _config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        SevenPointSolver::solve(&NormalizedCorrespondences::from_pixels(data))
    }

    fn for_each_solution_from_indices<F>(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
        mut emit: F,
    ) -> SolverResult<usize>
    where
        F: FnMut(Self::Model),
    {
        if indices.len() != Self::min_sample_size() {
            return Err(if indices.len() < Self::min_sample_size() {
                SolverError::InsufficientData {
                    required: Self::min_sample_size(),
                    provided: indices.len(),
                }
            } else {
                SolverError::InvalidSampleSize {
                    expected: Self::min_sample_size(),
                    provided: indices.len(),
                }
            });
        }
        let normalized = config.0.as_slice();
        if normalized.len() != data.len() {
            return Err(SolverError::DegenerateConfiguration {
                reason: format!(
                    "prepared fundamental-matrix data has {} entries, the input has {}",
                    normalized.len(),
                    data.len()
                ),
            });
        }
        let first = *normalized
            .get(indices[0])
            .ok_or(SolverError::InvalidDataIndex {
                index: indices[0],
                data_len: data.len(),
            })?;
        let mut sample = [first; 7];
        for (sample_index, &data_index) in indices.iter().enumerate() {
            sample[sample_index] =
                *normalized
                    .get(data_index)
                    .ok_or(SolverError::InvalidDataIndex {
                        index: data_index,
                        data_len: data.len(),
                    })?;
        }
        let models = <SevenPointSolver as MultiSolutionSolver>::solve(&sample, &())?;
        let count = models.len();
        for model in &models {
            emit(denormalize(model, config.0.normalization()));
        }
        Ok(count)
    }
}

#[derive(Debug, Clone)]
struct PixelEightPointSolver;

impl Solver for PixelEightPointSolver {
    type Datum = PixelCorrespondence;
    type Model = FundamentalMatrix;
    type Config = PreparedFundamentalConfig;
    type Unit = openmvg_core::Pixels;

    fn min_sample_size() -> usize {
        8
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        EightPointSolver::residual(model, datum)
    }

    fn squared_residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_fundamental_squared(model.matrix(), datum)
    }

    fn is_valid(model: &Self::Model) -> bool {
        EightPointSolver::is_valid(model)
    }
}

impl SingleSolutionSolver for PixelEightPointSolver {
    fn solve_single(
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        let normalized = NormalizedCorrespondences::from_pixels(data);
        let model =
            <EightPointSolver as SingleSolutionSolver>::solve_single(normalized.as_slice(), &())?;
        Ok(model.map(|model| denormalize(&model, normalized.normalization())))
    }

    fn solve_single_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        if config.0.len() != data.len() {
            return Err(SolverError::DegenerateConfiguration {
                reason: format!(
                    "prepared fundamental-matrix data has {} entries, the input has {}",
                    config.0.len(),
                    data.len()
                ),
            });
        }
        let model = <EightPointSolver as SingleSolutionSolver>::solve_single_from_indices(
            indices,
            config.0.as_slice(),
            &(),
        )?;
        Ok(model.map(|model| denormalize(&model, config.0.normalization())))
    }
}

/// Estimate a fundamental matrix from pixel matches that include wrong ones.
///
/// Each match is scored by its Sampson error, a first-order approximation of its distance to
/// the epipolar geometry, in pixels. Set the threshold to a few times the pixel noise, for
/// example 2 px for 0.5 px of noise. Use it when the cameras are not calibrated; with a known
/// calibration, prefer [`estimate_essential`](crate::essential::estimate_essential) on
/// bearings.
///
/// # How it works
///
/// Hartley normalization is prepared once. Minimal and non-minimal solvers see normalized
/// correspondences, every emitted model is denormalized, and robust scoring uses the caller's
/// pixel threshold exactly.
///
/// ```text
/// pixels -> Hartley normalize -> F7 hypotheses -> denormalize + score
///        -> F8 consensus refit -> pixel-space result
/// ```
pub fn estimate_fundamental(
    data: &[PixelCorrespondence],
    threshold: PixelThreshold,
    options: &RobustOptions,
) -> Result<RobustEstimate<FundamentalMatrix>, RobustError> {
    estimate_fundamental_guided(data, threshold, options, RobustGuidance::default())
}

/// Estimate a fundamental matrix with quality ordering, inlier priors, or both.
pub fn estimate_fundamental_guided(
    data: &[PixelCorrespondence],
    threshold: PixelThreshold,
    options: &RobustOptions,
    guidance: RobustGuidance<'_>,
) -> Result<RobustEstimate<FundamentalMatrix>, RobustError> {
    let solver_config = PreparedFundamentalConfig::new(data);
    let refiner = SolverRefiner::<PixelEightPointSolver>::new(solver_config.clone());
    let search = GenRansacConfig::from_options(GenRansacConfig::fundamental(), options);
    GenRansacEstimator::estimate_guided::<PixelSevenPointSolver, _>(
        data,
        threshold.into(),
        &search,
        &solver_config,
        guidance,
        &refiner,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Rotation3, Vector3};

    fn skew(vector: Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(
            0.0, -vector.z, vector.y, vector.z, 0.0, -vector.x, -vector.y, vector.x, 0.0,
        )
    }

    fn synthetic_problem() -> (Vec<PixelCorrespondence>, usize, Matrix3<f64>) {
        let focal = 900.0;
        let principal_x = 620.0;
        let principal_y = 470.0;
        let intrinsic = Matrix3::new(
            focal,
            0.0,
            principal_x,
            0.0,
            focal,
            principal_y,
            0.0,
            0.0,
            1.0,
        );
        let inverse = intrinsic.try_inverse().unwrap();
        let rotation = Rotation3::from_euler_angles(0.05, -0.12, 0.07);
        let translation = Vector3::new(0.48, -0.14, 0.08);
        let mut data = (0..36)
            .map(|index| {
                let u = index as f64;
                let point = Vector3::new(
                    1.8 * (0.67 * u).sin(),
                    1.5 * (1.09 * u + 0.2).cos(),
                    4.0 + 0.1 * u + 0.6 * (0.39 * u).sin().abs(),
                );
                let second = rotation * point + translation;
                let first_pixel = intrinsic * point;
                let second_pixel = intrinsic * second;
                let noise = 0.08 * (1.37 * u).sin();
                PixelCorrespondence::from_coords(
                    first_pixel.x / first_pixel.z,
                    first_pixel.y / first_pixel.z,
                    second_pixel.x / second_pixel.z + noise,
                    second_pixel.y / second_pixel.z - 0.6 * noise,
                )
            })
            .collect::<Vec<_>>();
        let inlier_count = data.len();
        for index in 0..8 {
            data.push(PixelCorrespondence::new(
                data[index].point1,
                data[(index + 17) % inlier_count].point2,
            ));
        }
        let expected = inverse.transpose() * skew(translation) * rotation.matrix() * inverse;
        (data, inlier_count, expected)
    }

    fn projective_error(left: Matrix3<f64>, right: Matrix3<f64>) -> f64 {
        let left = left / left.norm();
        let right = right / right.norm();
        (left - right).norm().min((left + right).norm())
    }

    /// Robust Fundamental estimation: the typed pixel facade recovers the dominant Fundamental
    /// model with outliers; minimal fitting, pixel scoring, and consensus refitting remain one
    /// policy.
    #[test]
    fn pixel_facade_recovers_the_dominant_fundamental_matrix() {
        let (data, inlier_count, expected) = synthetic_problem();
        let options = RobustOptions::default().with_seed(31).with_max_draws(512);
        let result = estimate_fundamental(&data, PixelThreshold::try_new(1.0).unwrap(), &options)
            .expect("pixel-space GenRANSAC should recover the dominant epipolar geometry");

        assert!(projective_error(*result.model.matrix(), expected) < 0.03);
        assert!(
            result
                .inliers
                .iter()
                .filter(|&&index| index < inlier_count)
                .count()
                >= inlier_count - 2
        );
    }
}
