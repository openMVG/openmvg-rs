// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free observation and point-track quality evaluation.

use crate::numerics::refine::{Loss, NoiseModel};

use super::super::RelativePose;
use super::super::point_block::{POINT_BLOCK_RCOND_THRESHOLD, reciprocal_condition_number};
use super::model::{ObservationModel, transform};
use super::self_calibration::{SelfCalibratingBundleAdjustmentProblem, radial_model};
use super::{
    CalibratedBundleAdjustmentProblem, MultiViewBundleAdjustmentError,
    MultiViewBundleAdjustmentEvaluation, MultiViewObservationResidual,
    MultiViewTrackClassification, MultiViewTrackDiagnostics, MultiViewTrackLifecycleEvaluation,
    MultiViewTrackLifecycleSummary, ScaleGauge, validation,
};

impl CalibratedBundleAdjustmentProblem<'_> {
    /// Evaluate residuals and geometric track health without allocating.
    ///
    /// Outputs follow the point-major observation order of
    /// [`MultiViewMeasurements`](super::MultiViewMeasurements). Both caller
    /// slices must have exact lengths returned by [`observation_count`](Self::observation_count)
    /// and [`point_count`](Self::point_count). The function never changes poses,
    /// points, measurements, or problem topology. Pass the solve report's loss
    /// to reproduce its final point-block conditioning. Output contents are
    /// unspecified when evaluation returns an error.
    pub fn evaluate(
        self,
        poses: &[RelativePose],
        points: &[[f64; 3]],
        loss: Loss,
        observation_residuals: &mut [MultiViewObservationResidual],
        track_diagnostics: &mut [MultiViewTrackDiagnostics],
    ) -> Result<MultiViewBundleAdjustmentEvaluation, MultiViewBundleAdjustmentError> {
        evaluate_problem(
            EvaluationProblem {
                model: ObservationModel::from(self.measurements()),
                gauge: self.gauge(),
                noise: self.noise(),
            },
            poses,
            points,
            loss,
            observation_residuals,
            track_diagnostics,
        )
    }

    /// Evaluate and classify track lifecycle without allocating or mutating topology.
    ///
    /// An observation is retained when its standardized residual norm is finite
    /// and at most `maximum_standardized_residual_norm`. Track diagnostics and
    /// geometric conditioning are computed from that retained subset, not from
    /// observations that the same call rejects. A supported track is recommended
    /// for retriangulation when any observation was rejected. Weak geometry is
    /// reported separately because triangulating the same retained views cannot
    /// manufacture parallax. The caller decides whether and how to apply these
    /// recommendations. Output contents are unspecified on error.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_track_lifecycle(
        self,
        poses: &[RelativePose],
        points: &[[f64; 3]],
        loss: Loss,
        maximum_standardized_residual_norm: f64,
        observation_residuals: &mut [MultiViewObservationResidual],
        track_diagnostics: &mut [MultiViewTrackDiagnostics],
        observation_inliers: &mut [bool],
        track_classifications: &mut [MultiViewTrackClassification],
    ) -> Result<MultiViewTrackLifecycleEvaluation, MultiViewBundleAdjustmentError> {
        evaluate_lifecycle_problem(
            EvaluationProblem {
                model: ObservationModel::from(self.measurements()),
                gauge: self.gauge(),
                noise: self.noise(),
            },
            poses,
            points,
            loss,
            maximum_standardized_residual_norm,
            observation_residuals,
            track_diagnostics,
            observation_inliers,
            track_classifications,
        )
    }
}

impl SelfCalibratingBundleAdjustmentProblem<'_> {
    /// Evaluate grouped radial residuals and track health without allocating.
    ///
    /// Intrinsic groups are treated as fixed at the values borrowed by this
    /// problem view. Pass the solve report's loss to reproduce final
    /// observation weights and point-block conditioning.
    pub fn evaluate(
        self,
        poses: &[RelativePose],
        points: &[[f64; 3]],
        loss: Loss,
        observation_residuals: &mut [MultiViewObservationResidual],
        track_diagnostics: &mut [MultiViewTrackDiagnostics],
    ) -> Result<MultiViewBundleAdjustmentEvaluation, MultiViewBundleAdjustmentError> {
        evaluate_problem(
            EvaluationProblem {
                model: radial_model(
                    self.measurements(),
                    self.intrinsic_groups(),
                    self.camera_to_group(),
                ),
                gauge: self.gauge(),
                noise: self.noise(),
            },
            poses,
            points,
            loss,
            observation_residuals,
            track_diagnostics,
        )
    }

    /// Evaluate grouped radial observations and propose track lifecycle edits.
    ///
    /// Conditioning is recomputed from the retained radial observations.
    /// Intrinsics are never changed by evaluation, and all topology edits
    /// remain explicit caller actions.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_track_lifecycle(
        self,
        poses: &[RelativePose],
        points: &[[f64; 3]],
        loss: Loss,
        maximum_standardized_residual_norm: f64,
        observation_residuals: &mut [MultiViewObservationResidual],
        track_diagnostics: &mut [MultiViewTrackDiagnostics],
        observation_inliers: &mut [bool],
        track_classifications: &mut [MultiViewTrackClassification],
    ) -> Result<MultiViewTrackLifecycleEvaluation, MultiViewBundleAdjustmentError> {
        evaluate_lifecycle_problem(
            EvaluationProblem {
                model: radial_model(
                    self.measurements(),
                    self.intrinsic_groups(),
                    self.camera_to_group(),
                ),
                gauge: self.gauge(),
                noise: self.noise(),
            },
            poses,
            points,
            loss,
            maximum_standardized_residual_norm,
            observation_residuals,
            track_diagnostics,
            observation_inliers,
            track_classifications,
        )
    }
}

#[derive(Clone, Copy)]
struct EvaluationProblem<'a> {
    model: ObservationModel<'a>,
    gauge: ScaleGauge,
    noise: NoiseModel<'a>,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_lifecycle_problem(
    problem: EvaluationProblem<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    loss: Loss,
    maximum_standardized_residual_norm: f64,
    observation_residuals: &mut [MultiViewObservationResidual],
    track_diagnostics: &mut [MultiViewTrackDiagnostics],
    observation_inliers: &mut [bool],
    track_classifications: &mut [MultiViewTrackClassification],
) -> Result<MultiViewTrackLifecycleEvaluation, MultiViewBundleAdjustmentError> {
    let observation_count = problem.model.observation_count();
    let point_count = problem.model.point_count();
    validate_evaluation_storage(
        observation_count,
        point_count,
        observation_residuals,
        track_diagnostics,
    )?;
    if observation_inliers.len() != observation_count {
        return Err(
            MultiViewBundleAdjustmentError::ObservationInlierMaskCountMismatch {
                expected: observation_count,
                actual: observation_inliers.len(),
            },
        );
    }
    if track_classifications.len() != point_count {
        return Err(
            MultiViewBundleAdjustmentError::TrackClassificationCountMismatch {
                expected: point_count,
                actual: track_classifications.len(),
            },
        );
    }
    if !(maximum_standardized_residual_norm.is_finite()
        && maximum_standardized_residual_norm >= 0.0)
    {
        return Err(crate::numerics::refine::RefinementError::InvalidThreshold.into());
    }
    validate_evaluation_problem(problem, poses, points, loss)?;
    evaluate_track_lifecycle_model(
        problem.model,
        poses,
        points,
        loss,
        problem.noise,
        maximum_standardized_residual_norm,
        observation_residuals,
        track_diagnostics,
        observation_inliers,
        track_classifications,
    )
}

fn evaluate_problem(
    problem: EvaluationProblem<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    loss: Loss,
    observation_residuals: &mut [MultiViewObservationResidual],
    track_diagnostics: &mut [MultiViewTrackDiagnostics],
) -> Result<MultiViewBundleAdjustmentEvaluation, MultiViewBundleAdjustmentError> {
    validate_evaluation_storage(
        problem.model.observation_count(),
        problem.model.point_count(),
        observation_residuals,
        track_diagnostics,
    )?;
    validate_evaluation_problem(problem, poses, points, loss)?;
    evaluate_model(
        problem.model,
        poses,
        points,
        loss,
        problem.noise,
        observation_residuals,
        track_diagnostics,
    )
}

fn validate_evaluation_storage(
    observation_count: usize,
    point_count: usize,
    observation_residuals: &[MultiViewObservationResidual],
    track_diagnostics: &[MultiViewTrackDiagnostics],
) -> Result<(), MultiViewBundleAdjustmentError> {
    if observation_residuals.len() != observation_count {
        return Err(
            MultiViewBundleAdjustmentError::ObservationResidualCountMismatch {
                expected: observation_count,
                actual: observation_residuals.len(),
            },
        );
    }
    if track_diagnostics.len() != point_count {
        return Err(
            MultiViewBundleAdjustmentError::TrackDiagnosticCountMismatch {
                expected: point_count,
                actual: track_diagnostics.len(),
            },
        );
    }
    Ok(())
}

fn validate_evaluation_problem(
    problem: EvaluationProblem<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    loss: Loss,
) -> Result<(), MultiViewBundleAdjustmentError> {
    validation::problem_with_capacity(
        poses,
        points,
        problem.model,
        problem.gauge,
        poses.len(),
        points.len(),
    )?;
    problem
        .noise
        .try_validate(problem.model.observation_count())?;
    loss.try_validate()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_track_lifecycle_model(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    loss: Loss,
    noise: NoiseModel<'_>,
    maximum_standardized_residual_norm: f64,
    observation_residuals: &mut [MultiViewObservationResidual],
    track_diagnostics: &mut [MultiViewTrackDiagnostics],
    observation_inliers: &mut [bool],
    track_classifications: &mut [MultiViewTrackClassification],
) -> Result<MultiViewTrackLifecycleEvaluation, MultiViewBundleAdjustmentError> {
    let mut evaluation = MultiViewBundleAdjustmentEvaluation::default();
    let mut lifecycle = MultiViewTrackLifecycleSummary::default();
    let mut squared_residual = 0.0;

    for (point_index, point) in points.iter().enumerate() {
        let range = model.offsets()[point_index]..model.offsets()[point_index + 1];
        let mut complete_point_hessian = [[0.0; 3]; 3];
        let mut retained_point_hessian = [[0.0; 3]; 3];
        let mut retained_squared_residual = 0.0;
        let mut retained_maximum_standardized_norm: f64 = 0.0;
        let mut retained = 0;
        for observation in range.clone() {
            let camera = model.camera(observation);
            let camera_point = transform(&poses[camera], point);
            let projection = model.project(observation, &camera_point).ok_or(
                MultiViewBundleAdjustmentError::PointNotProjectable {
                    point: point_index,
                    camera,
                },
            )?;
            let norm = squared_norm(&projection.residual).sqrt();
            let inverse_sigma = noise.inverse_standard_deviation(observation);
            let standardized_norm = norm * inverse_sigma;
            observation_residuals[observation] = MultiViewObservationResidual {
                residual: projection.residual,
                standardized_norm,
            };
            evaluation.robust_cost += loss.cost(standardized_norm);
            squared_residual += norm * norm;
            evaluation.maximum_standardized_residual_norm = evaluation
                .maximum_standardized_residual_norm
                .max(standardized_norm);
            let sqrt_weight = inverse_sigma * loss.sqrt_weight(standardized_norm);
            let point_jacobian = multiply_2x3_3x3(&projection.jacobian, &poses[camera].rotation);
            accumulate_hessian(&mut complete_point_hessian, &point_jacobian, sqrt_weight);

            let is_inlier = standardized_norm.is_finite()
                && standardized_norm <= maximum_standardized_residual_norm;
            observation_inliers[observation] = is_inlier;
            if is_inlier {
                retained += 1;
                retained_squared_residual += norm * norm;
                retained_maximum_standardized_norm =
                    retained_maximum_standardized_norm.max(standardized_norm);
                accumulate_hessian(&mut retained_point_hessian, &point_jacobian, sqrt_weight);
            }
        }
        let rejected = range.len() - retained;
        lifecycle.retained_observation_count += retained;
        lifecycle.rejected_observation_count += rejected;
        let complete_point_rcond = reciprocal_condition_number(&complete_point_hessian);
        let complete_weak = complete_point_rcond <= POINT_BLOCK_RCOND_THRESHOLD;
        evaluation.weak_point_count += usize::from(complete_weak);
        evaluation.minimum_point_hessian_reciprocal_condition_number = if point_index == 0 {
            complete_point_rcond
        } else {
            evaluation
                .minimum_point_hessian_reciprocal_condition_number
                .min(complete_point_rcond)
        };
        let point_rcond = if retained < 2 {
            0.0
        } else {
            reciprocal_condition_number(&retained_point_hessian)
        };
        let weak = point_rcond <= POINT_BLOCK_RCOND_THRESHOLD;
        track_diagnostics[point_index] = MultiViewTrackDiagnostics {
            observation_count: retained,
            rmse: if retained == 0 {
                0.0
            } else {
                (retained_squared_residual / (2 * retained) as f64).sqrt()
            },
            maximum_standardized_residual_norm: retained_maximum_standardized_norm,
            point_hessian_reciprocal_condition_number: point_rcond,
            weak,
        };
        let classification = if retained < 2 {
            lifecycle.insufficient_support_track_count += 1;
            MultiViewTrackClassification::InsufficientSupport
        } else if weak {
            lifecycle.weak_geometry_track_count += 1;
            MultiViewTrackClassification::WeakGeometry
        } else if rejected > 0 {
            lifecycle.retriangulation_track_count += 1;
            MultiViewTrackClassification::Retriangulate
        } else {
            lifecycle.stable_track_count += 1;
            MultiViewTrackClassification::Stable
        };
        track_classifications[point_index] = classification;
    }
    evaluation.rmse = (squared_residual / (2 * model.observation_count()) as f64).sqrt();
    Ok(MultiViewTrackLifecycleEvaluation {
        evaluation,
        lifecycle,
    })
}

fn evaluate_model(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    loss: Loss,
    noise: crate::numerics::refine::NoiseModel<'_>,
    observation_residuals: &mut [MultiViewObservationResidual],
    track_diagnostics: &mut [MultiViewTrackDiagnostics],
) -> Result<MultiViewBundleAdjustmentEvaluation, MultiViewBundleAdjustmentError> {
    let mut robust_cost = 0.0;
    let mut squared_residual = 0.0;
    let mut maximum_standardized_residual_norm: f64 = 0.0;
    let mut weak_point_count = 0;
    let mut minimum_point_rcond: f64 = 1.0;

    for (point_index, point) in points.iter().enumerate() {
        let range = model.offsets()[point_index]..model.offsets()[point_index + 1];
        let mut point_hessian = [[0.0; 3]; 3];
        let mut track_squared_residual = 0.0;
        let mut track_maximum_standardized_norm: f64 = 0.0;

        for observation in range.clone() {
            let camera = model.camera(observation);
            let camera_point = transform(&poses[camera], point);
            let projection = model.project(observation, &camera_point).ok_or(
                MultiViewBundleAdjustmentError::PointNotProjectable {
                    point: point_index,
                    camera,
                },
            )?;
            let norm = squared_norm(&projection.residual).sqrt();
            let inverse_sigma = noise.inverse_standard_deviation(observation);
            let standardized_norm = norm * inverse_sigma;
            observation_residuals[observation] = MultiViewObservationResidual {
                residual: projection.residual,
                standardized_norm,
            };

            robust_cost += loss.cost(standardized_norm);
            let residual_squared = norm * norm;
            squared_residual += residual_squared;
            track_squared_residual += residual_squared;
            maximum_standardized_residual_norm =
                maximum_standardized_residual_norm.max(standardized_norm);
            track_maximum_standardized_norm =
                track_maximum_standardized_norm.max(standardized_norm);

            let sqrt_weight = inverse_sigma * loss.sqrt_weight(standardized_norm);
            let point_jacobian = multiply_2x3_3x3(&projection.jacobian, &poses[camera].rotation);
            accumulate_hessian(&mut point_hessian, &point_jacobian, sqrt_weight);
        }

        let point_rcond = reciprocal_condition_number(&point_hessian);
        let weak = point_rcond <= POINT_BLOCK_RCOND_THRESHOLD;
        weak_point_count += usize::from(weak);
        minimum_point_rcond = minimum_point_rcond.min(point_rcond);
        track_diagnostics[point_index] = MultiViewTrackDiagnostics {
            observation_count: range.len(),
            rmse: (track_squared_residual / (2 * range.len()) as f64).sqrt(),
            maximum_standardized_residual_norm: track_maximum_standardized_norm,
            point_hessian_reciprocal_condition_number: point_rcond,
            weak,
        };
    }

    Ok(MultiViewBundleAdjustmentEvaluation {
        robust_cost,
        rmse: (squared_residual / (2 * model.observation_count()) as f64).sqrt(),
        maximum_standardized_residual_norm,
        weak_point_count,
        minimum_point_hessian_reciprocal_condition_number: minimum_point_rcond,
    })
}

#[inline(always)]
fn squared_norm(vector: &[f64; 2]) -> f64 {
    vector[0] * vector[0] + vector[1] * vector[1]
}

#[inline]
fn multiply_2x3_3x3(left: &[[f64; 3]; 2], right: &[[f64; 3]; 3]) -> [[f64; 3]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            left[row][0] * right[0][column]
                + left[row][1] * right[1][column]
                + left[row][2] * right[2][column]
        })
    })
}

#[inline]
fn accumulate_hessian(hessian: &mut [[f64; 3]; 3], jacobian: &[[f64; 3]; 2], sqrt_weight: f64) {
    for jacobian_row in jacobian {
        let weighted = jacobian_row.map(|value| sqrt_weight * value);
        for row in 0..3 {
            for column in 0..3 {
                hessian[row][column] += weighted[row] * weighted[column];
            }
        }
    }
}
