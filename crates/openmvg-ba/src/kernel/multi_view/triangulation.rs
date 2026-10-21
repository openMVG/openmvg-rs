// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free calibrated multi-view track triangulation.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Track triangulation (triangulate_tracks, triangulate_selected_tracks)│
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: fixed poses, one track per point (optional inlier mask, optional
//!        track selection), the problem's camera model, noise and loss
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Multi-ray intersection                                       │
//! └──────────────────────────────────────────────────────────────────────┘
//! Each retained observation gives a world ray (center c, unit direction
//! d). Solve Σ (I − d dᵀ) X = Σ (I − d dᵀ) c, a 3×3 system; fewer than 2
//! rays or a reciprocal condition ≤ 192 ε leaves the track untriangulated.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Nonlinear refinement of X ∈ ℝ³                               │
//! └──────────────────────────────────────────────────────────────────────┘
//! The BA residual of every retained observation (pixels or tangent-plane
//! radians, whitened by 1/σ, robust loss) minimized by the shared LM loop;
//! a point behind a camera makes the cost undefined.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Certify the point                                            │
//! └──────────────────────────────────────────────────────────────────────┘
//! Status from LM progress and the reciprocal condition of the final 3×3
//! point Hessian, with the scaled gradient norm and the costs.
//!   │
//!   ▼
//! Output: one MultiViewTriangulationReport per point, written into caller
//!         storage; nothing is allocated and the map is not changed
//! ```

use crate::kernel::point_block::{
    POINT_BLOCK_RCOND_THRESHOLD, PointBlockFactor, reciprocal_condition_number,
};
use crate::numerics::refine::covariance::InformationDiagnostics;
use crate::numerics::refine::nonlinear::{LmProblem, LmSolution, minimize};
use crate::numerics::refine::{Loss, NoiseModel, NormalEquations, Options, RefinementStatus};

use super::super::RelativePose;
use super::model::{ObservationModel, transform};
use super::self_calibration::{SelfCalibratingBundleAdjustmentProblem, radial_model};
use super::{
    CalibratedBundleAdjustmentProblem, MultiViewBundleAdjustmentError,
    MultiViewTriangulationReport, validation,
};

impl CalibratedBundleAdjustmentProblem<'_> {
    /// Triangulate every point track into caller-provided report storage.
    ///
    /// `observation_inliers` is either empty (use every observation) or has
    /// exactly [`observation_count`](Self::observation_count) entries. Each
    /// track starts from a multi-ray least-squares intersection and is refined
    /// with the same projection, whitening, robust loss, and LM policy as BA.
    /// The method allocates nothing and does not mutate the map; callers decide
    /// which successful candidates to insert or replace.
    pub fn triangulate_tracks(
        self,
        poses: &[RelativePose],
        observation_inliers: &[bool],
        reports: &mut [MultiViewTriangulationReport],
        options: Options,
    ) -> Result<(), MultiViewBundleAdjustmentError> {
        self.triangulate_tracks_impl(poses, observation_inliers, &[], reports, options)
    }

    /// Triangulate only selected point tracks into point-aligned report storage.
    ///
    /// `track_selection` and `reports` must each have one entry per point.
    /// Reports for unselected tracks are reset to their default value. This is
    /// useful after [`evaluate_track_lifecycle`](Self::evaluate_track_lifecycle),
    /// where only tracks classified for replacement need expensive ray
    /// intersection and nonlinear refinement.
    pub fn triangulate_selected_tracks(
        self,
        poses: &[RelativePose],
        observation_inliers: &[bool],
        track_selection: &[bool],
        reports: &mut [MultiViewTriangulationReport],
        options: Options,
    ) -> Result<(), MultiViewBundleAdjustmentError> {
        let point_count = self.point_count();
        if track_selection.len() != point_count {
            return Err(
                MultiViewBundleAdjustmentError::TrackSelectionCountMismatch {
                    expected: point_count,
                    actual: track_selection.len(),
                },
            );
        }
        self.triangulate_tracks_impl(
            poses,
            observation_inliers,
            track_selection,
            reports,
            options,
        )
    }

    fn triangulate_tracks_impl(
        self,
        poses: &[RelativePose],
        observation_inliers: &[bool],
        track_selection: &[bool],
        reports: &mut [MultiViewTriangulationReport],
        options: Options,
    ) -> Result<(), MultiViewBundleAdjustmentError> {
        let model = ObservationModel::from(self.measurements());
        triangulate_tracks_impl(
            model,
            poses,
            self.point_count(),
            self.observation_count(),
            self.noise(),
            observation_inliers,
            track_selection,
            reports,
            options,
        )
    }
}

impl SelfCalibratingBundleAdjustmentProblem<'_> {
    /// Triangulate every grouped radial track into caller-provided reports.
    ///
    /// Radial measurements are inverted to camera rays with the intrinsic
    /// groups borrowed by this problem. Candidate points are never committed.
    pub fn triangulate_tracks(
        self,
        poses: &[RelativePose],
        observation_inliers: &[bool],
        reports: &mut [MultiViewTriangulationReport],
        options: Options,
    ) -> Result<(), MultiViewBundleAdjustmentError> {
        self.triangulate_tracks_impl(poses, observation_inliers, &[], reports, options)
    }

    /// Triangulate only selected grouped radial tracks.
    pub fn triangulate_selected_tracks(
        self,
        poses: &[RelativePose],
        observation_inliers: &[bool],
        track_selection: &[bool],
        reports: &mut [MultiViewTriangulationReport],
        options: Options,
    ) -> Result<(), MultiViewBundleAdjustmentError> {
        let point_count = self.point_count();
        if track_selection.len() != point_count {
            return Err(
                MultiViewBundleAdjustmentError::TrackSelectionCountMismatch {
                    expected: point_count,
                    actual: track_selection.len(),
                },
            );
        }
        self.triangulate_tracks_impl(
            poses,
            observation_inliers,
            track_selection,
            reports,
            options,
        )
    }

    fn triangulate_tracks_impl(
        self,
        poses: &[RelativePose],
        observation_inliers: &[bool],
        track_selection: &[bool],
        reports: &mut [MultiViewTriangulationReport],
        options: Options,
    ) -> Result<(), MultiViewBundleAdjustmentError> {
        let model = radial_model(
            self.measurements(),
            self.intrinsic_groups(),
            self.camera_to_group(),
        );
        triangulate_tracks_impl(
            model,
            poses,
            self.point_count(),
            self.observation_count(),
            self.noise(),
            observation_inliers,
            track_selection,
            reports,
            options,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn triangulate_tracks_impl(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point_count: usize,
    observation_count: usize,
    noise: NoiseModel<'_>,
    observation_inliers: &[bool],
    track_selection: &[bool],
    reports: &mut [MultiViewTriangulationReport],
    options: Options,
) -> Result<(), MultiViewBundleAdjustmentError> {
    if !observation_inliers.is_empty() && observation_inliers.len() != observation_count {
        return Err(
            MultiViewBundleAdjustmentError::ObservationInlierMaskCountMismatch {
                expected: observation_count,
                actual: observation_inliers.len(),
            },
        );
    }
    if reports.len() != point_count {
        return Err(
            MultiViewBundleAdjustmentError::TriangulationReportCountMismatch {
                expected: point_count,
                actual: reports.len(),
            },
        );
    }
    options.try_validate()?;
    noise.try_validate(observation_count)?;
    validation::triangulation_problem(poses, point_count, model)?;
    for (point, report) in reports.iter_mut().enumerate() {
        if !track_selection.is_empty() && !track_selection[point] {
            *report = MultiViewTriangulationReport::default();
            continue;
        }
        *report = triangulate_track(model, poses, point, observation_inliers, options, noise);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PointLinearization {
    cost: f64,
    gradient: [f64; 3],
    hessian: [[f64; 3]; 3],
}

/// One point track as a three-parameter least-squares problem.
struct TrackProblem<'a> {
    model: ObservationModel<'a>,
    poses: &'a [RelativePose],
    point_index: usize,
    observation_inliers: &'a [bool],
    noise: NoiseModel<'a>,
}

impl LmProblem<3> for TrackProblem<'_> {
    type State = [f64; 3];

    fn linearize(&self, point: &[f64; 3], loss: Loss) -> Option<(NormalEquations<3>, f64)> {
        let linearization = linearize_track(
            self.model,
            self.poses,
            self.point_index,
            point,
            self.observation_inliers,
            loss,
            self.noise,
        )?;
        Some((
            NormalEquations::from_parts(linearization.hessian, linearization.gradient),
            linearization.cost,
        ))
    }

    fn gradient_inf_norm(&self, equations: &NormalEquations<3>) -> f64 {
        scaled_gradient_norm(equations.gradient(), equations.hessian())
    }

    fn retract(&self, point: &[f64; 3], step: &[f64; 3]) -> Option<[f64; 3]> {
        Some(std::array::from_fn(|coordinate| {
            point[coordinate] + step[coordinate]
        }))
    }

    fn cost(&self, point: &[f64; 3], loss: Loss) -> Option<f64> {
        track_cost(
            self.model,
            self.poses,
            self.point_index,
            point,
            self.observation_inliers,
            loss,
            self.noise,
        )
    }

    /// Jacobi-scaled point block solve, without the default step cap.
    fn solve_step(&self, equations: &NormalEquations<3>, damping: f64) -> Option<[f64; 3]> {
        let factor = PointBlockFactor::new(equations.hessian(), damping)?;
        Some(factor.solve(equations.gradient()).map(|value| -value))
    }

    /// Step length relative to the point's distance from the origin.
    fn step_norm(&self, point: &[f64; 3], step: &[f64; 3]) -> f64 {
        norm(step) / (1.0 + norm(point))
    }
}

fn triangulate_track(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point_index: usize,
    observation_inliers: &[bool],
    options: Options,
    noise: NoiseModel<'_>,
) -> MultiViewTriangulationReport {
    let track = model.offsets()[point_index]..model.offsets()[point_index + 1];
    let observation_count = track
        .clone()
        .filter(|&observation| retained(observation_inliers, observation))
        .count();
    let Some(point) = ray_intersection(model, poses, track.clone(), observation_inliers) else {
        return MultiViewTriangulationReport {
            observation_count,
            ..MultiViewTriangulationReport::default()
        };
    };
    let Some(initial_cost) = track_cost(
        model,
        poses,
        point_index,
        &point,
        observation_inliers,
        options.loss,
        noise,
    ) else {
        return MultiViewTriangulationReport {
            point,
            observation_count,
            ..MultiViewTriangulationReport::default()
        };
    };

    let problem = TrackProblem {
        model,
        poses,
        point_index,
        observation_inliers,
        noise,
    };
    let LmSolution {
        state: point,
        progress,
        ..
    } = minimize(&problem, point, options);

    let Some(final_linearization) = linearize_track(
        model,
        poses,
        point_index,
        &point,
        observation_inliers,
        options.loss,
        noise,
    ) else {
        return MultiViewTriangulationReport {
            point,
            status: RefinementStatus::Stalled,
            observation_count,
            iteration_count: progress.iteration_count,
            accepted_step_count: progress.accepted_step_count,
            rejected_step_count: progress.rejected_step_count,
            initial_cost,
            ..MultiViewTriangulationReport::default()
        };
    };
    let rcond = reciprocal_condition_number(&final_linearization.hessian);
    let diagnostics = InformationDiagnostics::new(
        3,
        if rcond > POINT_BLOCK_RCOND_THRESHOLD {
            3
        } else {
            0
        },
        2 * observation_count,
        rcond,
    );
    MultiViewTriangulationReport {
        point,
        status: progress.status(diagnostics),
        observation_count,
        iteration_count: progress.iteration_count,
        accepted_step_count: progress.accepted_step_count,
        rejected_step_count: progress.rejected_step_count,
        initial_cost,
        final_cost: final_linearization.cost,
        gradient_inf_norm: scaled_gradient_norm(
            &final_linearization.gradient,
            &final_linearization.hessian,
        ),
        step_norm: progress.step_norm,
        point_hessian_reciprocal_condition_number: rcond,
    }
}

fn ray_intersection(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    track: std::ops::Range<usize>,
    observation_inliers: &[bool],
) -> Option<[f64; 3]> {
    let mut hessian = [[0.0; 3]; 3];
    let mut right = [0.0; 3];
    let mut count = 0;
    for observation in track {
        if !retained(observation_inliers, observation) {
            continue;
        }
        let camera = model.camera(observation);
        let direction_camera = model.observation_bearing(observation)?;
        let direction_world = normalize(&transpose_multiply(
            &poses[camera].rotation,
            &direction_camera,
        ))?;
        let center = transpose_multiply(
            &poses[camera].rotation,
            &poses[camera].translation.map(|value| -value),
        );
        let projector: [[f64; 3]; 3] = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                (row == column) as u8 as f64 - direction_world[row] * direction_world[column]
            })
        });
        for row in 0..3 {
            right[row] += dot(&projector[row], &center);
            for column in 0..3 {
                hessian[row][column] += projector[row][column];
            }
        }
        count += 1;
    }
    if count < 2 || reciprocal_condition_number(&hessian) <= POINT_BLOCK_RCOND_THRESHOLD {
        return None;
    }
    PointBlockFactor::new(&hessian, 0.0).map(|factor| factor.solve(&right))
}

fn linearize_track(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point_index: usize,
    point: &[f64; 3],
    observation_inliers: &[bool],
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Option<PointLinearization> {
    let mut result = PointLinearization {
        cost: 0.0,
        gradient: [0.0; 3],
        hessian: [[0.0; 3]; 3],
    };
    let track = model.offsets()[point_index]..model.offsets()[point_index + 1];
    for observation in track {
        if !retained(observation_inliers, observation) {
            continue;
        }
        let camera = model.camera(observation);
        let camera_point = transform(&poses[camera], point);
        if !model.positive_depth(observation, &camera_point)? {
            return None;
        }
        let projection = model.project(observation, &camera_point)?;
        let residual_norm = norm2(&projection.residual);
        let inverse_sigma = noise.inverse_standard_deviation(observation);
        let whitened_norm = residual_norm * inverse_sigma;
        result.cost += loss.cost(whitened_norm);
        let sqrt_weight = inverse_sigma * loss.sqrt_weight(whitened_norm);
        for residual in 0..2 {
            let weighted_residual = sqrt_weight * projection.residual[residual];
            let jacobian: [f64; 3] = std::array::from_fn(|world| {
                sqrt_weight
                    * (0..3)
                        .map(|camera_coordinate| {
                            projection.jacobian[residual][camera_coordinate]
                                * poses[camera].rotation[camera_coordinate][world]
                        })
                        .sum::<f64>()
            });
            for row in 0..3 {
                result.gradient[row] += jacobian[row] * weighted_residual;
                for column in 0..3 {
                    result.hessian[row][column] += jacobian[row] * jacobian[column];
                }
            }
        }
    }
    result.cost.is_finite().then_some(result).filter(|result| {
        result.gradient.iter().all(|value| value.is_finite())
            && result
                .hessian
                .iter()
                .flatten()
                .all(|value| value.is_finite())
    })
}

fn track_cost(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point_index: usize,
    point: &[f64; 3],
    observation_inliers: &[bool],
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Option<f64> {
    let mut cost = 0.0;
    for observation in model.offsets()[point_index]..model.offsets()[point_index + 1] {
        if !retained(observation_inliers, observation) {
            continue;
        }
        let camera_point = transform(&poses[model.camera(observation)], point);
        if !model.positive_depth(observation, &camera_point)? {
            return None;
        }
        let projection = model.project(observation, &camera_point)?;
        cost +=
            loss.cost(norm2(&projection.residual) * noise.inverse_standard_deviation(observation));
    }
    cost.is_finite().then_some(cost)
}

fn retained(mask: &[bool], observation: usize) -> bool {
    mask.is_empty() || mask[observation]
}

fn scaled_gradient_norm(gradient: &[f64; 3], hessian: &[[f64; 3]; 3]) -> f64 {
    let scale = (0..3)
        .map(|index| hessian[index][index])
        .fold(0.0, f64::max);
    let floor = 64.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    (0..3)
        .map(|index| gradient[index].abs() / hessian[index][index].max(floor).sqrt())
        .fold(0.0, f64::max)
}

fn transpose_multiply(matrix: &[[f64; 3]; 3], vector: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|column| (0..3).map(|row| matrix[row][column] * vector[row]).sum())
}

fn normalize(vector: &[f64; 3]) -> Option<[f64; 3]> {
    let magnitude = norm(vector);
    (magnitude.is_finite() && magnitude > f64::MIN_POSITIVE)
        .then(|| vector.map(|value| value / magnitude))
}

fn norm(vector: &[f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn norm2(vector: &[f64; 2]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1]).sqrt()
}

fn dot<const N: usize>(left: &[f64; N], right: &[f64; N]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}
