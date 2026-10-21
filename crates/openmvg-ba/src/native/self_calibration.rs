// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Self-calibrating solves that refine shared intrinsic groups.

use super::gauge::{common_pose_variables, gauge_plan};
use super::reporting::{map_native_error, solve_report};
use super::world_rebase::WorldRebase;
use super::{
    check_workspace_budget, iterative_schur_options, solver_options, sparse_memory_budget,
};
use crate::kernel::RelativePose;
use crate::kernel::multi_view::sparse::SparseSelfCalibrationWorkspace;
use crate::kernel::multi_view::sparse::refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors;
use crate::kernel::multi_view::sparse::{
    DirectSchurAssembly, SparseBundleAdjustmentReport,
    intrinsic_marginal_covariances_self_calibrating_grouped_sparse,
};
use crate::kernel::multi_view::{
    BundleAdjustmentPriors, CameraPoseVariables, FixedVariableBlocks, PixelObservation, ScaleGauge,
    self_calibration::RadialCamera, self_calibration::SelfCalibratingMeasurements,
};
use crate::numerics::refine::Options;
use crate::{BundleProblem, LinearSolver, ResidualSpace, Solution, SolveConfig, SolveError};
use openmvg_core::{
    camera::{BrownConradyCamera, Camera, CameraModel, RadialDistortion},
    model::CameraIntrinsics,
};

struct PreparedSelfCalibratingProblem {
    rebase: WorldRebase,
    poses: Vec<RelativePose>,
    points: Vec<[f64; 3]>,
    intrinsics: Vec<RadialCamera>,
    camera_to_intrinsic: Vec<usize>,
    observations: Vec<PixelObservation>,
    point_offsets: Vec<usize>,
    camera_source_order: Vec<usize>,
    fixed_poses: Vec<bool>,
    fixed_points: Vec<bool>,
    fixed_intrinsics: Vec<bool>,
    intrinsic_dimensions: Vec<usize>,
    gauge: ScaleGauge,
}

impl PreparedSelfCalibratingProblem {
    fn try_new(problem: &BundleProblem) -> Result<Self, SolveError> {
        let intrinsic_indices = super::single_pose_intrinsics(problem).ok_or_else(|| {
            SolveError::Unsupported(
                "native self-calibration requires one intrinsic group per observed pose".to_owned(),
            )
        })?;
        if !matches!(problem.residual_space(), ResidualSpace::Pixels) {
            return Err(SolveError::Unsupported(
                "self-calibration requires pixel residuals".to_owned(),
            ));
        }
        if common_pose_variables(problem)? != CameraPoseVariables::RotationAndTranslation {
            return Err(SolveError::Unsupported(
                "self-calibration requires full-pose or fixed camera blocks".to_owned(),
            ));
        }
        for intrinsic in problem.intrinsics() {
            let variables = intrinsic.variables();
            if variables.radial()
                && matches!(
                    intrinsic.intrinsics().camera(),
                    Camera::BrownConrady(camera)
                        if matches!(camera.radial_distortion(), RadialDistortion::K1K2K3(..))
                )
            {
                return Err(SolveError::Unsupported(
                    "native self-calibration refines at most two declared radial coefficients"
                        .to_owned(),
                ));
            }
            if !variables.is_fixed()
                && (!variables.focal() || variables.principal_point() || variables.tangential())
            {
                return Err(SolveError::Unsupported(
                    "self-calibration refines the focal length and radial coefficients with a fixed principal point"
                        .to_owned(),
                ));
            }
        }
        let gauge_plan = gauge_plan(problem)?;
        let rebase =
            WorldRebase::try_new(problem.poses()[gauge_plan.anchor].pose(), gauge_plan.scale)?;
        let camera_source_order = std::iter::once(gauge_plan.anchor)
            .chain((0..problem.poses().len()).filter(|index| *index != gauge_plan.anchor))
            .collect::<Vec<_>>();
        let mut camera_destination = vec![0; problem.poses().len()];
        for (destination, source) in camera_source_order.iter().copied().enumerate() {
            camera_destination[source] = destination;
        }
        let poses = camera_source_order
            .iter()
            .map(|&source| rebase.pose_to_native(problem.poses()[source].pose()))
            .collect::<Result<Vec<_>, _>>()?;
        let points = problem
            .points()
            .iter()
            .map(|point| rebase.point_to_native(point.position()))
            .collect::<Result<Vec<_>, _>>()?;
        let intrinsics = problem
            .intrinsics()
            .iter()
            .map(|intrinsic| radial_camera(intrinsic.intrinsics().camera()))
            .collect::<Result<Vec<_>, _>>()?;
        let camera_to_intrinsic = camera_source_order
            .iter()
            .map(|&source| intrinsic_indices[source])
            .collect::<Vec<_>>();
        let principal_points = problem
            .intrinsics()
            .iter()
            .map(|intrinsic| intrinsic.intrinsics().camera().principal_point())
            .collect::<Vec<_>>();
        let mut observation_source_order = (0..problem.observations().len()).collect::<Vec<_>>();
        observation_source_order.sort_by_key(|&source| {
            let observation = &problem.observations()[source];
            (observation.point(), source)
        });
        let observations = observation_source_order
            .iter()
            .map(|&source| {
                let observation = &problem.observations()[source];
                let principal = principal_points[observation.intrinsic()];
                PixelObservation::new(
                    camera_destination[observation.pose()],
                    [
                        observation.pixel().x - principal.x,
                        observation.pixel().y - principal.y,
                    ],
                )
            })
            .collect();
        let mut point_offsets = vec![0; problem.points().len() + 1];
        for observation in problem.observations() {
            point_offsets[observation.point() + 1] += 1;
        }
        for point in 0..problem.points().len() {
            point_offsets[point + 1] += point_offsets[point];
        }
        let fixed_poses = camera_source_order
            .iter()
            .map(|&source| {
                source == gauge_plan.anchor || problem.poses()[source].variables().is_fixed()
            })
            .collect();
        let fixed_points = problem
            .points()
            .iter()
            .map(|point| point.is_fixed())
            .collect();
        let fixed_intrinsics = problem
            .intrinsics()
            .iter()
            .map(|intrinsic| intrinsic.variables().is_fixed())
            .collect();
        let intrinsic_dimensions = problem
            .intrinsics()
            .iter()
            .map(|intrinsic| {
                free_radial_dimension(
                    intrinsic.intrinsics().camera(),
                    intrinsic.variables().radial(),
                )
            })
            .collect();
        Ok(Self {
            rebase,
            poses,
            points,
            intrinsics,
            camera_to_intrinsic,
            observations,
            point_offsets,
            camera_source_order,
            fixed_poses,
            fixed_points,
            fixed_intrinsics,
            intrinsic_dimensions,
            gauge: match gauge_plan.native {
                ScaleGauge::FixedBaseline { camera } => ScaleGauge::FixedBaseline {
                    camera: camera_destination[camera],
                },
                ScaleGauge::FixedTranslationComponent { camera, component } => {
                    ScaleGauge::FixedTranslationComponent {
                        camera: camera_destination[camera],
                        component,
                    }
                }
                ScaleGauge::Free => ScaleGauge::Free,
            },
        })
    }
}

impl PreparedSelfCalibratingProblem {
    fn measurements(&self) -> SelfCalibratingMeasurements<'_> {
        SelfCalibratingMeasurements::new(&self.observations, &self.point_offsets)
    }

    fn fixed(&self) -> FixedVariableBlocks<'_> {
        FixedVariableBlocks {
            camera_poses: &self.fixed_poses,
            points: &self.fixed_points,
            intrinsics: &self.fixed_intrinsics,
            intrinsic_dimensions: &self.intrinsic_dimensions,
        }
    }
}

/// Largest accepted marginal standard deviation, under unit pixel noise, of each free intrinsic
/// parameter's contribution to the relative radial displacement at the farthest image corner:
/// `σ(log f)`, `σ(k1)·r²`, and `σ(k2)·r⁴`, with `r` the corner's normalized radius. Exact rank
/// cannot certify self-calibration: near-critical motion leaves the reduced system numerically
/// full rank while the intrinsics drift along its weak direction. A group above this bound is not
/// constrained by its observations and keeps its input intrinsics.
const MAX_INTRINSIC_CORNER_DISPLACEMENT_STD: f64 = 0.01;

pub(super) fn solve_self_calibrating(
    problem: &BundleProblem,
    config: &SolveConfig,
) -> Result<Solution, SolveError> {
    let mut prepared = PreparedSelfCalibratingProblem::try_new(problem)?;
    let initial = (
        prepared.poses.clone(),
        prepared.points.clone(),
        prepared.intrinsics.clone(),
    );
    let options = solver_options(problem, config)?;
    let (report, mut workspace) = refine_self_calibrating(&mut prepared, config, options)?;
    let unconstrained = unconstrained_intrinsics(problem, &prepared, &mut workspace, options);
    if unconstrained.is_empty() {
        return raise_self_calibrating_solution(problem, prepared, &report, Vec::new());
    }
    // Re-solve from the input state so the held groups' poses and points do not keep what the
    // first solve's drift along the weak direction absorbed. Fixing parameters only shrinks the
    // other groups' marginals, so they stay constrained.
    drop(workspace);
    (prepared.poses, prepared.points, prepared.intrinsics) = initial;
    for &group in &unconstrained {
        prepared.fixed_intrinsics[group] = true;
    }
    let (report, _) = refine_self_calibrating(&mut prepared, config, options)?;
    raise_self_calibrating_solution(problem, prepared, &report, unconstrained)
}

/// Free intrinsic groups whose marginal uncertainty exceeds
/// [`MAX_INTRINSIC_CORNER_DISPLACEMENT_STD`]; every free group when the final information cannot
/// be factored, or its direct storage does not fit the workspace budget, because nothing then
/// certifies the intrinsics.
fn unconstrained_intrinsics(
    problem: &BundleProblem,
    prepared: &PreparedSelfCalibratingProblem,
    workspace: &mut SparseSelfCalibrationWorkspace,
    options: Options,
) -> Vec<usize> {
    let marginals = intrinsic_marginal_covariances_self_calibrating_grouped_sparse(
        &prepared.poses,
        &prepared.points,
        &prepared.intrinsics,
        &prepared.camera_to_intrinsic,
        prepared.measurements(),
        prepared.gauge,
        prepared.fixed(),
        workspace,
        options.loss,
        crate::numerics::refine::NoiseModel::Unit,
    );
    let Ok(marginals) = marginals else {
        return (0..prepared.intrinsics.len())
            .filter(|&group| !prepared.fixed_intrinsics[group])
            .collect();
    };
    marginals
        .iter()
        .filter(|marginal| {
            let data = problem.intrinsics()[marginal.group].intrinsics();
            let (width, height) = data.dimensions();
            let principal = data.camera().principal_point();
            let focal = prepared.intrinsics[marginal.group].focal;
            let radius_squared = (principal.x.max(f64::from(width) - principal.x) / focal).powi(2)
                + (principal.y.max(f64::from(height) - principal.y) / focal).powi(2);
            let scales = [1.0, radius_squared, radius_squared * radius_squared];
            let waived = problem
                .intrinsics()
                .get(marginal.group)
                .is_some_and(|block| {
                    block.certification() == crate::IntrinsicCertification::Waived
                });
            (0..marginal.dimension).any(|parameter| {
                // A negative variance (NaN deviation) is as unconstrained as a large one, even
                // for a waived block.
                let deviation = marginal.matrix[parameter][parameter].sqrt() * scales[parameter];
                deviation.is_nan() || (!waived && deviation > MAX_INTRINSIC_CORNER_DISPLACEMENT_STD)
            })
        })
        .map(|marginal| marginal.group)
        .collect()
}

fn refine_self_calibrating(
    prepared: &mut PreparedSelfCalibratingProblem,
    config: &SolveConfig,
    options: Options,
) -> Result<(SparseBundleAdjustmentReport, SparseSelfCalibrationWorkspace), SolveError> {
    // Field borrows, not the helper methods: the solve mutates poses, points, and intrinsics.
    let measurements =
        SelfCalibratingMeasurements::new(&prepared.observations, &prepared.point_offsets);
    let fixed = FixedVariableBlocks {
        camera_poses: &prepared.fixed_poses,
        points: &prepared.fixed_points,
        intrinsics: &prepared.fixed_intrinsics,
        intrinsic_dimensions: &prepared.intrinsic_dimensions,
    };
    let mut workspace = match config.linear_solver() {
        LinearSolver::Auto => SparseSelfCalibrationWorkspace::try_new_grouped_auto(
            prepared.poses.len(),
            prepared.intrinsics.len(),
            &prepared.camera_to_intrinsic,
            measurements,
            prepared.gauge,
            fixed,
            sparse_memory_budget(config),
        ),
        LinearSolver::Direct => {
            SparseSelfCalibrationWorkspace::try_new_grouped_with_direct_assembly(
                prepared.poses.len(),
                prepared.intrinsics.len(),
                &prepared.camera_to_intrinsic,
                measurements,
                prepared.gauge,
                fixed,
                DirectSchurAssembly::Auto,
            )
        }
        LinearSolver::Iterative => {
            SparseSelfCalibrationWorkspace::try_new_grouped_with_iterative_schur_options(
                prepared.poses.len(),
                prepared.intrinsics.len(),
                &prepared.camera_to_intrinsic,
                measurements,
                prepared.gauge,
                fixed,
                iterative_schur_options(),
            )
        }
    }
    .map_err(map_native_error)?;
    check_workspace_budget(
        "native self-calibration",
        workspace.execution_decision().selected_workspace_bytes(),
        config,
    )?;
    let report = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
        &mut prepared.poses,
        &mut prepared.points,
        &mut prepared.intrinsics,
        &prepared.camera_to_intrinsic,
        measurements,
        prepared.gauge,
        fixed,
        BundleAdjustmentPriors::NONE,
        &mut workspace,
        options,
        crate::numerics::refine::NoiseModel::Unit,
    )
    .map_err(map_native_error)?;
    Ok((report, workspace))
}

fn raise_self_calibrating_solution(
    problem: &BundleProblem,
    prepared: PreparedSelfCalibratingProblem,
    report: &SparseBundleAdjustmentReport,
    unconstrained_intrinsics: Vec<usize>,
) -> Result<Solution, SolveError> {
    let mut poses = problem
        .poses()
        .iter()
        .map(|camera| camera.pose().clone())
        .collect::<Vec<_>>();
    for (native, &source) in prepared
        .poses
        .iter()
        .copied()
        .zip(&prepared.camera_source_order)
    {
        if !problem.poses()[source].variables().is_fixed() {
            poses[source] = prepared.rebase.pose_from_native(native)?;
        }
    }
    let points = prepared
        .points
        .into_iter()
        .enumerate()
        .map(|(index, point)| {
            if problem.points()[index].is_fixed() {
                Ok(*problem.points()[index].position())
            } else {
                prepared.rebase.point_from_native(point)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let intrinsics = problem
        .intrinsics()
        .iter()
        .zip(prepared.intrinsics)
        .zip(&prepared.fixed_intrinsics)
        .map(|((input, radial), &fixed)| {
            if fixed {
                return Ok(input.intrinsics().clone());
            }
            let (width, height) = input.intrinsics().dimensions();
            CameraIntrinsics::try_new(
                with_radial_parameters(input.intrinsics().camera(), radial)
                    .map_err(|error| SolveError::Numerical(error.to_string()))?,
                width,
                height,
            )
            .map_err(|error| SolveError::Numerical(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Solution::try_new(
        problem,
        poses,
        intrinsics,
        points,
        solve_report(report, None, problem.gauge())?
            .with_unconstrained_intrinsics(unconstrained_intrinsics),
    )
    .map_err(super::reporting::map_solution_error)
}

/// Free leading `(log focal, k1, k2)` parameters of a refined intrinsic block: exactly the
/// parameters the camera model declares. A pinhole camera refines its focal length, a K1
/// Brown-Conrady camera adds `k1`, and K1K2 adds `k2`. Declared parameters the radial kernel
/// cannot represent are rejected by [`radial_camera`] unless zero, and then stay fixed.
fn free_radial_dimension(camera: &Camera, radial: bool) -> usize {
    match camera {
        Camera::BrownConrady(camera) if radial => match camera.radial_distortion() {
            RadialDistortion::None => 1,
            RadialDistortion::K1(_) => 2,
            RadialDistortion::K1K2(..) | RadialDistortion::K1K2K3(..) => 3,
        },
        _ => 1,
    }
}

/// Write refined radial parameters back into the input camera model, keeping its declared
/// distortion variant, principal point, and any parameters held fixed.
fn with_radial_parameters(
    camera: &Camera,
    radial: RadialCamera,
) -> Result<Camera, openmvg_core::camera::CameraError> {
    let principal = camera.principal_point();
    match camera {
        Camera::BrownConrady(camera) => {
            let distortion = match *camera.radial_distortion() {
                RadialDistortion::None => RadialDistortion::None,
                RadialDistortion::K1(_) => RadialDistortion::K1(radial.k1),
                RadialDistortion::K1K2(..) => RadialDistortion::K1K2(radial.k1, radial.k2),
                RadialDistortion::K1K2K3(_, _, k3) => {
                    RadialDistortion::K1K2K3(radial.k1, radial.k2, k3)
                }
            };
            BrownConradyCamera::try_with_focal(
                radial.focal,
                principal.x,
                principal.y,
                distortion,
                *camera.tangential_distortion(),
            )
            .map(Camera::BrownConrady)
        }
        _ => Camera::try_pinhole(radial.focal, radial.focal, principal.x, principal.y),
    }
}

fn radial_camera(camera: &Camera) -> Result<RadialCamera, SolveError> {
    let (focal_x, focal_y, k1, k2, k3, p1, p2) = match camera {
        Camera::Pinhole(camera) => (camera.focal_x(), camera.focal_y(), 0.0, 0.0, 0.0, 0.0, 0.0),
        Camera::BrownConrady(camera) => {
            let radial = camera.radial_distortion();
            let tangential = camera.tangential_distortion();
            (
                camera.focal_x(),
                camera.focal_y(),
                radial.k1(),
                radial.k2(),
                radial.k3(),
                tangential.p1(),
                tangential.p2(),
            )
        }
        Camera::Fisheye(_) | Camera::Spherical(_) => {
            return Err(SolveError::Unsupported(
                "self-calibration supports perspective cameras only".to_owned(),
            ));
        }
    };
    if focal_x != focal_y || k3 != 0.0 || p1 != 0.0 || p2 != 0.0 {
        return Err(SolveError::Unsupported(
            "self-calibration requires isotropic focal length, k1+k2 radial distortion, and no tangential distortion"
                .to_owned(),
        ));
    }
    Ok(RadialCamera::new(focal_x, k1, k2))
}
