// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `openmvg-ba` adapter used by SfM orchestration.

use std::sync::Mutex;

use openmvg_ba::{
    BundleAdjuster, BundleProblem, NativeBundleAdjuster, NativePreflightStatus, SolveConfig,
    SolveError,
};

use crate::ba::backend::{
    BaBackendFailure, BaBackendResult, BaBackendStatus, BaFailureKind, BaNotApplicable,
    BaPreflightDecision, BaPreflightDeferralReason, BaPreflightReport, BaPreflightResult,
    BaUnsupportedCapability, BundleAdjustmentBackend,
};
use crate::ba::config::BundleAdjustConfig;

/// Workspace budget SfM grants every bundle adjustment, in bytes: the budget the native solver
/// already selects its strategy against. Preflight defers, and the solve refuses, any linear-solver
/// workspace beyond it.
const WORKSPACE_BUDGET_BYTES: u128 = NativeBundleAdjuster::AUTO_WORKSPACE_BYTES as u128;

/// Bridges the stable mutable [`BundleAdjuster`] hook to the pipeline's shared backend handle.
///
/// The problem and solution cross this bridge unchanged. The mutex only
/// reconciles the adjuster's `&mut self` contract with the shared handle; it does not translate
/// or copy numerical state.
pub(crate) struct DirectBundleAdjustmentBackend<'a> {
    name: &'static str,
    adjuster: Mutex<&'a mut dyn BundleAdjuster>,
}

impl<'a> DirectBundleAdjustmentBackend<'a> {
    pub(crate) fn new(adjuster: &'a mut dyn BundleAdjuster) -> Self {
        Self {
            name: adjuster.name(),
            adjuster: Mutex::new(adjuster),
        }
    }
}

impl BundleAdjustmentBackend for DirectBundleAdjustmentBackend<'_> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn preflight(&self, problem: &BundleProblem, config: &BundleAdjustConfig) -> BaPreflightResult {
        preflight(problem, config, Some(WORKSPACE_BUDGET_BYTES), self.name)
    }

    fn adjust(&self, problem: &BundleProblem, config: &BundleAdjustConfig) -> BaBackendResult {
        let mut adjuster = self.adjuster.lock().map_err(|_| {
            failure_for(
                self.name,
                BaFailureKind::Internal,
                "bundle adjuster lock was poisoned",
            )
        })?;
        solve_with_adjuster(
            problem,
            config,
            Some(WORKSPACE_BUDGET_BYTES),
            &mut **adjuster,
            self.name,
        )
    }
}

fn solve_with_adjuster(
    problem: &BundleProblem,
    config: &BundleAdjustConfig,
    workspace_budget_bytes: Option<u128>,
    adjuster: &mut dyn BundleAdjuster,
    backend_name: &'static str,
) -> BaBackendResult {
    let solve_config = solve_config_for(config, workspace_budget_bytes, backend_name)?;
    match adjuster.solve(problem, solve_config) {
        Ok(solution) => Ok(BaBackendStatus::Proposed(Box::new(solution))),
        Err(SolveError::Unsupported(reason)) => Ok(BaBackendStatus::NotApplicable(
            BaNotApplicable::new(backend_name, BaUnsupportedCapability::Other, reason),
        )),
        Err(error) => Err(map_solve_error_for(backend_name, error)),
    }
}

fn preflight(
    problem: &BundleProblem,
    config: &BundleAdjustConfig,
    workspace_budget_bytes: Option<u128>,
    backend_name: &'static str,
) -> BaPreflightResult {
    let solve_config = solve_config_for(config, workspace_budget_bytes, backend_name)?;
    let preflight = NativeBundleAdjuster::new()
        .preflight(problem, &solve_config)
        .map_err(|error| map_solve_error_for(backend_name, error))?;
    let decision = match preflight.status {
        NativePreflightStatus::Proceed => BaPreflightDecision::Proceed,
        NativePreflightStatus::InsufficientCameraCoverage => {
            BaPreflightDecision::Deferred(BaPreflightDeferralReason::InsufficientCameraCoverage)
        }
        NativePreflightStatus::ResourceLimit => BaPreflightDecision::Deferred(
            BaPreflightDeferralReason::ObservabilityProbeResourceLimit,
        ),
        NativePreflightStatus::WeakPointGeometry => {
            BaPreflightDecision::Deferred(BaPreflightDeferralReason::WeakPointGeometry)
        }
        NativePreflightStatus::ObservabilityNotCertified => {
            BaPreflightDecision::Deferred(BaPreflightDeferralReason::ObservabilityNotCertified)
        }
        // A refusal this adapter does not know yet still fails closed.
        _ => BaPreflightDecision::Deferred(BaPreflightDeferralReason::ObservabilityNotCertified),
    };
    Ok(BaPreflightReport {
        decision,
        workspace_budget_bytes,
        pose_dimension: preflight.pose_dimension,
        covered_free_pose_count: preflight.covered_free_pose_count,
        weak_point_count: preflight.weak_point_count,
        dense_observability_escalation: preflight.dense_observability_escalation,
    })
}

/// The solve config a proposal from `backend_name` is audited against before commit. The audit
/// allocates no linear-solver workspace, so the config carries no workspace budget.
pub(crate) fn audit_solve_config(
    config: &BundleAdjustConfig,
    backend_name: &'static str,
) -> Result<SolveConfig, BaBackendFailure> {
    solve_config_for(config, None, backend_name)
}

fn solve_config_for(
    config: &BundleAdjustConfig,
    workspace_budget_bytes: Option<u128>,
    backend_name: &'static str,
) -> Result<SolveConfig, BaBackendFailure> {
    config.validate().map_err(|error| {
        failure_for(
            backend_name,
            BaFailureKind::InvalidConfig,
            error.to_string(),
        )
    })?;
    let mut solve_config = SolveConfig::default()
        .with_max_iterations(config.max_iterations())
        .with_function_tolerance(config.function_tolerance())
        .with_gradient_tolerance(config.gradient_tolerance())
        .with_parameter_tolerance(config.parameter_tolerance())
        .with_linear_solver(config.linear_solver());
    if let Some(bytes) = workspace_budget_bytes {
        solve_config = solve_config.with_workspace_budget_bytes(bytes);
    }
    solve_config
        .validate()
        .map_err(|error| map_solve_error_for(backend_name, error))?;
    Ok(solve_config)
}

fn failure_for(
    backend_name: &'static str,
    kind: BaFailureKind,
    reason: impl Into<String>,
) -> BaBackendFailure {
    BaBackendFailure::new(backend_name, kind, reason)
}

fn map_solve_error_for(backend_name: &'static str, error: SolveError) -> BaBackendFailure {
    let failure = |kind, reason: String| BaBackendFailure::new(backend_name, kind, reason);
    match error {
        SolveError::InvalidConfig(reason) => {
            failure(BaFailureKind::InvalidConfig, reason.to_owned())
        }
        SolveError::ResourceExhausted(reason) => {
            failure(BaFailureKind::ResourceExhausted, reason).with_retryable(true)
        }
        SolveError::Numerical(reason) => failure(BaFailureKind::Numerical, reason),
        SolveError::Unsupported(reason) => failure(BaFailureKind::InvalidConfig, reason),
        SolveError::Internal(reason) => failure(BaFailureKind::Internal, reason),
        _ => failure(BaFailureKind::Internal, error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point2, Point3, Rotation3, Vector3};
    use openmvg_ba::{
        BundleIntrinsic, BundleObservation, BundlePoint, BundlePose, Gauge, IntrinsicVariables,
        Observability, PoseVariables, SolveStatus,
    };
    use openmvg_core::{
        camera::{BrownConradyCamera, Camera, CameraModel, RadialDistortion, TangentialDistortion},
        model::{CameraIntrinsics, WorldToCameraPose},
    };

    use super::*;

    fn fixture() -> BundleProblem {
        let data = CameraIntrinsics::try_new(
            Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap(),
            64,
            48,
        )
        .unwrap();
        BundleProblem::builder(
            vec![BundlePose::new(
                openmvg_core::model::WorldToCameraPose::identity(),
                PoseVariables::FIXED,
            )],
            vec![BundleIntrinsic::try_new(data, IntrinsicVariables::FIXED).unwrap()],
            vec![BundlePoint::try_new(Point3::new(0.0, 0.0, 4.0)).unwrap()],
            vec![BundleObservation::try_new(0, 0, 0, Point2::new(32.0, 24.0)).unwrap()],
        )
        .build()
        .unwrap()
    }

    fn self_calibration_fixture(camera: Camera, observable_motion: bool) -> BundleProblem {
        // Pure translation is a critical motion for this self-calibration
        // problem. Independent rotations and a non-coplanar center make the
        // success fixture genuinely observable.
        let rotation = |angles: (f64, f64, f64)| {
            if observable_motion {
                Rotation3::from_euler_angles(angles.0, angles.1, angles.2)
            } else {
                Rotation3::identity()
            }
        };
        let truth_poses = [
            WorldToCameraPose::identity(),
            WorldToCameraPose::from_rotation_center(
                rotation((0.018, 0.065, -0.012)),
                Point3::new(0.6, 0.0, 0.0),
            )
            .unwrap(),
            WorldToCameraPose::from_rotation_center(
                rotation((-0.042, -0.052, 0.024)),
                Point3::new(-0.4, 0.1, if observable_motion { 0.13 } else { 0.0 }),
            )
            .unwrap(),
        ];
        let initial_poses = [
            WorldToCameraPose::identity(),
            WorldToCameraPose::from_rotation_center(
                rotation((0.020, 0.062, -0.010)),
                Point3::new(0.62, -0.01, 0.01),
            )
            .unwrap(),
            WorldToCameraPose::from_rotation_center(
                rotation((-0.039, -0.055, 0.026)),
                Point3::new(-0.42, 0.11, if observable_motion { 0.12 } else { -0.01 }),
            )
            .unwrap(),
        ];
        let truth_points = [
            Point3::new(-0.8, -0.5, 4.0),
            Point3::new(0.4, -0.3, 4.5),
            Point3::new(0.7, 0.8, 5.0),
            Point3::new(-0.3, 0.9, 5.5),
            Point3::new(-1.0, 0.2, 4.3),
            Point3::new(1.1, 0.1, 5.2),
            Point3::new(-0.6, -0.9, 6.0),
            Point3::new(0.2, 1.2, 6.4),
            Point3::new(0.9, -0.7, 5.8),
            Point3::new(-1.2, 0.7, 5.7),
            Point3::new(0.1, -1.1, 4.8),
            Point3::new(1.3, 0.9, 6.8),
        ];
        let observations = truth_poses
            .iter()
            .enumerate()
            .flat_map(|(camera_index, pose)| {
                let camera = camera.clone();
                truth_points
                    .into_iter()
                    .enumerate()
                    .map(move |(point_index, point)| {
                        let camera_point = pose.project_to_camera(&point);
                        BundleObservation::try_new(
                            camera_index,
                            0,
                            point_index,
                            camera.project(&camera_point.coords).unwrap(),
                        )
                        .unwrap()
                    })
            })
            .collect();
        let intrinsic = CameraIntrinsics::try_new(camera, 640, 480).unwrap();
        BundleProblem::builder(
            initial_poses
                .into_iter()
                .enumerate()
                .map(|(index, pose)| {
                    BundlePose::new(
                        pose,
                        if index == 0 {
                            PoseVariables::FIXED
                        } else {
                            PoseVariables::FREE
                        },
                    )
                })
                .collect(),
            vec![
                BundleIntrinsic::try_new(
                    intrinsic,
                    IntrinsicVariables::default()
                        .with_focal(true)
                        .with_radial(true),
                )
                .unwrap(),
            ],
            truth_points
                .into_iter()
                .map(|point| BundlePoint::try_new(point + Vector3::new(0.02, -0.01, 0.03)).unwrap())
                .collect(),
            observations,
        )
        .with_gauge(Gauge::FixedBaseline {
            first: 0,
            second: 1,
            distance: 0.6,
        })
        .build()
        .unwrap()
    }

    fn self_calibration_config() -> BundleAdjustConfig {
        BundleAdjustConfig::default()
            .with_scope(crate::ba::config::OptimizationScope::SelfCalibration)
            .with_robust_loss(openmvg_ba::RobustLoss::Squared)
            .with_linear_solver(openmvg_ba::LinearSolver::Direct)
    }

    #[test]
    fn direct_bridge_forwards_the_canonical_problem_without_translation() {
        struct InspectingAdjuster {
            called: bool,
        }

        impl BundleAdjuster for InspectingAdjuster {
            fn name(&self) -> &'static str {
                "inspecting-adjuster"
            }

            fn solve(
                &mut self,
                problem: &BundleProblem,
                _config: SolveConfig,
            ) -> Result<openmvg_ba::Solution, SolveError> {
                self.called = true;
                assert_eq!(problem.poses().len(), 1);
                assert_eq!(problem.points().len(), 1);
                Err(SolveError::Unsupported("test stop".to_owned()))
            }
        }

        let problem = fixture();
        let mut adjuster = InspectingAdjuster { called: false };
        {
            let backend = DirectBundleAdjustmentBackend::new(&mut adjuster);
            assert!(matches!(
                backend
                    .adjust(&problem, &BundleAdjustConfig::default())
                    .unwrap(),
                BaBackendStatus::NotApplicable(_)
            ));
        }
        assert!(adjuster.called);
    }

    #[test]
    fn direct_bridge_attributes_solve_config_errors_to_the_custom_adjuster() {
        struct NamedAdjuster;

        impl BundleAdjuster for NamedAdjuster {
            fn name(&self) -> &'static str {
                "named-adjuster"
            }

            fn solve(
                &mut self,
                _problem: &BundleProblem,
                _config: SolveConfig,
            ) -> Result<openmvg_ba::Solution, SolveError> {
                unreachable!("invalid solve config must be rejected before dispatch")
            }
        }

        let problem = fixture();
        let config = BundleAdjustConfig::default().with_max_iterations(0);
        let mut adjuster = NamedAdjuster;
        let backend = DirectBundleAdjustmentBackend::new(&mut adjuster);
        let error = backend.adjust(&problem, &config).unwrap_err();
        assert_eq!(error.backend_name, "named-adjuster");
        assert_eq!(error.kind, BaFailureKind::InvalidConfig);
    }

    #[test]
    fn direct_native_backend_accepts_observable_self_calibration_motion() {
        let problem = self_calibration_fixture(
            Camera::try_pinhole(700.0, 700.0, 320.0, 240.0).unwrap(),
            true,
        );
        let config = self_calibration_config();
        let solve_config = audit_solve_config(&config, "test").unwrap();
        let mut adjuster = NativeBundleAdjuster::new();
        let backend = DirectBundleAdjustmentBackend::new(&mut adjuster);
        let BaBackendStatus::Proposed(solution) = backend.adjust(&problem, &config).unwrap() else {
            panic!("observable self-calibration fixture must be supported");
        };

        assert_eq!(solution.intrinsics().len(), 1);
        assert!(solution.report().final_objective() <= solution.report().initial_objective());
        assert_eq!(
            solution.report().diagnostics().observability(),
            Observability::FullRank
        );
        assert_eq!(solution.audit_for_commit(&problem, &solve_config), Ok(()));
    }

    #[test]
    fn direct_native_backend_holds_intrinsics_under_critical_self_calibration_motion() {
        // Pure translation leaves focal exactly unobservable for a pinhole group, and only
        // nearly so once distortion is free: that solve stays numerically full rank while
        // focal drifts. Both groups keep their input intrinsics instead.
        let k1k2 = BrownConradyCamera::try_with_focal(
            700.0,
            320.0,
            240.0,
            RadialDistortion::K1K2(0.0, 0.0),
            TangentialDistortion::None,
        )
        .unwrap();
        for camera in [
            Camera::try_pinhole(700.0, 700.0, 320.0, 240.0).unwrap(),
            Camera::BrownConrady(k1k2),
        ] {
            let problem = self_calibration_fixture(camera, false);
            let config = self_calibration_config();
            let solve_config = audit_solve_config(&config, "test").unwrap();
            let mut adjuster = NativeBundleAdjuster::new();
            let backend = DirectBundleAdjustmentBackend::new(&mut adjuster);
            let BaBackendStatus::Proposed(solution) = backend.adjust(&problem, &config).unwrap()
            else {
                panic!("critical-motion fixture must reach the native solver");
            };

            assert_eq!(solution.report().unconstrained_intrinsics(), [0]);
            assert_eq!(
                solution.intrinsics()[0],
                *problem.intrinsics()[0].intrinsics()
            );
            assert!(matches!(
                solution.report().status(),
                SolveStatus::Converged(_)
            ));
            assert_eq!(
                solution.report().diagnostics().observability(),
                Observability::FullRank
            );
            assert_eq!(solution.audit_for_commit(&problem, &solve_config), Ok(()));
        }
    }
}
