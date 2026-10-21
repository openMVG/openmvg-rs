// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Small shared-state regression cases: physical BA variables, independent exposure losses,
//! fixed-state preservation and stored reconstruction compatibility.

use super::*;
use crate::ba::native::DirectBundleAdjustmentBackend;
use crate::model::{PairGeometry, ViewRecord};
use crate::{
    FeatureMatch, Keypoint, PoseId, Reconstruction, ReconstructionParts, SfmEngine, ViewGraph,
    ViewGraphParts, WorldToCameraPose,
};
use openmvg_ba::{
    BundleAdjuster, BundleProblem, NativeBundleAdjuster, Solution, SolveConfig, SolveError,
};

const VIEW_POSES: [usize; 4] = [0, 1, 1, 2];
const POINT_COUNT: usize = 12;

fn world_points() -> Vec<Point3<f64>> {
    (0..POINT_COUNT)
        .map(|i| {
            Point3::new(
                -0.9 + 0.5 * (i % 4) as f64,
                -0.6 + 0.4 * (i / 4) as f64,
                4.5 + 0.35 * (i % 5) as f64 + 0.1 * (i / 4) as f64,
            )
        })
        .collect()
}

fn scene(different_intrinsics: bool, points_only: bool) -> Reconstruction {
    let rotations = [
        Rotation3::identity(),
        Rotation3::from_euler_angles(0.02, 0.07, -0.015),
        Rotation3::from_euler_angles(-0.025, -0.06, 0.01),
    ];
    let centers = [
        Point3::origin(),
        Point3::new(0.6, 0.1, 0.0),
        Point3::new(1.2, -0.05, 0.15),
    ];
    let calibration = [[500.0, 510.0, 320.0, 240.0], [730.0, 680.0, 310.0, 250.0]];
    let intrinsic_ids = [0, 0, usize::from(different_intrinsics), 0];
    let cameras = calibration[..if different_intrinsics { 2 } else { 1 }]
        .iter()
        .map(|k| Camera::try_pinhole(k[0], k[1], k[2], k[3]).unwrap())
        .collect();
    let views = (0..4)
        .map(|view| {
            ViewRecord::new(
                ViewId::new(view),
                ImageMeta::external_id(view.to_string(), 640, 480),
                IntrinsicId::new(intrinsic_ids[view]),
                PoseId::new(VIEW_POSES[view]),
            )
        })
        .collect();
    let mut parts = ReconstructionParts::new(2026_u64, SfmEngine::Global, cameras, views)
        .with_calibrated_intrinsics(true);
    for pose in 0..3 {
        let (rotation, center) = if pose == 1 && !points_only {
            (
                Rotation3::from_euler_angles(0.004, -0.003, 0.002) * rotations[pose],
                centers[pose] + Vector3::new(0.03, -0.02, 0.01),
            )
        } else {
            (rotations[pose], centers[pose])
        };
        parts = parts.with_pose(
            PoseId::new(pose),
            WorldToCameraPose::from_rotation_center(rotation, center).unwrap(),
        );
    }
    let points = world_points();
    let volume = (points[1] - points[0])
        .cross(&(points[4] - points[0]))
        .dot(&(points[5] - points[0]));
    assert!(volume.abs() > 0.1, "the control points must be nonplanar");
    for (index, point) in points.iter().enumerate() {
        let observations = (0..4)
            .map(|view| {
                let pose = VIEW_POSES[view];
                let q = rotations[pose] * (point - centers[pose]);
                let k = calibration[intrinsic_ids[view]];
                // Independent pinhole equations. Two exposures at the shared pose
                // have small opposite measurement biases, never different PoseIds.
                let bias = if points_only {
                    0.0
                } else {
                    match view {
                        1 => 1.0,
                        2 => -1.0,
                        _ => 0.0,
                    }
                };
                (
                    ViewId::new(view),
                    Point2::new(
                        k[0] * q.x / q.z + k[2] + 0.75 * bias,
                        k[1] * q.y / q.z + k[3] + 0.35 * bias,
                    ),
                )
            })
            .collect();
        let initial = if points_only {
            point + Vector3::new(0.015, -0.01, 0.02 + 0.001 * index as f64)
        } else {
            *point
        };
        parts = parts.with_point(initial, observations);
    }
    Reconstruction::try_from_parts(parts).unwrap()
}

#[derive(Default)]
struct RecordingNative {
    native: NativeBundleAdjuster,
    calls: usize,
    summary: Option<serde_json::Value>,
}

impl BundleAdjuster for RecordingNative {
    fn name(&self) -> &'static str {
        "shared-pose-probe-native"
    }

    fn solve(
        &mut self,
        problem: &BundleProblem,
        config: SolveConfig,
    ) -> std::result::Result<Solution, SolveError> {
        self.calls += 1;
        let aliases: Vec<_> = (0..problem.poses().len())
            .flat_map(|a| (a + 1..problem.poses().len()).map(move |b| (a, b)))
            .filter(|&(a, b)| problem.poses()[a].pose() == problem.poses()[b].pose())
            .collect();
        let free_pose_dimension: usize = problem
            .poses()
            .iter()
            .map(|pose| {
                3 * usize::from(pose.variables().rotation())
                    + 3 * usize::from(pose.variables().translation())
            })
            .sum();
        let result = self.native.solve(problem, config);
        let mut summary = serde_json::json!({
            "pose_blocks": problem.poses().len(),
            "intrinsic_blocks": problem.intrinsics().len(),
            "observations": problem.observations().len(),
            "free_pose_dimension": free_pose_dimension,
            "input_alias_pairs": aliases,
        });
        match &result {
            Ok(solution) => {
                let center_gap = aliases
                    .iter()
                    .map(|&(a, b)| {
                        (solution.poses()[a].center() - solution.poses()[b].center()).norm()
                    })
                    .fold(0.0, f64::max);
                let rotation_gap = aliases
                    .iter()
                    .map(|&(a, b)| {
                        (solution.poses()[a].rotation().matrix()
                            - solution.poses()[b].rotation().matrix())
                        .norm()
                    })
                    .fold(0.0, f64::max);
                summary["native_status"] =
                    serde_json::json!(format!("{:?}", solution.report().status()));
                summary["native_rank"] = serde_json::json!(format!(
                    "{:?}",
                    solution.report().diagnostics().observability()
                ));
                summary["initial_objective"] =
                    serde_json::json!(solution.report().initial_objective());
                summary["final_objective"] = serde_json::json!(solution.report().final_objective());
                summary["alias_center_gap"] = serde_json::json!(center_gap);
                summary["alias_rotation_gap"] = serde_json::json!(rotation_gap);
            }
            Err(error) => summary["native_error"] = serde_json::json!(error.to_string()),
        }
        self.summary = Some(summary);
        result
    }
}

fn config(scope: OptimizationScope) -> BundleAdjustConfig {
    BundleAdjustConfig::default()
        .with_scope(scope)
        .with_fixed_views([ViewId::new(0), ViewId::new(3)].into_iter().collect())
        .with_robust_loss(openmvg_ba::RobustLoss::Squared)
        .with_max_iterations(60)
        .with_function_tolerance(1e-12)
        .with_gradient_tolerance(1e-9)
        .with_parameter_tolerance(1e-12)
}

// Independent scalar pinhole equations; do not obtain this oracle from the BA evaluator.
fn squared_objective(workspace: &SfmWorkspace) -> f64 {
    let mut objective = 0.0;
    for landmark in workspace.landmarks().iter_active() {
        let track = workspace.tracks().get(landmark.source_track).unwrap();
        for (_, observation) in landmark.supporting_observations(track) {
            let view = workspace.view(observation.view).unwrap();
            let Camera::Pinhole(camera) = workspace.camera_model(view.intrinsic_id).unwrap() else {
                panic!("this bounded oracle uses pinhole equations");
            };
            let pose = &workspace.pose(observation.view).unwrap().pose;
            let q = pose.rotation() * (landmark.point - pose.center());
            let measured = workspace.track_observation_pixel(observation).unwrap();
            let dx = camera.focal_x() * q.x / q.z + camera.principal_x() - measured.x;
            let dy = camera.focal_y() * q.y / q.z + camera.principal_y() - measured.y;
            objective += 0.5 * (dx * dx + dy * dy);
        }
    }
    objective
}

fn free_pose_probe(different_intrinsics: bool) {
    let mut workspace = scene(different_intrinsics, false).workspace.clone();
    let initial_poses: Vec<_> = workspace
        .pose_store()
        .iter_populated()
        .map(|(index, pose)| (index, pose.clone()))
        .collect();
    let initial_points = workspace.landmarks().clone();
    let initial_cameras = workspace.camera_models().to_vec();
    let initial_objective = squared_objective(&workspace);
    let mut adjuster = RecordingNative::default();
    let report = {
        let backend = DirectBundleAdjustmentBackend::new(&mut adjuster);
        bundle_adjust(
            &mut workspace,
            &backend,
            &config(OptimizationScope::PosesOnly),
        )
    }
    .expect("shared physical poses must be optimized jointly by the native backend");
    assert_eq!(adjuster.calls, 1);
    let summary = adjuster.summary.as_ref().unwrap();
    assert_eq!(summary["pose_blocks"], 3);
    assert_eq!(summary["free_pose_dimension"], 6);
    assert_eq!(summary["observations"], 48);
    assert_eq!(
        summary["intrinsic_blocks"],
        if different_intrinsics { 2 } else { 1 }
    );
    assert_eq!(report.ba.pose_count, 3);
    assert_eq!(report.adjusted_cameras, 3);
    assert_eq!(report.ba.observation_count, 48);
    assert_eq!(workspace.pose_store().populated_count(), 3);
    assert_eq!(workspace.registered_view_count(), 4);
    assert_eq!(
        workspace.pose(ViewId::new(1)).unwrap().pose,
        workspace.pose(ViewId::new(2)).unwrap().pose
    );
    for index in [0, 2] {
        assert_eq!(
            workspace.pose_by_id(PoseId::new(index)).unwrap(),
            &initial_poses[index].1
        );
    }
    for (actual, initial) in workspace
        .landmarks()
        .as_slice()
        .iter()
        .zip(initial_points.as_slice())
    {
        assert_eq!(actual.point, initial.point);
    }
    assert_eq!(workspace.camera_models(), initial_cameras);

    // Allow each world point any shared normalized image ray as a relaxation. For opposite
    // biases e and focal lengths a,b, the two half-squared terms have lower bound
    // 0.5 e^2 (a+b)^2/(a^2+b^2), independently in x and y. One shared rigid pose cannot beat it.
    let second = if different_intrinsics {
        [730.0_f64, 680.0]
    } else {
        [500.0, 510.0]
    };
    let per_point: f64 = [(500.0_f64, second[0], 0.75_f64), (510.0, second[1], 0.35)]
        .iter()
        .map(|&(a, b, e)| 0.5 * e * e * (a + b).powi(2) / (a * a + b * b))
        .sum();
    let lower_bound = POINT_COUNT as f64 * per_point;
    let final_objective = squared_objective(&workspace);
    assert!(
        final_objective >= lower_bound - 1e-7,
        "a shared-state objective {final_objective} cannot undercut {lower_bound}"
    );
    assert!(final_objective < initial_objective);
    if !different_intrinsics {
        // The true pose predicts the midpoint of both equal-calibration exposures exactly.
        assert!((final_objective - lower_bound).abs() < 1e-6);
    }
    assert!((report.ba.final_objective - final_objective).abs() < 1e-8);
    workspace.validate().unwrap();
    eprintln!(
        "SHARED_POSE_PROBE {}",
        serde_json::json!({
            "case": if different_intrinsics { "free_different_intrinsics" } else { "free_same_intrinsics" },
            "physical_poses": 3, "views": 4, "backend_calls": adjuster.calls,
            "native": summary, "independent_final_objective": final_objective,
            "shared_objective_lower_bound": lower_bound,
        })
    );
}

#[test]
fn shared_pose_probe_free_same_intrinsics() {
    free_pose_probe(false);
}

#[test]
fn shared_pose_probe_free_different_intrinsics() {
    free_pose_probe(true);
}

#[test]
fn shared_pose_probe_fixed_poses_refine_points_with_different_intrinsics() {
    let mut workspace = scene(true, true).workspace.clone();
    let initial_poses: Vec<_> = workspace
        .pose_store()
        .iter_populated()
        .map(|(index, pose)| (index, pose.clone()))
        .collect();
    let mut adjuster = RecordingNative::default();
    let outcome = {
        let backend = DirectBundleAdjustmentBackend::new(&mut adjuster);
        bundle_adjust(
            &mut workspace,
            &backend,
            &config(OptimizationScope::PointsOnly),
        )
    };
    eprintln!(
        "SHARED_POSE_PROBE {}",
        serde_json::json!({
            "case": "fixed_poses_different_intrinsics",
            "physical_poses": 3,
            "views": 4,
            "native": adjuster.summary,
            "sfm_error": outcome.as_ref().err().map(ToString::to_string),
        })
    );
    outcome.expect("fixed shared poses should permit points-only refinement");
    assert_eq!(
        workspace
            .pose_store()
            .iter_populated()
            .map(|(index, pose)| (index, pose.clone()))
            .collect::<Vec<_>>(),
        initial_poses,
    );
    assert_eq!(workspace.pose_store().populated_count(), 3);
    for (landmark, truth) in workspace.landmarks().iter_active().zip(world_points()) {
        assert!((landmark.point - truth).norm() < 1e-7);
        assert!(landmark.reprojection_error.value() < 1e-7);
    }
}

#[test]
fn shared_pose_probe_checkpoint_preserves_poses_intrinsics_and_observations() {
    let original = scene(true, true);
    let bytes = original.to_bytes().unwrap();
    let restored = Reconstruction::from_bytes(&bytes).unwrap();
    assert_eq!(restored.views(), original.views());
    assert_eq!(restored.camera_models(), original.camera_models());
    assert_eq!(restored.workspace.pose_store().populated_count(), 3);
    assert_eq!(restored.workspace.registered_view_count(), 4);
    assert_eq!(restored.observations().count(), 48);
    assert_eq!(
        restored.workspace.pose(ViewId::new(1)).unwrap().pose,
        restored.workspace.pose(ViewId::new(2)).unwrap().pose,
    );
    assert_eq!(restored.to_bytes().unwrap(), bytes);
}

#[test]
fn shared_pose_preflight_preserves_exposures_and_fixes_the_selected_alias_group() {
    let workspace = scene(true, false).workspace;
    let cfg = config(OptimizationScope::PosesOnly)
        .with_fixed_views([ViewId::new(2)].into_iter().collect());
    // This direct preparation entry is used by preflight; it must not rely on the solve wrapper.
    let problem = bundle_adjustment_problem(&workspace, &cfg).unwrap();
    assert_eq!(problem.poses().len(), 3);
    assert_eq!(problem.intrinsics().len(), 2);
    assert_eq!(problem.observations().len(), 48);
    assert_eq!(
        problem.poses()[1].variables(),
        openmvg_ba::PoseVariables::FIXED
    );
    assert_eq!(
        problem.poses()[0].variables(),
        openmvg_ba::PoseVariables::FREE
    );
    assert_eq!(
        problem.poses()[2].variables(),
        openmvg_ba::PoseVariables::FREE
    );
    for point in 0..POINT_COUNT {
        let exposures = &problem.observations()[4 * point..4 * point + 4];
        assert_eq!(
            exposures.iter().map(|obs| obs.pose()).collect::<Vec<_>>(),
            VIEW_POSES
        );
        assert_eq!(
            exposures
                .iter()
                .map(|obs| obs.intrinsic())
                .collect::<Vec<_>>(),
            [0, 0, 1, 0]
        );
        assert!(exposures.iter().all(|obs| obs.point() == point));
    }
}

#[test]
fn equal_transforms_do_not_merge_distinct_pose_identities() {
    let mut workspace = scene(true, false).workspace;
    let identical = workspace.pose_by_id(PoseId::new(0)).unwrap().pose.clone();
    for id in [1, 2] {
        workspace
            .set_pose(PoseRecord {
                id: PoseId::new(id),
                pose: identical.clone(),
                component: Some(ComponentId::new(0)),
            })
            .unwrap();
    }
    let problem =
        bundle_adjustment_problem(&workspace, &config(OptimizationScope::PosesOnly)).unwrap();
    assert_eq!(problem.poses().len(), 3);
    assert_eq!(problem.poses()[0].pose(), problem.poses()[1].pose());
    assert_eq!(problem.poses()[1].pose(), problem.poses()[2].pose());
    assert_eq!(
        problem.poses()[1].variables(),
        openmvg_ba::PoseVariables::FREE
    );
    assert_eq!(
        problem.poses()[0].variables(),
        openmvg_ba::PoseVariables::FIXED
    );
}

#[test]
fn alias_views_cannot_form_a_fixed_baseline_gauge() {
    let mut workspace = scene(true, false).workspace;
    let component = ComponentId::new(0);
    for pose in 0..3 {
        workspace.poses.get_mut(pose).unwrap().component = Some(component);
    }
    workspace.components = vec![crate::model::ReconstructionComponent::new(
        component,
        (0..4).map(ViewId::new).collect(),
        workspace
            .landmarks()
            .iter_active()
            .map(|landmark| landmark.id)
            .collect(),
        GaugeScalePolicy::Free,
    )];
    workspace.validate().unwrap();
    let distance = (workspace.pose(ViewId::new(3)).unwrap().pose.center()
        - workspace.pose(ViewId::new(2)).unwrap().pose.center())
    .norm();
    workspace.components[0].gauge_scale_policy = GaugeScalePolicy::FixedBaseline {
        first: ViewId::new(3),
        second: ViewId::new(2),
        distance,
        unit: crate::config::LengthUnit::Arbitrary,
    };
    let problem =
        bundle_adjustment_problem(&workspace, &config(OptimizationScope::PosesOnly)).unwrap();
    assert_eq!(
        problem.gauge(),
        openmvg_ba::Gauge::FixedBaseline {
            first: 2,
            second: 1,
            distance
        }
    );
    workspace.components[0].gauge_scale_policy = GaugeScalePolicy::FixedBaseline {
        first: ViewId::new(1),
        second: ViewId::new(2),
        distance: 1.0,
        unit: crate::config::LengthUnit::Arbitrary,
    };
    assert!(matches!(
        bundle_adjustment_problem(&workspace, &config(OptimizationScope::PosesOnly)),
        Err(SfmError::InvalidInput(_))
    ));
}

fn prepared_alias_graph() -> ViewGraph {
    let cameras = vec![
        Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap(),
        Camera::try_pinhole(120.0, 110.0, 30.0, 23.0).unwrap(),
    ];
    let views = (0..3)
        .map(|view| {
            ViewRecord::new(
                ViewId::new(view),
                ImageMeta::external_id(view.to_string(), 64, 48),
                IntrinsicId::new(usize::from(view == 1)),
                PoseId::new(usize::from(view == 2)),
            )
        })
        .collect();
    let keypoints = (0..3)
        .map(|view| {
            (0..8)
                .map(|feature| {
                    let x = -0.4 + 0.2 * (feature % 4) as f64 - if view == 2 { 1.0 } else { 0.0 };
                    let y = -0.2 + 0.4 * (feature / 4) as f64;
                    let k = if view == 1 {
                        [120.0, 110.0, 30.0, 23.0]
                    } else {
                        [100.0, 100.0, 32.0, 24.0]
                    };
                    Keypoint::try_new(
                        (k[0] * x / 5.0 + k[2]) as f32,
                        (k[1] * y / 5.0 + k[3]) as f32,
                        2.0,
                        0.0,
                    )
                    .unwrap()
                })
                .collect()
        })
        .collect();
    let mut parts =
        ViewGraphParts::new(11, cameras, views, keypoints).with_calibrated_intrinsics(true);
    for first in 0..3 {
        for second in first + 1..3 {
            let geometry = if (first, second) == (0, 1) {
                PairGeometry::RotationOnly {
                    rotation: Rotation3::identity(),
                }
            } else {
                let relative_pose = openmvg_core::model::RelativePose::from_baseline_direction(
                    Rotation3::identity(),
                    Vector3::x_axis(),
                )
                .unwrap();
                PairGeometry::Essential {
                    matrix: relative_pose.translation_source_to_target().cross_matrix(),
                    relative_pose: Some(relative_pose),
                }
            };
            let matches = (0..8)
                .map(|feature| {
                    FeatureMatch::try_new(
                        FeatureId::new(feature),
                        FeatureId::new(feature),
                        Some(0.25),
                    )
                    .unwrap()
                })
                .collect();
            parts = parts.with_pair(
                crate::ViewPair::new(ViewId::new(first), ViewId::new(second)),
                matches,
                Some((geometry, vec![true; 8])),
            );
        }
    }
    ViewGraph::try_from_parts(parts).unwrap()
}

#[test]
fn global_initialization_refuses_shared_poses_without_changing_the_graph() {
    let graph = prepared_alias_graph();
    let bytes = graph.to_bytes().unwrap();
    let error = crate::Global::default().run_prepared(&graph).unwrap_err();
    assert!(
        matches!(error, SfmError::UnsupportedSharedPoseInitialization { pose } if pose == PoseId::new(0))
    );
    assert_eq!(graph.to_bytes().unwrap(), bytes);
    #[cfg(feature = "unstable-internals")]
    {
        let mut workspace = graph.workspace.clone();
        let error = crate::Global::default()
            .runner()
            .run_in_place(&mut workspace)
            .unwrap_err();
        assert!(matches!(
            error,
            SfmError::UnsupportedSharedPoseInitialization { .. }
        ));
        assert_eq!(
            ViewGraph::try_from_workspace(workspace)
                .unwrap()
                .with_calibrated_intrinsics(graph.has_calibrated_intrinsics())
                .with_extraction_method(graph.extraction_method())
                .to_bytes()
                .unwrap(),
            bytes
        );
    }
}

#[test]
fn incremental_initialization_refuses_shared_poses_without_changing_the_graph() {
    let graph = prepared_alias_graph();
    let bytes = graph.to_bytes().unwrap();
    let error = crate::Incremental::default()
        .run_prepared(&graph)
        .unwrap_err();
    assert!(
        matches!(error, SfmError::UnsupportedSharedPoseInitialization { pose } if pose == PoseId::new(0))
    );
    assert_eq!(graph.to_bytes().unwrap(), bytes);
    #[cfg(feature = "unstable-internals")]
    {
        let mut workspace = graph.workspace.clone();
        let error = crate::Incremental::default()
            .runner()
            .run_in_place(&mut workspace)
            .unwrap_err();
        assert!(matches!(
            error,
            SfmError::UnsupportedSharedPoseInitialization { .. }
        ));
        assert_eq!(
            ViewGraph::try_from_workspace(workspace)
                .unwrap()
                .with_calibrated_intrinsics(graph.has_calibrated_intrinsics())
                .with_extraction_method(graph.extraction_method())
                .to_bytes()
                .unwrap(),
            bytes
        );
    }
}
