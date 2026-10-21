// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental-SfM driver loop end-to-end on a synthetic 3-camera scene.
//!
//! A perfect front-end (exact projections, identity Essential geometry for the seed
//! pair) is injected so the test isolates the *driver*: seed init places two cameras
//! and triangulates their shared tracks, then the grow loop registers the third
//! camera via P3P and triangulates the remainder. Poses are recovered up to the gauge
//! fixed by the seed (image 0 = origin/identity, unit baseline to image 1).

use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

use nalgebra::{Point2, Point3, Rotation3};
use openmvg_core::model::WorldToCameraPose;
use openmvg_sfm::ba::BundleProblem as BaProblem;
use openmvg_sfm::{
    config::*,
    events::*,
    model::*,
    testing::seams::*,
    unstable::*,
    unstable::{
        adapters::triangulation::*, backends::*, incremental::*, reports::ErrorStats, types::*,
    },
    *,
};

#[test]
fn raw_translation_edges_do_not_veto_incremental_registration_by_default() {
    let config = IncrementalSfmConfig::default();

    assert_eq!(Incremental::default().config(), &config);
    assert!(config.bundle_adjust_initial_pair());
    assert_eq!(
        config.registration_max_rotation_edge_error(),
        Some(openmvg_core::AngularThreshold::TWENTY_DEGREES)
    );
    assert_eq!(config.registration_max_translation_edge_error(), None);
    assert_eq!(
        config.registration_config().max_translation_edge_error(),
        None
    );
}

const FOCAL: f64 = 500.0;
const CX: f64 = 400.0;
const CY: f64 = 300.0;

fn three_camera_workspace() -> SfmWorkspace {
    SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("cam-0", 800, 600),
            ImageMeta::external_id("cam-1", 800, 600),
            ImageMeta::external_id("cam-2", 800, 600),
        ])
        .camera_models(vec![Camera::try_pinhole(FOCAL, FOCAL, CX, CY).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .exhaustive_pairs()
        .build()
        .unwrap()
}

/// Run the injected front end and build tracks, as the production runner does before
/// handing the workspace to the incremental engine.
fn prepare_frontend(
    workspace: &mut SfmWorkspace,
    extractor: &dyn FeatureExtractor,
    matcher: &dyn PairMatcher,
    verifier: &dyn GeometricVerifier,
) -> Result<()> {
    workspace
        .extract_features(extractor, ViewSelector::Missing)?
        .match_pairs(matcher, PairSelector::Unmatched)?
        .verify_pairs(verifier, PairSelector::Unverified)?;
    workspace.build_tracks()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn incremental_from_frontend(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    extractor: &dyn FeatureExtractor,
    matcher: &dyn PairMatcher,
    verifier: &dyn GeometricVerifier,
    registrar: &dyn PoseRegistrar,
    triangulator: &dyn Triangulator,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
) -> Result<IncrementalSfmReport> {
    prepare_frontend(workspace, extractor, matcher, verifier)?;
    incremental_sfm_with(workspace, config, registrar, triangulator, ba_backend, None)
}

#[test]
fn observed_incremental_events_are_ordered_and_behavior_neutral() {
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4)
        .with_triangulation_min_observations(2);
    let mut unobserved = three_camera_workspace();
    unobserved
        .extract_features(&InjectExtractor, openmvg_sfm::unstable::ViewSelector::All)
        .unwrap()
        .match_pairs(&InjectMatcher, openmvg_sfm::unstable::PairSelector::All)
        .unwrap()
        .verify_pairs(&InjectVerifier, openmvg_sfm::unstable::PairSelector::All)
        .unwrap();
    unobserved.build_tracks().unwrap();
    let mut observed = unobserved.clone();

    let expected_report = incremental_sfm_with(
        &mut unobserved,
        &config,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        None,
        None,
    )
    .unwrap();
    let mut recorded = Vec::new();
    let mut sink = |event: openmvg_sfm::events::SfmEvent<'_>| {
        recorded.push((
            event.sequence,
            event.phase,
            event.transition,
            event.snapshot.registered_pose_count(),
        ));
    };
    let actual_report = incremental_sfm_with(
        &mut observed,
        &config,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        None,
        Some(&mut sink),
    )
    .unwrap();

    assert_eq!(actual_report, expected_report);
    assert_eq!(observed.pose_store(), unobserved.pose_store());
    assert_eq!(observed.landmarks(), unobserved.landmarks());
    assert_eq!(observed.components(), unobserved.components());

    assert!(
        recorded
            .iter()
            .enumerate()
            .all(|(index, event)| event.0 == index as u64)
    );
    assert!(matches!(
        recorded.first(),
        Some((
            _,
            SfmEventPhase::IncrementalInitialPair,
            SfmEventTransition::Committed,
            2
        ))
    ));
    let registered_counts = recorded
        .iter()
        .filter_map(|(_, phase, transition, registered)| {
            matches!(
                (phase, transition),
                (
                    SfmEventPhase::IncrementalRegistration { .. },
                    SfmEventTransition::Committed
                )
            )
            .then_some(*registered)
        })
        .collect::<Vec<_>>();
    assert_eq!(registered_counts, vec![3]);
    assert!(matches!(
        recorded.last(),
        Some((
            _,
            SfmEventPhase::IncrementalFinal,
            SfmEventTransition::Completed,
            3
        ))
    ));
}

fn seven_camera_workspace() -> SfmWorkspace {
    SfmWorkspace::builder()
        .image_meta(
            (0..7)
                .map(|index| ImageMeta::external_id(format!("cam-{index}"), 800, 600))
                .collect::<Vec<_>>(),
        )
        .camera_models(vec![Camera::try_pinhole(FOCAL, FOCAL, CX, CY).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .exhaustive_pairs()
        .build()
        .unwrap()
}

/// Ground-truth camera poses. Image 0 at origin/identity; image 1 unit-baseline along
/// +X (identity rotation) so the injected Essential (identity R, +X translation) is
/// the *true* seed geometry; image 2 rotated + translated (recovered by P3P).
fn gt_poses() -> [WorldToCameraPose; 7] {
    [
        WorldToCameraPose::identity(),
        WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::new(1.0, 0.0, 0.0))
            .unwrap(),
        WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(0.08, -0.05, 0.03),
            Point3::new(0.35, -0.1, -0.4),
        )
        .unwrap(),
        WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(-0.04, 0.06, -0.02),
            Point3::new(-0.3, 0.15, -0.25),
        )
        .unwrap(),
        WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(0.02, 0.04, 0.05),
            Point3::new(0.6, 0.2, -0.3),
        )
        .unwrap(),
        WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(-0.03, -0.02, 0.04),
            Point3::new(-0.55, -0.2, -0.35),
        )
        .unwrap(),
        WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(0.05, -0.01, -0.03),
            Point3::new(0.15, 0.3, -0.5),
        )
        .unwrap(),
    ]
}

fn world_points() -> Vec<Point3<f64>> {
    vec![
        Point3::new(-0.5, -0.3, 4.0),
        Point3::new(0.4, -0.2, 4.4),
        Point3::new(-0.2, 0.5, 4.8),
        Point3::new(0.6, 0.4, 5.2),
        Point3::new(-0.7, 0.2, 5.5),
        Point3::new(0.1, -0.6, 5.8),
        Point3::new(0.8, -0.1, 6.1),
        Point3::new(-0.4, 0.7, 6.4),
    ]
}

fn project(pose: &WorldToCameraPose, point: Point3<f64>) -> Point2<f64> {
    let camera_point = pose.project_to_camera(&point);
    Point2::new(
        FOCAL * camera_point.x / camera_point.z + CX,
        FOCAL * camera_point.y / camera_point.z + CY,
    )
}

#[derive(Debug, Clone, Copy)]
struct SeedBaCall {
    gauge: openmvg_sfm::ba::Gauge,
    camera_count: usize,
    point_count: usize,
    scope: OptimizationScope,
    robust_loss: openmvg_sfm::ba::RobustLoss,
    fixed_view_count: usize,
    fixed_view: Option<usize>,
    fixed_point_count: usize,
    intrinsics_fixed: bool,
}

#[derive(Default)]
struct RecordingSeedBa {
    calls: AtomicUsize,
    details: Mutex<Vec<SeedBaCall>>,
}

impl BundleAdjustmentBackend for RecordingSeedBa {
    fn name(&self) -> &'static str {
        "RecordingSeedBa"
    }

    fn adjust(&self, problem: &BaProblem, config: &BundleAdjustConfig) -> BaBackendResult {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.details.lock().unwrap().push(SeedBaCall {
            gauge: problem.gauge(),
            camera_count: problem.poses().len(),
            point_count: problem.points().len(),
            scope: config.scope(),
            robust_loss: config.robust_loss(),
            fixed_view_count: config.fixed_views().len(),
            fixed_view: config
                .fixed_views()
                .iter()
                .copied()
                .next()
                .map(ViewId::index),
            fixed_point_count: config.fixed_points().len(),
            intrinsics_fixed: problem
                .intrinsics()
                .iter()
                .all(|intrinsic| intrinsic.intrinsics().is_fixed()),
        });
        Ok(BaBackendStatus::Proposed(Box::new(
            openmvg_sfm::testing::identity_bundle_solution(problem),
        )))
    }
}

struct BehindCameraSeedBa;

impl BundleAdjustmentBackend for BehindCameraSeedBa {
    fn adjust(&self, problem: &BaProblem, _config: &BundleAdjustConfig) -> BaBackendResult {
        let points = problem
            .points()
            .iter()
            .map(|_| Point3::new(0.0, 0.0, -1.0))
            .collect();
        Ok(BaBackendStatus::Proposed(Box::new(
            openmvg_sfm::testing::bundle_solution(
                problem,
                problem
                    .poses()
                    .iter()
                    .map(|camera| camera.pose().clone())
                    .collect(),
                problem
                    .intrinsics()
                    .iter()
                    .map(|intrinsic| intrinsic.intrinsics().clone())
                    .collect(),
                points,
                openmvg_sfm::ba::SolveStatus::Skipped,
                1.0,
                1.0,
                0,
                match problem.gauge() {
                    openmvg_sfm::ba::Gauge::Free => {
                        openmvg_sfm::ba::Observability::OnlyExpectedGauge
                    }
                    _ => openmvg_sfm::ba::Observability::FullRank,
                },
            ),
        )))
    }
}

struct RmseRegressingSeedBa;

impl BundleAdjustmentBackend for RmseRegressingSeedBa {
    fn adjust(&self, problem: &BaProblem, _config: &BundleAdjustConfig) -> BaBackendResult {
        let diagnostics = openmvg_sfm::ba::SolveDiagnostics::new(
            0,
            0,
            openmvg_sfm::ba::StationarityAudit::try_from_fresh_zero_damping(0.0).unwrap(),
            match problem.gauge() {
                openmvg_sfm::ba::Gauge::Free => openmvg_sfm::ba::Observability::OnlyExpectedGauge,
                _ => openmvg_sfm::ba::Observability::FullRank,
            },
            0,
            0,
        );
        let report = openmvg_sfm::ba::SolveReport::try_new(
            openmvg_sfm::ba::SolveStatus::Skipped,
            0,
            1.0,
            1.0,
            1.0,
            1.1,
            diagnostics,
        )
        .unwrap();
        Ok(BaBackendStatus::Proposed(Box::new(
            openmvg_sfm::ba::Solution::try_new(
                problem,
                problem
                    .poses()
                    .iter()
                    .map(|camera| camera.pose().clone())
                    .collect(),
                problem
                    .intrinsics()
                    .iter()
                    .map(|intrinsic| intrinsic.intrinsics().clone())
                    .collect(),
                problem
                    .points()
                    .iter()
                    .map(|point| *point.position())
                    .collect(),
                report,
            )
            .unwrap(),
        )))
    }
}

#[derive(Default)]
struct NonCommittableFinalBa {
    calls: AtomicUsize,
}

impl BundleAdjustmentBackend for NonCommittableFinalBa {
    fn adjust(&self, problem: &BaProblem, _config: &BundleAdjustConfig) -> BaBackendResult {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(BaBackendStatus::Proposed(Box::new(
            openmvg_sfm::testing::bundle_solution(
                problem,
                problem
                    .poses()
                    .iter()
                    .map(|camera| camera.pose().clone())
                    .collect(),
                problem
                    .intrinsics()
                    .iter()
                    .map(|intrinsic| intrinsic.intrinsics().clone())
                    .collect(),
                problem
                    .points()
                    .iter()
                    .map(|point| *point.position())
                    .collect(),
                openmvg_sfm::ba::SolveStatus::Skipped,
                1.0,
                2.0,
                0,
                openmvg_sfm::ba::Observability::OnlyExpectedGauge,
            ),
        )))
    }
}

struct FailingBa {
    kind: BaFailureKind,
    calls: AtomicUsize,
}

impl FailingBa {
    fn new(kind: BaFailureKind) -> Self {
        Self {
            kind,
            calls: AtomicUsize::new(0),
        }
    }
}

impl BundleAdjustmentBackend for FailingBa {
    fn adjust(&self, _problem: &BaProblem, _config: &BundleAdjustConfig) -> BaBackendResult {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(BaBackendFailure::new(
            "FailingBa",
            self.kind,
            "injected failure",
        ))
    }
}

struct SevenMatchMatcher;

impl PairMatcher for SevenMatchMatcher {
    fn match_features(
        &self,
        ctx: &PairContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        PairMatches::try_new(
            ctx.pair,
            (0..7)
                .map(|index| {
                    FeatureMatch::try_new(FeatureId::new(index), FeatureId::new(index), Some(0.0))
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
    }
}

struct SeedBaOrderRegistrar<'a> {
    ba_calls: &'a AtomicUsize,
}

impl PoseRegistrar for SeedBaOrderRegistrar<'_> {
    fn estimate_pose(
        &self,
        ctx: &PoseRegistrationContext,
        observations: &[PoseRegistrationObservation],
    ) -> Result<Option<PoseRegistrationEstimate>> {
        assert_eq!(
            self.ba_calls.load(Ordering::SeqCst),
            1,
            "the seed BA must finish before the first P3P registration"
        );
        GenRansacP3pRegistrar::default().estimate_pose(ctx, observations)
    }
}

#[test]
fn configured_seed_ba_requires_backend_before_reconstruction() {
    let mut workspace = three_camera_workspace();
    let config =
        openmvg_sfm::testing::minimal_incremental_config().with_bundle_adjust_initial_pair(true);

    let error = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        None,
    )
    .unwrap_err();

    assert!(error.to_string().contains("bundle_adjust_initial_pair"));
    assert_eq!(workspace.pose_store().populated_count(), 0);
    assert!(workspace.landmarks().is_empty());
}

#[test]
fn seed_ba_rejects_underconstrained_seed_before_calling_backend() {
    let mut workspace = three_camera_workspace();
    let backend = RecordingSeedBa::default();
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(7),
        ))
        .with_bundle_adjust_initial_pair(true);

    let error = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &SevenMatchMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        Some(&backend),
    )
    .unwrap_err();

    assert!(error.to_string().contains("at least 8 active landmarks"));
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
    assert_eq!(workspace.pose_store().populated_count(), 0);
    assert!(workspace.landmarks().is_empty());
    assert!(workspace.components().is_empty());
}

#[test]
fn seed_ba_uses_fixed_unit_baseline_before_p3p_growth() {
    let mut workspace = three_camera_workspace();
    let backend = RecordingSeedBa::default();
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4)
        .with_triangulation_min_observations(2)
        .with_retriangulate(false)
        .with_bundle_adjust_initial_pair(true);

    let report = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &SeedBaOrderRegistrar {
            ba_calls: &backend.calls,
        },
        &DltTriangulator::default(),
        Some(&backend),
    )
    .unwrap();

    assert!(
        report
            .growth
            .growth_registrations
            .iter()
            .all(|registration| registration.pre_ba_post_refit_support.is_none()),
        "support-survival diagnostics are emitted only when existing landmarks are refit"
    );

    let seed_bundle = report
        .initialization
        .seed_bundle_adjustment
        .expect("enabled seed BA must be reported");
    assert_eq!(seed_bundle.ba.backend_name, "RecordingSeedBa");
    assert_eq!(seed_bundle.rejected_landmarks, 0);

    let first = backend.details.lock().unwrap()[0];
    let [_view_a, view_b] = report.initialization.initialized_views.as_slice() else {
        panic!("pair seed must initialize exactly two images");
    };
    assert_eq!(
        first.gauge,
        openmvg_sfm::ba::Gauge::FixedBaseline {
            first: 0,
            second: 1,
            distance: 1.0,
        }
    );
    assert_eq!(first.camera_count, 2);
    assert_eq!(first.point_count, 8);
    assert_eq!(first.scope, OptimizationScope::PosesAndPoints);
    assert_eq!(first.robust_loss, openmvg_sfm::ba::RobustLoss::Squared);
    assert_eq!(first.fixed_view_count, 1);
    assert_eq!(first.fixed_view, Some(0));
    assert_eq!(first.fixed_point_count, 0);
    assert!(first.intrinsics_fixed);

    let pose_a = &workspace.pose(ViewId::new(0)).unwrap().pose;
    let pose_b = &workspace.pose(*view_b).unwrap().pose;
    assert!(((pose_b.center() - pose_a.center()).norm() - 1.0).abs() <= 256.0 * f64::EPSILON);
}

#[test]
fn rejected_seed_ba_geometry_rolls_back_seed_state() {
    let mut workspace = three_camera_workspace();
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_bundle_adjust_initial_pair(true);

    let error = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        Some(&BehindCameraSeedBa),
    )
    .unwrap_err();

    assert!(error.to_string().contains("rejected"));
    assert_eq!(workspace.pose_store().populated_count(), 0);
    assert!(workspace.landmarks().is_empty());
    assert!(workspace.components().is_empty());
}

#[test]
fn seed_ba_reprojection_regression_rolls_back_seed_state() {
    let mut workspace = three_camera_workspace();
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_bundle_adjust_initial_pair(true);

    let error = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        Some(&RmseRegressingSeedBa),
    )
    .unwrap_err();

    assert!(error.to_string().contains("increased reprojection RMSE"));
    assert_eq!(workspace.pose_store().populated_count(), 0);
    assert!(workspace.landmarks().is_empty());
    assert!(workspace.components().is_empty());
}

#[test]
fn rejected_final_ba_retains_pre_final_reconstruction_and_is_reported() {
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4)
        .with_triangulation_min_observations(2);

    let mut expected_workspace = three_camera_workspace();
    let expected_report = incremental_from_frontend(
        &mut expected_workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        None,
    )
    .unwrap();

    let mut workspace = three_camera_workspace();
    let backend = NonCommittableFinalBa::default();
    let report = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        Some(&backend),
    )
    .unwrap();

    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    assert_eq!(report.final_ba_rejected_count, 1);
    assert_eq!(report.final_cleaned_landmark_count, 0);
    assert!(report.final_retriangulation.is_none());
    assert_eq!(
        report
            .final_refinement
            .as_ref()
            .expect("rejected final refinement is reported")
            .outcome,
        IncrementalFinalRefinementStatus::RejectedRolledBack
    );
    assert_eq!(
        report.registered_view_count,
        expected_report.registered_view_count
    );
    assert_eq!(report.landmark_count, expected_report.landmark_count);
    assert_eq!(
        workspace
            .pose_store()
            .iter_populated()
            .map(|(index, record)| (index, record.clone()))
            .collect::<Vec<_>>(),
        expected_workspace
            .pose_store()
            .iter_populated()
            .map(|(index, record)| (index, record.clone()))
            .collect::<Vec<_>>()
    );
    assert_eq!(workspace.landmarks(), expected_workspace.landmarks());
    assert_eq!(workspace.components(), expected_workspace.components());
    assert_eq!(
        workspace.camera_models(),
        expected_workspace.camera_models()
    );
}

#[test]
fn numerical_final_ba_failure_rolls_back_without_aborting_reconstruction() {
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4)
        .with_triangulation_min_observations(2);
    let mut workspace = three_camera_workspace();
    let backend = FailingBa::new(BaFailureKind::Numerical);

    let report = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        Some(&backend),
    )
    .expect("a numerical final-BA failure retains the growth reconstruction");

    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    assert_eq!(report.registered_view_count, 3);
    assert_eq!(report.final_ba_rejected_count, 1);
    assert_eq!(report.termination, IncrementalTermination::Complete);
}

#[test]
fn invalid_configuration_final_ba_failure_is_propagated() {
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4)
        .with_triangulation_min_observations(2);
    let mut workspace = three_camera_workspace();
    let backend = FailingBa::new(BaFailureKind::InvalidConfig);

    let error = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        Some(&backend),
    )
    .expect_err("an invalid final-BA configuration must abort reconstruction");

    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        &error,
        SfmError::BundleAdjustmentBackendFailed(failure)
            if failure.kind() == BundleAdjustmentFailureKind::InvalidConfig
    ));
}

#[test]
fn invalid_configuration_periodic_ba_failure_is_propagated() {
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4)
        .with_bundle_adjust_growth_ratio(0.1);
    let mut workspace = three_camera_workspace();
    let backend = FailingBa::new(BaFailureKind::InvalidConfig);

    let error = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        Some(&backend),
    )
    .expect_err("an invalid BA configuration must not be hidden as a rejected proposal");

    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        &error,
        SfmError::BundleAdjustmentBackendFailed(failure)
            if failure.kind() == BundleAdjustmentFailureKind::InvalidConfig
    ));
}

#[test]
fn final_refinement_rollback_event_exposes_the_restored_workspace() {
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4)
        .with_triangulation_min_observations(2);
    let mut expected = three_camera_workspace();
    incremental_from_frontend(
        &mut expected,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        None,
    )
    .unwrap();

    let mut workspace = three_camera_workspace();
    let backend = NonCommittableFinalBa::default();
    let mut rolled_back_state = None;
    let mut sink = |event: openmvg_sfm::events::SfmEvent<'_>| {
        if event.phase == SfmEventPhase::IncrementalFinalBundleAdjustment
            && event.transition == SfmEventTransition::RolledBack
        {
            rolled_back_state = Some((
                event
                    .snapshot
                    .unstable_workspace()
                    .pose_store()
                    .iter_populated()
                    .map(|(index, record)| (index, record.clone()))
                    .collect::<Vec<_>>(),
                event.snapshot.unstable_workspace().landmarks().clone(),
                event.snapshot.unstable_workspace().components().to_vec(),
            ));
        }
    };
    prepare_frontend(
        &mut workspace,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
    )
    .unwrap();
    incremental_sfm_with(
        &mut workspace,
        &config,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        Some(&backend),
        Some(&mut sink),
    )
    .unwrap();

    let (poses, landmarks, components) =
        rolled_back_state.expect("final rollback must be observed after restoration");
    assert_eq!(
        poses,
        expected
            .pose_store()
            .iter_populated()
            .map(|(index, record)| (index, record.clone()))
            .collect::<Vec<_>>()
    );
    assert_eq!(landmarks, expected.landmarks().clone());
    assert_eq!(components, expected.components());
}

#[test]
fn incremental_driver_reconstructs_three_camera_scene() {
    let mut workspace = SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("cam-0", 800, 600),
            ImageMeta::external_id("cam-1", 800, 600),
            ImageMeta::external_id("cam-2", 800, 600),
        ])
        .camera_models(vec![Camera::try_pinhole(FOCAL, FOCAL, CX, CY).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .exhaustive_pairs()
        .build()
        .unwrap();

    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4)
        .with_triangulation_min_observations(2);

    let report = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        None,
    )
    .unwrap();

    // Seed placed two cameras + triangulated; the driver grew to the third.
    assert_eq!(report.initialization.seed.views.len(), 2);
    assert_eq!(report.initialization.initialized_views.len(), 2);
    assert!(
        report.initialization.initialized_landmarks >= 4,
        "seed should triangulate the shared points, got {}",
        report.initialization.initialized_landmarks
    );
    assert_eq!(
        report.registered_view_count, 3,
        "all three cameras should register, got {}",
        report.registered_view_count
    );
    assert_eq!(report.termination, IncrementalTermination::Complete);
    assert_eq!(report.growth.grown_view_count, 1);
    assert!(report.growth.frontier_selection_count >= report.growth.iteration_count);
    assert!(
        report.growth.frontier_eligible_candidate_evaluation_count
            >= report
                .growth
                .frontier_support_band_candidate_evaluation_count
    );
    let support = report.growth.growth_registrations[0]
        .pre_ba_post_refit_support
        .expect("accepted registration reports post-update support survival");
    assert!(support.supporting_landmark_count > 0);
    assert!(support.supporting_landmark_count <= support.active_landmark_count);
    assert!(
        report.growth.triangulation_rejections.total()
            >= report.initialization.triangulation_rejections.total()
    );
    assert!(
        report.landmark_count >= 8,
        "all world points should be triangulated, got {}",
        report.landmark_count
    );

    // Poses recovered up to the seed gauge (image 0 = origin/identity, unit baseline).
    let gt = gt_poses();
    for (idx, gt_pose) in gt[..3].iter().enumerate() {
        let record = workspace
            .pose_store()
            .get(idx)
            .unwrap_or_else(|| panic!("image {idx} not registered"));
        let center_error = (record.pose.center() - gt_pose.center()).norm();
        let rotation_error = (record.pose.rotation().inverse() * gt_pose.rotation()).angle();
        assert!(
            center_error < 1e-4,
            "image {idx} center error {center_error}: got {:?}, gt {:?}",
            record.pose.center(),
            gt_pose.center()
        );
        assert!(
            rotation_error < 1e-4,
            "image {idx} rotation error {rotation_error} rad",
        );
    }
}

struct FailMultiViewTriangulation;

impl Triangulator for FailMultiViewTriangulation {
    fn triangulate(
        &self,
        observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        if observations.len() > 2 {
            return Err(SfmError::InvalidInput(
                "injected track-update failure".to_owned(),
            ));
        }
        DltTriangulator::default().triangulate(observations)
    }
}

#[test]
fn single_view_track_update_failure_rolls_back_the_reconstruction() {
    let mut workspace = three_camera_workspace();
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4);

    let error = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &FailMultiViewTriangulation,
        None,
    )
    .unwrap_err();

    assert!(error.to_string().contains("track-update failure"));
    assert_eq!(workspace.pose_store().populated_count(), 0);
    assert!(workspace.landmarks().is_empty());
    assert!(workspace.registration_failures().is_empty());
    workspace.validate().unwrap();
}

#[test]
fn retriangulation_extends_seed_landmarks_to_third_view() {
    // With re-triangulation on, the seed's two-view landmarks must be re-fit to three
    // views once camera 2 registers (track continuation). With it off, they stay at two.
    let build = || {
        SfmWorkspace::builder()
            .image_meta(vec![
                ImageMeta::external_id("cam-0", 800, 600),
                ImageMeta::external_id("cam-1", 800, 600),
                ImageMeta::external_id("cam-2", 800, 600),
            ])
            .camera_models(vec![Camera::try_pinhole(FOCAL, FOCAL, CX, CY).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .exhaustive_pairs()
            .build()
            .unwrap()
    };
    let base = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4)
        .with_triangulation_min_observations(2);

    let run = |retriangulate: bool| {
        let mut ws = build();
        let cfg = base.clone().with_retriangulate(retriangulate);
        incremental_from_frontend(
            &mut ws,
            &cfg,
            &InjectExtractor,
            &InjectMatcher,
            &InjectVerifier,
            &GenRansacP3pRegistrar::default(),
            &DltTriangulator::default(),
            None,
        )
        .unwrap();
        // Max observation count over the seed-shared landmarks (all 8 are visible in
        // every camera in this synthetic scene).
        ws.landmarks()
            .as_slice()
            .iter()
            .map(|landmark| {
                let track = ws
                    .tracks()
                    .get(landmark.source_track)
                    .expect("landmark source track exists");
                landmark.observation_support.len(track.observations().len())
            })
            .max()
            .unwrap_or(0)
    };

    let with_retri = run(true);
    let without_retri = run(false);
    assert_eq!(
        with_retri, 3,
        "re-triangulation should extend seed landmarks to all 3 views, got {with_retri}"
    );
    assert_eq!(
        without_retri, 2,
        "append-only triangulation leaves seed landmarks at 2 views, got {without_retri}"
    );
}

#[test]
fn adaptive_growth_ba_reaches_the_backend_and_is_reported() {
    let mut workspace = three_camera_workspace();
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::BestPair(
            InitialPairConfig::default().with_min_inliers(8),
        ))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4)
        .with_bundle_adjust_growth_ratio(0.1);

    let ba = RecordingSeedBa::default();
    let report = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &GenRansacP3pRegistrar::default(),
        &DltTriangulator::default(),
        Some(&ba),
    )
    .unwrap();

    assert_eq!(report.registered_view_count, 3);
    assert_eq!(report.growth.adaptive_ba_attempt_count, 1);
    assert_eq!(report.growth.adaptive_ba_applied_count, 1);
    assert_eq!(report.growth.adaptive_ba_rejected_count, 0);
    assert_eq!(report.growth.periodic_ba_count, 1);
    assert_eq!(report.growth.periodic_ba_attempt_count, 1);
    assert!(report.growth.periodic_ba_snapshot_payload_peak_bytes > 0);
    assert!(report.growth.growth_registrations[0].adaptive_ba_after);
    let final_refinement = report
        .final_refinement
        .as_ref()
        .expect("successful final refinement is reported");
    assert_eq!(
        final_refinement.outcome,
        IncrementalFinalRefinementStatus::Applied
    );
    assert_eq!(final_refinement.schema_version, 1);
    assert!(final_refinement.before_retriangulation.is_some());
    assert!(report.final_retriangulation.is_some());
}

#[test]
fn failed_candidate_is_retried_only_after_the_growth_frontier_is_exhausted() {
    let mut workspace = seven_camera_workspace();
    let registrar = RejectFirstRegistration::default();
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::ExplicitPair(ViewPair::new(
            ViewId::new(0),
            ViewId::new(1),
        )))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4);

    let report = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &registrar,
        &DltTriangulator::default(),
        None,
    )
    .unwrap();

    assert_eq!(report.registered_view_count, 7);
    let attempts = registrar.attempts.lock().unwrap();
    let deferred = attempts[0];
    assert_eq!(attempts.len(), 6);
    assert_eq!(attempts.last(), Some(&deferred));
    assert_eq!(
        attempts.iter().filter(|&&image| image == deferred).count(),
        2
    );
    for image in 2..7 {
        assert!(attempts.contains(&ViewId::new(image)));
    }
}

#[test]
fn failed_frontier_without_retained_growth_is_not_retried() {
    let mut workspace = seven_camera_workspace();
    let registrar = RecordingRejectAllRegistrar::default();
    let config = openmvg_sfm::testing::minimal_incremental_config()
        .with_initial_pair(InitialPairPolicy::ExplicitPair(ViewPair::new(
            ViewId::new(0),
            ViewId::new(1),
        )))
        .with_min_registration_support(4)
        .with_registration_min_inliers(4);

    let report = incremental_from_frontend(
        &mut workspace,
        &config,
        &InjectExtractor,
        &InjectMatcher,
        &InjectVerifier,
        &registrar,
        &DltTriangulator::default(),
        None,
    )
    .unwrap();

    assert_eq!(report.registered_view_count, 2);
    assert_eq!(
        report.termination,
        IncrementalTermination::AllCandidatesRejected
    );
    let mut attempts = registrar.attempts.lock().unwrap().clone();
    attempts.sort_unstable();
    assert_eq!(attempts, (2..7).map(ViewId::new).collect::<Vec<_>>());
}

#[derive(Default)]
struct RejectFirstRegistration {
    attempts: Mutex<Vec<ViewId>>,
    calls: AtomicUsize,
}

impl PoseRegistrar for RejectFirstRegistration {
    fn estimate_pose(
        &self,
        ctx: &PoseRegistrationContext,
        observations: &[PoseRegistrationObservation],
    ) -> Result<Option<PoseRegistrationEstimate>> {
        self.attempts.lock().unwrap().push(ctx.view);
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Ok(None)
        } else {
            GenRansacP3pRegistrar::default().estimate_pose(ctx, observations)
        }
    }
}

#[derive(Default)]
struct RecordingRejectAllRegistrar {
    attempts: Mutex<Vec<ViewId>>,
}

impl PoseRegistrar for RecordingRejectAllRegistrar {
    fn estimate_pose(
        &self,
        ctx: &PoseRegistrationContext,
        _observations: &[PoseRegistrationObservation],
    ) -> Result<Option<PoseRegistrationEstimate>> {
        self.attempts.lock().unwrap().push(ctx.view);
        Ok(None)
    }
}

// ── Injected perfect front-end ──────────────────────────────────────────────

struct InjectExtractor;

impl FeatureExtractor for InjectExtractor {
    fn extract(&self, ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
        let pose = &gt_poses()[ctx.view.index()];
        let keypoints = world_points()
            .into_iter()
            .map(|point| project(pose, point))
            .map(|xy| Keypoint::try_new(xy.x as f32, xy.y as f32, 1.0, 0.0))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(ExtractedFeatures::try_new(
            keypoints,
            DescriptorSet::try_f32(vec![0.0; 8], 1)?,
        )?)
    }
}

struct InjectMatcher;

impl PairMatcher for InjectMatcher {
    fn match_features(
        &self,
        ctx: &PairContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        PairMatches::try_new(
            ctx.pair,
            (0..8)
                .map(|index| {
                    FeatureMatch::try_new(FeatureId::new(index), FeatureId::new(index), Some(0.0))
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
    }
}

struct InjectVerifier;

impl GeometricVerifier for InjectVerifier {
    fn verify_pair(
        &self,
        ctx: &VerificationContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
        matches: &PairMatches,
        _ki: &openmvg_sfm::Camera,
        _kj: &openmvg_sfm::Camera,
    ) -> Result<VerificationVerdict> {
        let evidence = ModelEvidence::try_new(
            InlierMask::all_inliers(matches.len()),
            ErrorStats::from_values(&vec![0.0; matches.len()], &[1.0]),
            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
            "incremental-driver-test",
            1,
            TerminationReason::Converged,
            Vec::new(),
        )?;
        // Emit the *true* relative pose for every pair from the GT camera poses.
        // Canonical pair (a, b): R_rel = R_b · R_aᵀ and t_rel = R_a · (C_b − C_a)
        // (direction of C_b in camera-a's frame). The driver's registration guards
        // (rotation + translation edge-consistency) consume all edges, so every edge
        // must carry its real geometry — not just the seed pair.
        let gt = gt_poses();
        let (a, b) = (ctx.view_i.index(), ctx.view_j.index());
        let r_rel = gt[b].rotation() * gt[a].rotation().inverse();
        let t_cam_a = gt[a].rotation() * (gt[b].center() - gt[a].center());
        let t_unit = nalgebra::Unit::new_normalize(t_cam_a);
        Ok(VerificationVerdict::Accepted {
            model: GeometricModel::Essential {
                matrix: nalgebra::Matrix3::identity(),
                relative_pose: RelativePose::from_baseline_direction(r_rel, t_unit)?,
                evidence,
            },
            report: PairVerificationReport {
                pair: ctx.pair,
                input_match_count: matches.len(),
                inlier_count: matches.len(),
                message: None,
            },
        })
    }
}
