// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Matrix3, Point3, Rotation3, Vector3};
use openmvg_core::model::{EstimationResult, RelativeRotation, RelativeTranslation};
use openmvg_geometry::averaging::{
    RotationConfig, RotationEstimate, TranslationConfig, TranslationEstimate,
};
use openmvg_sfm::ba::BundleProblem as BaProblem;
use openmvg_sfm::{
    config::*,
    model::*,
    testing::seams::*,
    unstable::*,
    unstable::{backends::*, reports::ErrorStats, types::*},
    *,
};

fn bundle_workspace() -> SfmWorkspace {
    bundle_workspace_with_gauge(GaugeScalePolicy::Free)
}

fn bundle_workspace_with_gauge(gauge_scale_policy: GaugeScalePolicy) -> SfmWorkspace {
    let mut workspace = SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("image-a", 64, 48),
            ImageMeta::external_id("image-b", 64, 48),
        ])
        .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs(vec![(ViewId::new(0), ViewId::new(1))])
        .build()
        .unwrap();
    let config = openmvg_sfm::config::GlobalAveragingConfig::default()
        .with_gauge_scale_policy(gauge_scale_policy);
    workspace
        .extract_features(&BundleExtractor, openmvg_sfm::unstable::ViewSelector::All)
        .unwrap();
    workspace
        .match_pairs(&BundleMatcher, openmvg_sfm::unstable::PairSelector::Active)
        .unwrap();
    workspace
        .verify_pairs(&BundleVerifier, openmvg_sfm::unstable::PairSelector::Active)
        .unwrap();
    openmvg_sfm::unstable::ops::averaging::run_global_averaging_scoped(
        &mut workspace,
        bundle_rotations,
        bundle_centers,
        config
            .min_translation_parallax()
            .map(|threshold| threshold.value().to_degrees()),
        config.min_edge_isotropy(),
    )
    .unwrap();
    workspace.build_tracks().unwrap();
    workspace.release_matching_data().unwrap();
    openmvg_sfm::unstable::ops::reconstruction::triangulate(
        &mut workspace,
        &BundleTriangulator,
        gauge_scale_policy,
    )
    .unwrap();
    workspace
}

struct BundleExtractor;

impl FeatureExtractor for BundleExtractor {
    fn extract(&self, ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
        let x = if ctx.view == ViewId::new(0) {
            32.0
        } else {
            42.0
        };
        Ok(ExtractedFeatures::try_new(
            vec![Keypoint::try_new(x, 24.0, 1.0, 0.0)?],
            DescriptorSet::try_f32(vec![0.0], 1)?,
        )?)
    }
}

struct BundleMatcher;

impl PairMatcher for BundleMatcher {
    fn match_features(
        &self,
        ctx: &PairContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        PairMatches::try_new(
            ctx.pair,
            vec![FeatureMatch::try_new(
                FeatureId::new(0),
                FeatureId::new(0),
                Some(0.0),
            )?],
        )
    }
}

struct BundleVerifier;

impl GeometricVerifier for BundleVerifier {
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
            "bundle-test",
            1,
            TerminationReason::Converged,
            Vec::new(),
        )?;
        Ok(VerificationVerdict::Accepted {
            model: GeometricModel::Essential {
                matrix: Matrix3::identity(),
                relative_pose: RelativePose::from_baseline_direction(
                    Rotation3::identity(),
                    Vector3::x_axis(),
                )?,
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

fn bundle_rotations(
    rotations: &[RelativeRotation],
    _config: &RotationConfig,
) -> EstimationResult<RotationEstimate> {
    Ok(RotationEstimate::new(
        rotations
            .iter()
            .flat_map(|rotation| [rotation.view_i(), rotation.view_j()])
            .map(|view| (view, Rotation3::identity()))
            .collect(),
        0,
        openmvg_core::AngularResidual::ZERO,
    ))
}

fn bundle_centers(
    _translations: &[RelativeTranslation],
    _rotations: &std::collections::BTreeMap<ViewId, Rotation3<f64>>,
    _config: &TranslationConfig,
) -> EstimationResult<TranslationEstimate> {
    Ok(TranslationEstimate::new(
        [
            (ViewId::new(0), Vector3::new(0.0, 0.0, 0.0)),
            (ViewId::new(1), Vector3::new(1.0, 0.0, 0.0)),
        ]
        .into_iter()
        .collect(),
        0,
        openmvg_core::AngularResidual::ZERO,
    ))
}

struct BundleTriangulator;

impl Triangulator for BundleTriangulator {
    fn triangulate(
        &self,
        observations: &[PosedPixelObservation<'_>],
    ) -> Result<openmvg_sfm::unstable::backends::TriangulationVerdict> {
        Ok(TriangulatedLandmark {
            point: Point3::new(0.5, 0.0, 5.0),
            reprojection_error: openmvg_core::PixelResidual::try_new(0.25).unwrap(),
            supporting_observation_indices: (0..observations.len())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
        .into())
    }
}

struct IdentityBackend;

impl BundleAdjustmentBackend for IdentityBackend {
    fn name(&self) -> &'static str {
        "IdentityTestBA"
    }

    fn adjust(
        &self,
        problem: &BaProblem,
        _cfg: &openmvg_sfm::unstable::BundleAdjustConfig,
    ) -> BaBackendResult {
        Ok(BaBackendStatus::Proposed(Box::new(
            openmvg_sfm::testing::identity_bundle_solution(problem),
        )))
    }
}

#[test]
fn bundle_adjust_preserves_tracks() {
    let mut workspace = bundle_workspace();
    let track_count = workspace.tracks().len();

    let report = bundle_adjust(
        &mut workspace,
        &IdentityBackend,
        &openmvg_sfm::unstable::BundleAdjustConfig::default(),
    )
    .unwrap();

    assert_eq!(report.ba.backend_name, "IdentityTestBA");
    assert_eq!(report.adjusted_cameras, 2);
    assert_eq!(report.adjusted_landmarks, 1);
    assert_eq!(report.rejected_landmarks, 0);
    assert_eq!(workspace.tracks().len(), track_count);
    assert_eq!(workspace.landmarks().len(), 1);
    assert!(
        (workspace.landmarks().as_slice()[0]
            .reprojection_error
            .value()
            - 15.0)
            .abs()
            < 1e-12
    );
    assert!(workspace.pose(ViewId::new(0)).is_some());
}

struct GaugeCheckingUnsupportedBackend {
    expected: openmvg_sfm::ba::Gauge,
    adjust_calls: AtomicUsize,
}

impl BundleAdjustmentBackend for GaugeCheckingUnsupportedBackend {
    fn adjust(
        &self,
        problem: &BaProblem,
        _cfg: &openmvg_sfm::unstable::BundleAdjustConfig,
    ) -> BaBackendResult {
        assert_eq!(problem.gauge(), self.expected);
        self.adjust_calls.fetch_add(1, Ordering::SeqCst);
        Ok(BaBackendStatus::NotApplicable(BaNotApplicable::new(
            "UnsupportedBA",
            BaUnsupportedCapability::Gauge,
            "injected unsupported gauge",
        )))
    }
}

#[test]
fn bundle_adjustment_propagates_gauge_and_rejects_non_applicability() {
    let gauge = GaugeScalePolicy::FixedBaseline {
        first: ViewId::new(0),
        second: ViewId::new(1),
        distance: 1.0,
        unit: LengthUnit::Meters,
    };
    let mut workspace = bundle_workspace_with_gauge(gauge);
    let poses_before = workspace
        .pose_store()
        .iter_populated()
        .map(|(index, record)| (index, record.pose.clone()))
        .collect::<Vec<_>>();
    let points_before = workspace
        .landmarks()
        .as_slice()
        .iter()
        .map(|landmark| landmark.point)
        .collect::<Vec<_>>();
    let primary = GaugeCheckingUnsupportedBackend {
        expected: openmvg_sfm::ba::Gauge::FixedBaseline {
            first: 0,
            second: 1,
            distance: 1.0,
        },
        adjust_calls: AtomicUsize::new(0),
    };
    let error = bundle_adjust(
        &mut workspace,
        &primary,
        &openmvg_sfm::unstable::BundleAdjustConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(error, SfmError::BundleAdjustmentNotApplicable(_)));
    assert_eq!(primary.adjust_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        workspace
            .pose_store()
            .iter_populated()
            .map(|(index, record)| (index, record.pose.clone()))
            .collect::<Vec<_>>(),
        poses_before
    );
    assert_eq!(
        workspace
            .landmarks()
            .as_slice()
            .iter()
            .map(|landmark| landmark.point)
            .collect::<Vec<_>>(),
        points_before
    );
}

#[test]
fn non_applicable_backend_does_not_mutate_workspace() {
    let mut workspace = bundle_workspace();
    let points_before = workspace
        .landmarks()
        .as_slice()
        .iter()
        .map(|landmark| landmark.point)
        .collect::<Vec<_>>();
    let backend = GaugeCheckingUnsupportedBackend {
        expected: openmvg_sfm::ba::Gauge::Free,
        adjust_calls: AtomicUsize::new(0),
    };

    let error = bundle_adjust(
        &mut workspace,
        &backend,
        &openmvg_sfm::unstable::BundleAdjustConfig::default(),
    )
    .unwrap_err();

    assert!(matches!(error, SfmError::BundleAdjustmentNotApplicable(_)));
    assert_eq!(backend.adjust_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        workspace
            .landmarks()
            .as_slice()
            .iter()
            .map(|landmark| landmark.point)
            .collect::<Vec<_>>(),
        points_before
    );
}

struct DegenerateProposalBackend;

impl BundleAdjustmentBackend for DegenerateProposalBackend {
    fn adjust(
        &self,
        problem: &BaProblem,
        _cfg: &openmvg_sfm::unstable::BundleAdjustConfig,
    ) -> BaBackendResult {
        let mut cameras = problem
            .poses()
            .iter()
            .map(|camera| camera.pose().clone())
            .collect::<Vec<_>>();
        let mut center = *cameras[0].center();
        center.x += 10.0;
        cameras[0] = openmvg_core::model::WorldToCameraPose::from_rotation_center(
            *cameras[0].rotation(),
            center,
        )
        .expect("finite shifted pose");
        Ok(BaBackendStatus::Proposed(Box::new(
            openmvg_sfm::testing::bundle_solution(
                problem,
                cameras,
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
                openmvg_sfm::ba::SolveStatus::Degenerate,
                1.0,
                1.0,
                0,
                openmvg_sfm::ba::Observability::OnlyExpectedGauge,
            ),
        )))
    }
}

#[test]
fn non_committable_proposal_is_rejected_before_workspace_mutation() {
    let mut workspace = bundle_workspace();
    let poses_before = workspace
        .pose_store()
        .iter_populated()
        .map(|(index, record)| (index, record.pose.clone()))
        .collect::<Vec<_>>();

    let error = bundle_adjust(
        &mut workspace,
        &DegenerateProposalBackend,
        &openmvg_sfm::unstable::BundleAdjustConfig::default(),
    )
    .unwrap_err();

    assert!(matches!(
        &error,
        SfmError::BundleAdjustmentProposalRejected { reason, .. }
            if reason.commit_rejection()
                == Some(BaCommitRejection::Status(BaSolveStatus::Degenerate))
    ));
    assert_eq!(
        workspace
            .pose_store()
            .iter_populated()
            .map(|(index, record)| (index, record.pose.clone()))
            .collect::<Vec<_>>(),
        poses_before
    );
}

struct MalformedCoverageBackend;

impl BundleAdjustmentBackend for MalformedCoverageBackend {
    fn adjust(
        &self,
        problem: &BaProblem,
        _cfg: &openmvg_sfm::unstable::BundleAdjustConfig,
    ) -> BaBackendResult {
        let _ = problem;
        Err(BaBackendFailure::new(
            "MalformedCoverageBA",
            BaFailureKind::InvalidProblem,
            "proposal does not cover the input state exactly",
        ))
    }
}

struct MutatesFixedPoseBackend;

impl BundleAdjustmentBackend for MutatesFixedPoseBackend {
    fn adjust(
        &self,
        problem: &BaProblem,
        _cfg: &openmvg_sfm::unstable::BundleAdjustConfig,
    ) -> BaBackendResult {
        let _ = problem;
        Err(BaBackendFailure::new(
            "MutatesFixedPoseBA",
            BaFailureKind::InvalidProblem,
            "proposal changed fixed pose",
        ))
    }
}

#[test]
fn malformed_coverage_and_fixed_state_are_rejected_transactionally() {
    let mut workspace = bundle_workspace();
    let poses_before = workspace
        .pose_store()
        .iter_populated()
        .map(|(index, record)| (index, record.pose.clone()))
        .collect::<Vec<_>>();
    let error = bundle_adjust(
        &mut workspace,
        &MalformedCoverageBackend,
        &openmvg_sfm::unstable::BundleAdjustConfig::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("does not cover"));
    assert_eq!(
        workspace
            .pose_store()
            .iter_populated()
            .map(|(index, record)| (index, record.pose.clone()))
            .collect::<Vec<_>>(),
        poses_before
    );

    let config = openmvg_sfm::unstable::BundleAdjustConfig::default()
        .with_fixed_views([ViewId::new(0)].into_iter().collect());
    let error = bundle_adjust(&mut workspace, &MutatesFixedPoseBackend, &config).unwrap_err();
    assert!(error.to_string().contains("changed fixed pose"));
    assert_eq!(
        workspace
            .pose_store()
            .iter_populated()
            .map(|(index, record)| (index, record.pose.clone()))
            .collect::<Vec<_>>(),
        poses_before
    );
}

struct InvalidPointBackend;

struct NonFinitePointBackend;

impl BundleAdjustmentBackend for NonFinitePointBackend {
    fn adjust(
        &self,
        _problem: &BaProblem,
        _cfg: &openmvg_sfm::unstable::BundleAdjustConfig,
    ) -> BaBackendResult {
        Err(BaBackendFailure::new(
            "NonFinitePointBA",
            BaFailureKind::InvalidProblem,
            "bundle adjustment returned non-finite landmark",
        ))
    }
}

#[test]
fn bundle_adjust_rejects_non_finite_point_before_mutating_workspace() {
    let mut workspace = bundle_workspace();
    let poses_before = workspace
        .pose_store()
        .iter_populated()
        .map(|(index, record)| (index, record.pose.clone()))
        .collect::<Vec<_>>();
    let points_before = workspace
        .landmarks()
        .as_slice()
        .iter()
        .map(|landmark| landmark.point)
        .collect::<Vec<_>>();

    let error = bundle_adjust(
        &mut workspace,
        &NonFinitePointBackend,
        &openmvg_sfm::unstable::BundleAdjustConfig::default(),
    )
    .unwrap_err();

    assert!(error.to_string().contains("non-finite landmark"));
    let poses_after = workspace
        .pose_store()
        .iter_populated()
        .map(|(index, record)| (index, record.pose.clone()))
        .collect::<Vec<_>>();
    let points_after = workspace
        .landmarks()
        .as_slice()
        .iter()
        .map(|landmark| landmark.point)
        .collect::<Vec<_>>();
    assert_eq!(poses_after, poses_before);
    assert_eq!(points_after, points_before);
}

struct IntrinsicUpdateBackend;

impl BundleAdjustmentBackend for IntrinsicUpdateBackend {
    fn adjust(
        &self,
        problem: &BaProblem,
        cfg: &openmvg_sfm::unstable::BundleAdjustConfig,
    ) -> BaBackendResult {
        assert_eq!(cfg.scope(), OptimizationScope::SelfCalibration);
        assert!(
            problem
                .intrinsics()
                .iter()
                .all(|intrinsic| !intrinsic.variables().is_fixed())
        );
        let input = problem.intrinsics()[0].intrinsics();
        let intrinsic = openmvg_core::model::CameraIntrinsics::try_new(
            Camera::try_pinhole(110.0, 110.0, 32.0, 24.0).unwrap(),
            input.width(),
            input.height(),
        )
        .unwrap();
        Ok(BaBackendStatus::Proposed(Box::new(
            openmvg_sfm::testing::bundle_solution(
                problem,
                problem
                    .poses()
                    .iter()
                    .map(|camera| camera.pose().clone())
                    .collect(),
                vec![intrinsic],
                problem
                    .points()
                    .iter()
                    .map(|point| *point.position())
                    .collect(),
                openmvg_sfm::ba::SolveStatus::Skipped,
                0.0,
                0.0,
                0,
                openmvg_sfm::ba::Observability::OnlyExpectedGauge,
            ),
        )))
    }
}

#[test]
fn self_calibration_unlocks_and_commits_intrinsic_groups() {
    let mut workspace = bundle_workspace();
    bundle_adjust(
        &mut workspace,
        &IntrinsicUpdateBackend,
        &openmvg_sfm::unstable::BundleAdjustConfig::default()
            .with_scope(OptimizationScope::SelfCalibration),
    )
    .unwrap();

    let Camera::Pinhole(camera) = workspace.camera_model(IntrinsicId::new(0)).unwrap() else {
        panic!("self-calibration must preserve the pinhole model");
    };
    assert_eq!(camera.focal_x(), 110.0);
    assert_eq!(camera.focal_y(), 110.0);
}

impl BundleAdjustmentBackend for InvalidPointBackend {
    fn adjust(
        &self,
        problem: &BaProblem,
        _cfg: &openmvg_sfm::unstable::BundleAdjustConfig,
    ) -> BaBackendResult {
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
                    .map(|_| Point3::new(0.0, 0.0, -5.0))
                    .collect(),
                openmvg_sfm::ba::SolveStatus::Skipped,
                0.0,
                0.0,
                0,
                openmvg_sfm::ba::Observability::OnlyExpectedGauge,
            ),
        )))
    }
}

#[derive(Default)]
struct RejectOnceBackend {
    calls: AtomicUsize,
}

impl BundleAdjustmentBackend for RejectOnceBackend {
    fn adjust(
        &self,
        problem: &BaProblem,
        _cfg: &openmvg_sfm::unstable::BundleAdjustConfig,
    ) -> BaBackendResult {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let points = problem
            .points()
            .iter()
            .map(|point| {
                if call == 0 {
                    Point3::new(0.0, 0.0, -5.0)
                } else {
                    *point.position()
                }
            })
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
                0.0,
                0.0,
                0,
                openmvg_sfm::ba::Observability::OnlyExpectedGauge,
            ),
        )))
    }
}

#[test]
fn bundle_adjust_rejects_geometry_outside_the_camera_domain() {
    let mut workspace = bundle_workspace();

    let report = bundle_adjust(
        &mut workspace,
        &InvalidPointBackend,
        &openmvg_sfm::unstable::BundleAdjustConfig::default(),
    )
    .unwrap();

    assert_eq!(report.rejected_landmarks, 1);
    assert_eq!(workspace.landmarks().active_count(), 0);
    assert_eq!(
        workspace.landmarks().as_slice()[0].status,
        LandmarkStatus::Rejected
    );
    assert!(workspace.components()[0].landmarks.is_empty());
}

#[test]
fn adjust_and_clean_stops_after_all_landmarks_are_rejected() {
    let mut workspace = bundle_workspace();
    let backend = RejectOnceBackend::default();

    let report = bundle_adjust_and_clean(
        &mut workspace,
        &backend,
        &openmvg_sfm::unstable::BundleAdjustConfig::default(),
        f64::INFINITY,
        0.0,
        2,
    )
    .unwrap();

    assert_eq!(report.total_removed_landmarks, 1);
    assert_eq!(report.total_removed_observations, 0);
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    assert_eq!(workspace.landmarks().active_count(), 0);
}

#[test]
fn nested_observability_does_not_change_flat_commit_semantics() {
    let report = BundleAdjustmentReport {
        backend_name: "typed-observability-test".to_owned(),
        initial_objective: 10.0,
        final_objective: 9.0,
        initial_rmse: 2.0,
        final_rmse: 1.0,
        residual_units: ResidualUnit::Pixels,
        iteration_count: 1,
        observation_count: 10,
        pose_count: 3,
        point_count: 4,
        intrinsic_group_count: 1,
        status: BaSolveStatus::Converged(BaConvergenceReason::Objective),
        diagnostics: BaDiagnostics {
            accepted_step_count: Some(1),
            rank: BaRankAssessment::FullRank,
            weak_point_count: Some(0),
            observability: Some(BaObservabilityDiagnostics {
                pose_rank: 4,
                pose_dimension: 6,
                weak_point_count: Some(2),
            }),
            ..BaDiagnostics::default()
        },
        termination_reason: "Converged(Objective)".to_owned(),
    };

    assert!(
        report
            .diagnostics
            .observability
            .as_ref()
            .unwrap()
            .pose_is_rank_deficient()
    );
}
use std::sync::atomic::{AtomicUsize, Ordering};
