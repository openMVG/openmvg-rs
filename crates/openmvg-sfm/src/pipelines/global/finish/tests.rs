// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

mod post_fixed_rotation_cleanup_tests {
    use super::super::report::{
        GlobalSelfCalibrationRejection, GlobalSelfCalibrationStatus, GlobalTrappedViewDiagnostic,
    };
    use super::super::{
        FixedRotationPreparationReport, GlobalBaIterationLimits, GlobalFinishReport,
        GlobalFinishTransaction, GlobalPostFixedRotationCleanupConfig,
        GlobalPostFixedRotationCleanupRejection, GlobalPostFixedRotationCleanupStatus,
        GraphConsistencyStats, allowed_post_cleanup_views,
        certify_post_cleanup_topology_from_graph, finish_global_ba_retriangulate_inner,
        finish_global_ba_retriangulate_prepared_with_events, fixed_rotation_bundle_config,
        prepare_fixed_rotation_core_structure, prepare_fixed_rotation_structure,
        prepared_domain_signature, reject_under_supported_landmarks_on_deferred_tracks,
        run_post_fixed_rotation_cleanup, run_self_calibration,
    };
    use super::super::{
        fixed_rotation::defer_prefit_trapped_views, pose_trust::trapped_view_diagnostic,
    };
    use crate::ba::backend::{
        BaBackendFailure, BaBackendResult, BaBackendStatus, BaFailureKind, BaPreflightReport,
        BaPreflightResult, BundleAdjustmentBackend,
    };
    use crate::backends::{
        PosedPixelObservation, TriangulatedLandmark, TriangulationVerdict, Triangulator,
    };
    use crate::config::GaugeScalePolicy;
    use crate::events::{
        EventEmitter, SfmEvent, SfmEventBranch, SfmEventPhase, SfmEventTransition,
    };
    use crate::model::{
        ImageMeta, Landmark, LandmarkStatus, Landmarks, ObservationSupport, PoseRecord,
    };
    use crate::ops::reconstruction::{refit_incremental_for_views, refresh_primary_component};
    use crate::two_view::{GeometricModel, InlierMask, ModelEvidence, TerminationReason};
    use crate::workspace::quality::ErrorStats;
    use crate::{
        Camera, FeatureId, FeatureSet, IntrinsicId, LandmarkId, PairId, RelativePose, Result,
        SfmWorkspace, Track, TrackId, TrackObservation, TrackSet, ViewId, WorldToCameraPose,
    };
    use nalgebra::{Matrix3, Point2, Point3, Rotation3, UnitVector3, Vector3};
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn finish_global_ba_retriangulate_report(
        workspace: &mut SfmWorkspace,
        ba: &dyn BundleAdjustmentBackend,
        robust_triangulator: &dyn Triangulator,
        max_reproj: f64,
        max_robust_rounds: usize,
    ) -> Result<Option<GlobalFinishReport>> {
        if workspace.pose_store().populated_count() < 2 {
            return Ok(None);
        }
        let transaction = GlobalFinishTransaction::capture(workspace);
        let result = prepare_fixed_rotation_structure(workspace).and_then(|removed| {
            finish_global_ba_retriangulate_inner(
                workspace,
                ba,
                robust_triangulator,
                max_reproj,
                max_robust_rounds,
                GlobalBaIterationLimits::default(),
                removed,
                None,
                SfmEventBranch::Main,
                &mut EventEmitter::disabled(),
            )
        });
        match result {
            Ok(report) if report.accepted => Ok(Some(report)),
            Ok(report) => {
                transaction.rollback(workspace);
                Ok(Some(report))
            }
            Err(error) => {
                transaction.rollback(workspace);
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_global_ba_retriangulate_prepared_report(
        workspace: &mut SfmWorkspace,
        ba: &dyn BundleAdjustmentBackend,
        robust_triangulator: &dyn Triangulator,
        max_reproj: f64,
        max_robust_rounds: usize,
        iteration_limits: GlobalBaIterationLimits,
        preparation: FixedRotationPreparationReport,
        post_fixed_rotation_cleanup: Option<&GlobalPostFixedRotationCleanupConfig>,
    ) -> Result<Option<GlobalFinishReport>> {
        let mut events = EventEmitter::disabled();
        finish_global_ba_retriangulate_prepared_with_events(
            workspace,
            ba,
            robust_triangulator,
            max_reproj,
            max_robust_rounds,
            iteration_limits,
            preparation,
            post_fixed_rotation_cleanup,
            SfmEventBranch::Main,
            &mut events,
        )
    }

    struct AcceptingTriangulator;

    impl Triangulator for AcceptingTriangulator {
        fn triangulate(
            &self,
            observations: &[PosedPixelObservation<'_>],
        ) -> Result<TriangulationVerdict> {
            Ok(TriangulatedLandmark {
                point: Point3::new(0.0, 0.0, 5.0),
                reprojection_error: openmvg_core::PixelResidual::ZERO,
                supporting_observation_indices: (0..observations.len())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            }
            .into())
        }
    }

    #[derive(Clone, Copy)]
    enum ExactBackendFailure {
        Never,
        Preflight,
        Adjust,
    }

    struct ExactCleanupBackend {
        calls: AtomicUsize,
        failure: ExactBackendFailure,
        observability: openmvg_ba::Observability,
    }

    impl ExactCleanupBackend {
        fn successful() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                failure: ExactBackendFailure::Never,
                observability: openmvg_ba::Observability::OnlyExpectedGauge,
            }
        }

        /// The sparse exact certificate reports `FullRank` on the gauge-reduced layout.
        fn full_rank() -> Self {
            Self {
                observability: openmvg_ba::Observability::FullRank,
                ..Self::successful()
            }
        }

        fn failing(failure: ExactBackendFailure) -> Self {
            Self {
                failure,
                ..Self::successful()
            }
        }
    }

    impl BundleAdjustmentBackend for ExactCleanupBackend {
        fn name(&self) -> &'static str {
            "ExactCleanupBackend"
        }

        fn preflight(
            &self,
            problem: &openmvg_ba::BundleProblem,
            _config: &crate::ba::config::BundleAdjustConfig,
        ) -> BaPreflightResult {
            if matches!(self.failure, ExactBackendFailure::Preflight) {
                return Err(BaBackendFailure::new(
                    "ExactCleanupBackend",
                    BaFailureKind::Numerical,
                    "injected preflight failure",
                ));
            }
            let pose_dimension = problem.poses().len().saturating_mul(3).saturating_sub(4);
            Ok(BaPreflightReport {
                decision: crate::ba::backend::BaPreflightDecision::Proceed,
                workspace_budget_bytes: Some(2),
                pose_dimension: Some(pose_dimension),
                // Model the bounded probe conservatively missing one otherwise observable
                // camera. The exact rerun below still certifies the complete retained system.
                covered_free_pose_count: Some(problem.poses().len().saturating_sub(2)),
                weak_point_count: Some(0),
                dense_observability_escalation: true,
            })
        }

        fn adjust(
            &self,
            problem: &openmvg_ba::BundleProblem,
            _config: &crate::ba::config::BundleAdjustConfig,
        ) -> BaBackendResult {
            if matches!(self.failure, ExactBackendFailure::Adjust) {
                return Err(BaBackendFailure::new(
                    "ExactCleanupBackend",
                    BaFailureKind::Numerical,
                    "injected solve failure",
                ));
            }
            self.propose(
                problem,
                problem
                    .intrinsics()
                    .iter()
                    .map(|intrinsic| intrinsic.intrinsics().clone())
                    .collect(),
            )
        }
    }

    impl ExactCleanupBackend {
        /// Proposes the problem's own poses and points with the given intrinsics.
        fn propose(
            &self,
            problem: &openmvg_ba::BundleProblem,
            intrinsics: Vec<openmvg_core::model::CameraIntrinsics>,
        ) -> BaBackendResult {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let pose_dimension = problem.poses().len().saturating_mul(3).saturating_sub(4);
            let details =
                openmvg_ba::ObservabilityDiagnostics::try_new(pose_dimension, pose_dimension, 0)
                    .unwrap();
            let diagnostics = openmvg_ba::SolveDiagnostics::new(
                0,
                0,
                openmvg_ba::StationarityAudit::try_from_fresh_zero_damping(0.0).unwrap(),
                self.observability,
                0,
                0,
            )
            .with_observability_details(details);
            let solution = crate::testing::bundle_solution_with_diagnostics(
                problem,
                problem
                    .poses()
                    .iter()
                    .map(|camera| camera.pose().clone())
                    .collect(),
                intrinsics,
                problem
                    .points()
                    .iter()
                    .map(|point| *point.position())
                    .collect(),
                openmvg_ba::SolveStatus::Skipped,
                0.0,
                0.0,
                0,
                diagnostics,
            );
            let _ = call;
            Ok(BaBackendStatus::Proposed(Box::new(solution)))
        }
    }

    #[test]
    fn cleanup_cap_uses_the_smaller_absolute_and_fractional_limit() {
        let config = GlobalPostFixedRotationCleanupConfig {
            enabled: true,
            max_views: 4,
            max_fraction: 0.005,
        };
        assert_eq!(allowed_post_cleanup_views(&config, 64), 1);
        assert_eq!(allowed_post_cleanup_views(&config, 1_000), 4);
        assert_eq!(allowed_post_cleanup_views(&config, 10_000), 4);
        assert!(
            GlobalPostFixedRotationCleanupConfig {
                max_fraction: 0.0,
                ..config
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn topology_accepts_a_leaf_but_rejects_articulation_and_joint_disconnection() {
        let images = (0..4).map(ViewId::new).collect::<Vec<_>>();
        let triangle_with_leaf = vec![(0, 1), (1, 2), (2, 0), (1, 3)];
        assert!(
            certify_post_cleanup_topology_from_graph(
                &images,
                &triangle_with_leaf,
                &BTreeSet::from([ViewId::new(3)]),
            )
            .is_ok()
        );
        assert!(matches!(
            certify_post_cleanup_topology_from_graph(
                &images,
                &triangle_with_leaf,
                &BTreeSet::from([ViewId::new(1)]),
            ),
            Err(GlobalPostFixedRotationCleanupRejection::ContainsArticulationViews { .. })
        ));

        let cycle = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        assert!(matches!(
            certify_post_cleanup_topology_from_graph(
                &images,
                &cycle,
                &BTreeSet::from([ViewId::new(1), ViewId::new(3)]),
            ),
            Err(GlobalPostFixedRotationCleanupRejection::RetainedGraphDisconnected { .. })
        ));
    }

    #[test]
    fn three_low_degree_non_articulation_translation_traps_are_a_bounded_defer_set() {
        let images = (0..6).map(ViewId::new).collect::<Vec<_>>();
        let core_with_redundant_tail_attachments = vec![
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 0),
            (3, 1),
            (4, 1),
            (4, 2),
            (5, 2),
            (5, 0),
        ];
        let candidates = [ViewId::new(3), ViewId::new(4), ViewId::new(5)];

        for image in candidates {
            let diagnostic = trapped_view_diagnostic(image, 2, vec![0.0; 2], vec![90.0; 2])
                .expect("a coherent 90-degree translation error must be typed as a trap");
            assert_eq!(diagnostic.active_pair_degree, 2);
            assert_eq!(diagnostic.median_rotation_residual_deg, Some(0.0));
            assert_eq!(diagnostic.median_translation_residual_deg, Some(90.0));
        }

        let candidate_set = candidates.into_iter().collect::<BTreeSet<_>>();
        assert!(
            certify_post_cleanup_topology_from_graph(
                &images,
                &core_with_redundant_tail_attachments,
                &candidate_set,
            )
            .is_ok()
        );
        assert!(
            candidate_set.len()
                <= allowed_post_cleanup_views(
                    &GlobalPostFixedRotationCleanupConfig {
                        enabled: true,
                        max_views: 4,
                        max_fraction: 0.005,
                    },
                    2_307,
                )
        );
    }

    #[test]
    fn scoped_cleanup_preserves_domains_and_ids_and_rolls_back_exactly() {
        let tracks = TrackSet::try_new(vec![
            test_track(0, &[0, 1, 2]),
            test_track(1, &[1, 2]),
            test_track(2, &[0, 1, 2]),
        ])
        .unwrap();
        let mut workspace = SfmWorkspace::builder()
            .image_meta(
                (0..3)
                    .map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48))
                    .collect::<Vec<_>>(),
            )
            .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .imported_pairs(vec![
                (ViewId::new(0), ViewId::new(1)),
                (ViewId::new(1), ViewId::new(2)),
            ])
            .tracks(
                tracks,
                (0..3).map(|image| {
                    FeatureSet::try_new(
                        (0..3)
                            .map(|track| Point2::new(track as f64 + image as f64, track as f64))
                            .collect(),
                    )
                    .unwrap()
                }),
            )
            .build()
            .unwrap();
        for image in 0..3 {
            workspace
                .set_pose(PoseRecord {
                    id: crate::PoseId::new(image),
                    pose: WorldToCameraPose::from_rotation_center(
                        Rotation3::identity(),
                        Point3::new(image as f64, 0.0, 0.0),
                    )
                    .unwrap(),
                    component: None,
                })
                .unwrap();
        }
        workspace.landmarks = Landmarks::try_new(vec![
            Landmark::active(
                LandmarkId::new(0),
                TrackId::new(0),
                Point3::new(0.0, 0.0, 5.0),
                openmvg_core::PixelResidual::ZERO,
                ObservationSupport::try_from_indices([1, 2], 3).unwrap(),
            ),
            Landmark::active(
                LandmarkId::new(1),
                TrackId::new(1),
                Point3::new(0.0, 0.0, 5.0),
                openmvg_core::PixelResidual::ZERO,
                ObservationSupport::default(),
            ),
        ])
        .unwrap();
        workspace.components = vec![crate::model::ReconstructionComponent::new(
            crate::ComponentId::new(0),
            (0..3).map(ViewId::new).collect(),
            vec![LandmarkId::new(0), LandmarkId::new(1)],
            GaugeScalePolicy::Free,
        )];

        let poses_before = workspace.pose_store().clone();
        let landmarks_before = workspace.landmarks().clone();
        let components_before = workspace.components().to_vec();
        let tracks_before = workspace.tracks().clone();
        let pairs_before = workspace.pair_graph().pairs().to_vec();
        let active_pairs_before = workspace.pair_graph().active_pair_ids().collect::<Vec<_>>();
        let domains_before = prepared_domain_signature(&workspace);
        let transaction = crate::ops::rollback::GeometryRollback::capture(&workspace);

        workspace.poses.take(2).unwrap();
        assert_eq!(
            reject_under_supported_landmarks_on_deferred_tracks(&mut workspace, &[ViewId::new(2)],)
                .unwrap(),
            2
        );
        assert_eq!(
            workspace
                .landmarks()
                .get(LandmarkId::new(1))
                .unwrap()
                .status,
            LandmarkStatus::Rejected
        );
        let retriangulation = refit_incremental_for_views(
            &mut workspace,
            &AcceptingTriangulator,
            2,
            f64::INFINITY,
            &[ViewId::new(2)],
        )
        .unwrap();
        refresh_primary_component(&mut workspace);

        assert_eq!(retriangulation.new_landmarks, 1);
        assert_eq!(retriangulation.revived_landmarks, 0);
        assert_eq!(
            workspace
                .landmarks()
                .get(LandmarkId::new(0))
                .unwrap()
                .status,
            LandmarkStatus::Rejected
        );
        assert_eq!(
            workspace
                .landmarks()
                .get(LandmarkId::new(0))
                .unwrap()
                .source_track,
            TrackId::new(0)
        );
        assert_eq!(
            workspace
                .landmarks()
                .get(LandmarkId::new(1))
                .unwrap()
                .source_track,
            TrackId::new(1)
        );
        assert_eq!(
            workspace
                .landmarks()
                .get(LandmarkId::new(2))
                .unwrap()
                .source_track,
            TrackId::new(2)
        );
        assert_eq!(workspace.tracks(), &tracks_before);
        assert_eq!(workspace.pair_graph().pairs(), pairs_before);
        assert_eq!(
            workspace.pair_graph().active_pair_ids().collect::<Vec<_>>(),
            active_pairs_before
        );
        let domains_after = super::super::prepared_domain_signature_with_landmark_prefix(
            &workspace,
            domains_before.landmark_prefix_count,
        );
        assert_eq!(domains_after.pair_digest, domains_before.pair_digest);
        assert_eq!(domains_after.track_digest, domains_before.track_digest);
        assert_eq!(
            domains_after.landmark_prefix_digest,
            domains_before.landmark_prefix_digest
        );

        transaction.rollback(&mut workspace);
        assert_eq!(workspace.pose_store(), &poses_before);
        assert_eq!(workspace.landmarks(), &landmarks_before);
        assert_eq!(workspace.components(), components_before);
        assert_eq!(workspace.tracks(), &tracks_before);
        assert_eq!(workspace.pair_graph().pairs(), pairs_before);
    }

    #[test]
    fn integrated_cleanup_applies_and_replaces_the_retained_fixed_rotation_receipt() {
        for backend in [
            ExactCleanupBackend::successful(),
            ExactCleanupBackend::full_rank(),
        ] {
            integrated_cleanup_applies_with(&backend);
        }
    }

    fn integrated_cleanup_applies_with(backend: &ExactCleanupBackend) {
        let mut workspace = cleanup_workspace();
        let config = fixed_rotation_bundle_config();
        let mut translation = workspace.bundle_adjust(backend, &config).unwrap();
        assert_eq!(translation.ba.backend_name, "ExactCleanupBackend");
        let pairs_before = workspace.pair_graph().pairs().to_vec();
        let tracks_before = workspace.tracks().clone();
        let domains_before = prepared_domain_signature(&workspace);

        let report = run_post_fixed_rotation_cleanup(
            &mut workspace,
            backend,
            &AcceptingTriangulator,
            f64::INFINITY,
            &config,
            &GraphConsistencyStats::default(),
            &graph_with_trapped_leaf(),
            &mut translation,
            Some(&GlobalPostFixedRotationCleanupConfig {
                enabled: true,
                max_views: 4,
                max_fraction: 1.0,
            }),
        );

        let GlobalPostFixedRotationCleanupStatus::Applied {
            deferred_views,
            original_translation_and_points,
            ..
        } = report.outcome
        else {
            panic!("cleanup should apply: {report:?}");
        };
        assert_eq!(deferred_views, vec![ViewId::new(3)]);
        assert_eq!(
            original_translation_and_points.ba.backend_name,
            "ExactCleanupBackend"
        );
        assert_eq!(translation.ba.backend_name, "ExactCleanupBackend");
        assert!(!workspace.pose_store().has(3));
        assert_eq!(workspace.pair_graph().pairs(), pairs_before);
        assert_eq!(workspace.tracks(), &tracks_before);
        let domains_after = super::super::prepared_domain_signature_with_landmark_prefix(
            &workspace,
            domains_before.landmark_prefix_count,
        );
        assert_eq!(domains_after.pair_digest, domains_before.pair_digest);
        assert_eq!(domains_after.track_digest, domains_before.track_digest);
        assert_eq!(
            domains_after.landmark_prefix_digest,
            domains_before.landmark_prefix_digest
        );
    }

    #[test]
    fn integrated_cleanup_rolls_back_preflight_and_solve_failures_exactly() {
        for failure in [ExactBackendFailure::Preflight, ExactBackendFailure::Adjust] {
            let mut workspace = cleanup_workspace();
            let config = fixed_rotation_bundle_config();
            let successful = ExactCleanupBackend::successful();
            let mut translation = workspace.bundle_adjust(&successful, &config).unwrap();
            let poses_before = workspace.pose_store().clone();
            let landmarks_before = workspace.landmarks().clone();
            let components_before = workspace.components().to_vec();
            let reports_before = workspace.reports().clone();
            let backend = ExactCleanupBackend::failing(failure);

            let report = run_post_fixed_rotation_cleanup(
                &mut workspace,
                &backend,
                &AcceptingTriangulator,
                f64::INFINITY,
                &config,
                &GraphConsistencyStats::default(),
                &graph_with_trapped_leaf(),
                &mut translation,
                Some(&GlobalPostFixedRotationCleanupConfig {
                    enabled: true,
                    max_views: 4,
                    max_fraction: 1.0,
                }),
            );

            assert!(matches!(
                report.outcome,
                GlobalPostFixedRotationCleanupStatus::Rejected { .. }
            ));
            assert_eq!(workspace.pose_store(), &poses_before);
            assert_eq!(workspace.landmarks(), &landmarks_before);
            assert_eq!(workspace.components(), components_before);
            assert_eq!(
                serde_json::to_value(workspace.reports()).unwrap(),
                serde_json::to_value(&reports_before).unwrap()
            );
            assert_eq!(translation.ba.backend_name, "ExactCleanupBackend");
        }
    }

    #[test]
    fn accepted_prepared_candidate_finish_matches_public_transactional_finish() {
        let mut transactional = cleanup_workspace();
        let mut candidate_owned = transactional.clone();
        let candidate_owned_removed =
            prepare_fixed_rotation_structure(&mut candidate_owned).unwrap();

        let transactional_report = finish_global_ba_retriangulate_report(
            &mut transactional,
            &ExactCleanupBackend::successful(),
            &AcceptingTriangulator,
            4.0,
            0,
        )
        .unwrap()
        .expect("the prepared reconstruction is large enough to finish");
        let candidate_owned_report = finish_global_ba_retriangulate_prepared_report(
            &mut candidate_owned,
            &ExactCleanupBackend::successful(),
            &AcceptingTriangulator,
            4.0,
            0,
            GlobalBaIterationLimits::default(),
            candidate_owned_removed,
            None,
        )
        .unwrap()
        .expect("the owned candidate is large enough to finish");

        assert!(transactional_report.accepted);
        assert_eq!(transactional_report, candidate_owned_report);
        assert_eq!(
            transactional.camera_models(),
            candidate_owned.camera_models()
        );
        assert_eq!(transactional.pose_store(), candidate_owned.pose_store());
        assert_eq!(transactional.landmarks(), candidate_owned.landmarks());
        assert_eq!(transactional.components(), candidate_owned.components());
        assert_eq!(
            serde_json::to_value(transactional.reports()).unwrap(),
            serde_json::to_value(candidate_owned.reports()).unwrap()
        );
    }

    #[test]
    fn observed_prepared_finish_preserves_result_and_orders_candidate_events() {
        let mut unobserved = cleanup_workspace();
        let mut observed = unobserved.clone();
        let unobserved_removed = prepare_fixed_rotation_structure(&mut unobserved).unwrap();
        let observed_removed = prepare_fixed_rotation_structure(&mut observed).unwrap();
        let expected = finish_global_ba_retriangulate_prepared_report(
            &mut unobserved,
            &ExactCleanupBackend::successful(),
            &AcceptingTriangulator,
            4.0,
            0,
            GlobalBaIterationLimits::default(),
            unobserved_removed,
            None,
        )
        .unwrap();

        let branch = SfmEventBranch::GlobalCandidate {
            attempt: 1,
            candidate: crate::pipelines::global::GlobalCandidate::GlobalPositioning,
        };
        let mut recorded = Vec::new();
        let mut sink = |event: SfmEvent<'_>| {
            recorded.push((event.sequence, event.branch, event.phase, event.transition));
        };
        let actual = {
            let mut events = EventEmitter::observing(&mut sink);
            finish_global_ba_retriangulate_prepared_with_events(
                &mut observed,
                &ExactCleanupBackend::successful(),
                &AcceptingTriangulator,
                4.0,
                0,
                GlobalBaIterationLimits::default(),
                observed_removed,
                None,
                branch,
                &mut events,
            )
            .unwrap()
        };

        assert_eq!(actual, expected);
        assert_eq!(observed.pose_store(), unobserved.pose_store());
        assert_eq!(observed.landmarks(), unobserved.landmarks());
        assert_eq!(observed.components(), unobserved.components());
        assert_eq!(
            recorded,
            vec![
                (
                    0,
                    branch,
                    SfmEventPhase::GlobalFixedRotation,
                    SfmEventTransition::Tentative,
                ),
                (
                    1,
                    branch,
                    SfmEventPhase::GlobalPostFixedRotationCleanup,
                    SfmEventTransition::Checkpoint,
                ),
                (
                    2,
                    branch,
                    SfmEventPhase::GlobalFixedRotation,
                    SfmEventTransition::Committed,
                ),
                (
                    3,
                    branch,
                    SfmEventPhase::GlobalBundleAdjustment,
                    SfmEventTransition::Tentative,
                ),
                (
                    4,
                    branch,
                    SfmEventPhase::GlobalBundleAdjustment,
                    SfmEventTransition::Committed,
                ),
                (
                    5,
                    branch,
                    SfmEventPhase::GlobalCandidateAdmission,
                    SfmEventTransition::Tentative,
                ),
            ]
        );
    }

    /// Moves the focal on every self-calibration solve, as a real refit would, and can fail the
    /// refit after the first solve has already changed the intrinsics. Every earlier stage matches
    /// the exact backend.
    struct FocalMovingBackend {
        inner: ExactCleanupBackend,
        self_calibration_solves: AtomicUsize,
        fail_refit: bool,
    }

    impl FocalMovingBackend {
        fn new(fail_refit: bool) -> Self {
            Self {
                inner: ExactCleanupBackend::successful(),
                self_calibration_solves: AtomicUsize::new(0),
                fail_refit,
            }
        }
    }

    impl BundleAdjustmentBackend for FocalMovingBackend {
        fn name(&self) -> &'static str {
            self.inner.name()
        }

        fn preflight(
            &self,
            problem: &openmvg_ba::BundleProblem,
            config: &crate::ba::config::BundleAdjustConfig,
        ) -> BaPreflightResult {
            self.inner.preflight(problem, config)
        }

        fn adjust(
            &self,
            problem: &openmvg_ba::BundleProblem,
            config: &crate::ba::config::BundleAdjustConfig,
        ) -> BaBackendResult {
            if !matches!(
                config.scope(),
                crate::ba::config::OptimizationScope::SelfCalibration
            ) {
                return self.inner.adjust(problem, config);
            }
            let solve = self.self_calibration_solves.fetch_add(1, Ordering::SeqCst);
            if self.fail_refit && solve > 0 {
                return Err(BaBackendFailure::new(
                    "FocalMovingBackend",
                    BaFailureKind::Numerical,
                    "injected self-calibration failure",
                ));
            }
            let intrinsics = problem
                .intrinsics()
                .iter()
                .map(|intrinsic| {
                    use openmvg_core::camera::CameraModel;
                    let data = intrinsic.intrinsics().clone();
                    let (fx, fy) = data.camera().focal_lengths().unwrap();
                    let center = data.camera().principal_point();
                    data.try_with_camera(
                        Camera::try_pinhole(1.05 * fx, 1.05 * fy, center.x, center.y).unwrap(),
                    )
                    .unwrap()
                })
                .collect();
            self.inner.propose(problem, intrinsics)
        }
    }

    type RecordedEvent = (u64, SfmEventBranch, SfmEventPhase, SfmEventTransition);

    /// The cleanup fixture after an accepted finish, as admission hands it to self-calibration.
    fn finished_cleanup_workspace() -> SfmWorkspace {
        let mut workspace = cleanup_workspace();
        let removed = prepare_fixed_rotation_structure(&mut workspace).unwrap();
        let report = finish_global_ba_retriangulate_prepared_with_events(
            &mut workspace,
            &ExactCleanupBackend::successful(),
            &AcceptingTriangulator,
            4.0,
            0,
            GlobalBaIterationLimits::default(),
            removed,
            None,
            SfmEventBranch::Main,
            &mut EventEmitter::disabled(),
        )
        .unwrap()
        .expect("the cleanup fixture has enough poses to finish");
        assert!(report.accepted);
        assert!(report.self_calibration.is_none());
        workspace
    }

    fn self_calibrate(
        workspace: &mut SfmWorkspace,
        ba: &dyn BundleAdjustmentBackend,
    ) -> (Result<GlobalSelfCalibrationStatus>, Vec<SfmEventTransition>) {
        let mut recorded: Vec<RecordedEvent> = Vec::new();
        let mut sink = |event: SfmEvent<'_>| {
            recorded.push((event.sequence, event.branch, event.phase, event.transition));
        };
        let result = {
            let mut events = EventEmitter::observing(&mut sink);
            run_self_calibration(
                workspace,
                ba,
                &AcceptingTriangulator,
                4.0,
                GlobalBaIterationLimits::default(),
                SfmEventBranch::Main,
                &mut events,
            )
        };
        assert!(
            recorded
                .iter()
                .all(|(_, _, phase, _)| *phase == SfmEventPhase::GlobalSelfCalibration)
        );
        let transitions = recorded
            .into_iter()
            .map(|(_, _, _, transition)| transition)
            .collect();
        (result, transitions)
    }

    fn assert_same_reconstruction(actual: &SfmWorkspace, expected: &SfmWorkspace) {
        assert_eq!(actual.pose_store(), expected.pose_store());
        assert_eq!(actual.landmarks(), expected.landmarks());
        assert_eq!(actual.components(), expected.components());
        assert_eq!(actual.camera_models(), expected.camera_models());
    }

    #[test]
    fn self_calibration_commits_on_an_admitted_reconstruction() {
        let mut workspace = finished_cleanup_workspace();
        let admitted_models = workspace.camera_models().to_vec();
        let (result, transitions) = self_calibrate(&mut workspace, &FocalMovingBackend::new(false));

        assert!(matches!(
            result.unwrap(),
            GlobalSelfCalibrationStatus::Committed(_)
        ));
        assert_ne!(workspace.camera_models(), admitted_models.as_slice());
        assert_eq!(
            transitions,
            vec![SfmEventTransition::Tentative, SfmEventTransition::Committed]
        );
    }

    #[test]
    fn failed_self_calibration_restores_the_admitted_reconstruction_exactly() {
        let admitted = finished_cleanup_workspace();
        let mut rolled_back = admitted.clone();
        let (result, transitions) =
            self_calibrate(&mut rolled_back, &FocalMovingBackend::new(true));

        assert!(matches!(
            result.unwrap(),
            GlobalSelfCalibrationStatus::RolledBack(GlobalSelfCalibrationRejection::Refinement {
                error
            }) if error.contains("injected self-calibration failure")
        ));
        assert_eq!(transitions, vec![SfmEventTransition::RolledBack]);
        assert_same_reconstruction(&rolled_back, &admitted);
    }

    fn graph_with_trapped_leaf() -> GraphConsistencyStats {
        GraphConsistencyStats {
            locally_trapped_cameras: 1,
            trapped_views: vec![GlobalTrappedViewDiagnostic {
                view: ViewId::new(3),
                active_pair_degree: 1,
                rotation_residual_support: 1,
                translation_residual_support: 1,
                median_rotation_residual_deg: Some(0.0),
                median_translation_residual_deg: Some(20.0),
            }],
            ..GraphConsistencyStats::default()
        }
    }

    fn cleanup_workspace() -> SfmWorkspace {
        cleanup_workspace_with_tracks([&[0, 1, 2, 3]; 4])
    }

    fn cleanup_workspace_with_tracks(track_views: [&[usize]; 4]) -> SfmWorkspace {
        let centers = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ];
        // View 3 has two edges so that a wrong pose for it is a strict incident majority.
        let pairs = [(0, 1), (1, 2), (2, 0), (1, 3), (2, 3)];
        let points = (0..4)
            .map(|index| Point3::new(-0.3 + index as f64 * 0.2, 0.1, 5.0))
            .collect::<Vec<_>>();
        let tracks = TrackSet::try_new(
            track_views
                .iter()
                .enumerate()
                .map(|(track, views)| test_track(track, views))
                .collect(),
        )
        .unwrap();
        let feature_sets = centers
            .iter()
            .map(|center| {
                FeatureSet::try_new(
                    points
                        .iter()
                        .map(|point| {
                            let camera_point = point.coords - center.coords;
                            Point2::new(
                                100.0 * camera_point.x / camera_point.z + 32.0,
                                100.0 * camera_point.y / camera_point.z + 24.0,
                            )
                        })
                        .collect(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut workspace = SfmWorkspace::builder()
            .image_meta(
                (0..4)
                    .map(|image| ImageMeta::external_id(format!("image-{image}"), 64, 48))
                    .collect::<Vec<_>>(),
            )
            .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .imported_pairs(
                pairs
                    .iter()
                    .map(|&(first, second)| (ViewId::new(first), ViewId::new(second))),
            )
            .tracks(tracks, feature_sets)
            .build()
            .unwrap();
        for (image, center) in centers.into_iter().enumerate() {
            workspace
                .set_pose(PoseRecord {
                    id: crate::PoseId::new(image),
                    pose: WorldToCameraPose::from_rotation_center(Rotation3::identity(), center)
                        .unwrap(),
                    component: Some(crate::ComponentId::new(0)),
                })
                .unwrap();
        }
        workspace.landmarks = Landmarks::try_new(
            points
                .into_iter()
                .enumerate()
                .map(|(index, point)| {
                    Landmark::active(
                        LandmarkId::new(index),
                        TrackId::new(index),
                        point,
                        openmvg_core::PixelResidual::ZERO,
                        ObservationSupport::default(),
                    )
                })
                .collect(),
        )
        .unwrap();
        refresh_primary_component(&mut workspace);
        for pair_index in 0..workspace.pair_graph().len() {
            let pair_id = PairId::new(pair_index);
            let pair = workspace.pair_graph().pair(pair_id).unwrap();
            let direction = centers[pair.second().index()] - centers[pair.first().index()];
            workspace
                .geometric_models
                .set(
                    pair_index,
                    GeometricModel::Essential {
                        matrix: Matrix3::identity(),
                        relative_pose: RelativePose::from_baseline_direction(
                            Rotation3::identity(),
                            UnitVector3::new_normalize(Vector3::from(direction)),
                        )
                        .unwrap(),
                        evidence: ModelEvidence::try_new(
                            InlierMask::all_inliers(1),
                            ErrorStats::from_values(&[0.0], &[1.0]),
                            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
                            "cleanup-test",
                            1,
                            TerminationReason::Converged,
                            Vec::new(),
                        )
                        .unwrap(),
                    },
                )
                .unwrap();
        }
        workspace
    }

    #[test]
    fn incidence_core_prefit_defers_one_topology_safe_trapped_view() {
        let mut workspace = cleanup_workspace();
        workspace
            .set_pose_for_view(
                ViewId::new(3),
                WorldToCameraPose::from_rotation_center(
                    Rotation3::identity(),
                    Point3::new(-2.0, 0.0, 0.0),
                )
                .unwrap(),
                Some(crate::ComponentId::new(0)),
            )
            .unwrap();
        let before = super::super::graph_consistency_stats(&workspace);
        assert_eq!(
            before
                .trapped_views
                .iter()
                .map(|diagnostic| diagnostic.view)
                .collect::<Vec<_>>(),
            vec![ViewId::new(3)]
        );
        let report = defer_prefit_trapped_views(&mut workspace).unwrap();

        assert_eq!(report.rejected_landmarks, 0);
        assert_eq!(report.removed_views, vec![ViewId::new(3)]);
        assert_eq!(workspace.registered_view_count(), 3);
        assert!(!workspace.has_pose(ViewId::new(3)));
        assert_eq!(workspace.landmarks().active_count(), 4);
    }

    #[test]
    fn near_pure_rotation_edges_cannot_trap_a_camera() {
        let mut workspace = cleanup_workspace();
        workspace
            .set_pose_for_view(
                ViewId::new(3),
                WorldToCameraPose::from_rotation_center(
                    Rotation3::identity(),
                    Point3::new(-2.0, 0.0, 0.0),
                )
                .unwrap(),
                Some(crate::ComponentId::new(0)),
            )
            .unwrap();
        assert_eq!(
            super::super::graph_consistency_stats(&workspace).locally_trapped_cameras,
            1
        );

        // The same bearing in both views of a zoom burst from one spot: zero parallax leaves
        // the baseline direction of view 3's edges unobserved, so they are no evidence.
        let keypoint = openmvg_features::model::Keypoint::try_new(32.0, 24.0, 1.0, 0.0).unwrap();
        for image in 0..4 {
            let features = openmvg_features::model::ExtractedFeatures::try_new(
                vec![keypoint],
                openmvg_features::model::DescriptorSet::try_f32(vec![0.0], 1).unwrap(),
            )
            .unwrap();
            workspace.set_features(image, features).unwrap();
        }
        for second in [1, 2] {
            let pair_id = workspace
                .pair_graph()
                .pair_id(ViewId::new(second), ViewId::new(3))
                .unwrap();
            let pair = workspace.pair_graph().pair(pair_id).unwrap();
            let matches = crate::two_view::PairMatches::try_new(
                pair,
                vec![
                    openmvg_features::model::FeatureMatch::try_new(
                        FeatureId::new(0),
                        FeatureId::new(0),
                        None,
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
            workspace.matches.set(pair_id.index(), matches).unwrap();
        }
        let stats = super::super::graph_consistency_stats(&workspace);
        assert_eq!(stats.locally_trapped_cameras, 0);
        // Only the parallax-carrying triangle edges, which agree with the poses, remain.
        assert!(stats.translation_p90_deg.is_some_and(|p90| p90 < 1e-3));
    }

    #[test]
    fn incidence_core_is_selected_after_prefit_deferral() {
        let mut workspace = cleanup_workspace();
        workspace
            .set_pose_for_view(
                ViewId::new(3),
                WorldToCameraPose::from_rotation_center(
                    Rotation3::identity(),
                    Point3::new(-2.0, 0.0, 0.0),
                )
                .unwrap(),
                Some(crate::ComponentId::new(0)),
            )
            .unwrap();

        let preparation = prepare_fixed_rotation_core_structure(&mut workspace).unwrap();
        let repeated = crate::ops::reconstruction::retain_primary_bundle_component(&mut workspace)
            .expect("prepared incidence core should already be stable");

        assert!(preparation.removed_views.contains(&ViewId::new(3)));
        assert!(repeated.is_empty());
        assert_eq!(repeated.rejected_landmarks, 0);
    }

    /// Two dense lobes joined only through view 2, plus view 5 held by a single landmark.
    fn articulated_workspace() -> SfmWorkspace {
        let centers = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.5, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.0),
            Point3::new(4.5, 0.0, 0.0),
            Point3::new(0.25, 0.5, 0.0),
        ];
        let track_views: [&[usize]; 4] = [&[0, 1, 2, 5], &[0, 1, 2], &[2, 3, 4], &[2, 3, 4]];
        let points = [
            Point3::new(-0.2, 0.0, 5.0),
            Point3::new(0.3, 0.2, 6.0),
            Point3::new(4.1, -0.1, 5.5),
            Point3::new(4.4, 0.2, 6.5),
        ];
        let tracks = TrackSet::try_new(
            track_views
                .iter()
                .enumerate()
                .map(|(track, views)| test_track(track, views))
                .collect(),
        )
        .unwrap();
        let feature_sets = centers
            .iter()
            .map(|center| {
                FeatureSet::try_new(
                    points
                        .iter()
                        .map(|point| {
                            let camera_point = point.coords - center.coords;
                            Point2::new(
                                100.0 * camera_point.x / camera_point.z + 32.0,
                                100.0 * camera_point.y / camera_point.z + 24.0,
                            )
                        })
                        .collect(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut workspace = SfmWorkspace::builder()
            .image_meta(
                (0..centers.len())
                    .map(|image| ImageMeta::external_id(format!("image-{image}"), 64, 48))
                    .collect::<Vec<_>>(),
            )
            .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .tracks(tracks, feature_sets)
            .build()
            .unwrap();
        for (image, center) in centers.into_iter().enumerate() {
            workspace
                .set_pose(PoseRecord {
                    id: crate::PoseId::new(image),
                    pose: WorldToCameraPose::from_rotation_center(Rotation3::identity(), center)
                        .unwrap(),
                    component: Some(crate::ComponentId::new(0)),
                })
                .unwrap();
        }
        workspace.landmarks = Landmarks::try_new(
            points
                .into_iter()
                .enumerate()
                .map(|(index, point)| {
                    Landmark::active(
                        LandmarkId::new(index),
                        TrackId::new(index),
                        point,
                        openmvg_core::PixelResidual::ZERO,
                        ObservationSupport::default(),
                    )
                })
                .collect(),
        )
        .unwrap();
        refresh_primary_component(&mut workspace);
        workspace
    }

    #[test]
    fn fixed_rotation_structure_keeps_one_biconnected_block() {
        let mut workspace = articulated_workspace();

        let report = prepare_fixed_rotation_structure(&mut workspace).unwrap();
        let repeated = prepare_fixed_rotation_structure(&mut workspace).unwrap();

        assert_eq!(
            report.removed_views,
            vec![ViewId::new(3), ViewId::new(4), ViewId::new(5)]
        );
        assert_eq!(report.rejected_landmarks, 2);
        assert_eq!(workspace.registered_view_count(), 3);
        assert_eq!(workspace.landmarks().active_count(), 2);
        assert!(repeated.removed_views.is_empty());
        assert_eq!(repeated.rejected_landmarks, 0);
    }

    #[test]
    fn released_rotation_peels_a_camera_that_observes_only_its_translation() {
        // View 3 sees two landmarks: enough bearings for a translation, not for a full pose.
        let mut workspace =
            cleanup_workspace_with_tracks([&[0, 1, 2, 3], &[0, 1, 2, 3], &[0, 1, 2], &[0, 1, 2]]);
        let preparation = prepare_fixed_rotation_structure(&mut workspace).unwrap();
        assert!(preparation.removed_views.is_empty());

        let report = finish_global_ba_retriangulate_inner(
            &mut workspace,
            &ExactCleanupBackend::successful(),
            &AcceptingTriangulator,
            f64::INFINITY,
            0,
            GlobalBaIterationLimits::default(),
            preparation,
            None,
            SfmEventBranch::Main,
            &mut EventEmitter::disabled(),
        )
        .unwrap();

        assert_eq!(
            report.translation_and_points.ba.backend_name,
            "ExactCleanupBackend"
        );
        assert!(report.full_bundle_adjustment.is_some());
        assert!(!report.rejected_full_bundle_adjustment);
        assert_eq!(
            report.removed_views_before_full_bundle_adjustment,
            vec![ViewId::new(3)]
        );
        assert!(!workspace.has_pose(ViewId::new(3)));
        assert_eq!(workspace.registered_view_count(), 3);
        assert_eq!(workspace.landmarks().active_count(), 4);
    }

    fn test_track(track: usize, images: &[usize]) -> Track {
        Track::try_new(
            TrackId::new(track),
            images
                .iter()
                .map(|&image| TrackObservation {
                    view: ViewId::new(image),
                    feature: FeatureId::new(track),
                })
                .collect(),
        )
        .unwrap()
    }
}
