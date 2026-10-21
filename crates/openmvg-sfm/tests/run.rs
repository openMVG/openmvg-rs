// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

use nalgebra::{Matrix3, Point3, Rotation3, Vector3};
use openmvg_core::model::{
    EstimationError, EstimationResult, RelativeRotation, RelativeTranslation,
};
use openmvg_features::tracking::{Track, TrackSet as Tracks};
use openmvg_geometry::averaging::{
    RotationConfig, RotationEstimate, TranslationConfig, TranslationEstimate,
};
use openmvg_sfm::ba::BundleProblem as BaProblem;
use openmvg_sfm::progress::{SfmProgressSink, SfmProgressStage};
use openmvg_sfm::unstable::adapters::triangulation::DltTriangulator;
use openmvg_sfm::{
    events::*,
    model::*,
    reports::*,
    testing::seams::*,
    unstable::reports::*,
    unstable::*,
    unstable::{backends::*, global::*, types::*},
    *,
};

const CAMERA_COUNT: usize = 4;
/// Rotation-solver calls of one graph preparation: the adaptive conditioning's nine solves (five
/// graduated, three scale-refining, one final) plus the one prepared solve every candidate reuses.
/// The tests below assert this does not grow with the number of candidates tried.
const ROTATION_SOLVES_PER_PREPARATION: usize = 10;
const CORE_TRACK_COUNT: usize = 12;
const BRIDGE_TRACK_COUNT: usize = 2;
const FEATURE_COUNT: usize = CORE_TRACK_COUNT + BRIDGE_TRACK_COUNT;
const FOCAL: f64 = 100.0;
const CX: f64 = 32.0;
const CY: f64 = 24.0;

fn candidate_bootstrap(
    outcome: &GlobalCandidateStatus,
) -> Option<&openmvg_sfm::unstable::global::GlobalTranslationReport> {
    match outcome {
        GlobalCandidateStatus::Accepted { translation, .. }
        | GlobalCandidateStatus::Rejected { translation, .. } => Some(translation),
        GlobalCandidateStatus::Deferred { translation, .. }
        | GlobalCandidateStatus::Failed { translation, .. } => translation.as_deref(),
    }
}

fn point(track_index: usize) -> Point3<f64> {
    Point3::new(
        -0.6 + 0.12 * track_index as f64,
        -0.3 + 0.18 * (track_index % 4) as f64,
        4.0 + 0.25 * (track_index % 3) as f64,
    )
}

fn observation_xy(camera: usize, point: Point3<f64>) -> nalgebra::Point2<f64> {
    let camera_point = point.coords - Vector3::new(camera as f64, 0.0, 0.0);
    nalgebra::Point2::new(
        FOCAL * camera_point.x / camera_point.z + CX,
        FOCAL * camera_point.y / camera_point.z + CY,
    )
}

fn prepared_workspace() -> SfmWorkspace {
    let tracks = Tracks::try_new(
        (0..CORE_TRACK_COUNT)
            .map(|track_index| {
                Track::try_new(
                    TrackId::new(track_index),
                    (0..CAMERA_COUNT)
                        .map(|camera| TrackObservation {
                            view: ViewId::new(camera),
                            feature: FeatureId::new(track_index),
                        })
                        .collect(),
                )
            })
            .chain((CORE_TRACK_COUNT..FEATURE_COUNT).map(|track_index| {
                Track::try_new(
                    TrackId::new(track_index),
                    (2..CAMERA_COUNT)
                        .map(|camera| TrackObservation {
                            view: ViewId::new(camera),
                            feature: FeatureId::new(track_index),
                        })
                        .collect(),
                )
            }))
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("synthetic tracks are valid"),
    )
    .expect("track identifiers are unique");
    let feature_sets = (0..CAMERA_COUNT).map(|camera| {
        FeatureSet::try_new(
            (0..FEATURE_COUNT)
                .map(|track_index| observation_xy(camera, point(track_index)))
                .collect(),
        )
        .unwrap()
    });
    let mut workspace = SfmWorkspace::builder()
        .image_meta(
            (0..CAMERA_COUNT).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)),
        )
        .camera_models([Camera::try_pinhole(FOCAL, FOCAL, CX, CY).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs([
            (ViewId::new(0), ViewId::new(1)),
            (ViewId::new(1), ViewId::new(2)),
            (ViewId::new(2), ViewId::new(3)),
        ])
        .tracks(tracks, feature_sets)
        .build()
        .expect("synthetic workspace is valid");
    workspace
        .extract_features(&SyntheticExtractor, ViewSelector::All)
        .expect("feature staging succeeds")
        .match_pairs(&SyntheticMatcher, PairSelector::Active)
        .expect("matching succeeds")
        .verify_pairs(&SyntheticVerifier, PairSelector::Active)
        .expect("verification succeeds");
    workspace
}

fn noncontiguous_prepared_workspace() -> SfmWorkspace {
    let retained = [0usize, 2, 4];
    let tracks = Tracks::try_new(
        (0..CORE_TRACK_COUNT)
            .map(|track_index| {
                Track::try_new(
                    TrackId::new(track_index),
                    retained
                        .into_iter()
                        .map(|camera| TrackObservation {
                            view: ViewId::new(camera),
                            feature: FeatureId::new(track_index),
                        })
                        .collect(),
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("non-contiguous tracks are valid"),
    )
    .expect("non-contiguous track identifiers are unique");
    let feature_sets = (0..5).map(|camera| {
        FeatureSet::try_new(
            (0..FEATURE_COUNT)
                .map(|track_index| observation_xy(camera, point(track_index)))
                .collect(),
        )
        .unwrap()
    });
    let mut workspace = SfmWorkspace::builder()
        .image_meta(
            (0..5).map(|index| ImageMeta::external_id(format!("noncontiguous-{index}"), 64, 48)),
        )
        .camera_models([Camera::try_pinhole(FOCAL, FOCAL, CX, CY).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs([
            (ViewId::new(0), ViewId::new(2)),
            (ViewId::new(2), ViewId::new(4)),
        ])
        .tracks(tracks, feature_sets)
        .build()
        .expect("non-contiguous workspace is valid");
    workspace
        .extract_features(&SyntheticExtractor, ViewSelector::All)
        .expect("feature staging succeeds")
        .match_pairs(&SyntheticMatcher, PairSelector::Active)
        .expect("matching succeeds")
        .verify_pairs(&SyntheticVerifier, PairSelector::Active)
        .expect("verification succeeds");
    workspace
}

struct SyntheticExtractor;

impl FeatureExtractor for SyntheticExtractor {
    fn extract(&self, ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
        let keypoints = (0..FEATURE_COUNT)
            .map(|track_index| {
                let xy = observation_xy(ctx.view.index(), point(track_index));
                Keypoint::try_new(xy.x as f32, xy.y as f32, 1.0, 0.0)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(ExtractedFeatures::try_new(
            keypoints,
            DescriptorSet::try_f32(vec![0.0; FEATURE_COUNT], 1)?,
        )?)
    }
}

struct SyntheticMatcher;

impl PairMatcher for SyntheticMatcher {
    fn match_features(
        &self,
        ctx: &PairContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        let feature_range = if ctx.pair == ViewPair::new(ViewId::new(2), ViewId::new(3)) {
            CORE_TRACK_COUNT..FEATURE_COUNT
        } else {
            0..CORE_TRACK_COUNT
        };
        PairMatches::try_new(
            ctx.pair,
            feature_range
                .map(|index| {
                    FeatureMatch::try_new(FeatureId::new(index), FeatureId::new(index), Some(0.0))
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
    }
}

struct SyntheticVerifier;

impl GeometricVerifier for SyntheticVerifier {
    fn verify_pair(
        &self,
        ctx: &VerificationContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
        matches: &PairMatches,
        _camera_i: &openmvg_sfm::Camera,
        _camera_j: &openmvg_sfm::Camera,
    ) -> Result<VerificationVerdict> {
        let evidence = ModelEvidence::try_new(
            InlierMask::all_inliers(matches.len()),
            ErrorStats::from_values(&vec![0.0; matches.len()], &[1.0]),
            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
            "run-facade-test",
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

/// Exact rotations for every active endpoint. The partial core comes only from the configured
/// GPRBA certificate, not from a missing numerical estimate.
fn active_rotation_solver(
    rotations: &[RelativeRotation],
    _config: &RotationConfig,
) -> EstimationResult<RotationEstimate> {
    assert!(!rotations.is_empty());
    let cameras = rotations
        .iter()
        .flat_map(|rotation| [rotation.view_i(), rotation.view_j()])
        .collect::<BTreeSet<_>>();
    Ok(RotationEstimate::new(
        cameras
            .into_iter()
            .map(|image| (image, Rotation3::identity()))
            .collect(),
        1,
        openmvg_core::AngularResidual::ZERO,
    ))
}

fn failing_rotation_solver(
    _rotations: &[RelativeRotation],
    _config: &RotationConfig,
) -> EstimationResult<RotationEstimate> {
    Err(EstimationError::InvalidInput(
        "injected rotation failure".to_owned(),
    ))
}

fn linear_translation_solver(
    translations: &[RelativeTranslation],
    rotations: &BTreeMap<ViewId, Rotation3<f64>>,
    _config: &TranslationConfig,
) -> EstimationResult<TranslationEstimate> {
    assert!(!translations.is_empty());
    Ok(TranslationEstimate::new(
        rotations
            .keys()
            .map(|&view| (view, Vector3::new(view.index() as f64, 0.0, 0.0)))
            .collect(),
        1,
        openmvg_core::AngularResidual::ZERO,
    ))
}

static SCHEDULE_ROTATION_CALLS: AtomicUsize = AtomicUsize::new(0);
static SCHEDULE_TRANSLATION_CALLS: AtomicUsize = AtomicUsize::new(0);
static NONCONTIGUOUS_ROTATION_CALLS: AtomicUsize = AtomicUsize::new(0);
static NONCONTIGUOUS_TRANSLATION_CALLS: AtomicUsize = AtomicUsize::new(0);

fn schedule_counting_rotation_solver(
    rotations: &[RelativeRotation],
    config: &RotationConfig,
) -> EstimationResult<RotationEstimate> {
    SCHEDULE_ROTATION_CALLS.fetch_add(1, Ordering::SeqCst);
    active_rotation_solver(rotations, config)
}

fn schedule_counting_translation_solver(
    translations: &[RelativeTranslation],
    rotations: &BTreeMap<ViewId, Rotation3<f64>>,
    config: &TranslationConfig,
) -> EstimationResult<TranslationEstimate> {
    SCHEDULE_TRANSLATION_CALLS.fetch_add(1, Ordering::SeqCst);
    linear_translation_solver(translations, rotations, config)
}

fn noncontiguous_counting_rotation_solver(
    rotations: &[RelativeRotation],
    config: &RotationConfig,
) -> EstimationResult<RotationEstimate> {
    NONCONTIGUOUS_ROTATION_CALLS.fetch_add(1, Ordering::SeqCst);
    active_rotation_solver(rotations, config)
}

fn noncontiguous_counting_translation_solver(
    translations: &[RelativeTranslation],
    rotations: &BTreeMap<ViewId, Rotation3<f64>>,
    config: &TranslationConfig,
) -> EstimationResult<TranslationEstimate> {
    NONCONTIGUOUS_TRANSLATION_CALLS.fetch_add(1, Ordering::SeqCst);
    linear_translation_solver(translations, rotations, config)
}

#[derive(Default)]
struct LinearRegistrar {
    calls: AtomicUsize,
}

impl PoseRegistrar for LinearRegistrar {
    fn estimate_pose(
        &self,
        ctx: &PoseRegistrationContext,
        observations: &[PoseRegistrationObservation],
    ) -> Result<Option<PoseRegistrationEstimate>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Some(PoseRegistrationEstimate {
            pose: WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(ctx.view.index() as f64, 0.0, 0.0),
            )
            .unwrap(),
            inliers: observations.len(),
            residual_median: openmvg_core::AngularResidual::ZERO,
            method: "linear-test".to_owned(),
            focal_px: None,
        }))
    }
}

struct FailingRegistrar;

impl PoseRegistrar for FailingRegistrar {
    fn estimate_pose(
        &self,
        _ctx: &PoseRegistrationContext,
        _observations: &[PoseRegistrationObservation],
    ) -> Result<Option<PoseRegistrationEstimate>> {
        Err(openmvg_sfm::SfmError::InvalidInput(
            "injected registration failure".to_owned(),
        ))
    }
}

#[derive(Default)]
struct ObservableBackend {
    scopes: Mutex<Vec<OptimizationScope>>,
}

impl BundleAdjustmentBackend for ObservableBackend {
    fn adjust(&self, problem: &BaProblem, config: &BundleAdjustConfig) -> BaBackendResult {
        self.scopes
            .lock()
            .expect("scope mutex")
            .push(config.scope());
        observable_proposal(
            problem,
            config,
            problem
                .poses()
                .iter()
                .map(|camera| camera.pose().clone())
                .collect(),
            "ObservableTestBA",
        )
    }
}

fn observable_proposal(
    problem: &BaProblem,
    config: &BundleAdjustConfig,
    cameras: Vec<WorldToCameraPose>,
    _backend_name: &str,
) -> BaBackendResult {
    let camera_dimension = cameras.len().saturating_mul(3).saturating_sub(4);
    let rank = match problem.gauge() {
        openmvg_sfm::ba::Gauge::Free => openmvg_sfm::ba::Observability::OnlyExpectedGauge,
        _ => openmvg_sfm::ba::Observability::FullRank,
    };
    let mut diagnostics = openmvg_sfm::ba::SolveDiagnostics::new(
        0,
        0,
        openmvg_sfm::ba::StationarityAudit::try_from_fresh_zero_damping(0.0).unwrap(),
        rank,
        0,
        0,
    );
    if matches!(config.scope(), OptimizationScope::TranslationsAndPoints) {
        diagnostics = diagnostics.with_observability_details(
            openmvg_sfm::ba::ObservabilityDiagnostics::try_new(
                camera_dimension,
                camera_dimension,
                0,
            )
            .unwrap(),
        );
    }
    Ok(BaBackendStatus::Proposed(Box::new(
        openmvg_sfm::testing::bundle_solution_with_diagnostics(
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
            openmvg_sfm::ba::SolveStatus::Skipped,
            0.0,
            0.0,
            0,
            diagnostics,
        ),
    )))
}

#[derive(Default)]
struct FirstCandidateMutationBackend {
    fixed_rotation_calls: AtomicUsize,
    full_pose_calls: AtomicUsize,
    fixed_rotation_input_center_y: Mutex<Vec<f64>>,
    fail_first_full_pose: bool,
    inner: ObservableBackend,
}

impl FirstCandidateMutationBackend {
    fn failing_first_full_pose() -> Self {
        Self {
            fail_first_full_pose: true,
            ..Self::default()
        }
    }
}

impl BundleAdjustmentBackend for FirstCandidateMutationBackend {
    fn adjust(&self, problem: &BaProblem, config: &BundleAdjustConfig) -> BaBackendResult {
        if matches!(config.scope(), OptimizationScope::TranslationsAndPoints) {
            let input_center_y = problem
                .poses()
                .get(1)
                .expect("each global candidate contains camera 1")
                .pose()
                .center()
                .y;
            self.fixed_rotation_input_center_y
                .lock()
                .expect("fixed-rotation input mutex")
                .push(input_center_y);
            if self.fixed_rotation_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                let mut cameras = problem
                    .poses()
                    .iter()
                    .map(|camera| camera.pose().clone())
                    .collect::<Vec<_>>();
                let pose = &mut cameras[1];
                let mut center = *pose.center();
                center.y += 0.02;
                *pose = WorldToCameraPose::from_rotation_center(*pose.rotation(), center).unwrap();
                return observable_proposal(
                    problem,
                    config,
                    cameras,
                    "AdmittedDirectionMutationBA",
                );
            }
        }
        if matches!(config.scope(), OptimizationScope::PosesAndPoints) {
            let call = self.full_pose_calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_first_full_pose && call == 0 {
                return Err(BaBackendFailure::new(
                    "AdmittedDirectionMutationBA",
                    BaFailureKind::Internal,
                    "injected non-recoverable Direction finish failure",
                ));
            }
        }
        self.inner.adjust(problem, config)
    }
}

#[derive(Default)]
struct FixedRotationResourceFailure {
    fixed_rotation_calls: AtomicUsize,
    fallback: ObservableBackend,
}

impl BundleAdjustmentBackend for FixedRotationResourceFailure {
    fn adjust(&self, problem: &BaProblem, config: &BundleAdjustConfig) -> BaBackendResult {
        if matches!(config.scope(), OptimizationScope::TranslationsAndPoints) {
            self.fixed_rotation_calls.fetch_add(1, Ordering::SeqCst);
            return Err(BaBackendFailure::new(
                "ResourceLimitedTestBA",
                BaFailureKind::ResourceExhausted,
                "injected fixed-rotation capacity limit",
            ));
        }
        self.fallback.adjust(problem, config)
    }
}

#[derive(Default)]
struct FirstCandidateDeferredBackend {
    preflight_calls: AtomicUsize,
    inner: ObservableBackend,
}

impl BundleAdjustmentBackend for FirstCandidateDeferredBackend {
    fn preflight(&self, _problem: &BaProblem, _config: &BundleAdjustConfig) -> BaPreflightResult {
        let call = self.preflight_calls.fetch_add(1, Ordering::SeqCst);
        Ok(if call == 0 {
            BaPreflightReport::deferred(BaPreflightDeferralReason::InsufficientCameraCoverage)
        } else {
            BaPreflightReport::proceed()
        })
    }

    fn adjust(&self, problem: &BaProblem, config: &BundleAdjustConfig) -> BaBackendResult {
        self.inner.adjust(problem, config)
    }
}

fn assert_reconstruction_state_eq(actual: &SfmWorkspace, expected: &SfmWorkspace) {
    assert_eq!(actual.root_seed(), expected.root_seed());
    assert_eq!(actual.views(), expected.views());
    assert_eq!(actual.camera_models(), expected.camera_models());
    assert_eq!(actual.pair_graph(), expected.pair_graph());
    assert_eq!(actual.feature_store(), expected.feature_store());
    assert_eq!(actual.match_store(), expected.match_store());
    assert_eq!(
        actual.geometric_model_store(),
        expected.geometric_model_store()
    );
    assert_eq!(
        actual.is_matching_data_released(),
        expected.is_matching_data_released()
    );
    assert_eq!(actual.rotation_estimates(), expected.rotation_estimates());
    assert_eq!(actual.camera_centers(), expected.camera_centers());
    assert_eq!(actual.pose_store(), expected.pose_store());
    assert_eq!(actual.tracks(), expected.tracks());
    assert_eq!(actual.landmarks(), expected.landmarks());
    assert_eq!(actual.components(), expected.components());
    assert_eq!(
        actual.registration_failures(),
        expected.registration_failures()
    );
    assert_eq!(
        serde_json::to_value(actual.reports()).expect("actual reports serialize"),
        serde_json::to_value(expected.reports()).expect("expected reports serialize")
    );
}

#[test]
fn incremental_bundle_requires_only_incremental_backends() {
    let mut workspace = prepared_workspace();
    let triangulator = DltTriangulator::default();
    let registrar = LinearRegistrar::default();
    let ba = ObservableBackend::default();

    let config = SfmConfig::for_engine(SfmEngine::Incremental);
    let report = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&ba)
        .incremental_backends(IncrementalBackends::new(&registrar, &triangulator))
        .run_in_place(&mut workspace)
        .expect("incremental route succeeds");

    assert_eq!(report.schema_version(), RUN_REPORT_SCHEMA_VERSION);
    assert_eq!(report.final_route(), FinalRoute::Incremental);
    assert!(report.has_reconstruction());
    assert!(matches!(report.outcome(), SfmOutcome::Incremental { .. }));
    assert_eq!(workspace.pose_store().populated_count(), CAMERA_COUNT);
    assert!(!workspace.is_matching_data_released());
    assert_eq!(workspace.feature_store().populated_count(), CAMERA_COUNT);
    assert_eq!(workspace.match_store().populated_count(), CAMERA_COUNT - 1);
}

#[test]
fn runner_rejects_a_workspace_without_tracks() {
    let config = SfmConfig::for_engine(SfmEngine::Global);
    let empty = SfmWorkspace::builder()
        .build()
        .expect("an empty workspace is a valid staging state");

    let error = Runner::for_config(config.clone())
        .run_workspace(empty)
        .expect_err("global needs precomputed tracks");

    assert!(error.to_string().contains("precomputed tracks"), "{error}");
}

#[test]
fn runner_rejects_backends_for_the_other_engine() {
    let triangulator = DltTriangulator::default();
    let registrar = LinearRegistrar::default();

    let global = SfmConfig::for_engine(SfmEngine::Global);
    let error = Runner::for_config(global.clone())
        .incremental_backends(IncrementalBackends::new(&registrar, &triangulator))
        .run_workspace(prepared_workspace())
        .expect_err("incremental backends cannot run the global engine");
    assert!(matches!(error, SfmError::InvalidConfig(_)), "{error}");

    let incremental = SfmConfig::for_engine(SfmEngine::Incremental);
    let error = Runner::for_config(incremental.clone())
        .global_backends(GlobalBackends::new(
            active_rotation_solver,
            linear_translation_solver,
            &triangulator,
            &triangulator,
        ))
        .run_workspace(prepared_workspace())
        .expect_err("global backends cannot run the incremental engine");
    assert!(matches!(error, SfmError::InvalidConfig(_)), "{error}");
}

#[test]
fn owned_incremental_releases_matching_data() {
    let prepared = prepared_workspace();
    let expected_tracks = prepared.tracks().clone();
    let ba = ObservableBackend::default();

    let config = SfmConfig::for_engine(SfmEngine::Incremental);
    let (workspace, report) = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&ba)
        .run_workspace(prepared)
        .expect("owned incremental route succeeds");

    assert_eq!(report.final_route(), FinalRoute::Incremental);
    assert!(workspace.is_matching_data_released());
    assert_eq!(workspace.feature_store().populated_count(), 0);
    assert_eq!(workspace.match_store().populated_count(), 0);
    assert_eq!(workspace.tracks(), &expected_tracks);
    assert_eq!(workspace.pose_store().populated_count(), CAMERA_COUNT);
}

#[test]
fn run_report_explains_every_incremental_input_view() {
    let prepared = prepared_workspace();
    let triangulator = DltTriangulator::default();
    let registrar = LinearRegistrar::default();
    let ba = ObservableBackend::default();
    let mut config = SfmConfig::for_engine(SfmEngine::Incremental);
    let incremental = config.incremental().clone().with_max_iterations(1);
    config.set_incremental(incremental);

    let (workspace, report) = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&ba)
        .incremental_backends(IncrementalBackends::new(&registrar, &triangulator))
        .run_workspace(prepared)
        .expect("bounded incremental run returns its partial reconstruction");

    assert_eq!(report.view_verdicts().len(), CAMERA_COUNT);
    assert!(
        report
            .view_verdicts()
            .iter()
            .enumerate()
            .all(|(index, verdict)| verdict.view() == ViewId::new(index))
    );
    assert_eq!(
        report
            .view_verdicts()
            .iter()
            .filter(|verdict| verdict.is_registered())
            .count(),
        workspace.registered_view_count()
    );
    let omitted = report
        .view_verdicts()
        .iter()
        .find(|verdict| !verdict.is_registered())
        .expect("the iteration bound leaves one view unregistered");
    assert!(matches!(
        omitted.status(),
        ViewVerdictStatus::Unregistered {
            reason: ViewOmissionReason::Incremental { diagnostic }
        } if diagnostic.view == omitted.view()
    ));
}

#[test]
fn observed_incremental_matches_unobserved_and_brackets_the_run() {
    let config = SfmConfig::for_engine(SfmEngine::Incremental);
    let triangulator = DltTriangulator::default();
    let mut unobserved = prepared_workspace();
    let unobserved_registrar = LinearRegistrar::default();
    let unobserved_ba = ObservableBackend::default();
    let expected_report = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&unobserved_ba)
        .incremental_backends(IncrementalBackends::new(
            &unobserved_registrar,
            &triangulator,
        ))
        .run_in_place(&mut unobserved)
        .expect("unobserved incremental route succeeds");

    let mut observed = prepared_workspace();
    let observed_registrar = LinearRegistrar::default();
    let observed_ba = ObservableBackend::default();
    let mut events = Vec::new();
    let mut sink = |event: openmvg_sfm::events::SfmEvent<'_>| {
        events.push((
            event.sequence,
            event.branch,
            event.phase,
            event.transition,
            event.snapshot.registered_pose_count(),
        ));
    };
    let actual_report = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&observed_ba)
        .incremental_backends(IncrementalBackends::new(&observed_registrar, &triangulator))
        .observer(&mut sink)
        .run_in_place(&mut observed)
        .expect("observed incremental route succeeds");

    let SfmOutcome::Incremental { report } = expected_report.into_outcome() else {
        panic!("incremental configuration must return an incremental report")
    };
    let expected_incremental = *report;
    let SfmOutcome::Incremental { report } = actual_report.into_outcome() else {
        panic!("incremental configuration must return an incremental report")
    };
    let actual_incremental = *report;
    assert_eq!(actual_incremental, expected_incremental);
    assert_reconstruction_state_eq(&observed, &unobserved);

    assert!(
        events
            .iter()
            .enumerate()
            .all(|(index, event)| event.0 == index as u64)
    );
    assert!(matches!(
        events.first(),
        Some((
            0,
            SfmEventBranch::Main,
            SfmEventPhase::Run {
                engine: SfmEngine::Incremental
            },
            SfmEventTransition::Started,
            0
        ))
    ));
    assert!(matches!(
        events.last(),
        Some((
            _,
            SfmEventBranch::Main,
            SfmEventPhase::Run {
                engine: SfmEngine::Incremental
            },
            SfmEventTransition::Completed,
            CAMERA_COUNT
        ))
    ));
}

/// Progress notification captured by [`RecordingProgress`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgressRecord {
    Begin(SfmProgressStage, usize),
    Advanced(SfmProgressStage, usize),
    Finish(SfmProgressStage),
}

#[derive(Debug, Default)]
struct RecordingProgress(Mutex<Vec<ProgressRecord>>);

impl RecordingProgress {
    /// Recorded notifications, with each run of consecutive `advance` calls for one stage
    /// collapsed into a single `Advanced` total.
    fn records(&self) -> Vec<ProgressRecord> {
        let mut records: Vec<ProgressRecord> = Vec::new();
        for &call in self.0.lock().unwrap().iter() {
            match (records.last_mut(), call) {
                (
                    Some(ProgressRecord::Advanced(stage, total)),
                    ProgressRecord::Advanced(next_stage, completed),
                ) if *stage == next_stage => *total += completed,
                _ => records.push(call),
            }
        }
        records
    }
}

impl SfmProgressSink for RecordingProgress {
    fn begin(&self, stage: SfmProgressStage, total: usize) {
        let record = ProgressRecord::Begin(stage, total);
        self.0.lock().unwrap().push(record);
    }

    fn advance(&self, stage: SfmProgressStage, completed: usize) {
        let record = ProgressRecord::Advanced(stage, completed);
        self.0.lock().unwrap().push(record);
    }

    fn finish(&self, stage: SfmProgressStage) {
        self.0.lock().unwrap().push(ProgressRecord::Finish(stage));
    }
}

/// One `begin`, the full `advance` total, and `finish` for every stage in order.
fn completed_stages(stages: &[(SfmProgressStage, usize)]) -> Vec<ProgressRecord> {
    stages
        .iter()
        .flat_map(|&(stage, total)| {
            [
                ProgressRecord::Begin(stage, total),
                ProgressRecord::Advanced(stage, total),
                ProgressRecord::Finish(stage),
            ]
        })
        .collect()
}

#[test]
fn incremental_reports_stage_progress_without_changing_the_reconstruction() {
    let config = SfmConfig::for_engine(SfmEngine::Incremental);
    let triangulator = DltTriangulator::default();
    let run = |progress: &dyn SfmProgressSink| {
        let mut workspace = prepared_workspace();
        let registrar = LinearRegistrar::default();
        let ba = ObservableBackend::default();
        Runner::for_config(config.clone())
            .bundle_adjustment_backend(&ba)
            .incremental_backends(IncrementalBackends::new(&registrar, &triangulator))
            .progress(progress)
            .run_in_place(&mut workspace)
            .expect("incremental route succeeds");
        workspace
    };

    let progress = RecordingProgress::default();
    let reported = run(&progress);
    assert_reconstruction_state_eq(&reported, &run(&()));
    // The two seed views are registered before growth starts.
    assert_eq!(
        progress.records(),
        completed_stages(&[
            (SfmProgressStage::InitialPair, 1),
            (SfmProgressStage::Registration, CAMERA_COUNT - 2),
            (SfmProgressStage::BundleAdjustment, 1),
        ])
    );
}

#[test]
fn global_reports_stage_progress_without_changing_the_reconstruction() {
    let config = SfmConfig::for_engine(SfmEngine::Global);
    let triangulator = DltTriangulator::default();
    let run = |progress: &dyn SfmProgressSink| {
        let ba = ObservableBackend::default();
        Runner::for_config(config.clone())
            .bundle_adjustment_backend(&ba)
            .global_backends(GlobalBackends::new(
                active_rotation_solver,
                linear_translation_solver,
                &triangulator,
                &triangulator,
            ))
            .progress(progress)
            .run_workspace(prepared_workspace())
            .expect("global core route succeeds")
            .0
    };

    let progress = RecordingProgress::default();
    let reported = run(&progress);
    assert_reconstruction_state_eq(&reported, &run(&()));
    assert_eq!(
        progress.records(),
        completed_stages(&[
            (SfmProgressStage::RotationAveraging, 1),
            (SfmProgressStage::TranslationAveraging, 1),
            (SfmProgressStage::Triangulation, FEATURE_COUNT),
            // Fixed-rotation and released-rotation solves, then the robust rounds.
            (
                SfmProgressStage::BundleAdjustment,
                2 + config.global().max_robust_rounds()
            ),
            (SfmProgressStage::SelfCalibration, 1),
        ])
    );
}

/// [`RecordingProgress`] that asks the run to stop once it has received `limit` notifications.
struct CancelAfter {
    limit: usize,
    progress: RecordingProgress,
}

impl CancelAfter {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            progress: RecordingProgress::default(),
        }
    }

    /// Every notification received after the run was asked to stop.
    fn after_cancellation(&self) -> Vec<ProgressRecord> {
        let calls = self.progress.0.lock().unwrap();
        calls[self.limit.min(calls.len())..].to_vec()
    }
}

impl SfmProgressSink for CancelAfter {
    fn begin(&self, stage: SfmProgressStage, total: usize) {
        self.progress.begin(stage, total);
    }

    fn advance(&self, stage: SfmProgressStage, completed: usize) {
        self.progress.advance(stage, completed);
    }

    fn finish(&self, stage: SfmProgressStage) {
        self.progress.finish(stage);
    }

    fn is_canceled(&self) -> bool {
        self.progress.0.lock().unwrap().len() >= self.limit
    }
}

#[test]
fn cancellation_at_every_notification_stops_and_restores_the_workspace() {
    let triangulator = DltTriangulator::default();
    let registrar = LinearRegistrar::default();
    for engine in [SfmEngine::Incremental, SfmEngine::Global] {
        let config = SfmConfig::for_engine(engine);
        let run = |progress: &dyn SfmProgressSink, workspace: &mut SfmWorkspace| {
            let ba = ObservableBackend::default();
            let runner = Runner::for_config(config.clone())
                .bundle_adjustment_backend(&ba)
                .progress(progress);
            match engine {
                SfmEngine::Incremental => {
                    runner.incremental_backends(IncrementalBackends::new(&registrar, &triangulator))
                }
                _ => runner.global_backends(GlobalBackends::new(
                    active_rotation_solver,
                    linear_translation_solver,
                    &triangulator,
                    &triangulator,
                )),
            }
            .run_in_place(workspace)
        };
        let complete = RecordingProgress::default();
        run(&complete, &mut prepared_workspace()).expect("an uncanceled run succeeds");
        let notifications = complete.0.lock().unwrap().len();

        // Canceling after the last notification still fails: the run checks once more before
        // it returns.
        for limit in 0..=notifications {
            let progress = CancelAfter::new(limit);
            let mut workspace = prepared_workspace();
            let error = run(&progress, &mut workspace).expect_err("a canceled run fails");
            assert!(
                matches!(error, SfmError::Canceled),
                "{engine:?} canceled after {limit} notifications failed with {error}"
            );
            assert_reconstruction_state_eq(&workspace, &prepared_workspace());
            assert!(
                progress
                    .after_cancellation()
                    .iter()
                    .all(|record| !matches!(record, ProgressRecord::Begin(..))),
                "{engine:?} began a stage after {limit} notifications"
            );
        }
    }
}

#[test]
fn observed_failure_restores_input_and_reports_rollback() {
    let mut workspace = prepared_workspace();
    let pristine = workspace.clone();
    let triangulator = DltTriangulator::default();
    let ba = ObservableBackend::default();
    let mut transitions = Vec::new();
    let mut sink = |event: openmvg_sfm::events::SfmEvent<'_>| {
        if matches!(event.phase, SfmEventPhase::Run { .. }) {
            transitions.push(event.transition);
        }
    };

    let config = SfmConfig::for_engine(SfmEngine::Incremental);
    let error = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&ba)
        .incremental_backends(IncrementalBackends::new(&FailingRegistrar, &triangulator))
        .observer(&mut sink)
        .run_in_place(&mut workspace)
        .expect_err("injected registration failure must propagate");

    assert!(error.to_string().contains("injected registration failure"));
    assert_reconstruction_state_eq(&workspace, &pristine);
    assert_eq!(
        transitions,
        vec![SfmEventTransition::Started, SfmEventTransition::RolledBack]
    );
}

#[test]
fn observed_owned_global_failure_has_a_terminal_rollback_event() {
    let prepared = prepared_workspace();
    let triangulator = DltTriangulator::default();
    let ba = ObservableBackend::default();
    let config = SfmConfig::for_engine(SfmEngine::Global);
    let mut transitions = Vec::new();
    let mut sink = |event: openmvg_sfm::events::SfmEvent<'_>| {
        if matches!(event.phase, SfmEventPhase::Run { .. }) {
            transitions.push(event.transition);
        }
    };

    let error = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&ba)
        .global_backends(GlobalBackends::new(
            failing_rotation_solver,
            linear_translation_solver,
            &triangulator,
            &triangulator,
        ))
        .observer(&mut sink)
        .run_workspace(prepared)
        .expect_err("injected rotation failure must propagate");

    assert!(error.to_string().contains("injected rotation failure"));
    assert_eq!(
        transitions,
        vec![SfmEventTransition::Started, SfmEventTransition::RolledBack]
    );
}

#[test]
fn observed_owned_incremental_failure_reports_the_pristine_workspace() {
    let prepared = prepared_workspace();
    let triangulator = DltTriangulator::default();
    let ba = ObservableBackend::default();
    let config = SfmConfig::for_engine(SfmEngine::Incremental);
    let mut rollback_retained_matching_data = false;
    let mut sink = |event: openmvg_sfm::events::SfmEvent<'_>| {
        if event.transition == SfmEventTransition::RolledBack {
            rollback_retained_matching_data = !event.snapshot.is_matching_data_released();
        }
    };

    let error = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&ba)
        .incremental_backends(IncrementalBackends::new(&FailingRegistrar, &triangulator))
        .observer(&mut sink)
        .run_workspace(prepared)
        .expect_err("injected registration failure must propagate");

    assert!(error.to_string().contains("injected registration failure"));
    assert!(rollback_retained_matching_data);
}

#[test]
fn global_bundle_has_no_incremental_registration_backend() {
    let mut workspace = prepared_workspace();
    let triangulator = DltTriangulator::default();
    let ba = ObservableBackend::default();
    let config = SfmConfig::for_engine(SfmEngine::Global);

    let report = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&ba)
        .global_backends(GlobalBackends::new(
            active_rotation_solver,
            linear_translation_solver,
            &triangulator,
            &triangulator,
        ))
        .run_in_place(&mut workspace)
        .expect("complete global core succeeds");

    let SfmOutcome::Global {
        preparation,
        result: GlobalOutcome::Reconstructed { report: adaptive },
    } = report.outcome()
    else {
        panic!(
            "expected a complete certified global core, got {:#?}",
            report.outcome()
        )
    };
    assert_eq!(preparation.candidate_domain_camera_count, CAMERA_COUNT);
    assert!(matches!(
        adaptive.completion(),
        GlobalCompletionReport::Reconstructed {
            registered_view_count: CAMERA_COUNT,
            input_view_count: CAMERA_COUNT,
            ..
        }
    ));
    assert_eq!(workspace.pose_store().populated_count(), CAMERA_COUNT);
}

#[test]
fn rotation_conditioning_keeps_the_complete_core() {
    let mut workspace = prepared_workspace();
    let triangulator = DltTriangulator::default();
    let ba = ObservableBackend::default();
    let config = SfmConfig::for_engine(SfmEngine::Global);

    let report = Runner::for_config(config)
        .bundle_adjustment_backend(&ba)
        .global_backends(GlobalBackends::new(
            active_rotation_solver,
            linear_translation_solver,
            &triangulator,
            &triangulator,
        ))
        .run_in_place(&mut workspace)
        .expect("adaptive conditioning leaves the complete core reconstructable");

    let SfmOutcome::Global {
        preparation,
        result: GlobalOutcome::Reconstructed { .. },
    } = report.outcome()
    else {
        panic!(
            "expected a reconstructed global core, got {:#?}",
            report.outcome()
        )
    };
    assert!(
        preparation
            .rotation_conditioning
            .method
            .starts_with("adaptive_cycle_gnc"),
        "{}",
        preparation.rotation_conditioning.method
    );
    assert_eq!(preparation.rotation_conditioning.hard_removed, 0);
    assert_eq!(
        preparation.rotation_conditioning.kept_edges,
        preparation.rotation_conditioning.input_edges
    );
    assert!(!preparation.rotation_conditioning.method.contains("peeled"));
    assert_eq!(preparation.candidate_domain_camera_count, CAMERA_COUNT);
    assert_eq!(workspace.pose_store().populated_count(), CAMERA_COUNT);
}

#[test]
fn global_schedules_direction_after_mixed_deferral() {
    let mut workspace = prepared_workspace();
    let triangulator = DltTriangulator::default();
    let ba = FirstCandidateDeferredBackend::default();
    SCHEDULE_ROTATION_CALLS.store(0, Ordering::SeqCst);
    SCHEDULE_TRANSLATION_CALLS.store(0, Ordering::SeqCst);

    let config = SfmConfig::for_engine(SfmEngine::Global);
    let report = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&ba)
        .global_backends(GlobalBackends::new(
            schedule_counting_rotation_solver,
            schedule_counting_translation_solver,
            &triangulator,
            &triangulator,
        ))
        .run_in_place(&mut workspace)
        .expect("Mixed-to-Direction scheduling completes");

    let SfmOutcome::Global { result, .. } = report.outcome() else {
        panic!("expected global outcome, got {:#?}", report.outcome())
    };
    let adaptive = match result {
        GlobalOutcome::Reconstructed { report } | GlobalOutcome::NoReconstruction { report } => {
            report
        }
        _ => panic!("unexpected future global result: {result:#?}"),
    };
    assert_eq!(adaptive.attempts().len(), 2);
    assert_eq!(
        adaptive.attempts()[0].candidate,
        GlobalCandidate::GlobalPositioning
    );
    assert!(matches!(
        adaptive.attempts()[0].outcome,
        GlobalCandidateStatus::Deferred { .. }
    ));
    assert_eq!(
        adaptive.attempts()[1].candidate,
        GlobalCandidate::TranslationAveraging
    );
    assert!(matches!(
        adaptive.attempts()[1].outcome,
        GlobalCandidateStatus::Accepted { .. }
    ));
    assert!(!matches!(
        adaptive.attempts()[1].outcome,
        GlobalCandidateStatus::Deferred { .. }
    ));
    let mixed_bootstrap = candidate_bootstrap(&adaptive.attempts()[0].outcome)
        .expect("Mixed reached its prepared translation");
    let direction_bootstrap = candidate_bootstrap(&adaptive.attempts()[1].outcome)
        .expect("Direction reached its prepared translation");
    assert_eq!(direction_bootstrap.probes, mixed_bootstrap.probes);
    assert_eq!(
        direction_bootstrap.averaging.rotation_count,
        mixed_bootstrap.averaging.rotation_count
    );
    assert_eq!(
        direction_bootstrap.averaging.rotation_iterations,
        mixed_bootstrap.averaging.rotation_iterations
    );
    assert_eq!(
        direction_bootstrap.averaging.rotation_residual,
        mixed_bootstrap.averaging.rotation_residual
    );
    assert_eq!(ba.preflight_calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        SCHEDULE_ROTATION_CALLS.load(Ordering::SeqCst),
        ROTATION_SOLVES_PER_PREPARATION
    );
    assert_eq!(SCHEDULE_TRANSLATION_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn rejected_global_candidate_is_atomic_and_returns_the_pristine_workspace() {
    let mut workspace = prepared_workspace();
    let pristine = workspace.clone();
    let triangulator = DltTriangulator::default();
    let ba = FixedRotationResourceFailure::default();

    let config = SfmConfig::for_engine(SfmEngine::Global);
    let report = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&ba)
        .global_backends(GlobalBackends::new(
            active_rotation_solver,
            linear_translation_solver,
            &triangulator,
            &triangulator,
        ))
        .run_in_place(&mut workspace)
        .expect("core-only resource rejection is a typed no-core result");

    let SfmOutcome::Global {
        result: GlobalOutcome::NoReconstruction { report: adaptive },
        ..
    } = report.outcome()
    else {
        panic!("expected a typed no-core result")
    };
    assert_eq!(report.final_route(), FinalRoute::GlobalEmpty);
    assert!(!report.has_reconstruction());
    assert!(
        matches!(
            adaptive.completion(),
            GlobalCompletionReport::NoReconstruction {
                reason: GlobalNoReconstructionReason::ResourceCapacity { .. },
                input_view_count: CAMERA_COUNT,
            }
        ),
        "unexpected core-only completion: {:#?}",
        adaptive.completion()
    );
    assert_eq!(ba.fixed_rotation_calls.load(Ordering::SeqCst), 1);
    assert_reconstruction_state_eq(&workspace, &pristine);
    assert!(report.view_verdicts().iter().all(|verdict| matches!(
        verdict.status(),
        ViewVerdictStatus::Unregistered {
            reason: ViewOmissionReason::NoConsistentGlobalSubgraph
        }
    )));
}

#[test]
fn failed_owned_mixed_finish_retries_direction_from_pristine_input() {
    let mut workspace = prepared_workspace();
    let triangulator = DltTriangulator::default();
    let ba = FirstCandidateMutationBackend::failing_first_full_pose();
    let mut config = SfmConfig::for_engine(SfmEngine::Global);
    let adaptive = config.global().clone();
    let policy = adaptive.policy().with_post_fit_trust(
        adaptive
            .policy()
            .post_fit_trust()
            .with_max_rotation_graph_p90_deg(30.0)
            .with_max_all_verified_camera_translation_median_deg(30.0),
    );
    let adaptive = adaptive
        .with_independent_track_validation(None)
        .with_policy(policy);
    config.set_global(adaptive);

    let mut admission_events = Vec::new();
    let mut sink = |event: openmvg_sfm::events::SfmEvent<'_>| {
        if event.phase == SfmEventPhase::GlobalCandidateAdmission
            && matches!(
                event.transition,
                SfmEventTransition::Discarded
                    | SfmEventTransition::Committed
                    | SfmEventTransition::Selected
            )
        {
            admission_events.push((event.branch, event.transition));
        }
    };
    let progress = RecordingProgress::default();
    let report = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&ba)
        .global_backends(GlobalBackends::new(
            active_rotation_solver,
            linear_translation_solver,
            &triangulator,
            &triangulator,
        ))
        .observer(&mut sink)
        .progress(&progress)
        .run_in_place(&mut workspace)
        .expect("Direction recovers after the owned Mixed finish fails");

    let SfmOutcome::Global {
        result: GlobalOutcome::Reconstructed { report: adaptive },
        ..
    } = report.outcome()
    else {
        panic!("expected a certified global outcome")
    };
    assert_eq!(adaptive.attempts().len(), 2);
    assert_eq!(
        adaptive.attempts()[0].candidate,
        GlobalCandidate::GlobalPositioning
    );
    assert!(matches!(
        adaptive.attempts()[0].outcome,
        GlobalCandidateStatus::Failed {
            stage: openmvg_sfm::unstable::global::GlobalCandidateStage::Finish,
            ..
        }
    ));
    assert_eq!(
        adaptive.attempts()[1].candidate,
        GlobalCandidate::TranslationAveraging
    );
    assert!(adaptive.attempts()[1].outcome.is_accepted());
    assert!(matches!(
        adaptive.completion(),
        GlobalCompletionReport::Reconstructed {
            selected_attempt: 1,
            ..
        }
    ));

    let fixed_rotation_inputs = ba
        .fixed_rotation_input_center_y
        .lock()
        .expect("fixed-rotation input mutex");
    assert_eq!(fixed_rotation_inputs.len(), 2);
    assert!(fixed_rotation_inputs[0].abs() < 1e-12);
    assert!(
        fixed_rotation_inputs[1].abs() < 1e-3,
        "Direction inherited the failed Mixed mutation: {fixed_rotation_inputs:?}"
    );
    assert_eq!(ba.fixed_rotation_calls.load(Ordering::SeqCst), 2);
    assert!(ba.full_pose_calls.load(Ordering::SeqCst) >= 2);
    assert_eq!(
        admission_events,
        vec![
            (
                SfmEventBranch::GlobalCandidate {
                    attempt: 0,
                    candidate: GlobalCandidate::GlobalPositioning,
                },
                SfmEventTransition::Discarded,
            ),
            (
                SfmEventBranch::GlobalCandidate {
                    attempt: 1,
                    candidate: GlobalCandidate::TranslationAveraging,
                },
                SfmEventTransition::Committed,
            ),
            (
                SfmEventBranch::GlobalCandidate {
                    attempt: 1,
                    candidate: GlobalCandidate::TranslationAveraging,
                },
                SfmEventTransition::Selected,
            ),
        ]
    );
    assert!(
        workspace
            .pose_store()
            .get(1)
            .expect("the committed Mixed reconstruction contains camera 1")
            .pose
            .center()
            .y
            .abs()
            < 1e-3
    );
    // The failed Direction finish abandons its bundle adjustment after the fixed-rotation solve,
    // and Mixed repeats every candidate stage.
    let solves = 2 + config.global().max_robust_rounds();
    let mut expected = completed_stages(&[
        (SfmProgressStage::RotationAveraging, 1),
        (SfmProgressStage::TranslationAveraging, 1),
        (SfmProgressStage::Triangulation, FEATURE_COUNT),
    ]);
    expected.extend([
        ProgressRecord::Begin(SfmProgressStage::BundleAdjustment, solves),
        ProgressRecord::Advanced(SfmProgressStage::BundleAdjustment, 1),
    ]);
    expected.extend(completed_stages(&[
        (SfmProgressStage::TranslationAveraging, 1),
        (SfmProgressStage::Triangulation, FEATURE_COUNT),
        (SfmProgressStage::BundleAdjustment, solves),
        (SfmProgressStage::SelfCalibration, 1),
    ]));
    assert_eq!(progress.records(), expected);
}

#[test]
fn prepared_candidates_preserve_noncontiguous_camera_ids() {
    let mut workspace = noncontiguous_prepared_workspace();
    NONCONTIGUOUS_ROTATION_CALLS.store(0, Ordering::SeqCst);
    NONCONTIGUOUS_TRANSLATION_CALLS.store(0, Ordering::SeqCst);
    let triangulator = DltTriangulator::default();
    let ba = FirstCandidateDeferredBackend::default();
    let config = SfmConfig::for_engine(SfmEngine::Global);

    let report = Runner::for_config(config.clone())
        .bundle_adjustment_backend(&ba)
        .global_backends(GlobalBackends::new(
            noncontiguous_counting_rotation_solver,
            noncontiguous_counting_translation_solver,
            &triangulator,
            &triangulator,
        ))
        .run_in_place(&mut workspace)
        .expect("both candidates run over the non-contiguous prepared domain");

    let expected = vec![ViewId::new(0), ViewId::new(2), ViewId::new(4)];
    let SfmOutcome::Global {
        preparation,
        result,
    } = report.outcome()
    else {
        panic!("expected global outcome")
    };
    assert_eq!(preparation.candidate_domain_camera_ids, expected);
    let adaptive = match result {
        GlobalOutcome::Reconstructed { report } | GlobalOutcome::NoReconstruction { report } => {
            report
        }
        _ => panic!("unexpected future global result: {result:#?}"),
    };
    assert_eq!(adaptive.attempts().len(), 2);
    assert_eq!(
        adaptive.attempts()[0].candidate,
        GlobalCandidate::GlobalPositioning
    );
    assert_eq!(
        adaptive.attempts()[1].candidate,
        GlobalCandidate::TranslationAveraging
    );
    assert!(matches!(
        adaptive.attempts()[1].outcome,
        GlobalCandidateStatus::Accepted { .. }
    ));
    let mixed = candidate_bootstrap(&adaptive.attempts()[0].outcome)
        .unwrap_or_else(|| panic!("Mixed missed the prepared rotations: {adaptive:#?}"));
    let direction = candidate_bootstrap(&adaptive.attempts()[1].outcome)
        .unwrap_or_else(|| panic!("Direction missed the prepared rotations: {adaptive:#?}"));
    assert_eq!(direction.probes, mixed.probes);
    assert_eq!(
        direction
            .probes
            .degrees
            .iter()
            .filter_map(|(image, degree)| (*degree > 0).then_some(*image))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(direction.averaging.rotation_count, expected.len());
    assert_eq!(mixed.averaging.rotation_count, expected.len());
    assert_eq!(
        NONCONTIGUOUS_ROTATION_CALLS.load(Ordering::SeqCst),
        ROTATION_SOLVES_PER_PREPARATION
    );
    assert_eq!(NONCONTIGUOUS_TRANSLATION_CALLS.load(Ordering::SeqCst), 1);
}
