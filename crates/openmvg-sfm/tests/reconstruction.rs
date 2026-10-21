// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Matrix3, Point3, Rotation3, Vector3};
use openmvg_core::model::{EstimationResult, RelativeRotation, RelativeTranslation};
use openmvg_geometry::averaging::{
    RotationConfig, RotationEstimate, TranslationConfig, TranslationEstimate,
};
#[cfg(feature = "parallel")]
use openmvg_sfm::unstable::ops::reconstruction::triangulate_incremental;
use openmvg_sfm::{
    model::*,
    unstable::*,
    unstable::{backends::*, reports::ErrorStats, types::*},
    *,
};

#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};

fn reconstruction_workspace() -> SfmWorkspace {
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
    initialize_global_geometry(
        &mut workspace,
        &ReconstructionExtractor,
        &ReconstructionMatcher,
        &FixedTriangulator,
    );
    workspace
}

fn initialize_global_geometry(
    workspace: &mut SfmWorkspace,
    extractor: &dyn FeatureExtractor,
    matcher: &dyn PairMatcher,
    triangulator: &dyn Triangulator,
) {
    let config = openmvg_sfm::config::GlobalAveragingConfig::default();
    workspace
        .extract_features(extractor, openmvg_sfm::unstable::ViewSelector::All)
        .unwrap();
    workspace
        .match_pairs(matcher, openmvg_sfm::unstable::PairSelector::Active)
        .unwrap();
    workspace
        .verify_pairs(
            &ReconstructionVerifier,
            openmvg_sfm::unstable::PairSelector::Active,
        )
        .unwrap();
    openmvg_sfm::unstable::ops::averaging::run_global_averaging_scoped(
        workspace,
        reconstruction_rotations,
        reconstruction_centers,
        config
            .min_translation_parallax()
            .map(|threshold| threshold.value().to_degrees()),
        config.min_edge_isotropy(),
    )
    .unwrap();
    workspace.build_tracks().unwrap();
    workspace.release_matching_data().unwrap();
    openmvg_sfm::unstable::ops::reconstruction::triangulate(
        workspace,
        triangulator,
        config.gauge_scale_policy(),
    )
    .unwrap();
}

struct ReconstructionExtractor;

impl FeatureExtractor for ReconstructionExtractor {
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

struct ReconstructionMatcher;

impl PairMatcher for ReconstructionMatcher {
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

struct ReconstructionVerifier;

impl GeometricVerifier for ReconstructionVerifier {
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
            "reconstruction-test",
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

fn reconstruction_rotations(
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

fn reconstruction_centers(
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

struct FixedTriangulator;

impl Triangulator for FixedTriangulator {
    fn triangulate(
        &self,
        observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        assert_eq!(observations.len(), 2);
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

#[test]
fn reconstruction_triangulates_tracks_into_distinct_landmarks() {
    let mut workspace = reconstruction_workspace();

    let report = workspace.triangulate(&FixedTriangulator).unwrap();

    assert_eq!(report.attempted_tracks, 1);
    assert_eq!(report.triangulated_landmarks, 1);
    assert_eq!(workspace.landmarks().len(), 1);
    let landmark = workspace
        .landmarks()
        .get(openmvg_sfm::LandmarkId::new(0))
        .unwrap();
    assert_eq!(landmark.source_track, openmvg_sfm::TrackId::new(0));
    assert_eq!(landmark.observation_support.len(2), 2);
    assert_eq!(
        workspace.components()[0].landmarks,
        vec![openmvg_sfm::LandmarkId::new(0)]
    );
}

#[test]
fn reconstruction_uses_retained_track_geometry_after_feature_and_match_drop() {
    let mut workspace = reconstruction_workspace();
    workspace.release_matching_data().unwrap();

    let report = workspace.triangulate(&FixedTriangulator).unwrap();

    assert_eq!(workspace.feature_store().populated_count(), 0);
    assert_eq!(workspace.match_store().populated_count(), 0);
    assert_eq!(report.triangulated_landmarks, 1);
}

#[cfg(feature = "parallel")]
struct ManyFeaturesExtractor;

#[cfg(feature = "parallel")]
impl FeatureExtractor for ManyFeaturesExtractor {
    fn extract(&self, ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
        let offset = if ctx.view == ViewId::new(0) { 0.0 } else { 5.0 };
        Ok(ExtractedFeatures::try_new(
            (0..128)
                .map(|index| Keypoint::try_new(1.0 + index as f32 + offset, 24.0, 1.0, 0.0))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            DescriptorSet::try_f32((0..128).map(|index| index as f32).collect(), 1)?,
        )?)
    }
}

#[cfg(feature = "parallel")]
struct ManyFeaturesMatcher;

#[cfg(feature = "parallel")]
impl PairMatcher for ManyFeaturesMatcher {
    fn match_features(
        &self,
        ctx: &PairContext,
        fi: &ExtractedFeatures,
        fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        assert_eq!(fi.len(), fj.len());
        PairMatches::try_new(
            ctx.pair,
            (0..fi.len())
                .map(|index| {
                    FeatureMatch::try_new(
                        FeatureId::new(index),
                        FeatureId::new(index),
                        Some(index as f32),
                    )
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
    }
}

#[cfg(feature = "parallel")]
#[derive(Default)]
struct ConcurrencyProbeTriangulator {
    active: AtomicUsize,
    peak: AtomicUsize,
}

#[cfg(feature = "parallel")]
struct RejectAllTriangulator;

#[cfg(feature = "parallel")]
impl Triangulator for RejectAllTriangulator {
    fn triangulate(
        &self,
        _observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        Ok(TriangulationVerdict::Rejected(
            TriangulationRejectReason::NoValidHypothesis,
        ))
    }
}

#[cfg(feature = "parallel")]
impl Triangulator for ConcurrencyProbeTriangulator {
    fn triangulate(
        &self,
        observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(1));
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(TriangulatedLandmark {
            point: Point3::new(observations[0].xy.x, 0.0, 5.0),
            reprojection_error: openmvg_core::PixelResidual::ZERO,
            supporting_observation_indices: (0..observations.len())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
        .into())
    }
}

#[cfg(feature = "parallel")]
#[test]
fn global_triangulation_runs_tracks_concurrently_and_commits_in_order() {
    let mut workspace = SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("image-a", 128, 48),
            ImageMeta::external_id("image-b", 128, 48),
        ])
        .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 64.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs(vec![(ViewId::new(0), ViewId::new(1))])
        .build()
        .unwrap();
    initialize_global_geometry(
        &mut workspace,
        &ManyFeaturesExtractor,
        &ManyFeaturesMatcher,
        &FixedTriangulator,
    );

    let triangulator = ConcurrencyProbeTriangulator::default();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let report = pool
        .install(|| workspace.triangulate(&triangulator))
        .unwrap();

    assert_eq!(report.triangulated_landmarks, 128);
    assert!(triangulator.peak.load(Ordering::SeqCst) > 1);
    assert!(
        workspace
            .landmarks()
            .as_slice()
            .iter()
            .enumerate()
            .all(|(index, landmark)| {
                landmark.id.index() == index && landmark.source_track.index() == index
            })
    );
}

#[cfg(feature = "parallel")]
#[test]
fn incremental_growth_is_thread_count_invariant() {
    let mut workspace = SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("image-a", 128, 48),
            ImageMeta::external_id("image-b", 128, 48),
        ])
        .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 64.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs(vec![(ViewId::new(0), ViewId::new(1))])
        .build()
        .unwrap();
    initialize_global_geometry(
        &mut workspace,
        &ManyFeaturesExtractor,
        &ManyFeaturesMatcher,
        &RejectAllTriangulator,
    );
    assert_eq!(workspace.landmarks().len(), 0);

    let mut serial_workspace = workspace.clone();
    let mut parallel_workspace = workspace;
    let serial_triangulator = ConcurrencyProbeTriangulator::default();
    let parallel_triangulator = ConcurrencyProbeTriangulator::default();
    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let serial_report = serial_pool
        .install(|| {
            triangulate_incremental(
                &mut serial_workspace,
                &serial_triangulator,
                2,
                f64::INFINITY,
            )
        })
        .unwrap();
    let parallel_report = parallel_pool
        .install(|| {
            triangulate_incremental(
                &mut parallel_workspace,
                &parallel_triangulator,
                2,
                f64::INFINITY,
            )
        })
        .unwrap();

    assert_eq!(serial_report, parallel_report);
    assert_eq!(serial_workspace.landmarks(), parallel_workspace.landmarks());
    assert_eq!(serial_triangulator.peak.load(Ordering::SeqCst), 1);
    assert!(parallel_triangulator.peak.load(Ordering::SeqCst) > 1);
}
