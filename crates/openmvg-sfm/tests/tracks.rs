// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Matrix3, Rotation3, Vector3};
use openmvg_sfm::{
    model::*,
    unstable::*,
    unstable::{backends::*, reports::ErrorStats, types::*},
    *,
};

fn track_workspace() -> SfmWorkspace {
    SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("image-a", 64, 48),
            ImageMeta::external_id("image-b", 64, 48),
        ])
        .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs(vec![(ViewId::new(0), ViewId::new(1))])
        .build()
        .unwrap()
}

struct TrackExtractor;

impl FeatureExtractor for TrackExtractor {
    fn extract(&self, ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
        let offset = ctx.view.index() as f32 * 10.0;
        Ok(ExtractedFeatures::try_new(
            vec![
                Keypoint::try_new(1.0 + offset, 2.0 + offset, 1.0, 0.0)?,
                Keypoint::try_new(3.0 + offset, 4.0 + offset, 1.0, 0.0)?,
            ],
            DescriptorSet::try_f32(vec![0.0, 1.0], 1)?,
        )?)
    }
}

struct TrackMatcher;

impl PairMatcher for TrackMatcher {
    fn match_features(
        &self,
        ctx: &PairContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        PairMatches::try_new(
            ctx.pair,
            vec![
                FeatureMatch::try_new(FeatureId::new(0), FeatureId::new(0), Some(0.0))?,
                FeatureMatch::try_new(FeatureId::new(1), FeatureId::new(1), Some(0.1))?,
            ],
        )
    }
}

struct TrackVerifier;

impl GeometricVerifier for TrackVerifier {
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
            InlierMask::new(vec![true, false]),
            ErrorStats::from_values(&[0.0], &[1.0]),
            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
            "track-test",
            1,
            TerminationReason::Converged,
            Vec::new(),
        )?;
        let model = GeometricModel::Essential {
            matrix: Matrix3::identity(),
            relative_pose: RelativePose::from_baseline_direction(
                Rotation3::identity(),
                Vector3::x_axis(),
            )?,
            evidence,
        };
        model.validate_for_matches(matches)?;
        Ok(VerificationVerdict::Accepted {
            model,
            report: PairVerificationReport {
                pair: ctx.pair,
                input_match_count: matches.len(),
                inlier_count: 1,
                message: None,
            },
        })
    }
}

fn conflict_workspace() -> SfmWorkspace {
    SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("image-a", 64, 48),
            ImageMeta::external_id("image-b", 64, 48),
            ImageMeta::external_id("image-c", 64, 48),
        ])
        .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs(vec![
            (ViewId::new(0), ViewId::new(1)),
            (ViewId::new(1), ViewId::new(2)),
            (ViewId::new(0), ViewId::new(2)),
        ])
        .build()
        .unwrap()
}

struct ConflictMatcher;

impl PairMatcher for ConflictMatcher {
    fn match_features(
        &self,
        ctx: &PairContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        let (feature_i, feature_j, distance) = match (ctx.view_i.index(), ctx.view_j.index()) {
            (0, 1) | (1, 2) => (0, 0, 0.1),
            (0, 2) => (1, 0, 0.2),
            pair => panic!("unexpected pair {pair:?}"),
        };
        PairMatches::try_new(
            ctx.pair,
            vec![FeatureMatch::try_new(
                FeatureId::new(feature_i),
                FeatureId::new(feature_j),
                Some(distance),
            )?],
        )
    }
}

struct AllInlierVerifier;

impl GeometricVerifier for AllInlierVerifier {
    fn verify_pair(
        &self,
        ctx: &VerificationContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
        matches: &PairMatches,
        _ki: &openmvg_sfm::Camera,
        _kj: &openmvg_sfm::Camera,
    ) -> Result<VerificationVerdict> {
        let residuals = vec![0.0; matches.len()];
        let evidence = ModelEvidence::try_new(
            InlierMask::new(vec![true; matches.len()]),
            ErrorStats::from_values(&residuals, &[1.0]),
            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
            "track-conflict-test",
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

#[test]
fn tracks_resolve_verified_inlier_coordinates_from_feature_sets() {
    let mut workspace = track_workspace();
    workspace
        .extract_features(&TrackExtractor, ViewSelector::All)
        .unwrap()
        .match_pairs(&TrackMatcher, PairSelector::Active)
        .unwrap()
        .verify_pairs(&TrackVerifier, PairSelector::Active)
        .unwrap();

    let report = workspace.build_tracks().unwrap();

    assert_eq!(report.track_count, 1);
    assert_eq!(report.observation_count, 2);
    assert_eq!(workspace.tracks().len(), 1);
    let observations = workspace
        .tracks()
        .get(openmvg_sfm::TrackId::new(0))
        .unwrap()
        .observations();
    assert_eq!(observations[0].view, ViewId::new(0));
    assert_eq!(observations[0].feature, FeatureId::new(0));
    assert_eq!(
        workspace
            .track_observation_pixel(&observations[0])
            .unwrap()
            .x,
        1.0
    );
    assert_eq!(observations[1].view, ViewId::new(1));
    assert_eq!(
        workspace
            .track_observation_pixel(&observations[1])
            .unwrap()
            .x,
        11.0
    );
}

#[test]
fn tracks_keep_resolvable_geometry_after_matching_data_is_released() {
    let mut workspace = track_workspace();
    workspace
        .extract_features(&TrackExtractor, ViewSelector::All)
        .unwrap()
        .match_pairs(&TrackMatcher, PairSelector::Active)
        .unwrap()
        .verify_pairs(&TrackVerifier, PairSelector::Active)
        .unwrap();
    workspace.build_tracks().unwrap();

    workspace.release_matching_data().unwrap();

    assert!(workspace.is_matching_data_released());
    assert_eq!(workspace.feature_store().populated_count(), 0);
    assert_eq!(workspace.match_store().populated_count(), 0);
    assert_eq!(workspace.tracks().len(), 1);
    let observation = &workspace
        .tracks()
        .get(openmvg_sfm::TrackId::new(0))
        .unwrap()
        .observations()[1];
    assert_eq!(
        workspace.track_observation_pixel(observation).unwrap().x,
        11.0
    );
    assert!(matches!(
        workspace.match_pairs(&TrackMatcher, PairSelector::Active),
        Err(openmvg_sfm::SfmError::MatchingDataReleased)
    ));
    workspace.release_matching_data().unwrap();
}

#[test]
fn pipeline_keeps_stronger_edges_when_track_components_conflict() {
    let mut workspace = conflict_workspace();
    workspace
        .extract_features(&TrackExtractor, ViewSelector::All)
        .unwrap()
        .match_pairs(&ConflictMatcher, PairSelector::Active)
        .unwrap()
        .verify_pairs(&AllInlierVerifier, PairSelector::Active)
        .unwrap();

    let report = workspace.build_tracks().unwrap();
    assert_eq!(report.track_count, 1);
    assert_eq!(report.observation_count, 3);
    assert_eq!(report.rejected_conflict_count, 1);
    assert_eq!(report.rejected_singleton_count, 1);

    let observations = workspace
        .tracks()
        .get(openmvg_sfm::TrackId::new(0))
        .unwrap()
        .observations();
    assert_eq!(observations.len(), 3);
    assert!(
        observations
            .iter()
            .all(|observation| observation.feature == FeatureId::new(0))
    );
}

struct MultiOrientationExtractor;

impl FeatureExtractor for MultiOrientationExtractor {
    fn extract(&self, ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
        let offset = ctx.view.index() as f32 * 10.0;
        Ok(ExtractedFeatures::try_new(
            vec![
                Keypoint::try_new(1.0 + offset, 2.0 + offset, 1.0, 0.0)?,
                Keypoint::try_new(1.0 + offset, 2.0 + offset, 1.0, 1.5)?,
                Keypoint::try_new(3.0 + offset, 4.0 + offset, 1.0, 0.0)?,
            ],
            DescriptorSet::try_f32(vec![0.0, 1.0, 2.0], 1)?,
        )?)
    }
}

struct IdentityMatcher;

impl PairMatcher for IdentityMatcher {
    fn match_features(
        &self,
        ctx: &PairContext,
        fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        let matches = (0..fi.len())
            .map(|feature| {
                FeatureMatch::try_new(FeatureId::new(feature), FeatureId::new(feature), Some(0.0))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        PairMatches::try_new(ctx.pair, matches)
    }
}

#[test]
fn tracks_count_one_measurement_per_pixel_for_multi_orientation_features() {
    let mut workspace = track_workspace();
    workspace
        .extract_features(&MultiOrientationExtractor, ViewSelector::All)
        .unwrap()
        .match_pairs(&IdentityMatcher, PairSelector::Active)
        .unwrap()
        .verify_pairs(&AllInlierVerifier, PairSelector::Active)
        .unwrap();

    let report = workspace.build_tracks().unwrap();

    assert_eq!(report.track_count, 2);
    assert_eq!(report.observation_count, 4);
    assert_eq!(report.rejected_conflict_count, 0);
    let features: Vec<Vec<FeatureId>> = (0..2)
        .map(|track| {
            workspace
                .tracks()
                .get(openmvg_sfm::TrackId::new(track))
                .unwrap()
                .observations()
                .iter()
                .map(|observation| observation.feature)
                .collect()
        })
        .collect();
    assert_eq!(
        features,
        vec![
            vec![FeatureId::new(0), FeatureId::new(0)],
            vec![FeatureId::new(2), FeatureId::new(2)],
        ]
    );
}
