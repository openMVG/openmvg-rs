// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Matrix3, Rotation3, Vector3};
use openmvg_sfm::{
    config::*,
    model::*,
    unstable::*,
    unstable::{backends::*, incremental::*, reports::ErrorStats, types::*},
    *,
};

fn seeded_workspace() -> SfmWorkspace {
    let mut workspace = SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("image-a", 64, 48),
            ImageMeta::external_id("image-b", 64, 48),
            ImageMeta::external_id("image-c", 64, 48),
        ])
        .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .exhaustive_pairs()
        .build()
        .unwrap();
    workspace
        .extract_features(&SeedExtractor, ViewSelector::All)
        .unwrap()
        .match_pairs(&SeedMatcher, PairSelector::Active)
        .unwrap()
        .verify_pairs(&SeedVerifier, PairSelector::Active)
        .unwrap();
    workspace
}

struct SeedExtractor;

impl FeatureExtractor for SeedExtractor {
    fn extract(&self, _ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
        Ok(ExtractedFeatures::try_new(
            (0..12)
                .map(|index| Keypoint::try_new(index as f32, index as f32, 1.0, 0.0))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            DescriptorSet::try_f32(vec![0.0; 12], 1)?,
        )?)
    }
}

struct SeedMatcher;

impl PairMatcher for SeedMatcher {
    fn match_features(
        &self,
        ctx: &PairContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        PairMatches::try_new(
            ctx.pair,
            (0..12)
                .map(|index| {
                    FeatureMatch::try_new(FeatureId::new(index), FeatureId::new(index), Some(0.0))
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
    }
}

struct SeedVerifier;

impl GeometricVerifier for SeedVerifier {
    fn verify_pair(
        &self,
        ctx: &VerificationContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
        matches: &PairMatches,
        _ki: &openmvg_sfm::Camera,
        _kj: &openmvg_sfm::Camera,
    ) -> Result<VerificationVerdict> {
        let inliers = match ctx.pair.endpoints() {
            (first, second) if (first, second) == (ViewId::new(0), ViewId::new(1)) => 9,
            (first, second) if (first, second) == (ViewId::new(0), ViewId::new(2)) => 12,
            _ => 10,
        };
        let mut mask = vec![false; matches.len()];
        for value in mask.iter_mut().take(inliers) {
            *value = true;
        }
        let evidence = ModelEvidence::try_new(
            InlierMask::new(mask),
            ErrorStats::from_values(&vec![0.0; inliers], &[1.0]),
            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
            "seed-test",
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
                inlier_count: inliers,
                message: None,
            },
        })
    }
}

#[test]
fn seed_best_pair_is_deterministic_by_inlier_score() {
    let workspace = seeded_workspace();

    let seed = select_initial_pair(
        &workspace,
        &InitialPairPolicy::BestPair(InitialPairConfig::default().with_min_inliers(8)),
    )
    .unwrap();

    assert_eq!(seed.views.len(), 2);
    assert_eq!(
        seed.pairs,
        vec![ViewPair::new(ViewId::new(0), ViewId::new(2))]
    );
    assert_eq!(seed.breakdown.inliers, 12);
}

#[test]
fn explicit_pair_keeps_the_seed_fixed() {
    let workspace = seeded_workspace();

    let seed = select_initial_pair(
        &workspace,
        &InitialPairPolicy::ExplicitPair(ViewPair::new(ViewId::new(0), ViewId::new(1))),
    )
    .unwrap();

    assert_eq!(seed.views.len(), 2);
    assert_eq!(seed.views, vec![ViewId::new(0), ViewId::new(1)]);
    assert_eq!(
        seed.pairs,
        vec![ViewPair::new(ViewId::new(0), ViewId::new(1))]
    );
    assert_eq!(seed.breakdown.inliers, 9);
}

#[test]
fn explicit_pair_rejects_a_repeated_image() {
    let workspace = seeded_workspace();

    let error = select_initial_pair(
        &workspace,
        &InitialPairPolicy::ExplicitPair(ViewPair::new(ViewId::new(0), ViewId::new(0))),
    )
    .unwrap_err();

    assert!(matches!(error, openmvg_sfm::SfmError::InvalidInput(_)));
}

#[test]
fn pair_seed_reports_component_and_registration_frontier() {
    let mut workspace = seeded_workspace();
    workspace.build_tracks().unwrap();

    let seed = select_initial_pair(
        &workspace,
        &InitialPairPolicy::BestPair(InitialPairConfig::default().with_min_inliers(8)),
    )
    .unwrap();

    assert_eq!(seed.breakdown.component_size, 3);
    assert_eq!(seed.breakdown.frontier_views, 1);
    assert!(seed.breakdown.frontier_tracks >= 6);
}
