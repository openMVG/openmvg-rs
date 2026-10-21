// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Mutex;

use openmvg_sfm::{
    model::*,
    unstable::*,
    unstable::{backends::*, types::*},
    *,
};

#[derive(Debug, Default, PartialEq, Eq)]
struct CapturedSeeds {
    features: BTreeMap<ViewId, RunSeed>,
    matching: Vec<RunSeed>,
    verification: Vec<RunSeed>,
}

#[derive(Default)]
struct SeedCapturingFrontend {
    seeds: Mutex<CapturedSeeds>,
}

impl FeatureExtractor for SeedCapturingFrontend {
    fn extract(&self, ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
        self.seeds
            .lock()
            .unwrap()
            .features
            .insert(ctx.view, ctx.seed);
        Ok(ExtractedFeatures::try_new(
            vec![
                Keypoint::try_new(0.0, 0.0, 1.0, 0.0)?,
                Keypoint::try_new(1.0, 1.0, 1.0, 0.0)?,
            ],
            DescriptorSet::try_f32(vec![0.0, 1.0], 1)?,
        )?)
    }
}

impl PairMatcher for SeedCapturingFrontend {
    fn match_features(
        &self,
        ctx: &PairContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        self.seeds.lock().unwrap().matching.push(ctx.seed);
        PairMatches::try_new(
            ctx.pair,
            vec![
                FeatureMatch::try_new(FeatureId::new(0), FeatureId::new(0), Some(0.0))?,
                FeatureMatch::try_new(FeatureId::new(1), FeatureId::new(1), Some(0.0))?,
            ],
        )
    }
}

impl GeometricVerifier for SeedCapturingFrontend {
    fn verify_pair(
        &self,
        ctx: &VerificationContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
        matches: &PairMatches,
        _ki: &openmvg_sfm::Camera,
        _kj: &openmvg_sfm::Camera,
    ) -> Result<VerificationVerdict> {
        self.seeds.lock().unwrap().verification.push(ctx.seed);
        Ok(VerificationVerdict::Rejected {
            reason: VerificationRejectReason::InsufficientInliers {
                required: 3,
                actual: matches.len(),
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

fn capture(root_seed: RunSeed) -> CapturedSeeds {
    let mut workspace = SfmWorkspace::builder()
        .root_seed(root_seed)
        .image_meta([
            ImageMeta::external_id("a", 64, 48),
            ImageMeta::external_id("b", 64, 48),
        ])
        .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .exhaustive_pairs()
        .build()
        .unwrap();
    let frontend = SeedCapturingFrontend::default();
    workspace
        .extract_features(&frontend, ViewSelector::All)
        .unwrap();
    workspace
        .match_pairs(&frontend, PairSelector::Active)
        .unwrap();
    workspace
        .verify_pairs(&frontend, PairSelector::Active)
        .unwrap();
    frontend.seeds.into_inner().unwrap()
}

#[test]
fn root_seed_reaches_frontend_stages_through_stable_domains() {
    let root_seed = RunSeed::from_u64(42);
    let captured = capture(root_seed);

    assert_eq!(
        captured.features[&ViewId::new(0)],
        root_seed.derive("pipeline/feature-extraction/v1", &[ViewId::new(0).value()])
    );
    assert_eq!(
        captured.matching,
        [root_seed.derive(
            "pipeline/pair-matching/v1",
            &[ViewId::new(0).value(), ViewId::new(1).value()]
        )]
    );
    assert_eq!(
        captured.verification,
        [root_seed.derive(
            "pipeline/geometric-verification/v1",
            &[ViewId::new(0).value(), ViewId::new(1).value()]
        )]
    );
    assert_ne!(captured.matching, captured.verification);
    assert_eq!(captured, capture(root_seed));
    assert_ne!(captured, capture(43.into()));
}
