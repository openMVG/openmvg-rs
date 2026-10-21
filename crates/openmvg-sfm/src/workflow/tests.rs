// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::engines::graph_config;
use super::*;
use crate::adapters::features::{SiftFrontendConfig, SiftFrontendPreset};
use crate::image::prepare::{ImageDatasetConfig, ImageIntrinsicPolicy, prepare_image_paths};
use crate::progress::SfmProgressStage;
use crate::progress::testing::{CancelAfter, ProgressRecord, RecordingProgress};
use crate::{Camera, Result, SfmWorkspace};
use openmvg_core::camera::{
    BrownConradyCamera, CameraModelType, RadialDistortion, TangentialDistortion,
};

fn save_noise_image(path: &std::path::Path) {
    let mut state = 7_u32;
    image::GrayImage::from_fn(96, 96, |_, _| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        image::Luma([(state >> 24) as u8])
    })
    .save(path)
    .unwrap();
}

/// Unrelated noise images: every pair is matched, and none survives verification.
fn noise_images(root: &std::path::Path, images: usize) -> Vec<std::path::PathBuf> {
    (0..images)
        .map(|index| {
            let path = root.join(format!("view-{index}.png"));
            save_noise_image(&path);
            path
        })
        .collect()
}

fn noise_dataset(root: &std::path::Path, config: &ImageDatasetConfig) -> Dataset {
    prepare_image_paths(&noise_images(root, 3), config)
        .unwrap()
        .build_dataset(3)
        .unwrap()
}

/// Default front-end progress over three images whose pairs all fail verification.
fn rejected_progress(radial_groups: Option<usize>) -> Vec<ProgressRecord> {
    let images = 3;
    // Every one of the n(n-1)/2 pairs is matched and verified.
    let pairs = 3;
    let mut expected = vec![
        ProgressRecord::Begin(SfmProgressStage::FeatureExtraction, images),
        ProgressRecord::Advanced(SfmProgressStage::FeatureExtraction, images),
        ProgressRecord::Finish(SfmProgressStage::FeatureExtraction),
    ];
    // The default HNSW matcher builds one graph per image before matching.
    let mut counts = vec![(SfmProgressStage::MatcherPreparation, images)];
    let stages = [SfmProgressStage::Matching, SfmProgressStage::Verification];
    counts.extend(stages.into_iter().map(|stage| (stage, pairs)));
    counts.extend(radial_groups.map(|groups| (SfmProgressStage::RadialCalibration, groups)));
    for (stage, count) in counts {
        expected.push(ProgressRecord::Begin(stage, count));
        expected.push(ProgressRecord::Advanced(stage, count));
        expected.push(ProgressRecord::Finish(stage));
    }
    // Verification rejects every pair, so no pair is left to merge into tracks.
    expected.push(ProgressRecord::Begin(SfmProgressStage::TrackBuilding, 0));
    expected.push(ProgressRecord::Finish(SfmProgressStage::TrackBuilding));
    expected
}

#[test]
fn frontend_reports_progress_without_changing_the_view_graph() {
    let root = tempfile::TempDir::new().unwrap();
    let dataset = noise_dataset(root.path(), &ImageDatasetConfig::default());
    let frontend = Frontend::default();

    let progress = RecordingProgress::default();
    let observed = frontend
        .build_with_report(&dataset, &progress)
        .map(|(graph, _)| graph);
    let unobserved = frontend.build(&dataset);

    let summary = |graph: Result<ViewGraph>| {
        graph
            .map(|graph| format!("{:?}", graph.workspace()))
            .map_err(|error| error.to_string())
    };
    assert_eq!(summary(observed), summary(unobserved));
    // Each image without EXIF is its own radial camera group.
    assert_eq!(progress.records(), rejected_progress(Some(3)));
}

#[test]
fn frontend_skips_radial_calibration_without_unknown_distortion() {
    let pinhole = ImageDatasetConfig::default().with_camera_model(CameraModelType::PinholeSingle);
    let zero_distortion = BrownConradyCamera::try_with_focal(
        100.0,
        48.0,
        48.0,
        RadialDistortion::K1(0.0),
        TangentialDistortion::None,
    )
    .map(Camera::BrownConrady)
    .unwrap();
    let calibrated =
        ImageDatasetConfig::default().with_intrinsic_policy(ImageIntrinsicPolicy::Shared {
            camera: zero_distortion,
        });

    for config in [pinhole, calibrated] {
        let root = tempfile::TempDir::new().unwrap();
        let dataset = noise_dataset(root.path(), &config);
        let progress = RecordingProgress::default();

        assert!(
            Frontend::default()
                .build_with_report(&dataset, &progress)
                .is_err()
        );
        assert_eq!(progress.records(), rejected_progress(None));
    }
}

#[test]
fn from_dir_keeps_a_discovered_calibration_fixed() {
    let root = tempfile::TempDir::new().unwrap();
    let images = root.path().join("images");
    std::fs::create_dir(&images).unwrap();
    noise_images(&images, 2);

    let estimated = Dataset::from_dir(root.path()).unwrap();
    std::fs::write(images.join("K.txt"), "100 0 48\n0 100 47\n0 0 1\n").unwrap();
    let calibrated = Dataset::from_dir(root.path()).unwrap();

    assert!(!estimated.has_calibrated_intrinsics());
    assert!(calibrated.has_calibrated_intrinsics());
    assert_eq!(
        calibrated.camera_models(),
        [Camera::try_pinhole(100.0, 100.0, 48.0, 47.0).unwrap()]
    );
    assert_eq!(calibrated.root_seed(), Dataset::DEFAULT_SEED.into());
}

#[test]
fn from_dir_with_defaults_matches_from_dir() {
    let root = tempfile::TempDir::new().unwrap();
    noise_images(root.path(), 2);

    let options = DatasetConfig::default();
    let defaults = Dataset::from_dir_with(root.path(), &options).unwrap();
    let discovered = Dataset::from_dir(root.path()).unwrap();

    assert_eq!(options.seed(), Dataset::DEFAULT_SEED.into());
    assert_eq!(options.intrinsics(), None);
    assert_eq!(defaults.root_seed(), discovered.root_seed());
    assert_eq!(defaults.views(), discovered.views());
    assert_eq!(defaults.camera_models(), discovered.camera_models());
    assert_eq!(
        defaults.has_calibrated_intrinsics(),
        discovered.has_calibrated_intrinsics()
    );
}

#[test]
fn from_dir_with_prefers_explicit_intrinsics_and_seed() {
    let root = tempfile::TempDir::new().unwrap();
    noise_images(root.path(), 2);
    std::fs::write(root.path().join("K.txt"), "100 0 48\n0 100 47\n0 0 1\n").unwrap();
    let explicit = root.path().join("explicit.txt");
    std::fs::write(&explicit, "200 0 40\n0 210 41\n0 0 1\n").unwrap();

    let options = DatasetConfig::default()
        .with_seed(7.into())
        .with_intrinsics(&explicit);
    let dataset = Dataset::from_dir_with(root.path(), &options).unwrap();

    assert_eq!(options.intrinsics(), Some(explicit.as_path()));
    assert_eq!(dataset.root_seed(), 7.into());
    assert!(dataset.has_calibrated_intrinsics());
    assert_eq!(
        dataset.camera_models(),
        [Camera::try_pinhole(200.0, 210.0, 40.0, 41.0).unwrap()]
    );
}

#[test]
fn from_dir_with_rejects_an_unusable_explicit_calibration() {
    let root = tempfile::TempDir::new().unwrap();
    noise_images(root.path(), 2);
    std::fs::write(root.path().join("K.txt"), "100 0 48\n0 100 47\n0 0 1\n").unwrap();
    let invalid = root.path().join("invalid.txt");
    std::fs::write(&invalid, "not a matrix\n").unwrap();

    let missing = DatasetConfig::default().with_intrinsics(root.path().join("missing.txt"));
    let malformed = DatasetConfig::default().with_intrinsics(&invalid);

    assert!(matches!(
        Dataset::from_dir_with(root.path(), &missing),
        Err(crate::SfmError::ReadFile { .. })
    ));
    assert!(matches!(
        Dataset::from_dir_with(root.path(), &malformed),
        Err(crate::SfmError::InvalidInput(_))
    ));
}

#[test]
fn engines_refine_only_estimated_intrinsics() {
    let estimated =
        ViewGraph::try_from_workspace(SfmWorkspace::builder().build().unwrap()).unwrap();
    let calibrated = estimated.clone().with_calibrated_intrinsics(true);

    for graph in [&estimated, &calibrated] {
        let refines = !graph.has_calibrated_intrinsics();
        let incremental = graph_config(Incremental::default().runner().config, graph);
        let global = graph_config(Global::default().runner().config, graph);
        assert_eq!(
            incremental.incremental().self_calibrate_intrinsics(),
            refines
        );
        assert_eq!(global.global().self_calibrate_intrinsics(), refines);
    }
}

#[test]
fn frontend_profiles_are_the_evaluated_lanes() {
    let lane = |preset, density| {
        SiftFrontendConfig::new(preset)
            .with_density(density)
            .with_max_dimension(1600)
    };

    let release = Frontend::default();
    assert_eq!(
        release.sift,
        lane(SiftFrontendPreset::Quality, crate::DetectorDensity::High)
    );
    assert_eq!(release.matching_method, MatchingMethod::Hnsw);
    assert!(release.radial_calibration);
}

#[test]
fn frontend_matches_with_hnsw_by_default_and_serializes_the_choice() {
    assert_eq!(Frontend::default().matching_method(), MatchingMethod::Hnsw);

    let brute_force = Frontend::default().with_matching_method(MatchingMethod::BruteForce);
    let json = serde_json::to_value(brute_force).unwrap();
    assert_eq!(json["matching_method"], "brute_force");
    let restored: Frontend = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(restored, brute_force);

    // Front-end JSON written before the key existed selects the default.
    let mut legacy = json;
    legacy.as_object_mut().unwrap().remove("matching_method");
    let restored: Frontend = serde_json::from_value(legacy).unwrap();
    assert_eq!(restored.matching_method(), MatchingMethod::Hnsw);
}

#[test]
fn frontend_pair_budget_is_off_by_default_and_serializes_the_choice() {
    assert_eq!(Frontend::default().max_pairs_per_view(), None);

    let filtered = Frontend::default().with_max_pairs_per_view(Some(15));
    assert_eq!(filtered.max_pairs_per_view(), Some(15));
    let json = serde_json::to_value(filtered).unwrap();
    assert_eq!(json["max_pairs_per_view"], 15);
    let restored: Frontend = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(restored, filtered);

    // Front-end JSON written before the key existed leaves the filter off.
    let mut legacy = json;
    legacy.as_object_mut().unwrap().remove("max_pairs_per_view");
    let restored: Frontend = serde_json::from_value(legacy).unwrap();
    assert_eq!(restored.max_pairs_per_view(), None);
}

#[test]
fn frontend_extracts_with_alp_by_default_and_serializes_the_choice() {
    assert_eq!(
        Frontend::default().extraction_method(),
        ExtractionMethod::Alp
    );
    assert_eq!(ExtractionMethod::default(), ExtractionMethod::Alp);

    let sift = Frontend::default().with_extraction_method(ExtractionMethod::Sift);
    let json = serde_json::to_value(sift).unwrap();
    assert_eq!(json["extraction_method"], "sift");
    let restored: Frontend = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(restored, sift);

    // Front-end JSON written before the key existed selects SIFT.
    let mut legacy = json;
    legacy.as_object_mut().unwrap().remove("extraction_method");
    let restored: Frontend = serde_json::from_value(legacy).unwrap();
    assert_eq!(restored.extraction_method(), ExtractionMethod::Sift);
}

#[test]
fn equirectangular_images_get_one_spherical_camera_per_width_and_no_guessed_focal() {
    let root = tempfile::TempDir::new().unwrap();
    let paths: Vec<_> = [(64, 32), (64, 32), (96, 48)]
        .into_iter()
        .enumerate()
        .map(|(index, (width, height))| {
            let path = root.path().join(format!("sphere-{index}.png"));
            image::GrayImage::new(width, height).save(&path).unwrap();
            path
        })
        .collect();
    let config = ImageDatasetConfig::default().with_camera_model(CameraModelType::Spherical);

    let prepared = prepare_image_paths(&paths, &config).unwrap();
    let dataset = prepared.build_dataset(3).unwrap();

    assert_eq!(dataset.camera_models().len(), 2);
    assert!(matches!(
        dataset.camera_models()[0],
        crate::Camera::Spherical(ref camera) if camera.width() == 64
    ));
    let views: Vec<_> = dataset
        .views()
        .iter()
        .map(|view| view.intrinsic_id)
        .collect();
    assert_eq!(views[0], views[1]);
    assert_ne!(views[0], views[2]);
    assert_eq!(
        dataset.intrinsic_prior(views[0]),
        Some(crate::model::IntrinsicPrior::Trusted)
    );

    let flat = root.path().join("flat.png");
    image::GrayImage::new(64, 48).save(&flat).unwrap();
    let error = prepare_image_paths(&[paths[0].clone(), flat], &config).unwrap_err();
    assert!(error.to_string().contains("equirectangular"), "{error}");
}

#[test]
fn images_without_a_camera_share_one_group_only_when_asked() {
    let root = tempfile::TempDir::new().unwrap();

    let separate = noise_dataset(root.path(), &ImageDatasetConfig::default());
    assert_eq!(separate.camera_models().len(), 3);

    let shared = noise_dataset(
        root.path(),
        &ImageDatasetConfig::default().with_share_unknown_cameras(true),
    );
    assert_eq!(shared.camera_models().len(), 1);
    assert!(
        shared
            .views()
            .iter()
            .all(|view| view.intrinsic_id == crate::IntrinsicId::new(0))
    );
    assert_eq!(
        shared.intrinsic_prior(crate::IntrinsicId::new(0)),
        Some(crate::model::IntrinsicPrior::Guess)
    );
}

#[cfg(feature = "unstable-internals")]
#[test]
fn frontend_extracts_with_the_selected_detector() {
    use crate::backends::{FeatureContext, FeatureExtractor};

    let root = tempfile::TempDir::new().unwrap();
    let paths = noise_images(root.path(), 2);
    let dataset = prepare_image_paths(&paths, &ImageDatasetConfig::default())
        .unwrap()
        .build_dataset(3)
        .unwrap();
    let meta = crate::model::ImageMeta::path(&paths[0], 96, 96);
    let ctx = FeatureContext {
        view: crate::ViewId::new(0),
        seed: dataset.root_seed(),
    };
    let frontend = Frontend::default();
    let sift = frontend.sift.extractor().extract(&ctx, &meta).unwrap();
    let alp = crate::adapters::features::AlpFeatureExtractor::new(frontend.sift.preset())
        .with_optional_max_dimension(frontend.sift.max_dimension())
        .extract(&ctx, &meta)
        .unwrap();
    assert_ne!(sift.keypoints().len(), alp.keypoints().len());

    for (detector, expected) in [
        (ExtractionMethod::Sift, &sift),
        (ExtractionMethod::Alp, &alp),
    ] {
        let mut workspace = dataset.workspace.clone();
        workspace
            .install_pair_graph_before_matching(
                crate::workspace::pair_graph::PairGraph::exhaustive(2).unwrap(),
            )
            .unwrap();
        // No noise pair passes verification, but the features stay installed.
        assert!(
            frontend
                .with_extraction_method(detector)
                .stage_with(&mut workspace, None, None, &())
                .is_err()
        );
        let installed = workspace.feature_set_store().get(0).unwrap();
        assert_eq!(installed.len(), expected.keypoints().len(), "{detector:?}");
    }
}

#[cfg(feature = "unstable-internals")]
#[test]
fn frontend_rejects_ultra_density_for_alp_before_extraction() {
    let root = tempfile::TempDir::new().unwrap();
    let paths = noise_images(root.path(), 2);
    let dataset = prepare_image_paths(&paths, &ImageDatasetConfig::default())
        .unwrap()
        .build_dataset(3)
        .unwrap();
    let mut workspace = dataset.workspace.clone();
    workspace
        .install_pair_graph_before_matching(
            crate::workspace::pair_graph::PairGraph::exhaustive(2).unwrap(),
        )
        .unwrap();
    let frontend = Frontend::default().with_density(crate::DetectorDensity::Ultra);
    let error = frontend
        .stage_with(&mut workspace, None, None, &())
        .unwrap_err();
    assert!(
        matches!(error, crate::SfmError::InvalidConfig(_)),
        "{error}"
    );
    assert!(workspace.feature_set_store().get(0).is_none());
}

#[cfg(feature = "unstable-internals")]
#[test]
fn frontend_applies_native_preset_density_and_shared_budget() {
    use openmvg_features::extraction::{
        AlpExtractor, AlpPreset, DescriptorStorage, FeatureBudgetPolicy, SiftExtractor, SiftPreset,
    };
    use openmvg_features::model::Keypoint;

    let root = tempfile::TempDir::new().unwrap();
    let paths = noise_images(root.path(), 2);
    let dataset = prepare_image_paths(&paths, &ImageDatasetConfig::default())
        .unwrap()
        .build_dataset(3)
        .unwrap();
    let (image, geometry) =
        crate::image::load_gray_image_max_dimension(&paths[0], Some(1600)).unwrap();
    for (method, budget) in [
        (ExtractionMethod::Alp, FeatureBudgetPolicy::strongest(8)),
        (ExtractionMethod::Sift, FeatureBudgetPolicy::strongest(8)),
        (ExtractionMethod::Alp, FeatureBudgetPolicy::strongest(0)),
        (ExtractionMethod::Sift, FeatureBudgetPolicy::strongest(0)),
    ] {
        let frontend = Frontend::default()
            .with_density(crate::DetectorDensity::Normal)
            .with_feature_budget(budget)
            .with_extraction_method(method);
        let native = match method {
            ExtractionMethod::Alp => AlpExtractor::with_preset(AlpPreset::Normal)
                .with_feature_budget(budget)
                .extract_u8(&image)
                .unwrap(),
            ExtractionMethod::Sift => SiftExtractor::with_preset(SiftPreset::Normal)
                .with_feature_budget(budget)
                .extract_u8(&image)
                .unwrap(),
        };
        let expected_geometry: Vec<_> = native
            .keypoints()
            .iter()
            .map(|keypoint| {
                let (x, y) = geometry.working_to_source(keypoint.x, keypoint.y);
                Keypoint::try_new(
                    x,
                    y,
                    geometry.working_length_to_source(keypoint.sigma),
                    keypoint.orientation,
                )
                .unwrap()
            })
            .collect();
        let expected_bytes: Vec<_> = native
            .descriptors()
            .iter()
            .flat_map(|descriptor| descriptor.as_slice().iter().copied())
            .collect();
        let mut workspace = dataset.workspace.clone();
        workspace
            .install_pair_graph_before_matching(
                crate::workspace::pair_graph::PairGraph::exhaustive(2).unwrap(),
            )
            .unwrap();
        // Feature records are installed before geometric-verification completion.
        let _ = frontend.stage_with(&mut workspace, None, None, &());
        let actual = workspace.feature_store().get(0).unwrap();
        assert_eq!(
            actual.keypoints(),
            expected_geometry.as_slice(),
            "{method:?} {budget:?}"
        );
        let (bytes, dimensions) = actual.descriptors().as_u8().unwrap();
        assert_eq!(dimensions, 128);
        assert_eq!(bytes, expected_bytes.as_slice(), "{method:?} {budget:?}");
    }
}

#[test]
fn prepared_engines_borrow_without_mutating_the_graph() {
    let graph = ViewGraph::try_from_workspace(SfmWorkspace::builder().build().unwrap()).unwrap();

    let incremental = Incremental::default().run_prepared(&graph);
    let global = Global::default().run_prepared(&graph);

    assert!(matches!(
        incremental,
        Err(crate::SfmError::MissingPrerequisite { .. })
    ));
    assert!(matches!(
        global,
        Err(crate::SfmError::MissingPrerequisite { .. })
    ));
    assert!(graph.validate().is_ok());
}

#[test]
fn frontend_rejects_images_without_verified_pairs() {
    let root = tempfile::TempDir::new().unwrap();
    let dataset = noise_dataset(root.path(), &ImageDatasetConfig::default());

    let error = Frontend::default().build(&dataset).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("none of the 3 image pairs passed geometric verification"),
        "{error}"
    );
}

#[test]
fn frontend_cancellation_at_every_notification_stops_before_the_next_stage() {
    let root = tempfile::TempDir::new().unwrap();
    let dataset = noise_dataset(root.path(), &ImageDatasetConfig::default());
    let complete = RecordingProgress::default();
    assert!(
        Frontend::default()
            .build_with_report(&dataset, &complete)
            .is_err()
    );

    // Cancellation wins over the verification failure the uncanceled run reports.
    for limit in 0..=complete.notification_count() {
        let progress = CancelAfter::new(limit);
        let error = Frontend::default()
            .build_with_report(&dataset, &progress)
            .unwrap_err();
        assert!(
            matches!(error, crate::SfmError::Canceled),
            "canceled after {limit} notifications: {error}"
        );
        assert!(!progress.began_after_cancellation(), "after {limit}");
    }
}
