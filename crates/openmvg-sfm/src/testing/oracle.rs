// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic fixtures and ground-truth-initialized incremental validation.
//!
//! These APIs isolate registration, triangulation, and bundle adjustment from
//! front-end and initial-pair estimation errors. They are available only through the
//! opt-in `unstable-test-support` feature and are not default reconstruction paths.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::Point2;
use serde::{Deserialize, Serialize};

use super::{SyntheticSceneBuilder, SyntheticSceneConfig};
use crate::evaluation::ground_truth::{GroundTruthReconstruction, GroundTruthScene};
use crate::model::ImageMeta;
use crate::{Camera, FeatureId, IntrinsicId, LandmarkId, Result, SfmError, ViewId, ViewPair};

pub use crate::pipelines::incremental::{
    IncrementalOracleSfmReport, incremental_sfm_oracle, incremental_sfm_oracle_with,
    initialize_incremental_seed_from_oracle,
};

/// One noise-free 2D observation of a ground-truth point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleObservation {
    /// Observing view.
    pub view: ViewId,
    /// Index of the point in the synthetic scene, before landmark renumbering.
    pub original_point_id: usize,
    /// Per-view feature index, assigned in increasing original-point order.
    pub feature_id: FeatureId,
    /// Projected position, in pixels.
    pub xy: Point2<f64>,
}

/// Ground-truth feature matches of one covisible view pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OraclePairMatches {
    /// Matched views, `first < second`.
    pub pair: ViewPair,
    /// `(first-view feature, second-view feature)` correspondences.
    pub matches: Vec<(FeatureId, FeatureId)>,
    /// Whether each entry of `matches` is geometrically correct; all `true` until
    /// [`OracleSfmFixture::inject_match_outliers`] corrupts the pair.
    pub inlier_mask: Vec<bool>,
}

/// Ground-truth track of one scene point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleTrack {
    /// Landmark id of the point in the fixture ground truth.
    pub landmark_id: LandmarkId,
    /// Index of the point in the synthetic scene, before landmark renumbering.
    pub original_point_id: usize,
    /// Observing `(view, feature)` pairs, sorted.
    pub observations: Vec<(ViewId, FeatureId)>,
}

/// Synthetic scene with exact correspondences, tracks, and ground truth for oracle runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleSfmFixture {
    /// Ground-truth scene; its pairs are the covisible view pairs.
    pub scene: GroundTruthScene,
    /// Ground-truth poses and points of `scene`.
    pub reconstruction_gt: GroundTruthReconstruction,
    /// Image metadata, indexed by view.
    pub image_meta: Vec<ImageMeta>,
    /// Camera models, indexed by intrinsic id.
    pub camera_models: Vec<Camera>,
    /// Ground-truth world-to-camera poses, sorted by view.
    pub poses_gt: Vec<(ViewId, openmvg_core::model::WorldToCameraPose)>,
    /// Ground-truth world points, sorted by landmark id.
    pub points_gt: Vec<(LandmarkId, nalgebra::Point3<f64>)>,
    /// Every observation, sorted by view, then original point.
    pub observations: Vec<OracleObservation>,
    /// View pairs sharing at least one point, sorted.
    pub pairs: Vec<ViewPair>,
    /// Matches of each pair in `pairs`, in the same order.
    pub matches: Vec<OraclePairMatches>,
    /// One track per ground-truth point, in landmark order.
    pub tracks: Vec<OracleTrack>,
}

/// Deterministic match corruption for end-to-end robustness tests.
///
/// `outlier_fraction` is applied independently to every ordinary pair. The optional
/// `poisoned_pair` is fully deranged, which tests whether verification removes a bad
/// view-graph edge rather than merely trimming individual correspondences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleMatchCorruptionConfig {
    /// Fraction of each ordinary pair's matches to derange, in `[0, 1]`, rounded per pair.
    pub outlier_fraction: f64,
    /// Pair whose matches are all deranged; `None` corrupts every pair uniformly.
    pub poisoned_pair: Option<ViewPair>,
}

impl OracleMatchCorruptionConfig {
    /// Corrupts every pair with the same outlier fraction.
    pub fn diffuse(outlier_fraction: f64) -> Self {
        Self {
            outlier_fraction,
            poisoned_pair: None,
        }
    }

    /// Corrupts every pair with `outlier_fraction` and fully deranges the pair `(view_i, view_j)`.
    pub fn with_poisoned_pair(outlier_fraction: f64, view_i: ViewId, view_j: ViewId) -> Self {
        Self {
            outlier_fraction,
            poisoned_pair: Some(ViewPair::new(view_i, view_j)),
        }
    }
}

/// Outliers injected into one pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OraclePairCorruptionReport {
    /// Corrupted pair.
    pub pair: ViewPair,
    /// Number of matches in the pair (unchanged by corruption).
    pub match_count: usize,
    /// Number of matches turned into outliers; never exactly 1.
    pub outlier_count: usize,
    /// Whether every match of a non-empty pair was deranged.
    pub fully_poisoned: bool,
}

/// Outliers injected by [`OracleSfmFixture::inject_match_outliers`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleMatchCorruptionReport {
    /// Matches over all pairs.
    pub total_matches: usize,
    /// Outliers injected over all pairs.
    pub total_outliers: usize,
    /// Per-pair breakdown, in fixture match order.
    pub pairs: Vec<OraclePairCorruptionReport>,
}

impl OracleSfmFixture {
    /// Replace selected right-image feature ids with a one-to-one cyclic derangement.
    ///
    /// The operation preserves match count and uniqueness while changing only geometric
    /// correctness. `OraclePairMatches::inlier_mask` remains aligned with `matches`, so a
    /// verifier can be scored with exact true/false-positive accounting. The fixture must
    /// be clean; refusing a second injection prevents labels from silently becoming stale.
    pub fn inject_match_outliers(
        &mut self,
        config: &OracleMatchCorruptionConfig,
    ) -> Result<OracleMatchCorruptionReport> {
        if !config.outlier_fraction.is_finite() || !(0.0..=1.0).contains(&config.outlier_fraction) {
            return Err(SfmError::InvalidInput(format!(
                "oracle match outlier fraction must be finite and in [0, 1], got {}",
                config.outlier_fraction
            )));
        }

        let poisoned_pair = config.poisoned_pair;
        if let Some(poisoned_pair) = poisoned_pair {
            let exists = self
                .matches
                .iter()
                .any(|pair_matches| pair_matches.pair == poisoned_pair);
            if !exists {
                return Err(SfmError::InvalidInput(format!(
                    "poisoned oracle pair ({}, {}) is not present",
                    poisoned_pair.first(),
                    poisoned_pair.second()
                )));
            }
        }

        if self.matches.iter().any(|pair_matches| {
            pair_matches.inlier_mask.len() != pair_matches.matches.len()
                || pair_matches.inlier_mask.iter().any(|is_inlier| !is_inlier)
        }) {
            return Err(SfmError::InvalidInput(
                "oracle match fixture is already corrupted or has stale labels".to_owned(),
            ));
        }

        // Plan and validate every pair before mutating any match. A malformed sparse
        // fixture must not be left half-corrupted when this method returns an error.
        let mut plans = Vec::with_capacity(self.matches.len());
        for pair_matches in &self.matches {
            let match_count = pair_matches.matches.len();
            let force_full = poisoned_pair.is_some_and(|pair| pair_matches.pair == pair);
            if force_full && match_count < 2 {
                return Err(SfmError::InvalidInput(format!(
                    "poisoned oracle pair ({}, {}) needs at least two matches",
                    pair_matches.pair.first(),
                    pair_matches.pair.second()
                )));
            }
            let mut outlier_count = if force_full {
                match_count
            } else {
                (config.outlier_fraction * match_count as f64).round() as usize
            }
            .min(match_count);
            // A one-element cyclic rotation is unchanged, so make any non-zero
            // corruption a genuine derangement when the pair has enough support.
            if outlier_count == 1 && match_count >= 2 {
                outlier_count = 2;
            }
            if outlier_count == 1 {
                return Err(SfmError::InvalidInput(format!(
                    "oracle pair ({}, {}) has only one match and cannot be deranged",
                    pair_matches.pair.first(),
                    pair_matches.pair.second()
                )));
            }
            plans.push((
                outlier_count,
                match_count > 0 && outlier_count == match_count,
            ));
        }

        let total_matches = self.matches.iter().map(|pair| pair.matches.len()).sum();
        let total_outliers = plans.iter().map(|(outliers, _)| outliers).sum();
        let mut pairs = Vec::with_capacity(self.matches.len());
        for (pair_matches, (outlier_count, fully_poisoned)) in self.matches.iter_mut().zip(plans) {
            let match_count = pair_matches.matches.len();

            if outlier_count > 0 {
                let donor_features: Vec<_> = pair_matches.matches[..outlier_count]
                    .iter()
                    .map(|(_, feature_j)| *feature_j)
                    .collect();
                for (index, (_, feature_j)) in
                    pair_matches.matches[..outlier_count].iter_mut().enumerate()
                {
                    *feature_j = donor_features[(index + 1) % outlier_count];
                    pair_matches.inlier_mask[index] = false;
                }
            }

            pairs.push(OraclePairCorruptionReport {
                pair: pair_matches.pair,
                match_count,
                outlier_count,
                fully_poisoned,
            });
        }

        Ok(OracleMatchCorruptionReport {
            total_matches,
            total_outliers,
            pairs,
        })
    }
}

/// Builds the fixed noise-free oracle fixture: 5 pinhole views of 40 points, seed 7.
///
/// # Errors
///
/// Returns [`SfmError::InvalidInput`] if the synthetic scene or its camera cannot be built.
pub fn synthetic_oracle_fixture() -> Result<OracleSfmFixture> {
    let synthetic = SyntheticSceneBuilder::new()
        .with_cameras(5)
        .with_points(40)
        .with_seed(7)
        .with_noise(0.0)
        .build()
        .map_err(|error| SfmError::InvalidInput(error.to_string()))?;
    let config: SyntheticSceneConfig = synthetic.config.clone();

    let mut valid_points = synthetic.points.clone();
    valid_points.sort_by_key(|(track_id, _)| *track_id);
    let valid_original_ids: Vec<_> = valid_points
        .iter()
        .map(|(track_id, _)| track_id.index())
        .collect();
    let valid_set: BTreeSet<_> = valid_original_ids.iter().copied().collect();

    let mut original_to_landmark = BTreeMap::new();
    let mut points_gt = Vec::with_capacity(valid_original_ids.len());
    for (landmark_index, (track_id, point)) in valid_points.iter().enumerate() {
        let original_point_id = track_id.index();
        let landmark_id = LandmarkId::new(landmark_index);
        original_to_landmark.insert(original_point_id, landmark_id);
        points_gt.push((landmark_id, *point));
    }

    let mut poses_gt = synthetic.poses.clone();
    poses_gt.sort_by_key(|(view_id, _)| *view_id);

    let image_meta: Vec<_> = (0..config.num_cameras)
        .map(|view_id| {
            ImageMeta::external_id(
                format!("synthetic/{:03}.png", view_id),
                config.image_size.0 as u32,
                config.image_size.1 as u32,
            )
        })
        .collect();
    let camera_models = vec![
        Camera::try_pinhole(
            config.focal_length,
            config.focal_length,
            config.image_size.0 as f64 / 2.0,
            config.image_size.1 as f64 / 2.0,
        )
        .map_err(|error| SfmError::InvalidInput(error.to_string()))?,
    ];

    let mut filtered_observations: Vec<_> = synthetic
        .observations
        .iter()
        .filter(|((_, original_point_id), _)| valid_set.contains(&original_point_id.index()))
        .map(|((view_id, original_point_id), xy)| (view_id.index(), original_point_id.index(), *xy))
        .collect();
    filtered_observations
        .sort_by_key(|(view_id, original_point_id, _)| (*view_id, *original_point_id));

    let mut feature_counts: BTreeMap<usize, usize> = BTreeMap::new();
    let mut feature_by_observation = BTreeMap::new();
    let mut observations = Vec::with_capacity(filtered_observations.len());
    for (view_id, original_point_id, xy) in filtered_observations {
        let next_feature = feature_counts.entry(view_id).or_insert(0);
        let feature_id = FeatureId::new(*next_feature);
        *next_feature += 1;
        feature_by_observation.insert((view_id, original_point_id), feature_id);
        observations.push(OracleObservation {
            view: ViewId::new(view_id),
            original_point_id,
            feature_id,
            xy,
        });
    }

    let mut tracks = Vec::with_capacity(valid_original_ids.len());
    let mut covisible: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for original_point_id in valid_original_ids {
        let mut track_observations: Vec<_> = observations
            .iter()
            .filter(|obs| obs.original_point_id == original_point_id)
            .map(|obs| (obs.view, obs.feature_id))
            .collect();
        track_observations.sort_by_key(|(view_id, feature_id)| (*view_id, *feature_id));

        for i in 0..track_observations.len() {
            for j in (i + 1)..track_observations.len() {
                let view_i = track_observations[i].0.index();
                let view_j = track_observations[j].0.index();
                covisible
                    .entry((view_i.min(view_j), view_i.max(view_j)))
                    .or_default()
                    .push(original_point_id);
            }
        }

        tracks.push(OracleTrack {
            landmark_id: original_to_landmark[&original_point_id],
            original_point_id,
            observations: track_observations,
        });
    }

    let pairs: Vec<_> = covisible
        .keys()
        .map(|(a, b)| ViewPair::new(ViewId::new(*a), ViewId::new(*b)))
        .collect();
    let mut matches = Vec::with_capacity(pairs.len());
    for (&(view_a, view_b), original_point_ids) in &covisible {
        let pair = ViewPair::new(ViewId::new(view_a), ViewId::new(view_b));
        let mut pair_matches = Vec::with_capacity(original_point_ids.len());
        for original_point_id in original_point_ids {
            let feature_a = feature_by_observation[&(pair.first().index(), *original_point_id)];
            let feature_b = feature_by_observation[&(pair.second().index(), *original_point_id)];
            pair_matches.push((feature_a, feature_b));
        }
        matches.push(OraclePairMatches {
            pair,
            inlier_mask: vec![true; pair_matches.len()],
            matches: pair_matches,
        });
    }

    let scene = GroundTruthScene {
        image_meta: image_meta
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, meta)| (ViewId::new(index), meta))
            .collect(),
        camera_models: camera_models
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, camera)| (IntrinsicId::new(index), camera))
            .collect(),
        poses: poses_gt.clone(),
        points: points_gt.clone(),
        pairs: pairs.clone(),
    };
    let reconstruction_gt = scene.reconstruction();

    Ok(OracleSfmFixture {
        scene,
        reconstruction_gt,
        image_meta,
        camera_models,
        poses_gt,
        points_gt,
        observations,
        pairs,
        matches,
        tracks,
    })
}
