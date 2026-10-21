// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reconstruction quality of a workspace: reprojection and bearing error statistics, track
//! coverage, and the worst-supported tracks.

use crate::ops::reconstruction::CameraResidualPolicy;
use crate::two_view::ResidualUnit;
use crate::{Result, SfmError, SfmWorkspace, TrackId, ViewPair};
use openmvg_core::camera::CameraModel;

/// Fraction of samples at or below a threshold — the error CDF evaluated there.
///
/// This is accuracy-at-threshold, not AUC or mAA, and is not comparable with published
/// mAA numbers: because the CDF is monotone, accuracy-at-threshold is always >= the mAA
/// that published benchmarks such as IMC report. Compare with those numbers only through an
/// mAA computed by the evaluation harness.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ThresholdRate {
    /// Error threshold, in the units of the distribution it summarizes.
    pub threshold: f64,
    /// Fraction of samples at or below `threshold`, in `[0, 1]`; 0 for an empty distribution.
    pub pass_rate: f64,
}

/// Summary statistics for an error distribution.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ErrorStats {
    /// Number of finite samples summarized.
    pub count: usize,
    /// Smallest sample; 0 when `count` is 0.
    pub min: f64,
    /// Largest sample; 0 when `count` is 0.
    pub max: f64,
    /// Arithmetic mean; 0 when `count` is 0.
    pub mean: f64,
    /// Median (mean of the two middle samples for an even count); 0 when `count` is 0.
    pub median: f64,
    /// Root mean square of the samples; 0 when `count` is 0.
    pub rmse: f64,
    /// Error CDF at each requested threshold, in increasing threshold order.
    pub threshold_pass_rates: Vec<ThresholdRate>,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl ErrorStats {
    /// Accuracy-at-threshold rates. Renamed from `auc()`, which it never was.
    pub fn threshold_pass_rates(&self) -> &[ThresholdRate] {
        &self.threshold_pass_rates
    }

    /// Validate a non-negative error distribution and its threshold CDF samples.
    pub fn validate(&self) -> crate::Result<()> {
        let values = [self.min, self.max, self.mean, self.median, self.rmse];
        let ordered = self.min <= self.mean
            && self.mean <= self.max
            && self.min <= self.median
            && self.median <= self.max;
        let empty_is_zero = self.count != 0 || values.iter().all(|value| *value == 0.0);
        let invalid_rates = self.threshold_pass_rates.iter().any(|rate| {
            !rate.threshold.is_finite()
                || rate.threshold < 0.0
                || !rate.pass_rate.is_finite()
                || !(0.0..=1.0).contains(&rate.pass_rate)
                || self.count == 0 && rate.pass_rate != 0.0
        });
        let invalid_cdf = self.threshold_pass_rates.windows(2).any(|rates| {
            rates[0].threshold >= rates[1].threshold || rates[0].pass_rate > rates[1].pass_rate
        });
        let tolerance = 1e-12 * self.max.max(1.0);
        let singleton_is_exact = self.count != 1
            || values
                .windows(2)
                .all(|pair| (pair[0] - pair[1]).abs() <= tolerance);
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || !ordered
            || !empty_is_zero
            || self.rmse + tolerance < self.mean
            || self.rmse > self.max + tolerance
            || !singleton_is_exact
            || invalid_rates
            || invalid_cdf
        {
            return Err(crate::SfmError::InvalidInput(
                "error statistics are not a valid non-negative distribution".to_owned(),
            ));
        }
        Ok(())
    }

    /// Build and validate statistics without discarding invalid samples.
    pub fn try_from_values(values: &[f64], thresholds: &[f64]) -> crate::Result<Self> {
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(crate::SfmError::InvalidInput(
                "error samples must be finite and non-negative".to_owned(),
            ));
        }
        let stats = Self::from_values(values, thresholds);
        stats.validate()?;
        Ok(stats)
    }

    /// Build stats from error values and threshold list.
    pub fn from_values(values: &[f64], thresholds: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                rmse: 0.0,
                threshold_pass_rates: thresholds
                    .iter()
                    .map(|&threshold| ThresholdRate {
                        threshold,
                        pass_rate: 0.0,
                    })
                    .collect(),
            };
        }

        let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(f64::total_cmp);

        if sorted.is_empty() {
            return Self::from_values(&[], thresholds);
        }

        let count = sorted.len();
        let mean = sorted.iter().enumerate().fold(0.0, |mean, (index, value)| {
            mean + (value - mean) / (index + 1) as f64
        });
        let median = if count.is_multiple_of(2) {
            let lower = sorted[count / 2 - 1];
            lower + (sorted[count / 2] - lower) * 0.5
        } else {
            sorted[count / 2]
        };
        let max = sorted[count - 1];
        let rmse = if max == 0.0 {
            0.0
        } else {
            max * (sorted
                .iter()
                .map(|value| (value / max).powi(2))
                .sum::<f64>()
                / count as f64)
                .sqrt()
        };
        let threshold_pass_rates = thresholds
            .iter()
            .map(|&threshold| {
                let n_in = sorted.iter().filter(|&&error| error <= threshold).count();
                ThresholdRate {
                    threshold,
                    pass_rate: n_in as f64 / count as f64,
                }
            })
            .collect();

        Self {
            count,
            min: sorted[0],
            max,
            mean,
            median,
            rmse,
            threshold_pass_rates,
        }
    }
}

/// Camera-model-independent landmark quality computed from calibrated bearing errors.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct BearingQualityReport {
    /// Distribution of per-landmark mean bearing errors, in radians.
    pub mean_error_stats_rad: ErrorStats,
    /// Distribution of per-landmark maximum bearing errors, in radians.
    pub max_error_stats_rad: ErrorStats,
    /// Active landmarks whose exact supporting geometry could not be evaluated.
    pub invalid_landmark_count: usize,
    /// Largest per-landmark maximum bearing errors, keyed by source track.
    pub top_offending_tracks: Vec<(TrackId, f64)>,
}

impl Default for BearingQualityReport {
    fn default() -> Self {
        Self {
            mean_error_stats_rad: ErrorStats::from_values(&[], &[]),
            max_error_stats_rad: ErrorStats::from_values(&[], &[]),
            invalid_landmark_count: 0,
            top_offending_tracks: Vec::new(),
        }
    }
}

/// Workspace-wide coverage and accuracy summary, computed from the current workspace state.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct QualityReport {
    /// Number of views in the workspace.
    pub view_count: usize,
    /// Number of views holding extracted features; 0 once matching data is released.
    #[serde(default)]
    pub featured_view_count: usize,
    /// Number of views whose pose is registered.
    pub registered_view_count: usize,
    /// `registered_view_count / view_count`, or 0 for an empty workspace.
    pub registered_fraction: f64,
    /// Number of candidate pairs, active or deactivated.
    pub pair_count: usize,
    /// Number of pairs holding putative matches; 0 once matching data is released.
    #[serde(default)]
    pub matched_pair_count: usize,
    /// Whether descriptors and raw matches had been released when the report was computed.
    #[serde(default)]
    pub matching_data_released: bool,
    /// Number of pairs that have not been deactivated.
    pub active_pair_count: usize,
    /// `active_pair_count / pair_count`, or 0 without pairs.
    pub active_pair_coverage: f64,
    /// Number of pairs with an accepted two-view geometric model.
    pub verified_pair_count: usize,
    /// `verified_pair_count / pair_count`, or 0 without pairs.
    pub verified_pair_coverage: f64,
    /// Distribution of verified-pair inlier ratios (in `[0, 1]`), with pass rates at 0.25, 0.5,
    /// and 0.75.
    pub inlier_ratio_stats: ErrorStats,
    /// Number of feature tracks.
    pub track_count: usize,
    /// Distribution of track lengths in observations, with pass rates at 2, 3, and 4.
    pub track_length_stats: ErrorStats,
    /// Number of active landmarks.
    pub landmark_count: usize,
    /// Unit of [`Self::observation_reprojection_error_stats`],
    /// [`Self::landmark_reprojection_error_stats`], and the [`Self::top_offending_tracks`]
    /// scores.
    pub reprojection_error_units: ResidualUnit,
    /// Distribution of reprojection errors over the observations that support the active
    /// landmarks, with pass rates at 1, 2, and 4 units. Its `rmse` is the reprojection RMSE the
    /// bundle adjustment minimizes and the one other SfM tools report. Observations whose
    /// residual cannot be evaluated are left out.
    pub observation_reprojection_error_stats: ErrorStats,
    /// Distribution of per-landmark mean reprojection errors over active landmarks, with pass
    /// rates at 1, 2, and 4 units. Averaging within a landmark hides large single-observation
    /// errors, so its `rmse` is lower than the per-observation one; use it to rank landmarks and
    /// tracks, not to compare with other tools.
    pub landmark_reprojection_error_stats: ErrorStats,
    /// Camera-model-independent quality.
    pub bearing_quality: BearingQualityReport,
    /// Lowest-inlier accepted pairs, stored as `(view_pair, 1 - inlier_ratio)`.
    pub top_offending_pairs: Vec<(ViewPair, f64)>,
    /// Highest-error active landmarks, keyed by their source track. Scores are
    /// mean reprojection residuals in [`Self::reprojection_error_units`].
    pub top_offending_tracks: Vec<(TrackId, f64)>,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl QualityReport {
    /// Validate count domains and normalized coverage values.
    pub fn validate(&self) -> Result<()> {
        validate_count(self.featured_view_count, self.view_count, "featured images")?;
        validate_count(
            self.registered_view_count,
            self.view_count,
            "registered images",
        )?;
        validate_count(self.matched_pair_count, self.pair_count, "matched pairs")?;
        if self.matching_data_released
            && (self.featured_view_count != 0 || self.matched_pair_count != 0)
        {
            return Err(SfmError::InvalidInput(
                "quality report marks matching data released but retained counts are non-zero"
                    .to_owned(),
            ));
        }
        validate_count(self.active_pair_count, self.pair_count, "active pairs")?;
        validate_count(self.verified_pair_count, self.pair_count, "verified pairs")?;
        validate_count(self.landmark_count, self.track_count, "landmarks")?;
        validate_fraction(
            self.registered_fraction,
            fraction(self.registered_view_count, self.view_count),
            "registered image fraction",
        )?;
        validate_fraction(
            self.active_pair_coverage,
            fraction(self.active_pair_count, self.pair_count),
            "active pair coverage",
        )?;
        validate_fraction(
            self.verified_pair_coverage,
            fraction(self.verified_pair_count, self.pair_count),
            "verified pair coverage",
        )?;
        validate_stats(
            &self.inlier_ratio_stats,
            self.verified_pair_count,
            "inlier ratios",
            Some(1.0),
        )?;
        validate_stats(
            &self.track_length_stats,
            self.track_count,
            "track lengths",
            None,
        )?;
        validate_stats(
            &self.landmark_reprojection_error_stats,
            self.landmark_count,
            "landmark reprojection errors",
            None,
        )?;
        self.observation_reprojection_error_stats.validate()?;
        validate_count(
            self.bearing_quality.invalid_landmark_count,
            self.landmark_count,
            "invalid bearing-quality landmarks",
        )?;
        let valid_bearing_landmarks = self
            .landmark_count
            .saturating_sub(self.bearing_quality.invalid_landmark_count);
        validate_stats(
            &self.bearing_quality.mean_error_stats_rad,
            valid_bearing_landmarks,
            "mean bearing errors",
            Some(std::f64::consts::PI),
        )?;
        validate_stats(
            &self.bearing_quality.max_error_stats_rad,
            valid_bearing_landmarks,
            "maximum bearing errors",
            Some(std::f64::consts::PI),
        )?;
        for &(pair, score) in &self.top_offending_pairs {
            if pair.first() == pair.second()
                || pair.first().index() >= self.view_count
                || pair.second().index() >= self.view_count
                || !score.is_finite()
                || !(0.0..=1.0).contains(&score)
            {
                return Err(SfmError::InvalidInput(
                    "quality report contains an invalid offending-pair score".to_owned(),
                ));
            }
        }
        for &(track, score) in &self.top_offending_tracks {
            if track.index() >= self.track_count || !score.is_finite() || score < 0.0 {
                return Err(SfmError::InvalidInput(
                    "quality report contains an invalid offending-track score".to_owned(),
                ));
            }
        }
        for &(track, score) in &self.bearing_quality.top_offending_tracks {
            if track.index() >= self.track_count
                || !score.is_finite()
                || !(0.0..=std::f64::consts::PI).contains(&score)
            {
                return Err(SfmError::InvalidInput(
                    "quality report contains an invalid offending-bearing score".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
fn bearing_quality_report(workspace: &SfmWorkspace) -> BearingQualityReport {
    let mut mean_errors = Vec::with_capacity(workspace.landmarks().active_count());
    let mut max_errors = Vec::with_capacity(workspace.landmarks().active_count());
    let mut top_offenders = Vec::with_capacity(workspace.landmarks().active_count());
    let mut invalid_landmark_count = 0usize;

    'landmarks: for landmark in workspace.landmarks().iter_active() {
        let Some(track) = workspace.tracks().get(landmark.source_track) else {
            invalid_landmark_count += 1;
            continue;
        };
        let mut sum = 0.0;
        let mut maximum = 0.0_f64;
        let mut count = 0usize;
        for (_, observation) in landmark.supporting_observations(track) {
            let Some(pose) = workspace.pose(observation.view) else {
                invalid_landmark_count += 1;
                continue 'landmarks;
            };
            let Some(image) = workspace.view(observation.view) else {
                invalid_landmark_count += 1;
                continue 'landmarks;
            };
            let Some(camera) = workspace.camera_model(image.intrinsic_id) else {
                invalid_landmark_count += 1;
                continue 'landmarks;
            };
            let camera_point = pose.pose.project_to_camera(&landmark.point);
            let Ok(pixel) = workspace.track_observation_pixel(observation) else {
                invalid_landmark_count += 1;
                continue 'landmarks;
            };
            let Ok(error) = camera.bearing_error_rad(&camera_point.coords, &pixel) else {
                invalid_landmark_count += 1;
                continue 'landmarks;
            };
            sum += error;
            maximum = maximum.max(error);
            count += 1;
        }
        if count == 0 {
            invalid_landmark_count += 1;
            continue;
        }
        mean_errors.push(sum / count as f64);
        max_errors.push(maximum);
        top_offenders.push((landmark.source_track, maximum));
    }

    top_offenders.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    top_offenders.truncate(5);
    let thresholds = [
        0.25_f64.to_radians(),
        0.5_f64.to_radians(),
        1.0_f64.to_radians(),
    ];
    BearingQualityReport {
        mean_error_stats_rad: ErrorStats::from_values(&mean_errors, &thresholds),
        max_error_stats_rad: ErrorStats::from_values(&max_errors, &thresholds),
        invalid_landmark_count,
        top_offending_tracks: top_offenders,
    }
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
fn validate_count(observed: usize, domain: usize, label: &str) -> Result<()> {
    if observed > domain {
        return Err(SfmError::InvalidInput(format!(
            "quality report has {observed} {label} in a domain of {domain}"
        )));
    }
    Ok(())
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
fn validate_fraction(observed: f64, expected: f64, label: &str) -> Result<()> {
    if !observed.is_finite()
        || !(0.0..=1.0).contains(&observed)
        || (observed - expected).abs() > 1e-12
    {
        return Err(SfmError::InvalidInput(format!(
            "quality report has inconsistent {label}: {observed}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
fn validate_stats(
    stats: &ErrorStats,
    expected_count: usize,
    label: &str,
    upper_bound: Option<f64>,
) -> Result<()> {
    stats.validate()?;
    if stats.count != expected_count || upper_bound.is_some_and(|bound| stats.max > bound) {
        return Err(SfmError::InvalidInput(format!(
            "quality report contains invalid {label} statistics"
        )));
    }
    Ok(())
}

/// Reprojection error of every observation that supports an active landmark, in pixel-equivalent
/// units. Observations without a pose, camera, or finite residual are skipped.
fn observation_reprojection_errors(workspace: &SfmWorkspace) -> Vec<f64> {
    let mut errors = Vec::new();
    for landmark in workspace.landmarks().iter_active() {
        let Some(track) = workspace.tracks().get(landmark.source_track) else {
            continue;
        };
        for (_, observation) in landmark.supporting_observations(track) {
            let (Some(pose), Some(image)) = (
                workspace.pose(observation.view),
                workspace.view(observation.view),
            ) else {
                continue;
            };
            let Some(camera) = workspace.camera_model(image.intrinsic_id) else {
                continue;
            };
            let Ok(policy) = CameraResidualPolicy::from_pixel_limit(camera, f64::INFINITY) else {
                continue;
            };
            let Ok(pixel) = workspace.track_observation_pixel(observation) else {
                continue;
            };
            let camera_point = pose.pose.project_to_camera(&landmark.point);
            if let Some(evaluation) = policy.evaluate(camera, &camera_point.coords, pixel) {
                errors.push(evaluation.pixel_equivalent);
            }
        }
    }
    errors
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub(crate) fn quality_report(workspace: &SfmWorkspace) -> QualityReport {
    let view_count = workspace.views().len();
    let featured_view_count = workspace.keypoint_store().populated_count();
    let registered_view_count = workspace.registered_view_count();
    let pair_count = workspace.pair_graph().len();
    let matched_pair_count = workspace.match_store().populated_count();
    let active_pair_count = workspace.pair_graph().active_count();
    let verified_pair_count = workspace.geometric_model_store().populated_count();

    let mut inlier_ratios = Vec::new();
    let mut pair_scores = Vec::new();
    for (pair_index, model) in workspace.geometric_model_store().iter_populated() {
        let ratio = model.evidence().inlier_ratio;
        if ratio.is_finite() {
            inlier_ratios.push(ratio);
            if let Some(pair) = workspace.pair_graph().pair(crate::PairId::new(pair_index)) {
                pair_scores.push((pair, 1.0 - ratio));
            }
        }
    }
    pair_scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pair_scores.truncate(5);

    let track_lengths: Vec<f64> = workspace
        .tracks()
        .as_slice()
        .iter()
        .map(|track| track.observations().len() as f64)
        .collect();
    // Only Active landmarks count toward quality — Rejected landmarks are a soft-delete
    // kept in the vec (so BA can reference ids); their stale points must not inflate the
    // landmark count or pollute reprojection stats.
    let mut track_scores: Vec<(TrackId, f64)> = workspace
        .landmarks()
        .iter_active()
        .map(|landmark| (landmark.source_track, landmark.reprojection_error.value()))
        .collect();
    let landmark_errors: Vec<f64> = track_scores.iter().map(|(_, error)| *error).collect();
    let observation_errors = observation_reprojection_errors(workspace);
    track_scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    track_scores.truncate(5);

    QualityReport {
        view_count,
        featured_view_count,
        registered_view_count,
        registered_fraction: fraction(registered_view_count, view_count),
        pair_count,
        matched_pair_count,
        matching_data_released: workspace.is_matching_data_released(),
        active_pair_count,
        active_pair_coverage: fraction(active_pair_count, pair_count),
        verified_pair_count,
        verified_pair_coverage: fraction(verified_pair_count, pair_count),
        inlier_ratio_stats: ErrorStats::from_values(&inlier_ratios, &[0.25, 0.5, 0.75]),
        track_count: workspace.tracks().len(),
        track_length_stats: ErrorStats::from_values(&track_lengths, &[2.0, 3.0, 4.0]),
        landmark_count: workspace.landmarks().active_count(),
        reprojection_error_units: ResidualUnit::Pixels,
        observation_reprojection_error_stats: ErrorStats::from_values(
            &observation_errors,
            &[1.0, 2.0, 4.0],
        ),
        landmark_reprojection_error_stats: ErrorStats::from_values(
            &landmark_errors,
            &[1.0, 2.0, 4.0],
        ),
        bearing_quality: bearing_quality_report(workspace),
        top_offending_pairs: pair_scores,
        top_offending_tracks: track_scores,
    }
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
fn fraction(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod quality_tests {
    use super::*;
    use crate::model::{ImageMeta, Landmark, Landmarks, ObservationSupport, PoseRecord};
    use crate::{
        Camera, FeatureId, IntrinsicId, LandmarkId, Track, TrackObservation, TrackSet, ViewId,
    };
    use nalgebra::{Point2, Point3, Rotation3};
    use openmvg_core::model::WorldToCameraPose;

    fn observation(image: usize, feature: usize) -> TrackObservation {
        TrackObservation {
            view: ViewId::new(image),
            feature: FeatureId::new(feature),
        }
    }

    fn exact_bearing_workspace(point: Point3<f64>) -> SfmWorkspace {
        let mut workspace = SfmWorkspace::builder()
            .image_meta([
                ImageMeta::external_id("a", 64, 48),
                ImageMeta::external_id("b", 64, 48),
            ])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        let poses = [
            WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::origin())
                .unwrap(),
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(1.0, 0.0, 0.0),
            )
            .unwrap(),
        ];
        let observed_pixels = poses
            .iter()
            .map(|pose| {
                let camera_point = pose.project_to_camera(&point);
                workspace
                    .camera_model(IntrinsicId::new(0))
                    .unwrap()
                    .project(&camera_point.coords)
                    .unwrap_or(Point2::new(32.0, 24.0))
            })
            .collect::<Vec<_>>();
        let observations = observed_pixels
            .iter()
            .enumerate()
            .map(|(index, _)| TrackObservation {
                view: ViewId::new(index),
                feature: FeatureId::new(0),
            })
            .collect::<Vec<_>>();
        for (index, pixel) in observed_pixels.into_iter().enumerate() {
            workspace
                .feature_sets
                .set(index, crate::FeatureSet::try_new(vec![pixel]).unwrap())
                .unwrap();
        }
        workspace.tracks =
            TrackSet::try_new(vec![Track::try_new(TrackId::new(0), observations).unwrap()])
                .unwrap();
        workspace.landmarks = Landmarks::try_new(vec![Landmark::active(
            LandmarkId::new(0),
            TrackId::new(0),
            point,
            openmvg_core::PixelResidual::ZERO,
            ObservationSupport::default(),
        )])
        .unwrap();
        for (index, pose) in poses.into_iter().enumerate() {
            workspace
                .set_pose(PoseRecord {
                    id: crate::PoseId::new(index),
                    pose,
                    component: None,
                })
                .unwrap();
        }
        workspace
    }

    #[test]
    fn observation_statistics_see_the_errors_a_landmark_mean_averages_away() {
        let mut workspace = exact_bearing_workspace(Point3::new(0.0, 0.0, 5.0));
        // One observation is exact and the other is 2 px off, so the landmark's mean error is
        // 1 px while the observation errors are 0 and 2 px.
        let off = TrackObservation {
            view: ViewId::new(1),
            feature: FeatureId::new(0),
        };
        let pixel = workspace.track_observation_pixel(&off).unwrap();
        workspace
            .feature_sets
            .set(
                1,
                crate::FeatureSet::try_new(vec![Point2::new(pixel.x + 2.0, pixel.y)]).unwrap(),
            )
            .unwrap();
        workspace.landmarks = Landmarks::try_new(vec![Landmark::active(
            LandmarkId::new(0),
            TrackId::new(0),
            Point3::new(0.0, 0.0, 5.0),
            openmvg_core::PixelResidual::try_new(1.0).unwrap(),
            ObservationSupport::default(),
        )])
        .unwrap();

        let report = quality_report(&workspace);

        let observations = &report.observation_reprojection_error_stats;
        assert_eq!(observations.count, 2);
        assert!((observations.mean - 1.0).abs() < 1e-9);
        assert!((observations.rmse - 2.0_f64.sqrt()).abs() < 1e-9);
        let landmarks = &report.landmark_reprojection_error_stats;
        assert_eq!(landmarks.count, 1);
        assert!((landmarks.rmse - 1.0).abs() < 1e-12);
        report.validate().unwrap();
    }

    #[test]
    fn bearing_quality_is_exact_and_invalid_geometry_fails_closed() {
        let exact = quality_report(&exact_bearing_workspace(Point3::new(0.0, 0.0, 5.0)));
        assert_eq!(exact.bearing_quality.mean_error_stats_rad.count, 1);
        assert_eq!(exact.bearing_quality.max_error_stats_rad.count, 1);
        assert_eq!(exact.bearing_quality.invalid_landmark_count, 0);
        assert!(exact.bearing_quality.max_error_stats_rad.max < 1e-14);

        let invalid = quality_report(&exact_bearing_workspace(Point3::new(0.0, 0.0, -5.0)));
        assert_eq!(invalid.bearing_quality.mean_error_stats_rad.count, 0);
        assert_eq!(invalid.bearing_quality.invalid_landmark_count, 1);
    }

    #[test]
    fn offending_tracks_are_ranked_by_geometry_not_length() {
        let mut workspace = SfmWorkspace::builder()
            .image_meta(vec![
                ImageMeta::external_id("a", 64, 48),
                ImageMeta::external_id("b", 64, 48),
                ImageMeta::external_id("c", 64, 48),
            ])
            .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        workspace.tracks = TrackSet::try_new(vec![
            Track::try_new(TrackId::new(0), vec![observation(0, 0), observation(1, 0)]).unwrap(),
            Track::try_new(
                TrackId::new(1),
                vec![observation(0, 1), observation(1, 1), observation(2, 1)],
            )
            .unwrap(),
        ])
        .unwrap();
        workspace.landmarks = Landmarks::try_new(vec![
            Landmark::active(
                LandmarkId::new(0),
                TrackId::new(0),
                Point3::new(0.0, 0.0, 5.0),
                openmvg_core::PixelResidual::try_new(0.5).unwrap(),
                ObservationSupport::default(),
            ),
            Landmark::active(
                LandmarkId::new(1),
                TrackId::new(1),
                Point3::new(1.0, 0.0, 5.0),
                openmvg_core::PixelResidual::try_new(8.0).unwrap(),
                ObservationSupport::default(),
            ),
        ])
        .unwrap();

        let report = quality_report(&workspace);

        assert_eq!(
            report.top_offending_tracks,
            vec![(TrackId::new(1), 8.0), (TrackId::new(0), 0.5)]
        );
        assert!(
            report
                .top_offending_tracks
                .iter()
                .all(|(_, score)| score.is_finite())
        );
    }

    #[test]
    fn empty_track_collection_has_no_synthetic_infinite_offenders() {
        let workspace = SfmWorkspace::builder().build().unwrap();
        let report = quality_report(&workspace);

        assert!(report.top_offending_tracks.is_empty());
        assert_eq!(report.track_length_stats.count, 0);
    }

    #[test]
    fn quality_validation_rejects_counts_outside_their_domain() {
        let workspace = SfmWorkspace::builder().build().unwrap();
        let mut report = quality_report(&workspace);
        report.registered_view_count = 1;

        assert!(report.validate().is_err());
    }
}
