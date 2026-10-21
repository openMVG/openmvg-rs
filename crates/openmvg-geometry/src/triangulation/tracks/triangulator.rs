// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pair-hypothesis consensus, refit, and acceptance for one track.

use crate::triangulation::{
    TwoViewMethod, parallax_angle, refine_angular_nview, triangulate_two_view,
};
use nalgebra::Point3;
use openmvg_core::camera::{Camera, CameraModel};
use openmvg_core::model::ViewId;
use openmvg_core::{AngularResidual, PixelResidual};

use super::config::TriangulationPreset;
use super::observation::{
    PreparedObservation, prepare_observations, prepare_resolved_observations,
};
use super::{
    ConsensusConfig, ConsensusThreshold, ResolvedTrackObservation, TrackTriangulationConfig,
    TrackTriangulationRejection, TriangulatedTrack, TriangulationContext, TriangulationError,
    TriangulationObservation, TriangulationPolicy, TriangulationQuality,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PairHypothesis {
    pub(super) first: usize,
    pub(super) second: usize,
    pub(super) parallax: f64,
}

/// Triangulate a whole track, with a robust policy for observations that can be wrong or a
/// plain one for clean data.
///
/// # How it works
///
/// Robust hypotheses use two observations, the calibrated geometric minimum.
/// Acceptance is separate: callers may still require three or more supporting
/// observations through [`ConsensusConfig::min_inliers`]. This distinction lets
/// a three-observation track recover from one outlier, which a three-view
/// hypothesis cannot do.
///
/// ```text
/// track observations -> pair sampler -> solver hypothesis -> residual score
///                    -> inlier consensus -> geometric refit -> landmark
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TrackTriangulator {
    config: TrackTriangulationConfig,
}

impl TrackTriangulator {
    /// Construct the recommended camera-model-independent robust estimator.
    pub fn robust() -> Self {
        Self::from_preset(TriangulationPreset::Robust)
    }

    /// Construct a clean-data estimator that retains every observation.
    pub fn all_observations() -> Self {
        Self::from_preset(TriangulationPreset::AllObservations)
    }

    /// Construct from a named preset.
    pub(crate) fn from_preset(preset: TriangulationPreset) -> Self {
        Self {
            config: TrackTriangulationConfig::from_preset(preset),
        }
    }

    /// Construct from a fully configurable policy.
    pub fn try_new(config: TrackTriangulationConfig) -> Result<Self, TriangulationError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Access the resolved configuration.
    pub fn config(&self) -> &TrackTriangulationConfig {
        &self.config
    }

    /// Triangulate one track using its configured policy.
    pub fn triangulate(
        &self,
        observations: &[TriangulationObservation],
        context: TriangulationContext<'_>,
    ) -> Result<TriangulatedTrack, TriangulationError> {
        self.config.validate()?;
        let prepared = prepare_observations(observations, context)?;
        self.triangulate_prepared(&prepared)
    }

    /// Triangulate observations whose camera context has already been resolved.
    ///
    /// Use it when you already hold each observation's camera, to skip the lookups that
    /// [`Self::triangulate`] does in its maps.
    pub fn triangulate_resolved(
        &self,
        observations: &[ResolvedTrackObservation<'_>],
    ) -> Result<TriangulatedTrack, TriangulationError> {
        self.triangulate_resolved_iter(observations.iter().copied())
    }

    /// [`Self::triangulate_resolved`] over an iterator, without collecting the observations.
    pub fn triangulate_resolved_iter<'a, I>(
        &self,
        observations: I,
    ) -> Result<TriangulatedTrack, TriangulationError>
    where
        I: IntoIterator<Item = ResolvedTrackObservation<'a>>,
        I::IntoIter: ExactSizeIterator,
    {
        self.config.validate()?;
        let prepared = prepare_resolved_observations(observations)?;
        self.triangulate_prepared(&prepared)
    }

    fn triangulate_prepared(
        &self,
        prepared: &[PreparedObservation<'_>],
    ) -> Result<TriangulatedTrack, TriangulationError> {
        match self.config.policy {
            TriangulationPolicy::AllObservations => {
                let inliers: Vec<usize> = (0..prepared.len()).collect();
                self.refit_and_validate(prepared, inliers, None, 0)
            }
            TriangulationPolicy::Consensus(config) if prepared.len() == 2 => {
                let inliers = vec![0, 1];
                ensure_consensus(&inliers, prepared.len(), config)?;
                self.refit_and_validate(prepared, inliers, None, 0)
            }
            TriangulationPolicy::Consensus(config) => self.consensus(prepared, config),
        }
    }

    fn consensus(
        &self,
        observations: &[PreparedObservation<'_>],
        config: ConsensusConfig,
    ) -> Result<TriangulatedTrack, TriangulationError> {
        let mut candidate_inliers = Vec::with_capacity(observations.len());
        let mut best_inliers = Vec::with_capacity(observations.len());
        let mut best_point = None;
        let mut best_cost = f64::INFINITY;
        let mut best_angle = 0.0;
        let mut best_view_pair = (ViewId::new(usize::MAX), ViewId::new(usize::MAX));

        let hypotheses = select_pair_hypotheses(
            observations,
            self.config.hypothesis_min_parallax.value(),
            config.pair_budget.for_observation_count(observations.len()),
        );
        let mut tested_hypothesis_count = 0;
        for hypothesis in hypotheses {
            let first = hypothesis.first;
            let second = hypothesis.second;
            let pair_angle = hypothesis.parallax;
            let view_pair = (
                observations[first].ray.view_id(),
                observations[second].ray.view_id(),
            );
            let Ok(point) = triangulate_two_view(
                &observations[first].ray,
                &observations[second].ray,
                self.config.two_view_method,
            ) else {
                continue;
            };
            tested_hypothesis_count += 1;
            let cost = classify_inliers(
                &point,
                observations,
                config,
                self.config.acceptance.min_depth.value(),
                &mut candidate_inliers,
            );
            let replace = best_point.is_none()
                || candidate_inliers.len() > best_inliers.len()
                || (candidate_inliers.len() == best_inliers.len()
                    && (cost < best_cost
                        || (cost == best_cost
                            && (pair_angle > best_angle
                                || (pair_angle == best_angle && view_pair < best_view_pair)))));
            if replace {
                best_point = Some(point);
                best_cost = cost;
                best_angle = pair_angle;
                best_view_pair = view_pair;
                std::mem::swap(&mut best_inliers, &mut candidate_inliers);
            }
        }

        let Some(mut point) = best_point else {
            return Err(TriangulationError::NoHypothesis);
        };
        ensure_consensus(&best_inliers, observations.len(), config)?;

        // Stabilize support and all-inlier refinement together. The returned
        // identities therefore classify as inliers to the returned point.
        for _ in 0..8 {
            point = refit(
                observations,
                &best_inliers,
                self.config.two_view_method,
                Some(point),
            )?;
            classify_inliers(
                &point,
                observations,
                config,
                self.config.acceptance.min_depth.value(),
                &mut candidate_inliers,
            );
            ensure_consensus(&candidate_inliers, observations.len(), config)?;
            if candidate_inliers == best_inliers {
                return self.validate_result(
                    observations,
                    best_inliers,
                    point,
                    tested_hypothesis_count,
                );
            }
            std::mem::swap(&mut best_inliers, &mut candidate_inliers);
        }
        Err(TriangulationError::Rejected(
            TrackTriangulationRejection::UnstableConsensus,
        ))
    }

    fn refit_and_validate(
        &self,
        observations: &[PreparedObservation<'_>],
        inliers: Vec<usize>,
        initial: Option<Point3<f64>>,
        tested_hypothesis_count: usize,
    ) -> Result<TriangulatedTrack, TriangulationError> {
        let point = refit(observations, &inliers, self.config.two_view_method, initial)?;
        self.validate_result(observations, inliers, point, tested_hypothesis_count)
    }

    fn validate_result(
        &self,
        observations: &[PreparedObservation<'_>],
        mut inliers: Vec<usize>,
        point: Point3<f64>,
        tested_hypothesis_count: usize,
    ) -> Result<TriangulatedTrack, TriangulationError> {
        let quality = quality(&point, observations, &inliers, tested_hypothesis_count)?;
        if quality.max_parallax.value() < self.config.acceptance.min_parallax.value() {
            return Err(TriangulationError::Rejected(
                TrackTriangulationRejection::InsufficientParallax,
            ));
        }
        if inliers.iter().any(|&index| {
            observations[index].ray.oriented_distance(&point)
                <= self.config.acceptance.min_depth.value()
        }) {
            return Err(TriangulationError::Rejected(
                TrackTriangulationRejection::Cheirality,
            ));
        }
        if quality.max_angular_residual.value()
            > self.config.acceptance.max_angular_residual.value()
        {
            return Err(TriangulationError::Rejected(
                TrackTriangulationRejection::AngularResidual,
            ));
        }
        if self
            .config
            .acceptance
            .max_pixel_residual
            .is_some_and(|threshold| quality.max_pixel_residual.value() > threshold.value())
        {
            return Err(TriangulationError::Rejected(
                TrackTriangulationRejection::PixelResidual,
            ));
        }

        for index in &mut inliers {
            *index = observations[*index].input_index;
        }
        inliers.sort_unstable();
        Ok(TriangulatedTrack {
            point,
            inlier_indices: inliers,
            quality,
        })
    }
}

pub(super) fn select_pair_hypotheses(
    observations: &[PreparedObservation<'_>],
    min_parallax: f64,
    max_hypotheses: usize,
) -> Vec<PairHypothesis> {
    let observation_count = observations.len();
    let pair_count = observation_count.saturating_mul(observation_count.saturating_sub(1)) / 2;
    if pair_count <= max_hypotheses {
        let mut hypotheses = Vec::with_capacity(pair_count);
        for first in 0..observation_count {
            for second in (first + 1)..observation_count {
                push_valid_pair(observations, min_parallax, first, second, &mut hypotheses);
            }
        }
        return hypotheses;
    }

    // The circle schedule enumerates every pair exactly once while each round
    // uses every observation at most once. This limits the influence of any
    // single contaminated observation without RNG state or repeated pairs.
    let player_count = observation_count + observation_count % 2;
    let bye = observation_count;
    let mut players: Vec<_> = (0..player_count).collect();
    let mut usage = vec![0usize; observation_count];
    let mut hypotheses = Vec::with_capacity(max_hypotheses);
    let mut round = Vec::with_capacity(player_count / 2);
    for _ in 0..player_count.saturating_sub(1) {
        round.clear();
        for index in 0..(player_count / 2) {
            let first = players[index];
            let second = players[player_count - 1 - index];
            if first == bye || second == bye {
                continue;
            }
            push_valid_pair(
                observations,
                min_parallax,
                first.min(second),
                first.max(second),
                &mut round,
            );
        }
        round.sort_by(|left, right| {
            let left_usage = (
                usage[left.first].max(usage[left.second]),
                usage[left.first] + usage[left.second],
            );
            let right_usage = (
                usage[right.first].max(usage[right.second]),
                usage[right.first] + usage[right.second],
            );
            left_usage
                .cmp(&right_usage)
                .then_with(|| right.parallax.total_cmp(&left.parallax))
                .then_with(|| {
                    let left_views = (
                        observations[left.first].ray.view_id(),
                        observations[left.second].ray.view_id(),
                    );
                    let right_views = (
                        observations[right.first].ray.view_id(),
                        observations[right.second].ray.view_id(),
                    );
                    left_views.cmp(&right_views)
                })
        });
        let remaining = max_hypotheses - hypotheses.len();
        for hypothesis in round.iter().take(remaining).copied() {
            usage[hypothesis.first] += 1;
            usage[hypothesis.second] += 1;
            hypotheses.push(hypothesis);
        }
        if hypotheses.len() == max_hypotheses {
            break;
        }
        players[1..].rotate_right(1);
    }
    hypotheses
}

fn push_valid_pair(
    observations: &[PreparedObservation<'_>],
    min_parallax: f64,
    first: usize,
    second: usize,
    hypotheses: &mut Vec<PairHypothesis>,
) {
    let angle = parallax_angle(&observations[first].ray, &observations[second].ray);
    if angle >= min_parallax {
        hypotheses.push(PairHypothesis {
            first,
            second,
            parallax: angle,
        });
    }
}

impl Default for TrackTriangulator {
    fn default() -> Self {
        Self::from_preset(TriangulationPreset::Robust)
    }
}

fn ensure_consensus(
    inliers: &[usize],
    observations: usize,
    config: ConsensusConfig,
) -> Result<(), TriangulationError> {
    let required_by_ratio = (config.min_inlier_ratio * observations as f64).ceil() as usize;
    if inliers.len() < config.min_inliers.max(required_by_ratio) {
        Err(TriangulationError::InsufficientConsensus {
            inliers: inliers.len(),
            observations,
        })
    } else {
        Ok(())
    }
}

fn refit(
    observations: &[PreparedObservation<'_>],
    inliers: &[usize],
    two_view_method: TwoViewMethod,
    initial: Option<Point3<f64>>,
) -> Result<Point3<f64>, TriangulationError> {
    if inliers.len() < 2 {
        return Err(TriangulationError::InsufficientObservations {
            required: 2,
            provided: inliers.len(),
        });
    }
    if inliers.len() == 2 {
        triangulate_two_view(
            &observations[inliers[0]].ray,
            &observations[inliers[1]].ray,
            two_view_method,
        )
        .map_err(TriangulationError::Numerical)
    } else {
        let rays: Vec<_> = inliers
            .iter()
            .map(|&index| observations[index].ray)
            .collect();
        let initializer = match initial {
            Some(point) => point,
            None => best_pair_initializer(observations, inliers, two_view_method)?,
        };
        refine_angular_nview(&rays, initializer, 8).map_err(TriangulationError::Numerical)
    }
}

fn best_pair_initializer(
    observations: &[PreparedObservation<'_>],
    inliers: &[usize],
    method: TwoViewMethod,
) -> Result<Point3<f64>, TriangulationError> {
    let mut best: Option<(f64, (ViewId, ViewId), Point3<f64>)> = None;
    for first in 0..inliers.len() {
        for second in (first + 1)..inliers.len() {
            let first = &observations[inliers[first]].ray;
            let second = &observations[inliers[second]].ray;
            let angle = parallax_angle(first, second);
            let pair = (first.view_id(), second.view_id());
            let Ok(point) = triangulate_two_view(first, second, method) else {
                continue;
            };
            if best.as_ref().is_none_or(|(best_angle, best_pair, _)| {
                angle > *best_angle || (angle == *best_angle && pair < *best_pair)
            }) {
                best = Some((angle, pair, point));
            }
        }
    }
    best.map(|(_, _, point)| point)
        .ok_or(TriangulationError::NoHypothesis)
}

fn classify_inliers(
    point: &Point3<f64>,
    observations: &[PreparedObservation<'_>],
    config: ConsensusConfig,
    min_depth: f64,
    inliers: &mut Vec<usize>,
) -> f64 {
    inliers.clear();
    let threshold_squared = match config.threshold {
        ConsensusThreshold::Angular(threshold) => threshold.squared(),
        ConsensusThreshold::Pixels(threshold) => threshold.squared(),
    };
    let mut cost = 0.0;
    for (index, observation) in observations.iter().enumerate() {
        let residual = consensus_residual(point, observation, config.threshold);
        let squared = if residual.is_finite() {
            residual * residual
        } else {
            f64::INFINITY
        };
        cost += squared.min(threshold_squared);
        if squared <= threshold_squared && observation.ray.oriented_distance(point) > min_depth {
            inliers.push(index);
        }
    }
    cost
}

fn consensus_residual(
    point: &Point3<f64>,
    observation: &PreparedObservation<'_>,
    threshold: ConsensusThreshold,
) -> f64 {
    match threshold {
        ConsensusThreshold::Angular(_) if pixel_residual(point, observation).is_finite() => {
            observation.ray.angular_residual(point)
        }
        ConsensusThreshold::Angular(_) => f64::INFINITY,
        ConsensusThreshold::Pixels(_) => pixel_residual(point, observation),
    }
}

fn pixel_residual(point: &Point3<f64>, observation: &PreparedObservation<'_>) -> f64 {
    let camera_point = observation.pose.project_to_camera(point);
    let Ok(projected) = observation.camera.project(&camera_point.coords) else {
        return f64::INFINITY;
    };
    let mut horizontal = (projected.x - observation.pixel.x).abs();
    if let Camera::Spherical(spherical) = observation.camera {
        let width = f64::from(spherical.width());
        horizontal = horizontal.rem_euclid(width);
        horizontal = horizontal.min(width - horizontal);
    }
    horizontal.hypot(projected.y - observation.pixel.y)
}

fn quality(
    point: &Point3<f64>,
    observations: &[PreparedObservation<'_>],
    inliers: &[usize],
    tested_hypothesis_count: usize,
) -> Result<TriangulationQuality, TriangulationError> {
    let mut max_parallax: f64 = 0.0;
    for first in 0..inliers.len() {
        for second in (first + 1)..inliers.len() {
            max_parallax = max_parallax.max(parallax_angle(
                &observations[inliers[first]].ray,
                &observations[inliers[second]].ray,
            ));
        }
    }

    let mut angular_sum: f64 = 0.0;
    let mut angular_max: f64 = 0.0;
    let mut pixel_sum: f64 = 0.0;
    let mut pixel_max: f64 = 0.0;
    // Prepared observations are canonicalized by view, and every inlier list is
    // produced in prepared-observation order.
    for &index in inliers {
        let angular = observations[index].ray.angular_residual(point);
        let pixel = pixel_residual(point, &observations[index]);
        angular_sum += angular;
        angular_max = angular_max.max(angular);
        pixel_sum += pixel;
        pixel_max = pixel_max.max(pixel);
    }
    let count = inliers.len() as f64;
    let invalid_quality =
        |_| TriangulationError::Rejected(TrackTriangulationRejection::ProjectionDomain);
    Ok(TriangulationQuality {
        max_parallax: AngularResidual::try_new(max_parallax).map_err(invalid_quality)?,
        mean_angular_residual: AngularResidual::try_new(angular_sum / count)
            .map_err(invalid_quality)?,
        max_angular_residual: AngularResidual::try_new(angular_max).map_err(invalid_quality)?,
        mean_pixel_residual: PixelResidual::try_new(pixel_sum / count).map_err(invalid_quality)?,
        max_pixel_residual: PixelResidual::try_new(pixel_max).map_err(invalid_quality)?,
        input_observation_count: observations.len(),
        inlier_observation_count: inliers.len(),
        tested_hypothesis_count,
    })
}
