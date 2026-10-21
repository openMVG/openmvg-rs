// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Huber and hard-cut reweighting of bearing and direction constraints.

use nalgebra::Vector3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::api::BataOptions;
use super::plan::{Direction, GlobalPositioningPlan};
use super::workspace::GlobalPositioningWorkspace;

/// True direction cosine, normalized after scaling to avoid squared-norm
/// overflow/underflow. Undefined baselines return None; coarse reweighting
/// and angular diagnostics assign them a right-angle penalty. Projected
/// polish has the stricter nonzero-baseline admissibility requirement.
pub(super) fn direction_cosine(unit: &Vector3<f64>, baseline: Vector3<f64>) -> Option<f64> {
    if !unit
        .iter()
        .chain(baseline.iter())
        .all(|value| value.is_finite())
    {
        return None;
    }
    let scale = baseline.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return None;
    }
    let scaled = baseline / scale;
    let cosine = scaled.dot(unit) / scaled.norm();
    cosine.is_finite().then(|| cosine.clamp(-1.0, 1.0))
}

/// A forward halfline admits at most a right angle, so larger angular
/// cutoffs cannot remove additional admissible forward rays.
pub(super) fn cosine_cutoff(angle_degrees: f64) -> f64 {
    if angle_degrees >= 90.0 {
        0.0
    } else {
        angle_degrees.to_radians().cos()
    }
}

pub(super) fn chordal_huber_width(angle_radians: f64) -> f64 {
    let angle = angle_radians.min(std::f64::consts::PI);
    let width = 2.0 * (0.5 * angle).sin();
    // Halving the smallest subnormal can underflow; its sine limit is angle.
    if width > 0.0 { width } else { angle }
}

pub(super) fn projected_huber_width(angle_radians: f64) -> f64 {
    angle_radians.min(std::f64::consts::FRAC_PI_2).sin()
}

#[derive(Clone, Copy)]
pub(super) struct RobustIteration {
    pub(super) outer: usize,
    /// Huber transition in the chordal residual chart.
    pub(super) delta: f64,
    pub(super) cosine_hard_cut: f64,
    pub(super) cosine_mixed_hard_cut: f64,
    pub(super) options: BataOptions,
    /// Whether gross outliers and the mixed-direction gate may zero a weight.
    pub(super) cut: bool,
}

impl RobustIteration {
    /// The same iteration with Huber weights only: no warmup special cases and no cuts, so every
    /// weight is positive. Used when the cuts would disconnect the active graph.
    pub(super) fn huber_only(mut self) -> Self {
        self.cut = false;
        self.options.warmup_iterations = 0;
        self
    }

    fn bearing(self, ray: Vector3<f64>, baseline: Vector3<f64>, cosine: f64) -> f64 {
        if self.outer < self.options.warmup_iterations {
            return 1.0;
        }
        let chordal_error = (2.0 - 2.0 * cosine).clamp(0.0, 4.0).sqrt();
        let mut weight = huber_weight(chordal_error, self.delta);
        if self.cut
            && self.options.hard_cut_enabled()
            && self.outer >= self.options.hard_cut_start_iteration
            && (ray.dot(&baseline) < 0.0 || cosine < self.cosine_hard_cut)
        {
            weight = 0.0;
        }
        weight
    }

    fn direction(self, edge: Direction, baseline: Vector3<f64>, cosine: f64, mixed: bool) -> f64 {
        if mixed && self.outer < self.options.warmup_iterations {
            return 0.0;
        }
        if self.outer < self.options.warmup_iterations {
            return 1.0;
        }
        let chordal_error = (2.0 - 2.0 * cosine).clamp(0.0, 4.0).sqrt();
        let mut weight = huber_weight(chordal_error, self.delta);
        let hard_cut_cosine = if mixed {
            self.cosine_mixed_hard_cut
        } else {
            self.cosine_hard_cut
        };
        if self.cut
            && self.options.hard_cut_enabled()
            && (mixed || self.outer >= self.options.hard_cut_start_iteration)
            && (edge.direction.dot(&baseline) < 0.0 || cosine < hard_cut_cosine)
        {
            weight = 0.0;
        }
        weight
    }
}

fn huber_weight(error: f64, delta: f64) -> f64 {
    if error <= delta {
        1.0
    } else {
        (delta / error).clamp(0.0, 1.0)
    }
}

pub(super) fn update_robust_weights(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    robust: RobustIteration,
    _parallel_execution: bool,
) {
    #[cfg(feature = "parallel")]
    if _parallel_execution {
        let cameras = &workspace.cameras;
        let points = &workspace.points;
        let scales = &workspace.bearing_scale;
        workspace
            .bearing_weight
            .par_iter_mut()
            .zip(workspace.bearing_effective_weight.par_iter_mut())
            .enumerate()
            .for_each(|(index, (weight_out, effective_out))| {
                let observation = plan.bearings[index];
                let baseline = points[observation.point] - cameras[observation.camera];
                let cosine = direction_cosine(&observation.ray, baseline).unwrap_or(0.0);
                let robust_weight = robust.bearing(observation.ray, baseline, cosine);
                let weight = robust_weight * observation.base_weight;
                *weight_out = weight;
                *effective_out = weight * scales[index] * scales[index];
            });

        let scales = &workspace.direction_scale;
        workspace
            .direction_weight
            .par_iter_mut()
            .zip(workspace.direction_effective_weight.par_iter_mut())
            .enumerate()
            .for_each(|(index, (weight_out, effective_out))| {
                let edge = plan.directions[index];
                let baseline = cameras[edge.target] - cameras[edge.source];
                let cosine = direction_cosine(&edge.direction, baseline).unwrap_or(0.0);
                let robust_weight =
                    robust.direction(edge, baseline, cosine, !plan.bearings.is_empty());
                let weight = robust_weight * edge.base_weight;
                *weight_out = weight;
                *effective_out = weight * scales[index] * scales[index];
            });
        return;
    }

    for (index, observation) in plan.bearings.iter().copied().enumerate() {
        let baseline = workspace.points[observation.point] - workspace.cameras[observation.camera];
        let cosine = direction_cosine(&observation.ray, baseline).unwrap_or(0.0);
        let robust_weight = robust.bearing(observation.ray, baseline, cosine);
        let weight = robust_weight * observation.base_weight;
        workspace.bearing_weight[index] = weight;
        workspace.bearing_effective_weight[index] =
            weight * workspace.bearing_scale[index] * workspace.bearing_scale[index];
    }
    for (index, edge) in plan.directions.iter().copied().enumerate() {
        let baseline = workspace.cameras[edge.target] - workspace.cameras[edge.source];
        let cosine = direction_cosine(&edge.direction, baseline).unwrap_or(0.0);
        let robust_weight = robust.direction(edge, baseline, cosine, !plan.bearings.is_empty());
        let weight = robust_weight * edge.base_weight;
        workspace.direction_weight[index] = weight;
        workspace.direction_effective_weight[index] =
            weight * workspace.direction_scale[index] * workspace.direction_scale[index];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iteration(outer: usize) -> RobustIteration {
        let options = BataOptions {
            warmup_iterations: 0,
            hard_cut_start_iteration: 5,
            ..BataOptions::default()
        };
        RobustIteration {
            outer,
            delta: chordal_huber_width(options.final_huber_delta_radians),
            cosine_hard_cut: options.hard_cut_deg.to_radians().cos(),
            cosine_mixed_hard_cut: 0.0,
            options,
            cut: true,
        }
    }

    #[test]
    fn hard_cut_starts_at_the_configured_iteration() {
        let ray = Vector3::z();
        let baseline = Vector3::x();
        // A ninety-degree disagreement is a gross outlier.
        assert!(iteration(4).bearing(ray, baseline, 0.0) > 0.0);
        assert_eq!(iteration(5).bearing(ray, baseline, 0.0), 0.0);
    }

    #[test]
    fn positioning_mixed_direction_cut_is_disabled_beyond_budget() {
        let mut robust = iteration(2);
        robust.options.max_outer_iterations = 3;
        robust.options.hard_cut_start_iteration = 3;
        robust.cosine_mixed_hard_cut = (0.25 * robust.options.hard_cut_deg).to_radians().cos();
        let theta = 20.0_f64.to_radians();
        let baseline = Vector3::new(theta.cos(), theta.sin(), 0.0);
        let edge = Direction {
            source: 0,
            target: 1,
            direction: Vector3::x(),
            base_weight: 1.0,
        };
        assert!(
            robust.direction(edge, baseline, theta.cos(), true) > 0.0,
            "the mixed early gate cannot revive a disabled angular cut"
        );
    }
}
