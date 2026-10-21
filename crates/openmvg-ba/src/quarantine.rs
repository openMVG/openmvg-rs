// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quarantine of landmarks the adjuster cannot adjust.
//!
//! A landmark seen fewer than twice, or that sits on an observing camera center, collapses onto its
//! cameras against the scene, or that its cameras see along one ray, has no depth information and
//! no bounded Jacobian.
//! Such points are nuisance parameters: they must neither be optimized nor veto the cameras. The
//! adjuster therefore solves and certifies the problem without them, keeps them at their input
//! position, and reports them. During the solve, a step that would make a point degenerate keeps
//! that point at its previous position instead.

use nalgebra::Point3;
use openmvg_core::model::WorldToCameraPose;

use crate::BundleProblem;
use crate::kernel::TrackGeometry;

/// Iteration budget of each warm-started solve that follows the removal of weak points.
#[cfg(feature = "faer")]
pub(crate) const RESOLVE_ITERATIONS: usize = 8;

/// Most warm-started re-solves of one problem: a re-solve can expose points that the removed ones
/// kept anchored, but a few rounds either settle or show that the problem is not repairable.
#[cfg(feature = "faer")]
pub(crate) const MAX_RESOLVES: usize = 4;

/// Free points of `problem` that its input cannot place, in ascending order: those seen fewer
/// than twice, which have no depth, and those that sit on an observing camera center, collapse
/// onto their cameras against the scene, or have no parallax.
#[cfg(feature = "faer")]
pub(crate) fn degenerate_input_points(problem: &BundleProblem) -> Vec<usize> {
    let mut observation_counts = vec![0_usize; problem.points().len()];
    for observation in problem.observations() {
        observation_counts[observation.point()] += 1;
    }
    let mut weak: Vec<bool> = observation_counts.iter().map(|&count| count < 2).collect();
    let degenerate_points = degenerate(
        problem.points().len(),
        problem.observations().iter().map(|observation| {
            let center = problem.poses()[observation.pose()].pose().center();
            let point = problem.points()[observation.point()].position();
            (observation.point(), (point - center).into())
        }),
    );
    for point in degenerate_points {
        weak[point] = true;
    }
    (0..weak.len())
        .filter(|&point| weak[point] && !problem.points()[point].is_fixed())
        .collect()
}

/// Points that sit on an observing camera center, collapse onto their cameras against the scene, or
/// have no parallax at the given state, in ascending order.
pub(crate) fn degenerate_points(
    problem: &BundleProblem,
    poses: &[WorldToCameraPose],
    points: &[Point3<f64>],
) -> Vec<usize> {
    degenerate(
        points.len(),
        problem.observations().iter().map(|observation| {
            let offset = points[observation.point()] - poses[observation.pose()].center();
            (observation.point(), offset.into())
        }),
    )
}

/// `offsets` yields `(point, vector from an observing camera center to the point)`. A point with
/// fewer than two observations has no geometry to judge. The scene length that a camera must not
/// collapse onto is the median camera-to-point distance.
fn degenerate(point_count: usize, offsets: impl Iterator<Item = (usize, [f64; 3])>) -> Vec<usize> {
    let offsets = offsets.collect::<Vec<_>>();
    let mut distances = offsets
        .iter()
        .map(|(_, [x, y, z])| (x * x + y * y + z * z).sqrt())
        .collect::<Vec<_>>();
    let scene_length = if distances.is_empty() {
        0.0
    } else {
        let middle = distances.len() / 2;
        *distances.select_nth_unstable_by(middle, f64::total_cmp).1
    };
    let mut geometry = vec![TrackGeometry::with_scene_length(scene_length); point_count];
    for (point, offset) in offsets {
        geometry[point].push(offset);
    }
    geometry
        .iter()
        .enumerate()
        .filter(|(_, track)| track.is_degenerate())
        .map(|(point, _)| point)
        .collect()
}
