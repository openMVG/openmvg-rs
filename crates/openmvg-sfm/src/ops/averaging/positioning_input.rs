// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Input filter for the global positioning solver, which rejects unusable evidence.

use std::collections::BTreeMap;

use nalgebra::{Rotation3, Vector3};

use crate::ViewId;

/// Drop the track observations and relative directions that the global positioning solver
/// rejects, so the evidence handed to it is exactly what it admits.
///
/// An observation is kept when its view has a rotation and its world-frame bearing
/// (`rotation⁻¹ · ray`) has a finite norm above `1e-12`. A direction is kept when it joins
/// two distinct views that both have a rotation and its norm is finite and above `1e-12`.
/// Tracks are never removed and the order of the kept entries is unchanged; the solver's
/// minimum-view rule still decides which tracks are admitted.
pub(crate) fn retain_global_positioning_input(
    rotations: &BTreeMap<ViewId, Rotation3<f64>>,
    tracks: &mut BTreeMap<usize, Vec<(ViewId, Vector3<f64>)>>,
    directions: &mut Vec<(ViewId, ViewId, Vector3<f64>)>,
) {
    for observations in tracks.values_mut() {
        observations.retain(|(view, ray)| {
            rotations
                .get(view)
                .is_some_and(|rotation| is_usable_norm((rotation.inverse() * ray).norm()))
        });
    }
    directions.retain(|(source, target, direction)| {
        source != target
            && rotations.contains_key(source)
            && rotations.contains_key(target)
            && is_usable_norm(direction.norm())
    });
}

fn is_usable_norm(norm: f64) -> bool {
    norm.is_finite() && norm > 1e-12
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_exactly_the_inputs_the_solver_rejects() {
        let rotations = [0, 1, 2]
            .into_iter()
            .map(|view| (ViewId::new(view), Rotation3::identity()))
            .collect::<BTreeMap<_, _>>();
        let ray = Vector3::new(0.0, 0.0, 1.0);
        let mut tracks = BTreeMap::from([
            (
                4,
                vec![
                    (ViewId::new(0), ray),
                    (ViewId::new(5), ray),
                    (ViewId::new(1), Vector3::zeros()),
                    (ViewId::new(2), Vector3::new(f64::NAN, 0.0, 1.0)),
                    (ViewId::new(1), 2.0 * ray),
                ],
            ),
            (7, vec![(ViewId::new(9), ray)]),
        ]);
        let mut directions = vec![
            (ViewId::new(0), ViewId::new(1), ray),
            (ViewId::new(0), ViewId::new(9), ray),
            (ViewId::new(9), ViewId::new(1), ray),
            (ViewId::new(2), ViewId::new(2), ray),
            (
                ViewId::new(1),
                ViewId::new(2),
                Vector3::new(f64::INFINITY, 0.0, 0.0),
            ),
            (
                ViewId::new(1),
                ViewId::new(2),
                Vector3::new(0.0, 1e-13, 0.0),
            ),
            (ViewId::new(2), ViewId::new(0), 3.0 * ray),
        ];

        retain_global_positioning_input(&rotations, &mut tracks, &mut directions);

        assert_eq!(
            tracks,
            BTreeMap::from([
                (4, vec![(ViewId::new(0), ray), (ViewId::new(1), 2.0 * ray)]),
                (7, Vec::new()),
            ])
        );
        assert_eq!(
            directions,
            vec![
                (ViewId::new(0), ViewId::new(1), ray),
                (ViewId::new(2), ViewId::new(0), 3.0 * ray),
            ]
        );
    }

    #[test]
    fn keeps_usable_input_untouched() {
        let rotations = BTreeMap::from([
            (ViewId::new(0), Rotation3::from_euler_angles(0.1, -0.2, 0.3)),
            (ViewId::new(1), Rotation3::identity()),
        ]);
        let mut tracks = BTreeMap::from([(
            0,
            vec![
                (ViewId::new(1), Vector3::new(0.6, 0.0, 0.8)),
                (ViewId::new(0), Vector3::new(0.0, 0.0, 1.0)),
            ],
        )]);
        let mut directions = vec![(ViewId::new(1), ViewId::new(0), Vector3::new(1.0, 0.0, 0.0))];
        let (expected_tracks, expected_directions) = (tracks.clone(), directions.clone());

        retain_global_positioning_input(&rotations, &mut tracks, &mut directions);

        assert_eq!(tracks, expected_tracks);
        assert_eq!(directions, expected_directions);
    }
}
