// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Geometric admissibility of a landmark relative to the cameras that observe it.
//!
//! A landmark that sits on one observing camera center, or that every observing camera sees along
//! the same ray, has a projection Jacobian that is either unbounded or without depth information.
//! Its point block is then singular or so badly scaled against the camera blocks that eliminating
//! it destroys the Schur complement by cancellation. "On a camera center" is relative to the
//! farther of the point's farthest camera and the scene length, so a cluster of cameras and points
//! that collapses onto itself is caught even though its cameras stay at similar distances.
//!
//! Like a point behind the camera, such a point is an invalid evaluation. The Levenberg-Marquardt
//! candidate keeps such a point at its previous position, and rejects the step only when the
//! camera step collapses a camera onto that position; [`candidate_rule`] is that rule.

/// Smallest admissible ratio of the nearest to the farthest observing camera distance.
///
/// Eliminating a point whose cameras are `1/ratio` apart in distance loses about `eps / ratio^2`
/// of relative precision in the camera system, so `1e-4` keeps eight digits.
pub(crate) const MIN_NEAR_FAR_RATIO: f64 = 1e-4;

/// Smallest admissible sine of the angle between two observing rays of a point: about `1e-3`
/// degrees, far below the parallax that any triangulation stage admits.
pub(crate) const MIN_PARALLAX_SINE: f64 = 1.7e-5;

/// Running distance and parallax summary of one landmark's observing cameras.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TrackGeometry {
    near_squared: f64,
    far_squared: f64,
    #[cfg_attr(not(feature = "faer"), allow(dead_code))]
    scene_squared: f64,
    first_ray: Option<[f64; 3]>,
    parallax_found: bool,
    observations: usize,
    invalid: bool,
}

impl TrackGeometry {
    /// A summary whose nearest camera must also clear [`MIN_NEAR_FAR_RATIO`] of `scene_length`.
    pub(crate) const fn with_scene_length(scene_length: f64) -> Self {
        Self {
            near_squared: f64::INFINITY,
            far_squared: scene_length * scene_length,
            scene_squared: scene_length * scene_length,
            first_ray: None,
            parallax_found: false,
            observations: 0,
            invalid: false,
        }
    }

    /// Record the vector from one observing camera center to the point, in world axes.
    #[inline]
    pub(crate) fn push(&mut self, offset: [f64; 3]) {
        let squared = offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2];
        if !(squared.is_finite() && squared > 0.0) {
            self.invalid = true;
            return;
        }
        self.observations += 1;
        self.near_squared = self.near_squared.min(squared);
        self.far_squared = self.far_squared.max(squared);
        let inverse = squared.sqrt().recip();
        let ray = offset.map(|value| value * inverse);
        let Some(first) = self.first_ray else {
            self.first_ray = Some(ray);
            return;
        };
        if self.parallax_found {
            return;
        }
        let dot = ray[0] * first[0] + ray[1] * first[1] + ray[2] * first[2];
        let cross = [
            ray[1] * first[2] - ray[2] * first[1],
            ray[2] * first[0] - ray[0] * first[2],
            ray[0] * first[1] - ray[1] * first[0],
        ];
        let sine_squared = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
        self.parallax_found = dot < 0.0 || sine_squared >= MIN_PARALLAX_SINE * MIN_PARALLAX_SINE;
    }

    /// Whether the point is degenerate for the cameras recorded so far. A track seen by fewer than
    /// two cameras has no geometry to judge.
    pub(crate) fn is_degenerate(&self) -> bool {
        self.invalid
            || (self.observations >= 2
                && (!self.parallax_found
                    || self.near_squared
                        < MIN_NEAR_FAR_RATIO * MIN_NEAR_FAR_RATIO * self.far_squared))
    }

    /// Whether an observing camera sits on the point relative to the scene length alone.
    #[cfg_attr(not(feature = "faer"), allow(dead_code))]
    pub(crate) fn is_collapsed(&self) -> bool {
        self.invalid
            || (self.observations >= 2
                && self.near_squared < MIN_NEAR_FAR_RATIO * MIN_NEAR_FAR_RATIO * self.scene_squared)
    }
}

/// Which test, beyond a valid projection by every observing camera, admits a point evaluation.
#[cfg(feature = "faer")]
#[derive(Clone, Copy)]
pub(crate) enum Admission {
    /// A valid projection alone.
    Projection,
    /// No observing camera on the point, against the scene length.
    Collapse,
    /// Neither collapsed nor without parallax: [`TrackGeometry::is_degenerate`].
    Geometry,
}

#[cfg(feature = "faer")]
impl Admission {
    /// Whether the observing cameras collected in `geometry` admit the point.
    pub(crate) fn admits(self, geometry: &TrackGeometry) -> bool {
        match self {
            Self::Projection => true,
            Self::Collapse => !geometry.is_collapsed(),
            Self::Geometry => !geometry.is_degenerate(),
        }
    }
}

/// The current or the candidate iterate of a Levenberg-Marquardt step.
#[cfg(feature = "faer")]
#[derive(Clone, Copy)]
pub(crate) enum Iterate {
    Current,
    Candidate,
}

/// Where a candidate step leaves one point, with the robust cost of its track there.
#[cfg(feature = "faer")]
pub(crate) enum PointCandidate {
    /// At its stepped position.
    Moved(f64),
    /// At its previous position.
    Held(f64),
}

/// The candidate rule for one point: it moves to its stepped position unless that position is
/// degenerate at the candidate cameras, and otherwise keeps its previous position, which the
/// candidate cameras must not collapse onto. A previous position the current cameras already
/// collapse onto, or do not project, needs only a valid projection. `None` rejects the step.
///
/// `cost(cameras, point, admission)` is the robust cost of the point's track with the `cameras`
/// iterate and the point at its stepped (`Candidate`) or previous (`Current`) position, or `None`
/// when that evaluation fails `admission`. The evaluations run in a fixed order.
#[cfg(feature = "faer")]
pub(crate) fn candidate_rule(
    mut cost: impl FnMut(Iterate, Iterate, Admission) -> Option<f64>,
) -> Option<PointCandidate> {
    use Iterate::{Candidate, Current};
    if let Some(moved) = cost(Candidate, Candidate, Admission::Geometry) {
        return Some(PointCandidate::Moved(moved));
    }
    let held = match cost(Candidate, Current, Admission::Collapse) {
        Some(held) => held,
        None if cost(Current, Current, Admission::Collapse).is_none() => {
            cost(Candidate, Current, Admission::Projection)?
        }
        None => return None,
    };
    Some(PointCandidate::Held(held))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(offsets: &[[f64; 3]]) -> TrackGeometry {
        let mut geometry = TrackGeometry::with_scene_length(0.0);
        for offset in offsets {
            geometry.push(*offset);
        }
        geometry
    }

    #[test]
    fn well_triangulated_point_is_admissible() {
        assert!(!geometry(&[[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [-1.0, 0.0, 6.0]]).is_degenerate());
    }

    #[test]
    fn point_on_a_camera_center_is_degenerate() {
        assert!(geometry(&[[0.0, 0.0, 5.0], [1e-7, 0.0, 1e-7]]).is_degenerate());
        assert!(geometry(&[[0.0, 0.0, 5.0], [0.0, 0.0, 0.0]]).is_degenerate());
    }

    #[test]
    fn point_seen_along_one_ray_is_degenerate() {
        assert!(geometry(&[[0.0, 0.0, 1e5], [1.0, 0.0, 1e5]]).is_degenerate());
    }

    #[test]
    fn point_between_two_cameras_has_parallax() {
        assert!(!geometry(&[[0.0, 0.0, 5.0], [0.0, 0.0, -5.0]]).is_degenerate());
    }

    #[test]
    fn point_collapsed_onto_its_cameras_is_degenerate_against_the_scene() {
        let collapsed = [[0.0, 0.0, 1e-6], [1e-6, 0.0, 1e-6]];
        assert!(!geometry(&collapsed).is_degenerate());
        let mut geometry = TrackGeometry::with_scene_length(10.0);
        for offset in collapsed {
            geometry.push(offset);
        }
        assert!(geometry.is_degenerate());
        assert!(geometry.is_collapsed());
    }

    #[test]
    fn far_point_without_parallax_is_not_collapsed() {
        let mut geometry = TrackGeometry::with_scene_length(10.0);
        geometry.push([0.0, 0.0, 1e5]);
        geometry.push([1.0, 0.0, 1e5]);
        assert!(geometry.is_degenerate());
        assert!(!geometry.is_collapsed());
    }

    #[test]
    fn single_observation_has_no_geometry_to_judge() {
        assert!(!geometry(&[[0.0, 0.0, 5.0]]).is_degenerate());
    }
}
