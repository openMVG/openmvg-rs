// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Private adjacent-octave duplicate removal for scale-space detectors.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::extraction::ScaleSpaceKeypoint;

/// Which representative survives when adjacent octaves describe the same site.
#[derive(Clone, Copy)]
pub(super) enum AdjacentOctavePreference {
    /// Use ALP's signed-response comparison.
    StrongestSignedResponse,
}

#[cfg(test)]
pub(super) fn remove_adjacent_octave_duplicates(
    previous: &mut Vec<ScaleSpaceKeypoint>,
    current: &mut Vec<ScaleSpaceKeypoint>,
    preference: AdjacentOctavePreference,
) {
    remove_adjacent_octave_duplicates_by(previous, current, preference, |keypoint| keypoint);
}

/// Remove duplicates while retaining detector-specific metadata alongside the
/// shared scale-space geometry.
pub(super) fn remove_adjacent_octave_duplicates_by<T>(
    previous: &mut Vec<T>,
    current: &mut Vec<T>,
    preference: AdjacentOctavePreference,
    geometry: impl Fn(&T) -> &ScaleSpaceKeypoint,
) {
    #[cfg(not(feature = "std"))]
    use alloc::collections::BTreeMap;
    #[cfg(feature = "std")]
    use std::collections::BTreeMap;

    const RADIUS_MERGE: f32 = 2.0;
    const SIGMA_MERGE: f32 = 0.4;

    let mut keep_previous = vec![true; previous.len()];
    let mut keep_current = vec![true; current.len()];
    // The reference stops at the first matching current keypoint. The grid
    // narrows the search, then `duplicate` retains the lowest original index.
    let mut cells: BTreeMap<(i32, i32), Vec<usize>> = BTreeMap::new();
    for (index, value) in current.iter().enumerate() {
        let keypoint = geometry(value);
        cells
            .entry((keypoint.x.floor() as i32, keypoint.y.floor() as i32))
            .or_default()
            .push(index);
    }

    for (previous_index, previous_value) in previous.iter().enumerate() {
        let previous_keypoint = geometry(previous_value);
        let center_x = (0.5 * previous_keypoint.x).floor() as i32;
        let center_y = (0.5 * previous_keypoint.y).floor() as i32;
        let mut duplicate = None;

        for cell_y in center_y - 1..=center_y + 1 {
            for cell_x in center_x - 1..=center_x + 1 {
                let Some(indices) = cells.get(&(cell_x, cell_y)) else {
                    continue;
                };
                for &current_index in indices {
                    if duplicate.is_some_and(|best| current_index >= best)
                        || !keep_current[current_index]
                    {
                        continue;
                    }
                    let current_keypoint = geometry(&current[current_index]);
                    if previous_keypoint.is_max != current_keypoint.is_max {
                        continue;
                    }
                    let spatial_distance = (2.0 * current_keypoint.x - previous_keypoint.x).abs()
                        + (2.0 * current_keypoint.y - previous_keypoint.y).abs();
                    let scale_distance =
                        (2.0 * current_keypoint.sigma - previous_keypoint.sigma).abs();
                    if spatial_distance < RADIUS_MERGE && scale_distance < SIGMA_MERGE {
                        duplicate = Some(current_index);
                    }
                }
            }
        }

        let Some(current_index) = duplicate else {
            continue;
        };
        let current_keypoint = geometry(&current[current_index]);
        let previous_loses = match preference {
            AdjacentOctavePreference::StrongestSignedResponse => {
                if previous_keypoint.is_max {
                    previous_keypoint.peak < current_keypoint.peak
                } else {
                    previous_keypoint.peak > current_keypoint.peak
                }
            }
        };
        if previous_loses {
            keep_previous[previous_index] = false;
        } else {
            keep_current[current_index] = false;
        }
    }

    retain_by_mask(previous, &keep_previous);
    retain_by_mask(current, &keep_current);
}

fn retain_by_mask<T>(values: &mut Vec<T>, keep: &[bool]) {
    let mut keep = keep.iter().copied();
    values.retain(|_| keep.next().unwrap_or(false));
}
