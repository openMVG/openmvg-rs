// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Detector-independent feature-count policies.
//!
//! A budget selects detection candidates before descriptor construction. The
//! detector adapter decides how its native response maps to the common
//! "larger is stronger" convention.
//!
//! [`FeatureBudgetPolicy`] is the public policy type. The selection trait
//! behind it and the standalone unlimited policy are crate-private.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Position and strength ranked by a [`FeatureBudgetPolicy`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct FeatureCandidate {
    /// Column coordinate in input-image pixels; used by spatial budgets to
    /// pick a grid cell (values outside the image clamp to the border cells).
    pub x: f32,
    /// Row coordinate in input-image pixels; see [`Self::x`].
    pub y: f32,
    /// Detector-specific strength, normalized only so larger is better.
    pub response: f32,
}

impl FeatureCandidate {
    /// Create a candidate from its pixel position and strength.
    ///
    /// A non-finite `response` ranks below every finite one.
    pub const fn new(x: f32, y: f32, response: f32) -> Self {
        Self { x, y, response }
    }
}

/// Select candidates to retain from a detector result.
///
/// Returned indices may use any order. Callers sanitize duplicate and
/// out-of-range indices and retain selected candidates in detector order.
pub(crate) trait FeatureBudget {
    fn select(
        &self,
        candidates: &[FeatureCandidate],
        image_width: usize,
        image_height: usize,
    ) -> Vec<usize>;

    /// Allows detectors to preserve their zero-allocation streaming path.
    ///
    /// Returning `true` promises that [`Self::select`] would retain every
    /// candidate in source order.
    fn is_unlimited(&self) -> bool {
        false
    }
}

/// A keypoint that a feature budget can rank: its position and the strength of its response.
///
/// [`ScaleSpaceKeypoint`](crate::extraction::ScaleSpaceKeypoint) implements it; implement it for
/// keypoints of your own to cap them with [`apply_feature_budget`].
pub trait FeatureBudgetKeypoint {
    /// Map this keypoint to the position and "larger is stronger" response
    /// that the budget ranks.
    fn feature_candidate(&self) -> FeatureCandidate;
}

/// Apply a budget policy in place while preserving the detector's original order.
pub fn apply_feature_budget<K: FeatureBudgetKeypoint>(
    keypoints: &mut Vec<K>,
    image_width: usize,
    image_height: usize,
    budget: &FeatureBudgetPolicy,
) {
    retain_selected(keypoints, image_width, image_height, budget);
}

/// Retain the keypoints selected by any crate budget, in detector order.
pub(crate) fn retain_selected<K, B>(
    keypoints: &mut Vec<K>,
    image_width: usize,
    image_height: usize,
    budget: &B,
) where
    K: FeatureBudgetKeypoint,
    B: FeatureBudget + ?Sized,
{
    if budget.is_unlimited() || keypoints.is_empty() {
        return;
    }
    let candidates: Vec<_> = keypoints
        .iter()
        .map(FeatureBudgetKeypoint::feature_candidate)
        .collect();
    let mut keep = selection_mask(&candidates, image_width, image_height, budget).into_iter();
    keypoints.retain(|_| keep.next().unwrap_or(false));
}

/// Sanitize a policy's selected indices into a detector-order retention mask.
pub(crate) fn selection_mask<B: FeatureBudget + ?Sized>(
    candidates: &[FeatureCandidate],
    image_width: usize,
    image_height: usize,
    budget: &B,
) -> Vec<bool> {
    let mut keep = vec![false; candidates.len()];
    for index in budget.select(candidates, image_width, image_height) {
        if let Some(slot) = keep.get_mut(index) {
            *slot = true;
        }
    }
    keep
}

/// No feature-count limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UnlimitedBudget;

impl FeatureBudget for UnlimitedBudget {
    fn select(
        &self,
        candidates: &[FeatureCandidate],
        _image_width: usize,
        _image_height: usize,
    ) -> Vec<usize> {
        (0..candidates.len()).collect()
    }

    fn is_unlimited(&self) -> bool {
        true
    }
}

/// Keep the globally strongest detector responses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct StrongestResponseBudget {
    max_features: usize,
}

impl StrongestResponseBudget {
    /// Keep at most `max_features` candidates, strongest first; ties keep the
    /// earlier candidate. Zero keeps nothing.
    pub const fn new(max_features: usize) -> Self {
        Self { max_features }
    }

    /// Maximum number of retained candidates.
    pub const fn max_features(self) -> usize {
        self.max_features
    }
}

impl FeatureBudget for StrongestResponseBudget {
    fn select(
        &self,
        candidates: &[FeatureCandidate],
        _image_width: usize,
        _image_height: usize,
    ) -> Vec<usize> {
        strongest_indices(candidates, self.max_features)
    }
}

/// Balance strong responses across a regular image grid.
///
/// Each occupied cell first receives an equal quota. Any unused slots are
/// filled by the strongest remaining responses globally, so sparse cells do
/// not cause the requested budget to be under-filled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(serde::Serialize))]
pub struct SpatialGridBudget {
    max_features: usize,
    rows: usize,
    columns: usize,
}

#[cfg(feature = "std")]
impl<'de> serde::Deserialize<'de> for SpatialGridBudget {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Fields {
            max_features: usize,
            rows: usize,
            columns: usize,
        }
        let fields = <Fields as serde::Deserialize>::deserialize(deserializer)?;
        Ok(Self::new(fields.max_features, fields.rows, fields.columns))
    }
}

#[cfg(all(test, feature = "std"))]
#[test]
fn deserialized_spatial_budget_preserves_grid_bounds() {
    let budget: FeatureBudgetPolicy = serde_json::from_value(serde_json::json!({
        "spatial_grid": { "max_features": 8, "rows": 0, "columns": 1000 }
    }))
    .expect("deserialize budget");
    let FeatureBudgetPolicy::SpatialGrid(grid) = budget else {
        panic!("expected spatial budget");
    };
    assert_eq!(grid.rows(), 1);
    assert_eq!(grid.columns(), 64);
    assert_eq!(grid.max_features(), 8);
}

impl SpatialGridBudget {
    const MAX_GRID_DIMENSION: usize = 64;

    /// Create a spatial response budget.
    ///
    /// Grid dimensions are clamped to `1..=64` to keep accidental or
    /// untrusted configuration from allocating an unbounded cell table.
    pub fn new(max_features: usize, rows: usize, columns: usize) -> Self {
        Self {
            max_features,
            rows: rows.clamp(1, Self::MAX_GRID_DIMENSION),
            columns: columns.clamp(1, Self::MAX_GRID_DIMENSION),
        }
    }

    /// Create a budget of `max_features` over a 4 × 4 grid, the same as
    /// [`new(max_features, 4, 4)`](Self::new).
    pub const fn four_by_four(max_features: usize) -> Self {
        Self {
            max_features,
            rows: 4,
            columns: 4,
        }
    }

    /// Maximum number of retained candidates.
    pub const fn max_features(self) -> usize {
        self.max_features
    }

    /// Number of grid rows, in `1..=64`.
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Number of grid columns, in `1..=64`.
    pub const fn columns(self) -> usize {
        self.columns
    }
}

impl FeatureBudget for SpatialGridBudget {
    fn select(
        &self,
        candidates: &[FeatureCandidate],
        image_width: usize,
        image_height: usize,
    ) -> Vec<usize> {
        let limit = self.max_features.min(candidates.len());
        if limit == 0 {
            return Vec::new();
        }
        if limit == candidates.len() || image_width == 0 || image_height == 0 {
            return (0..limit).collect();
        }

        let cell_count = self.rows * self.columns;
        let mut cells: Vec<Vec<usize>> = (0..cell_count).map(|_| Vec::new()).collect();
        for (index, candidate) in candidates.iter().enumerate() {
            let row = grid_coordinate(candidate.y, image_height, self.rows);
            let column = grid_coordinate(candidate.x, image_width, self.columns);
            cells[row * self.columns + column].push(index);
        }
        for cell in &mut cells {
            sort_by_strength(cell, candidates);
        }

        let occupied = cells.iter().filter(|cell| !cell.is_empty()).count();
        if occupied == 0 {
            return Vec::new();
        }
        let base_quota = limit / occupied;
        let mut selected = Vec::with_capacity(limit);
        let mut keep = vec![false; candidates.len()];
        for cell in &cells {
            for &index in cell.iter().take(base_quota) {
                keep[index] = true;
                selected.push(index);
            }
        }

        // Distribute the remainder across each cell's next-best candidate.
        let mut frontier: Vec<usize> = cells
            .iter()
            .filter_map(|cell| cell.get(base_quota).copied())
            .collect();
        sort_by_strength(&mut frontier, candidates);
        for index in frontier {
            if selected.len() == limit {
                break;
            }
            keep[index] = true;
            selected.push(index);
        }

        // Short cells can leave quota unused. Fill from all remaining responses.
        if selected.len() < limit {
            let mut remaining: Vec<usize> = (0..candidates.len())
                .filter(|&index| !keep[index])
                .collect();
            sort_by_strength(&mut remaining, candidates);
            for index in remaining.into_iter().take(limit - selected.len()) {
                selected.push(index);
            }
        }
        selected.sort_unstable();
        selected
    }
}

/// Built-in policies that can be stored directly in detector configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "std", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum FeatureBudgetPolicy {
    /// Keep every candidate (the default).
    #[default]
    Unlimited,
    /// Keep the globally strongest responses.
    Strongest(StrongestResponseBudget),
    /// Balance strong responses across a regular image grid.
    SpatialGrid(SpatialGridBudget),
}

impl FeatureBudgetPolicy {
    /// Shorthand for [`Self::Strongest`] with [`StrongestResponseBudget::new`].
    pub const fn strongest(max_features: usize) -> Self {
        Self::Strongest(StrongestResponseBudget::new(max_features))
    }

    /// Shorthand for [`Self::SpatialGrid`] with
    /// [`SpatialGridBudget::four_by_four`].
    pub const fn spatial_four_by_four(max_features: usize) -> Self {
        Self::SpatialGrid(SpatialGridBudget::four_by_four(max_features))
    }

    /// Shorthand for [`Self::SpatialGrid`] with [`SpatialGridBudget::new`]
    /// (grid dimensions clamped to `1..=64`).
    pub fn spatial_grid(max_features: usize, rows: usize, columns: usize) -> Self {
        Self::SpatialGrid(SpatialGridBudget::new(max_features, rows, columns))
    }
}

impl FeatureBudget for FeatureBudgetPolicy {
    fn select(
        &self,
        candidates: &[FeatureCandidate],
        image_width: usize,
        image_height: usize,
    ) -> Vec<usize> {
        match self {
            Self::Unlimited => UnlimitedBudget.select(candidates, image_width, image_height),
            Self::Strongest(budget) => budget.select(candidates, image_width, image_height),
            Self::SpatialGrid(budget) => budget.select(candidates, image_width, image_height),
        }
    }

    fn is_unlimited(&self) -> bool {
        matches!(self, Self::Unlimited)
    }
}

fn strongest_indices(candidates: &[FeatureCandidate], max_features: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    sort_by_strength(&mut indices, candidates);
    indices.truncate(max_features);
    indices.sort_unstable();
    indices
}

fn sort_by_strength(indices: &mut [usize], candidates: &[FeatureCandidate]) {
    indices.sort_unstable_by(|&left, &right| {
        response(candidates[right].response)
            .total_cmp(&response(candidates[left].response))
            .then_with(|| left.cmp(&right))
    });
}

fn response(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        f32::NEG_INFINITY
    }
}

fn grid_coordinate(value: f32, extent: usize, divisions: usize) -> usize {
    if !value.is_finite() || extent == 0 {
        return 0;
    }
    (((value.max(0.0) / extent as f32) * divisions as f32).floor() as usize).min(divisions - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(x: f32, y: f32, response: f32) -> FeatureCandidate {
        FeatureCandidate::new(x, y, response)
    }

    #[test]
    fn strongest_budget_is_deterministic_and_preserves_source_order() {
        let candidates = [
            candidate(0.0, 0.0, 1.0),
            candidate(0.0, 0.0, 5.0),
            candidate(0.0, 0.0, 5.0),
            candidate(0.0, 0.0, 2.0),
        ];
        assert_eq!(
            StrongestResponseBudget::new(2).select(&candidates, 10, 10),
            vec![1, 2]
        );
    }

    #[test]
    fn spatial_budget_represents_each_occupied_cell() {
        let candidates = [
            candidate(10.0, 10.0, 100.0),
            candidate(20.0, 20.0, 90.0),
            candidate(75.0, 10.0, 4.0),
            candidate(10.0, 75.0, 3.0),
            candidate(75.0, 75.0, 2.0),
        ];
        assert_eq!(
            SpatialGridBudget::new(4, 2, 2).select(&candidates, 100, 100),
            vec![0, 2, 3, 4]
        );
    }

    #[test]
    fn spatial_budget_fills_slots_from_strong_remaining_responses() {
        let candidates = [
            candidate(10.0, 10.0, 10.0),
            candidate(15.0, 15.0, 9.0),
            candidate(20.0, 20.0, 8.0),
            candidate(75.0, 75.0, 1.0),
        ];
        assert_eq!(
            SpatialGridBudget::new(3, 2, 2).select(&candidates, 100, 100),
            vec![0, 1, 3]
        );
    }

    #[derive(Debug, PartialEq)]
    struct Keypoint(FeatureCandidate);

    impl FeatureBudgetKeypoint for Keypoint {
        fn feature_candidate(&self) -> FeatureCandidate {
            self.0
        }
    }

    #[test]
    fn apply_budget_ignores_invalid_indices_and_keeps_detector_order() {
        struct InvalidBudget;
        impl FeatureBudget for InvalidBudget {
            fn select(
                &self,
                _candidates: &[FeatureCandidate],
                _image_width: usize,
                _image_height: usize,
            ) -> Vec<usize> {
                vec![2, 2, 99, 0]
            }
        }

        let mut keypoints = vec![
            Keypoint(candidate(0.0, 0.0, 1.0)),
            Keypoint(candidate(0.0, 0.0, 2.0)),
            Keypoint(candidate(0.0, 0.0, 3.0)),
        ];
        retain_selected(&mut keypoints, 10, 10, &InvalidBudget);
        assert_eq!(
            keypoints,
            vec![
                Keypoint(candidate(0.0, 0.0, 1.0)),
                Keypoint(candidate(0.0, 0.0, 3.0))
            ]
        );
    }
}
