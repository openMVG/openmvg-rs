// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The source-track observations that support a landmark, public only under
//! `unstable-internals` like the [`Landmark`](super::Landmark) field that holds them.

use crate::{Result, SfmError, TrackObservationIndex};

/// Exact source-track observations supporting a landmark's current geometry.
///
/// `All` avoids an allocation for fully accepted tracks. `Selected` stores
/// compact indices into the landmark's immutable source
/// [`Track`](openmvg_features::tracking::Track), so pixel
/// coordinates and feature IDs are never duplicated.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(transparent)]
pub struct ObservationSupport(Option<Box<[TrackObservationIndex]>>);

impl ObservationSupport {
    /// Build the support from strictly increasing observation indices into a track of
    /// `track_length` observations.
    ///
    /// Fails when an index is out of range, the indices are not strictly increasing, or fewer
    /// than two are given. Selecting every observation yields the allocation-free full support.
    pub fn try_from_indices(
        indices: impl IntoIterator<Item = usize>,
        track_length: usize,
    ) -> Result<Self> {
        let mut selected = Vec::new();
        let mut previous = None;
        for index in indices {
            if index >= track_length {
                return Err(SfmError::InvalidInput(format!(
                    "support index {index} is outside track length {track_length}"
                )));
            }
            if previous.is_some_and(|previous| index <= previous) {
                return Err(SfmError::InvalidInput(
                    "support indices must be strictly increasing".to_owned(),
                ));
            }
            let compact = TrackObservationIndex::try_new(index).ok_or_else(|| {
                SfmError::InvalidInput(format!(
                    "track observation index {index} exceeds compact storage"
                ))
            })?;
            selected.push(compact);
            previous = Some(index);
        }
        if selected.len() < 2 {
            return Err(SfmError::InvalidInput(
                "a landmark requires at least two supporting observations".to_owned(),
            ));
        }
        if selected.len() == track_length
            && selected
                .iter()
                .enumerate()
                .all(|(index, selected)| selected.index() == index)
        {
            Ok(Self::default())
        } else {
            Ok(Self(Some(selected.into_boxed_slice())))
        }
    }

    /// Number of supporting observations for a source track of `track_length` observations.
    pub fn len(&self, track_length: usize) -> usize {
        self.0
            .as_ref()
            .map_or(track_length, |indices| indices.len())
    }

    /// Check the support against a source track of `track_length` observations: at least two
    /// supporting observations, strictly increasing and in range.
    pub fn validate(&self, track_length: usize) -> Result<()> {
        if track_length < 2 {
            return Err(SfmError::InvalidInput(
                "landmark source track must contain at least two observations".to_owned(),
            ));
        }
        let Some(indices) = &self.0 else {
            return Ok(());
        };
        if indices.len() < 2 {
            return Err(SfmError::InvalidInput(
                "selected landmark support requires at least two observations".to_owned(),
            ));
        }
        if indices.windows(2).any(|window| window[0] >= window[1]) {
            return Err(SfmError::InvalidInput(
                "selected landmark support must be strictly increasing".to_owned(),
            ));
        }
        if indices
            .last()
            .is_some_and(|index| index.index() >= track_length)
        {
            return Err(SfmError::InvalidInput(format!(
                "selected landmark support exceeds source track length {track_length}"
            )));
        }
        Ok(())
    }

    /// Whether track observation `index` supports the landmark.
    pub fn contains(&self, index: usize) -> bool {
        self.0.as_ref().is_none_or(|indices| {
            indices
                .binary_search_by_key(&index, |selected| selected.index())
                .is_ok()
        })
    }

    /// Whether every observation of the source track supports the landmark.
    pub fn is_all(&self) -> bool {
        self.0.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_support_is_allocation_free_and_selected_support_is_exact() {
        let all = ObservationSupport::try_from_indices(0..4, 4).expect("valid full support");
        assert!(all.is_all());
        assert_eq!(all.len(4), 4);

        let selected =
            ObservationSupport::try_from_indices([0, 2], 4).expect("valid selected support");
        assert!(!selected.is_all());
        assert_eq!(selected.len(4), 2);
        assert!(selected.contains(0));
        assert!(!selected.contains(1));
        assert!(selected.contains(2));
    }
}
