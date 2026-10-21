// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed dense identifiers for views, poses, intrinsics, features, tracks, landmarks, and
//! components.

use std::fmt;

pub use openmvg_core::model::{
    ComponentId, FeatureId, IntrinsicId, LandmarkId, PoseId, TrackId, ViewId,
};

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident, $label:literal) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(
            Debug,
            Clone,
            Copy,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
        )]
        pub(crate) struct $name(usize);

        impl $name {
            pub const fn new(index: usize) -> Self {
                Self(index)
            }

            pub const fn index(self) -> usize {
                self.0
            }

            pub const fn kind() -> &'static str {
                $label
            }
        }

        impl From<usize> for $name {
            fn from(index: usize) -> Self {
                Self::new(index)
            }
        }

        impl From<$name> for usize {
            fn from(id: $name) -> Self {
                id.index()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{}({})", $label, self.0)
            }
        }
    };
}

define_id!(PairId, "pair");

/// Stable index of an observation within an immutable source track.
///
/// The index is 32-bit because feature tracks cannot approach this limit
/// and landmarks may store one index per supporting observation.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct TrackObservationIndex(u32);

impl TrackObservationIndex {
    /// Wrap a zero-based observation index; `None` when it does not fit in `u32`.
    pub fn try_new(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self)
    }

    /// Zero-based position of the observation within its track.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<TrackObservationIndex> for usize {
    fn from(index: TrackObservationIndex) -> Self {
        index.index()
    }
}

impl fmt::Display for TrackObservationIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "track observation {}", self.0)
    }
}
