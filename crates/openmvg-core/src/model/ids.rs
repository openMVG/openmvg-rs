// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Strongly typed identifiers shared by the SfM crates.
//!
//! Persistent identifiers use a fixed-width `u64` representation. Every public
//! construction and deserialization path proves that the value fits the local
//! address space, so later storage indexing is infallible without panicking.

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! define_id {
    ($name:ident, $kind:literal, $description:literal) => {
        #[doc = $description]
        #[repr(transparent)]
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Creates an identifier from its numeric index.
            #[inline]
            pub const fn new(index: usize) -> Self {
                Self(index as u64)
            }

            /// Creates an identifier from its serialized value, or `None` when the value does
            /// not fit in a `usize` on this platform.
            #[inline]
            pub fn try_from_u64(value: u64) -> Option<Self> {
                usize::try_from(value).ok().map(|_| Self(value))
            }

            /// Returns the stable serialized value.
            #[inline]
            pub const fn value(self) -> u64 {
                self.0
            }

            /// Returns the identifier as an index, for example into a vector of views.
            #[inline]
            pub const fn index(self) -> usize {
                self.0 as usize
            }

            /// Returns the kind of identifier, such as `"view"`, as error messages name it.
            #[inline]
            pub const fn kind() -> &'static str {
                $kind
            }
        }

        impl From<usize> for $name {
            #[inline]
            fn from(index: usize) -> Self {
                Self::new(index)
            }
        }

        impl From<$name> for usize {
            #[inline]
            fn from(id: $name) -> Self {
                id.index()
            }
        }

        impl From<$name> for u64 {
            #[inline]
            fn from(id: $name) -> Self {
                id.value()
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let value = u64::deserialize(deserializer)?;
                Self::try_from_u64(value).ok_or_else(|| {
                    serde::de::Error::custom(concat!(
                        $kind,
                        " identifier exceeds the local index range"
                    ))
                })
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

define_id!(
    ViewId,
    "view",
    "Unique identifier for a view (image) in a reconstruction."
);
define_id!(
    TrackId,
    "track",
    "Unique identifier for a multi-view feature track."
);
define_id!(
    LandmarkId,
    "landmark",
    "Unique identifier for a reconstructed 3D landmark."
);
define_id!(
    IntrinsicId,
    "intrinsic",
    "Unique identifier for a shared camera-intrinsic group."
);
define_id!(
    PoseId,
    "pose",
    "Unique identifier for a shared world-to-camera pose."
);
define_id!(
    FeatureId,
    "feature",
    "Unique identifier for an image feature."
);
define_id!(
    ComponentId,
    "component",
    "Stable identifier for a connected reconstruction component."
);

/// Canonical identity of an unordered pair of views.
///
/// Endpoints are stored in increasing [`ViewId`] order. Equal endpoints are allowed here; the
/// APIs that need two distinct views reject them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct ViewPair {
    first: ViewId,
    second: ViewId,
}

impl ViewPair {
    /// Canonicalizes two view identifiers into an unordered pair.
    #[must_use]
    pub fn new(left: ViewId, right: ViewId) -> Self {
        if left <= right {
            Self {
                first: left,
                second: right,
            }
        } else {
            Self {
                first: right,
                second: left,
            }
        }
    }

    /// Returns whether the supplied endpoints are already in canonical order.
    #[must_use]
    pub fn is_canonical(left: ViewId, right: ViewId) -> bool {
        left <= right
    }

    /// Returns the lower ordered endpoint.
    #[must_use]
    pub const fn first(self) -> ViewId {
        self.first
    }

    /// Returns the upper ordered endpoint.
    #[must_use]
    pub const fn second(self) -> ViewId {
        self.second
    }

    /// Returns both endpoints in canonical order.
    #[must_use]
    pub const fn endpoints(self) -> (ViewId, ViewId) {
        (self.first, self.second)
    }
}

impl<'de> Deserialize<'de> for ViewPair {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Fields {
            first: ViewId,
            second: ViewId,
        }

        let fields = Fields::deserialize(deserializer)?;
        Ok(Self::new(fields.first, fields.second))
    }
}

impl fmt::Display for ViewPair {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "({}, {})", self.first, self.second)
    }
}

impl From<(ViewId, ViewId)> for ViewPair {
    fn from((first, second): (ViewId, ViewId)) -> Self {
        Self::new(first, second)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ComponentId, FeatureId, IntrinsicId, LandmarkId, PoseId, TrackId, ViewId, ViewPair,
    };
    use std::collections::{BTreeSet, HashMap, HashSet};
    use std::mem::{align_of, size_of};

    #[test]
    fn persistent_identifiers_have_u64_layout() {
        macro_rules! assert_u64_layout {
            ($id:ty) => {
                assert_eq!(size_of::<$id>(), size_of::<u64>());
                assert_eq!(align_of::<$id>(), align_of::<u64>());
            };
        }

        assert_u64_layout!(ViewId);
        assert_u64_layout!(TrackId);
        assert_u64_layout!(LandmarkId);
        assert_u64_layout!(IntrinsicId);
        assert_u64_layout!(PoseId);
        assert_u64_layout!(FeatureId);
        assert_u64_layout!(ComponentId);
    }

    #[test]
    fn stable_values_are_checked_before_local_indexing() {
        assert_eq!(ViewId::try_from_u64(7), Some(ViewId::new(7)));
        assert_eq!(
            ViewId::try_from_u64(u64::MAX).is_some(),
            usize::BITS == u64::BITS
        );
    }

    #[test]
    fn view_pair_is_canonical_and_deserializes_canonically() {
        let pair = ViewPair::new(ViewId::new(9), ViewId::new(2));
        assert_eq!(pair.endpoints(), (ViewId::new(2), ViewId::new(9)));

        let json = serde_json::to_string(&pair).unwrap();
        assert_eq!(json, r#"{"first":2,"second":9}"#);
        let reversed: ViewPair = serde_json::from_str(r#"{"first":9,"second":2}"#).unwrap();
        assert_eq!(reversed, pair);
    }

    #[test]
    fn identifiers_preserve_scalar_and_map_key_serde() {
        let view = ViewId::new(7);
        assert_eq!(serde_json::to_string(&view).unwrap(), "7");
        assert_eq!(serde_json::from_str::<ViewId>("7").unwrap(), view);

        let map = HashMap::from([(ViewId::new(3), LandmarkId::new(11))]);
        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(json, r#"{"3":11}"#);
        assert_eq!(
            serde_json::from_str::<HashMap<ViewId, LandmarkId>>(&json).unwrap(),
            map
        );
    }

    #[test]
    fn identifiers_support_deterministic_ordering_and_hashing() {
        let ordered = BTreeSet::from([ViewId::new(9), ViewId::new(2), ViewId::new(9)]);
        assert_eq!(
            ordered.into_iter().collect::<Vec<_>>(),
            [ViewId::new(2), ViewId::new(9)]
        );

        let hashed = HashSet::from([TrackId::new(4), TrackId::new(4), TrackId::new(8)]);
        assert_eq!(hashed.len(), 2);
        assert!(hashed.contains(&TrackId::new(4)));
    }
}
