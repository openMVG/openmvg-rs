// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The error returned by invalid feature, descriptor, or match data.

use openmvg_core::model::FeatureId;

/// Invalid feature, descriptor, or match data.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum FeatureDataError {
    /// A keypoint coordinate, scale or orientation is NaN or infinite.
    #[error("keypoint coordinates, scale, and orientation must be finite")]
    NonFiniteKeypoint,
    /// A keypoint scale is negative.
    #[error("keypoint scale must be non-negative")]
    NegativeKeypointScale,
    /// A descriptor set was declared with dimension 0.
    #[error("{kind} descriptor dimension must be greater than zero")]
    ZeroDescriptorDimension {
        /// Storage kind: `"f32"`, `"u8"` or `"binary"`.
        kind: &'static str,
    },
    /// The descriptor buffer length is not a whole number of descriptors, or, for
    /// [`DescriptorSet::try_binary`](crate::model::DescriptorSet::try_binary), does not hold
    /// exactly the declared count.
    #[error("{kind} descriptor data length {len} is not divisible by dimension {dimension}")]
    InvalidDescriptorLength {
        /// Storage kind: `"f32"`, `"u8"` or `"binary"`.
        kind: &'static str,
        /// Buffer length in elements (`u64` words for binary storage).
        len: usize,
        /// Elements per descriptor (`u64` words for binary storage).
        dimension: usize,
    },
    /// The binary descriptor buffer length `count * words` overflows `usize`.
    #[error("descriptor length overflow")]
    DescriptorLengthOverflow,
    /// An `f32` descriptor element is NaN or infinite.
    #[error("f32 descriptors must contain only finite values")]
    NonFiniteDescriptor,
    /// Keypoints and descriptors have different counts.
    #[error("feature keypoint count {keypoints} does not match descriptor count {descriptors}")]
    FeatureDescriptorCountMismatch {
        /// Number of keypoints.
        keypoints: usize,
        /// Number of descriptors.
        descriptors: usize,
    },
    /// A feature-set pixel coordinate is NaN or infinite.
    #[error("feature set contains a non-finite pixel")]
    NonFinitePixel,
    /// A feature index is not in the feature set.
    #[error("feature {feature} is outside feature set of length {count}")]
    FeatureOutOfRange {
        /// Requested feature.
        feature: FeatureId,
        /// Number of features in the set.
        count: usize,
    },
    /// A match distance is NaN, infinite or negative.
    #[error("match distance must be finite and non-negative")]
    InvalidMatchDistance,
    /// Matches are not strictly increasing by `(first, second)`, which
    /// includes duplicate pairs.
    #[error("match set is not strictly ordered by feature pair")]
    NonCanonicalMatchOrder,
}
