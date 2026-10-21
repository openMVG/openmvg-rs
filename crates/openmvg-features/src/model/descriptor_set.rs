// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Homogeneous per-image descriptor storage.

use super::FeatureDataError;

/// Homogeneous descriptor storage for one image.
///
/// Serializes as its storage variant; deserialization applies
/// [`Self::validate`].
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorSet {
    storage: DescriptorStorage,
}

// Hand-written `#[serde(transparent)]`: serde does not combine `transparent`
// with `try_from`.
impl serde::Serialize for DescriptorSet {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.storage.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for DescriptorSet {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let descriptors = Self {
            storage: DescriptorStorage::deserialize(deserializer)?,
        };
        descriptors
            .validate()
            .map_err(<D::Error as serde::de::Error>::custom)?;
        Ok(descriptors)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum DescriptorStorage {
    F32 { data: Vec<f32>, dim: usize },
    U8 { data: Vec<u8>, dim: usize },
    Binary { data: Vec<u64>, dim_bits: usize },
}

impl DescriptorSet {
    /// Dense `f32` descriptors stored row-major, `dim` elements each.
    ///
    /// # Errors
    ///
    /// [`FeatureDataError::NonFiniteDescriptor`],
    /// [`FeatureDataError::ZeroDescriptorDimension`] or
    /// [`FeatureDataError::InvalidDescriptorLength`] (see [`Self::validate`]).
    pub fn try_f32(data: Vec<f32>, dim: usize) -> Result<Self, FeatureDataError> {
        let descriptors = Self {
            storage: DescriptorStorage::F32 { data, dim },
        };
        descriptors.validate()?;
        Ok(descriptors)
    }

    /// Dense byte descriptors stored row-major, `dim` elements each.
    ///
    /// # Errors
    ///
    /// [`FeatureDataError::ZeroDescriptorDimension`] or
    /// [`FeatureDataError::InvalidDescriptorLength`] (see [`Self::validate`]).
    pub fn try_u8(data: Vec<u8>, dim: usize) -> Result<Self, FeatureDataError> {
        let descriptors = Self {
            storage: DescriptorStorage::U8 { data, dim },
        };
        descriptors.validate()?;
        Ok(descriptors)
    }

    /// `count` packed binary descriptors of `dim_bits` bits each; every
    /// descriptor occupies `dim_bits.div_ceil(64)` consecutive `u64` words.
    ///
    /// # Errors
    ///
    /// [`FeatureDataError::ZeroDescriptorDimension`] if `dim_bits` is 0,
    /// [`FeatureDataError::DescriptorLengthOverflow`] if the word count
    /// overflows, [`FeatureDataError::InvalidDescriptorLength`] if `data` does
    /// not hold exactly `count` descriptors.
    pub fn try_binary(
        data: Vec<u64>,
        dim_bits: usize,
        count: usize,
    ) -> Result<Self, FeatureDataError> {
        if dim_bits == 0 {
            return Err(FeatureDataError::ZeroDescriptorDimension { kind: "binary" });
        }
        let words_per_descriptor = dim_bits.div_ceil(64);
        let expected_len = count
            .checked_mul(words_per_descriptor)
            .ok_or(FeatureDataError::DescriptorLengthOverflow)?;
        if data.len() != expected_len {
            return Err(FeatureDataError::InvalidDescriptorLength {
                kind: "binary",
                len: data.len(),
                dimension: words_per_descriptor,
            });
        }
        Ok(Self {
            storage: DescriptorStorage::Binary { data, dim_bits },
        })
    }

    /// Number of descriptors.
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.storage {
            DescriptorStorage::F32 { data, dim } => data.len().checked_div(*dim).unwrap_or(0),
            DescriptorStorage::U8 { data, dim } => data.len().checked_div(*dim).unwrap_or(0),
            DescriptorStorage::Binary { data, dim_bits } => {
                if *dim_bits == 0 {
                    0
                } else {
                    data.len() / dim_bits.div_ceil(64)
                }
            }
        }
    }

    /// Whether the set holds no descriptors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Descriptor dimension: elements per descriptor for `f32`/`u8` storage,
    /// bits per descriptor for binary storage.
    #[must_use]
    pub fn dimension(&self) -> usize {
        match &self.storage {
            DescriptorStorage::F32 { dim, .. } | DescriptorStorage::U8 { dim, .. } => *dim,
            DescriptorStorage::Binary { dim_bits, .. } => *dim_bits,
        }
    }

    /// Borrow dense floating-point descriptors and their element dimension.
    #[must_use]
    pub fn as_f32(&self) -> Option<(&[f32], usize)> {
        match &self.storage {
            DescriptorStorage::F32 { data, dim } => Some((data, *dim)),
            _ => None,
        }
    }

    /// Borrow dense byte descriptors and their element dimension.
    #[must_use]
    pub fn as_u8(&self) -> Option<(&[u8], usize)> {
        match &self.storage {
            DescriptorStorage::U8 { data, dim } => Some((data, *dim)),
            _ => None,
        }
    }

    /// Borrow packed binary descriptors and their bit dimension.
    #[must_use]
    pub fn as_binary(&self) -> Option<(&[u64], usize)> {
        match &self.storage {
            DescriptorStorage::Binary { data, dim_bits } => Some((data, *dim_bits)),
            _ => None,
        }
    }

    /// Re-check the invariants enforced by the constructors: non-zero
    /// dimension, whole number of descriptors, finite `f32` values.
    ///
    /// # Errors
    ///
    /// The first violated invariant, as in the constructors.
    pub fn validate(&self) -> Result<(), FeatureDataError> {
        match &self.storage {
            DescriptorStorage::F32 { data, dim } => {
                if data.iter().any(|value| !value.is_finite()) {
                    return Err(FeatureDataError::NonFiniteDescriptor);
                }
                validate_dense_descriptor(data.len(), *dim, "f32")
            }
            DescriptorStorage::U8 { data, dim } => {
                validate_dense_descriptor(data.len(), *dim, "u8")
            }
            DescriptorStorage::Binary { data, dim_bits } => {
                if *dim_bits == 0 {
                    return Err(FeatureDataError::ZeroDescriptorDimension { kind: "binary" });
                }
                let words_per_descriptor = dim_bits.div_ceil(64);
                if !data.len().is_multiple_of(words_per_descriptor) {
                    return Err(FeatureDataError::InvalidDescriptorLength {
                        kind: "binary",
                        len: data.len(),
                        dimension: words_per_descriptor,
                    });
                }
                Ok(())
            }
        }
    }
}

fn validate_dense_descriptor(
    len: usize,
    dimension: usize,
    kind: &'static str,
) -> Result<(), FeatureDataError> {
    if dimension == 0 {
        return Err(FeatureDataError::ZeroDescriptorDimension { kind });
    }
    if !len.is_multiple_of(dimension) {
        return Err(FeatureDataError::InvalidDescriptorLength {
            kind,
            len,
            dimension,
        });
    }
    Ok(())
}
