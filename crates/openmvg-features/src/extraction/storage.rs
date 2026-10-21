// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Descriptor storage traits for different data types (f32, u8).
//!
//! A storage trait with two implementations: full-precision f32 and
//! quantized u8.
//!
//! # Storage types
//!
//! - [`F32Descriptor`]: Full precision 32-bit floating point (128 × 4 = 512 bytes per descriptor)
//! - [`U8Descriptor`]: Quantized 8-bit unsigned integer (128 × 1 = 128 bytes per descriptor)
//!
//! # Normalization pipeline
//!
//! The detectors finalize every descriptor on one shared scale:
//!
//! 1. Compute raw SIFT descriptor (128-D float)
//! 2. Apply RootSIFT normalization (L1 normalize → sqrt → L2 normalize)
//! 3. Scale to an L2 norm of 512 and cap each value at 255
//! 4. For u8: round to bytes ([`U8Descriptor::try_from_finalized`])
//!
//! # L2 distance computation
//!
//! The storage types only hold descriptor values. Distances come from the
//! matching kernels, `matching::squared_l2_f32` for float descriptors and
//! `matching::squared_l2_u8` for raw bytes on one shared grid, which accumulates
//! in integer arithmetic:
//!
//! ```text
//! squared_l2(a, b) = Σ(a[i] - b[i])² computed in u16/u32 arithmetic
//! ```
//!
//! # Examples
//!
//! ```
//! use openmvg_features::extraction::{
//!     DescriptorStorage, F32Descriptor, U8Descriptor, SIFT_DESCRIPTOR_DIM,
//! };
//! use openmvg_features::matching::{squared_l2_f32, squared_l2_u8};
//!
//! let rootsift_desc = vec![0.0; SIFT_DESCRIPTOR_DIM];
//!
//! let f32_desc = F32Descriptor::from_root_sift(&rootsift_desc).unwrap();
//! let u8_desc = U8Descriptor::try_from_finalized(&rootsift_desc).unwrap();
//!
//! assert_eq!(squared_l2_f32(f32_desc.as_slice(), f32_desc.as_slice()), 0.0);
//! assert_eq!(squared_l2_u8(u8_desc.as_slice(), u8_desc.as_slice()), 0);
//! ```

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// The standard SIFT descriptor dimension (4×4×8 = 128).
pub const SIFT_DESCRIPTOR_DIM: usize = 128;

// =============================================================================
// DescriptorError
// =============================================================================

/// Errors that can occur when constructing descriptors.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DescriptorError {
    /// The input slice length does not match the expected descriptor dimension.
    DimensionMismatch {
        /// Required number of descriptor elements.
        expected: usize,
        /// Number of elements supplied.
        actual: usize,
    },
    /// A descriptor element is NaN or infinite.
    NonFiniteValue {
        /// Zero-based index of the first non-finite element.
        index: usize,
    },
    /// The [`U8Descriptor`] quantization scale is NaN, infinite, zero or
    /// negative. The scale maps a stored byte back to the descriptor value
    /// (`value = byte * scale / 255`), so it must be finite and strictly
    /// positive.
    InvalidScale,
}

#[cfg(feature = "std")]
impl std::error::Error for DescriptorError {}

impl core::fmt::Display for DescriptorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "descriptor dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::NonFiniteValue { index } => {
                write!(f, "descriptor element {index} is not finite")
            }
            Self::InvalidScale => {
                f.write_str("descriptor quantization scale must be finite and positive")
            }
        }
    }
}

fn validate_finite_values(values: &[f32]) -> Result<(), DescriptorError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(DescriptorError::NonFiniteValue { index });
    }
    Ok(())
}

// =============================================================================
// DescriptorStorage Trait
// =============================================================================

/// Storage of one 128-value descriptor: [`F32Descriptor`] or [`U8Descriptor`].
///
/// The detectors' `extract_typed` methods are generic over it. `Element` is the stored value
/// type, `f32` or `u8`.
pub trait DescriptorStorage: Sized + Clone {
    /// The descriptor dimension as a compile-time constant.
    const DIM: usize;

    /// The underlying element type (f32 or u8).
    type Element: Copy;

    /// Create a descriptor from a RootSIFT-normalized f32 slice.
    ///
    /// `root_sift` is a 128-D RootSIFT-normalized descriptor: L1-normalized, then square-rooted
    /// element-wise, so it has unit L2 norm and values typically in [0, 1].
    ///
    /// # Warning: per-descriptor rescaling
    ///
    /// For [`U8Descriptor`] this stretches **each** descriptor so its own largest value maps to
    /// 255, and records that factor in [`U8Descriptor::scale`]. Such bytes compare only after
    /// converting each vector back with its own scale (`F32Descriptor::from`), **not** byte for
    /// byte. The matchers compare raw bytes, so give them descriptors on the shared scale,
    /// from a detector or from [`U8Descriptor::try_from_finalized`].
    ///
    /// # Errors
    /// Returns [`DescriptorError::DimensionMismatch`] if the input slice length != 128,
    /// or [`DescriptorError::NonFiniteValue`] for a NaN or infinite bin.
    fn from_root_sift(root_sift: &[f32]) -> Result<Self, DescriptorError>;

    /// Store a descriptor already on the shared SIFT scale that [`U8Descriptor`] describes.
    /// Matchers that compare raw bytes need this one scale across descriptors.
    fn try_from_finalized(values: &[f32]) -> Result<Self, DescriptorError> {
        Self::from_root_sift(values)
    }

    /// Access the underlying descriptor data.
    fn as_slice(&self) -> &[Self::Element];

    /// Get the dimension of the descriptor.
    fn dimension(&self) -> usize {
        Self::DIM
    }
}

// =============================================================================
// F32Descriptor
// =============================================================================

/// Descriptor stored as 128 `f32` values, 512 bytes.
///
/// The detectors fill it with the values that [`U8Descriptor`] rounds to bytes, on the same
/// shared scale. Use it when you need the unrounded values, for example to transform the
/// descriptors further.
#[derive(Clone, Debug, PartialEq)]
pub struct F32Descriptor {
    data: [f32; SIFT_DESCRIPTOR_DIM],
}

impl F32Descriptor {
    /// Create a descriptor from a fixed-size array, rejecting NaN and infinite
    /// values.
    ///
    /// # Errors
    /// Returns [`DescriptorError::NonFiniteValue`] with the index of the first
    /// element that is NaN or infinite.
    pub fn try_new(data: [f32; SIFT_DESCRIPTOR_DIM]) -> Result<Self, DescriptorError> {
        validate_finite_values(&data)?;
        Ok(Self { data })
    }

    /// Create a descriptor from a fixed-size array without validation.
    ///
    /// Crate-internal detector paths use this for values they produced.
    pub(crate) fn new(data: [f32; SIFT_DESCRIPTOR_DIM]) -> Self {
        Self { data }
    }

    /// Create a zero-initialized descriptor.
    pub fn zeros() -> Self {
        Self {
            data: [0.0; SIFT_DESCRIPTOR_DIM],
        }
    }

    /// Borrow the fixed-size descriptor without losing its dimensional type.
    #[must_use]
    pub const fn as_array(&self) -> &[f32; SIFT_DESCRIPTOR_DIM] {
        &self.data
    }
}

impl DescriptorStorage for F32Descriptor {
    const DIM: usize = SIFT_DESCRIPTOR_DIM;
    type Element = f32;

    fn from_root_sift(root_sift: &[f32]) -> Result<Self, DescriptorError> {
        if root_sift.len() != SIFT_DESCRIPTOR_DIM {
            return Err(DescriptorError::DimensionMismatch {
                expected: SIFT_DESCRIPTOR_DIM,
                actual: root_sift.len(),
            });
        }

        let mut data = [0.0f32; SIFT_DESCRIPTOR_DIM];
        data.copy_from_slice(root_sift);
        Self::try_new(data)
    }

    fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

impl Default for F32Descriptor {
    fn default() -> Self {
        Self::zeros()
    }
}

impl TryFrom<&[f32]> for F32Descriptor {
    type Error = DescriptorError;
    fn try_from(slice: &[f32]) -> Result<Self, Self::Error> {
        Self::from_root_sift(slice)
    }
}

impl TryFrom<Vec<f32>> for F32Descriptor {
    type Error = DescriptorError;
    fn try_from(vec: Vec<f32>) -> Result<Self, Self::Error> {
        Self::from_root_sift(&vec)
    }
}

// =============================================================================
// U8Descriptor
// =============================================================================

/// Descriptor stored as 128 bytes, a quarter of [`F32Descriptor`]; the storage `openmvg-sfm`
/// uses.
///
/// # Scale
///
/// The detectors finalize every descriptor on one shared scale, the convention of SIFT
/// implementations: the RootSIFT vector is scaled to an L2 norm of 512, each value is capped at
/// 255, and `U8Descriptor` rounds it to a byte, an error of at most half a unit. Because the
/// scale is shared, descriptors compare byte for byte: the `u8` distance metrics and the
/// matchers read [`as_slice`](DescriptorStorage::as_slice) directly.
/// [`Self::try_from_finalized`] stores values already on that scale.
///
/// [`DescriptorStorage::from_root_sift`] instead stretches each descriptor so that its own
/// largest value maps to 255, and records that factor in [`Self::scale`]. Such bytes do not
/// share a scale: compare them only after converting each vector back with
/// `F32Descriptor::from`.
///
/// # Memory layout
/// - 128 × u8 = 128 bytes per descriptor
/// - 4 bytes for the quantization scale: `value = byte · scale / 255`, with a scale of 255 for
///   the shared scale
/// - Total: 132 bytes (vs 512 bytes for f32)
#[derive(Clone, Debug)]
pub struct U8Descriptor {
    data: [u8; SIFT_DESCRIPTOR_DIM],
    /// Scale factor used during quantization (max value in original descriptor).
    /// Used to convert distances back to f32 space.
    scale: f32,
}

impl U8Descriptor {
    fn from_root_sift_array(root_sift: &[f32; SIFT_DESCRIPTOR_DIM]) -> Self {
        let max_val = root_sift.iter().copied().fold(0.0f32, f32::max);
        let mut data = [0u8; SIFT_DESCRIPTOR_DIM];
        if max_val > 1e-10 {
            let scale_factor = 255.0 / max_val;
            for (slot, &value) in data.iter_mut().zip(root_sift) {
                *slot = (value * scale_factor).round().clamp(0.0, 255.0) as u8;
            }
            Self {
                data,
                scale: max_val,
            }
        } else {
            Self { data, scale: 1.0 }
        }
    }

    /// Create a descriptor from quantized bytes and their quantization scale.
    ///
    /// `scale` maps a stored byte back to the descriptor value:
    /// `value = byte * scale / 255`. Bytes on the shared scale, such as descriptors read back
    /// from a file, use `scale = 255.0`.
    ///
    /// # Errors
    /// Returns [`DescriptorError::InvalidScale`] if `scale` is NaN, infinite,
    /// zero or negative.
    pub fn try_new(data: [u8; SIFT_DESCRIPTOR_DIM], scale: f32) -> Result<Self, DescriptorError> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(DescriptorError::InvalidScale);
        }
        Ok(Self { data, scale })
    }

    /// Create a zero-initialized descriptor.
    pub fn zeros() -> Self {
        Self {
            data: [0u8; SIFT_DESCRIPTOR_DIM],
            scale: 1.0,
        }
    }

    /// Borrow the fixed-size byte descriptor without losing its dimensional type.
    #[must_use]
    pub const fn as_array(&self) -> &[u8; SIFT_DESCRIPTOR_DIM] {
        &self.data
    }

    /// Get the scale factor used during quantization.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Round a descriptor that is **already on the shared scale** (L2 norm 512, values capped
    /// at 255), as the ALP and SIFT detectors produce, to bytes.
    ///
    /// Unlike [`from_root_sift`](DescriptorStorage::from_root_sift), this applies **no
    /// per-descriptor rescaling**: values are rounded and clamped in place, and the scale is
    /// 255. Use it whenever descriptors are compared byte for byte, as the matchers do.
    ///
    /// # Why this exists
    ///
    /// `from_root_sift` stretches every descriptor so its own largest value maps to 255.
    /// That factor varies between descriptors of the same image, so two descriptors that are
    /// close on the shared scale can land far apart as bytes, and byte nearest neighbors then
    /// disagree with float nearest neighbors. Rounding on the shared scale keeps byte distances
    /// faithful to float distances.
    ///
    /// # Errors
    /// Returns [`DescriptorError::DimensionMismatch`] if `values.len() != 128`, or
    /// [`DescriptorError::NonFiniteValue`] for a NaN or infinite bin.
    pub fn try_from_finalized(values: &[f32]) -> Result<Self, DescriptorError> {
        let values: &[f32; SIFT_DESCRIPTOR_DIM] =
            values
                .try_into()
                .map_err(|_| DescriptorError::DimensionMismatch {
                    expected: SIFT_DESCRIPTOR_DIM,
                    actual: values.len(),
                })?;
        validate_finite_values(values)?;
        Ok(Self::from_finalized_array(values))
    }

    /// Quantize a statically-sized descriptor already on the shared `[0, 255]`
    /// grid. This avoids a redundant dimension check in typed detector paths.
    #[inline]
    pub(crate) fn from_finalized_array(values: &[f32; SIFT_DESCRIPTOR_DIM]) -> Self {
        let mut data = [0u8; SIFT_DESCRIPTOR_DIM];
        for (slot, &val) in data.iter_mut().zip(values) {
            *slot = val.round().clamp(0.0, 255.0) as u8;
        }
        // `scale` means "original = byte * scale / 255" (see `From<&U8Descriptor>` for
        // `F32Descriptor`). These bytes ARE the original grid values, so the identity scale
        // is 255.0, not 1.0 — with 1.0 the reconstruction would come back 255x too small and
        // would not be comparable with a `from_root_sift` descriptor.
        Self { data, scale: 255.0 }
    }
}

impl DescriptorStorage for U8Descriptor {
    const DIM: usize = SIFT_DESCRIPTOR_DIM;
    type Element = u8;

    fn from_root_sift(root_sift: &[f32]) -> Result<Self, DescriptorError> {
        if root_sift.len() != SIFT_DESCRIPTOR_DIM {
            return Err(DescriptorError::DimensionMismatch {
                expected: SIFT_DESCRIPTOR_DIM,
                actual: root_sift.len(),
            });
        }

        let root_sift: &[f32; SIFT_DESCRIPTOR_DIM] =
            root_sift
                .try_into()
                .map_err(|_| DescriptorError::DimensionMismatch {
                    expected: SIFT_DESCRIPTOR_DIM,
                    actual: root_sift.len(),
                })?;
        validate_finite_values(root_sift)?;
        Ok(Self::from_root_sift_array(root_sift))
    }

    fn try_from_finalized(values: &[f32]) -> Result<Self, DescriptorError> {
        U8Descriptor::try_from_finalized(values)
    }

    fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

impl PartialEq for U8Descriptor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && (self.scale - other.scale).abs() < f32::EPSILON
    }
}

impl Default for U8Descriptor {
    fn default() -> Self {
        Self::zeros()
    }
}

// =============================================================================
// Conversion Functions
// =============================================================================

/// Convert an [`F32Descriptor`] to a [`U8Descriptor`], rescaling this one descriptor.
///
/// Like [`from_root_sift`](DescriptorStorage::from_root_sift), this stretches the descriptor so
/// its own largest value maps to 255 and records that factor in [`U8Descriptor::scale`], which
/// `F32Descriptor::from` uses to get the values back. The bytes are not on the shared scale, so
/// they do not compare byte for byte; give the matchers [`U8Descriptor::try_from_finalized`]
/// descriptors instead.
impl From<&F32Descriptor> for U8Descriptor {
    fn from(f32_desc: &F32Descriptor) -> Self {
        // The generic conversion preserves the scale-aware round-trip contract.
        // Raw-byte matcher paths must opt into the shared grid explicitly with
        // `try_from_finalized`.
        U8Descriptor::from_root_sift_array(&f32_desc.data)
    }
}

/// Convert a U8Descriptor back to F32Descriptor.
///
/// This conversion is lossy: the reconstructed f32 values approximate the
/// original values.
impl From<&U8Descriptor> for F32Descriptor {
    fn from(u8_desc: &U8Descriptor) -> Self {
        let scale = u8_desc.scale / 255.0;
        let mut data = [0.0f32; SIFT_DESCRIPTOR_DIM];
        for (i, &val) in u8_desc.data.iter().enumerate() {
            data[i] = val as f32 * scale;
        }
        F32Descriptor { data }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_root_sift() -> Vec<f32> {
        // Create a simple test descriptor
        let mut desc = vec![0.0f32; SIFT_DESCRIPTOR_DIM];
        for (i, val) in desc.iter_mut().enumerate().take(SIFT_DESCRIPTOR_DIM) {
            *val = (i as f32 + 1.0) / (SIFT_DESCRIPTOR_DIM as f32);
        }
        // L2 normalize (mimicking RootSIFT output)
        let norm: f32 = desc.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut desc {
            *x /= norm;
        }
        desc
    }

    /// Squared L2 distance between the scale-reconstructed float vectors.
    fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    fn reconstructed(desc: &U8Descriptor) -> F32Descriptor {
        F32Descriptor::from(desc)
    }

    #[test]
    fn f32_try_new_rejects_non_finite_values() {
        let mut data = [0.25f32; SIFT_DESCRIPTOR_DIM];
        assert_eq!(F32Descriptor::try_new(data).unwrap().as_array(), &data);

        data[7] = f32::NAN;
        assert_eq!(
            F32Descriptor::try_new(data),
            Err(DescriptorError::NonFiniteValue { index: 7 })
        );
        data[7] = 0.0;
        data[3] = f32::NEG_INFINITY;
        assert_eq!(
            F32Descriptor::try_new(data),
            Err(DescriptorError::NonFiniteValue { index: 3 })
        );
    }

    #[test]
    fn u8_try_new_rejects_invalid_scales() {
        let data = [9u8; SIFT_DESCRIPTOR_DIM];
        let descriptor = U8Descriptor::try_new(data, 255.0).unwrap();
        assert_eq!(descriptor.as_array(), &data);
        assert_eq!(descriptor.scale(), 255.0);

        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                U8Descriptor::try_new(data, scale).unwrap_err(),
                DescriptorError::InvalidScale
            );
        }
    }

    #[test]
    fn descriptor_storage_enforces_dimension_and_preserves_values() {
        let root_sift = make_test_root_sift();
        let f32_descriptor = F32Descriptor::from_root_sift(&root_sift).unwrap();
        let u8_descriptor = U8Descriptor::from_root_sift(&root_sift).unwrap();

        assert_eq!(f32_descriptor.as_slice(), root_sift);
        assert_eq!(u8_descriptor.dimension(), SIFT_DESCRIPTOR_DIM);
        assert_eq!(*u8_descriptor.as_slice().iter().max().unwrap(), 255);
        assert!(matches!(
            F32Descriptor::from_root_sift(&root_sift[..SIFT_DESCRIPTOR_DIM - 1]),
            Err(DescriptorError::DimensionMismatch {
                expected: SIFT_DESCRIPTOR_DIM,
                actual
            }) if actual == SIFT_DESCRIPTOR_DIM - 1
        ));
        assert!(U8Descriptor::from_root_sift(&root_sift[..SIFT_DESCRIPTOR_DIM - 1]).is_err());
    }

    /// `try_from_finalized` must quantize in place (no per-descriptor rescaling) so that
    /// raw-byte comparisons stay faithful to the float values.
    #[test]
    fn test_uint8_from_finalized_preserves_grid() {
        let mut values = vec![0.0f32; SIFT_DESCRIPTOR_DIM];
        values[0] = 200.0;
        values[1] = 100.4; // rounds down
        values[2] = 100.6; // rounds up
        values[3] = 300.0; // clamps to 255

        let desc = U8Descriptor::try_from_finalized(&values).unwrap();
        assert_eq!(desc.as_slice()[0], 200);
        assert_eq!(desc.as_slice()[1], 100);
        assert_eq!(desc.as_slice()[2], 101);
        assert_eq!(desc.as_slice()[3], 255);

        // `scale` is the identity for grid values: reconstruction must recover the
        // original magnitudes, not shrink them by 255x.
        assert_eq!(desc.scale(), 255.0);
        let recovered = F32Descriptor::from(&desc);
        assert!((recovered.as_slice()[0] - 200.0).abs() < 1e-3);
        let zero = U8Descriptor::try_from_finalized(&vec![0.0f32; SIFT_DESCRIPTOR_DIM]).unwrap();
        // 200² + 100² + 101² + 255² = 40000 + 10000 + 10201 + 65025
        let distance = squared_l2(
            reconstructed(&desc).as_slice(),
            reconstructed(&zero).as_slice(),
        );
        assert!((distance - 125_226.0).abs() < 1.0);
    }

    #[test]
    fn per_descriptor_quantization_records_its_own_scale() {
        let values = make_test_root_sift();
        let desc = U8Descriptor::from_root_sift(&values).unwrap();
        assert_eq!(desc.scale(), values.iter().copied().fold(0.0f32, f32::max));
    }

    /// Regression guard for the per-descriptor-max quantization defect: on the shared
    /// `[0, 255]` grid, `b` is a far-away descriptor (`q / 2`) while `a` is nearly
    /// identical to `q`. `from_root_sift` stretches each descriptor by its own peak,
    /// which makes `b` look *identical* to `q` in raw-byte space and flips the nearest
    /// neighbor; `try_from_finalized` keeps the correct ordering.
    #[test]
    fn test_from_finalized_keeps_raw_byte_nearest_neighbour_correct() {
        let mk = |a: f32, b: f32| {
            let mut v = vec![0.0f32; SIFT_DESCRIPTOR_DIM];
            v[0] = a;
            v[1] = b;
            v
        };
        let (q, near, far) = (mk(200.0, 100.0), mk(198.0, 102.0), mk(100.0, 50.0));

        let raw_sq = |x: &U8Descriptor, y: &U8Descriptor| -> u32 {
            x.as_slice()
                .iter()
                .zip(y.as_slice())
                .map(|(p, r)| {
                    let d = *p as i32 - *r as i32;
                    (d * d) as u32
                })
                .sum()
        };

        // Correct behavior on the shared grid.
        let (qf, nf, ff) = (
            U8Descriptor::try_from_finalized(&q).unwrap(),
            U8Descriptor::try_from_finalized(&near).unwrap(),
            U8Descriptor::try_from_finalized(&far).unwrap(),
        );
        assert!(
            raw_sq(&qf, &nf) < raw_sq(&qf, &ff),
            "try_from_finalized must rank the genuinely-near descriptor first"
        );

        // The old path inverts it: `far` is a scalar multiple of `q`, so peak-stretching
        // maps both onto the same bytes (distance 0) while `near` drifts away.
        let (qr, nr, fr) = (
            U8Descriptor::from_root_sift(&q).unwrap(),
            U8Descriptor::from_root_sift(&near).unwrap(),
            U8Descriptor::from_root_sift(&far).unwrap(),
        );
        assert!(
            raw_sq(&qr, &fr) < raw_sq(&qr, &nr),
            "documents why raw-byte matchers must not use from_root_sift"
        );
    }

    #[test]
    fn test_f32_vs_uint8_distance_correlation() {
        // Create two different descriptors
        let rootsift_a = make_test_root_sift();
        let mut rootsift_b = make_test_root_sift();

        // Modify B to be different
        for item in rootsift_b.iter_mut().take(SIFT_DESCRIPTOR_DIM / 2) {
            *item *= 0.5;
        }
        // Re-normalize B
        let norm: f32 = rootsift_b.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut rootsift_b {
            *x /= norm;
        }

        let f32_a = F32Descriptor::from_root_sift(&rootsift_a).unwrap();
        let f32_b = F32Descriptor::from_root_sift(&rootsift_b).unwrap();
        let u8_a = U8Descriptor::from_root_sift(&rootsift_a).unwrap();
        let u8_b = U8Descriptor::from_root_sift(&rootsift_b).unwrap();

        let f32_dist = squared_l2(f32_a.as_slice(), f32_b.as_slice());
        let u8_dist = squared_l2(
            reconstructed(&u8_a).as_slice(),
            reconstructed(&u8_b).as_slice(),
        );

        // Distances should be correlated (not necessarily equal due to quantization)
        // Check that they're in the same ballpark (within 10% for this test case)
        let ratio = u8_dist / f32_dist;
        assert!(
            ratio > 0.8 && ratio < 1.2,
            "u8 and f32 distances should be correlated: f32={}, u8={}, ratio={}",
            f32_dist,
            u8_dist,
            ratio
        );
    }

    #[test]
    fn test_conversion_roundtrip() {
        let root_sift = make_test_root_sift();
        let f32_original = F32Descriptor::from_root_sift(&root_sift).unwrap();

        // Convert to u8 and back
        let u8_desc = U8Descriptor::from(&f32_original);
        let f32_reconstructed = F32Descriptor::from(&u8_desc);

        // Check reconstruction error (should be small but not zero due to quantization)
        let reconstruction_dist = squared_l2(f32_original.as_slice(), f32_reconstructed.as_slice());
        assert!(
            reconstruction_dist < 0.01, // Allow some quantization error
            "Reconstruction error too high: {}",
            reconstruction_dist
        );
    }

    #[test]
    fn test_distance_ordering_preserved() {
        // Test that u8 quantization preserves distance ordering
        let base = make_test_root_sift();

        // Create descriptors at different distances from base
        let make_variant = |offset: f32| {
            let mut desc = base.clone();
            for val in desc.iter_mut().take(SIFT_DESCRIPTOR_DIM / 4) {
                *val += offset;
            }
            let norm: f32 = desc.iter().map(|x| x * x).sum::<f32>().sqrt();
            desc.iter().map(|x| x / norm).collect::<Vec<f32>>()
        };

        let near = make_variant(0.1);
        let far = make_variant(0.5);

        let base_f32 = F32Descriptor::from_root_sift(&base).unwrap();
        let near_f32 = F32Descriptor::from_root_sift(&near).unwrap();
        let far_f32 = F32Descriptor::from_root_sift(&far).unwrap();

        let base_u8 = U8Descriptor::from_root_sift(&base).unwrap();
        let near_u8 = U8Descriptor::from_root_sift(&near).unwrap();
        let far_u8 = U8Descriptor::from_root_sift(&far).unwrap();

        // Check that ordering is preserved
        let f32_near_dist = squared_l2(base_f32.as_slice(), near_f32.as_slice());
        let f32_far_dist = squared_l2(base_f32.as_slice(), far_f32.as_slice());
        let base_u8 = reconstructed(&base_u8);
        let u8_near_dist = squared_l2(base_u8.as_slice(), reconstructed(&near_u8).as_slice());
        let u8_far_dist = squared_l2(base_u8.as_slice(), reconstructed(&far_u8).as_slice());

        assert!(
            f32_near_dist < f32_far_dist,
            "f32: near should be closer than far"
        );
        assert!(
            u8_near_dist < u8_far_dist,
            "u8: near should be closer than far"
        );
    }

    #[test]
    fn every_public_descriptor_conversion_rejects_non_finite_bins() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = [0.0; SIFT_DESCRIPTOR_DIM];
            values[7] = value;
            let rejected = |result: Result<(), DescriptorError>| {
                assert_eq!(result, Err(DescriptorError::NonFiniteValue { index: 7 }));
            };
            rejected(F32Descriptor::try_new(values).map(drop));
            rejected(F32Descriptor::from_root_sift(&values).map(drop));
            rejected(F32Descriptor::try_from_finalized(&values).map(drop));
            rejected(F32Descriptor::try_from(values.as_slice()).map(drop));
            rejected(F32Descriptor::try_from(values.to_vec()).map(drop));
            rejected(U8Descriptor::from_root_sift(&values).map(drop));
            rejected(U8Descriptor::try_from_finalized(&values).map(drop));
        }
        for value in [1e-40, 1e-10, 1.00001e-10, f32::MAX] {
            let values = [value; SIFT_DESCRIPTOR_DIM];
            let float = F32Descriptor::try_new(values).unwrap();
            let quantized = U8Descriptor::from(&float);
            assert!(quantized.scale().is_finite());
            assert!(
                F32Descriptor::from(&quantized)
                    .as_array()
                    .iter()
                    .all(|v| v.is_finite())
            );
        }
    }

    #[test]
    fn test_zero_descriptor() {
        let zeros = vec![0.0f32; SIFT_DESCRIPTOR_DIM];

        let f32_desc = F32Descriptor::from_root_sift(&zeros).unwrap();
        let u8_desc = U8Descriptor::from_root_sift(&zeros).unwrap();

        // Zero descriptor should have zero distance to itself
        assert_eq!(squared_l2(f32_desc.as_slice(), f32_desc.as_slice()), 0.0);
        // u8 zero descriptor - scale factor defaults to 1.0 for edge case
        assert_eq!(u8_desc.scale(), 1.0);
    }

    #[test]
    fn test_edge_case_single_nonzero() {
        // Descriptor with only one non-zero value
        let mut desc = vec![0.0f32; SIFT_DESCRIPTOR_DIM];
        desc[0] = 1.0;

        let u8_desc = U8Descriptor::from_root_sift(&desc).unwrap();
        // First element should be 255, rest should be 0
        assert_eq!(u8_desc.as_slice()[0], 255);
        for i in 1..SIFT_DESCRIPTOR_DIM {
            assert_eq!(u8_desc.as_slice()[i], 0);
        }
    }
}
