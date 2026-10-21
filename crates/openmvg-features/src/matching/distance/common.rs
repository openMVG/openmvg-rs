// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared distance computation utilities.
//!
//! This module provides distance computations that LLVM can auto-vectorize.
//! All functions return 0.0 for empty descriptors.

/// Conservative per-element bound that keeps every pairwise squared distance
/// representable as `f32`, with headroom for floating-point accumulation.
fn f32_distance_element_bound(dimension: usize) -> f64 {
    ((f32::MAX as f64) / (8.0 * dimension.max(1) as f64)).sqrt()
}

pub(crate) fn is_valid_f32_distance_descriptor(descriptor: &[f32]) -> bool {
    let bound = f32_distance_element_bound(descriptor.len()) as f32;
    descriptor
        .iter()
        .all(|value| value.is_finite() && value.abs() <= bound)
}

pub(crate) fn is_valid_f64_distance_descriptor(descriptor: &[f64]) -> bool {
    let bound = f32_distance_element_bound(descriptor.len());
    descriptor
        .iter()
        .all(|value| value.is_finite() && value.abs() <= bound)
}

// =============================================================================
// Primary API
// =============================================================================

/// Compute the squared L2 distance between two f32 slices.
///
/// Uses an AVX2 kernel for 128-D descriptors on supported x86-64 hosts and an
/// allocation-free portable fallback elsewhere. Both sum in the same order, so
/// every target returns bit-identical distances. Returns 0.0 for empty descriptors.
///
/// # Panics
///
/// Panics if the slice lengths differ.
///
/// # Examples
/// ```
/// use openmvg_features::matching::squared_l2_f32;
/// let a = vec![1.0f32, 2.0, 3.0];
/// let b = vec![4.0f32, 5.0, 6.0];
/// let dist = squared_l2_f32(&a, &b);
/// assert!((dist - 27.0).abs() < 1e-6); // (3² + 3² + 3²)
/// ```
#[inline]
pub fn squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "descriptor lengths must match");

    if a.is_empty() {
        return 0.0;
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    if a.len() == 128
        && let pulp::Arch::V3(simd) = runtime_arch()
    {
        return squared_l2_f32_v3(simd, a, b);
    }

    squared_l2_f32_blocked(a, b)
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn runtime_arch() -> pulp::Arch {
    static ARCH: std::sync::OnceLock<pulp::Arch> = std::sync::OnceLock::new();
    *ARCH.get_or_init(pulp::Arch::new)
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub(crate) fn squared_l2_f32_v3(simd: pulp::x86::V3, a: &[f32], b: &[f32]) -> f32 {
    let Ok(a) = <&[f32; 128]>::try_from(a) else {
        return squared_l2_f32_blocked(a, b);
    };
    let Ok(b) = <&[f32; 128]>::try_from(b) else {
        return squared_l2_f32_blocked(a, b);
    };

    struct Operation<'a> {
        simd: pulp::x86::V3,
        a: &'a [f32; 128],
        b: &'a [f32; 128],
    }

    impl pulp::NullaryFnOnce for Operation<'_> {
        type Output = f32;

        #[inline(always)]
        fn call(self) -> Self::Output {
            squared_l2_f32_v3_inner(self.simd, self.a, self.b)
        }
    }

    simd.vectorize(Operation { simd, a, b })
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub(crate) fn squared_l2_f32_v3_inner(simd: pulp::x86::V3, a: &[f32; 128], b: &[f32; 128]) -> f32 {
    use pulp::{Simd, cast, f32x8};

    let (a_blocks, a_tail) = pulp::as_arrays::<8, _>(a);
    let (b_blocks, b_tail) = pulp::as_arrays::<8, _>(b);
    let mut sums = [simd.splat_f32x8(0.0); 4];
    for (block, (left, right)) in a_blocks.iter().zip(b_blocks).enumerate() {
        let left: f32x8 = cast(*left);
        let right: f32x8 = cast(*right);
        let difference = simd.sub_f32x8(left, right);
        sums[block % 4] = simd.add_f32x8(sums[block % 4], simd.mul_f32x8(difference, difference));
    }
    let total = simd.add_f32x8(
        simd.add_f32x8(sums[0], sums[1]),
        simd.add_f32x8(sums[2], sums[3]),
    );
    debug_assert!(a_tail.is_empty() && b_tail.is_empty());
    simd.reduce_sum_f32s(total)
}

/// Portable squared L2 in the summation order of the AVX2 kernel: element `i`
/// accumulates into lane `i % 8` of partial sum `(i / 8) % 4`, then partial
/// sums and lanes are reduced pairwise exactly as `pulp`'s x86-v3 reduction.
fn squared_l2_f32_blocked(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [[0.0f32; 8]; 4];
    for (block, (left, right)) in a.chunks(8).zip(b.chunks(8)).enumerate() {
        let sum = &mut sums[block % 4];
        for (lane, (&left, &right)) in left.iter().zip(right).enumerate() {
            let difference = left - right;
            sum[lane] += difference * difference;
        }
    }
    let lanes: [f32; 8] = core::array::from_fn(|lane| {
        (sums[0][lane] + sums[1][lane]) + (sums[2][lane] + sums[3][lane])
    });
    let halves: [f32; 4] = core::array::from_fn(|lane| lanes[lane] + lanes[lane + 4]);
    (halves[0] + halves[2]) + (halves[1] + halves[3])
}

/// Compute the squared L2 distance between two f64 slices.
///
/// Returns 0.0 for empty descriptors.
///
/// # Panics
///
/// Panics if the slice lengths differ.
#[inline]
pub fn squared_l2_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "descriptor lengths must match");

    if a.is_empty() {
        return 0.0;
    }

    squared_l2_f64_loop(a, b)
}

/// Compute the L2 (Euclidean) distance between two f32 slices.
///
/// This is `sqrt(squared_l2_f32(a, b))`.
/// For nearest neighbor search, prefer `squared_l2_f32` to avoid sqrt overhead.
///
/// # Panics
///
/// Panics if the slice lengths differ.
#[inline]
pub fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
    squared_l2_f32(a, b).sqrt()
}

/// Compute the L2 (Euclidean) distance between two f64 slices.
///
/// This is `sqrt(squared_l2_f64(a, b))`.
/// For nearest neighbor search, prefer `squared_l2_f64` to avoid sqrt overhead.
///
/// # Panics
///
/// Panics if the slice lengths differ.
#[inline]
pub fn l2_f64(a: &[f64], b: &[f64]) -> f64 {
    squared_l2_f64(a, b).sqrt()
}

// =============================================================================
// Scalar reference implementations (crate-private)
// =============================================================================

/// Sequential-loop squared L2 distance (f32), the scalar test oracle for
/// [`squared_l2_f32`].
#[cfg(test)]
#[inline]
pub(crate) fn squared_l2_f32_loop(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "descriptor lengths must match");

    if a.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Sequential-loop squared L2 distance (f64), the kernel behind
/// [`squared_l2_f64`].
#[inline]
pub(crate) fn squared_l2_f64_loop(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "descriptor lengths must match");

    if a.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_l2_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let result = squared_l2_f32(&a, &b);
        assert!((result - 27.0).abs() < 1e-6); // 3² + 3² + 3² = 27
    }

    #[test]
    fn test_squared_l2_f64() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let result = squared_l2_f64(&a, &b);
        assert!((result - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_f32() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![3.0f32, 4.0, 0.0];
        let result = l2_f32(&a, &b);
        assert!((result - 5.0).abs() < 1e-6); // 3-4-5 triangle
    }

    #[test]
    fn test_l2_f64() {
        let a = vec![0.0f64, 0.0, 0.0];
        let b = vec![3.0f64, 4.0, 0.0];
        let result = l2_f64(&a, &b);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_identical_descriptors() {
        let a = vec![1.0f32, 2.0, 3.0];
        assert!(squared_l2_f32(&a, &a).abs() < 1e-10);
        assert!(l2_f32(&a, &a).abs() < 1e-10);
    }

    #[test]
    fn test_empty_descriptors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(squared_l2_f32(&a, &b), 0.0);
        assert_eq!(l2_f32(&a, &b), 0.0);

        let a64: Vec<f64> = vec![];
        let b64: Vec<f64> = vec![];
        assert_eq!(squared_l2_f64(&a64, &b64), 0.0);
        assert_eq!(l2_f64(&a64, &b64), 0.0);
    }

    #[test]
    #[should_panic(expected = "descriptor lengths must match")]
    fn mismatched_lengths_are_rejected_before_simd() {
        let left = vec![0.0f32; 128];
        let right = vec![0.0f32; 127];
        let _ = squared_l2_f32(&left, &right);
    }

    #[test]
    fn test_single_element() {
        let a = vec![5.0f32];
        let b = vec![3.0f32];
        assert!((squared_l2_f32(&a, &b) - 4.0).abs() < 1e-6); // (5-3)² = 4
        assert!((l2_f32(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn portable_and_simd_distances_are_bit_identical() {
        let mut state = 0x9e37_79b9_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 24) as f32
        };
        for (dimension, scale) in [(128, 1e-3), (128, 1.0), (128, 255.0), (61, 1.0)] {
            for _ in 0..200 {
                let a: Vec<f32> = (0..dimension).map(|_| next() * scale).collect();
                let b: Vec<f32> = (0..dimension).map(|_| next() * scale).collect();
                assert_eq!(
                    squared_l2_f32(&a, &b).to_bits(),
                    squared_l2_f32_blocked(&a, &b).to_bits()
                );
            }
        }
    }

    #[test]
    fn test_primary_vs_loop_equivalence() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32 * 0.1).collect();

        let primary_result = squared_l2_f32(&a, &b);
        let loop_result = squared_l2_f32_loop(&a, &b);

        assert!(
            (primary_result - loop_result).abs() < 1e-4,
            "primary and loop results should match: {} vs {}",
            primary_result,
            loop_result
        );
    }
}
