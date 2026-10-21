// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! L2 (Euclidean) distance for u8 quantized descriptors.
//!
//! This module provides L2 distance computation for quantized
//! SIFT/RootSIFT descriptors stored as u8 arrays.
//!
//! The 128-D SIFT path uses AVX2 when available and falls back to the direct
//! scalar accumulation loop on other dimensions and targets.

use crate::matching::traits::DistanceMetric;

pub(crate) const MAX_SQUARED_L2_U8_DIMENSION: usize = u32::MAX as usize / 65_025;

/// Compute squared L2 distance between two u8 descriptor slices.
///
/// The integer sum is exact, so every target returns the same distance. Returns 0 for empty
/// descriptors.
///
/// # Panics
///
/// Panics if the slice lengths differ, or if a descriptor is longer than 66 051 bytes, where
/// the distance could overflow a `u32`.
///
/// # Examples
/// ```
/// use openmvg_features::matching::squared_l2_u8;
/// let a = vec![100u8, 150, 200];
/// let b = vec![110u8, 140, 190];
/// let dist = squared_l2_u8(&a, &b);
/// assert_eq!(dist, 300); // 10² + 10² + 10² = 300
/// ```
#[inline]
pub fn squared_l2_u8(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "descriptor lengths must match");
    assert!(
        a.len() <= MAX_SQUARED_L2_U8_DIMENSION,
        "u8 descriptor is too large for a u32 squared distance"
    );

    if a.is_empty() {
        return 0;
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    if a.len() == 128
        && let pulp::Arch::V3(simd) = super::common::runtime_arch()
    {
        return squared_l2_u8_v3(simd, a, b);
    }

    squared_l2_u8_scalar(a, b)
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub(crate) fn squared_l2_u8_v3(simd: pulp::x86::V3, a: &[u8], b: &[u8]) -> u32 {
    let Ok(a) = <&[u8; 128]>::try_from(a) else {
        return squared_l2_u8_scalar(a, b);
    };
    let Ok(b) = <&[u8; 128]>::try_from(b) else {
        return squared_l2_u8_scalar(a, b);
    };

    struct Operation<'a> {
        simd: pulp::x86::V3,
        a: &'a [u8; 128],
        b: &'a [u8; 128],
    }

    impl pulp::NullaryFnOnce for Operation<'_> {
        type Output = u32;

        #[inline(always)]
        fn call(self) -> Self::Output {
            squared_l2_u8_v3_inner(self.simd, self.a, self.b)
        }
    }

    simd.vectorize(Operation { simd, a, b })
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub(crate) fn squared_l2_u8_v3_inner(simd: pulp::x86::V3, a: &[u8; 128], b: &[u8; 128]) -> u32 {
    use pulp::{cast, i32x8, u8x16};

    let (a_blocks, a_tail) = pulp::as_arrays::<16, _>(a);
    let (b_blocks, b_tail) = pulp::as_arrays::<16, _>(b);
    let mut sum = simd.splat_i32x8(0);
    for (left, right) in a_blocks.iter().zip(b_blocks) {
        let left = simd.convert_u8x16_to_i16x16(cast::<_, u8x16>(*left));
        let right = simd.convert_u8x16_to_i16x16(cast::<_, u8x16>(*right));
        let difference = simd.wrapping_sub_i16x16(left, right);
        sum = simd.wrapping_add_i32x8(
            sum,
            simd.multiply_wrapping_add_adjacent_i16x16(difference, difference),
        );
    }
    let lanes: [i32; 8] = cast::<i32x8, _>(sum);
    let vector_sum = lanes.into_iter().sum::<i32>() as u32;
    debug_assert!(a_tail.is_empty() && b_tail.is_empty());
    vector_sum
}

/// Generic dynamic-size path using scalar loop (avoids heap allocation).
///
/// LLVM can auto-vectorize this loop for common descriptor sizes (32, 64, 256).
#[inline]
fn squared_l2_u8_scalar(a: &[u8], b: &[u8]) -> u32 {
    let mut sum: u32 = 0;
    for i in 0..a.len() {
        let diff = a[i] as i32 - b[i] as i32;
        sum += (diff * diff) as u32;
    }
    sum
}

/// Direct-loop squared L2 distance for u8 descriptors, the scalar test oracle
/// for [`squared_l2_u8`].
#[cfg(test)]
#[inline]
pub(crate) fn squared_l2_u8_loop(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "descriptor lengths must match");
    assert!(
        a.len() <= MAX_SQUARED_L2_U8_DIMENSION,
        "u8 descriptor is too large for a u32 squared distance"
    );

    if a.is_empty() {
        return 0;
    }

    let mut sum: u32 = 0;
    for i in 0..a.len() {
        let diff = a[i] as i32 - b[i] as i32;
        sum += (diff * diff) as u32;
    }
    sum
}

/// Compute L2 (Euclidean) distance between two u8 descriptor slices.
///
/// This is `sqrt(squared_l2_u8(a, b))`. For nearest neighbor search where only ordering
/// matters, prefer `squared_l2_u8` to avoid the sqrt overhead.
///
/// # Panics
///
/// As [`squared_l2_u8`]: on different slice lengths or a descriptor too long for a `u32` sum.
///
/// # Examples
/// ```
/// use openmvg_features::matching::l2_u8;
/// let a = vec![0u8, 0, 0];
/// let b = vec![3u8, 4, 0];
/// let dist = l2_u8(&a, &b);
/// assert!((dist - 5.0).abs() < 0.001); // 3-4-5 triangle
/// ```
#[inline]
pub fn l2_u8(a: &[u8], b: &[u8]) -> f32 {
    (squared_l2_u8(a, b) as f32).sqrt()
}

/// Squared L2 distance metric for u8 descriptors.
///
/// Implements the `DistanceMetric<u8>` trait for use with generic matchers.
/// Returns the squared distance as f32 (suitable for ordering).
///
/// For nearest neighbor search where only relative ordering matters,
/// this is more efficient than `L2DistanceU8` as it avoids the sqrt.
/// Its 128-dimensional path selects AVX2 automatically on supported x86-64
/// hosts and otherwise uses the scalar implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredL2DistanceU8;

impl SquaredL2DistanceU8 {
    /// Create a squared L2 distance metric for u8 descriptors.
    pub fn new() -> Self {
        Self
    }
}

impl DistanceMetric<u8> for SquaredL2DistanceU8 {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        squared_l2_u8(a, b) as f32
    }

    fn is_valid_descriptor(&self, descriptor: &[u8]) -> bool {
        descriptor.len() <= MAX_SQUARED_L2_U8_DIMENSION
    }

    fn fold_distances<'a, I, R, F>(&self, query: &[u8], candidates: I, initial: R, mut fold: F) -> R
    where
        I: IntoIterator<Item = &'a [u8]>,
        F: FnMut(R, usize, f32) -> R,
    {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if query.len() == 128
            && let pulp::Arch::V3(simd) = super::common::runtime_arch()
        {
            struct Operation<'query, I, R, F> {
                simd: pulp::x86::V3,
                query: &'query [u8; 128],
                candidates: I,
                initial: R,
                fold: F,
            }

            impl<'query, 'candidate, I, R, F> pulp::NullaryFnOnce for Operation<'query, I, R, F>
            where
                I: IntoIterator<Item = &'candidate [u8]>,
                F: FnMut(R, usize, f32) -> R,
            {
                type Output = R;

                // A plain loop keeps the kernel in the AVX2 function body;
                // `Iterator::fold` may be outlined without the target feature.
                #[inline(always)]
                fn call(self) -> Self::Output {
                    let mut fold = self.fold;
                    let mut state = self.initial;
                    for (index, candidate) in self.candidates.into_iter().enumerate() {
                        let distance = if let Ok(candidate) = candidate.try_into() {
                            squared_l2_u8_v3_inner(self.simd, self.query, candidate) as f32
                        } else {
                            squared_l2_u8(self.query, candidate) as f32
                        };
                        state = fold(state, index, distance);
                    }
                    state
                }
            }

            if let Ok(query) = query.try_into() {
                return simd.vectorize(Operation {
                    simd,
                    query,
                    candidates,
                    initial,
                    fold,
                });
            }
        }

        candidates
            .into_iter()
            .enumerate()
            .fold(initial, |state, (index, candidate)| {
                fold(state, index, self.distance(query, candidate))
            })
    }
}

/// L2 (Euclidean) distance metric for u8 descriptors.
///
/// Implements the `DistanceMetric<u8>` trait for use with generic matchers.
/// Returns the actual L2 distance (with sqrt).
///
/// For nearest neighbor search where only relative ordering matters,
/// consider using `SquaredL2DistanceU8` instead to avoid the sqrt overhead.
/// This metric shares the same automatic 128-dimensional SIMD dispatcher.
#[derive(Debug, Clone, Copy, Default)]
pub struct L2DistanceU8;

impl L2DistanceU8 {
    /// Create an L2 distance metric for u8 descriptors.
    pub fn new() -> Self {
        Self
    }
}

impl DistanceMetric<u8> for L2DistanceU8 {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        l2_u8(a, b)
    }

    fn is_valid_descriptor(&self, descriptor: &[u8]) -> bool {
        descriptor.len() <= MAX_SQUARED_L2_U8_DIMENSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_l2_u8_basic() {
        let a = vec![100u8, 150, 200, 50];
        let b = vec![110u8, 140, 190, 60];

        // Expected: (10)² + (10)² + (10)² + (10)² = 400
        let dist = squared_l2_u8(&a, &b);
        assert_eq!(dist, 400);
    }

    #[test]
    fn test_squared_l2_u8_empty() {
        let a: Vec<u8> = vec![];
        let b: Vec<u8> = vec![];
        assert_eq!(squared_l2_u8(&a, &b), 0);
    }

    #[test]
    fn test_l2_u8_345_triangle() {
        let a = vec![0u8, 0, 0];
        let b = vec![3u8, 4, 0];

        // Expected: sqrt(9 + 16) = 5
        let dist = l2_u8(&a, &b);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_ordering_preserved() {
        // Verify that distance ordering is consistent
        let base = vec![100u8; 128];
        let near = {
            let mut v = vec![100u8; 128];
            v[0] = 101; // Small change
            v
        };
        let far = {
            let mut v = vec![100u8; 128];
            for x in v.iter_mut().take(10) {
                *x = 200; // Larger change
            }
            v
        };

        let sq_near = squared_l2_u8(&base, &near);
        let sq_far = squared_l2_u8(&base, &far);
        let l2_near = l2_u8(&base, &near);
        let l2_far = l2_u8(&base, &far);

        assert!(
            sq_near < sq_far,
            "Squared distance ordering must be preserved"
        );
        assert!(l2_near < l2_far, "L2 distance ordering must be preserved");
    }

    #[test]
    fn test_primary_vs_loop_equivalence_128d() {
        // Verify that the primary and explicit-loop entry points agree for
        // 128-D SIFT descriptors.
        let a: Vec<u8> = (0..128).map(|i| (i * 2) as u8).collect();
        let b: Vec<u8> = (0..128).map(|i| (i * 2 + 1) as u8).collect();

        let primary = squared_l2_u8(&a, &b);
        let scalar = squared_l2_u8_loop(&a, &b);
        assert_eq!(
            primary, scalar,
            "primary and loop must agree: primary={}, loop={}",
            primary, scalar
        );
    }

    #[test]
    fn test_primary_vs_loop_max_difference() {
        let a = vec![0u8; 128];
        let b = vec![255u8; 128];

        let primary = squared_l2_u8(&a, &b);
        let scalar = squared_l2_u8_loop(&a, &b);
        assert_eq!(primary, scalar);
        assert_eq!(primary, 128 * 255 * 255);
    }

    #[test]
    fn test_primary_vs_loop_non_128d() {
        // Verify the generic (non-128) path also matches the loop
        let a: Vec<u8> = (0..64).map(|i| (i * 3) as u8).collect();
        let b: Vec<u8> = (0..64).map(|i| (i * 3 + 5) as u8).collect();

        let primary = squared_l2_u8(&a, &b);
        let scalar = squared_l2_u8_loop(&a, &b);
        assert_eq!(primary, scalar);
    }

    #[test]
    fn test_primary_vs_loop_random_128d() {
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..4_096 {
            let mut a = [0u8; 128];
            let mut b = [0u8; 128];
            for (left, right) in a.iter_mut().zip(&mut b) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *left = state as u8;
                *right = (state >> 24) as u8;
            }
            assert_eq!(squared_l2_u8(&a, &b), squared_l2_u8_loop(&a, &b));
        }
    }

    #[test]
    fn test_primary_vs_loop_single_lane_extremes_128d() {
        let b = [0u8; 128];
        for index in 0..128 {
            let mut a = [0u8; 128];
            a[index] = u8::MAX;
            assert_eq!(squared_l2_u8(&a, &b), 255 * 255);
        }
    }

    #[test]
    fn operation_scoped_fold_matches_individual_distances() {
        let metric = SquaredL2DistanceU8::new();
        let query: Vec<u8> = (0..128).map(|index| index as u8).collect();
        let candidates: Vec<Vec<u8>> = (0..5)
            .map(|offset| {
                (0..128)
                    .map(|index| (index as u8).wrapping_add(offset))
                    .collect()
            })
            .collect();
        let expected: Vec<f32> = candidates
            .iter()
            .map(|candidate| metric.distance(&query, candidate))
            .collect();

        let actual = metric.fold_distances(
            &query,
            candidates.iter().map(Vec::as_slice),
            Vec::new(),
            |mut distances, _, distance| {
                distances.push(distance);
                distances
            },
        );

        assert_eq!(actual, expected);
    }
}
