// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Squared L2 (Euclidean) distance metric implementation.

use crate::matching::distance::common::{
    is_valid_f32_distance_descriptor, is_valid_f64_distance_descriptor, squared_l2_f32,
    squared_l2_f64,
};
use crate::matching::traits::DistanceMetric;

/// Squared L2 (Euclidean) distance metric.
///
/// Computes the squared L2 norm: sum((a_i - b_i)²)
///
/// This is faster than [`L2Distance`](super::L2Distance) as it avoids the
/// sqrt() operation. Use this when you only need to compare relative
/// distances (e.g., finding nearest neighbors) since sqrt is monotonic.
///
/// # Performance
///
/// 128-value `f32` descriptors use AVX2 on x86-64 hosts that have it, with no option to
/// select; every target returns the same distances bit for bit.
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredL2Distance;

impl SquaredL2Distance {
    /// Create a squared L2 distance metric.
    pub fn new() -> Self {
        Self
    }
}

impl DistanceMetric<f32> for SquaredL2Distance {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        squared_l2_f32(a, b)
    }

    fn is_valid_descriptor(&self, descriptor: &[f32]) -> bool {
        is_valid_f32_distance_descriptor(descriptor)
    }

    fn fold_distances<'a, I, R, F>(
        &self,
        query: &[f32],
        candidates: I,
        initial: R,
        mut fold: F,
    ) -> R
    where
        I: IntoIterator<Item = &'a [f32]>,
        F: FnMut(R, usize, f32) -> R,
    {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if query.len() == 128
            && let pulp::Arch::V3(simd) = super::common::runtime_arch()
        {
            struct Operation<'query, I, R, F> {
                simd: pulp::x86::V3,
                query: &'query [f32; 128],
                candidates: I,
                initial: R,
                fold: F,
            }

            impl<'query, 'candidate, I, R, F> pulp::NullaryFnOnce for Operation<'query, I, R, F>
            where
                I: IntoIterator<Item = &'candidate [f32]>,
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
                            super::common::squared_l2_f32_v3_inner(self.simd, self.query, candidate)
                        } else {
                            squared_l2_f32(self.query, candidate)
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

impl DistanceMetric<f64> for SquaredL2Distance {
    #[inline]
    fn distance(&self, a: &[f64], b: &[f64]) -> f32 {
        squared_l2_f64(a, b) as f32
    }

    fn is_valid_descriptor(&self, descriptor: &[f64]) -> bool {
        is_valid_f64_distance_descriptor(descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_l2_distance_f64() {
        let metric = SquaredL2Distance::new();
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let distance = metric.distance(&a, &b);
        assert!((distance - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_squared_vs_regular_l2() {
        use crate::matching::distance::L2Distance;

        let metric_sq = SquaredL2Distance::new();
        let metric_l2 = L2Distance::new();

        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];

        let sq_dist = metric_sq.distance(&a, &b);
        let l2_dist = metric_l2.distance(&a, &b);

        // sqrt(squared_dist) should equal l2_dist
        assert!((sq_dist.sqrt() - l2_dist).abs() < 1e-5);
    }

    #[test]
    fn operation_scoped_fold_matches_individual_distances() {
        let metric = SquaredL2Distance::new();
        let query: Vec<f32> = (0..128).map(|index| index as f32 * 0.25).collect();
        let candidates: Vec<Vec<f32>> = (0..5)
            .map(|offset| {
                (0..128)
                    .map(|index| (index + offset) as f32 * 0.25)
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
