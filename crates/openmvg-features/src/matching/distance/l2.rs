// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! L2 (Euclidean) distance metric implementation.

use crate::matching::distance::common::{
    is_valid_f32_distance_descriptor, is_valid_f64_distance_descriptor, l2_f32, l2_f64,
};
use crate::matching::traits::DistanceMetric;

/// L2 (Euclidean) distance metric.
///
/// Computes the L2 norm: sqrt(sum((a_i - b_i)²))
///
/// For nearest neighbor search where only relative ordering matters,
/// consider using [`SquaredL2Distance`](super::SquaredL2Distance) instead
/// to avoid the sqrt operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct L2Distance;

impl L2Distance {
    /// Create an L2 distance metric.
    pub fn new() -> Self {
        Self
    }
}

impl DistanceMetric<f32> for L2Distance {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        l2_f32(a, b)
    }

    fn is_valid_descriptor(&self, descriptor: &[f32]) -> bool {
        is_valid_f32_distance_descriptor(descriptor)
    }
}

impl DistanceMetric<f64> for L2Distance {
    #[inline]
    fn distance(&self, a: &[f64], b: &[f64]) -> f32 {
        l2_f64(a, b) as f32
    }

    fn is_valid_descriptor(&self, descriptor: &[f64]) -> bool {
        is_valid_f64_distance_descriptor(descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance_f32() {
        let metric = L2Distance::new();
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let distance = metric.distance(&a, &b);
        let expected = 27.0_f32.sqrt(); // sqrt(3² + 3² + 3²)
        assert!((distance - expected).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_f64() {
        let metric = L2Distance::new();
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let distance = metric.distance(&a, &b);
        let expected = 27.0_f64.sqrt() as f32;
        assert!((distance - expected).abs() < 1e-6);
    }
}
