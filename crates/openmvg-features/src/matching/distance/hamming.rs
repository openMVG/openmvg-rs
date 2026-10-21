// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hamming distance metric for binary descriptors.
//!
//! Hamming distance counts the number of bit positions that differ between
//! two binary strings. This is the standard metric for binary descriptors
//! like ORB, BRIEF, BRISK, FREAK, etc.

use crate::matching::traits::DistanceMetric;

const MAX_HAMMING_BYTES: usize = u32::MAX as usize / u8::BITS as usize;

/// Hamming distance metric for binary descriptors.
///
/// Counts the number of differing bits between two binary descriptor vectors.
/// Works with descriptors stored as `u8` byte arrays where each byte contains
/// 8 bits of the descriptor.
///
/// # Descriptor formats
///
/// - ORB: 256 bits (32 bytes)
/// - BRIEF: 128, 256, or 512 bits (16, 32, or 64 bytes)
/// - BRISK: 512 bits (64 bytes)
/// - FREAK: 512 bits (64 bytes)
///
/// # Examples
///
/// ```rust
/// use openmvg_features::matching::HammingDistance;
/// use openmvg_features::matching::DistanceMetric;
///
/// let metric = HammingDistance::new();
///
/// // Two ORB-like descriptors (32 bytes each)
/// let desc1 = vec![0b11110000u8; 32];
/// let desc2 = vec![0b11111111u8; 32];
///
/// let dist = metric.distance(&desc1, &desc2);
/// // Each byte differs by 4 bits, 32 bytes = 128 bits difference
/// assert_eq!(dist, 128.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct HammingDistance;

impl HammingDistance {
    /// Create a Hamming distance metric.
    pub fn new() -> Self {
        Self
    }
}

impl DistanceMetric<u8> for HammingDistance {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        hamming_u8(a, b) as f32
    }

    fn is_valid_descriptor(&self, descriptor: &[u8]) -> bool {
        descriptor.len() <= MAX_HAMMING_BYTES
    }
}

/// Compute the Hamming distance between two u8 slices (binary descriptors).
///
/// Counts the differing bits across all bytes. A build for a CPU with the POPCNT instruction
/// (for example `-C target-cpu=x86-64-v2`) counts them faster.
///
/// # Panics
///
/// Panics if the descriptor lengths differ or if the distance cannot fit in
/// the `u32` return type.
///
/// # Examples
///
/// ```rust
/// use openmvg_features::matching::hamming_u8;
///
/// let a = vec![0b11110000u8, 0b00001111];
/// let b = vec![0b11111111u8, 0b00000000];
///
/// let dist = hamming_u8(&a, &b);
/// assert_eq!(dist, 8); // 4 bits differ in each byte
/// ```
#[inline]
pub fn hamming_u8(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "descriptor lengths must match");
    assert!(
        a.len() <= MAX_HAMMING_BYTES,
        "Hamming distance exceeds u32 capacity"
    );

    let (a_words, a_tail) = a.as_chunks::<8>();
    let (b_words, b_tail) = b.as_chunks::<8>();
    let words: u32 = a_words
        .iter()
        .zip(b_words)
        .map(|(x, y)| (u64::from_le_bytes(*x) ^ u64::from_le_bytes(*y)).count_ones())
        .sum();
    let tail: u32 = a_tail
        .iter()
        .zip(b_tail)
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    words + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_u8_identical() {
        let a = vec![0b11110000u8, 0b00001111, 0b10101010];
        let b = vec![0b11110000u8, 0b00001111, 0b10101010];

        assert_eq!(hamming_u8(&a, &b), 0);
    }

    #[test]
    fn test_hamming_u8_all_different() {
        let a = vec![0b00000000u8, 0b00000000];
        let b = vec![0b11111111u8, 0b11111111];

        // 2 bytes * 8 bits = 16 bits difference
        assert_eq!(hamming_u8(&a, &b), 16);
    }

    #[test]
    fn test_hamming_u8_partial() {
        let a = vec![0b11110000u8];
        let b = vec![0b11111111u8];

        // 4 bits differ
        assert_eq!(hamming_u8(&a, &b), 4);
    }

    #[test]
    fn test_hamming_u8_empty() {
        let a: Vec<u8> = vec![];
        let b: Vec<u8> = vec![];

        assert_eq!(hamming_u8(&a, &b), 0);
    }

    #[test]
    fn test_hamming_distance_trait() {
        let metric = HammingDistance::new();

        let a = vec![0b11110000u8, 0b00001111];
        let b = vec![0b11111111u8, 0b00000000];

        // 4 bits differ in each byte = 8 total
        let dist = metric.distance(&a, &b);
        assert_eq!(dist, 8.0);
    }

    #[test]
    fn test_orb_descriptor_size() {
        // ORB descriptors are 256 bits = 32 bytes
        let orb1 = vec![0u8; 32];
        let orb2 = vec![255u8; 32];

        // All bits different
        let dist = hamming_u8(&orb1, &orb2);
        assert_eq!(dist, 256);
    }

    #[test]
    fn test_brief_descriptor_size() {
        // BRIEF-256 descriptors are 256 bits = 32 bytes
        let brief1 = vec![0u8; 32];
        let brief2 = vec![0u8; 32];

        let dist = hamming_u8(&brief1, &brief2);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_brisk_descriptor_size() {
        // BRISK descriptors are 512 bits = 64 bytes
        let brisk1 = vec![0b10101010u8; 64];
        let brisk2 = vec![0b01010101u8; 64];

        // Every bit differs
        let dist = hamming_u8(&brisk1, &brisk2);
        assert_eq!(dist, 512);
    }

    #[test]
    fn test_hamming_u8_matches_bytewise_reference_for_all_tail_lengths() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        };
        for len in 0..=67 {
            let a: Vec<u8> = (0..len).map(|_| next()).collect();
            let b: Vec<u8> = (0..len).map(|_| next()).collect();
            let reference: u32 = a.iter().zip(&b).map(|(x, y)| (x ^ y).count_ones()).sum();
            assert_eq!(hamming_u8(&a, &b), reference, "length {len}");
        }
    }
}
