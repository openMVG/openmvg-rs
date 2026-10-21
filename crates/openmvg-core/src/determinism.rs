// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Versioned deterministic seed derivation and random transformations.
//!
//! This is the replay protocol shared by the OpenMVG-rs crates. Algorithms take a
//! caller-owned [`RunSeed`], derive child seeds per domain and stable ID with
//! [`RunSeed::derive`], and draw from an [`AlgorithmRng`]. They never depend on
//! a random engine or on `rand`'s evolving distribution implementations
//! directly, so dependency updates do not change the draws, and a seed gives the
//! same draws on every platform except where a method says otherwise, as
//! [`AlgorithmRng::standard_normal_f32`] does. A whole run with the same seed and
//! inputs replays exactly on the same target and toolchain, at any thread count;
//! across targets, floating-point results may differ.
//!
//! [`SEED_DERIVATION_ALGORITHM`] and [`RNG_ALGORITHM`] identify the two halves of
//! the protocol. Any change that alters derived
//! seeds or drawn values gets a new identifier; reports record it so a run can
//! be replayed.
//!
//! # Deriving streams
//!
//! Give each random task its own child seed instead of sharing one generator. The `domain`
//! names the task, and the `stable_ids` name the item it works on, such as a view or an image
//! pair. The draws for an item then depend only on the root seed and that item, not on the
//! order in which items are processed or on the thread count. `stable_ids` may be empty for a
//! task that runs once. Any string works as a domain. By convention the library's domains
//! start with the crate's name or stage, such as `geometry/...`, `sfm/...`, or `pipeline/...`,
//! and end with a version; start yours with your application's name so that the two never
//! meet.
//!
//! ```rust
//! use openmvg_core::{AlgorithmRng, RunSeed};
//!
//! let root = RunSeed::from_u64(2026);
//! // One stream per view, keyed by the view's ID.
//! let view_seed = |view: u64| root.derive("my-app/jitter/v1", &[view]);
//!
//! // The same seed gives the same draws, wherever and whenever it runs.
//! let (mut first, mut again) = (AlgorithmRng::new(view_seed(7)), AlgorithmRng::new(view_seed(7)));
//! assert_eq!(first.unit_f64(), again.unit_f64());
//! // Another view, or another domain, gives an independent stream.
//! assert_ne!(view_seed(7), view_seed(8));
//! assert_ne!(view_seed(7), root.derive("my-app/noise/v1", &[7]));
//! ```

use rand_chacha::ChaCha12Rng;
use rand_core::{RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

const SEED_TREE_DOMAIN: &[u8] = b"openmvg-rs/seed-tree/v2\0";
const UNIT_F64_SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);

/// Versioned root-to-child derivation used by [`RunSeed::derive`].
pub const SEED_DERIVATION_ALGORITHM: &str = "sha256-domain-separated-v2";

/// Versioned engine and transformation protocol used by [`AlgorithmRng`].
pub const RNG_ALGORITHM: &str = "chacha12-openmvg-transforms-v1";

/// A caller-owned 256-bit deterministic seed.
///
/// [`Self::from_u64`] builds one from a number; [`Self::from_bytes`] restores a complete seed.
/// `Display` writes the 32 bytes as 64 lowercase hexadecimal digits, in byte order, and `parse`
/// reads them back, so a seed recorded in a report replays exactly:
///
/// ```rust
/// use openmvg_core::RunSeed;
///
/// let text = RunSeed::from_u64(1).to_string();
/// assert_eq!(text, format!("01{}", "0".repeat(62)));
/// assert_eq!(text.parse::<RunSeed>()?, RunSeed::from_u64(1));
/// # Ok::<(), openmvg_core::determinism::ParseRunSeedError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunSeed([u8; 32]);

impl RunSeed {
    /// The all-zero seed, which `RunSeed::default()` returns.
    ///
    /// The library's own defaults use other seeds: `openmvg-geometry`'s `RobustOptions` and
    /// `openmvg-sfm`'s `Dataset` both start from `RunSeed::from_u64(42)`.
    pub const ZERO: Self = Self([0; 32]);

    /// Expand a 64-bit seed into the 256-bit seed layout.
    ///
    /// The little-endian value occupies the first eight bytes and the
    /// remaining bytes are zero. This mapping is part of the protocol.
    pub const fn from_u64(value: u64) -> Self {
        let bytes = value.to_le_bytes();
        Self([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ])
    }

    /// Construct a seed without transforming its bytes.
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Return the exact seed bytes.
    pub(crate) const fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    /// Derive an independent child seed for the task `domain` and the item `stable_ids`.
    ///
    /// The child depends on the parent seed, the domain and every ID in order, and on nothing
    /// else; [Deriving streams](crate::determinism#deriving-streams) shows how to choose them.
    pub fn derive(self, domain: &str, stable_ids: &[u64]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(SEED_TREE_DOMAIN);
        hasher.update(self.0);
        hasher.update((domain.len() as u64).to_le_bytes());
        hasher.update(domain.as_bytes());
        hasher.update((stable_ids.len() as u64).to_le_bytes());
        for id in stable_ids {
            hasher.update(id.to_le_bytes());
        }
        Self(hasher.finalize().into())
    }
}

impl From<u64> for RunSeed {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl From<[u8; 32]> for RunSeed {
    fn from(value: [u8; 32]) -> Self {
        Self::from_bytes(value)
    }
}

impl fmt::Display for RunSeed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Why a text is not a [`RunSeed`]: a seed is the 64 hexadecimal digits that its `Display`
/// writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ParseRunSeedError {
    /// The text, without surrounding whitespace, does not have 64 characters.
    #[error("a run seed has 64 hexadecimal digits, not {0} characters")]
    Length(usize),
    /// The character at this position, counted in characters from zero, is not a hexadecimal
    /// digit.
    #[error("the run seed has a non-hexadecimal character at position {0}")]
    NotHex(usize),
}

impl std::str::FromStr for RunSeed {
    type Err = ParseRunSeedError;

    /// Read the 64 hexadecimal digits that `Display` writes, in either case, ignoring
    /// surrounding whitespace.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let text = text.trim();
        let count = text.chars().count();
        if count != 64 {
            return Err(ParseRunSeedError::Length(count));
        }
        let mut digits = text.chars().enumerate().map(|(position, character)| {
            character
                .to_digit(16)
                .ok_or(ParseRunSeedError::NotHex(position))
        });
        let mut bytes = [0; 32];
        for byte in &mut bytes {
            // The length check guarantees two characters per byte.
            let (Some(high), Some(low)) = (digits.next(), digits.next()) else {
                return Err(ParseRunSeedError::Length(count));
            };
            // Two hexadecimal digits always fit in one byte.
            *byte = (high? << 4 | low?) as u8;
        }
        Ok(Self(bytes))
    }
}

/// The random engine used by every supported randomized algorithm.
///
/// The engine and every draw method are versioned together by [`RNG_ALGORITHM`]: the same seed
/// gives the same draws on every platform, except where a method says otherwise.
#[derive(Clone, Debug)]
pub struct AlgorithmRng {
    inner: ChaCha12Rng,
    normal_spare: Option<f64>,
}

impl AlgorithmRng {
    /// Start a deterministic stream from a complete seed.
    pub fn new(seed: RunSeed) -> Self {
        Self {
            inner: ChaCha12Rng::from_seed(seed.into_bytes()),
            normal_spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Draw uniformly from `[0, 1)` using the high 53 bits of one word.
    pub fn unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * UNIT_F64_SCALE
    }

    /// Draw uniformly from `(0, 1)` using the high 53 bits of one word.
    pub fn open_unit_f64(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) * UNIT_F64_SCALE
    }

    /// Draw uniformly from `[-1, 1)`.
    pub fn symmetric_unit_f64(&mut self) -> f64 {
        2.0 * self.unit_f64() - 1.0
    }

    /// Draw an unbiased index from `0..upper`, or return `None` for an empty range.
    pub fn index(&mut self, upper: usize) -> Option<usize> {
        let upper = u64::try_from(upper).ok()?;
        if upper == 0 {
            return None;
        }
        let rejection_floor = 0_u64.wrapping_sub(upper) % upper;
        loop {
            let word = self.next_u64();
            if word >= rejection_floor {
                return usize::try_from(word % upper).ok();
            }
        }
    }

    /// Shuffle a slice with the versioned Fisher-Yates transformation.
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for upper in (2..=values.len()).rev() {
            if let Some(index) = self.index(upper) {
                values.swap(upper - 1, index);
            }
        }
    }

    /// Draw `count` distinct indices from `0..population` into the first `count` slots of
    /// `output`, without allocating.
    ///
    /// Returns `false` and leaves `output` unchanged when `population` is zero, `count` exceeds
    /// it, or `output` holds fewer than `count` slots.
    #[must_use]
    pub fn sample_indices(
        &mut self,
        population: usize,
        count: usize,
        output: &mut [usize],
    ) -> bool {
        if population == 0 || count > population || output.len() < count {
            return false;
        }
        for (slot, upper) in (population - count..population).enumerate() {
            let Some(candidate) = self.index(upper + 1) else {
                return false;
            };
            output[slot] = if output[..slot].contains(&candidate) {
                upper
            } else {
                candidate
            };
        }
        true
    }

    /// Draw one standard-normal `f32` with a versioned Marsaglia polar transform.
    ///
    /// This transform uses the target toolchain's `ln` and `sqrt`; bit-exact
    /// replay is promised within the declared target/toolchain, not across targets.
    pub fn standard_normal_f32(&mut self) -> f32 {
        if let Some(spare) = self.normal_spare.take() {
            return spare as f32;
        }
        loop {
            let first = 2.0 * self.unit_f64() - 1.0;
            let second = 2.0 * self.unit_f64() - 1.0;
            let radius_squared = first * first + second * second;
            if radius_squared > 0.0 && radius_squared < 1.0 {
                let scale = (-2.0 * radius_squared.ln() / radius_squared).sqrt();
                self.normal_spare = Some(second * scale);
                return (first * scale) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u64_seed_layout_is_explicit() {
        assert_eq!(
            RunSeed::from_u64(0x0807_0605_0403_0201).into_bytes(),
            [
                1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ]
        );
    }

    #[test]
    fn seed_domains_and_ids_are_separated() {
        let root = RunSeed::from_u64(7);
        assert_eq!(
            root.derive("features/v1", &[1, 2]),
            root.derive("features/v1", &[1, 2])
        );
        assert_ne!(
            root.derive("features/v1", &[1, 2]),
            root.derive("matching/v1", &[1, 2])
        );
        assert_ne!(
            root.derive("features/v1", &[1, 2]),
            root.derive("features/v1", &[2, 1])
        );
    }

    /// Versioned deterministic RNG protocol: the pinned ChaCha12 stream has fixed raw-word vectors;
    /// shuffle and sampling transformations have fixed outputs; the project-owned standard-normal
    /// transform has fixed output bits.
    #[test]
    fn algorithm_stream_and_transforms_are_replayable() {
        let seed = RunSeed::from_u64(42);
        let mut first = AlgorithmRng::new(seed);
        let mut second = AlgorithmRng::new(seed);
        let first_words = [first.next_u64(), first.next_u64(), first.next_u64()];
        let second_words = [second.next_u64(), second.next_u64(), second.next_u64()];
        assert_eq!(first_words, second_words);
        assert_eq!(
            first_words,
            [
                7_969_544_298_883_215_725,
                2_971_366_125_662_693_780,
                12_399_091_586_092_985_278,
            ]
        );

        let mut first = AlgorithmRng::new(seed);
        let mut second = AlgorithmRng::new(seed);
        let mut first_values: Vec<_> = (0..32).collect();
        let mut second_values = first_values.clone();
        first.shuffle(&mut first_values);
        second.shuffle(&mut second_values);
        assert_eq!(first_values, second_values);
        assert_eq!(
            first_values,
            [
                20, 27, 23, 22, 17, 12, 10, 29, 14, 0, 24, 8, 5, 26, 30, 21, 7, 6, 25, 15, 4, 2,
                16, 11, 3, 18, 31, 1, 19, 28, 9, 13,
            ]
        );

        let mut first_sample = [0; 8];
        let mut second_sample = [0; 8];
        assert!(first.sample_indices(100, 8, &mut first_sample));
        assert!(second.sample_indices(100, 8, &mut second_sample));
        assert_eq!(first_sample, second_sample);
        assert_eq!(first_sample, [83, 43, 38, 95, 45, 56, 35, 59]);

        let mut normal = AlgorithmRng::new(seed);
        assert_eq!(
            core::array::from_fn::<_, 4, _>(|_| normal.standard_normal_f32().to_bits()),
            [3_195_316_773, 3_214_442_332, 1_066_355_758, 1_067_114_093]
        );
    }

    #[test]
    fn chacha12_known_answer_crosses_the_64_byte_block_boundary() {
        let seed = RunSeed::from_bytes(core::array::from_fn(|index| index as u8));
        let mut rng = AlgorithmRng::new(seed);

        assert_eq!(
            core::array::from_fn::<_, 10, _>(|_| rng.next_u64()),
            [
                6_829_280_927_315_210_738,
                12_268_062_495_221_155_140,
                13_566_740_668_459_520_841,
                3_898_457_950_037_656_553,
                11_492_207_417_558_777_945,
                11_156_586_730_023_474_729,
                1_611_574_972_052_613_985,
                8_757_309_153_920_058_853,
                16_173_054_648_362_138_220,
                5_081_134_865_904_393_435,
            ]
        );
    }

    #[test]
    fn chacha12_known_answer_uses_high_seed_bytes() {
        let mut seed = [0_u8; 32];
        seed[31] = 0xa5;
        let mut rng = AlgorithmRng::new(RunSeed::from_bytes(seed));

        assert_eq!(
            core::array::from_fn::<_, 3, _>(|_| rng.next_u64()),
            [
                9_038_260_209_222_205_421,
                8_020_762_970_600_673_850,
                10_829_766_227_627_304_226,
            ]
        );
    }

    #[test]
    fn displayed_seeds_parse_back_exactly() {
        let seed = RunSeed::from_bytes(core::array::from_fn(|index| (index * 37 + 5) as u8));
        let text = seed.to_string();
        assert_eq!(text.parse::<RunSeed>(), Ok(seed));
        assert_eq!(text.to_uppercase().parse::<RunSeed>(), Ok(seed));
        assert_eq!(format!("  {text}\n").parse::<RunSeed>(), Ok(seed));
    }

    #[test]
    fn malformed_seed_texts_are_rejected() {
        let valid = RunSeed::from_u64(42).to_string();
        assert_eq!("2a".parse::<RunSeed>(), Err(ParseRunSeedError::Length(2)));
        assert_eq!(
            format!("{valid}0").parse::<RunSeed>(),
            Err(ParseRunSeedError::Length(65))
        );
        let mut wrong = valid.clone();
        wrong.replace_range(10..11, "g");
        assert_eq!(wrong.parse::<RunSeed>(), Err(ParseRunSeedError::NotHex(10)));
        // Positions and lengths count characters, not bytes: "é" takes two bytes.
        let accented = format!("é{}", &valid[1..]);
        assert_eq!(
            accented.parse::<RunSeed>(),
            Err(ParseRunSeedError::NotHex(0))
        );
        let short = format!("é{}", &valid[2..]);
        assert_eq!(short.parse::<RunSeed>(), Err(ParseRunSeedError::Length(63)));
    }

    #[test]
    fn seed_derivation_has_a_full_width_known_answer() {
        let root = RunSeed::from_bytes(core::array::from_fn(|index| index as u8));

        assert_eq!(
            root.derive("geometry/genransac/hypothesis/v1", &[1, u64::MAX])
                .to_string(),
            "60b96ee607b40719d0de251636a46f616b00786724a6218ff3010dc8f6bc0091"
        );
    }

    #[test]
    fn sampled_indices_are_unique_and_bounded() {
        let mut rng = AlgorithmRng::new(RunSeed::from_u64(9));
        let mut sample = [usize::MAX; 32];
        assert!(rng.sample_indices(32, 32, &mut sample));
        sample.sort_unstable();
        assert_eq!(sample, core::array::from_fn::<_, 32, _>(|index| index));
        assert!(!rng.sample_indices(3, 4, &mut sample));
        assert_eq!(rng.index(0), None);
    }

    #[test]
    fn generated_floats_respect_their_intervals() {
        let mut rng = AlgorithmRng::new(RunSeed::from_u64(11));
        for _ in 0..10_000 {
            let closed_open = rng.unit_f64();
            let open = rng.open_unit_f64();
            let normal = rng.standard_normal_f32();
            assert!((0.0..1.0).contains(&closed_open));
            assert!(open > 0.0 && open < 1.0);
            assert!(normal.is_finite());
        }
    }
}
