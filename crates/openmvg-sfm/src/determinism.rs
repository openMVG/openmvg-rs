// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Seed domains of the pipeline stages and the canonical JSON and SHA-256 hashing used to
//! fingerprint configurations and reports.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{Result, SfmError};
use openmvg_core::RunSeed;

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
const HASH_PREFIX: &str = "sha256:";

pub(crate) const DEFAULT_ROOT_SEED: RunSeed = RunSeed::ZERO;

pub(crate) const FEATURE_EXTRACTION_SEED_DOMAIN: &str = "pipeline/feature-extraction/v1";
pub(crate) const PAIR_MATCHING_SEED_DOMAIN: &str = "pipeline/pair-matching/v1";
pub(crate) const GEOMETRIC_VERIFICATION_SEED_DOMAIN: &str = "pipeline/geometric-verification/v1";
pub(crate) const POSE_REGISTRATION_SEED_DOMAIN: &str = "pipeline/pose-registration/v1";
pub(crate) const GLOBAL_MIXED_POSITIONING_SEED_DOMAIN: &str =
    "pipeline/global/mixed-positioning/v1";

/// Serialize a value to canonical JSON bytes.
///
/// Object keys are sorted recursively. Arrays retain their semantic order.
/// Serialization failures, including non-finite JSON numbers, are propagated.
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub(crate) fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let value = serde_json::to_value(value).map_err(|error| {
        SfmError::InvalidInput(format!("canonical JSON serialization failed: {error}"))
    })?;
    serde_json::to_vec(&canonicalize_json(value)).map_err(|error| {
        SfmError::InvalidInput(format!("canonical JSON serialization failed: {error}"))
    })
}

/// Hash canonical JSON with SHA-256.
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub fn stable_json_hash<T: Serialize>(value: &T) -> Result<String> {
    Ok(stable_hash_bytes(&canonical_json_bytes(value)?))
}

/// Hash bytes with SHA-256 and an explicit algorithm prefix.
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub fn stable_hash_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(HASH_PREFIX.len() + digest.len() * 2);
    encoded.push_str(HASH_PREFIX);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
fn canonicalize_json(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize_json).collect()),
        Value::Object(map) => {
            let mut entries: Vec<_> = map.into_iter().collect();
            entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = serde_json::Map::new();
            for (key, value) in entries {
                sorted.insert(key, canonicalize_json(value));
            }
            Value::Object(sorted)
        }
        scalar => scalar,
    }
}
