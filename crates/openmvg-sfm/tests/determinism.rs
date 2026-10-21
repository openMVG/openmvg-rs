// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_sfm::RunSeed;
use openmvg_sfm::testing::determinism::{stable_hash_bytes, stable_json_hash};
use serde::{Serialize, Serializer};
use std::collections::HashMap;

#[derive(Serialize)]
struct DeterminismFixture {
    ids: Vec<usize>,
    label: &'static str,
}

struct SerializationFailure;

impl Serialize for SerializationFailure {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Err(serde::ser::Error::custom("intentional failure"))
    }
}

#[test]
fn determinism_json_hash_is_stable_for_sorted_dtos() {
    let fixture = DeterminismFixture {
        ids: vec![1, 2, 3],
        label: "fixture",
    };

    assert_eq!(
        stable_json_hash(&fixture).unwrap(),
        stable_json_hash(&fixture).unwrap()
    );
    assert_ne!(
        stable_json_hash(&fixture).unwrap(),
        stable_hash_bytes(b"fixture")
    );
}

#[test]
fn determinism_seed_derivation_uses_label_and_ids() {
    let root = RunSeed::from_u64(7);
    let seed_a = root.derive("features", &[1, 2]);
    let seed_b = root.derive("features", &[1, 2]);
    let seed_c = root.derive("matches", &[1, 2]);
    let seed_d = root.derive("features", &[2, 1]);

    assert_eq!(seed_a, seed_b);
    assert_ne!(seed_a, seed_c);
    assert_ne!(seed_a, seed_d);
}

#[test]
fn canonical_json_hash_is_independent_of_map_insertion_order() {
    let mut first = HashMap::new();
    first.insert("alpha", 1);
    first.insert("beta", 2);
    let mut second = HashMap::new();
    second.insert("beta", 2);
    second.insert("alpha", 1);

    assert_eq!(
        stable_json_hash(&first).unwrap(),
        stable_json_hash(&second).unwrap()
    );
}

#[test]
fn hashes_and_seeds_have_stable_golden_vectors() {
    assert_eq!(
        stable_hash_bytes(b"abc"),
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        RunSeed::from_u64(7).derive("features", &[1, 2]).to_string(),
        "fb3bba9fd7b2c83c20f3a6e5458a18c6f686ff3ffed6c9d10556c84b4ee31588"
    );
}

#[test]
fn json_serialization_failure_is_propagated() {
    assert!(stable_json_hash(&SerializationFailure).is_err());
}
