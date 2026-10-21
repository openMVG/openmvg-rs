// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The conformance vectors of §10 and the writer round trip over them.
//!
//! A valid vector must pass every rule and recompute the hashes in its manifest; an
//! invalid vector must break exactly the one rule its `expected.json` names. The writer must
//! publish every valid vector again with the same hashes.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use super::json::Json;
use super::write::{ManifestFields, write};
use super::*;
use crate::Error;

/// The conformance vectors live in the repository, not in the published crate: `None`, with a
/// note, when this source tree lacks them.
pub(crate) fn testdata() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/format/testdata");
    if root.is_dir() {
        return Some(root);
    }
    eprintln!("skipped: this source tree has no conformance vectors in docs/format/testdata");
    None
}

fn vectors(kind: &str) -> Vec<PathBuf> {
    let Some(root) = testdata() else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = fs::read_dir(root.join(kind))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    assert!(!dirs.is_empty(), "no {kind} vectors");
    dirs
}

fn expected(vector: &Path) -> Json {
    Json::parse(&fs::read(vector.join("expected.json")).unwrap()).unwrap()
}

fn name(vector: &Path) -> String {
    vector.file_name().unwrap().to_string_lossy().into_owned()
}

#[test]
fn valid_vectors_conform_and_ignore_physical_encoding() {
    let mut manifests = BTreeMap::new();
    for vector in vectors("valid") {
        let loaded = load(&vector);
        assert!(
            loaded.violations.is_empty(),
            "valid/{}: {:?}",
            name(&vector),
            loaded.violations
        );
        manifests.insert(name(&vector), loaded.manifest.unwrap());
    }
    if !manifests.is_empty() {
        assert_eq!(
            manifests["reconstruction"],
            manifests["reconstruction_plain"]
        );
    }
}

#[test]
fn invalid_vectors_break_exactly_their_rule() {
    let mut failures = Vec::new();
    for vector in vectors("invalid") {
        let rule = expected(&vector)
            .get("rule")
            .and_then(Json::as_str)
            .unwrap()
            .to_owned();
        let found: BTreeSet<&str> = check(&vector).iter().map(|v| v.rule().id()).collect();
        if found.len() != 1 || !found.contains(rule.as_str()) {
            failures.push(format!(
                "invalid/{}: expected exactly [{rule}], found {found:?}",
                name(&vector)
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn options(manifest: &Manifest) -> ManifestFields {
    ManifestFields {
        profiles: manifest.profiles.clone(),
        length_unit: manifest.length_unit,
        generator: Some("openmvg-io conformance test".to_owned()),
        extensions_used: manifest.extensions_used.clone(),
    }
}

#[test]
fn writer_republishes_every_valid_vector_with_the_same_hashes() {
    let scratch = tempfile::tempdir().unwrap();
    for (i, vector) in vectors("valid").into_iter().enumerate() {
        let loaded = load(&vector);
        let source = loaded.manifest.clone().unwrap();
        let target = scratch.path().join(format!("{i}.omvg"));
        let written = write(&target, loaded.tables, &options(&source)).unwrap();
        assert_eq!(
            written.content_hash,
            source.content_hash,
            "valid/{}",
            name(&vector)
        );
        assert_eq!(written.tables, source.tables, "valid/{}", name(&vector));
        let found = check(&target);
        assert!(
            found.is_empty(),
            "valid/{} written: {found:?}",
            name(&vector)
        );
    }
    // Nothing but the published datasets is left behind.
    let leftovers: Vec<_> = fs::read_dir(scratch.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with('.'))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn writer_sorts_rows_by_key() {
    let Some(root) = testdata() else {
        return;
    };
    let loaded = load(&root.join("valid/full"));
    let source = loaded.manifest.clone().unwrap();
    let mut tables = loaded.tables;
    for t in tables.values_mut() {
        let reversed = arrow_array::UInt32Array::from_iter_values((0..t.rows as u32).rev());
        for column in &mut t.columns {
            let array = arrow_select::take::take(column.values.array(), &reversed, None).unwrap();
            column.values = table::Values::from_array(&array, column.ltype).unwrap();
        }
    }
    let scratch = tempfile::tempdir().unwrap();
    let written = write(
        &scratch.path().join("sorted.omvg"),
        tables,
        &options(&source),
    )
    .unwrap();
    assert_eq!(written.content_hash, source.content_hash);
}

#[test]
fn writer_refuses_invalid_tables_and_existing_targets() {
    let Some(root) = testdata() else {
        return;
    };
    let loaded = load(&root.join("valid/full"));
    let source = loaded.manifest.clone().unwrap();
    let scratch = tempfile::tempdir().unwrap();

    let mut without_views = loaded.tables.clone();
    without_views.remove("views");
    let target = scratch.path().join("no_views.omvg");
    match write(&target, without_views, &options(&source)) {
        Err(Error::Invalid(violations)) => {
            assert_eq!(
                violations.iter().map(Violation::rule).collect::<Vec<_>>(),
                [Rule::ManifestProfileTables]
            );
        }
        other => panic!("expected manifest.profile_tables, found {other:?}"),
    }
    assert!(!target.exists());
    let message = write(&target, Tables::new(), &options(&source))
        .unwrap_err()
        .to_string();
    assert!(
        message.starts_with("the dataset breaks 1 rule(s), first: [manifest.profile_tables]"),
        "{message}"
    );

    fs::create_dir(&target).unwrap();
    assert!(matches!(
        write(&target, loaded.tables, &options(&source)),
        Err(Error::Exists(_))
    ));
    assert_eq!(
        fs::read_dir(&target).unwrap().count(),
        0,
        "an existing target is left untouched"
    );
}

#[test]
fn a_missing_manifest_is_one_violation() {
    let scratch = tempfile::tempdir().unwrap();
    let found = check(scratch.path());
    assert_eq!(
        found.iter().map(Violation::rule).collect::<Vec<_>>(),
        [Rule::ManifestFile]
    );
}
