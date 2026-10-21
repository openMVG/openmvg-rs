// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `omvg validate` over the conformance vectors of §10: exit 0 and `conforms` on every valid
//! vector, exit 1 with the expected rule on every invalid one, and exit 2 on a usage error.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn vectors(kind: &str) -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/format/testdata");
    let mut dirs: Vec<PathBuf> = fs::read_dir(root.join(kind))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    assert!(!dirs.is_empty(), "no {kind} vectors");
    dirs
}

fn omvg(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_omvg"))
        .args(args)
        .output()
        .unwrap()
}

fn validate(vector: &Path) -> (Option<i32>, Vec<String>) {
    let output = omvg(&["validate", vector.to_str().unwrap()]);
    let stdout = String::from_utf8(output.stdout).unwrap();
    (
        output.status.code(),
        stdout.lines().map(str::to_owned).collect(),
    )
}

#[test]
fn every_valid_vector_conforms() {
    for vector in vectors("valid") {
        let (code, lines) = validate(&vector);
        assert_eq!(code, Some(0), "{}: {lines:?}", vector.display());
        assert_eq!(lines, ["conforms"], "{}", vector.display());
    }
}

#[test]
fn every_invalid_vector_reports_its_rule() {
    for vector in vectors("invalid") {
        let expected: serde_json::Value =
            serde_json::from_slice(&fs::read(vector.join("expected.json")).unwrap()).unwrap();
        let rule = format!("[{}]", expected["rule"].as_str().unwrap());
        let (code, lines) = validate(&vector);
        assert_eq!(code, Some(1), "{}: {lines:?}", vector.display());
        let (verdict, violations) = lines.split_last().unwrap();
        assert_eq!(*verdict, format!("{} violation(s)", violations.len()));
        assert!(
            violations.iter().all(|line| line.starts_with(&rule)),
            "{}: expected {rule}, found {violations:?}",
            vector.display()
        );
    }
}

#[test]
fn usage_errors_and_native_paths_have_controlled_exit_codes() {
    for args in [
        &[][..],
        &["validate"],
        &["check", "scene.omvg"],
        &["validate", "a", "b"],
    ] {
        let output = omvg(args);
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).starts_with("usage: omvg validate"));
    }
    assert_eq!(omvg(&["--help"]).status.code(), Some(0));

    #[cfg(unix)]
    {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let scratch = tempfile::tempdir().unwrap();
        let path = scratch
            .path()
            .join(OsString::from_vec(b"missing-\xff".to_vec()));
        let output = Command::new(env!("CARGO_BIN_EXE_omvg"))
            .arg("validate")
            .arg(path)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&output.stdout).starts_with("[manifest.file]"));
        assert!(output.stderr.is_empty());
    }
}
