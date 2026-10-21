// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The OMVG format core: manifest, table files, hashes, and every rule of the specification.
//!
//! ```text
//! manifest.json ──> §3 checks ──> content hash
//!        │
//!        v
//! each listed table ──> §8 file checks ──> structural checks ──> table hash
//!        │                                 (a failing table is broken)
//!        v
//! table rules and cross-table rules over the unbroken §5 tables
//! ```

pub(crate) mod checks;
mod footer;
mod hash;
mod json;
pub(crate) mod manifest;
mod read;
mod rule;
pub(crate) mod schema;
pub(crate) mod table;
pub(crate) mod write;

#[cfg(test)]
pub(crate) mod conformance;

use std::path::Path;

pub use rule::{Rule, Violation};

use checks::{Broken, Tables};
use manifest::Manifest;

/// The version of the OMVG format specification this crate implements.
pub const FORMAT_VERSION: &str = "0.1-draft";

/// A dataset as loaded: its manifest when the manifest passed §3, the tables that loaded and
/// passed their structural checks, and every rule broken.
pub(crate) struct Loaded {
    pub(crate) manifest: Option<Manifest>,
    pub(crate) tables: Tables,
    pub(crate) violations: Vec<Violation>,
}

/// Load and check the dataset at `root`. Table files are opened only when the manifest passes.
/// A table §5 does not define gets the §8 file checks, the structural checks, and its hash.
pub(crate) fn load(root: &Path) -> Loaded {
    let manifest = match manifest::read_manifest(root) {
        Ok(manifest) => manifest,
        Err(violations) => {
            return Loaded {
                manifest: None,
                tables: Tables::new(),
                violations,
            };
        }
    };

    let mut violations = Vec::new();
    if manifest.recomputed_content_hash() != manifest.content_hash {
        violations.push(Violation::in_manifest(
            Rule::HashContent,
            "content_hash does not match the manifest's table hashes",
        ));
    }
    let mut tables = Tables::new();
    let mut broken = Broken::new();
    for entry in &manifest.tables {
        let problems = match read::load_table(root, &entry.name, entry.rows) {
            Ok(table) => {
                let problems = checks::structural_checks(&table);
                if problems.is_empty() {
                    if hash::table_hash(&table) != entry.hash {
                        violations.push(Violation::in_table(
                            Rule::HashTable,
                            &entry.name,
                            "table hash does not match the manifest",
                        ));
                    }
                    tables.insert(entry.name.clone(), table);
                }
                problems
            }
            Err(problems) => problems,
        };
        if !problems.is_empty() {
            broken.insert(entry.name.clone());
            violations.extend(problems);
        }
    }
    violations.extend(checks::validate_tables(
        &tables,
        &broken,
        &manifest.profiles,
    ));
    Loaded {
        manifest: Some(manifest),
        tables,
        violations,
    }
}

/// Check the OMVG dataset at `path` against every rule of the specification.
///
/// Returns every rule broken; an empty list means the dataset conforms. Unreadable files are
/// reported as violations, not errors, so the result is always complete for what could be read.
///
/// ```no_run
/// let violations = openmvg_io::check("scene.omvg");
/// for violation in &violations {
///     eprintln!("{violation}");
/// }
/// assert!(violations.is_empty());
/// ```
pub fn check(path: impl AsRef<Path>) -> Vec<Violation> {
    load(path.as_ref()).violations
}
