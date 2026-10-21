// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The manifest (§3): its checks, its typed form, and `manifest_core` (§7.4).

use std::path::Path;

use super::FORMAT_VERSION;
use super::hash::{self, Digest32};
use super::json::Json;
use super::rule::{Rule, Violation};
use super::schema::{self, PROFILES};
use crate::LengthUnit;

/// Members every manifest carries, in the order the specification lists them.
const REQUIRED_MEMBERS: &[&str] = &[
    "omvg",
    "profiles",
    "length_unit",
    "extensions_used",
    "extensions_required",
    "tables",
    "content_hash",
];

/// One `tables` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TableEntry {
    pub(crate) name: String,
    pub(crate) rows: u64,
    pub(crate) hash: Digest32,
}

/// A manifest that passed every check of §3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Manifest {
    pub(crate) profiles: Vec<String>,
    pub(crate) length_unit: LengthUnit,
    pub(crate) extensions_used: Vec<String>,
    pub(crate) extensions_required: Vec<String>,
    /// Entries in manifest order.
    pub(crate) tables: Vec<TableEntry>,
    pub(crate) content_hash: Digest32,
}

impl Manifest {
    /// The `manifest_core` byte string of §7.4. The §3 character rules leave nothing to escape.
    pub(crate) fn core(&self) -> Vec<u8> {
        fn strings(values: &[String]) -> String {
            values
                .iter()
                .map(|v| format!("\"{v}\""))
                .collect::<Vec<_>>()
                .join(",")
        }
        format!(
            "{{\"omvg\":\"{FORMAT_VERSION}\",\"profiles\":[{}],\"length_unit\":\"{}\",\
             \"extensions_used\":[{}],\"extensions_required\":[{}]}}",
            strings(&self.profiles),
            self.length_unit.name(),
            strings(&self.extensions_used),
            strings(&self.extensions_required),
        )
        .into_bytes()
    }

    /// The content hash that the manifest's own table hashes give.
    pub(crate) fn recomputed_content_hash(&self) -> Digest32 {
        hash::content_hash(
            &self.core(),
            self.tables.iter().map(|t| (t.name.as_str(), t.hash)),
        )
    }
}

/// A sorted array of distinct strings that each satisfy `is_valid`.
fn sorted_names(value: &Json, is_valid: fn(&str) -> bool) -> Option<Vec<String>> {
    let Json::Array(values) = value else {
        return None;
    };
    let names: Vec<String> = values
        .iter()
        .map(|v| v.as_str().filter(|s| is_valid(s)).map(str::to_owned))
        .collect::<Option<_>>()?;
    names.windows(2).all(|w| w[0] < w[1]).then_some(names)
}

fn table_entry(name: &str, entry: &Json) -> Option<TableEntry> {
    if !schema::is_table_name(name) {
        return None;
    }
    let rows = match entry.get("rows")? {
        Json::Integer(rows) => u64::try_from(*rows).ok()?,
        _ => return None,
    };
    let hash = entry.get("hash")?.as_str().and_then(parse_hash)?;
    Some(TableEntry {
        name: name.to_owned(),
        rows,
        hash,
    })
}

/// The digest of a recorded hash: `sha256:` followed by 64 lowercase hexadecimal digits.
fn parse_hash(s: &str) -> Option<Digest32> {
    let hex = s.strip_prefix("sha256:")?.as_bytes();
    let lower = |b: &u8| b.is_ascii_digit() || (b'a'..=b'f').contains(b);
    if hex.len() != 64 || !hex.iter().all(lower) {
        return None;
    }
    let mut digest = [0u8; 32];
    for (byte, pair) in digest.iter_mut().zip(hex.as_chunks::<2>().0) {
        let text = std::str::from_utf8(pair).ok()?;
        *byte = u8::from_str_radix(text, 16).ok()?;
    }
    Some(digest)
}

/// Read `manifest.json` under `root` and apply every check of §3.
pub(crate) fn read_manifest(root: &Path) -> Result<Manifest, Vec<Violation>> {
    let file = |message: String| vec![Violation::in_manifest(Rule::ManifestFile, message)];
    let text = std::fs::read(root.join("manifest.json"))
        .map_err(|error| file(format!("cannot read manifest.json: {error}")))?;
    let json =
        Json::parse(&text).map_err(|error| file(format!("cannot parse manifest.json: {error}")))?;
    check_manifest(&json)
}

/// Apply every check of §3 in the reference order; return the typed manifest when it passes.
pub(crate) fn check_manifest(m: &Json) -> Result<Manifest, Vec<Violation>> {
    let mut out = Vec::new();
    let bad = |out: &mut Vec<Violation>, rule: Rule, message: String| {
        out.push(Violation::in_manifest(rule, message))
    };

    if !matches!(m, Json::Object(_)) {
        bad(
            &mut out,
            Rule::ManifestFile,
            "manifest.json is not a JSON object".into(),
        );
        return Err(out);
    }
    for member in REQUIRED_MEMBERS {
        if m.get(member).is_none() {
            bad(
                &mut out,
                Rule::ManifestMember,
                format!("missing required member {member:?}"),
            );
        }
    }
    let (
        Some(omvg),
        Some(profiles),
        Some(length_unit),
        Some(used),
        Some(required),
        Some(tables),
        Some(content_hash),
    ) = (
        m.get("omvg"),
        m.get("profiles"),
        m.get("length_unit"),
        m.get("extensions_used"),
        m.get("extensions_required"),
        m.get("tables"),
        m.get("content_hash"),
    )
    else {
        return Err(out);
    };

    if omvg.as_str() != Some(FORMAT_VERSION) {
        bad(
            &mut out,
            Rule::ManifestVersion,
            format!("omvg must be {FORMAT_VERSION:?}, found {omvg:?}"),
        );
    }
    let profiles = sorted_names(profiles, schema::is_name).filter(|p| !p.is_empty());
    match &profiles {
        None => bad(
            &mut out,
            Rule::ManifestProfiles,
            "profiles must be a non-empty sorted array of distinct names".into(),
        ),
        Some(names) if names.iter().any(|p| schema::profile_tables(p).is_none()) => {
            let known: Vec<&str> = PROFILES.iter().map(|(name, _)| *name).collect();
            bad(
                &mut out,
                Rule::ManifestProfiles,
                format!("unknown profile in {names:?}; known: {known:?}"),
            );
        }
        Some(_) => {}
    }
    let length_unit = length_unit.as_str().and_then(LengthUnit::from_name);
    if length_unit.is_none() {
        bad(
            &mut out,
            Rule::ManifestLengthUnit,
            "length_unit must be one of [\"unknown\", \"meter\"]".to_owned(),
        );
    }
    let used = sorted_names(used, schema::is_extension_name);
    let required = sorted_names(required, schema::is_extension_name);
    let extensions = match (used, required) {
        (Some(used), Some(required)) if required.iter().all(|r| used.contains(r)) => {
            Some((used, required))
        }
        _ => None,
    };
    match &extensions {
        None => bad(
            &mut out,
            Rule::ManifestExtensions,
            "extensions must be sorted arrays of distinct OMVG_ names, required a subset of used"
                .into(),
        ),
        Some((_, required)) if !required.is_empty() => bad(
            &mut out,
            Rule::ManifestRequiredExtension,
            format!("this reader implements no extension and cannot read {required:?}"),
        ),
        _ => {}
    }
    let entries = match tables {
        Json::Object(members) => members
            .iter()
            .map(|(name, entry)| table_entry(name, entry))
            .collect::<Option<Vec<_>>>(),
        _ => None,
    };
    if entries.is_none() {
        bad(
            &mut out,
            Rule::ManifestTables,
            "tables must map table names to {rows: integer >= 0, hash: sha256:<hex>}".into(),
        );
    }
    let content_hash = content_hash.as_str().and_then(parse_hash);
    if content_hash.is_none() {
        bad(
            &mut out,
            Rule::ManifestContentHash,
            "content_hash must be sha256: followed by 64 lowercase hex digits".into(),
        );
    }
    if !out.is_empty() {
        return Err(out);
    }
    let (
        Some(profiles),
        Some(length_unit),
        Some((extensions_used, extensions_required)),
        Some(tables),
        Some(content_hash),
    ) = (profiles, length_unit, extensions, entries, content_hash)
    else {
        return Err(out);
    };
    let mut missing: Vec<&str> = profiles
        .iter()
        .filter_map(|p| schema::profile_tables(p))
        .flatten()
        .copied()
        .filter(|t| !tables.iter().any(|entry| entry.name == *t))
        .collect();
    missing.sort_unstable();
    missing.dedup();
    if !missing.is_empty() {
        bad(
            &mut out,
            Rule::ManifestProfileTables,
            format!("tables required by {profiles:?} are not listed: {missing:?}"),
        );
        return Err(out);
    }
    // This reader implements no extension, so it can tell that a table is unowned only when
    // no extension is used.
    let unowned: Vec<&str> = tables
        .iter()
        .map(|entry| entry.name.as_str())
        .filter(|name| extensions_used.is_empty() && schema::table_spec(name).name.is_empty())
        .collect();
    if !unowned.is_empty() {
        bad(
            &mut out,
            Rule::ManifestTables,
            format!("§5 does not define {unowned:?}, and no extension is used to define them"),
        );
        return Err(out);
    }
    Ok(Manifest {
        profiles,
        length_unit,
        extensions_used,
        extensions_required,
        tables,
        content_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules(text: &str) -> Vec<Rule> {
        match check_manifest(&Json::parse(text.as_bytes()).unwrap()) {
            Ok(_) => Vec::new(),
            Err(violations) => violations.iter().map(Violation::rule).collect(),
        }
    }

    fn manifest(edit: impl FnOnce(&mut serde_json::Value)) -> String {
        let mut value = serde_json::json!({
            "omvg": FORMAT_VERSION,
            "profiles": ["dataset"],
            "length_unit": "unknown",
            "extensions_used": [],
            "extensions_required": [],
            "tables": {
                "cameras": {"rows": 1, "hash": format!("sha256:{}", "0".repeat(64))},
                "keypoints": {"rows": 0, "hash": format!("sha256:{}", "1".repeat(64))},
                "views": {"rows": 2, "hash": format!("sha256:{}", "a".repeat(64))},
            },
            "content_hash": format!("sha256:{}", "f".repeat(64)),
        });
        edit(&mut value);
        value.to_string()
    }

    #[test]
    fn manifest_checks_stop_in_order() {
        let cases = [
            ("[]".to_owned(), Rule::ManifestFile),
            (
                manifest(|m| {
                    m.as_object_mut().unwrap().remove("tables");
                    m["omvg"] = "0.0".into();
                }),
                Rule::ManifestMember,
            ),
            (
                manifest(|m| {
                    m["tables"].as_object_mut().unwrap().remove("keypoints");
                    m["length_unit"] = "inch".into();
                }),
                Rule::ManifestLengthUnit,
            ),
            (
                manifest(|m| m["tables"]["views"]["hash"] = "sha256:00".into()),
                Rule::ManifestTables,
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(rules(&text), [expected], "{text}");
        }
    }
}
