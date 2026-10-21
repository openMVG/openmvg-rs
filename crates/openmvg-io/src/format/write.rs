// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validating, hashing, and atomically publishing a dataset (§3, §7, §8, §10).
//!
//! ```text
//! manifest fields ──> §3 checks
//!        │
//! canonical tables ──> sort by key ──> structural, table, and cross-table rules
//!        │
//!        v
//! table hashes ──> content hash ──> manifest
//!        │
//!        v
//! .<name>.staging-<pid>-<n>/  <table>.parquet ... manifest.json (last), each synced
//!        │  rename
//!        v
//! <name>/
//! ```
//!
//! A dataset is published to a new path in one rename, so a reader never sees a partial
//! dataset. An error before the rename removes the staging directory and leaves the target
//! absent. Once the rename succeeds the dataset is complete; if its parent directory then cannot
//! be flushed, the error says so ([`Error::Unsynced`]), since a crash may undo the rename.
//! The writer checks that the target is absent before it stages; a directory created at the
//! target by another process in between is not guarded against.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{Array, ArrayRef, ListArray, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use parquet::arrow::{ArrowWriter, arrow_writer::ArrowWriterOptions};
use parquet::basic::{Compression, Encoding};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use parquet::schema::types::ColumnPath;
use serde::Serialize;

use crate::{Error, LengthUnit};

use super::FORMAT_VERSION;
use super::checks::{self, Broken, Tables};
use super::hash;
use super::json::Json;
use super::manifest::{self, Manifest, TableEntry};
use super::rule::{Rule, Violation};
use super::schema::{ID_COLUMNS, LType, Scalar};
use super::table::{Column, Table, Values, scalar_type};

/// Rows per row group (§8).
const ROW_GROUP_ROWS: usize = 131_072;

/// The manifest fields a writer chooses.
#[derive(Debug, Clone)]
pub(crate) struct ManifestFields {
    pub(crate) profiles: Vec<String>,
    pub(crate) length_unit: LengthUnit,
    pub(crate) generator: Option<String>,
    /// The extensions whose tables the dataset carries.
    pub(crate) extensions_used: Vec<String>,
}

/// `manifest.json` in the member order of §3.
#[derive(Serialize)]
struct ManifestJson<'a> {
    omvg: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    generator: Option<&'a str>,
    profiles: &'a [String],
    length_unit: &'a str,
    extensions_used: &'a [String],
    extensions_required: &'a [String],
    tables: BTreeMap<&'a str, TableJson>,
    content_hash: String,
}

#[derive(Serialize)]
struct TableJson {
    rows: u64,
    hash: String,
}

/// The text of `manifest.json` for `manifest`.
fn manifest_text(manifest: &Manifest, generator: Option<&str>) -> serde_json::Result<String> {
    let json = ManifestJson {
        omvg: FORMAT_VERSION,
        generator,
        profiles: &manifest.profiles,
        length_unit: manifest.length_unit.name(),
        extensions_used: &manifest.extensions_used,
        extensions_required: &manifest.extensions_required,
        tables: manifest
            .tables
            .iter()
            .map(|t| {
                let hash = hash::hex(&t.hash);
                (t.name.as_str(), TableJson { rows: t.rows, hash })
            })
            .collect(),
        content_hash: hash::hex(&manifest.content_hash),
    };
    Ok(serde_json::to_string_pretty(&json)? + "\n")
}

/// Check the manifest fields a writer chooses under §3; return them as a manifest whose
/// hashes are still zero.
fn checked_manifest(tables: &Tables, options: &ManifestFields) -> Result<Manifest, Vec<Violation>> {
    let distinct = |values: &[String]| {
        let mut values = values.to_vec();
        values.sort_unstable();
        values.dedup();
        values
    };
    let draft = Manifest {
        profiles: distinct(&options.profiles),
        length_unit: options.length_unit,
        extensions_used: distinct(&options.extensions_used),
        extensions_required: Vec::new(),
        tables: tables
            .iter()
            .map(|(name, table)| TableEntry {
                name: name.clone(),
                rows: u64::try_from(table.rows).unwrap_or(u64::MAX),
                hash: [0; 32],
            })
            .collect(),
        content_hash: [0; 32],
    };
    let invalid = |error: String| vec![Violation::in_manifest(Rule::ManifestFile, error)];
    let text = manifest_text(&draft, options.generator.as_deref())
        .map_err(|error| invalid(error.to_string()))?;
    let json = Json::parse(text.as_bytes()).map_err(|error| invalid(error.to_string()))?;
    manifest::check_manifest(&json)
}

/// Put the rows of `table` in ascending key order; equal keys keep their order.
fn sort(table: &mut Table) -> Result<(), ArrowError> {
    let order = checks::sort_order(&checks::key_arrays(table, table.spec.key));
    if order.iter().enumerate().all(|(i, &row)| i == row) {
        return Ok(());
    }
    let indices = UInt64Array::from_iter_values(
        order
            .iter()
            .map(|&row| u64::try_from(row).unwrap_or(u64::MAX)),
    );
    for column in &mut table.columns {
        let array = arrow_select::take::take(column.values.array(), &indices, None)?;
        column.values = Values::from_array(&array, column.ltype).ok_or_else(|| {
            ArrowError::ComputeError(format!("column {:?} changed type when sorted", column.name))
        })?;
    }
    Ok(())
}

/// The leaf path of a column: the column itself, or the element of its three-level `LIST`.
fn leaf_path(column: &Column) -> ColumnPath {
    match column.ltype {
        LType::FixedSizeList(..) | LType::List(_) => ColumnPath::new(vec![
            column.name.clone(),
            "list".to_owned(),
            "element".to_owned(),
        ]),
        _ => ColumnPath::new(vec![column.name.clone()]),
    }
}

/// The §8 encoding of a leaf, or `None` for dictionary encoding.
fn leaf_encoding(lt: LType) -> Option<Encoding> {
    let scalar = match lt {
        LType::String => return None,
        LType::Binary | LType::FixedSizeBinary(_) => return Some(Encoding::PLAIN),
        LType::Scalar(scalar) | LType::FixedSizeList(scalar, _) | LType::List(scalar) => scalar,
    };
    Some(match scalar {
        Scalar::Int32 | Scalar::Int64 => Encoding::DELTA_BINARY_PACKED,
        Scalar::Float | Scalar::Double => Encoding::BYTE_STREAM_SPLIT,
        Scalar::Boolean => Encoding::RLE,
    })
}

/// The Arrow field and array a column is written as: `STRING` as UTF-8, lists with a required
/// `element`, and a required §5 column as a required field.
fn written(column: &Column) -> Result<(Field, ArrayRef), ArrowError> {
    let nullable = !column.required();
    let array: ArrayRef = match (&column.values, column.ltype) {
        (Values::Bytes(bytes), LType::String) => {
            Arc::new(StringArray::try_from_binary(bytes.clone())?)
        }
        (Values::List(list, _), LType::FixedSizeList(element, _) | LType::List(element)) => {
            let field = Arc::new(Field::new("element", scalar_type(element), false));
            Arc::new(ListArray::try_new(
                field,
                list.offsets().clone(),
                list.values().clone(),
                list.nulls().cloned(),
            )?)
        }
        (values, _) => values.array().slice(0, values.len()),
    };
    let data_type: DataType = array.data_type().clone();
    Ok((Field::new(column.name.clone(), data_type, nullable), array))
}

/// Write one table file: Snappy, the §8 encodings, statistics on ID columns, and the table name
/// in the key-value metadata.
fn write_table(table: &Table, path: &Path) -> Result<(), Error> {
    // An all-null nullable §5 column hashes like an absent one (§7.3), so it is left out.
    let columns: Vec<&Column> = table
        .columns
        .iter()
        .filter(|c| !(c.spec.is_some_and(|s| s.nullable) && c.values.null_count() == table.rows))
        .collect();
    let mut properties = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(false)
        .set_statistics_enabled(EnabledStatistics::None)
        .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
        .set_key_value_metadata(Some(vec![KeyValue::new(
            "omvg.table".to_owned(),
            table.name().to_owned(),
        )]));
    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays = Vec::with_capacity(columns.len());
    for column in columns {
        let path = leaf_path(column);
        properties = match leaf_encoding(column.ltype) {
            Some(encoding) => properties.set_column_encoding(path.clone(), encoding),
            None => properties.set_column_dictionary_enabled(path.clone(), true),
        };
        if ID_COLUMNS.contains(&column.name.as_str()) {
            properties = properties.set_column_statistics_enabled(path, EnabledStatistics::Chunk);
        }
        let (field, array) = written(column)?;
        fields.push(field);
        arrays.push(array);
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new_with_options(
        schema.clone(),
        arrays,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(table.rows)),
    )?;
    let file = File::create_new(path)?;
    let options = ArrowWriterOptions::new()
        .with_properties(properties.build())
        .with_skip_arrow_metadata(true)
        .with_schema_root("schema".to_owned());
    let mut writer = ArrowWriter::try_new_with_options(file, schema, options)?;
    writer.write(&batch)?;
    writer.into_inner()?.sync_all()?;
    Ok(())
}

/// Flush a directory entry to disk, where the platform allows opening a directory.
fn sync_dir(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// Write every file into `staging`, the manifest last, and publish it as `target`.
fn publish(staging: &Path, target: &Path, tables: &Tables, manifest: &str) -> Result<(), Error> {
    for (name, table) in tables {
        write_table(table, &staging.join(format!("{name}.parquet")))?;
    }
    {
        // Closed before the rename: Windows refuses to rename a directory with an open file.
        let mut file = File::create_new(staging.join("manifest.json"))?;
        file.write_all(manifest.as_bytes())?;
        file.sync_all()?;
    }
    sync_dir(staging)?;
    fs::rename(staging, target)?;
    sync_dir(parent_dir(target)).map_err(|error| Error::Unsynced {
        path: target.to_owned(),
        source: error,
    })
}

/// Validate, sort, and hash `tables`, then publish them as a new dataset at `path`; return the
/// manifest written. A table §5 does not define is written as given, under its extension.
pub(crate) fn write(
    path: &Path,
    mut tables: Tables,
    options: &ManifestFields,
) -> Result<Manifest, Error> {
    let mut manifest = checked_manifest(&tables, options).map_err(Error::Invalid)?;

    let mut problems = Vec::new();
    let mut broken = Broken::new();
    for (name, table) in &mut tables {
        sort(table)?;
        let found = checks::structural_checks(table);
        if !found.is_empty() {
            broken.insert(name.clone());
        }
        problems.extend(found);
    }
    problems.extend(checks::validate_tables(
        &tables,
        &broken,
        &manifest.profiles,
    ));
    if !problems.is_empty() {
        return Err(Error::Invalid(problems));
    }

    for entry in &mut manifest.tables {
        entry.hash = tables.get(&entry.name).map_or([0; 32], hash::table_hash);
    }
    manifest.content_hash = manifest.recomputed_content_hash();
    let text =
        manifest_text(&manifest, options.generator.as_deref()).map_err(|e| Error::Io(e.into()))?;

    if fs::symlink_metadata(path).is_ok() {
        return Err(Error::Exists(path.to_owned()));
    }
    let staging = staging_dir(path)?;
    match publish(&staging, path, &tables, &text) {
        Ok(()) => Ok(manifest),
        // The staging directory is the published dataset now.
        Err(error @ Error::Unsynced { .. }) => Err(error),
        Err(error) => {
            // Only the directory this call created is removed.
            let _ = fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

/// The directory holding `path`; a bare file name lives in the current directory.
fn parent_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

/// Create a staging directory next to `target`, unique to this process and call.
fn staging_dir(target: &Path) -> std::io::Result<PathBuf> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = target
        .file_name()
        .ok_or_else(|| std::io::Error::other("the dataset path has no file name"))?;
    let staging = parent_dir(target).join(format!(
        ".{}.staging-{}-{}",
        name.to_string_lossy(),
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&staging)?;
    Ok(staging)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_file_name_lives_in_the_current_directory() {
        assert_eq!(parent_dir(Path::new("scene.omvg")), Path::new("."));
        assert_eq!(parent_dir(Path::new("out/scene.omvg")), Path::new("out"));
        assert_eq!(parent_dir(Path::new("/scene.omvg")), Path::new("/"));
    }
}
