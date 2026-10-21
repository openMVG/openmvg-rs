// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Column, table, and content hashes (§7).
//!
//! ```text
//! column buffer   u64 rows ‖ mode (1 when the column holds a null, else 0)
//!                 ‖ per row: [validity byte, in mode 1] canonical bytes (§7.1)
//! column hash     SHA-256(column buffer)
//! table hash      SHA-256(u64 rows ‖ for each column with a value, in bytewise name order:
//!                                    name ‖ 0x00 ‖ stored type ‖ 0x00 ‖ column hash)
//! content hash    SHA-256(SHA-256(manifest_core) ‖ for each table: name ‖ 0x00 ‖ table hash)
//! ```
//!
//! The table hash reads only the stored columns, never §5, so any reader can verify any table.
//! Column buffers stream into the hasher, so hashing needs no copy of the table.

use arrow_array::Array;
use sha2::{Digest, Sha256};

use super::table::{Table, Values};

/// A SHA-256 digest.
pub(crate) type Digest32 = [u8; 32];

/// Bytes buffered before they reach the hasher, so that small values hash in bulk.
const CHUNK: usize = 1 << 16;

/// A column buffer that streams into SHA-256.
struct Sink {
    hasher: Sha256,
    buffer: Vec<u8>,
}

impl Sink {
    fn new() -> Self {
        Self {
            hasher: Sha256::new(),
            buffer: Vec::with_capacity(CHUNK),
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
        if self.buffer.len() >= CHUNK {
            self.hasher.update(&self.buffer);
            self.buffer.clear();
        }
    }

    fn finish(mut self) -> Digest32 {
        self.hasher.update(&self.buffer);
        self.hasher.finalize().into()
    }
}

/// Push the canonical bytes of element `i` of scalar values; a null element is zero.
fn push_scalar(sink: &mut Sink, values: &Values, i: usize) {
    match values {
        Values::Int32(a) => sink.push(&if a.is_valid(i) { a.value(i) } else { 0 }.to_le_bytes()),
        Values::Int64(a) => sink.push(&if a.is_valid(i) { a.value(i) } else { 0 }.to_le_bytes()),
        Values::Float(a) => sink.push(&if a.is_valid(i) { a.value(i) } else { 0.0 }.to_le_bytes()),
        Values::Double(a) => sink.push(&if a.is_valid(i) { a.value(i) } else { 0.0 }.to_le_bytes()),
        Values::Boolean(a) => sink.push(&[u8::from(a.is_valid(i) && a.value(i))]),
        Values::Bytes(_) | Values::FixedBytes(_) | Values::List(..) => {}
    }
}

fn length_prefix(len: usize) -> [u8; 8] {
    u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes()
}

/// The column hash (§7.2) of one column. A list, fixed-size or not, is length-prefixed as it
/// is stored.
pub(crate) fn column_hash(values: &Values) -> Digest32 {
    let nullable = values.null_count() > 0;
    let mut sink = Sink::new();
    // The row count and the mode make the buffer decodable, so distinct columns never share it.
    sink.push(&length_prefix(values.len()));
    sink.push(&[u8::from(nullable)]);
    for row in 0..values.len() {
        let valid = values.is_valid(row);
        if nullable {
            sink.push(&[u8::from(valid)]);
        }
        if !valid {
            continue;
        }
        match values {
            Values::Bytes(a) => {
                let bytes = a.value(row);
                sink.push(&length_prefix(bytes.len()));
                sink.push(bytes);
            }
            Values::FixedBytes(a) => sink.push(a.value(row)),
            Values::List(list, elements) => {
                let range = Values::list_range(list, row);
                sink.push(&length_prefix(range.len()));
                for i in range {
                    push_scalar(&mut sink, elements, i);
                }
            }
            scalar => push_scalar(&mut sink, scalar, row),
        }
    }
    sink.finish()
}

/// The table hash (§7.3) of a table.
pub(crate) fn table_hash(table: &Table) -> Digest32 {
    // A column without a value hashes like an absent one.
    let mut columns: Vec<_> = table
        .columns
        .iter()
        .filter(|c| c.values.null_count() < c.values.len())
        .collect();
    columns.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    let mut outer = Sha256::new();
    // The row count binds cardinality even when no column has a value.
    outer.update(length_prefix(table.rows));
    for column in columns {
        outer.update(column.name.as_bytes());
        outer.update([0u8]);
        outer.update(column.ltype.stored().to_string().as_bytes());
        outer.update([0u8]);
        outer.update(column_hash(&column.values));
    }
    outer.finalize().into()
}

/// The content hash (§7.4) from `manifest_core` and the raw table hashes.
pub(crate) fn content_hash<'a>(
    core: &[u8],
    tables: impl IntoIterator<Item = (&'a str, Digest32)>,
) -> Digest32 {
    let mut tables: Vec<(&str, Digest32)> = tables.into_iter().collect();
    tables.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut outer = Sha256::new();
    outer.update(Sha256::digest(core));
    for (name, digest) in tables {
        outer.update(name.as_bytes());
        outer.update([0u8]);
        outer.update(digest);
    }
    outer.finalize().into()
}

/// The recorded form of a digest: `sha256:` and 64 lowercase hexadecimal digits.
pub(crate) fn hex(digest: &Digest32) -> String {
    use std::fmt::Write;
    let mut text = String::with_capacity(71);
    text.push_str("sha256:");
    for byte in digest {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array};

    use super::*;
    use crate::format::schema::LType;

    fn values(array: ArrayRef, lt: LType) -> Values {
        Values::from_array(&array, lt).unwrap()
    }

    /// The row count and mode that open every column buffer.
    fn header(rows: u64, mode: u8) -> Vec<u8> {
        let mut bytes = rows.to_le_bytes().to_vec();
        bytes.push(mode);
        bytes
    }

    #[test]
    fn a_null_and_its_validity_bytes_never_read_as_values() {
        // Without the mode byte both columns had the buffer 00 01000000 00 01000000 00 01000000.
        let with_null = values(
            Arc::new(Int32Array::from(vec![None, Some(0), Some(0), Some(0)])),
            LType::INT32,
        );
        let without = values(
            Arc::new(Int32Array::from(vec![256, 65536, 16_777_216, 0])),
            LType::INT32,
        );
        let mut first = header(4, 1);
        first.extend_from_slice(&[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
        let mut second = header(4, 0);
        second.extend_from_slice(&[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
        assert_eq!(
            column_hash(&with_null),
            <[u8; 32]>::from(Sha256::digest(&first))
        );
        assert_eq!(
            column_hash(&without),
            <[u8; 32]>::from(Sha256::digest(&second))
        );
        assert_ne!(column_hash(&with_null), column_hash(&without));
    }

    #[test]
    fn tables_without_a_value_differ_by_row_count() {
        let table = |rows: usize| Table {
            name: "OMVG_example".to_owned(),
            spec: crate::format::schema::table_spec("OMVG_example"),
            rows,
            columns: vec![crate::format::table::Column {
                name: "score".to_owned(),
                ltype: LType::DOUBLE,
                spec: None,
                values: Values::nulls(LType::DOUBLE, rows).unwrap(),
            }],
        };
        assert_eq!(
            table_hash(&table(1)),
            <[u8; 32]>::from(Sha256::digest(1u64.to_le_bytes()))
        );
        assert_ne!(table_hash(&table(1)), table_hash(&table(9)));
    }
}
