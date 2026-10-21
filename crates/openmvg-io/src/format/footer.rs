// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reading a Parquet footer and bounding its counts before the Parquet decoder sees it.
//!
//! ```text
//! last 8 bytes ──> footer length ≤ file length − 12, magic PAR1
//!   │
//!   v
//! footer bytes ──> walk the Thrift compact encoding ──> every list count ≤ bytes left
//!                                                    └> every num_children ≤ schema length
//! ```
//!
//! The decoder reserves memory from these counts before it reads the elements, so a few
//! bytes could claim billions of row groups. Every element takes at least one byte, so a count
//! larger than the bytes left is malformed, and rejecting it bounds the reservation by the
//! file size.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// How deeply structs and lists may nest; Parquet metadata nests a few levels.
const MAX_DEPTH: usize = 64;

// Thrift compact type codes.
const BOOL_TRUE: u8 = 1;
const BOOL_FALSE: u8 = 2;
const I8: u8 = 3;
const I16: u8 = 4;
const I32: u8 = 5;
const I64: u8 = 6;
const DOUBLE: u8 = 7;
const BINARY: u8 = 8;
const LIST: u8 = 9;
const SET: u8 = 10;
const MAP: u8 = 11;
const STRUCT: u8 = 12;
const UUID: u8 = 13;

/// A cursor over Thrift compact bytes.
struct Compact<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Compact<'_> {
    fn left(&self) -> u64 {
        u64::try_from(self.bytes.len() - self.at).unwrap_or(u64::MAX)
    }

    fn byte(&mut self) -> Result<u8, String> {
        let byte = *self
            .bytes
            .get(self.at)
            .ok_or("the footer ends inside a value")?;
        self.at += 1;
        Ok(byte)
    }

    fn skip(&mut self, n: u64) -> Result<(), String> {
        if n > self.left() {
            return Err("the footer ends inside a value".to_owned());
        }
        self.at += usize::try_from(n).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("the footer holds a varint longer than 10 bytes".to_owned())
    }

    fn zigzag(&mut self) -> Result<i64, String> {
        let v = self.varint()?;
        Ok(((v >> 1) as i64) ^ -((v & 1) as i64))
    }

    /// Reject a count of elements that the bytes left cannot hold.
    fn bound(&self, count: u64) -> Result<(), String> {
        if count > self.left() {
            return Err(format!(
                "the footer claims {count} elements in {} bytes",
                self.left()
            ));
        }
        Ok(())
    }

    /// A list or set header: the element type and the bounded count.
    fn list(&mut self) -> Result<(u8, u64), String> {
        let header = self.byte()?;
        let count = match header >> 4 {
            15 => self.varint()?,
            short => u64::from(short),
        };
        self.bound(count)?;
        Ok((header & 0x0f, count))
    }

    /// Skip one value of type `ty`; a boolean field carries its value in the field header.
    fn value(&mut self, ty: u8, depth: usize, in_field: bool) -> Result<(), String> {
        if depth > MAX_DEPTH {
            return Err(format!("the footer nests deeper than {MAX_DEPTH} levels"));
        }
        match ty {
            BOOL_TRUE | BOOL_FALSE if in_field => Ok(()),
            BOOL_TRUE | BOOL_FALSE | I8 => self.skip(1),
            I16 | I32 | I64 => self.varint().map(drop),
            DOUBLE => self.skip(8),
            UUID => self.skip(16),
            BINARY => {
                let len = self.varint()?;
                self.skip(len)
            }
            LIST | SET => {
                let (element, count) = self.list()?;
                for _ in 0..count {
                    self.value(element, depth + 1, false)?;
                }
                Ok(())
            }
            MAP => {
                let count = self.varint()?;
                if count == 0 {
                    return Ok(());
                }
                let types = self.byte()?;
                self.bound(count.saturating_mul(2))?;
                for _ in 0..count {
                    self.value(types >> 4, depth + 1, false)?;
                    self.value(types & 0x0f, depth + 1, false)?;
                }
                Ok(())
            }
            STRUCT => self.fields(depth + 1, |_, _, _| Ok(false)),
            _ => Err(format!("the footer holds the unknown Thrift type {ty}")),
        }
    }

    /// Walk the fields of a struct. `visit` sees each field id and type and returns whether it
    /// read the value; the walker skips the values it did not.
    fn fields(
        &mut self,
        depth: usize,
        mut visit: impl FnMut(&mut Self, i16, u8) -> Result<bool, String>,
    ) -> Result<(), String> {
        if depth > MAX_DEPTH {
            return Err(format!("the footer nests deeper than {MAX_DEPTH} levels"));
        }
        let mut id = 0i16;
        loop {
            let header = self.byte()?;
            if header == 0 {
                return Ok(());
            }
            let ty = header & 0x0f;
            id = match header >> 4 {
                0 => i16::try_from(self.zigzag()?).map_err(|e| e.to_string())?,
                delta => id.wrapping_add(i16::from(delta)),
            };
            if !visit(self, id, ty)? {
                self.value(ty, depth, true)?;
            }
        }
    }
}

/// Bound every count of a `FileMetaData` footer: each list by the bytes left, and each schema
/// element's `num_children` (field 5 of field 2) by the number of schema elements.
fn check(footer: &[u8]) -> Result<(), String> {
    let mut walker = Compact {
        bytes: footer,
        at: 0,
    };
    walker.fields(0, |walker, id, ty| {
        if (id, ty) != (2, LIST) {
            return Ok(false);
        }
        let (element, elements) = walker.list()?;
        for _ in 0..elements {
            if element != STRUCT {
                walker.value(element, 1, false)?;
                continue;
            }
            walker.fields(2, |walker, id, ty| {
                if (id, ty) != (5, I32) {
                    return Ok(false);
                }
                let children = walker.zigzag()?;
                if !u64::try_from(children).is_ok_and(|c| c <= elements) {
                    return Err(format!(
                        "a schema element claims {children} children of {elements} elements"
                    ));
                }
                Ok(true)
            })?;
        }
        Ok(true)
    })
}

/// The footer of a Parquet file, after the length, magic, and count checks.
pub(crate) fn read(mut file: &File) -> Result<Vec<u8>, String> {
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    if len < 12 {
        return Err(format!("{len} bytes are too few for a Parquet file"));
    }
    let mut tail = [0u8; 8];
    file.seek(SeekFrom::End(-8)).map_err(|e| e.to_string())?;
    file.read_exact(&mut tail).map_err(|e| e.to_string())?;
    let [l0, l1, l2, l3, m0, m1, m2, m3] = tail;
    if [m0, m1, m2, m3] != *b"PAR1" {
        return Err("the file does not end with the Parquet magic PAR1".to_owned());
    }
    let footer_len = u32::from_le_bytes([l0, l1, l2, l3]);
    if u64::from(footer_len) > len - 12 {
        return Err(format!(
            "the footer claims {footer_len} bytes of a {len}-byte file"
        ));
    }
    let mut footer = vec![0u8; usize::try_from(footer_len).map_err(|e| e.to_string())?];
    file.seek(SeekFrom::End(-8 - i64::from(footer_len)))
        .map_err(|e| e.to_string())?;
    file.read_exact(&mut footer).map_err(|e| e.to_string())?;
    check(&footer)?;
    Ok(footer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_nesting_obeys_the_footer_depth_limit() {
        // An unknown field holding nested singleton lists ends in one byte, not a struct.
        // Struct-only depth checks therefore cannot bound this valid compact envelope.
        let nested_lists = |depth| {
            let mut footer = vec![0x19];
            footer.extend(std::iter::repeat_n(0x19, depth));
            footer.extend([0x13, 0x00, 0x00]);
            footer
        };
        assert!(check(&nested_lists(MAX_DEPTH - 1)).is_ok());
        assert!(
            check(&nested_lists(MAX_DEPTH))
                .unwrap_err()
                .contains("nests deeper")
        );
    }

    #[test]
    fn a_schema_element_cannot_claim_more_children_than_the_schema_holds() {
        // Field 2 (schema), one struct whose field 5 (num_children) is 1000.
        let footer = [0x29, 0x1c, 0x55, 0xd0, 0x0f, 0x00, 0x00];
        assert!(check(&footer).unwrap_err().contains("1000 children"));
    }
}
