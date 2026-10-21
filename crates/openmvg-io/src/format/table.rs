// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical in-memory tables: every §5 column of a table in §5 order, followed by its extra
//! columns, each held in the one Arrow array type of its logical type. Every column of an
//! extension table is an extra column.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, FixedSizeBinaryArray, Float32Array, Float64Array,
    Int32Array, Int64Array, ListArray, new_null_array,
};
use arrow_schema::{DataType, Field};

use super::schema::{ColumnSpec, LType, Scalar, TableSpec};

/// The values of one column, typed by its logical type.
///
/// `STRING` and `BINARY` both live in `Bytes`; the structural checks validate the UTF-8 of a
/// `STRING` column, and the writer converts it back to an Arrow string array. `FIXED_SIZE_LIST`
/// and `LIST` both live in `List`, a plain Arrow list, as §8 stores them on disk.
#[derive(Debug, Clone)]
pub(crate) enum Values {
    Int32(Int32Array),
    Int64(Int64Array),
    Float(Float32Array),
    Double(Float64Array),
    Boolean(BooleanArray),
    Bytes(BinaryArray),
    FixedBytes(FixedSizeBinaryArray),
    /// A list array and its element values.
    List(ListArray, Box<Values>),
}

/// The in-memory Arrow type of a scalar.
pub(crate) fn scalar_type(scalar: Scalar) -> DataType {
    match scalar {
        Scalar::Int32 => DataType::Int32,
        Scalar::Int64 => DataType::Int64,
        Scalar::Float => DataType::Float32,
        Scalar::Double => DataType::Float64,
        Scalar::Boolean => DataType::Boolean,
    }
}

/// The list element field of a canonical list. It is nullable so that a file with null list
/// elements still loads and fails `value.list_null` instead of a decoding error.
pub(crate) fn element_field(scalar: Scalar) -> Arc<Field> {
    Arc::new(Field::new("element", scalar_type(scalar), true))
}

/// The in-memory Arrow type that holds a column of logical type `lt`.
pub(crate) fn arrow_type(lt: LType) -> DataType {
    match lt {
        LType::Scalar(scalar) => scalar_type(scalar),
        LType::String | LType::Binary => DataType::Binary,
        LType::FixedSizeBinary(n) => {
            DataType::FixedSizeBinary(i32::try_from(n).unwrap_or(i32::MAX))
        }
        LType::FixedSizeList(element, _) | LType::List(element) => {
            DataType::List(element_field(element))
        }
    }
}

impl Values {
    /// Type `array` as the canonical values of a column of logical type `lt`, or `None` when
    /// its Arrow type is not the canonical one.
    pub(crate) fn from_array(array: &ArrayRef, lt: LType) -> Option<Values> {
        let any = array.as_any();
        Some(match lt {
            LType::Scalar(Scalar::Int32) => {
                Values::Int32(any.downcast_ref::<Int32Array>()?.clone())
            }
            LType::Scalar(Scalar::Int64) => {
                Values::Int64(any.downcast_ref::<Int64Array>()?.clone())
            }
            LType::Scalar(Scalar::Float) => {
                Values::Float(any.downcast_ref::<Float32Array>()?.clone())
            }
            LType::Scalar(Scalar::Double) => {
                Values::Double(any.downcast_ref::<Float64Array>()?.clone())
            }
            LType::Scalar(Scalar::Boolean) => {
                Values::Boolean(any.downcast_ref::<BooleanArray>()?.clone())
            }
            LType::String | LType::Binary => {
                Values::Bytes(any.downcast_ref::<BinaryArray>()?.clone())
            }
            LType::FixedSizeBinary(n) => {
                let fixed = any.downcast_ref::<FixedSizeBinaryArray>()?;
                if usize::try_from(fixed.value_length()).ok()? != n {
                    return None;
                }
                Values::FixedBytes(fixed.clone())
            }
            LType::FixedSizeList(element, _) | LType::List(element) => {
                let list = any.downcast_ref::<ListArray>()?;
                // Rebuild with the canonical element field, whatever the file named it.
                let list = ListArray::try_new(
                    element_field(element),
                    list.offsets().clone(),
                    list.values().clone(),
                    list.nulls().cloned(),
                )
                .ok()?;
                let elements = Values::from_array(list.values(), LType::Scalar(element))?;
                Values::List(list, Box::new(elements))
            }
        })
    }

    /// An all-null column of logical type `lt`.
    pub(crate) fn nulls(lt: LType, rows: usize) -> Option<Values> {
        Values::from_array(&new_null_array(&arrow_type(lt), rows), lt)
    }

    pub(crate) fn array(&self) -> &dyn Array {
        match self {
            Values::Int32(a) => a,
            Values::Int64(a) => a,
            Values::Float(a) => a,
            Values::Double(a) => a,
            Values::Boolean(a) => a,
            Values::Bytes(a) => a,
            Values::FixedBytes(a) => a,
            Values::List(a, _) => a,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.array().len()
    }

    pub(crate) fn null_count(&self) -> usize {
        self.array().null_count()
    }

    pub(crate) fn is_valid(&self, row: usize) -> bool {
        self.array().is_valid(row)
    }

    /// Per row: whether the value is present.
    pub(crate) fn valid_mask(&self) -> Vec<bool> {
        (0..self.len()).map(|row| self.is_valid(row)).collect()
    }

    /// An integer column as `i64` values, with nulls as zero.
    pub(crate) fn ints(&self) -> Option<Vec<i64>> {
        match self {
            Values::Int32(a) => Some(
                (0..a.len())
                    .map(|i| {
                        if a.is_valid(i) {
                            i64::from(a.value(i))
                        } else {
                            0
                        }
                    })
                    .collect(),
            ),
            Values::Int64(a) => Some(
                (0..a.len())
                    .map(|i| if a.is_valid(i) { a.value(i) } else { 0 })
                    .collect(),
            ),
            _ => None,
        }
    }

    /// The element range of list row `row`.
    pub(crate) fn list_range(list: &ListArray, row: usize) -> std::ops::Range<usize> {
        let offsets = list.value_offsets();
        let start = usize::try_from(offsets[row]).unwrap_or(0);
        let end = usize::try_from(offsets[row + 1]).unwrap_or(start);
        start..end
    }

    /// The valid rows of a `DOUBLE` list column of fixed `width`, as row indices and their
    /// row-major values. The structural checks have already checked every length.
    pub(crate) fn double_rows(&self, width: usize) -> (Vec<usize>, Vec<f64>) {
        let mut rows = Vec::new();
        let mut values = Vec::new();
        if let Values::List(list, elements) = self
            && let Values::Double(elements) = elements.as_ref()
        {
            for row in 0..list.len() {
                let range = Values::list_range(list, row);
                if list.is_valid(row) && range.len() == width {
                    rows.push(row);
                    values.extend(range.map(|i| elements.value(i)));
                }
            }
        }
        (rows, values)
    }

    /// Row `row` of a `STRING` or `BINARY` column, when present.
    pub(crate) fn bytes(&self, row: usize) -> Option<&[u8]> {
        match self {
            Values::Bytes(a) if a.is_valid(row) => Some(a.value(row)),
            _ => None,
        }
    }

    /// Row `row` of a `STRING` column, when present and valid UTF-8.
    pub(crate) fn str(&self, row: usize) -> Option<&str> {
        self.bytes(row)
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
    }

    /// Row `row` of a `DOUBLE` column, when present.
    pub(crate) fn double(&self, row: usize) -> Option<f64> {
        match self {
            Values::Double(a) if a.is_valid(row) => Some(a.value(row)),
            _ => None,
        }
    }
}

/// One column of a canonical table.
#[derive(Debug, Clone)]
pub(crate) struct Column {
    pub(crate) name: String,
    pub(crate) ltype: LType,
    /// The §5 column, or `None` for an extra column.
    pub(crate) spec: Option<&'static ColumnSpec>,
    pub(crate) values: Values,
}

impl Column {
    /// Whether §5 requires every value.
    pub(crate) fn required(&self) -> bool {
        self.spec.is_some_and(|c| !c.nullable)
    }
}

/// A canonical table: every §5 column of its schema in §5 order (an absent nullable column as
/// an all-null column), followed by its extra columns.
#[derive(Debug, Clone)]
pub(crate) struct Table {
    pub(crate) name: String,
    /// The §5 table, or the empty schema of an extension table.
    pub(crate) spec: &'static TableSpec,
    pub(crate) rows: usize,
    pub(crate) columns: Vec<Column>,
}

impl Table {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }

    pub(crate) fn values(&self, name: &str) -> Option<&Values> {
        self.column(name).map(|c| &c.values)
    }

    /// An integer column as `i64` values with nulls as zero; empty when the column is absent.
    pub(crate) fn ints(&self, name: &str) -> Vec<i64> {
        self.values(name).and_then(Values::ints).unwrap_or_default()
    }
}
