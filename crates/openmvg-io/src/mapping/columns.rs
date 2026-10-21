// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical tables from Rust values, and Rust values from the rows of canonical tables.

use std::sync::Arc;

use arrow_array::types::Float64Type;
use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    ListArray,
};

use crate::error::{Error, Result};
use crate::format::schema::{TableSpec, table_spec};
use crate::format::table::{Column, Table, Values};

/// An ID as the `INT64` that §4 requires.
pub(super) fn id(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::Unsupported(format!("ID {value} is above 2^63 - 1")))
}

/// A feature index or image size as the `INT32` that §4 requires.
pub(super) fn int32<T: Copy + std::fmt::Display + TryInto<i32>>(value: T) -> Result<i32> {
    value
        .try_into()
        .map_err(|_| Error::Unsupported(format!("value {value} is above 2^31 - 1")))
}

/// The columns of one table as Arrow arrays, typed against §5 when the table is built.
pub(super) struct TableBuilder {
    spec: &'static TableSpec,
    rows: usize,
    columns: Vec<(&'static str, ArrayRef)>,
}

impl TableBuilder {
    pub(super) fn new(name: &str, rows: usize) -> Self {
        Self {
            spec: table_spec(name),
            rows,
            columns: Vec::new(),
        }
    }

    fn with(mut self, name: &'static str, array: ArrayRef) -> Self {
        self.columns.push((name, array));
        self
    }

    pub(super) fn int64(self, name: &'static str, values: Vec<i64>) -> Self {
        self.with(name, Arc::new(Int64Array::from(values)))
    }

    pub(super) fn int32(self, name: &'static str, values: Vec<i32>) -> Self {
        self.with(name, Arc::new(Int32Array::from(values)))
    }

    pub(super) fn nullable_int32(self, name: &'static str, values: Vec<Option<i32>>) -> Self {
        self.with(name, Arc::new(Int32Array::from(values)))
    }

    pub(super) fn double(self, name: &'static str, values: Vec<f64>) -> Self {
        self.with(name, Arc::new(Float64Array::from(values)))
    }

    pub(super) fn nullable_double(self, name: &'static str, values: Vec<Option<f64>>) -> Self {
        self.with(name, Arc::new(Float64Array::from(values)))
    }

    pub(super) fn nullable_float(self, name: &'static str, values: Vec<Option<f32>>) -> Self {
        self.with(name, Arc::new(Float32Array::from(values)))
    }

    pub(super) fn nullable_boolean(self, name: &'static str, values: Vec<Option<bool>>) -> Self {
        self.with(name, Arc::new(BooleanArray::from(values)))
    }

    pub(super) fn nullable_string(self, name: &'static str, values: &[Option<String>]) -> Self {
        let array = BinaryArray::from_iter(values.iter().map(|v| v.as_deref().map(str::as_bytes)));
        self.with(name, Arc::new(array))
    }

    /// A `DOUBLE` list column; `None` rows are null.
    pub(super) fn double_list<const N: usize>(
        self,
        name: &'static str,
        values: &[Option<[f64; N]>],
    ) -> Self {
        let array = ListArray::from_iter_primitive::<Float64Type, _, _>(
            values.iter().map(|row| row.map(|row| row.map(Some))),
        );
        self.with(name, Arc::new(array))
    }

    /// The canonical table: every §5 column in §5 order, an unset nullable one all null.
    pub(super) fn build(self) -> Result<Table> {
        let internal = |message: String| Error::Encoding(message.into());
        let mut columns = Vec::with_capacity(self.spec.columns.len());
        for spec in self.spec.columns {
            let values = match self.columns.iter().find(|(name, _)| *name == spec.name) {
                Some((_, array)) if array.len() == self.rows => {
                    Values::from_array(array, spec.ltype)
                }
                Some(_) => None,
                None if spec.nullable => Values::nulls(spec.ltype, self.rows),
                None => {
                    return Err(internal(format!(
                        "{}.{} is unset",
                        self.spec.name, spec.name
                    )));
                }
            }
            .ok_or_else(|| {
                internal(format!(
                    "{}.{} has the wrong shape",
                    self.spec.name, spec.name
                ))
            })?;
            columns.push(Column {
                name: spec.name.to_owned(),
                ltype: spec.ltype,
                spec: Some(spec),
                values,
            });
        }
        Ok(Table {
            name: self.spec.name.to_owned(),
            spec: self.spec,
            rows: self.rows,
            columns,
        })
    }
}

/// Typed access to the rows of one table that passed every check, so that each value has its
/// §5 type and every required value is present.
pub(super) struct Rows<'a> {
    table: &'a Table,
}

impl<'a> Rows<'a> {
    pub(super) const fn new(table: &'a Table) -> Self {
        Self { table }
    }

    pub(super) const fn len(&self) -> usize {
        self.table.rows
    }

    fn values(&self, column: &str) -> Option<&'a Values> {
        self.table.values(column)
    }

    /// An `INT64` or `INT32` column as indices; the checks bound both below by zero.
    pub(super) fn indices(&self, column: &str) -> Result<Vec<usize>> {
        self.table
            .ints(column)
            .into_iter()
            .map(|value| {
                usize::try_from(value).map_err(|_| {
                    Error::Unsupported(format!(
                        "ID {value} in {}.{column} is outside this target's index range",
                        self.table.name()
                    ))
                })
            })
            .collect()
    }

    pub(super) fn double(&self, column: &str, row: usize) -> Option<f64> {
        self.values(column).and_then(|values| values.double(row))
    }

    pub(super) fn float(&self, column: &str, row: usize) -> Option<f32> {
        match self.values(column) {
            Some(Values::Float(a)) if a.is_valid(row) => Some(a.value(row)),
            _ => None,
        }
    }

    pub(super) fn int(&self, column: &str, row: usize) -> Option<i64> {
        match self.values(column) {
            Some(Values::Int32(a)) if a.is_valid(row) => Some(i64::from(a.value(row))),
            Some(Values::Int64(a)) if a.is_valid(row) => Some(a.value(row)),
            _ => None,
        }
    }

    pub(super) fn boolean(&self, column: &str, row: usize) -> Option<bool> {
        match self.values(column) {
            Some(Values::Boolean(a)) if a.is_valid(row) => Some(a.value(row)),
            _ => None,
        }
    }

    pub(super) fn string(&self, column: &str, row: usize) -> Option<&'a str> {
        self.values(column).and_then(|values| values.str(row))
    }

    /// Row `row` of a `DOUBLE` list column of `N` elements, when present.
    pub(super) fn double_list<const N: usize>(&self, column: &str, row: usize) -> Option<[f64; N]> {
        let Some(Values::List(list, elements)) = self.values(column) else {
            return None;
        };
        let Values::Double(elements) = elements.as_ref() else {
            return None;
        };
        if !list.is_valid(row) {
            return None;
        }
        elements
            .values()
            .get(Values::list_range(list, row))?
            .try_into()
            .ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::schema::LType;

    #[test]
    fn source_ids_never_alias_when_the_target_index_range_is_smaller() {
        let values = vec![i64::from(u32::MAX) + 1, i64::from(u32::MAX) + 2];
        let table = Table {
            name: "example_ids".to_owned(),
            spec: table_spec("example_ids"),
            rows: values.len(),
            columns: vec![Column {
                name: "id".to_owned(),
                ltype: LType::INT64,
                spec: None,
                values: Values::Int64(Int64Array::from(values.clone())),
            }],
        };
        let indices = Rows::new(&table).indices("id");
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            indices.unwrap(),
            values
                .into_iter()
                .map(|value| value as usize)
                .collect::<Vec<_>>()
        );
        #[cfg(target_pointer_width = "32")]
        assert!(matches!(indices, Err(Error::Unsupported(_))));
    }
}
