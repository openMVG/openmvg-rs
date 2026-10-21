// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The value, table, and cross-table rules (§4 to §6) over canonical tables.
//!
//! ```text
//! structural checks   per column: nulls, list elements and lengths, finiteness, UTF-8
//!        │            a table that fails one is broken
//!        v
//! table rules         per unbroken table: key order, ranges, rotations, camera models,
//!        │            image paths, view order, pair models
//!        v
//! cross-table rules   references and the rules spanning tables; a rule that reads a
//!                     broken table is skipped instead of reporting a consequence
//! ```
//!
//! Reader and writer share these checks, so a file this crate writes passes this crate's reader.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};

use arrow_array::Array;

use super::rule::{Rule, Violation};
use super::schema::{Bound, CAMERA_MODELS, LType, PAIR_MODELS, SCHEMA};
use super::table::{Table, Values};

/// Tolerance on ‖RᵀR − I‖_F and |det R − 1| (§4).
const ROTATION_TOLERANCE: f64 = 1e-8;
/// Tolerance on |‖t‖ − 1| for a unit direction (§4).
const DIRECTION_TOLERANCE: f64 = 1e-8;

/// Canonical tables by name, and the names of the tables that failed a structural check.
pub(crate) type Tables = BTreeMap<String, Table>;
pub(crate) type Broken = BTreeSet<String>;

/// How many rows a rule flags and the first of them.
fn rows_text(rows: &[usize]) -> String {
    match rows.first() {
        Some(first) => format!("{} row(s), first at row {first}", rows.len()),
        None => "no rows".to_owned(),
    }
}

fn flagged(mask: impl IntoIterator<Item = bool>) -> Vec<usize> {
    mask.into_iter()
        .enumerate()
        .filter_map(|(i, bad)| bad.then_some(i))
        .collect()
}

/// The floating-point value at index `i`, when present.
fn floating(values: &Values, i: usize) -> Option<f64> {
    match values {
        Values::Float(a) if a.is_valid(i) => Some(f64::from(a.value(i))),
        Values::Double(a) if a.is_valid(i) => Some(a.value(i)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Structural checks
// ---------------------------------------------------------------------------------------------

/// Nulls in required columns, list element nulls and lengths, non-finite floats, and UTF-8.
pub(crate) fn structural_checks(table: &Table) -> Vec<Violation> {
    let name = table.name();
    let mut out = Vec::new();
    for column in &table.columns {
        let (values, lt, c) = (&column.values, column.ltype, &column.name);
        if column.required() && values.null_count() > 0 {
            out.push(Violation::in_table(
                Rule::ValueNull,
                name,
                format!("required column {c:?} has {} null(s)", values.null_count()),
            ));
        }
        let mut non_finite = 0usize;
        if let Values::List(list, elements) = values {
            let mut null_elements = 0usize;
            let mut wrong_length = Vec::new();
            for row in (0..list.len()).filter(|&row| list.is_valid(row)) {
                let range = Values::list_range(list, row);
                if let LType::FixedSizeList(_, n) = lt
                    && range.len() != n
                {
                    wrong_length.push(row);
                }
                for i in range {
                    if !elements.is_valid(i) {
                        null_elements += 1;
                    } else if floating(elements, i).is_some_and(|v| !v.is_finite()) {
                        non_finite += 1;
                    }
                }
            }
            if null_elements > 0 {
                out.push(Violation::in_table(
                    Rule::ValueListNull,
                    name,
                    format!("column {c:?} has {null_elements} null list element(s)"),
                ));
            }
            if let LType::FixedSizeList(_, n) = lt
                && !wrong_length.is_empty()
            {
                out.push(Violation::in_table(
                    Rule::ValueListLength,
                    name,
                    format!(
                        "column {c:?} needs {n} elements: {}",
                        rows_text(&wrong_length)
                    ),
                ));
            }
        } else if lt.is_floating() {
            non_finite = (0..values.len())
                .filter(|&i| floating(values, i).is_some_and(|v| !v.is_finite()))
                .count();
        }
        if non_finite > 0 {
            out.push(Violation::in_table(
                Rule::ValueNonFinite,
                name,
                format!("column {c:?} has {non_finite} non-finite value(s)"),
            ));
        }
        if lt == LType::String {
            let invalid: Vec<usize> = (0..values.len())
                .filter(|&row| {
                    values
                        .bytes(row)
                        .is_some_and(|b| std::str::from_utf8(b).is_err())
                })
                .collect();
            if !invalid.is_empty() {
                out.push(Violation::in_table(
                    Rule::ValueUtf8,
                    name,
                    format!("column {c:?} is not valid UTF-8: {}", rows_text(&invalid)),
                ));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Row keys and lookups
// ---------------------------------------------------------------------------------------------

/// Key columns as integer arrays; a key column holding nulls becomes (validity, value), so that
/// a null sorts first.
pub(crate) fn key_arrays(table: &Table, key: &[&str]) -> Vec<Vec<i64>> {
    let mut arrays = Vec::new();
    for column in key {
        let Some(values) = table.values(column) else {
            continue;
        };
        if values.null_count() > 0 {
            arrays.push(values.valid_mask().into_iter().map(i64::from).collect());
        }
        arrays.push(values.ints().unwrap_or_default());
    }
    arrays
}

fn compare_rows(a: &[Vec<i64>], i: usize, b: &[Vec<i64>], j: usize) -> Ordering {
    a.iter()
        .zip(b)
        .map(|(x, y)| x[i].cmp(&y[j]))
        .find(|o| o.is_ne())
        .unwrap_or(Ordering::Equal)
}

/// Row indices of `columns` in ascending key order; equal rows keep their order.
pub(crate) fn sort_order(columns: &[Vec<i64>]) -> Vec<usize> {
    let rows = columns.first().map_or(0, Vec::len);
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&i, &j| compare_rows(columns, i, columns, j));
    order
}

/// For each row of `query`, the first row of `reference` with equal values, if any.
fn lookup(reference: &[Vec<i64>], query: &[Vec<i64>]) -> Vec<Option<usize>> {
    let order = sort_order(reference);
    let rows = query.first().map_or(0, Vec::len);
    (0..rows)
        .map(|q| {
            let position = order.partition_point(|&r| compare_rows(reference, r, query, q).is_lt());
            order
                .get(position)
                .copied()
                .filter(|&r| compare_rows(reference, r, query, q).is_eq())
        })
        .collect()
}

/// Start index of every run of equal rows in sorted columns.
fn runs(columns: &[&[i64]]) -> Vec<usize> {
    let rows = columns.first().map_or(0, |c| c.len());
    (0..rows)
        .filter(|&i| i == 0 || columns.iter().any(|c| c[i] != c[i - 1]))
        .collect()
}

fn permuted(values: &[i64], order: &[usize]) -> Vec<i64> {
    order.iter().map(|&i| values[i]).collect()
}

// ---------------------------------------------------------------------------------------------
// Table rules
// ---------------------------------------------------------------------------------------------

/// Rows of a bounded column whose present value breaks the bound.
fn out_of_range(values: &Values, bound: Bound) -> Vec<usize> {
    let below = |sign: Ordering| match bound {
        Bound::NonNegative => sign.is_lt(),
        Bound::Positive => sign.is_le(),
    };
    (0..values.len())
        .filter(|&row| {
            values.is_valid(row)
                && match values {
                    Values::Int32(a) => below(a.value(row).cmp(&0)),
                    Values::Int64(a) => below(a.value(row).cmp(&0)),
                    // NaN compares as neither; `value.non_finite` reports it.
                    Values::Float(a) => a.value(row).partial_cmp(&0.0).is_some_and(below),
                    Values::Double(a) => a.value(row).partial_cmp(&0.0).is_some_and(below),
                    _ => false,
                }
        })
        .collect()
}

/// Whether a row-major 3×3 matrix is not a rotation within the tolerance.
fn rotation_defect(r: &[f64; 9]) -> bool {
    let at = |i: usize, j: usize| r[3 * i + j];
    let mut frobenius = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let gram =
                (0..3).map(|k| at(k, i) * at(k, j)).sum::<f64>() - if i == j { 1.0 } else { 0.0 };
            frobenius += gram * gram;
        }
    }
    let det = at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1))
        - at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0))
        + at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
    let frobenius = frobenius.sqrt();
    !frobenius.is_finite()
        || !det.is_finite()
        || frobenius > ROTATION_TOLERANCE
        || (det - 1.0).abs() > ROTATION_TOLERANCE
}

/// Whether `path` is a relative POSIX path that stays inside the dataset (§5.2).
pub(crate) fn image_path_ok(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && path.split('/').all(|part| !matches!(part, "" | "." | ".."))
}

/// Rules local to one canonical table: keys, ranges, rotations, camera models, and paths.
fn table_rules(table: &Table) -> Vec<Violation> {
    let (name, spec) = (table.name(), table.spec);
    let mut out = Vec::new();

    let keys = key_arrays(table, spec.key);
    let order: Vec<Ordering> = (1..table.rows)
        .map(|i| compare_rows(&keys, i, &keys, i - 1))
        .collect();
    let descending = flagged(order.iter().map(|o| o.is_lt()));
    if !descending.is_empty() {
        out.push(Violation::in_table(
            Rule::KeyUnsorted,
            name,
            format!(
                "rows are not sorted by {:?}: {}",
                spec.key,
                rows_text(&descending)
            ),
        ));
    }
    let equal = flagged(order.iter().map(|o| o.is_eq()));
    if !equal.is_empty() {
        out.push(Violation::in_table(
            Rule::KeyDuplicate,
            name,
            format!("duplicate key {:?}: {}", spec.key, rows_text(&equal)),
        ));
    }

    for c in spec.columns {
        let (Some(bound), Some(values)) = (c.bound, table.values(c.name)) else {
            continue;
        };
        let bad = out_of_range(values, bound);
        if !bad.is_empty() {
            out.push(Violation::in_table(
                Rule::ValueRange,
                name,
                format!("column {:?} must be {bound}: {}", c.name, rows_text(&bad)),
            ));
        }
    }

    if spec.column("rotation").is_some()
        && let Some(values) = table.values("rotation")
    {
        let (rows, flat) = values.double_rows(9);
        let bad: Vec<usize> = rows
            .iter()
            .zip(flat.as_chunks::<9>().0)
            .filter(|(_, r)| rotation_defect(r))
            .map(|(row, _)| *row)
            .collect();
        if !bad.is_empty() {
            out.push(Violation::in_table(
                Rule::Rotation,
                name,
                format!("improper rotation: {}", rows_text(&bad)),
            ));
        }
    }
    if spec.column("translation").is_some()
        && let Some(values) = table.values("translation")
    {
        let (rows, flat) = values.double_rows(3);
        let bad: Vec<usize> = rows
            .iter()
            .zip(flat.as_chunks::<3>().0)
            .filter(|(_, t)| {
                ((t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt() - 1.0).abs() > DIRECTION_TOLERANCE
            })
            .map(|(row, _)| *row)
            .collect();
        if !bad.is_empty() {
            out.push(Violation::in_table(
                Rule::UnitDirection,
                name,
                format!("translation is not a unit vector: {}", rows_text(&bad)),
            ));
        }
    }

    match name {
        "cameras" => out.extend(camera_rules(table)),
        "views" => {
            if let Some(images) = table.values("image") {
                let bad = flagged(
                    (0..table.rows)
                        .map(|row| images.str(row).is_some_and(|path| !image_path_ok(path))),
                );
                if !bad.is_empty() {
                    out.push(Violation::in_table(
                        Rule::ViewsImagePath,
                        name,
                        format!(
                            "image is not a relative POSIX path inside the dataset: {}",
                            rows_text(&bad)
                        ),
                    ));
                }
                let mut seen = HashSet::new();
                let repeated = flagged(
                    (0..table.rows)
                        .map(|row| images.str(row).is_some_and(|path| !seen.insert(path))),
                );
                if !repeated.is_empty() {
                    out.push(Violation::in_table(
                        Rule::ViewsImageDuplicate,
                        name,
                        format!(
                            "image repeats an earlier row's image: {}",
                            rows_text(&repeated)
                        ),
                    ));
                }
            }
        }
        "matches" | "pairs" => {
            let (first, second) = (table.ints("view_first"), table.ints("view_second"));
            let bad = flagged(first.iter().zip(&second).map(|(a, b)| a >= b));
            if !bad.is_empty() {
                let rule = if name == "pairs" {
                    Rule::PairsViewOrder
                } else {
                    Rule::MatchesViewOrder
                };
                out.push(Violation::in_table(
                    rule,
                    name,
                    format!("view_first must be < view_second: {}", rows_text(&bad)),
                ));
            }
            if name == "pairs" {
                out.extend(pair_model_rules(table));
            }
        }
        _ => {}
    }
    out
}

const CAMERA_COLUMNS: [&str; 10] = ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2"];
const INTRINSICS: &[&str] = &["fx", "fy", "cx", "cy"];

/// §6: the columns each model uses, the prefix rule, and the spherical image size.
fn camera_rules(table: &Table) -> Vec<Violation> {
    let ids = table.ints("id");
    let (widths, heights) = (table.ints("width"), table.ints("height"));
    let Some(models) = table.values("model") else {
        return Vec::new();
    };
    let coefficients: Vec<(&str, Option<&Values>)> = CAMERA_COLUMNS
        .iter()
        .map(|c| (*c, table.values(c)))
        .collect();
    let (mut model_bad, mut columns_bad, mut prefix_bad, mut size_bad) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for row in 0..table.rows {
        let is_set = |c: &str| {
            coefficients
                .iter()
                .any(|(name, values)| *name == c && values.is_some_and(|v| v.is_valid(row)))
        };
        let present: Vec<&str> = CAMERA_COLUMNS
            .iter()
            .copied()
            .filter(|c| is_set(c))
            .collect();
        let Some(model) = models.str(row).filter(|m| CAMERA_MODELS.contains(m)) else {
            model_bad.push(ids[row]);
            continue;
        };
        let (required, allowed, prefixes): (&[&str], &[&str], &[&[&str]]) = match model {
            "SPHERICAL" => {
                if widths[row] % 2 != 0 || heights[row] * 2 != widths[row] {
                    size_bad.push(ids[row]);
                }
                (&[], &[], &[])
            }
            "PINHOLE" => (INTRINSICS, INTRINSICS, &[]),
            "BROWN_CONRADY" => (
                INTRINSICS,
                &["fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2"],
                &[&["k1", "k2", "k3"], &["p1", "p2"]],
            ),
            _ => (
                INTRINSICS,
                &["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
                &[&["k1", "k2", "k3", "k4"]],
            ),
        };
        if !required.iter().all(|c| present.contains(c))
            || !present.iter().all(|c| allowed.contains(c))
        {
            columns_bad.push(ids[row]);
        } else if prefixes
            .iter()
            .any(|seq| seq.windows(2).any(|w| is_set(w[1]) && !is_set(w[0])))
        {
            prefix_bad.push(ids[row]);
        }
    }
    let mut out = Vec::new();
    for (rule, ids, what) in [
        (
            Rule::CameraModel,
            model_bad,
            format!("model must be one of {CAMERA_MODELS:?}"),
        ),
        (
            Rule::CameraColumns,
            columns_bad,
            "parameter columns do not match the model".to_owned(),
        ),
        (
            Rule::CameraPrefix,
            prefix_bad,
            "coefficients break the prefix rule".to_owned(),
        ),
        (
            Rule::CameraSphericalSize,
            size_bad,
            "SPHERICAL needs an even width and height = width / 2".to_owned(),
        ),
    ] {
        if !ids.is_empty() {
            let shown = &ids[..ids.len().min(5)];
            out.push(Violation::in_table(
                rule,
                "cameras",
                format!("{what}: camera id(s) {shown:?}"),
            ));
        }
    }
    out
}

/// §5.8: which of matrix, rotation, and translation each pair model carries.
fn pair_model_rules(table: &Table) -> Vec<Violation> {
    let mut out = Vec::new();
    let has = |column: &str| {
        table
            .values(column)
            .map(Values::valid_mask)
            .unwrap_or_else(|| vec![false; table.rows])
    };
    let (matrix, rotation, translation) = (has("matrix"), has("rotation"), has("translation"));
    let model: Vec<Option<&str>> = (0..table.rows)
        .map(|row| table.values("model").and_then(|m| m.str(row)))
        .collect();

    let unknown = flagged(
        model
            .iter()
            .map(|m| m.is_some_and(|m| !PAIR_MODELS.contains(&m))),
    );
    if !unknown.is_empty() {
        out.push(Violation::in_table(
            Rule::PairsModel,
            "pairs",
            format!(
                "model must be one of {PAIR_MODELS:?} or null: {}",
                rows_text(&unknown)
            ),
        ));
    }
    let mismatched = flagged((0..table.rows).map(|row| {
        let (m, r, t) = (matrix[row], rotation[row], translation[row]);
        let ok = match model[row] {
            Some("essential" | "fundamental" | "homography") => m && r == t,
            Some("rotation") => !m && r && !t,
            Some(other) if !PAIR_MODELS.contains(&other) => return false, // reported as pairs.model
            _ => !m && !r && !t,
        };
        !ok
    }));
    if !mismatched.is_empty() {
        out.push(Violation::in_table(
            Rule::PairsModelColumns,
            "pairs",
            format!(
                "matrix/rotation/translation do not match the model: {}",
                rows_text(&mismatched)
            ),
        ));
    }
    if let Some(values) = table.values("matrix") {
        let (rows, flat) = values.double_rows(9);
        let zero: Vec<usize> = rows
            .iter()
            .zip(flat.as_chunks::<9>().0)
            .filter(|(_, m)| m.iter().all(|v| *v == 0.0))
            .map(|(row, _)| *row)
            .collect();
        if !zero.is_empty() {
            let shown = &zero[..zero.len().min(5)];
            out.push(Violation::in_table(
                Rule::PairsMatrixZero,
                "pairs",
                format!("matrix is all zero at row(s) {shown:?}"),
            ));
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Cross-table rules
// ---------------------------------------------------------------------------------------------

/// The usable tables, the broken ones, and the violations found so far.
struct Cross<'a> {
    tables: &'a Tables,
    broken: &'a Broken,
    out: Vec<Violation>,
}

impl<'a> Cross<'a> {
    /// A table that is present and passed its structural checks.
    fn usable(&self, name: &str) -> Option<&'a Table> {
        self.tables
            .get(name)
            .filter(|_| !self.broken.contains(name))
    }

    /// Check that `source.columns` references `target.target_columns` on the rows `mask` keeps;
    /// return, per kept row, the first matching target row. `None` when the target is broken
    /// or absent.
    fn reference(
        &mut self,
        source: &Table,
        columns: &[&str],
        target: &str,
        target_columns: &[&str],
        mask: Option<&[bool]>,
    ) -> Option<Vec<Option<usize>>> {
        if self.broken.contains(target) {
            return None;
        }
        let Some(dst) = self.usable(target) else {
            self.out.push(Violation::in_table(
                Rule::ReferenceDangling,
                source.name(),
                format!("{columns:?} references table {target:?}, which is absent"),
            ));
            return None;
        };
        let query: Vec<Vec<i64>> = columns
            .iter()
            .map(|c| {
                let values = source.ints(c);
                match mask {
                    Some(mask) => values
                        .into_iter()
                        .zip(mask)
                        .filter_map(|(v, keep)| keep.then_some(v))
                        .collect(),
                    None => values,
                }
            })
            .collect();
        let reference: Vec<Vec<i64>> = target_columns.iter().map(|c| dst.ints(c)).collect();
        let index = lookup(&reference, &query);
        let dangling = flagged(index.iter().map(Option::is_none));
        if !dangling.is_empty() {
            self.out.push(Violation::in_table(
                Rule::ReferenceDangling,
                source.name(),
                format!(
                    "{columns:?} has no match in {target}{target_columns:?}: {}",
                    rows_text(&dangling)
                ),
            ));
        }
        Some(index)
    }

    fn push(&mut self, rule: Rule, table: &str, message: String) {
        self.out.push(Violation::in_table(rule, table, message));
    }
}

/// How many pairs of views have a null `inlier` on some, but not all, of their `matches` rows.
fn mixed_inlier_pairs(matches: &Table) -> usize {
    let (first, second) = (matches.ints("view_first"), matches.ints("view_second"));
    let order = sort_order(&[first.clone(), second.clone()]);
    let (first, second) = (permuted(&first, &order), permuted(&second, &order));
    let inlier = matches.values("inlier");
    let is_null: Vec<bool> = order
        .iter()
        .map(|&i| inlier.is_none_or(|v| !v.is_valid(i)))
        .collect();
    let starts = runs(&[&first, &second]);
    let ends = starts.iter().skip(1).copied().chain([first.len()]);
    starts
        .iter()
        .zip(ends)
        .filter(|&(&start, end)| {
            let pair = &is_null[start..end];
            pair.contains(&true) && pair.contains(&false)
        })
        .count()
}

/// References between tables and the rules that span tables (§5).
fn cross_table_rules(tables: &Tables, broken: &Broken, profiles: &[String]) -> Vec<Violation> {
    let mut x = Cross {
        tables,
        broken,
        out: Vec::new(),
    };

    if let Some(views) = x.usable("views") {
        x.reference(views, &["camera_id"], "cameras", &["id"], None);
    }
    if let (Some(poses), Some(views)) = (x.usable("poses"), x.usable("views")) {
        let used = lookup(&[views.ints("pose_id")], &[poses.ints("id")]);
        let unused = flagged(used.iter().map(Option::is_none));
        if !unused.is_empty() {
            x.push(
                Rule::ReferenceUnusedPose,
                "poses",
                format!(
                    "pose not referenced by any views.pose_id: {}",
                    rows_text(&unused)
                ),
            );
        }
    }
    if let Some(obs) = x.usable("observations") {
        x.reference(obs, &["point_id"], "points", &["id"], None);
        let view_rows = x.reference(obs, &["view_id"], "views", &["id"], None);
        if let (Some(view_rows), Some(views)) = (view_rows, x.usable("views"))
            && !broken.contains("poses")
        {
            let pose_of_view = views.ints("pose_id");
            let pose_ids: Vec<i64> = view_rows
                .iter()
                .flatten()
                .map(|&r| pose_of_view[r])
                .collect();
            let poses = x.usable("poses").map(|p| p.ints("id")).unwrap_or_default();
            let unregistered = lookup(&[poses], &[pose_ids])
                .iter()
                .filter(|r| r.is_none())
                .count();
            if unregistered > 0 {
                x.push(
                    Rule::ObservationsUnregisteredView,
                    "observations",
                    format!("view's pose_id has no poses row: {unregistered} row(s)"),
                );
            }
        }
        if let Some(keypoints) = x.usable("keypoints") {
            let has_feature = obs
                .values("feature")
                .map(Values::valid_mask)
                .unwrap_or_else(|| vec![false; obs.rows]);
            let missing = has_feature.iter().filter(|b| !**b).count();
            if missing > 0 {
                x.push(
                    Rule::ObservationsFeatureNull,
                    "observations",
                    format!("feature must be non-null when keypoints is present: {missing} row(s)"),
                );
            }
            let rows = x.reference(
                obs,
                &["view_id", "feature"],
                "keypoints",
                &["view_id", "feature"],
                Some(&has_feature),
            );
            if let Some(rows) = rows {
                let observed: Vec<usize> = flagged(has_feature.iter().copied());
                for axis in ["x", "y"] {
                    let (Some(o), Some(k)) = (obs.values(axis), keypoints.values(axis)) else {
                        continue;
                    };
                    let differ = observed
                        .iter()
                        .zip(&rows)
                        .filter_map(|(&o_row, k_row)| k_row.map(|k_row| (o_row, k_row)))
                        .filter(|&(o_row, k_row)| {
                            o.double(o_row).map(f64::to_bits) != k.double(k_row).map(f64::to_bits)
                        })
                        .count();
                    if differ > 0 {
                        x.push(
                            Rule::ObservationsPixelMismatch,
                            "observations",
                            format!("{axis} differs bitwise from the keypoint: {differ} row(s)"),
                        );
                        break;
                    }
                }
            }
        }
    }
    if let Some(keypoints) = x.usable("keypoints") {
        x.reference(keypoints, &["view_id"], "views", &["id"], None);
        let (view, feature) = (keypoints.ints("view_id"), keypoints.ints("feature"));
        let order = sort_order(&[view.clone(), feature.clone()]);
        let (view, feature) = (permuted(&view, &order), permuted(&feature, &order));
        let starts = runs(&[&view]);
        let mut start = 0usize;
        let mut run = starts.iter().peekable();
        let sparse = flagged((0..view.len()).map(|i| {
            if run.peek().is_some_and(|&&s| s == i) {
                start = i;
                run.next();
            }
            usize::try_from(feature[i]).ok() != Some(i - start)
        }));
        if !sparse.is_empty() {
            x.push(
                Rule::KeypointsDense,
                "keypoints",
                format!(
                    "features must be 0, 1, 2, ... within each view: {}",
                    rows_text(&sparse)
                ),
            );
        }
    }
    if let Some(matches) = x.usable("matches") {
        for side in ["first", "second"] {
            let (view, feature) = (format!("view_{side}"), format!("feature_{side}"));
            x.reference(
                matches,
                &[&view, &feature],
                "keypoints",
                &["view_id", "feature"],
                None,
            );
        }
        let mixed = mixed_inlier_pairs(matches);
        if mixed > 0 {
            x.push(
                Rule::MatchesInlierMixed,
                "matches",
                format!("inlier is null on some but not all rows of {mixed} pair(s)"),
            );
        }
    }
    if let Some(pairs) = x.usable("pairs") {
        x.reference(pairs, &["view_first"], "views", &["id"], None);
        x.reference(pairs, &["view_second"], "views", &["id"], None);
    }
    if let Some(gt) = x.usable("gt_poses") {
        let view_rows = x.reference(gt, &["view_id"], "views", &["id"], None);
        // A benchmark is evaluated on images, so every view with ground truth names one.
        if let (Some(view_rows), Some(views)) = (view_rows, x.usable("views"))
            && profiles.iter().any(|p| p == "benchmark")
        {
            let (ids, images) = (views.ints("id"), views.values("image"));
            let missing: Vec<i64> = view_rows
                .iter()
                .flatten()
                .filter(|&&row| images.is_none_or(|v| !v.is_valid(row)))
                .map(|&row| ids[row])
                .collect();
            if !missing.is_empty() {
                let shown = &missing[..missing.len().min(5)];
                x.push(
                    Rule::ViewsImageNull,
                    "views",
                    format!(
                        "image must be non-null for a view with a gt_poses row: view id(s) {shown:?}"
                    ),
                );
            }
        }
    }
    x.out
}

/// Every §4 to §6 rule over canonical tables. The rules of a table in `broken`, and the
/// cross-table rules that read it, are skipped; a table §5 does not define gets no rule here.
pub(crate) fn validate_tables(
    tables: &Tables,
    broken: &Broken,
    profiles: &[String],
) -> Vec<Violation> {
    let mut out: Vec<Violation> = SCHEMA
        .iter()
        .filter(|spec| !broken.contains(spec.name))
        .filter_map(|spec| tables.get(spec.name))
        .flat_map(table_rules)
        .collect();
    out.extend(cross_table_rules(tables, broken, profiles));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_rule_rejects_overflowing_finite_statistics() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(!rotation_defect(&identity));

        // The input is finite, but its dot products and determinant contain inf - inf.
        // A validator must reject this matrix when the derived errors become NaN.
        let overflowing = [
            1e308, 1e308, 1e308, 1e308, -1e308, 1e308, 1e308, 1e308, -1e308,
        ];
        assert!(rotation_defect(&overflowing));
    }

    #[test]
    fn image_paths_stay_inside_the_dataset() {
        assert!(image_path_ok("images/a.jpg") && image_path_ok("a..b/c"));
        for bad in ["", "/a.jpg", "a\\b.jpg", "a//b", "./a", "a/../b", "a/"] {
            assert!(!image_path_ok(bad), "{bad}");
        }
    }
}
