// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Logical types (§7.1), the table schema (§5), profiles (§3), and the name rules.

use std::fmt;

/// The scalar logical types, which are also the only list element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Scalar {
    Int32,
    Int64,
    Float,
    Double,
    Boolean,
}

impl Scalar {
    pub(crate) const fn is_floating(self) -> bool {
        matches!(self, Scalar::Float | Scalar::Double)
    }

    const fn name(self) -> &'static str {
        match self {
            Scalar::Int32 => "INT32",
            Scalar::Int64 => "INT64",
            Scalar::Float => "FLOAT",
            Scalar::Double => "DOUBLE",
            Scalar::Boolean => "BOOLEAN",
        }
    }
}

/// A logical type of §7.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LType {
    Scalar(Scalar),
    String,
    Binary,
    FixedSizeBinary(usize),
    FixedSizeList(Scalar, usize),
    List(Scalar),
}

impl LType {
    pub(crate) const INT32: LType = LType::Scalar(Scalar::Int32);
    pub(crate) const INT64: LType = LType::Scalar(Scalar::Int64);
    pub(crate) const FLOAT: LType = LType::Scalar(Scalar::Float);
    pub(crate) const DOUBLE: LType = LType::Scalar(Scalar::Double);
    pub(crate) const BOOLEAN: LType = LType::Scalar(Scalar::Boolean);

    pub(crate) const fn is_floating(self) -> bool {
        match self {
            LType::Scalar(scalar) | LType::FixedSizeList(scalar, _) | LType::List(scalar) => {
                scalar.is_floating()
            }
            _ => false,
        }
    }

    /// Whether a column read as `found` from its Parquet type stores a §5 column of this type.
    pub(crate) fn accepts(self, found: LType) -> bool {
        self.stored() == found
    }

    /// The type of the column as stored, which §7.3 hashes: a fixed-size list is a `LIST<T>`.
    pub(crate) const fn stored(self) -> LType {
        match self {
            LType::FixedSizeList(element, _) => LType::List(element),
            other => other,
        }
    }
}

impl fmt::Display for LType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LType::Scalar(scalar) => f.write_str(scalar.name()),
            LType::String => f.write_str("STRING"),
            LType::Binary => f.write_str("BINARY"),
            LType::FixedSizeBinary(n) => write!(f, "FIXED_SIZE_BINARY({n})"),
            LType::FixedSizeList(element, n) => {
                write!(f, "FIXED_SIZE_LIST<{}, {n}>", element.name())
            }
            LType::List(element) => write!(f, "LIST<{}>", element.name()),
        }
    }
}

/// A lower bound on a column's values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bound {
    /// `>= 0`
    NonNegative,
    /// `> 0`
    Positive,
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Bound::NonNegative => ">= 0",
            Bound::Positive => "> 0",
        })
    }
}

/// One §5 column.
#[derive(Debug)]
pub(crate) struct ColumnSpec {
    pub(crate) name: &'static str,
    pub(crate) ltype: LType,
    pub(crate) nullable: bool,
    pub(crate) bound: Option<Bound>,
}

/// One §5 table.
#[derive(Debug)]
pub(crate) struct TableSpec {
    pub(crate) name: &'static str,
    pub(crate) key: &'static [&'static str],
    pub(crate) columns: &'static [ColumnSpec],
}

impl TableSpec {
    pub(crate) fn column(&self, name: &str) -> Option<&'static ColumnSpec> {
        self.columns.iter().find(|c| c.name == name)
    }
}

const fn required(name: &'static str, ltype: LType) -> ColumnSpec {
    ColumnSpec {
        name,
        ltype,
        nullable: false,
        bound: None,
    }
}

const fn nullable(name: &'static str, ltype: LType) -> ColumnSpec {
    ColumnSpec {
        name,
        ltype,
        nullable: true,
        bound: None,
    }
}

const fn bounded(column: ColumnSpec, bound: Bound) -> ColumnSpec {
    ColumnSpec {
        bound: Some(bound),
        ..column
    }
}

const fn id(name: &'static str) -> ColumnSpec {
    bounded(required(name, LType::INT64), Bound::NonNegative)
}

const ROTATION: LType = LType::FixedSizeList(Scalar::Double, 9);
const VECTOR: LType = LType::FixedSizeList(Scalar::Double, 3);

/// The §5 tables, in the order the specification lists them.
pub(crate) static SCHEMA: &[TableSpec] = &[
    TableSpec {
        name: "cameras",
        key: &["id"],
        columns: &[
            id("id"),
            required("model", LType::String),
            bounded(required("width", LType::INT32), Bound::Positive),
            bounded(required("height", LType::INT32), Bound::Positive),
            bounded(nullable("fx", LType::DOUBLE), Bound::Positive),
            bounded(nullable("fy", LType::DOUBLE), Bound::Positive),
            nullable("cx", LType::DOUBLE),
            nullable("cy", LType::DOUBLE),
            nullable("k1", LType::DOUBLE),
            nullable("k2", LType::DOUBLE),
            nullable("k3", LType::DOUBLE),
            nullable("k4", LType::DOUBLE),
            nullable("p1", LType::DOUBLE),
            nullable("p2", LType::DOUBLE),
            nullable("calibrated", LType::BOOLEAN),
        ],
    },
    TableSpec {
        name: "views",
        key: &["id"],
        columns: &[
            id("id"),
            id("camera_id"),
            id("pose_id"),
            nullable("image", LType::String),
            nullable("image_sha256", LType::FixedSizeBinary(32)),
        ],
    },
    TableSpec {
        name: "poses",
        key: &["id"],
        columns: &[
            id("id"),
            required("rotation", ROTATION),
            required("center", VECTOR),
        ],
    },
    TableSpec {
        name: "points",
        key: &["id"],
        columns: &[
            id("id"),
            required("xyz", VECTOR),
            nullable("color", LType::FixedSizeBinary(3)),
            bounded(nullable("error_px", LType::DOUBLE), Bound::NonNegative),
        ],
    },
    TableSpec {
        name: "observations",
        key: &["point_id", "view_id"],
        columns: &[
            id("point_id"),
            id("view_id"),
            bounded(nullable("feature", LType::INT32), Bound::NonNegative),
            required("x", LType::DOUBLE),
            required("y", LType::DOUBLE),
        ],
    },
    TableSpec {
        name: "keypoints",
        key: &["view_id", "feature"],
        columns: &[
            id("view_id"),
            bounded(required("feature", LType::INT32), Bound::NonNegative),
            required("x", LType::DOUBLE),
            required("y", LType::DOUBLE),
            bounded(nullable("scale", LType::FLOAT), Bound::Positive),
            nullable("orientation", LType::FLOAT),
        ],
    },
    TableSpec {
        name: "matches",
        key: &[
            "view_first",
            "view_second",
            "feature_first",
            "feature_second",
        ],
        columns: &[
            id("view_first"),
            id("view_second"),
            bounded(required("feature_first", LType::INT32), Bound::NonNegative),
            bounded(required("feature_second", LType::INT32), Bound::NonNegative),
            bounded(nullable("distance", LType::FLOAT), Bound::NonNegative),
            nullable("inlier", LType::BOOLEAN),
        ],
    },
    TableSpec {
        name: "pairs",
        key: &["view_first", "view_second"],
        columns: &[
            id("view_first"),
            id("view_second"),
            nullable("model", LType::String),
            nullable("matrix", LType::FixedSizeList(Scalar::Double, 9)),
            nullable("rotation", ROTATION),
            nullable("translation", VECTOR),
        ],
    },
    TableSpec {
        name: "gt_poses",
        key: &["view_id"],
        columns: &[
            id("view_id"),
            required("rotation", ROTATION),
            required("center", VECTOR),
        ],
    },
];

/// A table §5 does not define: every column is an extra column, and there is no key.
static EXTENSION_TABLE: TableSpec = TableSpec {
    name: "",
    key: &[],
    columns: &[],
};

/// The §5 table named `name`, or the empty schema of an extension table.
pub(crate) fn table_spec(name: &str) -> &'static TableSpec {
    SCHEMA
        .iter()
        .find(|t| t.name == name)
        .unwrap_or(&EXTENSION_TABLE)
}

/// The profiles of §3 and the tables each requires.
pub(crate) const PROFILES: &[(&str, &[&str])] = &[
    ("benchmark", &["cameras", "views", "gt_poses"]),
    ("dataset", &["cameras", "views", "keypoints"]),
    (
        "reconstruction",
        &["cameras", "views", "poses", "points", "observations"],
    ),
];

pub(crate) fn profile_tables(profile: &str) -> Option<&'static [&'static str]> {
    PROFILES
        .iter()
        .find(|(name, _)| *name == profile)
        .map(|(_, tables)| *tables)
}

pub(crate) const CAMERA_MODELS: &[&str] = &["PINHOLE", "BROWN_CONRADY", "FISHEYE", "SPHERICAL"];
pub(crate) const PAIR_MODELS: &[&str] = &["essential", "fundamental", "homography", "rotation"];

/// Columns that get min/max statistics (§8).
pub(crate) const ID_COLUMNS: &[&str] = &[
    "id",
    "camera_id",
    "pose_id",
    "point_id",
    "view_id",
    "view_first",
    "view_second",
];

/// A profile or other manifest name: `[A-Za-z0-9._-]+`.
pub(crate) fn is_name(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// A table name: `[a-z0-9_]+`.
pub(crate) fn is_table_name(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// A column name: `[A-Za-z0-9_]+`.
pub(crate) fn is_column_name(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// An extension name: `OMVG_[A-Za-z0-9_]+`.
pub(crate) fn is_extension_name(s: &str) -> bool {
    s.strip_prefix("OMVG_").is_some_and(is_column_name)
}
