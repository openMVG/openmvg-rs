// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rule identifiers and the violations that name them.

use std::fmt;

macro_rules! rules {
    ($($(#[$doc:meta])* $variant:ident => $id:literal,)+) => {
        /// A rule of the OMVG format specification that a dataset can break.
        ///
        /// Each rule has an identifier, such as `key.unsorted` or `camera.prefix`, which the
        /// Python checker of the specification (`docs/format/reference/omvg.py` in the repository) reports too, so the results of the
        /// two tools compare directly.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum Rule {
            $($(#[$doc])* $variant,)+
        }

        impl Rule {
            /// The rule identifier, for example `"key.unsorted"`.
            #[must_use]
            pub const fn id(self) -> &'static str {
                match self {
                    $(Rule::$variant => $id,)+
                }
            }
        }
    };
}

rules! {
    /// `manifest.json` is missing, is not UTF-8 JSON, repeats a member, or is not an object.
    ManifestFile => "manifest.file",
    /// The manifest lacks a required member.
    ManifestMember => "manifest.member",
    /// The manifest names a format version other than this crate's.
    ManifestVersion => "manifest.version",
    /// `profiles` is not a non-empty sorted array of distinct known profile names.
    ManifestProfiles => "manifest.profiles",
    /// `length_unit` is not one of the allowed units.
    ManifestLengthUnit => "manifest.length_unit",
    /// The extension arrays are not sorted arrays of distinct `OMVG_` names, or
    /// `extensions_required` is not a subset of `extensions_used`.
    ManifestExtensions => "manifest.extensions",
    /// The dataset requires an extension that this crate does not implement.
    ManifestRequiredExtension => "manifest.required_extension",
    /// `tables` does not map table names to a row count and a hash, or lists a table that the
    /// specification does not define while no extension is used.
    ManifestTables => "manifest.tables",
    /// `content_hash` is not `sha256:` followed by 64 lowercase hexadecimal digits.
    ManifestContentHash => "manifest.content_hash",
    /// A table that one of the declared profiles requires is not listed.
    ManifestProfileTables => "manifest.profile_tables",
    /// A table file holds a different number of rows than the manifest records.
    ManifestRows => "manifest.rows",
    /// `content_hash` disagrees with the manifest's table hashes.
    HashContent => "hash.content",
    /// A table's values disagree with the hash the manifest records for it.
    HashTable => "hash.table",
    /// A table file cannot be opened or read as Parquet.
    ParquetFile => "parquet.file",
    /// A table file uses a codec other than Snappy or none.
    ParquetCodec => "parquet.codec",
    /// A column has a Parquet type outside the format's type mapping, or not the type its
    /// table schema expects.
    ParquetType => "parquet.type",
    /// A column name repeats or uses a character outside `A-Z a-z 0-9 _`.
    SchemaColumnName => "schema.column_name",
    /// A required column is absent.
    SchemaMissingColumn => "schema.missing_column",
    /// A required column holds a null.
    ValueNull => "value.null",
    /// A list holds a null element.
    ValueListNull => "value.list_null",
    /// A fixed-size list holds the wrong number of elements.
    ValueListLength => "value.list_length",
    /// A floating-point value is NaN or infinite.
    ValueNonFinite => "value.non_finite",
    /// A string is not valid UTF-8.
    ValueUtf8 => "value.utf8",
    /// A value breaks its column's lower bound.
    ValueRange => "value.range",
    /// Rows are not sorted by the table key.
    KeyUnsorted => "key.unsorted",
    /// Two rows share a key.
    KeyDuplicate => "key.duplicate",
    /// A rotation matrix is not a proper rotation.
    Rotation => "rotation",
    /// A translation direction is not a unit vector.
    UnitDirection => "unit_direction",
    /// A camera names an unknown model.
    CameraModel => "camera.model",
    /// A camera's non-null parameter columns do not match its model.
    CameraColumns => "camera.columns",
    /// A camera's distortion coefficients break the prefix rule.
    CameraPrefix => "camera.prefix",
    /// A spherical camera's image size is not `width = 2 * height`.
    CameraSphericalSize => "camera.spherical_size",
    /// An image path is not a relative POSIX path below the image root.
    ViewsImagePath => "views.image_path",
    /// Two views name the same image.
    ViewsImageDuplicate => "views.image_duplicate",
    /// A view with ground truth has no image in a benchmark dataset.
    ViewsImageNull => "views.image_null",
    /// A match row has `view_first >= view_second`.
    MatchesViewOrder => "matches.view_order",
    /// A pair row has `view_first >= view_second`.
    PairsViewOrder => "pairs.view_order",
    /// A pair names an unknown model.
    PairsModel => "pairs.model",
    /// A pair's matrix, rotation, and translation do not match its model.
    PairsModelColumns => "pairs.model_columns",
    /// A pair's matrix is all zero.
    PairsMatrixZero => "pairs.matrix_zero",
    /// A reference names a row, or a table, that does not exist.
    ReferenceDangling => "reference.dangling",
    /// A pose is not referenced by any view.
    ReferenceUnusedPose => "reference.unused_pose",
    /// An observation lies in a view whose pose is absent.
    ObservationsUnregisteredView => "observations.unregistered_view",
    /// An observation has no feature although the dataset stores keypoints.
    ObservationsFeatureNull => "observations.feature_null",
    /// An observation's pixel differs bitwise from its keypoint's.
    ObservationsPixelMismatch => "observations.pixel_mismatch",
    /// A view's features are not numbered `0, 1, 2, ...`.
    KeypointsDense => "keypoints.dense",
    /// A pair's inlier flags are null on some, but not all, of its matches.
    MatchesInlierMixed => "matches.inlier_mixed",
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// One broken rule: which rule, in which table, and a message for people.
///
/// It displays as `[rule] table: message`, or `[rule] message` for a manifest rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    rule: Rule,
    table: Option<String>,
    message: String,
}

impl Violation {
    fn new(rule: Rule, table: Option<String>, message: impl Into<String>) -> Self {
        Self {
            rule,
            table,
            message: message.into(),
        }
    }

    pub(crate) fn in_manifest(rule: Rule, message: impl Into<String>) -> Self {
        Self::new(rule, None, message)
    }

    pub(crate) fn in_table(rule: Rule, table: &str, message: impl Into<String>) -> Self {
        Self::new(rule, Some(table.to_owned()), message)
    }

    /// The broken rule.
    #[must_use]
    pub fn rule(&self) -> Rule {
        self.rule
    }

    /// The table the violation is in, or `None` for a rule of the manifest.
    #[must_use]
    pub fn table(&self) -> Option<&str> {
        self.table.as_deref()
    }

    /// What is wrong, for people; the wording is not part of the API.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.table {
            Some(table) => write!(f, "[{}] {table}: {}", self.rule, self.message),
            None => write!(f, "[{}] {}", self.rule, self.message),
        }
    }
}
