# OMVG format specification

**Version:** `0.1-draft`. Breaking changes are allowed until 1.0.

**Scope:** a compact, language-neutral format for sharing multi-view datasets, benchmarks, and
sparse reconstructions: a JSON manifest plus one Parquet file per table.

**Companion files:** [`testdata/`](testdata/) holds the conformance vectors (§10), and
[`reference/omvg.py`](reference/omvg.py) is a Python reference reader, writer, and checker built
on pyarrow. Where either disagrees with this text, this text is normative.

The key words MUST, MUST NOT, SHOULD, and MAY are used as in RFC 2119.

## 1. Goals and non-goals

Goals:

- Readable in any language with commodity tools: JSON plus Parquet (pandas, polars, duckdb,
  pyarrow, arrow-rs).
- Compact enough to share large benchmark corpora (Appendix C).
- Exact values under explicit conventions.
- Integrity that any implementation can verify, independent of the Parquet encoding.
- Additive evolution through extensions.

Non-goals:

- Mirroring any implementation's in-memory types.
- Dense reconstructions, meshes, or the images themselves (images are referenced by path).
- A write-optimized working store.

## 2. Layout

An OMVG dataset is a directory:

```text
scene.omvg/
  manifest.json
  cameras.parquet
  views.parquet
  ...
```

- The directory holds `manifest.json` and one Parquet file per table listed in the manifest.
  Table `<name>` is stored in `<name>.parquet` at the directory root, and each file holds exactly
  one table.
- A directory without a valid manifest is not an OMVG dataset.
- Readers ignore files that the manifest does not list.
- Images are not part of the dataset. `views.image` references them by a path relative to the
  image root: the dataset root unless the user supplies another (§5.2).
- Writers SHOULD publish atomically: write a temporary sibling directory with `manifest.json`
  last, flush every file, rename the directory onto a path that does not exist, and flush the
  parent directory. Readers then see either no dataset or a complete one. A writer that cannot
  flush the parent after the rename SHOULD report that the dataset is published but may not
  survive a crash, rather than that it was not written.

## 3. Manifest (`manifest.json`)

```json
{
  "omvg": "0.1-draft",
  "generator": "openmvg-io 0.1.0",
  "profiles": ["reconstruction"],
  "length_unit": "unknown",
  "extensions_used": [],
  "extensions_required": [],
  "dataset": {
    "name": "Strecha castle-P30",
    "license": "CC-BY-4.0",
    "url": "https://example.org/dataset",
    "citation": "<BibTeX or DOI>"
  },
  "tables": {
    "cameras":      { "rows": 1,   "hash": "sha256:<64 hex digits>" },
    "views":        { "rows": 2,   "hash": "sha256:<64 hex digits>" },
    "poses":        { "rows": 2,   "hash": "sha256:<64 hex digits>" },
    "points":       { "rows": 120, "hash": "sha256:<64 hex digits>" },
    "observations": { "rows": 240, "hash": "sha256:<64 hex digits>" }
  },
  "content_hash": "sha256:<64 hex digits>"
}
```

| Member | Required | Value |
|---|---|---|
| `omvg` | yes | The spec version the dataset conforms to: `"0.1-draft"`. |
| `profiles` | yes | Non-empty array of profile names (§5), sorted ascending, without duplicates. |
| `length_unit` | yes | `"unknown"` or `"meter"`. |
| `extensions_used` | yes | Array of extension names (§9), sorted ascending, without duplicates; `[]` when none. |
| `extensions_required` | yes | Subset of `extensions_used`, sorted ascending, without duplicates; `[]` when none. |
| `tables` | yes | Object mapping each table name to `{ "rows": <integer>, "hash": "sha256:<hex>" }`. |
| `content_hash` | yes | `"sha256:<hex>"`, defined in §7. |
| `generator` | no | Free text naming the software that wrote the dataset. |
| `dataset` | no | `name`, `license`, `url`, and `citation`, each an optional string. `license` is an SPDX license expression (for example `CC-BY-4.0 AND LicenseRef-flickr-per-image`), the license of the stored values. SHOULD be present for shared data. |

Rules:

- `manifest.json` is a UTF-8 JSON object (RFC 8259), and no object in it repeats a member name.
  `NaN`, `Infinity`, and `-Infinity` are not JSON, so readers reject them even in a member they
  ignore.
- Version, profile, and length-unit names use only the ASCII characters `A-Z a-z 0-9 . _ -`.
  Extension names have the form given in §9. Table names use only `a-z 0-9 _`. Hashes are
  `sha256:` followed by 64 lowercase hexadecimal digits.
- `rows` is a JSON integer, written without a fraction or an exponent. It MUST equal the row
  count of the table's file. Readers check it.
- `length_unit` applies to every length in the dataset: `poses.center`, `points.xyz`, and
  `gt_poses.center`. With `meter`, every length is metric, and `poses`/`points` relate to
  `gt_poses` by a rigid transform. With `unknown`, each is defined up to its own similarity, so
  evaluation aligns them by a similarity. A view without a `gt_poses` row has no ground truth.
- Readers MUST ignore unknown top-level members. They MUST NOT reject a dataset because of its
  `generator` or `dataset` members.
- **Versions.** `omvg` is `MAJOR.MINOR`, and readers accept every minor version of a major
  version they implement. A minor version only adds tables, nullable columns, profiles, and
  manifest members; readers ignore the profiles they do not know and check those they know.
  Until 1.0, `omvg` MUST equal `0.1-draft` exactly, and readers MUST reject a profile name that
  §5 does not define.
- Readers MUST reject a dataset whose `extensions_required` names an extension they do not
  implement.
- Every table required by a declared profile MUST be listed. Writers list only §5 tables and
  tables defined by an extension in `extensions_used`. When `extensions_used` is empty, readers
  MUST reject a listed table that §5 does not define. Otherwise readers check a listed table they
  do not know as §9 describes; it counts toward `content_hash`.

## 4. Conventions

- **Pixels:** origin at the top-left of the image, `x` to the right, `y` down. Integer
  coordinates are pixel centers: the center of the top-left pixel is `(0, 0)`, and an image `W`
  pixels wide spans `x ∈ [−0.5, W − 0.5]`. COLMAP puts the same point at `(0.5, 0.5)`
  (Appendix B). Pixel coordinates, `width`, and `height` refer to the pixel array stored in the
  image file, decoded without applying the EXIF Orientation tag and without resizing.
- **Poses** are world-to-camera: `x_cam = R · (X_world − C)`, where `R` is a proper rotation and
  `C` is the camera center in world coordinates. The equivalent translation is `t = −R · C`.
- **Rotation matrices** are stored row-major as 9 values: element `(i, j)` is at index `3i + j`.
- **Camera frame:** right-handed, `+z` into the scene, `+x` right, `+y` down, matching the image
  axes.
- **Angles** are in radians.
- **Floating-point values** MUST be finite; NaN and ±∞ MUST NOT appear. `−0.0` is a legal value,
  and the hash distinguishes it from `+0.0` (§7).
- **Integer columns are signed** (`INT32`, `INT64`), so every Parquet consumer reads them
  without unsigned support.
  - IDs are non-negative `INT64`, unique within their table. Writers MUST reject a source ID
    above `2^63 − 1`.
  - Feature indices, image sizes, and counts are non-negative `INT32`. Writers MUST reject a
    value above `2^31 − 1` and never truncate.
- **IDs are references only.** They carry no meaning beyond identity within one dataset. Readers
  MAY renumber them; writers SHOULD use dense IDs starting at 0.
- **Strings** MUST be valid UTF-8.
- **Rotations** (`rotation` in `poses`, `gt_poses`, and `pairs`) MUST satisfy
  `‖RᵀR − I‖_F ≤ 1e-8` (Frobenius norm) and `|det R − 1| ≤ 1e-8`. **Unit directions**
  (`pairs.translation`) MUST satisfy `|‖t‖₂ − 1| ≤ 1e-8`. Both are evaluated in double precision,
  and readers MUST check them. Writers SHOULD emit values well inside these bounds (an
  orthonormalized rotation is within about `1e-15`), so rounding differences between readers
  cannot change the verdict. The rotation tolerances are those the OpenMVG-rs core accepts.

## 5. Tables

Common rules:

- Every column listed is REQUIRED in its table unless marked nullable. A nullable column MAY be
  omitted, and readers MUST treat an absent nullable column as all null. Both forms have the
  same hash (§7).
- Each column has the logical type stated; §8 maps logical types to Parquet.
- Constraints in a Notes column, such as `> 0`, `≥ 0`, or `view_first < view_second`, are rules
  like any other: readers MUST reject values that break them.
- Rows MUST be sorted by the stated key, compared column by column in ascending order, with null
  before any value. Readers MUST reject unsorted tables and duplicate keys.
- A column documented as "references `table.column`" MUST match a row of that table, and the
  referenced table MUST be present. Readers MUST reject dangling references.
- Extra columns MAY be present. Readers that do not know them ignore them. They are hashed as
  described in §7.3.
- Column names, including those of extra and extension columns, use only the ASCII characters
  `A-Z a-z 0-9 _` and are unique within a table.

Profiles:

| Profile | Required tables | Optional tables |
|---|---|---|
| `reconstruction` | `cameras`, `views`, `poses`, `points`, `observations` | |
| `dataset` | `cameras`, `views`, `keypoints` | `matches`, `pairs` |
| `benchmark` | `cameras`, `views`, `gt_poses` | |

A dataset MAY declare several profiles; its required table set is the union. Any §5 table MAY
appear in any dataset whose references it satisfies.

### 5.1 `cameras`: key `id`

A `cameras` row is one set of intrinsic parameters. Views that share calibration share a row,
and a dataset with per-image calibration has one row per view.

| Column | Type | Notes |
|---|---|---|
| `id` | INT64 | camera ID |
| `model` | STRING | `PINHOLE`, `BROWN_CONRADY`, `FISHEYE`, or `SPHERICAL` (§6) |
| `width`, `height` | INT32 | image size in pixels, > 0 |
| `fx`, `fy` | DOUBLE, nullable | focal lengths in pixels |
| `cx`, `cy` | DOUBLE, nullable | principal point in pixels (§4) |
| `k1`, `k2`, `k3` | DOUBLE, nullable | radial coefficients |
| `k4` | DOUBLE, nullable | fourth fisheye coefficient |
| `p1`, `p2` | DOUBLE, nullable | tangential coefficients |
| `calibrated` | BOOLEAN, nullable | whether the parameters are a known calibration (§6) |

§6 states which columns each model uses.

### 5.2 `views`: key `id`

| Column | Type | Notes |
|---|---|---|
| `id` | INT64 | view ID |
| `camera_id` | INT64 | references `cameras.id` |
| `pose_id` | INT64 | pose identity, shared by views captured from one pose |
| `image` | STRING, nullable | path of the image file, relative to the image root (§2) |
| `image_sha256` | FIXED_SIZE_BINARY(32), nullable | SHA-256 of the image file's bytes |

- `pose_id` is an identity, not a reference: a view whose `pose_id` has no `poses` row is not
  registered. Writers without shared poses use `pose_id = id`.
- `image` is a POSIX path with `/` separators. It MUST NOT be empty, start with `/`, contain a
  backslash, or have an empty, `.`, or `..` component. A file name such as `a..JPG` is legal.
  `image` SHOULD be in Unicode Normalization Form C.
- `image` is the view's identity across datasets (§5.9): non-null values MUST be unique. In a
  dataset that declares `benchmark`, every view with a `gt_poses` row MUST have an `image`.

### 5.3 `poses`: key `id`

| Column | Type | Notes |
|---|---|---|
| `id` | INT64 | referenced by at least one `views.pose_id` |
| `rotation` | FIXED_SIZE_LIST\<DOUBLE, 9\> | `R`, row-major (§4) |
| `center` | FIXED_SIZE_LIST\<DOUBLE, 3\> | `C`, in `length_unit` |

### 5.4 `points`: key `id`

| Column | Type | Notes |
|---|---|---|
| `id` | INT64 | point ID |
| `xyz` | FIXED_SIZE_LIST\<DOUBLE, 3\> | world coordinates, in `length_unit` |
| `color` | FIXED_SIZE_BINARY(3), nullable | 8-bit sRGB, in R, G, B order |
| `error_px` | DOUBLE, nullable | mean reprojection error in pixels, ≥ 0 |

`error_px` is the mean, over the point's `observations` rows, of the Euclidean distance between
the observed pixel and the projection of `xyz` through the observing view's pose and camera. It
is computed by the writer; readers do not check it.

### 5.5 `observations`: key `(point_id, view_id)`

| Column | Type | Notes |
|---|---|---|
| `point_id` | INT64 | references `points.id` |
| `view_id` | INT64 | references `views.id`; the view's `pose_id` MUST have a `poses` row |
| `feature` | INT32, nullable | references `keypoints (view_id, feature)` when that table is present |
| `x`, `y` | DOUBLE | observed pixel (§4) |

When `keypoints` is present, `feature` MUST be non-null, and `x`, `y` MUST equal that keypoint's
`x`, `y` bitwise, so the file has one source of truth for each pixel. Without `keypoints`,
`feature` MAY carry the producer's own feature index.

### 5.6 `keypoints`: key `(view_id, feature)`

| Column | Type | Notes |
|---|---|---|
| `view_id` | INT64 | references `views.id` |
| `feature` | INT32 | dense per view: a view's `k`-th row, counting from 0, has `feature = k` |
| `x`, `y` | DOUBLE | pixel (§4) |
| `scale` | FLOAT, nullable | detection scale in pixels, > 0 |
| `orientation` | FLOAT, nullable | dominant direction in radians, measured from `+x` toward `+y` |

`scale` is the scale the detector reports, in pixels (§4); for SIFT it is the Gaussian
`σ` of the keypoint's scale-space level. A view without keypoints has no rows.

### 5.7 `matches`: key `(view_first, view_second, feature_first, feature_second)`

`matches` holds putative correspondences: the matcher output for each pair. Geometric
verification does not remove rows; it sets their `inlier` flag.

| Column | Type | Notes |
|---|---|---|
| `view_first`, `view_second` | INT64 | reference `views.id`; `view_first < view_second` |
| `feature_first` | INT32 | references `keypoints (view_first, feature)` |
| `feature_second` | INT32 | references `keypoints (view_second, feature)` |
| `distance` | FLOAT, nullable | descriptor distance in the matcher's metric, ≥ 0, lower is better |
| `inlier` | BOOLEAN, nullable | `true` if the pair's verified model kept the match; null if the pair was not verified |

- Within one pair, `inlier` MUST be null on every row or on none.
- A writer that only has verified correspondences writes them with `inlier = true`.
- Readers that want verified correspondences select `inlier = true`.
- A pair MAY have non-null `inlier` flags without a `pairs` row.

### 5.8 `pairs`: key `(view_first, view_second)`

| Column | Type | Notes |
|---|---|---|
| `view_first`, `view_second` | INT64 | reference `views.id`; `view_first < view_second` |
| `model` | STRING, nullable | `essential`, `fundamental`, `homography`, or `rotation` |
| `matrix` | FIXED_SIZE_LIST\<DOUBLE, 9\>, nullable | row-major; not all zero, defined up to a non-zero scale |
| `rotation` | FIXED_SIZE_LIST\<DOUBLE, 9\>, nullable | relative rotation `R`, row-major |
| `translation` | FIXED_SIZE_LIST\<DOUBLE, 3\>, nullable | unit translation direction `t` |

The relative pose maps first-camera coordinates to second-camera coordinates:
`x_second = R · x_first + s · t`, with an unknown scale `s > 0`. `F` and `H` act on homogeneous
observed pixel coordinates (§4), as stored in `keypoints` (distortion not removed). `E` acts on
unit bearings obtained by unprojecting pixels through the view's camera (Appendix A).

Under lens distortion, no exact `F` or `H` exists in raw pixel coordinates. A stored `F` or `H`
is then the empirical fit that verification used to select inliers, not an exact two-view model:
it documents how `matches.inlier` was decided, and evaluation SHOULD NOT score geometry against
it. An exact relative model is `essential` on unit bearings, or the `rotation`/`translation` pose.

| `model` | `matrix` | `rotation` and `translation` | Meaning |
|---|---|---|---|
| `essential` | required | both or neither | `x̂_secondᵀ E x̂_first = 0`; with a pose, `E` SHOULD be proportional to `[t]× R` |
| `fundamental` | required | both or neither | `x_secondᵀ F x_first = 0` |
| `homography` | required | both or neither | `x_second ∼ H x_first` |
| `rotation` | null | `rotation` required, `translation` null | relative rotation without a translation estimate |
| null | null | both null | verified pair without a usable two-view model |

- Readers derive a pair's verified match count from `matches` (`inlier = true` rows).
- A null `model` records a pair that verification examined but could not describe by a two-view
  model; for example, COLMAP rejects pairs whose motion is a watermark (Appendix B). Converters
  MAY keep the source label in an extra column.
- Model diagnostics such as residual statistics, inlier thresholds, or solver names are not part
  of 0.1. Writers MAY store them in extra columns.

### 5.9 `gt_poses`: key `view_id`

| Column | Type | Notes |
|---|---|---|
| `view_id` | INT64 | references `views.id` |
| `rotation` | FIXED_SIZE_LIST\<DOUBLE, 9\> | `R`, row-major (§4) |
| `center` | FIXED_SIZE_LIST\<DOUBLE, 3\> | `C`, in `length_unit` |

Ground truth is keyed by view rather than by pose, because it is measured per image: views that
share a `pose_id` MAY have distinct ground-truth poses.

**Evaluation.** Ground truth ships as its own `benchmark` dataset, whose `cameras` hold the
ground-truth intrinsics. A prediction lists every input view, registered or not, and views of
the two datasets match by `image`. Evaluation aligns the prediction to the ground truth by a
similarity, or by a rigid transform when both datasets use `meter`.

## 6. Camera models

| `model` | `fx`, `fy`, `cx`, `cy` | `k1`, `k2`, `k3` | `k4` | `p1`, `p2` |
|---|---|---|---|---|
| `PINHOLE` | required | null | null | null |
| `BROWN_CONRADY` | required | radial terms, a prefix | null | tangential terms, a prefix |
| `FISHEYE` | required | fisheye terms, a prefix that continues into `k4` | fourth fisheye term | null |
| `SPHERICAL` | null | null | null | null |

- **Prefix rule.** The present coefficients form a prefix of their sequence: `k2` requires `k1`,
  `k3` requires `k2`, `k4` requires `k3`, and `p2` requires `p1`. The radial and tangential
  prefixes of `BROWN_CONRADY` are independent, and either may be empty.
- **Null and zero differ.** A null coefficient is not part of the model; `0.0` is a modeled term
  whose current value is zero. Self-calibration refines modeled terms and leaves absent ones
  out, so the distinction is part of the data.
- **Calibrated cameras.** `calibrated` is `true` when the parameters come from a calibration
  that a reconstruction SHOULD keep fixed, `false` when they are an estimate or a starting guess
  that it may refine, and null when the producer does not know. It applies to every model.
- `fx` and `fy` MUST be positive.
- `SPHERICAL` is the equirectangular model: `width` MUST be even and `height` MUST equal
  `width / 2`.
- `FISHEYE` projects a ray only while its angle `θ` from the optical axis stays on the first
  increasing branch of the distortion polynomial: `θ ≤ θ_max`, where `θ_max` is the first zero of
  `dθ_d/dθ` in `(0, π)`, or `π` if there is none. Readers compute `θ_max`; it is not stored.
- `BROWN_CONRADY` coefficients mean what they mean in OpenCV's `distCoeffs` `(k1, k2, p1, p2, k3)`,
  with OpenCV's tangential formula. `FISHEYE` with four coefficients is OpenCV's equidistant
  fisheye model (COLMAP `OPENCV_FISHEYE`), extended to rays beyond 90°.
- Readers MUST reject an unknown `model` value, a non-null column where this section requires
  null, a null column where it requires a value, and a coefficient set that breaks the prefix
  rule.
- New models arrive in a later spec version or a required extension, and new parameters arrive
  as new nullable columns. Multi-sensor rigs are an extension (§9).

The projection equations are normative and given in Appendix A.

## 7. Integrity

Each table's hash is computed from its values, independent of the Parquet encoding, the codec,
the row-group layout, and the order of columns in the file. Hashes are defined for a manifest
that meets §3 and for a table whose column names, Parquet types, and fixed-size list lengths meet
§4, §5, and §8. Readers check those rules before they compare hashes.

### 7.1 Logical types and canonical bytes

| Logical type | Canonical bytes of one value |
|---|---|
| `INT32`, `INT64` | little-endian two's complement, 4 or 8 bytes |
| `FLOAT`, `DOUBLE` | little-endian IEEE 754 bits, 4 or 8 bytes |
| `BOOLEAN` | one byte, `0` or `1` |
| `STRING`, `BINARY` | `u64` little-endian byte length, then the bytes (UTF-8 for `STRING`) |
| `FIXED_SIZE_BINARY(n)` | the `n` bytes |
| `LIST<T>` | `u64` little-endian element count, then the elements' canonical bytes |
| `FIXED_SIZE_LIST<T, n>` | as `LIST<T>` (§7.3) |

List elements are `INT32`, `INT64`, `FLOAT`, `DOUBLE`, or `BOOLEAN`, and MUST NOT be null. Every
column of a dataset, including extra and extension columns, has one of these types.

### 7.2 Column hash

The column buffer starts with the row count as a `u64` little-endian and a mode byte: `1` when
the column contains at least one null, else `0`. The canonical bytes of each value follow,
over the rows in file order (key order, in a conforming table). In mode `1`, every value is
preceded by a validity byte: `1` followed by the value's bytes when present, or the single byte
`0` when null. The row count and the mode make the buffer decodable, so two different columns
never share a buffer.

```text
DOUBLE [1.5]        01 00 00 00 00 00 00 00 | 00 | 00 00 00 00 00 00 f8 3f
INT32  [7, null]    02 00 00 00 00 00 00 00 | 01 | 01 07 00 00 00 | 00
```

`column_hash = SHA-256(column buffer)`.

### 7.3 Table hash

```text
table_hash = SHA-256( u64 little-endian row count
                      ‖ for each hashed column, in ascending bytewise order of its UTF-8 name:
                        UTF-8(name) ‖ 0x00 ‖ ASCII(type) ‖ 0x00 ‖ column_hash )
```

- Every column in the file is hashed, whether §5, extra, or extension, except a column with no
  non-null value, which is skipped. Omitting an all-null column (§5) therefore leaves the hash
  unchanged. The row count comes first, so a table without a hashed column, an empty table for
  example, still hashes its number of rows.
- `type` is the logical type of the column's Parquet type alone (§8): `INT32`, `INT64`, `FLOAT`,
  `DOUBLE`, `BOOLEAN`, `STRING`, `BINARY`, `FIXED_SIZE_BINARY(n)` with `n` in decimal, or
  `LIST<T>` with `T` a scalar name. A fixed-size list is stored as a `LIST` and hashed as
  `LIST<T>`, with the element count before each value. The hash therefore needs no schema.

### 7.4 Content hash

`manifest_core` is the exact byte string

```text
{"omvg":<omvg>,"profiles":[<profile>,...],"length_unit":<unit>,"extensions_used":[<name>,...],"extensions_required":[<name>,...]}
```

with no whitespace, members in this order, and each string written as `"`, its characters, and
`"`. The character rules of §3 leave nothing to escape. For example:

```text
{"omvg":"0.1-draft","profiles":["reconstruction"],"length_unit":"unknown","extensions_used":[],"extensions_required":[]}
```

```text
content_hash = SHA-256( SHA-256(manifest_core)
                        ‖ for each listed table, in ascending bytewise name order:
                          UTF-8(name) ‖ 0x00 ‖ table_hash )
```

The `generator` and `dataset` members are descriptive and outside the hash.

### 7.5 Recording and verification

- Writers MUST record `tables.<name>.hash = "sha256:" + hex(table_hash)` and
  `content_hash = "sha256:" + hex(content_hash)`, in lowercase hexadecimal.
- Readers MUST verify `content_hash` from the manifest's table hashes, and MUST verify a table's
  hash before trusting its values. They MAY defer verification of tables they do not load.
- For a fixed-width column without nulls, the column hash is one line, for example with numpy:
  `hashlib.sha256(struct.pack("<QB", len(col), 0) + np.ascontiguousarray(col, dtype="<f8").tobytes())`.

## 8. Parquet encoding

Rules for writers are normative; readers MUST accept any valid Parquet file that maps to the §5
logical types.

| Logical type | Parquet type |
|---|---|
| `INT32`, `INT64` | `INT32`, `INT64`, unannotated or annotated as a signed integer of the same width |
| `FLOAT`, `DOUBLE` | `FLOAT`, `DOUBLE`, unannotated |
| `BOOLEAN` | `BOOLEAN`, unannotated |
| `STRING` | `BYTE_ARRAY` annotated `STRING` (converted type `UTF8`) |
| `BINARY` | `BYTE_ARRAY`, unannotated |
| `FIXED_SIZE_BINARY(n)` | `FIXED_LEN_BYTE_ARRAY(n)`, unannotated |
| `FIXED_SIZE_LIST<T, n>` | three-level `LIST` of `T`, with exactly `n` elements in every non-null value |
| `LIST<T>` | three-level `LIST` of `T` |

- **Columns.** Every column is a top-level primitive or a three-level `LIST` of a primitive.
  Readers MUST reject any other annotation (an unsigned integer, `DATE`, `DECIMAL`, `JSON`, and
  so on), groups other than a `LIST`, and repeated fields outside a `LIST`.
- **Fixed-size lists.** Parquet has no fixed-size list type, so writers store them as `LIST`, and
  readers MUST check every length.
- **Arrow schema.** The Parquet schema alone defines a column's type. A writer that stores
  `ARROW:schema` MUST keep it consistent with the Parquet schema and MUST declare fixed-size
  lists in it as Arrow lists, not as `FixedSizeList`: pyarrow 14 to 25 cannot read back a
  nullable `FixedSizeList` column that contains a null.
- **Nullability.** A required §5 column SHOULD be written with Parquet repetition `REQUIRED`;
  readers check for nulls whatever the repetition.
- **Row counts.** Readers MUST check that the row groups hold the footer's row count and that
  every column decodes to the manifest's `rows`. They MUST reject a footer whose element counts
  exceed its size before reserving memory for them.
- **Codec.** Writers MUST use Snappy or no compression; readers MUST support both and MUST reject
  any other codec. One small codec set keeps pure-Rust and WASM readers possible.
- **Encodings.** For every leaf column, including list elements, writers SHOULD use:
  - `DELTA_BINARY_PACKED` for `INT32` and `INT64`;
  - `BYTE_STREAM_SPLIT` for `FLOAT` and `DOUBLE`;
  - `RLE` for `BOOLEAN`;
  - dictionary encoding for `STRING`, and for no other type;
  - plain encoding for everything else.

  Snappy finds repeated byte runs but cannot model numbers. These encodings turn sorted keys
  into small deltas and group the bytes of floating-point values, so Snappy can compress them
  (Appendix C).
- **Layout.** Writers SHOULD use row groups of about 131,072 rows and data page version 2.
- **Key-value metadata.** Writers MAY record `omvg.table = <table name>`; readers ignore it.
- **Statistics.** Writers SHOULD write min/max statistics for ID columns. Bloom filters are not
  used.
- Byte-identical files are not required; integrity is defined by §7.

## 9. Extensions

- Extension names have the form `OMVG_<name>`, where `<name>` uses `A-Z a-z 0-9 _`.
- An extension may add tables, add columns to §5 tables, or change semantics. Its definition MUST
  give the schema of each table and column it adds in the §5 form, with types from §7.1, and MUST
  NOT reuse a §5 table name.
- §7.3 hashes a table from its file alone, so every reader computes the same hashes whether or
  not it implements an extension. A reader checks every listed table: one that §5 does not
  define and the reader does not implement gets the file and value checks of §4 and §8 (column
  names and types, codec, row count, non-finite floats, null list elements, UTF-8) and hash
  verification, but no key, reference, or range rule.
- Every table belongs to §5 or to an extension in `extensions_used`. A reader that implements
  none of the used extensions cannot tell which tables they own, so it applies the checks above
  to every unknown table; with no extension used, an unknown table is an error (§3).
- An extension MUST be listed in `extensions_used` whenever the dataset uses it. It MUST also be
  listed in `extensions_required` when ignoring it changes how a reader interprets or evaluates
  the data: it changes the meaning of a §5 table or column, or a declared profile depends on it.
  An extension that only adds tables or columns a reader can skip is optional. Its definition
  states which case applies.

## 10. Conformance

A conforming reader:

1. validates the manifest (§3), including member types, character sets, ordering, and `rows`;
2. rejects unknown required extensions and an unknown table when no extension is used (§3, §9);
3. verifies `content_hash` and the hashes of the tables it uses (§7);
4. checks the codec, the Parquet types, the column names, and the presence of required columns
   (§5, §8), in every listed table (§9);
5. rejects unsorted tables, duplicate keys, dangling references, non-finite floats, nulls in
   required columns or list elements, fixed-size lists of the wrong length, improper rotations,
   non-unit directions, and every other MUST of §4 to §6;
6. never resolves `views.image` outside the image root and does not follow symbolic links out
   of it.

A conforming writer:

1. emits every table sorted by its key;
2. writes correct row counts and hashes;
3. follows the conventions of §4 to §6 and the type mapping of §8;
4. SHOULD publish atomically (§2) and follow the encoding recommendations of §8.

Test vectors live in [`testdata/`](testdata/):

- `valid/<vector>/` holds conforming datasets. Each vector's manifest gives its table and content
  hashes, and every implementation MUST reproduce them; `expected.json` gives its description.
  Together the vectors cover every
  table and logical type, a nullable column with some nulls, an omitted all-null column, an
  empty table, `−0.0`, a non-ASCII image path, extra columns (one with nulls, one list-typed),
  a table of an unknown extension, every pair model including null, a pair with null `inlier`
  flags, partial ground truth, and one camera of each model.
  `reconstruction_plain` stores the values of `reconstruction` uncompressed, with plain
  encodings, one-row row groups, every all-null column present, and the columns in reverse
  order; both have the same hashes.
- `invalid/<vector>/` holds datasets that break exactly one rule. A conforming reader MUST
  reject each. The vector's `expected.json` names the rule with the identifier that
  `reference/omvg.py` reports, for example `camera.prefix`; other readers need not use these
  identifiers. Each vector is written with true hashes and then edited. Outside the `hash_*`
  vectors, an edit that leaves a hash stale also breaks a rule that readers check first (§7). A
  table that §7 cannot hash records 64 zeros.
- `projection.json` lists Appendix A cases as a camera row, a camera-frame point, and the
  expected pixel, or null when the point does not project. Implementations MUST reproduce each
  pixel within `1e-9` px.

`reference/make_testdata.py` generated the vectors. The committed files are the vectors: a
rerun on another platform may change the last bits of a trigonometric value, and with it the
hashes. `python reference/omvg.py conformance testdata` runs the reference implementation over
every vector.

## 11. Open questions (resolve before 1.0)

1. Per-point visibility statistics (track length, triangulation angle): in `points` or in an
   extension.
2. Feature descriptors: a `descriptors_<kind>` table per descriptor kind, with the element type
   (`u8`, `f32`, or packed bits, including the bit order) declared in the manifest.
3. Single-file packaging for distribution.
4. Model diagnostics in `pairs` (residual statistics, inlier thresholds with their units), which
   an exact round trip of a verified view graph through the format would need.

## Appendix A. Camera projection (normative)

A camera-frame point `X = (X, Y, Z)` projects to the pixel `(u, v)` as follows. An absent (null)
coefficient enters every formula as `0`, so a model without a term projects like the same model
with that term set to zero.

**`PINHOLE`**, for `Z > 0`, with `x = X / Z` and `y = Y / Z`:

```text
u = fx · x + cx
v = fy · y + cy
```

**`BROWN_CONRADY`**, for `Z > 0`, with `x = X / Z`, `y = Y / Z`, and `r² = x² + y²`:

```text
radial = 1 + k1·r² + k2·r⁴ + k3·r⁶
x_d = x · radial + 2·p1·x·y + p2·(r² + 2·x²)
y_d = y · radial + p1·(r² + 2·y²) + 2·p2·x·y
u = fx · x_d + cx
v = fy · y_d + cy
```

**`FISHEYE`**, with `r = √(X² + Y²)` and `θ = atan2(r, Z)`, for `θ ≤ θ_max` (§6):

```text
θ_d = θ · (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
u = fx · θ_d · X / r + cx
v = fy · θ_d · Y / r + cy
```

At `r = 0`, the pixel is `(cx, cy)` when `Z > 0`, and the ray does not project when `Z < 0`.

**`SPHERICAL`**, with `W = width`, `H = height = W / 2`, longitude `λ = atan2(X, Z)`, and
latitude `φ = atan2(Y, √(X² + Z²))`:

```text
u = (λ + π) · W / (2π) − 0.5,   wrapped into [−0.5, W − 0.5) by adding or subtracting W
v = (φ + π/2) · H / π − 0.5,    in [−0.5, H − 0.5]
```

The image's left and right edges are longitude `±π`, its top and bottom edges are latitude
`∓π/2` (`+y` points down), and the forward ray `(0, 0, 1)` projects to the image center
`((W − 1) / 2, (H − 1) / 2)`.

### Test values

[`testdata/projection.json`](testdata/projection.json) is the normative list of camera rows,
camera-frame points, and expected pixels, including rays that do not project. Implementations
MUST reproduce those pixels within `1e-9` px. The `PINHOLE`, `BROWN_CONRADY`, and `FISHEYE`
values were checked against the OpenMVG-rs core camera models and an independent evaluation
of the formulas above.

## Appendix B. Conversions (informative)

### COLMAP

- **Pixels.** `x_omvg = x_colmap − 0.5` and `y_omvg = y_colmap − 0.5`, for keypoints,
  observations, and the principal point `cx`, `cy`.
- **Poses.** COLMAP stores world-to-camera `qvec = (qw, qx, qy, qz)` (Hamilton convention) and
  `tvec`. `R` is the rotation of `qvec`, and `C = −Rᵀ · tvec`.
- **Two-view matrices.** With `T = [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]`, which maps OMVG pixels
  to COLMAP pixels: `F_omvg = Tᵀ · F_colmap · T` and `H_omvg = T⁻¹ · H_colmap · T`. `E` acts on
  bearings and does not change.
- **Pair models.** `two_view_geometries.config`: 2 (calibrated) → `essential`, 3 (uncalibrated)
  → `fundamental`, 4, 5, and 6 (planar, panoramic, planar or panoramic) → `homography`, 7
  (watermark) → null. The databases of Appendix C store `E` without a relative pose, so
  `rotation` and `translation` stay null.
- **Pair IDs.** `pair_id = id1 · 2147483647 + id2`, with `id1 < id2`.
- **Tracks.** When a track sees a point twice in one image, keep one observation: `observations`
  is keyed by `(point_id, view_id)`.
- **Cameras.** After the principal-point shift:

  | COLMAP model and parameters | OMVG |
  |---|---|
  | `SIMPLE_PINHOLE` `f, cx, cy` | `PINHOLE`, `fx = fy = f` |
  | `PINHOLE` `fx, fy, cx, cy` | `PINHOLE` |
  | `SIMPLE_RADIAL` `f, cx, cy, k` | `BROWN_CONRADY`, `fx = fy = f`, `k1 = k` |
  | `RADIAL` `f, cx, cy, k1, k2` | `BROWN_CONRADY`, `fx = fy = f`, `k1`, `k2` |
  | `OPENCV` `fx, fy, cx, cy, k1, k2, p1, p2` | `BROWN_CONRADY`, `k1`, `k2`, `p1`, `p2` |
  | `FULL_OPENCV` with `k4 = k5 = k6 = 0` | `BROWN_CONRADY`, `k1`, `k2`, `k3`, `p1`, `p2` |
  | `OPENCV_FISHEYE` `fx, fy, cx, cy, k1, k2, k3, k4` | `FISHEYE`, `k1` to `k4` |
  | `SIMPLE_RADIAL_FISHEYE` `f, cx, cy, k` | `FISHEYE`, `fx = fy = f`, `k1 = k` |
  | `RADIAL_FISHEYE` `f, cx, cy, k1, k2` | `FISHEYE`, `fx = fy = f`, `k1`, `k2` |

  COLMAP's fisheye models compute `θ = atan(r)` on normalized coordinates and so cover only rays
  with `Z > 0`. COLMAP's `prior_focal_length` marks a trusted focal prior that COLMAP still
  refines, not a calibration, so `calibrated` stays null.

The pixel, pose, two-view matrix, pair model, and pair ID rules were checked on the corpus of
Appendix C.

### Ground-truth sources

- Normalize quaternions before converting them to `R`.
- Project a matrix `M` that is not exactly a rotation to `R = U · diag(1, 1, det(U·Vᵀ)) · Vᵀ`,
  where `M = U · Σ · Vᵀ`. For a world-to-camera translation `t`, compute `C = −M⁻¹ · t` from the
  source matrix before projecting it.
- COLMAP `THIN_PRISM_FISHEYE`, and `FULL_OPENCV` with a non-zero `k4`, `k5`, or `k6`, have no
  OMVG model: convert the undistorted release instead.
- BlendedMVS and MegaDepth use `length_unit = "unknown"`; ETH3D, Strecha, and Tanks and Temples
  use `"meter"`.

### Camera-to-world matrices (nerfstudio, OpenGL)

A camera-to-world matrix `[R_c2w | C]` in OpenGL camera axes (`+x` right, `+y` up, `+z` toward
the viewer) gives `R = D · R_c2wᵀ` and the same `C`, with `D = diag(1, −1, −1)`. When the matrix
already uses this spec's camera axes, `D` is the identity.

## Appendix C. Size reference (informative)

128 MegaDepth scenes stored as COLMAP databases (8.1 GB, 26,674 images) were converted with the
reference writer to the `dataset` and `benchmark` profiles, without quantization. The output
holds 381 million keypoints and 354 million putative matches with their inlier flags. Every
scene was written in each configuration below and read back: all give the same
`content_hash`, stored values are bit-exact, and derived values (centers from COLMAP's `tvec`,
shifted `F` and `H`) agree with the source within `1e-14` relative.

| | SQLite | Snappy, default encodings | Snappy, §8 encodings | zstd level 9 (not allowed) |
|---|---|---|---|---|
| Total | 8.1 GB | 6.8 GB | **2.9 GB** (2.8× smaller) | 2.3 GB |
| Keypoints | | 4.6 GB | 2.2 GB (5.6 B per keypoint) | 1.6 GB |
| Matches | | 2.2 GB | 0.76 GB (2.1 B per match) | 0.68 GB |

- With default encodings, Snappy output can be larger than the source database: it is 1.4 and
  1.5 times the database size on two of the three sub-corpora.
- On top of the §8 encodings, zstd gains 1.27× (1.15× to 1.51× per scene), mostly on keypoint
  coordinates.
- Keypoint coordinates are three quarters of the output.
- Matches are near their entropy floor, because `feature_second` is effectively random within a
  view. The `inlier` flag costs under one bit per match with `RLE`.
- Storing single-precision source coordinates as `DOUBLE` costs about 15% more than `FLOAT`
  with `BYTE_STREAM_SPLIT`, since most of the extra mantissa bytes are zero and compress away.
