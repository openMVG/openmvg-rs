#!/usr/bin/env python3
"""Reference reader, writer, and checker for the OMVG format, version 0.1-draft.

The normative text is ``docs/format/OMVG_FORMAT_SPEC.md``; section numbers in this module refer
to it. Where this module and the specification disagree, the specification wins.

The module favors clarity over speed, but every per-row rule is vectorized with numpy, so it
handles benchmark-sized tables with tens of millions of rows. It needs Python 3.9 or later,
numpy, and pyarrow 14 or later.

Command line::

    python omvg.py check DATASET          # print every rule the dataset breaks
    python omvg.py hash DATASET           # recompute table and content hashes from the files
    python omvg.py conformance TESTDATA   # run the conformance vectors (§10)

Library::

    dataset = read(path)                  # raises OmvgError when any rule is broken
    violations = check(path)              # list of Violation(rule, table, message)
    write(path, tables, profiles=["reconstruction"])

Each violation names a rule ID such as ``key.unsorted`` or ``camera.prefix``; the invalid
conformance vectors name the rule they break with the same IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = "0.1-draft"
LENGTH_UNITS = ("unknown", "meter")
ALLOWED_CODECS = ("SNAPPY", "UNCOMPRESSED")
ROTATION_TOLERANCE = 1e-8
DIRECTION_TOLERANCE = 1e-8
ROW_GROUP_SIZE = 131_072

_NAME = re.compile(r"[A-Za-z0-9._-]+\Z")
_TABLE_NAME = re.compile(r"[a-z0-9_]+\Z")
_COLUMN_NAME = re.compile(r"[A-Za-z0-9_]+\Z")
_EXTENSION_NAME = re.compile(r"OMVG_[A-Za-z0-9_]+\Z")
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")


# --------------------------------------------------------------------------------------------
# Logical types (§7.1) and the table schema (§5)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LType:
    """A logical type of §7.1. ``size`` is the fixed width or length; ``element`` the list
    element type."""

    kind: str
    size: int = 0
    element: str = ""

    def __str__(self) -> str:
        if self.kind == "FIXED_SIZE_BINARY":
            return f"FIXED_SIZE_BINARY({self.size})"
        if self.kind == "FIXED_SIZE_LIST":
            return f"FIXED_SIZE_LIST<{self.element}, {self.size}>"
        if self.kind == "LIST":
            return f"LIST<{self.element}>"
        return self.kind

    @property
    def is_list(self) -> bool:
        return self.kind in ("FIXED_SIZE_LIST", "LIST")

    @property
    def is_floating(self) -> bool:
        return self.kind in ("FLOAT", "DOUBLE") or (
            self.is_list and self.element in ("FLOAT", "DOUBLE")
        )


# Scalar kind -> (Arrow type, numpy little-endian dtype, canonical byte width).
_SCALARS = {
    "INT32": (pa.int32(), "<i4", 4),
    "INT64": (pa.int64(), "<i8", 8),
    "FLOAT": (pa.float32(), "<f4", 4),
    "DOUBLE": (pa.float64(), "<f8", 8),
    "BOOLEAN": (pa.bool_(), "u1", 1),
}

INT32 = LType("INT32")
INT64 = LType("INT64")
FLOAT = LType("FLOAT")
DOUBLE = LType("DOUBLE")
BOOLEAN = LType("BOOLEAN")
STRING = LType("STRING")
BINARY = LType("BINARY")


def fixed_size_binary(n: int) -> LType:
    return LType("FIXED_SIZE_BINARY", size=n)


def fixed_size_list(element: LType, n: int) -> LType:
    return LType("FIXED_SIZE_LIST", size=n, element=element.kind)


def list_of(element: LType) -> LType:
    return LType("LIST", element=element.kind)


def arrow_type(lt: LType) -> pa.DataType:
    """The Arrow type that holds a column of logical type ``lt`` in memory.

    Fixed-size lists are plain Arrow lists, as §8 requires on disk: pyarrow cannot read back a
    nullable ``FixedSizeList`` column that contains a null.
    """
    if lt.kind in _SCALARS:
        return _SCALARS[lt.kind][0]
    if lt.kind == "STRING":
        return pa.string()
    if lt.kind == "BINARY":
        return pa.binary()
    if lt.kind == "FIXED_SIZE_BINARY":
        return pa.binary(lt.size)
    return pa.list_(pa.field("element", _SCALARS[lt.element][0], nullable=False))


@dataclass(frozen=True)
class Column:
    """One §5 column. ``minimum`` is ``(">=", v)`` or ``(">", v)`` when the column has a bound."""

    name: str
    ltype: LType
    nullable: bool = False
    minimum: tuple[str, float] | None = None


@dataclass(frozen=True)
class TableSpec:
    key: tuple[str, ...]
    columns: tuple[Column, ...]

    def column(self, name: str) -> Column | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None


_ID = (">=", 0)
_POSITIVE = (">", 0)
_ROTATION = fixed_size_list(DOUBLE, 9)
_VECTOR = fixed_size_list(DOUBLE, 3)

SCHEMA: dict[str, TableSpec] = {
    "cameras": TableSpec(
        ("id",),
        (
            Column("id", INT64, minimum=_ID),
            Column("model", STRING),
            Column("width", INT32, minimum=_POSITIVE),
            Column("height", INT32, minimum=_POSITIVE),
            Column("fx", DOUBLE, True, _POSITIVE),
            Column("fy", DOUBLE, True, _POSITIVE),
            Column("cx", DOUBLE, True),
            Column("cy", DOUBLE, True),
            Column("k1", DOUBLE, True),
            Column("k2", DOUBLE, True),
            Column("k3", DOUBLE, True),
            Column("k4", DOUBLE, True),
            Column("p1", DOUBLE, True),
            Column("p2", DOUBLE, True),
            Column("calibrated", BOOLEAN, True),
        ),
    ),
    "views": TableSpec(
        ("id",),
        (
            Column("id", INT64, minimum=_ID),
            Column("camera_id", INT64, minimum=_ID),
            Column("pose_id", INT64, minimum=_ID),
            Column("image", STRING, True),
            Column("image_sha256", fixed_size_binary(32), True),
        ),
    ),
    "poses": TableSpec(
        ("id",),
        (
            Column("id", INT64, minimum=_ID),
            Column("rotation", _ROTATION),
            Column("center", _VECTOR),
        ),
    ),
    "points": TableSpec(
        ("id",),
        (
            Column("id", INT64, minimum=_ID),
            Column("xyz", _VECTOR),
            Column("color", fixed_size_binary(3), True),
            Column("error_px", DOUBLE, True, (">=", 0)),
        ),
    ),
    "observations": TableSpec(
        ("point_id", "view_id"),
        (
            Column("point_id", INT64, minimum=_ID),
            Column("view_id", INT64, minimum=_ID),
            Column("feature", INT32, True, _ID),
            Column("x", DOUBLE),
            Column("y", DOUBLE),
        ),
    ),
    "keypoints": TableSpec(
        ("view_id", "feature"),
        (
            Column("view_id", INT64, minimum=_ID),
            Column("feature", INT32, minimum=_ID),
            Column("x", DOUBLE),
            Column("y", DOUBLE),
            Column("scale", FLOAT, True, _POSITIVE),
            Column("orientation", FLOAT, True),
        ),
    ),
    "matches": TableSpec(
        ("view_first", "view_second", "feature_first", "feature_second"),
        (
            Column("view_first", INT64, minimum=_ID),
            Column("view_second", INT64, minimum=_ID),
            Column("feature_first", INT32, minimum=_ID),
            Column("feature_second", INT32, minimum=_ID),
            Column("distance", FLOAT, True, (">=", 0)),
            Column("inlier", BOOLEAN, True),
        ),
    ),
    "pairs": TableSpec(
        ("view_first", "view_second"),
        (
            Column("view_first", INT64, minimum=_ID),
            Column("view_second", INT64, minimum=_ID),
            Column("model", STRING, True),
            Column("matrix", fixed_size_list(DOUBLE, 9), True),
            Column("rotation", _ROTATION, True),
            Column("translation", _VECTOR, True),
        ),
    ),
    "gt_poses": TableSpec(
        ("view_id",),
        (
            Column("view_id", INT64, minimum=_ID),
            Column("rotation", _ROTATION),
            Column("center", _VECTOR),
        ),
    ),
}

# A table §5 does not define: every column is an extra column, and there is no key.
_EXTENSION_TABLE = TableSpec((), ())


def _spec(name: str) -> TableSpec:
    return SCHEMA.get(name, _EXTENSION_TABLE)


PROFILES: dict[str, tuple[str, ...]] = {
    "reconstruction": ("cameras", "views", "poses", "points", "observations"),
    "dataset": ("cameras", "views", "keypoints"),
    "benchmark": ("cameras", "views", "gt_poses"),
}

CAMERA_MODELS = ("PINHOLE", "BROWN_CONRADY", "FISHEYE", "SPHERICAL")
PAIR_MODELS = ("essential", "fundamental", "homography", "rotation")

# Columns that get min/max statistics (§8).
_ID_COLUMNS = frozenset(
    {"id", "camera_id", "pose_id", "point_id", "view_id", "view_first", "view_second"}
)


# --------------------------------------------------------------------------------------------
# Errors and results
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    rule: str
    table: str | None
    message: str

    def __str__(self) -> str:
        where = f"{self.table}: " if self.table else ""
        return f"[{self.rule}] {where}{self.message}"


class OmvgError(ValueError):
    """Raised by ``read``, ``write``, and ``compute_hashes`` when data breaks a rule of the
    specification."""

    def __init__(self, violations: Sequence[Violation]):
        self.violations = list(violations)
        super().__init__("\n".join(str(v) for v in self.violations))


@dataclass
class Dataset:
    """A dataset read by ``read``: the manifest and every table it lists.

    Each §5 table holds every §5 column of its schema (absent nullable columns as all-null
    columns) followed by its extra columns; any other table holds its columns as stored.
    """

    manifest: dict
    tables: dict[str, pa.Table] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Hashing (§7)
# --------------------------------------------------------------------------------------------


def _array(column: pa.ChunkedArray | pa.Array) -> pa.Array:
    """Return a column as one contiguous Arrow array."""
    if isinstance(column, pa.Array):
        return column
    if column.num_chunks == 1:
        return column.chunk(0)
    if column.num_chunks == 0:
        return pa.array([], type=column.type)
    return pa.concat_arrays(column.chunks)


def _valid_mask(arr: pa.Array) -> np.ndarray:
    if arr.null_count == 0:
        return np.ones(len(arr), dtype=bool)
    return ~np.asarray(arr.is_null().to_numpy(zero_copy_only=False), dtype=bool)


def _scalar_bytes(arr: pa.Array, kind: str) -> np.ndarray:
    """Canonical bytes of a scalar or list-element array, one row per value (nulls zeroed)."""
    _, dtype, width = _SCALARS[kind]
    if arr.null_count:
        arr = arr.fill_null(False if kind == "BOOLEAN" else 0)
    values = np.asarray(arr.to_numpy(zero_copy_only=False)).astype(dtype, copy=False)
    return np.ascontiguousarray(values).view(np.uint8).reshape(len(arr), width)


def _fixed_width_bytes(arr: pa.Array, lt: LType) -> np.ndarray:
    """Canonical bytes of a scalar or fixed-size binary column as an ``(n, width)`` array."""
    n = len(arr)
    if lt.kind in _SCALARS:
        return _scalar_bytes(arr, lt.kind)
    width = lt.size
    buffer = arr.buffers()[1]
    if n == 0 or buffer is None:
        return np.zeros((n, width), dtype=np.uint8)
    data = np.frombuffer(buffer, dtype=np.uint8)
    return data[arr.offset * width : (arr.offset + n) * width].reshape(n, width)


def _gather(data: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate ``data[starts[i] : starts[i] + lengths[i]]`` over ``i``."""
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.uint8)
    out_starts = np.cumsum(lengths) - lengths
    index = np.repeat(starts - out_starts, lengths) + np.arange(total)
    return data[index]


def _interleave(
    heads: np.ndarray, head_lengths: np.ndarray, body: np.ndarray, body_lengths: np.ndarray
) -> np.ndarray:
    """Concatenate, row by row, a head segment and a body segment."""
    sizes = head_lengths + body_lengths
    total = int(sizes.sum())
    out = np.empty(total, dtype=np.uint8)
    starts = np.cumsum(sizes) - sizes
    for segment, lengths, offset in ((heads, head_lengths, 0), (body, body_lengths, head_lengths)):
        count = int(lengths.sum())
        if count:
            within = np.cumsum(lengths) - lengths
            out[np.repeat(starts + offset - within, lengths) + np.arange(count)] = segment
    return out


def column_buffer(arr: pa.Array, lt: LType) -> bytes:
    """The column buffer of §7.2 for one column of hash type ``lt`` (never a fixed-size list)."""
    nullable = arr.null_count > 0
    return struct.pack("<QB", len(arr), nullable) + _column_values(arr, lt)


def _column_values(arr: pa.Array, lt: LType) -> bytes:
    """The rows of a column buffer, after its row count and mode."""
    n = len(arr)
    valid = _valid_mask(arr)
    nullable = arr.null_count > 0
    if lt.kind in ("STRING", "BINARY", "LIST"):
        if lt.kind == "LIST":
            offsets = np.asarray(arr.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
            counts = np.where(valid, offsets[1:] - offsets[:-1], 0)
            width = _SCALARS[lt.element][2]
            values = _scalar_bytes(arr.values, lt.element).reshape(-1)
            body = _gather(values, offsets[:-1] * width, counts * width)
            body_lengths = counts * width
        else:
            buffers = arr.buffers()
            if n:
                offsets = np.frombuffer(
                    buffers[1], dtype="<i4", count=n + 1, offset=arr.offset * 4
                ).astype(np.int64)
            else:
                offsets = np.zeros(1, dtype=np.int64)
            data = (
                np.frombuffer(buffers[2], dtype=np.uint8)
                if len(buffers) > 2 and buffers[2] is not None
                else np.zeros(0, np.uint8)
            )
            counts = np.where(valid, offsets[1:] - offsets[:-1], 0)
            body = _gather(data, offsets[:-1], counts)
            body_lengths = counts
        # Head of each row: validity byte (when the column has a null) then u64 length (when present).
        heads = np.empty((n, 9), dtype=np.uint8)
        heads[:, 0] = valid
        heads[:, 1:] = np.ascontiguousarray(counts.astype("<u8")).view(np.uint8).reshape(n, 8)
        keep = np.empty((n, 9), dtype=bool)
        keep[:, 0] = nullable
        keep[:, 1:] = valid[:, None]
        return _interleave(heads[keep], keep.sum(axis=1), body, body_lengths).tobytes()
    values = _fixed_width_bytes(arr, lt)
    if not nullable:
        return np.ascontiguousarray(values).tobytes()
    rows = np.empty((n, 1 + values.shape[1]), dtype=np.uint8)
    rows[:, 0] = valid
    rows[:, 1:] = values
    keep = np.ones(rows.shape, dtype=bool)
    keep[:, 1:] = valid[:, None]
    return rows[keep].tobytes()


def table_hash(name: str, table: pa.Table) -> bytes:
    """The table hash of §7.3 for a canonical table (see ``Dataset``).

    Every column is hashed in bytewise name order under the logical type of its Parquet type,
    so the hash does not depend on §5; a column with no non-null value is skipped. The row
    count comes first, so tables of different lengths never share a hash.
    """
    outer = hashlib.sha256(struct.pack("<Q", table.num_rows))
    for column in sorted(table.column_names, key=lambda s: s.encode()):
        arr = _array(table.column(column))
        if arr.null_count == len(arr):
            continue  # absent and all-null columns hash alike
        lt = ltype_of_arrow(arr.type)
        if lt is None:
            raise OmvgError(
                [
                    Violation(
                        "parquet.type", name, f"column {column!r} has unsupported type {arr.type}"
                    )
                ]
            )
        digest = hashlib.sha256(column_buffer(arr, lt)).digest()
        outer.update(column.encode() + b"\x00" + str(lt).encode("ascii") + b"\x00" + digest)
    return outer.digest()


def manifest_core(
    version: str,
    profiles: Sequence[str],
    length_unit: str,
    used: Sequence[str],
    required: Sequence[str],
) -> bytes:
    """The ``manifest_core`` byte string of §7.4."""

    def strings(values: Sequence[str]) -> str:
        return ",".join(f'"{v}"' for v in values)

    return (
        f'{{"omvg":"{version}","profiles":[{strings(profiles)}],"length_unit":"{length_unit}",'
        f'"extensions_used":[{strings(used)}],"extensions_required":[{strings(required)}]}}'
    ).encode()


def content_hash(core: bytes, table_hashes: Mapping[str, bytes]) -> bytes:
    """The content hash of §7.4 from ``manifest_core`` and the raw table hashes."""
    h = hashlib.sha256(hashlib.sha256(core).digest())
    for name in sorted(table_hashes, key=lambda s: s.encode()):
        h.update(name.encode() + b"\x00" + table_hashes[name])
    return h.digest()


def _hex(digest: bytes) -> str:
    return "sha256:" + digest.hex()


# --------------------------------------------------------------------------------------------
# Type mapping (§8)
# --------------------------------------------------------------------------------------------


def ltype_of_arrow(t: pa.DataType) -> LType | None:
    """The logical type of a column held in Arrow type ``t``, or None if not allowed.

    A dictionary takes the type of its values. A ``FixedSizeList`` is ``LIST<T>``, since every
    list column is stored and hashed as ``LIST<T>`` (§7.3, §8).
    """
    if pa.types.is_dictionary(t):
        return ltype_of_arrow(t.value_type)
    for kind, (scalar, _, _) in _SCALARS.items():
        if t == scalar:
            return LType(kind)
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return STRING
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return BINARY
    if pa.types.is_fixed_size_binary(t):
        return fixed_size_binary(t.byte_width)
    is_list = pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)
    if is_list and t.value_type in [scalar for scalar, _, _ in _SCALARS.values()]:
        return list_of(ltype_of_arrow(t.value_type))
    return None


def _column_of(leaf) -> str:
    """The top-level column that a Parquet leaf belongs to.

    A top-level primitive column is its own leaf, so its path equals its name, dots included.
    A three-level ``LIST`` column has the leaf ``<column>.<list>.<element>``. Any other leaf is
    nested in a group the format does not allow, and is attributed to its outermost group.
    """
    if leaf.path == leaf.name:
        return leaf.path
    parts = leaf.path.split(".")
    if leaf.max_repetition_level == 1 and len(parts) >= 3:
        return ".".join(parts[:-2])
    return parts[0]


def _is_column_leaf(leaf, column: str) -> bool:
    """Whether ``leaf`` alone forms the column ``column``: a top-level primitive, or the
    element of a three-level ``LIST``."""
    if leaf.path == leaf.name:
        return leaf.path == column and leaf.max_repetition_level == 0
    return (
        leaf.max_repetition_level == 1
        and leaf.path.count(".") == column.count(".") + 2
        and leaf.path.startswith(column + ".")
    )


def _ltype_of_parquet(column: str, leaves: Sequence) -> LType | None:
    """The logical type of a Parquet column from its leaf schema, or None if not allowed."""
    if len(leaves) != 1 or not _is_column_leaf(leaves[0], column):
        return None
    c = leaves[0]
    physical, logical, converted = c.physical_type, str(c.logical_type), str(c.converted_type)
    kind: LType | None = None
    if (
        physical == "INT32"
        and logical in ("None", "Int(bitWidth=32, isSigned=true)")
        and converted in ("NONE", "INT_32")
    ):
        kind = INT32
    elif (
        physical == "INT64"
        and logical in ("None", "Int(bitWidth=64, isSigned=true)")
        and converted in ("NONE", "INT_64")
    ):
        kind = INT64
    elif physical in ("FLOAT", "DOUBLE", "BOOLEAN") and logical == "None" and converted == "NONE":
        kind = LType(physical)
    elif physical == "BYTE_ARRAY" and (
        logical == "String" or (logical == "None" and converted == "UTF8")
    ):
        kind = STRING
    elif physical == "BYTE_ARRAY" and logical == "None" and converted == "NONE":
        kind = BINARY
    elif physical == "FIXED_LEN_BYTE_ARRAY" and logical == "None" and converted == "NONE":
        kind = fixed_size_binary(c.length)
    if kind is None:
        return None
    if c.max_repetition_level == 1:
        return list_of(kind) if kind.kind in _SCALARS else None
    return kind


def _accepts(expected: LType, found: LType) -> bool:
    """Whether a Parquet column of logical type ``found`` stores a §5 column of ``expected``."""
    if expected.kind == "FIXED_SIZE_LIST":
        return found.kind == "LIST" and found.element == expected.element
    return expected == found


def _canonical(arr: pa.Array, lt: LType) -> pa.Array:
    """Convert an Arrow array to the in-memory Arrow type of ``lt`` (see ``arrow_type``)."""
    if pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_decode()
    if lt.is_list:
        element = _SCALARS[lt.element][0]
        if pa.types.is_fixed_size_list(arr.type):
            width = arr.type.list_size
            offsets = (np.arange(len(arr) + 1, dtype=np.int64) + arr.offset) * width
            values = arr.values
        else:
            offsets = np.asarray(arr.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
            values = arr.values
        if values.type != element:
            values = values.cast(element)
        mask = pa.array(~_valid_mask(arr)) if arr.null_count else None
        return pa.ListArray.from_arrays(
            pa.array(offsets, pa.int32()), values, type=arrow_type(lt), mask=mask
        )
    target = arrow_type(lt)
    return arr if arr.type == target else arr.cast(target)


def _to_arrow(values, lt: LType, table: str, column: str) -> pa.Array:
    """Convert writer input (Arrow, numpy, or Python sequences) to the in-memory type of ``lt``.

    Integers out of range raise instead of truncating (§4).
    """
    try:
        if isinstance(values, (pa.Array, pa.ChunkedArray)):
            return _canonical(_array(values), lt)
        if isinstance(values, np.ndarray) and lt.is_list and values.ndim == 2:
            n, width = values.shape
            flat = pa.array(np.ascontiguousarray(values).reshape(-1)).cast(
                _SCALARS[lt.element][0], safe=True
            )
            offsets = pa.array(np.arange(n + 1, dtype=np.int64) * width, pa.int32())
            return pa.ListArray.from_arrays(offsets, flat, type=arrow_type(lt))
        if isinstance(values, np.ndarray) and lt.kind == "FIXED_SIZE_BINARY" and values.ndim == 2:
            values = [bytes(row) for row in values.astype(np.uint8)]
        if isinstance(values, np.ndarray) and values.ndim == 1:
            return pa.array(values).cast(arrow_type(lt), safe=True)
        return pa.array(list(values), type=arrow_type(lt))
    except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError, TypeError, ValueError) as error:
        raise OmvgError(
            [
                Violation(
                    "value.range", table, f"column {column!r} cannot be stored as {lt}: {error}"
                )
            ]
        ) from None


# --------------------------------------------------------------------------------------------
# Column-level checks shared by the reader and the writer (§4, §5)
# --------------------------------------------------------------------------------------------


def _first(mask: np.ndarray) -> str:
    rows = np.flatnonzero(mask)
    return f"{rows.size} row(s), first at row {rows[0]}" if rows.size else "no rows"


def _columns(name: str, table: pa.Table) -> Iterable[tuple[str, pa.Array, LType, bool]]:
    """Yield ``(column, array, logical type, required)`` for every column of a canonical table."""
    spec = _spec(name)
    for column in table.column_names:
        c = spec.column(column)
        arr = _array(table.column(column))
        if c is not None:
            yield column, arr, c.ltype, not c.nullable
        else:
            yield column, arr, ltype_of_arrow(arr.type), False


def _structural_checks(name: str, table: pa.Table) -> list[Violation]:
    """Nulls in required columns, list element nulls and lengths, non-finite floats, UTF-8."""
    out: list[Violation] = []
    for column, arr, lt, required in _columns(name, table):
        if required and arr.null_count:
            out.append(
                Violation(
                    "value.null", name, f"required column {column!r} has {arr.null_count} null(s)"
                )
            )
        if lt.is_list:
            flat = arr.flatten()
            if flat.null_count:
                out.append(
                    Violation(
                        "value.list_null",
                        name,
                        f"column {column!r} has {flat.null_count} null list element(s)",
                    )
                )
            if lt.kind == "FIXED_SIZE_LIST":
                lengths = np.asarray(
                    arr.value_lengths().fill_null(lt.size).to_numpy(zero_copy_only=False)
                )
                bad = lengths != lt.size
                if bad.any():
                    out.append(
                        Violation(
                            "value.list_length",
                            name,
                            f"column {column!r} needs {lt.size} elements: {_first(bad)}",
                        )
                    )
            values = flat.drop_null() if flat.null_count else flat
        else:
            values = arr.drop_null() if arr.null_count else arr
        if lt.is_floating:
            finite = np.isfinite(
                np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float64)
            )
            if not finite.all():
                out.append(
                    Violation(
                        "value.non_finite",
                        name,
                        f"column {column!r} has {int((~finite).sum())} non-finite value(s)",
                    )
                )
        if lt.kind == "STRING":
            try:
                arr.validate(full=True)
            except pa.ArrowInvalid as error:
                out.append(Violation("value.utf8", name, f"column {column!r}: {error}"))
    return out


# --------------------------------------------------------------------------------------------
# Table and cross-table rules (§4 to §6)
# --------------------------------------------------------------------------------------------


def _np(table: pa.Table, column: str) -> np.ndarray:
    """A numeric column as a numpy array, nulls replaced by zero so integers stay integers."""
    arr = _array(table.column(column))
    if arr.null_count:
        arr = arr.fill_null(0)
    return np.asarray(arr.to_numpy(zero_copy_only=False))


def _key_arrays(table: pa.Table, key: Sequence[str]) -> list[np.ndarray]:
    """Key columns as integer arrays; a nullable key column becomes (validity, value)."""
    arrays = []
    for column in key:
        arr = _array(table.column(column))
        if arr.null_count:
            arrays.append(_valid_mask(arr).astype(np.int8))
            arrays.append(np.asarray(arr.fill_null(0).to_numpy(zero_copy_only=False)))
        else:
            arrays.append(np.asarray(arr.to_numpy(zero_copy_only=False)))
    return arrays


def _key_order(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Per adjacent row pair: -1 when descending, 0 when equal, 1 when ascending."""
    n = len(arrays[0]) if arrays else 0
    result = np.zeros(max(n - 1, 0), dtype=np.int8)
    undecided = np.ones(max(n - 1, 0), dtype=bool)
    for a in arrays:
        step = (a[1:] > a[:-1]).astype(np.int8) - (a[1:] < a[:-1]).astype(np.int8)
        decide = undecided & (step != 0)
        result[decide] = step[decide]
        undecided &= step == 0
    return result


def _sort_order(table: pa.Table, key: Sequence[str]) -> np.ndarray:
    arrays = _key_arrays(table, key)
    return np.lexsort(arrays[::-1]) if arrays and len(arrays[0]) else np.arange(table.num_rows)


def _lookup(reference: Sequence[np.ndarray], query: Sequence[np.ndarray]) -> np.ndarray:
    """For each query row, the index of an equal row of ``reference``, or -1."""
    n_ref, n_query = len(reference[0]), len(query[0])
    ref_code = np.zeros(n_ref, dtype=np.int64)
    query_code = np.zeros(n_query, dtype=np.int64)
    found = np.ones(n_query, dtype=bool)
    for r, q in zip(reference, query, strict=True):
        unique = np.unique(r)
        if unique.size == 0:
            return np.full(n_query, -1, dtype=np.int64)
        position = np.minimum(np.searchsorted(unique, q), unique.size - 1)
        found &= unique[position] == q
        ref_code = ref_code * unique.size + np.searchsorted(unique, r)
        query_code = query_code * unique.size + np.where(found, position, 0)
    order = np.argsort(ref_code, kind="stable")
    ordered = ref_code[order]
    if ordered.size == 0:
        return np.full(n_query, -1, dtype=np.int64)
    position = np.minimum(np.searchsorted(ordered, query_code), ordered.size - 1)
    hit = found & (ordered[position] == query_code)
    return np.where(hit, order[position], -1)


def _runs(*columns: np.ndarray) -> np.ndarray:
    """Start index of every run of equal rows in sorted columns."""
    n = len(columns[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.zeros(n, dtype=bool)
    change[0] = True
    for c in columns:
        change[1:] |= c[1:] != c[:-1]
    return np.flatnonzero(change)


def _rotation_defects(flat: np.ndarray) -> np.ndarray:
    """Per rotation: whether ‖RᵀR − I‖_F or |det R − 1| exceeds the tolerance (§4)."""
    r = flat.reshape(-1, 3, 3)
    gram = np.einsum("nki,nkj->nij", r, r) - np.eye(3)
    frobenius = np.sqrt(np.einsum("nij,nij->n", gram, gram))
    det = (
        r[:, 0, 0] * (r[:, 1, 1] * r[:, 2, 2] - r[:, 1, 2] * r[:, 2, 1])
        - r[:, 0, 1] * (r[:, 1, 0] * r[:, 2, 2] - r[:, 1, 2] * r[:, 2, 0])
        + r[:, 0, 2] * (r[:, 1, 0] * r[:, 2, 1] - r[:, 1, 1] * r[:, 2, 0])
    )
    return (
        ~np.isfinite(frobenius)
        | ~np.isfinite(det)
        | (frobenius > ROTATION_TOLERANCE)
        | (np.abs(det - 1.0) > ROTATION_TOLERANCE)
    )


def _list_rows(table: pa.Table, column: str, width: int) -> tuple[np.ndarray, np.ndarray]:
    """The valid rows of a fixed-size list column as ``(row indices, (k, width) values)``."""
    arr = _array(table.column(column))
    valid = _valid_mask(arr)
    values = np.asarray(arr.flatten().to_numpy(zero_copy_only=False), dtype=np.float64).reshape(
        -1, width
    )
    return np.flatnonzero(valid), values


def _image_path_ok(path: str) -> bool:
    if not path or path.startswith("/") or "\\" in path:
        return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


def _table_rules(name: str, table: pa.Table) -> list[Violation]:
    """Rules local to one canonical table: keys, ranges, rotations, camera models, paths."""
    spec = SCHEMA[name]
    out: list[Violation] = []
    order = _key_order(_key_arrays(table, spec.key))
    if (order < 0).any():
        out.append(
            Violation(
                "key.unsorted", name, f"rows are not sorted by {spec.key}: {_first(order < 0)}"
            )
        )
    if (order == 0).any():
        out.append(
            Violation("key.duplicate", name, f"duplicate key {spec.key}: {_first(order == 0)}")
        )

    for c in spec.columns:
        if c.minimum is None:
            continue
        arr = _array(table.column(c.name))
        values = np.asarray(
            (arr.drop_null() if arr.null_count else arr).to_numpy(zero_copy_only=False)
        )
        op, bound = c.minimum
        bad = values < bound if op == ">=" else values <= bound
        if bad.any():
            out.append(
                Violation(
                    "value.range", name, f"column {c.name!r} must be {op} {bound}: {_first(bad)}"
                )
            )

    for column, width in (("rotation", 9), ("translation", 3)):
        c = spec.column(column)
        if c is None:
            continue
        _, values = _list_rows(table, column, width)
        if column == "rotation":
            bad = _rotation_defects(values)
            if bad.any():
                out.append(Violation("rotation", name, f"improper rotation: {_first(bad)}"))
        else:
            bad = np.abs(np.linalg.norm(values, axis=1) - 1.0) > DIRECTION_TOLERANCE
            if bad.any():
                out.append(
                    Violation(
                        "unit_direction", name, f"translation is not a unit vector: {_first(bad)}"
                    )
                )

    if name == "cameras":
        out += _camera_rules(table)
    elif name == "views":
        images = _array(table.column("image")).to_pylist()
        bad = np.array(
            [image is not None and not _image_path_ok(image) for image in images], dtype=bool
        )
        if bad.any():
            out.append(
                Violation(
                    "views.image_path",
                    name,
                    f"image is not a relative POSIX path inside the dataset: {_first(bad)}",
                )
            )
        seen: set[str] = set()
        repeated = np.zeros(len(images), dtype=bool)
        for row, image in enumerate(images):
            if image is not None:
                repeated[row] = image in seen
                seen.add(image)
        if repeated.any():
            out.append(
                Violation(
                    "views.image_duplicate",
                    name,
                    f"image repeats an earlier row's image: {_first(repeated)}",
                )
            )
    elif name in ("matches", "pairs"):
        bad = _np(table, "view_first") >= _np(table, "view_second")
        if bad.any():
            out.append(
                Violation(
                    f"{name}.view_order", name, f"view_first must be < view_second: {_first(bad)}"
                )
            )
        if name == "pairs":
            out += _pair_model_rules(table)
    return out


_CAMERA_COLUMNS = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2")


def _camera_rules(table: pa.Table) -> list[Violation]:
    """§6: the columns each model uses, the prefix rule, and the spherical image size."""
    out: list[Violation] = []
    rows = table.select(["id", "model", "width", "height", *_CAMERA_COLUMNS]).to_pylist()
    columns_bad, prefix_bad, model_bad, size_bad = [], [], [], []
    for row in rows:
        model = row["model"]
        present = {c for c in _CAMERA_COLUMNS if row[c] is not None}
        if model not in CAMERA_MODELS:
            model_bad.append(row["id"])
            continue
        if model == "SPHERICAL":
            required, allowed, prefixes = set(), set(), []
            if row["width"] % 2 or row["height"] * 2 != row["width"]:
                size_bad.append(row["id"])
        else:
            required = {"fx", "fy", "cx", "cy"}
            if model == "PINHOLE":
                allowed, prefixes = required, []
            elif model == "BROWN_CONRADY":
                allowed = required | {"k1", "k2", "k3", "p1", "p2"}
                prefixes = [("k1", "k2", "k3"), ("p1", "p2")]
            else:
                allowed = required | {"k1", "k2", "k3", "k4"}
                prefixes = [("k1", "k2", "k3", "k4")]
        if not required <= present or not present <= allowed:
            columns_bad.append(row["id"])
        elif any(
            row[later] is not None and row[earlier] is None
            for seq in prefixes
            for earlier, later in itertools.pairwise(seq)
        ):
            prefix_bad.append(row["id"])
    for rule, ids, what in (
        ("camera.model", model_bad, f"model must be one of {CAMERA_MODELS}"),
        ("camera.columns", columns_bad, "parameter columns do not match the model"),
        ("camera.prefix", prefix_bad, "coefficients break the prefix rule"),
        ("camera.spherical_size", size_bad, "SPHERICAL needs an even width and height = width / 2"),
    ):
        if ids:
            out.append(Violation(rule, "cameras", f"{what}: camera id(s) {ids[:5]}"))
    return out


def _pair_model_rules(table: pa.Table) -> list[Violation]:
    """§5.8: which of matrix, rotation, and translation each pair model carries."""
    out: list[Violation] = []
    model = _array(table.column("model")).to_pylist()
    has = {c: _valid_mask(_array(table.column(c))) for c in ("matrix", "rotation", "translation")}
    unknown = np.array([m is not None and m not in PAIR_MODELS for m in model], dtype=bool)
    if unknown.any():
        out.append(
            Violation(
                "pairs.model",
                "pairs",
                f"model must be one of {PAIR_MODELS} or null: {_first(unknown)}",
            )
        )
    m = np.array([x if x in PAIR_MODELS else "" for x in model], dtype=object)
    matrix, rotation, translation = has["matrix"], has["rotation"], has["translation"]
    both_or_neither = rotation == translation
    ok = np.where(
        (m == "essential") | (m == "fundamental") | (m == "homography"),
        matrix & both_or_neither,
        np.where(
            m == "rotation", ~matrix & rotation & ~translation, ~matrix & ~rotation & ~translation
        ),
    )
    bad = ~ok & ~unknown
    if bad.any():
        out.append(
            Violation(
                "pairs.model_columns",
                "pairs",
                f"matrix/rotation/translation do not match the model: {_first(bad)}",
            )
        )
    rows, values = _list_rows(table, "matrix", 9)
    zero = (values == 0).all(axis=1)
    if zero.any():
        out.append(
            Violation(
                "pairs.matrix_zero",
                "pairs",
                f"matrix is all zero at row(s) {rows[zero][:5].tolist()}",
            )
        )
    return out


def _reference(
    out: list[Violation],
    tables: Mapping[str, pa.Table],
    broken: set[str],
    source: str,
    columns: Sequence[str],
    target: str,
    target_columns: Sequence[str],
    mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Check that ``source.columns`` references ``target.target_columns``; return the row map."""
    if target in broken:
        return None
    if target not in tables:
        out.append(
            Violation(
                "reference.dangling",
                source,
                f"{list(columns)} references table {target!r}, which is absent",
            )
        )
        return None
    src, dst = tables[source], tables[target]
    query = [_np(src, c) for c in columns]
    if mask is not None:
        query = [q[mask] for q in query]
    index = _lookup([_np(dst, c) for c in target_columns], query)
    if (index < 0).any():
        out.append(
            Violation(
                "reference.dangling",
                source,
                f"{list(columns)} has no match in {target}{tuple(target_columns)}: {_first(index < 0)}",
            )
        )
    return index


def _cross_table_rules(
    tables: Mapping[str, pa.Table], broken: set[str], profiles: Sequence[str]
) -> list[Violation]:
    """References between tables and the rules that span tables (§5)."""
    out: list[Violation] = []
    t = tables
    if "views" in t:
        _reference(out, t, broken, "views", ["camera_id"], "cameras", ["id"])
    if "poses" in t and "views" in t and "views" not in broken:
        used = _lookup([_np(t["views"], "pose_id")], [_np(t["poses"], "id")])
        if (used < 0).any():
            out.append(
                Violation(
                    "reference.unused_pose",
                    "poses",
                    f"pose not referenced by any views.pose_id: {_first(used < 0)}",
                )
            )
    if "observations" in t:
        obs = t["observations"]
        _reference(out, t, broken, "observations", ["point_id"], "points", ["id"])
        view_rows = _reference(out, t, broken, "observations", ["view_id"], "views", ["id"])
        if view_rows is not None and "poses" not in broken:
            known = view_rows >= 0
            pose_ids = _np(t["views"], "pose_id")[view_rows[known]]
            poses = _np(t["poses"], "id") if "poses" in t else np.zeros(0, np.int64)
            unregistered = _lookup([poses], [pose_ids]) < 0
            if unregistered.any():
                out.append(
                    Violation(
                        "observations.unregistered_view",
                        "observations",
                        f"view's pose_id has no poses row: {unregistered.sum()} row(s)",
                    )
                )
        if "keypoints" in t and "keypoints" not in broken:
            feature = _array(obs.column("feature"))
            if feature.null_count:
                out.append(
                    Violation(
                        "observations.feature_null",
                        "observations",
                        f"feature must be non-null when keypoints is present: {feature.null_count} row(s)",
                    )
                )
            has_feature = _valid_mask(feature)
            rows = _reference(
                out,
                t,
                broken,
                "observations",
                ["view_id", "feature"],
                "keypoints",
                ["view_id", "feature"],
                mask=has_feature,
            )
            if rows is not None:
                hit = rows >= 0
                kp = t["keypoints"]
                for axis in ("x", "y"):
                    observed = _np(obs, axis)[has_feature][hit].view(np.uint64)
                    stored = _np(kp, axis)[rows[hit]].view(np.uint64)
                    if (observed != stored).any():
                        out.append(
                            Violation(
                                "observations.pixel_mismatch",
                                "observations",
                                f"{axis} differs bitwise from the keypoint: {int((observed != stored).sum())} row(s)",
                            )
                        )
                        break
    if "keypoints" in t and "keypoints" not in broken:
        _reference(out, t, broken, "keypoints", ["view_id"], "views", ["id"])
        kp = t["keypoints"]
        view, feature = _np(kp, "view_id"), _np(kp, "feature")
        order = np.lexsort((feature, view))
        view, feature = view[order], feature[order]
        starts = _runs(view)
        lengths = np.diff(np.append(starts, view.size))
        expected = np.arange(view.size) - np.repeat(starts, lengths)
        if (feature != expected).any():
            out.append(
                Violation(
                    "keypoints.dense",
                    "keypoints",
                    f"features must be 0, 1, 2, ... within each view: {_first(feature != expected)}",
                )
            )
    if "matches" in t:
        m = t["matches"]
        for side in ("first", "second"):
            _reference(
                out,
                t,
                broken,
                "matches",
                [f"view_{side}", f"feature_{side}"],
                "keypoints",
                ["view_id", "feature"],
            )
        first, second = _np(m, "view_first"), _np(m, "view_second")
        inlier = _array(m.column("inlier"))
        order = np.lexsort((second, first))
        first, second = first[order], second[order]
        is_null = (~_valid_mask(inlier))[order]
        starts = _runs(first, second)
        nulls = (
            np.add.reduceat(is_null.astype(np.int64), starts)
            if starts.size
            else np.zeros(0, np.int64)
        )
        sizes = np.diff(np.append(starts, first.size))
        mixed = (nulls > 0) & (nulls < sizes)
        if mixed.any():
            out.append(
                Violation(
                    "matches.inlier_mixed",
                    "matches",
                    f"inlier is null on some but not all rows of {int(mixed.sum())} pair(s)",
                )
            )
    if "pairs" in t:
        _reference(out, t, broken, "pairs", ["view_first"], "views", ["id"])
        _reference(out, t, broken, "pairs", ["view_second"], "views", ["id"])
    if "gt_poses" in t:
        view_rows = _reference(out, t, broken, "gt_poses", ["view_id"], "views", ["id"])
        if view_rows is not None and "benchmark" in profiles:
            known = view_rows[view_rows >= 0]
            missing = ~_valid_mask(_array(t["views"].column("image")))[known]
            if missing.any():
                ids = _np(t["views"], "id")[known[missing]].tolist()
                out.append(
                    Violation(
                        "views.image_null",
                        "views",
                        f"image must be non-null for a view with a gt_poses row: view id(s) {ids[:5]}",
                    )
                )
    return out


def validate_tables(
    tables: Mapping[str, pa.Table], broken: Iterable[str] = (), profiles: Sequence[str] = ()
) -> list[Violation]:
    """Every §4 to §6 rule over canonical tables. Tables in ``broken`` failed a structural
    check; their own rules and the cross-table rules that read them are skipped. Tables §5
    does not define get no rule here."""
    skip = set(broken)
    out: list[Violation] = []
    for name in SCHEMA:
        if name in tables and name not in skip:
            out += _table_rules(name, tables[name])
    usable = {n: tbl for n, tbl in tables.items() if n not in skip}
    return out + _cross_table_rules(usable, skip, profiles)


# --------------------------------------------------------------------------------------------
# Manifest (§3)
# --------------------------------------------------------------------------------------------


def _reject_duplicate_members(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate member {key!r}")
        seen[key] = value
    return seen


def _reject_constant(token: str):
    raise ValueError(f"{token} is not a JSON value")


def _parse_manifest(data: bytes) -> dict:
    """Strict JSON: no repeated member, and no ``NaN`` or ``Infinity``, even in ignored members."""
    return json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_members,
        parse_constant=_reject_constant,
    )


def _sorted_names(value, pattern: re.Pattern) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(v, str) and pattern.match(v) for v in value)
        and value == sorted(value)
        and len(set(value)) == len(value)
    )


def _check_manifest(m) -> list[Violation]:
    out: list[Violation] = []

    def bad(rule: str, message: str) -> None:
        out.append(Violation(rule, None, message))

    if not isinstance(m, dict):
        bad("manifest.file", "manifest.json is not a JSON object")
        return out
    for member in (
        "omvg",
        "profiles",
        "length_unit",
        "extensions_used",
        "extensions_required",
        "tables",
        "content_hash",
    ):
        if member not in m:
            bad("manifest.member", f"missing required member {member!r}")
    if out:
        return out
    if m["omvg"] != VERSION:
        bad("manifest.version", f"omvg must be {VERSION!r}, found {m['omvg']!r}")
    profiles = m["profiles"]
    if not _sorted_names(profiles, _NAME) or not profiles:
        bad("manifest.profiles", "profiles must be a non-empty sorted array of distinct names")
    elif any(p not in PROFILES for p in profiles):
        bad("manifest.profiles", f"unknown profile in {profiles}; known: {sorted(PROFILES)}")
    if m["length_unit"] not in LENGTH_UNITS:
        bad("manifest.length_unit", f"length_unit must be one of {LENGTH_UNITS}")
    used, required = m["extensions_used"], m["extensions_required"]
    if (
        not _sorted_names(used, _EXTENSION_NAME)
        or not _sorted_names(required, _EXTENSION_NAME)
        or not set(required) <= set(used)
    ):
        bad(
            "manifest.extensions",
            "extensions must be sorted arrays of distinct OMVG_ names, required a subset of used",
        )
    elif required:
        bad(
            "manifest.required_extension",
            f"this reader implements no extension and cannot read {required}",
        )
    tables = m["tables"]
    if not isinstance(tables, dict) or not all(
        isinstance(name, str)
        and _TABLE_NAME.match(name)
        and isinstance(entry, dict)
        and isinstance(entry.get("rows"), int)
        and not isinstance(entry.get("rows"), bool)
        and entry["rows"] >= 0
        and isinstance(entry.get("hash"), str)
        and _HASH.match(entry["hash"])
        for name, entry in tables.items()
    ):
        bad(
            "manifest.tables",
            "tables must map table names to {rows: integer >= 0, hash: sha256:<hex>}",
        )
    if not isinstance(m["content_hash"], str) or not _HASH.match(m["content_hash"]):
        bad(
            "manifest.content_hash",
            "content_hash must be sha256: followed by 64 lowercase hex digits",
        )
    if not out:
        missing = sorted({t for p in profiles for t in PROFILES[p]} - set(tables))
        if missing:
            bad(
                "manifest.profile_tables",
                f"tables required by {profiles} are not listed: {missing}",
            )
    if not out and not used:
        # This reader implements no extension, so it can tell a table is unowned only when none is used.
        unowned = [name for name in tables if name not in SCHEMA]
        if unowned:
            bad(
                "manifest.tables",
                f"§5 does not define {unowned}, and no extension is used to define them",
            )
    return out


def _manifest_content_hash(m: dict) -> bytes:
    core = manifest_core(
        m["omvg"], m["profiles"], m["length_unit"], m["extensions_used"], m["extensions_required"]
    )
    return content_hash(
        core, {name: bytes.fromhex(entry["hash"][7:]) for name, entry in m["tables"].items()}
    )


# --------------------------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------------------------


def _load_table(root: Path, name: str, rows: int) -> tuple[pa.Table | None, list[Violation]]:
    """Open one table file and apply the §8 checks; return the canonical table."""
    out: list[Violation] = []
    path = root / f"{name}.parquet"
    try:
        pf = pq.ParquetFile(path)
        md = pf.metadata
    except (OSError, pa.ArrowException) as error:
        return None, [Violation("parquet.file", name, f"cannot open {path.name}: {error}")]

    codecs = {
        md.row_group(g).column(c).compression
        for g in range(md.num_row_groups)
        for c in range(md.num_columns)
    }
    if not codecs <= set(ALLOWED_CODECS):
        out.append(
            Violation(
                "parquet.codec",
                name,
                f"codecs {sorted(codecs - set(ALLOWED_CODECS))} are not allowed; use Snappy or none",
            )
        )
    if md.num_rows != rows:
        out.append(
            Violation(
                "manifest.rows", name, f"manifest says {rows} rows, the file has {md.num_rows}"
            )
        )
    group_rows = sum(md.row_group(g).num_rows for g in range(md.num_row_groups))
    if group_rows != md.num_rows:
        out.append(
            Violation(
                "parquet.file",
                name,
                f"the footer says {md.num_rows} rows, its row groups hold {group_rows}",
            )
        )

    leaves: dict[str, list] = {}
    for i in range(len(md.schema)):
        leaf = md.schema.column(i)
        leaves.setdefault(_column_of(leaf), []).append(leaf)
    spec = _spec(name)
    found: dict[str, LType] = {}  # logical type of each column's Parquet type
    types: dict[str, LType] = {}  # logical type each column is read as
    for column, group in leaves.items():
        lt = _ltype_of_parquet(column, group)
        c = spec.column(column)
        if sum(_is_column_leaf(leaf, column) for leaf in group) > 1:
            out.append(
                Violation(
                    "schema.column_name", name, f"column name {column!r} is used more than once"
                )
            )
        elif not _COLUMN_NAME.match(column):
            out.append(
                Violation(
                    "schema.column_name",
                    name,
                    f"column name {column!r} must use only A-Z a-z 0-9 _",
                )
            )
        elif lt is None or (c is not None and not _accepts(c.ltype, lt)):
            expected = f" (expected {c.ltype})" if c is not None else ""
            out.append(
                Violation(
                    "parquet.type",
                    name,
                    f"column {column!r} has an unsupported Parquet type{expected}",
                )
            )
        else:
            found[column] = lt
            types[column] = c.ltype if c is not None else lt
    missing = [c.name for c in spec.columns if not c.nullable and c.name not in leaves]
    if missing:
        out.append(
            Violation("schema.missing_column", name, f"required column(s) {missing} are absent")
        )
    if out:
        return None, out
    try:
        raw = pf.read()
    except (OSError, pa.ArrowException) as error:
        return None, [Violation("parquet.file", name, f"cannot read {path.name}: {error}")]
    if raw.num_rows != rows or any(len(column) != rows for column in raw.columns):
        return None, [
            Violation(
                "parquet.file", name, f"{path.name} decodes to {raw.num_rows} rows, not {rows}"
            )
        ]
    for column, lt in found.items():
        if ltype_of_arrow(raw.schema.field(column).type) != lt:
            return None, [
                Violation(
                    "parquet.type",
                    name,
                    f"column {column!r} is not a primitive or three-level LIST column",
                )
            ]
    arrays, fields = [], []
    for c in spec.columns:
        if c.name in types:
            arrays.append(_canonical(_array(raw.column(c.name)), c.ltype))
        else:
            arrays.append(pa.nulls(raw.num_rows, type=arrow_type(c.ltype)))
        fields.append(pa.field(c.name, arrow_type(c.ltype), nullable=True))
    for column in raw.column_names:
        if spec.column(column) is None:
            arrays.append(_canonical(_array(raw.column(column)), types[column]))
            fields.append(pa.field(column, arrow_type(types[column]), nullable=True))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields)), []


def _load(path: str | os.PathLike) -> tuple[dict | None, dict[str, pa.Table], list[Violation]]:
    root = Path(path)
    try:
        manifest = _parse_manifest((root / "manifest.json").read_bytes())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return None, {}, [Violation("manifest.file", None, f"cannot read manifest.json: {error}")]
    violations = _check_manifest(manifest)
    if violations:
        return manifest, {}, violations
    if _hex(_manifest_content_hash(manifest)) != manifest["content_hash"]:
        violations.append(
            Violation(
                "hash.content", None, "content_hash does not match the manifest's table hashes"
            )
        )

    tables: dict[str, pa.Table] = {}
    broken: set[str] = set()
    for name, entry in manifest["tables"].items():
        table, problems = _load_table(root, name, entry["rows"])
        if table is not None:
            problems += _structural_checks(name, table)
        violations += problems
        if problems:
            broken.add(name)
            continue
        if _hex(table_hash(name, table)) != entry["hash"]:
            violations.append(
                Violation("hash.table", name, "table hash does not match the manifest")
            )
        tables[name] = table
    violations += validate_tables(tables, broken, manifest["profiles"])
    return manifest, tables, violations


def check(path: str | os.PathLike) -> list[Violation]:
    """Return every rule the dataset at ``path`` breaks; an empty list means it conforms."""
    return _load(path)[2]


def read(path: str | os.PathLike) -> Dataset:
    """Read and fully verify the dataset at ``path``."""
    manifest, tables, violations = _load(path)
    if violations:
        raise OmvgError(violations)
    return Dataset(manifest, tables)


def compute_hashes(path: str | os.PathLike) -> dict:
    """Recompute every table hash and the content hash from the files, ignoring the recorded
    hashes. The manifest must otherwise be valid, and every listed table must load and pass the
    value checks; ``OmvgError`` lists the problems otherwise."""
    root = Path(path)
    manifest = _parse_manifest((root / "manifest.json").read_bytes())
    hashes = {}
    for name, entry in manifest["tables"].items():
        table, problems = _load_table(root, name, entry["rows"])
        if table is not None:
            problems += _structural_checks(name, table)
        if problems:
            raise OmvgError(problems)
        hashes[name] = table_hash(name, table)
    core = manifest_core(
        manifest["omvg"],
        manifest["profiles"],
        manifest["length_unit"],
        manifest["extensions_used"],
        manifest["extensions_required"],
    )
    return {
        "tables": {n: _hex(h) for n, h in sorted(hashes.items())},
        "content_hash": _hex(content_hash(core, hashes)),
    }


def image_path(root: str | os.PathLike, image: str) -> Path:
    """Resolve ``views.image`` against the image root (the dataset root unless the user supplies
    another), refusing paths that leave it (§10).

    Symbolic links are resolved first, so a link that points outside the root is refused too.
    """
    if not _image_path_ok(image):
        raise ValueError(f"image path {image!r} is not a relative POSIX path")
    base = Path(root).resolve()
    resolved = (base / image).resolve()
    if base != resolved and base not in resolved.parents:
        raise ValueError(f"image path {image!r} resolves outside the image root")
    return resolved


# --------------------------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------------------------


def _canonical_input(name: str, data) -> pa.Table:
    """Build a canonical, key-sorted table from writer input: a pyarrow Table or a mapping from
    column name to values. A table §5 does not define keeps its columns and row order."""
    spec = _spec(name)
    columns = (
        {n: data.column(n) for n in data.column_names} if isinstance(data, pa.Table) else dict(data)
    )
    missing = [c.name for c in spec.columns if not c.nullable and c.name not in columns]
    if missing:
        raise OmvgError(
            [Violation("schema.missing_column", name, f"required column(s) {missing} are absent")]
        )
    for column in columns:
        if not _COLUMN_NAME.match(column):
            raise OmvgError(
                [
                    Violation(
                        "schema.column_name",
                        name,
                        f"column name {column!r} must use only A-Z a-z 0-9 _",
                    )
                ]
            )
    arrays, fields = [], []
    for c in spec.columns:
        if c.name in columns:
            arr = _to_arrow(columns[c.name], c.ltype, name, c.name)
            arrays.append(arr)
            fields.append(pa.field(c.name, arrow_type(c.ltype)))
    for column, values in columns.items():
        if spec.column(column) is not None:
            continue
        arr = values if isinstance(values, (pa.Array, pa.ChunkedArray)) else pa.array(values)
        arr = _array(arr)
        lt = ltype_of_arrow(arr.type)
        if lt is None:
            raise OmvgError(
                [
                    Violation(
                        "parquet.type",
                        name,
                        f"extra column {column!r} has type {arr.type}, outside §7.1",
                    )
                ]
            )
        arrays.append(_canonical(arr, lt))
        fields.append(pa.field(column, arrow_type(lt)))
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise OmvgError(
            [Violation("parquet.file", name, f"columns have different lengths {sorted(lengths)}")]
        )
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    # Absent nullable §5 columns become all-null, which hashes like absence (§7.3).
    for i, c in enumerate(spec.columns):
        if c.name not in table.column_names:
            table = table.add_column(
                i,
                pa.field(c.name, arrow_type(c.ltype)),
                pa.nulls(table.num_rows, type=arrow_type(c.ltype)),
            )
    return table.take(pa.array(_sort_order(table, spec.key)))


def _parquet_schema(name: str, table: pa.Table) -> pa.Schema:
    """REQUIRED repetition for required §5 columns (§8) and the optional ``omvg.table`` entry."""
    spec = _spec(name)
    fields = []
    for f in table.schema:
        c = spec.column(f.name)
        fields.append(pa.field(f.name, f.type, nullable=c is None or c.nullable))
    return pa.schema(fields, metadata={"omvg.table": name})


def _write_parquet(
    table: pa.Table,
    path: Path,
    name: str,
    *,
    compression: str = "snappy",
    encodings: str = "spec",
    row_group_size: int = ROW_GROUP_SIZE,
) -> None:
    """Write one table file. ``encodings="spec"`` applies the §8 recommendations; ``"plain"``
    uses plain encoding everywhere (for testing that hashes ignore the encoding)."""
    schema = _parquet_schema(name, table)
    table = pa.Table.from_arrays([table.column(f.name) for f in schema], schema=schema)
    column_encoding: dict[str, str] = {}
    dictionary: list[str] = []
    for f in schema:
        lt = ltype_of_arrow(f.type)
        leaf = f"{f.name}.list.element" if lt.is_list else f.name
        kind = lt.element if lt.is_list else lt.kind
        if encodings == "plain":
            column_encoding[leaf] = "PLAIN"
        elif kind == "STRING":
            dictionary.append(f.name)
        elif kind in ("INT32", "INT64"):
            column_encoding[leaf] = "DELTA_BINARY_PACKED"
        elif kind in ("FLOAT", "DOUBLE"):
            column_encoding[leaf] = "BYTE_STREAM_SPLIT"
        elif kind == "BOOLEAN":
            column_encoding[leaf] = "RLE"
        else:
            column_encoding[leaf] = "PLAIN"
    statistics = [f.name for f in schema if f.name in _ID_COLUMNS]
    pq.write_table(
        table,
        path,
        compression=compression,
        use_dictionary=dictionary if dictionary else False,
        column_encoding=column_encoding or None,
        row_group_size=row_group_size,
        data_page_version="2.0",
        write_statistics=statistics if statistics else False,
    )


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _check_descriptive(dataset) -> None:
    """Type-check the members §3 defines in ``dataset``; other members pass."""
    if dataset is not None and not (
        isinstance(dataset, Mapping)
        and all(
            isinstance(dataset[k], str)
            for k in ("name", "license", "url", "citation")
            if k in dataset
        )
    ):
        raise OmvgError(
            [
                Violation(
                    "manifest.member",
                    None,
                    "dataset members name, license, url, and citation are strings",
                )
            ]
        )


def write(
    path: str | os.PathLike,
    tables: Mapping[str, object],
    *,
    profiles: Sequence[str],
    length_unit: str = "unknown",
    extensions_used: Sequence[str] = (),
    extensions_required: Sequence[str] = (),
    generator: str | None = "omvg.py (OMVG reference implementation)",
    dataset: Mapping[str, str] | None = None,
    compression: str = "snappy",
    encodings: str = "spec",
    row_group_size: int = ROW_GROUP_SIZE,
    omit_null_columns: bool = True,
) -> dict:
    """Validate, sort, and atomically publish a dataset at ``path``; return its manifest.

    ``tables`` maps table names to a pyarrow Table or to a mapping from column name to values
    (Arrow arrays, numpy arrays with one row per value, or Python sequences). Columns outside
    §5 are written as extra columns, and a table outside §5 as it is (§9). ``path`` must not
    exist. ``omit_null_columns``
    drops nullable §5 columns that are entirely null, which leaves every hash unchanged.
    """
    if compression not in ("snappy", "none"):
        raise ValueError("compression must be 'snappy' or 'none' (§8)")
    if encodings not in ("spec", "plain"):
        raise ValueError("encodings must be 'spec' or 'plain'")
    manifest_fields = {
        "omvg": VERSION,
        "profiles": sorted(set(profiles)),
        "length_unit": length_unit,
        "extensions_used": sorted(set(extensions_used)),
        "extensions_required": sorted(set(extensions_required)),
        "tables": {n: {"rows": 0, "hash": "sha256:" + "0" * 64} for n in tables},
        "content_hash": "sha256:" + "0" * 64,
    }
    problems = _check_manifest(manifest_fields)
    if problems:
        raise OmvgError(problems)
    _check_descriptive(dataset)

    canonical = {name: _canonical_input(name, data) for name, data in tables.items()}
    problems = []
    broken = set()
    for name, table in canonical.items():
        found = _structural_checks(name, table)
        problems += found
        if found:
            broken.add(name)
    problems += validate_tables(canonical, broken, manifest_fields["profiles"])
    if problems:
        raise OmvgError(problems)

    hashes = {name: table_hash(name, table) for name, table in canonical.items()}
    core = manifest_core(
        VERSION,
        manifest_fields["profiles"],
        length_unit,
        manifest_fields["extensions_used"],
        manifest_fields["extensions_required"],
    )
    manifest: dict = {"omvg": VERSION}
    if generator is not None:
        manifest["generator"] = generator
    manifest.update(
        profiles=manifest_fields["profiles"],
        length_unit=length_unit,
        extensions_used=manifest_fields["extensions_used"],
        extensions_required=manifest_fields["extensions_required"],
    )
    if dataset is not None:
        manifest["dataset"] = dict(dataset)
    manifest["tables"] = {
        name: {"rows": canonical[name].num_rows, "hash": _hex(hashes[name])}
        for name in sorted(canonical, key=lambda s: s.encode())
    }
    manifest["content_hash"] = _hex(content_hash(core, hashes))

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"{target} exists; OMVG datasets are published to a new path")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for name, table in canonical.items():
            if omit_null_columns:
                drop = [
                    c.name
                    for c in _spec(name).columns
                    if c.nullable and table.column(c.name).null_count == table.num_rows
                ]
                table = table.drop(drop) if drop else table
            file = staging / f"{name}.parquet"
            _write_parquet(
                table,
                file,
                name,
                compression="none" if compression == "none" else "snappy",
                encodings=encodings,
                row_group_size=row_group_size,
            )
            _fsync(file)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _fsync(staging / "manifest.json")
        _fsync(staging)
        os.rename(staging, target)
        _fsync(target.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


# --------------------------------------------------------------------------------------------
# Camera projection (Appendix A)
# --------------------------------------------------------------------------------------------


def fisheye_theta_max(k1: float, k2: float, k3: float, k4: float) -> float:
    """The first zero of dθ_d/dθ = 1 + 3k1θ² + 5k2θ⁴ + 7k3θ⁶ + 9k4θ⁸ in (0, π), or π (§6)."""
    roots = (
        np.roots([9 * k4, 7 * k3, 5 * k2, 3 * k1, 1.0]) if any((k1, k2, k3, k4)) else np.zeros(0)
    )
    candidates = [
        math.sqrt(s.real)
        for s in roots
        if abs(s.imag) <= 1e-12 * max(1.0, abs(s.real))
        and s.real > 0
        and math.sqrt(s.real) < math.pi
    ]
    return min(candidates, default=math.pi)


def project(camera: Mapping[str, object], point: Sequence[float]) -> tuple[float, float] | None:
    """Project a camera-frame point through a ``cameras`` row; None when the ray does not
    project. Null coefficients enter as zero."""
    x, y, z = (float(v) for v in point)

    def coefficient(name: str) -> float:
        value = camera.get(name)
        return 0.0 if value is None else float(value)

    model = camera["model"]
    if model == "SPHERICAL":
        width, height = float(camera["width"]), float(camera["height"])
        longitude = math.atan2(x, z)
        latitude = math.atan2(y, math.hypot(x, z))
        u = (longitude + math.pi) * width / (2 * math.pi) - 0.5
        if u >= width - 0.5:
            u -= width
        elif u < -0.5:
            u += width
        return u, (latitude + math.pi / 2) * height / math.pi - 0.5
    fx, fy, cx, cy = (coefficient(n) for n in ("fx", "fy", "cx", "cy"))
    if model in ("PINHOLE", "BROWN_CONRADY"):
        if z <= 0:
            return None
        xn, yn = x / z, y / z
        if model == "BROWN_CONRADY":
            k1, k2, k3, p1, p2 = (coefficient(n) for n in ("k1", "k2", "k3", "p1", "p2"))
            r2 = xn * xn + yn * yn
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            xn, yn = (
                xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn),
                yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn,
            )
        return fx * xn + cx, fy * yn + cy
    if model == "FISHEYE":
        k1, k2, k3, k4 = (coefficient(n) for n in ("k1", "k2", "k3", "k4"))
        r = math.hypot(x, y)
        if r == 0:
            return (cx, cy) if z > 0 else None
        theta = math.atan2(r, z)
        if theta > fisheye_theta_max(k1, k2, k3, k4):
            return None
        t2 = theta * theta
        theta_d = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
        return fx * theta_d * x / r + cx, fy * theta_d * y / r + cy
    raise ValueError(f"unknown camera model {model!r}")


# --------------------------------------------------------------------------------------------
# Conformance (§10)
# --------------------------------------------------------------------------------------------


def conformance(testdata: str | os.PathLike) -> list[str]:
    """Run the conformance vectors; return a list of failures (empty when all pass)."""
    root = Path(testdata)
    failures: list[str] = []
    for vector in sorted((root / "valid").iterdir()):
        found = check(vector)
        if found:
            failures.append(f"valid/{vector.name}: {[str(v) for v in found]}")
            continue
    a = json.loads((root / "valid/reconstruction/manifest.json").read_text("utf-8"))
    b = json.loads((root / "valid/reconstruction_plain/manifest.json").read_text("utf-8"))
    if (a["tables"], a["content_hash"]) != (b["tables"], b["content_hash"]):
        failures.append("valid/reconstruction_plain: physical encoding changed a hash")
    for vector in sorted((root / "invalid").iterdir()):
        rule = json.loads((vector / "expected.json").read_text("utf-8"))["rule"]
        rules = sorted({v.rule for v in check(vector)})
        if rules != [rule]:
            failures.append(f"invalid/{vector.name}: expected exactly [{rule}], found {rules}")
    for case in json.loads((root / "projection.json").read_text("utf-8")):
        pixel = project(case["camera"], case["point"])
        want = case["pixel"]
        if (pixel is None) != (want is None) or (
            pixel is not None and max(abs(a - b) for a, b in zip(pixel, want, strict=True)) > 1e-9
        ):
            failures.append(
                f"projection {case['camera']['model']} {case['point']}: expected {want}, found {pixel}"
            )
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("check", "print every rule a dataset breaks"),
        ("hash", "recompute table and content hashes from the files"),
        ("conformance", "run the conformance vectors in a testdata directory"),
    ):
        commands.add_parser(name, help=text).add_argument("path")
    args = parser.parse_args(argv)
    if args.command == "check":
        violations = check(args.path)
        for v in violations:
            print(v)
        print("conforms" if not violations else f"{len(violations)} violation(s)")
        return 1 if violations else 0
    if args.command == "hash":
        try:
            print(json.dumps(compute_hashes(args.path), indent=2))
        except OmvgError as error:
            for v in error.violations:
                print(v)
            return 1
        return 0
    failures = conformance(args.path)
    for f in failures:
        print(f)
    print("all vectors pass" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
