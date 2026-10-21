#!/usr/bin/env python3
"""Generate the OMVG conformance vectors in ``docs/format/testdata/``.

    python make_testdata.py OUTPUT_DIR

The vectors are the committed files, not this script: rerunning it on another platform may
change the last bits of a trigonometric value and therefore the recorded hashes. Every vector
is checked with ``omvg.py`` before the script returns:

- ``valid/<name>/`` must conform, and its manifest records the hashes recomputed from the
  files. ``reconstruction`` and ``reconstruction_plain`` hold the same values in different
  Parquet encodings, codecs, row groups, and column orders, so their hashes are equal.
- ``invalid/<name>/`` breaks exactly the rule named in ``expected.json`` and no other. Its
  manifest records honest hashes unless the hash itself is the broken rule.
- ``projection.json`` lists camera-frame points and the pixels of Appendix A.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omvg

# --------------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------------


def rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Rodrigues rotation, re-orthonormalized so it sits far inside the §4 tolerance."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    r = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)
    u, _, vt = np.linalg.svd(r)
    return u @ vt


def skew(t: np.ndarray) -> np.ndarray:
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def to_camera(r: np.ndarray, c: np.ndarray, x: np.ndarray) -> np.ndarray:
    return r @ (x - c)


# --------------------------------------------------------------------------------------------
# Vector contents
# --------------------------------------------------------------------------------------------


def reconstruction_tables() -> dict:
    """One camera of each model, a shared pose, an unregistered view, and extra columns."""
    cameras = {
        "id": [0, 1, 2, 3],
        "model": ["PINHOLE", "BROWN_CONRADY", "FISHEYE", "SPHERICAL"],
        "width": [1280, 1280, 1280, 2048],
        "height": [960, 960, 960, 1024],
        "fx": [1000.0, 1000.0, 400.0, None],
        "fy": [1010.0, 1010.0, 400.0, None],
        "cx": [639.5, 639.5, 639.5, None],
        "cy": [479.5, 479.5, 479.5, None],
        # BROWN_CONRADY: a radial prefix (k1, k2) whose k2 is a modeled zero, and p1 alone.
        "k1": [None, -0.12, 0.05, None],
        "k2": [None, 0.0, -0.01, None],
        "k3": [None, None, 0.002, None],
        "k4": [None, None, -0.0003, None],
        "p1": [None, 0.0005, None, None],
        "calibrated": [True, False, None, None],
    }
    views = {
        "id": [0, 1, 2, 3, 4],
        "camera_id": [0, 1, 2, 3, 0],
        # Views 0 and 1 share pose 0; view 4's pose has no poses row, so it is not registered.
        "pose_id": [0, 0, 2, 3, 4],
        "image": [
            "images/0000.jpg",
            "images/caméra_1.jpg",
            None,
            "pano/équirect 3.jpg",
            "images/a..JPG",
        ],
        "image_sha256": [
            hashlib.sha256(b"0000").digest(),
            None,
            None,
            hashlib.sha256(b"pano").digest(),
            None,
        ],
        "exposure_us": pa.array([1000, None, 250, None, 4000], pa.int32()),
        "thumbnail": pa.array([b"\x89PNG", None, b"", None, b"\x00\x01"], pa.binary()),
    }
    poses = {
        0: (np.eye(3), np.array([-0.0, 0.0, 0.0])),
        2: (rotation((0, 1, 0), 0.2), np.array([-0.8, 0.1, 0.3])),
        3: (rotation((1, 0.2, 0), -0.15), np.array([0.6, -0.2, 0.5])),
    }
    xyz = np.array(
        [
            [0.3, -0.2, 5.0],
            [-0.6, 0.4, 6.0],
            [0.0, -0.0, 4.0],
            [1.1, 0.7, 7.5],
            [-1.4, -0.9, 5.5],
            [0.2, 1.3, 8.0],
        ]
    )
    camera_rows = pa.table(cameras).to_pylist()
    observations = {"point_id": [], "view_id": [], "feature": [], "x": [], "y": []}
    for p, point in enumerate(xyz):
        for v in (0, 1, 2, 3):
            r, c = poses[views["pose_id"][v]]
            pixel = omvg.project(camera_rows[views["camera_id"][v]], to_camera(r, c, point))
            if pixel is None:
                continue
            observations["point_id"].append(p)
            observations["view_id"].append(v)
            # The spherical view carries no producer feature index.
            observations["feature"].append(None if v == 3 else 10 * p + v)
            observations["x"].append(pixel[0])
            observations["y"].append(pixel[1])
    return {
        "cameras": cameras,
        "views": views,
        "poses": {
            "id": list(poses),
            "rotation": np.array([r.reshape(9) for r, _ in poses.values()]),
            "center": np.array([c for _, c in poses.values()]),
        },
        "points": {
            "id": list(range(len(xyz))),
            "xyz": xyz,
            "color": [
                b"\xff\x00\x00",
                None,
                b"\x00\x80\xff",
                None,
                b"\x10\x20\x30",
                b"\x00\x00\x00",
            ],
            "observers": pa.array([[0, 1, 2], [2], None, [], [0, 3], None], pa.list_(pa.int64())),
        },
        "observations": observations,
    }


def full_tables() -> dict:
    """Every table: keypoints, putative matches, every pair model, and ground truth."""
    k = np.array([[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]])
    camera = {
        "id": 0,
        "model": "PINHOLE",
        "width": 640,
        "height": 480,
        "fx": 500.0,
        "fy": 500.0,
        "cx": 319.5,
        "cy": 239.5,
        "calibrated": True,
    }
    gt = [
        (np.eye(3), np.zeros(3)),
        (rotation((0, 1, 0), -0.08), np.array([0.5, 0.0, 0.05])),
        (rotation((0.1, 1, 0), -0.16), np.array([1.0, 0.05, 0.12])),
        (rotation((0.2, 1, 0.1), -0.24), np.array([1.5, -0.05, 0.2])),
    ]
    xyz = np.array(
        [
            [x, y, z]
            for x, y, z in (
                (0.2, -0.3, 6.0),
                (1.0, 0.4, 7.0),
                (-0.5, 0.2, 5.5),
                (0.8, -0.6, 8.0),
                (1.6, 0.1, 6.5),
                (0.4, 0.7, 7.5),
                (-0.2, -0.5, 9.0),
                (1.2, -0.2, 5.0),
            )
        ]
    )
    keypoints = {"view_id": [], "feature": [], "x": [], "y": [], "scale": []}
    feature_of: dict[tuple[int, int], int] = {}
    for v, (r, c) in enumerate(gt):
        f = 0
        for p, point in enumerate(xyz):
            x, y = omvg.project(camera, to_camera(r, c, point))
            feature_of[(v, p)] = f
            keypoints["view_id"].append(v)
            keypoints["feature"].append(f)
            keypoints["x"].append(x)
            keypoints["y"].append(y)
            keypoints["scale"].append(1.6 * (1 + (p % 3)))
            f += 1
        for extra in range(2):  # trailing keypoints that no match or observation uses
            keypoints["view_id"].append(v)
            keypoints["feature"].append(f)
            keypoints["x"].append(12.25 + 40 * extra + v)
            keypoints["y"].append(400.75 - 30 * extra)
            keypoints["scale"].append(2.0)
            f += 1
    n_points, n_features = len(xyz), len(xyz) + 2
    observations = {"point_id": [], "view_id": [], "feature": [], "x": [], "y": []}
    for p in range(n_points):
        for v in range(len(gt)):
            f = feature_of[(v, p)]
            observations["point_id"].append(p)
            observations["view_id"].append(v)
            observations["feature"].append(f)
            observations["x"].append(keypoints["x"][v * n_features + f])
            observations["y"].append(keypoints["y"][v * n_features + f])

    # Inlier flags per pair: None = not verified; points listed in `outliers` are rejected.
    verification = {
        (0, 1): {6},
        (0, 2): None,
        (0, 3): {1, 4},
        (1, 2): {7},
        (1, 3): set(),
        (2, 3): set(range(n_points)),
    }
    matches = {
        "view_first": [],
        "view_second": [],
        "feature_first": [],
        "feature_second": [],
        "distance": [],
        "inlier": [],
    }
    for (a, b), outliers in verification.items():
        for p in range(n_points):
            matches["view_first"].append(a)
            matches["view_second"].append(b)
            matches["feature_first"].append(feature_of[(a, p)])
            matches["feature_second"].append(feature_of[(b, p)])
            matches["distance"].append(0.05 * (p + 1) + 0.01 * (a + b))
            matches["inlier"].append(None if outliers is None else p not in outliers)
        # A wrong match between the trailing keypoints of both views.
        matches["view_first"].append(a)
        matches["view_second"].append(b)
        matches["feature_first"].append(n_points)
        matches["feature_second"].append(n_points + 1)
        matches["distance"].append(0.9)
        matches["inlier"].append(None if outliers is None else False)

    def relative(a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        (ra, ca), (rb, cb) = gt[a], gt[b]
        r = rb @ ra.T
        t = rb @ (ca - cb)
        return r, t / np.linalg.norm(t)

    r01, t01 = relative(0, 1)
    r03, t03 = relative(0, 3)
    r12, _ = relative(1, 2)
    r13, _ = relative(1, 3)
    k_inv = np.linalg.inv(k)
    e01 = skew(t01) @ r01
    f03 = k_inv.T @ (skew(t03) @ r03) @ k_inv
    h12 = k @ r12 @ k_inv
    pairs = {
        "view_first": [0, 0, 1, 1, 2],
        "view_second": [1, 3, 2, 3, 3],
        "model": ["essential", "fundamental", "homography", "rotation", None],
        "matrix": [
            e01.reshape(9).tolist(),
            (f03 / np.linalg.norm(f03)).reshape(9).tolist(),
            (h12 / h12[2, 2]).reshape(9).tolist(),
            None,
            None,
        ],
        "rotation": [
            r01.reshape(9).tolist(),
            r03.reshape(9).tolist(),
            None,
            r13.reshape(9).tolist(),
            None,
        ],
        "translation": [t01.tolist(), t03.tolist(), None, None, None],
        "colmap_config": pa.array([2, 3, 4, 5, 7], pa.int32()),
    }
    rotations = np.array([r.reshape(9) for r, _ in gt])
    centers = np.array([c for _, c in gt])
    return {
        "cameras": {key: [value] for key, value in camera.items()},
        # Partial ground truth: view 3 has no gt_poses row, so it may have no image.
        "views": {
            "id": [0, 1, 2, 3],
            "camera_id": [0, 0, 0, 0],
            "pose_id": [0, 1, 2, 3],
            "image": ["images/0000.png", "images/0001.png", "images/0002.png", None],
        },
        "keypoints": keypoints,
        "matches": matches,
        "pairs": pairs,
        "gt_poses": {"view_id": [0, 1, 2], "rotation": rotations[:3], "center": centers[:3]},
        "poses": {"id": [0, 1, 2, 3], "rotation": rotations, "center": centers},
        "points": {
            "id": list(range(n_points)),
            "xyz": xyz,
            "color": [bytes([20 * p, 255 - 20 * p, 128]) for p in range(n_points)],
            "error_px": [0.0, 0.25, 0.5, -0.0, 1.0, 0.125, 0.75, 0.375],
        },
        "observations": observations,
    }


def empty_tables() -> dict:
    """A dataset whose keypoints and matches tables have no rows."""
    spec = omvg.SCHEMA
    return {
        "cameras": {"id": [0], "model": ["SPHERICAL"], "width": [4096], "height": [2048]},
        "views": {"id": [0], "camera_id": [0], "pose_id": [0]},
        "keypoints": {
            c.name: pa.array([], omvg.arrow_type(c.ltype)) for c in spec["keypoints"].columns
        },
        "matches": {
            c.name: pa.array([], omvg.arrow_type(c.ltype)) for c in spec["matches"].columns
        },
    }


def example_scores() -> dict:
    """A table of the made-up extension OMVG_example, which no reader implements."""
    return {
        "view_id": [0, 0, 0],
        "rank": pa.array([2, 0, 1], pa.int32()),  # no key: rows keep their order
        "score": pa.array([0.5, None, 0.25], pa.float32()),
        "label": ["a", "b", None],
        "weights": pa.array([[1.0, 2.0], [3.0, 4.0], None], pa.list_(pa.float64(), 2)),
        "note": pa.nulls(3, pa.string()),  # all-null, so not hashed
    }


FULL_PROFILES = ["benchmark", "dataset", "reconstruction"]
EXTENSIONS = ["OMVG_example"]


# --------------------------------------------------------------------------------------------
# Raw writing for invalid vectors
# --------------------------------------------------------------------------------------------


def canonical(tables: dict) -> dict[str, pa.Table]:
    return {name: omvg._canonical_input(name, data) for name, data in tables.items()}


def with_value(table: pa.Table, column: str, row: int, value) -> pa.Table:
    values = table.column(column).to_pylist()
    values[row] = value
    index = table.column_names.index(column)
    return table.set_column(
        index, table.schema.field(index), pa.array(values, table.schema.field(index).type)
    )


def append_row(table: pa.Table, row: dict) -> pa.Table:
    rows = table.to_pylist() + [{c: row.get(c) for c in table.column_names}]
    return pa.Table.from_pylist(rows, schema=table.schema)


def raw_write(
    directory: Path,
    tables: dict[str, pa.Table],
    profiles: list[str],
    *,
    extensions: Sequence[str] = (),
    sort: bool = True,
    manifest_edit: Callable[[dict], dict] | None = None,
    file_edit: Callable[[Path], None] | None = None,
) -> None:
    """Write tables and an honest manifest without validation."""
    directory.mkdir(parents=True)
    hashes = {}
    for name, table in tables.items():
        if sort:
            table = table.take(pa.array(omvg._sort_order(table, omvg._spec(name).key)))
        try:
            hashes[name] = omvg.table_hash(name, table)
        except (KeyError, ValueError, pa.ArrowException):
            hashes[name] = bytes(32)  # §7 defines no hash for a table that breaks a §4 or §8 rule
        pq.write_table(
            table.replace_schema_metadata({"omvg.table": name}),
            directory / f"{name}.parquet",
            compression="snappy",
        )
    core = omvg.manifest_core(omvg.VERSION, profiles, "unknown", extensions, [])
    manifest = {
        "omvg": omvg.VERSION,
        "generator": "make_testdata.py",
        "profiles": list(profiles),
        "length_unit": "unknown",
        "extensions_used": list(extensions),
        "extensions_required": [],
        "tables": {
            n: {"rows": tables[n].num_rows, "hash": omvg._hex(hashes[n])} for n in sorted(tables)
        },
        "content_hash": omvg._hex(omvg.content_hash(core, hashes)),
    }
    if manifest_edit is not None:
        manifest = manifest_edit(manifest)
    text = (
        manifest
        if isinstance(manifest, str)
        else json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    (directory / "manifest.json").write_text(text, encoding="utf-8")
    if file_edit is not None:
        file_edit(directory)


def rewrite(directory: Path, name: str, edit: Callable[[pa.Table], pa.Table], **options) -> None:
    """Replace one table file, keeping the manifest as it is."""
    path = directory / f"{name}.parquet"
    table = edit(pq.read_table(path))
    pq.write_table(
        table.replace_schema_metadata({"omvg.table": name}),
        path,
        **{"compression": "snappy", **options},
    )


class Compact:
    """A cursor over the Thrift compact bytes of a Parquet footer, enough to find a field."""

    def __init__(self, data: bytes):
        self.data, self.at = data, 0

    def byte(self) -> int:
        self.at += 1
        return self.data[self.at - 1]

    def varint(self) -> int:
        value, shift = 0, 0
        while True:
            byte = self.byte()
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                return value
            shift += 7

    def header(self, last: int) -> tuple[int, int]:
        """The id and type of the next field after field ``last``; type 0 ends the struct."""
        byte = self.byte()
        if byte == 0:
            return last, 0
        if byte >> 4:
            return last + (byte >> 4), byte & 0x0F
        u = self.varint()
        return (u >> 1) ^ -(u & 1), byte & 0x0F

    def skip(self, kind: int) -> None:
        if kind == 3:
            self.at += 1
        elif kind in (4, 5, 6):
            self.varint()
        elif kind == 7:
            self.at += 8
        elif kind == 8:
            length = self.varint()
            self.at += length
        elif kind in (9, 10):
            header = self.byte()
            count = self.varint() if header >> 4 == 15 else header >> 4
            element = header & 0x0F
            for _ in range(count):
                self.skip(3 if element in (1, 2) else element)  # a boolean element takes a byte
        elif kind == 11:
            count = self.varint()
            types = self.byte() if count else 0
            for _ in range(count):
                self.skip(types >> 4)
                self.skip(types & 0x0F)
        elif kind == 12:
            field, kind = self.header(0)
            while kind:
                self.skip(kind)
                field, kind = self.header(field)
        elif kind not in (1, 2):  # a boolean field carries its value in the header
            raise ValueError(f"unknown Thrift type {kind}")

    def find(self, wanted: int) -> None:
        """Move to the value of top-level field ``wanted``."""
        field, kind = self.header(0)
        while field != wanted:
            assert kind, f"no field {wanted}"
            self.skip(kind)
            field, kind = self.header(field)


def varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append(value & 0x7F | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def patch_footer(path: Path, field: int, patch: Callable[[Compact], bytes]) -> None:
    """Replace the value of a top-level footer field; ``patch`` reads the old value."""
    blob = path.read_bytes()
    length = struct.unpack("<I", blob[-8:-4])[0]
    footer = blob[-8 - length : -8]
    cursor = Compact(footer)
    cursor.find(field)
    start = cursor.at
    value = patch(cursor)
    footer = footer[:start] + value + footer[cursor.at :]
    path.write_bytes(blob[: -8 - length] + footer + struct.pack("<I", len(footer)) + b"PAR1")


# --------------------------------------------------------------------------------------------
# Invalid vectors: (name, rule, description, builder)
# --------------------------------------------------------------------------------------------


def _manifest(edit: Callable[[dict], None]) -> dict:
    def apply(m: dict) -> dict:
        edit(m)
        return m

    return {"manifest_edit": apply}


def _invalid_string(t: dict) -> None:
    views = t["views"]
    images = pa.array([b"images/\xff.png", b"a.png", b"b.png", b"c.png"], pa.binary()).view(
        pa.string()
    )
    t["views"] = views.set_column(views.column_names.index("image"), "image", images)


def _null_rotation_element(t: dict) -> None:
    poses = t["poses"]
    values = poses.column("rotation").to_pylist()
    values[1][4] = None
    loose = pa.list_(pa.float64())
    t["poses"] = poses.set_column(1, pa.field("rotation", loose), pa.array(values, loose))


def _unsorted(t: dict) -> None:
    t["points"] = t["points"].take(pa.array([1, 0, *range(2, t["points"].num_rows)]))


def _duplicate(t: dict) -> None:
    pts = t["points"]
    t["points"] = pa.concat_tables([pts.slice(0, 2), pts.slice(1, 1), pts.slice(2)])


def _dangling(t: dict) -> None:
    t["observations"] = append_row(
        t["observations"], {"point_id": 7, "view_id": 9, "feature": 9, "x": 1.0, "y": 1.0}
    )


def _unused_pose(t: dict) -> None:
    t["poses"] = append_row(
        t["poses"], {"id": 9, "rotation": np.eye(3).reshape(9).tolist(), "center": [0.0, 0.0, 0.0]}
    )


def _unregistered(t: dict) -> None:
    t["poses"] = t["poses"].slice(0, 3)  # view 3 keeps pose_id 3 and its observations


def _sparse_keypoints(t: dict) -> None:
    kp = t["keypoints"]
    last = 9  # view 0's last keypoint, used by nothing
    t["keypoints"] = with_value(kp, "feature", last, 10)


def _swapped_match(t: dict) -> None:
    t["matches"] = append_row(
        t["matches"],
        {
            "view_first": 3,
            "view_second": 1,
            "feature_first": 8,
            "feature_second": 9,
            "distance": 0.95,
            "inlier": None,
        },
    )


def _mixed_inliers(t: dict) -> None:
    m = t["matches"]
    first = m.column("view_first").to_pylist()
    second = m.column("view_second").to_pylist()
    row = next(i for i, (a, b) in enumerate(zip(first, second, strict=True)) if (a, b) == (0, 2))
    t["matches"] = with_value(m, "inlier", row, True)


def _swapped_pair(t: dict) -> None:
    t["pairs"] = append_row(
        t["pairs"],
        {
            "view_first": 3,
            "view_second": 0,
            "model": "rotation",
            "rotation": np.eye(3).reshape(9).tolist(),
        },
    )


def _matrix_zero(t: dict) -> None:
    t["pairs"] = with_value(t["pairs"], "matrix", 2, [0.0] * 9)


def _improper_rotation(t: dict) -> None:
    values = t["poses"].column("rotation").to_pylist()
    t["poses"] = with_value(t["poses"], "rotation", 1, [v * (1 + 1e-6) for v in values[1]])


def _long_direction(t: dict) -> None:
    values = t["pairs"].column("translation").to_pylist()
    t["pairs"] = with_value(t["pairs"], "translation", 0, [2 * v for v in values[0]])


def _spherical_size(t: dict) -> None:
    cams = t["cameras"]
    for column, value in (
        ("model", "SPHERICAL"),
        ("fx", None),
        ("fy", None),
        ("cx", None),
        ("cy", None),
        ("width", 2048),
        ("height", 1000),
    ):
        cams = with_value(cams, column, 0, value)
    t["cameras"] = cams


def _pixel_mismatch(t: dict) -> None:
    obs = t["observations"]
    t["observations"] = with_value(
        obs, "x", 3, math.nextafter(obs.column("x")[3].as_py(), math.inf)
    )


def _reflection(t: dict) -> None:
    values = t["gt_poses"].column("rotation").to_pylist()
    t["gt_poses"] = with_value(t["gt_poses"], "rotation", 2, [-v for v in values[2]])


def _add_column(table: str, column: str, values: pa.Array) -> Callable[[dict], None]:
    def edit(t: dict) -> None:
        t[table] = t[table].append_column(column, values)

    return edit


def _duplicate_column(t: dict) -> None:
    t["views"] = t["views"].append_column("image", t["views"].column("image"))


def _drop(*names: str) -> Callable[[dict], None]:
    def edit(t: dict) -> None:
        for name in names:
            t.pop(name)

    return edit


def _dangling_match(t: dict) -> None:
    m = t["matches"]
    pairs = list(
        zip(m.column("view_first").to_pylist(), m.column("view_second").to_pylist(), strict=True)
    )
    t["matches"] = with_value(m, "feature_second", pairs.index((0, 2)), 99)


def _add_example_scores(t: dict) -> None:
    t["example_scores"] = omvg._canonical_input("example_scores", example_scores())


def _set_rows(table: str, edits: dict[str, object], row: int = 0) -> Callable[[dict], None]:
    def edit(t: dict) -> None:
        for column, value in edits.items():
            t[table] = with_value(t[table], column, row, value)

    return edit


def _file_edit(edit: Callable[[Path], None]) -> dict:
    return {"file_edit": edit}


def _tamper_points(directory: Path) -> None:
    rewrite(directory, "points", lambda t: with_value(t, "error_px", 2, 0.5000001))


def _tamper_example_scores(directory: Path) -> None:
    rewrite(directory, "example_scores", lambda t: with_value(t, "rank", 0, 3))


def _missing_file(directory: Path) -> None:
    (directory / "gt_poses.parquet").unlink()


def _overcount_points(directory: Path) -> None:
    """The footer's num_rows (field 3) says 9 while the row groups hold the 8 points."""

    def nine(cursor: Compact) -> bytes:
        cursor.varint()
        return varint(2 * 9)  # zigzag

    patch_footer(directory / "points.parquet", 3, nine)


def _billions_of_row_groups(directory: Path) -> None:
    """The row_groups list (field 4) claims 2^31 - 1 elements in a footer of a few hundred bytes."""

    def billions(cursor: Compact) -> bytes:
        assert cursor.byte() == 0x1C  # one row group
        return bytes([0xFC]) + varint(2**31 - 1)  # a struct list whose count follows

    patch_footer(directory / "points.parquet", 4, billions)


INVALID: list[tuple[str, str, str, Callable[[dict], None] | None, dict]] = [
    # Manifest (§3).
    (
        "manifest_duplicate_member",
        "manifest.file",
        "manifest.json repeats a member",
        None,
        {"manifest_edit": lambda m: json.dumps(m)[:-1] + ', "omvg": "0.1-draft"}'},
    ),
    (
        "manifest_nan",
        "manifest.file",
        "an unknown member of manifest.json is NaN, which JSON does not allow",
        None,
        {"manifest_edit": lambda m: json.dumps(m)[:-1] + ', "note": NaN}'},
    ),
    (
        "manifest_missing_member",
        "manifest.member",
        "content_hash is absent",
        None,
        _manifest(lambda m: m.pop("content_hash")),
    ),
    (
        "manifest_version",
        "manifest.version",
        "omvg names an unknown version",
        None,
        _manifest(lambda m: m.update(omvg="0.2")),
    ),
    (
        "manifest_profiles_unsorted",
        "manifest.profiles",
        "profiles are not sorted",
        None,
        _manifest(lambda m: m.update(profiles=["reconstruction", "dataset", "benchmark"])),
    ),
    (
        "manifest_profile_unknown",
        "manifest.profiles",
        "an unknown profile is declared",
        None,
        _manifest(lambda m: m["profiles"].append("stereo")),
    ),
    (
        "manifest_length_unit",
        "manifest.length_unit",
        "length_unit is not a known unit",
        None,
        _manifest(lambda m: m.update(length_unit="feet")),
    ),
    (
        "manifest_extensions",
        "manifest.extensions",
        "extensions_required is not a subset of extensions_used",
        None,
        _manifest(lambda m: m.update(extensions_required=["OMVG_example_required"])),
    ),
    (
        "manifest_required_extension",
        "manifest.required_extension",
        "an extension this reader does not implement is required",
        None,
        _manifest(
            lambda m: m.update(
                extensions_used=["OMVG_example_required"],
                extensions_required=["OMVG_example_required"],
            )
        ),
    ),
    (
        "manifest_tables_rows",
        "manifest.tables",
        "a row count is negative",
        None,
        _manifest(lambda m: m["tables"]["views"].update(rows=-1)),
    ),
    (
        "manifest_tables_hash",
        "manifest.tables",
        "a table hash uses uppercase hexadecimal",
        None,
        _manifest(
            lambda m: m["tables"]["views"].update(
                hash=m["tables"]["views"]["hash"].upper().replace("SHA256", "sha256")
            )
        ),
    ),
    (
        "manifest_content_hash_format",
        "manifest.content_hash",
        "content_hash lacks the sha256: prefix",
        None,
        _manifest(lambda m: m.update(content_hash=m["content_hash"][7:])),
    ),
    (
        "manifest_profile_tables",
        "manifest.profile_tables",
        "the benchmark profile's gt_poses table is not listed",
        lambda t: t.pop("gt_poses"),
        {},
    ),
    (
        "manifest_rows",
        "manifest.rows",
        "a row count disagrees with the file",
        None,
        _manifest(lambda m: m["tables"]["points"].update(rows=9)),
    ),
    (
        "manifest_rows_fraction",
        "manifest.tables",
        "a row count is written as 4.0",
        None,
        _manifest(lambda m: m["tables"]["views"].update(rows=4.0)),
    ),
    (
        "manifest_profiles_empty",
        "manifest.profiles",
        "profiles is empty",
        None,
        _manifest(lambda m: m.update(profiles=[])),
    ),
    (
        "manifest_extension_name",
        "manifest.extensions",
        "an extension name lacks the OMVG_ prefix",
        None,
        _manifest(lambda m: m.update(extensions_used=["rigs"])),
    ),
    (
        "manifest_unowned_table",
        "manifest.tables",
        "a table that section 5 does not define is listed while no extension is used",
        _add_example_scores,
        {},
    ),
    # Integrity (§7).
    (
        "hash_content",
        "hash.content",
        "content_hash does not match the table hashes",
        None,
        _manifest(lambda m: m.update(content_hash="sha256:" + "ab" * 32)),
    ),
    (
        "hash_table",
        "hash.table",
        "a value changed after the table was hashed",
        None,
        _file_edit(_tamper_points),
    ),
    (
        "hash_table_extension",
        "hash.table",
        "a value of an unknown extension table changed after it was hashed",
        _add_example_scores,
        {"extensions": EXTENSIONS, **_file_edit(_tamper_example_scores)},
    ),
    # Parquet (§8).
    (
        "parquet_file",
        "parquet.file",
        "a listed table file is missing",
        None,
        _file_edit(_missing_file),
    ),
    (
        "parquet_row_groups",
        "parquet.file",
        "the footer and the manifest say 9 rows, the row groups hold 8",
        None,
        {
            **_manifest(lambda m: m["tables"]["points"].update(rows=9)),
            **_file_edit(_overcount_points),
        },
    ),
    (
        "parquet_row_group_count",
        "parquet.file",
        "the footer claims 2^31 - 1 row groups in a few hundred bytes",
        None,
        _file_edit(_billions_of_row_groups),
    ),
    (
        "parquet_codec",
        "parquet.codec",
        "a file uses zstd",
        None,
        _file_edit(lambda d: rewrite(d, "points", lambda t: t, compression="zstd")),
    ),
    (
        "parquet_type_unsigned",
        "parquet.type",
        "an ID column is unsigned",
        None,
        _file_edit(
            lambda d: rewrite(
                d, "views", lambda t: t.set_column(0, "id", t.column("id").cast(pa.uint64()))
            )
        ),
    ),
    (
        "parquet_type_width",
        "parquet.type",
        "width is stored as INT64",
        None,
        _file_edit(
            lambda d: rewrite(
                d, "cameras", lambda t: t.set_column(2, "width", t.column("width").cast(pa.int64()))
            )
        ),
    ),
    (
        "parquet_type_element",
        "parquet.type",
        "rotation elements are stored as FLOAT",
        None,
        _file_edit(
            lambda d: rewrite(
                d,
                "poses",
                lambda t: t.set_column(
                    1, "rotation", t.column("rotation").cast(pa.list_(pa.float32()))
                ),
            )
        ),
    ),
    (
        "parquet_type_struct",
        "parquet.type",
        "an extra column is a struct with one field",
        _add_column(
            "views",
            "exposure",
            pa.array([{"us": v} for v in (1, 2, 3, 4)], pa.struct([("us", pa.int32())])),
        ),
        {},
    ),
    (
        "schema_missing_column",
        "schema.missing_column",
        "observations has no y column",
        lambda t: t.update(observations=t["observations"].drop(["y"])),
        {},
    ),
    (
        "schema_column_name_dot",
        "schema.column_name",
        "an extra column name contains a dot",
        _add_column("views", "exposure.us", pa.array([1, 2, 3, 4], pa.int32())),
        {},
    ),
    (
        "schema_column_duplicate",
        "schema.column_name",
        "views has two image columns",
        _duplicate_column,
        {},
    ),
    # Values (§4).
    (
        "value_null",
        "value.null",
        "a required column holds a null",
        _set_rows("points", {"xyz": None}, row=3),
        {},
    ),
    (
        "value_non_finite",
        "value.non_finite",
        "a coordinate is NaN",
        _set_rows("keypoints", {"x": math.nan}, row=2),
        {},
    ),
    (
        "value_list_length",
        "value.list_length",
        "a center has two elements",
        _set_rows("gt_poses", {"center": [1.0, 2.0]}, row=1),
        {},
    ),
    (
        "value_list_null",
        "value.list_null",
        "a rotation element is null",
        _null_rotation_element,
        {},
    ),
    ("value_utf8", "value.utf8", "an image path is not valid UTF-8", _invalid_string, {}),
    (
        "value_range",
        "value.range",
        "a keypoint scale is negative",
        _set_rows("keypoints", {"scale": -1.0}),
        {},
    ),
    (
        "value_range_id",
        "value.range",
        "a view id is -1",
        lambda t: t.update(views=append_row(t["views"], {"id": -1, "camera_id": 0, "pose_id": 0})),
        {},
    ),
    (
        "value_range_width",
        "value.range",
        "a camera width is 0",
        _set_rows("cameras", {"width": 0}),
        {},
    ),
    (
        "value_range_distance",
        "value.range",
        "a match distance is negative",
        _set_rows("matches", {"distance": -0.5}),
        {},
    ),
    ("rotation", "rotation", "a pose rotation is scaled by 1 + 1e-6", _improper_rotation, {}),
    (
        "rotation_reflection",
        "rotation",
        "a ground-truth rotation is a reflection (det -1)",
        _reflection,
        {},
    ),
    ("unit_direction", "unit_direction", "a pair translation has length 2", _long_direction, {}),
    # Tables (§5) and camera models (§6).
    ("key_unsorted", "key.unsorted", "points rows are out of order", _unsorted, {"sort": False}),
    ("key_duplicate", "key.duplicate", "two points rows share an id", _duplicate, {}),
    (
        "reference_dangling",
        "reference.dangling",
        "an observation names a missing point and view",
        _dangling,
        {},
    ),
    (
        "reference_dangling_keypoint",
        "reference.dangling",
        "an observation names a feature its view does not have",
        _set_rows("observations", {"feature": 99}),
        {},
    ),
    (
        "reference_dangling_match",
        "reference.dangling",
        "a match names a feature its view does not have",
        _dangling_match,
        {},
    ),
    (
        "reference_dangling_ground_truth",
        "reference.dangling",
        "a ground-truth pose names a missing view",
        lambda t: t.update(
            gt_poses=append_row(
                t["gt_poses"],
                {
                    "view_id": 9,
                    "rotation": np.eye(3).reshape(9).tolist(),
                    "center": [0.0, 0.0, 0.0],
                },
            )
        ),
        {},
    ),
    (
        "reference_absent_table",
        "reference.dangling",
        "observations is present but points is absent",
        _drop("points"),
        {"profiles": ["benchmark", "dataset"]},
    ),
    (
        "reference_unused_pose",
        "reference.unused_pose",
        "a pose no view references",
        _unused_pose,
        {},
    ),
    (
        "camera_model",
        "camera.model",
        "an unknown camera model",
        _set_rows("cameras", {"model": "OPENCV"}),
        {},
    ),
    (
        "camera_columns",
        "camera.columns",
        "a PINHOLE camera carries k1",
        _set_rows("cameras", {"k1": 0.0}),
        {},
    ),
    (
        "camera_columns_missing",
        "camera.columns",
        "a PINHOLE camera has no fy",
        _set_rows("cameras", {"fy": None}),
        {},
    ),
    (
        "camera_prefix",
        "camera.prefix",
        "a BROWN_CONRADY camera has k2 without k1",
        _set_rows("cameras", {"model": "BROWN_CONRADY", "k2": 0.01}),
        {},
    ),
    (
        "camera_prefix_tangential",
        "camera.prefix",
        "a BROWN_CONRADY camera has p2 without p1",
        _set_rows("cameras", {"model": "BROWN_CONRADY", "p2": 0.001}),
        {},
    ),
    (
        "camera_prefix_fisheye",
        "camera.prefix",
        "a FISHEYE camera has k4 without k3",
        _set_rows("cameras", {"model": "FISHEYE", "k1": 0.0, "k2": 0.0, "k4": 0.01}),
        {},
    ),
    (
        "camera_spherical_size",
        "camera.spherical_size",
        "a SPHERICAL camera's height is not width / 2",
        _spherical_size,
        {},
    ),
    (
        "camera_spherical_odd",
        "camera.spherical_size",
        "a SPHERICAL camera's width is odd",
        _set_rows(
            "cameras",
            {
                "model": "SPHERICAL",
                "fx": None,
                "fy": None,
                "cx": None,
                "cy": None,
                "width": 2049,
                "height": 1024,
            },
        ),
        {},
    ),
    (
        "views_image_path",
        "views.image_path",
        "an image path leaves the dataset root",
        _set_rows("views", {"image": "../escape.png"}, row=2),
        {},
    ),
    (
        "views_image_absolute",
        "views.image_path",
        "an image path is absolute",
        _set_rows("views", {"image": "/images/0001.png"}, row=1),
        {},
    ),
    (
        "views_image_backslash",
        "views.image_path",
        "an image path uses a backslash",
        _set_rows("views", {"image": "images\\0001.png"}, row=1),
        {},
    ),
    (
        "views_image_duplicate",
        "views.image_duplicate",
        "two views share an image",
        _set_rows("views", {"image": "images/0000.png"}, row=1),
        {},
    ),
    (
        "views_image_null",
        "views.image_null",
        "a view with a ground-truth pose has no image",
        _set_rows("views", {"image": None}),
        {},
    ),
    (
        "observations_unregistered_view",
        "observations.unregistered_view",
        "observations in a view whose pose has no row",
        _unregistered,
        {},
    ),
    (
        "observations_feature_null",
        "observations.feature_null",
        "an observation without a feature while keypoints is present",
        _set_rows("observations", {"feature": None}, row=5),
        {},
    ),
    (
        "observations_pixel_mismatch",
        "observations.pixel_mismatch",
        "an observed x differs from its keypoint by one ulp",
        _pixel_mismatch,
        {},
    ),
    ("keypoints_dense", "keypoints.dense", "a view skips feature index 9", _sparse_keypoints, {}),
    (
        "matches_view_order",
        "matches.view_order",
        "a match has view_first > view_second",
        _swapped_match,
        {},
    ),
    (
        "matches_inlier_mixed",
        "matches.inlier_mixed",
        "an unverified pair has one non-null inlier flag",
        _mixed_inliers,
        {},
    ),
    (
        "pairs_view_order",
        "pairs.view_order",
        "a pair has view_first > view_second",
        _swapped_pair,
        {},
    ),
    (
        "pairs_model",
        "pairs.model",
        "an unknown pair model",
        _set_rows("pairs", {"model": "affine"}, row=2),
        {},
    ),
    (
        "pairs_model_columns",
        "pairs.model_columns",
        "an essential pair with a rotation but no translation",
        _set_rows("pairs", {"translation": None}),
        {},
    ),
    ("pairs_matrix_zero", "pairs.matrix_zero", "a homography matrix is all zero", _matrix_zero, {}),
]


# --------------------------------------------------------------------------------------------
# Camera projection vectors (Appendix A)
# --------------------------------------------------------------------------------------------

_INTRINSICS = {"fx": 1000.0, "fy": 1010.0, "cx": 639.5, "cy": 479.5, "width": 1280, "height": 960}
_BROWN = {
    **_INTRINSICS,
    "model": "BROWN_CONRADY",
    "k1": -0.12,
    "k2": 0.03,
    "k3": -0.004,
    "p1": 0.0005,
    "p2": -0.0003,
}
_BROWN_PARTIAL = {**_INTRINSICS, "model": "BROWN_CONRADY", "k1": -0.12, "p1": 0.0005}
_FISHEYE = {
    "model": "FISHEYE",
    "width": 1280,
    "height": 960,
    "fx": 400.0,
    "fy": 400.0,
    "cx": 639.5,
    "cy": 479.5,
    "k1": 0.05,
    "k2": -0.01,
    "k3": 0.002,
    "k4": -0.0003,
}
_SPHERICAL = {"model": "SPHERICAL", "width": 2048, "height": 1024}
A, B = [0.3, -0.2, 2.0], [-0.45, 0.35, 1.5]

PROJECTIONS = [
    # The pixels of the Appendix A tables.
    ({**_INTRINSICS, "model": "PINHOLE"}, A, [789.5, 378.5]),
    ({**_INTRINSICS, "model": "PINHOLE"}, B, [339.5, 715.1666666666666]),
    (_BROWN, A, [788.881482528125, 378.9263159310625]),
    (_BROWN, B, [344.3485053497942, 711.3968000566987]),
    (_BROWN_PARTIAL, A, [788.9, 378.9204125]),
    (_BROWN_PARTIAL, B, [344.63, 711.2097111111111]),
    (_FISHEYE, A, [698.956215580877, 439.862522946082]),
    (_FISHEYE, [1.0, -0.5, 0.2], [1172.9527581932998, 212.77362090335004]),
    (_FISHEYE, [1.0, 0.5, -0.3], [1356.0873362966172, 837.7936681483086]),
    (_SPHERICAL, A, [1072.0305776921493, 479.37015747028164]),
    (_SPHERICAL, [-1.0, 0.5, -2.0], [150.6256244740437, 583.2049323996682]),
    (_SPHERICAL, [0.0, 0.0, 1.0], [1023.5, 511.5]),
    (_SPHERICAL, [0.0, -1.0, 0.0], [1023.5, -0.5]),
    # Boundary cases.
    ({**_INTRINSICS, "model": "PINHOLE"}, [0.1, 0.1, -1.0], None),  # behind the camera
    (_FISHEYE, [0.0, 0.0, 1.0], [639.5, 479.5]),  # on the axis
    (_FISHEYE, [0.0, 0.0, -1.0], None),  # opposite the axis
    (_FISHEYE, [1.0, 0.0, -1.8], None),  # θ ≈ 151° is past θ_max ≈ 136°
    (_SPHERICAL, [0.0, 0.0, -1.0], [-0.5, 511.5]),  # longitude π wraps onto the left edge
]


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def write_valid(out: Path) -> None:
    valid = out / "valid"
    valid.mkdir(parents=True)
    common = {"generator": "make_testdata.py"}
    omvg.write(
        valid / "reconstruction",
        reconstruction_tables(),
        profiles=["reconstruction"],
        length_unit="meter",
        dataset={
            "name": "OMVG conformance vector",
            "license": "CC0-1.0",
            "url": "https://example.org/omvg",
            "citation": "none",
        },
        **common,
    )
    # Same values: no compression, plain encodings, one row per row group, every all-null
    # column present, and columns in reverse order.
    plain = valid / "reconstruction_plain"
    omvg.write(
        plain,
        reconstruction_tables(),
        profiles=["reconstruction"],
        length_unit="meter",
        compression="none",
        encodings="plain",
        row_group_size=1,
        omit_null_columns=False,
        **common,
    )
    for name in omvg.PROFILES["reconstruction"]:
        table = pq.read_table(plain / f"{name}.parquet")
        reversed_table = table.select(table.column_names[::-1])
        omvg._write_parquet(
            reversed_table,
            plain / f"{name}.parquet",
            name,
            compression="none",
            encodings="plain",
            row_group_size=1,
        )
    omvg.write(valid / "full", full_tables(), profiles=FULL_PROFILES, **common)
    omvg.write(valid / "empty", empty_tables(), profiles=["dataset"], **common)
    omvg.write(
        valid / "extension",
        {**empty_tables(), "example_scores": example_scores()},
        profiles=["dataset"],
        extensions_used=EXTENSIONS,
        **common,
    )
    descriptions = {
        "reconstruction": "one camera of each model, a shared pose, an unregistered view, nulls, -0.0, a non-ASCII path, extra columns",
        "reconstruction_plain": "the values of reconstruction with no compression, plain encodings, one-row row groups, all-null columns present, and reversed column order",
        "full": "every table, every pair model including null, a pair with null inlier flags, partial ground truth (view 3 has none and no image)",
        "empty": "empty keypoints and matches tables",
        "extension": "a table of an unknown extension: no key, nulls, a fixed-size list hashed as LIST<DOUBLE>, an all-null column",
    }
    for vector in sorted(valid.iterdir()):
        expected = {"description": descriptions[vector.name]}
        (vector / "expected.json").write_text(
            json.dumps(expected, indent=2) + "\n", encoding="utf-8"
        )
    a = json.loads((valid / "reconstruction" / "manifest.json").read_text())
    b = json.loads((valid / "reconstruction_plain" / "manifest.json").read_text())
    assert (a["tables"], a["content_hash"]) == (b["tables"], b["content_hash"]), (
        "encoding changed a hash"
    )


def write_invalid(out: Path) -> None:
    invalid = out / "invalid"
    invalid.mkdir(parents=True)
    names = set()
    for name, rule, description, edit, options in INVALID:
        assert name not in names, name
        names.add(name)
        tables = canonical(full_tables())
        if edit is not None:
            edit(tables)
        directory = invalid / name
        options = dict(options)
        profiles = options.pop("profiles", FULL_PROFILES)
        raw_write(directory, tables, profiles, **options)
        (directory / "expected.json").write_text(
            json.dumps({"rule": rule, "description": description}, indent=2) + "\n",
            encoding="utf-8",
        )


def write_projection(out: Path) -> None:
    cases = []
    for camera, point, pixel in PROJECTIONS:
        found = omvg.project(camera, point)
        assert (found is None) == (pixel is None), (camera["model"], point, found)
        if pixel is not None:
            assert max(abs(f - p) for f, p in zip(found, pixel, strict=True)) <= 1e-9, (
                camera["model"],
                point,
                found,
                pixel,
            )
        cases.append({"camera": camera, "point": point, "pixel": pixel})
    (out / "projection.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    write_valid(out)
    write_invalid(out)
    write_projection(out)
    failures = omvg.conformance(out)
    for failure in failures:
        print(failure)
    print(
        f"{len(list((out / 'valid').iterdir()))} valid and {len(INVALID)} invalid vectors; "
        + ("all pass" if not failures else f"{len(failures)} failure(s)")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
