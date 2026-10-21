# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Saving, loading, and pickling a Reconstruction."""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import colmap_bytes
from openmvg_rs import (
    ExportError,
    OpenMVGError,
    Reconstruction,
    load_reconstruction,
    reconstruct,
)


@pytest.fixture(scope="module")
def result(rendered: Path) -> Reconstruction:
    return reconstruct(rendered, engine="global", seed=7, progress=False)


def test_save_and_load_round_trip(result: Reconstruction, tmp_path: Path) -> None:
    path = tmp_path / "scene.omvgrs"
    result.save(path)
    loaded = load_reconstruction(str(path))
    assert loaded.seed == 7
    assert loaded.engine == "global"
    assert loaded.intrinsics_mode == "fixed"
    assert loaded.num_images == result.num_images
    assert loaded.num_registered_images == result.num_registered_images
    assert loaded.num_points == result.num_points
    assert loaded.registered_images == result.registered_images
    assert loaded.unregistered_images == result.unregistered_images
    assert repr(loaded) == repr(result)

    # The exports carry every camera, pose and point, so equal exports mean equal scenes.
    result.write_colmap(tmp_path / "saved")
    loaded.write_colmap(tmp_path / "loaded")
    assert colmap_bytes(tmp_path / "loaded") == colmap_bytes(tmp_path / "saved")

    loaded.save(tmp_path / "again.omvgrs")
    assert (tmp_path / "again.omvgrs").read_bytes() == path.read_bytes()

    np.testing.assert_array_equal(loaded.points, result.points)
    for mine, theirs in zip(loaded.views, result.views, strict=True):
        assert mine.image == theirs.image
        np.testing.assert_array_equal(mine.pose.rotation, theirs.pose.rotation)
        np.testing.assert_array_equal(mine.pose.translation, theirs.pose.translation)
        assert dict(mine.camera.parameters) == dict(theirs.camera.parameters)


def test_pickle_round_trip(result: Reconstruction, tmp_path: Path) -> None:
    restored = pickle.loads(pickle.dumps(result))
    assert (restored.engine, restored.seed, restored.intrinsics_mode) == ("global", 7, "fixed")
    result.write_colmap(tmp_path / "original")
    restored.write_colmap(tmp_path / "restored")
    assert colmap_bytes(tmp_path / "restored") == colmap_bytes(tmp_path / "original")


def test_views_are_immutable_and_pickle(result: Reconstruction) -> None:
    view = result.views[0]
    restored = pickle.loads(pickle.dumps(view))
    assert restored.image == view.image
    assert dict(restored.camera.parameters) == dict(view.camera.parameters)
    np.testing.assert_array_equal(restored.pose.rotation, view.pose.rotation)
    for array in (restored.pose.rotation, restored.pose.translation, restored.camera.matrix):
        assert array is not None
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0
    with pytest.raises(TypeError):
        view.camera.parameters["fx"] = 1.0  # type: ignore[index]
    assert repr(view) == f"<openmvg_rs.View image={view.image.name!r} camera='PINHOLE'>"


def test_corrupt_file_raises(result: Reconstruction, tmp_path: Path) -> None:
    path = tmp_path / "scene.omvgrs"
    result.save(path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(OpenMVGError, match="checksum"):
        load_reconstruction(path)

    path.write_bytes(b"not a reconstruction")
    with pytest.raises(OpenMVGError, match="not an OpenMVG-rs reconstruction file"):
        load_reconstruction(path)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "missing.omvgrs"
    with pytest.raises(FileNotFoundError) as caught:
        load_reconstruction(missing)
    assert caught.value.filename == str(missing)
    with pytest.raises(TypeError):
        load_reconstruction(b"scene.omvgrs")  # type: ignore[arg-type]


def test_save_errors(result: Reconstruction, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        result.save(tmp_path / "missing" / "scene.omvgrs")
    assert caught.value.filename == str(tmp_path / "missing" / "scene.omvgrs")
    with pytest.raises(ExportError):
        result.save(tmp_path)


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions as non-root"
)
def test_refused_access_raises_permission_error(result: Reconstruction, tmp_path: Path) -> None:
    path = tmp_path / "scene.omvgrs"
    result.save(path)
    path.chmod(0o000)
    try:
        with pytest.raises(PermissionError) as caught:
            load_reconstruction(path)
        assert caught.value.filename == str(path)
    finally:
        path.chmod(0o600)

    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        with pytest.raises(PermissionError) as caught:
            result.save(locked / "scene.omvgrs")
        assert caught.value.filename == str(locked / "scene.omvgrs")
    finally:
        locked.chmod(0o700)
