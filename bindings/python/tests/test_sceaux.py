# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Real-data contract on the Sceaux Castle reference dataset (11 images and a K.txt)."""

import os
import subprocess
from pathlib import Path

import pytest
from conftest import colmap_bytes
from openmvg_rs import reconstruct

IMAGES = 11
POINT_FLOOR = 4000


@pytest.mark.parametrize("engine", ["incremental", "global"])
def test_registers_every_image_deterministically(sceaux: Path, tmp_path: Path, engine: str) -> None:
    first = reconstruct(sceaux, engine=engine, progress=False)
    assert first.num_images == IMAGES
    assert first.num_registered_images == IMAGES
    assert first.intrinsics_mode == "fixed"
    assert first.num_points > POINT_FLOOR
    second = reconstruct(sceaux, engine=engine, progress=False)
    first.write_colmap(tmp_path / "first")
    second.write_colmap(tmp_path / "second")
    assert colmap_bytes(tmp_path / "first") == colmap_bytes(tmp_path / "second")


def test_matches_the_rust_example(sceaux: Path, tmp_path: Path) -> None:
    """The binding takes the library's own path: same defaults, byte-identical model."""
    example = os.environ.get("OPENMVG_SFM_EXAMPLE")
    if not example:
        if os.environ.get("CI"):
            pytest.fail("OPENMVG_SFM_EXAMPLE must name the built `sfm` example in CI")
        pytest.skip("set OPENMVG_SFM_EXAMPLE to the built `sfm` example to compare")
    subprocess.run([Path(example).resolve(), sceaux, tmp_path / "rust"], check=True)
    reconstruct(sceaux, progress=False).write_colmap(tmp_path / "python")
    assert colmap_bytes(tmp_path / "python") == colmap_bytes(tmp_path / "rust")
