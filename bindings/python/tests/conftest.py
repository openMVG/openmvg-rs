# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

import http.client
import importlib.util
import os
import shutil
from pathlib import Path
from types import ModuleType

import pytest
from synthetic import render_scene

COLMAP_FILES = ("cameras.txt", "images.txt", "points3D.txt")


def colmap_bytes(directory: Path) -> dict[str, bytes]:
    model = directory / "sparse" / "0"
    return {name: (model / name).read_bytes() for name in COLMAP_FILES}


@pytest.fixture(scope="session")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic scene, rendered once per session; tests must not modify it."""
    directory = tmp_path_factory.mktemp("rendered")
    render_scene(directory)
    return directory


@pytest.fixture
def scene(rendered: Path, tmp_path: Path) -> Path:
    """A private copy of the synthetic scene that a test may modify."""
    return Path(shutil.copytree(rendered, tmp_path / "scene"))


def _load_sceaux() -> ModuleType:
    """The repository's Sceaux fetcher, shared with the examples."""
    path = Path(__file__).resolve().parents[3] / "scripts" / "sceaux.py"
    spec = importlib.util.spec_from_file_location("sceaux", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCEAUX = _load_sceaux()


@pytest.fixture(scope="session")
def sceaux() -> Path:
    """The Sceaux Castle images, fetched and verified by ``scripts/sceaux.py``.

    Without network access the tests are skipped, except in CI where they fail.
    """
    try:
        images: Path = SCEAUX.fetch()
    except SCEAUX.ChecksumError as error:
        pytest.fail(str(error))
    except (OSError, http.client.HTTPException) as error:
        if os.environ.get("CI"):
            pytest.fail(f"cannot download the Sceaux dataset: {error}")
        pytest.skip(f"Sceaux dataset unavailable offline: {error}")
    return images
