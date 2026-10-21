# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""The module surface, argument validation, and the Reconstruction object protocol."""

import copy
import importlib.metadata
import pickle
import re
import sys
import sysconfig
from pathlib import Path

import numpy
import openmvg_rs
import pytest
from openmvg_rs import (
    DatasetError,
    EstimationError,
    ExportError,
    NoReconstructionError,
    OpenMVGError,
    Reconstruction,
    ReconstructionError,
    reconstruct,
)


@pytest.fixture(scope="module")
def result(rendered: Path) -> Reconstruction:
    return reconstruct(rendered, progress=False)


def test_version_matches_the_distribution() -> None:
    assert openmvg_rs.__version__ == importlib.metadata.version("openmvg-rs")
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", openmvg_rs.__version__)


def test_public_names() -> None:
    assert sorted(openmvg_rs.__all__) == sorted(
        [
            "AbsolutePoseEstimate",
            "Camera",
            "DatasetError",
            "EssentialEstimate",
            "EstimationError",
            "ExportError",
            "FundamentalEstimate",
            "HomographyEstimate",
            "NoReconstructionError",
            "OpenMVGError",
            "PlyFormat",
            "Pose",
            "Reconstruction",
            "ReconstructionError",
            "View",
            "__version__",
            "estimate_absolute_pose",
            "estimate_essential",
            "estimate_fundamental",
            "estimate_homography",
            "load_reconstruction",
            "reconstruct",
        ]
    )
    for name in openmvg_rs.__all__:
        assert hasattr(openmvg_rs, name)


def test_exception_hierarchy() -> None:
    assert issubclass(OpenMVGError, Exception)
    for error in (DatasetError, ReconstructionError, ExportError, EstimationError):
        assert error.__bases__ == (OpenMVGError,)
    assert NoReconstructionError.__bases__ == (ReconstructionError,)
    for error in (
        OpenMVGError,
        DatasetError,
        ReconstructionError,
        NoReconstructionError,
        ExportError,
        EstimationError,
    ):
        assert error.__module__ == "openmvg_rs"


def test_free_threaded_build_keeps_the_gil_disabled() -> None:
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("not a free-threaded interpreter")
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"engine": "sequential"}, ValueError),
        ({"engine": None}, ValueError),
        ({"extraction_method": "unknown"}, ValueError),
        ({"extraction_method": None}, TypeError),
        ({"extraction_method": True}, TypeError),
        ({"seed": -1}, ValueError),
        ({"seed": 2**64}, ValueError),
        ({"seed": True}, TypeError),
        ({"seed": 1.0}, TypeError),
        ({"seed": "42"}, TypeError),
        ({"intrinsics": b"K.txt"}, TypeError),
        ({"intrinsics": 3}, TypeError),
        ({"progress": 1}, TypeError),
        ({"progress": None}, TypeError),
        ({"max_workers": 0}, ValueError),
        ({"max_workers": -2}, ValueError),
        ({"max_workers": True}, TypeError),
        ({"max_workers": 2.0}, TypeError),
    ],
)
def test_invalid_arguments(rendered: Path, arguments: dict[str, object], error: type) -> None:
    with pytest.raises(error):
        reconstruct(rendered, progress=False, **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["sift", "alp"])
def test_extraction_method_is_forwarded_without_changing_other_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    result = object()
    calls = []

    def native(*args: object) -> object:
        calls.append(args)
        return result

    monkeypatch.setattr(openmvg_rs, "_reconstruct", native)
    assert (
        reconstruct(tmp_path, extraction_method=method, seed=7, progress=False, max_workers=1)
        is result
    )
    assert calls == [(str(tmp_path), "incremental", 7, None, False, 1, method)]


def test_extraction_default_is_named_alp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(openmvg_rs, "_reconstruct", lambda *args: calls.append(args))
    reconstruct(tmp_path, progress=False, max_workers=1)
    assert calls[0][-1] == "alp"


def test_image_dir_must_be_a_text_path(rendered: Path) -> None:
    with pytest.raises(TypeError):
        reconstruct(bytes(rendered))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        reconstruct(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        reconstruct(image_dir=rendered)  # type: ignore[call-arg]


def test_any_integer_type_is_accepted(rendered: Path) -> None:
    result = reconstruct(rendered, seed=numpy.uint64(7), progress=False, max_workers=numpy.int8(2))
    assert result.seed == 7
    assert type(result.seed) is int


def test_seed_bounds_are_inclusive_exclusive(rendered: Path) -> None:
    assert reconstruct(rendered, seed=0, progress=False).seed == 0
    assert reconstruct(rendered, seed=2**64 - 1, progress=False).seed == 2**64 - 1


def test_reconstruction_is_not_constructible() -> None:
    with pytest.raises(TypeError):
        Reconstruction()


def test_reconstruction_is_read_only(result: Reconstruction) -> None:
    with pytest.raises(AttributeError):
        result.engine = "global"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        result.extra = 1  # type: ignore[attr-defined]


def test_pickle_keeps_the_run_details(result: Reconstruction) -> None:
    restored = pickle.loads(pickle.dumps(result))
    assert type(restored) is Reconstruction
    assert repr(restored) == repr(result)
    assert (restored.engine, restored.seed, restored.intrinsics_mode) == (
        "incremental",
        42,
        "fixed",
    )
    assert restored.registered_images == result.registered_images


def test_copies_are_the_object_itself(result: Reconstruction) -> None:
    assert copy.copy(result) is result
    assert copy.deepcopy(result) is result


def test_repr(result: Reconstruction) -> None:
    assert repr(result) == (
        f"<openmvg_rs.Reconstruction engine='incremental' "
        f"registered={result.num_registered_images}/{result.num_images} "
        f"points={result.num_points}>"
    )


def test_attribute_types(result: Reconstruction) -> None:
    assert result.engine == "incremental"
    assert result.seed == 42
    assert result.intrinsics_mode == "fixed"
    assert type(result.num_images) is int
    assert type(result.num_registered_images) is int
    assert type(result.num_points) is int
    assert isinstance(result.registered_images, tuple)
    assert isinstance(result.unregistered_images, tuple)
    assert all(isinstance(path, Path) for path in result.registered_images)
