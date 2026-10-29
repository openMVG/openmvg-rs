# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Sparse 3D reconstruction from a folder of images.

``reconstruct`` turns a folder of overlapping photographs into camera poses and a sparse point
cloud, which can be read as NumPy arrays or written as a COLMAP model or a PLY file::

    import openmvg_rs

    scene = openmvg_rs.reconstruct("ImageDataset_SceauxCastle/images")
    scene.points  # (n, 3) array of 3D points
    scene.views[0].pose.center  # where the first registered image was taken
    scene.write_colmap("sceaux")
    scene.write_ply("sceaux/points.ply")
    scene.save("sceaux/scene.omvgrs")

``load_reconstruction`` reads a saved reconstruction back, and pickling uses the same bytes.

``estimate_fundamental``, ``estimate_homography``, ``estimate_essential`` and
``estimate_absolute_pose`` robustly fit two-view geometry or a camera pose to NumPy arrays of
correspondences.
"""

import os
from importlib.metadata import version as _distribution_version
from typing import Literal, SupportsIndex

from . import _arguments
from ._geometry import (
    AbsolutePoseEstimate,
    EssentialEstimate,
    FundamentalEstimate,
    HomographyEstimate,
    Pose,
    estimate_absolute_pose,
    estimate_essential,
    estimate_fundamental,
    estimate_homography,
)
from ._native import (
    DatasetError,
    EstimationError,
    ExportError,
    NoReconstructionError,
    OpenMVGError,
    PlyFormat,
    Reconstruction,
    ReconstructionError,
)
from ._native import load_reconstruction as _load_reconstruction
from ._native import reconstruct as _reconstruct
from ._scene import Camera, View

# The installed distribution's version, which matches what pip reports for every release, release
# candidates included; the native module's `__version__` is the Rust crate's.
__version__ = _distribution_version("openmvg-rs")

__all__ = [
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

_ENGINES = ("incremental", "global")
_EXTRACTION_METHODS = ("sift", "alp")
_CAMERA_MODELS = ("pinhole", "pinhole-radial1", "pinhole-radial2", "spherical")

# Rayon's global thread pool does not survive fork(): a child forked after a run would wait
# forever for threads that do not exist in the child, so its runs use a private pool instead (0
# asks the native layer for one of the default size). A run with max_workers never touches the
# global pool.
_global_pool_started = False
_global_pool_lost = False


def _after_fork_in_child() -> None:
    global _global_pool_lost
    _global_pool_lost = _global_pool_started


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _path(name: str, value: str | os.PathLike[str]) -> str:
    path = os.fspath(value)
    if not isinstance(path, str):
        raise TypeError(f"{name} must be a str or os.PathLike[str], not {type(value).__name__}")
    return path


def reconstruct(
    image_dir: str | os.PathLike[str],
    /,
    *,
    engine: Literal["incremental", "global"] = "incremental",
    extraction_method: Literal["sift", "alp"] = "alp",
    seed: SupportsIndex = 42,
    intrinsics: str | os.PathLike[str] | None = None,
    camera_model: Literal[
        "pinhole", "pinhole-radial1", "pinhole-radial2", "spherical"
    ] = "pinhole-radial1",
    share_intrinsics: bool = False,
    progress: bool = True,
    max_workers: SupportsIndex | None = None,
) -> Reconstruction:
    """Reconstruct camera poses and 3D points from a folder of images.

    ``image_dir`` holds the images directly or in an ``images/`` subfolder. A ``K.txt`` file (a
    3x3 camera matrix) in ``image_dir`` or next to the images gives every image the same fixed
    intrinsics; otherwise they are initialized from EXIF metadata and refined during the run.
    An image without EXIF camera make and model gets a camera of its own unless
    ``share_intrinsics`` is set.

    Args:
        image_dir: The image folder.
        engine: ``"incremental"`` adds views one at a time; ``"global"`` estimates all their
            poses at once from the pairwise geometry.
        extraction_method: ``"alp"`` (the default) uses ALP High; ``"sift"`` uses SIFT
            High. Both emit RootSIFT-u8 descriptors. This choice is independent of the SfM engine.
        seed: Seed of every randomized step; the same inputs and seed give the same result.
            An integer in ``0 <= seed < 2**64``.
        intrinsics: A ``K.txt`` file to use instead of the one in the image folder.
        camera_model: Model of the cameras estimated from EXIF metadata: ``"pinhole"`` (no
            distortion), ``"pinhole-radial1"`` (the default) or ``"pinhole-radial2"`` (one or
            two radial coefficients, estimated during the run), or ``"spherical"`` for 2:1
            equirectangular images. A ``K.txt`` calibration keeps its own pinhole camera.
        share_intrinsics: Give the images with no EXIF camera make and model, and the same
            dimensions, one shared camera, as frames extracted from one video need; one focal
            length and distortion are then estimated from all of them. Use it only when they
            come from one camera at one zoom setting.
        progress: Draw progress bars on stderr when it is a terminal.
        max_workers: Maximum number of compute worker threads; ``None`` uses all cores.

    Returns:
        The reconstruction. Images whose views could not be registered are listed in
        ``unregistered_images``.

    Raises:
        TypeError: An argument has the wrong type.
        ValueError: ``engine``, ``extraction_method``, ``camera_model``, ``seed`` or
            ``max_workers`` has an invalid value.
        FileNotFoundError: ``image_dir``, an image or the calibration file does not exist.
        PermissionError: The operating system refuses to read an image or the calibration file.
        DatasetError: The folder has no usable images or the calibration is malformed.
        NoReconstructionError: No view could be registered.
        ReconstructionError: Matching or reconstruction failed otherwise.
        KeyboardInterrupt: The run was interrupted; it stops promptly and returns nothing.

    The GIL is released during the run. Ctrl-C cancels it when ``reconstruct`` is called from
    the main thread; from other threads, Python does not deliver signals and the run completes.
    A process forked after a run (multiprocessing's ``fork`` start method) can call it as well.
    """
    global _global_pool_started
    directory = _path("image_dir", image_dir)
    if engine not in _ENGINES:
        raise ValueError(f"engine must be 'incremental' or 'global', not {engine!r}")
    if not isinstance(extraction_method, str):
        raise TypeError("extraction_method must be a str")
    if extraction_method not in _EXTRACTION_METHODS:
        raise ValueError("extraction_method must be 'sift' or 'alp'")
    root_seed = _arguments.seed(seed)
    calibration = None if intrinsics is None else _path("intrinsics", intrinsics)
    if not isinstance(camera_model, str):
        raise TypeError(f"camera_model must be a str, not {type(camera_model).__name__}")
    if camera_model not in _CAMERA_MODELS:
        raise ValueError(
            "camera_model must be 'pinhole', 'pinhole-radial1', 'pinhole-radial2' or "
            f"'spherical', not {camera_model!r}"
        )
    if not isinstance(share_intrinsics, bool):
        raise TypeError(f"share_intrinsics must be a bool, not {type(share_intrinsics).__name__}")
    if not isinstance(progress, bool):
        raise TypeError(f"progress must be a bool, not {type(progress).__name__}")
    workers = None
    if max_workers is not None:
        workers = _arguments.integer("max_workers", max_workers)
        if workers < 1:
            raise ValueError(f"max_workers must be a positive integer or None, not {workers}")
    elif _global_pool_lost:
        workers = 0
    else:
        _global_pool_started = True
    return _reconstruct(
        directory,
        engine,
        root_seed,
        calibration,
        progress,
        workers,
        extraction_method,
        camera_model,
        share_intrinsics,
    )


def load_reconstruction(path: str | os.PathLike[str], /) -> Reconstruction:
    """Load a reconstruction written by ``Reconstruction.save``.

    The file records the image paths, cameras, poses, 3D points, seed, engine and intrinsics
    mode. Only a release that writes the same schema version reads it.

    Raises:
        TypeError: ``path`` is not a ``str`` or ``os.PathLike[str]``.
        FileNotFoundError: ``path`` does not exist.
        PermissionError: The operating system refuses to read ``path``.
        OpenMVGError: ``path`` is not a reconstruction file, was written by an incompatible
            version, or is corrupt.
    """
    return _load_reconstruction(_path("path", path))
