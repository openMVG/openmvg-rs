# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Native extension of openmvg_rs; use the names re-exported by the package."""

import os
import pathlib
from typing import ClassVar, Literal, final

import numpy as np
import numpy.typing as npt

from ._scene import View

__version__: str
__all__ = [
    "DatasetError",
    "EstimationError",
    "ExportError",
    "NoReconstructionError",
    "OpenMVGError",
    "PlyFormat",
    "Reconstruction",
    "ReconstructionError",
    "__version__",
    "_restore_reconstruction",
    "estimate_absolute_pose",
    "estimate_essential",
    "estimate_fundamental",
    "estimate_homography",
    "load_reconstruction",
    "reconstruct",
]

def reconstruct(
    image_dir: str,
    engine: str,
    seed: int,
    intrinsics: str | None,
    progress: bool,
    max_workers: int | None,
    extraction_method: str = "alp",
) -> Reconstruction: ...
def load_reconstruction(path: str, /) -> Reconstruction: ...
def _restore_reconstruction(data: bytes, /) -> Reconstruction: ...
def estimate_fundamental(
    points1: npt.NDArray[np.float64],
    points2: npt.NDArray[np.float64],
    threshold: float,
    seed: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]: ...
def estimate_homography(
    points1: npt.NDArray[np.float64],
    points2: npt.NDArray[np.float64],
    threshold: float,
    seed: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]: ...
def estimate_essential(
    points1: npt.NDArray[np.float64],
    points2: npt.NDArray[np.float64],
    k1: npt.NDArray[np.float64],
    k2: npt.NDArray[np.float64],
    threshold: float,
    seed: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]: ...
def estimate_absolute_pose(
    points3d: npt.NDArray[np.float64],
    points2d: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
    threshold: float,
    seed: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]: ...

@final
class PlyFormat:
    """PLY storage encoding; binary output uses little-endian f32 coordinates."""

    Ascii: ClassVar[PlyFormat]
    Binary: ClassVar[PlyFormat]

@final
class Reconstruction:
    """An immutable sparse reconstruction returned by ``reconstruct``.

    It holds the poses of the registered views, the camera models, and the 3D points; read them
    as ``views`` and ``points``, write them out with ``write_colmap`` or ``write_ply``, or save
    them with ``save``.
    """

    @property
    def engine(self) -> Literal["incremental", "global"]:
        """The reconstruction engine: ``"incremental"`` or ``"global"``."""
    @property
    def seed(self) -> int:
        """The seed the run was started with."""
    @property
    def intrinsics_mode(self) -> Literal["fixed", "estimated"]:
        """``"fixed"`` when a calibration file supplied the camera intrinsics, which then stay
        unchanged; ``"estimated"`` when they were initialized from EXIF metadata and refined."""
    @property
    def num_images(self) -> int:
        """Number of views, one per image in the input folder."""
    @property
    def num_registered_images(self) -> int:
        """Number of registered views: input images with a reconstructed pose."""
    @property
    def num_points(self) -> int:
        """Number of reconstructed 3D points."""
    @property
    def points(self) -> npt.NDArray[np.float64]:
        """The reconstructed 3D points in world coordinates, a read-only ``(n, 3)`` array."""
    @property
    def views(self) -> tuple[View, ...]:
        """The registered views as ``View`` values, in the order of ``registered_images``."""
    @property
    def registered_images(self) -> tuple[pathlib.Path, ...]:
        """Image paths of the registered views, in input order."""
    @property
    def unregistered_images(self) -> tuple[pathlib.Path, ...]:
        """Image paths of the views that were not registered, in input order."""
    def write_colmap(self, directory: str | os.PathLike[str], /) -> None:
        """Write a COLMAP text model to ``directory/sparse/0/``.

        Missing directories are created. ``cameras.txt``, ``images.txt`` and ``points3D.txt``
        are replaced; other files are left alone.
        """
    def write_ply(
        self, path: str | os.PathLike[str], /, *, format: PlyFormat = PlyFormat.Binary
    ) -> None:
        """Write white 3D points and green registered camera centers as a PLY point cloud.

        Binary little-endian is the default; ``PlyFormat.Ascii`` selects text output.
        Missing parent directories are created and an existing file is replaced.
        """
    def save(self, path: str | os.PathLike[str], /) -> None:
        """Save the reconstruction to a binary file that ``load_reconstruction`` reads back.

        The parent directory must exist; an existing file is replaced atomically.
        Only OpenMVG-rs reads the file back, with a release that writes the same schema version;
        it is not an exchange format, so share results with ``write_colmap`` or ``write_ply``.
        The conventional extension is ``.omvgrs``.
        """
    def __reduce__(self) -> tuple[object, tuple[bytes]]:
        """Pickle as the bytes ``save`` writes."""
    def __copy__(self) -> Reconstruction:
        """A reconstruction is immutable, so a copy is the object itself."""
    def __deepcopy__(self, memo: object, /) -> Reconstruction:
        """A reconstruction is immutable, so a copy is the object itself."""

class OpenMVGError(Exception):
    """Base class of the errors raised by openmvg_rs."""

class DatasetError(OpenMVGError):
    """The image folder or its calibration could not be loaded."""

class ReconstructionError(OpenMVGError):
    """Feature matching or reconstruction failed."""

class NoReconstructionError(ReconstructionError):
    """The run completed without registering any view."""

class ExportError(OpenMVGError):
    """A reconstruction could not be written."""

class EstimationError(OpenMVGError):
    """No geometric model could be estimated from the correspondences."""
