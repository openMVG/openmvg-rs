# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Robust estimation of two-view geometry and of a camera pose from correspondences.

The estimators take real-valued NumPy arrays of pixel coordinates (and 3D points for
``estimate_absolute_pose``), convert them to ``float64``, copy them into the library's types,
and run with the GIL released. Thresholds are in pixels. The same inputs and ``seed`` give the
same result.
"""

import math
from dataclasses import dataclass, fields
from typing import Any, SupportsIndex, TypeVar

import numpy as np
import numpy.typing as npt

from . import _arguments
from ._native import estimate_absolute_pose as _estimate_absolute_pose
from ._native import estimate_essential as _estimate_essential
from ._native import estimate_fundamental as _estimate_fundamental
from ._native import estimate_homography as _estimate_homography

__all__ = [
    "AbsolutePoseEstimate",
    "EssentialEstimate",
    "FundamentalEstimate",
    "HomographyEstimate",
    "Pose",
    "estimate_absolute_pose",
    "estimate_essential",
    "estimate_fundamental",
    "estimate_homography",
]

_Array = TypeVar("_Array", bound=np.ndarray[Any, Any])


def _frozen(array: _Array) -> _Array:
    array.flags.writeable = False
    return array


def _restore_readonly_state(value: Any, state: list[Any]) -> None:
    for member, item in zip(fields(value), state, strict=True):
        if isinstance(item, np.ndarray):
            item = _frozen(item)
        object.__setattr__(value, member.name, item)


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class Pose:
    """A rigid transform into a camera's frame: ``X_cam = rotation @ X_ref + translation``.

    The reference frame is the world for ``View.pose`` and ``AbsolutePoseEstimate.pose``, and the
    first camera for ``EssentialEstimate.pose``, whose translation has unit length because the
    scale is unknown. The camera frame is right-handed, with ``+z`` into the scene, ``+x`` right
    and ``+y`` down, matching the image axes.

    Attributes:
        rotation: The 3x3 rotation from the reference frame to the camera's, a read-only copy.
        translation: The translation, 3 values, a read-only copy.
    """

    rotation: npt.NDArray[np.float64]
    translation: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        for name, shape in (("rotation", (3, 3)), ("translation", (3,))):
            array = np.array(_real_array(name, getattr(self, name)))
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, not {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, _frozen(array))

    @property
    def center(self) -> npt.NDArray[np.float64]:
        """The camera center in the reference frame, ``-rotation.T @ translation``; for an
        ``EssentialEstimate``, the unit direction from the first camera to the second."""
        return _frozen(-self.rotation.T @ self.translation)

    def __repr__(self) -> str:
        x, y, z = self.center
        return f"<openmvg_rs.Pose center=({x:.6g}, {y:.6g}, {z:.6g})>"

    __setstate__ = _restore_readonly_state


def _summary(estimate: Any) -> str:
    mask = estimate.inlier_mask
    return f"<openmvg_rs.{type(estimate).__name__} inliers={np.count_nonzero(mask)}/{len(mask)}>"


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class FundamentalEstimate:
    """A fundamental matrix and its inliers, returned by ``estimate_fundamental``.

    Attributes:
        matrix: The 3x3 fundamental matrix ``F``, with ``x2^T F x1 = 0`` for homogeneous pixels.
        inlier_mask: ``True`` for each inlier correspondence.
    """

    matrix: npt.NDArray[np.float64]
    inlier_mask: npt.NDArray[np.bool_]
    __repr__ = _summary
    __setstate__ = _restore_readonly_state


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class HomographyEstimate:
    """A homography and its inliers, returned by ``estimate_homography``.

    Attributes:
        matrix: The 3x3 homography ``H``, with ``x2 ~ H x1`` for homogeneous pixels.
        inlier_mask: ``True`` for each inlier correspondence.
    """

    matrix: npt.NDArray[np.float64]
    inlier_mask: npt.NDArray[np.bool_]
    __repr__ = _summary
    __setstate__ = _restore_readonly_state


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class EssentialEstimate:
    """An essential matrix, the relative pose it encodes, and its inliers.

    Returned by ``estimate_essential``. A point ``X1`` in the first camera's frame is
    ``X2 = pose.rotation @ X1 + s * pose.translation`` in the second camera's frame, for an
    unknown scale ``s > 0``.

    Attributes:
        matrix: The 3x3 essential matrix ``E``, with ``y2^T E y1 = 0`` for normalized
            coordinates ``y = K^-1 x``.
        pose: The transform from the first camera's frame to the second's; its translation has
            unit length.
        inlier_mask: ``True`` for each inlier correspondence.
    """

    matrix: npt.NDArray[np.float64]
    pose: Pose
    inlier_mask: npt.NDArray[np.bool_]
    __repr__ = _summary
    __setstate__ = _restore_readonly_state


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class AbsolutePoseEstimate:
    """A camera pose and its inliers, returned by ``estimate_absolute_pose``.

    Attributes:
        pose: The world-to-camera pose of the camera.
        inlier_mask: ``True`` for each inlier correspondence.
    """

    pose: Pose
    inlier_mask: npt.NDArray[np.bool_]
    __repr__ = _summary
    __setstate__ = _restore_readonly_state


# Pickles and reprs name the public module, so this file can move without breaking them.
for _result in (
    Pose,
    FundamentalEstimate,
    HomographyEstimate,
    EssentialEstimate,
    AbsolutePoseEstimate,
):
    _result.__module__ = "openmvg_rs"


def _real_array(name: str, value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or (
        raw.dtype.kind == "O" and any(np.iscomplexobj(item) for item in raw.flat)
    ):
        raise TypeError(f"{name} must contain only real values")
    return np.asarray(raw, dtype=np.float64)


def _points(name: str, value: npt.ArrayLike, dimension: int) -> npt.NDArray[np.float64]:
    array = _real_array(name, value)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(f"{name} must have shape (n, {dimension}), not {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _same_length(
    name1: str, array1: npt.NDArray[Any], name2: str, array2: npt.NDArray[Any]
) -> None:
    if len(array1) != len(array2):
        raise ValueError(f"{name1} and {name2} differ in length: {len(array1)} and {len(array2)}")


def _camera_matrix(name: str, value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    k = _real_array(name, value)
    if k.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), not {k.shape}")
    if not np.isfinite(k).all():
        raise ValueError(f"{name} must be finite")
    if k[0, 1] != 0 or k[1, 0] != 0 or k[2, 0] != 0 or k[2, 1] != 0 or k[2, 2] != 1:
        raise ValueError(
            f"{name} must be [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]: zero skew, last row [0, 0, 1]"
        )
    if not (k[0, 0] > 0 and k[1, 1] > 0):
        raise ValueError(f"{name} must have positive focal lengths")
    return k


def _threshold(value: float) -> float:
    threshold = float(value)
    if not (math.isfinite(threshold) and threshold > 0):
        raise ValueError(f"threshold must be a positive number of pixels, not {threshold}")
    return threshold


def estimate_fundamental(
    points1: npt.ArrayLike,
    points2: npt.ArrayLike,
    /,
    *,
    threshold: float = 4.0,
    seed: SupportsIndex = 42,
) -> FundamentalEstimate:
    """Estimate the fundamental matrix of two uncalibrated views.

    Args:
        points1: ``(n, 2)`` pixel coordinates in the first image.
        points2: ``(n, 2)`` pixel coordinates of the same points in the second image.
        threshold: Inlier threshold on the Sampson distance, in pixels.
        seed: Seed of the random sampling, an integer in ``0 <= seed < 2**64``.

    Returns:
        A ``FundamentalEstimate`` with the 3x3 matrix and the inlier mask, as read-only arrays.

    Raises:
        TypeError: An argument has the wrong type.
        ValueError: An argument has the wrong shape or value.
        EstimationError: Too few correspondences, or no model has enough support.
    """
    a, b = _points("points1", points1, 2), _points("points2", points2, 2)
    _same_length("points1", a, "points2", b)
    matrix, mask = _estimate_fundamental(a, b, _threshold(threshold), _arguments.seed(seed))
    return FundamentalEstimate(_frozen(matrix), _frozen(mask))


def estimate_homography(
    points1: npt.ArrayLike,
    points2: npt.ArrayLike,
    /,
    *,
    threshold: float = 4.0,
    seed: SupportsIndex = 42,
) -> HomographyEstimate:
    """Estimate the homography mapping the first image's points onto the second's.

    Args:
        points1: ``(n, 2)`` pixel coordinates in the first image.
        points2: ``(n, 2)`` pixel coordinates of the same points in the second image.
        threshold: Inlier threshold on the symmetric transfer error, in pixels.
        seed: Seed of the random sampling, an integer in ``0 <= seed < 2**64``.

    Returns:
        A ``HomographyEstimate`` with the 3x3 matrix and the inlier mask, as read-only arrays.

    Raises:
        TypeError: An argument has the wrong type.
        ValueError: An argument has the wrong shape or value.
        EstimationError: Too few correspondences, or no model has enough support.
    """
    a, b = _points("points1", points1, 2), _points("points2", points2, 2)
    _same_length("points1", a, "points2", b)
    matrix, mask = _estimate_homography(a, b, _threshold(threshold), _arguments.seed(seed))
    return HomographyEstimate(_frozen(matrix), _frozen(mask))


def estimate_essential(
    points1: npt.ArrayLike,
    points2: npt.ArrayLike,
    camera_matrix1: npt.ArrayLike,
    camera_matrix2: npt.ArrayLike | None = None,
    /,
    *,
    threshold: float = 4.0,
    seed: SupportsIndex = 42,
) -> EssentialEstimate:
    """Estimate the essential matrix and relative pose of two calibrated views.

    The five-point solver generates the hypotheses; the pose is the decomposition of the final
    essential matrix that puts the most inliers in front of both cameras.

    Args:
        points1: ``(n, 2)`` pixel coordinates in the first image.
        points2: ``(n, 2)`` pixel coordinates of the same points in the second image.
        camera_matrix1: The first camera matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
        camera_matrix2: The second camera matrix; ``None`` uses ``camera_matrix1``.
        threshold: Inlier threshold on the Sampson distance, in pixels of the camera with the
            smallest focal length.
        seed: Seed of the random sampling, an integer in ``0 <= seed < 2**64``.

    Returns:
        An ``EssentialEstimate`` with the 3x3 matrix, the relative pose with a unit
        translation, and the inlier mask, as read-only arrays.

    Raises:
        TypeError: An argument has the wrong type.
        ValueError: An argument has the wrong shape or value.
        EstimationError: Too few correspondences, no model has enough support, or no pose puts
            the inliers in front of both cameras.
    """
    a, b = _points("points1", points1, 2), _points("points2", points2, 2)
    _same_length("points1", a, "points2", b)
    k1 = _camera_matrix("camera_matrix1", camera_matrix1)
    k2 = k1 if camera_matrix2 is None else _camera_matrix("camera_matrix2", camera_matrix2)
    matrix, rotation, translation, mask = _estimate_essential(
        a, b, k1, k2, _threshold(threshold), _arguments.seed(seed)
    )
    return EssentialEstimate(_frozen(matrix), Pose(rotation, translation), _frozen(mask))


def estimate_absolute_pose(
    points3d: npt.ArrayLike,
    points2d: npt.ArrayLike,
    camera_matrix: npt.ArrayLike,
    /,
    *,
    threshold: float = 4.0,
    seed: SupportsIndex = 42,
) -> AbsolutePoseEstimate:
    """Estimate the pose of a calibrated camera from 2D-3D correspondences.

    The P3P solver generates the hypotheses, and the pose is refined on the reprojection error
    of its inliers.

    Args:
        points3d: ``(n, 3)`` world points.
        points2d: ``(n, 2)`` pixel coordinates where the camera observes them.
        camera_matrix: The camera matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
        threshold: Inlier threshold in pixels, converted to an angular error through the
            smaller focal length.
        seed: Seed of the random sampling, an integer in ``0 <= seed < 2**64``.

    Returns:
        An ``AbsolutePoseEstimate`` with the world-to-camera pose and the inlier mask, as
        read-only arrays.

    Raises:
        TypeError: An argument has the wrong type.
        ValueError: An argument has the wrong shape or value.
        EstimationError: Too few correspondences, or no pose has enough support.
    """
    world, pixels = _points("points3d", points3d, 3), _points("points2d", points2d, 2)
    _same_length("points3d", world, "points2d", pixels)
    k = _camera_matrix("camera_matrix", camera_matrix)
    rotation, translation, mask = _estimate_absolute_pose(
        world, pixels, k, _threshold(threshold), _arguments.seed(seed)
    )
    return AbsolutePoseEstimate(Pose(rotation, translation), _frozen(mask))
