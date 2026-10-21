# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""The registered views of a reconstruction: image, pose and camera.

``Reconstruction.views`` builds these values from the native reconstruction; they hold copies,
so they stay valid on their own.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from ._geometry import Pose, _frozen

__all__ = ["Camera", "View"]


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class Camera:
    """A camera model and its parameters, shared by the views that use the same camera.

    Pixel coordinates put the center of the top-left pixel at ``(0, 0)``, with ``x`` to the right
    and ``y`` down. COLMAP puts the same point at ``(0.5, 0.5)``, so ``write_colmap`` adds 0.5 to
    ``cx`` and ``cy``.

    Attributes:
        model: ``"PINHOLE"``, ``"BROWN_CONRADY"``, ``"FISHEYE"`` or ``"SPHERICAL"``, the names of
            the OMVG format's ``cameras`` table.
        width: The image width in pixels.
        height: The image height in pixels.
        parameters: The model's parameters by name, read-only: ``fx``, ``fy`` (focal lengths in
            pixels) and ``cx``, ``cy`` (principal point) for every model but ``"SPHERICAL"``,
            which has none; the radial coefficients ``k1``, ``k2``, ``k3`` and tangential ``p1``,
            ``p2`` that a ``"BROWN_CONRADY"`` camera models; and ``k1`` to ``k4`` for a
            ``"FISHEYE"`` camera. A coefficient the camera does not model is absent.
    """

    model: str
    width: int
    height: int
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def matrix(self) -> npt.NDArray[np.float64] | None:
        """The 3x3 camera matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``; ``None`` for a
        ``"SPHERICAL"`` camera."""
        if "fx" not in self.parameters:
            return None
        p = self.parameters
        return _frozen(np.array([[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1.0]]))

    def __repr__(self) -> str:
        return f"<openmvg_rs.Camera model={self.model!r} size={self.width}x{self.height}>"

    def __reduce__(self) -> tuple[Any, ...]:
        # A mappingproxy does not pickle; rebuild through __init__ from a plain dict.
        return (Camera, (self.model, self.width, self.height, dict(self.parameters)))


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class View:
    """A registered image, the pose the reconstruction gave it, and its camera.

    Attributes:
        image: The path of the image file.
        pose: The world-to-camera pose: ``X_cam = pose.rotation @ X_world + pose.translation``.
        camera: The camera model; views that share a camera share this object.
    """

    image: Path
    pose: Pose
    camera: Camera

    def __repr__(self) -> str:
        return f"<openmvg_rs.View image={self.image.name!r} camera={self.camera.model!r}>"


# Pickles and reprs name the public module, so this file can move without breaking them.
for _value in (Camera, View):
    _value.__module__ = "openmvg_rs"
