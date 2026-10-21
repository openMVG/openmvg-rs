# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Argument checks shared by the public functions."""

import operator
from typing import SupportsIndex


def integer(name: str, value: SupportsIndex) -> int:
    # Any integer type is accepted (NumPy's included); bool is one, but True as a seed or a thread
    # count is almost certainly a mistake.
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    return operator.index(value)


def seed(value: SupportsIndex) -> int:
    root_seed = integer("seed", value)
    if not 0 <= root_seed < 2**64:
        raise ValueError(f"seed must satisfy 0 <= seed < 2**64, not {root_seed}")
    return root_seed
