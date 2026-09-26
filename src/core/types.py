"""Shared type aliases and small validation helpers."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


def as_float_array(values: npt.ArrayLike, *, name: str = "values") -> FloatArray:
    """Convert ``values`` to a 1-D float64 array, rejecting NaN and infinities.

    Args:
        values: Any array-like of numbers.
        name: Name used in error messages.

    Raises:
        ValueError: If the data is not one-dimensional or is not finite.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numbers")
    return array
