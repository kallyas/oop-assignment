"""Point-forecast accuracy metrics implemented from first principles.

References:
    Hyndman, R. J. & Koehler, A. B. (2006). Another look at measures of
        forecast accuracy. *International Journal of Forecasting*, 22(4),
        679-688.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.core.types import FloatArray, as_float_array


def _paired(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    a = as_float_array(actual, name="actual")
    p = as_float_array(predicted, name="predicted")
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs predicted {p.shape}")
    if a.size == 0:
        raise ValueError("cannot score an empty forecast")
    return a, p


def mae(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Mean absolute error, in the units of the data."""
    a, p = _paired(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Root mean squared error; penalises large misses more heavily than MAE."""
    a, p = _paired(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Mean absolute percentage error, expressed in percent.

    Raises:
        ValueError: If any actual value is zero, where MAPE is undefined.
    """
    a, p = _paired(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(100.0 * np.mean(np.abs((a - p) / a)))
