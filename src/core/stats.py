"""Descriptive statistics computed two independent ways.

The assignment repeatedly asks for the same four summary statistics, computed
with the standard-library :mod:`statistics` module and cross-checked with NumPy.
Keeping the calculation in one place keeps every mini-project consistent.

References:
    Python Software Foundation. ``statistics`` -- Mathematical statistics
        functions. https://docs.python.org/3/library/statistics.html
    NumPy Developers. ``numpy.var`` (the ``ddof`` argument).
        https://numpy.org/doc/stable/reference/generated/numpy.var.html
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from src.core.types import as_float_array


@dataclass(frozen=True, slots=True)
class DescriptiveStats:
    """Immutable bundle of location and dispersion statistics.

    Attributes:
        n: Number of observations.
        mean: Arithmetic mean.
        median: Median.
        variance: Variance (sample or population, see ``ddof``).
        stdev: Standard deviation, consistent with ``variance``.
        ddof: Delta degrees of freedom used for the variance (1 = sample).
    """

    n: int
    mean: float
    median: float
    variance: float
    stdev: float
    ddof: Literal[0, 1]

    @property
    def cv(self) -> float:
        """Coefficient of variation (stdev / mean), a unit-free dispersion measure.

        Raises:
            ZeroDivisionError: If the mean is zero, where the CV is undefined.
        """
        if self.mean == 0:
            raise ZeroDivisionError("coefficient of variation is undefined for a zero mean")
        return self.stdev / abs(self.mean)

    @classmethod
    def from_statistics(
        cls, data: npt.ArrayLike, *, ddof: Literal[0, 1] = 1
    ) -> DescriptiveStats:
        """Compute the statistics with the standard-library ``statistics`` module.

        ``statistics.variance``/``stdev`` use the sample (n - 1) denominator,
        whereas ``pvariance``/``pstdev`` use the population (n) denominator.

        Raises:
            ValueError: If there are too few observations for the chosen ``ddof``.
        """
        values = [float(v) for v in _validated(data, ddof)]
        if ddof == 1:
            variance, stdev = statistics.variance(values), statistics.stdev(values)
        else:
            variance, stdev = statistics.pvariance(values), statistics.pstdev(values)
        return cls(
            n=len(values),
            mean=statistics.fmean(values),
            median=float(statistics.median(values)),
            variance=float(variance),
            stdev=float(stdev),
            ddof=ddof,
        )

    @classmethod
    def from_numpy(cls, data: npt.ArrayLike, *, ddof: Literal[0, 1] = 1) -> DescriptiveStats:
        """Compute the statistics with NumPy.

        ``np.var`` defaults to ``ddof=0`` (population variance); pass ``ddof=1``
        to reproduce ``statistics.variance``.
        """
        values = _validated(data, ddof)
        return cls(
            n=int(values.size),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            variance=float(np.var(values, ddof=ddof)),
            stdev=float(np.std(values, ddof=ddof)),
            ddof=ddof,
        )

    def as_dict(self) -> dict[str, float]:
        """Return the statistics as a plain dictionary (handy for tables)."""
        return {
            "n": float(self.n),
            "mean": self.mean,
            "median": self.median,
            "variance": self.variance,
            "stdev": self.stdev,
        }


def _validated(data: npt.ArrayLike, ddof: int) -> npt.NDArray[np.float64]:
    values = as_float_array(data, name="data")
    if values.size <= ddof:
        raise ValueError(f"need more than {ddof} observation(s), got {values.size}")
    return values
