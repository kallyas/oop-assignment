"""The :class:`DistrictPopulation` domain object."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np
import numpy.typing as npt

from src.core.stats import DescriptiveStats
from src.core.types import FloatArray, as_float_array


class DistrictPopulation:
    """An annual population series for one district.

    Populations are stored in **thousands of people**, following the UBOS
    figures supplied with the assignment. The arrays are exposed read-only so a
    ``DistrictPopulation`` cannot be silently corrupted after validation.

    Args:
        name: District name, e.g. ``"Kampala"``.
        years: Calendar years, strictly increasing and consecutive.
        populations: Population (thousands) for each year.

    Raises:
        ValueError: On empty input, unequal lengths, negative populations or
            non-consecutive years.
    """

    def __init__(self, name: str, years: npt.ArrayLike, populations: npt.ArrayLike) -> None:
        if not name.strip():
            raise ValueError("district name must be a non-empty string")
        year_arr = np.asarray(years, dtype=np.int64)
        pop_arr = as_float_array(populations, name="populations")
        if year_arr.ndim != 1:
            raise ValueError("years must be one-dimensional")
        if year_arr.size == 0:
            raise ValueError("a district needs at least one observation")
        if year_arr.size != pop_arr.size:
            raise ValueError(
                f"years ({year_arr.size}) and populations ({pop_arr.size}) differ in length"
            )
        if np.any(pop_arr < 0):
            raise ValueError("populations cannot be negative")
        if year_arr.size > 1 and np.any(np.diff(year_arr) != 1):
            raise ValueError("years must be consecutive and increasing")
        year_arr.setflags(write=False)
        pop_arr.setflags(write=False)
        self._name = name
        self._years = year_arr
        self._populations = pop_arr

    # ------------------------------------------------------------ dunders
    def __repr__(self) -> str:
        return (
            f"DistrictPopulation(name={self._name!r}, "
            f"years={int(self._years[0])}-{int(self._years[-1])}, "
            f"n={len(self)}, latest={self._populations[-1]:,.0f}k)"
        )

    def __len__(self) -> int:
        return int(self._populations.size)

    def __iter__(self) -> Iterator[tuple[int, float]]:
        """Iterate over ``(year, population)`` pairs."""
        return zip(self._years.tolist(), self._populations.tolist())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DistrictPopulation):
            return NotImplemented
        return (
            self._name == other._name
            and np.array_equal(self._years, other._years)
            and np.array_equal(self._populations, other._populations)
        )

    # --------------------------------------------------------- properties
    @property
    def name(self) -> str:
        return self._name

    @property
    def years(self) -> npt.NDArray[np.int64]:
        """Read-only array of calendar years."""
        return self._years

    @property
    def populations(self) -> FloatArray:
        """Read-only array of populations in thousands."""
        return self._populations

    # --------------------------------------------------------- statistics
    def statistics_summary(self, *, ddof: Literal[0, 1] = 1) -> DescriptiveStats:
        """Mean, median, variance and stdev via the ``statistics`` module."""
        return DescriptiveStats.from_statistics(self._populations, ddof=ddof)

    def numpy_summary(self, *, ddof: Literal[0, 1] = 0) -> DescriptiveStats:
        """The same statistics via NumPy (``ddof=0`` is NumPy's default)."""
        return DescriptiveStats.from_numpy(self._populations, ddof=ddof)

    # ------------------------------------------------------------- growth
    def growth_rates(self) -> FloatArray:
        """Year-on-year growth rates ``(P_t - P_{t-1}) / P_{t-1}`` as fractions.

        Raises:
            ValueError: If there are fewer than two years or a zero population.
        """
        self._require_growth_computable()
        p = self._populations
        return np.diff(p) / p[:-1]

    def cagr(self) -> float:
        """Compound annual growth rate ``(P_end / P_start) ** (1 / years) - 1``."""
        self._require_growth_computable()
        p = self._populations
        return float((p[-1] / p[0]) ** (1.0 / (p.size - 1)) - 1.0)

    # ----------------------------------------------------------- slicing
    def split(self, last_train_year: int) -> tuple[DistrictPopulation, DistrictPopulation]:
        """Split into a training series (``<= last_train_year``) and a test series.

        Raises:
            ValueError: If either side of the split would be empty.
        """
        mask = self._years <= last_train_year
        if mask.all() or not mask.any():
            raise ValueError(f"split at {last_train_year} leaves an empty train or test set")
        return (
            DistrictPopulation(self._name, self._years[mask], self._populations[mask]),
            DistrictPopulation(self._name, self._years[~mask], self._populations[~mask]),
        )

    def _require_growth_computable(self) -> None:
        if len(self) < 2:
            raise ValueError("growth rates need at least two years of data")
        if np.any(self._populations[:-1] == 0):
            raise ValueError("growth rates are undefined when a base-year population is zero")
