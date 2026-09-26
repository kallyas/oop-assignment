"""The :class:`Region` domain object: twelve months of rainfall."""

from __future__ import annotations

import calendar

import numpy as np
import numpy.typing as npt

from src.core.stats import DescriptiveStats
from src.core.types import FloatArray, as_float_array

MONTHS: tuple[str, ...] = tuple(calendar.month_abbr[1:])


class Region:
    """Mean monthly rainfall (mm) for one region, January to December.

    Args:
        name: Region name.
        monthly_rainfall: Twelve non-negative values in mm.

    Raises:
        ValueError: If there are not exactly twelve finite, non-negative values.
    """

    def __init__(self, name: str, monthly_rainfall: npt.ArrayLike) -> None:
        if not name.strip():
            raise ValueError("region name must be non-empty")
        rain = as_float_array(monthly_rainfall, name="monthly_rainfall")
        if rain.size != 12:
            raise ValueError(f"expected 12 monthly values, got {rain.size}")
        if np.any(rain < 0):
            raise ValueError("rainfall cannot be negative")
        rain.setflags(write=False)
        self._name = name
        self._rain = rain

    def __repr__(self) -> str:
        return f"Region(name={self._name!r}, annual_total={self.annual_total():,.0f} mm)"

    def __len__(self) -> int:
        return 12

    def __getitem__(self, month: int | str) -> float:
        """Rainfall for a month, by 0-based index or abbreviation (``"Mar"``)."""
        idx = MONTHS.index(month) if isinstance(month, str) else month
        return float(self._rain[idx])

    @property
    def name(self) -> str:
        return self._name

    @property
    def rainfall(self) -> FloatArray:
        """Read-only array of the twelve monthly totals (mm)."""
        return self._rain

    def annual_total(self) -> float:
        return float(self._rain.sum())

    def mean(self) -> float:
        return float(self._rain.mean())

    def wettest_month(self) -> str:
        return MONTHS[int(np.argmax(self._rain))]

    def driest_month(self) -> str:
        return MONTHS[int(np.argmin(self._rain))]

    def coefficient_of_variation(self) -> float:
        """Seasonality index: sample stdev / mean of the monthly values.

        Raises:
            ZeroDivisionError: For a region with no rain at all.
        """
        return DescriptiveStats.from_statistics(self._rain).cv
