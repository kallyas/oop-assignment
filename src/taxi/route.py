"""The :class:`Route` domain object for a matatu route."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.core.stats import DescriptiveStats
from src.core.types import FloatArray, as_float_array


class Route:
    """Daily passenger counts and a flat fare for one taxi route.

    Args:
        name: Route name, e.g. ``"Kampala-Ntinda"``.
        passengers: Passengers carried per day (non-negative).
        fare: Fare per passenger in UGX (positive).

    Raises:
        ValueError: On an empty series, negative counts or a non-positive fare.
    """

    def __init__(self, name: str, passengers: npt.ArrayLike, fare: float) -> None:
        if not name.strip():
            raise ValueError("route name must be non-empty")
        counts = as_float_array(passengers, name="passengers")
        if counts.size == 0:
            raise ValueError("a route needs at least one day of data")
        if np.any(counts < 0):
            raise ValueError("passenger counts cannot be negative")
        if fare <= 0:
            raise ValueError("fare must be positive")
        counts.setflags(write=False)
        self._name = name
        self._passengers = counts
        self._fare = float(fare)

    def __repr__(self) -> str:
        return f"Route(name={self._name!r}, days={len(self)}, fare=UGX {self._fare:,.0f})"

    def __len__(self) -> int:
        return int(self._passengers.size)

    @property
    def name(self) -> str:
        return self._name

    @property
    def passengers(self) -> FloatArray:
        return self._passengers

    @property
    def fare(self) -> float:
        return self._fare

    def daily_revenue(self) -> FloatArray:
        """UGX collected each day (passengers × fare)."""
        return self._passengers * self._fare

    def total_revenue(self) -> float:
        return float(self.daily_revenue().sum())

    def statistics(self) -> DescriptiveStats:
        """Passenger mean/median/variance/stdev via the ``statistics`` module.

        Raises:
            ValueError: With fewer than two days (sample variance undefined).
        """
        return DescriptiveStats.from_statistics(self._passengers)
