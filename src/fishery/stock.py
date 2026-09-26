"""Discrete logistic fish-stock dynamics with proportional harvesting.

References:
    Verhulst, P.-F. (1838). Notice sur la loi que la population suit dans son
        accroissement. *Correspondance mathématique et physique*, 10, 113-121.
    Schaefer, M. B. (1954). Some aspects of the dynamics of populations
        important to the management of the commercial marine fisheries.
        *Bulletin of the Inter-American Tropical Tuna Commission*, 1(2), 27-56.
    Clark, C. W. (1990). *Mathematical Bioeconomics* (2nd ed.). Wiley, ch. 1
        (equilibrium yield and maximum sustainable yield).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from src.core.types import FloatArray

WEEKS_PER_YEAR = 52


@dataclass(frozen=True, slots=True)
class StockTrajectory:
    """Output of :meth:`FishStock.simulate`.

    Attributes:
        biomass: Stock (tonnes) at the start of each week, length ``weeks + 1``.
        harvest: Catch (tonnes) taken during each week, length ``weeks``.
    """

    biomass: FloatArray
    harvest: FloatArray

    @property
    def final_stock(self) -> float:
        return float(self.biomass[-1])

    @property
    def total_harvest(self) -> float:
        return float(self.harvest.sum())


class FishStock:
    """Logistic growth with proportional harvesting (Schaefer surplus-production model).

    ``N(t+1) = N(t) + r N(t) (1 - N(t)/K) - h N(t)``, floored at zero.

    Args:
        r: Intrinsic growth rate per week.
        carrying_capacity: ``K`` in tonnes.
        initial_stock: ``N(0)`` in tonnes.
        harvest_rate: Fraction ``h`` of the stock caught per week.

    Raises:
        ValueError: If a parameter is outside its meaningful range.
    """

    def __init__(
        self,
        r: float = 0.4,
        carrying_capacity: float = 10_000.0,
        initial_stock: float = 4_000.0,
        harvest_rate: float = 0.1,
    ) -> None:
        if r <= 0:
            raise ValueError("growth rate r must be positive")
        if carrying_capacity <= 0:
            raise ValueError("carrying capacity K must be positive")
        if initial_stock < 0:
            raise ValueError("initial stock cannot be negative")
        if not 0.0 <= harvest_rate < 1.0:
            raise ValueError("harvest rate h must lie in [0, 1)")
        self.r = r
        self.carrying_capacity = carrying_capacity
        self.initial_stock = initial_stock
        self.harvest_rate = harvest_rate

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(r={self.r}, K={self.carrying_capacity:,.0f}, "
            f"N0={self.initial_stock:,.0f}, h={self.harvest_rate})"
        )

    # ----------------------------------------------------------- dynamics
    def harvest_rate_at(self, week: int) -> float:
        """Harvest fraction applied in ``week`` (0-based). Subclasses may vary it."""
        return self.harvest_rate

    def simulate(self, weeks: int = WEEKS_PER_YEAR) -> StockTrajectory:
        """Iterate the model for ``weeks`` steps.

        Raises:
            ValueError: If ``weeks`` is negative.
        """
        if weeks < 0:
            raise ValueError("weeks cannot be negative")
        biomass = np.empty(weeks + 1)
        harvest = np.empty(weeks)
        biomass[0] = self.initial_stock
        r, k = self.r, self.carrying_capacity
        for t in range(weeks):
            n = biomass[t]
            harvest[t] = self.harvest_rate_at(t) * n
            biomass[t + 1] = max(0.0, n + r * n * (1.0 - n / k) - harvest[t])
        return StockTrajectory(biomass, harvest)

    # ------------------------------------------------------- equilibrium
    @property
    def equilibrium_stock(self) -> float:
        """Non-trivial steady state ``K (1 - h / r)`` (0 if ``h >= r``)."""
        return max(0.0, self.carrying_capacity * (1.0 - self.harvest_rate / self.r))

    @property
    def equilibrium_yield(self) -> float:
        """Weekly catch at the steady state, ``h * N*`` (tonnes/week)."""
        return self.harvest_rate * self.equilibrium_stock

    @property
    def msy(self) -> float:
        """Maximum sustainable yield ``r K / 4`` (tonnes/week), reached at ``h = r/2``."""
        return self.r * self.carrying_capacity / 4.0

    @property
    def msy_harvest_rate(self) -> float:
        return self.r / 2.0

    @property
    def is_sustainable(self) -> bool:
        """True if the harvest rate does not exceed the MSY rate ``r/2``.

        Rates in ``(r/2, r)`` still converge to a positive stock, but a smaller
        one that yields *less* than MSY: the fishery is growth-overfished.
        """
        return self.harvest_rate <= self.msy_harvest_rate


class ClosedSeasonFishStock(FishStock):
    """A fish stock with an annual closed season (no harvesting in given weeks).

    Args:
        closed_weeks: Week-of-year indices (0-51) in which fishing is banned.
        r, carrying_capacity, initial_stock, harvest_rate: As for :class:`FishStock`.
    """

    def __init__(
        self,
        closed_weeks: Iterable[int],
        *,
        r: float = 0.4,
        carrying_capacity: float = 10_000.0,
        initial_stock: float = 4_000.0,
        harvest_rate: float = 0.1,
    ) -> None:
        super().__init__(r, carrying_capacity, initial_stock, harvest_rate)
        weeks = frozenset(closed_weeks)
        if any(not 0 <= w < WEEKS_PER_YEAR for w in weeks):
            raise ValueError(f"closed weeks must lie in 0..{WEEKS_PER_YEAR - 1}")
        self.closed_weeks = weeks

    def harvest_rate_at(self, week: int) -> float:
        return 0.0 if week % WEEKS_PER_YEAR in self.closed_weeks else self.harvest_rate

    def __repr__(self) -> str:
        return f"{super().__repr__()[:-1]}, closed_weeks={len(self.closed_weeks)}/yr)"
