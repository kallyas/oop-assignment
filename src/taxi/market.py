"""Linear supply-demand equilibrium solved as a 2×2 linear system.

References:
    Mankiw, N. G. (2021). *Principles of Economics* (9th ed.). Cengage,
        ch. 4 (market equilibrium, shortages and surpluses).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from src.core.types import FloatArray


@dataclass(frozen=True, slots=True)
class LinearCurve:
    """``Q = intercept + slope * P`` (slope < 0 for demand, > 0 for supply)."""

    intercept: float
    slope: float

    def quantity(self, price: float) -> float:
        return self.intercept + self.slope * price


@dataclass(frozen=True, slots=True)
class Equilibrium:
    price: float
    quantity: float


class LinearMarket:
    """A market with linear demand and supply curves.

    Writing ``Q - slope * P = intercept`` for each curve gives the system::

        [ -b_d  1 ] [P]   [a_d]
        [ -b_s  1 ] [Q] = [a_s]

    Raises:
        ValueError: If demand does not slope down or supply does not slope up.
    """

    def __init__(self, demand: LinearCurve, supply: LinearCurve) -> None:
        if demand.slope >= 0:
            raise ValueError("demand must slope downwards")
        if supply.slope <= 0:
            raise ValueError("supply must slope upwards")
        self.demand = demand
        self.supply = supply

    @property
    def system(self) -> tuple[FloatArray, FloatArray]:
        """The coefficient matrix ``A`` and right-hand side ``b``."""
        a = np.array([[-self.demand.slope, 1.0], [-self.supply.slope, 1.0]])
        b = np.array([self.demand.intercept, self.supply.intercept])
        return a, b

    def equilibrium(self) -> Equilibrium:
        a, b = self.system
        price, quantity = scipy.linalg.solve(a, b)
        return Equilibrium(float(price), float(quantity))

    def excess_demand(self, price: float) -> float:
        """``Qd - Qs`` at ``price``: positive means a shortage of seats."""
        return self.demand.quantity(price) - self.supply.quantity(price)
