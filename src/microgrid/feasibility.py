"""Policies for turning a mathematically valid dispatch into a physical one.

The linear solve can return negative energy (a battery "absorbing" daytime
load), which no real source can deliver. Policies are interchangeable
strategy objects so the notebook can compare them on the same data.

References:
    Lawson, C. L. & Hanson, R. J. (1995). *Solving Least Squares Problems*.
        SIAM (reprint of 1974 ed.), ch. 23 (non-negative least squares).
    Huangfu, Q. & Hall, J. A. J. (2018). Parallelizing the dual revised
        simplex method. *Mathematical Programming Computation*, 10, 119-142.
        (the HiGHS solver behind ``scipy.optimize.linprog``)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.optimize

from src.core.types import FloatArray
from src.microgrid.grid import MicroGrid


class FeasibilityPolicy(ABC):
    """Maps a raw day's solution to a non-negative dispatch."""

    name: str = "abstract"

    @abstractmethod
    def repair(self, grid: MicroGrid, raw: FloatArray, demand: FloatArray) -> FloatArray:
        """Return a non-negative dispatch for one day (``raw`` has a negative entry)."""


class ClipPolicy(FeasibilityPolicy):
    """Set negative sources to zero and keep the rest unchanged."""

    name = "clip"

    def repair(self, grid: MicroGrid, raw: FloatArray, demand: FloatArray) -> FloatArray:
        return np.clip(raw, 0.0, None)


class NNLSPolicy(FeasibilityPolicy):
    """Non-negative least squares: the non-negative dispatch closest to meeting demand."""

    name = "nnls"

    def repair(self, grid: MicroGrid, raw: FloatArray, demand: FloatArray) -> FloatArray:
        solution, _ = scipy.optimize.nnls(grid.coefficients, demand)
        return np.asarray(solution, dtype=np.float64)


class LeastCostPolicy(FeasibilityPolicy):
    """Cheapest non-negative dispatch that meets *every* demand (``A s >= d``).

    Solved as a linear programme with :func:`scipy.optimize.linprog`. Unlike
    NNLS, which trades under-supply against over-supply symmetrically, this
    never leaves load unserved; any surplus on a constraint is curtailed. For
    a health centre, losing power to a vaccine fridge is far more serious
    than curtailing a few kWh of surplus.

    Raises:
        RuntimeError: If no non-negative dispatch can cover the demand.
    """

    name = "least-cost LP"

    def repair(self, grid: MicroGrid, raw: FloatArray, demand: FloatArray) -> FloatArray:
        result = scipy.optimize.linprog(
            c=grid.tariffs,
            A_ub=-grid.coefficients,
            b_ub=-demand,
            bounds=(0.0, None),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"no non-negative dispatch covers demand {demand}: {result.message}")
        return np.asarray(result.x, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    """Result of dispatching many days under a feasibility policy.

    All matrices have shape ``(n_sources, n_days)`` except where noted.
    """

    grid: MicroGrid
    demands: FloatArray
    raw: FloatArray
    dispatch: FloatArray
    policy_name: str

    @property
    def infeasible(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of days whose raw solution had a negative source (tolerance 1e-9)."""
        return np.asarray(np.any(self.raw < -1e-9, axis=0))

    @property
    def infeasible_days(self) -> list[int]:
        return [int(i) for i in np.flatnonzero(self.infeasible)]

    @property
    def residual(self) -> FloatArray:
        """Demand minus delivered load per constraint (positive = unmet demand, kWh)."""
        return self.demands - self.grid.coefficients @ self.dispatch

    @property
    def unmet(self) -> FloatArray:
        """Demand left unserved per constraint (kWh, >= 0)."""
        return np.clip(self.residual, 0.0, None)

    @property
    def curtailed(self) -> FloatArray:
        """Energy delivered beyond demand per constraint (kWh, >= 0)."""
        return np.clip(-self.residual, 0.0, None)

    @property
    def daily_cost(self) -> FloatArray:
        """UGX per day (length ``n_days``)."""
        return self.grid.daily_cost(self.dispatch)

    @property
    def total_cost(self) -> float:
        return float(self.daily_cost.sum())


def dispatch_with_policy(grid: MicroGrid, demands: FloatArray, policy: FeasibilityPolicy) -> DispatchPlan:
    """Solve every day vectorised, then repair infeasible days with ``policy``."""
    raw = grid.solve_days(demands)
    repaired = raw.copy()
    for j in np.flatnonzero(np.any(raw < -1e-9, axis=0)):
        repaired[:, j] = policy.repair(grid, raw[:, j], demands[:, j])
    return DispatchPlan(grid, demands, raw, repaired, policy.name)
