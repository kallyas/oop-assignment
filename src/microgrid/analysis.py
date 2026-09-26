"""Performance timing and Monte Carlo sensitivity analysis for a micro-grid."""

from __future__ import annotations

import timeit
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.core.types import FloatArray
from src.microgrid.grid import MicroGrid


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Best-of-``repeat`` wall time per call for the two solving strategies."""

    loop_seconds: float
    vectorised_seconds: float
    n_days: int

    @property
    def speedup(self) -> float:
        return self.loop_seconds / self.vectorised_seconds


def time_solvers(
    grid: MicroGrid, demands: FloatArray, *, number: int = 200, repeat: int = 5
) -> TimingResult:
    """Time the per-day loop against the single vectorised solve with :mod:`timeit`.

    The minimum over ``repeat`` runs is reported, as recommended by the
    ``timeit`` docs, because larger values reflect interference from other
    processes rather than the code under test.
    """
    loop = min(timeit.repeat(lambda: grid.solve_days_loop(demands), number=number, repeat=repeat))
    vec = min(timeit.repeat(lambda: grid.solve_days(demands), number=number, repeat=repeat))
    return TimingResult(loop / number, vec / number, int(demands.shape[1]))


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """Monte Carlo response of the dispatch to perturbed demand.

    Attributes:
        base_demand: Unperturbed demand vector.
        base_solution: Dispatch for ``base_demand``.
        demands: Perturbed demands, shape ``(n_draws, n_constraints)``.
        solutions: Corresponding dispatches, shape ``(n_draws, n_sources)``.
    """

    base_demand: FloatArray
    base_solution: FloatArray
    demands: FloatArray
    solutions: FloatArray

    @property
    def relative_source_change(self) -> FloatArray:
        """Per-draw, per-source relative change ``(s - s0) / |s0|``."""
        change = (self.solutions - self.base_solution) / np.abs(self.base_solution)
        return np.asarray(change, dtype=np.float64)

    @property
    def amplification(self) -> FloatArray:
        """Per-draw ratio of relative solution change to relative demand change (2-norms).

        Linear-algebra theory bounds this by ``cond(A)``.
        """
        rel_x = np.linalg.norm(self.solutions - self.base_solution, axis=1) / np.linalg.norm(self.base_solution)
        rel_b = np.linalg.norm(self.demands - self.base_demand, axis=1) / np.linalg.norm(self.base_demand)
        return np.asarray(rel_x / rel_b, dtype=np.float64)


def monte_carlo_sensitivity(
    grid: MicroGrid,
    base_demand: npt.ArrayLike,
    *,
    relative_perturbation: float = 0.05,
    n_draws: int = 1000,
    rng: np.random.Generator,
) -> SensitivityResult:
    """Perturb each demand independently by ``U(-p, +p)`` and re-solve.

    Raises:
        ValueError: If ``relative_perturbation`` is not in ``(0, 1)`` or ``n_draws < 1``.
    """
    if not 0.0 < relative_perturbation < 1.0:
        raise ValueError("relative_perturbation must lie in (0, 1)")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    b0 = np.asarray(base_demand, dtype=np.float64)
    factors = 1.0 + rng.uniform(-relative_perturbation, relative_perturbation, (n_draws, b0.size))
    demands = b0 * factors
    solutions = grid.solve_days(demands.T).T
    return SensitivityResult(b0, grid.solve_day(*b0), demands, solutions)
