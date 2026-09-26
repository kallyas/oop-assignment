"""Bounded random-walk model of the weekly fish price."""

from __future__ import annotations

import numpy as np

from src.core.types import FloatArray


class PriceModel:
    """Seeded Gaussian random walk in UGX/kg, reflected at price bounds.

    ``p_0 = start`` and ``p_t = reflect(p_{t-1} + e_t)`` with ``e_t ~ N(0, sigma)``.
    Reflection (rather than clipping) avoids the price "sticking" at a bound.

    Args:
        start: Initial price (UGX/kg).
        lower: Lower bound (UGX/kg).
        upper: Upper bound (UGX/kg).
        step_sd: Standard deviation of the weekly change (UGX/kg).
        seed: Seed for :func:`numpy.random.default_rng`.

    Raises:
        ValueError: If the bounds are inconsistent or ``step_sd`` is negative.
    """

    def __init__(
        self,
        start: float = 12_000.0,
        lower: float = 9_000.0,
        upper: float = 16_000.0,
        step_sd: float = 400.0,
        seed: int = 42,
    ) -> None:
        if not lower < upper:
            raise ValueError("lower bound must be below upper bound")
        if not lower <= start <= upper:
            raise ValueError("start price must lie within the bounds")
        if step_sd < 0:
            raise ValueError("step_sd cannot be negative")
        self.start = start
        self.lower = lower
        self.upper = upper
        self.step_sd = step_sd
        self.seed = seed

    def __repr__(self) -> str:
        return (
            f"PriceModel(start={self.start:,.0f}, bounds=({self.lower:,.0f}, {self.upper:,.0f}), "
            f"step_sd={self.step_sd:,.0f}, seed={self.seed})"
        )

    def reflect(self, prices: FloatArray) -> FloatArray:
        """Fold values back into ``[lower, upper]`` as if bouncing off the bounds."""
        width = self.upper - self.lower
        y = np.mod(prices - self.lower, 2.0 * width)
        return self.lower + np.where(y > width, 2.0 * width - y, y)

    def simulate(
        self, weeks: int, n_paths: int = 1, rng: np.random.Generator | None = None
    ) -> FloatArray:
        """Simulate ``n_paths`` price paths of length ``weeks``.

        Args:
            weeks: Number of weekly prices per path (the first equals ``start``).
            n_paths: Number of independent paths.
            rng: Generator to draw from; defaults to a fresh one seeded with ``seed``
                so repeated calls are reproducible.

        Returns:
            Array of shape ``(n_paths, weeks)``.
        """
        if weeks < 1 or n_paths < 1:
            raise ValueError("weeks and n_paths must be positive")
        gen = rng if rng is not None else np.random.default_rng(self.seed)
        steps = gen.normal(0.0, self.step_sd, size=(n_paths, weeks - 1))
        paths = np.empty((n_paths, weeks))
        paths[:, 0] = self.start
        for t in range(1, weeks):
            paths[:, t] = self.reflect(paths[:, t - 1] + steps[:, t - 1])
        return paths
