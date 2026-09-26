"""Rolling-origin evaluation and selection of forecasters across routes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from src.core.forecasting import (
    Forecaster,
    SimpleExponentialSmoothingForecaster,
    walk_forward_forecasts,
)
from src.core.metrics import mae
from src.core.types import FloatArray, as_float_array
from src.taxi.route import Route

ModelFactory = Callable[[], Forecaster]


@dataclass(frozen=True, slots=True)
class RouteBacktest:
    """Walk-forward forecasts for one route and every candidate model.

    Attributes:
        route: The route evaluated.
        first_origin: 0-based index of the first forecast day (3 = day 4).
        forecasts: One-step-ahead passenger forecasts per model name.
    """

    route: Route
    first_origin: int
    forecasts: dict[str, FloatArray]

    @property
    def actual(self) -> FloatArray:
        return self.route.passengers[self.first_origin :]

    def mae_passengers(self) -> dict[str, float]:
        return {m: mae(self.actual, f) for m, f in self.forecasts.items()}

    def mae_revenue(self) -> dict[str, float]:
        """MAE in UGX; revenue = passengers × fixed fare, so the ranking is identical."""
        return {m: v * self.route.fare for m, v in self.mae_passengers().items()}

    @property
    def best_model(self) -> str:
        scores = self.mae_passengers()
        return min(scores, key=scores.__getitem__)


def backtest_route(
    route: Route, factories: Sequence[ModelFactory], *, first_origin: int = 3
) -> RouteBacktest:
    """Walk-forward one-step forecasts for days ``first_origin+1 .. n``."""
    forecasts = {}
    for factory in factories:
        model = factory()
        forecasts[model.name] = walk_forward_forecasts(model, route.passengers, first_origin)
    return RouteBacktest(route, first_origin, forecasts)


def forecast_next_day(route: Route, factory: ModelFactory) -> tuple[float, Forecaster]:
    """Refit ``factory()`` on all days and forecast the next day's passengers."""
    model = factory().fit(route.passengers)
    return float(model.predict(1)[0]), model


def grid_search_alpha(
    values: Sequence[float] | FloatArray,
    *,
    first_origin: int = 3,
    grid: Sequence[float] = SimpleExponentialSmoothingForecaster.DEFAULT_GRID,
) -> tuple[float, dict[float, float]]:
    """Choose one global SES alpha by minimising walk-forward MAE.

    This uses the evaluation days themselves, so its MAE is optimistically
    biased; it is reported only to compare against the leak-free, per-origin
    tuning performed by ``SimpleExponentialSmoothingForecaster(alpha=None)``.

    Returns:
        The best alpha and the MAE for every alpha in ``grid``.
    """
    y = as_float_array(values)
    scores = {
        float(a): mae(
            y[first_origin:],
            walk_forward_forecasts(SimpleExponentialSmoothingForecaster(alpha=a), y, first_origin),
        )
        for a in grid
    }
    best = min(scores, key=scores.__getitem__)
    return best, scores


def simulate_weekly_demand(
    days: int = 60,
    *,
    base: float = 50.0,
    weekday_effects: Sequence[float] = (0.0, -2.0, 0.0, 2.0, 14.0, 6.0, -16.0),
    noise_sd: float = 3.0,
    seed: int = 11,
) -> FloatArray:
    """Synthetic daily passengers with a weekly cycle (day 0 is a Monday).

    Default effects make Friday the busiest and Sunday the quietest day.

    Raises:
        ValueError: If ``weekday_effects`` does not have seven entries.
    """
    if len(weekday_effects) != 7:
        raise ValueError("weekday_effects must have exactly seven entries")
    if days < 1:
        raise ValueError("days must be positive")
    rng = np.random.default_rng(seed)
    effects = np.asarray(weekday_effects, dtype=np.float64)[np.arange(days) % 7]
    counts = np.maximum(0.0, np.round(base + effects + rng.normal(0.0, noise_sd, days)))
    return np.asarray(counts, dtype=np.float64)
