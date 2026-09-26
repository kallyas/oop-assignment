"""Tests for mini-project 5 (taxi route planner)."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.forecasting import MovingAverageForecaster, SimpleExponentialSmoothingForecaster
from src.taxi import (
    FleetPlan,
    LinearCurve,
    LinearMarket,
    Route,
    backtest_route,
    grid_search_alpha,
    simulate_weekly_demand,
)
from src.taxi.data import load_routes


def test_route_revenue_and_statistics() -> None:
    ntinda = load_routes()[0]
    assert ntinda.total_revenue() == pytest.approx(474 * 2_000)
    assert ntinda.statistics().mean == pytest.approx(47.4)


@pytest.mark.parametrize(("counts", "fare"), [([], 2000.0), ([10, -1], 2000.0), ([10], 0.0)])
def test_route_validation(counts: list[int], fare: float) -> None:
    with pytest.raises(ValueError):
        Route("Bad", counts, fare)


def test_equilibrium_hand_calculation() -> None:
    # 120 - 0.02P = 10 + 0.03P  ->  P = 2200, Q = 76
    market = LinearMarket(LinearCurve(120.0, -0.02), LinearCurve(10.0, 0.03))
    eq = market.equilibrium()
    assert eq.price == pytest.approx(2200.0)
    assert eq.quantity == pytest.approx(76.0)
    assert market.excess_demand(2000.0) == pytest.approx(10.0)


def test_market_rejects_wrong_slopes() -> None:
    with pytest.raises(ValueError):
        LinearMarket(LinearCurve(120.0, 0.02), LinearCurve(10.0, 0.03))


def test_backtest_moving_average_first_forecast() -> None:
    ntinda = load_routes()[0]
    bt = backtest_route(ntinda, [lambda: MovingAverageForecaster(3)])
    first = bt.forecasts["3-day moving average"][0]
    assert first == pytest.approx((35 + 40 + 42) / 3)
    assert len(bt.actual) == 7  # days 4..10


def test_grid_search_returns_alpha_from_grid() -> None:
    best, scores = grid_search_alpha(load_routes()[0].passengers)
    assert best in SimpleExponentialSmoothingForecaster.DEFAULT_GRID
    assert scores[best] == min(scores.values())


def test_fleet_plan_rounds_up_with_buffer() -> None:
    plan = FleetPlan("R", forecast_passengers=200.0)
    assert plan.capacity_per_vehicle == 112
    assert plan.exact_vehicles == pytest.approx(230 / 112)
    assert plan.vehicles == 3


def test_fleet_plan_zero_demand_keeps_minimum_service() -> None:
    assert FleetPlan("Quiet", forecast_passengers=0.0).vehicles == 1


def test_simulated_week_has_friday_peak() -> None:
    y = simulate_weekly_demand(70, noise_sd=0.0)
    by_weekday = y.reshape(-1, 7).mean(axis=0)
    assert int(np.argmax(by_weekday)) == 4 and int(np.argmin(by_weekday)) == 6
