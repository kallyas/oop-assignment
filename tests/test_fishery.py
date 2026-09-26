"""Tests for mini-project 3 (fish stock and export risk)."""

from __future__ import annotations

import numpy as np
import pytest

from src.fishery import (
    ClosedSeasonFishStock,
    CooperativeModel,
    FishStock,
    PriceModel,
    RiskAssessor,
    RiskLevel,
    run_scenarios,
)


def test_one_step_hand_calculation() -> None:
    # N1 = 4000 + 0.4*4000*(1 - 0.4) - 0.1*4000 = 4000 + 960 - 400 = 4560
    traj = FishStock(harvest_rate=0.1).simulate(1)
    assert traj.biomass[1] == pytest.approx(4560.0)
    assert traj.harvest[0] == pytest.approx(400.0)


@pytest.mark.parametrize("h", [0.05, 0.10, 0.20, 0.30])
def test_converges_to_analytic_equilibrium(h: float) -> None:
    stock = FishStock(harvest_rate=h)
    assert stock.simulate(400).final_stock == pytest.approx(stock.equilibrium_stock, rel=1e-6)


def test_msy_reached_at_half_r() -> None:
    stock = FishStock(harvest_rate=0.2)
    assert stock.msy == pytest.approx(1000.0)
    assert stock.equilibrium_yield == pytest.approx(stock.msy)


def test_zero_initial_stock_stays_extinct() -> None:
    assert FishStock(initial_stock=0.0).simulate(52).total_harvest == 0.0


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        FishStock(harvest_rate=1.2)
    with pytest.raises(ValueError):
        PriceModel(start=20_000)


def test_price_paths_stay_in_bounds_and_are_reproducible() -> None:
    model = PriceModel(step_sd=3_000.0, seed=5)
    paths = model.simulate(52, n_paths=200)
    assert paths.min() >= model.lower and paths.max() <= model.upper
    np.testing.assert_array_equal(paths, model.simulate(52, n_paths=200))


def test_risk_classification_uses_cv_not_variance() -> None:
    # Same relative dispersion at two very different scales -> same rating.
    small = RiskAssessor([90.0, 100.0, 110.0])
    large = RiskAssessor([9e9, 10e9, 11e9])
    assert small.cv == pytest.approx(large.cv)
    assert small.classify() is large.classify() is RiskLevel.MODERATE


def test_var_on_known_distribution() -> None:
    samples = np.arange(1, 101, dtype=float)  # 1..100
    var = RiskAssessor.value_at_risk(samples, 0.95)
    assert var.quantile == pytest.approx(np.quantile(samples, 0.05))
    assert var.var == pytest.approx(50.5 - var.quantile)


def test_closed_season_skips_harvest() -> None:
    stock = ClosedSeasonFishStock(closed_weeks=range(10, 18), harvest_rate=0.2)
    harvest = stock.simulate(60).harvest
    assert np.all(harvest[10:18] == 0.0) and np.all(harvest[52 + 10 : 60] == 0.0)
    assert harvest[9] > 0


def test_scenarios_share_price_paths() -> None:
    a, b = run_scenarios([0.1, 0.2], n_paths=50)
    # With common random numbers, revenue ratio equals catch ratio path by path.
    ratio = b.annual_revenue_samples / a.annual_revenue_samples
    assert np.ptp(ratio) < 0.05 * ratio.mean()
    assert CooperativeModel(FishStock(), PriceModel()).simulate().weekly_revenue.shape == (52,)
