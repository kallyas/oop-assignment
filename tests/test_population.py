"""Tests for mini-project 1 (district population forecaster)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.population import (
    ClassroomPlan,
    DistrictPopulation,
    ExponentialCAGRForecaster,
    FibonacciRatioForecaster,
    LinearTrendForecaster,
    bootstrap_interval,
    fibonacci_numbers,
    validate_models,
)
from src.population.data import load_districts


@pytest.fixture
def kampala() -> DistrictPopulation:
    return next(d for d in load_districts() if d.name == "Kampala")


def test_repr_and_len(kampala: DistrictPopulation) -> None:
    assert len(kampala) == 10
    assert "Kampala" in repr(kampala) and "2015-2024" in repr(kampala)


@pytest.mark.parametrize(
    ("years", "pops", "message"),
    [
        ([2020, 2021], [1.0], "differ in length"),
        ([2020, 2021], [1.0, -2.0], "negative"),
        ([], [], "at least one"),
        ([2020, 2022], [1.0, 2.0], "consecutive"),
    ],
)
def test_validation_rejects_bad_input(years: list[int], pops: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DistrictPopulation("X", years, pops)


def test_arrays_are_read_only(kampala: DistrictPopulation) -> None:
    with pytest.raises(ValueError):
        kampala.populations[0] = 0.0


def test_growth_rates_and_cagr_hand_calculation() -> None:
    d = DistrictPopulation("Test", [2020, 2021, 2022], [100.0, 110.0, 121.0])
    np.testing.assert_allclose(d.growth_rates(), [0.10, 0.10])
    assert d.cagr() == pytest.approx(0.10)


def test_growth_undefined_for_zero_base() -> None:
    with pytest.raises(ValueError, match="zero"):
        DistrictPopulation("Empty", [2020, 2021], [0.0, 5.0]).cagr()


def test_cagr_forecaster_extrapolates_geometrically() -> None:
    model = ExponentialCAGRForecaster().fit([100.0, 110.0, 121.0], [2020, 2021, 2022])
    np.testing.assert_allclose(model.predict(2), [133.1, 146.41])


def test_fibonacci_forecaster_uses_successive_ratios() -> None:
    assert fibonacci_numbers(7) == [1, 1, 2, 3, 5, 8, 13]
    model = FibonacciRatioForecaster().fit([100.0])
    # Ratios 1/1, 2/1, 3/2 -> 100, 200, 300.
    np.testing.assert_allclose(model.predict(3), [100.0, 200.0, 300.0])


def test_validation_selects_exact_linear_model() -> None:
    d = DistrictPopulation("Linear", range(2015, 2025), [100 + 10 * i for i in range(10)])
    report = validate_models(
        d, [LinearTrendForecaster, ExponentialCAGRForecaster], last_train_year=2021
    )
    assert report.best.model_name == "Linear trend"
    assert report.best.mae == pytest.approx(0.0, abs=1e-8)


def test_bootstrap_interval_brackets_point_forecast(kampala: DistrictPopulation) -> None:
    model = LinearTrendForecaster().fit(kampala.populations, kampala.years)
    lower, upper = bootstrap_interval(model, 5, n_bootstrap=300, rng=np.random.default_rng(0))
    point = model.predict(5)
    assert np.all(lower < point) and np.all(point < upper)
    # Uncertainty should not shrink as we look further ahead.
    assert (upper - lower)[-1] > (upper - lower)[0]


def test_classroom_plan_hand_calculation() -> None:
    plan = ClassroomPlan("X", 2024, 2029, 100.0, 110.0, 0.18, 53)
    assert plan.classrooms_base == math.ceil(18_000 / 53)
    assert plan.classrooms_target == math.ceil(19_800 / 53)
    assert plan.additional_classrooms == plan.classrooms_target - plan.classrooms_base


def test_classroom_plan_never_negative() -> None:
    assert ClassroomPlan("Shrinking", 2024, 2029, 100.0, 90.0, 0.18, 53).additional_classrooms == 0
