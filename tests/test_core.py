"""Tests for the shared statistics, metrics and forecasting framework."""

from __future__ import annotations

import statistics

import numpy as np
import pytest

from src.core.forecasting import (
    LinearTrendForecaster,
    MovingAverageForecaster,
    NotFittedError,
    SeasonalNaiveForecaster,
    SimpleExponentialSmoothingForecaster,
    walk_forward_forecasts,
)
from src.core.metrics import mae, mape, rmse
from src.core.stats import DescriptiveStats


def test_statistics_and_numpy_agree_when_ddof_matches() -> None:
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    s = DescriptiveStats.from_statistics(data, ddof=1)
    n = DescriptiveStats.from_numpy(data, ddof=1)
    assert s.variance == pytest.approx(n.variance)
    assert s.variance == pytest.approx(statistics.variance(data))
    # Population variance of this classic example is exactly 4.
    assert DescriptiveStats.from_numpy(data, ddof=0).variance == pytest.approx(4.0)


def test_stats_reject_too_few_observations() -> None:
    with pytest.raises(ValueError):
        DescriptiveStats.from_statistics([5.0], ddof=1)


def test_metrics_hand_calculation() -> None:
    actual, pred = [100.0, 200.0], [110.0, 180.0]
    assert mae(actual, pred) == pytest.approx(15.0)
    assert rmse(actual, pred) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert mape(actual, pred) == pytest.approx(10.0)


def test_mape_undefined_for_zero_actual() -> None:
    with pytest.raises(ValueError, match="zero"):
        mape([0.0, 1.0], [1.0, 1.0])


def test_linear_trend_recovers_exact_line() -> None:
    model = LinearTrendForecaster().fit([3.0, 5.0, 7.0], [2000, 2001, 2002])
    np.testing.assert_allclose(model.predict(2), [9.0, 11.0])
    np.testing.assert_allclose(model.future_times(2), [2003, 2004])


def test_predict_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        MovingAverageForecaster(3).predict(1)


def test_ses_alpha_one_is_naive_forecast() -> None:
    model = SimpleExponentialSmoothingForecaster(alpha=1.0).fit([1.0, 5.0, 9.0])
    assert model.predict(1)[0] == pytest.approx(9.0)


def test_ses_hand_calculation() -> None:
    # l0 = 10, l1 = 0.5*20 + 0.5*10 = 15, l2 = 0.5*30 + 0.5*15 = 22.5
    model = SimpleExponentialSmoothingForecaster(alpha=0.5).fit([10.0, 20.0, 30.0])
    assert model.predict(3).tolist() == [22.5, 22.5, 22.5]


def test_seasonal_naive_repeats_last_season() -> None:
    model = SeasonalNaiveForecaster(period=3).fit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert model.predict(4).tolist() == [4.0, 5.0, 6.0, 4.0]


def test_walk_forward_uses_only_past_data() -> None:
    fc = walk_forward_forecasts(MovingAverageForecaster(3), [1, 2, 3, 100, 5], first_origin=3)
    # Day-4 forecast must not see the 100 it is predicting.
    assert fc.tolist() == [2.0, 35.0]


def test_invalid_hyperparameters_and_history() -> None:
    with pytest.raises(ValueError):
        SimpleExponentialSmoothingForecaster(alpha=0.0)
    with pytest.raises(ValueError):
        MovingAverageForecaster(3).fit([1.0, 2.0])
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit([1.0, 2.0, 3.0], [2000, 2001, 2003])
