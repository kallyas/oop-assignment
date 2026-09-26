"""Domain-agnostic building blocks shared by several mini-projects."""

from src.core.forecasting import (
    Forecaster,
    LinearTrendForecaster,
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    SimpleExponentialSmoothingForecaster,
    walk_forward_forecasts,
)
from src.core.metrics import mae, mape, rmse
from src.core.sequences import fibonacci_numbers
from src.core.stats import DescriptiveStats
from src.core.types import FloatArray

__all__ = [
    "DescriptiveStats",
    "FloatArray",
    "Forecaster",
    "LinearTrendForecaster",
    "MovingAverageForecaster",
    "SeasonalNaiveForecaster",
    "SimpleExponentialSmoothingForecaster",
    "fibonacci_numbers",
    "mae",
    "mape",
    "rmse",
    "walk_forward_forecasts",
]
