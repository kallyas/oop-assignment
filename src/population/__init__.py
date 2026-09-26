"""Mini-project 1: UBOS district population forecaster."""

from src.population.district import DistrictPopulation
from src.population.evaluation import (
    Forecast,
    ModelScore,
    ValidationReport,
    bootstrap_interval,
    forecast_district,
    validate_models,
)
from src.population.forecasters import (
    ExponentialCAGRForecaster,
    FibonacciRatioForecaster,
    LinearTrendForecaster,
    fibonacci_numbers,
)
from src.population.planning import ClassroomPlan

__all__ = [
    "ClassroomPlan",
    "DistrictPopulation",
    "ExponentialCAGRForecaster",
    "FibonacciRatioForecaster",
    "Forecast",
    "LinearTrendForecaster",
    "ModelScore",
    "ValidationReport",
    "bootstrap_interval",
    "fibonacci_numbers",
    "forecast_district",
    "validate_models",
]
