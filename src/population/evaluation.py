"""Hold-out validation, model selection, forecasting and bootstrap intervals.

References:
    Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
        Chapman & Hall. (residual resampling, percentile intervals)
    Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
        Practice* (3rd ed.), sec. 5.5 (bootstrapped prediction intervals)
        and 5.8 (train/test evaluation). OTexts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from src.core.forecasting import Forecaster
from src.core.metrics import mae, mape, rmse
from src.core.types import FloatArray, IntArray
from src.population.district import DistrictPopulation

ModelFactory = Callable[[], Forecaster]
"""A zero-argument callable returning a fresh, unfitted model."""


@dataclass(frozen=True, slots=True)
class ModelScore:
    """Hold-out accuracy of one model on one district."""

    model_name: str
    mae: float
    rmse: float
    mape: float
    test_predictions: FloatArray = field(repr=False)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Train/test comparison of several models on one district.

    Attributes:
        district: The full series.
        train: Series used for fitting.
        test: Held-out series used for scoring.
        scores: One :class:`ModelScore` per candidate model, in input order.
        selection_metric: Metric used to pick :attr:`best`.
    """

    district: DistrictPopulation
    train: DistrictPopulation
    test: DistrictPopulation
    scores: tuple[ModelScore, ...]
    selection_metric: str = "rmse"

    @property
    def best(self) -> ModelScore:
        """The model with the lowest value of :attr:`selection_metric`."""
        return min(self.scores, key=lambda s: float(getattr(s, self.selection_metric)))

    def metrics_agree(self) -> bool:
        """True if MAE, RMSE and MAPE all rank the same model first."""
        winners = {
            min(self.scores, key=lambda s: float(getattr(s, metric))).model_name
            for metric in ("mae", "rmse", "mape")
        }
        return len(winners) == 1

    def table(self) -> list[dict[str, str | float]]:
        """Rows suitable for ``pandas.DataFrame`` or pretty printing."""
        return [
            {"model": s.model_name, "MAE (k)": s.mae, "RMSE (k)": s.rmse, "MAPE (%)": s.mape}
            for s in self.scores
        ]


def validate_models(
    district: DistrictPopulation,
    factories: Sequence[ModelFactory],
    *,
    last_train_year: int,
    selection_metric: str = "rmse",
) -> ValidationReport:
    """Fit each model on years ``<= last_train_year`` and score it on the rest.

    Raises:
        ValueError: If ``factories`` is empty or ``selection_metric`` is unknown.
    """
    if not factories:
        raise ValueError("at least one model factory is required")
    if selection_metric not in {"mae", "rmse", "mape"}:
        raise ValueError(f"unknown selection metric {selection_metric!r}")
    train, test = district.split(last_train_year)
    scores = []
    for factory in factories:
        model = factory().fit(train.populations, train.years)
        pred = model.predict(len(test))
        scores.append(
            ModelScore(
                model_name=model.name,
                mae=mae(test.populations, pred),
                rmse=rmse(test.populations, pred),
                mape=mape(test.populations, pred),
                test_predictions=pred,
            )
        )
    return ValidationReport(district, train, test, tuple(scores), selection_metric)


@dataclass(frozen=True, slots=True)
class Forecast:
    """A point forecast with an optional bootstrap prediction interval."""

    district: str
    model_name: str
    years: IntArray
    point: FloatArray
    lower: FloatArray | None = None
    upper: FloatArray | None = None
    level: float | None = None

    @property
    def final_value(self) -> float:
        return float(self.point[-1])


def forecast_district(
    district: DistrictPopulation,
    factory: ModelFactory,
    horizon: int,
    *,
    n_bootstrap: int = 0,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[Forecast, Forecaster]:
    """Refit ``factory()`` on the whole series and forecast ``horizon`` years.

    If ``n_bootstrap > 0`` a residual-bootstrap prediction interval is added
    (see :func:`bootstrap_interval`).

    Returns:
        The forecast and the model fitted on the full series.
    """
    model = factory().fit(district.populations, district.years)
    point = model.predict(horizon)
    years = model.future_times(horizon).astype(np.int64)
    if n_bootstrap <= 0:
        return Forecast(district.name, model.name, years, point), model
    lower, upper = bootstrap_interval(
        model, horizon, n_bootstrap=n_bootstrap, level=level, rng=rng or np.random.default_rng()
    )
    return Forecast(district.name, model.name, years, point, lower, upper, level), model


def bootstrap_interval(
    fitted_model: Forecaster,
    horizon: int,
    *,
    n_bootstrap: int = 1000,
    level: float = 0.95,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Residual-bootstrap prediction interval for a fitted model.

    Algorithm (for ``b = 1..B``):

    1. Build a pseudo-history ``y* = fitted + e*`` by resampling in-sample
       residuals with replacement (parameter uncertainty).
    2. Refit a clone of the model on ``y*`` and forecast ``horizon`` steps.
    3. Add freshly resampled residuals to that path (future shock uncertainty).

    The interval is the equal-tailed percentile range of the ``B`` paths.
    Residuals are re-centred to mean zero first: a curve that is not a
    least-squares fit (e.g. the CAGR curve through the end points) leaves
    one-sided residuals that would otherwise bias every bootstrap path.
    Residuals are assumed exchangeable (i.i.d.); serial correlation would make
    the interval too narrow.

    Raises:
        ValueError: On invalid ``n_bootstrap``/``level`` or too few residuals.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    times, values = fitted_model.training_data
    fitted = fitted_model.fitted_values()
    resid = values - fitted
    usable = ~np.isnan(resid)
    if np.count_nonzero(usable) < 2:
        raise ValueError("need at least two residuals to bootstrap")
    pool = resid[usable] - resid[usable].mean()
    base = np.where(usable, fitted, values)

    paths = np.empty((n_bootstrap, horizon))
    for b in range(n_bootstrap):
        pseudo = base + np.where(usable, rng.choice(pool, size=values.size), 0.0)
        path = fitted_model.clone().fit(pseudo, times).predict(horizon)
        paths[b] = path + rng.choice(pool, size=horizon)
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(paths, [alpha, 1.0 - alpha], axis=0)
    return lower, upper
