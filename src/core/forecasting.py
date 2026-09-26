"""Forecasting framework: an abstract base class plus generic univariate models.

Every model follows the same two-step contract, ``fit(values, times)`` then
``predict(horizon)``, so evaluation code (back-testing, model selection,
bootstrapping) can treat all models polymorphically. Domain-specific models
(e.g. the CAGR and Fibonacci-ratio population models) subclass
:class:`Forecaster` in their own packages.

The base class uses the *template method* pattern: the public ``fit`` and
``predict`` methods do validation and bookkeeping once, and subclasses only
implement the model-specific ``_fit`` and ``_predict`` hooks.

References:
    Brown, R. G. (1959). *Statistical Forecasting for Inventory Control*.
        McGraw-Hill. (simple exponential smoothing)
    Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
        Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/ -- sec. 5.2
        (mean, naive and seasonal naive methods), 5.10 (time-series
        cross-validation) and 8.1 (simple exponential smoothing).
    Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an
        analysis and review. *International Journal of Forecasting*, 16(4),
        437-450. (rolling-origin evaluation)
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, TypeVar

import numpy as np
import numpy.typing as npt

from src.core.types import FloatArray, as_float_array

_F = TypeVar("_F", bound="Forecaster")


class NotFittedError(RuntimeError):
    """Raised when ``predict`` is called before ``fit``."""


class Forecaster(ABC):
    """Abstract base class for univariate forecasting models.

    Subclasses implement :meth:`_fit` and :meth:`_predict`, and override
    :attr:`min_history` if they need more than one observation.
    """

    def __init__(self) -> None:
        self._times: FloatArray | None = None
        self._values: FloatArray | None = None

    # ----------------------------------------------------------------- API
    @property
    def min_history(self) -> int:
        """Fewest observations the model can be fitted on."""
        return 1

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model label used in tables and legends."""

    def fit(self: _F, values: npt.ArrayLike, times: npt.ArrayLike | None = None) -> _F:
        """Fit the model to an equally spaced series.

        Args:
            values: Observed values in chronological order.
            times: Time index (e.g. calendar years). Defaults to ``0..n-1``.

        Returns:
            The fitted model itself, to allow ``model.fit(y).predict(h)``.

        Raises:
            ValueError: If there is too little history, if ``times`` and
                ``values`` differ in length, or if ``times`` is not strictly
                increasing and equally spaced.
        """
        y = as_float_array(values, name="values")
        if y.size < self.min_history:
            raise ValueError(
                f"{self.name} needs at least {self.min_history} observations, got {y.size}"
            )
        t = np.arange(y.size, dtype=np.float64) if times is None else as_float_array(times, name="times")
        if t.shape != y.shape:
            raise ValueError(f"times and values differ in length ({t.size} vs {y.size})")
        if t.size > 1:
            steps = np.diff(t)
            if np.any(steps <= 0) or not np.allclose(steps, steps[0]):
                raise ValueError("times must be strictly increasing and equally spaced")
        self._times, self._values = t, y
        self._fit(t, y)
        return self

    def predict(self, horizon: int) -> FloatArray:
        """Forecast the next ``horizon`` periods after the end of the training data.

        Raises:
            NotFittedError: If the model has not been fitted.
            ValueError: If ``horizon`` is not a positive integer.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")
        return self._predict(self.future_times(horizon))

    @property
    def training_data(self) -> tuple[FloatArray, FloatArray]:
        """The ``(times, values)`` the model was fitted on.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        return self._require_fitted()

    def future_times(self, horizon: int) -> FloatArray:
        """Time index of the ``horizon`` periods following the training data."""
        times, _ = self._require_fitted()
        step = float(times[1] - times[0]) if times.size > 1 else 1.0
        return np.asarray(times[-1] + step * np.arange(1, horizon + 1), dtype=np.float64)

    def fitted_values(self) -> FloatArray:
        """In-sample fitted values.

        The default is the honest *one-step-ahead* fit: the value at index
        ``i`` is predicted by a copy of the model trained on ``values[:i]``.
        Indices with too little history are ``NaN``. Models with a closed-form
        curve (e.g. a regression line) override this.
        """
        _, values = self._require_fitted()
        fitted = np.full(values.size, np.nan)
        for i in range(self.min_history, values.size):
            fitted[i] = self.clone().fit(values[:i])._predict_one_step()
        return fitted

    def residuals(self) -> FloatArray:
        """Actual minus fitted values (``NaN`` where no fit is available)."""
        _, values = self._require_fitted()
        return values - self.fitted_values()

    def clone(self: _F) -> _F:
        """Return an unfitted copy carrying the same hyper-parameters."""
        twin = copy.copy(self)
        Forecaster.__init__(twin)
        return twin

    def __repr__(self) -> str:
        state = "fitted" if self._values is not None else "unfitted"
        return f"{type(self).__name__}(name={self.name!r}, {state})"

    # ------------------------------------------------------------- hooks
    @abstractmethod
    def _fit(self, times: FloatArray, values: FloatArray) -> None:
        """Estimate model parameters from validated training data."""

    @abstractmethod
    def _predict(self, future_times: FloatArray) -> FloatArray:
        """Return forecasts at ``future_times`` (always after the training data)."""

    # ----------------------------------------------------------- helpers
    def _require_fitted(self) -> tuple[FloatArray, FloatArray]:
        if self._times is None or self._values is None:
            raise NotFittedError(f"{self.name} must be fitted before use")
        return self._times, self._values

    def _predict_one_step(self) -> float:
        return float(self.predict(1)[0])


class LinearTrendForecaster(Forecaster):
    """Ordinary least-squares straight line, ``y = slope * t + intercept``."""

    @property
    def min_history(self) -> int:
        return 2

    def __init__(self) -> None:
        super().__init__()
        self.slope_: float = float("nan")
        self.intercept_: float = float("nan")

    @property
    def name(self) -> str:
        return "Linear trend"

    def _fit(self, times: FloatArray, values: FloatArray) -> None:
        slope, intercept = np.polyfit(times, values, deg=1)
        self.slope_, self.intercept_ = float(slope), float(intercept)

    def _predict(self, future_times: FloatArray) -> FloatArray:
        return self.slope_ * future_times + self.intercept_

    def fitted_values(self) -> FloatArray:
        times, _ = self._require_fitted()
        return self._predict(times)


class MovingAverageForecaster(Forecaster):
    """Flat forecast equal to the mean of the last ``window`` observations."""

    def __init__(self, window: int = 3) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        super().__init__()
        self.window = window
        self.level_: float = float("nan")

    @property
    def min_history(self) -> int:
        return self.window

    @property
    def name(self) -> str:
        return f"{self.window}-day moving average"

    def _fit(self, times: FloatArray, values: FloatArray) -> None:
        self.level_ = float(np.mean(values[-self.window :]))

    def _predict(self, future_times: FloatArray) -> FloatArray:
        return np.full(future_times.size, self.level_)


class SimpleExponentialSmoothingForecaster(Forecaster):
    """Simple exponential smoothing (SES) with an optional grid-searched alpha.

    The level is updated as ``l_t = alpha * y_t + (1 - alpha) * l_{t-1}``
    with ``l_0 = y_0``; the forecast for every horizon is the final level.

    If ``alpha`` is ``None`` the smoothing constant is chosen on *each* call to
    :meth:`fit` by minimising the in-sample one-step-ahead squared error over
    ``alpha_grid``. Because tuning only sees the training window, the model
    can be back-tested without look-ahead bias.
    """

    DEFAULT_GRID: ClassVar[tuple[float, ...]] = tuple(round(0.05 * k, 2) for k in range(1, 21))

    def __init__(
        self, alpha: float | None = None, alpha_grid: Sequence[float] = DEFAULT_GRID
    ) -> None:
        if alpha is not None and not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
        if alpha is None and (len(alpha_grid) == 0 or not all(0.0 < a <= 1.0 for a in alpha_grid)):
            raise ValueError("alpha_grid must be a non-empty sequence of values in (0, 1]")
        super().__init__()
        self.alpha = alpha
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.alpha_: float = float("nan") if alpha is None else alpha
        self.level_: float = float("nan")

    @property
    def min_history(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "SES (tuned α)" if self.alpha is None else f"SES (α={self.alpha:.2f})"

    @staticmethod
    def smooth(values: FloatArray, alpha: float) -> FloatArray:
        """Return the level after each observation (``levels[i]`` uses ``values[:i+1]``)."""
        levels = np.empty_like(values)
        levels[0] = values[0]
        for i in range(1, values.size):
            levels[i] = alpha * values[i] + (1.0 - alpha) * levels[i - 1]
        return levels

    @classmethod
    def in_sample_sse(cls, values: FloatArray, alpha: float) -> float:
        """Sum of squared one-step-ahead errors ``y_t - l_{t-1}`` for ``t >= 1``."""
        levels = cls.smooth(values, alpha)
        return float(np.sum((values[1:] - levels[:-1]) ** 2))

    def _fit(self, times: FloatArray, values: FloatArray) -> None:
        if self.alpha is None:
            sse = [self.in_sample_sse(values, a) for a in self.alpha_grid]
            self.alpha_ = self.alpha_grid[int(np.argmin(sse))]
        self.level_ = float(self.smooth(values, self.alpha_)[-1])

    def _predict(self, future_times: FloatArray) -> FloatArray:
        return np.full(future_times.size, self.level_)


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the value observed one season (``period`` steps) earlier."""

    def __init__(self, period: int = 7) -> None:
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        super().__init__()
        self.period = period
        self._last_season: FloatArray = np.empty(0)

    @property
    def min_history(self) -> int:
        return self.period

    @property
    def name(self) -> str:
        return f"Seasonal naïve (m={self.period})"

    def _fit(self, times: FloatArray, values: FloatArray) -> None:
        self._last_season = values[-self.period :].copy()

    def _predict(self, future_times: FloatArray) -> FloatArray:
        idx = np.arange(future_times.size) % self.period
        return np.asarray(self._last_season[idx], dtype=np.float64)


def walk_forward_forecasts(
    model: Forecaster, values: npt.ArrayLike, first_origin: int
) -> FloatArray:
    """Rolling-origin (walk-forward) one-step-ahead forecasts.

    For each origin ``i`` in ``first_origin .. n-1`` a fresh clone of ``model``
    is fitted on ``values[:i]`` and asked for one step ahead, so every forecast
    only uses information that would have been available at the time.

    Returns:
        Array of length ``n - first_origin`` aligned with ``values[first_origin:]``.

    Raises:
        ValueError: If ``first_origin`` leaves the model too little history or
            no observations to forecast.
    """
    y = as_float_array(values, name="values")
    if first_origin < model.min_history:
        raise ValueError(
            f"first_origin={first_origin} is below {model.name}'s min_history={model.min_history}"
        )
    if first_origin >= y.size:
        raise ValueError("first_origin must leave at least one observation to forecast")
    return np.array(
        [float(model.clone().fit(y[:i]).predict(1)[0]) for i in range(first_origin, y.size)]
    )
