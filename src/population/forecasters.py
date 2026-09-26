"""Population-specific forecasting models.

The generic linear trend lives in :mod:`src.core.forecasting` and is reused
here; this module adds the two growth-rate models the assignment asks for.

References:
    Sigler, L. E. (2002). *Fibonacci's Liber Abaci: A Translation into Modern
        English*. Springer. (the rabbit-population recurrence)
    Preston, S. H., Heuveline, P. & Guillot, M. (2001). *Demography:
        Measuring and Modeling Population Processes*. Blackwell, ch. 1
        (growth rates and exponential growth).
"""

from __future__ import annotations

import numpy as np

from src.core.forecasting import Forecaster, LinearTrendForecaster
from src.core.sequences import fibonacci_numbers
from src.core.types import FloatArray

__all__ = [
    "ExponentialCAGRForecaster",
    "FibonacciRatioForecaster",
    "LinearTrendForecaster",
    "fibonacci_numbers",
]


class ExponentialCAGRForecaster(Forecaster):
    """Constant-percentage growth at the historical CAGR.

    ``P_{T+k} = P_T * (1 + g) ** k`` where ``g`` is the compound annual growth
    rate of the training window. This is the discrete analogue of exponential
    (Malthusian) growth.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cagr_: float = float("nan")
        self._base: float = float("nan")
        self._step: float = 1.0

    @property
    def min_history(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "Exponential (CAGR)"

    def _fit(self, times: FloatArray, values: FloatArray) -> None:
        if np.any(values <= 0):
            raise ValueError("CAGR requires strictly positive values")
        periods = values.size - 1
        self.cagr_ = float((values[-1] / values[0]) ** (1.0 / periods) - 1.0)
        self._base = float(values[-1])
        self._step = float(times[1] - times[0])

    def _predict(self, future_times: FloatArray) -> FloatArray:
        times, _ = self._require_fitted()
        steps_ahead = (future_times - times[-1]) / self._step
        return np.asarray(self._base * (1.0 + self.cagr_) ** steps_ahead, dtype=np.float64)

    def fitted_values(self) -> FloatArray:
        """The exponential curve through the first and last training points."""
        times, values = self._require_fitted()
        steps = (times - times[0]) / self._step
        return np.asarray(values[0] * (1.0 + self.cagr_) ** steps, dtype=np.float64)


class FibonacciRatioForecaster(Forecaster):
    """The previous cohort's model: scale the last value by Fibonacci ratios.

    The ``k``-th forecast is ``P_T * prod_{j=1..k} F(s+j) / F(s+j-1)`` where
    ``s = start_index``. The ratios converge to the golden ratio
    (about 1.618), so the model implies roughly 62 % growth *per step*
    regardless of the data; the only information it takes from the history
    is the last observation. It is included as a baseline to be critiqued.

    Args:
        start_index: Which Fibonacci ratio to start from (1 = ``F2/F1 = 1``).
    """

    def __init__(self, start_index: int = 1) -> None:
        if start_index < 1:
            raise ValueError(f"start_index must be >= 1, got {start_index}")
        super().__init__()
        self.start_index = start_index
        self._last: float = float("nan")

    @property
    def name(self) -> str:
        return "Fibonacci ratio"

    def ratios(self, horizon: int) -> FloatArray:
        """The ``horizon`` successive ratios ``F(n+1)/F(n)`` applied by the model."""
        fib = np.array(fibonacci_numbers(self.start_index + horizon + 1), dtype=np.float64)
        s = self.start_index
        return fib[s : s + horizon] / fib[s - 1 : s - 1 + horizon]

    def _fit(self, times: FloatArray, values: FloatArray) -> None:
        self._last = float(values[-1])

    def _predict(self, future_times: FloatArray) -> FloatArray:
        return np.asarray(self._last * np.cumprod(self.ratios(future_times.size)), dtype=np.float64)
