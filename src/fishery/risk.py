"""Revenue risk measurement: dispersion statistics, CV-based rating and VaR.

References:
    Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing
        Financial Risk* (3rd ed.). McGraw-Hill. (historical/Monte Carlo VaR)
    Everitt, B. S. & Skrondal, A. (2010). *The Cambridge Dictionary of
        Statistics* (4th ed.). Cambridge University Press ("coefficient of
        variation").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from src.core.stats import DescriptiveStats
from src.core.types import as_float_array


class RiskLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


@dataclass(frozen=True, slots=True)
class ValueAtRisk:
    """Value-at-Risk of a revenue distribution.

    Attributes:
        confidence: e.g. 0.95 for the 5 % VaR.
        expected: Mean simulated revenue.
        quantile: Revenue at the ``1 - confidence`` quantile ("bad year" level).
    """

    confidence: float
    expected: float
    quantile: float

    @property
    def var(self) -> float:
        """Shortfall versus expectation that is exceeded only ``1 - confidence`` of the time."""
        return self.expected - self.quantile


class RiskAssessor:
    """Rates revenue risk by the coefficient of variation (CV).

    Variance is measured in *squared* currency (UGX²), so any fixed variance
    threshold changes meaning with the scale of the business and the currency.
    The CV = stdev / mean is dimensionless and comparable across scenarios.

    Default thresholds (weekly revenue):

    * CV < 0.10 -> Low: under a normal approximation ~95 % of weeks fall within
      ±20 % of the mean, which ordinary working capital can absorb.
    * 0.10 <= CV < 0.25 -> Moderate: swings up to ±50 % are plausible; the
      cooperative needs a cash reserve or forward sales.
    * CV >= 0.25 -> High: a typical week can lose a quarter of revenue.

    Args:
        revenues: Revenue observations (e.g. weekly UGX).
        thresholds: ``(low_upper, moderate_upper)`` CV cut-offs.

    Raises:
        ValueError: On fewer than two observations, a non-positive mean or
            badly ordered thresholds.
    """

    def __init__(self, revenues: npt.ArrayLike, thresholds: tuple[float, float] = (0.10, 0.25)) -> None:
        values = as_float_array(revenues, name="revenues")
        if values.size < 2:
            raise ValueError("need at least two revenue observations")
        if not 0 < thresholds[0] < thresholds[1]:
            raise ValueError("thresholds must satisfy 0 < low < moderate")
        self.stats = DescriptiveStats.from_statistics(values)
        if self.stats.mean <= 0:
            raise ValueError("mean revenue must be positive for a CV-based rating")
        self.thresholds = thresholds

    def __repr__(self) -> str:
        return f"RiskAssessor(n={self.stats.n}, cv={self.cv:.3f}, level={self.classify().value})"

    @property
    def cv(self) -> float:
        return self.stats.cv

    def classify(self) -> RiskLevel:
        low, moderate = self.thresholds
        if self.cv < low:
            return RiskLevel.LOW
        if self.cv < moderate:
            return RiskLevel.MODERATE
        return RiskLevel.HIGH

    @staticmethod
    def value_at_risk(samples: npt.ArrayLike, confidence: float = 0.95) -> ValueAtRisk:
        """Historical-simulation VaR from Monte Carlo samples.

        Raises:
            ValueError: If ``confidence`` is not in ``(0, 1)`` or samples are empty.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        values = as_float_array(samples, name="samples")
        if values.size == 0:
            raise ValueError("no samples supplied")
        return ValueAtRisk(
            confidence=confidence,
            expected=float(np.mean(values)),
            quantile=float(np.quantile(values, 1.0 - confidence)),
        )
