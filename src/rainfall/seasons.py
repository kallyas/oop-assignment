"""Automatic rainy-season detection with :func:`scipy.signal.find_peaks`.

References:
    Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for
        scientific computing in Python. *Nature Methods*, 17, 261-272.
        (``find_peaks`` and topographic prominence)
    Basalirwa, C. P. K. (1995). Delineation of Uganda into climatological
        rainfall zones using the method of principal component analysis.
        *International Journal of Climatology*, 15(10), 1161-1177.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.signal import find_peaks

from src.rainfall.region import MONTHS, Region


class Modality(Enum):
    UNIMODAL = "Unimodal"
    BIMODAL = "Bimodal"
    MULTIMODAL = "Multimodal"
    NO_SEASON = "No distinct season"

    @classmethod
    def from_count(cls, n_peaks: int) -> Modality:
        return {0: cls.NO_SEASON, 1: cls.UNIMODAL, 2: cls.BIMODAL}.get(n_peaks, cls.MULTIMODAL)


@dataclass(frozen=True, slots=True)
class SeasonProfile:
    """Detected rainfall peaks of one region.

    Attributes:
        region: Region name.
        peak_months: 0-based month index of each peak, in calendar order.
        prominences: Prominence (mm) of each peak.
        relative_prominences: Prominence divided by the annual range.
    """

    region: str
    peak_months: tuple[int, ...]
    prominences: tuple[float, ...]
    relative_prominences: tuple[float, ...]

    @property
    def modality(self) -> Modality:
        return Modality.from_count(len(self.peak_months))

    @property
    def peak_names(self) -> tuple[str, ...]:
        return tuple(MONTHS[m] for m in self.peak_months)


def detect_seasons(region: Region, *, min_relative_prominence: float = 0.25) -> SeasonProfile:
    """Find rainy-season peaks, treating the year as circular.

    ``find_peaks`` never reports the first or last sample, and a December or
    January peak would be missed on a Jan-Dec array. The series is therefore
    tiled three times and only peaks in the middle copy are kept, which gives
    every month two real neighbours and lets prominences wrap across the year.

    A peak counts as a *season* only if its prominence (how far it stands above
    the higher of the troughs on either side) is at least
    ``min_relative_prominence`` of the annual range. This filters out a brief
    mid-season dip, which would otherwise split one long season into two.

    Raises:
        ValueError: If ``min_relative_prominence`` is outside ``[0, 1]``.
    """
    if not 0.0 <= min_relative_prominence <= 1.0:
        raise ValueError("min_relative_prominence must lie in [0, 1]")
    rain = region.rainfall
    span = float(rain.max() - rain.min())
    if span == 0.0:
        return SeasonProfile(region.name, (), (), ())
    tiled = np.tile(rain, 3)
    peaks, props = find_peaks(tiled, prominence=(min_relative_prominence * span, None))
    in_middle = (peaks >= 12) & (peaks < 24)
    months = tuple(int(p - 12) for p in peaks[in_middle])
    prominences = tuple(float(p) for p in props["prominences"][in_middle])
    return SeasonProfile(region.name, months, prominences, tuple(p / span for p in prominences))
