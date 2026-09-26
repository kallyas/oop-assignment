"""Monthly rainfall suitability rules for crops."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from src.core.types import as_float_array
from src.rainfall.region import Region


class Suitability(Enum):
    """Outcome of comparing one month's rainfall with a crop's range.

    The integer value is the ordinal code used by heatmaps.
    """

    DROUGHT = -1
    GOOD = 0
    WATERLOGGING = 1

    def label(self, crop: str) -> str:
        return {
            Suitability.DROUGHT: "Drought risk",
            Suitability.GOOD: f"Good for {crop.lower()}",
            Suitability.WATERLOGGING: "Waterlogging risk",
        }[self]


@dataclass(frozen=True, slots=True)
class CropRule:
    """A crop's acceptable monthly rainfall band during its growing season.

    Attributes:
        crop: Crop name.
        min_mm: Below this monthly total the crop is water-stressed.
        max_mm: Above this monthly total waterlogging/disease risk rises.
        source: Citation for the thresholds.
        rationale: How the source figure was converted to a monthly band.
    """

    crop: str
    min_mm: float
    max_mm: float
    source: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.min_mm < 0 or self.max_mm <= self.min_mm:
            raise ValueError("require 0 <= min_mm < max_mm")

    def classify_month(self, rainfall_mm: float) -> Suitability:
        """Classify one month's rainfall (boundaries count as *Good*)."""
        if rainfall_mm < 0:
            raise ValueError("rainfall cannot be negative")
        if rainfall_mm < self.min_mm:
            return Suitability.DROUGHT
        if rainfall_mm > self.max_mm:
            return Suitability.WATERLOGGING
        return Suitability.GOOD

    def classify(self, monthly_rainfall: npt.ArrayLike) -> list[Suitability]:
        """Classify every month of a rainfall series."""
        return [self.classify_month(float(v)) for v in as_float_array(monthly_rainfall)]

    def codes(self, region: Region) -> npt.NDArray[np.int64]:
        """Ordinal suitability codes (-1, 0, 1) for each month of ``region``."""
        return np.array([s.value for s in self.classify(region.rainfall)], dtype=np.int64)


# Thresholds convert seasonal crop water needs to a monthly band by dividing by
# the length of the growing season (months). Sources are listed per crop.
FAO_MANUAL_3 = (
    "Brouwer, C. & Heibloem, M. (1986). Irrigation Water Management: Irrigation "
    "Water Needs, Training Manual No. 3, Table 5. Rome: FAO."
)
DEFAULT_CROP_RULES: tuple[CropRule, ...] = (
    CropRule(
        "Maize",
        min_mm=125.0,
        max_mm=200.0,
        source=FAO_MANUAL_3,
        rationale="500-800 mm per growing period of ~4 months (125-140 days).",
    ),
    CropRule(
        "Beans",
        min_mm=100.0,
        max_mm=170.0,
        source=FAO_MANUAL_3,
        rationale="300-500 mm per growing period of ~3 months (95-110 days).",
    ),
    CropRule(
        "Coffee",
        min_mm=100.0,
        max_mm=200.0,
        source=(
            "DaMatta, F. M. & Ramalho, J. D. C. (2006). Impacts of drought and temperature "
            "stress on coffee physiology and production: a review. Brazilian Journal of "
            "Plant Physiology, 18(1), 55-81."
        ),
        rationale=(
            "optimum annual rainfall ~1,200-1,800 mm, i.e. ~100-150 mm per month; the upper "
            "bound is widened to 200 mm because the perennial root system tolerates wet "
            "months better than annual crops."
        ),
    ),
)
