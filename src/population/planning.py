"""Translate population forecasts into primary-school classroom requirements."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassroomPlan:
    """Classroom requirement for one district between a base and a target year.

    Populations are in thousands (UBOS convention); pupil counts are in people.
    """

    district: str
    base_year: int
    target_year: int
    base_population_k: float
    target_population_k: float
    school_age_share: float
    pupils_per_classroom: int

    def __post_init__(self) -> None:
        if not 0.0 < self.school_age_share <= 1.0:
            raise ValueError("school_age_share must lie in (0, 1]")
        if self.pupils_per_classroom <= 0:
            raise ValueError("pupils_per_classroom must be positive")
        if self.base_population_k < 0 or self.target_population_k < 0:
            raise ValueError("populations cannot be negative")

    @property
    def base_pupils(self) -> float:
        return self.base_population_k * 1_000 * self.school_age_share

    @property
    def target_pupils(self) -> float:
        return self.target_population_k * 1_000 * self.school_age_share

    @property
    def classrooms_base(self) -> int:
        """Classrooms needed for the base-year cohort (rounded up)."""
        return math.ceil(self.base_pupils / self.pupils_per_classroom)

    @property
    def classrooms_target(self) -> int:
        """Classrooms needed for the target-year cohort (rounded up)."""
        return math.ceil(self.target_pupils / self.pupils_per_classroom)

    @property
    def additional_classrooms(self) -> int:
        """Extra classrooms to build, assuming base-year need is already met.

        Never negative: a shrinking cohort frees classrooms but does not
        imply demolition.
        """
        return max(0, self.classrooms_target - self.classrooms_base)
