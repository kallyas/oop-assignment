"""Turn suitability classifications into planting windows."""

from __future__ import annotations

from src.rainfall.crops import CropRule, Suitability
from src.rainfall.region import MONTHS, Region


def good_runs(rule: CropRule, region: Region) -> list[tuple[str, str, int]]:
    """Maximal runs of consecutive *Good* months, wrapping over New Year.

    Returns:
        ``(first_month, last_month, length)`` tuples, longest first.
    """
    good = [s is Suitability.GOOD for s in rule.classify(region.rainfall)]
    if all(good):
        return [(MONTHS[0], MONTHS[-1], 12)]
    if not any(good):
        return []
    start = good.index(False) + 1  # begin just after a non-good month so runs don't wrap mid-way
    runs: list[tuple[str, str, int]] = []
    length = 0
    for k in range(12):
        m = (start + k) % 12
        if good[m]:
            length += 1
        elif length:
            runs.append((MONTHS[(m - length) % 12], MONTHS[(m - 1) % 12], length))
            length = 0
    if length:
        end = (start + 11) % 12
        runs.append((MONTHS[(end - length + 1) % 12], MONTHS[end], length))
    return sorted(runs, key=lambda r: -r[2])
