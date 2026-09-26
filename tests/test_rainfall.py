"""Tests for mini-project 4 (rainfall and crop suitability)."""

from __future__ import annotations

import numpy as np
import pytest

from src.rainfall import (
    CropRule,
    Modality,
    Region,
    Suitability,
    cosine_similarity,
    detect_seasons,
    euclidean_distance,
    good_runs,
    pearson_correlation,
    scipy_cosine_similarity,
)
from src.rainfall.data import load_regions


def test_region_summary_methods() -> None:
    r = Region("Test", [10.0] * 11 + [120.0])
    assert r.annual_total() == pytest.approx(230.0)
    assert r.wettest_month() == "Dec"
    assert r["Dec"] == 120.0


@pytest.mark.parametrize("values", [[1.0] * 11, [1.0] * 11 + [-1.0]])
def test_region_validation(values: list[float]) -> None:
    with pytest.raises(ValueError):
        Region("Bad", values)


def test_cosine_similarity_matches_scipy() -> None:
    regions = load_regions()
    for a in regions:
        for b in regions:
            assert cosine_similarity(a.rainfall, b.rainfall) == pytest.approx(
                scipy_cosine_similarity(a.rainfall, b.rainfall)
            )


def test_cosine_is_scale_invariant_but_euclidean_is_not() -> None:
    a = np.array([10.0, 50.0, 100.0])
    assert cosine_similarity(a, 3 * a) == pytest.approx(1.0)
    assert pearson_correlation(a, 3 * a) == pytest.approx(1.0)
    assert euclidean_distance(a, 3 * a) > 0


def test_cosine_undefined_for_zero_vector() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 2.0])


def test_crop_rule_boundaries() -> None:
    rule = CropRule("Maize", 100.0, 200.0)
    assert rule.classify_month(99.9) is Suitability.DROUGHT
    assert rule.classify_month(100.0) is Suitability.GOOD
    assert rule.classify_month(200.0) is Suitability.GOOD
    assert rule.classify_month(200.1) is Suitability.WATERLOGGING
    assert Suitability.GOOD.label("Maize") == "Good for maize"
    with pytest.raises(ValueError):
        CropRule("Bad", 200.0, 100.0)


def test_season_detection_handles_year_wrap() -> None:
    # Peaks in January and July: the January peak is only visible with circular padding.
    rain = [200, 120, 60, 40, 60, 120, 200, 120, 60, 40, 60, 120]
    profile = detect_seasons(Region("Wrap", rain))
    assert profile.peak_names == ("Jan", "Jul")
    assert profile.modality is Modality.BIMODAL


def test_flat_rainfall_has_no_season() -> None:
    assert detect_seasons(Region("Flat", [50.0] * 12)).modality is Modality.NO_SEASON


def test_good_runs_wrap_over_new_year() -> None:
    rule = CropRule("X", 100.0, 200.0)
    rain = [150, 150, 10, 10, 10, 10, 10, 10, 10, 10, 150, 150]
    assert good_runs(rule, Region("R", rain)) == [("Nov", "Feb", 4)]
