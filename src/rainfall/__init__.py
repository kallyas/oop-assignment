"""Mini-project 4: rainfall pattern and crop suitability analyser."""

from src.rainfall.advisory import good_runs
from src.rainfall.crops import DEFAULT_CROP_RULES, CropRule, Suitability
from src.rainfall.region import MONTHS, Region
from src.rainfall.seasons import Modality, SeasonProfile, detect_seasons
from src.rainfall.similarity import (
    SimilarityMatrix,
    all_matrices,
    cosine_similarity,
    euclidean_distance,
    pairwise_matrix,
    pearson_correlation,
    scipy_cosine_similarity,
)

__all__ = [
    "DEFAULT_CROP_RULES",
    "MONTHS",
    "CropRule",
    "Modality",
    "Region",
    "SeasonProfile",
    "SimilarityMatrix",
    "Suitability",
    "all_matrices",
    "cosine_similarity",
    "detect_seasons",
    "euclidean_distance",
    "good_runs",
    "pairwise_matrix",
    "pearson_correlation",
    "scipy_cosine_similarity",
]
