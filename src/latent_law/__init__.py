"""Latent Coordinate Law discovery engine."""

from latent_law.data import generate_igp24_synthetic
from latent_law.discovery import discover_coordinates
from latent_law.evaluation import evaluate_holdout
from latent_law.features import extract_features
from latent_law.laws import Law, induce_laws

__all__ = [
    "Law",
    "discover_coordinates",
    "evaluate_holdout",
    "extract_features",
    "generate_igp24_synthetic",
    "induce_laws",
]
