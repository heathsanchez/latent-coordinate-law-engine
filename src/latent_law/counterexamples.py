from __future__ import annotations

import pandas as pd

from latent_law.data import generate_igp24_synthetic
from latent_law.features import extract_features
from latent_law.laws import Law, law_condition_mask


def search_counterexamples(law: Law, df: pd.DataFrame) -> pd.DataFrame:
    """Return rows satisfying a law condition but violating its prediction."""

    features = extract_features(df)
    mask = law_condition_mask(features, law.condition)
    violations = features[mask & (features[law.target] != law.predicted_value)].copy()
    return violations


def generate_counterexample_candidates(law: Law, n: int, seed: int) -> pd.DataFrame:
    """Generate synthetic rows near a law's condition for active probing."""

    candidates = generate_igp24_synthetic(n=n, seed=seed, holdout_mode="none")
    features = extract_features(candidates)
    mask = law_condition_mask(features, law.condition)
    if mask.any():
        return candidates.loc[mask].reset_index(drop=True)
    return candidates.reset_index(drop=True)
