from pathlib import Path

import pandas as pd

from latent_law.challenge import generate_feynman_like, run_a_plus_plus_challenge
from latent_law.coordinates import synthesize_coordinates


def test_feynman_like_coordinate_synthesis_recovers_energy_terms():
    probe = generate_feynman_like(seed=7, n=220)
    hidden = probe.data.drop(columns=[probe.hidden_coordinate])
    synthesized = synthesize_coordinates(hidden, targets=probe.targets, max_base_features=6, max_new_features=300)

    assert any(
        name.startswith("coord_mul__")
        and "mass" in name
        and "velocity" in name
        and name.count("velocity") >= 2
        for name in synthesized.columns
    )


def test_a_plus_plus_challenge_writes_outputs(tmp_path: Path):
    result = run_a_plus_plus_challenge(str(tmp_path), seed=8)

    expected = {
        "dataset_sources.json",
        "scientific_rediscovery.csv",
        "unknown_prediction.csv",
        "human_comparison.csv",
        "a_plus_plus_scores.csv",
        "a_plus_plus_summary.json",
        "a_plus_plus_report.md",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    assert result["invention_score"] >= 0.5
    scores = pd.read_csv(tmp_path / "a_plus_plus_scores.csv")
    assert "rediscovery_score" in scores.columns
