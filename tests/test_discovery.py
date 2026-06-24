from latent_law.data import generate_igp24_synthetic
from latent_law.discovery import discover_coordinates
from latent_law.features import extract_features


def _rank_features(report, target):
    return [row["feature"] for row in report["targets"][target]["rankings"]]


def test_coordinate_discovery_ranks_a6_highly_for_r():
    df = extract_features(generate_igp24_synthetic(n=700, seed=1))
    report = discover_coordinates(df)

    ranking = _rank_features(report, "r")
    assert "a6" in ranking[:3] or "threshold_zone" in ranking[:3]
    assert ranking.index("a6") < ranking.index("a18")


def test_coordinate_discovery_ranks_support_highly_for_t():
    df = extract_features(generate_igp24_synthetic(n=700, seed=2))
    report = discover_coordinates(df)

    ranking = _rank_features(report, "t")
    assert "support_index" in ranking[:4] or "support_face" in ranking[:4]
    assert ranking.index("support_index") < ranking.index("a18")
