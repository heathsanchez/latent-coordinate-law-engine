from pathlib import Path

from latent_law.benchmarks import run_coordinate_invention_probe
from latent_law.coordinates import synthesize_coordinates


def test_synthesize_coordinates_creates_second_order_ratio():
    import pandas as pd

    df = pd.DataFrame(
        {
            "control_parameter": [0.2, 0.4, 0.8],
            "coupling": [1.0, 1.5, 2.0],
            "temperature": [2.0, 2.0, 1.0],
            "phase": ["low", "low", "high"],
        }
    )

    out = synthesize_coordinates(df, targets=["phase"], max_base_features=3)

    assert any(
        name.startswith("coord_mul__")
        and "control_parameter" in name
        and "coupling" in name
        and "temperature" in name
        for name in out.columns
    )


def test_coordinate_invention_probe_recovers_withheld_phase_coordinate(tmp_path: Path):
    result = run_coordinate_invention_probe(str(tmp_path), seed=5)

    assert result["success"]
    assert (tmp_path / "coordinate_invention_probe.json").exists()
