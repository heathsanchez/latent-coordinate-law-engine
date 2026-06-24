from pathlib import Path

from latent_law.cli import main
from latent_law.data import generate_igp24_synthetic
from latent_law.evaluation import evaluate_holdout


def test_heldout_prediction_reaches_high_accuracy_clean_synthetic():
    df = generate_igp24_synthetic(n=900, seed=10)
    train = df[~df["holdout"]]
    holdout = df[df["holdout"]]

    report = evaluate_holdout(train, holdout)

    assert report["t"]["accuracy"] >= 0.90
    assert report["r"]["accuracy"] >= 0.90
    assert report["combined"]["accuracy"] >= 0.90


def test_cli_demo_creates_all_output_files(tmp_path: Path):
    out = tmp_path / "out"

    code = main(["demo", "--out", str(out), "--n", "400", "--seed", "11"])

    assert code == 0
    expected = {
        "dataset.csv",
        "coordinate_report.json",
        "lawbook.json",
        "holdout_report.json",
        "counterexamples.csv",
        "summary.md",
    }
    assert expected == {path.name for path in out.iterdir()}
