from pathlib import Path

import pandas as pd

from latent_law.benchmarks import generate_phase_transition, run_conclusive_benchmark
from latent_law.discovery import discover_coordinates


def test_phase_transition_blind_domain_recovers_control_coordinate():
    df = generate_phase_transition(n=360, seed=30)
    report = discover_coordinates(df, target_cols=["phase", "outbreak"])
    top = [row["feature"] for row in report["targets"]["combined"]["rankings"][:4]]

    assert "effective_control" in top or "control_parameter" in top


def test_conclusive_benchmark_writes_required_outputs(tmp_path: Path):
    out = tmp_path / "benchmark"

    summary = run_conclusive_benchmark(str(out), seed=40)

    expected = {
        "lawbook.json",
        "thresholds.json",
        "invariants.json",
        "coordinate_rankings.csv",
        "transfer_results.csv",
        "compression_results.csv",
        "counterexamples.csv",
        "benchmark_report.md",
        "final_conclusion.md",
        "benchmark_summary.json",
    }
    assert expected.issubset({path.name for path in out.iterdir()})
    assert summary["blind_transfer_accuracy"] >= 0.7
    assert summary["mean_compression_ratio"] < 1.0
    transfer = pd.read_csv(out / "transfer_results.csv")
    assert set(["IGP24", "ETP", "ARC", "MAZE", "CA", "PHASE"]).issubset(set(transfer["domain"]))
