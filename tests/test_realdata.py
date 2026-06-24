from pathlib import Path

import numpy as np
import pandas as pd

from latent_law.challenge import run_a_plus_plus_challenge
from latent_law.realdata import infer_target_column, load_real_datasets, run_real_dataset_benchmark


def _write_fake_dataset(path: Path, target_name: str = "quality", rows: int = 140) -> None:
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "x3": rng.normal(size=rows),
            "x4": rng.normal(size=rows),
        }
    )
    df[target_name] = np.where(df["x1"] + df["x2"] * 0.5 > 0, "high", "low")
    if path.suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)


def test_real_dataset_loader_finds_recursive_files_and_infers_target(tmp_path: Path):
    data_dir = tmp_path / "data" / "real"
    nested = data_dir / "uci"
    nested.mkdir(parents=True)
    _write_fake_dataset(nested / "fake_quality.csv", target_name="quality")

    loaded, skipped = load_real_datasets(data_dir)

    assert len(loaded) == 1
    assert skipped == []
    assert loaded[0]["target"] == "quality"
    assert infer_target_column(loaded[0]["dataframe"]) == "quality"


def test_real_dataset_loader_uses_last_column_when_no_common_target(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_fake_dataset(data_dir / "custom.tsv", target_name="mystery")

    loaded, _ = load_real_datasets(data_dir)

    assert loaded[0]["target"] == "mystery"


def test_real_dataset_loader_reports_skipped_files(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"x1": [1, 2], "x2": [3, 4], "target": [0, 1]}).to_csv(data_dir / "tiny.csv", index=False)
    pd.DataFrame({"x1": range(120), "target": [0, 1] * 60}).to_csv(data_dir / "too_few_numeric.csv", index=False)

    loaded, skipped = load_real_datasets(data_dir)

    assert loaded == []
    assert {row["reason"] for row in skipped} == {"fewer_than_100_rows", "fewer_than_3_numeric_features"}


def test_real_dataset_benchmark_writes_reports(tmp_path: Path):
    data_dir = tmp_path / "real"
    data_dir.mkdir()
    _write_fake_dataset(data_dir / "fake.csv")
    pd.DataFrame({"x1": [1, 2], "target": [0, 1]}).to_csv(data_dir / "skip.csv", index=False)

    result = run_real_dataset_benchmark(data_dir, tmp_path / "out")

    assert result["report"]["loaded_count"] == 1
    assert result["report"]["skipped_count"] == 1
    assert (tmp_path / "out" / "real_dataset_report.json").exists()
    assert (tmp_path / "out" / "real_dataset_results.csv").exists()
    assert (tmp_path / "out" / "real_dataset_summary.md").exists()


def test_a_plus_plus_data_dir_no_internet_mode_does_not_crash(tmp_path: Path):
    data_dir = tmp_path / "real"
    data_dir.mkdir()
    _write_fake_dataset(data_dir / "fake.csv")

    result = run_a_plus_plus_challenge(str(tmp_path / "app"), seed=9, data_dir=str(data_dir))

    assert result["real_dataset_validation_count"] == 1
    assert (tmp_path / "app" / "real_dataset_report.json").exists()
    assert (tmp_path / "app" / "dataset_sources.json").exists()
