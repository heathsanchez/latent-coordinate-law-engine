from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from latent_law.coordinates import synthesize_coordinates
from latent_law.discovery import discover_coordinates
from latent_law.reporting import write_json


COMMON_TARGET_NAMES = ["target", "y", "label", "class", "quality", "strength", "output"]
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".parquet"}


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            df = pd.read_csv(path)
        except pd.errors.ParserError:
            df = pd.read_csv(path, sep=";")
        if len(df.columns) == 1:
            df_semicolon = pd.read_csv(path, sep=";")
            if len(df_semicolon.columns) > 1:
                return df_semicolon
        return df
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported suffix: {suffix}")


def infer_target_column(df: pd.DataFrame) -> str:
    normalized = {str(col).strip().lower().replace(" ", "_"): col for col in df.columns}
    for name in COMMON_TARGET_NAMES:
        if name in normalized:
            return str(normalized[name])
    return str(df.columns[-1])


def scan_dataset_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def load_real_datasets(data_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loaded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path in scan_dataset_files(data_dir):
        record = {"path": str(path)}
        try:
            df = _read_table(path)
        except Exception as exc:
            skipped.append({**record, "reason": f"read_failed: {exc}"})
            continue
        if len(df) < 100:
            skipped.append({**record, "rows": int(len(df)), "reason": "fewer_than_100_rows"})
            continue
        target = infer_target_column(df)
        numeric_features = [
            col
            for col in df.columns
            if col != target and pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 1
        ]
        if len(numeric_features) < 3:
            skipped.append(
                {
                    **record,
                    "rows": int(len(df)),
                    "target": target,
                    "numeric_feature_count": len(numeric_features),
                    "reason": "fewer_than_3_numeric_features",
                }
            )
            continue
        loaded.append(
            {
                **record,
                "name": path.stem,
                "dataframe": df,
                "target": target,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "numeric_feature_count": len(numeric_features),
            }
        )
    return loaded, skipped


def _problem_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 12:
        return "regression"
    return "classification"


def _feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    return [col for col in df.columns if col != target and df[col].nunique(dropna=False) > 1]


def _model(kind: str, problem: str, x: pd.DataFrame) -> Pipeline:
    categorical = [c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c]) and not pd.api.types.is_bool_dtype(x[c])]
    numeric = [c for c in x.columns if c not in categorical]
    if problem == "regression":
        estimator = (
            DecisionTreeRegressor(max_depth=5, min_samples_leaf=4, random_state=53)
            if kind == "coordinate"
            else RandomForestRegressor(n_estimators=180, random_state=53)
        )
    else:
        estimator = (
            DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=53)
            if kind == "coordinate"
            else RandomForestClassifier(n_estimators=180, random_state=53, class_weight="balanced")
        )
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("num", StandardScaler(), numeric),
                    ],
                    remainder="drop",
                ),
            ),
            ("model", estimator),
        ]
    )


def _evaluate(df: pd.DataFrame, target: str, raw_cols: list[str], coord_cols: list[str], problem: str) -> dict[str, Any]:
    y = df[target]
    stratify = y if problem == "classification" and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=61, stratify=stratify)

    raw_model = _model("raw", problem, x_train[raw_cols])
    raw_model.fit(x_train[raw_cols], y_train)
    raw_pred = raw_model.predict(x_test[raw_cols])

    coord_model = _model("coordinate", problem, x_train[coord_cols])
    coord_model.fit(x_train[coord_cols], y_train)
    coord_pred = coord_model.predict(x_test[coord_cols])

    if problem == "regression":
        raw_score = float(r2_score(y_test, raw_pred))
        coord_score = float(r2_score(y_test, coord_pred))
        metric_name = "r2"
        raw_f1 = None
        coord_f1 = None
    else:
        raw_score = float(accuracy_score(y_test, raw_pred))
        coord_score = float(accuracy_score(y_test, coord_pred))
        metric_name = "accuracy"
        raw_f1 = float(f1_score(y_test, raw_pred, average="macro", zero_division=0))
        coord_f1 = float(f1_score(y_test, coord_pred, average="macro", zero_division=0))

    return {
        "metric": metric_name,
        "raw_score": raw_score,
        "coordinate_score": coord_score,
        "raw_macro_f1": raw_f1,
        "coordinate_macro_f1": coord_f1,
    }


def evaluate_real_dataset(entry: dict[str, Any]) -> dict[str, Any]:
    df = entry["dataframe"].copy()
    target = entry["target"]
    raw_cols = _feature_columns(df, target)
    problem = _problem_type(df[target])
    synthesized = synthesize_coordinates(df, targets=[target], max_base_features=8, max_new_features=240)
    report = discover_coordinates(synthesized, target_cols=[target])
    ranking = report["targets"]["combined"]["rankings"]
    top_coords = [row["feature"] for row in ranking if row["feature"].startswith("coord_")][:5]
    if len(top_coords) < 2:
        top_coords = [row["feature"] for row in ranking[:5]]
    eval_cols = list(dict.fromkeys(top_coords))
    metrics = _evaluate(synthesized, target, raw_cols, eval_cols, problem)
    raw_score = metrics["raw_score"]
    coord_score = metrics["coordinate_score"]
    close_threshold = 0.95 * raw_score if raw_score >= 0 else raw_score * 1.05
    within_95 = coord_score >= close_threshold
    compression_ratio = len(eval_cols) / max(len(raw_cols), 1)
    return {
        "dataset": entry["name"],
        "path": entry["path"],
        "target": target,
        "problem_type": problem,
        "rows": entry["rows"],
        "raw_feature_count": len(raw_cols),
        "coordinate_feature_count": len(eval_cols),
        "compression_ratio": compression_ratio,
        "top_invented_coordinates": json.dumps(eval_cols),
        "metric": metrics["metric"],
        "raw_accuracy": raw_score if problem == "classification" else np.nan,
        "coordinate_accuracy": coord_score if problem == "classification" else np.nan,
        "raw_r2": raw_score if problem == "regression" else np.nan,
        "coordinate_r2": coord_score if problem == "regression" else np.nan,
        "raw_macro_f1": metrics["raw_macro_f1"],
        "coordinate_macro_f1": metrics["coordinate_macro_f1"],
        "coordinate_beats_raw": bool(coord_score > raw_score),
        "coordinate_within_95pct_raw_with_fewer_features": bool(within_95 and len(eval_cols) < len(raw_cols)),
    }


def run_real_dataset_benchmark(data_dir: str | Path, out: str | Path) -> dict[str, Any]:
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    loaded, skipped = load_real_datasets(data_dir)
    results: list[dict[str, Any]] = []
    for entry in loaded:
        try:
            results.append(evaluate_real_dataset(entry))
        except Exception as exc:
            skipped.append({"path": entry["path"], "target": entry["target"], "reason": f"evaluation_failed: {exc}"})

    report = {
        "data_dir": str(data_dir),
        "loaded": [
            {k: v for k, v in entry.items() if k != "dataframe"}
            for entry in loaded
            if not any(skip.get("path") == entry["path"] and str(skip.get("reason", "")).startswith("evaluation_failed") for skip in skipped)
        ],
        "skipped": skipped,
        "loaded_count": len(results),
        "skipped_count": len(skipped),
    }
    write_json(report, str(out_path / "real_dataset_report.json"))
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path / "real_dataset_results.csv", index=False)
    _write_summary(out_path / "real_dataset_summary.md", report, results_df)
    return {"report": report, "results": results}


def _write_summary(path: Path, report: dict[str, Any], results_df: pd.DataFrame) -> None:
    lines = [
        "# Real Dataset Coordinate Benchmark",
        "",
        f"Data dir: `{report['data_dir']}`",
        f"Loaded datasets: {report['loaded_count']}",
        f"Skipped files: {report['skipped_count']}",
        "",
    ]
    if not results_df.empty:
        beats = int(results_df["coordinate_beats_raw"].sum())
        close = int(results_df["coordinate_within_95pct_raw_with_fewer_features"].sum())
        lines.extend(
            [
                f"Coordinate model beats raw baseline: {beats}/{len(results_df)}",
                f"Within 95% of raw with fewer features: {close}/{len(results_df)}",
                "",
                "## Datasets",
                "",
            ]
        )
        for _, row in results_df.iterrows():
            score = row["coordinate_accuracy"] if row["problem_type"] == "classification" else row["coordinate_r2"]
            raw = row["raw_accuracy"] if row["problem_type"] == "classification" else row["raw_r2"]
            lines.extend(
                [
                    f"### {row['dataset']}",
                    f"- target: {row['target']}",
                    f"- problem: {row['problem_type']}",
                    f"- coordinate score: {score:.3f}",
                    f"- raw score: {raw:.3f}",
                    f"- compression ratio: {row['compression_ratio']:.3f}",
                    "",
                ]
            )
    if report["skipped"]:
        lines.extend(["## Skipped Files", ""])
        for row in report["skipped"]:
            lines.append(f"- `{row['path']}`: {row['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
