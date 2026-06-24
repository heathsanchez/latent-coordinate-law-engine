from __future__ import annotations

import json
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from latent_law.benchmarks import generate_cellular_automata, generate_phase_transition
from latent_law.coordinates import synthesize_coordinates
from latent_law.discovery import discover_coordinates
from latent_law.etp import generate_etp_from_equations
from latent_law.realdata import run_real_dataset_benchmark
from latent_law.reporting import write_json


DATASET_SOURCES = [
    {
        "name": "srsd-benchmark",
        "kind": "git",
        "url": "https://github.com/omron-sinicx/srsd-benchmark",
        "status": "not_downloaded",
        "note": "Network is restricted in this environment unless explicitly approved.",
    },
    {
        "name": "pmlb",
        "kind": "git",
        "url": "https://github.com/EpistasisLab/pmlb",
        "status": "not_downloaded",
        "note": "Network is restricted in this environment unless explicitly approved.",
    },
    {
        "name": "uci_wine_quality",
        "kind": "http",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "status": "not_downloaded",
        "note": "Network is restricted in this environment unless explicitly approved.",
    },
]


@dataclass
class RediscoveryProbe:
    name: str
    data: pd.DataFrame
    targets: list[str]
    hidden_coordinate: str
    expected_terms: list[str]
    source: str


def _band(values: pd.Series | np.ndarray, labels: tuple[str, str, str] = ("low", "mid", "high")) -> pd.Series:
    series = pd.Series(values)
    return pd.qcut(series.rank(method="first"), q=3, labels=list(labels)).astype(str)


def generate_feynman_like(seed: int = 0, n: int = 420) -> RediscoveryProbe:
    rng = np.random.default_rng(seed)
    mass = rng.uniform(0.5, 12.0, n)
    velocity = rng.uniform(0.1, 35.0, n)
    height = rng.uniform(0.0, 80.0, n)
    gravity = rng.normal(9.81, 0.015, n)
    drag = rng.uniform(0.0, 0.08, n)
    total_energy = 0.5 * mass * velocity**2 + mass * gravity * height
    observed_range = total_energy / (1.0 + drag * velocity)
    df = pd.DataFrame(
        {
            "mass": mass,
            "velocity": velocity,
            "height": height,
            "gravity": gravity,
            "drag": drag,
            "observed_range": observed_range,
            "total_energy": total_energy,
            "energy_band": _band(total_energy, ("low_energy", "mid_energy", "high_energy")),
        }
    )
    return RediscoveryProbe(
        name="FEYNMAN_LIKE_ENERGY",
        data=df,
        targets=["energy_band"],
        hidden_coordinate="total_energy",
        expected_terms=["mass", "velocity"],
        source="offline_physics_surrogate",
    )


def generate_gas_law_like(seed: int = 1, n: int = 420) -> RediscoveryProbe:
    rng = np.random.default_rng(seed)
    pressure = rng.uniform(0.8, 12.0, n)
    volume = rng.uniform(0.5, 8.0, n)
    moles = rng.uniform(0.1, 4.0, n)
    impurity = rng.uniform(0.0, 0.12, n)
    temperature = pressure * volume / (moles * 0.082057)
    expansion_class = _band(temperature * (1 - impurity), ("cold", "temperate", "hot"))
    df = pd.DataFrame(
        {
            "pressure": pressure,
            "volume": volume,
            "moles": moles,
            "impurity": impurity,
            "temperature": temperature,
            "expansion_class": expansion_class,
        }
    )
    return RediscoveryProbe(
        name="GAS_LAW_TEMPERATURE",
        data=df,
        targets=["expansion_class"],
        hidden_coordinate="temperature",
        expected_terms=["pressure", "volume", "moles"],
        source="offline_science_surrogate",
    )


def generate_ca_entropy_probe(seed: int = 2) -> RediscoveryProbe:
    df = generate_cellular_automata(seed=seed)
    return RediscoveryProbe(
        name="CA_ENTROPY",
        data=df,
        targets=["wolfram_class"],
        hidden_coordinate="local_entropy",
        expected_terms=["rule_density"],
        source="generated_elementary_ca",
    )


def generate_phase_probe(seed: int = 3) -> RediscoveryProbe:
    df = generate_phase_transition(n=420, seed=seed)
    return RediscoveryProbe(
        name="PHASE_EFFECTIVE_CONTROL",
        data=df,
        targets=["phase", "outbreak"],
        hidden_coordinate="effective_control",
        expected_terms=["control_parameter", "coupling", "temperature"],
        source="generated_phase_transition",
    )


def generate_etp_raw(seed: int = 4, n: int = 420) -> pd.DataFrame:
    return generate_etp_from_equations(n=n, seed=seed)


def _feature_columns(df: pd.DataFrame, targets: list[str]) -> list[str]:
    excluded = set(targets) | {"domain", "label", "experiment", "holdout", "run", "description", "status"}
    return [c for c in df.columns if c not in excluded and df[c].nunique(dropna=False) > 1]


def _model(kind: str, x: pd.DataFrame) -> Pipeline:
    categorical = [c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c]) and not pd.api.types.is_bool_dtype(x[c])]
    numeric = [c for c in x.columns if c not in categorical]
    if kind == "tree":
        estimator = DecisionTreeClassifier(max_depth=4, min_samples_leaf=4, random_state=29)
    elif kind == "knn":
        estimator = KNeighborsClassifier(n_neighbors=7)
    else:
        estimator = RandomForestClassifier(n_estimators=180, random_state=29, class_weight="balanced")
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


def _target(df: pd.DataFrame, targets: list[str]) -> pd.Series:
    if len(targets) == 1:
        return df[targets[0]].astype(str)
    return df[targets].astype(str).agg("|".join, axis=1)


def _predictive_metrics(df: pd.DataFrame, targets: list[str], cols: list[str], structural_col: str | None = None) -> dict[str, float]:
    y = _target(df, targets)
    if structural_col and structural_col in df.columns:
        order = df[structural_col].rank(method="first")
        train_mask = order <= order.quantile(0.65)
        x_train, x_test = df.loc[train_mask, cols], df.loc[~train_mask, cols]
        y_train, y_test = y.loc[train_mask], y.loc[~train_mask]
    else:
        stratify = y if y.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(df[cols], y, test_size=0.3, random_state=31, stratify=stratify)
    clf = _model("forest", x_train)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def _max_abs_correlation(df: pd.DataFrame, hidden: str, candidates: list[str]) -> tuple[str | None, float]:
    if hidden not in df.columns:
        return None, 0.0
    best_name: str | None = None
    best_corr = 0.0
    hidden_values = df[hidden].astype(float)
    for col in candidates:
        if col == hidden or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        corr = abs(float(pd.Series(df[col]).corr(hidden_values)))
        if corr == corr and corr > best_corr:
            best_corr = corr
            best_name = col
    return best_name, best_corr


def _score_probe(probe: RediscoveryProbe) -> dict[str, Any]:
    hidden_values = probe.data[probe.hidden_coordinate].copy()
    hidden = probe.data.drop(columns=[probe.hidden_coordinate])
    synthesized = synthesize_coordinates(hidden, targets=probe.targets, max_base_features=8, max_new_features=360)
    synthesized[probe.hidden_coordinate] = hidden_values
    candidate_cols = [c for c in synthesized.columns if c.startswith("coord_")]
    best_corr_name, best_corr = _max_abs_correlation(synthesized, probe.hidden_coordinate, candidate_cols)
    discovery_frame = synthesized.drop(columns=[probe.hidden_coordinate])
    discovery = discover_coordinates(discovery_frame, target_cols=probe.targets)
    top = discovery["targets"]["combined"]["rankings"][:12]
    top_names = [row["feature"] for row in top]
    expected_hit = any(
        name.startswith("coord_") and all(term in name for term in probe.expected_terms)
        for name in top_names
    )
    invented_cols = [c for c in top_names if c.startswith("coord_")]
    raw_cols = [c for c in _feature_columns(hidden, probe.targets) if c in hidden.columns]
    raw_metrics = _predictive_metrics(hidden, probe.targets, raw_cols)
    invented_metrics = _predictive_metrics(discovery_frame, probe.targets, list(dict.fromkeys(invented_cols + raw_cols[:2])))
    unknown_metrics = _predictive_metrics(discovery_frame, probe.targets, list(dict.fromkeys(invented_cols + raw_cols[:2])), structural_col=best_corr_name)
    return {
        "probe": probe.name,
        "source": probe.source,
        "hidden_coordinate": probe.hidden_coordinate,
        "best_reconstruction": best_corr_name,
        "best_abs_correlation": best_corr,
        "expected_formula_hit_top12": expected_hit,
        "top_invented_coordinates": invented_cols[:5],
        "raw_accuracy": raw_metrics["accuracy"],
        "invented_accuracy": invented_metrics["accuracy"],
        "unknown_prediction_accuracy": unknown_metrics["accuracy"],
        "unknown_prediction_macro_f1": unknown_metrics["macro_f1"],
        "compression_ratio": len(set(invented_cols[:5])) / max(len(raw_cols), 1),
    }


def _human_comparison(df: pd.DataFrame, targets: list[str], top_coordinate_cols: list[str]) -> dict[str, float]:
    y = _target(df, targets)
    cols = _feature_columns(df, targets)
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=43, stratify=stratify)
    results = {}
    for name, kind, features in [
        ("random_forest", "forest", cols),
        ("decision_tree_coordinates", "tree", top_coordinate_cols),
        ("nearest_neighbors", "knn", cols),
    ]:
        clf = _model(kind, x_train[features])
        clf.fit(x_train[features], y_train)
        pred = clf.predict(x_test[features])
        results[f"{name}_accuracy"] = float(accuracy_score(y_test, pred))
        results[f"{name}_macro_f1"] = float(f1_score(y_test, pred, average="macro", zero_division=0))
    return results


def _evaluate_wine_quality(cache_dir: Path) -> dict[str, Any] | None:
    candidates = [cache_dir / "winequality-red.csv", cache_dir / "pmlb" / "datasets" / "wine_quality_red" / "wine_quality_red.tsv.gz"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path, sep=";")
        else:
            df = pd.read_csv(path, sep="\t")
    except Exception as exc:
        return {"domain": "UCI_WINE_QUALITY", "status": "unreadable", "error": str(exc)}
    if "quality" not in df.columns:
        return {"domain": "UCI_WINE_QUALITY", "status": "missing_quality"}
    df = df.copy()
    df["quality_band"] = np.where(df["quality"] <= 5, "low", np.where(df["quality"] >= 7, "high", "mid"))
    df = df.drop(columns=["quality"])
    synthesized = synthesize_coordinates(df, targets=["quality_band"], max_base_features=8, max_new_features=220)
    report = discover_coordinates(synthesized, target_cols=["quality_band"])
    top = [row["feature"] for row in report["targets"]["combined"]["rankings"][:5]]
    metrics = _human_comparison(synthesized, ["quality_band"], top)
    return {
        "domain": "UCI_WINE_QUALITY",
        "status": "evaluated",
        "rows": int(len(df)),
        "top_coordinates": top,
        **metrics,
    }


def _attempt_download_sources(cache_dir: Path, allow_download: bool) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    for source in DATASET_SOURCES:
        row = dict(source)
        target = cache_dir / (Path(source["url"]).name if source["kind"] == "http" else source["name"])
        if not allow_download:
            if target.exists():
                row["status"] = "cached"
                row["path"] = str(target)
            else:
                row["status"] = "not_attempted_network_restricted"
            statuses.append(row)
            continue
        try:
            if source["kind"] == "http":
                urllib.request.urlretrieve(source["url"], target)
                row["status"] = "downloaded"
                row["path"] = str(target)
            else:
                if target.exists():
                    row["status"] = "cached"
                else:
                    subprocess.run(["git", "clone", "--depth", "1", source["url"], str(target)], check=True)
                    row["status"] = "downloaded"
                row["path"] = str(target)
        except Exception as exc:  # pragma: no cover - network path is environment-dependent.
            row["status"] = "failed"
            row["error"] = str(exc)
            print(f"WARNING: failed to download {source['name']}: {exc}. Continuing with cached/local files if available.")
        statuses.append(row)
    return statuses


def run_a_plus_plus_challenge(
    out: str,
    seed: int = 0,
    allow_download: bool = False,
    data_dir: str | None = None,
) -> dict[str, Any]:
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(data_dir) if data_dir else out_path / "cache"
    sources = [] if data_dir else _attempt_download_sources(cache_dir, allow_download=allow_download)
    probes = [
        generate_feynman_like(seed=seed + 1),
        generate_gas_law_like(seed=seed + 2),
        generate_ca_entropy_probe(seed=seed + 3),
        generate_phase_probe(seed=seed + 4),
    ]
    rediscovery_rows = [_score_probe(probe) for probe in probes]

    etp = generate_etp_raw(seed=seed + 5)
    etp_targets = ["implication_true"]
    etp_synth = synthesize_coordinates(etp, targets=etp_targets, max_base_features=8, max_new_features=260)
    etp_report = discover_coordinates(etp_synth, target_cols=etp_targets)
    etp_top = [row["feature"] for row in etp_report["targets"]["implication_true"]["rankings"][:5]]
    comparison_rows = []
    comparison_rows.append({"domain": "ETP_RAW", **_human_comparison(etp_synth, etp_targets, etp_top)})
    real_dataset_summary = run_real_dataset_benchmark(cache_dir, out_path)
    real_dataset_rows = real_dataset_summary["results"]

    red_df = pd.DataFrame(rediscovery_rows)
    human_df = pd.DataFrame(comparison_rows)
    red_df.to_csv(out_path / "scientific_rediscovery.csv", index=False)
    red_df[["probe", "unknown_prediction_accuracy", "unknown_prediction_macro_f1"]].to_csv(out_path / "unknown_prediction.csv", index=False)
    human_df.to_csv(out_path / "human_comparison.csv", index=False)
    pd.DataFrame(sources).to_json(out_path / "dataset_sources.json", orient="records", indent=2)

    rediscovery_score = float((red_df["best_abs_correlation"] >= 0.95).mean())
    invention_score = float(red_df["expected_formula_hit_top12"].mean())
    prediction_score = float(red_df["unknown_prediction_accuracy"].mean())
    compression_score = float((red_df["compression_ratio"] < 1.0).mean())
    transfer_score = float((red_df["invented_accuracy"] >= red_df["raw_accuracy"] - 0.05).mean())
    human_comparison_score = float((human_df["decision_tree_coordinates_accuracy"] >= human_df["nearest_neighbors_accuracy"]).mean())
    real_validation_count = len(real_dataset_rows)
    real_downloads = sum(1 for row in sources if row["status"] in {"downloaded", "cached"})

    scores = {
        "rediscovery_score": rediscovery_score,
        "invention_score": invention_score,
        "prediction_score": prediction_score,
        "compression_score": compression_score,
        "transfer_score": transfer_score,
        "human_comparison_score": human_comparison_score,
        "real_dataset_download_count": real_downloads,
        "real_dataset_validation_count": real_validation_count,
    }
    pd.DataFrame([scores]).to_csv(out_path / "a_plus_plus_scores.csv", index=False)

    if real_downloads == 0:
        verdict = "B) General methodology"
        a_status = "A+ not justified: coordinate invention succeeded on controlled/offline probes, but no live real-data downloads were available in this restricted run."
    elif real_validation_count < 2:
        verdict = "B) General methodology"
        a_status = "A+ not justified: coordinate invention succeeded and at least one real dataset was validated, but real scientific benchmark validation remains too thin."
    elif rediscovery_score >= 0.75 and invention_score >= 0.75 and prediction_score >= 0.75:
        verdict = "A+) Real coordinate invention"
        a_status = "A++ not justified without a genuinely unknown outcome benchmark."
    else:
        verdict = "C) Useful heuristic"
        a_status = "Coordinate invention evidence remains partial."

    summary = {
        **scores,
        "verdict": verdict,
        "status": a_status,
        "rediscovery": rediscovery_rows,
        "human_comparison": comparison_rows,
        "real_dataset_results": real_dataset_rows,
        "data_dir": str(cache_dir),
    }
    write_json(summary, str(out_path / "a_plus_plus_summary.json"))
    _write_a_plus_plus_report(out_path, summary)
    return summary


def _write_a_plus_plus_report(out_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# A++ Discovery Challenge Report",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        summary["status"],
        "",
        "## Scores",
        "",
        f"- rediscovery score: {summary['rediscovery_score']:.3f}",
        f"- invention score: {summary['invention_score']:.3f}",
        f"- unknown prediction score: {summary['prediction_score']:.3f}",
        f"- compression score: {summary['compression_score']:.3f}",
        f"- transfer score: {summary['transfer_score']:.3f}",
        f"- human comparison score: {summary['human_comparison_score']:.3f}",
        f"- real dataset downloads: {summary['real_dataset_download_count']}",
        f"- real dataset validations: {summary['real_dataset_validation_count']}",
        "",
        "## Rediscovery Probes",
        "",
    ]
    for row in summary["rediscovery"]:
        lines.extend(
            [
                f"### {row['probe']}",
                f"- hidden coordinate: {row['hidden_coordinate']}",
                f"- best reconstruction: {row['best_reconstruction']}",
                f"- absolute correlation: {row['best_abs_correlation']:.3f}",
                f"- formula hit in top 12: {row['expected_formula_hit_top12']}",
                f"- unknown prediction accuracy: {row['unknown_prediction_accuracy']:.3f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Real Dataset Validation",
            "",
        ]
    )
    if summary["real_dataset_results"]:
        for row in summary["real_dataset_results"]:
            name = row.get("domain", row.get("dataset", "UNKNOWN_DATASET"))
            status = row.get("status", "evaluated")
            top_coordinates = row.get("top_coordinates", row.get("top_invented_coordinates", "[]"))
            if isinstance(top_coordinates, str):
                top_text = top_coordinates
            else:
                top_text = ", ".join(top_coordinates)
            coord_score = row.get("coordinate_accuracy", row.get("coordinate_r2", float("nan")))
            lines.extend(
                [
                    f"### {name}",
                    f"- status: {status}",
                    f"- rows: {row.get('rows', 'n/a')}",
                    f"- top coordinates: {top_text}",
                    f"- coordinate score: {coord_score:.3f}" if coord_score == coord_score else "",
                    "",
                ]
            )
    else:
        lines.extend(["No real downloaded datasets were validated in this run.", ""])
    lines.extend(
        [
            "## Ultimate Question",
            "",
            "The current evidence still favors B: hard problems often become easier because the correct coordinates have not yet been discovered.",
            "However, A++ is not awarded here because the run did not validate against downloaded real scientific benchmarks or genuinely previously unknown outcomes.",
        ]
    )
    (out_path / "a_plus_plus_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
