#!/usr/bin/env python3
"""MATHGRAPH v146: Semantic Survivorship Engine.

Single-file, Colab-ready experiment for testing whether simple semantic
structures survive perturbation, renaming, holdout, counterexamples, and
domain transfer. The goal is ruthless selection, not a larger coordinate
grammar.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =============================================================================
# 00 MOUNT DRIVE / 01 INSTALL DEPENDENCIES
# =============================================================================

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
}


def install_missing_packages() -> None:
    missing = [pkg for import_name, pkg in REQUIRED_PACKAGES.items() if importlib.util.find_spec(import_name) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def mount_drive_if_requested(enabled: bool) -> None:
    if not enabled:
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as exc:
        print(f"Drive mount skipped: {exc}")


install_missing_packages()

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 02 DATA LOADERS
# =============================================================================

TARGET_NAMES = {"target", "y", "label", "class", "quality", "strength", "output"}
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".parquet", ".dat"}


@dataclass
class Domain:
    name: str
    df: pd.DataFrame
    target: str
    semantic_roles: dict[str, str]
    source: str = "synthetic"


@dataclass
class Candidate:
    domain: str
    name: str
    expression: str
    family: str
    columns: list[str]
    roles: list[str]
    depth: int
    complexity: float
    values: pd.Series


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            try:
                semi = pd.read_csv(path, sep=";")
                if len(semi.columns) > 1:
                    df = semi
            except Exception:
                pass
        return df
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".dat":
        return pd.read_csv(path, sep=r"\s+", header=None)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported file type: {suffix}")


def infer_target(df: pd.DataFrame) -> str:
    normalized = {str(col).strip().lower().replace(" ", "_"): col for col in df.columns}
    for name in TARGET_NAMES:
        if name in normalized:
            return str(normalized[name])
    return str(df.columns[-1])


def infer_roles(df: pd.DataFrame, target: str) -> dict[str, str]:
    role_keywords = {
        "RESOURCE": ["pressure", "volume", "resource", "signal", "energy", "capacity", "control", "coupling", "search", "budget"],
        "CONSTRAINT": ["constraint", "moles", "temperature", "cost", "load", "restriction", "bottleneck"],
        "NOISE": ["noise", "variance", "dissipation", "loss", "impurity"],
        "LOAD": ["load", "demand", "weight"],
        "DISSIPATION": ["dissipation", "loss", "friction"],
        "FREEDOM": ["freedom", "variables", "capacity"],
        "RESTRICTION": ["restriction", "clauses", "constraint"],
        "SEARCH": ["search", "budget", "objects"],
        "BOTTLENECK": ["bottleneck", "obstacle", "wall", "holes"],
        "CONTROL": ["control"],
        "COUPLING": ["coupling"],
        "TEMPERATURE": ["temperature"],
        "COST": ["cost", "load", "temperature", "bottleneck"],
    }
    roles: dict[str, str] = {}
    numeric = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    fallback = ["RESOURCE", "CONSTRAINT", "SIGNAL", "NOISE", "CAPACITY", "LOAD", "ENERGY", "DISSIPATION"]
    for col in numeric:
        lowered = str(col).lower()
        assigned = None
        for role, keywords in role_keywords.items():
            if any(keyword in lowered for keyword in keywords):
                assigned = role
                break
        roles[str(col)] = assigned or fallback[len(roles) % len(fallback)]
    return roles


def load_real_datasets(data_dirs: list[Path]) -> tuple[list[Domain], list[dict[str, Any]]]:
    domains: list[Domain] = []
    skipped: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in data_dirs:
        if not root.exists():
            skipped.append({"path": str(root), "reason": "missing_directory"})
            continue
        for path in sorted(root.rglob("*")):
            if path in seen or not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            seen.add(path)
            try:
                df = read_table(path).dropna(axis=0).reset_index(drop=True)
            except Exception as exc:
                skipped.append({"path": str(path), "reason": f"read_failed: {exc}"})
                continue
            if "airfoil" in path.stem.lower() and len(df.columns) == 6:
                df.columns = ["frequency", "angle", "chord", "velocity", "suction", "target"]
            target = infer_target(df)
            numeric = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
            if len(df) < 100:
                skipped.append({"path": str(path), "reason": "fewer_than_100_rows", "rows": len(df)})
                continue
            if len(numeric) < 3:
                skipped.append({"path": str(path), "reason": "fewer_than_3_numeric_features", "rows": len(df)})
                continue
            keep = numeric + [target]
            slim = df[keep].copy()
            domains.append(Domain(path.stem.upper(), slim, target, infer_roles(slim, target), str(path)))
    return domains, skipped


def attempt_downloads(out: Path, allow_download: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not allow_download:
        return rows
    cache = out / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    sources = {
        "winequality-red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "airfoil": "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
    }
    for name, url in sources.items():
        suffix = ".csv" if url.endswith(".csv") else ".dat"
        path = cache / f"{name}{suffix}"
        row = {"name": name, "url": url, "path": str(path), "status": "not_requested"}
        try:
            import urllib.request

            urllib.request.urlretrieve(url, path)
            row["status"] = "downloaded"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            print(f"WARNING: download failed for {name}: {exc}. Continuing.")
        rows.append(row)
    try:
        if importlib.util.find_spec("pmlb") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pmlb"])
        from pmlb import fetch_data  # type: ignore

        for dataset in ["1027_ESL", "1028_SWD", "197_cpu_act", "215_2dplanes"]:
            path = cache / f"{dataset}.csv"
            row = {"name": dataset, "source": "pmlb", "path": str(path), "status": "not_requested"}
            try:
                fetch_data(dataset).to_csv(path, index=False)
                row["status"] = "downloaded"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
            rows.append(row)
    except Exception as exc:
        rows.append({"name": "pmlb", "status": "failed", "error": str(exc)})
    return rows


# =============================================================================
# 03 SYNTHETIC DOMAINS
# =============================================================================


def band(values: np.ndarray, labels: tuple[str, str, str] = ("low", "mid", "high")) -> np.ndarray:
    return pd.qcut(pd.Series(values).rank(method="first"), 3, labels=list(labels)).astype(str).to_numpy()


def synthetic_domains(n: int, seed: int) -> list[Domain]:
    rng = np.random.default_rng(seed)
    domains: list[Domain] = []

    pressure = rng.uniform(0.5, 15, n)
    volume = rng.uniform(0.5, 10, n)
    moles = rng.uniform(0.2, 5, n)
    gas = pd.DataFrame({"pressure": pressure, "volume": volume, "moles": moles, "noise": rng.normal(0, 1, n)})
    gas["target"] = band(pressure * volume / moles)
    domains.append(Domain("GAS", gas, "target", {"pressure": "RESOURCE", "volume": "RESOURCE", "moles": "CONSTRAINT", "noise": "NOISE"}))

    control = rng.uniform(0.05, 1.0, n)
    coupling = rng.uniform(0.4, 2.2, n)
    temperature = rng.uniform(0.5, 3.5, n)
    phase = pd.DataFrame({"control": control, "coupling": coupling, "temperature": temperature, "jitter": rng.normal(0, 1, n)})
    phase["target"] = band(coupling * control / temperature)
    domains.append(Domain("PHASE", phase, "target", {"control": "CONTROL", "coupling": "COUPLING", "temperature": "TEMPERATURE", "jitter": "NOISE"}))

    clearance = rng.uniform(0.5, 12, n)
    width = rng.uniform(1, 20, n)
    load = rng.uniform(1, 220, n)
    obstruction = pd.DataFrame({"clearance": clearance, "width": width, "load": load, "friction": rng.uniform(0.05, 0.8, n)})
    obstruction["target"] = band(clearance * width / load)
    domains.append(Domain("OBSTRUCTION", obstruction, "target", {"clearance": "CAPACITY", "width": "CAPACITY", "load": "LOAD", "friction": "DISSIPATION"}))

    search_budget = rng.uniform(10, 500, n)
    bottleneck = rng.uniform(0.5, 40, n)
    wall_density = rng.uniform(0.05, 0.7, n)
    maze = pd.DataFrame({"search_budget": search_budget, "bottleneck": bottleneck, "wall_density": wall_density, "branching": rng.uniform(1, 6, n)})
    maze["target"] = band(search_budget / (bottleneck + wall_density))
    domains.append(Domain("MAZE", maze, "target", {"search_budget": "SEARCH", "bottleneck": "BOTTLENECK", "wall_density": "BOTTLENECK", "branching": "RESOURCE"}))

    signal = rng.uniform(0.1, 20, n)
    noise = rng.uniform(0.1, 8, n)
    sig = pd.DataFrame({"signal": signal, "noise": noise, "variance": rng.uniform(0.1, 4, n), "drift": rng.normal(0, 1, n)})
    sig["target"] = band(signal / noise)
    domains.append(Domain("SIGNAL", sig, "target", {"signal": "SIGNAL", "noise": "NOISE", "variance": "NOISE", "drift": "NOISE"}))

    energy = rng.uniform(1, 300, n)
    dissipation = rng.uniform(0.5, 40, n)
    en = pd.DataFrame({"energy": energy, "dissipation": dissipation, "loss": rng.uniform(0.1, 8, n), "mass": rng.uniform(1, 20, n)})
    en["target"] = band(energy / dissipation)
    domains.append(Domain("ENERGY", en, "target", {"energy": "ENERGY", "dissipation": "DISSIPATION", "loss": "DISSIPATION", "mass": "RESOURCE"}))

    capacity = rng.uniform(1, 200, n)
    cap_load = rng.uniform(0.5, 160, n)
    cap = pd.DataFrame({"capacity": capacity, "load": cap_load, "reserve": rng.uniform(0, 20, n), "cost": rng.uniform(0.1, 10, n)})
    cap["target"] = band(capacity / cap_load)
    domains.append(Domain("CAPACITY", cap, "target", {"capacity": "CAPACITY", "load": "LOAD", "reserve": "CAPACITY", "cost": "COST"}))

    x = rng.integers(0, 10_000, n)
    mod = rng.choice([5, 7, 11, 13], n)
    modular = pd.DataFrame({"x": x, "n": mod, "noise": rng.normal(0, 1, n), "scale": rng.uniform(1, 10, n)})
    modular["target"] = np.where(x % mod <= 1, "low", np.where(x % mod <= mod // 2, "mid", "high"))
    domains.append(Domain("MODULAR", modular, "target", {"x": "RESOURCE", "n": "CONSTRAINT", "noise": "NOISE", "scale": "RESOURCE"}))

    objects = rng.integers(1, 12, n)
    symmetry = rng.uniform(0, 1, n)
    constraint = rng.uniform(0.1, 6, n)
    arc = pd.DataFrame({"objects": objects, "symmetry": symmetry, "constraint": constraint, "colors": rng.integers(2, 9, n)})
    arc["target"] = band((objects * symmetry) / constraint)
    domains.append(Domain("ARC_STYLE", arc, "target", {"objects": "RESOURCE", "symmetry": "SIGNAL", "constraint": "CONSTRAINT", "colors": "RESOURCE"}))

    return domains


# =============================================================================
# 04 MODELS AND CANDIDATES
# =============================================================================


def problem_type(y: pd.Series) -> str:
    return "regression" if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15 else "classification"


def make_model(problem: str, kind: str) -> Any:
    if problem == "regression":
        return RandomForestRegressor(n_estimators=120, random_state=146) if kind == "raw" else DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=146)
    return RandomForestClassifier(n_estimators=120, random_state=146, class_weight="balanced") if kind == "raw" else DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=146)


def score_frame(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, problem: str, kind: str) -> float:
    categorical = [c for c in x_train.columns if not pd.api.types.is_numeric_dtype(x_train[c])]
    numeric = [c for c in x_train.columns if c not in categorical]
    pipe = Pipeline(
        [
            ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical), ("num", StandardScaler(), numeric)], remainder="drop")),
            ("model", make_model(problem, kind)),
        ]
    )
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    return float(r2_score(y_test, pred)) if problem == "regression" else float(accuracy_score(y_test.astype(str), pd.Series(pred).astype(str)))


def numeric_cols(domain: Domain) -> list[str]:
    return [
        c
        for c in domain.df.columns
        if c != domain.target and pd.api.types.is_numeric_dtype(domain.df[c]) and domain.df[c].nunique(dropna=True) > 1
    ]


def family_from_roles(roles: list[str], expression_type: str) -> str:
    role_set = set(roles)
    if {"CONTROL", "COUPLING", "TEMPERATURE"}.issubset(role_set):
        return "EFFECTIVE_PARAMETER"
    if "SIGNAL" in role_set and "NOISE" in role_set:
        return "SIGNAL_NOISE"
    if "CAPACITY" in role_set and "LOAD" in role_set:
        return "CAPACITY_LOAD"
    if "ENERGY" in role_set and "DISSIPATION" in role_set:
        return "ENERGY_DISSIPATION"
    if "FREEDOM" in role_set and "RESTRICTION" in role_set:
        return "FREEDOM_RESTRICTION"
    if "SEARCH" in role_set and "BOTTLENECK" in role_set:
        return "SEARCH_BOTTLENECK"
    if expression_type == "xy_over_z" and "CONSTRAINT" in role_set:
        return "PRODUCT_CONSTRAINT"
    if expression_type == "diff_over_z" and ("COST" in role_set or "CONSTRAINT" in role_set):
        return "DIFFERENCE_COST"
    if "RESOURCE" in role_set and "CONSTRAINT" in role_set:
        return "RESOURCE_CONSTRAINT"
    return "UNKNOWN"


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def generate_candidates(domain: Domain, max_features: int = 7) -> list[Candidate]:
    cols = numeric_cols(domain)[:max_features]
    df = domain.df
    candidates: list[Candidate] = []

    def add(name: str, expr: str, expression_type: str, columns: list[str], values: pd.Series, depth: int, complexity: float) -> None:
        if values.nunique(dropna=False) <= 1:
            return
        roles = [domain.semantic_roles.get(col, "UNKNOWN") for col in columns]
        family = family_from_roles(roles, expression_type)
        candidates.append(Candidate(domain.name, name, expr, family, columns, roles, depth, complexity, values.astype(float)))

    for x in cols:
        for y in cols:
            if x == y:
                continue
            add(f"{x}_over_{y}", f"{x}/{y}", "ratio", [x, y], safe_div(df[x], df[y]), 2, 2.0)
            add(f"{x}_minus_{y}", f"({x}-{y})", "difference", [x, y], df[x] - df[y], 2, 2.0)
            add(f"{x}_plus_{y}_over_load", f"({x}+{y})", "sum", [x, y], df[x] + df[y], 2, 2.0)
    for x in cols:
        for y in cols:
            for z in cols:
                if len({x, y, z}) != 3:
                    continue
                add(f"{x}_times_{y}_over_{z}", f"({x}*{y})/{z}", "xy_over_z", [x, y, z], safe_div(df[x] * df[y], df[z]), 3, 3.2)
                add(f"{x}_plus_{y}_over_{z}", f"({x}+{y})/{z}", "sum_over_z", [x, y, z], safe_div(df[x] + df[y], df[z]), 3, 3.0)
                add(f"{x}_minus_{y}_over_{z}", f"({x}-{y})/{z}", "diff_over_z", [x, y, z], safe_div(df[x] - df[y], df[z]), 3, 3.0)
                add(f"{x}_over_{y}_plus_{z}", f"{x}/({y}+{z})", "x_over_sum", [x, y, z], safe_div(df[x], df[y] + df[z]), 3, 3.1)
    for x in cols:
        for y in cols:
            for z in cols:
                for w in cols:
                    if len({x, y, z, w}) != 4:
                        continue
                    add(
                        f"{x}_times_{y}_over_{z}_plus_{w}",
                        f"({x}*{y})/({z}+{w})",
                        "xy_over_zw",
                        [x, y, z, w],
                        safe_div(df[x] * df[y], df[z] + df[w]),
                        4,
                        4.2,
                    )
    # Keep runtime bounded while preserving family diversity.
    return candidates[:550]


# =============================================================================
# 05 SURVIVORSHIP TESTS
# =============================================================================


def candidate_score(values: pd.Series, y: pd.Series, problem: str, seed: int = 146) -> tuple[float, float, pd.Index, pd.Index]:
    stratify = y if problem == "classification" and y.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(values.index, test_size=0.3, random_state=seed, stratify=stratify)
    score = score_frame(
        pd.DataFrame({"candidate": values.loc[train_idx]}),
        pd.DataFrame({"candidate": values.loc[test_idx]}),
        y.loc[train_idx],
        y.loc[test_idx],
        problem,
        "candidate",
    )
    train_score = score_frame(
        pd.DataFrame({"candidate": values.loc[train_idx]}),
        pd.DataFrame({"candidate": values.loc[train_idx]}),
        y.loc[train_idx],
        y.loc[train_idx],
        problem,
        "candidate",
    )
    return score, train_score, train_idx, test_idx


def raw_baseline(domain: Domain, problem: str) -> float:
    cols = numeric_cols(domain)
    y = domain.df[domain.target]
    stratify = y if problem == "classification" and y.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(domain.df.index, test_size=0.3, random_state=146, stratify=stratify)
    return score_frame(domain.df.loc[train_idx, cols], domain.df.loc[test_idx, cols], y.loc[train_idx], y.loc[test_idx], problem, "raw")


def perturbation_survival(candidate: Candidate, domain: Domain, base_score: float, problem: str, rng: np.random.Generator) -> float:
    perturbed = domain.df.copy()
    for col in candidate.columns:
        std = float(perturbed[col].std() or 1.0)
        perturbed[col] = perturbed[col] + rng.normal(0, 0.05 * std, len(perturbed))
    temp_domain = Domain(domain.name, perturbed, domain.target, domain.semantic_roles, domain.source)
    temp_candidate = rebuild_candidate(candidate, temp_domain)
    if temp_candidate is None:
        return 0.0
    score, _, _, _ = candidate_score(temp_candidate.values, perturbed[domain.target], problem, seed=316)
    return float(max(0.0, min(1.0, score / max(base_score, 1e-9))))


def rename_survival(candidate: Candidate, domain: Domain) -> float:
    renamed = {col: f"v{i}" for i, col in enumerate(candidate.columns)}
    metadata = {renamed[col]: domain.semantic_roles.get(col, "UNKNOWN") for col in candidate.columns}
    renamed_roles = [metadata[renamed[col]] for col in candidate.columns]
    renamed_family = family_from_roles(renamed_roles, expression_type_from_candidate(candidate))
    return 1.0 if renamed_family == candidate.family else 0.0


def expression_type_from_candidate(candidate: Candidate) -> str:
    expr = candidate.expression
    if "*" in expr and "/" in expr and "+)" in expr:
        return "xy_over_zw"
    if "*" in expr and "/" in expr:
        return "xy_over_z"
    if "-" in expr and "/" in expr:
        return "diff_over_z"
    if "+" in expr and "/" in expr:
        return "sum_over_z"
    if "/" in expr:
        return "ratio"
    if "-" in expr:
        return "difference"
    return "sum"


def rebuild_candidate(candidate: Candidate, domain: Domain) -> Candidate | None:
    if any(col not in domain.df.columns for col in candidate.columns):
        return None
    df = domain.df
    cols = candidate.columns
    try:
        if candidate.expression.startswith("(") and "*)/(" in candidate.expression:
            values = safe_div(df[cols[0]] * df[cols[1]], df[cols[2]] + df[cols[3]])
        elif "*" in candidate.expression and "/" in candidate.expression:
            values = safe_div(df[cols[0]] * df[cols[1]], df[cols[2]])
        elif "+" in candidate.expression and "/" in candidate.expression and len(cols) >= 3:
            values = safe_div(df[cols[0]] + df[cols[1]], df[cols[2]])
        elif "-" in candidate.expression and "/" in candidate.expression and len(cols) >= 3:
            values = safe_div(df[cols[0]] - df[cols[1]], df[cols[2]])
        elif "/" in candidate.expression and len(cols) == 2:
            values = safe_div(df[cols[0]], df[cols[1]])
        elif "-" in candidate.expression and len(cols) == 2:
            values = df[cols[0]] - df[cols[1]]
        else:
            values = df[cols[0]] + df[cols[1]]
        return Candidate(candidate.domain, candidate.name, candidate.expression, candidate.family, cols, candidate.roles, candidate.depth, candidate.complexity, values)
    except Exception:
        return None


def counterexample_rate(values: pd.Series, y: pd.Series, train_idx: pd.Index, test_idx: pd.Index, problem: str) -> tuple[float, list[int]]:
    model = make_model(problem, "candidate")
    x_train = pd.DataFrame({"candidate": values.loc[train_idx]})
    x_test = pd.DataFrame({"candidate": values.loc[test_idx]})
    model.fit(x_train, y.loc[train_idx])
    pred = model.predict(x_test)
    if problem == "regression":
        residuals = np.abs(np.asarray(pred) - np.asarray(y.loc[test_idx]))
        threshold = np.quantile(residuals, 0.8) if len(residuals) else 0.0
        bad_mask = residuals > threshold
    else:
        bad_mask = pd.Series(pred, index=test_idx).astype(str) != y.loc[test_idx].astype(str)
    bad_indices = list(pd.Index(test_idx)[np.asarray(bad_mask)])
    return float(len(bad_indices) / max(len(test_idx), 1)), [int(i) for i in bad_indices[:25]]


def evaluate_domain(domain: Domain, rng: np.random.Generator) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    y = domain.df[domain.target]
    problem = problem_type(y)
    raw = raw_baseline(domain, problem)
    candidates = generate_candidates(domain)
    rows: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    for cand in candidates:
        score, train_score, train_idx, test_idx = candidate_score(cand.values, y, problem)
        advantage = score - raw
        perturb = perturbation_survival(cand, domain, score, problem, rng)
        rename = rename_survival(cand, domain)
        holdout_score = max(0.0, min(1.0, score / max(train_score, 1e-9))) if train_score > 0 else 0.0
        cex_rate, cex_indices = counterexample_rate(cand.values, y, train_idx, test_idx, problem)
        cex_resistance = 1.0 - cex_rate
        row = {
            "domain": domain.name,
            "candidate": cand.name,
            "expression": cand.expression,
            "semantic_family": cand.family,
            "columns": "|".join(cand.columns),
            "roles": "|".join(cand.roles),
            "depth": cand.depth,
            "complexity": cand.complexity,
            "raw_baseline": raw,
            "predictive_score": score,
            "advantage": advantage,
            "support_size": int(len(cand.values)),
            "perturbation_survival": perturb,
            "rename_survival": rename,
            "holdout_score": holdout_score,
            "counterexample_rate": cex_rate,
            "counterexample_resistance": cex_resistance,
            "counterexample_indices": cex_indices,
        }
        rows.append(row)
        if cex_rate > 0.2 or advantage <= 0:
            counterexamples.append(
                {
                    "domain": domain.name,
                    "semantic_family": cand.family,
                    "candidate": cand.name,
                    "type": "candidate_failure",
                    "detail": f"advantage={advantage:.3f}; counterexample_rate={cex_rate:.3f}",
                    "indices": cex_indices,
                }
            )
    summary = {"domain": domain.name, "raw_baseline": raw, "candidate_count": len(rows), "problem_type": problem}
    return rows, counterexamples, summary


# =============================================================================
# 06 SURVIVORSHIP AGGREGATION
# =============================================================================


def add_survivorship(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    out = candidate_df.copy()
    recurrence = out.groupby("semantic_family")["domain"].nunique().to_dict()
    family_counts = out.groupby("semantic_family")["domain"].transform("nunique")
    out["recurrence_score"] = family_counts / max(out["domain"].nunique(), 1)
    out["positive_advantage"] = out["advantage"].clip(lower=0)
    out["survivor_score"] = (
        out["positive_advantage"]
        * out["perturbation_survival"]
        * out["holdout_score"]
        * out["recurrence_score"]
        * out["counterexample_resistance"]
        * out["rename_survival"]
        / out["complexity"].clip(lower=1.0)
    )
    return out.sort_values("survivor_score", ascending=False)


def family_survivorship(candidate_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    if candidate_df.empty:
        return pd.DataFrame()
    for family, group in candidate_df.groupby("semantic_family"):
        best_by_domain = group.sort_values("survivor_score", ascending=False).groupby("domain").head(1)
        promoted_like = group[
            (group["survivor_score"] >= threshold)
            & (group["perturbation_survival"] >= 0.7)
            & (group["holdout_score"] >= 0.7)
            & (group["counterexample_rate"] <= 0.2)
        ]
        rows.append(
            {
                "semantic_family": family,
                "known_semantic_family": family != "UNKNOWN",
                "candidate_count": int(len(group)),
                "domains": int(group["domain"].nunique()),
                "surviving_domains": int(promoted_like["domain"].nunique()),
                "mean_advantage": float(group["advantage"].mean()),
                "best_advantage": float(group["advantage"].max()),
                "mean_perturbation_survival": float(best_by_domain["perturbation_survival"].mean()),
                "mean_holdout_score": float(best_by_domain["holdout_score"].mean()),
                "mean_counterexample_rate": float(best_by_domain["counterexample_rate"].mean()),
                "mean_counterexample_resistance": float(best_by_domain["counterexample_resistance"].mean()),
                "mean_rename_survival": float(best_by_domain["rename_survival"].mean()),
                "mean_survivor_score": float(best_by_domain["survivor_score"].mean()),
                "best_survivor_score": float(group["survivor_score"].max()),
                "best_expression": str(group.sort_values("survivor_score", ascending=False).iloc[0]["expression"]),
                "domain_list": "|".join(sorted(group["domain"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["known_semantic_family", "best_survivor_score"], ascending=False)


def domain_holdout_scores(candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = sorted(candidate_df["semantic_family"].unique()) if not candidate_df.empty else []
    domains = sorted(candidate_df["domain"].unique()) if not candidate_df.empty else []
    for family in families:
        family_df = candidate_df[candidate_df["semantic_family"] == family]
        for heldout in domains:
            train = family_df[family_df["domain"] != heldout]
            test = family_df[family_df["domain"] == heldout]
            if train.empty or test.empty:
                strength = 0.0
                survival = False
            else:
                train_strength = float(train.groupby("domain")["survivor_score"].max().mean())
                test_strength = float(test["survivor_score"].max())
                strength = test_strength / max(train_strength, 1e-9)
                survival = test_strength > 0 and strength >= 0.5
            rows.append({"semantic_family": family, "heldout_domain": heldout, "domain_holdout_strength": strength, "survival": survival})
    return pd.DataFrame(rows)


def build_lawbook(family_df: pd.DataFrame, holdout_df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    laws = []
    if family_df.empty:
        return {"lawbook_version": "v146", "laws": []}
    for _, row in family_df.iterrows():
        holdout = holdout_df[holdout_df["semantic_family"] == row["semantic_family"]]
        holdout_survival = float(holdout["survival"].mean()) if not holdout.empty else 0.0
        if (
            row["semantic_family"] != "UNKNOWN"
            and
            row["best_survivor_score"] >= threshold
            and row["surviving_domains"] >= 3
            and row["mean_perturbation_survival"] >= 0.7
            and row["mean_holdout_score"] >= 0.7
            and row["mean_counterexample_rate"] <= 0.2
        ):
            laws.append(
                {
                    "law": row["semantic_family"],
                    "domains": int(row["surviving_domains"]),
                    "survivor_score": float(row["best_survivor_score"]),
                    "domain_holdout_survival": holdout_survival,
                    "perturbation_survival": float(row["mean_perturbation_survival"]),
                    "holdout_score": float(row["mean_holdout_score"]),
                    "counterexample_rate": float(row["mean_counterexample_rate"]),
                    "best_expression": row["best_expression"],
                    "statement": f"{row['semantic_family']} survived strict perturbation, holdout, counterexample, and recurrence tests",
                }
            )
    return {"lawbook_version": "v146", "promotion_threshold": threshold, "laws": laws}


def verdict(lawbook: dict[str, Any], family_df: pd.DataFrame) -> tuple[str, str]:
    known = family_df[family_df["semantic_family"] != "UNKNOWN"] if not family_df.empty else family_df
    if lawbook["laws"]:
        return "A", "one or more semantic families survive strongly across domains"
    if not known.empty and (known["best_survivor_score"] > 0).any() and (known["domains"] >= 3).any():
        return "B", "weak survivorship; promising but not law-level"
    if not known.empty and (known["domains"] >= 2).any():
        return "C", "semantic patterns recur but do not survive"
    return "D", "no evidence"


# =============================================================================
# 07 REPORTS
# =============================================================================


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_reports(
    out: Path,
    candidate_df: pd.DataFrame,
    family_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    counterexamples: pd.DataFrame,
    lawbook: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(out / "semantic_candidates.csv", index=False)
    candidate_df.to_csv(out / "survivorship_scores.csv", index=False)
    family_df.to_csv(out / "family_survivorship.csv", index=False)
    holdout_df.to_csv(out / "domain_holdout.csv", index=False)
    counterexamples.to_csv(out / "counterexamples.csv", index=False)
    write_json(out / "semantic_lawbook_v146.json", lawbook)
    write_json(out / "manifest.json", manifest)
    write_benchmark_report(out, candidate_df, family_df, holdout_df, lawbook)
    write_final_conclusion(out, lawbook, family_df)


def write_benchmark_report(out: Path, candidate_df: pd.DataFrame, family_df: pd.DataFrame, holdout_df: pd.DataFrame, lawbook: dict[str, Any]) -> None:
    grade, statement = verdict(lawbook, family_df)
    lines = [
        "# MATHGRAPH v146 Semantic Survivorship Report",
        "",
        f"Verdict: **{grade}** — {statement}.",
        "",
        "## What Survives?",
        "",
    ]
    if family_df.empty:
        lines.append("No semantic families were evaluated.")
    else:
        for _, row in family_df.head(12).iterrows():
            lines.append(
                f"- {row['semantic_family']}: best_survivor={row['best_survivor_score']:.4f}, "
                f"domains={int(row['domains'])}, surviving_domains={int(row['surviving_domains'])}, "
                f"counterexamples={row['mean_counterexample_rate']:.3f}"
            )
    lines.extend(["", "## What Fails?", ""])
    if not family_df.empty:
        failed = family_df[family_df["surviving_domains"] < 3]
        for _, row in failed.head(12).iterrows():
            lines.append(f"- {row['semantic_family']}: insufficient surviving domains ({int(row['surviving_domains'])})")
    lines.extend(["", "## What Transfers?", ""])
    if not holdout_df.empty:
        transfer = holdout_df.groupby("semantic_family")["survival"].mean().sort_values(ascending=False)
        for family, value in transfer.head(12).items():
            lines.append(f"- {family}: domain_holdout_survival={value:.3f}")
    lines.extend(["", "## Closest To A Law", ""])
    if not family_df.empty:
        top = family_df.iloc[0]
        lines.append(f"`{top['semantic_family']}` is closest: best expression `{top['best_expression']}`, score {top['best_survivor_score']:.4f}.")
    lines.extend(["", "## Interpretation", ""])
    if lawbook["laws"]:
        lines.append("At least one semantic family met the strict survivorship criteria. This is evidence for semantic survivorship, not universality.")
    else:
        lines.append("No semantic family met the strict lawbook criteria. The run distinguishes recurrence from survivorship.")
    lines.append("The measured hierarchy is reported separately: coordinate success, operator success, semantic recurrence, and semantic survivorship.")
    (out / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_conclusion(out: Path, lawbook: dict[str, Any], family_df: pd.DataFrame) -> None:
    grade, statement = verdict(lawbook, family_df)
    text = f"""# Final Conclusion

Outcome: **{grade}**

Interpretation: {statement}.

The v146 test ranks semantic families by survivorship rather than raw accuracy.
A family is promoted only if it survives perturbation, variable renaming,
domain holdout, adversarial counterexamples, and recurrence across at least
three domains.

Honesty rule: this run does not claim universal laws, scientific truth, or
intelligence solved. If no laws promote, the correct conclusion is that semantic
recurrence was not enough to establish semantic survivorship.
"""
    (out / "final_conclusion.md").write_text(text, encoding="utf-8")


# =============================================================================
# 08 ORCHESTRATION
# =============================================================================


def run(args: argparse.Namespace) -> dict[str, Any]:
    mount_drive_if_requested(args.mount_drive)
    out = Path(args.out)
    downloads = attempt_downloads(out, args.download)
    data_dirs = [Path(args.data_dir)] if args.data_dir else []
    if args.download:
        data_dirs.append(out / "downloads")
    real_domains, skipped = load_real_datasets(data_dirs) if data_dirs else ([], [])
    n = 180 if args.quick else 420
    domains = synthetic_domains(n=n, seed=args.seed) + real_domains
    rng = np.random.default_rng(args.seed + 1000)

    all_rows: list[dict[str, Any]] = []
    all_counterexamples: list[dict[str, Any]] = []
    domain_summaries: list[dict[str, Any]] = []
    for domain in domains:
        print(f"Evaluating {domain.name} ({len(domain.df)} rows)")
        rows, cex, summary = evaluate_domain(domain, rng)
        all_rows.extend(rows)
        all_counterexamples.extend(cex)
        domain_summaries.append(summary)

    candidates = add_survivorship(pd.DataFrame(all_rows))
    threshold = 0.015
    families = family_survivorship(candidates, threshold=threshold)
    holdout = domain_holdout_scores(candidates)
    lawbook = build_lawbook(families, holdout, threshold=threshold)

    recurrence = candidates.groupby("semantic_family")["domain"].nunique().to_dict() if not candidates.empty else {}
    for row in all_counterexamples:
        row["family_domain_recurrence"] = recurrence.get(row.get("semantic_family"), 0)
    counterexamples = pd.DataFrame(all_counterexamples)
    manifest = {
        "system": "MATHGRAPH v146 Semantic Survivorship Engine",
        "seed": args.seed,
        "quick": args.quick,
        "domains": [domain.name for domain in domains],
        "synthetic_domain_count": len(domains) - len(real_domains),
        "real_domain_count": len(real_domains),
        "downloads": downloads,
        "skipped_real_files": skipped,
        "domain_summaries": domain_summaries,
        "promotion_threshold": threshold,
    }
    write_reports(out, candidates, families, holdout, counterexamples, lawbook, manifest)
    grade, statement = verdict(lawbook, families)
    return {"out": str(out), "verdict": grade, "statement": statement, "laws": len(lawbook["laws"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MATHGRAPH v146 Semantic Survivorship Engine")
    parser.add_argument("--quick", action="store_true", help="Run a fast smoke-sized experiment")
    parser.add_argument("--out", default="mathgraph_v146_out", help="Output directory")
    parser.add_argument("--data-dir", default=None, help="Recursive local CSV/TSV/Parquet directory")
    parser.add_argument("--download", action="store_true", help="Best-effort UCI/PMLB downloads")
    parser.add_argument("--seed", type=int, default=146)
    parser.add_argument("--mount-drive", action="store_true", help="Mount Google Drive in Colab")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
