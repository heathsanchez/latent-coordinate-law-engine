#!/usr/bin/env python3
"""MATHGRAPH v145: Semantic Portal Discovery Engine.

Single-file, Colab-ready research engine for testing the hypothesis:

    Multiple coordinates may instantiate the same semantic continuation law.

The script runs the v144 coordinate/operator pass, then parses coordinates into
semantic continuation templates such as resource/constraint, capacity/load, and
search/bottleneck. It clusters semantic basins, evaluates semantic transfer,
evolves failed meanings, and exports a measured semantic lawbook. It
intentionally reports weak evidence as weak evidence.
"""

from __future__ import annotations

# =============================================================================
# 00 MOUNT DRIVE
# =============================================================================

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import textwrap
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


def mount_drive_if_requested(enabled: bool) -> None:
    if not enabled:
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        print("Mounted Google Drive at /content/drive")
    except Exception as exc:
        print(f"Drive mount skipped: {exc}")


# =============================================================================
# 01 INSTALL DEPENDENCIES
# =============================================================================

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "networkx": "networkx",
    "matplotlib": "matplotlib",
}

OPTIONAL_PACKAGES = {"openml": "openml", "pmlb": "pmlb"}


def install_missing_packages(include_optional: bool = False) -> None:
    packages = dict(REQUIRED_PACKAGES)
    if include_optional:
        packages.update(OPTIONAL_PACKAGES)
    missing: list[str] = []
    for import_name, package_name in packages.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(package_name)
    if not missing:
        return
    print(f"Installing missing packages: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


install_missing_packages(include_optional=False)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 02 DATASET DOWNLOADS
# =============================================================================

UCI_SOURCES = {
    "winequality-red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "airfoil_self_noise": "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
}

PMLB_DATASETS = [
    "1027_ESL",
    "1028_SWD",
    "1029_LEV",
    "1030_ERA",
    "201_pol",
    "197_cpu_act",
    "215_2dplanes",
    "529_pollen",
    "537_houses",
]

OPENML_DATASET_IDS = [287, 216, 42225]


def attempt_dataset_downloads(cache_dir: Path, allow_download: bool) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, url in UCI_SOURCES.items():
        suffix = ".csv" if url.endswith(".csv") else ".tsv"
        path = cache_dir / f"{name}{suffix}"
        row = {"name": name, "source": "uci", "url": url, "path": str(path), "status": "not_requested"}
        if path.exists():
            row["status"] = "cached"
        elif allow_download:
            try:
                import urllib.request

                urllib.request.urlretrieve(url, path)
                row["status"] = "downloaded"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                print(f"WARNING: failed to download {name}: {exc}. Continuing.")
        rows.append(row)

    if allow_download:
        try:
            install_missing_packages(include_optional=True)
            from pmlb import fetch_data  # type: ignore

            pmlb_dir = cache_dir / "pmlb"
            pmlb_dir.mkdir(exist_ok=True)
            for dataset in PMLB_DATASETS:
                path = pmlb_dir / f"{dataset}.csv"
                row = {"name": dataset, "source": "pmlb", "path": str(path), "status": "not_requested"}
                if path.exists():
                    row["status"] = "cached"
                else:
                    try:
                        data = fetch_data(dataset, local_cache_dir=str(pmlb_dir))
                        data.to_csv(path, index=False)
                        row["status"] = "downloaded"
                    except Exception as exc:
                        row["status"] = "failed"
                        row["error"] = str(exc)
                        print(f"WARNING: failed to download PMLB {dataset}: {exc}. Continuing.")
                rows.append(row)
        except Exception as exc:
            rows.append({"name": "pmlb", "source": "pmlb", "status": "failed", "error": str(exc)})
            print(f"WARNING: PMLB unavailable: {exc}. Continuing.")
        try:
            install_missing_packages(include_optional=True)
            import openml  # type: ignore

            openml_dir = cache_dir / "openml"
            openml_dir.mkdir(exist_ok=True)
            for dataset_id in OPENML_DATASET_IDS:
                path = openml_dir / f"openml_{dataset_id}.csv"
                row = {"name": f"openml_{dataset_id}", "source": "openml", "path": str(path), "status": "not_requested"}
                if path.exists():
                    row["status"] = "cached"
                else:
                    try:
                        dataset = openml.datasets.get_dataset(dataset_id)
                        target = dataset.default_target_attribute
                        x, y, _, _ = dataset.get_data(target=target, dataset_format="dataframe")
                        frame = x.copy()
                        frame["target"] = y
                        frame.to_csv(path, index=False)
                        row["status"] = "downloaded"
                    except Exception as exc:
                        row["status"] = "failed"
                        row["error"] = str(exc)
                        print(f"WARNING: failed to download OpenML {dataset_id}: {exc}. Continuing.")
                rows.append(row)
        except Exception as exc:
            rows.append({"name": "openml", "source": "openml", "status": "failed", "error": str(exc)})
            print(f"WARNING: OpenML unavailable: {exc}. Continuing.")
    return rows


# =============================================================================
# 03 REAL DATASET LOADERS
# =============================================================================

TARGET_NAMES = {"target", "y", "label", "class", "quality", "strength", "output"}


@dataclass
class DatasetSpec:
    name: str
    domain: str
    df: pd.DataFrame
    target: str
    source: str


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            try:
                semicolon = pd.read_csv(path, sep=";")
                if len(semicolon.columns) > 1:
                    df = semicolon
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


def load_real_datasets(data_dirs: list[Path]) -> tuple[list[DatasetSpec], list[dict[str, Any]]]:
    datasets: list[DatasetSpec] = []
    skipped: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in data_dirs:
        if not root.exists():
            skipped.append({"path": str(root), "reason": "missing_directory"})
            continue
        for path in sorted(root.rglob("*")):
            if path in seen or not path.is_file() or path.suffix.lower() not in {".csv", ".tsv", ".dat", ".parquet"}:
                continue
            seen.add(path)
            try:
                df = read_table(path)
            except Exception as exc:
                skipped.append({"path": str(path), "reason": f"read_failed: {exc}"})
                continue
            if path.name == "airfoil_self_noise.tsv" or "airfoil" in path.stem.lower():
                if len(df.columns) == 6:
                    df.columns = ["frequency", "angle", "chord", "velocity", "suction", "target"]
            target = infer_target(df)
            numeric_features = [
                col
                for col in df.columns
                if col != target and pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 1
            ]
            if len(df) < 100:
                skipped.append({"path": str(path), "reason": "fewer_than_100_rows", "rows": int(len(df))})
                continue
            if len(numeric_features) < 3:
                skipped.append(
                    {
                        "path": str(path),
                        "reason": "fewer_than_3_numeric_features",
                        "rows": int(len(df)),
                        "numeric_features": len(numeric_features),
                    }
                )
                continue
            datasets.append(DatasetSpec(path.stem, "REAL", df, target, str(path)))
    return datasets, skipped


# =============================================================================
# 04 SYNTHETIC WORLDS
# =============================================================================


def _band(values: np.ndarray, labels: tuple[str, str, str]) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first")
    return pd.qcut(ranks, 3, labels=list(labels)).astype(str).to_numpy()


def world_phase(n: int, rng: np.random.Generator) -> DatasetSpec:
    control = rng.uniform(0.05, 1.0, n)
    coupling = rng.uniform(0.4, 2.2, n)
    temperature = rng.uniform(0.5, 3.5, n)
    noise = rng.uniform(0, 0.12, n)
    effective = coupling * control / temperature
    phase = np.where(effective < 0.32, "low", np.where(effective < 0.47, "critical", "high"))
    df = pd.DataFrame(
        {
            "control": control,
            "coupling": coupling,
            "temperature": temperature,
            "noise": noise,
            "phase": phase,
            "hidden_effective": effective,
        }
    )
    return DatasetSpec("PHASE", "PHASE", df, "phase", "synthetic")


def world_gas(n: int, rng: np.random.Generator) -> DatasetSpec:
    pressure = rng.uniform(0.5, 14.0, n)
    volume = rng.uniform(0.4, 9.0, n)
    moles = rng.uniform(0.1, 5.0, n)
    impurity = rng.uniform(0.0, 0.1, n)
    pv_over_n = pressure * volume / moles
    target = _band(pv_over_n * (1 - impurity), ("cold", "medium", "hot"))
    df = pd.DataFrame(
        {
            "pressure": pressure,
            "volume": volume,
            "moles": moles,
            "impurity": impurity,
            "target": target,
            "hidden_pv_over_n": pv_over_n,
        }
    )
    return DatasetSpec("GAS", "GAS", df, "target", "synthetic")


def world_maze(n: int, rng: np.random.Generator) -> DatasetSpec:
    grid_size = rng.integers(12, 80, n)
    wall_density = rng.uniform(0.1, 0.65, n)
    branch_factor = rng.uniform(1.0, 5.0, n)
    choke_count = rng.integers(0, 12, n)
    corridor_width = rng.integers(1, 6, n)
    bottleneck = choke_count * wall_density * branch_factor / corridor_width
    target = np.where(bottleneck > 3.8, "bottleneck", np.where(wall_density > 0.5, "blocked", "open"))
    df = pd.DataFrame(
        {
            "grid_size": grid_size,
            "wall_density": wall_density,
            "branch_factor": branch_factor,
            "choke_count": choke_count,
            "corridor_width": corridor_width,
            "target": target,
            "hidden_bottleneck": bottleneck,
        }
    )
    return DatasetSpec("MAZE", "MAZE", df, "target", "synthetic")


def world_ca(rng: np.random.Generator) -> DatasetSpec:
    rows = []
    for rule in range(256):
        bits = np.array([(rule >> i) & 1 for i in range(8)])
        density = bits.mean()
        entropy = -(density * np.log2(density + 1e-9) + (1 - density) * np.log2(1 - density + 1e-9))
        transitions = int(np.sum(bits != np.roll(bits, 1)))
        asymmetry = int(np.sum(bits != bits[::-1]))
        target = "chaotic" if entropy > 0.92 and transitions >= 5 else ("fixed" if transitions <= 1 else "structured")
        if rng.random() < 0.02:
            target = str(rng.choice(["chaotic", "fixed", "structured"]))
        rows.append(
            {
                "rule": rule,
                "density": density,
                "transitions": transitions,
                "asymmetry": asymmetry,
                "target": target,
                "hidden_entropy": entropy,
            }
        )
    return DatasetSpec("CA", "CA", pd.DataFrame(rows), "target", "synthetic")


def world_obstruction(n: int, rng: np.random.Generator) -> DatasetSpec:
    width = rng.uniform(1.0, 20.0, n)
    load = rng.uniform(1.0, 200.0, n)
    clearance = rng.uniform(0.5, 12.0, n)
    friction = rng.uniform(0.05, 0.8, n)
    hidden = load / (width * clearance) - friction
    target = np.where(hidden > 2.6, "blocked", "clear")
    df = pd.DataFrame(
        {
            "width": width,
            "load": load,
            "clearance": clearance,
            "friction": friction,
            "target": target,
            "hidden_obstruction": hidden,
        }
    )
    return DatasetSpec("OBSTRUCTION", "OBSTRUCTION", df, "target", "synthetic")


def world_sat(n: int, rng: np.random.Generator) -> DatasetSpec:
    clauses = rng.integers(20, 240, n)
    variables = rng.integers(5, 80, n)
    unit_ratio = rng.uniform(0.0, 0.6, n)
    horn_ratio = rng.uniform(0.0, 1.0, n)
    balance = rng.uniform(0.1, 1.8, n)
    hidden = (clauses / variables) * (1 - horn_ratio) * balance - unit_ratio
    target = np.where(hidden > 3.3, "hard", "easy")
    df = pd.DataFrame(
        {
            "clauses": clauses,
            "variables": variables,
            "unit_ratio": unit_ratio,
            "horn_ratio": horn_ratio,
            "balance": balance,
            "target": target,
            "hidden_boolean_coordinate": hidden,
        }
    )
    return DatasetSpec("SAT", "SAT", df, "target", "synthetic")


def world_modular(n: int, rng: np.random.Generator) -> DatasetSpec:
    x = rng.integers(0, 10_000, n)
    n_mod = rng.choice([5, 7, 11, 13], n)
    residue = x % n_mod
    target = np.where(residue <= 1, "low_residue", np.where(residue <= n_mod // 2, "mid_residue", "high_residue"))
    df = pd.DataFrame({"x": x, "n": n_mod, "noise": rng.normal(0, 1, n), "target": target, "hidden_residue": residue})
    return DatasetSpec("MODULAR", "MODULAR", df, "target", "synthetic")


def equation_stats(eq: str, prefix: str) -> dict[str, float]:
    left, _, right = eq.partition("=")
    text = eq.replace("◇", "*")
    variables = [ch for ch in text if ch.isalpha() and ch.islower()]
    depth = 0
    max_depth = 0
    for char in text:
        if char == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ")":
            depth = max(0, depth - 1)
    return {
        f"{prefix}_len": len(eq),
        f"{prefix}_vars": len(set(variables)),
        f"{prefix}_leaves": len(variables),
        f"{prefix}_ops": text.count("*"),
        f"{prefix}_depth": max_depth,
        f"{prefix}_repeat": sum(max(0, variables.count(v) - 1) for v in set(variables)),
        f"{prefix}_side_delta": len(left) - len(right),
    }


def world_etp(n: int, rng: np.random.Generator) -> DatasetSpec:
    atoms = ["x", "y", "z", "w"]
    terms = atoms + [
        "(x ◇ x)",
        "(x ◇ y)",
        "(y ◇ x)",
        "(x ◇ (y ◇ z))",
        "((x ◇ y) ◇ z)",
        "((x ◇ y) ◇ (z ◇ x))",
        "((x ◇ x) ◇ (y ◇ y))",
    ]
    equations = [f"{a} = {b}" for a in terms for b in terms]
    rows = []
    for _ in range(n):
        premise = str(rng.choice(equations))
        conclusion = str(rng.choice(equations))
        ps = equation_stats(premise, "premise")
        cs = equation_stats(conclusion, "conclusion")
        shared = len(set(ch for ch in premise if ch in "xyzw") & set(ch for ch in conclusion if ch in "xyzw"))
        hidden = shared + ps["premise_repeat"] + cs["conclusion_repeat"] - abs(ps["premise_depth"] - cs["conclusion_depth"])
        target = "implies" if hidden >= 3 else "independent"
        row = {**ps, **cs, "shared_vars": shared, "pair_token_sum": ps["premise_len"] + cs["conclusion_len"], "target": target}
        rows.append(row)
    return DatasetSpec("ETP_RAW", "ETP", pd.DataFrame(rows), "target", "synthetic")


def world_arc(n: int, rng: np.random.Generator) -> DatasetSpec:
    objects = rng.integers(1, 12, n)
    colors = rng.integers(2, 9, n)
    holes = rng.integers(0, 5, n)
    symmetry = rng.uniform(0, 1, n)
    displacement = rng.integers(0, 8, n)
    hidden = symmetry * colors + holes - displacement / np.maximum(objects, 1)
    target = np.where(hidden > 4.2, "mirror_color", np.where(holes >= 3, "fill_holes", "move_object"))
    df = pd.DataFrame(
        {
            "objects": objects,
            "colors": colors,
            "holes": holes,
            "symmetry": symmetry,
            "displacement": displacement,
            "target": target,
            "hidden_symbolic_transform": hidden,
        }
    )
    return DatasetSpec("ARC_STYLE", "ARC", df, "target", "synthetic")


def build_synthetic_worlds(seed: int, n: int) -> list[DatasetSpec]:
    rng = np.random.default_rng(seed)
    return [
        world_phase(n, rng),
        world_gas(n, rng),
        world_maze(n, rng),
        world_ca(rng),
        world_obstruction(n, rng),
        world_sat(n, rng),
        world_modular(n, rng),
        world_etp(n, rng),
        world_arc(n, rng),
    ]


# =============================================================================
# 05 EXPRESSION COORDINATE ENGINE
# =============================================================================


@dataclass
class Coordinate:
    name: str
    depth: int
    parents: list[str]
    complexity: float
    func: Callable[[pd.DataFrame], pd.Series]
    expression: str
    score: float = 0.0

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        values = self.func(df)
        values = pd.Series(values, index=df.index).astype(float)
        return values.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def export(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "depth": self.depth,
            "parents": self.parents,
            "complexity": self.complexity,
            "expression": self.expression,
            "score": self.score,
        }


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(text)).strip("_")


def numeric_features(df: pd.DataFrame, target: str, max_features: int = 8) -> list[str]:
    cols = [
        col
        for col in df.columns
        if col != target
        and not str(col).startswith("hidden_")
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique(dropna=True) > 1
    ]
    cols.sort(key=lambda c: (df[c].nunique(dropna=True), float(df[c].std() or 0)), reverse=True)
    return cols[:max_features]


def build_expression_coordinates(df: pd.DataFrame, target: str, max_base: int = 8, max_coords: int = 260) -> list[Coordinate]:
    bases = numeric_features(df, target, max_base)
    coords: list[Coordinate] = []

    def add(coord: Coordinate) -> None:
        if len(coords) >= max_coords:
            return
        try:
            values = coord.evaluate(df)
        except Exception:
            return
        if values.nunique(dropna=False) <= 1:
            return
        coords.append(coord)

    for x in bases:
        sx = safe_name(x)
        add(Coordinate(sx, 0, [sx], 1.0, lambda d, x=x: d[x], x))
        add(Coordinate(f"abs_{sx}", 1, [sx], 1.2, lambda d, x=x: d[x].abs(), f"abs({x})"))
        add(Coordinate(f"sqrt_{sx}", 1, [sx], 1.3, lambda d, x=x: np.sqrt(pd.Series(d[x]).clip(lower=0)), f"sqrt({x})"))
        add(Coordinate(f"log_{sx}", 1, [sx], 1.3, lambda d, x=x: np.log1p(pd.Series(d[x]).clip(lower=0)), f"log({x})"))
        add(Coordinate(f"square_{sx}", 1, [sx], 1.3, lambda d, x=x: d[x] ** 2, f"{x}^2"))

    for i, x in enumerate(bases):
        for y in bases[i + 1 :]:
            sx, sy = safe_name(x), safe_name(y)
            add(Coordinate(f"{sx}_plus_{sy}", 2, [sx, sy], 2.0, lambda d, x=x, y=y: d[x] + d[y], f"{x}+{y}"))
            add(Coordinate(f"{sx}_minus_{sy}", 2, [sx, sy], 2.0, lambda d, x=x, y=y: d[x] - d[y], f"{x}-{y}"))
            add(Coordinate(f"{sx}_times_{sy}", 2, [sx, sy], 2.1, lambda d, x=x, y=y: d[x] * d[y], f"{x}*{y}"))
            add(
                Coordinate(
                    f"min_{sx}_{sy}",
                    2,
                    [sx, sy],
                    2.1,
                    lambda d, x=x, y=y: np.minimum(d[x], d[y]),
                    f"min({x},{y})",
                )
            )
            add(
                Coordinate(
                    f"max_{sx}_{sy}",
                    2,
                    [sx, sy],
                    2.1,
                    lambda d, x=x, y=y: np.maximum(d[x], d[y]),
                    f"max({x},{y})",
                )
            )
            add(
                Coordinate(
                    f"{sx}_over_{sy}",
                    2,
                    [sx, sy],
                    2.2,
                    lambda d, x=x, y=y: d[x] / pd.Series(d[y]).replace(0, np.nan),
                    f"{x}/{y}",
                )
            )
            add(
                Coordinate(
                    f"{sy}_over_{sx}",
                    2,
                    [sx, sy],
                    2.2,
                    lambda d, x=x, y=y: d[y] / pd.Series(d[x]).replace(0, np.nan),
                    f"{y}/{x}",
                )
            )

    for x in bases:
        for y in bases:
            for z in bases:
                if len({x, y, z}) != 3 or len(coords) >= max_coords:
                    continue
                sx, sy, sz = safe_name(x), safe_name(y), safe_name(z)
                add(
                    Coordinate(
                        f"{sx}_times_{sy}_over_{sz}",
                        3,
                        [sx, sy, sz],
                        3.4,
                        lambda d, x=x, y=y, z=z: (d[x] * d[y]) / pd.Series(d[z]).replace(0, np.nan),
                        f"({x}*{y})/{z}",
                    )
                )
                add(
                    Coordinate(
                        f"{sx}_plus_{sy}_over_{sz}",
                        3,
                        [sx, sy, sz],
                        3.3,
                        lambda d, x=x, y=y, z=z: (d[x] + d[y]) / pd.Series(d[z]).replace(0, np.nan),
                        f"({x}+{y})/{z}",
                    )
                )
                add(
                    Coordinate(
                        f"{sx}_times_{sy}_minus_{sz}",
                        3,
                        [sx, sy, sz],
                        3.2,
                        lambda d, x=x, y=y, z=z: d[x] * (d[y] - d[z]),
                        f"{x}*({y}-{z})",
                    )
                )
                add(
                    Coordinate(
                        f"{sx}_over_{sy}_plus_{sz}",
                        3,
                        [sx, sy, sz],
                        3.4,
                        lambda d, x=x, y=y, z=z: d[x] / (d[y] + d[z]).replace(0, np.nan),
                        f"{x}/({y}+{z})",
                    )
                )

    depth3 = [coord for coord in coords if coord.depth == 3][: max(12, max_coords // 10)]
    for base in depth3:
        if len(coords) >= max_coords:
            break
        add(
            Coordinate(
                f"log_{base.name}",
                4,
                [base.name],
                base.complexity + 1.2,
                lambda d, base=base: np.log1p(base.evaluate(d).clip(lower=0)),
                f"log({base.expression})",
            )
        )
        add(
            Coordinate(
                f"sqrt_{base.name}",
                4,
                [base.name],
                base.complexity + 1.2,
                lambda d, base=base: np.sqrt(base.evaluate(d).clip(lower=0)),
                f"sqrt({base.expression})",
            )
        )
        add(
            Coordinate(
                f"square_{base.name}",
                4,
                [base.name],
                base.complexity + 1.2,
                lambda d, base=base: base.evaluate(d) ** 2,
                f"({base.expression})^2",
            )
        )

    return coords[:max_coords]


# =============================================================================
# 06 PORTAL SEARCH
# =============================================================================


def parse_operators(expression: str) -> list[str]:
    ops: list[str] = []
    if "abs(" in expression:
        ops.append("absolute")
    if "sqrt(" in expression:
        ops.append("sqrt")
    if "log(" in expression:
        ops.append("log")
    if "min(" in expression:
        ops.append("min")
    if "max(" in expression:
        ops.append("max")
    if "^2" in expression:
        ops.append("square")
    if "*" in expression:
        ops.append("multiply")
    if "/" in expression:
        ops.append("divide")
    if "+" in expression:
        ops.append("add")
    if "-" in expression:
        ops.append("subtract")
    if not ops:
        ops.append("identity")
    return ops


def operator_family(expression: str, operators: list[str]) -> str:
    op_set = set(operators)
    if {"multiply", "divide"}.issubset(op_set):
        return "EFFECTIVE_PARAMETER"
    if "divide" in op_set and ("add" in op_set or "subtract" in op_set):
        return "CONSTRAINT"
    if "divide" in op_set:
        return "RATIO"
    if "multiply" in op_set:
        return "PRODUCT"
    if "subtract" in op_set:
        return "DIFFERENCE"
    if "add" in op_set and "absolute" in op_set:
        return "ENERGY"
    if "add" in op_set:
        return "SUM"
    if "min" in op_set or "max" in op_set:
        return "SYMMETRY"
    if "sqrt" in op_set or "log" in op_set or "square" in op_set:
        return "TRANSFORM"
    return "IDENTITY"


def residual_failure_mode(score: float, raw_single: float, portal_score: float) -> str:
    if score <= 0:
        return "no_signal"
    if portal_score < 0.8:
        return "negative_transfer"
    if portal_score < 1.0:
        return "underperforms_raw_coordinate"
    if portal_score < 1.1:
        return "weak_advantage"
    return "survived"


def evolved_operator_candidates(family: str) -> list[str]:
    evolution = {
        "RATIO": ["EFFECTIVE_PARAMETER", "CONSTRAINT"],
        "DIFFERENCE": ["CONSTRAINT", "SUM"],
        "PRODUCT": ["EFFECTIVE_PARAMETER", "RATIO"],
        "SUM": ["CONSTRAINT", "ENERGY"],
        "TRANSFORM": ["RATIO", "EFFECTIVE_PARAMETER"],
        "SYMMETRY": ["DIFFERENCE", "CONSTRAINT"],
        "IDENTITY": ["RATIO", "PRODUCT", "DIFFERENCE"],
    }
    return evolution.get(family, ["RATIO", "EFFECTIVE_PARAMETER"])


SEMANTIC_TEMPLATES = [
    "RESOURCE_CONSTRAINT",
    "EFFECTIVE_PARAMETER",
    "AVAILABLE_CAPACITY",
    "SIGNAL_NOISE",
    "FREEDOM_CONSTRAINT",
    "ENERGY_DISSIPATION",
    "CONTINUATION_COST",
    "SEARCH_BOTTLENECK",
]

EMERGENT_CONCEPTS = {
    "RESOURCE_CONSTRAINT": "AVAILABLE_CONTINUATION",
    "EFFECTIVE_PARAMETER": "EFFECTIVE_FREEDOM",
    "AVAILABLE_CAPACITY": "CAPACITY",
    "SIGNAL_NOISE": "VIABILITY",
    "FREEDOM_CONSTRAINT": "CONSTRAINT_RELIEF",
    "ENERGY_DISSIPATION": "ENERGY_BALANCE",
    "CONTINUATION_COST": "SEARCH_PRESSURE",
    "SEARCH_BOTTLENECK": "SEARCH_PRESSURE",
}

ROLE_KEYWORDS = {
    "resource": {
        "pressure",
        "volume",
        "control",
        "coupling",
        "clearance",
        "width",
        "energy",
        "capacity",
        "objects",
        "colors",
        "clauses",
        "variables",
        "choke_count",
        "grid_size",
    },
    "constraint": {
        "moles",
        "temperature",
        "load",
        "friction",
        "wall_density",
        "corridor_width",
        "noise",
        "impurity",
        "unit_ratio",
        "horn_ratio",
        "balance",
    },
    "capacity": {"clearance", "width", "volume", "control", "coupling", "grid_size", "corridor_width"},
    "load": {"load", "pressure", "clauses", "wall_density", "choke_count"},
    "signal": {"control", "coupling", "symmetry", "density", "transitions", "shared_vars"},
    "variance": {"noise", "impurity", "asymmetry", "balance"},
    "freedom": {"variables", "degrees", "corridor_width", "moles", "clearance", "width"},
    "restriction": {"clauses", "temperature", "load", "wall_density", "friction", "unit_ratio"},
    "energy": {"energy", "pressure", "volume", "load", "coupling"},
    "loss": {"entropy", "friction", "noise", "impurity", "temperature", "asymmetry"},
    "continuation": {"control", "capacity", "clearance", "width", "volume", "objects", "shared_vars"},
    "obstacle": {"bottleneck", "wall_density", "load", "friction", "holes", "choke_count"},
    "search": {"search", "clauses", "grid_size", "objects", "pair_token_sum"},
    "bottleneck": {"bottleneck", "choke_count", "wall_density", "load", "corridor_width"},
}


def expression_variables(expression: str) -> list[str]:
    cleaned = expression
    for token in ["sqrt", "log", "abs", "min", "max"]:
        cleaned = cleaned.replace(token, " ")
    for char in "()+-*/^,":
        cleaned = cleaned.replace(char, " ")
    return [part for part in cleaned.split() if part and not part.isnumeric()]


def parse_semantics(expression: str, operators: list[str]) -> dict[str, Any]:
    variables = expression_variables(expression)
    numerator: list[str] = []
    denominator: list[str] = []
    if "/" in expression:
        left, right = expression.split("/", 1)
        numerator = expression_variables(left)
        denominator = expression_variables(right)
    elif "*" in expression or "+" in expression:
        numerator = variables
    else:
        numerator = variables[:1]

    roles = semantic_roles(variables)
    numerator_roles = semantic_roles(numerator)
    denominator_roles = semantic_roles(denominator)
    template = assign_semantic_template(operators, numerator_roles, denominator_roles, roles)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "variables": variables,
        "roles": sorted(roles),
        "numerator_roles": sorted(numerator_roles),
        "denominator_roles": sorted(denominator_roles),
        "pattern": template,
        "emergent_concept": EMERGENT_CONCEPTS.get(template, "VIABILITY"),
    }


def semantic_roles(variables: list[str]) -> set[str]:
    roles: set[str] = set()
    for var in variables:
        lowered = var.lower()
        for role, keywords in ROLE_KEYWORDS.items():
            if lowered in keywords or any(keyword in lowered for keyword in keywords):
                roles.add(role)
    return roles


def assign_semantic_template(
    operators: list[str],
    numerator_roles: set[str],
    denominator_roles: set[str],
    roles: set[str],
) -> str:
    has_ratio = "divide" in operators
    has_product = "multiply" in operators
    if has_ratio and ({"resource", "constraint"} & numerator_roles or "resource" in roles) and (
        denominator_roles & {"constraint", "restriction", "loss"}
    ):
        return "RESOURCE_CONSTRAINT"
    if has_ratio and has_product:
        return "EFFECTIVE_PARAMETER"
    if has_ratio and "capacity" in numerator_roles and (denominator_roles & {"load", "constraint"}):
        return "AVAILABLE_CAPACITY"
    if has_ratio and "signal" in numerator_roles and (denominator_roles & {"variance", "loss"}):
        return "SIGNAL_NOISE"
    if has_ratio and "freedom" in numerator_roles and (denominator_roles & {"restriction", "constraint"}):
        return "FREEDOM_CONSTRAINT"
    if has_ratio and "energy" in numerator_roles and (denominator_roles & {"loss", "constraint"}):
        return "ENERGY_DISSIPATION"
    if has_ratio and "continuation" in numerator_roles and (denominator_roles & {"obstacle", "constraint"}):
        return "CONTINUATION_COST"
    if has_ratio and "search" in numerator_roles and (denominator_roles & {"bottleneck", "obstacle"}):
        return "SEARCH_BOTTLENECK"
    if has_ratio:
        return "RESOURCE_CONSTRAINT"
    if has_product:
        return "EFFECTIVE_PARAMETER"
    return "AVAILABLE_CAPACITY" if "capacity" in roles else "CONTINUATION_COST"


def semantic_evolution_candidates(pattern: str) -> list[str]:
    candidates = {
        "RESOURCE_CONSTRAINT": ["(resource*capacity)/constraint", "(resource-loss)/constraint", "(resource*time)/cost"],
        "EFFECTIVE_PARAMETER": ["(resource*capacity)/constraint", "(signal*resource)/noise"],
        "AVAILABLE_CAPACITY": ["capacity/load", "(capacity-loss)/load"],
        "SIGNAL_NOISE": ["signal/variance", "(signal*resource)/noise"],
        "FREEDOM_CONSTRAINT": ["degrees_of_freedom/restriction", "(freedom*capacity)/constraint"],
        "ENERGY_DISSIPATION": ["energy/loss", "(energy-loss)/constraint"],
        "CONTINUATION_COST": ["continuation/obstacle", "(continuation*capacity)/cost"],
        "SEARCH_BOTTLENECK": ["search/bottleneck", "(search-bottleneck)/cost"],
    }
    return candidates.get(pattern, ["resource/constraint", "(resource*capacity)/constraint"])


def semantic_embedding(row: pd.Series) -> np.ndarray:
    roles = set(row.get("semantic_roles", []))
    operators = set(row.get("operators", []))
    family = row.get("semantic_pattern", "")
    values = []
    for template in SEMANTIC_TEMPLATES:
        values.append(1.0 if family == template else 0.0)
    for role in sorted(ROLE_KEYWORDS):
        values.append(1.0 if role in roles else 0.0)
    for op in ["add", "subtract", "multiply", "divide", "absolute", "square", "sqrt", "log", "min", "max"]:
        values.append(1.0 if op in operators else 0.0)
    values.extend(
        [
            float(row.get("depth", 0)) / 4.0,
            float(row.get("portal_score", 0)),
            float(row.get("stability", 0)),
        ]
    )
    return np.array(values, dtype=float)


def problem_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 15:
        return "regression"
    return "classification"


def model_for(problem: str, kind: str) -> Any:
    if problem == "regression":
        if kind == "tree":
            return DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=143)
        return RandomForestRegressor(n_estimators=160, min_samples_leaf=3, random_state=143)
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=143)
    return RandomForestClassifier(n_estimators=160, min_samples_leaf=3, random_state=143, class_weight="balanced")


def fit_score(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, problem: str, kind: str) -> float:
    categorical = [c for c in x_train.columns if not pd.api.types.is_numeric_dtype(x_train[c])]
    numeric = [c for c in x_train.columns if c not in categorical]
    pipe = Pipeline(
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
            ("model", model_for(problem, kind)),
        ]
    )
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    if problem == "regression":
        return float(r2_score(y_test, pred))
    return float(accuracy_score(y_test.astype(str), pd.Series(pred).astype(str)))


@dataclass
class PortalResult:
    domain: str
    coordinate: str
    expression: str
    operators: list[str]
    operator_family: str
    semantic_pattern: str
    emergent_concept: str
    numerator: list[str]
    denominator: list[str]
    semantic_roles: list[str]
    depth: int
    complexity: float
    raw_score: float
    raw_best_single_score: float
    coordinate_score: float
    portal_score: float
    advantage: float
    compressive_score: float
    is_portal: bool
    is_strong_portal: bool
    stability: float
    residual: float
    failure_mode: str
    evolved_candidates: list[str]
    semantic_evolution: list[str]
    parents: list[str] = field(default_factory=list)


def score_coordinate(
    spec: DatasetSpec,
    coord: Coordinate,
    train_idx: pd.Index,
    test_idx: pd.Index,
    y_train: pd.Series,
    y_test: pd.Series,
    problem: str,
) -> float:
    values = coord.evaluate(spec.df)
    x_train = pd.DataFrame({coord.name: values.loc[train_idx]})
    x_test = pd.DataFrame({coord.name: values.loc[test_idx]})
    return fit_score(x_train, x_test, y_train, y_test, problem, "tree")


def portal_search(spec: DatasetSpec, max_coords: int) -> tuple[list[PortalResult], dict[str, Any], list[Coordinate]]:
    df = spec.df.dropna(axis=0).reset_index(drop=True)
    spec = DatasetSpec(spec.name, spec.domain, df, spec.target, spec.source)
    y = df[spec.target]
    ptype = problem_type(y)
    raw_cols = numeric_features(df, spec.target, max_features=16)
    if len(raw_cols) < 1 or y.nunique(dropna=True) < 2:
        return [], {"domain": spec.domain, "status": "skipped", "reason": "insufficient_variation"}, []
    stratify = y if ptype == "classification" and y.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(df.index, test_size=0.3, random_state=143, stratify=stratify)
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    raw_score = fit_score(df.loc[train_idx, raw_cols], df.loc[test_idx, raw_cols], y_train, y_test, ptype, "forest")
    raw_single_scores = []
    for col in raw_cols:
        score = fit_score(df.loc[train_idx, [col]], df.loc[test_idx, [col]], y_train, y_test, ptype, "tree")
        raw_single_scores.append(score)
    raw_best_single = max(raw_single_scores) if raw_single_scores else max(raw_score, 1e-9)

    coords = build_expression_coordinates(df, spec.target, max_base=8, max_coords=max_coords)
    results: list[PortalResult] = []
    for coord in coords:
        try:
            score = score_coordinate(spec, coord, train_idx, test_idx, y_train, y_test, ptype)
        except Exception:
            continue
        coord.score = score
        denominator = max(raw_best_single, 1e-9)
        portal_score = score / denominator if denominator > 0 else 0.0
        advantage = score - raw_best_single
        compressive_score = score / max(raw_score, 1e-9) / max(coord.complexity, 1.0)
        stability = coordinate_stability(spec, coord, ptype)
        operators = parse_operators(coord.expression)
        family = operator_family(coord.expression, operators)
        semantics = parse_semantics(coord.expression, operators)
        residual = max(0.0, raw_best_single - score)
        failure_mode = residual_failure_mode(score, raw_best_single, portal_score)
        results.append(
            PortalResult(
                domain=spec.domain,
                coordinate=coord.name,
                expression=coord.expression,
                operators=operators,
                operator_family=family,
                semantic_pattern=semantics["pattern"],
                emergent_concept=semantics["emergent_concept"],
                numerator=semantics["numerator"],
                denominator=semantics["denominator"],
                semantic_roles=semantics["roles"],
                depth=coord.depth,
                complexity=coord.complexity,
                raw_score=raw_score,
                raw_best_single_score=raw_best_single,
                coordinate_score=score,
                portal_score=portal_score,
                advantage=advantage,
                compressive_score=compressive_score,
                is_portal=portal_score > 1.1,
                is_strong_portal=portal_score > 1.25,
                stability=stability,
                residual=residual,
                failure_mode=failure_mode,
                evolved_candidates=evolved_operator_candidates(family) if failure_mode != "survived" else [],
                semantic_evolution=semantic_evolution_candidates(semantics["pattern"]) if failure_mode != "survived" else [],
                parents=coord.parents,
            )
        )
    results.sort(key=lambda r: (r.is_strong_portal, r.portal_score, r.coordinate_score, -r.complexity), reverse=True)
    representation = {
        "domain": spec.domain,
        "dataset": spec.name,
        "source": spec.source,
        "target": spec.target,
        "problem_type": ptype,
        "rows": len(df),
        "raw_feature_count": len(raw_cols),
        "raw_score": raw_score,
        "raw_best_single_score": raw_best_single,
        "best_coordinate_score": results[0].coordinate_score if results else np.nan,
        "best_portal_score": results[0].portal_score if results else np.nan,
        "portal_count": sum(r.is_portal for r in results),
        "strong_portal_count": sum(r.is_strong_portal for r in results),
        "coordinate_count": len(results),
        "status": "evaluated",
    }
    return results, representation, coords


def coordinate_stability(spec: DatasetSpec, coord: Coordinate, problem: str, repeats: int = 3) -> float:
    y = spec.df[spec.target]
    scores = []
    for seed in range(repeats):
        stratify = y if problem == "classification" and y.value_counts().min() >= 2 else None
        try:
            train_idx, test_idx = train_test_split(spec.df.index, test_size=0.3, random_state=900 + seed, stratify=stratify)
            score = score_coordinate(spec, coord, train_idx, test_idx, y.loc[train_idx], y.loc[test_idx], problem)
            scores.append(score)
        except Exception:
            pass
    if len(scores) < 2:
        return 0.0
    return float(max(0.0, 1.0 - np.std(scores)))


# =============================================================================
# 07 PORTAL BASINS
# =============================================================================


def build_portal_basins(portals: list[PortalResult]) -> tuple[nx.DiGraph, pd.DataFrame]:
    graph = nx.DiGraph()
    lookup = {portal.coordinate: portal for portal in portals}
    for portal in portals:
        graph.add_node(
            portal.coordinate,
            domain=portal.domain,
            depth=portal.depth,
            operator_family=portal.operator_family,
            portal_score=portal.portal_score,
            is_portal=portal.is_portal,
        )
        for parent in portal.parents:
            if parent == portal.coordinate:
                continue
            graph.add_node(parent, domain=portal.domain, depth=max(0, portal.depth - 1), portal_score=0.0, is_portal=False)
            graph.add_edge(parent, portal.coordinate)

    rows = []
    components = list(nx.weakly_connected_components(graph))
    for i, nodes in enumerate(components):
        sub = graph.subgraph(nodes)
        portal_nodes = [node for node in nodes if lookup.get(node, None) and lookup[node].is_portal]
        families = sorted({graph.nodes[node].get("operator_family") for node in nodes if graph.nodes[node].get("operator_family")})
        rows.append(
            {
                "basin_id": i,
                "basin_label": basin_label(families),
                "basin_size": len(nodes),
                "portal_count": len(portal_nodes),
                "portal_density": len(portal_nodes) / max(len(nodes), 1),
                "max_portal_depth": max((graph.nodes[node].get("depth", 0) for node in nodes), default=0),
                "operator_families": "|".join(families),
                "edge_count": sub.number_of_edges(),
            }
        )
    return graph, pd.DataFrame(rows)


def basin_label(families: list[str]) -> str:
    family_set = set(families)
    if "EFFECTIVE_PARAMETER" in family_set and "RATIO" in family_set:
        return "effective_freedom"
    if "RATIO" in family_set or "CONSTRAINT" in family_set:
        return "constraint_ratio"
    if "DIFFERENCE" in family_set or "SYMMETRY" in family_set:
        return "structural_contrast"
    if "PRODUCT" in family_set:
        return "interaction_product"
    return families[0].lower() if families else "unclassified"


def plot_basin_graph(graph: nx.DiGraph, path: Path) -> None:
    if graph.number_of_nodes() == 0:
        return
    sample_nodes = list(graph.nodes())[:120]
    sub = graph.subgraph(sample_nodes)
    colors = ["tab:red" if sub.nodes[n].get("is_portal") else "tab:blue" for n in sub.nodes()]
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(sub, seed=143)
    nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=60, alpha=0.8)
    nx.draw_networkx_edges(sub, pos, alpha=0.25, arrows=False)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# =============================================================================
# 08 PORTAL TRANSFER
# =============================================================================


def build_operator_grammar(portal_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    if portal_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    exploded = portal_df.explode("operators")
    for operator, group in exploded.groupby("operators"):
        survived = group[group["is_portal"]]
        domains = sorted(group["domain"].unique())
        survived_domains = sorted(survived["domain"].unique())
        rows.append(
            {
                "operator": operator,
                "frequency": int(len(group)),
                "survivor_count": int(len(survived)),
                "survival_rate": float(len(survived) / max(len(group), 1)),
                "average_advantage": float(group["advantage"].mean()),
                "average_portal_score": float(group["portal_score"].mean()),
                "domains_appearing": len(domains),
                "domains_surviving": len(survived_domains),
                "domain_list": "|".join(domains),
                "surviving_domain_list": "|".join(survived_domains),
            }
        )
    grammar = pd.DataFrame(rows)
    family_rows = []
    for family, group in portal_df.groupby("operator_family"):
        survived = group[group["is_portal"]]
        domains = sorted(group["domain"].unique())
        survived_domains = sorted(survived["domain"].unique())
        family_rows.append(
            {
                "operator_family": family,
                "frequency": int(len(group)),
                "survivor_count": int(len(survived)),
                "survival_rate": float(len(survived) / max(len(group), 1)),
                "average_advantage": float(group["advantage"].mean()),
                "average_portal_score": float(group["portal_score"].mean()),
                "domains_appearing": len(domains),
                "domains_surviving": len(survived_domains),
                "domain_list": "|".join(domains),
                "surviving_domain_list": "|".join(survived_domains),
            }
        )
    families = pd.DataFrame(family_rows)
    return grammar, families


def build_family_transfer_matrix(portal_df: pd.DataFrame) -> pd.DataFrame:
    if portal_df.empty:
        return pd.DataFrame()
    rows = []
    domain_families = {
        domain: group.groupby("operator_family")
        for domain, group in portal_df.groupby("domain")
    }
    domains = sorted(portal_df["domain"].unique())
    for source in domains:
        source_group = portal_df[portal_df["domain"] == source]
        source_surviving = set(source_group[source_group["is_portal"]]["operator_family"].unique())
        for target in domains:
            target_group = portal_df[portal_df["domain"] == target]
            for family in sorted(source_group["operator_family"].unique()):
                target_family = target_group[target_group["operator_family"] == family]
                if target_family.empty:
                    rows.append(
                        {
                            "source_domain": source,
                            "target_domain": target,
                            "operator_family": family,
                            "performance_retention": 0.0,
                            "coordinate_usefulness": 0.0,
                            "survival": False,
                            "status": "absent",
                        }
                    )
                    continue
                source_score = float(source_group[source_group["operator_family"] == family]["portal_score"].mean())
                target_score = float(target_family["portal_score"].mean())
                target_survives = bool(target_family["is_portal"].any())
                rows.append(
                    {
                        "source_domain": source,
                        "target_domain": target,
                        "operator_family": family,
                        "performance_retention": float(target_score / max(source_score, 1e-9)),
                        "coordinate_usefulness": float(target_family["coordinate_score"].max()),
                        "survival": target_survives,
                        "status": "evaluated",
                    }
                )
    return pd.DataFrame(rows)


def add_transfer_rates(families: pd.DataFrame, transfer_df: pd.DataFrame) -> pd.DataFrame:
    if families.empty:
        return families
    out = families.copy()
    rates = []
    for family in out["operator_family"]:
        group = transfer_df[(transfer_df["operator_family"] == family) & (transfer_df["source_domain"] != transfer_df["target_domain"])]
        if group.empty:
            rates.append(0.0)
        else:
            rates.append(float(group["survival"].mean()))
    out["transfer_rate"] = rates
    out["portal_score"] = (
        out["frequency"].clip(lower=1)
        * out["transfer_rate"]
        * out["average_portal_score"].clip(lower=0)
        * out["domains_surviving"].clip(lower=1)
    )
    return out.sort_values("portal_score", ascending=False)


def build_semantic_families(portal_df: pd.DataFrame) -> pd.DataFrame:
    if portal_df.empty:
        return pd.DataFrame()
    rows = []
    for pattern, group in portal_df.groupby("semantic_pattern"):
        survived = group[group["is_portal"]]
        domains = sorted(group["domain"].unique())
        survived_domains = sorted(survived["domain"].unique())
        rows.append(
            {
                "semantic_pattern": pattern,
                "emergent_concept": EMERGENT_CONCEPTS.get(pattern, "VIABILITY"),
                "frequency": int(len(group)),
                "survivor_count": int(len(survived)),
                "semantic_survival": float(len(survived) / max(len(group), 1)),
                "semantic_advantage": float(group["advantage"].mean()),
                "average_portal_score": float(group["portal_score"].mean()),
                "semantic_stability": float(survived["stability"].mean()) if not survived.empty else 0.0,
                "domains_appearing": len(domains),
                "domains_surviving": len(survived_domains),
                "domain_list": "|".join(domains),
                "surviving_domain_list": "|".join(survived_domains),
            }
        )
    return pd.DataFrame(rows)


def build_semantic_transfer(portal_df: pd.DataFrame) -> pd.DataFrame:
    if portal_df.empty:
        return pd.DataFrame()
    rows = []
    domains = sorted(portal_df["domain"].unique())
    for source in domains:
        source_group = portal_df[portal_df["domain"] == source]
        for target in domains:
            target_group = portal_df[portal_df["domain"] == target]
            for pattern in sorted(source_group["semantic_pattern"].unique()):
                target_pattern = target_group[target_group["semantic_pattern"] == pattern]
                if target_pattern.empty:
                    rows.append(
                        {
                            "source_domain": source,
                            "target_domain": target,
                            "semantic_pattern": pattern,
                            "semantic_transfer": 0.0,
                            "semantic_advantage": 0.0,
                            "survival": False,
                            "status": "absent",
                        }
                    )
                    continue
                source_score = float(source_group[source_group["semantic_pattern"] == pattern]["portal_score"].mean())
                target_score = float(target_pattern["portal_score"].mean())
                target_survives = bool(target_pattern["is_portal"].any())
                rows.append(
                    {
                        "source_domain": source,
                        "target_domain": target,
                        "semantic_pattern": pattern,
                        "semantic_transfer": float(target_score / max(source_score, 1e-9)),
                        "semantic_advantage": float(target_pattern["advantage"].mean()),
                        "survival": target_survives,
                        "status": "evaluated",
                    }
                )
    return pd.DataFrame(rows)


def add_semantic_transfer_rates(semantic_df: pd.DataFrame, transfer_df: pd.DataFrame) -> pd.DataFrame:
    if semantic_df.empty:
        return semantic_df
    out = semantic_df.copy()
    transfer_rates = []
    transfer_means = []
    for pattern in out["semantic_pattern"]:
        group = transfer_df[(transfer_df["semantic_pattern"] == pattern) & (transfer_df["source_domain"] != transfer_df["target_domain"])]
        if group.empty:
            transfer_rates.append(0.0)
            transfer_means.append(0.0)
        else:
            transfer_rates.append(float(group["survival"].mean()))
            transfer_means.append(float(group["semantic_transfer"].mean()))
    out["transfer_rate"] = transfer_rates
    out["mean_semantic_transfer"] = transfer_means
    out["semantic_recurrence"] = out["domains_surviving"]
    out["semantic_rarity"] = 1.0 - (out["survivor_count"] / out["frequency"].clip(lower=1))
    out["semantic_score"] = (
        out["semantic_recurrence"].clip(lower=1)
        * out["transfer_rate"]
        * out["average_portal_score"].clip(lower=0)
        * out["semantic_stability"].clip(lower=0)
    )
    return out.sort_values("semantic_score", ascending=False)


def build_semantic_basins(portal_df: pd.DataFrame) -> pd.DataFrame:
    if portal_df.empty:
        return pd.DataFrame()
    embeddings = np.vstack([semantic_embedding(row) for _, row in portal_df.iterrows()])
    n_clusters = max(2, min(8, len(portal_df) // 25))
    if len(portal_df) < n_clusters:
        n_clusters = len(portal_df)
    labels = KMeans(n_clusters=n_clusters, random_state=145, n_init=10).fit_predict(embeddings) if n_clusters > 1 else np.zeros(len(portal_df), dtype=int)
    reduced = PCA(n_components=2, random_state=145).fit_transform(embeddings) if embeddings.shape[1] >= 2 and len(portal_df) >= 2 else np.zeros((len(portal_df), 2))
    rows = []
    enriched = portal_df.copy()
    enriched["semantic_cluster"] = labels
    enriched["semantic_x"] = reduced[:, 0]
    enriched["semantic_y"] = reduced[:, 1]
    for cluster, group in enriched.groupby("semantic_cluster"):
        patterns = sorted(group["semantic_pattern"].unique())
        concepts = sorted(group["emergent_concept"].unique())
        survived = group[group["is_portal"]]
        rows.append(
            {
                "semantic_basin": int(cluster),
                "basin_label": infer_basin_label(patterns, concepts),
                "size": int(len(group)),
                "portal_count": int(len(survived)),
                "portal_density": float(len(survived) / max(len(group), 1)),
                "dominant_semantic_pattern": group["semantic_pattern"].mode().iloc[0],
                "emergent_concepts": "|".join(concepts),
                "domains": "|".join(sorted(group["domain"].unique())),
                "mean_portal_score": float(group["portal_score"].mean()),
                "semantic_x": float(group["semantic_x"].mean()),
                "semantic_y": float(group["semantic_y"].mean()),
            }
        )
    return pd.DataFrame(rows)


def infer_basin_label(patterns: list[str], concepts: list[str]) -> str:
    if "AVAILABLE_CONTINUATION" in concepts or "RESOURCE_CONSTRAINT" in patterns:
        return "available_continuation"
    if "EFFECTIVE_FREEDOM" in concepts or "EFFECTIVE_PARAMETER" in patterns:
        return "effective_freedom"
    if "SEARCH_PRESSURE" in concepts:
        return "search_pressure"
    if "ENERGY_BALANCE" in concepts:
        return "energy_balance"
    if "CONSTRAINT_RELIEF" in concepts:
        return "constraint_relief"
    if concepts:
        return concepts[0].lower()
    return "semantic_basin"


# =============================================================================
# 09 LAWBOOK
# =============================================================================


def build_lawbook(family_rankings: pd.DataFrame, transfer_df: pd.DataFrame) -> dict[str, Any]:
    laws = []
    if family_rankings.empty:
        return {"lawbook_version": "v144", "hypothesis": "reusable_portal_operator_grammars", "laws": []}
    for _, family in family_rankings.iterrows():
        if family["domains_surviving"] < 2 or family["average_portal_score"] <= 1.05 or family["transfer_rate"] <= 0:
            continue
        transfer_group = transfer_df[
            (transfer_df["operator_family"] == family["operator_family"])
            & (transfer_df["source_domain"] != transfer_df["target_domain"])
        ]
        laws.append(
            {
                "law": f"{family['operator_family'].lower()} operator family repeatedly exposes portal coordinates",
                "operator_family": family["operator_family"],
                "domains": int(family["domains_surviving"]),
                "score": float(family["portal_score"]),
                "average_portal_score": float(family["average_portal_score"]),
                "average_advantage": float(family["average_advantage"]),
                "transfer_rate": float(family["transfer_rate"]),
                "survival_rate": float(family["survival_rate"]),
                "domains_list": family["surviving_domain_list"],
                "negative_transfers": int((transfer_group["survival"] == False).sum()) if not transfer_group.empty else 0,
                "confidence": "high" if family["transfer_rate"] >= 0.5 and family["domains_surviving"] >= 3 else "medium",
            }
        )
    return {"lawbook_version": "v144", "hypothesis": "reusable_portal_operator_grammars", "laws": laws}


def build_semantic_lawbook(semantic_rankings: pd.DataFrame, semantic_transfer: pd.DataFrame) -> dict[str, Any]:
    laws = []
    if semantic_rankings.empty:
        return {"lawbook_version": "v145", "hypothesis": "semantic_continuation_structures", "laws": []}
    for _, row in semantic_rankings.iterrows():
        if (
            row["domains_surviving"] < 2
            or row["transfer_rate"] <= 0
            or row["semantic_stability"] < 0.75
            or row["average_portal_score"] <= 1.05
        ):
            continue
        transfer_group = semantic_transfer[
            (semantic_transfer["semantic_pattern"] == row["semantic_pattern"])
            & (semantic_transfer["source_domain"] != semantic_transfer["target_domain"])
        ]
        laws.append(
            {
                "law": row["semantic_pattern"],
                "emergent_concept": row["emergent_concept"],
                "domains": int(row["domains_surviving"]),
                "transfer": float(row["transfer_rate"]),
                "mean_semantic_transfer": float(row["mean_semantic_transfer"]),
                "advantage": float(row["semantic_advantage"]),
                "stability": float(row["semantic_stability"]),
                "score": float(row["semantic_score"]),
                "negative_transfers": int((transfer_group["survival"] == False).sum()) if not transfer_group.empty else 0,
                "statement": f"{row['semantic_pattern'].lower()} recurs as {row['emergent_concept'].lower()} across measured domains",
            }
        )
    return {"lawbook_version": "v145", "hypothesis": "semantic_continuation_structures", "laws": laws}


# =============================================================================
# 10 REPORTS
# =============================================================================


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def final_metrics(
    portal_df: pd.DataFrame,
    basin_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    representation_df: pd.DataFrame,
    grammar_rankings: pd.DataFrame,
    semantic_rankings: pd.DataFrame,
    semantic_transfer: pd.DataFrame,
) -> dict[str, float]:
    total = len(portal_df)
    portals = portal_df[portal_df["is_portal"]] if total else pd.DataFrame()
    strong = portal_df[portal_df["is_strong_portal"]] if total else pd.DataFrame()
    evaluated_transfer = transfer_df[transfer_df["status"] == "evaluated"] if not transfer_df.empty else pd.DataFrame()
    cross_transfer = (
        evaluated_transfer[evaluated_transfer["source_domain"] != evaluated_transfer["target_domain"]]
        if not evaluated_transfer.empty
        else pd.DataFrame()
    )
    possible_cross = (
        len(transfer_df[transfer_df["source_domain"] != transfer_df["target_domain"]])
        if not transfer_df.empty and {"source_domain", "target_domain"}.issubset(transfer_df.columns)
        else 0
    )
    return {
        "portal_density": float(len(portals) / max(total, 1)),
        "strong_portal_density": float(len(strong) / max(total, 1)),
        "portal_depth": float(portals["depth"].mean()) if not portals.empty else 0.0,
        "portal_transfer": float(evaluated_transfer["performance_retention"].mean()) if not evaluated_transfer.empty else 0.0,
        "portal_cross_transfer": float(cross_transfer["performance_retention"].mean()) if not cross_transfer.empty else 0.0,
        "portal_cross_transfer_coverage": float(len(cross_transfer) / max(possible_cross, 1)),
        "portal_compression": float(portals["compressive_score"].mean()) if not portals.empty else 0.0,
        "portal_rarity": float(1.0 - len(portals) / max(total, 1)),
        "portal_stability": float(portals["stability"].mean()) if not portals.empty else 0.0,
        "mean_raw_score": float(representation_df["raw_score"].mean()) if not representation_df.empty else 0.0,
        "mean_best_portal_score": float(representation_df["best_portal_score"].mean()) if not representation_df.empty else 0.0,
        "mean_basin_density": float(basin_df["portal_density"].mean()) if not basin_df.empty else 0.0,
        "surviving_operator_families": float((grammar_rankings["domains_surviving"] >= 2).sum()) if not grammar_rankings.empty else 0.0,
        "best_operator_grammar_score": float(grammar_rankings["portal_score"].max()) if not grammar_rankings.empty else 0.0,
        "semantic_transfer": float(semantic_transfer[semantic_transfer["source_domain"] != semantic_transfer["target_domain"]]["semantic_transfer"].mean()) if not semantic_transfer.empty else 0.0,
        "semantic_stability": float(semantic_rankings["semantic_stability"].mean()) if not semantic_rankings.empty else 0.0,
        "semantic_recurrence": float(semantic_rankings["domains_surviving"].mean()) if not semantic_rankings.empty else 0.0,
        "semantic_advantage": float(semantic_rankings["semantic_advantage"].mean()) if not semantic_rankings.empty else 0.0,
        "semantic_rarity": float(semantic_rankings["semantic_rarity"].mean()) if not semantic_rankings.empty else 0.0,
        "semantic_survival": float(semantic_rankings["semantic_survival"].mean()) if not semantic_rankings.empty else 0.0,
        "best_semantic_score": float(semantic_rankings["semantic_score"].max()) if not semantic_rankings.empty else 0.0,
    }


def write_reports(
    out: Path,
    portal_df: pd.DataFrame,
    representation_df: pd.DataFrame,
    basin_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    operator_df: pd.DataFrame,
    family_rankings: pd.DataFrame,
    survival_df: pd.DataFrame,
    semantic_basins: pd.DataFrame,
    semantic_transfer: pd.DataFrame,
    semantic_families: pd.DataFrame,
    semantic_rankings: pd.DataFrame,
    lawbook: dict[str, Any],
    semantic_lawbook: dict[str, Any],
    counterexamples: pd.DataFrame,
    manifest: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    portal_df.to_csv(out / "portal_rankings.csv", index=False)
    portal_df.to_csv(out / "coordinate_rankings.csv", index=False)
    operator_df.to_csv(out / "portal_grammar.csv", index=False)
    family_rankings.to_csv(out / "portal_grammar_rankings.csv", index=False)
    survival_df.to_csv(out / "operator_survival.csv", index=False)
    semantic_basins.to_csv(out / "semantic_basins.csv", index=False)
    semantic_transfer.to_csv(out / "semantic_transfer.csv", index=False)
    semantic_families.to_csv(out / "semantic_families.csv", index=False)
    semantic_rankings.to_csv(out / "semantic_rankings.csv", index=False)
    representation_df.to_csv(out / "representation_scores.csv", index=False)
    basin_df.to_csv(out / "portal_basins.csv", index=False)
    transfer_df.to_csv(out / "transfer_matrix.csv", index=False)
    counterexamples.to_csv(out / "counterexamples.csv", index=False)
    counterexamples.to_csv(out / "semantic_counterexamples.csv", index=False)
    write_json(out / "lawbook_v144.json", lawbook)
    write_json(out / "semantic_lawbook.json", semantic_lawbook)
    write_json(out / "lawbook_v145.json", semantic_lawbook)
    write_json(out / "manifest.json", manifest)
    write_json(out / "final_metrics.json", metrics)
    write_markdown_report(out, portal_df, representation_df, family_rankings, semantic_rankings, semantic_lawbook, metrics, manifest)
    write_final_conclusion(out, metrics, semantic_lawbook)


def write_markdown_report(
    out: Path,
    portal_df: pd.DataFrame,
    representation_df: pd.DataFrame,
    family_rankings: pd.DataFrame,
    semantic_rankings: pd.DataFrame,
    lawbook: dict[str, Any],
    metrics: dict[str, float],
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# MATHGRAPH v145 Semantic Portal Discovery Report",
        "",
        "Hypothesis: multiple coordinates may instantiate the same semantic continuation law.",
        "",
        "## Run Manifest",
        "",
        f"- datasets evaluated: {manifest['datasets_evaluated']}",
        f"- synthetic datasets: {manifest['synthetic_datasets']}",
        f"- real datasets: {manifest['real_datasets']}",
        f"- coordinates evaluated: {len(portal_df)}",
        f"- operator families evaluated: {0 if family_rankings.empty else len(family_rankings)}",
        f"- semantic families evaluated: {0 if semantic_rankings.empty else len(semantic_rankings)}",
        "",
        "## Final Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value:.4f}")
    lines.extend(["", "## Semantic Families", ""])
    if semantic_rankings.empty:
        lines.append("No semantic families were evaluated.")
    else:
        for _, row in semantic_rankings.head(12).iterrows():
            lines.append(
                f"- {row['semantic_pattern']} -> {row['emergent_concept']}: "
                f"domains={int(row['domains_surviving'])}, transfer={row['transfer_rate']:.3f}, "
                f"stability={row['semantic_stability']:.3f}, score={row['semantic_score']:.3f}"
            )
    lines.extend(["", "## Operator Families", ""])
    if family_rankings.empty:
        lines.append("No operator families were evaluated.")
    else:
        for _, row in family_rankings.head(12).iterrows():
            lines.append(
                f"- {row['operator_family']}: domains={int(row['domains_surviving'])}, "
                f"survival={row['survival_rate']:.3f}, transfer={row['transfer_rate']:.3f}, "
                f"grammar_score={row['portal_score']:.3f}"
            )
    lines.extend(["", "## 2. Which Coordinates Survive?", ""])
    if portal_df.empty:
        lines.append("No coordinates were evaluated.")
    else:
        cols = ["domain", "coordinate", "expression", "depth", "coordinate_score", "raw_best_single_score", "portal_score", "stability"]
        for _, row in portal_df.sort_values("portal_score", ascending=False).head(20).iterrows():
            lines.append(
                f"- {row['domain']}: `{row['expression']}` "
                f"score={row['coordinate_score']:.3f}, portal={row['portal_score']:.3f}, depth={int(row['depth'])}"
            )
    lines.extend(["", "## 3. Which Operators Fail?", ""])
    if not portal_df.empty:
        failures = portal_df[portal_df["failure_mode"] != "survived"]
        for family, group in failures.groupby("operator_family"):
            lines.append(f"- {family}: failures={len(group)}, common_mode={group['failure_mode'].mode().iloc[0]}")
    lines.extend(["", "## Do Semantic Continuation Structures Transfer?", ""])
    if lawbook["laws"]:
        lines.append("Some semantic structures met the promotion rule. Universality is not claimed.")
    else:
        lines.append("No semantic structure met the promotion rule for cross-domain reuse.")
    lines.extend(["", "## Is Representation Search Actually Semantic Search?", ""])
    lines.append(
        "This run measures that claim indirectly: coordinate successes are aggregated into semantic continuation "
        "patterns, basin clusters, transfer, and residual-evolution evidence."
    )
    lines.extend(["", "## Representation Scores", ""])
    for _, row in representation_df.iterrows():
        lines.append(
            f"- {row['domain']}: raw={row['raw_score']:.3f}, "
            f"best_coordinate={row['best_coordinate_score']:.3f}, portals={int(row['portal_count'])}"
        )
    lines.extend(["", "## Semantic Lawbook", "", f"Promoted semantic laws: {len(lawbook['laws'])}", ""])
    for law in lawbook["laws"][:20]:
        lines.append(
            f"- {law['law']} as {law['emergent_concept']}: domains={law['domains']}, "
            f"score={law['score']:.3f}, transfer={law['transfer']:.3f}"
        )
    (out / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 11 FINAL CONCLUSION
# =============================================================================


def verdict(metrics: dict[str, float], lawbook: dict[str, Any]) -> tuple[str, str]:
    law_count = len(lawbook["laws"])
    coordinate_signal = metrics["portal_density"] > 0 and metrics["mean_best_portal_score"] > 1.05
    grammar_signal = metrics["surviving_operator_families"] >= 2
    semantic_signal = law_count >= 1 or (metrics["semantic_recurrence"] >= 2 and metrics["semantic_survival"] > 0.03)
    if semantic_signal and law_count >= 1:
        return "C", "semantic continuation structures transfer"
    if grammar_signal:
        return "B", "operator grammars transfer more clearly than semantic laws in this run"
    if coordinate_signal:
        return "A", "coordinates transfer more clearly than higher-level abstractions in this run"
    return "D", "nothing transfers convincingly in this run"


def write_final_conclusion(out: Path, metrics: dict[str, float], lawbook: dict[str, Any]) -> None:
    grade, statement = verdict(metrics, lawbook)
    text = f"""# Final Conclusion

Outcome: **{grade}**

Interpretation: {statement}.

This is a research result, not a proof. A semantic law is promoted only when a
semantic continuation pattern appears in multiple domains, survives transfer,
is stable under perturbation, and exceeds the advantage threshold.

Scientific caution: this run must not be read as universal portals, universal
semantics, universal coordinates, or intelligence solved. It only reports
measured semantic-family evidence.
"""
    (out / "final_conclusion.md").write_text(text, encoding="utf-8")


# =============================================================================
# COUNTEREXAMPLES
# =============================================================================


def build_counterexamples(portal_df: pd.DataFrame, transfer_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not portal_df.empty:
        failures = portal_df[portal_df["failure_mode"] != "survived"]
        for _, row in failures.iterrows():
            rows.append(
                {
                    "type": "semantic_failure",
                    "domain": row["domain"],
                    "coordinate": row["coordinate"],
                    "semantic_pattern": row["semantic_pattern"],
                    "emergent_concept": row["emergent_concept"],
                    "detail": f"{row['failure_mode']}; residual={row['residual']:.3f}; try={','.join(row['semantic_evolution'])}",
                }
            )
    if not transfer_df.empty:
        failures = transfer_df[(transfer_df["status"] != "evaluated") | ((transfer_df["status"] == "evaluated") & (~transfer_df["survival"]))]
        for _, row in failures.iterrows():
            pattern = row.get("semantic_pattern", row.get("operator_family", "unknown"))
            rows.append(
                {
                    "type": "negative_semantic_transfer",
                    "domain": row["target_domain"],
                    "coordinate": pattern,
                    "semantic_pattern": pattern,
                    "emergent_concept": EMERGENT_CONCEPTS.get(pattern, "unknown"),
                    "detail": row["status"],
                }
            )
    return pd.DataFrame(rows, columns=["type", "domain", "coordinate", "semantic_pattern", "emergent_concept", "detail"])


# =============================================================================
# ORCHESTRATION
# =============================================================================


def run_engine(args: argparse.Namespace) -> dict[str, Any]:
    mount_drive_if_requested(args.mount_drive)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    download_rows = attempt_dataset_downloads(out / "downloads", allow_download=args.download)
    data_dirs = [Path(path) for path in args.data_dir] if args.data_dir else []
    if args.download:
        data_dirs.append(out / "downloads")
    real_datasets, skipped = load_real_datasets(data_dirs) if data_dirs else ([], [])
    synthetic = build_synthetic_worlds(seed=args.seed, n=args.n)
    datasets = synthetic + real_datasets

    all_portals: list[PortalResult] = []
    all_coords_by_domain: dict[str, list[Coordinate]] = {}
    best_coords: dict[str, Coordinate] = {}
    representation_rows: list[dict[str, Any]] = []

    for spec in datasets:
        print(f"Evaluating {spec.domain} ({len(spec.df)} rows)")
        results, representation, coords = portal_search(spec, max_coords=args.max_coords)
        representation_rows.append(representation)
        all_portals.extend(results)
        all_coords_by_domain[spec.domain] = coords
        if results:
            coord_map = {coord.name: coord for coord in coords}
            best = results[0]
            if best.coordinate in coord_map:
                best_coords[spec.domain] = coord_map[best.coordinate]

    portal_df = pd.DataFrame([asdict(row) for row in all_portals])
    representation_df = pd.DataFrame(representation_rows)
    graph, basin_df = build_portal_basins(all_portals)
    plot_basin_graph(graph, out / "portal_basins.png")
    operator_df, family_df = build_operator_grammar(portal_df)
    transfer_df = build_family_transfer_matrix(portal_df)
    family_rankings = add_transfer_rates(family_df, transfer_df)
    semantic_families = build_semantic_families(portal_df)
    semantic_transfer = build_semantic_transfer(portal_df)
    semantic_rankings = add_semantic_transfer_rates(semantic_families, semantic_transfer)
    semantic_basins = build_semantic_basins(portal_df)
    survival_df = family_rankings[
        [
            "operator_family",
            "frequency",
            "survivor_count",
            "survival_rate",
            "transfer_rate",
            "domains_appearing",
            "domains_surviving",
        ]
    ].copy() if not family_rankings.empty else pd.DataFrame()
    lawbook = build_lawbook(family_rankings, transfer_df)
    semantic_lawbook = build_semantic_lawbook(semantic_rankings, semantic_transfer)
    counterexamples = build_counterexamples(portal_df, semantic_transfer)
    metrics = final_metrics(portal_df, basin_df, transfer_df, representation_df, family_rankings, semantic_rankings, semantic_transfer)
    manifest = {
        "system": "MATHGRAPH v145 Semantic Portal Discovery Engine",
        "out": str(out),
        "seed": args.seed,
        "n": args.n,
        "max_coords": args.max_coords,
        "datasets_evaluated": len(datasets),
        "synthetic_datasets": len(synthetic),
        "real_datasets": len(real_datasets),
        "skipped_real_files": skipped,
        "downloads": download_rows,
        "dependencies": list(REQUIRED_PACKAGES.values()),
    }
    write_reports(
        out,
        portal_df,
        representation_df,
        basin_df,
        transfer_df,
        operator_df,
        family_rankings,
        survival_df,
        semantic_basins,
        semantic_transfer,
        semantic_families,
        semantic_rankings,
        lawbook,
        semantic_lawbook,
        counterexamples,
        manifest,
        metrics,
    )
    return {"metrics": metrics, "law_count": len(semantic_lawbook["laws"]), "out": str(out), "verdict": verdict(metrics, semantic_lawbook)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MATHGRAPH v145 Semantic Portal Discovery Engine")
    parser.add_argument("--out", default="mathgraph_v145_out", help="Output directory")
    parser.add_argument("--data-dir", action="append", default=[], help="Recursive local data directory; may be passed multiple times")
    parser.add_argument("--download", action="store_true", help="Best-effort public dataset downloads")
    parser.add_argument("--mount-drive", action="store_true", help="Mount Google Drive when running in Colab")
    parser.add_argument("--seed", type=int, default=143)
    parser.add_argument("--n", type=int, default=420, help="Rows per synthetic dataset")
    parser.add_argument("--max-coords", type=int, default=260, help="Maximum expression coordinates per dataset")
    parser.add_argument("--quick", action="store_true", help="Fast smoke run")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.quick:
        args.n = min(args.n, 180)
        args.max_coords = min(args.max_coords, 120)
    result = run_engine(args)
    grade, statement = result["verdict"]
    print(json.dumps({"out": result["out"], "law_count": result["law_count"], "verdict": grade, "statement": statement}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
