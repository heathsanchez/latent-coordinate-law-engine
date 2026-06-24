#!/usr/bin/env python3
"""MATHGRAPH v149: Continuation Kernel Discovery Engine.

The experiment asks whether successful abstractions preserve future
continuation: viable downstream search branches, rediscoverability, and
stability under perturbation/intervention. Prediction is measured, but it is
not sufficient for promotion.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REQUIRED = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn"}


def install_missing() -> None:
    missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


install_missing()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


PROMOTION = {
    "continuation": 0.60,
    "perturbation_survival": 0.60,
    "counterexample_rate": 0.20,
    "rediscovery": 0.50,
    "search_acceleration": 1.50,
    "holdout_success": 0.50,
    "minimum_worlds": 3,
}

FAMILY_ORDER = [
    "RESOURCE_CONSTRAINT",
    "CAPACITY_LOAD",
    "SIGNAL_NOISE",
    "SEARCH_BOTTLENECK",
    "ENERGY_DISSIPATION",
    "CONTROL_PARAMETER",
    "FREEDOM_RESTRICTION",
    "CONTINUATION_COST",
    "AVAILABLE_CAPACITY",
    "UNKNOWN",
]


@dataclass
class World:
    name: str
    family: str
    df: pd.DataFrame
    target: str
    latent: np.ndarray
    role_map: dict[str, str]
    source_columns: dict[str, str]
    noise_level: float


@dataclass
class Candidate:
    name: str
    expression: str
    operator: str
    columns: list[str]
    depth: int
    values: pd.Series


def mount_drive(enabled: bool) -> None:
    if not enabled:
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as exc:
        print(f"Drive mount skipped: {exc}")


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def random_name(rng: np.random.Generator, used: set[str]) -> str:
    while True:
        name = "x" + "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=5))
        if name not in used:
            used.add(name)
            return name


def transform(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sign = float(rng.choice([1, 1, 1, -1]))
    scale = float(10 ** rng.uniform(-1.0, 1.0))
    offset = float(rng.uniform(-4.0, 4.0))
    out = sign * values * scale + offset
    if out.min() <= 0:
        out = out - out.min() + rng.uniform(0.25, 2.0)
    return out


def build_world(
    name: str,
    family: str,
    roles: list[str],
    formula: Callable[[dict[str, np.ndarray]], np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> World:
    used: set[str] = set()
    raw: dict[str, np.ndarray] = {}
    df = pd.DataFrame()
    role_map: dict[str, str] = {}
    source_columns: dict[str, str] = {}
    denominator_roles = {
        "constraint",
        "moles",
        "load",
        "noise",
        "dissipation",
        "temperature",
        "restriction",
        "bottleneck",
        "obstacle",
        "cost",
        "clauses",
    }
    for role in roles:
        low, high = (0.7, 12.0) if role in denominator_roles else (0.5, 22.0)
        values = rng.uniform(low, high, n)
        raw[role] = values
        col = random_name(rng, used)
        df[col] = transform(values, rng)
        role_map[col] = role
        source_columns[role] = col
    for _ in range(4):
        col = random_name(rng, used)
        df[col] = transform(rng.uniform(0.3, 28.0, n), rng)
        role_map[col] = "distractor"
    latent = formula(raw)
    noise_level = float(rng.uniform(0.01, 0.08))
    df["outcome"] = latent + rng.normal(0, noise_level * float(np.std(latent) or 1.0), n)
    return World(name, family, df, "outcome", latent, role_map, source_columns, noise_level)


def generate_worlds(n: int, seed: int, variants: int) -> list[World]:
    rng = np.random.default_rng(seed)
    specs: list[tuple[str, str, list[str], Callable[[dict[str, np.ndarray]], np.ndarray]]] = [
        ("GAS", "RESOURCE_CONSTRAINT", ["pressure", "volume", "moles"], lambda v: (v["pressure"] * v["volume"]) / v["moles"]),
        ("PHASE", "CONTROL_PARAMETER", ["control", "coupling", "temperature"], lambda v: (v["control"] * v["coupling"]) / v["temperature"]),
        ("CAPACITY", "CAPACITY_LOAD", ["capacity", "load"], lambda v: v["capacity"] / v["load"]),
        ("OBSTRUCTION", "AVAILABLE_CAPACITY", ["clearance", "width", "load"], lambda v: (v["clearance"] * v["width"]) / v["load"]),
        ("SIGNAL", "SIGNAL_NOISE", ["signal", "noise"], lambda v: v["signal"] / v["noise"]),
        ("ENERGY", "ENERGY_DISSIPATION", ["energy", "dissipation"], lambda v: v["energy"] / v["dissipation"]),
        ("CONTROL", "CONTROL_PARAMETER", ["control", "coupling", "temperature"], lambda v: (v["control"] * v["coupling"]) / v["temperature"]),
        ("MAZE", "SEARCH_BOTTLENECK", ["search_budget", "bottleneck"], lambda v: v["search_budget"] / v["bottleneck"]),
        ("MODULAR", "UNKNOWN", ["x", "n", "offset"], lambda v: ((v["x"] + v["offset"]) % (v["n"] + 1.0)) / (v["n"] + 1.0)),
        ("ARC_STYLE", "FREEDOM_RESTRICTION", ["objects", "symmetry", "constraint"], lambda v: (v["objects"] + v["symmetry"]) / v["constraint"]),
        ("SEARCH", "CONTINUATION_COST", ["continuation", "obstacle", "cost"], lambda v: v["continuation"] / (v["obstacle"] + v["cost"])),
        ("SAT", "FREEDOM_RESTRICTION", ["variables", "clauses", "slack"], lambda v: (v["variables"] + v["slack"]) / v["clauses"]),
        ("ETP_STYLE", "FREEDOM_RESTRICTION", ["projection", "restriction", "freedom"], lambda v: (v["projection"] + v["freedom"]) / v["restriction"]),
    ]
    worlds = []
    for base, family, roles, formula in specs:
        for i in range(variants):
            worlds.append(build_world(f"{base}_{i + 1}", family, roles, formula, n, rng))
    return worlds


def generate_candidates(world: World, limit: int | None = None) -> list[Candidate]:
    cols = [c for c in world.df.columns if c != world.target]
    df = world.df
    candidates: list[Candidate] = []
    seen: set[str] = set()

    def add(name: str, expression: str, operator: str, columns: list[str], depth: int, values: pd.Series) -> None:
        if name in seen:
            return
        seen.add(name)
        values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if values.nunique(dropna=False) > 1:
            candidates.append(Candidate(name, expression, operator, columns, depth, values))

    for x in cols:
        add(x, x, "identity", [x], 1, df[x])
    for x in cols:
        for y in cols:
            if x == y:
                continue
            add(f"{x}_over_{y}", f"{x}/{y}", "ratio", [x, y], 2, safe_div(df[x], df[y]))
            add(f"{x}_minus_{y}", f"{x}-{y}", "difference", [x, y], 2, df[x] - df[y])
            add(f"{x}_plus_{y}", f"{x}+{y}", "sum", [x, y], 2, df[x] + df[y])
            add(f"{x}_times_{y}", f"{x}*{y}", "product", [x, y], 2, df[x] * df[y])
            add(f"min_{x}_{y}", f"min({x},{y})", "min", [x, y], 2, pd.Series(np.minimum(df[x], df[y]), index=df.index))
            add(f"max_{x}_{y}", f"max({x},{y})", "max", [x, y], 2, pd.Series(np.maximum(df[x], df[y]), index=df.index))
    for x in cols:
        for y in cols:
            for z in cols:
                if len({x, y, z}) != 3:
                    continue
                add(f"{x}_times_{y}_over_{z}", f"({x}*{y})/{z}", "product_ratio", [x, y, z], 3, safe_div(df[x] * df[y], df[z]))
                add(f"{x}_plus_{y}_over_{z}", f"({x}+{y})/{z}", "sum_ratio", [x, y, z], 3, safe_div(df[x] + df[y], df[z]))
                add(f"{x}_minus_{y}_over_{z}", f"({x}-{y})/{z}", "difference_ratio", [x, y, z], 3, safe_div(df[x] - df[y], df[z]))
                add(f"{x}_over_{y}_plus_{z}", f"{x}/({y}+{z})", "ratio_sum", [x, y, z], 3, safe_div(df[x], df[y] + df[z]))
                add(f"{x}_times_{y}_over_{z}_plus_1", f"({x}*{y})/({z}+1)", "product_sum_ratio", [x, y, z], 3, safe_div(df[x] * df[y], df[z] + 1.0))
                if limit and len(candidates) >= limit:
                    return candidates
    return candidates


def score_candidate(values: pd.Series, target: pd.Series, seed: int) -> float:
    train_idx, test_idx = train_test_split(values.index, test_size=0.30, random_state=seed)
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=seed)
    model.fit(pd.DataFrame({"c": values.loc[train_idx]}), target.loc[train_idx])
    pred = model.predict(pd.DataFrame({"c": values.loc[test_idx]}))
    return float(max(-1.0, r2_score(target.loc[test_idx], pred)))


def raw_baseline(world: World) -> float:
    cols = [c for c in world.df.columns if c != world.target]
    train_idx, test_idx = train_test_split(world.df.index, test_size=0.30, random_state=149)
    model = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, random_state=149)
    model.fit(world.df.loc[train_idx, cols], world.df.loc[train_idx, world.target])
    pred = model.predict(world.df.loc[test_idx, cols])
    return float(max(-1.0, r2_score(world.df.loc[test_idx, world.target], pred)))


def rebuild(candidate: Candidate, df: pd.DataFrame) -> pd.Series | None:
    c = candidate.columns
    try:
        if candidate.operator == "identity":
            return df[c[0]]
        if candidate.operator == "ratio":
            return safe_div(df[c[0]], df[c[1]])
        if candidate.operator == "difference":
            return df[c[0]] - df[c[1]]
        if candidate.operator == "sum":
            return df[c[0]] + df[c[1]]
        if candidate.operator == "product":
            return df[c[0]] * df[c[1]]
        if candidate.operator == "min":
            return pd.Series(np.minimum(df[c[0]], df[c[1]]), index=df.index)
        if candidate.operator == "max":
            return pd.Series(np.maximum(df[c[0]], df[c[1]]), index=df.index)
        if candidate.operator == "product_ratio":
            return safe_div(df[c[0]] * df[c[1]], df[c[2]])
        if candidate.operator == "sum_ratio":
            return safe_div(df[c[0]] + df[c[1]], df[c[2]])
        if candidate.operator == "difference_ratio":
            return safe_div(df[c[0]] - df[c[1]], df[c[2]])
        if candidate.operator == "ratio_sum":
            return safe_div(df[c[0]], df[c[1]] + df[c[2]])
        if candidate.operator == "product_sum_ratio":
            return safe_div(df[c[0]] * df[c[1]], df[c[2]] + 1.0)
    except Exception:
        return None
    return None


def roles(candidate: Candidate, world: World) -> list[str]:
    return [world.role_map.get(c, "distractor") for c in candidate.columns]


def infer_family(candidate: Candidate, world: World) -> str:
    r = set(roles(candidate, world))
    op = candidate.operator
    if {"capacity", "load"}.issubset(r) and op in {"ratio", "product_ratio", "sum_ratio", "ratio_sum"}:
        return "CAPACITY_LOAD"
    if {"signal", "noise"}.issubset(r) and op in {"ratio", "difference_ratio", "ratio_sum"}:
        return "SIGNAL_NOISE"
    if {"search_budget", "bottleneck"}.issubset(r) and op in {"ratio", "sum_ratio", "ratio_sum"}:
        return "SEARCH_BOTTLENECK"
    if {"energy", "dissipation"}.issubset(r) and op in {"ratio", "product_ratio", "difference_ratio"}:
        return "ENERGY_DISSIPATION"
    if {"control", "coupling", "temperature"}.issubset(r) and op in {"product_ratio", "product_sum_ratio"}:
        return "CONTROL_PARAMETER"
    if r & {"constraint", "moles"} and op in {"ratio", "product_ratio", "sum_ratio", "ratio_sum"}:
        return "RESOURCE_CONSTRAINT"
    if r & {"restriction", "clauses"} and op in {"ratio", "sum_ratio", "ratio_sum"}:
        return "FREEDOM_RESTRICTION"
    if r & {"cost", "obstacle"} and op in {"ratio_sum", "sum_ratio", "ratio"}:
        return "CONTINUATION_COST"
    if {"clearance", "width", "load"}.issubset(r) and op in {"product_ratio", "product_sum_ratio"}:
        return "AVAILABLE_CAPACITY"
    return "UNKNOWN"


def perturbation_survival(candidate: Candidate, world: World, base_score: float) -> float:
    rng = np.random.default_rng(9149)
    df = world.df.copy()
    for col in candidate.columns:
        df[col] = df[col] + rng.normal(0, 0.05 * float(df[col].std() or 1.0), len(df))
    rebuilt = rebuild(candidate, df)
    if rebuilt is None:
        return 0.0
    score = score_candidate(rebuilt, df[world.target], 9149)
    return float(np.clip((score + 1.0) / max(base_score + 1.0, 1e-9), 0.0, 1.25))


def holdout_success(candidate: Candidate, world: World) -> float:
    return float(score_candidate(candidate.values, world.df[world.target], 3149) > 0.45)


def counterexample_rate(candidate: Candidate, world: World) -> float:
    train_idx, test_idx = train_test_split(candidate.values.index, test_size=0.30, random_state=4149)
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=4149)
    model.fit(pd.DataFrame({"c": candidate.values.loc[train_idx]}), world.df.loc[train_idx, world.target])
    pred = model.predict(pd.DataFrame({"c": candidate.values.loc[test_idx]}))
    residual = np.abs(pred - np.asarray(world.df.loc[test_idx, world.target]))
    scale = float(np.std(world.df.loc[test_idx, world.target]) or 1.0)
    return float(np.mean(residual > 0.45 * scale))


def intervention_stability(candidate: Candidate, world: World, base_score: float) -> float:
    df = world.df.copy()
    for col in candidate.columns:
        role = world.role_map.get(col, "distractor")
        sign = -1.0 if role in {"constraint", "load", "noise", "dissipation", "restriction", "bottleneck", "temperature", "obstacle", "cost", "clauses", "moles"} else 1.0
        df[col] = df[col] * (1.0 + 0.10 * sign)
    rebuilt = rebuild(candidate, df)
    if rebuilt is None:
        return 0.0
    shifted = score_candidate(rebuilt, df[world.target], 5149)
    return float(np.clip((shifted + 1.0) / max(base_score + 1.0, 1e-9), 0.0, 1.25))


def continuation_for_candidate(
    candidate: Candidate,
    world: World,
    all_candidates: list[Candidate],
    score_map: dict[str, float],
    base_viable: int,
    threshold: float,
) -> dict[str, float]:
    relevant_roles = set(roles(candidate, world)) - {"distractor"}
    if not relevant_roles:
        family_candidates = [c for c in all_candidates if c.operator == candidate.operator]
    else:
        family_candidates = [
            c
            for c in all_candidates
            if (set(roles(c, world)) & relevant_roles) or c.operator == candidate.operator
        ]
    family_candidates = family_candidates[:70]
    remaining = sum(1 for cand in family_candidates if score_map.get(cand.name, -1.0) >= threshold)
    continuation_density = remaining / max(base_viable, 1)

    branch_scores = []
    for branch, noise in enumerate([0.00, 0.06, 0.12]):
        df = world.df.copy()
        rng = np.random.default_rng(7000 + branch)
        for col in candidate.columns:
            df[col] = df[col] + rng.normal(0, noise * float(df[col].std() or 1.0), len(df))
        rebuilt = rebuild(candidate, df)
        if rebuilt is None:
            branch_scores.append(0.0)
            continue
        branch_scores.append(float(score_candidate(rebuilt, df[world.target], 7149 + branch) >= threshold))
    search_branch_survival = float(np.mean(branch_scores))
    strong_density = min(1.0, continuation_density) * search_branch_survival
    return {
        "remaining_viable_futures": float(remaining),
        "possible_futures": float(base_viable),
        "future_volume": float(remaining),
        "continuation_density": float(np.clip(continuation_density, 0.0, 1.5)),
        "strong_continuation_density": float(np.clip(strong_density, 0.0, 1.0)),
        "search_branch_survival": search_branch_survival,
        "continuation_survival": float(np.clip((continuation_density + search_branch_survival) / 2.0, 0.0, 1.0)),
        "continuation_compression": float(base_viable / max(remaining, 1)),
        "continuation_gain": float(remaining / max(len(family_candidates), 1)),
    }


def search_costs(candidates: list[Candidate], score_map: dict[str, float], threshold: float) -> tuple[float, float, int]:
    start = time.perf_counter()
    blind_rank = len(candidates)
    guided = sorted(candidates, key=lambda c: (c.depth, c.operator not in {"ratio", "product_ratio", "sum_ratio", "ratio_sum"}, c.name))
    guided_rank = len(guided)
    for i, cand in enumerate(candidates, 1):
        if score_map.get(cand.name, -1.0) >= threshold:
            blind_rank = i
            break
    for i, cand in enumerate(guided, 1):
        if score_map.get(cand.name, -1.0) >= threshold:
            guided_rank = i
            break
    elapsed = time.perf_counter() - start
    blind_cost = blind_rank + elapsed
    guided_cost = guided_rank + elapsed
    return blind_cost, guided_cost, guided_rank


def evaluate_world(world: World, threshold: float, candidate_limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = generate_candidates(world, candidate_limit)[: candidate_limit or None]
    raw = raw_baseline(world)
    scored_candidates = candidates[: candidate_limit or 120]
    score_map = {
        cand.name: score_candidate(cand.values, world.df[world.target], 1000 + idx)
        for idx, cand in enumerate(scored_candidates)
    }
    base_viable = sum(1 for value in score_map.values() if value >= threshold)
    blind_cost, guided_cost, rediscovery_rank = search_costs(scored_candidates, score_map, threshold)
    search_acceleration = blind_cost / max(guided_cost, 1e-9)
    rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    counter_rows: list[dict[str, Any]] = []
    for idx, cand in enumerate(scored_candidates):
        pred = score_map[cand.name]
        pert = perturbation_survival(cand, world, pred)
        ce = counterexample_rate(cand, world)
        holdout = holdout_success(cand, world)
        intervention = intervention_stability(cand, world, pred)
        cont = continuation_for_candidate(cand, world, scored_candidates, score_map, base_viable, threshold)
        family = infer_family(cand, world)
        rediscovery = float(family == world.family and pred >= threshold)
        law_value = (
            cont["continuation_survival"]
            * pert
            * (1.0 - ce)
            * (1.0 + max(search_acceleration - 1.0, 0.0))
            * (0.5 + holdout)
            * (0.5 + rediscovery)
        )
        row = {
            "world": world.name,
            "hidden_family": world.family,
            "candidate": cand.name,
            "expression": cand.expression,
            "operator": cand.operator,
            "family": family,
            "depth": cand.depth,
            "prediction_score": pred,
            "raw_score": raw,
            "prediction_advantage": pred - raw,
            "perturbation_survival": pert,
            "counterexample_rate": ce,
            "holdout_success": holdout,
            "intervention_stability": intervention,
            "rediscovery": rediscovery,
            "search_acceleration": search_acceleration,
            "rediscovery_rank": rediscovery_rank,
            "law_value": law_value,
            **cont,
        }
        rows.append(row)
        branch_rows.append(
            {
                "world": world.name,
                "candidate": cand.name,
                "family": family,
                "search_branch_survival": cont["search_branch_survival"],
                "remaining_viable_futures": cont["remaining_viable_futures"],
                "possible_futures": cont["possible_futures"],
            }
        )
        if ce >= PROMOTION["counterexample_rate"] or cont["continuation_survival"] < 0.35:
            counter_rows.append(
                {
                    "world": world.name,
                    "candidate": cand.name,
                    "family": family,
                    "counterexample_rate": ce,
                    "continuation_survival": cont["continuation_survival"],
                    "reason": "high_counterexample_rate" if ce >= PROMOTION["counterexample_rate"] else "low_continuation",
                }
            )
    return rows, branch_rows, counter_rows


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        rows.groupby("family", dropna=False)
        .agg(
            candidate_count=("candidate", "count"),
            worlds=("world", "nunique"),
            mean_prediction=("prediction_score", "mean"),
            mean_prediction_advantage=("prediction_advantage", "mean"),
            continuation_density=("continuation_density", "mean"),
            strong_continuation_density=("strong_continuation_density", "mean"),
            continuation_stability=("continuation_survival", "mean"),
            continuation_survival=("continuation_survival", "mean"),
            future_volume=("future_volume", "mean"),
            search_branch_survival=("search_branch_survival", "mean"),
            perturbation_survival=("perturbation_survival", "mean"),
            counterexample_rate=("counterexample_rate", "mean"),
            rediscovery=("rediscovery", "mean"),
            search_acceleration=("search_acceleration", "mean"),
            holdout_success=("holdout_success", "mean"),
            intervention_stability=("intervention_stability", "mean"),
            continuation_gain=("continuation_gain", "mean"),
            law_value=("law_value", "mean"),
        )
        .reset_index()
    )
    grouped["promote"] = (
        (grouped["continuation_survival"] > PROMOTION["continuation"])
        & (grouped["perturbation_survival"] > PROMOTION["perturbation_survival"])
        & (grouped["counterexample_rate"] < PROMOTION["counterexample_rate"])
        & (grouped["rediscovery"] > PROMOTION["rediscovery"])
        & (grouped["search_acceleration"] > PROMOTION["search_acceleration"])
        & (grouped["holdout_success"] > PROMOTION["holdout_success"])
        & (grouped["worlds"] >= PROMOTION["minimum_worlds"])
    )
    return grouped.sort_values(["promote", "law_value"], ascending=[False, False])


def correlations(rows: pd.DataFrame) -> dict[str, float]:
    target_cols = ["rediscovery", "holdout_success", "perturbation_survival", "counterexample_rate"]
    out: dict[str, float] = {}
    for col in target_cols:
        if rows["continuation_survival"].nunique() > 1 and rows[col].nunique() > 1:
            out[f"continuation_vs_{col}"] = float(rows["continuation_survival"].corr(rows[col]))
        else:
            out[f"continuation_vs_{col}"] = 0.0
        if rows["prediction_score"].nunique() > 1 and rows[col].nunique() > 1:
            out[f"prediction_vs_{col}"] = float(rows["prediction_score"].corr(rows[col]))
        else:
            out[f"prediction_vs_{col}"] = 0.0
    return out


def verdict(summary: pd.DataFrame, corr: dict[str, float]) -> tuple[str, str]:
    if summary["promote"].any() and corr.get("continuation_vs_rediscovery", 0.0) > corr.get("prediction_vs_rediscovery", 0.0):
        return "A", "Continuation strongly predicts abstraction survival."
    if (summary["continuation_survival"] > 0.50).any() and corr.get("continuation_vs_rediscovery", 0.0) > 0:
        return "B", "Continuation partially predicts survival."
    if corr.get("prediction_vs_rediscovery", 0.0) >= corr.get("continuation_vs_rediscovery", 0.0):
        return "C", "Continuation no better than prediction."
    return "D", "No measurable continuation effect."


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def write_outputs(
    out: Path,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    branch: pd.DataFrame,
    counters: pd.DataFrame,
    examples: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    rankings = scores.sort_values("law_value", ascending=False)
    corr = correlations(scores)
    grade, statement = verdict(summary, corr)
    promoted = summary[summary["promote"]].copy()
    lawbook = {
        "lawbook_version": "v149",
        "hypothesis": "continuation_preservation",
        "promotion_rules": PROMOTION,
        "verdict": grade,
        "laws": [
            {
                "family": row["family"],
                "law_value": float(row["law_value"]),
                "continuation_survival": float(row["continuation_survival"]),
                "search_branch_survival": float(row["search_branch_survival"]),
                "future_volume": float(row["future_volume"]),
                "worlds": int(row["worlds"]),
                "statement": f"{row['family']} preserves continuation across hidden worlds.",
            }
            for _, row in promoted.iterrows()
        ],
    }
    scores.to_csv(out / "continuation_scores.csv", index=False)
    rankings.to_csv(out / "continuation_rankings.csv", index=False)
    summary.to_csv(out / "continuation_survival.csv", index=False)
    branch.to_csv(out / "branch_survival.csv", index=False)
    scores[["world", "candidate", "family", "future_volume", "possible_futures", "remaining_viable_futures", "continuation_density"]].to_csv(out / "future_volume.csv", index=False)
    counters.to_csv(out / "counterexamples.csv", index=False)
    examples.to_csv(out / "continuation_examples.csv", index=False)
    write_json(out / "lawbook_v149.json", lawbook)
    write_json(
        out / "manifest.json",
        {
            "system": "MATHGRAPH v149 Continuation Kernel Discovery Engine",
            "quick": args.quick,
            "seed": args.seed,
            "worlds": int(scores["world"].nunique()),
            "candidates_scored": int(len(scores)),
            "correlations": corr,
        },
    )
    write_reports(out, summary, rankings, lawbook, corr, grade, statement)
    return {"out": str(out), "verdict": grade, "statement": statement, "laws": len(lawbook["laws"])}


def write_reports(out: Path, summary: pd.DataFrame, rankings: pd.DataFrame, lawbook: dict[str, Any], corr: dict[str, float], grade: str, statement: str) -> None:
    lines = [
        "# MATHGRAPH v149 Continuation Kernel Discovery Report",
        "",
        f"Verdict: **{grade}** — {statement}",
        "",
        "## What Preserves Continuation?",
    ]
    for _, row in summary.head(12).iterrows():
        lines.append(
            f"- {row['family']}: continuation={row['continuation_survival']:.3f}, "
            f"branch={row['search_branch_survival']:.3f}, future_volume={row['future_volume']:.1f}, "
            f"rediscovery={row['rediscovery']:.3f}, law_value={row['law_value']:.3f}, promote={bool(row['promote'])}"
        )
    lines.extend(
        [
            "",
            "## Does Continuation Beat Prediction?",
            f"- continuation vs rediscovery: {corr.get('continuation_vs_rediscovery', 0.0):.3f}",
            f"- prediction vs rediscovery: {corr.get('prediction_vs_rediscovery', 0.0):.3f}",
            f"- continuation vs holdout: {corr.get('continuation_vs_holdout_success', 0.0):.3f}",
            f"- prediction vs holdout: {corr.get('prediction_vs_holdout_success', 0.0):.3f}",
            "",
            "## Top Candidate Examples",
        ]
    )
    for _, row in rankings.head(10).iterrows():
        lines.append(
            f"- {row['world']} `{row['expression']}` [{row['family']}]: "
            f"continuation={row['continuation_survival']:.3f}, prediction={row['prediction_score']:.3f}, law_value={row['law_value']:.3f}"
        )
    lines.extend(["", "## Lawbook"])
    if lawbook["laws"]:
        for law in lawbook["laws"]:
            lines.append(f"- {law['family']}: {law['statement']}")
    else:
        lines.append("- No law promoted. Continuation evidence was insufficient under the promotion gates.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "This experiment does not reward prediction, recurrence, semantics, or transfer alone. "
            "A candidate must preserve viable future search under perturbation, intervention, holdout, rediscovery, and counterexample pressure.",
            "The result is evidence about continuation preservation as an operational invariant, not a claim of universal truth.",
        ]
    )
    (out / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    final = f"""# Final Conclusion

Outcome: **{grade}**

Interpretation: {statement}

v149 asks whether abstractions survive because they preserve future possibility.
The core metric is continuation preservation: how many viable downstream
candidate futures remain discoverable after perturbation, intervention,
hidden-world search, and changed search branches.

Promoted laws: {len(lawbook["laws"])}
"""
    (out / "final_conclusion.md").write_text(final, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    mount_drive(args.mount_drive)
    n = 70 if args.quick else 320
    variants = 2 if args.quick else 3
    candidate_limit = 80 if args.quick else 260
    worlds = generate_worlds(n, args.seed, variants)
    all_rows: list[dict[str, Any]] = []
    all_branch: list[dict[str, Any]] = []
    all_counters: list[dict[str, Any]] = []
    for world in worlds:
        print(f"Evaluating {world.name} ({world.family})")
        rows, branch, counters = evaluate_world(world, threshold=0.42, candidate_limit=candidate_limit)
        all_rows.extend(rows)
        all_branch.extend(branch)
        all_counters.extend(counters)
    scores = pd.DataFrame(all_rows)
    branch_df = pd.DataFrame(all_branch)
    counter_df = pd.DataFrame(all_counters)
    summary = aggregate(scores)
    examples = scores.sort_values("law_value", ascending=False).head(30)
    return write_outputs(Path(args.out), scores, summary, branch_df, counter_df, examples, args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MATHGRAPH v149 Continuation Kernel Discovery Engine")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="mathgraph_v149_out")
    p.add_argument("--seed", type=int, default=149)
    p.add_argument("--mount-drive", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
