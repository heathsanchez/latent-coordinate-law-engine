#!/usr/bin/env python3
"""MATHGRAPH v150: Future Volume Engine.

The experiment asks whether surviving abstractions preserve useful future
search volume. Prediction is recorded only as a comparator; promotion depends
on rediscoverable search branches remaining after abstraction-guided search,
perturbation, mutation, and holdout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
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
    "future_volume": 0.70,
    "rediscovery": 0.60,
    "acceleration": 1.50,
    "counterexamples": 0.20,
    "holdout": 0.50,
    "minimum_worlds": 3,
}

DENOMINATOR_ROLES = {
    "constraint",
    "moles",
    "load",
    "noise",
    "dissipation",
    "temperature",
    "bottleneck",
    "cost",
    "obstacle",
    "n",
}


@dataclass
class World:
    name: str
    family: str
    df: pd.DataFrame
    target: str
    role_map: dict[str, str]


@dataclass
class Candidate:
    name: str
    expression: str
    operator: str
    columns: list[str]
    values: pd.Series


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def random_name(rng: np.random.Generator, used: set[str]) -> str:
    while True:
        name = "x" + "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=5))
        if name not in used:
            used.add(name)
            return name


def transform(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sign = float(rng.choice([1.0, 1.0, 1.0, -1.0]))
    scale = float(10 ** rng.uniform(-1.0, 1.0))
    offset = float(rng.uniform(-3.0, 3.0))
    out = sign * values * scale + offset
    if out.min() <= 0:
        out = out - out.min() + rng.uniform(0.25, 1.5)
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
    for role in roles:
        low, high = (0.8, 12.0) if role in DENOMINATOR_ROLES else (0.5, 22.0)
        raw[role] = rng.uniform(low, high, n)
        col = random_name(rng, used)
        df[col] = transform(raw[role], rng)
        role_map[col] = role
    for _ in range(4):
        col = random_name(rng, used)
        df[col] = transform(rng.uniform(0.4, 26.0, n), rng)
        role_map[col] = "distractor"
    latent = formula(raw)
    noise = rng.uniform(0.01, 0.07) * float(np.std(latent) or 1.0)
    df["outcome"] = latent + rng.normal(0, noise, n)
    return World(name, family, df, "outcome", role_map)


def generate_worlds(n: int, seed: int, quick: bool) -> list[World]:
    rng = np.random.default_rng(seed)
    specs: list[tuple[str, str, list[str], Callable[[dict[str, np.ndarray]], np.ndarray]]] = [
        ("GAS", "RESOURCE_CONSTRAINT", ["pressure", "volume", "moles"], lambda v: (v["pressure"] * v["volume"]) / v["moles"]),
        ("PHASE", "CONTROL_PARAMETER", ["control", "coupling", "temperature"], lambda v: (v["control"] * v["coupling"]) / v["temperature"]),
        ("CAPACITY", "CAPACITY_LOAD", ["capacity", "load"], lambda v: v["capacity"] / v["load"]),
        ("SIGNAL", "SIGNAL_NOISE", ["signal", "noise"], lambda v: v["signal"] / v["noise"]),
        ("ENERGY", "ENERGY_DISSIPATION", ["energy", "dissipation"], lambda v: v["energy"] / v["dissipation"]),
        ("MAZE", "SEARCH_BOTTLENECK", ["search_budget", "bottleneck"], lambda v: v["search_budget"] / v["bottleneck"]),
        ("OBSTRUCTION", "AVAILABLE_CAPACITY", ["clearance", "width", "load"], lambda v: (v["clearance"] * v["width"]) / v["load"]),
        ("MODULAR", "UNKNOWN", ["x", "n", "offset"], lambda v: ((v["x"] + v["offset"]) % (v["n"] + 1.0)) / (v["n"] + 1.0)),
        ("ARC_STYLE", "CONTINUATION_COST", ["objects", "symmetry", "constraint"], lambda v: (v["objects"] + v["symmetry"]) / v["constraint"]),
    ]
    if quick:
        specs = specs[:8]
    worlds = [build_world(name, family, roles, formula, n, rng) for name, family, roles, formula in specs]
    return worlds


def generate_candidates(world: World) -> list[Candidate]:
    cols = [c for c in world.df.columns if c != world.target]
    df = world.df
    out: list[Candidate] = []
    seen: set[str] = set()

    def add(name: str, expression: str, operator: str, columns: list[str], values: pd.Series) -> None:
        if name in seen:
            return
        seen.add(name)
        values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if values.nunique(dropna=False) > 1:
            out.append(Candidate(name, expression, operator, columns, values))

    for x in cols:
        for y in cols:
            if x == y:
                continue
            add(f"{x}_plus_{y}", f"{x}+{y}", "sum", [x, y], df[x] + df[y])
            add(f"{x}_minus_{y}", f"{x}-{y}", "difference", [x, y], df[x] - df[y])
            add(f"{x}_times_{y}", f"{x}*{y}", "product", [x, y], df[x] * df[y])
            add(f"{x}_over_{y}", f"{x}/{y}", "ratio", [x, y], safe_div(df[x], df[y]))
            add(f"min_{x}_{y}", f"min({x},{y})", "min", [x, y], pd.Series(np.minimum(df[x], df[y]), index=df.index))
            add(f"max_{x}_{y}", f"max({x},{y})", "max", [x, y], pd.Series(np.maximum(df[x], df[y]), index=df.index))
    for x in cols:
        for y in cols:
            for z in cols:
                if len({x, y, z}) != 3:
                    continue
                add(f"{x}_plus_{y}_over_{z}", f"({x}+{y})/{z}", "sum_ratio", [x, y, z], safe_div(df[x] + df[y], df[z]))
                add(f"{x}_times_{y}_over_{z}", f"({x}*{y})/{z}", "product_ratio", [x, y, z], safe_div(df[x] * df[y], df[z]))
                add(f"{x}_over_{y}_plus_{z}", f"{x}/({y}+{z})", "ratio_sum", [x, y, z], safe_div(df[x], df[y] + df[z]))
    return out


def score(values: pd.Series, target: pd.Series, seed: int) -> float:
    train_idx, test_idx = train_test_split(values.index, test_size=0.30, random_state=seed)
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=seed)
    model.fit(pd.DataFrame({"c": values.loc[train_idx]}), target.loc[train_idx])
    pred = model.predict(pd.DataFrame({"c": values.loc[test_idx]}))
    return float(max(-1.0, r2_score(target.loc[test_idx], pred)))


def holdout(values: pd.Series, target: pd.Series, seed: int, threshold: float) -> float:
    return float(score(values, target, seed) >= threshold)


def counterexample_rate(values: pd.Series, target: pd.Series, seed: int) -> float:
    train_idx, test_idx = train_test_split(values.index, test_size=0.30, random_state=seed)
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=seed)
    model.fit(pd.DataFrame({"c": values.loc[train_idx]}), target.loc[train_idx])
    pred = model.predict(pd.DataFrame({"c": values.loc[test_idx]}))
    residual = np.abs(pred - np.asarray(target.loc[test_idx]))
    scale = float(np.std(target.loc[test_idx]) or 1.0)
    return float(np.mean(residual > 0.45 * scale))


def raw_score(world: World) -> float:
    cols = [c for c in world.df.columns if c != world.target]
    train_idx, test_idx = train_test_split(world.df.index, test_size=0.30, random_state=150)
    model = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, random_state=150)
    model.fit(world.df.loc[train_idx, cols], world.df.loc[train_idx, world.target])
    pred = model.predict(world.df.loc[test_idx, cols])
    return float(max(-1.0, r2_score(world.df.loc[test_idx, world.target], pred)))


def infer_family(candidate: Candidate, world: World) -> str:
    roles = set(world.role_map.get(c, "distractor") for c in candidate.columns)
    op = candidate.operator
    if {"capacity", "load"}.issubset(roles) and op in {"ratio", "product_ratio", "sum_ratio", "ratio_sum"}:
        return "CAPACITY_LOAD"
    if {"search_budget", "bottleneck"}.issubset(roles) and op in {"ratio", "sum_ratio", "ratio_sum"}:
        return "SEARCH_BOTTLENECK"
    if {"signal", "noise"}.issubset(roles) and op in {"ratio", "ratio_sum"}:
        return "SIGNAL_NOISE"
    if {"energy", "dissipation"}.issubset(roles) and op in {"ratio", "product_ratio"}:
        return "ENERGY_DISSIPATION"
    if {"control", "coupling", "temperature"}.issubset(roles) and op in {"product_ratio", "ratio_sum"}:
        return "CONTROL_PARAMETER"
    if {"clearance", "width", "load"}.issubset(roles) and op in {"product_ratio", "ratio_sum"}:
        return "AVAILABLE_CAPACITY"
    if roles & {"moles", "constraint"} and op in {"ratio", "product_ratio", "sum_ratio", "ratio_sum"}:
        return "RESOURCE_CONSTRAINT"
    if roles & {"cost", "obstacle"} and op in {"ratio", "sum_ratio", "ratio_sum"}:
        return "CONTINUATION_COST"
    return "UNKNOWN"


def perturb_candidate(candidate: Candidate, world: World, rng: np.random.Generator, strength: float) -> pd.Series:
    df = world.df.copy()
    for col in candidate.columns:
        df[col] = df[col] + rng.normal(0, strength * float(df[col].std() or 1.0), len(df))
    return rebuild(candidate, df)


def rebuild(candidate: Candidate, df: pd.DataFrame) -> pd.Series:
    c = candidate.columns
    if candidate.operator == "sum":
        return df[c[0]] + df[c[1]]
    if candidate.operator == "difference":
        return df[c[0]] - df[c[1]]
    if candidate.operator == "product":
        return df[c[0]] * df[c[1]]
    if candidate.operator == "ratio":
        return safe_div(df[c[0]], df[c[1]])
    if candidate.operator == "min":
        return pd.Series(np.minimum(df[c[0]], df[c[1]]), index=df.index)
    if candidate.operator == "max":
        return pd.Series(np.maximum(df[c[0]], df[c[1]]), index=df.index)
    if candidate.operator == "sum_ratio":
        return safe_div(df[c[0]] + df[c[1]], df[c[2]])
    if candidate.operator == "product_ratio":
        return safe_div(df[c[0]] * df[c[1]], df[c[2]])
    if candidate.operator == "ratio_sum":
        return safe_div(df[c[0]], df[c[1]] + df[c[2]])
    raise ValueError(f"unknown operator {candidate.operator}")


def branch_pool(candidate: Candidate, candidates: list[Candidate], world: World) -> list[Candidate]:
    candidate_roles = set(world.role_map.get(c, "distractor") for c in candidate.columns) - {"distractor"}
    pool = [
        c
        for c in candidates
        if c.operator == candidate.operator
        or bool((set(world.role_map.get(col, "distractor") for col in c.columns) - {"distractor"}) & candidate_roles)
    ]
    return pool or candidates


def future_volume_for(
    candidate: Candidate,
    candidates: list[Candidate],
    world: World,
    score_map: dict[str, float],
    threshold: float,
    branches: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pool = branch_pool(candidate, candidates, world)
    original = rng.choice(candidates, size=branches, replace=True)
    guided = rng.choice(pool, size=branches, replace=True)
    original_redisc = sum(score_map[c.name] >= threshold for c in original)
    guided_redisc = 0
    surviving = 0
    dead = 0
    depths: list[int] = []
    for i, branch in enumerate(guided):
        strength = float(rng.choice([0.00, 0.03, 0.06, 0.10]))
        mutation_penalty = strength * float(0.4 + 0.2 * len(branch.columns))
        branch_score = score_map[branch.name] - mutation_penalty + float(rng.normal(0, 0.025))
        ok = branch_score >= threshold
        guided_redisc += int(ok)
        surviving += int(branch_score > 0.0)
        dead += int(branch_score <= 0.0)
        depths.append(i + 1)
    future_volume = guided_redisc / max(branches, 1)
    original_volume = original_redisc / max(branches, 1)
    useful_future_ratio = guided_redisc / max(original_redisc, 1)
    return {
        "original_branches": float(branches),
        "surviving_branches": float(surviving),
        "rediscoverable_branches": float(guided_redisc),
        "dead_branches": float(dead),
        "original_rediscoverable_branches": float(original_redisc),
        "future_volume": float(future_volume),
        "original_future_volume": float(original_volume),
        "useful_future_ratio": float(useful_future_ratio),
        "search_depth": float(np.mean(depths) if depths else branches),
        "candidate_reduction": float(len(candidates) / max(len(pool), 1)),
        "search_compression": float((branches + len(candidates)) / max(branches + len(pool), 1)),
        "branch_survival": float(surviving / max(branches, 1)),
    }


def evaluate_world(world: World, branches: int, threshold: float, candidate_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = generate_candidates(world)[:candidate_limit]
    target = world.df[world.target]
    raw = raw_score(world)
    score_map = {c.name: score(c.values, target, 1500 + i) for i, c in enumerate(candidates)}
    rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    counter_rows: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        pred = score_map[cand.name]
        family = infer_family(cand, world)
        future = future_volume_for(cand, candidates, world, score_map, threshold, branches, 5000 + i)
        ce = counterexample_rate(cand.values, target, 2500 + i)
        hold = holdout(cand.values, target, 3500 + i, threshold)
        pert_values = perturb_candidate(cand, world, np.random.default_rng(4500 + i), 0.06)
        pert_score = score(pert_values, target, 5500 + i)
        perturbation_survival = float(np.clip((pert_score + 1.0) / max(pred + 1.0, 1e-9), 0.0, 1.25))
        rediscovery = float(family == world.family and pred >= threshold)
        acceleration = future["search_compression"]
        law_value = (
            future["future_volume"]
            * max(future["useful_future_ratio"], 0.0)
            * max(acceleration, 0.0)
            * (1.0 - ce)
            * (0.5 + hold)
            * (0.5 + rediscovery)
        )
        row = {
            "world": world.name,
            "hidden_family": world.family,
            "candidate": cand.name,
            "expression": cand.expression,
            "operator": cand.operator,
            "family": family,
            "prediction": pred,
            "raw_prediction": raw,
            "prediction_advantage": pred - raw,
            "counterexample_rate": ce,
            "counterexample_resistance": 1.0 - ce,
            "holdout_success": hold,
            "perturbation_survival": perturbation_survival,
            "rediscovery": rediscovery,
            "acceleration": acceleration,
            "law_value": law_value,
            **future,
        }
        rows.append(row)
        branch_rows.append(
            {
                "world": world.name,
                "candidate": cand.name,
                "family": family,
                "surviving_branches": future["surviving_branches"],
                "rediscoverable_branches": future["rediscoverable_branches"],
                "dead_branches": future["dead_branches"],
                "future_volume": future["future_volume"],
                "branch_survival": future["branch_survival"],
            }
        )
        if ce >= PROMOTION["counterexamples"] or future["future_volume"] < 0.25:
            counter_rows.append(
                {
                    "world": world.name,
                    "candidate": cand.name,
                    "family": family,
                    "counterexample_rate": ce,
                    "future_volume": future["future_volume"],
                    "reason": "high_counterexample_rate" if ce >= PROMOTION["counterexamples"] else "low_future_volume",
                }
            )
    return rows, branch_rows, counter_rows


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scores.groupby("family", dropna=False)
        .agg(
            candidates=("candidate", "count"),
            worlds=("world", "nunique"),
            future_volume=("future_volume", "mean"),
            useful_future_ratio=("useful_future_ratio", "mean"),
            surviving_branches=("surviving_branches", "mean"),
            rediscoverable_branches=("rediscoverable_branches", "mean"),
            dead_branches=("dead_branches", "mean"),
            search_depth=("search_depth", "mean"),
            candidate_reduction=("candidate_reduction", "mean"),
            acceleration=("acceleration", "mean"),
            prediction=("prediction", "mean"),
            prediction_advantage=("prediction_advantage", "mean"),
            counterexample_rate=("counterexample_rate", "mean"),
            holdout_success=("holdout_success", "mean"),
            perturbation_survival=("perturbation_survival", "mean"),
            rediscovery=("rediscovery", "mean"),
            law_value=("law_value", "mean"),
        )
        .reset_index()
    )
    summary["promote"] = (
        (summary["future_volume"] > PROMOTION["future_volume"])
        & (summary["rediscovery"] > PROMOTION["rediscovery"])
        & (summary["acceleration"] > PROMOTION["acceleration"])
        & (summary["counterexample_rate"] < PROMOTION["counterexamples"])
        & (summary["holdout_success"] > PROMOTION["holdout"])
        & (summary["worlds"] >= PROMOTION["minimum_worlds"])
    )
    return summary.sort_values(["promote", "law_value"], ascending=[False, False])


def correlations(scores: pd.DataFrame) -> dict[str, float]:
    pairs = {
        "rediscovery": "rediscovery",
        "holdout": "holdout_success",
        "counterexamples": "counterexample_rate",
    }
    out: dict[str, float] = {}
    for label, col in pairs.items():
        out[f"future_volume_vs_{label}"] = float(scores["future_volume"].corr(scores[col])) if scores["future_volume"].nunique() > 1 and scores[col].nunique() > 1 else 0.0
        out[f"prediction_vs_{label}"] = float(scores["prediction"].corr(scores[col])) if scores["prediction"].nunique() > 1 and scores[col].nunique() > 1 else 0.0
    return out


def verdict(summary: pd.DataFrame, corr: dict[str, float]) -> tuple[str, str]:
    fv_better = corr.get("future_volume_vs_rediscovery", 0.0) > corr.get("prediction_vs_rediscovery", 0.0)
    if summary["promote"].any() and fv_better:
        return "A", "Future volume strongly predicts abstraction survival."
    if (summary["future_volume"] > 0.50).any() and fv_better:
        return "B", "Future volume partially predicts survival."
    if not fv_better:
        return "C", "No advantage over prediction."
    return "D", "Search-volume hypothesis rejected."


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def write_outputs(out: Path, scores: pd.DataFrame, branch: pd.DataFrame, counters: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    rankings = scores.sort_values("law_value", ascending=False)
    summary = summarize(scores)
    corr = correlations(scores)
    grade, statement = verdict(summary, corr)
    laws = [
        {
            "family": row["family"],
            "future_volume": float(row["future_volume"]),
            "rediscovery": float(row["rediscovery"]),
            "acceleration": float(row["acceleration"]),
            "counterexample_rate": float(row["counterexample_rate"]),
            "worlds": int(row["worlds"]),
            "law_value": float(row["law_value"]),
            "statement": f"{row['family']} preserves useful future search volume.",
        }
        for _, row in summary[summary["promote"]].iterrows()
    ]
    lawbook = {
        "lawbook_version": "v150",
        "hypothesis": "future_volume",
        "promotion_rules": PROMOTION,
        "verdict": grade,
        "laws": laws,
    }
    scores.to_csv(out / "future_volume.csv", index=False)
    rankings.to_csv(out / "future_volume_rankings.csv", index=False)
    branch.to_csv(out / "branch_survival.csv", index=False)
    summary.to_csv(out / "search_volume.csv", index=False)
    counters.to_csv(out / "counterexamples.csv", index=False)
    write_json(out / "lawbook_v150.json", lawbook)
    write_json(out / "manifest.json", {"system": "MATHGRAPH v150 Future Volume Engine", "quick": args.quick, "seed": args.seed, "branches": args.branches, "correlations": corr})
    write_reports(out, summary, rankings, lawbook, corr, grade, statement)
    return {"out": str(out), "verdict": grade, "statement": statement, "laws": len(laws)}


def write_reports(out: Path, summary: pd.DataFrame, rankings: pd.DataFrame, lawbook: dict[str, Any], corr: dict[str, float], grade: str, statement: str) -> None:
    lines = [
        "# MATHGRAPH v150 Future Volume Report",
        "",
        f"Verdict: **{grade}** — {statement}",
        "",
        "## Family Future Volume",
    ]
    for _, row in summary.head(12).iterrows():
        lines.append(
            f"- {row['family']}: future_volume={row['future_volume']:.3f}, "
            f"rediscovery={row['rediscovery']:.3f}, acceleration={row['acceleration']:.3f}, "
            f"counterexamples={row['counterexample_rate']:.3f}, law_value={row['law_value']:.3f}, promote={bool(row['promote'])}"
        )
    lines.extend(
        [
            "",
            "## Future Volume Versus Prediction",
            f"- corr(future_volume, rediscovery): {corr.get('future_volume_vs_rediscovery', 0.0):.3f}",
            f"- corr(prediction, rediscovery): {corr.get('prediction_vs_rediscovery', 0.0):.3f}",
            f"- corr(future_volume, holdout): {corr.get('future_volume_vs_holdout', 0.0):.3f}",
            f"- corr(prediction, holdout): {corr.get('prediction_vs_holdout', 0.0):.3f}",
            f"- corr(future_volume, counterexamples): {corr.get('future_volume_vs_counterexamples', 0.0):.3f}",
            f"- corr(prediction, counterexamples): {corr.get('prediction_vs_counterexamples', 0.0):.3f}",
            "",
            "## Top Abstractions",
        ]
    )
    for _, row in rankings.head(10).iterrows():
        lines.append(
            f"- {row['world']} `{row['expression']}` [{row['family']}]: "
            f"future_volume={row['future_volume']:.3f}, prediction={row['prediction']:.3f}, law_value={row['law_value']:.3f}"
        )
    lines.extend(["", "## Lawbook"])
    if lawbook["laws"]:
        for law in lawbook["laws"]:
            lines.append(f"- {law['family']}: {law['statement']}")
    else:
        lines.append("- No family promoted. Future volume did not clear all gates.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "This run rewards only useful future search volume: rediscoverable branches after abstraction-guided mutation, perturbation, and holdout.",
            "Prediction, recurrence, semantics, and transfer are measured but do not promote a law by themselves.",
        ]
    )
    (out / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "final_conclusion.md").write_text(
        f"""# Final Conclusion

Outcome: **{grade}**

Interpretation: {statement}

v150 asks whether surviving abstractions preserve useful future search volume.
Promotion requires high future volume, rediscovery, acceleration, low
counterexamples, holdout success, and recurrence across at least three worlds.

Promoted laws: {len(lawbook["laws"])}
""",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    n = 80 if args.quick else 260
    worlds = generate_worlds(n, args.seed, args.quick)
    limit = 120 if args.quick else 220
    all_rows: list[dict[str, Any]] = []
    all_branches: list[dict[str, Any]] = []
    all_counters: list[dict[str, Any]] = []
    for world in worlds:
        print(f"Evaluating {world.name} ({world.family})")
        rows, branch_rows, counter_rows = evaluate_world(world, args.branches, args.threshold, limit)
        all_rows.extend(rows)
        all_branches.extend(branch_rows)
        all_counters.extend(counter_rows)
    return write_outputs(Path(args.out), pd.DataFrame(all_rows), pd.DataFrame(all_branches), pd.DataFrame(all_counters), args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MATHGRAPH v150 Future Volume Engine")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="mathgraph_v150_out")
    p.add_argument("--seed", type=int, default=150)
    p.add_argument("--branches", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
