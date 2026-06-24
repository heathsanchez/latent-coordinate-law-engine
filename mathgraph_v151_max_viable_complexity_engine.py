#!/usr/bin/env python3
"""MATHGRAPH v151: Max Viable Complexity Engine.

The experiment asks whether good abstractions let search climb to higher
complexity before viability collapses. Raw accuracy is diagnostic only; the
main score is the highest complexity level that remains viable under
rediscovery, holdout, perturbation, and counterexample pressure.
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
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


PROMOTION = {
    "mvc_level_gain": 2,
    "holdout_success": 0.60,
    "perturbation_survival": 0.60,
    "counterexample_rate": 0.25,
    "rediscovery": 0.60,
    "minimum_worlds": 3,
}

AGENTS = ["blind", "prediction_guided", "future_volume_guided", "mvc_guided"]
FAMILIES = [
    "CAPACITY_LOAD",
    "SEARCH_BOTTLENECK",
    "RESOURCE_CONSTRAINT",
    "SIGNAL_NOISE",
    "ENERGY_DISSIPATION",
    "CONTROL_PARAMETER",
    "AVAILABLE_CAPACITY",
    "UNKNOWN",
]

DENOMINATOR_ROLES = {"moles", "load", "noise", "dissipation", "temperature", "bottleneck", "constraint", "n"}


@dataclass
class World:
    name: str
    family: str
    roles: list[str]
    formula: Callable[[dict[str, np.ndarray], int, np.random.Generator], np.ndarray]


@dataclass
class Task:
    world: str
    hidden_family: str
    level: int
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
    complexity: int


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def random_name(rng: np.random.Generator, used: set[str]) -> str:
    while True:
        name = "x" + "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=5))
        if name not in used:
            used.add(name)
            return name


def transform(values: np.ndarray, rng: np.random.Generator, level: int) -> np.ndarray:
    sign = float(rng.choice([1.0, 1.0, 1.0, -1.0]))
    scale = float(10 ** rng.uniform(-1.0, 1.0))
    offset = float(rng.uniform(-2.0, 2.0) * (1 + 0.1 * level))
    out = sign * values * scale + offset
    if out.min() <= 0:
        out = out - out.min() + rng.uniform(0.25, 1.5)
    return out


def worlds() -> list[World]:
    return [
        World("GAS", "RESOURCE_CONSTRAINT", ["pressure", "volume", "moles"], lambda v, l, r: (v["pressure"] * v["volume"]) / (v["moles"] + 0.15 * l)),
        World("PHASE", "CONTROL_PARAMETER", ["control", "coupling", "temperature"], lambda v, l, r: (v["control"] * v["coupling"]) / (v["temperature"] + 0.2 * l)),
        World("CAPACITY", "CAPACITY_LOAD", ["capacity", "load"], lambda v, l, r: v["capacity"] / (v["load"] + 0.1 * l)),
        World("SIGNAL", "SIGNAL_NOISE", ["signal", "noise"], lambda v, l, r: v["signal"] / (v["noise"] + 0.15 * l)),
        World("ENERGY", "ENERGY_DISSIPATION", ["energy", "dissipation"], lambda v, l, r: v["energy"] / (v["dissipation"] + 0.2 * l)),
        World("MAZE", "SEARCH_BOTTLENECK", ["search_budget", "bottleneck"], lambda v, l, r: v["search_budget"] / (v["bottleneck"] + 0.2 * l)),
        World("OBSTRUCTION", "AVAILABLE_CAPACITY", ["clearance", "width", "load"], lambda v, l, r: (v["clearance"] * v["width"]) / (v["load"] + 0.2 * l)),
        World("MODULAR", "UNKNOWN", ["x", "n", "offset"], lambda v, l, r: ((v["x"] + v["offset"]) % (v["n"] + 1.0)) / (v["n"] + 1.0)),
        World("ARC_STYLE", "UNKNOWN", ["objects", "symmetry", "constraint"], lambda v, l, r: (v["objects"] + v["symmetry"]) / (v["constraint"] + 0.2 * l)),
    ]


def build_task(world: World, level: int, n: int, seed: int) -> Task:
    rng = np.random.default_rng(seed + 100 * level + sum(ord(c) for c in world.name))
    used: set[str] = set()
    raw: dict[str, np.ndarray] = {}
    df = pd.DataFrame()
    role_map: dict[str, str] = {}
    for role in world.roles:
        low, high = (0.8, 12.0) if role in DENOMINATOR_ROLES else (0.5, 22.0)
        raw[role] = rng.uniform(low, high, n)
        col = random_name(rng, used)
        df[col] = transform(raw[role], rng, level)
        role_map[col] = role
    distractors = 2 + level
    for i in range(distractors):
        col = random_name(rng, used)
        values = rng.uniform(0.3, 28.0 + 3 * level, n)
        if level >= 6 and i % 2 == 0:
            values = values + 0.4 * next(iter(raw.values()))
        df[col] = transform(values, rng, level)
        role_map[col] = "distractor"
    latent = world.formula(raw, level, rng)
    if level >= 3:
        latent = latent / (1.0 + 0.025 * level * raw[world.roles[-1]])
    if level >= 4:
        latent = latent + 0.08 * safe_np(raw[world.roles[0]], raw[world.roles[-1]] + raw[world.roles[1 % len(world.roles)]])
    if level >= 5:
        latent = latent + rng.normal(0, 0.10 * float(np.std(latent) or 1.0), n)
    noise = (0.02 + 0.018 * level) * float(np.std(latent) or 1.0)
    df["outcome"] = latent + rng.normal(0, noise, n)
    return Task(world.name, world.family, level, df, "outcome", role_map)


def safe_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / np.where(np.abs(b) < 1e-9, 1e-9, b)


def generate_candidates(task: Task, limit: int) -> list[Candidate]:
    cols = [c for c in task.df.columns if c != task.target]
    df = task.df
    out: list[Candidate] = []
    seen: set[str] = set()

    def add(name: str, expression: str, operator: str, columns: list[str], values: pd.Series, complexity: int) -> None:
        if name in seen or len(out) >= limit:
            return
        seen.add(name)
        values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if values.nunique(dropna=False) > 1:
            out.append(Candidate(name, expression, operator, columns, values, complexity))

    for x in cols:
        for y in cols:
            if x == y:
                continue
            add(f"{x}_over_{y}", f"{x}/{y}", "ratio", [x, y], safe_div(df[x], df[y]), 1)
            add(f"{x}_plus_{y}", f"{x}+{y}", "sum", [x, y], df[x] + df[y], 1)
            add(f"{x}_minus_{y}", f"{x}-{y}", "difference", [x, y], df[x] - df[y], 1)
            add(f"{x}_times_{y}", f"{x}*{y}", "product", [x, y], df[x] * df[y], 2)
            add(f"min_{x}_{y}", f"min({x},{y})", "min", [x, y], pd.Series(np.minimum(df[x], df[y]), index=df.index), 2)
            add(f"max_{x}_{y}", f"max({x},{y})", "max", [x, y], pd.Series(np.maximum(df[x], df[y]), index=df.index), 2)
    for x in cols:
        for y in cols:
            for z in cols:
                if len(out) >= limit:
                    return out
                if len({x, y, z}) != 3:
                    continue
                add(f"{x}_times_{y}_over_{z}", f"({x}*{y})/{z}", "product_ratio", [x, y, z], safe_div(df[x] * df[y], df[z]), 3)
                add(f"{x}_plus_{y}_over_{z}", f"({x}+{y})/{z}", "sum_ratio", [x, y, z], safe_div(df[x] + df[y], df[z]), 3)
                add(f"{x}_over_{y}_plus_{z}", f"{x}/({y}+{z})", "ratio_sum", [x, y, z], safe_div(df[x], df[y] + df[z]), 3)
    return out


def score(values: pd.Series, target: pd.Series, seed: int) -> float:
    train_idx, test_idx = train_test_split(values.index, test_size=0.30, random_state=seed)
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=seed)
    model.fit(pd.DataFrame({"c": values.loc[train_idx]}), target.loc[train_idx])
    pred = model.predict(pd.DataFrame({"c": values.loc[test_idx]}))
    return float(max(-1.0, r2_score(target.loc[test_idx], pred)))


def counterexample_rate(values: pd.Series, target: pd.Series, seed: int) -> float:
    train_idx, test_idx = train_test_split(values.index, test_size=0.30, random_state=seed)
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=seed)
    model.fit(pd.DataFrame({"c": values.loc[train_idx]}), target.loc[train_idx])
    pred = model.predict(pd.DataFrame({"c": values.loc[test_idx]}))
    residual = np.abs(pred - np.asarray(target.loc[test_idx]))
    scale = float(np.std(target.loc[test_idx]) or 1.0)
    return float(np.mean(residual > 0.55 * scale))


def perturbation_survival(candidate: Candidate, task: Task, base_score: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    df = task.df.copy()
    for col in candidate.columns:
        df[col] = df[col] + rng.normal(0, 0.05 * float(df[col].std() or 1.0), len(df))
    rebuilt = rebuild(candidate, df)
    perturbed = score(rebuilt, df[task.target], seed + 11)
    return float(np.clip((perturbed + 1.0) / max(base_score + 1.0, 1e-9), 0.0, 1.25))


def rebuild(candidate: Candidate, df: pd.DataFrame) -> pd.Series:
    c = candidate.columns
    if candidate.operator == "ratio":
        return safe_div(df[c[0]], df[c[1]])
    if candidate.operator == "sum":
        return df[c[0]] + df[c[1]]
    if candidate.operator == "difference":
        return df[c[0]] - df[c[1]]
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
    if candidate.operator == "ratio_sum":
        return safe_div(df[c[0]], df[c[1]] + df[c[2]])
    raise ValueError(candidate.operator)


def infer_family(candidate: Candidate, task: Task) -> str:
    roles = set(task.role_map.get(c, "distractor") for c in candidate.columns)
    op = candidate.operator
    if {"capacity", "load"}.issubset(roles) and op in {"ratio", "product_ratio", "sum_ratio", "ratio_sum"}:
        return "CAPACITY_LOAD"
    if {"search_budget", "bottleneck"}.issubset(roles) and op in {"ratio", "sum_ratio", "ratio_sum"}:
        return "SEARCH_BOTTLENECK"
    if {"pressure", "volume", "moles"}.issubset(roles) and op in {"product_ratio", "ratio_sum"}:
        return "RESOURCE_CONSTRAINT"
    if {"signal", "noise"}.issubset(roles) and op in {"ratio", "ratio_sum"}:
        return "SIGNAL_NOISE"
    if {"energy", "dissipation"}.issubset(roles) and op in {"ratio", "product_ratio"}:
        return "ENERGY_DISSIPATION"
    if {"control", "coupling", "temperature"}.issubset(roles) and op in {"product_ratio", "ratio_sum"}:
        return "CONTROL_PARAMETER"
    if {"clearance", "width", "load"}.issubset(roles) and op in {"product_ratio", "ratio_sum"}:
        return "AVAILABLE_CAPACITY"
    return "UNKNOWN"


def future_volume(candidate: Candidate, candidates: list[Candidate], score_map: dict[str, float], threshold: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    roles = set(candidate.columns)
    same = [c for c in candidates if c.operator == candidate.operator or bool(set(c.columns) & roles)]
    pool = same or candidates
    sample = rng.choice(pool, size=60, replace=True)
    return float(np.mean([score_map[c.name] + rng.normal(0, 0.02) >= threshold for c in sample]))


def order_candidates(agent: str, candidates: list[Candidate], score_map: dict[str, float], future_map: dict[str, float]) -> list[Candidate]:
    if agent == "blind":
        return candidates
    if agent == "prediction_guided":
        return sorted(candidates, key=lambda c: score_map[c.name], reverse=True)
    if agent == "future_volume_guided":
        return sorted(candidates, key=lambda c: future_map[c.name], reverse=True)
    return sorted(
        candidates,
        key=lambda c: (
            future_map[c.name] * max(score_map[c.name], 0.0) * (1.0 + 0.08 * c.complexity),
            c.complexity,
        ),
        reverse=True,
    )


def evaluate_agent(task: Task, agent: str, candidates: list[Candidate], score_map: dict[str, float], threshold: float) -> dict[str, Any]:
    future_map = {c.name: future_volume(c, candidates, score_map, threshold, 9100 + i) for i, c in enumerate(candidates)}
    ordered = order_candidates(agent, candidates, score_map, future_map)
    best = ordered[0]
    for cand in ordered:
        if score_map[cand.name] >= threshold:
            best = cand
            break
    search_cost = ordered.index(best) + 1
    pred = score_map[best.name]
    ce = counterexample_rate(best.values, task.df[task.target], 5200 + task.level)
    holdout_success = float(pred >= threshold)
    perturb = perturbation_survival(best, task, pred, 6100 + task.level)
    family = infer_family(best, task)
    rediscovery = float(family == task.hidden_family and pred >= threshold)
    viable = (
        holdout_success >= 0.6
        and perturb >= 0.6
        and ce <= 0.25
        and rediscovery >= 0.6
    )
    viable_branches = int(round(future_map[best.name] * 60))
    mvc_score = (
        task.level
        * holdout_success
        * perturb
        * (1.0 - ce)
        * rediscovery
        / max(search_cost, 1)
    )
    return {
        "world": task.world,
        "hidden_family": task.hidden_family,
        "level": task.level,
        "agent": agent,
        "candidate": best.name,
        "expression": best.expression,
        "candidate_family": family,
        "prediction": pred,
        "holdout_success": holdout_success,
        "perturbation_survival": perturb,
        "counterexample_rate": ce,
        "counterexample_resistance": 1.0 - ce,
        "rediscovery": rediscovery,
        "search_cost": search_cost,
        "viable_branch_count": viable_branches,
        "future_volume": future_map[best.name],
        "candidate_complexity": best.complexity,
        "viable": viable,
        "mvc_score": mvc_score,
    }


def evaluate_world(world: World, n: int, seed: int, threshold: float, candidate_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in range(1, 7):
        task = build_task(world, level, n, seed)
        candidates = generate_candidates(task, candidate_limit)
        score_map = {c.name: score(c.values, task.df[task.target], 1500 + 17 * level + i) for i, c in enumerate(candidates)}
        for agent in AGENTS:
            rows.append(evaluate_agent(task, agent, candidates, score_map, threshold))
    return rows


def summarize_ladder(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ladder = rows.copy()
    max_rows = []
    collapse_rows = []
    for (world, family, agent), group in rows.groupby(["world", "hidden_family", "agent"]):
        viable_levels = sorted(group[group["viable"]]["level"].tolist())
        max_level = max(viable_levels) if viable_levels else 0
        collapse = next((level for level in range(1, 7) if level not in viable_levels), 7)
        max_rows.append(
            {
                "world": world,
                "hidden_family": family,
                "agent": agent,
                "max_complexity_solved": max_level,
                "collapse_point": collapse,
                "mean_mvc_score": group["mvc_score"].mean(),
                "mean_holdout_success": group["holdout_success"].mean(),
                "mean_perturbation_survival": group["perturbation_survival"].mean(),
                "mean_counterexample_rate": group["counterexample_rate"].mean(),
                "mean_rediscovery": group["rediscovery"].mean(),
                "mean_search_cost": group["search_cost"].mean(),
                "mean_viable_branch_count": group["viable_branch_count"].mean(),
            }
        )
        collapse_rows.append(
            {
                "world": world,
                "hidden_family": family,
                "agent": agent,
                "collapse_point": collapse,
                "max_complexity_solved": max_level,
                "collapse_reason": "no_viable_level" if max_level == 0 else "next_level_failed",
            }
        )
    scores = pd.DataFrame(max_rows)
    collapse = pd.DataFrame(collapse_rows)
    family = (
        scores.groupby(["hidden_family", "agent"])
        .agg(
            worlds=("world", "nunique"),
            max_complexity_solved=("max_complexity_solved", "mean"),
            collapse_point=("collapse_point", "mean"),
            mvc_score=("mean_mvc_score", "mean"),
            holdout_success=("mean_holdout_success", "mean"),
            perturbation_survival=("mean_perturbation_survival", "mean"),
            counterexample_rate=("mean_counterexample_rate", "mean"),
            rediscovery=("mean_rediscovery", "mean"),
            search_cost=("mean_search_cost", "mean"),
            viable_branch_count=("mean_viable_branch_count", "mean"),
        )
        .reset_index()
    )
    blind = family[family["agent"] == "blind"][["hidden_family", "max_complexity_solved"]].rename(columns={"max_complexity_solved": "blind_complexity"})
    family = family.merge(blind, on="hidden_family", how="left")
    family["mvc_beats_blind_by_levels"] = family["max_complexity_solved"] - family["blind_complexity"].fillna(0)
    family["promote"] = (
        (family["agent"] == "mvc_guided")
        & (family["mvc_beats_blind_by_levels"] >= PROMOTION["mvc_level_gain"])
        & (family["holdout_success"] >= PROMOTION["holdout_success"])
        & (family["perturbation_survival"] >= PROMOTION["perturbation_survival"])
        & (family["counterexample_rate"] <= PROMOTION["counterexample_rate"])
        & (family["rediscovery"] >= PROMOTION["rediscovery"])
        & (family["worlds"] >= PROMOTION["minimum_worlds"])
    )
    rankings = family.sort_values(["promote", "mvc_score", "max_complexity_solved"], ascending=[False, False, False])
    return ladder, scores, collapse, rankings


def verdict(rankings: pd.DataFrame) -> tuple[str, str]:
    mvc = rankings[rankings["agent"] == "mvc_guided"]
    pred = rankings[rankings["agent"] == "prediction_guided"]
    future = rankings[rankings["agent"] == "future_volume_guided"]
    if rankings["promote"].any():
        return "A", "MVC-guided abstractions reach much higher viable complexity."
    mvc_mean = mvc["max_complexity_solved"].mean() if not mvc.empty else 0.0
    other_mean = pd.concat([pred["max_complexity_solved"], future["max_complexity_solved"]]).mean() if not pred.empty or not future.empty else 0.0
    if mvc_mean > other_mean:
        return "B", "MVC helps partially."
    if mvc_mean == other_mean:
        return "C", "MVC no better than prediction/future volume."
    return "D", "hypothesis rejected."


def write_outputs(out: Path, ladder: pd.DataFrame, scores: pd.DataFrame, collapse: pd.DataFrame, rankings: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    grade, statement = verdict(rankings)
    counterexamples = ladder[(ladder["counterexample_rate"] > 0.25) | (~ladder["viable"])].copy()
    laws = [
        {
            "family": row["hidden_family"],
            "mvc_score": float(row["mvc_score"]),
            "max_complexity_solved": float(row["max_complexity_solved"]),
            "beats_blind_by_levels": float(row["mvc_beats_blind_by_levels"]),
            "worlds": int(row["worlds"]),
            "statement": f"{row['hidden_family']} preserves max viable complexity under constraint.",
        }
        for _, row in rankings[rankings["promote"]].iterrows()
    ]
    lawbook = {"lawbook_version": "v151", "hypothesis": "max_viable_complexity", "promotion_rules": PROMOTION, "verdict": grade, "laws": laws}
    scores.to_csv(out / "mvc_scores.csv", index=False)
    ladder.to_csv(out / "complexity_ladder.csv", index=False)
    collapse.to_csv(out / "collapse_points.csv", index=False)
    rankings.to_csv(out / "viable_complexity_rankings.csv", index=False)
    counterexamples.to_csv(out / "counterexamples.csv", index=False)
    write_json(out / "lawbook_v151.json", lawbook)
    write_json(out / "manifest.json", {"system": "MATHGRAPH v151 Max Viable Complexity Engine", "seed": args.seed, "quick": args.quick, "verdict": grade})
    write_reports(out, ladder, rankings, lawbook, grade, statement)
    return {"out": str(out), "verdict": grade, "statement": statement, "laws": len(laws)}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def write_reports(out: Path, ladder: pd.DataFrame, rankings: pd.DataFrame, lawbook: dict[str, Any], grade: str, statement: str) -> None:
    lines = [
        "# MATHGRAPH v151 Max Viable Complexity Report",
        "",
        f"Verdict: **{grade}** — {statement}",
        "",
        "## Which Abstractions Allow Complexity To Keep Increasing?",
    ]
    for _, row in rankings.head(14).iterrows():
        lines.append(
            f"- {row['hidden_family']} / {row['agent']}: max_complexity={row['max_complexity_solved']:.2f}, "
            f"collapse={row['collapse_point']:.2f}, mvc_score={row['mvc_score']:.4f}, "
            f"beats_blind={row['mvc_beats_blind_by_levels']:.2f}, promote={bool(row['promote'])}"
        )
    lines.extend(["", "## Where Does Each Abstraction Collapse?"])
    for _, row in rankings[rankings["agent"] == "mvc_guided"].head(12).iterrows():
        lines.append(f"- {row['hidden_family']}: collapse point {row['collapse_point']:.2f}, max viable complexity {row['max_complexity_solved']:.2f}")
    mvc_mean = rankings[rankings["agent"] == "mvc_guided"]["max_complexity_solved"].mean()
    pred_mean = rankings[rankings["agent"] == "prediction_guided"]["max_complexity_solved"].mean()
    future_mean = rankings[rankings["agent"] == "future_volume_guided"]["max_complexity_solved"].mean()
    lines.extend(
        [
            "",
            "## Does MVC Explain Survival Better?",
            f"- MVC-guided mean max complexity: {mvc_mean:.3f}",
            f"- Prediction-guided mean max complexity: {pred_mean:.3f}",
            f"- Future-volume-guided mean max complexity: {future_mean:.3f}",
            "",
            "## Lawbook",
        ]
    )
    if lawbook["laws"]:
        for law in lawbook["laws"]:
            lines.append(f"- {law['family']}: {law['statement']}")
    else:
        lines.append("- No abstraction promoted under the MVC gates.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "The test does not reward raw accuracy or branch count alone. A level counts only when rediscovery, holdout, perturbation survival, and counterexample resistance remain viable.",
            "The claim `truth = maximum viable complexity under constraint` is supported only if MVC-guided search reaches higher viable complexity than blind, prediction-guided, and future-volume-guided search under these gates.",
        ]
    )
    (out / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "final_conclusion.md").write_text(
        f"""# Final Conclusion

Outcome: **{grade}**

Interpretation: {statement}

v151 tests whether successful abstractions preserve maximum viable complexity.
Complexity only counts when rediscovery, holdout success, perturbation survival,
and counterexample resistance all remain viable.

Promoted laws: {len(lawbook["laws"])}
""",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    n = 80 if args.quick else 240
    limit = 100 if args.quick else 220
    selected = worlds()[:8] if args.quick else worlds()
    rows: list[dict[str, Any]] = []
    for world in selected:
        print(f"Evaluating {world.name} ({world.family})")
        rows.extend(evaluate_world(world, n, args.seed, args.threshold, limit))
    ladder, scores, collapse, rankings = summarize_ladder(pd.DataFrame(rows))
    return write_outputs(Path(args.out), ladder, scores, collapse, rankings, args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MATHGRAPH v151 Max Viable Complexity Engine")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="mathgraph_v151_out")
    p.add_argument("--seed", type=int, default=151)
    p.add_argument("--threshold", type=float, default=0.36)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
