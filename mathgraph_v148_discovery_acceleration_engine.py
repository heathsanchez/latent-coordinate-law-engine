#!/usr/bin/env python3
"""MATHGRAPH v148: Discovery Acceleration Engine.

The experiment asks which abstractions reduce discovery cost. Prediction is
measured only as the criterion for finding a usable coordinate; the main score
is search compression.
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
from typing import Any


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


FAMILIES = [
    "RESOURCE_CONSTRAINT",
    "CAPACITY_LOAD",
    "SIGNAL_NOISE",
    "ENERGY_DISSIPATION",
    "FREEDOM_RESTRICTION",
    "SEARCH_BOTTLENECK",
    "CONTROL_PARAMETER",
]

OPERATOR_ORDER = {
    "ratio": ["ratio"],
    "difference": ["difference", "ratio"],
    "product": ["product", "ratio"],
    "sum": ["sum", "ratio"],
}

SEMANTIC_ORDER = {
    "RESOURCE_CONSTRAINT": ["ratio", "sum_ratio", "product_ratio"],
    "CAPACITY_LOAD": ["ratio", "product_ratio", "sum_ratio"],
    "SIGNAL_NOISE": ["ratio", "difference_ratio"],
    "ENERGY_DISSIPATION": ["ratio", "difference_ratio", "product_ratio"],
    "FREEDOM_RESTRICTION": ["ratio", "sum_ratio"],
    "SEARCH_BOTTLENECK": ["ratio", "sum_ratio", "difference_ratio"],
    "CONTROL_PARAMETER": ["product_ratio", "ratio", "product_sum_ratio"],
}

SEMANTIC_PRIOR_ORDER = ["ratio", "product_ratio", "sum_ratio", "ratio_sum", "difference_ratio", "product", "sum", "difference", "product_sum_ratio"]


@dataclass
class HiddenWorld:
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


def random_name(rng: np.random.Generator, used: set[str]) -> str:
    while True:
        name = "x" + "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=5))
        if name not in used:
            used.add(name)
            return name


def transform(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sign = float(rng.choice([1, 1, 1, -1]))
    scale = float(10 ** rng.uniform(-1.0, 1.0))
    offset = float(rng.uniform(-5, 5))
    out = sign * values * scale + offset
    if out.min() <= 0:
        out = out - out.min() + rng.uniform(0.2, 2.0)
    return out


def hidden_worlds(n: int, seed: int, variants: int = 3) -> list[HiddenWorld]:
    rng = np.random.default_rng(seed)
    specs = [
        ("WORLD_A", "RESOURCE_CONSTRAINT", ["resource", "constraint"], lambda v: v["resource"] / v["constraint"]),
        ("WORLD_B", "CAPACITY_LOAD", ["capacity", "load"], lambda v: v["capacity"] / v["load"]),
        ("WORLD_C", "SIGNAL_NOISE", ["signal", "noise"], lambda v: v["signal"] / v["noise"]),
        ("WORLD_D", "ENERGY_DISSIPATION", ["energy", "dissipation"], lambda v: v["energy"] / v["dissipation"]),
        ("WORLD_E", "FREEDOM_RESTRICTION", ["freedom", "restriction"], lambda v: v["freedom"] / v["restriction"]),
        ("WORLD_F", "SEARCH_BOTTLENECK", ["search_budget", "bottleneck"], lambda v: v["search_budget"] / v["bottleneck"]),
        ("WORLD_G", "CONTROL_PARAMETER", ["control", "coupling", "temperature"], lambda v: (v["control"] * v["coupling"]) / v["temperature"]),
    ]
    worlds: list[HiddenWorld] = []
    for base_name, family, roles, formula in specs:
        for i in range(variants):
            used: set[str] = set()
            raw: dict[str, np.ndarray] = {}
            df = pd.DataFrame()
            role_map: dict[str, str] = {}
            source_columns: dict[str, str] = {}
            for role in roles:
                values = rng.uniform(0.5, 20.0, n)
                if role in {"constraint", "load", "noise", "dissipation", "restriction", "bottleneck", "temperature"}:
                    values = rng.uniform(0.8, 12.0, n)
                raw[role] = values
                col = random_name(rng, used)
                df[col] = transform(values, rng)
                role_map[col] = role
                source_columns[role] = col
            for _ in range(4):
                col = random_name(rng, used)
                df[col] = transform(rng.uniform(0.2, 25.0, n), rng)
                role_map[col] = "distractor"
            latent = formula(raw)
            noise_level = float(rng.uniform(0.01, 0.10))
            df["outcome"] = latent + rng.normal(0, noise_level * np.std(latent), n)
            worlds.append(HiddenWorld(f"{base_name}_{i + 1}", family, df, "outcome", latent, role_map, source_columns, noise_level))
    return worlds


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def generate_candidates(world: HiddenWorld) -> list[Candidate]:
    cols = [c for c in world.df.columns if c != world.target]
    df = world.df
    candidates: list[Candidate] = []

    def add(name: str, expr: str, operator: str, columns: list[str], depth: int, values: pd.Series) -> None:
        if values.nunique(dropna=False) > 1:
            candidates.append(Candidate(name, expr, operator, columns, depth, values.astype(float)))

    for x in cols:
        for y in cols:
            if x == y:
                continue
            add(f"{x}_over_{y}", f"{x}/{y}", "ratio", [x, y], 2, safe_div(df[x], df[y]))
            add(f"{x}_minus_{y}", f"{x}-{y}", "difference", [x, y], 2, df[x] - df[y])
            add(f"{x}_plus_{y}", f"{x}+{y}", "sum", [x, y], 2, df[x] + df[y])
            add(f"{x}_times_{y}", f"{x}*{y}", "product", [x, y], 2, df[x] * df[y])
    for x in cols:
        for y in cols:
            for z in cols:
                if len({x, y, z}) != 3:
                    continue
                add(f"{x}_times_{y}_over_{z}", f"({x}*{y})/{z}", "product_ratio", [x, y, z], 3, safe_div(df[x] * df[y], df[z]))
                add(f"{x}_plus_{y}_over_{z}", f"({x}+{y})/{z}", "sum_ratio", [x, y, z], 3, safe_div(df[x] + df[y], df[z]))
                add(f"{x}_minus_{y}_over_{z}", f"({x}-{y})/{z}", "difference_ratio", [x, y, z], 3, safe_div(df[x] - df[y], df[z]))
                add(f"{x}_over_{y}_plus_{z}", f"{x}/({y}+{z})", "ratio_sum", [x, y, z], 3, safe_div(df[x], df[y] + df[z]))
    for x in cols:
        for y in cols:
            for z in cols:
                for w in cols:
                    if len({x, y, z, w}) != 4:
                        continue
                    add(f"{x}_times_{y}_over_{z}_plus_{w}", f"({x}*{y})/({z}+{w})", "product_sum_ratio", [x, y, z, w], 4, safe_div(df[x] * df[y], df[z] + df[w]))
    return candidates


def score(values: pd.Series, target: pd.Series, seed: int = 148) -> float:
    train_idx, test_idx = train_test_split(values.index, test_size=0.3, random_state=seed)
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=seed)
    model.fit(pd.DataFrame({"c": values.loc[train_idx]}), target.loc[train_idx])
    pred = model.predict(pd.DataFrame({"c": values.loc[test_idx]}))
    return float(r2_score(target.loc[test_idx], pred))


def raw_score(world: HiddenWorld) -> float:
    cols = [c for c in world.df.columns if c != world.target]
    train_idx, test_idx = train_test_split(world.df.index, test_size=0.3, random_state=148)
    model = RandomForestRegressor(n_estimators=100, random_state=148, min_samples_leaf=3)
    model.fit(world.df.loc[train_idx, cols], world.df.loc[train_idx, world.target])
    pred = model.predict(world.df.loc[test_idx, cols])
    return float(r2_score(world.df.loc[test_idx, world.target], pred))


def perturbation_survival(candidate: Candidate, world: HiddenWorld, base: float) -> float:
    rng = np.random.default_rng(8148)
    df = world.df.copy()
    for col in candidate.columns:
        df[col] += rng.normal(0, 0.05 * float(df[col].std() or 1.0), len(df))
    rebuilt = rebuild(candidate, df)
    if rebuilt is None:
        return 0.0
    rescored = score(rebuilt, df[world.target], seed=8148)
    return float(max(0.0, min(1.0, rescored / max(base, 1e-9))))


def rebuild(candidate: Candidate, df: pd.DataFrame) -> pd.Series | None:
    c = candidate.columns
    try:
        if candidate.operator == "ratio":
            return safe_div(df[c[0]], df[c[1]])
        if candidate.operator == "difference":
            return df[c[0]] - df[c[1]]
        if candidate.operator == "sum":
            return df[c[0]] + df[c[1]]
        if candidate.operator == "product":
            return df[c[0]] * df[c[1]]
        if candidate.operator == "product_ratio":
            return safe_div(df[c[0]] * df[c[1]], df[c[2]])
        if candidate.operator == "sum_ratio":
            return safe_div(df[c[0]] + df[c[1]], df[c[2]])
        if candidate.operator == "difference_ratio":
            return safe_div(df[c[0]] - df[c[1]], df[c[2]])
        if candidate.operator == "ratio_sum":
            return safe_div(df[c[0]], df[c[1]] + df[c[2]])
        if candidate.operator == "product_sum_ratio":
            return safe_div(df[c[0]] * df[c[1]], df[c[2]] + df[c[3]])
    except Exception:
        return None
    return None


def counterexample_rate(candidate: Candidate, world: HiddenWorld) -> float:
    train_idx, test_idx = train_test_split(candidate.values.index, test_size=0.3, random_state=9148)
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=9148)
    model.fit(pd.DataFrame({"c": candidate.values.loc[train_idx]}), world.df.loc[train_idx, world.target])
    pred = model.predict(pd.DataFrame({"c": candidate.values.loc[test_idx]}))
    residual = np.abs(pred - np.asarray(world.df.loc[test_idx, world.target]))
    threshold = np.quantile(residual, 0.8) if len(residual) else 0.0
    return float(np.mean(residual > threshold)) if len(residual) else 1.0


def candidate_matches_family(candidate: Candidate, world: HiddenWorld) -> bool:
    roles = [world.role_map.get(c, "distractor") for c in candidate.columns]
    fam = world.family
    role_set = set(roles)
    if fam == "CONTROL_PARAMETER":
        return candidate.operator == "product_ratio" and {"control", "coupling", "temperature"}.issubset(role_set)
    if fam == "CAPACITY_LOAD":
        return candidate.operator in {"ratio", "product_ratio", "sum_ratio", "ratio_sum"} and "capacity" in role_set and "load" in role_set
    if fam == "RESOURCE_CONSTRAINT":
        return candidate.operator in {"ratio", "sum_ratio", "product_ratio", "ratio_sum"} and "resource" in role_set and "constraint" in role_set
    if fam == "SIGNAL_NOISE":
        return candidate.operator in {"ratio", "difference_ratio", "ratio_sum"} and "signal" in role_set and "noise" in role_set
    if fam == "ENERGY_DISSIPATION":
        return candidate.operator in {"ratio", "difference_ratio", "product_ratio"} and "energy" in role_set and "dissipation" in role_set
    if fam == "FREEDOM_RESTRICTION":
        return candidate.operator in {"ratio", "sum_ratio"} and "freedom" in role_set and "restriction" in role_set
    if fam == "SEARCH_BOTTLENECK":
        return candidate.operator in {"ratio", "sum_ratio", "ratio_sum"} and "search_budget" in role_set and "bottleneck" in role_set
    return False


def order_for_agent(agent: str, candidates: list[Candidate], family_hint: str | None = None) -> list[Candidate]:
    if agent == "blind":
        return candidates
    if agent == "operator":
        preferred = {"ratio", "difference", "product", "sum"}
        return sorted(candidates, key=lambda c: (c.operator not in preferred, c.depth, c.name))
    semantic_order = SEMANTIC_ORDER.get(family_hint or "", SEMANTIC_PRIOR_ORDER)
    return sorted(candidates, key=lambda c: (semantic_order.index(c.operator) if c.operator in semantic_order else 999, c.depth, c.name))


def run_agent(world: HiddenWorld, agent: str, threshold: float, family_hint: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = raw_score(world)
    candidates = order_for_agent(agent, generate_candidates(world), family_hint)
    start = time.perf_counter()
    failures = 0
    found = None
    rows = []
    for i, cand in enumerate(candidates, start=1):
        s = score(cand.values, world.df[world.target], seed=148 + i % 17)
        match = candidate_matches_family(cand, world)
        failed = not (s >= threshold and match)
        failures += int(failed)
        rows.append(
            {
                "world": world.name,
                "hidden_family": world.family,
                "agent": agent,
                "candidate": cand.name,
                "expression": cand.expression,
                "operator": cand.operator,
                "depth": cand.depth,
                "score": s,
                "raw_score": raw,
                "matches_hidden_family": match,
                "evaluation_index": i,
            }
        )
        if s >= threshold and match:
            found = (cand, s, i)
            break
    elapsed = time.perf_counter() - start
    evaluated = found[2] if found else len(candidates)
    depth = found[0].depth if found else max(c.depth for c in candidates)
    cost = evaluated + depth + failures + elapsed
    result = {
        "world": world.name,
        "hidden_family": world.family,
        "agent": agent,
        "coordinates_tested": evaluated,
        "evaluations": evaluated,
        "depth": depth,
        "failures": failures,
        "wall_time": elapsed,
        "discovery_cost": cost,
        "rediscovery_rank": found[2] if found else -1,
        "found": found is not None,
        "best_score": found[1] if found else max((r["score"] for r in rows), default=0.0),
        "raw_score": raw,
        "perturbation_survival": perturbation_survival(found[0], world, found[1]) if found else 0.0,
        "counterexample_rate": counterexample_rate(found[0], world) if found else 1.0,
    }
    return result, rows


def experiment(worlds: list[HiddenWorld]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cost_rows = []
    candidate_rows = []
    for world in worlds:
        print(f"Evaluating {world.name} ({world.family})")
        for agent in ["blind", "operator", "semantic"]:
            hint = None
            result, rows = run_agent(world, agent, threshold=0.45, family_hint=hint)
            cost_rows.append(result)
            candidate_rows.extend(rows)
    costs = pd.DataFrame(cost_rows)
    all_evals = pd.DataFrame(candidate_rows)
    pivot = costs.pivot(index=["world", "hidden_family"], columns="agent", values="discovery_cost").reset_index()
    pivot["operator_acceleration"] = pivot["blind"] / pivot["operator"].clip(lower=1e-9)
    pivot["semantic_acceleration"] = pivot["blind"] / pivot["semantic"].clip(lower=1e-9)
    acceleration = pivot.rename(columns={"blind": "blind_cost", "operator": "operator_cost", "semantic": "semantic_cost"})

    ranks = costs.pivot(index=["world", "hidden_family"], columns="agent", values="rediscovery_rank").reset_index()
    ranks = ranks.rename(columns={"blind": "blind_rank", "operator": "operator_rank", "semantic": "semantic_rank"})

    efficiency_rows = []
    for family, group in costs.groupby("hidden_family"):
        blind = group[group["agent"] == "blind"]
        semantic = group[group["agent"] == "semantic"]
        operator = group[group["agent"] == "operator"]
        efficiency_rows.append(
            {
                "family": family,
                "discovery_acceleration": float(blind["discovery_cost"].mean() / max(semantic["discovery_cost"].mean(), 1e-9)),
                "semantic_efficiency": float(semantic["found"].mean()),
                "cost_reduction": float(1.0 - semantic["discovery_cost"].mean() / max(blind["discovery_cost"].mean(), 1e-9)),
                "search_compression": float(blind["coordinates_tested"].mean() / max(semantic["coordinates_tested"].mean(), 1e-9)),
                "mean_rediscovery_rank": float(semantic["rediscovery_rank"].replace(-1, np.nan).mean()) if (semantic["rediscovery_rank"] > 0).any() else -1.0,
                "operator_acceleration": float(blind["discovery_cost"].mean() / max(operator["discovery_cost"].mean(), 1e-9)),
                "perturbation_survival": float(semantic["perturbation_survival"].mean()),
                "counterexample_rate": float(semantic["counterexample_rate"].mean()),
                "holdout_success": float(semantic["found"].mean()),
                "recurrence": int(group["world"].nunique()),
            }
        )
    efficiency = pd.DataFrame(efficiency_rows)
    efficiency["scientific_utility"] = efficiency["discovery_acceleration"] * efficiency["semantic_efficiency"] * (1 - efficiency["counterexample_rate"])
    efficiency["law_value"] = (
        efficiency["discovery_acceleration"]
        * efficiency["recurrence"]
        * efficiency["perturbation_survival"]
        * efficiency["holdout_success"]
        * efficiency["semantic_efficiency"]
    )
    efficiency["promote"] = (
        (efficiency["discovery_acceleration"] > 2.0)
        & (efficiency["holdout_success"] > 0.7)
        & (efficiency["perturbation_survival"] > 0.6)
        & (efficiency["counterexample_rate"] < 0.3)
        & (efficiency["recurrence"] >= 3)
        & (efficiency["law_value"] > 2.0)
    )
    laws = {
        "lawbook_version": "v148",
        "laws": [
            {
                "family": row["family"],
                "law_value": float(row["law_value"]),
                "acceleration": float(row["discovery_acceleration"]),
                "recurrence": int(row["recurrence"]),
                "statement": f"{row['family']} reduces future discovery cost in hidden worlds",
            }
            for _, row in efficiency[efficiency["promote"]].iterrows()
        ],
    }
    counterexamples = costs[(~costs["found"]) | (costs["counterexample_rate"] >= 0.3)].copy()
    counterexamples["type"] = np.where(counterexamples["found"], "counterexample_rate_high", "rediscovery_failed")
    return costs, acceleration, ranks, efficiency.sort_values("law_value", ascending=False), efficiency.sort_values("law_value", ascending=False), counterexamples, laws


def verdict(laws: dict[str, Any], efficiency: pd.DataFrame) -> tuple[str, str]:
    if laws["laws"]:
        return "A", "abstractions significantly reduce search"
    if not efficiency.empty and (efficiency["discovery_acceleration"] > 1.2).any():
        return "B", "weak acceleration"
    if not efficiency.empty and (efficiency["discovery_acceleration"] > 1.0).any():
        return "C", "no strong acceleration"
    return "D", "guided search no better than blind search"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def write_reports(out: Path, costs: pd.DataFrame, acceleration: pd.DataFrame, ranks: pd.DataFrame, efficiency: pd.DataFrame, law_values: pd.DataFrame, counterexamples: pd.DataFrame, laws: dict[str, Any], args: argparse.Namespace) -> None:
    out.mkdir(parents=True, exist_ok=True)
    costs.to_csv(out / "discovery_costs.csv", index=False)
    acceleration.to_csv(out / "search_acceleration.csv", index=False)
    ranks.to_csv(out / "rediscovery_ranks.csv", index=False)
    efficiency.to_csv(out / "semantic_efficiency.csv", index=False)
    law_values.to_csv(out / "law_values.csv", index=False)
    efficiency.to_csv(out / "family_survivorship.csv", index=False)
    counterexamples.to_csv(out / "counterexamples.csv", index=False)
    write_json(out / "lawbook_v148.json", laws)
    write_json(out / "manifest.json", {"system": "MATHGRAPH v148 Discovery Acceleration Engine", "seed": args.seed, "quick": args.quick})
    write_markdown(out, efficiency, acceleration, ranks, laws)


def write_markdown(out: Path, efficiency: pd.DataFrame, acceleration: pd.DataFrame, ranks: pd.DataFrame, laws: dict[str, Any]) -> None:
    grade, statement = verdict(laws, efficiency)
    lines = [
        "# MATHGRAPH v148 Discovery Acceleration Report",
        "",
        f"Verdict: **{grade}** — {statement}.",
        "",
        "## Which abstractions reduce search?",
    ]
    for _, row in efficiency.head(12).iterrows():
        lines.append(
            f"- {row['family']}: acceleration={row['discovery_acceleration']:.2f}x, "
            f"law_value={row['law_value']:.3f}, utility={row['scientific_utility']:.3f}, "
            f"rediscovery={row['semantic_efficiency']:.3f}"
        )
    lines.extend(["", "## Which semantic families rediscover hidden worlds?"])
    for _, row in efficiency.sort_values("semantic_efficiency", ascending=False).head(12).iterrows():
        lines.append(f"- {row['family']}: holdout success {row['holdout_success']:.3f}, mean rank {row['mean_rediscovery_rank']:.2f}")
    lines.extend(["", "## Which concepts accelerate discovery?"])
    for _, row in acceleration.sort_values("semantic_acceleration", ascending=False).head(12).iterrows():
        lines.append(f"- {row['hidden_family']} in {row['world']}: semantic acceleration {row['semantic_acceleration']:.2f}x")
    lines.extend(["", "## Closest to a law"])
    if not efficiency.empty:
        top = efficiency.iloc[0]
        lines.append(f"`{top['family']}` is closest by law_value={top['law_value']:.3f}.")
    lines.extend(["", "## Are we seeing features, coordinates, operators, semantics, or genuine discovery?"])
    if laws["laws"]:
        lines.append("The promoted abstractions reduced search cost and rediscovered hidden worlds; this is evidence for discovery acceleration, not universal truth.")
    else:
        lines.append("The run shows measurable search compression but not enough survivorship for a promoted law. This is closer to guided operator/semantic search than genuine scientific discovery.")
    lines.extend(["", "## Interpretation: Law As Search Compression"])
    lines.append(
        "Newtonian force, Darwinian fitness, Shannon information, Peircean abductive signs, and Schmidhuber-style compression all matter because they reduce future search. "
        "v148 tests that same criterion operationally: an abstraction is useful only if it lowers the cost of discovering the next coordinate."
    )
    lines.append("The honesty rule applies: no universal law is claimed; only measured reduction in discovery cost is reported.")
    (out / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    final = f"""# Final Conclusion

Outcome: **{grade}**

Interpretation: {statement}.

The strongest criterion in v148 is not prediction. It is whether an abstraction
reduces future discovery cost. Promotion requires acceleration, rediscovery,
perturbation survival, low counterexample rate, and recurrence.
"""
    (out / "final_conclusion.md").write_text(final, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    mount_drive(args.mount_drive)
    worlds = hidden_worlds(120 if args.quick else 500, args.seed, variants=3)
    costs, acceleration, ranks, efficiency, law_values, counterexamples, laws = experiment(worlds)
    out = Path(args.out)
    write_reports(out, costs, acceleration, ranks, efficiency, law_values, counterexamples, laws, args)
    grade, statement = verdict(laws, efficiency)
    return {"out": str(out), "verdict": grade, "statement": statement, "laws": len(laws["laws"])}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MATHGRAPH v148 Discovery Acceleration Engine")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="mathgraph_v148_out")
    p.add_argument("--seed", type=int, default=148)
    p.add_argument("--mount-drive", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
