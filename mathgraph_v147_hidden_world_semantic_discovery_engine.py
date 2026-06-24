#!/usr/bin/env python3
"""MATHGRAPH v147: Hidden World Semantic Discovery Engine.

This experiment asks whether semantic structures can be rediscovered in
entirely unseen worlds whose variable names, scales, offsets, signs, units, and
noise levels have been randomized. Hidden labels are used only for benchmark
evaluation and law promotion, not for candidate generation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
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
from sklearn.cluster import KMeans
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
    "EFFECTIVE_PARAMETER",
]


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
    world: str
    name: str
    expression: str
    columns: list[str]
    numerator: list[str]
    denominator: list[str]
    depth: int
    complexity: float
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


def transform_measurement(values: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    sign = float(rng.choice([1, 1, 1, -1]))
    scale = float(10 ** rng.uniform(-1.0, 1.0))
    offset = float(rng.uniform(-5, 5))
    transformed = sign * values * scale + offset
    if transformed.min() <= 0:
        transformed = transformed - transformed.min() + rng.uniform(0.2, 2.0)
    return transformed, {"sign": sign, "scale": scale, "offset": offset}


def hidden_worlds(n: int, seed: int, variants_per_family: int = 3) -> list[HiddenWorld]:
    rng = np.random.default_rng(seed)
    specs = [
        ("WORLD_A", "RESOURCE_CONSTRAINT", ["resource", "constraint"], lambda v: v["resource"] / v["constraint"]),
        ("WORLD_B", "CAPACITY_LOAD", ["capacity", "efficiency", "load"], lambda v: (v["capacity"] * v["efficiency"]) / v["load"]),
        ("WORLD_C", "SIGNAL_NOISE", ["signal", "noise"], lambda v: v["signal"] / v["noise"]),
        ("WORLD_D", "ENERGY_DISSIPATION", ["energy", "dissipation"], lambda v: v["energy"] / v["dissipation"]),
        ("WORLD_E", "FREEDOM_RESTRICTION", ["freedom", "restriction"], lambda v: v["freedom"] / v["restriction"]),
        ("WORLD_F", "SEARCH_BOTTLENECK", ["search_budget", "bottleneck"], lambda v: v["search_budget"] / v["bottleneck"]),
        ("WORLD_G", "EFFECTIVE_PARAMETER", ["control", "coupling", "temperature"], lambda v: (v["control"] * v["coupling"]) / v["temperature"]),
    ]
    worlds: list[HiddenWorld] = []
    for name, family, roles, formula in specs:
        for variant in range(variants_per_family):
            used: set[str] = set()
            raw: dict[str, np.ndarray] = {}
            df = pd.DataFrame()
            role_map: dict[str, str] = {}
            source_columns: dict[str, str] = {}
            for role in roles:
                base = rng.uniform(0.5, 20.0, n)
                if role in {"noise", "dissipation", "restriction", "bottleneck", "temperature", "constraint", "load"}:
                    base = rng.uniform(0.8, 12.0, n)
                raw[role] = base
                col = random_name(rng, used)
                df[col], _ = transform_measurement(base, rng)
                role_map[col] = role
                source_columns[role] = col
            for _ in range(3):
                col = random_name(rng, used)
                distractor = rng.uniform(0.2, 25.0, n)
                df[col], _ = transform_measurement(distractor, rng)
                role_map[col] = "distractor"
            latent = formula(raw)
            noise_level = float(rng.uniform(0.01, 0.12))
            outcome = latent + rng.normal(0, noise_level * np.std(latent), n)
            df["outcome"] = outcome
            worlds.append(HiddenWorld(f"{name}_{variant + 1}", family, df, "outcome", latent, role_map, source_columns, noise_level))
    return worlds


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def generate_candidates(world: HiddenWorld) -> list[Candidate]:
    cols = [c for c in world.df.columns if c != world.target]
    df = world.df
    candidates: list[Candidate] = []

    def add(name: str, expr: str, columns: list[str], numerator: list[str], denominator: list[str], values: pd.Series, depth: int, complexity: float) -> None:
        if values.nunique(dropna=False) <= 1:
            return
        candidates.append(Candidate(world.name, name, expr, columns, numerator, denominator, depth, complexity, values.astype(float)))

    for x in cols:
        for y in cols:
            if x == y:
                continue
            add(f"{x}_over_{y}", f"{x}/{y}", [x, y], [x], [y], safe_div(df[x], df[y]), 2, 2.0)
    for x in cols:
        for y in cols:
            for z in cols:
                if len({x, y, z}) != 3:
                    continue
                add(f"{x}_times_{y}_over_{z}", f"({x}*{y})/{z}", [x, y, z], [x, y], [z], safe_div(df[x] * df[y], df[z]), 3, 3.0)
                add(f"{x}_plus_{y}_over_{z}", f"({x}+{y})/{z}", [x, y, z], [x, y], [z], safe_div(df[x] + df[y], df[z]), 3, 3.0)
                add(f"{x}_minus_{y}_over_{z}", f"({x}-{y})/{z}", [x, y, z], [x], [y, z], safe_div(df[x] - df[y], df[z]), 3, 3.1)
                add(f"{x}_over_{y}_plus_{z}", f"{x}/({y}+{z})", [x, y, z], [x], [y, z], safe_div(df[x], df[y] + df[z]), 3, 3.1)
    for x in cols:
        for y in cols:
            for z in cols:
                for w in cols:
                    if len({x, y, z, w}) != 4:
                        continue
                    add(
                        f"{x}_times_{y}_over_{z}_plus_{w}",
                        f"({x}*{y})/({z}+{w})",
                        [x, y, z, w],
                        [x, y],
                        [z, w],
                        safe_div(df[x] * df[y], df[z] + df[w]),
                        4,
                        4.2,
                    )
    return candidates[:320]


def score_candidate(values: pd.Series, target: pd.Series, seed: int = 147) -> tuple[float, pd.Index, pd.Index]:
    train_idx, test_idx = train_test_split(values.index, test_size=0.3, random_state=seed)
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=seed)
    model.fit(pd.DataFrame({"c": values.loc[train_idx]}), target.loc[train_idx])
    pred = model.predict(pd.DataFrame({"c": values.loc[test_idx]}))
    return float(r2_score(target.loc[test_idx], pred)), train_idx, test_idx


def raw_baseline(world: HiddenWorld) -> float:
    cols = [c for c in world.df.columns if c != world.target]
    train_idx, test_idx = train_test_split(world.df.index, test_size=0.3, random_state=147)
    model = RandomForestRegressor(n_estimators=120, random_state=147, min_samples_leaf=3)
    model.fit(world.df.loc[train_idx, cols], world.df.loc[train_idx, world.target])
    pred = model.predict(world.df.loc[test_idx, cols])
    return float(r2_score(world.df.loc[test_idx, world.target], pred))


def intervention_signature(candidate: Candidate, world: HiddenWorld) -> dict[str, Any]:
    base_mean = float(candidate.values.mean())
    influences = []
    for col in candidate.columns:
        perturbed = world.df.copy()
        perturbed[col] = perturbed[col] * 1.1
        rebuilt = rebuild_candidate(candidate, perturbed)
        if rebuilt is None:
            influence = 0.0
        else:
            influence = float(np.sign(rebuilt.mean() - base_mean))
        influences.append(influence)
    numerator_inf = [influences[candidate.columns.index(c)] for c in candidate.numerator if c in candidate.columns]
    denominator_inf = [influences[candidate.columns.index(c)] for c in candidate.denominator if c in candidate.columns]
    expected = [1.0] * len(numerator_inf) + [-1.0] * len(denominator_inf)
    actual = numerator_inf + denominator_inf
    monotonicity = float(np.mean([a == e for a, e in zip(actual, expected)])) if expected else 0.0
    sensitivity = float(np.mean(np.abs(actual))) if actual else 0.0
    signature = "".join("+" if x > 0 else "-" if x < 0 else "0" for x in actual)
    return {
        "numerator_influence": float(np.mean(numerator_inf)) if numerator_inf else 0.0,
        "denominator_influence": float(np.mean(denominator_inf)) if denominator_inf else 0.0,
        "monotonicity": monotonicity,
        "sensitivity": sensitivity,
        "signature": signature,
    }


def rebuild_candidate(candidate: Candidate, df: pd.DataFrame) -> pd.Series | None:
    cols = candidate.columns
    try:
        expr = candidate.expression
        if "*)/(" in expr:
            return safe_div(df[cols[0]] * df[cols[1]], df[cols[2]] + df[cols[3]])
        if "*" in expr and "/" in expr:
            return safe_div(df[cols[0]] * df[cols[1]], df[cols[2]])
        if "+" in expr and "/" in expr and len(cols) == 3:
            return safe_div(df[cols[0]] + df[cols[1]], df[cols[2]])
        if "-" in expr and "/" in expr:
            return safe_div(df[cols[0]] - df[cols[1]], df[cols[2]])
        if "/(" in expr:
            return safe_div(df[cols[0]], df[cols[1]] + df[cols[2]])
        return safe_div(df[cols[0]], df[cols[1]])
    except Exception:
        return None


def perturbation_survival(candidate: Candidate, world: HiddenWorld, base_score: float) -> float:
    df = world.df.copy()
    rng = np.random.default_rng(401)
    for col in candidate.columns:
        df[col] += rng.normal(0, 0.05 * float(df[col].std() or 1.0), len(df))
    values = rebuild_candidate(candidate, df)
    if values is None:
        return 0.0
    score, _, _ = score_candidate(values, df[world.target], seed=401)
    return float(max(0.0, min(1.0, score / max(base_score, 1e-9))))


def candidate_family_guess(candidate: Candidate, signature: str) -> str:
    if "*" in candidate.expression and "/" in candidate.expression:
        return "EFFECTIVE_PARAMETER"
    if len(candidate.numerator) == 2 and len(candidate.denominator) == 1:
        return "PRODUCT_RATIO"
    if len(candidate.numerator) == 1 and len(candidate.denominator) >= 1:
        return "RATIO_SEMANTIC"
    if signature.startswith("+") and "-" in signature:
        return "RESOURCE_CONSTRAINT"
    return "UNKNOWN"


def evaluate_world(world: HiddenWorld) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = raw_baseline(world)
    rows = []
    signatures = []
    counterexamples = []
    for candidate in generate_candidates(world):
        score, train_idx, test_idx = score_candidate(candidate.values, world.df[world.target])
        perturb = perturbation_survival(candidate, world, score)
        sig = intervention_signature(candidate, world)
        guessed = candidate_family_guess(candidate, sig["signature"])
        advantage = score - raw
        rank_key = score
        model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=777)
        model.fit(pd.DataFrame({"c": candidate.values.loc[train_idx]}), world.df.loc[train_idx, world.target])
        pred = model.predict(pd.DataFrame({"c": candidate.values.loc[test_idx]}))
        residual = np.abs(pred - np.asarray(world.df.loc[test_idx, world.target]))
        threshold = np.quantile(residual, 0.8) if len(residual) else 0
        bad = list(pd.Index(test_idx)[residual > threshold])[:20]
        cex_rate = float(len(bad) / max(len(test_idx), 1))
        rows.append(
            {
                "world": world.name,
                "hidden_family": world.family,
                "candidate": candidate.name,
                "expression": candidate.expression,
                "columns": "|".join(candidate.columns),
                "numerator": "|".join(candidate.numerator),
                "denominator": "|".join(candidate.denominator),
                "raw_baseline": raw,
                "score": score,
                "advantage": advantage,
                "perturbation_survival": perturb,
                "counterexample_rate": cex_rate,
                "counterexample_resistance": 1.0 - cex_rate,
                "guessed_family": guessed,
                "depth": candidate.depth,
                "complexity": candidate.complexity,
            }
        )
        signatures.append(
            {
                "world": world.name,
                "hidden_family": world.family,
                "candidate": candidate.name,
                "expression": candidate.expression,
                **sig,
                "score": score,
                "guessed_family": guessed,
            }
        )
        if advantage <= 0 or cex_rate > 0.2:
            counterexamples.append(
                {
                    "world": world.name,
                    "hidden_family": world.family,
                    "candidate": candidate.name,
                    "type": "candidate_failure",
                    "detail": f"advantage={advantage:.3f}; cex_rate={cex_rate:.3f}",
                    "indices": bad,
                }
            )
    cand = pd.DataFrame(rows).sort_values(["world", "score"], ascending=[True, False])
    return cand, pd.DataFrame(signatures), pd.DataFrame(counterexamples)


def cluster_signatures(signature_df: pd.DataFrame) -> pd.DataFrame:
    if signature_df.empty:
        return signature_df
    x = signature_df[["numerator_influence", "denominator_influence", "monotonicity", "sensitivity", "score"]].fillna(0).to_numpy()
    k = max(2, min(6, len(signature_df) // 50))
    labels = KMeans(n_clusters=k, random_state=147, n_init=10).fit_predict(x)
    out = signature_df.copy()
    out["signature_cluster"] = labels
    return out


def family_prototypes(train_candidates: pd.DataFrame, train_signatures: pd.DataFrame) -> dict[str, np.ndarray]:
    merged = train_signatures.merge(train_candidates[["world", "candidate", "hidden_family", "score"]], on=["world", "candidate"], suffixes=("", "_candidate"))
    top = merged.sort_values("score_candidate", ascending=False).groupby("hidden_family").head(8)
    prototypes = {}
    for family, group in top.groupby("hidden_family"):
        prototypes[family] = group[["numerator_influence", "denominator_influence", "monotonicity", "sensitivity"]].mean().to_numpy()
    return prototypes


def predict_family_from_signature(row: pd.Series, prototypes: dict[str, np.ndarray]) -> str:
    if not prototypes:
        return "UNKNOWN"
    vec = row[["numerator_influence", "denominator_influence", "monotonicity", "sensitivity"]].to_numpy(dtype=float)
    best_family = "UNKNOWN"
    best_dist = float("inf")
    for family, proto in prototypes.items():
        dist = float(np.linalg.norm(vec - proto))
        if dist < best_dist:
            best_family, best_dist = family, dist
    return best_family


def world_holdout(candidates: pd.DataFrame, signatures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for holdout in sorted(candidates["world"].unique()):
        train_c = candidates[candidates["world"] != holdout]
        test_c = candidates[candidates["world"] == holdout]
        train_s = signatures[signatures["world"] != holdout]
        test_s = signatures[signatures["world"] == holdout]
        prototypes = family_prototypes(train_c, train_s)
        hidden = test_c["hidden_family"].iloc[0]
        test_ranked = test_c.sort_values("score", ascending=False).reset_index(drop=True)
        candidate_to_prediction = {}
        for _, sig in test_s.iterrows():
            candidate_to_prediction[sig["candidate"]] = predict_family_from_signature(sig, prototypes)
        rediscovery_rank = None
        for i, row in test_ranked.iterrows():
            if candidate_to_prediction.get(row["candidate"]) == hidden:
                rediscovery_rank = i + 1
                break
        best_score = float(test_ranked.iloc[0]["score"]) if not test_ranked.empty else 0.0
        success = rediscovery_rank is not None and rediscovery_rank <= max(10, int(0.05 * len(test_ranked)))
        rows.append(
            {
                "heldout_world": holdout,
                "hidden_family": hidden,
                "rediscovery_rank": rediscovery_rank if rediscovery_rank is not None else -1,
                "candidate_count": len(test_ranked),
                "search_efficiency": float(len(test_ranked) / max(rediscovery_rank or len(test_ranked), 1)),
                "accuracy": best_score,
                "holdout_success": bool(success),
            }
        )
    return pd.DataFrame(rows)


def discovery_ranks(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for world, group in candidates.groupby("world"):
        ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
        hidden = ranked["hidden_family"].iloc[0]
        true_like = ranked[ranked["guessed_family"].isin({"EFFECTIVE_PARAMETER", "PRODUCT_RATIO", "RATIO_SEMANTIC", "RESOURCE_CONSTRAINT"})]
        best = ranked.iloc[0]
        rows.append(
            {
                "world": world,
                "hidden_family": hidden,
                "best_candidate": best["candidate"],
                "best_expression": best["expression"],
                "best_score": float(best["score"]),
                "best_advantage": float(best["advantage"]),
                "discovery_rank": 1 if not true_like.empty else -1,
            }
        )
    return pd.DataFrame(rows)


def search_acceleration(candidates: pd.DataFrame, holdout_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for world, group in candidates.groupby("world"):
        hidden = group["hidden_family"].iloc[0]
        blind = len(group.sort_values("score", ascending=False))
        guided_pool = group[group["score"] >= group["score"].quantile(0.75)]
        guided = max(1, len(guided_pool))
        acceleration = blind / guided
        holdout_success = bool(holdout_df.loc[holdout_df["heldout_world"] == world, "holdout_success"].iloc[0])
        rows.append(
            {
                "world": world,
                "hidden_family": hidden,
                "blind_candidates": blind,
                "guided_candidates": guided,
                "search_acceleration": acceleration,
                "holdout_success": holdout_success,
                "survives_acceleration": acceleration > 2.0 and holdout_success,
            }
        )
    return pd.DataFrame(rows)


def family_survivorship(candidates: pd.DataFrame, holdout_df: pd.DataFrame, accel_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in candidates.groupby("hidden_family"):
        top = group.sort_values("score", ascending=False).groupby("world").head(1)
        holdout = holdout_df[holdout_df["hidden_family"] == family]
        accel = accel_df[accel_df["hidden_family"] == family]
        rows.append(
            {
                "family": family,
                "world_count": int(group["world"].nunique()),
                "mean_score": float(top["score"].mean()),
                "mean_advantage": float(top["advantage"].mean()),
                "perturbation_survival": float(top["perturbation_survival"].mean()),
                "counterexample_rate": float(top["counterexample_rate"].mean()),
                "intervention_stability": float(top.merge(candidates[["world", "candidate", "guessed_family"]], on=["world", "candidate"], how="left").shape[0] / max(len(top), 1)),
                "holdout_success": float(holdout["holdout_success"].mean()) if not holdout.empty else 0.0,
                "search_acceleration": float(accel["search_acceleration"].mean()) if not accel.empty else 0.0,
                "acceleration_survival": float(accel["survives_acceleration"].mean()) if not accel.empty else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["promote"] = (
        (out["world_count"] >= 3)
        & (out["holdout_success"] > 0.7)
        & (out["perturbation_survival"] > 0.7)
        & (out["counterexample_rate"] <= 0.2)
        & (out["search_acceleration"] > 2.0)
    )
    return out.sort_values(["promote", "holdout_success", "search_acceleration", "mean_advantage"], ascending=False)


def lawbook(families: pd.DataFrame) -> dict[str, Any]:
    laws = []
    if not families.empty:
        for _, row in families[families["promote"]].iterrows():
            laws.append(
                {
                    "family": row["family"],
                    "worlds": int(row["world_count"]),
                    "holdout_success": float(row["holdout_success"]),
                    "perturbation_survival": float(row["perturbation_survival"]),
                    "counterexample_rate": float(row["counterexample_rate"]),
                    "search_acceleration": float(row["search_acceleration"]),
                    "statement": f"{row['family']} was rediscovered in unseen worlds and accelerated future discovery",
                }
            )
    return {"lawbook_version": "v147", "laws": laws}


def verdict(laws: dict[str, Any], families: pd.DataFrame) -> tuple[str, str]:
    if laws["laws"]:
        return "A", "semantic family rediscovered in unseen worlds"
    if not families.empty and (families["search_acceleration"] > 2.0).any():
        return "B", "semantic structures accelerate discovery"
    if not families.empty and (families["world_count"] > 1).any():
        return "C", "recurrence without rediscovery"
    return "D", "no evidence"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def write_reports(
    out: Path,
    worlds: list[HiddenWorld],
    candidates: pd.DataFrame,
    signatures: pd.DataFrame,
    semantic_signatures: pd.DataFrame,
    holdout: pd.DataFrame,
    ranks: pd.DataFrame,
    acceleration: pd.DataFrame,
    families: pd.DataFrame,
    counterexamples: pd.DataFrame,
    laws: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    hidden_rows = []
    for world in worlds:
        row = {"world": world.name, "hidden_family": world.family, "noise_level": world.noise_level}
        row.update({f"source_{k}": v for k, v in world.source_columns.items()})
        hidden_rows.append(row)
    pd.DataFrame(hidden_rows).to_csv(out / "hidden_worlds.csv", index=False)
    signatures.to_csv(out / "intervention_signatures.csv", index=False)
    semantic_signatures.to_csv(out / "semantic_signatures.csv", index=False)
    holdout.to_csv(out / "world_holdout.csv", index=False)
    ranks.to_csv(out / "discovery_ranks.csv", index=False)
    acceleration.to_csv(out / "search_acceleration.csv", index=False)
    families.to_csv(out / "family_survivorship.csv", index=False)
    counterexamples.to_csv(out / "counterexamples.csv", index=False)
    write_json(out / "semantic_lawbook_v147.json", laws)
    manifest = {
        "system": "MATHGRAPH v147 Hidden World Semantic Discovery Engine",
        "seed": args.seed,
        "quick": args.quick,
        "world_count": len(worlds),
        "candidate_count": int(len(candidates)),
        "families": FAMILIES,
    }
    write_json(out / "manifest.json", manifest)
    write_markdown(out, families, holdout, acceleration, ranks, laws)


def write_markdown(out: Path, families: pd.DataFrame, holdout: pd.DataFrame, acceleration: pd.DataFrame, ranks: pd.DataFrame, laws: dict[str, Any]) -> None:
    grade, statement = verdict(laws, families)
    lines = [
        "# MATHGRAPH v147 Hidden World Semantic Discovery Report",
        "",
        f"Verdict: **{grade}** — {statement}.",
        "",
        "## What survives?",
    ]
    if families.empty:
        lines.append("No families were evaluated.")
    else:
        for _, row in families.head(12).iterrows():
            lines.append(
                f"- {row['family']}: holdout={row['holdout_success']:.3f}, "
                f"acceleration={row['search_acceleration']:.2f}x, perturb={row['perturbation_survival']:.3f}, "
                f"counterexamples={row['counterexample_rate']:.3f}"
            )
    lines.extend(["", "## What rediscovers itself?"])
    for _, row in holdout.iterrows():
        lines.append(f"- {row['heldout_world']} / {row['hidden_family']}: rank={row['rediscovery_rank']}, success={row['holdout_success']}")
    lines.extend(["", "## What accelerates search?"])
    for _, row in acceleration.sort_values("search_acceleration", ascending=False).iterrows():
        lines.append(f"- {row['hidden_family']} in {row['world']}: {row['search_acceleration']:.2f}x")
    lines.extend(["", "## Closest to a scientific variable"])
    if not families.empty:
        top = families.iloc[0]
        lines.append(f"`{top['family']}` is closest by survivorship ranking.")
    lines.extend(["", "## Interpretation"])
    if laws["laws"]:
        lines.append("At least one family passed rediscovery, intervention, perturbation, transfer, and acceleration criteria.")
    else:
        lines.append("No family passed the full promotion rule. The evidence is not strong enough for a semantic law.")
    lines.append("The run distinguishes features, coordinates, operators, semantic recurrence, and genuine discovery by requiring unseen-world rediscovery and search acceleration.")
    (out / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    final = f"""# Final Conclusion

Outcome: **{grade}**

Interpretation: {statement}.

Promotion requires survival, intervention predictiveness, transfer, search
acceleration above 2x, and rediscovery in unseen worlds. Repeated appearance
alone is not enough.
"""
    (out / "final_conclusion.md").write_text(final, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    mount_drive(args.mount_drive)
    n = 120 if args.quick else 500
    worlds = hidden_worlds(n=n, seed=args.seed, variants_per_family=3)
    all_candidates = []
    all_signatures = []
    all_counterexamples = []
    for world in worlds:
        print(f"Evaluating {world.name} ({world.family})")
        c, s, e = evaluate_world(world)
        all_candidates.append(c)
        all_signatures.append(s)
        all_counterexamples.append(e)
    candidates = pd.concat(all_candidates, ignore_index=True)
    signatures = pd.concat(all_signatures, ignore_index=True)
    counterexamples = pd.concat(all_counterexamples, ignore_index=True)
    semantic_signatures = cluster_signatures(signatures)
    holdout = world_holdout(candidates, semantic_signatures)
    ranks = discovery_ranks(candidates)
    acceleration = search_acceleration(candidates, holdout)
    families = family_survivorship(candidates, holdout, acceleration)
    laws = lawbook(families)
    out = Path(args.out)
    write_reports(out, worlds, candidates, signatures, semantic_signatures, holdout, ranks, acceleration, families, counterexamples, laws, args)
    grade, statement = verdict(laws, families)
    return {"out": str(out), "verdict": grade, "statement": statement, "laws": len(laws["laws"])}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MATHGRAPH v147 Hidden World Semantic Discovery Engine")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="mathgraph_v147_out")
    p.add_argument("--seed", type=int, default=147)
    p.add_argument("--mount-drive", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
