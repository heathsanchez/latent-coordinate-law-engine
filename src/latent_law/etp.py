from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_EQUATIONS_PATH = Path("/Users/heath/Desktop/LOGOS Papers/Maths Derivations/equations.txt")
DEFAULT_MATRIX_PATH = Path("/Users/heath/Desktop/LOGOS Papers/Maths Derivations/etp_matrix_full_best_bool.npy")

_FORBIDDEN_COORDINATE_NAMES = ("affine", "projection", "idempotent", "idempotence")
_TOKEN_RE = re.compile(r"[A-Za-z]+|[()=◇*]")
_VAR_RE = re.compile(r"\b[a-z]\b")


def load_equations(path: str | Path) -> list[str]:
    source = Path(path)
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _max_parenthesis_depth(text: str) -> int:
    depth = 0
    best = 0
    for char in text:
        if char == "(":
            depth += 1
            best = max(best, depth)
        elif char == ")":
            depth = max(0, depth - 1)
    return best


def _side_features(text: str, prefix: str) -> dict[str, int | float]:
    variables = _VAR_RE.findall(text)
    counts = {var: variables.count(var) for var in sorted(set(variables))}
    repeated = sum(max(0, count - 1) for count in counts.values())
    return {
        f"{prefix}_char_len": len(text),
        f"{prefix}_token_count": len(_TOKEN_RE.findall(text)),
        f"{prefix}_op_count": text.count("◇") + text.count("*"),
        f"{prefix}_depth": _max_parenthesis_depth(text),
        f"{prefix}_variable_count": len(set(variables)),
        f"{prefix}_leaf_count": len(variables),
        f"{prefix}_repeated_variable_count": repeated,
        f"{prefix}_max_variable_multiplicity": max(counts.values(), default=0),
        f"{prefix}_x_count": counts.get("x", 0),
        f"{prefix}_y_count": counts.get("y", 0),
        f"{prefix}_z_count": counts.get("z", 0),
        f"{prefix}_w_count": counts.get("w", 0),
        f"{prefix}_single_symbol": int(len(variables) == 1 and (text.strip() == variables[0])),
    }


def equation_features(equation: str, prefix: str = "eq") -> dict[str, int | float]:
    left, sep, right = equation.partition("=")
    if not sep:
        left, right = equation, ""
    left = left.strip()
    right = right.strip()
    all_vars = _VAR_RE.findall(equation)
    left_vars = set(_VAR_RE.findall(left))
    right_vars = set(_VAR_RE.findall(right))
    repeated = sum(max(0, all_vars.count(var) - 1) for var in set(all_vars))

    features: dict[str, int | float] = {
        f"{prefix}_char_len": len(equation),
        f"{prefix}_token_count": len(_TOKEN_RE.findall(equation)),
        f"{prefix}_op_count": equation.count("◇") + equation.count("*"),
        f"{prefix}_depth": _max_parenthesis_depth(equation),
        f"{prefix}_variable_count": len(set(all_vars)),
        f"{prefix}_leaf_count": len(all_vars),
        f"{prefix}_repeated_variable_count": repeated,
        f"{prefix}_max_variable_multiplicity": max((all_vars.count(var) for var in set(all_vars)), default=0),
        f"{prefix}_same_side_variable_set": int(left_vars == right_vars),
        f"{prefix}_shared_side_variable_count": len(left_vars & right_vars),
        f"{prefix}_lhs_rhs_leaf_delta": len(_VAR_RE.findall(left)) - len(_VAR_RE.findall(right)),
        f"{prefix}_lhs_rhs_op_delta": (left.count("◇") + left.count("*")) - (right.count("◇") + right.count("*")),
        f"{prefix}_lhs_rhs_depth_delta": _max_parenthesis_depth(left) - _max_parenthesis_depth(right),
    }
    features.update(_side_features(left, f"{prefix}_lhs"))
    features.update(_side_features(right, f"{prefix}_rhs"))
    _assert_no_forbidden_feature_names(features.keys())
    return features


def pair_equation_features(premise: str, conclusion: str) -> dict[str, int | float]:
    premise_vars = set(_VAR_RE.findall(premise))
    conclusion_vars = set(_VAR_RE.findall(conclusion))
    premise_base = equation_features(premise, "premise")
    conclusion_base = equation_features(conclusion, "conclusion")
    features = {
        **premise_base,
        **conclusion_base,
        "pair_shared_variable_count": len(premise_vars & conclusion_vars),
        "pair_same_variable_set": int(premise_vars == conclusion_vars),
        "pair_variable_union_count": len(premise_vars | conclusion_vars),
        "pair_token_delta": premise_base["premise_token_count"] - conclusion_base["conclusion_token_count"],
        "pair_op_delta": premise_base["premise_op_count"] - conclusion_base["conclusion_op_count"],
        "pair_depth_delta": premise_base["premise_depth"] - conclusion_base["conclusion_depth"],
        "pair_repeat_delta": premise_base["premise_repeated_variable_count"] - conclusion_base["conclusion_repeated_variable_count"],
        "pair_total_tokens": premise_base["premise_token_count"] + conclusion_base["conclusion_token_count"],
        "pair_total_ops": premise_base["premise_op_count"] + conclusion_base["conclusion_op_count"],
        "pair_total_depth": premise_base["premise_depth"] + conclusion_base["conclusion_depth"],
        "pair_total_repeats": premise_base["premise_repeated_variable_count"] + conclusion_base["conclusion_repeated_variable_count"],
    }
    _assert_no_forbidden_feature_names(features.keys())
    return features


def generate_etp_from_equations(
    equations_path: str | Path | None = None,
    matrix_path: str | Path | None = None,
    n: int = 300,
    seed: int = 1,
) -> pd.DataFrame:
    eq_path = Path(equations_path) if equations_path is not None else DEFAULT_EQUATIONS_PATH
    mat_path = Path(matrix_path) if matrix_path is not None else DEFAULT_MATRIX_PATH
    if eq_path.exists() and mat_path.exists():
        equations = load_equations(eq_path)
        matrix = np.load(mat_path)
        return _sample_from_matrix(equations, matrix, n=n, seed=seed)
    return _fallback_equation_pairs(n=n, seed=seed)


def _sample_from_matrix(equations: list[str], matrix: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    limit = min(len(equations), matrix.shape[0], matrix.shape[1])
    positives_needed = max(1, n // 2)
    negatives_needed = n - positives_needed
    positives: list[tuple[int, int]] = []
    negatives: list[tuple[int, int]] = []
    max_attempts = max(20_000, n * 300)
    attempts = 0
    while (len(positives) < positives_needed or len(negatives) < negatives_needed) and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, limit))
        j = int(rng.integers(0, limit))
        value = int(matrix[i, j])
        if value and len(positives) < positives_needed:
            positives.append((i, j))
        elif not value and len(negatives) < negatives_needed:
            negatives.append((i, j))

    pairs = positives + negatives
    rng.shuffle(pairs)
    rows = [_row_from_pair(equations, i, j, int(matrix[i, j])) for i, j in pairs[:n]]
    return pd.DataFrame(rows)


def _fallback_equation_pairs(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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
    equations = [f"{left} = {right}" for left in terms for right in terms]
    rows = []
    for _ in range(n):
        i = int(rng.integers(0, len(equations)))
        j = int(rng.integers(0, len(equations)))
        features = pair_equation_features(equations[i], equations[j])
        implication = int(
            features["pair_same_variable_set"] == 1
            and features["pair_total_repeats"] >= 2
            and features["pair_depth_delta"] >= -1
        )
        rows.append(_row_from_pair(equations, i, j, implication))
    return pd.DataFrame(rows)


def _row_from_pair(equations: list[str], i: int, j: int, implication: int) -> dict[str, int | float | str]:
    premise = equations[i]
    conclusion = equations[j]
    features = pair_equation_features(premise, conclusion)
    map_cost = features["pair_total_tokens"] + features["pair_variable_union_count"]
    route_cost = int((features["pair_total_tokens"] + 1) * (features["pair_total_ops"] + 1))
    return {
        "domain": "ETP",
        "premise_index": i,
        "conclusion_index": j,
        "premise_equation": premise,
        "conclusion_equation": conclusion,
        **features,
        "implication_true": int(implication),
        "map_cost": int(map_cost),
        "route_cost": int(max(route_cost, map_cost + 1)),
    }


def _assert_no_forbidden_feature_names(names: Iterable[str]) -> None:
    for name in names:
        lowered = name.lower()
        if any(term in lowered for term in _FORBIDDEN_COORDINATE_NAMES):
            raise ValueError(f"ETP feature leaks forbidden coordinate name: {name}")
