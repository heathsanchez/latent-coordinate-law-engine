from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from latent_law.features import extract_features


@dataclass
class Law:
    name: str
    target: str
    condition: dict
    predicted_value: Any
    precision: float
    recall: float
    support: int
    exceptions: list[int]
    confidence: str
    statement: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _condition_mask(df: pd.DataFrame, condition: dict) -> pd.Series:
    if "all" in condition:
        masks = [_condition_mask(df, part) for part in condition["all"]]
        if not masks:
            return pd.Series(True, index=df.index)
        mask = masks[0].copy()
        for part in masks[1:]:
            mask &= part
        return mask

    feature = condition["feature"]
    op = condition["op"]
    value = condition["value"]
    series = df[feature]
    if op == "<=":
        return series <= value
    if op == ">=":
        return series >= value
    if op == "==":
        return series == value
    if op == "in":
        return series.isin(value)
    raise ValueError(f"unsupported condition operator: {op}")


def law_condition_mask(df: pd.DataFrame, condition: dict) -> pd.Series:
    return _condition_mask(df, condition)


def _condition_text(condition: dict) -> str:
    if "all" in condition:
        return " and ".join(_condition_text(part) for part in condition["all"])
    if condition["op"] == "in":
        values = ", ".join(map(str, condition["value"]))
        return f"{condition['feature']} in {{{values}}}"
    return f"{condition['feature']} {condition['op']} {condition['value']}"


def _confidence(precision: float, recall: float, support: int) -> str:
    if precision >= 0.98 and recall >= 0.75 and support >= 5:
        return "high"
    if precision >= 0.9 and support >= 3:
        return "medium"
    return "low"


def _candidate_conditions(df: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    excluded = {target, "label", "experiment", "holdout", "run", "description", "status"}
    conditions: list[dict[str, Any]] = []
    for feature in df.columns:
        if feature in excluded or feature.startswith("coeff_"):
            continue
        series = df[feature]
        if pd.api.types.is_bool_dtype(series):
            for value in [False, True]:
                conditions.append({"feature": feature, "op": "==", "value": value})
        elif pd.api.types.is_numeric_dtype(series):
            values = sorted(v for v in series.dropna().unique().tolist() if np.isfinite(v))
            if len(values) <= 20:
                thresholds = values
            else:
                quantiles = np.linspace(0.05, 0.95, 19)
                thresholds = sorted(set(np.quantile(values, quantiles).round(6).tolist() + values))
            for value in thresholds:
                scalar = int(value) if float(value).is_integer() else float(value)
                conditions.extend(
                    [
                        {"feature": feature, "op": "<=", "value": scalar},
                        {"feature": feature, "op": "==", "value": scalar},
                        {"feature": feature, "op": ">=", "value": scalar},
                    ]
                )
        else:
            values = sorted(series.dropna().astype(str).unique().tolist())
            for value in values:
                conditions.append({"feature": feature, "op": "==", "value": value})
            if 1 < len(values) <= 5:
                conditions.append({"feature": feature, "op": "in", "value": values})

    return conditions


def _score_condition(df: pd.DataFrame, target: str, condition: dict) -> Law | None:
    mask = _condition_mask(df, condition)
    support = int(mask.sum())
    if support == 0:
        return None

    target_values = df.loc[mask, target]
    predicted_value = target_values.mode(dropna=False).iloc[0]
    correct = mask & (df[target] == predicted_value)
    predicted_total = int((df[target] == predicted_value).sum())
    precision = float(correct.sum() / support)
    recall = float(correct.sum() / predicted_total) if predicted_total else 0.0
    exceptions = df.index[mask & (df[target] != predicted_value)].astype(int).tolist()
    condition_text = _condition_text(condition)
    statement = f"if {condition_text} then {target} = {predicted_value}"
    name = f"{target}_{condition_text.replace(' ', '_').replace('{', '').replace('}', '')}"
    return Law(
        name=name,
        target=target,
        condition=condition,
        predicted_value=predicted_value.item() if hasattr(predicted_value, "item") else predicted_value,
        precision=precision,
        recall=recall,
        support=support,
        exceptions=exceptions,
        confidence=_confidence(precision, recall, support),
        statement=statement,
    )


def induce_laws(
    df: pd.DataFrame,
    target: str,
    min_precision: float = 0.9,
    min_recall: float = 0.5,
) -> list[Law]:
    """Mine simple threshold and membership laws for a target column."""

    features = extract_features(df)
    if target not in features.columns:
        raise ValueError(f"target column not found: {target}")

    laws: list[Law] = []
    seen: set[tuple[str, str, str]] = set()
    for condition in _candidate_conditions(features, target):
        law = _score_condition(features, target, condition)
        if law is None:
            continue
        key = (law.target, str(law.condition), str(law.predicted_value))
        if key in seen:
            continue
        seen.add(key)
        if law.precision >= min_precision and law.recall >= min_recall:
            laws.append(law)

    laws.sort(key=lambda law: (law.precision, law.recall, law.support), reverse=True)
    return laws
