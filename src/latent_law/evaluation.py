from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from latent_law.features import extract_features


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"t", "r", "label", "experiment", "holdout", "run", "description", "status"}
    return [
        col
        for col in df.columns
        if col not in excluded and not col.startswith("coeff_") and df[col].nunique(dropna=False) > 1
    ]


def _build_model(x: pd.DataFrame) -> Pipeline:
    categorical = [col for col in x.columns if not pd.api.types.is_numeric_dtype(x[col]) and not pd.api.types.is_bool_dtype(x[col])]
    numeric = [col for col in x.columns if col not in categorical]
    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("numeric", "passthrough", numeric),
                    ],
                    remainder="drop",
                ),
            ),
            ("classifier", RandomForestClassifier(n_estimators=160, random_state=7, class_weight="balanced")),
        ]
    )


def _evaluate_target(train: pd.DataFrame, holdout: pd.DataFrame, features: list[str], target: str) -> tuple[dict[str, Any], list[Any]]:
    model = _build_model(train[features])
    y_train = train[target].astype(str)
    y_true = holdout[target].astype(str)
    model.fit(train[features], y_train)
    pred = model.predict(holdout[features])
    labels = sorted(set(y_true.tolist()) | set(pred.tolist()))
    failed = [
        {
            "index": int(idx),
            "actual": actual,
            "predicted": predicted,
        }
        for idx, actual, predicted in zip(holdout.index, y_true.tolist(), pred.tolist())
        if actual != predicted
    ]
    return (
        {
            "accuracy": float(accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
            "failed_predictions": failed,
        },
        pred.tolist(),
    )


def evaluate_holdout(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> dict[str, Any]:
    """Train on train rows and evaluate t, r, and combined predictions."""

    train = extract_features(train_df)
    holdout = extract_features(holdout_df)
    if train.empty or holdout.empty:
        raise ValueError("train_df and holdout_df must both contain rows")

    features = _feature_columns(pd.concat([train, holdout], axis=0, ignore_index=True))
    t_report, t_pred = _evaluate_target(train, holdout, features, "t")
    r_report, r_pred = _evaluate_target(train, holdout, features, "r")

    combined_true = holdout[["t", "r"]].astype(str).agg("|".join, axis=1).tolist()
    combined_pred = [f"{t}|{r}" for t, r in zip(t_pred, r_pred)]
    labels = sorted(set(combined_true) | set(combined_pred))
    combined_failed = [
        {"index": int(idx), "actual": actual, "predicted": predicted}
        for idx, actual, predicted in zip(holdout.index, combined_true, combined_pred)
        if actual != predicted
    ]
    combined = {
        "accuracy": float(accuracy_score(combined_true, combined_pred)),
        "macro_f1": float(f1_score(combined_true, combined_pred, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(combined_true, combined_pred, labels=labels).tolist(),
        "failed_predictions": combined_failed,
    }
    return {"features": features, "t": t_report, "r": r_report, "combined": combined}
