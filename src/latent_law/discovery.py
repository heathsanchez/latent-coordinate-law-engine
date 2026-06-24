from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from latent_law.features import extract_features


def _combined_target(df: pd.DataFrame, target_cols: list[str]) -> pd.Series:
    return df[target_cols].astype(str).agg("|".join, axis=1)


def _candidate_features(df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    excluded = set(target_cols) | {"label", "experiment", "holdout", "run", "description", "status"}
    return [
        col
        for col in df.columns
        if col not in excluded and not col.startswith("coeff_") and df[col].nunique(dropna=False) > 1
    ]


def _encode_feature(series: pd.Series) -> pd.DataFrame:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return pd.DataFrame({series.name: series.astype(float)})
    dummies = pd.get_dummies(series.astype(str), prefix=series.name)
    return dummies


def _single_feature_score(df: pd.DataFrame, feature: str, target: pd.Series) -> dict[str, Any]:
    x_encoded = _encode_feature(df[feature])
    y = target.astype(str)
    if y.nunique() < 2:
        return {"feature": feature, "mutual_info": 0.0, "tree_accuracy": 1.0, "tree_macro_f1": 1.0, "tree_rule": ""}

    discrete = [not pd.api.types.is_numeric_dtype(df[feature]) or pd.api.types.is_bool_dtype(df[feature])] * x_encoded.shape[1]
    mi = float(mutual_info_classif(x_encoded, y, discrete_features=discrete, random_state=0).sum())

    counts = y.value_counts()
    stratify = y if len(y) >= 10 and counts.min() >= 2 and y.nunique() > 1 else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x_encoded, y, test_size=0.3, random_state=0, stratify=stratify
        )
    except ValueError:
        x_train, x_test, y_train, y_test = x_encoded, x_encoded, y, y

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=0)
    tree.fit(x_train, y_train)
    pred = tree.predict(x_test)
    rule = export_text(tree, feature_names=list(x_encoded.columns))
    return {
        "feature": feature,
        "mutual_info": mi,
        "tree_accuracy": float(accuracy_score(y_test, pred)),
        "tree_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "tree_rule": rule,
    }


def _baseline(df: pd.DataFrame, features: list[str], target: pd.Series) -> dict[str, float]:
    y = target.astype(str)
    if not features or y.nunique() < 2:
        return {"accuracy": 1.0, "macro_f1": 1.0}

    x = df[features].copy()
    categorical = [col for col in features if not pd.api.types.is_numeric_dtype(x[col]) and not pd.api.types.is_bool_dtype(x[col])]
    numeric = [col for col in features if col not in categorical]
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=120, random_state=0, class_weight="balanced")),
        ]
    )
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 and y.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=stratify)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }


def discover_coordinates(
    df: pd.DataFrame,
    target_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Rank candidate latent coordinates by predictive value for targets."""

    target_cols = target_cols or ["t", "r"]
    features_df = extract_features(df)
    for target in target_cols:
        if target not in features_df.columns:
            raise ValueError(f"target column not found: {target}")

    features = _candidate_features(features_df, target_cols)
    targets: dict[str, pd.Series] = {target: features_df[target] for target in target_cols}
    targets["combined"] = _combined_target(features_df, target_cols)

    target_reports: dict[str, Any] = {}
    for target_name, target_values in targets.items():
        rankings = [_single_feature_score(features_df, feature, target_values) for feature in features]
        rankings.sort(key=lambda row: (row["mutual_info"], row["tree_macro_f1"], row["tree_accuracy"]), reverse=True)
        target_reports[target_name] = {
            "rankings": rankings,
            "baseline": _baseline(features_df, features, target_values),
        }

    identified = {
        "t_coordinate": target_reports.get("t", {}).get("rankings", [{}])[0].get("feature"),
        "r_coordinate": target_reports.get("r", {}).get("rankings", [{}])[0].get("feature"),
        "a18_predictive_value_for_r": next(
            (row for row in target_reports.get("r", {}).get("rankings", []) if row["feature"] in {"a18", "a18_abs"}),
            None,
        ),
        "reverse_orientation": next(
            (row for row in target_reports.get("r", {}).get("rankings", []) if row["feature"] == "reverse_resonance"),
            None,
        ),
    }
    return {
        "n_rows": int(len(features_df)),
        "target_cols": target_cols,
        "candidate_features": features,
        "targets": target_reports,
        "identified_coordinates": identified,
    }
