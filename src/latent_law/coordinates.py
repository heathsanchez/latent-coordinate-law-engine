from __future__ import annotations

import itertools
import re
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_EXCLUDED = {
    "domain",
    "label",
    "experiment",
    "holdout",
    "run",
    "description",
    "status",
    "map_cost",
    "route_cost",
    "lowest_search_complexity",
}


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")


def _numeric_base_columns(df: pd.DataFrame, targets: list[str], max_base_features: int) -> list[str]:
    excluded = set(targets) | DEFAULT_EXCLUDED
    numeric = [
        col
        for col in df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique(dropna=False) > 1
        and not col.startswith("coeff_")
    ]
    # Prefer continuous and moderately varying columns; binary flags are still allowed later if capacity remains.
    numeric.sort(key=lambda col: (df[col].nunique(dropna=False), df[col].std() if df[col].std() == df[col].std() else 0), reverse=True)
    return numeric[:max_base_features]


def synthesize_coordinates(
    df: pd.DataFrame,
    targets: list[str],
    max_base_features: int = 8,
    max_new_features: int = 160,
    second_order: bool = True,
    include_pca: bool = True,
) -> pd.DataFrame:
    """Create generic candidate coordinates from existing numeric observations.

    This intentionally knows nothing about a domain. It proposes algebraic
    coordinates such as ratios, products, differences, absolute values, and a
    limited second-order ratio of a product by another base variable.
    """

    warnings.filterwarnings("ignore", category=PerformanceWarning)
    out = df.copy()
    bases = _numeric_base_columns(out, targets, max_base_features=max_base_features)
    created: list[str] = []

    for col in bases:
        safe = _safe_name(col)
        series = out[col].astype(float)
        specs = [
            (f"coord_abs__{safe}", series.abs()),
            (f"coord_square__{safe}", series**2),
            (f"coord_sqrt__{safe}", np.sqrt(series.clip(lower=0))),
            (f"coord_log__{safe}", np.log1p(series.clip(lower=0))),
        ]
        if float(series.min()) >= 0 and float(series.max()) <= 1 and series.nunique(dropna=False) > 2:
            clipped = series.clip(1e-9, 1 - 1e-9)
            entropy = -(clipped * np.log2(clipped) + (1 - clipped) * np.log2(1 - clipped))
            specs.append((f"coord_entropy__{safe}", entropy))
        for name, values in specs:
            out[name] = values.replace([np.inf, -np.inf], np.nan).fillna(0)
            if out[name].nunique(dropna=False) <= 1:
                out = out.drop(columns=[name])
                continue
            created.append(name)
            if len(created) >= max_new_features:
                return out

    for a, b in itertools.combinations(bases, 2):
        a_name = _safe_name(a)
        b_name = _safe_name(b)
        specs = [
            (f"coord_diff__{a_name}__minus__{b_name}", out[a] - out[b]),
            (f"coord_sum__{a_name}__plus__{b_name}", out[a] + out[b]),
            (f"coord_mul__{a_name}__times__{b_name}", out[a] * out[b]),
        ]
        denom = out[b].replace(0, np.nan)
        specs.append((f"coord_ratio__{a_name}__over__{b_name}", out[a] / denom))
        denom = out[a].replace(0, np.nan)
        specs.append((f"coord_ratio__{b_name}__over__{a_name}", out[b] / denom))
        for name, values in specs:
            out[name] = values.replace([np.inf, -np.inf], np.nan).fillna(0)
            if out[name].nunique(dropna=False) <= 1:
                out = out.drop(columns=[name])
                continue
            created.append(name)
            if len(created) >= max_new_features:
                return out

    if not second_order:
        return out

    product_cols = [col for col in created if col.startswith("coord_mul__")]
    for product_col in product_cols:
        for denom_col in bases:
            if denom_col not in product_col:
                name = f"{product_col}__over__{_safe_name(denom_col)}"
                denom = out[denom_col].replace(0, np.nan)
                out[name] = (out[product_col] / denom).replace([np.inf, -np.inf], np.nan).fillna(0)
                if out[name].nunique(dropna=False) <= 1:
                    out = out.drop(columns=[name])
                else:
                    created.append(name)
                    if len(created) >= max_new_features:
                        return out
            name = f"{product_col}__times__{_safe_name(denom_col)}"
            out[name] = (out[product_col] * out[denom_col]).replace([np.inf, -np.inf], np.nan).fillna(0)
            if out[name].nunique(dropna=False) <= 1:
                out = out.drop(columns=[name])
                continue
            created.append(name)
            if len(created) >= max_new_features:
                return out

    if include_pca and len(bases) >= 2:
        numeric = out[bases].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
        components = min(3, len(bases), max_new_features - len(created))
        if components > 0:
            transformed = PCA(n_components=components, random_state=0).fit_transform(StandardScaler().fit_transform(numeric))
            for i in range(components):
                name = f"coord_pca_{i + 1}"
                out[name] = transformed[:, i]
                if out[name].nunique(dropna=False) > 1:
                    created.append(name)
                if len(created) >= max_new_features:
                    return out
    return out
