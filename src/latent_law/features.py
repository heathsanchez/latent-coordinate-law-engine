from __future__ import annotations

import numpy as np
import pandas as pd


COEFF_COLUMNS = [f"coeff_{i}" for i in range(25)]
SUPPORT_EXCLUDED_INDICES = {6, 18}


def _ensure_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in COEFF_COLUMNS:
        if col not in out.columns:
            out[col] = 0
    return out


def _support_candidate_indices(df: pd.DataFrame) -> list[int]:
    candidates = [
        i
        for i in range(25)
        if i not in SUPPORT_EXCLUDED_INDICES and df[f"coeff_{i}"].nunique(dropna=False) > 1
    ]
    if candidates:
        return candidates
    return [i for i in range(25) if i not in SUPPORT_EXCLUDED_INDICES]


def _derive_support(row: pd.Series, candidates: list[int]) -> tuple[int, float]:
    values = row[[f"coeff_{i}" for i in candidates]].astype(float)
    abs_values = values.abs()
    if float(abs_values.max()) == 0:
        return -1, 0.0
    col = str(abs_values.idxmax())
    index = int(col.split("_")[1])
    return index, float(row[col])


def _support_face(index: int) -> str:
    if index == 12:
        return "x12_base"
    if index in {13, 15, 17}:
        return "x13_x15_x17"
    if index == -1:
        return "none"
    return "other"


def _threshold_zone(a6: float) -> str:
    if a6 == 71:
        return "boundary"
    if a6 >= 72:
        return "high"
    return "low"


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature-enriched copy of polynomial-like observations."""

    out = _ensure_coefficients(df)
    support_candidates = _support_candidate_indices(out)
    support = out.apply(lambda row: _derive_support(row, support_candidates), axis=1, result_type="expand")
    derived_support_index = support[0].astype(int)
    derived_support_amplitude = support[1]

    if "support_index" not in out.columns:
        out["support_index"] = derived_support_index
    else:
        out["support_index"] = out["support_index"].fillna(derived_support_index).astype(int)

    if "support_amplitude" not in out.columns:
        out["support_amplitude"] = derived_support_amplitude
    else:
        out["support_amplitude"] = out["support_amplitude"].fillna(derived_support_amplitude)

    if "a6" not in out.columns:
        out["a6"] = out["coeff_6"]
    else:
        out["a6"] = out["a6"].fillna(out["coeff_6"])

    if "a18" not in out.columns:
        out["a18"] = out["coeff_18"]
    else:
        out["a18"] = out["a18"].fillna(out["coeff_18"])

    coeff_values = out[COEFF_COLUMNS].astype(float)
    out["abs_support_amplitude"] = out["support_amplitude"].abs()
    out["a6_positive"] = out["a6"] > 0
    out["a6_negative"] = out["a6"] < 0
    out["a6_abs"] = out["a6"].abs()
    out["a18_abs"] = out["a18"].abs()
    support_columns = [f"coeff_{i}" for i in support_candidates]
    out["support_nonzero_count"] = coeff_values[support_columns].ne(0).sum(axis=1)
    out["active_support_slots"] = coeff_values[support_columns].apply(
        lambda row: ",".join(str(int(col.split("_")[1])) for col, value in row.items() if value != 0),
        axis=1,
    )
    out["paired_resonance"] = (out["a6"] > 0) & (out["a18"] != 0)
    out["reverse_resonance"] = out["a6"] < 0
    out["support_face"] = out["support_index"].map(_support_face)
    out["support_bucket"] = np.where(out["support_index"].isin([12, 13, 15, 17]), out["support_face"], "other")
    out["threshold_zone"] = out["a6"].map(_threshold_zone)
    return out
