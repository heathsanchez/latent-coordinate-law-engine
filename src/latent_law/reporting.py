from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from latent_law.laws import Law


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def export_lawbook(laws: list[Law], path: str) -> None:
    payload = {
        "lawbook_version": "1.0",
        "laws": [
            {
                "name": law.name,
                "target": law.target,
                "statement": law.statement,
                "condition": law.condition,
                "predicted_value": law.predicted_value,
                "precision": law.precision,
                "recall": law.recall,
                "support": law.support,
                "exceptions": law.exceptions,
                "confidence": law.confidence,
            }
            for law in laws
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_json(payload: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_summary(
    path: str,
    coordinate_report: dict[str, Any],
    laws: list[Law],
    holdout_report: dict[str, Any],
    counterexamples: pd.DataFrame,
) -> None:
    lines = [
        "# Latent Coordinate Law Demo Summary",
        "",
        f"Rows analyzed: {coordinate_report.get('n_rows')}",
        f"Top t coordinate: {coordinate_report['identified_coordinates'].get('t_coordinate')}",
        f"Top r coordinate: {coordinate_report['identified_coordinates'].get('r_coordinate')}",
        f"Induced laws: {len(laws)}",
        f"Combined holdout accuracy: {holdout_report['combined']['accuracy']:.3f}",
        f"Counterexamples found: {len(counterexamples)}",
        "",
        "## Laws",
        "",
    ]
    lines.extend(f"- {law.statement} (precision={law.precision:.3f}, recall={law.recall:.3f})" for law in laws[:12])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
