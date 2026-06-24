from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


COEFF_COLUMNS = [f"coeff_{i}" for i in range(25)]


def _target_t(support_index: int) -> int:
    return 25000 if support_index in {13, 15, 17} else 24979


def _target_r(a6: int) -> int:
    if a6 == 71:
        return 4
    if a6 >= 72:
        return 6
    return 2


def generate_igp24_synthetic(
    n: int,
    noise_rate: float = 0.0,
    seed: int = 0,
    holdout_mode: Literal["structured", "random", "none"] = "structured",
) -> pd.DataFrame:
    """Generate synthetic polynomial-like records with IGP24-style laws.

    The generator keeps support-face, a6 shell, and a18 variation separable so
    downstream discovery can identify that support predicts t, a6 predicts r,
    and a18 alone is inert.
    """

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    support_choices = np.array([12, 13, 15, 17])
    a6_choices = np.array([-90, -72, -30, 0, 20, 70, 71, 72, 85, 110])
    a18_choices = np.array([-140, -40, 0, 35, 71, 120])

    for i in range(n):
        support_index = int(support_choices[i % len(support_choices)])
        if i >= len(support_choices):
            support_index = int(rng.choice(support_choices))

        a6 = int(a6_choices[(i // len(support_choices)) % len(a6_choices)])
        if i >= len(support_choices) * len(a6_choices):
            a6 = int(rng.choice(a6_choices))

        a18 = int(a18_choices[(i // (len(support_choices) * len(a6_choices))) % len(a18_choices)])
        if i >= len(support_choices) * len(a6_choices) * len(a18_choices):
            a18 = int(rng.choice(a18_choices))

        coeffs = {col: 0 for col in COEFF_COLUMNS}
        support_amplitude = int(rng.choice([-1, 1]) * rng.integers(95, 141))
        coeffs[f"coeff_{support_index}"] = support_amplitude
        coeffs["coeff_6"] = a6
        coeffs["coeff_18"] = a18

        t = _target_t(support_index)
        r = _target_r(a6)
        label = f"t={t};r={r}"

        if noise_rate > 0 and rng.random() < noise_rate:
            if rng.random() < 0.5:
                t = 24979 if t == 25000 else 25000
            else:
                r = int(rng.choice([v for v in [2, 4, 6] if v != r]))
            label = f"noisy:t={t};r={r}"

        if holdout_mode == "structured":
            holdout = bool((support_index == 17 and a6 >= 72) or (support_index == 12 and a6 == 71))
        elif holdout_mode == "random":
            holdout = bool(rng.random() < 0.2)
        elif holdout_mode == "none":
            holdout = False
        else:
            raise ValueError(f"unknown holdout_mode: {holdout_mode}")

        rows.append(
            {
                **coeffs,
                "support_index": support_index,
                "support_amplitude": support_amplitude,
                "a6": a6,
                "a18": a18,
                "t": t,
                "r": r,
                "label": label,
                "experiment": "igp24_synthetic",
                "holdout": holdout,
            }
        )

    df = pd.DataFrame(rows)
    if holdout_mode == "structured" and n > 0 and not df["holdout"].any():
        df.loc[df.index[-1], "holdout"] = True
    return df
