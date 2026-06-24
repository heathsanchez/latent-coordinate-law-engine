from pathlib import Path

import numpy as np

from latent_law.etp import equation_features, generate_etp_from_equations, pair_equation_features


FORBIDDEN = ("affine", "projection", "idempotent", "idempotence")


def test_equation_features_are_raw_structure_only():
    features = equation_features("x = (y ◇ x) ◇ (x ◇ z)")

    assert features["eq_depth"] == 1
    assert features["eq_variable_count"] == 3
    assert features["eq_repeated_variable_count"] >= 1
    assert not any(term in column for column in features for term in FORBIDDEN)


def test_pair_features_are_raw_structure_only():
    features = pair_equation_features("x = x ◇ y", "x = (y ◇ x) ◇ z")

    assert features["pair_shared_variable_count"] == 2
    assert features["pair_total_ops"] == 3
    assert not any(term in column for column in features for term in FORBIDDEN)


def test_generate_etp_from_local_equations_and_matrix(tmp_path: Path):
    equations = [
        "x = x",
        "x = y",
        "x = x ◇ x",
        "x = (y ◇ x) ◇ (x ◇ z)",
    ]
    equations_path = tmp_path / "equations.txt"
    equations_path.write_text("\n".join(equations), encoding="utf-8")
    matrix = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, matrix)

    df = generate_etp_from_equations(equations_path, matrix_path, n=30, seed=3)

    assert len(df) == 30
    assert {"premise_equation", "conclusion_equation", "implication_true"}.issubset(df.columns)
    assert df["implication_true"].nunique() == 2
    assert not any(term in column for column in df.columns for term in FORBIDDEN)
