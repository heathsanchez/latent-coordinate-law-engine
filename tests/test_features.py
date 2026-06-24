import pandas as pd

from latent_law.features import extract_features


def test_feature_extraction_derives_support_and_a6_a18_fields():
    df = pd.DataFrame(
        [
            {
                **{f"coeff_{i}": 0 for i in range(25)},
                "coeff_12": 111,
                "coeff_6": -72,
                "coeff_18": 35,
                "t": 24979,
                "r": 2,
            }
        ]
    )

    features = extract_features(df)
    row = features.iloc[0]

    assert row["support_index"] == 12
    assert row["support_amplitude"] == 111
    assert row["abs_support_amplitude"] == 111
    assert row["a6"] == -72
    assert row["a18"] == 35
    assert bool(row["a6_negative"]) is True
    assert bool(row["a6_positive"]) is False
    assert row["a6_abs"] == 72
    assert row["a18_abs"] == 35
    assert row["support_nonzero_count"] == 1
    assert row["active_support_slots"] == "12"
    assert row["support_face"] == "x12_base"
    assert row["threshold_zone"] == "low"
