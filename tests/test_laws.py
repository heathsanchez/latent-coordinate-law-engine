from latent_law.data import generate_igp24_synthetic
from latent_law.features import extract_features
from latent_law.laws import induce_laws


def _has_condition(laws, feature, op, value, predicted):
    return any(
        law.condition.get("feature") == feature
        and law.condition.get("op") == op
        and law.condition.get("value") == value
        and law.predicted_value == predicted
        for law in laws
    )


def test_synthetic_generator_obeys_clean_law():
    df = generate_igp24_synthetic(n=500, seed=3, holdout_mode="none")
    features = extract_features(df)

    assert (features.loc[features["support_index"] == 12, "t"] == 24979).all()
    assert (features.loc[features["support_index"].isin([13, 15, 17]), "t"] == 25000).all()
    assert (features.loc[features["a6"] <= 70, "r"] == 2).all()
    assert (features.loc[features["a6"] == 71, "r"] == 4).all()
    assert (features.loc[features["a6"] >= 72, "r"] == 6).all()


def test_law_induction_discovers_a6_high_shell():
    df = generate_igp24_synthetic(n=700, seed=4, holdout_mode="none")
    laws = induce_laws(df, target="r")

    assert _has_condition(laws, "a6", ">=", 72, 6)


def test_law_induction_discovers_a6_low_shell():
    df = generate_igp24_synthetic(n=700, seed=5, holdout_mode="none")
    laws = induce_laws(df, target="r")

    assert _has_condition(laws, "a6", "<=", 70, 2)


def test_negative_a6_does_not_trigger_r6():
    df = extract_features(generate_igp24_synthetic(n=500, seed=6, holdout_mode="none"))
    negative = df[df["a6"] < 0]

    assert not negative.empty
    assert (negative["r"] != 6).all()


def test_a18_only_variation_does_not_trigger_r6():
    df = extract_features(generate_igp24_synthetic(n=800, seed=7, holdout_mode="none"))
    inert = df[df["a6"] <= 70]

    assert inert["a18"].nunique() > 1
    assert (inert["r"] == 2).all()
