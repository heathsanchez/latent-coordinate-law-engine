from latent_law.counterexamples import search_counterexamples
from latent_law.data import generate_igp24_synthetic
from latent_law.laws import Law


def test_counterexample_search_returns_zero_on_clean_data():
    df = generate_igp24_synthetic(n=500, seed=8, holdout_mode="none")
    law = Law(
        name="high_shell",
        target="r",
        condition={"feature": "a6", "op": ">=", "value": 72},
        predicted_value=6,
        precision=1.0,
        recall=1.0,
        support=1,
        exceptions=[],
        confidence="high",
        statement="if a6 >= 72 then r = 6",
    )

    assert search_counterexamples(law, df).empty


def test_counterexample_search_returns_exceptions_with_noise():
    df = generate_igp24_synthetic(n=1000, noise_rate=0.35, seed=9, holdout_mode="none")
    law = Law(
        name="high_shell",
        target="r",
        condition={"feature": "a6", "op": ">=", "value": 72},
        predicted_value=6,
        precision=1.0,
        recall=1.0,
        support=1,
        exceptions=[],
        confidence="high",
        statement="if a6 >= 72 then r = 6",
    )

    counterexamples = search_counterexamples(law, df)
    assert not counterexamples.empty
    assert (counterexamples["a6"] >= 72).all()
    assert (counterexamples["r"] != 6).all()
