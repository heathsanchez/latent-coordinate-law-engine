from latent_law.data import generate_igp24_synthetic
from latent_law.discovery import discover_coordinates
from latent_law.features import extract_features
from latent_law.laws import induce_laws


df = generate_igp24_synthetic(n=400, seed=12)
features = extract_features(df)
report = discover_coordinates(features)
laws = induce_laws(features, target="r")

print(report["identified_coordinates"])
for law in laws[:5]:
    print(law.statement, law.precision, law.recall)
