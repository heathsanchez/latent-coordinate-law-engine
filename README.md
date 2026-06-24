# Latent Coordinate Law Engine

Latent Coordinate Law:
A system becomes predictable when represented in coordinates where observations factorize.
This engine tests that claim by discovering coordinates, inducing threshold laws, predicting held-out cases, and searching for counterexamples.

The reusable pipeline is:

```text
raw observations
-> interventions
-> feature extraction
-> latent coordinate discovery
-> law induction
-> held-out prediction
-> counterexample search
-> Lawbook export
```

The included synthetic generator creates IGP24-style polynomial records with `coeff_0` through `coeff_24`, support-face structure, shell behavior in `a6`, inert `a18` variation, controlled noise, and structured holdouts.

## Quickstart

```bash
pip install -e .
pytest
latent-law demo --out out/
latent-law benchmark --out conclusive_out/
latent-law a-plus-plus --out a_plus_plus_out/
```

## Python Usage

```python
from latent_law.data import generate_igp24_synthetic
from latent_law.features import extract_features
from latent_law.discovery import discover_coordinates
from latent_law.laws import induce_laws

df = generate_igp24_synthetic(n=500, seed=3)
features = extract_features(df)
report = discover_coordinates(features, target_cols=["t", "r"])
laws = induce_laws(features, target="r")
```

## CLI

```bash
latent-law demo --out out/
latent-law discover --csv data.csv --out out/
latent-law benchmark --igp24-csv real_dataset.csv --out conclusive_out/
latent-law a-plus-plus --out a_plus_plus_out/
latent-law test
```

The demo writes:

- `dataset.csv`
- `coordinate_report.json`
- `lawbook.json`
- `holdout_report.json`
- `counterexamples.csv`
- `summary.md`

## Generalization Benchmark

The conclusive benchmark command runs the same coordinate discovery protocol across:

- IGP24 polynomial records
- ETP-style equation statistics
- ARC-style object/topology features
- maze/search representations
- elementary cellular automata
- hidden-threshold phase systems

It writes:

- `lawbook.json`
- `thresholds.json`
- `invariants.json`
- `coordinate_rankings.csv`
- `transfer_results.csv`
- `compression_results.csv`
- `counterexamples.csv`
- `benchmark_report.md`
- `final_conclusion.md`

The stricter A++ challenge adds withheld-coordinate rediscovery probes, unknown-prediction splits, human-comparison baselines, and live dataset-source manifests:

```bash
latent-law a-plus-plus --out a_plus_plus_out/
```

It writes `a_plus_plus_report.md`, `a_plus_plus_scores.csv`, `scientific_rediscovery.csv`, `unknown_prediction.csv`, `human_comparison.csv`, and `dataset_sources.json`. Network-backed real benchmark downloads are available through `--download`, subject to the execution environment's network permissions.
