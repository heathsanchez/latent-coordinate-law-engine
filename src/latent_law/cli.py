from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from latent_law.counterexamples import search_counterexamples
from latent_law.data import generate_igp24_synthetic
from latent_law.discovery import discover_coordinates
from latent_law.evaluation import evaluate_holdout
from latent_law.features import extract_features
from latent_law.laws import induce_laws
from latent_law.reporting import export_lawbook, write_json, write_summary
from latent_law.benchmarks import run_conclusive_benchmark, run_coordinate_invention_probe
from latent_law.challenge import run_a_plus_plus_challenge


def _run_pipeline(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    features = extract_features(df)
    coordinate_report = discover_coordinates(features, target_cols=["t", "r"])
    laws = induce_laws(features, target="t", min_precision=0.9, min_recall=0.4)
    laws.extend(induce_laws(features, target="r", min_precision=0.9, min_recall=0.4))

    holdout_mask = features["holdout"].astype(bool) if "holdout" in features.columns else pd.Series(False, index=features.index)
    if not holdout_mask.any():
        holdout_mask = pd.Series(False, index=features.index)
        holdout_mask.loc[features.sample(frac=0.2, random_state=0).index] = True
    holdout_report = evaluate_holdout(features.loc[~holdout_mask], features.loc[holdout_mask])

    counterexample_frames = [search_counterexamples(law, features) for law in laws]
    counterexamples = pd.concat(counterexample_frames, axis=0).drop_duplicates() if counterexample_frames else features.iloc[0:0]

    features.to_csv(out / "dataset.csv", index=False)
    write_json(coordinate_report, str(out / "coordinate_report.json"))
    export_lawbook(laws, str(out / "lawbook.json"))
    write_json(holdout_report, str(out / "holdout_report.json"))
    counterexamples.to_csv(out / "counterexamples.csv", index=False)
    write_summary(str(out / "summary.md"), coordinate_report, laws, holdout_report, counterexamples)


def demo(args: argparse.Namespace) -> int:
    df = generate_igp24_synthetic(n=args.n, noise_rate=args.noise_rate, seed=args.seed)
    _run_pipeline(df, Path(args.out))
    return 0


def discover(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.csv)
    _run_pipeline(df, Path(args.out))
    return 0


def test(_: argparse.Namespace) -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-q"])


def benchmark(args: argparse.Namespace) -> int:
    run_conclusive_benchmark(out=args.out, igp24_csv=args.igp24_csv, seed=args.seed)
    run_coordinate_invention_probe(out=args.out, seed=args.seed)
    return 0


def a_plus_plus(args: argparse.Namespace) -> int:
    run_a_plus_plus_challenge(out=args.out, seed=args.seed, allow_download=args.download)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="latent-law")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--out", default="out/")
    demo_parser.add_argument("--n", type=int, default=600)
    demo_parser.add_argument("--noise-rate", type=float, default=0.0)
    demo_parser.add_argument("--seed", type=int, default=0)
    demo_parser.set_defaults(func=demo)

    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--csv", required=True)
    discover_parser.add_argument("--out", default="out/")
    discover_parser.set_defaults(func=discover)

    test_parser = subparsers.add_parser("test")
    test_parser.set_defaults(func=test)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--out", default="benchmark_out/")
    benchmark_parser.add_argument("--igp24-csv", default=None)
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.set_defaults(func=benchmark)

    app_parser = subparsers.add_parser("a-plus-plus")
    app_parser.add_argument("--out", default="a_plus_plus_out/")
    app_parser.add_argument("--seed", type=int, default=0)
    app_parser.add_argument("--download", action="store_true")
    app_parser.set_defaults(func=a_plus_plus)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
