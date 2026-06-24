from pathlib import Path

from latent_law.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["demo", "--out", str(Path("out_v130_like")), "--n", "800", "--seed", "130"]))
