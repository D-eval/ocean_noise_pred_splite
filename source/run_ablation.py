"""Run all registered variants and calculate paired improvements over baseline."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .metrics import paired_bootstrap_interval


def _read_examples(path: Path) -> dict[tuple[str, str], dict]:
    with path.open(encoding="utf-8") as stream:
        return {(row["deployment"], row["timestamp"]): row for row in csv.DictReader(stream)}


def summarise(root: Path) -> dict:
    results = {
        variant: json.loads((root / variant / "result.json").read_text(encoding="utf-8"))
        for variant in ("none", "latent", "spectrum")
    }
    baseline = _read_examples(root / "none" / "test_examples.csv")
    comparisons = {}
    for variant in ("latent", "spectrum"):
        conditioned = _read_examples(root / variant / "test_examples.csv")
        keys = sorted(set(baseline) & set(conditioned))
        delta = np.asarray(
            [float(conditioned[key]["source_si_sdri_db"]) - float(baseline[key]["source_si_sdri_db"]) for key in keys]
        )
        low, high = paired_bootstrap_interval(delta)
        comparisons[variant] = {
            "paired_examples": len(keys),
            "mean_si_sdri_gain_over_unconditioned_db": float(delta.mean()),
            "paired_bootstrap_ci95_db": [low, high],
            "wins_fraction": float(np.mean(delta > 0)),
        }
    summary = {"variants": results, "comparisons": comparisons}
    (root / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/ablation"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=0.5)
    parser.add_argument("--validation-examples", type=int, default=128)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--train-examples-per-deployment", type=int)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for variant in ("none", "latent", "spectrum"):
        command = [
            sys.executable,
            "-m",
            "source.train",
            "--variant",
            variant,
            "--output",
            str(args.output / variant),
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--seconds",
            str(args.seconds),
            "--validation-examples",
            str(args.validation_examples),
            "--test-examples",
            str(args.test_examples),
            "--validate-every",
            str(args.validate_every),
            "--device",
            args.device,
        ]
        if args.train_examples_per_deployment:
            command += ["--train-examples-per-deployment", str(args.train_examples_per_deployment)]
        if args.small:
            command.append("--small")
        subprocess.run(command, check=True)
    print(json.dumps(summarise(args.output), indent=2))


if __name__ == "__main__":
    main()
