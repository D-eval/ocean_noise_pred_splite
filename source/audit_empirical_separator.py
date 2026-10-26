"""Check regression on the original fixed synthetic test, by source family."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .synthesis import draw_injection


OUTPUT = Path("results/dclde_empirical_residual_legacy_test_audit.json")


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize(names: tuple[str, ...] = ("persistence", "forecast")) -> dict:
    if not names or any(name not in {"persistence", "forecast"} for name in names):
        raise ValueError("choose persistence and/or forecast")
    result = {"protocol": "unchanged 4,096 DCLDE test mixtures; original seven-family whale/vessel generator",
              "training_target_pool": "50% procedural (calibrated whale or retained vessel), 50% train-day pseudo-real whale",
              "arms": {}}
    seed = 20261022 + 20_000_000
    for name in names:
        old = read(Path(f"experiments/dclde_{name}_residual_128/clean_synthetic_test.csv"))
        new = read(Path(f"experiments/dclde_empirical_{name}_residual_128/clean_synthetic_test.csv"))
        if len(old) != len(new) or len(new) != 4096:
            raise ValueError("old/new test row counts differ")
        groups = {}
        paired = []
        for item, (prior, updated) in enumerate(zip(old, new, strict=True)):
            if prior["example_id"] != updated["example_id"]:
                raise ValueError("old/new test examples differ")
            for field in ("day", "snr_db", "mixture_si_sdr_db", "bsr_si_sdri_db"):
                if field == "day":
                    if prior[field] != updated[field]:
                        raise ValueError("test day mismatch")
                elif abs(float(prior[field])-float(updated[field])) > 1e-4:
                    raise ValueError(f"test mixture mismatch: {field}")
            family = draw_injection(seed+item*104729, snr_range=(0.0, 0.0),
                                    profile="marine").family
            old_score = float(prior[f"{name}_si_sdri_db"])
            new_score = float(updated[f"{name}_si_sdri_db"])
            groups.setdefault(family, []).append((old_score, new_score))
            paired.append(new_score-old_score)
        result["arms"][name] = {
            "old_mean_si_sdri_db": float(np.mean([float(row[f"{name}_si_sdri_db"])
                                                 for row in old])),
            "new_mean_si_sdri_db": float(np.mean([float(row[f"{name}_si_sdri_db"])
                                                 for row in new])),
            "new_minus_old_mean_db": float(np.mean(paired)),
            "families": {family: {"n": len(values),
                                  "old_mean_db": float(np.mean(values, axis=0)[0]),
                                  "new_mean_db": float(np.mean(values, axis=0)[1])}
                         for family, values in sorted(groups.items())},
        }
    output = (OUTPUT if len(names) == 2 else OUTPUT.with_name(
        "dclde_empirical_" + names[0] + "_legacy_test_audit.json"))
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("persistence", "forecast"))
    args = parser.parse_args()
    print(json.dumps(summarize((args.only,) if args.only else
                               ("persistence", "forecast")), indent=2))
