"""Paired equal-size coverage acceptance, with no classifier-based selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OLD = Path("results/dclde_target_gap_diagnostic.json")
NEW = Path("results/dclde_empirical_target_gap_diagnostic.json")
OUTPUT = Path("results/dclde_empirical_target_acceptance.json")


def compare() -> dict:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    new = json.loads(NEW.read_text(encoding="utf-8"))
    for key in ("real_events", "same_WAV_clean_controls", "broad_test_clean_controls",
                "synthetic_whale_mixtures", "synthetic_draws_per_real_event",
                "classifier_checkpoint_sha256", "classifier_input_samples"):
        if old["protocol"][key] != new["protocol"][key]:
            raise ValueError(f"mismatched diagnostic cohort: {key}")
    if old["coverage"]["equal_size_repeats"] != new["coverage"]["equal_size_repeats"]:
        raise ValueError("unequal size-matched coverage repetitions")
    old_nn = np.asarray(old["coverage"]["equal_size_nn_median_across_repeats"])
    new_nn = np.asarray(new["coverage"]["equal_size_nn_median_across_repeats"])
    old_out = np.asarray(old["coverage"]["equal_size_outside_fraction_across_repeats"])
    new_out = np.asarray(new["coverage"]["equal_size_outside_fraction_across_repeats"])
    result = {
        "real_event_count": old["protocol"]["real_events"],
        "equal_size_candidates_per_repeat": old["coverage"]["equal_size_reference_count"],
        "repeats": len(old_nn),
        "old_equal_size_nn_median": float(np.median(old_nn)),
        "new_equal_size_nn_median": float(np.median(new_nn)),
        "paired_repeat_nn_delta_median": float(np.median(new_nn-old_nn)),
        "repeats_with_lower_nn_distance": int(np.sum(new_nn < old_nn)),
        "old_outside_p95_fraction_median": float(np.median(old_out)),
        "new_outside_p95_fraction_median": float(np.median(new_out)),
        "repeats_with_lower_outside_fraction": int(np.sum(new_out < old_out)),
        "old_synthetic_mean_classifier_probability": old["classifier_probability"]["synthetic_whale_mixture_mean"],
        "new_synthetic_mean_classifier_probability": new["classifier_probability"]["synthetic_whale_mixture_mean"],
        "classifier_used_for_pool_calibration_or_sample_selection": False,
        "acceptance": "pass" if (np.median(new_nn) < .35
                                and np.median(new_out) < .2
                                and int(np.sum(new_nn < old_nn)) >= 24) else "fail",
        "limitations": "classifier-space proxy; no isolated real whale GT; preceding clean WAV is an imperfect current-background reference",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
