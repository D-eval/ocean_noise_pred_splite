"""One-shot held-out audit after validation-only weak checkpoint selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dclde import CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE
from .dclde_causal_highres_experiment import (CausalResidualBSR,
    ConditionedInjectedDataset, HighResolutionStore, separation_test_rows)
from .dclde_separation_experiment import CausalFeatureStore, ExperimentConfig
from .evaluate_empirical_residual import evaluate as real_evaluate
from .dclde_real_residual_eval import CLASSIFIER_PATH, file_sha256
from .metrics import paired_bootstrap_interval
from .weak_real_finetune import OUT as FIT_OUT


OUT = Path("results/dclde_weak_real_persistence_audit.json")
BASE_REAL = Path("results/dclde_empirical_persistence_real_event_results.csv")
BASE_SYNTH = Path("experiments/dclde_empirical_persistence_residual_128/clean_synthetic_test.csv")
CLEAN = Path("results/dclde_target_gap_diagnostic.csv")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-5, 1 - 1e-5))
    return float(np.log(p / (1 - p)))


def paired(values: np.ndarray, days: np.ndarray) -> dict:
    by_day = {str(day): float(values[days == day].mean()) for day in np.unique(days)}
    return {"mean": float(values.mean()), "day_means": by_day,
            "day_bootstrap_ci95": list(paired_bootstrap_interval(np.asarray(list(by_day.values()))))}


def audit(device_name: str = "cuda") -> dict:
    checkpoint_path = FIT_OUT / "best.pt"
    selected = json.loads((FIT_OUT / "result.json").read_text(encoding="utf-8"))
    if selected["best_step"] == 0:
        raise ValueError("no weakly supervised checkpoint met validation constraints")
    device = torch.device(device_name)
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    cache = HighResolutionStore(store)
    model = CausalResidualBSR(store.bands, 20261022, device,
                              Path("models/practical_real.pt")).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    test = ConditionedInjectedDataset(cache, "test", ExperimentConfig().validation_examples,
                                      20261022 + 20_000_000, "persistence", spread_rows=True)
    synthetic_rows = separation_test_rows(model, DataLoader(test, batch_size=24,
                                                           num_workers=0), device, "persistence")
    prior_synth = read_rows(BASE_SYNTH)
    if [x["example_id"] for x in synthetic_rows] != [x["example_id"] for x in prior_synth]:
        raise ValueError("controlled synthetic cohorts do not match")
    synth_days = np.asarray([r["day"] for r in synthetic_rows])
    new_si = np.asarray([float(r["persistence_si_sdri_db"]) for r in synthetic_rows])
    old_si = np.asarray([float(r["persistence_si_sdri_db"]) for r in prior_synth])
    real_path = OUT.with_name("dclde_weak_real_persistence_event_results.json")
    real_evaluate(device_name, names=("persistence",),
                  checkpoint_overrides={"persistence": checkpoint_path},
                  output_override=real_path)
    weak = read_rows(real_path.with_suffix(".csv"))
    baseline = read_rows(BASE_REAL)
    clean = read_rows(CLEAN)
    ids = [row["example_id"] for row in weak]
    if ids != [row["example_id"] for row in baseline] or ids != [row["example_id"] for row in clean]:
        raise ValueError("real event and matched clean cohorts differ")
    real_days = np.asarray([r["day"] for r in weak])
    p_clean = np.asarray([float(r["same_WAV_clean_probability"]) for r in clean])
    def values(rows, key):
        return np.asarray([float(row[key]) for row in rows])
    metrics = {}
    for metric in ("background_event_probability", "event_stem_probability",
                   "monotonic_fraction", "off_event_relative_error"):
        current = values(weak, f"new_persistence_{metric}")
        previous = values(baseline, f"new_persistence_{metric}")
        metrics[metric] = {"pretrained_mean": float(previous.mean()),
                           "weak_finetuned_mean": float(current.mean()),
                           "weak_minus_pretrained": paired(current - previous, real_days)}
    previous_pn = values(baseline, "new_persistence_background_event_probability")
    current_pn = values(weak, "new_persistence_background_event_probability")
    old_residue = np.asarray([logit(p) - logit(c) for p, c in zip(previous_pn, p_clean)])
    new_residue = np.asarray([logit(p) - logit(c) for p, c in zip(current_pn, p_clean)])
    result = {
        "protocol": {"weak_checkpoint_selected_on_validation_only": True,
                     "train_detector_independent_of_evaluation_detector": True,
                     "real_test_events": len(weak), "controlled_test_examples": len(synthetic_rows),
                     "test_days_used_to_change_training_or_checkpoint": False,
                     "real_clean_stem_gt": False,
                     "train_detector_sha256": file_sha256(Path("models/weak_train_frame_detector.pt")),
                     "evaluation_detector_sha256": file_sha256(CLASSIFIER_PATH),
                     "separator_sha256": file_sha256(checkpoint_path)},
        "selected_step": selected["best_step"],
        "controlled_test": {"pretrained_si_sdri_db": float(old_si.mean()),
                            "weak_finetuned_si_sdri_db": float(new_si.mean()),
                            "weak_minus_pretrained_db": paired(new_si-old_si, synth_days)},
        "real_independent_evaluator": metrics,
        "matched_clean_mean_probability": float(p_clean.mean()),
        "clean_normalized_residue": {
            "definition": "logit p(background stem)-logit p(immediately prior same-WAV clean 5-s background)",
            "pretrained_mean": float(old_residue.mean()),
            "weak_finetuned_mean": float(new_residue.mean()),
            "weak_minus_pretrained": paired(new_residue-old_residue, real_days)},
        "real_rows": str(real_path.with_suffix(".csv")),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(audit(args.device), indent=2))
