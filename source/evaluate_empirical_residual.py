"""Evaluate 50/50 train-day target-pool separators on untouched real calls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dclde import (CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE,
                    EVENT_MASKED_FEATURE_CACHE, MANIFEST)
from .dclde_causal_highres_experiment import (CausalResidualBSR, HighResGRU,
                                               HighResolutionStore)
from .dclde_real_residual_eval import (CLASSIFIER_PATH, METRICS, PREDICTOR_PATH,
                                        RealResidualEventDataset, file_sha256)
from .dclde_separation_experiment import CausalFeatureStore
from .event_classifier import load_classifier
from .metrics import paired_bootstrap_interval
from .reference_free_eval import restoration_metrics


NAMES = ("persistence", "forecast")
OLD_CSV = Path("results/dclde_real_residual_event_results.csv")
OUTPUT = Path("results/dclde_empirical_residual_real_event_results.json")


def paired_summary(rows: list[dict], name: str, metric: str) -> dict:
    days = np.asarray([row["day"] for row in rows])
    difference = np.asarray([float(row[f"new_{name}_{metric}"]) -
                             float(row[f"old_{name}_{metric}"]) for row in rows])
    by_day = {str(day): float(difference[days == day].mean())
              for day in np.unique(days)}
    return {"mean": float(difference.mean()),
            "day_block_ci95": [float(value) for value in
                               paired_bootstrap_interval(np.asarray(list(by_day.values())))],
            "by_day": by_day}


def paired_new_conditions(rows: list[dict], metric: str) -> dict:
    days = np.asarray([row["day"] for row in rows])
    difference = np.asarray([float(row[f"new_forecast_{metric}"]) -
                             float(row[f"new_persistence_{metric}"]) for row in rows])
    by_day = {str(day): float(difference[days == day].mean())
              for day in np.unique(days)}
    return {"mean": float(difference.mean()),
            "day_block_ci95": [float(value) for value in
                               paired_bootstrap_interval(np.asarray(list(by_day.values())))],
            "by_day": by_day}


@torch.inference_mode()
def evaluate(device_name: str = "cuda", batch_size: int = 8,
             names: tuple[str, ...] = NAMES) -> dict:
    if not names or any(name not in NAMES for name in names):
        raise ValueError("choose persistence and/or forecast")
    device = torch.device(device_name)
    event_store = CausalFeatureStore(MANIFEST, EVENT_MASKED_FEATURE_CACHE)
    clean_store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    cache = HighResolutionStore(clean_store)
    dataset = RealResidualEventDataset(event_store, cache)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    forecaster = HighResGRU(cache.mean, cache.std).to(device)
    forecaster.load_state_dict(torch.load(PREDICTOR_PATH, map_location=device,
                                          weights_only=False)["model"], strict=True)
    forecaster.eval()
    models = {}
    checkpoint_paths = {}
    for name in names:
        path = Path(f"models/empirical_{name}_residual_128.pt")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if (checkpoint.get("target_pool") != "empirical_50_50"
                or checkpoint.get("interface") != f"{name}_residual_128"):
            raise ValueError(f"{name} checkpoint does not match the empirical protocol")
        model = CausalResidualBSR(clean_store.bands, 20261022, device,
                                  Path("models/practical_real.pt")).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        models[name] = model.eval()
        checkpoint_paths[name] = path
    classifier, temperature = load_classifier(CLASSIFIER_PATH, device)
    classifier.eval()
    with OLD_CSV.open(newline="", encoding="utf-8") as stream:
        old_rows = list(csv.DictReader(stream))
    rows = []
    for batch in loader:
        mixture = batch["waveform"].to(device)
        event_mask = batch["event_mask"].to(device)
        history = batch["history_map"].to(device)
        conditions = {"persistence": history[:, -1],
                      "forecast": forecaster(history)}
        per_example = {}
        for name in names:
            estimate = models[name](mixture, conditions[name])
            length = min(estimate.shape[-1], mixture.shape[-1])
            _, _, examples = restoration_metrics(
                classifier, mixture[..., :length], estimate[..., :length],
                event_mask[..., :length], temperature, return_per_example=True)
            per_example[name] = examples
        for item, example_id in enumerate(batch["example_id"]):
            previous = old_rows[len(rows)]
            if previous["example_id"] != example_id:
                raise ValueError("new and old real-event cohorts differ")
            row = {"example_id": example_id, "day": previous["day"]}
            for name in names:
                for metric in METRICS:
                    row[f"old_{name}_{metric}"] = float(previous[f"{name}_{metric}"])
                    row[f"new_{name}_{metric}"] = float(per_example[name][metric][item])
            rows.append(row)
        print(json.dumps({"evaluated": len(rows), "total": len(dataset)}), flush=True)
    result = {
        "protocol": {"real_event_examples": len(rows),
                     "days": sorted({row["day"] for row in rows}),
                     "target_pool": "50% calibrated procedural, 50% train-day pseudo-real",
                     "condition_predictor_retrained": False,
                     "classifier_retrained": False,
                     "real_stem_gt": False,
                     "same_cohort_as_old": True,
                     "predictor_checkpoint_sha256": file_sha256(PREDICTOR_PATH),
                     "classifier_checkpoint_sha256": file_sha256(CLASSIFIER_PATH),
                     "separator_checkpoint_sha256": {name: file_sha256(path)
                                                     for name, path in checkpoint_paths.items()}},
        "means": {name: {version: {metric: float(np.mean([
            row[f"{version}_{name}_{metric}"] for row in rows]))
            for metric in METRICS} for version in ("old", "new")}
            for name in names},
        "paired_new_minus_old": {name: {metric: paired_summary(rows, name, metric)
                                         for metric in METRICS} for name in names},
    }
    if names == NAMES:
        result["paired_new_forecast_minus_persistence"] = {
            metric: paired_new_conditions(rows, metric) for metric in METRICS}
    output = (OUTPUT if names == NAMES else OUTPUT.with_name(
        "dclde_empirical_" + "_".join(names) + "_real_event_results.json"))
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--only", choices=NAMES)
    args = parser.parse_args()
    result = evaluate(args.device, args.batch_size,
                      (args.only,) if args.only else NAMES)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
