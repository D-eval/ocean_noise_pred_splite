"""Frozen 128-band residual separators on the original 103 real whale calls.

No isolated real-event stems are used or assumed. The current call waveform is
only the separator mixture; all four condition maps come from training means,
another test day, or annotation-clean windows strictly before the call.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dclde import (CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE,
                    EVENT_MASKED_FEATURE_CACHE, MANIFEST)
from .dclde_causal_highres_experiment import (
    FRAMES, GROUPS, CausalResidualBSR, HighResGRU, HighResolutionStore,
)
from .dclde_separation_experiment import CausalFeatureStore, RealEventDataset
from .event_classifier import load_classifier
from .metrics import paired_bootstrap_interval
from .reference_free_eval import RESTORATION_GAINS_DB, restoration_metrics, trivial_estimates


MODEL_PATHS = {
    name: Path(f"models/{name}_residual_128.pt")
    for name in ("zero", "shuffled", "persistence", "forecast")
}
PREDICTOR_PATH = Path("models/highres_gru_128.pt")
CLASSIFIER_PATH = Path("experiments/dclde_event_classifier/best.pt")
OLD_ROWS = Path("results/dclde_real_event_results.csv")
OUTPUT = Path("results/dclde_real_residual_event_results.json")
METRICS = (
    "background_event_probability", "event_stem_probability",
    "monotonic_fraction", "off_event_relative_error",
    "removal_logit", "restoration_auc", "leakage_outside_event",
    "reconstruction_relative_error",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RealResidualEventDataset(Dataset):
    """Attach strictly past, annotation-clean 128-band maps to old call clips."""

    def __init__(self, event_store: CausalFeatureStore, cache: HighResolutionStore,
                 seed: int = 20261022):
        self.base = RealEventDataset(event_store)
        self.cache = cache
        self.seed = seed
        clean = cache.store.frame
        lookup = {(str(row.path), int(round(float(row.start_seconds) / 5))): index
                  for index, row in enumerate(clean.itertuples())}
        if len(lookup) != len(clean):
            raise ValueError("clean manifest has duplicate WAV/block keys")
        test_clean = cache.store.indices("test", label=0)
        test_days = clean.iloc[test_clean]["day"].to_numpy()
        donors = {str(day): test_clean[test_days == day] for day in np.unique(test_days)}
        self.history_rows = []
        self.shuffled_donors = []
        self.imputed_history_windows = 0
        self.imputed_examples = 0
        for item, event_index in enumerate(self.base.rows):
            row = event_store.frame.iloc[int(event_index)]
            current_block = int(round(float(row.start_seconds) / 5))
            if abs(float(row.start_seconds) - current_block * 5) > 1e-5:
                raise ValueError("real event is not aligned to a 5-s window")
            if current_block < 12:
                raise ValueError("real event has less than 60 s of same-WAV history")
            prior = [lookup.get((str(row.path), current_block - offset), -1)
                     for offset in range(12, 0, -1)]
            observed = np.flatnonzero(np.asarray(prior) >= 0)
            if not len(observed):
                raise ValueError("real event has no annotation-clean historical windows")
            missing = np.flatnonzero(np.asarray(prior) < 0)
            self.imputed_history_windows += len(missing)
            self.imputed_examples += bool(len(missing))
            for slot in missing:
                nearest = observed[np.argmin(np.abs(observed - slot))]
                prior[slot] = prior[nearest]
            for index in prior:
                clean_row = clean.iloc[index]
                if (clean_row["split"] != "test" or clean_row["label"] != 0
                        or clean_row["day"] != row.day
                        or str(clean_row["path"]) != str(row.path)
                        or float(clean_row["start_seconds"]) >= float(row.start_seconds)):
                    raise ValueError("history contains a noncausal or nonclean window")
            self.history_rows.append(np.asarray(prior, dtype=np.int32))
            candidates = sorted(day for day in donors if day != str(row.day))
            if not candidates:
                raise ValueError("Shuffled control requires a different test day")
            rng = np.random.default_rng(seed + item * 104729 + 271828)
            donor_day = candidates[int(rng.integers(len(candidates)))]
            pool = donors[donor_day]
            donor_index = int(pool[int(rng.integers(len(pool)))])
            self.shuffled_donors.append((donor_index, donor_day))
        self.history_rows = np.stack(self.history_rows)
        with OLD_ROWS.open(newline="", encoding="utf-8") as stream:
            old_ids = [row["example_id"] for row in csv.DictReader(stream)]
        new_ids = [f"{row.day}:{Path(row.path).stem}:{float(row.start_seconds):.1f}"
                   for row in event_store.frame.iloc[self.base.rows].itertuples()]
        if new_ids != old_ids or len(new_ids) != 103:
            raise ValueError("real residual evaluation differs from the prior 103-call test")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> dict:
        answer = self.base[item]
        history = np.asarray(self.cache.maps[self.history_rows[item]], dtype=np.float32)
        donor_index, donor_day = self.shuffled_donors[item]
        answer["history_map"] = torch.from_numpy(history)
        answer["shuffled_map"] = torch.from_numpy(self.cache.current(donor_index))
        answer["shuffled_donor_day"] = donor_day
        answer["shuffled_donor_manifest_index"] = donor_index
        return answer


def paired_contrast(rows: list[dict], left: str, right: str, metric: str) -> dict:
    days = np.asarray([row["day"] for row in rows])
    differences = np.asarray([float(row[f"{left}_{metric}"]) - float(row[f"{right}_{metric}"])
                              for row in rows])
    by_day = {str(day): float(differences[days == day].mean()) for day in np.unique(days)}
    interval = paired_bootstrap_interval(np.asarray(list(by_day.values())))
    return {"mean": float(differences.mean()),
            "day_block_ci95": [float(x) for x in interval], "by_day": by_day}


@torch.inference_mode()
def evaluate(device_name: str = "cuda", batch_size: int = 8) -> dict:
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
    for name, path in MODEL_PATHS.items():
        model = CausalResidualBSR(clean_store.bands, 20261022, device,
                                  Path("models/practical_real.pt")).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint["interface"] != f"{name}_residual_128":
            raise ValueError(f"{name} checkpoint has the wrong residual interface")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models[name] = model
    classifier, temperature = load_classifier(CLASSIFIER_PATH, device)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    names = (*MODEL_PATHS, "identity", "all_to_event")
    totals = {name: {} for name in names}
    curves = {name: np.zeros(len(RESTORATION_GAINS_DB), dtype=np.float64) for name in names}
    rows: list[dict] = []
    for batch in loader:
        mixture = batch["waveform"].to(device)
        mask = batch["event_mask"].to(device)
        history = batch["history_map"].to(device)
        if history.shape[1:] != (12, FRAMES, GROUPS):
            raise ValueError("history map has an unexpected shape")
        condition = {
            "zero": torch.as_tensor(cache.mean, device=device).view(1, 1, GROUPS)
                    .expand(len(mixture), FRAMES, GROUPS),
            "shuffled": batch["shuffled_map"].to(device),
            "persistence": history[:, -1],
            "forecast": forecaster(history),
        }
        estimates = {name: model(mixture, condition[name]) for name, model in models.items()}
        if any(estimate.ndim != 2 or len(estimate) != len(mixture)
               or not torch.isfinite(estimate).all()
               for estimate in estimates.values()):
            raise ValueError("separator returned an invalid real-event estimate")
        length = min(mixture.shape[-1], *(estimate.shape[-1] for estimate in estimates.values()))
        mixture, mask = mixture[..., :length], mask[..., :length]
        estimates = {name: estimate[..., :length] for name, estimate in estimates.items()}
        estimates.update(trivial_estimates(mixture))
        batch_examples = {}
        for name, event in estimates.items():
            metrics, curve, examples = restoration_metrics(
                classifier, mixture, event, mask, temperature, return_per_example=True,
            )
            if any(not np.isfinite(value) for value in metrics.values()):
                raise ValueError(f"nonfinite classifier result for {name}")
            for key, value in metrics.items():
                totals[name][key] = totals[name].get(key, 0.0) + value * len(mixture)
            curves[name] += np.asarray(curve) * len(mixture)
            batch_examples[name] = examples
        for item, example_id in enumerate(batch["example_id"]):
            row = {"example_id": example_id, "day": example_id.split(":", 1)[0],
                   "shuffled_donor_day": batch["shuffled_donor_day"][item],
                   "shuffled_donor_manifest_index": int(batch["shuffled_donor_manifest_index"][item])}
            if row["shuffled_donor_day"] == row["day"]:
                raise ValueError("Shuffled background comes from the event day")
            for name in names:
                for metric, values in batch_examples[name].items():
                    row[f"{name}_{metric}"] = values[item]
            rows.append(row)
        print(json.dumps({"evaluated": len(rows), "total": len(dataset)}), flush=True)

    pairs = (("persistence", "zero"), ("persistence", "shuffled"),
             ("forecast", "zero"), ("forecast", "shuffled"),
             ("forecast", "persistence"), ("zero", "shuffled"))
    result = {
        "protocol": {
            "test_examples": len(rows), "days": sorted({row["day"] for row in rows}),
            "same_events_as_preliminary_film": True,
            "real_event_source_gt": False, "classifier_frozen": True,
            "history": "12 same-WAV strictly previous 5-s annotation-clean 79x128 maps; missing slots use nearest clean window within the past 60 s",
            "history_imputed_examples": dataset.imputed_examples,
            "history_imputed_windows": dataset.imputed_history_windows,
            "zero": "fixed per-band train mean log background repeated for 79 frames",
            "shuffled": "annotation-clean 5-s background map from a different held-out test day",
            "forecast": "frozen high-resolution GRU applied to real-event clean history; no current event in the condition",
            "separator_training": "real annotation-clean ocean plus synthetic whale/vessel proxies",
            "mixture_consistency": "background = mixture - estimated event",
            "restoration_gains_db": list(RESTORATION_GAINS_DB),
            "metric_directions": {"background_event_probability": "lower",
                                  "event_stem_probability": "higher",
                                  "monotonic_fraction": "higher",
                                  "off_event_relative_error": "lower"},
            "classifier_checkpoint_sha256": file_sha256(CLASSIFIER_PATH),
            "predictor_checkpoint_sha256": file_sha256(PREDICTOR_PATH),
            "separator_checkpoint_sha256": {name: file_sha256(path) for name, path in MODEL_PATHS.items()},
        },
        "conditions": {
            name: {**{key: value / len(rows) for key, value in totals[name].items()},
                   "restoration_curve": (curves[name] / len(rows)).tolist()}
            for name in names
        },
        "paired_contrasts": {
            f"{left}_minus_{right}": {metric: paired_contrast(rows, left, right, metric)
                                      for metric in METRICS}
            for left, right in pairs
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with OUTPUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = evaluate(args.device, args.batch_size)
    print(json.dumps({"protocol": result["protocol"],
                      "conditions": result["conditions"],
                      "paired_contrasts": result["paired_contrasts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
