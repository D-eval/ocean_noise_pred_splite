"""One-shot frozen-model evaluation on a prespecified new-site GSL cohort.

GSL annotations provide call centers, not clean whale stems or onset/offset.
The event mask is therefore a declared +/-1.5 s proxy and cannot establish
full-reference separation quality. No GSL event is used in training or tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from .dclde import CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE, SAMPLE_RATE
from .dclde_causal_highres_experiment import (
    FRAMES, GROUPS, CausalResidualBSR, HighResGRU, HighResolutionStore,
    log_band_map,
)
from .dclde_real_residual_eval import CLASSIFIER_PATH, PREDICTOR_PATH, file_sha256
from .dclde_separation_experiment import CausalFeatureStore, _make_model
from .event_classifier import load_classifier
from .gsl_new_site_data import ROOT, selected_names
from .metrics import paired_bootstrap_interval
from .reference_free_eval import (
    RESTORATION_GAINS_DB, event_probabilities, restoration_metrics,
    trivial_estimates,
)


OUTPUT = Path("results/gsl_new_site_frozen_eval.json")
WINDOW = 5.0
HISTORY = 12
EVENT_HALF_WIDTH = 1.5  # GSL gives centers only, not precise call intervals.
GUARD = 2.0             # Avoid calls within 2 s of a history window.
BATCH = 8
MODELS = {
    "zero": (Path("models/zero_residual_128.pt"), "zero"),
    "shuffled": (Path("models/shuffled_residual_128.pt"), "shuffled"),
    "persistence": (Path("models/persistence_residual_128.pt"), "persistence"),
    "forecast": (Path("models/forecast_residual_128.pt"), "forecast"),
    "empirical_persistence": (Path("models/empirical_persistence_residual_128.pt"), "persistence"),
    "empirical_forecast": (Path("models/empirical_forecast_residual_128.pt"), "forecast"),
    "weak_adapter_persistence": (Path("models/adapter_weak_persistence_residual_128.pt"), "persistence"),
}


def _clip(audio: np.ndarray, start: float) -> np.ndarray:
    left = int(round(start * SAMPLE_RATE))
    waveform = audio[left:left + int(WINDOW * SAMPLE_RATE)]
    if len(waveform) != int(WINDOW * SAMPLE_RATE):
        raise ValueError("incomplete 5-s clip")
    # Match DCLDEClipDataset: remove each 5-s clip's DC component before
    # passing either mixtures or background-history windows to frozen models.
    return (waveform - np.mean(waveform, dtype=np.float64)).astype(np.float32)


class GSLEvents(Dataset):
    def __init__(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        if manifest["names"] != selected_names():
            raise ValueError("GSL cohort selection changed")
        with (ROOT / "annotations.csv").open(newline="", encoding="utf-8") as stream:
            annotations = list(csv.DictReader(stream))
        by_file = defaultdict(list)
        for row in annotations:
            by_file[row["filename"]].append((float(row["timestamp"]), float(row["snr"])))
        self.audio = {}
        self.centers = {}
        self.events = []
        self.imputed_windows = 0
        self.skipped = defaultdict(int)
        for name in manifest["names"]:
            audio, rate = sf.read(ROOT / name, dtype="float32", always_2d=False)
            if rate != SAMPLE_RATE or audio.ndim != 1:
                raise ValueError(f"bad prepared WAV: {name}")
            self.audio[name] = audio
            centers = np.asarray([value[0] for value in by_file[name]], dtype=np.float64)
            self.centers[name] = centers
            for center, snr in by_file[name]:
                start = center - WINDOW / 2
                if start < HISTORY * WINDOW or start + WINDOW > len(audio) / rate:
                    self.skipped["boundary"] += 1
                    continue
                history_starts = [start - HISTORY * WINDOW + k * WINDOW for k in range(HISTORY)]
                clean = [not self._has_call(name, prior, prior + WINDOW, GUARD)
                         for prior in history_starts]
                if not any(clean):
                    self.skipped["no_clean_history"] += 1
                    continue
                observed = np.flatnonzero(clean)
                matched = int(observed[-1])
                self.imputed_windows += sum(not value for value in clean)
                self.events.append({"name": name, "center": center, "start": start,
                                    "snr": snr, "history_starts": history_starts,
                                    "clean": clean, "matched": matched})
        self.donors = {}
        for name in manifest["names"]:
            duration = len(self.audio[name]) / SAMPLE_RATE
            options = [float(t) for t in np.arange(60, duration - WINDOW, WINDOW)
                       if not self._has_call(name, t, t + WINDOW, GUARD)]
            if not options:
                raise ValueError(f"no annotation-negative wrong-file donor: {name}")
            self.donors[name] = options[len(options) // 2]
        if len(self.events) < 100:
            raise ValueError("new-site cohort has too few evaluable events")

    def _has_call(self, name: str, left: float, right: float, guard: float) -> bool:
        centers = self.centers[name]
        return bool(np.any((centers >= left - guard) & (centers < right + guard)))

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, item: int) -> dict:
        row = self.events[item]
        name = row["name"]
        history = []
        clean_indices = np.flatnonzero(row["clean"])
        for slot, start in enumerate(row["history_starts"]):
            use_slot = slot if row["clean"][slot] else int(clean_indices[np.argmin(np.abs(clean_indices-slot))])
            history.append(_clip(self.audio[name], row["history_starts"][use_slot]))
        mixture = _clip(self.audio[name], row["start"])
        mask = np.zeros(len(mixture), dtype=np.float32)
        for center in self.centers[name]:
            left = int(max(0, round((center - EVENT_HALF_WIDTH - row["start"]) * SAMPLE_RATE)))
            right = int(min(len(mask), round((center + EVENT_HALF_WIDTH - row["start"]) * SAMPLE_RATE)))
            if right > left:
                mask[left:right] = 1
        if not mask.any():
            raise ValueError("center annotation is absent from its event clip")
        names = selected_names()
        donor_name = names[(names.index(name) + len(names) // 2) % len(names)]
        if donor_name == name:
            raise ValueError("wrong-file donor equals current file")
        return {"example_id": f"{name}:{row['center']:.3f}", "file": name,
                "center": row["center"], "snr": row["snr"],
                "waveform": torch.from_numpy(mixture.copy()),
                "event_mask": torch.from_numpy(mask),
                "history_wave": torch.from_numpy(np.stack(history)),
                "matched_clean_wave": torch.from_numpy(_clip(
                    self.audio[name], row["history_starts"][row["matched"]]).copy()),
                "shuffled_wave": torch.from_numpy(_clip(self.audio[donor_name], self.donors[donor_name]).copy()),
                "shuffled_donor_file": donor_name,
                "imputed_history_windows": int(HISTORY - sum(row["clean"]))}


def _load_models(device: torch.device, bands: int) -> tuple[dict, dict]:
    paths = {"bsr": Path("experiments/dclde_true_bsr/best.pt")}
    paths.update({name: path for name, (path, _) in MODELS.items()})
    paths["predictor"] = PREDICTOR_PATH
    paths["evaluation_detector"] = CLASSIFIER_PATH
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    bsr = _make_model("bsr", bands, 20261022, device)
    bsr.load_state_dict(torch.load(paths["bsr"], map_location=device, weights_only=False)["model"], strict=True)
    bsr.eval()
    models = {"bsr": bsr}
    for name, (path, _) in MODELS.items():
        model = CausalResidualBSR(bands, 20261022, device,
                                  Path("models/practical_real.pt")).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"], strict=True)
        model.eval()
        models[name] = model
    return models, hashes


def _paired(rows: list[dict], left: str, right: str, metric: str) -> dict:
    delta = np.asarray([float(row[f"{left}_{metric}"]) - float(row[f"{right}_{metric}"])
                        for row in rows], dtype=np.float64)
    files = np.asarray([row["file"] for row in rows])
    by_file = {name: float(delta[files == name].mean()) for name in sorted(set(files))}
    file_values = np.asarray(list(by_file.values()))
    return {"event_weighted_mean": float(delta.mean()),
            "file_equal_mean": float(file_values.mean()), "by_file": by_file,
            "file_block_ci95": list(paired_bootstrap_interval(file_values))}


@torch.inference_mode()
def evaluate(device_name: str = "cuda") -> dict:
    if OUTPUT.exists():
        raise FileExistsError("one-shot GSL result already exists; do not re-evaluate for model selection")
    device = torch.device(device_name)
    data_manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    dataset = GSLEvents()
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    cache = HighResolutionStore(store)
    models, hashes = _load_models(device, store.bands)
    forecaster = HighResGRU(cache.mean, cache.std).to(device)
    forecaster.load_state_dict(torch.load(PREDICTOR_PATH, map_location=device,
                                          weights_only=False)["model"], strict=True)
    forecaster.eval()
    classifier, temperature = load_classifier(CLASSIFIER_PATH, device)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    names = (*models, "identity", "all_to_event")
    rows = []
    for batch in DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0):
        mixture = batch["waveform"].to(device)
        mask = batch["event_mask"].to(device)
        history = log_band_map(batch["history_wave"].to(device).flatten(0, 1)).reshape(
            len(mixture), HISTORY, FRAMES, GROUPS)
        shuffled = log_band_map(batch["shuffled_wave"].to(device))
        constant = torch.as_tensor(cache.mean, device=device).view(1, 1, GROUPS).expand(
            len(mixture), FRAMES, GROUPS)
        condition = {"zero": constant, "shuffled": shuffled,
                     "persistence": history[:, -1], "forecast": forecaster(history)}
        estimates = {"bsr": models["bsr"](mixture, None)["estimate"]}
        for name, (_, kind) in MODELS.items():
            estimates[name] = models[name](mixture, condition[kind])
        estimates.update(trivial_estimates(mixture))
        clean_p = event_probabilities(classifier, batch["matched_clean_wave"].to(device), temperature)
        details = {}
        for name, event in estimates.items():
            _, _, examples = restoration_metrics(classifier, mixture, event, mask,
                                                  temperature, return_per_example=True)
            details[name] = examples
        for item, example_id in enumerate(batch["example_id"]):
            row = {"example_id": example_id, "file": batch["file"][item],
                   "center_seconds": float(batch["center"][item]),
                   "annotated_snr_db": float(batch["snr"][item]),
                   "shuffled_donor_file": batch["shuffled_donor_file"][item],
                   "imputed_history_windows": int(batch["imputed_history_windows"][item]),
                   "matched_annotation_negative_probability": float(clean_p[item])}
            for name in names:
                for metric, values in details[name].items():
                    row[f"{name}_{metric}"] = values[item]
            rows.append(row)
        print(json.dumps({"evaluated": len(rows), "total": len(dataset)}), flush=True)
    metrics = ("mixture_event_probability", "background_event_probability",
               "event_stem_probability", "monotonic_fraction", "off_event_relative_error",
               "removal_logit", "restoration_auc")
    result = {"protocol": {"dataset": "Gulf of St. Lawrence continuous dataset B, 2015-2017",
                            "source_doi": "10.20383/101.0241",
                            "selection": data_manifest["selection"],
                            "selected_files": selected_names(),
                            "annotated_events": data_manifest["selected_annotation_count"],
                            "evaluable_events": len(rows), "skipped": dict(dataset.skipped),
                            "event_mask": "call-center +/-1.5 s proxy; original annotation has no onset/offset",
                            "history": "12 strictly preceding 5-s windows, excluding annotated calls +/-2 s; nearest clean historical imputation",
                            "shuffled": "annotation-negative condition from a different selected file; exact dates unavailable",
                            "sample_rate": SAMPLE_RATE,
                            "input_preprocessing": "each 5-s clip demeaned, matching DCLDEClipDataset",
                            "source_sample_rates_hz": sorted({value["source_sample_rate"]
                                                               for value in data_manifest["audio"].values()}),
                            "model_and_detector_sha256": hashes,
                            "test_used_for_training_or_selection": False,
                            "real_clean_event_stems_available": False,
                            "restoration_gains_db": RESTORATION_GAINS_DB},
              "matched_annotation_negative_probability": float(np.mean([
                  row["matched_annotation_negative_probability"] for row in rows])),
              "methods": {name: {metric: float(np.mean([row[f"{name}_{metric}"] for row in rows]))
                                  for metric in metrics} for name in names},
              "contrasts": {}}
    for left, right in (("persistence", "zero"), ("persistence", "shuffled"),
                        ("forecast", "persistence"),
                        ("weak_adapter_persistence", "empirical_persistence"),
                        ("empirical_persistence", "persistence")):
        result["contrasts"][f"{left}_minus_{right}"] = {
            metric: _paired(rows, left, right, metric)
            for metric in ("background_event_probability", "event_stem_probability",
                           "off_event_relative_error")}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = evaluate(args.device)
    print(json.dumps({"protocol": result["protocol"], "methods": result["methods"]}, indent=2))


if __name__ == "__main__":
    main()
