"""Frozen-classifier floor and whale-target coverage diagnostic on DCLDE.

Real whale stems do not exist. We compare real event-bearing five-second WAVs
with synthetic whale events inserted into annotation-clean WAV from the same
recording, using the classifier's 64-d penultimate response relative to that
same clean reference. This is a proxy for target coverage, not stem distance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader

from .dclde import (CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE,
                    DCLDEClipDataset, EVENT_MASKED_FEATURE_CACHE, MANIFEST)
from .dclde_causal_highres_experiment import HighResolutionStore
from .dclde_real_residual_eval import CLASSIFIER_PATH, RealResidualEventDataset, file_sha256
from .dclde_separation_experiment import CausalFeatureStore, render_marine_training_target
from .event_classifier import EventClassifier, load_classifier
from .empirical_target_pool import EmpiricalTargetPool, REPORT as EMPIRICAL_POOL_REPORT
from .metrics import paired_bootstrap_interval


WHALE_FAMILIES = (
    "whale_upsweep", "whale_downsweep", "whale_arch", "whale_s_curve",
    "whale_pulse_packet",
)
OUTPUT = Path("results/dclde_target_gap_diagnostic.json")
MIXED_OUTPUT = Path("results/dclde_empirical_target_gap_diagnostic.json")
EVAL_SAMPLES = 9984  # 128-band BSR output length for a 10,000-sample input.


def empirical_route(owner: int, draw: int) -> str:
    """Balance both target routes within each equal-size 103-example draw."""
    return "empirical_procedural" if (owner + draw) % 2 == 0 else "pseudo_real"


@torch.inference_mode()
def classify_and_embed(model: EventClassifier, waveforms: torch.Tensor,
                       temperature: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact eval-mode penultimate features and calibrated probability."""
    encoded = model.network(model.features(waveforms))
    pooled = torch.cat((encoded.mean(dim=(-2, -1)),
                        encoded.amax(dim=(-2, -1))), dim=1)
    embedding = model.head[:2](pooled)
    logits = model.head[2:](embedding).squeeze(1)
    probability = torch.sigmoid(logits / temperature)
    return probability.cpu().numpy(), embedding.cpu().numpy()


def selected_broad_clean(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Deterministic, day-proportional coverage of all held-out clean windows."""
    test = frame.loc[frame.split.eq("test")].copy()
    days = sorted(test.day.unique())
    sizes = np.asarray([(test.day == day).sum() for day in days])
    allocation = np.floor(count * sizes / sizes.sum()).astype(int)
    for index in np.argsort(-(count * sizes / sizes.sum() - allocation))[:count - allocation.sum()]:
        allocation[index] += 1
    selected = []
    for day, amount in zip(days, allocation, strict=True):
        group = test.loc[test.day.eq(day)]
        positions = np.linspace(0, len(group) - 1, amount, dtype=int)
        selected.append(group.iloc[positions])
    return pd.concat(selected).reset_index(drop=True)


@torch.inference_mode()
def clean_probabilities(model: EventClassifier, temperature: float,
                        frame: pd.DataFrame, batch_size: int,
                        device: torch.device) -> list[dict]:
    dataset = DCLDEClipDataset(frame, "test")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    for batch in loader:
        values, _ = classify_and_embed(model, batch["waveform"][:, :EVAL_SAMPLES].to(device),
                                       temperature)
        for index, value in enumerate(values):
            rows.append({"example_id": batch["example_id"][index],
                         "day": batch["example_id"][index].split(":", 1)[0],
                         "probability": float(value)})
    return rows


def nearest_distance_audit(real: np.ndarray, synthetic: np.ndarray,
                           clean: np.ndarray, synthetic_owner: np.ndarray) -> dict:
    """Compare clean-referenced responses against density-matched real events."""
    real_delta = real - clean
    synthetic_delta = synthetic - clean[synthetic_owner]
    mean = synthetic_delta.mean(axis=0)
    scale = synthetic_delta.std(axis=0)
    floor = max(float(np.percentile(scale, 10)), 1e-4)
    scale = np.maximum(scale, floor)
    real_z = (real_delta - mean) / scale
    synthetic_z = (synthetic_delta - mean) / scale
    real_to_synth = cdist(real_z, synthetic_z) / np.sqrt(real.shape[1])
    real_to_real = cdist(real_z, real_z) / np.sqrt(real.shape[1])
    np.fill_diagonal(real_to_real, np.inf)
    nearest_synth = real_to_synth.min(axis=1)
    nearest_real = real_to_real.min(axis=1)
    reference_95 = float(np.quantile(nearest_real, .95))
    paired_nearest = np.asarray([
        real_to_synth[index, synthetic_owner == index].min()
        for index in range(len(real))
    ])
    # All 32 draws per event make a nearest synthetic artificially easy to find.
    # Each repeat uses one synthetic per real event, so reference-set sizes match.
    owner_indices = [np.flatnonzero(synthetic_owner == index)
                     for index in range(len(real))]
    equal_size_medians = []
    equal_size_outside = []
    for repeat in range(min(map(len, owner_indices))):
        selected = [indices[repeat] for indices in owner_indices]
        nearest = real_to_synth[:, selected].min(axis=1)
        equal_size_medians.append(float(np.median(nearest)))
        equal_size_outside.append(float(np.mean(nearest > reference_95)))
    return {
        "embedding": "frozen classifier 64-d penultimate SiLU response; event-bearing response minus same-WAV preceding clean response",
        "standardization": "per dimension, synthetic-event delta mean/std; std floored at synthetic 10th-percentile std",
        "distance": "Euclidean distance divided by sqrt(64)",
        "real_to_synthetic_nn_median": float(np.median(nearest_synth)),
        "real_to_real_loo_nn_median": float(np.median(nearest_real)),
        "real_to_synthetic_over_real_loo_ratio": float(np.median(nearest_synth) / np.median(nearest_real)),
        "real_to_synthetic_nn_p95": float(np.quantile(nearest_synth, .95)),
        "real_to_real_loo_nn_p95_threshold": reference_95,
        "real_fraction_outside_real_loo_p95": float(np.mean(nearest_synth > reference_95)),
        "equal_size_reference_count": len(real),
        "equal_size_repeats": len(equal_size_medians),
        "equal_size_real_to_synthetic_nn_median": float(np.median(equal_size_medians)),
        "equal_size_real_to_synthetic_over_real_loo_ratio": float(np.median(equal_size_medians) / np.median(nearest_real)),
        "equal_size_real_fraction_outside_real_loo_p95": float(np.median(equal_size_outside)),
        "equal_size_nn_median_across_repeats": equal_size_medians,
        "equal_size_outside_fraction_across_repeats": equal_size_outside,
        "same_WAV_paired_synthetic_nn_median": float(np.median(paired_nearest)),
        "per_real_nearest_synthetic": nearest_synth.tolist(),
        "per_real_nearest_real_loo": nearest_real.tolist(),
    }


def run(device_name: str = "cuda", broad_clean_count: int = 4096,
        synths_per_event: int = 32, batch_size: int = 64,
        target_pool: str = "legacy") -> dict:
    if target_pool not in {"legacy", "empirical_50_50"}:
        raise ValueError("unknown diagnostic target pool")
    if target_pool == "empirical_50_50" and synths_per_event % 2:
        raise ValueError("50/50 pool requires an even draw count")
    empirical = EmpiricalTargetPool() if target_pool == "empirical_50_50" else None
    device = torch.device(device_name)
    classifier, temperature = load_classifier(CLASSIFIER_PATH, device)
    classifier.eval()
    old = CausalFeatureStore(MANIFEST, EVENT_MASKED_FEATURE_CACHE)
    clean_store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    cohort = RealResidualEventDataset(old, HighResolutionStore(clean_store))
    clean_frame = clean_store.frame
    matched_indices = cohort.history_rows[:, -1]
    matched_frame = clean_frame.iloc[matched_indices].reset_index(drop=True)
    matched_dataset = DCLDEClipDataset(matched_frame, "test")
    real_waveforms = []
    clean_waveforms = []
    real_ids = []
    days = []
    for item in range(len(cohort)):
        real = cohort.base[item]
        clean = matched_dataset[item]
        if real["path"] != clean["path"] or clean["start_seconds"] >= real["start_seconds"]:
            raise ValueError("matched clean clip must precede the real call in the same WAV")
        real_waveforms.append(real["waveform"])
        clean_waveforms.append(clean["waveform"])
        real_ids.append(real["example_id"])
        days.append(real["example_id"].split(":", 1)[0])
    real_audio = torch.stack(real_waveforms)[..., :EVAL_SAMPLES]
    clean_audio = torch.stack(clean_waveforms)[..., :EVAL_SAMPLES]
    with torch.inference_mode():
        real_probability, real_embed = classify_and_embed(classifier, real_audio.to(device), temperature)
        matched_clean_probability, clean_embed = classify_and_embed(
            classifier, clean_audio.to(device), temperature)
        probe = torch.sigmoid(classifier(real_audio.to(device)) / temperature)
        reconstructed = torch.as_tensor(real_probability, device=device)
        if not torch.allclose(probe, reconstructed, rtol=1e-4, atol=1e-4):
            raise ValueError("embedding extractor does not reproduce classifier logits")

    broad_frame = selected_broad_clean(clean_frame, broad_clean_count)
    broad_clean = clean_probabilities(classifier, temperature, broad_frame, batch_size, device)
    synthetic_probability = []
    synthetic_embed = []
    synthetic_owner = []
    synthetic_rows = []
    for item, ocean_tensor in enumerate(clean_waveforms):
        ocean = ocean_tensor.numpy()
        ocean_rms = float(np.sqrt(np.mean(np.square(ocean), dtype=np.float64)))
        mixtures = []
        local_rows = []
        for draw in range(synths_per_event):
            family = WHALE_FAMILIES[(item * synths_per_event + draw) % len(WHALE_FAMILIES)]
            seed = 20261026 + item * 100003 + draw * 101
            if empirical is None:
                source, _ = render_marine_training_target(len(ocean), seed, source_family=family)
                route = "legacy_procedural"
            else:
                # Every equal-size repeat draws one target per real call; vary
                # route by owner as well as draw so each repeat stays 50/50.
                route = empirical_route(item, draw)
                source = empirical.sample(seed, route)
                family = route
            snr = float(np.random.default_rng(seed + 17).uniform(-18.0, 12.0))
            source *= ocean_rms * 10.0 ** (snr / 20.0)
            mixture = ocean + source
            peak = float(np.max(np.abs(mixture)))
            if peak > .98:
                mixture /= peak / .98
            mixtures.append(torch.from_numpy(mixture.astype(np.float32)))
            local_rows.append({"owner_real_example_id": real_ids[item], "day": days[item],
                               "family": family, "route": route, "snr_db": snr})
        for first in range(0, len(mixtures), batch_size):
            batch = torch.stack(mixtures[first:first + batch_size])[..., :EVAL_SAMPLES].to(device)
            probability, embedding = classify_and_embed(classifier, batch, temperature)
            synthetic_probability.extend(probability.tolist())
            synthetic_embed.extend(embedding)
            synthetic_owner.extend([item] * len(probability))
            for local, value in zip(local_rows[first:first + batch_size], probability, strict=True):
                local["probability"] = float(value)
                synthetic_rows.append(local)
        if (item + 1) % 25 == 0 or item + 1 == len(cohort):
            print(json.dumps({"generated_for_real_events": item + 1,
                              "total": len(cohort)}), flush=True)

    synthetic_probability = np.asarray(synthetic_probability)
    synthetic_embed = np.asarray(synthetic_embed)
    synthetic_owner = np.asarray(synthetic_owner)
    coverage = nearest_distance_audit(real_embed, synthetic_embed, clean_embed, synthetic_owner)
    broad_values = np.asarray([row["probability"] for row in broad_clean])
    snr_bins = (-18.0, -12.0, -6.0, 0.0, 6.0, 12.0)
    by_snr = {}
    for low, high in zip(snr_bins[:-1], snr_bins[1:], strict=True):
        values = [row["probability"] for row in synthetic_rows
                  if low <= row["snr_db"] < high]
        by_snr[f"[{low:g},{high:g})"] = {
            "count": len(values), "mean": float(np.mean(values)),
            "fraction_above_0_5": float(np.mean(np.asarray(values) >= .5)),
        }
    with Path("results/dclde_real_residual_event_results.csv").open(newline="", encoding="utf-8") as stream:
        separated = list(csv.DictReader(stream))
    if [row["example_id"] for row in separated] != real_ids:
        raise ValueError("real residual cohort differs from the clean-control cohort")
    residual_minus_clean = {}
    for name in ("zero", "shuffled", "persistence", "forecast"):
        difference = np.asarray([float(row[f"{name}_background_event_probability"])
                                 for row in separated]) - matched_clean_probability
        by_day = {day: float(difference[np.asarray(days) == day].mean())
                  for day in sorted(set(days))}
        residual_minus_clean[name] = {
            "mean": float(difference.mean()),
            "day_block_ci95": list(paired_bootstrap_interval(np.asarray(list(by_day.values())))),
            "by_day": by_day,
        }
    output = MIXED_OUTPUT if empirical is not None else OUTPUT
    result = {
        "protocol": {
            "real_events": len(cohort), "same_WAV_clean_controls": len(cohort),
            "broad_test_clean_controls": len(broad_clean),
            "synthetic_whale_mixtures": len(synthetic_probability),
            "synthetic_draws_per_real_event": synths_per_event,
            "target_pool": target_pool,
            "synthetic_families": (list(WHALE_FAMILIES) if empirical is None
                                   else ["empirical_procedural", "pseudo_real"]),
            "empirical_pool_report": (str(EMPIRICAL_POOL_REPORT) if empirical is not None else None),
            "synthetic_snr_db": [-18.0, 12.0],
            "synthetic_background": "immediately preceding annotation-clean 5-s WAV from the same recording as each real call",
            "real_whale_stem_gt": False,
            "classifier_input_samples": EVAL_SAMPLES,
            "classifier_frozen": True,
            "classifier_checkpoint_sha256": file_sha256(CLASSIFIER_PATH),
            "comparison_scope": "event-bearing real mixtures versus synthetic-whale mixtures; matched nearby backgrounds, not isolated real whale stems",
        },
        "classifier_probability": {
            "real_event_mean": float(real_probability.mean()),
            "matched_same_WAV_clean_mean": float(matched_clean_probability.mean()),
            "matched_same_WAV_clean_median": float(np.median(matched_clean_probability)),
            "matched_same_WAV_clean_fraction_above_0_5": float(np.mean(matched_clean_probability >= .5)),
            "broad_test_clean_mean": float(broad_values.mean()),
            "broad_test_clean_median": float(np.median(broad_values)),
            "broad_test_clean_fraction_above_0_5": float(np.mean(broad_values >= .5)),
            "synthetic_whale_mixture_mean": float(synthetic_probability.mean()),
            "synthetic_whale_mixture_median": float(np.median(synthetic_probability)),
            "synthetic_whale_mixture_fraction_above_0_5": float(np.mean(synthetic_probability >= .5)),
            "per_day_broad_clean_mean": {day: float(np.mean([r["probability"] for r in broad_clean if r["day"] == day]))
                                         for day in sorted({r["day"] for r in broad_clean})},
            "per_day_matched_clean_mean": {day: float(np.mean(matched_clean_probability[np.asarray(days) == day]))
                                           for day in sorted(set(days))},
            "per_family_synthetic_mean": {family: float(np.mean([r["probability"] for r in synthetic_rows if r["family"] == family]))
                                          for family in sorted({r["family"] for r in synthetic_rows})},
            "synthetic_by_snr_bin": by_snr,
            "residual_background_minus_matched_clean": residual_minus_clean,
        },
        "coverage": coverage,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ("example_id", "day", "real_event_probability",
                      "same_WAV_clean_probability", "real_to_synthetic_nn_distance",
                      "real_to_real_loo_nn_distance")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item, example_id in enumerate(real_ids):
            writer.writerow({"example_id": example_id, "day": days[item],
                             "real_event_probability": float(real_probability[item]),
                             "same_WAV_clean_probability": float(matched_clean_probability[item]),
                             "real_to_synthetic_nn_distance": coverage["per_real_nearest_synthetic"][item],
                             "real_to_real_loo_nn_distance": coverage["per_real_nearest_real_loo"][item]})
    with output.with_name(output.stem + "_synthetic_samples.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(synthetic_rows[0]))
        writer.writeheader()
        writer.writerows(synthetic_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--broad-clean-count", type=int, default=4096)
    parser.add_argument("--synths-per-event", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-pool", choices=("legacy", "empirical_50_50"),
                        default="legacy")
    args = parser.parse_args()
    print(json.dumps(run(args.device, args.broad_clean_count,
                         args.synths_per_event, args.batch_size,
                         args.target_pool), indent=2), flush=True)


if __name__ == "__main__":
    main()
