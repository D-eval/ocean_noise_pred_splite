"""Paired separation evaluation over broad source families and spectral shift."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DEFAULT_DATA_ROOT, SpectrumNormaliser, TOLMixtureDataset, load_experiment_frames
from .metrics import narrow_source_reference, paired_bootstrap_interval
from .model import ForecastConditionedSeparator
from .synthesis import BROAD_FAMILIES
from .train import evaluate


def load_checkpoint(path: Path, device: torch.device) -> tuple[ForecastConditionedSeparator, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ForecastConditionedSeparator(
        spectrum_dim=len(config["spectrum_columns"]),
        variant=config["variant"],
        latent_dim=int(config["latent_dim"]),
        initialization_seed=int(config["seed"]),
        model_dim=int(config["model_dim"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        dim_head=int(config["dim_head"]),
        n_fft=int(config["n_fft"]),
        hop_length=int(config["hop_length"]),
        bands=int(config["bands"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def summarise_pairs(rows: list[dict]) -> dict:
    gain = np.asarray([row["conditioned_gain_db"] for row in rows], dtype=float)
    low, high = paired_bootstrap_interval(gain)
    return {
        "examples": len(rows),
        "mean_js_bits": float(np.mean([row["source_js_bits"] for row in rows])),
        "mean_kl_bits": float(np.mean([row["source_kl_bits"] for row in rows])),
        "mean_source_ocean_spectral_overlap": float(
            np.mean([row["source_ocean_spectral_overlap"] for row in rows])
        ),
        "mean_source_spectral_entropy": float(
            np.mean([row["source_spectral_entropy"] for row in rows])
        ),
        "baseline_si_sdri_db": float(np.mean([row["baseline_si_sdri_db"] for row in rows])),
        "conditioned_si_sdri_db": float(np.mean([row["conditioned_si_sdri_db"] for row in rows])),
        "paired_conditioned_gain_db": float(gain.mean()),
        "paired_gain_ci95_db": [low, high],
        "conditioned_wins_fraction": float(np.mean(gain > 0.0)),
    }


def run(
    baseline_checkpoint: Path,
    conditioned_checkpoint: Path,
    output_json: Path,
    examples_per_family: int,
    snr_db: float,
    device_name: str,
    data_root: Path,
) -> dict:
    device = torch.device(device_name)
    baseline, baseline_config = load_checkpoint(baseline_checkpoint, device)
    conditioned, conditioned_config = load_checkpoint(conditioned_checkpoint, device)
    structural_keys = (
        "sample_rate", "seconds", "context_hours", "model_dim", "depth", "heads",
        "dim_head", "n_fft", "hop_length", "bands", "spectrum_columns",
        "normaliser_mean", "normaliser_scale",
    )
    mismatched = [key for key in structural_keys if baseline_config[key] != conditioned_config[key]]
    if mismatched:
        raise ValueError(f"checkpoints are not matched: {', '.join(mismatched)}")
    if baseline.variant != "none" or conditioned.variant == "none":
        raise ValueError("expected a none baseline and a forecast-conditioned checkpoint")

    sample_rate = int(baseline_config["sample_rate"])
    samples = round(sample_rate * float(baseline_config["seconds"]))
    reference = narrow_source_reference(sample_rate, samples)
    normaliser = SpectrumNormaliser(
        mean=np.asarray(baseline_config["normaliser_mean"], dtype=np.float32),
        scale=np.asarray(baseline_config["normaliser_scale"], dtype=np.float32),
    )
    frame = load_experiment_frames(data_root)["MB03_04"]
    paired_rows: list[dict] = []
    batch_size = int(baseline_config["batch_size"])
    for family in BROAD_FAMILIES:
        dataset = TOLMixtureDataset(
            frame,
            "MB03_04",
            normaliser=normaliser,
            context_hours=int(baseline_config["context_hours"]),
            sample_rate=sample_rate,
            seconds=float(baseline_config["seconds"]),
            seed=int(baseline_config["seed"]),
            max_examples=examples_per_family,
            snr_range=(snr_db, snr_db),
            source_profile="broad",
            source_family=family,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        _, baseline_rows = evaluate(baseline, loader, normaliser, device, reference)
        _, conditioned_rows = evaluate(conditioned, loader, normaliser, device, reference)
        for base, cond in zip(baseline_rows, conditioned_rows, strict=True):
            if (base["timestamp"], base["origin"], base["source_family"]) != (
                cond["timestamp"], cond["origin"], cond["source_family"]
            ):
                raise RuntimeError("paired evaluation lost sample alignment")
            paired_rows.append(
                {
                    "source_family": family,
                    "timestamp": base["timestamp"],
                    "origin": base["origin"],
                    "snr_db": base["snr_db"],
                    "source_js_bits": base["source_js_bits"],
                    "source_kl_bits": base["source_kl_bits"],
                    "source_ocean_spectral_overlap": base["source_ocean_spectral_overlap"],
                    "source_spectral_entropy": base["source_spectral_entropy"],
                    "baseline_si_sdri_db": base["source_si_sdri_db"],
                    "conditioned_si_sdri_db": cond["source_si_sdri_db"],
                    "conditioned_gain_db": cond["source_si_sdri_db"] - base["source_si_sdri_db"],
                }
            )

    js_values = np.asarray([row["source_js_bits"] for row in paired_rows])
    quantiles = np.quantile(js_values, np.linspace(0.0, 1.0, 6))
    bin_ids = np.clip(np.searchsorted(quantiles[1:-1], js_values, side="right"), 0, 4)
    divergence_bins = {}
    for bin_id in range(5):
        selected = [row for row, current in zip(paired_rows, bin_ids) if current == bin_id]
        summary = summarise_pairs(selected)
        summary["js_range_bits"] = [float(quantiles[bin_id]), float(quantiles[bin_id + 1])]
        divergence_bins[f"Q{bin_id + 1}"] = summary

    overlap_values = np.asarray([row["source_ocean_spectral_overlap"] for row in paired_rows])
    overlap_quantiles = np.quantile(overlap_values, np.linspace(0.0, 1.0, 6))
    overlap_bin_ids = np.clip(
        np.searchsorted(overlap_quantiles[1:-1], overlap_values, side="right"), 0, 4
    )
    overlap_bins = {}
    for bin_id in range(5):
        selected = [row for row, current in zip(paired_rows, overlap_bin_ids) if current == bin_id]
        summary = summarise_pairs(selected)
        summary["overlap_range"] = [
            float(overlap_quantiles[bin_id]), float(overlap_quantiles[bin_id + 1])
        ]
        overlap_bins[f"Q{bin_id + 1}"] = summary

    entropy_values = np.asarray([row["source_spectral_entropy"] for row in paired_rows])
    entropy_quantiles = np.quantile(entropy_values, np.linspace(0.0, 1.0, 6))
    entropy_bin_ids = np.clip(
        np.searchsorted(entropy_quantiles[1:-1], entropy_values, side="right"), 0, 4
    )
    entropy_bins = {}
    for bin_id in range(5):
        selected = [row for row, current in zip(paired_rows, entropy_bin_ids) if current == bin_id]
        summary = summarise_pairs(selected)
        summary["entropy_range"] = [
            float(entropy_quantiles[bin_id]), float(entropy_quantiles[bin_id + 1])
        ]
        entropy_bins[f"Q{bin_id + 1}"] = summary

    by_family = {
        family: summarise_pairs([row for row in paired_rows if row["source_family"] == family])
        for family in BROAD_FAMILIES
    }
    gains = np.asarray([row["conditioned_gain_db"] for row in paired_rows])
    result = {
        "metric": {
            "primary": "Jensen-Shannon divergence in bits between each source power spectrum and the original narrow harmonic-generator reference; symmetric, finite, range [0, 1]",
            "secondary": "KL(source || narrow reference) in bits; directional and unbounded",
            "task_relevant": "source-ocean spectral histogram intersection in [0, 1]; larger values mean the target occupies frequency regions similar to the background",
            "source_shape": "normalised spectral entropy in [0, 1]; larger values denote more noise-like, spectrally diffuse targets",
            "interpretation": "larger divergence means greater spectral departure from the original synthetic-signal distribution, not automatically greater real-world diversity",
        },
        "protocol": {
            "conditioning_variant": conditioned.variant,
            "families": list(BROAD_FAMILIES),
            "examples_per_family": examples_per_family,
            "examples": len(paired_rows),
            "snr_db": snr_db,
            "background": "held-out MB03_04 TOL-derived surrogate",
            "pairing": "same timestamp, ocean waveform, source waveform, and SNR for both models",
            "baseline_checkpoint": str(baseline_checkpoint),
            "conditioned_checkpoint": str(conditioned_checkpoint),
        },
        "overall": summarise_pairs(paired_rows),
        "by_divergence_quintile": divergence_bins,
        "by_source_ocean_overlap_quintile": overlap_bins,
        "by_spectral_entropy_quintile": entropy_bins,
        "by_family": by_family,
        "gain_js_pearson_r": float(np.corrcoef(js_values, gains)[0, 1]),
        "gain_overlap_pearson_r": float(np.corrcoef(overlap_values, gains)[0, 1]),
        "gain_entropy_pearson_r": float(np.corrcoef(entropy_values, gains)[0, 1]),
        "scope": "controlled surrogate-waveform result; not field-audio validation",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_path = output_json.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--conditioned-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/diversity_results.json"))
    parser.add_argument("--examples-per-family", type=int, default=128)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    result = run(
        args.baseline_checkpoint,
        args.conditioned_checkpoint,
        args.output,
        args.examples_per_family,
        args.snr_db,
        args.device,
        args.data_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
