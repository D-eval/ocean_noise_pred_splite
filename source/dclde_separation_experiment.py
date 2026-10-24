"""Matched BSR/FC-BSR training and reference-free evaluation on real calls.

Both separators train on identical annotation-clean DCLDE backgrounds with
identical diverse whale-like sweeps and vessel sounds. FC-BSR alone receives
twelve strictly preceding 5-second spectra. The learned residual forecaster must beat causal persistence
on a day-disjoint validation split; otherwise persistence remains selected.
Real annotated calls are used only for the final frozen-classifier evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .dclde import (
    CLEAN_BACKGROUND_MANIFEST,
    EVENT_MASKED_FEATURE_CACHE,
    DCLDEClipDataset,
    CLEAN_OCEAN_FEATURE_CACHE,
    FEATURE_CACHE,
    MANIFEST,
    SAMPLE_RATE,
    build_causal_feature_cache,
    build_clip_manifest,
    build_event_masked_feature_cache,
)
from .event_classifier import load_classifier
from .metrics import paired_bootstrap_interval, si_sdr
from .model import ForecastConditionedSeparator, NoiseForecaster, multi_resolution_loss
from .reference_free_eval import RESTORATION_GAINS_DB, restoration_metrics, trivial_estimates
from .synthesis import draw_injection, render_injection


def render_upcalls(samples: int, seed: int, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, list[list[float]]]:
    """Generate a diverse, time-localised North Atlantic right-whale-like upcall."""
    rng = np.random.default_rng(seed)
    output = np.zeros(samples, dtype=np.float32)
    intervals: list[list[float]] = []
    total_seconds = samples / sample_rate
    count = int(rng.integers(1, 4))
    for _ in range(count):
        duration = float(rng.uniform(0.55, min(2.4, total_seconds * 0.42)))
        onset = float(rng.uniform(0.15, max(0.151, total_seconds - duration - 0.15)))
        start = round(onset * sample_rate)
        length = min(samples - start, round(duration * sample_rate))
        if length < 8:
            continue
        time = np.arange(length, dtype=np.float64) / sample_rate
        f0 = float(rng.uniform(65.0, 145.0))
        f1 = float(rng.uniform(max(f0 + 45.0, 145.0), 360.0))
        curvature = float(rng.uniform(0.65, 1.8))
        phase = 2.0 * np.pi * (
            f0 * time + (f1 - f0) * duration * np.power(time / duration, curvature + 1.0)
            / (curvature + 1.0)
        )
        phase += rng.uniform(0.0, 2.0 * np.pi)
        call = np.sin(phase)
        if rng.random() < 0.65:
            call += rng.uniform(0.08, 0.35) * np.sin(2.0 * phase + rng.uniform(0.0, 2.0 * np.pi))
        taper = np.sin(np.pi * np.linspace(0.0, 1.0, length, endpoint=True)) ** rng.uniform(1.2, 2.8)
        call *= taper * (1.0 + rng.uniform(0.0, 0.25) * np.sin(2 * np.pi * rng.uniform(1.0, 5.0) * time))
        output[start : start + length] += call.astype(np.float32)
        intervals.append([start / sample_rate, (start + length) / sample_rate])
    rms = float(np.sqrt(np.mean(np.square(output), dtype=np.float64)))
    output /= max(rms, 1e-8)
    return output, intervals


def render_marine_training_target(
    samples: int,
    seed: int,
    sample_rate: int = SAMPLE_RATE,
    source_family: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Render whale contours and vessel proxies for clean-background training.

    Whale coverage includes rising, falling, arch-shaped, sigmoid, and pulsed
    sweeps. Vessel coverage includes harmonic machinery tones and cavitation;
    80% of vessel draws apply a 5--20 Hz amplitude LFO to tonal bands only.
    """
    parameters = draw_injection(
        seed, snr_range=(0.0, 0.0), profile="marine", family=source_family,
    )
    waveform = render_injection(parameters, sample_rate, samples)
    return waveform, asdict(parameters)


class CausalFeatureStore:
    """Cached causal histories with normalization fitted on training negatives."""

    def __init__(self, manifest: Path = MANIFEST, cache: Path = EVENT_MASKED_FEATURE_CACHE):
        self.frame = pd.read_csv(manifest)
        values = np.load(cache)
        self.history = values["history"].astype(np.float32)
        self.target = values["target"].astype(np.float32)
        self.eligible = values["eligible"].astype(bool)
        if len(self.frame) != len(self.history):
            raise ValueError("feature cache and manifest row counts differ")
        train_negative = (
            self.frame["split"].eq("train").to_numpy()
            & self.frame["label"].eq(0).to_numpy()
            & self.eligible
        )
        fit = np.concatenate((
            self.history[train_negative].reshape(-1, self.history.shape[-1]),
            self.target[train_negative],
        ))
        self.mean = np.nanmean(fit, axis=0).astype(np.float32)
        self.scale = np.nanstd(fit, axis=0).clip(1e-4).astype(np.float32)

    @property
    def bands(self) -> int:
        return int(self.history.shape[-1])

    def normalized_history(self, index: int) -> np.ndarray:
        return ((self.history[index] - self.mean) / self.scale).astype(np.float32)

    def normalized_target(self, index: int) -> np.ndarray:
        return ((self.target[index] - self.mean) / self.scale).astype(np.float32)

    def indices(self, split: str, label: int) -> np.ndarray:
        return np.flatnonzero(
            self.frame["split"].eq(split).to_numpy()
            & self.frame["label"].eq(label).to_numpy()
            & self.eligible
        )


class ForecastFeatureDataset(Dataset):
    def __init__(self, store: CausalFeatureStore, split: str):
        self.store = store
        self.rows = store.indices(split, label=0)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict:
        index = int(self.rows[item])
        return {
            "history": torch.from_numpy(self.store.normalized_history(index)),
            "target": torch.from_numpy(self.store.normalized_target(index)),
        }


class InjectedUpcallDataset(Dataset):
    """(H_t, N_t, s_t, x_t) tuples from annotated clean real backgrounds."""

    def __init__(
        self, store: CausalFeatureStore, split: str, examples: int, seed: int,
        source_family: str | None = None, snr_range: tuple[float, float] = (-18.0, 12.0),
        spread_rows: bool = False,
    ):
        self.store = store
        self.rows = store.indices(split, label=0)
        selected = store.frame.iloc[self.rows].copy()
        selected["split"] = split
        self.base = DCLDEClipDataset(selected, split)
        self.examples = examples
        self.seed = seed
        self.source_family = source_family
        self.snr_range = snr_range
        self.spread_rows = spread_rows

    def __len__(self) -> int:
        return self.examples

    def __getitem__(self, item: int) -> dict:
        local = (
            round(item * (len(self.rows) - 1) / max(1, self.examples - 1))
            if self.spread_rows and self.examples > 1 else item % len(self.rows)
        )
        manifest_index = int(self.rows[local])
        base_item = self.base[local]
        ocean = base_item["waveform"].numpy()
        source, parameters = render_marine_training_target(
            len(ocean), self.seed + item * 104729, source_family=self.source_family,
        )
        rng = np.random.default_rng(self.seed + item * 130363)
        snr_db = float(rng.uniform(*self.snr_range))
        ocean_rms = float(np.sqrt(np.mean(np.square(ocean), dtype=np.float64)))
        source *= ocean_rms * 10.0 ** (snr_db / 20.0)
        mixture = ocean + source
        peak = float(np.max(np.abs(mixture)))
        if peak > 0.98:
            scale = peak / 0.98
            mixture, ocean, source = mixture / scale, ocean / scale, source / scale
        row = self.store.frame.iloc[manifest_index]
        return {
            "mixture": torch.from_numpy(mixture.astype(np.float32)),
            "source": torch.from_numpy(source.astype(np.float32)),
            "ocean": torch.from_numpy(ocean.astype(np.float32)),
            "background": torch.from_numpy(ocean.astype(np.float32)),
            "event": torch.from_numpy(source.astype(np.float32)),
            "history": torch.from_numpy(self.store.normalized_history(manifest_index)),
            "target_spectrum": torch.from_numpy(self.store.normalized_target(manifest_index)),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            "source_family": parameters["family"],
            "example_id": f"{row.day}:{Path(row.path).stem}:{float(row.start_seconds):.1f}",
            "mixture_identity": "x_t = N_t + s_t",
        }


class RealEventDataset(Dataset):
    def __init__(self, store: CausalFeatureStore, split: str = "test"):
        self.store = store
        self.rows = store.indices(split, label=1)
        selected = store.frame.iloc[self.rows].copy()
        selected["split"] = split
        self.base = DCLDEClipDataset(selected, split)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict:
        value = self.base[item]
        value["history"] = torch.from_numpy(self.store.normalized_history(int(self.rows[item])))
        return value


@dataclass
class ExperimentConfig:
    forecast_steps: int = 5000
    forecast_minimum_steps: int = 2500
    forecast_validate_every: int = 100
    separation_steps: int = 10000
    separation_minimum_steps: int = 5000
    separation_validate_every: int = 100
    patience: int = 7
    batch_size: int = 4
    forecast_batch_size: int = 64
    learning_rate: float = 5e-5
    forecast_learning_rate: float = 3e-4
    train_examples: int = 48000
    validation_examples: int = 4096
    seed: int = 20261022
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    transfer_checkpoint: str = "models/practical_real.pt"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_model(variant: str, bands: int, seed: int, device: torch.device) -> ForecastConditionedSeparator:
    return ForecastConditionedSeparator(
        spectrum_dim=bands, variant=variant, latent_dim=16, initialization_seed=seed,
        forecast_residual=True, model_dim=32, depth=2, heads=4, dim_head=8,
        n_fft=256, hop_length=64, bands=20,
    ).to(device)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_transfer(model: ForecastConditionedSeparator, path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    destination = model.state_dict()
    compatible = {
        key: value for key, value in checkpoint["model"].items()
        if key in destination and destination[key].shape == value.shape
        and not key.startswith("forecaster.")
    }
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


@torch.no_grad()
def _forecast_metrics(
    forecaster: NoiseForecaster,
    loader: DataLoader,
    scale: np.ndarray,
    device: torch.device,
) -> dict:
    forecaster.eval()
    learned, persistence = [], []
    scale_tensor = torch.as_tensor(scale, device=device)
    for batch in loader:
        history = batch["history"].to(device)
        target = batch["target"].to(device)
        _, prediction = forecaster(history)
        learned.append(((prediction - target).abs() * scale_tensor).cpu())
        persistence.append(((history[:, -1] - target).abs() * scale_tensor).cpu())
    learned_error = torch.cat(learned)
    persistence_error = torch.cat(persistence)
    mae = float(learned_error.mean())
    persistence_mae = float(persistence_error.mean())
    return {
        "examples": int(len(learned_error)), "mae_db": mae,
        "persistence_mae_db": persistence_mae,
        "skill_over_persistence": 1.0 - mae / max(persistence_mae, 1e-12),
    }


def train_forecaster(config: ExperimentConfig, store: CausalFeatureStore, output: Path) -> dict:
    _seed_everything(config.seed)
    device = torch.device(config.device)
    train = ForecastFeatureDataset(store, "train")
    validation = ForecastFeatureDataset(store, "validation")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train, batch_size=config.forecast_batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation, batch_size=128, shuffle=False)
    forecaster = NoiseForecaster(store.bands, latent_dim=16, residual=True).to(device)
    optimiser = torch.optim.AdamW(forecaster.parameters(), lr=config.forecast_learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.forecast_steps)
    initial = _forecast_metrics(forecaster, validation_loader, store.scale, device)
    best_mae, best_step, stale = initial["mae_db"], 0, 0
    best_state = {key: value.detach().cpu().clone() for key, value in forecaster.state_dict().items()}
    history = [{"step": 0, "train_loss": None, **initial}]
    iterator = iter(train_loader)
    for step in range(1, config.forecast_steps + 1):
        forecaster.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        history_tensor = batch["history"].to(device)
        target = batch["target"].to(device)
        _, prediction = forecaster(history_tensor)
        loss = torch.nn.functional.smooth_l1_loss(prediction, target)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 5.0)
        optimiser.step()
        scheduler.step()
        if step % config.forecast_validate_every == 0 or step == config.forecast_steps:
            metrics = _forecast_metrics(forecaster, validation_loader, store.scale, device)
            row = {"step": step, "train_loss": float(loss.detach()), **metrics}
            history.append(row)
            print(json.dumps({"stage": "forecast", **row}), flush=True)
            if metrics["mae_db"] < best_mae - 1e-4:
                best_mae, best_step, stale = metrics["mae_db"], step, 0
                best_state = {key: value.detach().cpu().clone() for key, value in forecaster.state_dict().items()}
            else:
                stale += 1
            if step >= config.forecast_minimum_steps and stale >= config.patience:
                break
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "step": best_step, "normalizer_mean": store.mean, "normalizer_scale": store.scale}, output / "best.pt")
    (output / "training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    forecaster.load_state_dict(best_state)
    summary = {
        "best_step": best_step, "updates_run": history[-1]["step"],
        "validation": _forecast_metrics(forecaster, validation_loader, store.scale, device),
        "selection_rule": "learned residual must beat step-0 persistence on validation",
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@torch.no_grad()
def _validation_score(model: ForecastConditionedSeparator, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    improvements, source_scores = [], []
    for batch in loader:
        mixture = batch["mixture"].to(device)
        source = batch["source"].to(device)
        history = batch["history"].to(device)
        estimate = model(mixture, history)["estimate"]
        length = min(estimate.shape[-1], source.shape[-1])
        improvement = si_sdr(estimate[..., :length], source[..., :length]) - si_sdr(mixture[..., :length], source[..., :length])
        improvements.extend(improvement.cpu().tolist())
        source_scores.extend(si_sdr(estimate[..., :length], source[..., :length]).cpu().tolist())
    low, high = paired_bootstrap_interval(np.asarray(improvements))
    return {"examples": len(improvements), "si_sdri_db": float(np.mean(improvements)), "si_sdri_ci95_db": [low, high], "source_si_sdr_db": float(np.mean(source_scores))}


@torch.no_grad()
def _full_reference_test(
    model: ForecastConditionedSeparator,
    dataset: InjectedUpcallDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    model.eval()
    rows: list[dict] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        mixture = batch["mixture"].to(device)
        source = batch["source"].to(device)
        history = batch["history"].to(device)
        estimate = model(mixture, history)["estimate"]
        length = min(estimate.shape[-1], source.shape[-1], mixture.shape[-1])
        estimate = estimate[..., :length]
        source = source[..., :length]
        mixture = mixture[..., :length]
        ocean = batch["ocean"].to(device)[..., :length]
        mixture_scores = si_sdr(mixture, source)
        estimate_scores = si_sdr(estimate, source)
        ocean_scores = si_sdr(mixture - estimate, ocean)
        for index, example_id in enumerate(batch["example_id"]):
            rows.append({
                "example_id": example_id,
                "day": example_id.split(":", 1)[0],
                "source_family": batch["source_family"][index],
                "snr_db": float(batch["snr_db"][index]),
                "mixture_si_sdr_db": float(mixture_scores[index]),
                "source_si_sdr_db": float(estimate_scores[index]),
                "source_si_sdri_db": float(estimate_scores[index] - mixture_scores[index]),
                "background_si_sdr_db": float(ocean_scores[index]),
            })
    gains = np.asarray([row["source_si_sdri_db"] for row in rows], dtype=np.float64)
    days = sorted({row["day"] for row in rows})
    day_means = np.asarray([
        np.mean([row["source_si_sdri_db"] for row in rows if row["day"] == day])
        for day in days
    ])
    summary = {
        "examples": len(rows),
        "source_si_sdri_db": float(gains.mean()),
        "source_si_sdri_ci95_db": list(paired_bootstrap_interval(day_means)),
        "confidence_interval_unit": "held-out recording day",
        "source_si_sdr_db": float(np.mean([row["source_si_sdr_db"] for row in rows])),
        "background_si_sdr_db": float(np.mean([row["background_si_sdr_db"] for row in rows])),
    }
    return summary, rows


def train_separator(
    config: ExperimentConfig,
    store: CausalFeatureStore,
    variant: str,
    output: Path,
    forecast_checkpoint: Path | None = None,
) -> dict:
    _seed_everything(config.seed)
    device = torch.device(config.device)
    train_set = InjectedUpcallDataset(store, "train", config.train_examples, config.seed)
    validation_set = InjectedUpcallDataset(store, "validation", config.validation_examples, config.seed + 10_000_000)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_set, batch_size=config.batch_size * 2, shuffle=False)
    model = _make_model(variant, store.bands, config.seed, device)
    transferred = _load_transfer(model, Path(config.transfer_checkpoint))
    if forecast_checkpoint is not None:
        checkpoint = torch.load(forecast_checkpoint, map_location=device, weights_only=False)
        if model.forecaster is None:
            raise ValueError("forecast checkpoint supplied to an unconditioned model")
        model.forecaster.load_state_dict(checkpoint["model"])
        for parameter in model.forecaster.parameters():
            parameter.requires_grad_(False)
    optimiser = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.separation_steps)
    output.mkdir(parents=True, exist_ok=True)
    history_rows, best, best_step, stale = [], -math.inf, 0, 0
    iterator = iter(train_loader)
    for step in range(1, config.separation_steps + 1):
        model.train()
        if model.forecaster is not None:
            model.forecaster.eval()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        mixture = batch["mixture"].to(device)
        source = batch["source"].to(device)
        history_tensor = batch["history"].to(device)
        estimate = model(mixture, history_tensor)["estimate"]
        loss = multi_resolution_loss(estimate, source)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        scheduler.step()
        if step == 1 or step % config.separation_validate_every == 0 or step == config.separation_steps:
            row = {"step": step, "train_loss": float(loss.detach())}
            if step % config.separation_validate_every == 0 or step == config.separation_steps:
                metrics = _validation_score(model, validation_loader, device)
                row.update({f"validation_{key}": value for key, value in metrics.items()})
                if metrics["si_sdri_db"] > best + 1e-4:
                    best, best_step, stale = metrics["si_sdri_db"], step, 0
                    torch.save({"model": model.state_dict(), "config": asdict(config), "variant": variant, "step": step}, output / "best.pt")
                else:
                    stale += 1
            history_rows.append(row)
            print(json.dumps({"stage": "separation", "variant": variant, **row}), flush=True)
            (output / "training.json").write_text(json.dumps(history_rows, indent=2), encoding="utf-8")
        if step >= config.separation_minimum_steps and stale >= config.patience:
            break
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_set = InjectedUpcallDataset(
        store, "test", config.validation_examples, config.seed + 20_000_000,
        spread_rows=True,
    )
    summary = {
        "variant": variant, "config": asdict(config), "transferred_tensors": transferred,
        "best_step": best_step, "updates_run": history_rows[-1]["step"],
        "validation": _validation_score(model, validation_loader, device),
        "test_full_reference": None,
        "selection": "matched whale-sweep and vessel proxies in annotation-clean day-disjoint validation audio",
        "forecaster_frozen": forecast_checkpoint is not None,
    }
    test_summary, test_rows = _full_reference_test(
        model, test_set, config.batch_size * 2, device,
    )
    summary["test_full_reference"] = test_summary
    _write_rows(output / "clean_synthetic_test.csv", test_rows)
    (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@torch.no_grad()
def evaluate_real_events(
    checkpoints: dict[str, tuple[str, Path]],
    classifier_checkpoint: Path,
    store: CausalFeatureStore,
    output: Path,
    batch_size: int = 8,
    device_name: str = "cuda",
) -> dict:
    device = torch.device(device_name)
    models = {}
    for name, (variant, path) in checkpoints.items():
        model = _make_model(variant, store.bands, 20261022, device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
        model.eval()
        models[name] = model
    classifier, temperature = load_classifier(classifier_checkpoint, device)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    dataset = RealEventDataset(store)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    names = tuple(models) + ("identity", "all_to_event")
    totals = {name: {} for name in names}
    curves = {name: np.zeros(len(RESTORATION_GAINS_DB), dtype=np.float64) for name in names}
    rows: list[dict] = []
    seen = 0
    for batch in loader:
        mixture = batch["waveform"].to(device)
        mask = batch["event_mask"].to(device)
        history = batch["history"].to(device)
        estimates = {name: model(mixture, history)["estimate"] for name, model in models.items()}
        length = min([mixture.shape[-1], *(value.shape[-1] for value in estimates.values())])
        mixture, mask = mixture[..., :length], mask[..., :length]
        estimates = {name: value[..., :length] for name, value in estimates.items()}
        estimates.update(trivial_estimates(mixture))
        batch_examples: dict[str, dict[str, list[float]]] = {}
        for name, event in estimates.items():
            metrics, curve, examples = restoration_metrics(
                classifier, mixture, event, mask, temperature, return_per_example=True,
            )
            for key, value in metrics.items():
                totals[name][key] = totals[name].get(key, 0.0) + value * len(mixture)
            curves[name] += np.asarray(curve) * len(mixture)
            batch_examples[name] = examples
        for item, example_id in enumerate(batch["example_id"]):
            row = {"example_id": example_id, "day": example_id.split(":", 1)[0]}
            for name in names:
                for metric, values in batch_examples[name].items():
                    row[f"{name}_{metric}"] = values[item]
            rows.append(row)
        seen += len(mixture)
    paired = {}
    for metric in (
        "background_event_probability", "event_stem_probability", "removal_logit",
        "monotonic_fraction", "leakage_outside_event", "off_event_relative_error",
    ):
        differences = np.asarray([
            float(row[f"fc_bsr_{metric}"]) - float(row[f"bs_roformer_{metric}"])
            for row in rows
        ])
        days = sorted({row["day"] for row in rows})
        day_means = np.asarray([
            np.mean([difference for row, difference in zip(rows, differences, strict=True) if row["day"] == day])
            for day in days
        ])
        low, high = paired_bootstrap_interval(day_means)
        paired[metric] = {
            "fc_bsr_minus_bs_roformer": float(differences.mean()),
            "day_block_bootstrap_ci95": [low, high],
            "independent_day_blocks": len(days),
        }
    summary = {
        "protocol": {
            "test_examples": seen, "split": "three held-out recording days",
            "comparison": "paired BS-RoFormer versus FC-BSR on identical eligible real events",
            "history": "12 strictly preceding 5-s 30-band features; annotated call intervals and 2-s guards are excluded",
            "separator_training": "real annotation-clean 5-s ocean + identical whale-sweep and vessel proxies",
            "real_event_source_gt": False, "classifier_frozen": True,
            "mixture_consistency": "background = mixture - event",
            "restoration_gains_db": list(RESTORATION_GAINS_DB),
        },
        "conditions": {
            name: {**{key: value / seen for key, value in totals[name].items()}, "restoration_curve": (curves[name] / seen).tolist()}
            for name in names
        },
        "paired_fc_bsr_minus_bs_roformer": paired,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def compare_clean_synthetic_tests(
    baseline_csv: Path,
    conditioned_csv: Path,
    output: Path,
) -> dict:
    """Paired held-out test on clean current audio with injected synthetic events."""
    def read_rows(path: Path) -> dict[str, dict]:
        with path.open(encoding="utf-8", newline="") as stream:
            return {row["example_id"]: row for row in csv.DictReader(stream)}

    baseline = read_rows(baseline_csv)
    conditioned = read_rows(conditioned_csv)
    common = sorted(set(baseline) & set(conditioned))
    if not common:
        raise ValueError("full-reference test files contain no matched examples")
    rows = []
    for example_id in common:
        left, right = baseline[example_id], conditioned[example_id]
        rows.append({
            "example_id": example_id,
            "day": left["day"],
            "source_family": left["source_family"],
            "snr_db": float(left["snr_db"]),
            "baseline_si_sdri_db": float(left["source_si_sdri_db"]),
            "fc_bsr_si_sdri_db": float(right["source_si_sdri_db"]),
            "paired_gain_db": float(right["source_si_sdri_db"]) - float(left["source_si_sdri_db"]),
        })
    days = sorted({row["day"] for row in rows})
    day_means = np.asarray([
        np.mean([row["paired_gain_db"] for row in rows if row["day"] == day]) for day in days
    ])
    ci = paired_bootstrap_interval(day_means)
    by_family = {}
    for family in sorted({row["source_family"] for row in rows}):
        subset = [row for row in rows if row["source_family"] == family]
        by_family[family] = {
            "examples": len(subset),
            "baseline_si_sdri_db": float(np.mean([row["baseline_si_sdri_db"] for row in subset])),
            "fc_bsr_si_sdri_db": float(np.mean([row["fc_bsr_si_sdri_db"] for row in subset])),
            "paired_gain_db": float(np.mean([row["paired_gain_db"] for row in subset])),
        }
    summary = {
        "protocol": {
            "examples": len(rows), "split": "three held-out recording days",
            "current_background": "annotation-clean real 5-s WAV",
            "target": "marine-aware synthetic event with exact stem references",
            "history": "12 preceding event-masked 5-s background features (60 s total)",
            "models": "matched BS-RoFormer versus GRU + FiLM + BS-RoFormer",
            "independent_day_blocks": len(days),
        },
        "baseline_mean_si_sdri_db": float(np.mean([row["baseline_si_sdri_db"] for row in rows])),
        "fc_bsr_mean_si_sdri_db": float(np.mean([row["fc_bsr_si_sdri_db"] for row in rows])),
        "paired_gain_db": float(np.mean([row["paired_gain_db"] for row in rows])),
        "paired_gain_day_block_ci95_db": list(ci),
        "by_family": by_family,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def evaluate_saved_clean_separator(
    store: CausalFeatureStore,
    variant: str,
    checkpoint_path: Path,
    config: ExperimentConfig,
    output: Path,
) -> dict:
    """Run the held-out full-reference test for an already-trained separator."""
    device = torch.device(config.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _make_model(variant, store.bands, config.seed, device)
    model.load_state_dict(checkpoint["model"])
    dataset = InjectedUpcallDataset(
        store, "test", config.validation_examples, config.seed + 20_000_000,
        spread_rows=True,
    )
    summary, rows = _full_reference_test(model, dataset, config.batch_size * 2, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(output.parent / "clean_synthetic_test.csv", rows)
    result = {
        "variant": variant, "checkpoint": str(checkpoint_path),
        "best_step": checkpoint.get("step"), "test_full_reference": summary,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-steps", type=int, default=5000)
    parser.add_argument("--separation-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare-features-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-forecast", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    if not MANIFEST.exists() or not CLEAN_BACKGROUND_MANIFEST.exists():
        _, summary = build_clip_manifest()
        print(json.dumps(summary, indent=2))
    if not CLEAN_OCEAN_FEATURE_CACHE.exists():
        print(json.dumps(build_event_masked_feature_cache(
            CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE,
        ), indent=2))
    if not EVENT_MASKED_FEATURE_CACHE.exists():
        print(json.dumps(build_event_masked_feature_cache(), indent=2))
    if not FEATURE_CACHE.exists():
        print(json.dumps(build_causal_feature_cache(), indent=2))
    if args.prepare_features_only:
        return
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    config = ExperimentConfig(forecast_steps=args.forecast_steps, separation_steps=args.separation_steps, batch_size=args.batch_size, device=args.device)
    train_clean_hours = float(
        store.frame.loc[store.indices("train", label=0), "duration_seconds"].sum() / 3600.0
    )
    if train_clean_hours < 24.0:
        raise RuntimeError(f"main setup needs >=24 h of clean training background; found {train_clean_hours:.2f} h")
    forecast_folder = Path("experiments/dclde_forecaster")
    baseline_folder = Path("experiments/dclde_bsr")
    conditioned_folder = Path("experiments/dclde_fc_bsr")
    if not args.skip_train:
        if not args.skip_forecast:
            print(json.dumps(train_forecaster(config, store, forecast_folder), indent=2))
        if not args.skip_baseline:
            print(json.dumps(train_separator(config, store, "none_film", baseline_folder), indent=2))
        print(json.dumps(train_separator(config, store, "film", conditioned_folder, forecast_folder / "best.pt"), indent=2))
        deployment_config = {
            "variant": "film", "latent_dim": 16, "seed": config.seed,
            "model_dim": 32, "depth": 2, "heads": 4, "dim_head": 8,
            "n_fft": 256, "hop_length": 64, "bands": 20,
            "sample_rate": SAMPLE_RATE, "seconds": 5.0, "history_steps": 12,
            "normaliser_mean": store.mean.tolist(), "normaliser_scale": store.scale.tolist(),
            "spectrum_columns": [f"event-masked-band-{index + 1:02d}" for index in range(store.bands)],
            "history_definition": "12 consecutive preceding 5-s background spectra; annotated call intervals and 2-s guards excluded",
            "current_background_definition": "annotation-clean real 5-s DCLDE recording",
            "model_family": "BS-RoFormer",
            "conditioning": "FiLM",
            "forecaster": "causal GRU",
            "waveform_window_seconds": 5.0,
            "clean_training_background_hours": train_clean_hours,
            "clean_background_pool_hours": float(
                store.frame.loc[store.frame["label"].eq(0), "duration_seconds"].sum() / 3600.0
            ),
            "main_setup": "12 event-masked 5-s history features -> GRU forecast; current annotated-clean 5-s ocean N_t + synthetic marine event s_t; FiLM-conditioned BS-RoFormer separates x_t",
        }
        main_checkpoint = torch.load(conditioned_folder / "best.pt", map_location="cpu", weights_only=False)
        Path("models/practical_film.pt").parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": main_checkpoint["model"], "config": deployment_config,
                    "step": main_checkpoint["step"]}, "models/practical_film.pt")
        print(json.dumps({"exported_main_checkpoint": "models/practical_film.pt", "config": deployment_config}, indent=2))
    print(json.dumps(compare_clean_synthetic_tests(
        baseline_folder / "clean_synthetic_test.csv",
        conditioned_folder / "clean_synthetic_test.csv",
        Path("results/dclde_clean_synthetic_test.json"),
    ), indent=2))
    evaluation_store = CausalFeatureStore(MANIFEST, EVENT_MASKED_FEATURE_CACHE)
    evaluation_store.mean = store.mean.copy()
    evaluation_store.scale = store.scale.copy()
    print(json.dumps(evaluate_real_events(
        {"bs_roformer": ("none_film", baseline_folder / "best.pt"), "fc_bsr": ("film", conditioned_folder / "best.pt")},
        Path("experiments/dclde_event_classifier/best.pt"), evaluation_store,
        Path("results/dclde_real_event_results.json"), device_name=args.device,
    ), indent=2))


if __name__ == "__main__":
    main()
