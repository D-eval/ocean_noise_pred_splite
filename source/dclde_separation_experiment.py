"""Train on clean real ocean + synthetic upcalls, evaluate on real labelled calls.

The separator never sees a separated real event target.  Its checkpoint is
selected only with synthetic injections into annotation-clean validation audio.
The independent event classifier is then frozen and used for an interventional
restoration-curve evaluation on day-disjoint real events.
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

from .dclde import DCLDEClipDataset, MANIFEST, SAMPLE_RATE
from .event_classifier import load_classifier
from .metrics import paired_bootstrap_interval, si_sdr
from .model import ForecastConditionedSeparator, multi_resolution_loss
from .reference_free_eval import RESTORATION_GAINS_DB, restoration_metrics, trivial_estimates


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


class InjectedUpcallDataset(Dataset):
    def __init__(self, manifest: Path, split: str, examples: int, seed: int):
        frame = pd.read_csv(manifest)
        frame = frame[(frame["split"] == split) & (frame["label"] == 0)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"no clean negative clips in {split}")
        self.base = DCLDEClipDataset(frame.assign(split=split), split)
        self.examples = examples
        self.seed = seed

    def __len__(self) -> int:
        return self.examples

    def __getitem__(self, item: int) -> dict:
        base_item = self.base[item % len(self.base)]
        ocean = base_item["waveform"].numpy()
        source, _ = render_upcalls(len(ocean), self.seed + item * 104729)
        rng = np.random.default_rng(self.seed + item * 130363)
        snr_db = float(rng.uniform(-18.0, 12.0))
        ocean_rms = float(np.sqrt(np.mean(np.square(ocean), dtype=np.float64)))
        source *= ocean_rms * 10.0 ** (snr_db / 20.0)
        mixture = ocean + source
        peak = float(np.max(np.abs(mixture)))
        if peak > 0.98:
            mixture /= peak / 0.98
            ocean /= peak / 0.98
            source /= peak / 0.98
        return {
            "mixture": torch.from_numpy(mixture.astype(np.float32)),
            "source": torch.from_numpy(source.astype(np.float32)),
            "ocean": torch.from_numpy(ocean.astype(np.float32)),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }


@dataclass
class SeparationConfig:
    steps: int = 2500
    minimum_steps: int = 1500
    validate_every: int = 100
    patience: int = 7
    batch_size: int = 4
    learning_rate: float = 5e-5
    train_examples: int = 12000
    validation_examples: int = 512
    seed: int = 20261022
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    transfer_checkpoint: str = "models/practical_real.pt"


def _make_model(seed: int, device: torch.device) -> ForecastConditionedSeparator:
    return ForecastConditionedSeparator(
        spectrum_dim=30, variant="none_film", latent_dim=16, initialization_seed=seed,
        model_dim=32, depth=2, heads=4, dim_head=8, n_fft=256, hop_length=64, bands=20,
    ).to(device)


def _load_transfer(model: ForecastConditionedSeparator, path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    destination = model.state_dict()
    compatible = {key: value for key, value in checkpoint["model"].items()
                  if key in destination and destination[key].shape == value.shape}
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


@torch.no_grad()
def _validation_score(model: ForecastConditionedSeparator, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    improvements, source_scores = [], []
    for batch in loader:
        mixture = batch["mixture"].to(device)
        source = batch["source"].to(device)
        history = torch.zeros((len(mixture), 1, 30), device=device)
        estimate = model(mixture, history)["estimate"]
        length = min(estimate.shape[-1], source.shape[-1])
        improvement = si_sdr(estimate[..., :length], source[..., :length]) - si_sdr(
            mixture[..., :length], source[..., :length]
        )
        improvements.extend(improvement.cpu().tolist())
        source_scores.extend(si_sdr(estimate[..., :length], source[..., :length]).cpu().tolist())
    low, high = paired_bootstrap_interval(np.asarray(improvements))
    return {
        "examples": len(improvements), "si_sdri_db": float(np.mean(improvements)),
        "si_sdri_ci95_db": [low, high], "source_si_sdr_db": float(np.mean(source_scores)),
    }


def train_separator(config: SeparationConfig, output: Path, manifest: Path = MANIFEST) -> dict:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device)
    train_set = InjectedUpcallDataset(manifest, "train", config.train_examples, config.seed)
    validation_set = InjectedUpcallDataset(
        manifest, "validation", config.validation_examples, config.seed + 10_000_000
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_set, batch_size=config.batch_size * 2, shuffle=False)
    model = _make_model(config.seed, device)
    transferred = _load_transfer(model, Path(config.transfer_checkpoint))
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.steps)
    output.mkdir(parents=True, exist_ok=True)
    history, best, best_step, stale = [], -math.inf, 0, 0
    iterator = iter(train_loader)
    for step in range(1, config.steps + 1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        mixture = batch["mixture"].to(device)
        source = batch["source"].to(device)
        history_tensor = torch.zeros((len(mixture), 1, 30), device=device)
        estimate = model(mixture, history_tensor)["estimate"]
        loss = multi_resolution_loss(estimate, source)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        scheduler.step()
        if step == 1 or step % config.validate_every == 0 or step == config.steps:
            row = {"step": step, "train_loss": float(loss.detach())}
            if step % config.validate_every == 0 or step == config.steps:
                metrics = _validation_score(model, validation_loader, device)
                row.update({f"validation_{key}": value for key, value in metrics.items()})
                if metrics["si_sdri_db"] > best + 1e-4:
                    best, best_step, stale = metrics["si_sdri_db"], step, 0
                    torch.save({"model": model.state_dict(), "config": asdict(config), "step": step}, output / "best.pt")
                else:
                    stale += 1
            history.append(row)
            print(json.dumps(row), flush=True)
            (output / "training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if step >= config.minimum_steps and stale >= config.patience:
            break
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    summary = {
        "config": asdict(config), "transferred_tensors": transferred,
        "best_step": best_step, "updates_run": history[-1]["step"],
        "validation": _validation_score(model, validation_loader, device),
        "selection": "synthetic upcalls in annotation-clean, day-disjoint validation audio only",
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@torch.no_grad()
def evaluate_real_events(
    separator_checkpoint: Path, classifier_checkpoint: Path, output: Path,
    manifest: Path = MANIFEST, batch_size: int = 8, device_name: str = "cuda",
) -> dict:
    device = torch.device(device_name)
    separator = _make_model(20261022, device)
    separator.load_state_dict(torch.load(separator_checkpoint, map_location=device, weights_only=False)["model"])
    separator.eval()
    classifier, temperature = load_classifier(classifier_checkpoint, device)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    test = DCLDEClipDataset(manifest, "test")
    positive_indices = np.flatnonzero(test.frame["label"].to_numpy() == 1).tolist()
    positive = torch.utils.data.Subset(test, positive_indices)
    loader = DataLoader(positive, batch_size=batch_size, shuffle=False)
    names = ("separator", "identity", "all_to_event")
    totals = {name: {} for name in names}
    curves = {name: np.zeros(len(RESTORATION_GAINS_DB), dtype=np.float64) for name in names}
    rows: list[dict] = []
    seen = 0
    for batch in loader:
        mixture = batch["waveform"].to(device)
        mask = batch["event_mask"].to(device)
        history = torch.zeros((len(mixture), 1, 30), device=device)
        estimate = separator(mixture, history)["estimate"]
        length = min(mixture.shape[-1], estimate.shape[-1])
        mixture, mask, estimate = mixture[..., :length], mask[..., :length], estimate[..., :length]
        estimates = {"separator": estimate, **trivial_estimates(mixture)}
        for name, event in estimates.items():
            metrics, curve = restoration_metrics(classifier, mixture, event, mask, temperature)
            for key, value in metrics.items():
                totals[name][key] = totals[name].get(key, 0.0) + value * len(mixture)
            curves[name] += np.asarray(curve) * len(mixture)
        for example_id in batch["example_id"]:
            rows.append({"example_id": example_id})
        seen += len(mixture)
    summary = {
        "protocol": {
            "test_examples": seen, "split": "three held-out recording days",
            "separator_training": "real annotation-clean ocean + synthetic upcalls",
            "real_event_source_gt": False, "classifier_frozen": True,
            "mixture_consistency": "background = mixture - event",
            "restoration_gains_db": list(RESTORATION_GAINS_DB),
        },
        "conditions": {
            name: {
                **{key: value / seen for key, value in totals[name].items()},
                "restoration_curve": (curves[name] / seen).tolist(),
            }
            for name in names
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["example_id"])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--minimum-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    folder = Path("experiments/dclde_separator")
    if not args.skip_train:
        print(json.dumps(train_separator(SeparationConfig(
            steps=args.steps, minimum_steps=args.minimum_steps,
            batch_size=args.batch_size, device=args.device,
        ), folder), indent=2))
    print(json.dumps(evaluate_real_events(
        folder / "best.pt", Path("experiments/dclde_event_classifier/best.pt"),
        Path("results/dclde_real_event_results.json"), device_name=args.device,
    ), indent=2))


if __name__ == "__main__":
    main()
