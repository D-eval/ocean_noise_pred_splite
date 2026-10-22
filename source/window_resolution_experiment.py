"""Train forecast-conditioned separation at 5 s, 30 s, 1 min, and 10 min."""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .data import SpectrumNormaliser
from .metrics import paired_bootstrap_interval, si_sdr
from .model import ForecastConditionedSeparator, NoiseForecaster, multi_resolution_loss
from .real_audio_data import load_manifest
from .train import _to_device, engineering_usability_gate, evaluate
from .windowed_audio import (
    BAND_CENTRES,
    CONTEXT_SECONDS,
    FEATURE_CACHE,
    WINDOW_SECONDS,
    WindowForecastDataset,
    WindowedRealAudioDataset,
    build_feature_cache,
    fit_window_normaliser,
    prepare_window_features,
)


@dataclass
class WindowExperimentConfig:
    window_seconds: int
    forecast_steps: int = 2500
    forecast_min_steps: int = 1500
    separation_steps: int = 5000
    separation_min_steps: int = 3000
    forecast_validate_every: int = 100
    separation_validate_every: int = 250
    patience: int = 6
    batch_size: int = 8
    forecast_batch_size: int = 64
    learning_rate: float = 1e-4
    forecast_learning_rate: float = 3e-4
    forecast_weight: float = 0.2
    sample_rate: int = 8000
    seconds: float = 0.5
    train_examples_per_deployment: int = 4096
    validation_examples: int = 512
    test_examples: int = 1024
    strength_examples: int = 256
    train_snr_low_db: float = -12.0
    train_snr_high_db: float = 12.0
    seed: int = 20261022
    model_dim: int = 32
    latent_dim: int = 16
    depth: int = 2
    heads: int = 4
    dim_head: int = 8
    n_fft: int = 256
    hop_length: int = 64
    bands: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    separator_checkpoint: str = "models/practical_real.pt"

    def __post_init__(self) -> None:
        if self.window_seconds not in WINDOW_SECONDS:
            raise ValueError(f"window_seconds must be one of {WINDOW_SECONDS}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forecast_sets(
    features: dict[str, np.ndarray], normaliser: SpectrumNormaliser, window_seconds: int
) -> tuple[DatasetLike, WindowForecastDataset, WindowForecastDataset]:
    train = ConcatDataset([
        WindowForecastDataset(features["MB03_01"], normaliser, window_seconds),
        WindowForecastDataset(features["MB03_02"], normaliser, window_seconds),
    ])
    validation = WindowForecastDataset(
        features["MB03_03"], normaliser, window_seconds, max_examples=512
    )
    test = WindowForecastDataset(
        features["MB03_04"], normaliser, window_seconds, max_examples=2048
    )
    return train, validation, test


# A small alias keeps the public helper's annotation readable without requiring
# a custom base class shared by Dataset and ConcatDataset.
DatasetLike = torch.utils.data.Dataset


@torch.no_grad()
def _forecast_metrics(
    forecaster: NoiseForecaster,
    dataset: DatasetLike,
    normaliser: SpectrumNormaliser,
    device: torch.device,
) -> dict:
    forecaster.eval()
    absolute = []
    squared = []
    persistence = []
    scale = torch.as_tensor(normaliser.scale, device=device)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    for batch in loader:
        history = batch["history"].to(device)
        target = batch["target_spectrum"].to(device)
        _, prediction = forecaster(history)
        error_db = (prediction - target).abs() * scale
        persistence_db = (history[:, -1] - target).abs() * scale
        absolute.append(error_db.cpu())
        squared.append(error_db.square().cpu())
        persistence.append(persistence_db.cpu())
    errors = torch.cat(absolute)
    square_errors = torch.cat(squared)
    persistence_errors = torch.cat(persistence)
    mae = float(errors.mean())
    persistence_mae = float(persistence_errors.mean())
    return {
        "examples": len(dataset),
        "mae_db": mae,
        "rmse_db": float(square_errors.mean().sqrt()),
        "persistence_mae_db": persistence_mae,
        "skill_over_persistence": 1.0 - mae / max(persistence_mae, 1e-12),
    }


def _pretrain_forecaster(
    config: WindowExperimentConfig,
    train: DatasetLike,
    validation: WindowForecastDataset,
    normaliser: SpectrumNormaliser,
    output: Path,
) -> tuple[dict, dict[str, torch.Tensor]]:
    device = torch.device(config.device)
    forecaster = NoiseForecaster(
        len(BAND_CENTRES), config.latent_dim, residual=True
    ).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        train, batch_size=config.forecast_batch_size, shuffle=True,
        generator=generator, drop_last=len(train) >= config.forecast_batch_size,
    )
    optimiser = torch.optim.AdamW(
        forecaster.parameters(), lr=config.forecast_learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.forecast_steps)
    iterator = iter(loader)
    # With the residual decoder zero-initialised, step 0 is exactly the strong
    # persistence forecast. Learned dynamics must beat it on the validation
    # deployment to be selected.
    initial_metrics = _forecast_metrics(forecaster, validation, normaliser, device)
    best_mae = initial_metrics["mae_db"]
    best_state = {
        key: value.detach().cpu().clone() for key, value in forecaster.state_dict().items()
    }
    best_step = 0
    stale = 0
    history = [{"step": 0, "train_loss": None, **initial_metrics}]
    for step in range(1, config.forecast_steps + 1):
        forecaster.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        history_tensor = batch["history"].to(device)
        target = batch["target_spectrum"].to(device)
        _, prediction = forecaster(history_tensor)
        loss = torch.nn.functional.smooth_l1_loss(prediction, target)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 5.0)
        optimiser.step()
        scheduler.step()
        if step % config.forecast_validate_every == 0 or step == config.forecast_steps:
            metrics = _forecast_metrics(forecaster, validation, normaliser, device)
            row = {"step": step, "train_loss": float(loss.detach()), **metrics}
            history.append(row)
            print(json.dumps({"stage": "forecast", "window": config.window_seconds, **row}), flush=True)
            if metrics["mae_db"] < best_mae - 1e-4:
                best_mae = metrics["mae_db"]
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in forecaster.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if step >= config.forecast_min_steps and stale >= config.patience:
                break
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "step": best_step}, output / "forecast_best.pt")
    (output / "forecast_training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    forecaster.load_state_dict(best_state)
    summary = {
        "best_step": best_step,
        "updates_run": history[-1]["step"],
        "validation": _forecast_metrics(forecaster, validation, normaliser, device),
    }
    return summary, best_state


def _load_separator_transfer(model: ForecastConditionedSeparator, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    destination = model.state_dict()
    compatible = {
        key: value for key, value in checkpoint["model"].items()
        if key in destination and destination[key].shape == value.shape and not key.startswith("forecaster.")
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_tensors": len(compatible),
        "skipped_tensors": len(checkpoint["model"]) - len(compatible),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
    }


def _mixture_sets(
    config: WindowExperimentConfig,
    features: dict[str, np.ndarray],
    normaliser: SpectrumNormaliser,
    snr_range: tuple[float, float] | None = None,
) -> dict[str, WindowedRealAudioDataset]:
    records = load_manifest()["records"]
    common = dict(
        normaliser=normaliser, records=records, window_seconds=config.window_seconds,
        sample_rate=config.sample_rate, seconds=config.seconds, seed=config.seed,
        snr_range=snr_range or (config.train_snr_low_db, config.train_snr_high_db),
    )
    return {
        "MB03_01": WindowedRealAudioDataset(
            features["MB03_01"], deployment="MB03_01",
            examples=config.train_examples_per_deployment, max_cached_hours=6, **common,
        ),
        "MB03_02": WindowedRealAudioDataset(
            features["MB03_02"], deployment="MB03_02",
            examples=config.train_examples_per_deployment, max_cached_hours=6, **common,
        ),
        "MB03_03": WindowedRealAudioDataset(
            features["MB03_03"], deployment="MB03_03",
            examples=config.validation_examples, max_cached_hours=2, **common,
        ),
        "MB03_04": WindowedRealAudioDataset(
            features["MB03_04"], deployment="MB03_04",
            examples=config.test_examples, max_cached_hours=2, **common,
        ),
    }


@torch.no_grad()
def _quick_si_sdri(model, loader, device: torch.device) -> float:
    model.eval()
    values = []
    for batch in loader:
        batch = _to_device(batch, device)
        output = model(batch["mixture"], batch["history"], batch.get("external_condition"))
        estimate = output["estimate"]
        length = min(estimate.shape[-1], batch["source"].shape[-1])
        score = si_sdr(estimate[..., :length], batch["source"][..., :length])
        mixture = si_sdr(batch["mixture"][..., :length], batch["source"][..., :length])
        values.append((score - mixture).cpu())
    return float(torch.cat(values).mean())


def _make_conditioned_model(config: WindowExperimentConfig) -> ForecastConditionedSeparator:
    return ForecastConditionedSeparator(
        spectrum_dim=len(BAND_CENTRES), variant="film", latent_dim=config.latent_dim,
        initialization_seed=config.seed, forecast_residual=True,
        model_dim=config.model_dim, depth=config.depth,
        heads=config.heads, dim_head=config.dim_head, n_fft=config.n_fft,
        hop_length=config.hop_length, bands=config.bands,
    )


def _load_baseline(config: WindowExperimentConfig, device: torch.device) -> ForecastConditionedSeparator:
    checkpoint = torch.load(config.separator_checkpoint, map_location="cpu", weights_only=False)
    old = checkpoint["config"]
    model = ForecastConditionedSeparator(
        spectrum_dim=len(old["spectrum_columns"]), variant="none_film",
        latent_dim=old["latent_dim"], initialization_seed=old["seed"],
        model_dim=old["model_dim"], depth=old["depth"], heads=old["heads"],
        dim_head=old["dim_head"], n_fft=old["n_fft"],
        hop_length=old["hop_length"], bands=old["bands"],
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device)


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_window(
    config: WindowExperimentConfig,
    features: dict[str, np.ndarray],
    output: Path,
) -> dict:
    _seed_everything(config.seed)
    device = torch.device(config.device)
    normaliser = fit_window_normaliser(features)
    forecast_train, forecast_validation, forecast_test = _forecast_sets(
        features, normaliser, config.window_seconds
    )
    forecast_summary, forecast_state = _pretrain_forecaster(
        config, forecast_train, forecast_validation, normaliser, output
    )
    sets = _mixture_sets(config, features, normaliser)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        ConcatDataset([sets["MB03_01"], sets["MB03_02"]]),
        batch_size=config.batch_size, shuffle=True, generator=generator,
    )
    validation_loader = DataLoader(
        sets["MB03_03"], batch_size=config.batch_size, shuffle=False
    )
    test_loader = DataLoader(sets["MB03_04"], batch_size=config.batch_size, shuffle=False)
    model = _make_conditioned_model(config).to(device)
    transfer = _load_separator_transfer(model, Path(config.separator_checkpoint))
    model.forecaster.load_state_dict(forecast_state, strict=True)
    for parameter in model.forecaster.parameters():
        parameter.requires_grad_(False)
    optimiser = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.separation_steps)
    output.mkdir(parents=True, exist_ok=True)
    metadata = asdict(config) | {
        "history_duration_seconds": CONTEXT_SECONDS,
        "history_steps": CONTEXT_SECONDS // config.window_seconds,
        "prediction_target": "next/current non-overlapping real-audio window",
        "mixture": "real current-window clip plus broad synthetic source",
        "bands_hz": BAND_CENTRES.tolist(),
        "normaliser_mean": normaliser.mean.tolist(),
        "normaliser_scale": normaliser.scale.tolist(),
        "train_deployments": ["MB03_01", "MB03_02"],
        "validation_deployment": "MB03_03",
        "test_deployment": "MB03_04",
        "transfer": transfer,
        "forecast_pretraining": forecast_summary,
        "forecaster_frozen_during_separation": True,
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    iterator = iter(train_loader)
    best_score = float("-inf")
    best_step = 0
    stale = 0
    history = []
    for step in range(1, config.separation_steps + 1):
        model.train()
        model.forecaster.eval()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _to_device(batch, device)
        output_values = model(batch["mixture"], batch["history"])
        separation_loss = multi_resolution_loss(output_values["estimate"], batch["source"])
        forecast_loss = torch.nn.functional.smooth_l1_loss(
            output_values["prediction"], batch["target_spectrum"]
        )
        loss = separation_loss + config.forecast_weight * forecast_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        scheduler.step()
        if step % config.separation_validate_every == 0 or step == config.separation_steps:
            score = _quick_si_sdri(model, validation_loader, device)
            row = {
                "step": step, "loss": float(loss.detach()),
                "separation_loss": float(separation_loss.detach()),
                "forecast_loss": float(forecast_loss.detach()),
                "validation_source_si_sdri_db": score,
            }
            history.append(row)
            print(json.dumps({"stage": "separation", "window": config.window_seconds, **row}), flush=True)
            if score > best_score + 1e-4:
                best_score = score
                best_step = step
                stale = 0
                torch.save({"model": model.state_dict(), "config": metadata, "step": step}, output / "best.pt")
            else:
                stale += 1
            (output / "separation_training.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            if step >= config.separation_min_steps and stale >= config.patience:
                break
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    conditioned_test, conditioned_rows = evaluate(model, test_loader, normaliser, device)
    forecast_after_joint = _forecast_metrics(model.forecaster, forecast_test, normaliser, device)
    baseline = _load_baseline(config, device)
    baseline_test, baseline_rows = evaluate(baseline, test_loader, normaliser, device)
    baseline_by_id = {row["example_id"]: row["source_si_sdri_db"] for row in baseline_rows}
    differences = np.asarray([
        row["source_si_sdri_db"] - baseline_by_id[row["example_id"]]
        for row in conditioned_rows
    ])
    low, high = paired_bootstrap_interval(differences)
    strength = {}
    for snr in (-12.0, -6.0, 0.0, 6.0, 12.0):
        strength_config = WindowExperimentConfig(**(
            asdict(config) | {"test_examples": config.strength_examples}
        ))
        strength_set = _mixture_sets(
            strength_config, features, normaliser, snr_range=(snr, snr)
        )["MB03_04"]
        summary, _ = evaluate(
            model, DataLoader(strength_set, batch_size=config.batch_size), normaliser, device
        )
        strength[f"{snr:+.0f}_dB"] = summary
    result = {
        "window_seconds": config.window_seconds,
        "history_seconds": CONTEXT_SECONDS,
        "history_steps": CONTEXT_SECONDS // config.window_seconds,
        "forecast_pretrained": forecast_summary,
        "forecast_after_joint": forecast_after_joint,
        "best_separation_step": best_step,
        "separation_updates_run": history[-1]["step"],
        "conditioned_test": conditioned_test,
        "baseline_test": baseline_test,
        "paired_gain_over_baseline_db": float(differences.mean()),
        "paired_gain_ci95_db": [low, high],
        "strength_sweep": strength,
        "engineering_usability_gate": engineering_usability_gate(conditioned_test, strength),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_rows(output / "test_examples.csv", conditioned_rows)
    _write_rows(output / "baseline_examples.csv", baseline_rows)
    return result


def aggregate(experiments_root: Path, output: Path) -> dict:
    rows = []
    for window in WINDOW_SECONDS:
        result = json.loads(
            (experiments_root / f"window_{window}s" / "result.json").read_text(encoding="utf-8")
        )
        rows.append(result)
    best = max(rows, key=lambda value: value["conditioned_test"]["source_si_sdri_db"])
    summary = {
        "protocol": {
            "windows_seconds": list(WINDOW_SECONDS),
            "fixed_history_seconds": CONTEXT_SECONDS,
            "causality": "history ends at t-1; spectrum t is supervision only",
            "target_background": "real NOAA SanctSound waveform from window t",
            "target_signal": "broad synthetic source injected into a clip inside window t",
            "selection": "early stopping uses MB03_03 only; MB03_04 is held-out test",
        },
        "results": rows,
        "best_window_seconds": best["window_seconds"],
        "best_test_si_sdri_db": best["conditioned_test"]["source_si_sdri_db"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", choices=[str(value) for value in WINDOW_SECONDS] + ["all"], default="all")
    parser.add_argument("--build-features-only", action="store_true")
    parser.add_argument("--feature-cache", type=Path, default=FEATURE_CACHE)
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments/window_resolution"))
    parser.add_argument("--forecast-steps", type=int, default=2500)
    parser.add_argument("--separation-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.feature_cache.exists():
        build_feature_cache(args.feature_cache)
    if args.build_features_only:
        return
    _, all_features = prepare_window_features(args.feature_cache)
    windows = WINDOW_SECONDS if args.window == "all" else (int(args.window),)
    for window in windows:
        config = WindowExperimentConfig(
            window_seconds=window, forecast_steps=args.forecast_steps,
            separation_steps=args.separation_steps, batch_size=args.batch_size,
            device=args.device,
        )
        run_window(config, all_features[window], args.experiments_root / f"window_{window}s")
    if args.window == "all":
        summary = aggregate(args.experiments_root, Path("results/window_resolution_results.json"))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

