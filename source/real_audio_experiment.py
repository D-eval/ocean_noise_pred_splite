"""Four-condition field-audio experiment with paired evaluation."""
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

from .data import SpectrumNormaliser, load_experiment_frames
from .metrics import paired_bootstrap_interval
from .model import ForecastConditionedSeparator, multi_resolution_loss
from .real_audio_data import REAL_AUDIO_ROOT, RealAudioMixtureDataset, load_manifest
from .train import _to_device, engineering_usability_gate, evaluate


DISPLAY_NAMES = {
    "none_film": "BS-RoFormer",
    "oracle_film": "Oracle-FC-BSR",
    "film": "Forecast-FC-BSR",
    "shuffled_film": "Shuffled-FC-BSR",
}
CONDITIONS = {
    "none_film": "none",
    "oracle_film": "oracle",
    "film": "forecast",
    "shuffled_film": "shuffled",
}
STRENGTH_LEVELS_DB = tuple(float(value) for value in range(-18, 13, 3))


@dataclass
class RealExperimentConfig:
    variant: str
    steps: int = 2500
    batch_size: int = 4
    learning_rate: float = 1e-4
    forecast_weight: float = 0.2
    validate_every: int = 250
    context_hours: int = 24
    sample_rate: int = 8000
    seconds: float = 5.0
    clips_per_hour: int = 96
    validation_clips_per_hour: int = 24
    test_clips_per_hour: int = 48
    strength_clips_per_hour: int = 24
    train_snr_low_db: float = -12.0
    train_snr_high_db: float = 12.0
    seed: int = 20261021
    model_dim: int = 32
    latent_dim: int = 16
    depth: int = 2
    heads: int = 4
    dim_head: int = 8
    n_fft: int = 256
    hop_length: int = 64
    bands: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    initial_checkpoint: str = "models/practical_film.pt"


def _datasets(config: RealExperimentConfig, snr_range: tuple[float, float] | None = None):
    frames = load_experiment_frames()
    normaliser = SpectrumNormaliser.fit([frames["MB03_01"], frames["MB03_02"]])
    manifest = load_manifest()
    condition = CONDITIONS[config.variant]
    common = dict(
        normaliser=normaliser,
        records=manifest["records"],
        condition=condition,
        context_hours=config.context_hours,
        sample_rate=config.sample_rate,
        seconds=config.seconds,
        seed=config.seed,
        source_profile="broad",
        snr_range=snr_range or (config.train_snr_low_db, config.train_snr_high_db),
    )
    sets = {}
    for name in frames:
        clips = (
            config.validation_clips_per_hour if name == "MB03_03"
            else config.test_clips_per_hour if name == "MB03_04"
            else config.clips_per_hour
        )
        sets[name] = RealAudioMixtureDataset(frames[name], name, clips_per_hour=clips, **common)
    return frames, normaliser, sets


def _initialise_transfer(model: ForecastConditionedSeparator, checkpoint_path: Path) -> dict:
    """Give every arm the same synthetic-pretrained separator; load forecast only where used."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    separator = {
        key.removeprefix("separator."): value
        for key, value in state.items()
        if key.startswith("separator.")
    }
    missing, unexpected = model.separator.load_state_dict(separator, strict=False)
    # All four field-audio arms use the same FiLM separator layout.
    if unexpected or missing:
        raise RuntimeError(f"separator transfer mismatch: missing={missing}, unexpected={unexpected}")
    forecast_loaded = False
    if model.forecaster is not None:
        forecaster = {
            key.removeprefix("forecaster."): value
            for key, value in state.items()
            if key.startswith("forecaster.")
        }
        model.forecaster.load_state_dict(forecaster, strict=True)
        forecast_loaded = True
    return {"separator_loaded": True, "forecaster_loaded": forecast_loaded}


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_variant(config: RealExperimentConfig, output: Path) -> dict:
    if config.variant not in DISPLAY_NAMES:
        raise ValueError(f"variant must be one of {tuple(DISPLAY_NAMES)}")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device)
    frames, normaliser, sets = _datasets(config)
    generator = torch.Generator().manual_seed(config.seed)
    train = ConcatDataset([sets["MB03_01"], sets["MB03_02"]])
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(sets["MB03_03"], batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(sets["MB03_04"], batch_size=config.batch_size, shuffle=False)
    model = ForecastConditionedSeparator(
        spectrum_dim=len(frames["MB03_01"].columns), variant=config.variant,
        latent_dim=config.latent_dim, initialization_seed=config.seed,
        model_dim=config.model_dim, depth=config.depth, heads=config.heads,
        dim_head=config.dim_head, n_fft=config.n_fft,
        hop_length=config.hop_length, bands=config.bands,
    ).to(device)
    transfer = _initialise_transfer(model, Path(config.initial_checkpoint))
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.steps)
    output.mkdir(parents=True, exist_ok=True)
    metadata = asdict(config) | {
        "display_name": DISPLAY_NAMES[config.variant],
        "condition": CONDITIONS[config.variant],
        "background_waveform": "time-aligned real NOAA SanctSound current-hour FLAC",
        "target": "clean broad synthetic source injected at measured waveform RMS",
        "source_distribution": "general acoustic, five whale-contour, and two vessel families; vessel tonal bands optionally use 5--20 Hz LFO",
        "history": "24 hourly TOL rows ending at t-1",
        "normalisation": "mean/std fitted on MB03_01+02 only",
        "train_deployments": ["MB03_01", "MB03_02"],
        "validation_deployment": "MB03_03",
        "test_deployment": "MB03_04",
        "transfer": transfer,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "normaliser_mean": normaliser.mean.tolist(),
        "normaliser_scale": normaliser.scale.tolist(),
        "spectrum_columns": list(frames["MB03_01"].columns),
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history = []
    best_score = float("-inf")
    iterator = iter(train_loader)
    for step in range(1, config.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _to_device(batch, device)
        values = model(batch["mixture"], batch["history"], batch.get("external_condition"))
        separation_loss = multi_resolution_loss(values["estimate"], batch["source"])
        forecast_loss = separation_loss.new_zeros(())
        if values["prediction"] is not None:
            forecast_loss = torch.nn.functional.l1_loss(values["prediction"], batch["target_spectrum"])
        loss = separation_loss + config.forecast_weight * forecast_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        scheduler.step()
        if step == 1 or step % config.validate_every == 0 or step == config.steps:
            row = {
                "step": step, "loss": float(loss.detach()),
                "separation_loss": float(separation_loss.detach()),
                "forecast_loss": float(forecast_loss.detach()),
            }
            if step % config.validate_every == 0 or step == config.steps:
                validation, _ = evaluate(model, validation_loader, normaliser, device)
                row["validation_source_si_sdri_db"] = validation["source_si_sdri_db"]
                if validation["source_si_sdri_db"] > best_score:
                    best_score = validation["source_si_sdri_db"]
                    torch.save({"model": model.state_dict(), "config": metadata, "step": step}, output / "best.pt")
            history.append(row)
            (output / "training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps({"variant": config.variant} | row), flush=True)

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test, rows = evaluate(model, test_loader, normaliser, device)
    strength = {}
    for snr in STRENGTH_LEVELS_DB:
        strength_config = RealExperimentConfig(**(
            asdict(config) | {"test_clips_per_hour": config.strength_clips_per_hour}
        ))
        _, _, strength_sets = _datasets(strength_config, (snr, snr))
        summary, _ = evaluate(
            model, DataLoader(strength_sets["MB03_04"], batch_size=config.batch_size), normaliser, device
        )
        strength[f"{snr:+.0f}_dB"] = summary
    result = {
        "variant": config.variant,
        "display_name": DISPLAY_NAMES[config.variant],
        "condition": CONDITIONS[config.variant],
        "best_step": checkpoint["step"],
        "test": test,
        "strength_sweep": strength,
        "waveform_window_seconds": config.seconds,
        "engineering_usability_gate": engineering_usability_gate(test, strength),
        "scope": "real current-hour NOAA SanctSound waveform; synthetic injected target",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_rows(output / "test_examples.csv", rows)
    return result


def aggregate(experiments_root: Path, output_path: Path) -> dict:
    results = {}
    rows = {}
    for variant in DISPLAY_NAMES:
        folder = experiments_root / f"real_{variant}"
        results[variant] = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        with (folder / "test_examples.csv").open(encoding="utf-8", newline="") as stream:
            rows[variant] = list(csv.DictReader(stream))
    baseline = {row["example_id"]: float(row["source_si_sdri_db"]) for row in rows["none_film"]}
    table = {}
    for variant in DISPLAY_NAMES:
        paired = np.asarray([
            float(row["source_si_sdri_db"]) - baseline[row["example_id"]]
            for row in rows[variant]
        ])
        # Clips from one continuous hour are not independent. Bootstrap the six
        # hour-level means rather than pretending all 576 clips are iid.
        by_hour: dict[str, list[float]] = {}
        for row, difference in zip(rows[variant], paired, strict=True):
            by_hour.setdefault(row["timestamp"], []).append(float(difference))
        hour_means = np.asarray([np.mean(values) for values in by_hour.values()])
        low, high = paired_bootstrap_interval(hour_means)
        table[variant] = {
            "model": DISPLAY_NAMES[variant],
            "condition": CONDITIONS[variant],
            "source_si_sdri_db": results[variant]["test"]["source_si_sdri_db"],
            "source_si_sdri_ci95_db": results[variant]["test"]["source_si_sdri_ci95_db"],
            "paired_gain_over_baseline_db": float(paired.mean()),
            "paired_gain_ci95_db": [low, high],
            "paired_gain_ci_method": "bootstrap over six current-recording hours",
            "ocean_si_sdr_db": results[variant]["test"]["ocean_si_sdr_db"],
            "forecast_mae_db": results[variant]["test"]["forecast_mae_db"],
            "real_time_factor": results[variant]["test"]["real_time_factor"],
            "engineering_gate_passed": results[variant]["engineering_usability_gate"]["passed"],
            "strength_sweep": results[variant]["strength_sweep"],
        }
    proxy = json.loads(Path("results/broad_training_results.json").read_text(encoding="utf-8"))
    summary = {
        "protocol": {
            "current_background": "real NOAA SanctSound audio at t",
            "history": "real-recording-derived TOL at t-24...t-1",
            "ground_truth": "noise=N_t, signal=s_t, mixture=N_t+alpha*s_t",
            "paired_test_examples": len(baseline),
            "waveform_window_seconds": results["film"]["waveform_window_seconds"],
            "strength_sweep_snr_db": list(STRENGTH_LEVELS_DB),
            "selection": "best MB03_03 validation checkpoint; MB03_04 test used once",
        },
        "conditions": table,
        "surrogate_comparison": {
            "surrogate_baseline_si_sdri_db": proxy["unconditioned"]["source_si_sdri_db"],
            "surrogate_forecast_film_si_sdri_db": proxy["conditioning_ablation"]["film"]["test_si_sdri_db"],
            "real_baseline_si_sdri_db": table["none_film"]["source_si_sdri_db"],
            "real_forecast_si_sdri_db": table["film"]["source_si_sdri_db"],
            "warning": "absolute proxy-vs-real differences include a transfer/fine-tuning protocol change; paired gains within each background domain are the causal comparison",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(DISPLAY_NAMES) + ("all",), default="all")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--clips-per-hour", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    variants = tuple(DISPLAY_NAMES) if args.variant == "all" else (args.variant,)
    for variant in variants:
        config = RealExperimentConfig(
            variant=variant, steps=args.steps, clips_per_hour=args.clips_per_hour,
            batch_size=args.batch_size, device=args.device,
        )
        run_variant(config, args.experiments_root / f"real_{variant}")
    if args.variant == "all":
        print(json.dumps(aggregate(args.experiments_root, Path("results/real_audio_ablation.json")), indent=2))


if __name__ == "__main__":
    main()
