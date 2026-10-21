"""Train one forecast-conditioned separation ablation and evaluate it once."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .data import DEFAULT_DATA_ROOT, SpectrumNormaliser, TOLMixtureDataset, load_experiment_frames
from .metrics import paired_bootstrap_interval, si_sdr
from .model import ForecastConditionedSeparator, multi_resolution_loss


@dataclass
class ExperimentConfig:
    variant: str = "latent"
    context_hours: int = 24
    sample_rate: int = 8000
    seconds: float = 0.5
    steps: int = 1000
    batch_size: int = 8
    learning_rate: float = 2e-4
    forecast_weight: float = 0.2
    validate_every: int = 100
    validation_examples: int = 128
    test_examples: int = 256
    strength_examples: int = 128
    train_snr_low_db: float = -12.0
    train_snr_high_db: float = 12.0
    train_examples_per_deployment: int | None = None
    seed: int = 20261021
    model_dim: int = 64
    latent_dim: int = 32
    depth: int = 2
    heads: int = 4
    dim_head: int = 16
    n_fft: int = 256
    hop_length: int = 64
    bands: int = 24
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: ForecastConditionedSeparator,
    loader: DataLoader,
    normaliser: SpectrumNormaliser,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    model.eval()
    rows: list[dict] = []
    elapsed = 0.0
    audio_seconds = 0.0
    for batch in loader:
        batch = _to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        began = time.perf_counter()
        output = model(batch["mixture"], batch["history"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - began
        audio_seconds += batch["mixture"].numel() / loader.dataset.sample_rate
        estimate = output["estimate"]
        length = min(estimate.shape[-1], batch["mixture"].shape[-1])
        estimate = estimate[..., :length]
        source = batch["source"][..., :length]
        mixture = batch["mixture"][..., :length]
        ocean = batch["ocean"][..., :length]
        source_score = si_sdr(estimate, source)
        mixture_score = si_sdr(mixture, source)
        ocean_estimate = mixture - estimate
        ocean_score = si_sdr(ocean_estimate, ocean)
        prediction = output["prediction"]
        if prediction is None:
            forecast_mae_db = torch.full_like(source_score, float("nan"))
        else:
            scale = torch.as_tensor(normaliser.scale, device=device)
            forecast_mae_db = ((prediction - batch["target_spectrum"]).abs() * scale).mean(-1)
        for index in range(len(source_score)):
            rows.append(
                {
                    "deployment": batch["deployment"][index],
                    "timestamp": batch["timestamp"][index],
                    "origin": int(batch["origin"][index]),
                    "snr_db": float(batch["snr_db"][index]),
                    "source_si_sdr_db": float(source_score[index]),
                    "mixture_si_sdr_db": float(mixture_score[index]),
                    "source_si_sdri_db": float(source_score[index] - mixture_score[index]),
                    "ocean_si_sdr_db": float(ocean_score[index]),
                    "forecast_mae_db": float(forecast_mae_db[index]),
                }
            )
    improvements = np.asarray([row["source_si_sdri_db"] for row in rows])
    low, high = paired_bootstrap_interval(improvements)
    summary = {
        "examples": len(rows),
        "source_si_sdr_db": float(np.mean([row["source_si_sdr_db"] for row in rows])),
        "mixture_si_sdr_db": float(np.mean([row["mixture_si_sdr_db"] for row in rows])),
        "source_si_sdri_db": float(improvements.mean()),
        "source_si_sdri_ci95_db": [low, high],
        "ocean_si_sdr_db": float(np.mean([row["ocean_si_sdr_db"] for row in rows])),
        "forecast_mae_db": float(np.nanmean([row["forecast_mae_db"] for row in rows]))
        if model.variant != "none"
        else None,
        "real_time_factor": elapsed / max(audio_seconds, 1e-9),
        "usable_real_time": elapsed < audio_seconds,
    }
    model.train()
    return summary, rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def engineering_usability_gate(test_summary: dict, strength_sweep: dict) -> dict:
    """Pre-deployment gate separating enhancement from already-strong quality.

    At low input SNR the meaningful criterion is improvement over the mixture;
    at high input SNR the mixture is already target-dominant, so absolute target
    SI-SDR is the appropriate guard against damage.
    """
    checks = {
        "overall_si_sdri_at_least_5_db": test_summary["source_si_sdri_db"] >= 5.0,
        "weak_to_equal_snr_si_sdri_at_least_3_db": all(
            strength_sweep[key]["source_si_sdri_db"] >= 3.0
            for key in ("-12_dB", "-6_dB", "+0_dB")
        ),
        "strong_signal_source_si_sdr_at_least_8_db": all(
            strength_sweep[key]["source_si_sdr_db"] >= 8.0 for key in ("+6_dB", "+12_dB")
        ),
        "real_time_factor_below_1": test_summary["real_time_factor"] < 1.0,
    }
    return checks | {"passed": all(checks.values())}


def run_experiment(config: ExperimentConfig, output: Path, data_root: Path = DEFAULT_DATA_ROOT) -> dict:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device)
    frames = load_experiment_frames(data_root)
    normaliser = SpectrumNormaliser.fit([frames["MB03_01"], frames["MB03_02"]])
    dataset_kwargs = dict(
        normaliser=normaliser,
        context_hours=config.context_hours,
        sample_rate=config.sample_rate,
        seconds=config.seconds,
        seed=config.seed,
    )
    train_dataset_kwargs = dataset_kwargs | {
        "snr_range": (config.train_snr_low_db, config.train_snr_high_db)
    }
    train_sets = [
        TOLMixtureDataset(
            frames[name], name, max_examples=config.train_examples_per_deployment, **train_dataset_kwargs
        )
        for name in ("MB03_01", "MB03_02")
    ]
    val_set = TOLMixtureDataset(
        frames["MB03_03"], "MB03_03", max_examples=config.validation_examples, **dataset_kwargs
    )
    test_set = TOLMixtureDataset(
        frames["MB03_04"], "MB03_04", max_examples=config.test_examples, **dataset_kwargs
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        ConcatDataset(train_sets), batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0
    )
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = ForecastConditionedSeparator(
        spectrum_dim=len(frames["MB03_01"].columns),
        variant=config.variant,
        latent_dim=config.latent_dim,
        initialization_seed=config.seed,
        model_dim=config.model_dim,
        depth=config.depth,
        heads=config.heads,
        dim_head=config.dim_head,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        bands=config.bands,
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.steps)
    output.mkdir(parents=True, exist_ok=True)
    metadata = asdict(config) | {
        "data_root": str(data_root.resolve()),
        "train_deployments": ["MB03_01", "MB03_02"],
        "validation_deployment": "MB03_03",
        "test_deployment": "MB03_04",
        "condition_cutoff": "history ends at t-1; target/background spectrum is t",
        "background_waveform": "deterministic random-phase reconstruction from measured hourly TOL envelope",
        "normalisation": "mean/std fitted on MB03_01+02 only",
        "initialisation": "component-local seeds: separator seed+1, forecaster seed+2; separator core matched across variants",
        "normaliser_mean": normaliser.mean.tolist(),
        "normaliser_scale": normaliser.scale.tolist(),
        "spectrum_columns": list(frames["MB03_01"].columns),
        "train_examples": sum(len(dataset) for dataset in train_sets),
        "validation_examples_actual": len(val_set),
        "test_examples_actual": len(test_set),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history: list[dict] = []
    best_score = float("-inf")
    iterator = iter(train_loader)
    for step in range(1, config.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _to_device(batch, device)
        output_values = model(batch["mixture"], batch["history"])
        separation_loss = multi_resolution_loss(output_values["estimate"], batch["source"])
        if output_values["prediction"] is None:
            forecast_loss = separation_loss.new_zeros(())
        else:
            forecast_loss = torch.nn.functional.l1_loss(
                output_values["prediction"], batch["target_spectrum"]
            )
        loss = separation_loss + config.forecast_weight * forecast_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        scheduler.step()
        if step == 1 or step % config.validate_every == 0 or step == config.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "separation_loss": float(separation_loss.detach()),
                "forecast_loss": float(forecast_loss.detach()),
                "learning_rate": optimiser.param_groups[0]["lr"],
            }
            if step % config.validate_every == 0 or step == config.steps:
                validation, _ = evaluate(model, val_loader, normaliser, device)
                row.update({f"validation_{key}": value for key, value in validation.items()})
                if validation["source_si_sdri_db"] > best_score:
                    best_score = validation["source_si_sdri_db"]
                    torch.save({"model": model.state_dict(), "config": metadata, "step": step}, output / "best.pt")
            history.append(row)
            (output / "training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_summary, test_rows = evaluate(model, test_loader, normaliser, device)
    strength_sweep = {}
    for snr_db in (-12.0, -6.0, 0.0, 6.0, 12.0):
        strength_set = TOLMixtureDataset(
            frames["MB03_04"], "MB03_04", max_examples=config.strength_examples,
            snr_range=(snr_db, snr_db), **dataset_kwargs
        )
        strength_loader = DataLoader(
            strength_set, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        strength_summary, _ = evaluate(model, strength_loader, normaliser, device)
        strength_sweep[f"{snr_db:+.0f}_dB"] = strength_summary
    result = {
        "variant": config.variant,
        "best_step": checkpoint["step"],
        "test": test_summary,
        "strength_sweep": strength_sweep,
        "engineering_usability_gate": engineering_usability_gate(test_summary, strength_sweep),
        "scope": "surrogate-waveform experiment; not raw-audio field validation",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_rows(output / "test_examples.csv", test_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=ForecastConditionedSeparator.VARIANTS, default="latent")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-hours", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=0.5)
    parser.add_argument("--validation-examples", type=int, default=128)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--strength-examples", type=int, default=128)
    parser.add_argument("--train-snr-low-db", type=float, default=-12.0)
    parser.add_argument("--train-snr-high-db", type=float, default=12.0)
    parser.add_argument("--train-examples-per-deployment", type=int)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--small", action="store_true", help="use the smoke-test BS-RoFormer size")
    parser.add_argument("--practical", action="store_true", help="balanced local deployment profile")
    args = parser.parse_args()
    kwargs = vars(args).copy()
    output = kwargs.pop("output") or Path("experiments") / args.variant
    data_root = kwargs.pop("data_root")
    small = kwargs.pop("small")
    practical = kwargs.pop("practical")
    config = ExperimentConfig(**kwargs)
    if small:
        config.model_dim = 16
        config.latent_dim = 8
        config.depth = 1
        config.heads = 2
        config.dim_head = 8
        config.n_fft = 128
        config.hop_length = 32
        config.bands = 12
    if practical:
        config.model_dim = 32
        config.latent_dim = 16
        config.depth = 2
        config.heads = 4
        config.dim_head = 8
        config.n_fft = 256
        config.hop_length = 64
        config.bands = 20
    print(json.dumps(run_experiment(config, output, data_root), indent=2))


if __name__ == "__main__":
    main()
