"""Train a time-resolved 128-group Oracle residual interface on DCLDE audio.

This experiment does not train or call a GRU. The only Oracle input is the
current, annotation-clean real ocean waveform N_t in a controlled mixture.
It is compared on exactly the old BSR training and held-out mixtures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dclde import CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE
from .dclde_full_oracle_experiment import BASELINE_CHECKPOINT, _check_test_identity
from .dclde_separation_experiment import (
    CausalFeatureStore, ExperimentConfig, InjectedUpcallDataset, _load_transfer,
    _make_model, _seed_everything,
)
from .metrics import paired_bootstrap_interval, si_sdr
from .model import multi_resolution_loss


N_FFT = 512
HOP = 128
GROUPS = 128
BASELINE_ROWS = Path("experiments/dclde_true_bsr/clean_synthetic_test.csv")
BASELINE_RESULT = Path("experiments/dclde_true_bsr/result.json")


class ResidualBandSplit(nn.Module):
    """Fuse X-derived BSR tokens with [log|B|, log|X|-log|B|] before attention."""

    def __init__(self, original: nn.Module, token_bands: int, token_dim: int):
        super().__init__()
        self.original = original
        self.normalise = nn.LayerNorm(GROUPS * 2, elementwise_affine=False)
        self.project = nn.Linear(GROUPS * 2, token_bands * token_dim, bias=False)
        nn.init.zeros_(self.project.weight)
        self.token_bands = token_bands
        self.token_dim = token_dim
        self.context: torch.Tensor | None = None

    def forward(self, spectral_tokens: torch.Tensor) -> torch.Tensor:
        base = self.original(spectral_tokens)
        if self.context is None:
            return base
        if self.context.shape[0] != base.shape[0]:
            raise ValueError("Oracle condition and mixture batch sizes differ")
        aligned = F.interpolate(self.context.transpose(1, 2), size=base.shape[1],
                                mode="linear", align_corners=True).transpose(1, 2)
        correction = self.project(self.normalise(aligned)).view(
            base.shape[0], base.shape[1], self.token_bands, self.token_dim)
        return base + correction


class OracleResidualBSR(nn.Module):
    def __init__(self, bands: int, seed: int, device: torch.device, transfer: Path):
        super().__init__()
        self.backbone = _make_model("bsr", bands, seed, device)
        self.transferred_tensors = _load_transfer(self.backbone, transfer)
        original = self.backbone.separator.injector
        self.residual_split = ResidualBandSplit(original, token_bands=20, token_dim=32)
        self.backbone.separator.separator.band_split = self.residual_split
        self.register_buffer("oracle_window", torch.hann_window(N_FFT), persistent=False)

    @torch.no_grad()
    def oracle_map(self, mixture: torch.Tensor, ocean: torch.Tensor) -> torch.Tensor:
        """128 equal-width STFT power groups per 64-ms frame, retaining time."""
        if mixture.shape != ocean.shape or mixture.ndim != 2:
            raise ValueError("mixture and current background must be matched waveforms")
        joined = torch.cat((mixture, ocean), dim=0)
        spectrum = torch.stft(joined, N_FFT, hop_length=HOP,
                              window=self.oracle_window, return_complex=True)
        powers = spectrum.abs().square()
        ids = torch.arange(powers.shape[1], device=powers.device) * GROUPS // powers.shape[1]
        sums = powers.new_zeros((len(joined), GROUPS, powers.shape[-1]))
        sums.scatter_add_(1, ids[None, :, None].expand(len(joined), -1, powers.shape[-1]), powers)
        counts = torch.bincount(ids, minlength=GROUPS).to(powers).clamp_min(1)
        log_magnitude = 0.5 * torch.log(sums / counts[None, :, None] + 1e-8)
        log_x, log_b = log_magnitude.chunk(2, dim=0)
        innovation = log_x - log_b
        return torch.cat((log_b, innovation), dim=1).transpose(1, 2).contiguous()

    def forward(self, mixture: torch.Tensor, ocean: torch.Tensor) -> torch.Tensor:
        self.residual_split.context = self.oracle_map(mixture, ocean)
        try:
            return self.backbone(mixture, None)["estimate"]
        finally:
            self.residual_split.context = None


@torch.no_grad()
def validation_score(model: OracleResidualBSR, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores = []
    for batch in loader:
        mixture, source, ocean = (batch[key].to(device) for key in ("mixture", "source", "ocean"))
        estimate = model(mixture, ocean)
        length = min(estimate.shape[-1], source.shape[-1])
        improvement = si_sdr(estimate[..., :length], source[..., :length]) - si_sdr(
            mixture[..., :length], source[..., :length])
        scores.extend(improvement.cpu().tolist())
    return float(np.mean(scores))


@torch.no_grad()
def test_rows(model: OracleResidualBSR, loader: DataLoader, device: torch.device) -> list[dict]:
    model.eval()
    rows = []
    for batch in loader:
        mixture, source, ocean = (batch[key].to(device) for key in ("mixture", "source", "ocean"))
        estimate = model(mixture, ocean)
        length = min(estimate.shape[-1], source.shape[-1])
        mixture, source, estimate = mixture[..., :length], source[..., :length], estimate[..., :length]
        mixed_score = si_sdr(mixture, source)
        target_score = si_sdr(estimate, source)
        for index, example_id in enumerate(batch["example_id"]):
            rows.append({
                "example_id": example_id, "day": example_id.split(":", 1)[0],
                "snr_db": float(batch["snr_db"][index]),
                "mixture_si_sdr_db": float(mixed_score[index]),
                "bsr_si_sdri_db": 0.0,  # populated from the checked old baseline below
                "oracle_residual_si_sdri_db": float(target_score[index] - mixed_score[index]),
            })
    with BASELINE_ROWS.open(newline="", encoding="utf-8") as stream:
        baseline = list(csv.DictReader(stream))
    if len(rows) != len(baseline):
        raise ValueError("test count differs from the matched BSR baseline")
    for row, reference in zip(rows, baseline, strict=True):
        row["bsr_si_sdri_db"] = float(reference["source_si_sdri_db"])
    _check_test_identity(rows)
    return rows


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(output: Path, device_name: str, steps: int = 6000) -> dict:
    reference = json.loads(BASELINE_RESULT.read_text(encoding="utf-8"))
    config = ExperimentConfig(device=device_name)
    if steps != reference["updates_run"] or config.seed != reference["config"]["seed"]:
        raise ValueError("Oracle run must match the existing BSR update count and seed")
    if config.batch_size != reference["config"]["batch_size"] or config.separation_validate_every != 100:
        raise ValueError("training batch or validation cadence differs from BSR")
    _seed_everything(config.seed)
    device = torch.device(device_name)
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    model = OracleResidualBSR(store.bands, config.seed, device, Path(config.transfer_checkpoint)).to(device)
    train = InjectedUpcallDataset(store, "train", config.train_examples, config.seed)
    validation = InjectedUpcallDataset(store, "validation", config.validation_examples,
                                       config.seed + 10_000_000, cache_waveforms=True)
    test = InjectedUpcallDataset(store, "test", config.validation_examples,
                                 config.seed + 20_000_000, spread_rows=True)
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(config.seed), num_workers=0)
    validation_loader = DataLoader(validation, batch_size=24, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=24, shuffle=False, num_workers=0)

    # The added map path is exactly zero at initialization. Check actual
    # waveforms against the same transferred, injector-bypassed BSR.
    ordinary = _make_model("bsr", store.bands, config.seed, device)
    ordinary_transferred = _load_transfer(ordinary, Path(config.transfer_checkpoint))
    ordinary.eval(); model.eval()
    probe = next(iter(DataLoader(validation, batch_size=2, shuffle=False)))
    with torch.inference_mode():
        mixed = probe["mixture"].to(device)
        initial_difference = (model(mixed, probe["ocean"].to(device)) -
                              ordinary(mixed, None)["estimate"]).abs().max().item()
    del ordinary
    if initial_difference > 1e-6 or ordinary_transferred != model.transferred_tensors:
        raise ValueError(f"backbone initialization mismatch: {initial_difference:g}")
    print(json.dumps({"stage": "initial", "max_waveform_difference": initial_difference,
                      "transferred_tensors": model.transferred_tensors}), flush=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.separation_steps)
    output.mkdir(parents=True, exist_ok=True)
    training_rows, best, best_step = [], -math.inf, 0
    iterator = iter(train_loader)
    for step in range(1, steps + 1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        mixture, source, ocean = (batch[key].to(device) for key in ("mixture", "source", "ocean"))
        estimate = model(mixture, ocean)
        loss = multi_resolution_loss(estimate, source)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step(); scheduler.step()
        if step == 1 or step % config.separation_validate_every == 0:
            row = {"step": step, "train_loss": float(loss.detach())}
            if step % config.separation_validate_every == 0:
                score = validation_score(model, validation_loader, device)
                row["validation_si_sdri_db"] = score
                if score > best + 1e-4:
                    best, best_step = score, step
                    torch.save({"model": model.state_dict(), "config": vars(config),
                                "step": step, "interface": "oracle_residual_128"}, output / "best.pt")
            training_rows.append(row)
            print(json.dumps({"stage": "separation", **row}), flush=True)
            (output / "training.json").write_text(json.dumps(training_rows, indent=2), encoding="utf-8")

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    rows = test_rows(model, test_loader, device)
    sample_path = output / "clean_synthetic_test.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    days = np.asarray([row["day"] for row in rows])
    baseline = np.asarray([row["bsr_si_sdri_db"] for row in rows])
    oracle = np.asarray([row["oracle_residual_si_sdri_db"] for row in rows])
    delta = oracle - baseline
    day_means = np.asarray([delta[days == day].mean() for day in np.unique(days)])
    summary = {
        "protocol": {
            "dataset": "same annotation-clean DCLDE 5-s real ocean plus synthetic events",
            "examples": len(rows), "train_examples": config.train_examples,
            "backbone": "same transferred mixture-only BS-RoFormer (256-point STFT, 20 tokens, dim 32)",
            "oracle": "current N_t, 512-point STFT, 128 equal-width power groups per 64-ms frame",
            "innovation": "log|X| - log|B| in the same 128-group, time-resolved grid",
            "fusion": "[X-derived BSR tokens, log|B|, innovation] via zero-initialized additive linear projection before axial attention; no FiLM",
            "baseline_training": "existing dclde_true_bsr checkpoint; same seed, transferred tensors, 6000 updates, 100-step validation selection",
            "oracle_training": "6000 updates, same batch/data/seed/optimizer/scheduler/loss and best-validation checkpoint rule",
            "initial_waveform_max_difference": initial_difference,
            "baseline_checkpoint_sha256": _hash_file(BASELINE_CHECKPOINT),
            "interval": "paired bootstrap over held-out recording-day means",
        },
        "baseline_si_sdri_db": float(baseline.mean()),
        "oracle_residual_si_sdri_db": float(oracle.mean()),
        "delta_si_sdri_db": float(delta.mean()),
        "delta_day_ci95_db": [float(value) for value in paired_bootstrap_interval(day_means)],
        "days": {str(day): float(delta[days == day].mean()) for day in np.unique(days)},
        "baseline_best_step": reference["best_step"], "oracle_best_step": best_step,
        "oracle_best_validation_si_sdri_db": best,
        "checkpoint": str(output / "best.pt"), "samples_csv": str(sample_path),
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--output", type=Path, default=Path("experiments/dclde_oracle_residual_128"))
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.device, args.steps), indent=2), flush=True)


if __name__ == "__main__":
    main()
