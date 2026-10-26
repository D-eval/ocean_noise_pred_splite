"""Causal 60-s high-resolution background forecasting and residual separation.

The main conditions are the preceding clean window (persistence) and a GRU
prediction from twelve strictly preceding event-masked 5-s maps. Matched
fixed-mean and wrong-day controls test the adapter and temporal information.
The current clean waveform is used for forecast supervision, never as a
causal separator input. The separate Oracle checkpoint is an upper reference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .dclde import CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE, SAMPLE_RATE
from .dclde_full_oracle_experiment import _check_test_identity
from .dclde_oracle_residual_experiment import (
    BASELINE_RESULT, GROUPS, HOP, N_FFT, OracleResidualBSR, _hash_file,
)
from .dclde_separation_experiment import (
    CausalFeatureStore, ExperimentConfig, InjectedUpcallDataset, _make_model,
    _load_transfer, _seed_everything,
)
from .metrics import paired_bootstrap_interval, si_sdr
from .model import multi_resolution_loss


FRAMES = 79
STEPS_HISTORY = 12
DATA_ROOT = Path("data/dclde2013")
MAPS_PATH = DATA_ROOT / "clean_128x79_logmag.f16.npy"
HISTORY_PATH = DATA_ROOT / "clean_128x79_history_indices.npy"
NORMALIZER_PATH = DATA_ROOT / "clean_128x79_normalizer.npz"
CACHE_META_PATH = DATA_ROOT / "clean_128x79_meta.json"
FORECAST_MAPS_PATH = DATA_ROOT / "forecast_128x79_logmag.f16.npy"
PREDICTOR_DIR = Path("experiments/dclde_highres_gru")
BASELINE_ROWS = Path("experiments/dclde_true_bsr/clean_synthetic_test.csv")
ORACLE_RESULT = Path("results/dclde_oracle_residual_128.json")
FORECAST_META_PATH = DATA_ROOT / "forecast_128x79_meta.json"
OUTPUT_ROOT = Path("experiments")
SUMMARY_PATH = Path("results/dclde_causal_highres_comparison.json")
SUMMARY_ROWS_PATH = Path("results/dclde_causal_highres_test_samples.csv")


@torch.no_grad()
def log_band_map(waveforms: torch.Tensor) -> torch.Tensor:
    """Exactly the Oracle arm's 512-point/128 equal-width log-amplitude map."""
    window = torch.hann_window(N_FFT, device=waveforms.device, dtype=waveforms.dtype)
    spectrum = torch.stft(waveforms, N_FFT, hop_length=HOP,
                          window=window, return_complex=True)
    power = spectrum.abs().square()
    ids = torch.arange(power.shape[1], device=power.device) * GROUPS // power.shape[1]
    sums = power.new_zeros((len(waveforms), GROUPS, power.shape[-1]))
    sums.scatter_add_(1, ids[None, :, None].expand(len(waveforms), -1, power.shape[-1]), power)
    counts = torch.bincount(ids, minlength=GROUPS).to(power).clamp_min(1)
    return (0.5 * torch.log(sums / counts[None, :, None] + 1e-8)).transpose(1, 2)


def prepare_cache(device_name: str = "cuda", batch_size: int = 128) -> dict:
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    frame = store.frame
    signature = _hash_file(CLEAN_BACKGROUND_MANIFEST)
    if CACHE_META_PATH.exists():
        metadata = json.loads(CACHE_META_PATH.read_text(encoding="utf-8"))
        if (metadata.get("manifest_sha256") == signature
                and MAPS_PATH.exists() and HISTORY_PATH.exists() and NORMALIZER_PATH.exists()):
            return metadata
    lookup = {(str(row.path), int(round(float(row.start_seconds) / 5))): index
              for index, row in enumerate(frame.itertuples())}
    history = np.full((len(frame), STEPS_HISTORY), -1, dtype=np.int32)
    masked_windows = masked_rows = 0
    for index in np.flatnonzero(store.eligible):
        row = frame.iloc[int(index)]
        block = int(round(float(row.start_seconds) / 5))
        previous = [lookup.get((str(row.path), block - offset), -1)
                    for offset in range(STEPS_HISTORY, 0, -1)]
        if min(previous) < 0:
            masked_rows += 1
            masked_windows += sum(position < 0 for position in previous)
            observed = np.flatnonzero(np.asarray(previous) >= 0)
            if len(observed) == 0:
                raise ValueError(f"eligible row {index} has no clean historical window")
            for missing in np.flatnonzero(np.asarray(previous) < 0):
                nearest = observed[np.argmin(np.abs(observed - missing))]
                previous[missing] = previous[nearest]
        if any(frame.iloc[position]["split"] != row["split"] for position in previous):
            raise ValueError("history crosses a recording-day split")
        history[index] = previous
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(HISTORY_PATH, history)
    maps = np.lib.format.open_memmap(MAPS_PATH, mode="w+", dtype=np.float16,
                                     shape=(len(frame), FRAMES, GROUPS))
    device = torch.device(device_name)
    for file_index, (path, group) in enumerate(frame.groupby("path", sort=False), start=1):
        audio, rate = sf.read(path, dtype="float32")
        if rate != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError(f"expected 2-kHz mono WAV: {path}")
        row_indices = group.index.to_numpy()
        starts = np.rint(group["start_seconds"].to_numpy() * SAMPLE_RATE).astype(np.int64)
        for first in range(0, len(group), batch_size):
            indices = row_indices[first:first + batch_size]
            offsets = starts[first:first + batch_size]
            batch = np.stack([audio[offset:offset + 5 * SAMPLE_RATE] for offset in offsets])
            if batch.shape[-1] != 5 * SAMPLE_RATE:
                raise ValueError(f"short 5-s window in {path}")
            features = log_band_map(torch.from_numpy(batch).to(device)).cpu().numpy()
            if features.shape[1:] != (FRAMES, GROUPS):
                raise ValueError(f"unexpected STFT map shape {features.shape}")
            maps[indices] = features.astype(np.float16)
        if file_index % 100 == 0:
            maps.flush()
            print(json.dumps({"stage": "feature_cache", "files": file_index,
                              "of": frame.path.nunique()}), flush=True)
    maps.flush()
    train_indices = np.flatnonzero(frame["split"].eq("train").to_numpy())
    total = np.zeros(GROUPS, dtype=np.float64)
    square = np.zeros(GROUPS, dtype=np.float64)
    count = 0
    for first in range(0, len(train_indices), 512):
        values = np.asarray(maps[train_indices[first:first + 512]], dtype=np.float32)
        total += values.sum(axis=(0, 1), dtype=np.float64)
        square += np.square(values, dtype=np.float64).sum(axis=(0, 1))
        count += values.shape[0] * values.shape[1]
    mean = (total / count).astype(np.float32)
    std = np.sqrt(np.maximum(square / count - np.square(total / count), 1e-6)).astype(np.float32)
    np.savez(NORMALIZER_PATH, mean=mean, std=std)
    metadata = {
        "manifest_sha256": signature, "rows": len(frame),
        "eligible": int(store.eligible.sum()), "history_steps": STEPS_HISTORY,
        "shape": [len(frame), FRAMES, GROUPS], "dtype": "float16",
        "stft_n_fft": N_FFT, "stft_hop": HOP,
        "rule": "only annotation-clean, same-file, strictly earlier 12 windows; masked events filled by nearest clean window inside past 60 seconds",
        "masked_history_rows": masked_rows, "masked_history_windows": masked_windows,
        "maps_path": str(MAPS_PATH), "history_path": str(HISTORY_PATH),
        "normalizer_path": str(NORMALIZER_PATH),
    }
    CACHE_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class HighResolutionStore:
    def __init__(self, store: CausalFeatureStore):
        self.store = store
        self.metadata = json.loads(CACHE_META_PATH.read_text(encoding="utf-8"))
        if self.metadata["manifest_sha256"] != _hash_file(CLEAN_BACKGROUND_MANIFEST):
            raise ValueError("high-resolution cache was built for a different manifest")
        self.maps = np.load(MAPS_PATH, mmap_mode="r")
        self.history_indices = np.load(HISTORY_PATH, mmap_mode="r")
        normalizer = np.load(NORMALIZER_PATH)
        self.mean = normalizer["mean"].astype(np.float32)
        self.std = normalizer["std"].astype(np.float32)
        if self.maps.shape != (len(store.frame), FRAMES, GROUPS):
            raise ValueError("high-resolution map cache shape differs from manifest")

    def history(self, index: int) -> np.ndarray:
        indices = self.history_indices[index]
        if np.any(indices < 0):
            raise ValueError("current window lacks a full clean causal history")
        return np.asarray(self.maps[indices], dtype=np.float32)

    def current(self, index: int) -> np.ndarray:
        return np.asarray(self.maps[index], dtype=np.float32)


class ForecastDataset(Dataset):
    def __init__(self, cache: HighResolutionStore, split: str):
        self.cache = cache
        self.rows = cache.store.indices(split, label=0)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict:
        index = int(self.rows[item])
        return {
            "history": torch.from_numpy(self.cache.history(index)),
            "target": torch.from_numpy(self.cache.current(index)),
        }


class HighResGRU(nn.Module):
    """Small GRU over 12 encoded maps; zero-initialized residual to persistence."""
    def __init__(self, mean: np.ndarray, std: np.ndarray, hidden: int = 256):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).view(1, 1, 1, GROUPS))
        self.register_buffer("std", torch.as_tensor(std).view(1, 1, 1, GROUPS))
        self.encoder = nn.Sequential(nn.Linear(4 * GROUPS, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.decoder = nn.Linear(hidden, FRAMES * GROUPS)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[-3:] != (STEPS_HISTORY, FRAMES, GROUPS):
            raise ValueError(f"expected [batch,12,{FRAMES},{GROUPS}] history")
        normalized = (history - self.mean) / self.std
        batch = history.shape[0]
        pooled = F.adaptive_avg_pool2d(normalized.reshape(batch * STEPS_HISTORY, 1, FRAMES, GROUPS),
                                       (4, GROUPS)).reshape(batch, STEPS_HISTORY, 4 * GROUPS)
        _, hidden = self.gru(self.encoder(pooled))
        predicted_normalized = normalized[:, -1] + self.decoder(hidden[-1]).view(batch, FRAMES, GROUPS)
        return predicted_normalized * self.std[:, 0] + self.mean[:, 0]


@torch.no_grad()
def forecast_error(model: HighResGRU, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_forecast = total_persistence = count = 0
    for batch in loader:
        history = batch["history"].to(device)
        target = batch["target"].to(device)
        predicted = model(history)
        total_forecast += (predicted - target).abs().sum().item()
        total_persistence += (history[:, -1] - target).abs().sum().item()
        count += target.numel()
    return {"forecast_mae_log": total_forecast / count,
            "persistence_mae_log": total_persistence / count}


def train_forecaster(cache: HighResolutionStore, device_name: str = "cuda",
                     steps: int = 3000) -> dict:
    device = torch.device(device_name)
    _seed_everything(20261025)
    train = ForecastDataset(cache, "train")
    validation = ForecastDataset(cache, "validation")
    loader = DataLoader(train, batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(20261025), num_workers=0)
    validation_loader = DataLoader(validation, batch_size=32, shuffle=False, num_workers=0)
    model = HighResGRU(cache.mean, cache.std).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    initial = forecast_error(model, validation_loader, device)
    best = math.inf
    history_rows = [{"step": 0, **initial}]
    PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    for step in range(1, steps + 1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        source = batch["history"].to(device)
        target = batch["target"].to(device)
        estimate = model(source)
        loss = F.smooth_l1_loss((estimate - model.mean[:, 0]) / model.std[:, 0],
                                (target - model.mean[:, 0]) / model.std[:, 0])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step(); scheduler.step()
        if step % 250 == 0 or step == steps:
            metrics = forecast_error(model, validation_loader, device)
            row = {"step": step, "train_loss": float(loss.detach()), **metrics}
            history_rows.append(row)
            print(json.dumps({"stage": "forecast", **row}), flush=True)
            if metrics["forecast_mae_log"] < best - 1e-5:
                best = metrics["forecast_mae_log"]
                torch.save({"model": model.state_dict(), "step": step}, PREDICTOR_DIR / "best.pt")
            (PREDICTOR_DIR / "training.json").write_text(json.dumps(history_rows, indent=2), encoding="utf-8")
    selected = torch.load(PREDICTOR_DIR / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    final = forecast_error(model, validation_loader, device)
    result = {"best_step": selected["step"], "train_examples": len(train),
              "validation_examples": len(validation), "steps_run": steps,
              "validation": final, "skill_over_persistence": 1 - final["forecast_mae_log"] / final["persistence_mae_log"],
              "checkpoint": str(PREDICTOR_DIR / "best.pt")}
    (PREDICTOR_DIR / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@torch.no_grad()
def prepare_forecasts(cache: HighResolutionStore, device_name: str = "cuda",
                      batch_size: int = 64) -> dict:
    checkpoint_path = PREDICTOR_DIR / "best.pt"
    signature = _hash_file(checkpoint_path)
    if FORECAST_META_PATH.exists():
        metadata = json.loads(FORECAST_META_PATH.read_text(encoding="utf-8"))
        if (metadata.get("predictor_sha256") == signature and
                metadata.get("manifest_sha256") == cache.metadata["manifest_sha256"] and
                metadata.get("rows") == int(cache.store.eligible.sum()) and
                FORECAST_MAPS_PATH.exists()):
            return metadata
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = HighResGRU(cache.mean, cache.std).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    indices = np.flatnonzero(cache.store.eligible).astype(np.int32)
    output = np.lib.format.open_memmap(FORECAST_MAPS_PATH, mode="w+", dtype=np.float16,
                                       shape=(len(indices), FRAMES, GROUPS))
    for first in range(0, len(indices), batch_size):
        selected = indices[first:first + batch_size]
        prior = cache.history_indices[selected]
        if np.any(prior < 0) or np.any(prior >= selected[:, None]):
            raise ValueError("forecast inputs must contain only earlier clean rows")
        history = np.asarray(cache.maps[prior], dtype=np.float32)
        predicted = model(torch.from_numpy(history).to(device)).cpu().numpy()
        output[first:first + len(selected)] = predicted.astype(np.float16)
        if first and first % (batch_size * 100) == 0:
            output.flush()
            print(json.dumps({"stage": "forecast_maps", "rows": first,
                              "of": len(indices)}), flush=True)
    output.flush()
    metadata = {"predictor_sha256": signature, "manifest_sha256": cache.metadata["manifest_sha256"],
                "rows": len(indices), "shape": [len(indices), FRAMES, GROUPS],
                "causal": "all 12 input maps precede current window; no current target passed to model"}
    FORECAST_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class ConditionedInjectedDataset(InjectedUpcallDataset):
    def __init__(self, cache: HighResolutionStore, split: str, examples: int,
                 seed: int, condition: str, spread_rows: bool = False,
                 cache_waveforms: bool = False, target_pool=None):
        super().__init__(cache.store, split, examples, seed, spread_rows=spread_rows,
                         cache_waveforms=cache_waveforms, target_pool=target_pool)
        if condition not in {"zero", "shuffled", "persistence", "forecast"}:
            raise ValueError("unknown residual condition")
        self.cache = cache
        self.condition = condition
        if condition == "zero":
            # A fixed training-set prior is a better-formed null condition than
            # B=0 in log magnitude (which implies unit amplitude at every bin).
            self.constant_background = np.broadcast_to(
                cache.mean[None, :], (FRAMES, GROUPS)).copy()
        if condition == "shuffled":
            # Validation contains only one day. Its wrong-date donors come
            # from train days; train/test donors stay inside their own split.
            donor_split = "train" if split == "validation" else split
            donor_rows = cache.store.indices(donor_split, label=0)
            donor_days = cache.store.frame.iloc[donor_rows]["day"].to_numpy()
            self.donors_by_day = {
                str(day): donor_rows[donor_days == day]
                for day in np.unique(donor_days)
            }
            self.current_days = cache.store.frame.iloc[self.rows]["day"].to_numpy()
            if not all(any(day != str(current) for day in self.donors_by_day)
                       for current in np.unique(self.current_days)):
                raise ValueError("shuffled control requires a different recording day")
        if condition == "forecast":
            self.forecasts = np.load(FORECAST_MAPS_PATH, mmap_mode="r")
            self.positions = np.full(len(cache.store.frame), -1, dtype=np.int32)
            self.positions[np.flatnonzero(cache.store.eligible)] = np.arange(len(self.forecasts))

    def __getitem__(self, item: int) -> dict:
        answer = super().__getitem__(item)
        local = (round(item * (len(self.rows) - 1) / max(1, self.examples - 1))
                 if self.spread_rows and self.examples > 1 else item % len(self.rows))
        index = int(self.rows[local])
        if self.condition == "zero":
            condition_map = self.constant_background
        elif self.condition == "shuffled":
            current_day = str(self.current_days[local])
            candidates = sorted(day for day in self.donors_by_day if day != current_day)
            rng = np.random.default_rng(self.seed + item * 104729 + 271828)
            donor_day = candidates[int(rng.integers(len(candidates)))]
            pool = self.donors_by_day[donor_day]
            donor = int(pool[int(rng.integers(len(pool)))])
            if donor == index:
                raise ValueError("shuffled condition selected the current background")
            condition_map = self.cache.current(donor)
            answer["condition_day"] = donor_day
            answer["condition_manifest_index"] = donor
        elif self.condition == "persistence":
            prior = int(self.cache.history_indices[index, -1])
            if prior >= index or prior < 0:
                raise ValueError("persistence condition is not causal")
            condition_map = self.cache.current(prior)
        else:
            position = int(self.positions[index])
            if position < 0:
                raise ValueError("missing frozen GRU prediction")
            condition_map = np.asarray(self.forecasts[position], dtype=np.float32)
        answer["condition_map"] = torch.from_numpy(condition_map)
        return answer


class CausalResidualBSR(OracleResidualBSR):
    """Same Oracle residual adapter and backbone, with an external map."""
    def forward(self, mixture: torch.Tensor, condition_map: torch.Tensor) -> torch.Tensor:
        if condition_map.shape != (len(mixture), FRAMES, GROUPS):
            raise ValueError("condition map must be [batch,79,128]")
        with torch.no_grad():
            mixture_map = log_band_map(mixture)
            innovation = mixture_map - condition_map
            context = torch.cat((condition_map, innovation), dim=-1)
        self.residual_split.context = context
        try:
            return self.backbone(mixture, None)["estimate"]
        finally:
            self.residual_split.context = None


@torch.no_grad()
def separation_validation(model: CausalResidualBSR, loader: DataLoader,
                          device: torch.device) -> float:
    model.eval()
    scores = []
    for batch in loader:
        mixture, source, condition = (batch[key].to(device) for key in
                                      ("mixture", "source", "condition_map"))
        estimate = model(mixture, condition)
        length = min(estimate.shape[-1], source.shape[-1])
        improvement = si_sdr(estimate[..., :length], source[..., :length]) - si_sdr(
            mixture[..., :length], source[..., :length])
        scores.extend(improvement.cpu().tolist())
    return float(np.mean(scores))


@torch.no_grad()
def separation_test_rows(model: CausalResidualBSR, loader: DataLoader,
                         device: torch.device, condition: str) -> list[dict]:
    model.eval()
    rows = []
    with BASELINE_ROWS.open(newline="", encoding="utf-8") as stream:
        baseline = list(csv.DictReader(stream))
    for batch in loader:
        mixture, source, condition_map = (batch[key].to(device) for key in
                                          ("mixture", "source", "condition_map"))
        estimate = model(mixture, condition_map)
        length = min(estimate.shape[-1], source.shape[-1])
        mixed = si_sdr(mixture[..., :length], source[..., :length])
        score = si_sdr(estimate[..., :length], source[..., :length])
        for position, example_id in enumerate(batch["example_id"]):
            reference = baseline[len(rows)]
            row = {
                "example_id": example_id, "day": example_id.split(":", 1)[0],
                "snr_db": float(batch["snr_db"][position]),
                "mixture_si_sdr_db": float(mixed[position]),
                "bsr_si_sdri_db": float(reference["source_si_sdri_db"]),
                f"{condition}_si_sdri_db": float(score[position] - mixed[position]),
            }
            if condition == "shuffled":
                row["condition_day"] = batch["condition_day"][position]
                row["condition_manifest_index"] = int(batch["condition_manifest_index"][position])
                if row["condition_day"] == row["day"]:
                    raise ValueError("shuffled test donor is from the current day")
            rows.append(row)
    _check_test_identity(rows)
    return rows


def train_separator(cache: HighResolutionStore, condition: str,
                    device_name: str = "cuda", steps: int = 6000,
                    target_pool: str = "legacy") -> dict:
    if target_pool not in {"legacy", "empirical_50_50"}:
        raise ValueError("unknown residual training target pool")
    if target_pool == "empirical_50_50" and condition not in {"persistence", "forecast"}:
        raise ValueError("new target-pool training is scoped to causal residual arms")
    if target_pool == "empirical_50_50":
        from .empirical_target_pool import EmpiricalTargetPool
        empirical = EmpiricalTargetPool()
    else:
        empirical = None
    reference = json.loads(BASELINE_RESULT.read_text(encoding="utf-8"))
    config = ExperimentConfig(device=device_name)
    if empirical is not None:
        # Keep all 4,096 validation examples but avoid repeatedly scoring them
        # between updates; both empirical arms use this same checkpoint rule.
        config.separation_validate_every = 500
    if (steps != reference["updates_run"] or config.seed != reference["config"]["seed"]
            or config.batch_size != reference["config"]["batch_size"]):
        raise ValueError("residual arm must use matched BSR updates, seed and batch")
    _seed_everything(config.seed)
    device = torch.device(device_name)
    model = CausalResidualBSR(cache.store.bands, config.seed, device,
                              Path(config.transfer_checkpoint)).to(device)
    train = ConditionedInjectedDataset(cache, "train", config.train_examples,
                                       config.seed, condition, target_pool=empirical)
    validation = ConditionedInjectedDataset(cache, "validation", config.validation_examples,
                                            config.seed + 10_000_000, condition,
                                            cache_waveforms=True, target_pool=empirical)
    test = ConditionedInjectedDataset(cache, "test", config.validation_examples,
                                      config.seed + 20_000_000, condition, spread_rows=True)
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(config.seed), num_workers=0)
    validation_loader = DataLoader(validation, batch_size=24, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=24, shuffle=False, num_workers=0)
    ordinary = _make_model("bsr", cache.store.bands, config.seed, device)
    transferred = _load_transfer(ordinary, Path(config.transfer_checkpoint))
    ordinary.eval(); model.eval()
    probe = next(iter(DataLoader(validation, batch_size=2)))
    with torch.inference_mode():
        mixed = probe["mixture"].to(device)
        initial_difference = (model(mixed, probe["condition_map"].to(device)) -
                              ordinary(mixed, None)["estimate"]).abs().max().item()
    del ordinary
    if initial_difference > 1e-6 or transferred != model.transferred_tensors:
        raise ValueError(f"unmatched backbone initialization: {initial_difference}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.separation_steps)
    directory = (f"dclde_{condition}_residual_128" if empirical is None
                 else f"dclde_empirical_{condition}_residual_128")
    output = OUTPUT_ROOT / directory
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
        mixture, source, condition_map = (batch[key].to(device) for key in
                                          ("mixture", "source", "condition_map"))
        estimate = model(mixture, condition_map)
        loss = multi_resolution_loss(estimate, source)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step(); scheduler.step()
        if step == 1 or step % config.separation_validate_every == 0:
            row = {"step": step, "train_loss": float(loss.detach())}
            if step % config.separation_validate_every == 0:
                score = separation_validation(model, validation_loader, device)
                row["validation_si_sdri_db"] = score
                if score > best + 1e-4:
                    best, best_step = score, step
                    torch.save({"model": model.state_dict(), "config": vars(config),
                                "step": step, "interface": f"{condition}_residual_128",
                                "target_pool": target_pool}, output / "best.pt")
            training_rows.append(row)
            print(json.dumps({"stage": f"{condition}_separator", **row}), flush=True)
            (output / "training.json").write_text(json.dumps(training_rows, indent=2), encoding="utf-8")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    rows = separation_test_rows(model, test_loader, device, condition)
    sample_path = output / "clean_synthetic_test.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    values = np.asarray([row[f"{condition}_si_sdri_db"] for row in rows])
    baseline = np.asarray([row["bsr_si_sdri_db"] for row in rows])
    days = np.asarray([row["day"] for row in rows])
    delta = values - baseline
    day_means = np.asarray([delta[days == day].mean() for day in np.unique(days)])
    result = {"condition": condition, "training_target_pool": target_pool,
              "test_pool": "legacy procedural synthetic (unchanged held-out test mixtures)",
              "test_examples": len(rows),
              "mean_si_sdri_db": float(values.mean()), "delta_vs_bsr_db": float(delta.mean()),
              "delta_vs_bsr_day_ci95_db": [float(x) for x in paired_bootstrap_interval(day_means)],
              "days": {str(day): float(delta[days == day].mean()) for day in np.unique(days)},
              "best_step": best_step, "best_validation_si_sdri_db": best,
              "initial_waveform_max_difference": initial_difference,
              "checkpoint": str(output / "best.pt"), "samples_csv": str(sample_path)}
    if condition == "zero":
        result["condition_definition"] = "fixed per-band training mean log background over all 79 frames"
    if condition == "shuffled":
        result["condition_definition"] = "annotation-clean donor from a different day; train/test donors from same split, validation donors from train because validation has one day"
        result["test_donor_days"] = {
            day: sum(row["condition_day"] == day for row in rows)
            for day in sorted({row["condition_day"] for row in rows})
        }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summarize() -> dict:
    oracle = json.loads(ORACLE_RESULT.read_text(encoding="utf-8"))
    predictor = json.loads((PREDICTOR_DIR / "result.json").read_text(encoding="utf-8"))
    conditions = ("zero", "shuffled", "persistence", "forecast")
    causal = {name: json.loads((OUTPUT_ROOT / f"dclde_{name}_residual_128" / "result.json").read_text(encoding="utf-8"))
              for name in conditions}
    samples = {}
    for name in conditions:
        path = OUTPUT_ROOT / f"dclde_{name}_residual_128" / "clean_synthetic_test.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            samples[name] = list(csv.DictReader(stream))
    with (OUTPUT_ROOT / "dclde_oracle_residual_128" / "clean_synthetic_test.csv").open(newline="", encoding="utf-8") as stream:
        oracle_rows = list(csv.DictReader(stream))
    reference_ids = [row["example_id"] for row in oracle_rows]
    for name in conditions:
        if [row["example_id"] for row in samples[name]] != reference_ids:
            raise ValueError(f"{name} and Oracle evaluated on different examples")
        for reference, candidate in zip(oracle_rows, samples[name]):
            if candidate["day"] != reference["day"]:
                raise ValueError(f"{name} and Oracle recording days differ")
            for key in ("snr_db", "mixture_si_sdr_db", "bsr_si_sdri_db"):
                if abs(float(candidate[key]) - float(reference[key])) > 1e-5:
                    raise ValueError(f"{name} and Oracle differ in {key}")
    if any(row["condition_day"] == row["day"] for row in samples["shuffled"]):
        raise ValueError("shuffled test donor shares the target recording day")
    paired_rows = []
    for position, reference in enumerate(oracle_rows):
        row = {key: reference[key] for key in
               ("example_id", "day", "snr_db", "mixture_si_sdr_db", "bsr_si_sdri_db")}
        for name in conditions:
            row[f"{name}_si_sdri_db"] = samples[name][position][f"{name}_si_sdri_db"]
        row["oracle_si_sdri_db"] = reference["oracle_residual_si_sdri_db"]
        row["shuffled_donor_day"] = samples["shuffled"][position]["condition_day"]
        paired_rows.append(row)
    with SUMMARY_ROWS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader(); writer.writerows(paired_rows)
    days = np.asarray([row["day"] for row in paired_rows])
    scores = {name: np.asarray([float(row[f"{name}_si_sdri_db"]) for row in paired_rows])
              for name in conditions}
    scores["oracle"] = np.asarray([float(row["oracle_si_sdri_db"]) for row in paired_rows])
    scores["bsr"] = np.asarray([float(row["bsr_si_sdri_db"]) for row in paired_rows])

    def contrast(left: str, right: str) -> dict:
        difference = scores[left] - scores[right]
        by_day = {str(day): float(difference[days == day].mean()) for day in np.unique(days)}
        return {"mean_db": float(difference.mean()),
                "day_ci95_db": [float(x) for x in paired_bootstrap_interval(np.asarray(list(by_day.values())))],
                "days": by_day}

    comparisons = {
        f"{left}_minus_{right}": contrast(left, right)
        for left, right in (("zero", "bsr"), ("shuffled", "bsr"),
                            ("persistence", "zero"), ("persistence", "shuffled"),
                            ("forecast", "zero"), ("forecast", "shuffled"),
                            ("forecast", "persistence"), ("oracle", "forecast"))
    }
    result = {"protocol": "DCLDE clean 5-s real ocean plus synthetic target; same 4096 test mixtures, same BSR backbone initialization, training steps and validation rule",
              "history": "past 12 annotated event-masked 5-s windows from same WAV and split",
              "map": "512-point STFT, 128 equal-width groups, 79 time frames; causal predictions are log magnitudes",
              "innovation": "log|X|-predicted log|B|; identical residual adapter to Oracle",
              "bsr_si_sdri_db": oracle["baseline_si_sdri_db"],
              "zero_si_sdri_db": causal["zero"]["mean_si_sdri_db"],
              "shuffled_si_sdri_db": causal["shuffled"]["mean_si_sdri_db"],
              "persistence_si_sdri_db": causal["persistence"]["mean_si_sdri_db"],
              "forecast_si_sdri_db": causal["forecast"]["mean_si_sdri_db"],
              "oracle_si_sdri_db": oracle["oracle_residual_si_sdri_db"],
              "paired_test_samples_csv": str(SUMMARY_ROWS_PATH),
              "contrasts": comparisons,
              "forecast_minus_persistence_db": comparisons["forecast_minus_persistence"]["mean_db"],
              "forecast_minus_persistence_day_ci95_db": comparisons["forecast_minus_persistence"]["day_ci95_db"],
              "oracle_minus_forecast_db": comparisons["oracle_minus_forecast"]["mean_db"],
              "oracle_minus_forecast_day_ci95_db": comparisons["oracle_minus_forecast"]["day_ci95_db"],
              "oracle_gain_recovered_by_forecast_fraction": float(
                  causal["forecast"]["delta_vs_bsr_db"] / oracle["delta_si_sdri_db"]),
              "predictor": predictor, "causal_arms": causal}
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    cache = HighResolutionStore(store)
    result["predictor_test"] = cached_forecast_error(cache, "test")
    SUMMARY_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def cached_forecast_error(cache: HighResolutionStore, split: str) -> dict:
    """Full split diagnostic; predictions are the frozen causal maps."""
    indices = cache.store.indices(split, label=0)
    positions = np.full(len(cache.store.frame), -1, dtype=np.int32)
    positions[np.flatnonzero(cache.store.eligible)] = np.arange(int(cache.store.eligible.sum()))
    predicted_maps = np.load(FORECAST_MAPS_PATH, mmap_mode="r")
    forecast_sum = persistence_sum = 0.0
    count = 0
    for first in range(0, len(indices), 256):
        batch = indices[first:first + 256]
        target = np.asarray(cache.maps[batch], dtype=np.float32)
        forecast = np.asarray(predicted_maps[positions[batch]], dtype=np.float32)
        persistence = np.asarray(cache.maps[cache.history_indices[batch, -1]], dtype=np.float32)
        forecast_sum += np.abs(forecast - target).sum(dtype=np.float64)
        persistence_sum += np.abs(persistence - target).sum(dtype=np.float64)
        count += target.size
    return {"examples": len(indices), "forecast_mae_log": float(forecast_sum / count),
            "persistence_mae_log": float(persistence_sum / count)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("cache", "predictor", "forecasts", "zero", "shuffled", "persistence", "forecast", "summary"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-pool", choices=("legacy", "empirical_50_50"),
                        default="legacy")
    args = parser.parse_args()
    if args.stage == "cache":
        result = prepare_cache(args.device)
    else:
        store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
        cache = HighResolutionStore(store)
        if args.stage == "predictor":
            result = train_forecaster(cache, args.device)
        elif args.stage == "forecasts":
            result = prepare_forecasts(cache, args.device)
        elif args.stage in ("zero", "shuffled", "persistence", "forecast"):
            result = train_separator(cache, args.stage, args.device,
                                     target_pool=args.target_pool)
        else:
            result = summarize()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
