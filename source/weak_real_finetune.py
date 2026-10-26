"""Train-day real-call weak supervision for the causal residual separator.

Checkpoint selection uses validation-day calls and validation synthetic stems.
The independent clip evaluator and all test-day calls are excluded here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .dclde import (CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE,
                    EVENT_MASKED_FEATURE_CACHE, MANIFEST)
from .dclde_causal_highres_experiment import (CausalResidualBSR,
    ConditionedInjectedDataset, HighResolutionStore)
from .dclde_separation_experiment import CausalFeatureStore, RealEventDataset, _seed_everything
from .empirical_target_pool import EmpiricalTargetPool
from .metrics import si_sdr
from .model import multi_resolution_loss
from .weak_event_detector import OUT as DETECTOR_OUT, frame_targets, load


OUT = Path("experiments/dclde_weak_real_persistence")
INIT = Path("models/empirical_persistence_residual_128.pt")


class WeakRealDataset(Dataset):
    """Annotated events paired with preceding annotation-clean 128-band maps."""

    def __init__(self, event_store: CausalFeatureStore, cache: HighResolutionStore, split: str):
        if split not in {"train", "validation"}:
            raise ValueError("weak fitting and selection exclude test calls")
        self.base = RealEventDataset(event_store, split)
        self.cache = cache
        clean = cache.store.frame
        lookup = {(str(row.path), int(round(float(row.start_seconds) / 5))): index
                  for index, row in enumerate(clean.itertuples())}
        histories = []
        self.imputed = 0
        for event_index in self.base.rows:
            row = event_store.frame.iloc[int(event_index)]
            block = int(round(float(row.start_seconds) / 5))
            prior = [lookup.get((str(row.path), block - offset), -1)
                     for offset in range(12, 0, -1)]
            observed = np.flatnonzero(np.asarray(prior) >= 0)
            if not len(observed):
                raise ValueError("real event lacks a clean historical window")
            for slot in np.flatnonzero(np.asarray(prior) < 0):
                nearest = observed[np.argmin(np.abs(observed - slot))]
                prior[slot] = prior[nearest]
                self.imputed += 1
            for index in prior:
                clean_row = clean.iloc[index]
                if (clean_row["split"] != split or clean_row["label"] != 0
                        or clean_row["day"] != row.day
                        or str(clean_row["path"]) != str(row.path)
                        or float(clean_row["start_seconds"]) >= float(row.start_seconds)):
                    raise ValueError("weak history violates split or causality")
            histories.append(prior)
        self.histories = np.asarray(histories, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> dict:
        value = self.base[item]
        value["condition_map"] = torch.from_numpy(
            np.asarray(self.cache.maps[self.histories[item, -1]], dtype=np.float32))
        return value


def weak_loss(detector: nn.Module, mixture: torch.Tensor, event: torch.Tensor,
              sample_mask: torch.Tensor, bg_weight: float = 0.25,
              event_weight: float = 0.10, preserve_weight: float = 2.0) -> tuple[torch.Tensor, dict]:
    """Mixture-consistent real loss; only annotated frames are positive."""
    length = min(mixture.shape[-1], event.shape[-1])
    mixture, event, sample_mask = [x[..., :length] for x in (mixture, event, sample_mask)]
    background = mixture - event
    bg_logits = detector(background)
    event_logits = detector(event)
    positive = frame_targets(sample_mask, bg_logits.shape[1])
    valid = positive.sum().clamp_min(1.0)
    bg = (F.softplus(bg_logits) * positive).sum() / valid
    ev = (F.softplus(-event_logits) * positive).sum() / valid
    off = 1.0 - sample_mask
    # Per-clip RMS normalization makes this loss comparable across recording days.
    relative_off = ((event.square() * off).sum(dim=1) /
                    ((mixture.square() * off).sum(dim=1).clamp_min(1e-8))).sqrt().mean()
    loss = bg_weight * bg + event_weight * ev + preserve_weight * relative_off
    return loss, {"bg": float(bg.detach()), "event": float(ev.detach()),
                  "off": float(relative_off.detach())}


@torch.inference_mode()
def validate(model, detector, real_loader, synthetic_loader, device) -> dict:
    model.eval()
    bg_probs, event_probs, mix_probs, off_values, gains = [], [], [], [], []
    for batch in real_loader:
        mixture, mask, condition = (batch[key].to(device) for key in
                                    ("waveform", "event_mask", "condition_map"))
        event = model(mixture, condition)
        length = min(event.shape[1], mixture.shape[1])
        mixture, mask, event = (x[..., :length] for x in (mixture, mask, event))
        background = mixture - event
        frame = frame_targets(mask, detector(background).shape[1])
        for wave, collection in ((background, bg_probs), (event, event_probs),
                                 (mixture, mix_probs)):
            probability = torch.sigmoid(detector(wave))
            collection.extend(((probability * frame).sum(dim=1) /
                               frame.sum(dim=1).clamp_min(1)).cpu().tolist())
        off = 1.0 - mask
        off_values.extend((((event.square() * off).sum(dim=1) /
                            (mixture.square() * off).sum(dim=1).clamp_min(1e-8))
                           .sqrt()).cpu().tolist())
    for batch in synthetic_loader:
        mixture, target, condition = (batch[key].to(device) for key in
                                      ("mixture", "source", "condition_map"))
        estimate = model(mixture, condition)
        length = min(estimate.shape[-1], mixture.shape[-1], target.shape[-1])
        mixture, target, estimate = (x[..., :length] for x in
                                     (mixture, target, estimate))
        gains.extend((si_sdr(estimate, target) - si_sdr(mixture, target)).cpu().tolist())
    return {"real_bg_frame_probability": float(np.mean(bg_probs)),
            "real_event_frame_probability": float(np.mean(event_probs)),
            "real_mixture_frame_probability": float(np.mean(mix_probs)),
            "real_off_event_relative_error": float(np.mean(off_values)),
            "synthetic_si_sdri_db": float(np.mean(gains))}


def train(steps: int = 3000, validate_every: int = 250, device_name: str = "cuda") -> dict:
    if not DETECTOR_OUT.joinpath("best.pt").exists():
        raise FileNotFoundError("train the independent frame detector first")
    device = torch.device(device_name)
    _seed_everything(20261026)
    event_store = CausalFeatureStore(MANIFEST, EVENT_MASKED_FEATURE_CACHE)
    clean_store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    cache = HighResolutionStore(clean_store)
    real_train = WeakRealDataset(event_store, cache, "train")
    real_val = WeakRealDataset(event_store, cache, "validation")
    empirical = EmpiricalTargetPool()
    synth_train = ConditionedInjectedDataset(cache, "train", 12000, 20261026,
                                             "persistence", target_pool=empirical)
    synth_val = ConditionedInjectedDataset(cache, "validation", 256, 20261027,
                                           "persistence", cache_waveforms=True)
    real_loader = DataLoader(real_train, batch_size=2, shuffle=True, num_workers=0,
                             generator=torch.Generator().manual_seed(20261026))
    synth_loader = DataLoader(synth_train, batch_size=2, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(20261026))
    val_real_loader = DataLoader(real_val, batch_size=8, shuffle=False, num_workers=0)
    val_synth_loader = DataLoader(synth_val, batch_size=16, shuffle=False, num_workers=0)
    model = CausalResidualBSR(clean_store.bands, 20261022, device,
                              Path("models/practical_real.pt")).to(device)
    starting = torch.load(INIT, map_location=device, weights_only=False)
    if starting["interface"] != "persistence_residual_128":
        raise ValueError("starting checkpoint interface mismatch")
    model.load_state_dict(starting["model"], strict=True)
    detector = load(DETECTOR_OUT / "best.pt", device)
    if any(parameter.requires_grad for parameter in detector.parameters()):
        raise ValueError("weak detector must remain frozen")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-5)
    OUT.mkdir(parents=True, exist_ok=True)
    history = []
    initial = validate(model, detector, val_real_loader, val_synth_loader, device)
    print(json.dumps({"step": 0, **initial}), flush=True)
    # Fixed, predeclared checkpoint rule: minimize real background detector
    # response subject to event retention, off-event preservation, and a
    # bounded controlled-domain SI-SDR drop on validation day.
    min_event = max(0.55, initial["real_event_frame_probability"] - 0.05)
    max_off = max(0.06, initial["real_off_event_relative_error"] + 0.02)
    min_sisdr = initial["synthetic_si_sdri_db"] - 1.0
    best_bg, best_step = initial["real_bg_frame_probability"], 0
    def save(step: int):
        torch.save({"model": model.state_dict(), "interface": "persistence_residual_128",
                    "target_pool": "empirical_50_50_plus_real_weak",
                    "training_target_pool": "50% train-day synthetic/pseudo-real + 50% train-day real annotated",
                    "training_detector": str(DETECTOR_OUT / "best.pt"), "step": step,
                    "selection": {"min_event": min_event, "max_off": max_off,
                                  "min_synthetic_si_sdri": min_sisdr}}, OUT / "best.pt")
    save(0)
    real_iter, synth_iter = iter(real_loader), iter(synth_loader)
    for step in range(1, steps + 1):
        try:
            real = next(real_iter)
        except StopIteration:
            real_iter = iter(real_loader); real = next(real_iter)
        try:
            synth = next(synth_iter)
        except StopIteration:
            synth_iter = iter(synth_loader); synth = next(synth_iter)
        model.train()
        mixture, target, condition = (synth[key].to(device) for key in
                                      ("mixture", "source", "condition_map"))
        estimate = model(mixture, condition)
        supervised = multi_resolution_loss(estimate, target)
        real_mix, real_mask, real_condition = (real[key].to(device) for key in
                                               ("waveform", "event_mask", "condition_map"))
        real_event = model(real_mix, real_condition)
        weak, parts = weak_loss(detector, real_mix, real_event, real_mask)
        # The detector BCE is numerically much larger than the spectral
        # supervised loss at the pretrained checkpoint. Keep the real term
        # gentle so it cannot erase controlled-domain separation in a few
        # hundred updates.
        loss = supervised + 0.1 * weak
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step % validate_every == 0 or step == steps:
            score = validate(model, detector, val_real_loader, val_synth_loader, device)
            eligible = (score["real_event_frame_probability"] >= min_event and
                        score["real_off_event_relative_error"] <= max_off and
                        score["synthetic_si_sdri_db"] >= min_sisdr)
            row = {"step": step, "weak_multiplier": 0.1,
                   "supervised_loss": float(supervised.detach()),
                   "weak_loss": float(weak.detach()), **parts, **score,
                   "eligible": eligible}
            history.append(row)
            print(json.dumps(row), flush=True)
            (OUT / "training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if eligible and score["real_bg_frame_probability"] < best_bg - 1e-3:
                best_bg, best_step = score["real_bg_frame_probability"], step
                save(step)
    result = {"best_step": best_step, "initial_validation": initial,
              "best_validation_background_probability": best_bg,
              "selection_thresholds": {"min_event": min_event, "max_off": max_off,
                                       "min_synthetic_si_sdri": min_sisdr},
              "real_train_events": len(real_train), "real_validation_events": len(real_val),
              "train_history_imputed_windows": real_train.imputed,
              "validation_history_imputed_windows": real_val.imputed,
              "test_days_used_for_training_or_selection": False,
              "evaluation_detector_used_for_training_or_selection": False}
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    print(json.dumps(train(arguments.steps, arguments.validate_every, arguments.device), indent=2))
