"""Persistence-only real adaptation with frozen separator representation.

Only the 128-band residual projection and BS-RoFormer mask estimators update.
The train-day frame detector supplies matched-clean weak targets; a frozen
50/50 separator teacher protects synthetic/pseudo-real behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dclde import (CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE,
                    EVENT_MASKED_FEATURE_CACHE, MANIFEST, SAMPLE_RATE)
from .dclde_causal_highres_experiment import (CausalResidualBSR,
    ConditionedInjectedDataset, HighResolutionStore)
from .dclde_separation_experiment import CausalFeatureStore, _seed_everything
from .empirical_target_pool import EmpiricalTargetPool
from .model import multi_resolution_loss
from .weak_event_detector import OUT as DETECTOR_OUT, frame_targets, load
from .weak_real_finetune import INIT, WeakRealDataset, validate as validate_previous


OUT = Path("experiments/dclde_adapter_weak_persistence")


class MatchedCleanRealDataset(WeakRealDataset):
    """Use the last annotation-clean, same-WAV window as a response reference."""

    def __getitem__(self, item: int) -> dict:
        value = super().__getitem__(item)
        row = self.cache.store.frame.iloc[int(self.histories[item, -1])]
        with sf.SoundFile(row.path) as stream:
            if stream.samplerate != SAMPLE_RATE:
                raise ValueError("unexpected sample rate for matched clean WAV")
            stream.seek(round(float(row.start_seconds) * SAMPLE_RATE))
            clean = stream.read(5 * SAMPLE_RATE, dtype="float32", always_2d=True)[:, 0]
        if len(clean) != 5 * SAMPLE_RATE:
            raise ValueError("short matched clean WAV")
        clean = clean - np.mean(clean, dtype=np.float64)
        value["clean_waveform"] = torch.from_numpy(clean.astype(np.float32))
        return value


def freeze_for_adapter_head(model: CausalResidualBSR) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.residual_split.project.weight.requires_grad_(True)
    for parameter in model.backbone.separator.separator.mask_estimators.parameters():
        parameter.requires_grad_(True)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names or not any("band_split.project" in name for name in names):
        raise ValueError("residual adapter is not trainable")
    if not any("mask_estimators" in name for name in names):
        raise ValueError("mask/output head is not trainable")
    if any("mask_estimators" not in name and "band_split.project" not in name
           for name in names):
        raise ValueError("backbone representation unexpectedly unfrozen")
    return {"parameter_names": names,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)}


def matched_clean_weak_loss(detector: nn.Module, mixture: torch.Tensor,
                            event: torch.Tensor, clean: torch.Tensor,
                            sample_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    length = min(mixture.shape[-1], event.shape[-1], clean.shape[-1])
    mixture, event, clean, sample_mask = [x[..., :length] for x in
                                         (mixture, event, clean, sample_mask)]
    background = mixture - event
    bg_logits = detector(background)
    event_logits = detector(event)
    with torch.no_grad():
        clean_logits = detector(clean)
    positive = frame_targets(sample_mask, bg_logits.shape[1])
    valid = positive.sum().clamp_min(1.0)
    # Match the same recording's natural detector floor, rather than
    # forcing every clean background to an artificial zero probability.
    residue = (F.smooth_l1_loss(bg_logits, clean_logits, reduction="none")
               * positive).sum() / valid
    event_term = (F.softplus(-event_logits) * positive).sum() / valid
    off = 1.0 - sample_mask
    off_term = ((event.square() * off).sum(dim=1) /
                (mixture.square() * off).sum(dim=1).clamp_min(1e-8)).sqrt().mean()
    combined = 0.25 * residue + 0.1 * event_term + 2.0 * off_term
    return combined, {"matched_logit_loss": float(residue.detach()),
                      "event_loss": float(event_term.detach()),
                      "off_loss": float(off_term.detach())}


def teacher_consistency(mixture: torch.Tensor, student: torch.Tensor,
                        teacher: torch.Tensor) -> torch.Tensor:
    length = min(mixture.shape[-1], student.shape[-1], teacher.shape[-1])
    mixture, student, teacher = [x[..., :length] for x in (mixture, student, teacher)]
    # The two stems are mixture-consistent; writing both terms keeps the
    # constraint explicit and exposes any future change to that convention.
    event_l1 = (student - teacher).abs().mean()
    background_l1 = ((mixture - student) - (mixture - teacher)).abs().mean()
    return (event_l1 + background_l1) / mixture.abs().mean().clamp_min(1e-6)


@torch.inference_mode()
def validate(model, detector, real_loader, synth_loader, device) -> dict:
    basic = validate_previous(model, detector, real_loader, synth_loader, device)
    model.eval()
    residues, gaps, clean_probs = [], [], []
    for batch in real_loader:
        mixture, mask, condition, clean = (batch[key].to(device) for key in
                                           ("waveform", "event_mask", "condition_map",
                                            "clean_waveform"))
        event = model(mixture, condition)
        length = min(event.shape[-1], mixture.shape[-1])
        background = mixture[..., :length] - event[..., :length]
        positive = frame_targets(mask[..., :length], detector(background).shape[1])
        bg_logit, clean_logit = detector(background), detector(clean[..., :length])
        difference = bg_logit - clean_logit
        residues.extend(((difference * positive).sum(dim=1) /
                         positive.sum(dim=1).clamp_min(1)).cpu().tolist())
        gaps.extend(((difference.abs() * positive).sum(dim=1) /
                     positive.sum(dim=1).clamp_min(1)).cpu().tolist())
        clean_probs.extend(((torch.sigmoid(clean_logit) * positive).sum(dim=1) /
                            positive.sum(dim=1).clamp_min(1)).cpu().tolist())
    basic["real_clean_matched_logit_residue"] = float(np.mean(residues))
    basic["real_clean_matched_logit_gap_abs"] = float(np.mean(gaps))
    basic["real_clean_frame_probability"] = float(np.mean(clean_probs))
    return basic


def train(steps: int = 3000, validate_every: int = 100,
          lambda_real: float = 0.01, lambda_distill: float = 0.1,
          device_name: str = "cuda", output: Path = OUT) -> dict:
    if not DETECTOR_OUT.joinpath("best.pt").exists():
        raise FileNotFoundError("train-day frame detector is required")
    _seed_everything(20261027)
    device = torch.device(device_name)
    event_store = CausalFeatureStore(MANIFEST, EVENT_MASKED_FEATURE_CACHE)
    clean_store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    cache = HighResolutionStore(clean_store)
    real_train = MatchedCleanRealDataset(event_store, cache, "train")
    real_val = MatchedCleanRealDataset(event_store, cache, "validation")
    synth_train = ConditionedInjectedDataset(cache, "train", 12000, 20261027,
                                             "persistence", target_pool=EmpiricalTargetPool())
    synth_val = ConditionedInjectedDataset(cache, "validation", 256, 20261027,
                                           "persistence", cache_waveforms=True)
    real_loader = DataLoader(real_train, batch_size=2, shuffle=True,
                             generator=torch.Generator().manual_seed(20261027))
    synth_loader = DataLoader(synth_train, batch_size=2, shuffle=True,
                              generator=torch.Generator().manual_seed(20261027))
    val_real = DataLoader(real_val, batch_size=8)
    val_synth = DataLoader(synth_val, batch_size=16)
    def new_model():
        instance = CausalResidualBSR(clean_store.bands, 20261022, device,
                                     Path("models/practical_real.pt")).to(device)
        checkpoint = torch.load(INIT, map_location=device, weights_only=False)
        if checkpoint.get("interface") != "persistence_residual_128":
            raise ValueError("teacher checkpoint interface mismatch")
        instance.load_state_dict(checkpoint["model"], strict=True)
        return instance
    student, teacher = new_model(), new_model().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    trainable = freeze_for_adapter_head(student)
    detector = load(DETECTOR_OUT / "best.pt", device)
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),
                                  lr=1e-5, weight_decay=1e-5)
    output.mkdir(parents=True, exist_ok=True)
    initial = validate(student, detector, val_real, val_synth, device)
    min_event = initial["real_event_frame_probability"] - 0.03
    max_off = initial["real_off_event_relative_error"] + 0.01
    min_sisdr = initial["synthetic_si_sdri_db"] - 0.2
    best_gap, best_step = initial["real_clean_matched_logit_gap_abs"], 0
    history = [{"step": 0, **initial, "eligible": True}]
    print(json.dumps(history[0]), flush=True)
    def save(step):
        torch.save({"model": student.state_dict(), "interface": "persistence_residual_128",
                    "target_pool": "empirical_50_50_plus_matched_clean_real_weak",
                    "training_target_pool": "50% synthetic/pseudo-real + 50% train-day real annotated",
                    "trainable": trainable, "step": step,
                    "lambda_real": lambda_real, "lambda_distill": lambda_distill}, output / "best.pt")
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
        # Keep frozen BatchNorm running statistics fixed as well as weights.
        # Evaluation mode still permits gradients through adapter and head.
        student.eval()
        mixture, target, condition = (synth[key].to(device) for key in
                                      ("mixture", "source", "condition_map"))
        estimate = student(mixture, condition)
        with torch.no_grad():
            teacher_estimate = teacher(mixture, condition)
        gt = multi_resolution_loss(estimate, target)
        distill = teacher_consistency(mixture, estimate, teacher_estimate)
        real_mix, mask, clean, real_condition = (real[key].to(device) for key in
                                                 ("waveform", "event_mask", "clean_waveform",
                                                  "condition_map"))
        real_event = student(real_mix, real_condition)
        weak, parts = matched_clean_weak_loss(detector, real_mix, real_event,
                                              clean, mask)
        loss = gt + lambda_real * weak + lambda_distill * distill
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_((p for p in student.parameters() if p.requires_grad), 2.0)
        optimizer.step()
        if step % validate_every == 0 or step == steps:
            score = validate(student, detector, val_real, val_synth, device)
            eligible = (score["real_event_frame_probability"] >= min_event and
                        score["real_off_event_relative_error"] <= max_off and
                        score["synthetic_si_sdri_db"] >= min_sisdr)
            row = {"step": step, "gt_loss": float(gt.detach()),
                   "weak_loss": float(weak.detach()),
                   "teacher_loss": float(distill.detach()), **parts,
                   **score, "eligible": eligible}
            history.append(row)
            print(json.dumps(row), flush=True)
            (output / "training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if eligible and score["real_clean_matched_logit_gap_abs"] < best_gap - 0.01:
                best_gap, best_step = score["real_clean_matched_logit_gap_abs"], step
                save(step)
    result = {"best_step": best_step, "initial_validation": initial,
              "best_validation_absolute_clean_gap": best_gap,
              "lambda_real": lambda_real, "lambda_distill": lambda_distill,
              "validation_thresholds": {"min_event": min_event, "max_off": max_off,
                                        "min_synthetic_si_sdri": min_sisdr},
              "trainable": trainable,
              "real_train_events": len(real_train), "real_validation_events": len(real_val),
              "evaluation_detector_used_for_training_or_selection": False,
              "test_calls_used_for_training_or_selection": False}
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=False)["model"]
    original = torch.load(INIT, map_location="cpu", weights_only=False)["model"]
    mutable = set(trainable["parameter_names"]) | {"residual_split.project.weight"}
    unchanged = [key for key in original if key not in mutable]
    if any(not torch.equal(selected[key], original[key]) for key in unchanged):
        raise ValueError("a frozen representation parameter or buffer changed")
    result["frozen_state_tensors_verified_unchanged"] = len(unchanged)
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--lambda-real", type=float, default=0.01)
    parser.add_argument("--lambda-distill", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(train(args.steps, args.validate_every, args.lambda_real,
                           args.lambda_distill, args.device, args.output), indent=2))
