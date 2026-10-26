"""Train-day-only framewise detector for weak separator supervision.

This model is independent of the frozen clip-level evaluation classifier.
No test waveform, detector score, or test event is used for fitting or selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dclde import DCLDEClipDataset, MANIFEST, SAMPLE_RATE
from .event_classifier import _auc, _average_precision


OUT = Path("experiments/dclde_weak_train_detector")


class FrameEventDetector(nn.Module):
    """Temporal dilated CNN; unlike the evaluation model it has frame logits."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(256), persistent=False)
        self.net = nn.Sequential(
            nn.Conv1d(61, 96, 5, padding=2), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.Conv1d(96, 96, 5, dilation=2, padding=4), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.Conv1d(96, 96, 5, dilation=4, padding=8), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.Conv1d(96, 64, 5, dilation=8, padding=16), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv1d(64, 1, 1),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(waveform, 256, 128, window=self.window,
                              return_complex=True).abs().square()
        # 30–500 Hz gives exactly 61 bins at 2 kHz / 256.
        features = torch.log1p(1e4 * spectrum[:, 4:65])
        return self.net(features).squeeze(1)


def frame_targets(mask: torch.Tensor, frames: int) -> torch.Tensor:
    """A frame is positive when its center lies in an annotated event."""
    centers = torch.arange(frames, device=mask.device) * 128
    return mask[:, centers.clamp_max(mask.shape[1] - 1)]


@torch.no_grad()
def score(model: FrameEventDetector, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    scores, labels, clip_scores, clip_labels = [], [], [], []
    for batch in loader:
        logits = model(batch["waveform"].to(device))
        truth = frame_targets(batch["event_mask"].to(device), logits.shape[1])
        scores.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
        labels.append(truth.cpu().numpy().reshape(-1))
        clip_scores.append(torch.sigmoid(logits).amax(dim=1).cpu().numpy())
        clip_labels.append(batch["label"].numpy())
    p, y = np.concatenate(scores), np.concatenate(labels)
    cp, cy = np.concatenate(clip_scores), np.concatenate(clip_labels)
    return {"frame_ap": _average_precision(y, p), "frame_auc": _auc(y, p),
            "clip_ap": _average_precision(cy, cp), "clip_auc": _auc(cy, cp),
            "frame_positive_fraction": float(y.mean())}


def train(epochs: int = 30, batch_size: int = 64, device_name: str = "cuda") -> dict:
    torch.manual_seed(20261026)
    np.random.seed(20261026)
    device = torch.device(device_name)
    train_data = DCLDEClipDataset(MANIFEST, "train", augment=True, seed=20261026)
    val_data = DCLDEClipDataset(MANIFEST, "validation")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(20261026), num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=0)
    model = FrameEventDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    OUT.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch, stale = [], -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            wave = batch["waveform"].to(device)
            logits = model(wave)
            truth = frame_targets(batch["event_mask"].to(device), logits.shape[1])
            # Balanced frame weighting keeps short calls visible without
            # teaching the detector that every frame is an event.
            loss = F.binary_cross_entropy_with_logits(logits, truth,
                       pos_weight=torch.tensor(4.0, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = score(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["frame_ap"] > best + 1e-4:
            best, best_epoch, stale = metrics["frame_ap"], epoch, 0
            torch.save({"model": model.state_dict(), "architecture": "dilated_frame_cnn",
                        "train_days_only": True, "best_epoch": epoch}, OUT / "best.pt")
        else:
            stale += 1
        (OUT / "training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epoch >= 12 and stale >= 6:
            break
    result = {"best_epoch": best_epoch, "validation": history[best_epoch-1],
              "evaluation_classifier_used": False, "test_days_used": False}
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load(path: Path, device: torch.device) -> FrameEventDetector:
    model = FrameEventDetector().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    print(json.dumps(train(arguments.epochs, arguments.batch_size, arguments.device), indent=2))
