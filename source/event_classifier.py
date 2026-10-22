"""Independent right-whale event classifier used only as a frozen evaluator."""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dclde import DCLDEClipDataset, MANIFEST, SAMPLE_RATE, build_clip_manifest


class EventClassifier(nn.Module):
    """Compact spectrogram CNN for 2 kHz right-whale upcall detection."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_fft: int = 256, hop_length: int = 64):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Linear(192, 64), nn.SiLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def features(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(
            waveform, self.n_fft, self.hop_length, window=self.window,
            return_complex=True, center=True,
        ).abs().square()
        frequencies = torch.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate).to(waveform.device)
        spectrum = spectrum[:, (frequencies >= 30.0) & (frequencies <= 500.0)]
        values = torch.log1p(1e4 * spectrum)
        # Do not standardise each clip independently: absolute event energy is
        # intentionally part of the restoration intervention. BatchNorm learns
        # training-set statistics without making the classifier gain-invariant.
        return values.unsqueeze(1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        hidden = self.network(self.features(waveform))
        pooled = torch.cat((hidden.mean(dim=(-2, -1)), hidden.amax(dim=(-2, -1))), dim=1)
        return self.head(pooled).squeeze(1)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive = labels == 1
    p, n = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - p * (p + 1) / 2) / max(p * n, 1))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    truth = labels[order]
    cumulative = np.cumsum(truth)
    precision = cumulative / np.arange(1, len(truth) + 1)
    return float((precision * truth).sum() / max(truth.sum(), 1))


def classification_metrics(labels: np.ndarray, logits: np.ndarray, temperature: float = 1.0) -> dict:
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40.0, 40.0)))
    prediction = probability >= 0.5
    labels_bool = labels.astype(bool)
    tp = int(np.sum(prediction & labels_bool))
    fp = int(np.sum(prediction & ~labels_bool))
    fn = int(np.sum(~prediction & labels_bool))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        member = (probability >= low) & (probability < high if high < 1.0 else probability <= high)
        if member.any():
            ece += member.mean() * abs(probability[member].mean() - labels[member].mean())
    return {
        "examples": int(len(labels)),
        "positive_fraction": float(labels.mean()),
        "auroc": _auc(labels, probability),
        "average_precision": _average_precision(labels, probability),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "brier": float(np.mean(np.square(probability - labels))),
        "ece_10_bin": float(ece),
    }


@torch.no_grad()
def predict(model: EventClassifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, logits = [], []
    for batch in loader:
        labels.append(batch["label"].numpy())
        logits.append(model(batch["waveform"].to(device)).cpu().numpy())
    return np.concatenate(labels), np.concatenate(logits)


def fit_temperature(labels: np.ndarray, logits: np.ndarray) -> float:
    candidates = np.geomspace(0.1, 10.0, 201)
    losses = []
    for temperature in candidates:
        scaled = logits / temperature
        losses.append(float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled)))
    return float(candidates[int(np.argmin(losses))])


@dataclass
class ClassifierConfig:
    epochs: int = 30
    minimum_epochs: int = 15
    patience: int = 7
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 20261022
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_classifier(manifest: Path, output: Path, config: ClassifierConfig) -> dict:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device)
    train_set = DCLDEClipDataset(manifest, "train", augment=True, seed=config.seed)
    val_set = DCLDEClipDataset(manifest, "validation")
    test_set = DCLDEClipDataset(manifest, "test")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    model = EventClassifier().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.epochs)
    output.mkdir(parents=True, exist_ok=True)
    history, best_ap, best_epoch, stale = [], -math.inf, 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            waveform = batch["waveform"].to(device)
            target = batch["label"].to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(waveform), target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        val_labels, val_logits = predict(model, val_loader, device)
        metrics = classification_metrics(val_labels, val_logits)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["average_precision"] > best_ap + 1e-4:
            best_ap, best_epoch, stale = metrics["average_precision"], epoch, 0
            torch.save({"model": model.state_dict(), "config": asdict(config), "epoch": epoch}, output / "best.pt")
        else:
            stale += 1
        if epoch >= config.minimum_epochs and stale >= config.patience:
            break
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_labels, val_logits = predict(model, val_loader, device)
    temperature = fit_temperature(val_labels, val_logits)
    test_labels, test_logits = predict(model, test_loader, device)
    summary = {
        "config": asdict(config),
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"],
        "temperature": temperature,
        "validation": classification_metrics(val_labels, val_logits, temperature),
        "test": classification_metrics(test_labels, test_logits, temperature),
        "protocol": "day-disjoint; classifier frozen before separator evaluation",
    }
    checkpoint["temperature"] = temperature
    torch.save(checkpoint, output / "best.pt")
    (output / "training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_classifier(path: Path, device: torch.device) -> tuple[EventClassifier, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = EventClassifier().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, float(checkpoint.get("temperature", 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=Path("experiments/dclde_event_classifier"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.manifest.exists():
        _, summary = build_clip_manifest(args.manifest)
        print(json.dumps(summary, indent=2))
    if not args.prepare_only:
        print(json.dumps(train_classifier(
            args.manifest, args.output,
            ClassifierConfig(epochs=args.epochs, batch_size=args.batch_size, device=args.device),
        ), indent=2))


if __name__ == "__main__":
    main()
