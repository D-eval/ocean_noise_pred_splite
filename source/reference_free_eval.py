"""Interventional, reference-free evaluation for strongly annotated real events."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


RESTORATION_GAINS_DB = (-60.0, -30.0, -24.0, -18.0, -12.0, -6.0, 0.0)


def _logit(probability: float, epsilon: float = 1e-5) -> float:
    value = float(np.clip(float(probability), epsilon, 1.0 - epsilon))
    return float(np.log(value / (1.0 - value)))


@torch.no_grad()
def event_probabilities(
    classifier: Callable[[torch.Tensor], torch.Tensor],
    waveform: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    logits = classifier(waveform) / temperature
    return torch.sigmoid(logits)


@torch.no_grad()
def restoration_metrics(
    classifier: Callable[[torch.Tensor], torch.Tensor],
    mixture: torch.Tensor,
    event: torch.Tensor,
    event_mask: torch.Tensor,
    temperature: float = 1.0,
    gains_db: tuple[float, ...] = RESTORATION_GAINS_DB,
) -> tuple[dict, list[float]]:
    """Evaluate semantic removal plus partial-GT preservation constraints.

    Background is defined by exact mixture consistency, so the endpoint at
    0 dB is bit-identical to the input mixture.
    """
    if mixture.ndim == 1:
        mixture, event, event_mask = mixture[None], event[None], event_mask[None]
    length = min(mixture.shape[-1], event.shape[-1], event_mask.shape[-1])
    mixture = mixture[..., :length]
    event = event[..., :length]
    event_mask = event_mask[..., :length]
    background = mixture - event
    gains = torch.as_tensor(gains_db, dtype=mixture.dtype, device=mixture.device)
    scales = torch.pow(10.0, gains / 20.0)
    restored = background[:, None, :] + scales[None, :, None] * event[:, None, :]
    flat = restored.reshape(-1, length)
    probabilities = event_probabilities(classifier, flat, temperature).reshape(len(mixture), -1)
    mix_probability = event_probabilities(classifier, mixture, temperature)
    background_probability = event_probabilities(classifier, background, temperature)
    event_probability = event_probabilities(classifier, event, temperature)
    x = (gains - gains[0]) / (gains[-1] - gains[0])
    normalized = (probabilities - probabilities[:, :1]) / (
        probabilities[:, -1:] - probabilities[:, :1]
    ).abs().clamp_min(1e-5)
    auc = torch.trapezoid(normalized, x, dim=1)
    monotonic = (probabilities[:, 1:] >= probabilities[:, :-1] - 0.01).float().mean(dim=1)
    outside = 1.0 - event_mask
    leakage = (event.square() * outside).sum(dim=1) / event.square().sum(dim=1).clamp_min(1e-8)
    off_error = ((background - mixture).square() * outside).sum(dim=1).sqrt() / (
        (mixture.square() * outside).sum(dim=1).sqrt().clamp_min(1e-8)
    )
    reconstruction = (background + event - mixture).square().sum(dim=1).sqrt() / (
        mixture.square().sum(dim=1).sqrt().clamp_min(1e-8)
    )
    removal = torch.as_tensor([
        _logit(mix_value) - _logit(background_value)
        for mix_value, background_value in zip(mix_probability.cpu(), background_probability.cpu(), strict=True)
    ])
    metrics = {
        "mixture_event_probability": float(mix_probability.mean()),
        "background_event_probability": float(background_probability.mean()),
        "event_stem_probability": float(event_probability.mean()),
        "removal_logit": float(removal.mean()),
        "restoration_auc": float(auc.mean()),
        "monotonic_fraction": float(monotonic.mean()),
        "leakage_outside_event": float(leakage.mean()),
        "off_event_relative_error": float(off_error.mean()),
        "reconstruction_relative_error": float(reconstruction.mean()),
    }
    return metrics, probabilities.mean(dim=0).cpu().tolist()


def trivial_estimates(mixture: torch.Tensor) -> dict[str, torch.Tensor]:
    """Degenerate controls that any trustworthy metric suite must reject."""
    return {
        "identity": torch.zeros_like(mixture),
        "all_to_event": mixture.clone(),
    }
