"""Metrics and paired uncertainty summaries for separation experiments."""
from __future__ import annotations

import numpy as np
import torch


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    projection = (
        (estimate * reference).sum(dim=-1, keepdim=True)
        * reference
        / (reference.square().sum(dim=-1, keepdim=True) + eps)
    )
    noise = estimate - projection
    return 10.0 * torch.log10(
        (projection.square().sum(dim=-1) + eps) / (noise.square().sum(dim=-1) + eps)
    )


def paired_bootstrap_interval(
    values: np.ndarray, seed: int = 20261021, draws: int = 4000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)
