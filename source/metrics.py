"""Metrics and paired uncertainty summaries for separation experiments."""
from __future__ import annotations

import numpy as np
import torch

from .synthesis import draw_injection, render_injection


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


def spectral_probability(audio: np.ndarray, sample_rate: int, bins: int = 64) -> np.ndarray:
    """Map a waveform to a smoothed log-frequency power probability distribution."""
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if not len(audio):
        raise ValueError("audio must not be empty")
    spectrum = np.fft.rfft(audio * np.hanning(len(audio)))
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    edges = np.geomspace(20.0, sample_rate / 2.0, bins + 1)
    indices = np.clip(np.digitize(frequencies, edges) - 1, 0, bins - 1)
    distribution = np.bincount(indices, weights=power, minlength=bins).astype(np.float64)
    distribution += 1e-12
    return distribution / distribution.sum()


def kl_divergence_bits(distribution: np.ndarray, reference: np.ndarray) -> float:
    """Directional KL(source || narrow-reference) in bits."""
    distribution = np.asarray(distribution, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(np.sum(distribution * np.log2(distribution / reference)))


def jensen_shannon_bits(distribution: np.ndarray, reference: np.ndarray) -> float:
    """Symmetric, finite spectral distribution shift in bits (range 0 to 1)."""
    midpoint = 0.5 * (distribution + reference)
    return 0.5 * (
        kl_divergence_bits(distribution, midpoint) + kl_divergence_bits(reference, midpoint)
    )


def spectral_overlap(distribution: np.ndarray, background: np.ndarray) -> float:
    """Histogram-intersection overlap in [0, 1] between source and background power."""
    return float(np.minimum(distribution, background).sum())


def normalised_spectral_entropy(distribution: np.ndarray) -> float:
    """Spectral entropy in [0, 1], from concentrated tones to diffuse spectra."""
    distribution = np.asarray(distribution, dtype=np.float64)
    return float(-np.sum(distribution * np.log2(distribution)) / np.log2(len(distribution)))


def narrow_source_reference(
    sample_rate: int,
    samples: int,
    examples: int = 512,
    bins: int = 64,
    seed: int = 9137,
) -> np.ndarray:
    """Estimate the original harmonic-generator spectral distribution."""
    distributions = []
    for index in range(examples):
        params = draw_injection(seed + index * 104729, profile="narrow")
        source = render_injection(params, sample_rate, samples)
        distributions.append(spectral_probability(source, sample_rate, bins))
    reference = np.mean(distributions, axis=0)
    return reference / reference.sum()
