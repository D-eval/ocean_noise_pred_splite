"""Joint causal noise forecaster and condition-injected BS-RoFormer."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "SpliteAnySound" / "third_party" / "BS-RoFormer"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from bs_roformer import BSRoformer  # noqa: E402


def balanced_bands(frequency_bins: int, count: int) -> tuple[int, ...]:
    count = min(max(2, count), frequency_bins)
    quotient, remainder = divmod(frequency_bins, count)
    return tuple(quotient + (index < remainder) for index in range(count))


class NoiseForecaster(nn.Module):
    """Encode historical spectra and predict the next spectrum."""

    def __init__(self, spectrum_dim: int, latent_dim: int = 32, layers: int = 1):
        super().__init__()
        self.encoder = nn.GRU(spectrum_dim, latent_dim, num_layers=layers, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2), nn.GELU(), nn.Linear(latent_dim * 2, spectrum_dim)
        )

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.encoder(history)
        latent = self.norm(hidden[-1])
        return latent, self.decoder(latent)


class ConditionedBandSplit(nn.Module):
    """Adds a frequency-specific condition bias after BS-RoFormer's band split."""

    def __init__(self, band_split: nn.Module, condition_dim: int, bands: int, model_dim: int):
        super().__init__()
        self.band_split = band_split
        self.projection = nn.Sequential(
            nn.LayerNorm(condition_dim), nn.Linear(condition_dim, bands * model_dim)
        )
        self.bands = bands
        self.model_dim = model_dim
        self._condition: torch.Tensor | None = None

    def set_condition(self, condition: torch.Tensor | None) -> None:
        self._condition = condition

    def forward(self, spectral_tokens: torch.Tensor) -> torch.Tensor:
        result = self.band_split(spectral_tokens)
        if self._condition is None:
            return result
        bias = self.projection(self._condition).view(-1, 1, self.bands, self.model_dim)
        return result + bias


class ConditionalBSRoformer(nn.Module):
    """BS-RoFormer with a differentiable condition path into every TF token."""

    def __init__(
        self,
        condition_dim: int,
        model_dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        dim_head: int = 16,
        n_fft: int = 256,
        hop_length: int = 64,
        bands: int = 24,
    ):
        super().__init__()
        band_sizes = balanced_bands(n_fft // 2 + 1, bands)
        self.separator = BSRoformer(
            dim=model_dim,
            depth=depth,
            time_transformer_depth=1,
            freq_transformer_depth=1,
            heads=heads,
            dim_head=dim_head,
            num_residual_streams=1,
            stft_n_fft=n_fft,
            stft_hop_length=hop_length,
            stft_win_length=n_fft,
            freqs_per_bands=band_sizes,
            multi_stft_resolutions_window_sizes=(n_fft,),
            multi_stft_hop_size=hop_length,
            multi_stft_resolution_loss_weight=0.0,
            flash_attn=False,
            use_pope=False,
        )
        injector = ConditionedBandSplit(
            self.separator.band_split, condition_dim, len(band_sizes), model_dim
        )
        self.separator.band_split = injector
        self.injector = injector

    def forward(self, mixture: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        self.injector.set_condition(condition)
        try:
            estimate = self.separator(mixture)
        finally:
            self.injector.set_condition(None)
        if estimate.ndim == 3:
            estimate = estimate[:, 0]
        return estimate[..., : mixture.shape[-1]]


class ForecastConditionedSeparator(nn.Module):
    """The three registered ablations: none, latent, and predicted spectrum."""

    VARIANTS = ("none", "latent", "spectrum")

    def __init__(
        self,
        spectrum_dim: int,
        variant: str,
        latent_dim: int = 32,
        initialization_seed: int = 0,
        **separator_kwargs,
    ):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        self.variant = variant
        self.spectrum_dim = spectrum_dim
        self.latent_dim = latent_dim
        condition_dim = latent_dim if variant == "latent" else spectrum_dim
        # Component-local RNG streams make the BS-RoFormer core identical
        # across ablations and the forecaster identical across the two
        # conditioned variants. Module construction order cannot bias results.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed + 1)
            self.separator = ConditionalBSRoformer(condition_dim=condition_dim, **separator_kwargs)
        if variant == "none":
            self.forecaster = None
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(initialization_seed + 2)
                self.forecaster = NoiseForecaster(spectrum_dim, latent_dim)

    def forward(self, mixture: torch.Tensor, history: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if self.forecaster is None:
            condition = torch.zeros(
                (mixture.shape[0], self.spectrum_dim), dtype=mixture.dtype, device=mixture.device
            )
            latent = prediction = None
        else:
            latent, prediction = self.forecaster(history)
            condition = latent if self.variant == "latent" else prediction
        estimate = self.separator(mixture, condition)
        return {"estimate": estimate, "prediction": prediction, "latent": latent}


def multi_resolution_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    length = min(estimate.shape[-1], target.shape[-1])
    estimate = estimate[..., :length]
    target = target[..., :length]
    waveform = torch.nn.functional.l1_loss(estimate, target)
    spectral = estimate.new_zeros(())
    for n_fft in (128, 256, 512):
        if target.shape[-1] < n_fft:
            continue
        window = torch.hann_window(n_fft, device=target.device)
        est = torch.stft(estimate, n_fft, n_fft // 4, window=window, return_complex=True)
        ref = torch.stft(target, n_fft, n_fft // 4, window=window, return_complex=True)
        spectral = spectral + torch.nn.functional.l1_loss(torch.view_as_real(est), torch.view_as_real(ref))
    return waveform + 0.5 * spectral
