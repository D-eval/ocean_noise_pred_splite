"""Deterministic waveform construction for TOL backgrounds and injected signals."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class InjectionParameters:
    seed: int
    fundamental_hz: float
    partials: int
    spectral_slope: float
    noise_low_hz: float
    noise_high_hz: float
    noise_mix: float
    attack_seconds: float
    release_seconds: float
    modulation_hz: float
    modulation_depth: float
    snr_db: float


def _normalise_rms(audio: np.ndarray, target: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    return (audio * (target / max(rms, 1e-8))).astype(np.float32)


def render_tol_noise(
    levels_db: np.ndarray,
    band_centres_hz: np.ndarray,
    sample_rate: int,
    samples: int,
    seed: int,
    rms: float = 0.06,
) -> np.ndarray:
    """Render a random-phase surrogate with the measured one-third-octave envelope.

    SanctSound's public sound-level product contains hourly spectra, not waveform
    phase. The reconstruction is therefore a controlled background surrogate and
    is never described as recovered raw audio.
    """
    rng = np.random.default_rng(seed)
    fft_freqs = np.fft.rfftfreq(samples, 1.0 / sample_rate)
    valid = (band_centres_hz > 0) & np.isfinite(levels_db)
    if valid.sum() < 2:
        raise ValueError("at least two finite TOL bands are required")
    centres = band_centres_hz[valid]
    db = levels_db[valid]
    # Interpolate in log-frequency; edge bands extend as constants.
    log_centres = np.log(np.maximum(centres, 1.0))
    log_freqs = np.log(np.maximum(fft_freqs, 1.0))
    envelope_db = np.interp(log_freqs, log_centres, db, left=db[0], right=db[-1])
    envelope = np.power(10.0, (envelope_db - np.max(envelope_db)) / 20.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, len(fft_freqs))
    magnitude_jitter = np.sqrt(rng.exponential(1.0, len(fft_freqs)))
    spectrum = envelope * magnitude_jitter * np.exp(1j * phase)
    spectrum[0] = 0.0
    if samples % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    audio = np.fft.irfft(spectrum, n=samples).astype(np.float32)
    return _normalise_rms(audio, rms)


def draw_injection(seed: int, snr_range: tuple[float, float] = (-8.0, 8.0)) -> InjectionParameters:
    """Draw an expressive MINUS-inspired tonal/noise source configuration."""
    rng = np.random.default_rng(seed)
    return InjectionParameters(
        seed=seed,
        fundamental_hz=float(rng.uniform(55.0, 420.0)),
        partials=int(rng.integers(2, 8)),
        spectral_slope=float(rng.uniform(0.65, 1.8)),
        noise_low_hz=float(rng.uniform(80.0, 1200.0)),
        noise_high_hz=float(rng.uniform(1800.0, 7000.0)),
        noise_mix=float(rng.uniform(0.0, 0.45)),
        attack_seconds=float(rng.uniform(0.005, 0.12)),
        release_seconds=float(rng.uniform(0.04, 0.35)),
        modulation_hz=float(rng.uniform(0.08, 2.5)),
        modulation_depth=float(rng.uniform(0.0, 0.5)),
        snr_db=float(rng.uniform(*snr_range)),
    )


def render_injection(params: InjectionParameters, sample_rate: int, samples: int) -> np.ndarray:
    """Render harmonic partials, band noise, AHDSR-like edges, and amplitude LFO."""
    rng = np.random.default_rng(params.seed)
    time = np.arange(samples, dtype=np.float32) / sample_rate
    signal = np.zeros(samples, dtype=np.float32)
    for harmonic in range(1, params.partials + 1):
        frequency = params.fundamental_hz * harmonic
        if frequency >= sample_rate / 2:
            break
        amplitude = harmonic ** (-params.spectral_slope)
        signal += amplitude * np.sin(2 * np.pi * frequency * time + rng.uniform(0, 2 * np.pi))

    white = rng.standard_normal(samples).astype(np.float32)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(samples, 1.0 / sample_rate)
    high = min(params.noise_high_hz, sample_rate / 2)
    spectrum[(frequencies < params.noise_low_hz) | (frequencies > high)] = 0
    noise = _normalise_rms(np.fft.irfft(spectrum, n=samples).astype(np.float32), 1.0)
    signal = (1.0 - params.noise_mix) * signal + params.noise_mix * noise

    attack = min(samples, round(params.attack_seconds * sample_rate))
    release = min(samples, round(params.release_seconds * sample_rate))
    envelope = np.ones(samples, dtype=np.float32)
    if attack:
        envelope[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False, dtype=np.float32)
    if release:
        envelope[-release:] = np.minimum(
            envelope[-release:], np.linspace(1.0, 0.0, release, dtype=np.float32)
        )
    lfo = 1.0 - params.modulation_depth + params.modulation_depth * (
        0.5 + 0.5 * np.sin(2 * np.pi * params.modulation_hz * time)
    )
    return _normalise_rms((signal * envelope * lfo).astype(np.float32), 1.0)


def make_mixture(
    levels_db: np.ndarray,
    band_centres_hz: np.ndarray,
    sample_rate: int,
    samples: int,
    seed: int,
    snr_range: tuple[float, float] = (-8.0, 8.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Create ocean + injected-source mixture with a reproducible SNR."""
    ocean = render_tol_noise(levels_db, band_centres_hz, sample_rate, samples, seed)
    params = draw_injection(seed + 1, snr_range)
    source = render_injection(params, sample_rate, samples)
    ocean_rms = float(np.sqrt(np.mean(ocean * ocean)))
    source_rms = ocean_rms * 10.0 ** (params.snr_db / 20.0)
    source = _normalise_rms(source, source_rms)
    mixture = ocean + source
    peak = float(np.max(np.abs(mixture)))
    if peak > 0.98:
        scale = 0.98 / peak
        mixture, ocean, source = mixture * scale, ocean * scale, source * scale
    return (
        mixture.astype(np.float32),
        source.astype(np.float32),
        ocean.astype(np.float32),
        asdict(params),
    )
