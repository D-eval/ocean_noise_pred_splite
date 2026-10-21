"""Deterministic waveform construction for TOL backgrounds and injected signals."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class InjectionParameters:
    seed: int
    family: str
    fundamental_hz: float
    sweep_end_hz: float
    partials: int
    spectral_slope: float
    noise_low_hz: float
    noise_high_hz: float
    noise_mix: float
    attack_seconds: float
    release_seconds: float
    modulation_hz: float
    modulation_depth: float
    pulse_rate_hz: float
    pulse_width_seconds: float
    fm_index: float
    snr_db: float


BROAD_FAMILIES = (
    "harmonic",
    "chirp",
    "inharmonic",
    "impulse_train",
    "frequency_modulated",
    "band_noise",
)


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


def draw_injection(
    seed: int,
    snr_range: tuple[float, float] = (-8.0, 8.0),
    profile: str = "narrow",
    family: str | None = None,
) -> InjectionParameters:
    """Draw a deterministic source from either the original or broad distribution."""
    if profile not in {"narrow", "broad"}:
        raise ValueError("profile must be 'narrow' or 'broad'")
    rng = np.random.default_rng(seed)
    chosen_family = "harmonic" if profile == "narrow" else (family or str(rng.choice(BROAD_FAMILIES)))
    if chosen_family not in BROAD_FAMILIES:
        raise ValueError(f"unknown source family: {chosen_family}")
    broad = profile == "broad"
    fundamental = float(rng.uniform(25.0 if broad else 55.0, 1500.0 if broad else 420.0))
    sweep_end = float(rng.uniform(40.0, 3800.0)) if broad else fundamental
    noise_low = float(rng.uniform(20.0 if broad else 80.0, 2800.0 if broad else 1200.0))
    noise_high = float(rng.uniform(max(noise_low + 80.0, 400.0), 3900.0 if broad else 7000.0))
    return InjectionParameters(
        seed=seed,
        family=chosen_family,
        fundamental_hz=fundamental,
        sweep_end_hz=sweep_end,
        partials=int(rng.integers(1 if broad else 2, 13 if broad else 8)),
        spectral_slope=float(rng.uniform(0.25 if broad else 0.65, 2.6 if broad else 1.8)),
        noise_low_hz=noise_low,
        noise_high_hz=noise_high,
        noise_mix=float(rng.uniform(0.0, 0.9 if broad else 0.45)),
        attack_seconds=float(rng.uniform(0.001 if broad else 0.005, 0.2 if broad else 0.12)),
        release_seconds=float(rng.uniform(0.005 if broad else 0.04, 0.4 if broad else 0.35)),
        modulation_hz=float(rng.uniform(0.05 if broad else 0.08, 30.0 if broad else 2.5)),
        modulation_depth=float(rng.uniform(0.0, 1.0 if broad else 0.5)),
        pulse_rate_hz=float(rng.uniform(0.5, 35.0)),
        pulse_width_seconds=float(rng.uniform(0.001, 0.08)),
        fm_index=float(rng.uniform(0.1, 12.0)),
        snr_db=float(rng.uniform(*snr_range)),
    )


def render_injection(params: InjectionParameters, sample_rate: int, samples: int) -> np.ndarray:
    """Render tonal, swept, transient, modulated, or noise-like synthetic sources."""
    rng = np.random.default_rng(params.seed)
    time = np.arange(samples, dtype=np.float32) / sample_rate
    signal = np.zeros(samples, dtype=np.float32)

    if params.family == "harmonic":
        for harmonic in range(1, params.partials + 1):
            frequency = params.fundamental_hz * harmonic
            if frequency >= sample_rate / 2:
                break
            amplitude = harmonic ** (-params.spectral_slope)
            signal += amplitude * np.sin(2 * np.pi * frequency * time + rng.uniform(0, 2 * np.pi))
    elif params.family == "chirp":
        duration = max(samples / sample_rate, 1e-6)
        end = min(params.sweep_end_hz, sample_rate * 0.48)
        rate = (end - params.fundamental_hz) / duration
        phase = 2 * np.pi * (params.fundamental_hz * time + 0.5 * rate * np.square(time))
        signal = np.sin(phase + rng.uniform(0, 2 * np.pi)).astype(np.float32)
    elif params.family == "inharmonic":
        ratios = np.sort(rng.uniform(0.7, 8.5, params.partials))
        for index, ratio in enumerate(ratios, start=1):
            frequency = params.fundamental_hz * ratio
            if frequency < sample_rate / 2:
                signal += index ** (-params.spectral_slope) * np.sin(
                    2 * np.pi * frequency * time + rng.uniform(0, 2 * np.pi)
                )
    elif params.family == "impulse_train":
        event_count = max(1, round(params.pulse_rate_hz * samples / sample_rate))
        positions = rng.choice(samples, size=min(event_count, samples), replace=False)
        impulse = np.zeros(samples, dtype=np.float32)
        impulse[positions] = rng.uniform(0.5, 1.0, len(positions)) * rng.choice((-1.0, 1.0), len(positions))
        kernel_samples = max(2, round(params.pulse_width_seconds * sample_rate))
        kernel_time = np.arange(kernel_samples, dtype=np.float32) / sample_rate
        kernel = np.exp(-5.0 * kernel_time / max(params.pulse_width_seconds, 1e-4))
        kernel *= np.sin(2 * np.pi * min(params.fundamental_hz, sample_rate * 0.45) * kernel_time)
        signal = np.convolve(impulse, kernel, mode="full")[:samples].astype(np.float32)
    elif params.family == "frequency_modulated":
        carrier = min(params.fundamental_hz, sample_rate * 0.42)
        phase = 2 * np.pi * carrier * time + params.fm_index * np.sin(
            2 * np.pi * params.modulation_hz * time
        )
        signal = np.sin(phase + rng.uniform(0, 2 * np.pi)).astype(np.float32)

    white = rng.standard_normal(samples).astype(np.float32)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(samples, 1.0 / sample_rate)
    high = min(params.noise_high_hz, sample_rate / 2)
    spectrum[(frequencies < params.noise_low_hz) | (frequencies > high)] = 0
    noise = _normalise_rms(np.fft.irfft(spectrum, n=samples).astype(np.float32), 1.0)
    if params.family == "band_noise":
        signal = noise
    else:
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
    source_profile: str = "narrow",
    source_family: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Create ocean + injected-source mixture with a reproducible SNR."""
    ocean = render_tol_noise(levels_db, band_centres_hz, sample_rate, samples, seed)
    params = draw_injection(seed + 1, snr_range, source_profile, source_family)
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
