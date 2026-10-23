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
    event_count: int
    event_duration_seconds: float
    sweep_curvature: float
    lfo_enabled: bool
    snr_db: float


BROAD_FAMILIES = (
    "harmonic",
    "chirp",
    "inharmonic",
    "impulse_train",
    "frequency_modulated",
    "band_noise",
    "whale_upsweep",
    "whale_downsweep",
    "whale_arch",
    "whale_s_curve",
    "whale_pulse_packet",
    "vessel_tonal",
    "vessel_cavitation",
)

MARINE_FAMILIES = tuple(name for name in BROAD_FAMILIES if name.startswith(("whale_", "vessel_")))


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
    if profile not in {"narrow", "broad", "marine"}:
        raise ValueError("profile must be 'narrow', 'broad', or 'marine'")
    rng = np.random.default_rng(seed)
    if profile == "narrow":
        chosen_family = "harmonic"
    elif profile == "marine":
        chosen_family = family or str(rng.choice(MARINE_FAMILIES))
    else:
        chosen_family = family or str(rng.choice(BROAD_FAMILIES))
    if chosen_family not in BROAD_FAMILIES:
        raise ValueError(f"unknown source family: {chosen_family}")
    broad = profile != "narrow"
    whale = chosen_family.startswith("whale_")
    vessel = chosen_family.startswith("vessel_")
    if whale:
        fundamental = float(rng.uniform(45.0, 220.0))
        sweep_end = float(rng.uniform(160.0, 850.0))
    elif vessel:
        fundamental = float(rng.uniform(12.0, 120.0))
        sweep_end = fundamental
    else:
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
        noise_mix=float(
            rng.uniform(0.0, 0.25) if whale
            else rng.uniform(0.0, 0.35) if vessel
            else rng.uniform(0.0, 0.9 if broad else 0.45)
        ),
        attack_seconds=float(rng.uniform(0.001 if broad else 0.005, 0.2 if broad else 0.12)),
        release_seconds=float(rng.uniform(0.005 if broad else 0.04, 0.4 if broad else 0.35)),
        modulation_hz=float(rng.uniform(5.0, 20.0) if vessel else rng.uniform(0.05 if broad else 0.08, 30.0 if broad else 2.5)),
        modulation_depth=float(rng.uniform(0.2, 0.85) if vessel else rng.uniform(0.0, 1.0 if broad else 0.5)),
        pulse_rate_hz=float(rng.uniform(0.5, 35.0)),
        pulse_width_seconds=float(rng.uniform(0.001, 0.08)),
        fm_index=float(rng.uniform(0.1, 12.0)),
        event_count=int(rng.integers(1, 4 if whale else 2)),
        event_duration_seconds=float(rng.uniform(0.35, 2.5) if whale else rng.uniform(0.1, 1.0)),
        sweep_curvature=float(rng.uniform(0.45, 2.4)),
        lfo_enabled=bool(vessel and rng.random() < 0.8),
        snr_db=float(rng.uniform(*snr_range)),
    )


def _whale_phase(params: InjectionParameters, time: np.ndarray, duration: float) -> np.ndarray:
    """Integrate several smooth whale-like instantaneous-frequency contours."""
    u = np.clip(time / max(duration, 1e-6), 0.0, 1.0)
    low = min(params.fundamental_hz, params.sweep_end_hz)
    high = max(params.fundamental_hz, params.sweep_end_hz)
    if params.family == "whale_upsweep":
        frequency = low + (high - low) * np.power(u, params.sweep_curvature)
    elif params.family == "whale_downsweep":
        frequency = high - (high - low) * np.power(u, params.sweep_curvature)
    elif params.family == "whale_arch":
        frequency = low + (high - low) * np.sin(np.pi * u) ** params.sweep_curvature
    elif params.family == "whale_s_curve":
        sigmoid = 0.5 + 0.5 * np.tanh((u - 0.5) * (2.0 + 2.0 * params.sweep_curvature))
        frequency = low + (high - low) * sigmoid
    else:  # pulse packet: repeated short upsweeps inside the event envelope
        cycles = max(2.0, min(8.0, params.pulse_rate_hz * duration))
        local = np.mod(u * cycles, 1.0)
        frequency = low + (high - low) * np.power(local, params.sweep_curvature)
    return 2.0 * np.pi * np.cumsum(frequency, dtype=np.float64) / max(len(time) / duration, 1.0)


def _render_whale(params: InjectionParameters, sample_rate: int, samples: int, rng: np.random.Generator) -> np.ndarray:
    output = np.zeros(samples, dtype=np.float32)
    total_seconds = samples / sample_rate
    for _ in range(params.event_count):
        duration = min(params.event_duration_seconds * rng.uniform(0.75, 1.25), total_seconds * 0.8)
        length = max(8, min(samples, round(duration * sample_rate)))
        maximum_start = max(0, samples - length)
        start = int(rng.integers(0, maximum_start + 1)) if maximum_start else 0
        local_time = np.arange(length, dtype=np.float64) / sample_rate
        phase = _whale_phase(params, local_time, duration) + rng.uniform(0.0, 2.0 * np.pi)
        call = np.sin(phase)
        if rng.random() < 0.75:
            call += rng.uniform(0.08, 0.35) * np.sin(2.0 * phase + rng.uniform(0.0, 2.0 * np.pi))
        taper = np.sin(np.pi * np.linspace(0.0, 1.0, length)) ** rng.uniform(1.1, 2.8)
        flutter = 1.0 + rng.uniform(0.0, 0.22) * np.sin(2 * np.pi * rng.uniform(1.0, 5.0) * local_time)
        output[start : start + length] += (call * taper * flutter).astype(np.float32)
    return output


def _render_vessel(params: InjectionParameters, sample_rate: int, samples: int, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(samples, dtype=np.float32) / sample_rate
    tonal = np.zeros(samples, dtype=np.float32)
    for harmonic in range(1, params.partials + 1):
        frequency = params.fundamental_hz * harmonic
        if frequency >= sample_rate / 2:
            break
        tonal += harmonic ** (-params.spectral_slope) * np.sin(
            2 * np.pi * frequency * time + rng.uniform(0.0, 2.0 * np.pi)
        )
    # Apply the 5--20 Hz machinery LFO only to the tonal/narrow-band component.
    if params.lfo_enabled:
        lfo = 1.0 - params.modulation_depth + params.modulation_depth * (
            0.5 + 0.5 * np.sin(2 * np.pi * params.modulation_hz * time + rng.uniform(0.0, 2.0 * np.pi))
        )
        tonal *= lfo.astype(np.float32)
    if params.family == "vessel_tonal":
        return tonal
    white = rng.standard_normal(samples).astype(np.float32)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(samples, 1.0 / sample_rate)
    upper = min(sample_rate * 0.48, max(300.0, params.noise_high_hz))
    spectrum[(frequencies < max(15.0, params.noise_low_hz * 0.25)) | (frequencies > upper)] = 0
    cavitation = _normalise_rms(np.fft.irfft(spectrum, n=samples).astype(np.float32), 1.0)
    return (0.35 * tonal + 0.65 * cavitation).astype(np.float32)


def render_injection(params: InjectionParameters, sample_rate: int, samples: int) -> np.ndarray:
    """Render tonal, swept, transient, modulated, or noise-like synthetic sources."""
    rng = np.random.default_rng(params.seed)
    time = np.arange(samples, dtype=np.float32) / sample_rate
    signal = np.zeros(samples, dtype=np.float32)

    if params.family.startswith("whale_"):
        signal = _render_whale(params, sample_rate, samples, rng)
    elif params.family.startswith("vessel_"):
        signal = _render_vessel(params, sample_rate, samples, rng)
    elif params.family == "harmonic":
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
    # Vessel LFO is frequency-selective inside ``_render_vessel``; applying it
    # again here would unrealistically modulate the entire broadband source.
    if params.family.startswith("vessel_"):
        lfo = np.ones(samples, dtype=np.float32)
    else:
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


def mix_with_real_ocean(
    ocean: np.ndarray,
    sample_rate: int,
    seed: int,
    snr_range: tuple[float, float] = (-8.0, 8.0),
    source_profile: str = "broad",
    source_family: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Inject a clean synthetic source into an unmodified real ocean clip.

    The source gain is defined from the measured RMS of the current waveform.
    A common anti-clipping gain is applied to mixture and both references, so
    additivity and the requested SNR are preserved exactly.
    """
    ocean = np.asarray(ocean, dtype=np.float32).reshape(-1).copy()
    if not np.isfinite(ocean).all() or len(ocean) == 0:
        raise ValueError("ocean clip must be a non-empty finite waveform")
    ocean = ocean - np.mean(ocean, dtype=np.float64)
    ocean_rms = float(np.sqrt(np.mean(np.square(ocean), dtype=np.float64)))
    if ocean_rms < 1e-8:
        raise ValueError("ocean clip is silent")
    params = draw_injection(seed + 1, snr_range, source_profile, source_family)
    source = render_injection(params, sample_rate, len(ocean))
    source = _normalise_rms(source, ocean_rms * 10.0 ** (params.snr_db / 20.0))
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
