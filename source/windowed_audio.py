"""Sub-hour real-audio spectra and causally aligned mixture datasets."""
from __future__ import annotations

import json
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .data import SpectrumNormaliser
from .real_audio_data import REAL_AUDIO_ROOT, load_manifest
from .synthesis import mix_with_real_ocean


BASE_WINDOW_SECONDS = 5
WINDOW_SECONDS = (5, 30, 60, 600)
CONTEXT_SECONDS = 3600
FEATURE_CACHE = Path("data/window_features_5s.npz")
BAND_CENTRES = np.asarray(
    (25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
     400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000),
    dtype=np.float32,
)


def _group_records(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["deployment"], []).append(record)
    for deployment, values in grouped.items():
        values.sort(key=lambda value: value["timestamp"])
        times = pd.to_datetime([value["timestamp"] for value in values], utc=True)
        if len(times) > 1 and not np.all(np.diff(times) == pd.Timedelta(hours=1)):
            raise ValueError(f"{deployment} audio files are not contiguous hourly records")
    return grouped


def _one_second_band_power(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return integrated one-third-octave power for non-overlapping 1 s frames."""
    complete = len(audio) // sample_rate
    if complete == 0:
        raise ValueError("audio is shorter than one second")
    frames = audio[: complete * sample_rate].reshape(complete, sample_rate)
    window = np.hanning(sample_rate).astype(np.float32)
    scale = float(np.square(window, dtype=np.float64).sum())
    frequencies = np.fft.rfftfreq(sample_rate, 1.0 / sample_rate)
    lower = BAND_CENTRES / 2.0 ** (1.0 / 6.0)
    upper = BAND_CENTRES * 2.0 ** (1.0 / 6.0)
    masks = [(frequencies >= lo) & (frequencies < min(hi, sample_rate / 2 + 1e-6))
             for lo, hi in zip(lower, upper, strict=True)]
    output = np.empty((complete, len(BAND_CENTRES)), dtype=np.float32)
    for start in range(0, complete, 120):
        block = frames[start : start + 120].astype(np.float32, copy=True)
        block -= block.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(block * window, axis=1)
        power = np.square(np.abs(spectrum)) / scale
        if power.shape[1] > 2:
            power[:, 1:-1] *= 2.0
        for band, mask in enumerate(masks):
            output[start : start + len(block), band] = power[:, mask].sum(axis=1)
    return np.maximum(output, np.finfo(np.float32).tiny)


def build_feature_cache(
    output: Path = FEATURE_CACHE,
    manifest_path: Path = REAL_AUDIO_ROOT / "manifest.json",
) -> dict[str, np.ndarray]:
    """Scan each FLAC once and cache 5 s linear-power third-octave features."""
    manifest = load_manifest(manifest_path)
    sample_rate = int(manifest.get("sample_rate", 8000))
    if sample_rate != 8000:
        raise ValueError("the sub-hour experiment expects the 8 kHz audio cache")
    grouped = _group_records(manifest["records"])
    arrays: dict[str, np.ndarray] = {}
    starts: dict[str, str] = {}
    for deployment, records in grouped.items():
        chunks = []
        for number, record in enumerate(records, start=1):
            audio, rate = sf.read(record["local"], dtype="float32", always_2d=True)
            if rate != sample_rate:
                raise ValueError(f"unexpected sample rate in {record['local']}")
            mono = np.mean(audio, axis=1, dtype=np.float32)
            seconds = _one_second_band_power(mono, sample_rate)
            usable = len(seconds) // BASE_WINDOW_SECONDS * BASE_WINDOW_SECONDS
            five_second = seconds[:usable].reshape(
                -1, BASE_WINDOW_SECONDS, len(BAND_CENTRES)
            ).mean(axis=1)
            chunks.append(five_second.astype(np.float32))
            print(f"features {deployment} {number}/{len(records)}", flush=True)
        arrays[deployment] = np.concatenate(chunks)
        starts[deployment] = records[0]["timestamp"]
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sample_rate": sample_rate,
        "base_window_seconds": BASE_WINDOW_SECONDS,
        "band_centres_hz": BAND_CENTRES.tolist(),
        "representation": "integrated Hann-periodogram power, mean over 1 s frames",
        "deployment_starts": starts,
    }
    np.savez_compressed(output, metadata=json.dumps(metadata), **arrays)
    return arrays


def load_base_features(path: Path = FEATURE_CACHE) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
        arrays = {key: cache[key].astype(np.float32) for key in cache.files if key != "metadata"}
    return metadata, arrays


def aggregate_features(base_power: np.ndarray, window_seconds: int) -> np.ndarray:
    if window_seconds not in WINDOW_SECONDS:
        raise ValueError(f"window_seconds must be one of {WINDOW_SECONDS}")
    factor = window_seconds // BASE_WINDOW_SECONDS
    usable = len(base_power) // factor * factor
    power = base_power[:usable].reshape(-1, factor, base_power.shape[1]).mean(axis=1)
    return (10.0 * np.log10(np.maximum(power, np.finfo(np.float32).tiny))).astype(np.float32)


def fit_window_normaliser(features: dict[str, np.ndarray]) -> SpectrumNormaliser:
    train = np.concatenate([features["MB03_01"], features["MB03_02"]])
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-4] = 1.0
    return SpectrumNormaliser(mean, scale)


class WindowForecastDataset(Dataset):
    """Past-only spectra ending at t-1, paired with spectrum t."""

    def __init__(
        self,
        spectra_db: np.ndarray,
        normaliser: SpectrumNormaliser,
        window_seconds: int,
        max_examples: int | None = None,
    ):
        self.window_seconds = window_seconds
        self.context_steps = CONTEXT_SECONDS // window_seconds
        self.normalised = normaliser.transform(spectra_db)
        self.origins = np.arange(self.context_steps, len(spectra_db), dtype=np.int64)
        if max_examples and len(self.origins) > max_examples:
            indices = np.linspace(0, len(self.origins) - 1, max_examples, dtype=int)
            self.origins = self.origins[indices]

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        origin = int(self.origins[item])
        return {
            "history": torch.from_numpy(
                self.normalised[origin - self.context_steps : origin].copy()
            ),
            "target_spectrum": torch.from_numpy(self.normalised[origin].copy()),
        }


class WindowedRealAudioDataset(Dataset):
    """Canonical real clips whose condition uses only windows preceding the current one."""

    def __init__(
        self,
        spectra_db: np.ndarray,
        normaliser: SpectrumNormaliser,
        records: list[dict],
        deployment: str,
        window_seconds: int,
        examples: int,
        sample_rate: int = 8000,
        seconds: float = 0.5,
        seed: int = 20261022,
        snr_range: tuple[float, float] = (-12.0, 12.0),
        max_cached_hours: int = 6,
    ):
        self.deployment = deployment
        self.window_seconds = window_seconds
        self.context_steps = CONTEXT_SECONDS // window_seconds
        self.sample_rate = sample_rate
        self.samples = round(sample_rate * seconds)
        self.seed = seed
        self.snr_range = snr_range
        self.normalised = normaliser.transform(spectra_db)
        self.records = sorted(
            (value for value in records if value["deployment"] == deployment),
            key=lambda value: value["timestamp"],
        )
        if not self.records:
            raise ValueError(f"no audio records for {deployment}")
        self.file_samples = 3600 * sample_rate
        self.total_samples = len(self.records) * self.file_samples
        self.window_samples = window_seconds * sample_rate
        self.examples = examples
        self.max_cached_hours = max_cached_hours
        self._audio: OrderedDict[int, np.ndarray] = OrderedDict()
        first = pd.Timestamp(self.records[0]["timestamp"])
        self.start_timestamp = first.to_pydatetime()
        minimum = CONTEXT_SECONDS * sample_rate
        maximum = self.total_samples - self.samples
        if maximum <= minimum:
            raise ValueError("deployment is too short for one hour of history")
        # Stratified deterministic positions keep train/validation/test coverage broad.
        width = maximum - minimum + 1
        rng = np.random.default_rng(seed + sum(ord(char) for char in deployment))
        cell = width / examples
        positions = minimum + ((np.arange(examples) + rng.random(examples)) * cell).astype(np.int64)
        target = positions // self.window_samples
        upper = (target + 1) * self.window_samples - self.samples
        self.positions = np.minimum(positions, upper)
        if np.any(target < self.context_steps):
            raise AssertionError("a target window lacks its full causal history")

    def __len__(self) -> int:
        return self.examples

    def _hour_audio(self, hour: int) -> np.ndarray:
        if hour in self._audio:
            self._audio.move_to_end(hour)
            return self._audio[hour]
        audio, rate = sf.read(self.records[hour]["local"], dtype="float32", always_2d=True)
        if rate != self.sample_rate:
            raise ValueError("unexpected cached audio sample rate")
        mono = np.mean(audio, axis=1, dtype=np.float32)
        self._audio[hour] = mono
        while len(self._audio) > self.max_cached_hours:
            self._audio.popitem(last=False)
        return mono

    def __getitem__(self, item: int) -> dict:
        absolute = int(self.positions[item])
        target = absolute // self.window_samples
        hour, offset = divmod(absolute, self.file_samples)
        ocean_clip = self._hour_audio(hour)[offset : offset + self.samples]
        if len(ocean_clip) != self.samples:
            raise AssertionError("clip crosses an hourly-file boundary")
        mixture, source, ocean, parameters = mix_with_real_ocean(
            ocean_clip, self.sample_rate, self.seed + item * 1009,
            self.snr_range, "broad",
        )
        timestamp = self.start_timestamp + timedelta(seconds=absolute / self.sample_rate)
        return {
            "history": torch.from_numpy(
                self.normalised[target - self.context_steps : target].copy()
            ),
            "target_spectrum": torch.from_numpy(self.normalised[target].copy()),
            "external_condition": torch.from_numpy(self.normalised[target].copy()),
            "mixture": torch.from_numpy(mixture),
            "source": torch.from_numpy(source),
            "ocean": torch.from_numpy(ocean),
            "origin": target,
            "offset": absolute,
            "example_id": f"{self.deployment}:{item:06d}",
            "timestamp": timestamp.isoformat(),
            "deployment": self.deployment,
            "snr_db": float(parameters["snr_db"]),
            "source_family": parameters["family"],
            "parameters": parameters,
        }


def prepare_window_features(
    cache_path: Path = FEATURE_CACHE,
) -> tuple[dict, dict[int, dict[str, np.ndarray]]]:
    if not cache_path.exists():
        build_feature_cache(cache_path)
    metadata, base = load_base_features(cache_path)
    by_window = {
        window: {name: aggregate_features(values, window) for name, values in base.items()}
        for window in WINDOW_SECONDS
    }
    return metadata, by_window

