"""Leakage-safe loading of SanctSound TOL histories and deterministic mixtures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .synthesis import make_mixture


DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "soundscape"
    / "datasets"
    / "noaa-passive-bioacoustic"
    / "sanctsound"
    / "products"
    / "sound_level_metrics"
)


def load_tol_csv(data_root: Path, deployment: str) -> pd.DataFrame:
    """Load original NOAA CSV files so the pipeline does not require PyArrow."""
    site = deployment.split("_")[0].lower()
    pattern = f"{site}/sanctsound_{deployment.lower()}_tol_1h/data/*TOL_1h.csv"
    paths = sorted(data_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no TOL CSV files matched {data_root / pattern}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path, parse_dates=[0])
        frame = frame.rename(columns={frame.columns[0]: "timestamp"}).set_index("timestamp")
        # A few NOAA disk files contain leading/trailing spaces in every TOL
        # header. Strip before concatenation or pandas silently creates a
        # second, all-NaN set of bands and invalidates every causal window.
        frame.columns = [str(column).strip() for column in frame.columns]
        frames.append(frame)
    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    result.columns = [str(column).strip() for column in result.columns]
    return result.astype(np.float32)


def eligible_origins(frame: pd.DataFrame, context_hours: int) -> np.ndarray:
    """Return t where [t-context,t] is finite, hourly, and never crosses a gap."""
    values = frame.to_numpy(dtype=np.float32)
    finite = np.isfinite(values).all(axis=1)
    index = pd.DatetimeIndex(frame.index)
    origins = []
    for current in range(context_hours, len(frame)):
        start = current - context_hours
        if not finite[start : current + 1].all():
            continue
        # Compare timedeltas rather than integer epoch units: pandas may store
        # datetime64 at ns or us resolution depending on its version.
        deltas = index[start + 1 : current + 1] - index[start:current]
        if np.all(deltas == pd.Timedelta(hours=1)):
            origins.append(current)
    return np.asarray(origins, dtype=np.int64)


@dataclass(frozen=True)
class SpectrumNormaliser:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frames: list[pd.DataFrame]) -> "SpectrumNormaliser":
        values = np.concatenate([frame.to_numpy(np.float32) for frame in frames])
        mean = np.nanmean(values, axis=0).astype(np.float32)
        scale = np.nanstd(values, axis=0).astype(np.float32)
        scale[scale < 1e-4] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (values * self.scale + self.mean).astype(np.float32)


class TOLMixtureDataset(Dataset):
    """Past-only history paired with a present-hour background surrogate."""

    def __init__(
        self,
        frame: pd.DataFrame,
        deployment: str,
        normaliser: SpectrumNormaliser,
        context_hours: int = 24,
        sample_rate: int = 8000,
        seconds: float = 0.5,
        seed: int = 20261021,
        max_examples: int | None = None,
        snr_range: tuple[float, float] = (-8.0, 8.0),
        source_profile: str = "narrow",
        source_family: str | None = None,
    ):
        self.frame = frame
        self.deployment = deployment
        self.normaliser = normaliser
        self.context_hours = context_hours
        self.sample_rate = sample_rate
        self.samples = round(sample_rate * seconds)
        self.seed = seed
        self.snr_range = snr_range
        self.source_profile = source_profile
        self.source_family = source_family
        self.values = frame.to_numpy(np.float32)
        self.normalised = normaliser.transform(self.values)
        self.origins = eligible_origins(frame, context_hours)
        if max_examples and len(self.origins) > max_examples:
            selected = np.linspace(0, len(self.origins) - 1, max_examples, dtype=int)
            self.origins = self.origins[selected]
        self.band_centres = np.asarray(
            [float(column.removeprefix("TOL_")) for column in frame.columns], dtype=np.float32
        )

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str | int | dict]:
        origin = int(self.origins[item])
        # The condition ends at t-1. The target/current TOL at t is never input.
        history = self.normalised[origin - self.context_hours : origin]
        target_spectrum = self.normalised[origin]
        waveform_seed = self.seed + origin * 1009
        mixture, source, ocean, parameters = make_mixture(
            self.values[origin], self.band_centres, self.sample_rate, self.samples,
            waveform_seed, self.snr_range, self.source_profile, self.source_family
        )
        return {
            "history": torch.from_numpy(history),
            "target_spectrum": torch.from_numpy(target_spectrum),
            "mixture": torch.from_numpy(mixture),
            "source": torch.from_numpy(source),
            "ocean": torch.from_numpy(ocean),
            "origin": origin,
            "timestamp": str(self.frame.index[origin]),
            "deployment": self.deployment,
            "snr_db": float(parameters["snr_db"]),
            "source_family": parameters["family"],
            "parameters": parameters,
        }


def load_experiment_frames(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, pd.DataFrame]:
    deployments = ("MB03_01", "MB03_02", "MB03_03", "MB03_04")
    return {deployment: load_tol_csv(data_root, deployment) for deployment in deployments}
