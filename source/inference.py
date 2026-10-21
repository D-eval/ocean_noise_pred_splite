"""Checkpoint loading and chunked waveform inference for the local frontend."""
from __future__ import annotations

import csv
import io
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from .model import ForecastConditionedSeparator


class SeparationEngine:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        required = ("normaliser_mean", "normaliser_scale", "spectrum_columns")
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(
                f"checkpoint predates deployable metadata ({', '.join(missing)}); retrain with current source.train"
            )
        self.mean = np.asarray(self.config["normaliser_mean"], dtype=np.float32)
        self.scale = np.asarray(self.config["normaliser_scale"], dtype=np.float32)
        self.columns = list(self.config["spectrum_columns"])
        self.sample_rate = int(self.config["sample_rate"])
        self.context_hours = int(self.config["context_hours"])
        self.chunk_samples = round(float(self.config["seconds"]) * self.sample_rate)
        self.model = ForecastConditionedSeparator(
            spectrum_dim=len(self.columns),
            variant=self.config["variant"],
            latent_dim=int(self.config["latent_dim"]),
            initialization_seed=int(self.config["seed"]),
            model_dim=int(self.config["model_dim"]),
            depth=int(self.config["depth"]),
            heads=int(self.config["heads"]),
            dim_head=int(self.config["dim_head"]),
            n_fft=int(self.config["n_fft"]),
            hop_length=int(self.config["hop_length"]),
            bands=int(self.config["bands"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.requires_history = self.model.forecaster is not None

    def parse_history_csv(self, text: str) -> np.ndarray:
        raw_rows = csv.DictReader(io.StringIO(text.lstrip("\ufeff")), skipinitialspace=True)
        rows = [
            {key.strip(): value for key, value in row.items() if key is not None}
            for row in raw_rows
        ]
        if len(rows) < self.context_hours:
            raise ValueError(f"history needs at least {self.context_hours} hourly rows")
        values = []
        for row in rows[-self.context_hours :]:
            try:
                values.append([float(row[column.strip()]) for column in self.columns])
            except KeyError as error:
                raise ValueError(f"missing TOL column: {error.args[0]}") from error
            except ValueError as error:
                raise ValueError("history contains a non-numeric TOL value") from error
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("history contains missing or infinite values")
        return (array - self.mean) / self.scale

    def read_audio(self, payload: bytes) -> np.ndarray:
        audio, source_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if not len(audio):
            raise ValueError("audio file is empty")
        if source_rate != self.sample_rate:
            ratio = Fraction(self.sample_rate, source_rate).limit_denominator()
            audio = resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak
        return audio.astype(np.float32)

    @torch.inference_mode()
    def separate(self, audio: np.ndarray, history: np.ndarray | None = None) -> dict:
        if len(audio) / self.sample_rate > 120:
            raise ValueError("frontend limits each request to 120 seconds")
        chunk = self.chunk_samples
        stride = max(1, chunk * 3 // 4)
        padded = max(chunk, len(audio))
        starts = list(range(0, max(1, padded - chunk + 1), stride))
        if starts[-1] + chunk < len(audio):
            starts.append(len(audio) - chunk)
        result = np.zeros(len(audio), dtype=np.float32)
        weight = np.zeros(len(audio), dtype=np.float32)
        fade = max(1, chunk - stride)
        window = np.ones(chunk, dtype=np.float32)
        ramp = np.linspace(1e-3, 1.0, fade, endpoint=False, dtype=np.float32)
        window[:fade] = ramp
        window[-fade:] = ramp[::-1]
        if history is None:
            history = np.zeros((self.context_hours, len(self.columns)), dtype=np.float32)
        history_tensor = torch.from_numpy(history[None]).to(self.device)
        prediction = None
        began = time.perf_counter()
        for start in starts:
            valid = min(chunk, len(audio) - start)
            block = np.zeros(chunk, dtype=np.float32)
            block[:valid] = audio[start : start + valid]
            # Extra right padding compensates for upstream iSTFT length truncation.
            n_fft = int(self.config["n_fft"])
            model_input = np.pad(block, (0, n_fft))
            tensor = torch.from_numpy(model_input[None]).to(self.device)
            output = self.model(tensor, history_tensor)
            estimate = output["estimate"][0].detach().cpu().numpy()
            estimate = np.pad(estimate, (0, max(0, chunk - len(estimate))))[:valid]
            local_window = window[:valid]
            result[start : start + valid] += estimate * local_window
            weight[start : start + valid] += local_window
            if prediction is None and output["prediction"] is not None:
                prediction = output["prediction"][0].detach().cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        source = result / np.maximum(weight, 1e-6)
        ocean = audio - source
        forecast_db = None if prediction is None else (prediction * self.scale + self.mean).tolist()
        return {
            "source": source.astype(np.float32),
            "ocean": ocean.astype(np.float32),
            "forecast_db": forecast_db,
            "elapsed_seconds": elapsed,
            "duration_seconds": len(audio) / self.sample_rate,
            "real_time_factor": elapsed / max(len(audio) / self.sample_rate, 1e-9),
        }


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    stream = io.BytesIO()
    sf.write(stream, audio, sample_rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()
