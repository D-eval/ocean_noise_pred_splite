"""Summarise one continuous day of ocean audio for interactive inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
import soundfile as sf

from .real_audio_data import REAL_AUDIO_ROOT, load_manifest


def jensen_shannon_bits(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (first + second)
    return 0.5 * np.sum(first * np.log2(first / midpoint), axis=-1) + 0.5 * np.sum(
        second * np.log2(second / midpoint), axis=-1
    )


def analyse_day(
    deployment: str = "MB03_04",
    minute_seconds: int = 60,
    frequency_bins: int = 48,
) -> dict:
    records = [
        record for record in load_manifest()["records"]
        if record["deployment"] == deployment
    ]
    records = sorted(records, key=lambda record: record["timestamp"])
    if len(records) < 24:
        raise ValueError(f"{deployment} has only {len(records)} prepared hours; 24 required")
    records = records[:24]
    rates = {sf.info(record["local"]).samplerate for record in records}
    if rates != {8000}:
        raise ValueError(f"expected one 8 kHz cache, found {rates}")
    sample_rate = rates.pop()
    edges = np.geomspace(20.0, sample_rate / 2, frequency_bins + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    minute_spectra = []
    broadband = []
    timestamps = []
    frames_per_minute = minute_seconds * sample_rate
    for record in records:
        with sf.SoundFile(record["local"]) as stream:
            for minute in range(60):
                audio = stream.read(frames_per_minute, dtype="float32", always_2d=True).mean(axis=1)
                if len(audio) != frames_per_minute:
                    raise ValueError(f"short minute in {record['local']}")
                audio = audio - np.mean(audio, dtype=np.float64)
                frequencies, power = welch(
                    audio, fs=sample_rate, window="hann", nperseg=4096,
                    noverlap=2048, detrend=False, scaling="spectrum",
                )
                band_power = np.asarray([
                    np.sum(power[(frequencies >= low) & (frequencies < high)])
                    for low, high in zip(edges[:-1], edges[1:], strict=True)
                ])
                minute_spectra.append(np.maximum(band_power, 1e-20))
                broadband.append(20.0 * np.log10(max(float(np.sqrt(np.mean(audio * audio))), 1e-10)))
                base_time = np.datetime64(record["timestamp"].replace("+00:00", ""))
                timestamps.append((base_time + np.timedelta64(minute, "m")).astype(str))
    power = np.asarray(minute_spectra)
    db = 10.0 * np.log10(power)
    probability = np.maximum(power, 1e-20)
    probability /= probability.sum(axis=1, keepdims=True)
    lags = np.asarray([1, 2, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720])
    lag_rows = []
    for lag in lags:
        js = jensen_shannon_bits(probability[:-lag], probability[lag:])
        level_delta = np.abs(np.asarray(broadband[:-lag]) - np.asarray(broadband[lag:]))
        lag_rows.append({
            "minutes": int(lag),
            "js_median": round(float(np.median(js)), 6),
            "js_q25": round(float(np.quantile(js, 0.25)), 6),
            "js_q75": round(float(np.quantile(js, 0.75)), 6),
            "js_q90": round(float(np.quantile(js, 0.90)), 6),
            "level_delta_median_db": round(float(np.median(level_delta)), 3),
            "level_delta_q90_db": round(float(np.quantile(level_delta, 0.90)), 3),
        })
    adjacent_js = jensen_shannon_bits(probability[:-1], probability[1:])
    short_term_threshold = float(np.quantile(adjacent_js, 0.90))
    horizon = next(
        (row["minutes"] for row in lag_rows if row["js_median"] >= short_term_threshold),
        None,
    )
    median = np.median(db, axis=0)
    result = {
        "deployment": deployment,
        "date": timestamps[0][:10],
        "sample_rate": sample_rate,
        "aggregation_seconds": minute_seconds,
        "timestamps": timestamps,
        "frequency_hz": [round(float(value), 2) for value in centres],
        "spectrum_dbfs": np.round(db, 1).tolist(),
        "anomaly_db": np.round(db - median, 1).tolist(),
        "broadband_dbfs": np.round(broadband, 2).tolist(),
        "lag_statistics": lag_rows,
        "adjacent_js_p90_bits": round(short_term_threshold, 6),
        "change_horizon_minutes": horizon,
        "definition": "first lag whose median spectral-shape JS divergence reaches the 90th percentile of adjacent one-minute changes",
        "source_hours": [record["local"] for record in records],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", default="MB03_04")
    parser.add_argument("--output", type=Path, default=Path("results/ocean_noise_day.json"))
    args = parser.parse_args()
    result = analyse_day(args.deployment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "deployment", "date", "aggregation_seconds", "adjacent_js_p90_bits", "change_horizon_minutes"
    )}, indent=2))


if __name__ == "__main__":
    main()
