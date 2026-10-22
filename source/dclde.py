"""DCLDE 2013 time annotations, clip manifests, and waveform datasets.

The NOAA mirror contains one 2 kHz mono WAV every 15 minutes and strong
right-whale upcall annotations.  Splits are made by recording day so adjacent
clips can never leak across train, validation, and test.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


DCLDE_ROOT = Path("data/dclde2013/nefsc_sbnms_200903_nopp6_ch10")
AUDIO_ROOT = DCLDE_ROOT / "source-audio"
ANNOTATIONS = DCLDE_ROOT / "detections/NEFSC_SBNMS_200903_NOPP6_CH10_upcall-detection-log.csv"
MANIFEST = Path("data/dclde2013/clips_8s.csv")
SAMPLE_RATE = 2000
_EST = timezone(timedelta(hours=-5))
_STAMP = re.compile(r"_(\d{8})_(\d{6})_CH\d+\.wav$", re.IGNORECASE)


def recording_start(path: str | Path) -> pd.Timestamp:
    match = _STAMP.search(Path(path).name)
    if not match:
        raise ValueError(f"cannot parse DCLDE timestamp from {path}")
    value = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=_EST)
    return pd.Timestamp(value).tz_convert("UTC")


def load_annotations(path: Path = ANNOTATIONS) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={
        "Start_DateTime_ISO8601": "start",
        "End_DateTime_ISO8601": "end",
        "Detection_Confidence": "confidence",
    })
    frame["start"] = pd.to_datetime(frame["start"], utc=True)
    frame["end"] = pd.to_datetime(frame["end"], utc=True)
    frame["certain"] = frame["confidence"].eq("Detected")
    return frame.sort_values("start").reset_index(drop=True)


def _default_day_splits(days: list[str]) -> dict[str, str]:
    if len(days) < 4:
        raise ValueError("at least four recording days are required for day-disjoint splits")
    # Reserve at least three full dates plus the potentially partial final date
    # for validation/test.  This makes the split deliberately harder and avoids
    # treating adjacent clips from one continuous deployment as independent.
    train_end = max(2, len(days) - 4)
    return {
        day: ("train" if index < train_end else "validation" if index == train_end else "test")
        for index, day in enumerate(days)
    }


def build_clip_manifest(
    output: Path = MANIFEST,
    audio_root: Path = AUDIO_ROOT,
    annotations_path: Path = ANNOTATIONS,
    clip_seconds: float = 8.0,
    exclusion_buffer_seconds: float = 2.0,
    negative_ratio: float = 1.0,
    seed: int = 20261022,
) -> tuple[pd.DataFrame, dict]:
    """Build balanced fixed-grid clips; ambiguous calls are never negatives."""
    events = load_annotations(annotations_path)
    files = sorted(audio_root.rglob("*.wav"), key=recording_start)
    if not files:
        raise FileNotFoundError(f"no DCLDE WAV files below {audio_root}")
    starts = [recording_start(path) for path in files]
    days = sorted({stamp.tz_convert(_EST).strftime("%Y-%m-%d") for stamp in starts})
    split_by_day = _default_day_splits(days)
    rows: list[dict] = []
    for path, file_start in zip(files, starts, strict=True):
        info = sf.info(path)
        if info.samplerate != SAMPLE_RATE or info.channels != 1:
            raise ValueError(f"expected 2 kHz mono DCLDE audio: {path}")
        duration = info.frames / info.samplerate
        file_end = file_start + pd.Timedelta(seconds=duration)
        local_events = events[(events["end"] > file_start) & (events["start"] < file_end)]
        certain = local_events[local_events["certain"]]
        positive_bins: dict[int, list[list[float]]] = {}
        for event in certain.itertuples():
            centre = (event.start + (event.end - event.start) / 2 - file_start).total_seconds()
            bin_index = int(np.clip(np.floor(centre / clip_seconds), 0, duration // clip_seconds - 1))
            clip_start = bin_index * clip_seconds
            onset = max(0.0, (event.start - file_start).total_seconds() - clip_start)
            offset = min(clip_seconds, (event.end - file_start).total_seconds() - clip_start)
            positive_bins.setdefault(bin_index, []).append([onset, offset])
        excluded: set[int] = set()
        for event in local_events.itertuples():
            left = (event.start - file_start).total_seconds() - exclusion_buffer_seconds
            right = (event.end - file_start).total_seconds() + exclusion_buffer_seconds
            first = max(0, int(np.floor(left / clip_seconds)))
            last = min(int(duration // clip_seconds) - 1, int(np.floor(right / clip_seconds)))
            excluded.update(range(first, last + 1))
        day = file_start.tz_convert(_EST).strftime("%Y-%m-%d")
        relative = path.resolve().relative_to(Path.cwd().resolve())
        for bin_index in range(int(duration // clip_seconds)):
            if bin_index in positive_bins:
                label = 1
                intervals = positive_bins[bin_index]
            elif bin_index not in excluded:
                label = 0
                intervals = []
            else:
                continue
            rows.append({
                "path": str(relative),
                "start_seconds": bin_index * clip_seconds,
                "duration_seconds": clip_seconds,
                "label": label,
                "event_intervals": json.dumps(intervals, separators=(",", ":")),
                "day": day,
                "split": split_by_day[day],
            })
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    selected = []
    for split, group in frame.groupby("split", sort=False):
        positive = group[group["label"] == 1]
        negative = group[group["label"] == 0]
        count = min(len(negative), max(1, round(len(positive) * negative_ratio)))
        chosen = rng.choice(negative.index.to_numpy(), count, replace=False)
        selected.append(pd.concat([positive, negative.loc[chosen]]))
    frame = pd.concat(selected).sort_values(["day", "path", "start_seconds"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "source": "NOAA DCLDE 2013 public GCS mirror",
        "annotation": str(annotations_path),
        "clip_seconds": clip_seconds,
        "exclusion_buffer_seconds": exclusion_buffer_seconds,
        "day_splits": split_by_day,
        "rows": len(frame),
        "by_split_and_label": {
            f"{split}:{label}": int(count)
            for (split, label), count in frame.groupby(["split", "label"]).size().items()
        },
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return frame, summary


class DCLDEClipDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame | Path, split: str, augment: bool = False, seed: int = 0):
        frame = pd.read_csv(manifest) if isinstance(manifest, Path) else manifest.copy()
        self.frame = frame[frame["split"] == split].reset_index(drop=True)
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, item: int) -> dict:
        row = self.frame.iloc[item]
        start = round(float(row.start_seconds) * SAMPLE_RATE)
        samples = round(float(row.duration_seconds) * SAMPLE_RATE)
        with sf.SoundFile(row.path) as stream:
            stream.seek(start)
            waveform = stream.read(samples, dtype="float32", always_2d=True)[:, 0]
        if len(waveform) != samples:
            waveform = np.pad(waveform, (0, samples - len(waveform)))
        waveform = waveform - np.mean(waveform, dtype=np.float64)
        if self.augment:
            rng = np.random.default_rng(self.seed + item * 1009)
            waveform = waveform * np.float32(10.0 ** rng.uniform(-6.0, 6.0) / 20.0)
        mask = np.zeros(samples, dtype=np.float32)
        for onset, offset in json.loads(row.event_intervals):
            mask[max(0, round(onset * SAMPLE_RATE)) : min(samples, round(offset * SAMPLE_RATE))] = 1.0
        return {
            "waveform": torch.from_numpy(waveform.astype(np.float32)),
            "label": torch.tensor(float(row.label), dtype=torch.float32),
            "event_mask": torch.from_numpy(mask),
            "example_id": f"{row.day}:{Path(row.path).stem}:{float(row.start_seconds):.1f}",
            "path": row.path,
            "start_seconds": float(row.start_seconds),
        }


@dataclass(frozen=True)
class CurveMetrics:
    removal_logit: float
    restoration_auc: float
    monotonic_fraction: float
    event_stem_probability: float
    leakage_outside_event: float
    off_event_relative_error: float
    reconstruction_relative_error: float

