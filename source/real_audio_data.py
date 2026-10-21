"""Space-bounded preparation and loading of time-aligned SanctSound audio."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch
from torch.utils.data import Dataset

from .data import DEFAULT_DATA_ROOT, SpectrumNormaliser, eligible_origins, load_experiment_frames
from .synthesis import mix_with_real_ocean


REPO_ROOT = Path(__file__).resolve().parents[1]
SANCTSOUND_ROOT = DEFAULT_DATA_ROOT.parents[1]
REAL_AUDIO_ROOT = SANCTSOUND_ROOT / "audio_8khz"
PLAN_PATH = REPO_ROOT / "results" / "real_audio_download_plan.json"
GSUTIL = shutil.which("gsutil.cmd") or shutil.which("gsutil") or "gsutil"


@dataclass(frozen=True)
class DeploymentSource:
    deployment: str
    prefix: str
    recorder: str
    filename_style: str

    def uri(self, timestamp: pd.Timestamp) -> str:
        if self.filename_style == "sanctsound":
            name = (
                f"SanctSound_{self.deployment}_{self.recorder}_"
                f"{timestamp.strftime('%y%m%d%H0000')}.x.flac"
            )
        else:
            name = f"SurRidge-{self.recorder}-{timestamp.strftime('%y%m%d-%H0000')}.x.flac"
        return f"gs://noaa-passive-bioacoustic/sanctsound/audio/mb03/{self.prefix}/audio/{name}"


SOURCES = {
    "MB03_01": DeploymentSource("MB03_01", "01", "D143", "sanctsound"),
    "MB03_02": DeploymentSource("MB03_02", "02", "D113", "sanctsound"),
    "MB03_03": DeploymentSource("MB03_03", "03_disk01", "D143", "surridge"),
    "MB03_04": DeploymentSource("MB03_04", "04_disk01", "D113", "surridge"),
}


def _selected_timestamps(frame: pd.DataFrame, hours: int, context_hours: int) -> list[pd.Timestamp]:
    origins = eligible_origins(frame, context_hours)
    if len(origins) < hours:
        raise ValueError(f"only {len(origins)} eligible hours, requested {hours}")
    # A contiguous block makes the 'current WAV' portion itself continuous and
    # begins only after a complete causal history exists.
    for start in range(len(origins) - hours + 1):
        block = origins[start : start + hours]
        if np.all(np.diff(block) == 1):
            return [pd.Timestamp(frame.index[index]) for index in block]
    raise ValueError(f"no contiguous {hours}-hour block with complete history")


def build_plan(
    hours_per_deployment: int = 6,
    context_hours: int = 24,
    deployments: tuple[str, ...] | None = None,
) -> dict:
    frames = load_experiment_frames()
    deployments = deployments or tuple(SOURCES)
    requested = []
    for deployment in deployments:
        source = SOURCES[deployment]
        for timestamp in _selected_timestamps(frames[deployment], hours_per_deployment, context_hours):
            requested.append((deployment, timestamp, source.uri(timestamp)))
    command = [GSUTIL, "ls", "-l", *[item[2] for item in requested]]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    sizes = {}
    for line in completed.stdout.splitlines():
        match = re.match(r"\s*(\d+)\s+\S+\s+(gs://\S+\.flac)$", line)
        if match:
            sizes[match.group(2)] = int(match.group(1))
    missing = [uri for _, _, uri in requested if uri not in sizes]
    if missing:
        raise FileNotFoundError(f"objects missing from GCS listing: {missing}")
    files = []
    for deployment, timestamp, uri in requested:
        files.append(
            {
                "deployment": deployment,
                "timestamp": timestamp.isoformat(),
                "uri": uri,
                "remote_bytes": sizes[uri],
                "local": str((REAL_AUDIO_ROOT / deployment / f"{timestamp:%Y%m%dT%H0000}.flac").resolve()),
            }
        )
    total = sum(item["remote_bytes"] for item in files)
    plan = {
        "source": "NOAA SanctSound public GCS bucket",
        "selection": f"first contiguous {hours_per_deployment} eligible hours after a {context_hours}-hour causal history",
        "hours_per_deployment": hours_per_deployment,
        "context_hours": context_hours,
        "files": files,
        "remote_bytes": total,
        "remote_gib": total / 2**30,
        "storage_policy": "download one original FLAC, stream-resample to 8 kHz mono FLAC, then delete the original",
    }
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def _resample_file(source: Path, destination: Path, sample_rate: int = 8000) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(source) as input_file:
        stream = soxr.ResampleStream(input_file.samplerate, sample_rate, 1, dtype="float32", quality="HQ")
        with sf.SoundFile(destination, "w", samplerate=sample_rate, channels=1, format="FLAC", subtype="PCM_16") as output_file:
            while True:
                block = input_file.read(frames=input_file.samplerate * 30, dtype="float32", always_2d=True)
                if not len(block):
                    tail = stream.resample_chunk(np.empty((0, 1), np.float32), last=True)
                    if len(tail):
                        output_file.write(tail)
                    break
                mono = np.mean(block, axis=1, keepdims=True, dtype=np.float32)
                converted = stream.resample_chunk(mono, last=False)
                if len(converted):
                    output_file.write(converted)
        return {
            "source_sample_rate": input_file.samplerate,
            "source_channels": input_file.channels,
            "source_frames": input_file.frames,
        }


def _download_public_object(uri: str, destination: Path, expected_bytes: int) -> None:
    """Download one public GCS object sequentially with gsutil hash verification."""
    if destination.exists() and destination.stat().st_size == expected_bytes:
        return
    tracker_root = destination.parent / "trackers"
    tracker_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            GSUTIL,
            "-o", f"GSUtil:resumable_tracker_dir={tracker_root}",
            "-o", "GSUtil:sliced_object_download_max_components=1",
            "cp", uri, str(destination),
        ],
        check=True,
    )
    if destination.stat().st_size != expected_bytes:
        destination.unlink(missing_ok=True)
        raise IOError(f"size mismatch for {uri}")


def prepare_audio(
    plan: dict,
    max_download_gib: float = 12.0,
    min_free_gib: float = 100.0,
) -> dict:
    if plan["remote_bytes"] > max_download_gib * 2**30:
        raise RuntimeError(f"plan is {plan['remote_gib']:.2f} GiB, above {max_download_gib:.2f} GiB cap")
    usage = shutil.disk_usage(REPO_ROOT.drive + "\\")
    largest = max(item["remote_bytes"] for item in plan["files"])
    if usage.free - largest < min_free_gib * 2**30:
        raise RuntimeError("insufficient free space for one-file-at-a-time preparation")
    temp_root = REAL_AUDIO_ROOT / ".downloads"
    temp_root.mkdir(parents=True, exist_ok=True)
    records = []
    for number, item in enumerate(plan["files"], start=1):
        destination = Path(item["local"])
        if destination.exists():
            info = sf.info(destination)
            if info.samplerate == 8000 and info.channels == 1 and info.frames > 0:
                records.append(item | {"status": "reused", "local_bytes": destination.stat().st_size})
                print(f"[{number}/{len(plan['files'])}] reuse {destination.name}", flush=True)
                continue
        temp = temp_root / Path(item["uri"]).name
        print(f"[{number}/{len(plan['files'])}] download {item['uri']}", flush=True)
        try:
            _download_public_object(item["uri"], temp, item["remote_bytes"])
            if temp.stat().st_size != item["remote_bytes"]:
                raise IOError(f"size mismatch for {item['uri']}")
            audio_info = _resample_file(temp, destination)
        finally:
            temp.unlink(missing_ok=True)
        records.append(
            item
            | audio_info
            | {"status": "prepared", "sample_rate": 8000, "local_bytes": destination.stat().st_size}
        )
        free_gib = shutil.disk_usage(REPO_ROOT.drive + "\\").free / 2**30
        if free_gib < min_free_gib:
            raise RuntimeError(f"free space fell below {min_free_gib:.1f} GiB")
    manifest_path = REAL_AUDIO_ROOT / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8")).get("records", [])
        merged = {(record["deployment"], record["timestamp"]): record for record in existing}
        merged.update({(record["deployment"], record["timestamp"]): record for record in records})
        records = sorted(merged.values(), key=lambda record: (record["deployment"], record["timestamp"]))
    manifest = {
        "records": records,
        "local_bytes": sum(record["local_bytes"] for record in records),
        "sample_rate": 8000,
        "continuous_audio": True,
    }
    (REAL_AUDIO_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class RealAudioMixtureDataset(Dataset):
    """Real current ocean clips with causal TOL history and clean injected GT."""

    def __init__(
        self,
        frame: pd.DataFrame,
        deployment: str,
        normaliser: SpectrumNormaliser,
        records: list[dict],
        condition: str = "forecast",
        context_hours: int = 24,
        sample_rate: int = 8000,
        seconds: float = 0.5,
        clips_per_hour: int = 96,
        seed: int = 20261021,
        snr_range: tuple[float, float] = (-12.0, 12.0),
        source_profile: str = "broad",
    ):
        if condition not in {"none", "forecast", "oracle", "shuffled"}:
            raise ValueError("unknown condition")
        self.frame = frame
        self.deployment = deployment
        self.normaliser = normaliser
        self.condition = condition
        self.context_hours = context_hours
        self.sample_rate = sample_rate
        self.samples = round(sample_rate * seconds)
        self.clips_per_hour = clips_per_hour
        self.seed = seed
        self.snr_range = snr_range
        self.source_profile = source_profile
        self.values = frame.to_numpy(np.float32)
        self.normalised = normaliser.transform(self.values)
        timestamp_to_origin = {pd.Timestamp(value): index for index, value in enumerate(frame.index)}
        self.records = [record for record in records if record["deployment"] == deployment]
        self.origins = np.asarray(
            [timestamp_to_origin[pd.Timestamp(record["timestamp"])] for record in self.records], dtype=np.int64
        )
        eligible = set(eligible_origins(frame, context_hours).tolist())
        if not set(self.origins).issubset(eligible):
            raise ValueError("audio records do not all have complete causal TOL histories")
        self._audio: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records) * self.clips_per_hour

    def _hour_audio(self, hour_index: int) -> np.ndarray:
        if hour_index not in self._audio:
            audio, rate = sf.read(self.records[hour_index]["local"], dtype="float32", always_2d=True)
            if rate != self.sample_rate:
                raise ValueError(f"expected {self.sample_rate} Hz cache, found {rate}")
            self._audio[hour_index] = np.mean(audio, axis=1, dtype=np.float32)
        return self._audio[hour_index]

    def __getitem__(self, item: int) -> dict:
        hour_index, clip_index = divmod(item, self.clips_per_hour)
        origin = int(self.origins[hour_index])
        audio = self._hour_audio(hour_index)
        maximum = len(audio) - self.samples
        if maximum < 0:
            raise ValueError("audio file is shorter than one requested clip")
        rng = np.random.default_rng(self.seed + origin * 1009 + clip_index * 9176)
        offset = int(rng.integers(0, maximum + 1))
        ocean_clip = audio[offset : offset + self.samples]
        mixture, source, ocean, parameters = mix_with_real_ocean(
            ocean_clip, self.sample_rate, self.seed + origin * 1009 + clip_index,
            self.snr_range, self.source_profile,
        )
        if self.condition == "shuffled":
            shuffled_hour = (hour_index + max(1, len(self.records) // 2)) % len(self.records)
            condition_origin = int(self.origins[shuffled_hour])
            if condition_origin == origin:
                raise AssertionError("shuffled condition equals the current hour")
        else:
            condition_origin = origin
        return {
            "history": torch.from_numpy(
                self.normalised[origin - self.context_hours : origin].copy()
            ),
            "target_spectrum": torch.from_numpy(self.normalised[origin].copy()),
            "external_condition": torch.from_numpy(self.normalised[condition_origin].copy()),
            "mixture": torch.from_numpy(mixture),
            "source": torch.from_numpy(source),
            "ocean": torch.from_numpy(ocean),
            "origin": origin,
            "offset": offset,
            "example_id": f"{self.deployment}:{self.records[hour_index]['timestamp']}:{clip_index}:{offset}",
            "timestamp": self.records[hour_index]["timestamp"],
            "deployment": self.deployment,
            "snr_db": float(parameters["snr_db"]),
            "source_family": parameters["family"],
            "parameters": parameters,
        }


def load_manifest(path: Path = REAL_AUDIO_ROOT / "manifest.json") -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "prepare"))
    parser.add_argument("--hours-per-deployment", type=int, default=6)
    parser.add_argument("--context-hours", type=int, default=24)
    parser.add_argument("--deployments", nargs="+", choices=tuple(SOURCES))
    parser.add_argument("--max-download-gib", type=float, default=12.0)
    parser.add_argument("--min-free-gib", type=float, default=100.0)
    args = parser.parse_args()
    plan = build_plan(
        args.hours_per_deployment, args.context_hours,
        tuple(args.deployments) if args.deployments else None,
    )
    if args.command == "plan":
        print(json.dumps({key: value for key, value in plan.items() if key != "files"}, indent=2))
    else:
        manifest = prepare_audio(plan, args.max_download_gib, args.min_free_gib)
        print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
