"""Fetch a fixed, small GSL test cohort from the public FRDR archive.

Selection depends only on SHA-256 of the 50 continuous filenames, before
model evaluation. HTTP byte ranges avoid storing the 4.6 GB source ZIP.
The 32,768 Hz WAVs are downsampled to the model's 2 kHz rate and kept under data/.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly


URL = "https://www.frdr-dfdr.ca/repo/files/1/published/publication_239/submitted_data/narw-data.zip"
ARCHIVE_SIZE = 4634189342
ANNOTATION_MEMBER = "data/continuous/dataset_B/annotations_B_cont.csv"
AUDIO_PREFIX = "data/continuous/dataset_B/audio/"
ROOT = Path("data/gsl_new_site_confirm")
COHORT_SIZE = 16
COHORT_OFFSET = 12  # The first 12 hashes were only used in a preprocessing audit.


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0
        self.cache_start = -1
        self.cache = b""
        self.session = requests.Session()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.size + offset
        else:
            raise ValueError("invalid whence")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        if size == 0:
            return b""
        start = self.position
        end = min(self.size, start + size)
        if not (self.cache_start <= start and end <= self.cache_start + len(self.cache)):
            response = self.session.get(
                self.url, headers={"Range": f"bytes={start}-{min(self.size, max(end, start + 2**20))-1}"},
                timeout=180,
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise ValueError(f"server ignored range request: {response.status_code}")
            self.cache_start, self.cache = start, response.content
        self.position = end
        return self.cache[start-self.cache_start:end-self.cache_start]

    def close(self) -> None:
        self.session.close()
        super().close()


def selected_names() -> list[str]:
    names = [f"B_cont_{index}.wav" for index in range(50)]
    ranked = sorted(names, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    return ranked[COHORT_OFFSET:COHORT_OFFSET + COHORT_SIZE]


def prepare() -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    selected = selected_names()
    with RangeFile(URL, ARCHIVE_SIZE) as stream, zipfile.ZipFile(stream) as archive:
        member_names = set(archive.namelist())
        if not all(AUDIO_PREFIX + name in member_names for name in selected):
            raise ValueError("fixed GSL cohort is missing from archive")
        annotation_bytes = archive.read(ANNOTATION_MEMBER)
        annotations = list(csv.DictReader(io.StringIO(annotation_bytes.decode("utf-8-sig")), delimiter=";"))
        selected_annotations = [row for row in annotations if row["filename"] in selected]
        annotation_path = ROOT / "annotations.csv"
        with annotation_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=("filename", "timestamp", "snr"))
            writer.writeheader()
            writer.writerows(selected_annotations)
        audio_info = {}
        for name in selected:
            output = ROOT / name
            if output.exists():
                info = sf.info(output)
                if info.samplerate == 2000 and 1790 <= info.duration <= 1810:
                    audio_info[name] = {"source_sample_rate": 32768,
                                        "sample_rate": info.samplerate, "duration": info.duration,
                                        "bytes": output.stat().st_size}
                    print(json.dumps({"reused": name, **audio_info[name]}), flush=True)
                    continue
            raw = archive.read(AUDIO_PREFIX + name)
            audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            del raw
            if audio.ndim != 1 or rate not in (32000, 32768):
                raise ValueError(f"unexpected source WAV format for {name}: {rate} Hz, {audio.shape}")
            divisor = math.gcd(rate, 2000)
            reduced = resample_poly(audio, 2000 // divisor, rate // divisor).astype(np.float32)
            del audio
            sf.write(output, reduced, 2000, subtype="FLOAT")
            info = sf.info(output)
            if info.samplerate != 2000 or not 1790 <= info.duration <= 1810:
                raise ValueError(f"bad downsampled WAV: {name}")
            audio_info[name] = {"source_sample_rate": rate, "sample_rate": info.samplerate, "duration": info.duration,
                                "bytes": output.stat().st_size}
            print(json.dumps({"downloaded": name, **audio_info[name]}), flush=True)
    result = {"source_url": URL, "source_archive_bytes": ARCHIVE_SIZE,
              "selection": "SHA-256 filename ranks 13-28 among B_cont_0.wav through B_cont_49.wav; untouched after ranks 1-12 revealed a per-clip demeaning protocol error",
              "names": selected, "source_annotation_count": len(annotations),
              "selected_annotation_count": len(selected_annotations),
              "audio": audio_info,
              "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest()}
    (ROOT / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2), flush=True)
