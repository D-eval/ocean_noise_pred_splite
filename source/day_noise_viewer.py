"""Local vector spectrogram viewer with raw-audio recomputation on box zoom."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
import webbrowser

import numpy as np
import soundfile as sf

from .real_audio_data import load_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPO_ROOT / "day_viewer"
OVERVIEW_PATH = REPO_ROOT / "results" / "ocean_noise_day.json"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


def _power_spectrogram(
    windows: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    frequency_bins: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return band-mean PSD in dBFS/Hz for a matrix of waveform windows."""
    n_fft = windows.shape[1]
    taper = np.hanning(n_fft).astype(np.float32)
    normalisation = sample_rate * float(np.sum(taper * taper))
    edges = np.linspace(low_hz, high_hz, frequency_bins + 1, dtype=np.float64)
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    band_index = np.searchsorted(frequencies, edges)
    result = np.empty((len(windows), frequency_bins), dtype=np.float32)
    batch_size = 32
    for first in range(0, len(windows), batch_size):
        batch = windows[first : first + batch_size] * taper
        spectrum = np.fft.rfft(batch, axis=1)
        psd = (np.abs(spectrum) ** 2) / normalisation
        if psd.shape[1] > 2:
            psd[:, 1:-1] *= 2.0
        for band in range(frequency_bins):
            left = min(int(band_index[band]), psd.shape[1] - 1)
            right = max(left + 1, min(int(band_index[band + 1]), psd.shape[1]))
            result[first : first + len(batch), band] = np.mean(psd[:, left:right], axis=1)
    return 10.0 * np.log10(np.maximum(result, 1e-20)), edges, sample_rate / n_fft


class DayAudio:
    def __init__(self, deployment: str = "MB03_04") -> None:
        records = [
            record for record in load_manifest()["records"]
            if record["deployment"] == deployment
        ]
        self.records = sorted(records, key=lambda item: item["timestamp"])[:24]
        if len(self.records) != 24:
            raise ValueError(f"{deployment} requires 24 prepared hours; found {len(self.records)}")
        self.deployment = deployment
        self.sample_rate = 8000
        self.samples_per_hour = self.sample_rate * 3600
        self.duration_seconds = 24 * 3600
        self.date = self.records[0]["timestamp"][:10]
        for record in self.records:
            info = sf.info(record["local"])
            if info.samplerate != self.sample_rate or info.channels != 1:
                raise ValueError(f"invalid cache format: {record['local']}")
            if info.frames < self.samples_per_hour:
                raise ValueError(f"short hourly cache: {record['local']}")

    def read_windows(self, centres_seconds: np.ndarray, n_fft: int) -> np.ndarray:
        output = np.empty((len(centres_seconds), n_fft), dtype=np.float32)
        total_samples = self.duration_seconds * self.sample_rate
        with ExitStack() as stack:
            streams = [stack.enter_context(sf.SoundFile(record["local"])) for record in self.records]
            for row, centre in enumerate(centres_seconds):
                start = int(round(centre * self.sample_rate)) - n_fft // 2
                start = max(0, min(start, total_samples - n_fft))
                position = start
                written = 0
                while written < n_fft:
                    hour = min(position // self.samples_per_hour, len(streams) - 1)
                    offset = position - hour * self.samples_per_hour
                    count = min(n_fft - written, self.samples_per_hour - offset)
                    streams[hour].seek(offset)
                    block = streams[hour].read(count, dtype="float32", always_2d=False)
                    if len(block) != count:
                        raise ValueError(f"short read in {self.records[hour]['local']}")
                    output[row, written : written + count] = block
                    position += count
                    written += count
        output -= output.mean(axis=1, keepdims=True, dtype=np.float64)
        return output

    @lru_cache(maxsize=16)
    def wav_clip(self, start_millis: int, duration_millis: int) -> bytes:
        start = start_millis / 1000.0
        duration = duration_millis / 1000.0
        if not (0 <= start < self.duration_seconds):
            raise ValueError("audio start must stay within the prepared day")
        if not (1 <= duration <= 120):
            raise ValueError("audio duration must be between 1 and 120 seconds")
        duration = min(duration, self.duration_seconds - start)
        first_sample = int(round(start * self.sample_rate))
        sample_count = int(round(duration * self.sample_rate))
        audio = np.empty(sample_count, dtype=np.float32)
        position = first_sample
        written = 0
        with ExitStack() as stack:
            streams = [stack.enter_context(sf.SoundFile(record["local"])) for record in self.records]
            while written < sample_count:
                hour = min(position // self.samples_per_hour, len(streams) - 1)
                offset = position - hour * self.samples_per_hour
                count = min(sample_count - written, self.samples_per_hour - offset)
                streams[hour].seek(offset)
                block = streams[hour].read(count, dtype="float32", always_2d=False)
                if len(block) != count:
                    raise ValueError(f"short read in {self.records[hour]['local']}")
                audio[written : written + count] = block
                position += count
                written += count
        stream = io.BytesIO()
        sf.write(stream, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        return stream.getvalue()

    @lru_cache(maxsize=32)
    def detail(
        self,
        start_millis: int,
        end_millis: int,
        low_hz_tenths: int,
        high_hz_tenths: int,
        requested_time_bins: int = 480,
        requested_frequency_bins: int = 128,
    ) -> dict:
        start = start_millis / 1000.0
        end = end_millis / 1000.0
        low_hz = low_hz_tenths / 10.0
        high_hz = high_hz_tenths / 10.0
        if not (0 <= start < end <= self.duration_seconds):
            raise ValueError("time range must stay within the prepared 24-hour recording")
        if end - start < 0.25:
            raise ValueError("select at least 0.25 seconds")
        if not (0 <= low_hz < high_hz <= self.sample_rate / 2):
            raise ValueError("frequency range must stay within 0-4000 Hz")
        time_bins = min(max(int(requested_time_bins), 32), 600)
        frequency_bins = min(max(int(requested_frequency_bins), 24), 160)
        span_hz = high_hz - low_hz
        desired_df = max(span_hz / (frequency_bins * 1.5), self.sample_rate / 32768)
        proposed = self.sample_rate / desired_df
        n_fft = 2 ** int(math.ceil(math.log2(max(1024.0, proposed))))
        duration_samples = max(1, int((end - start) * self.sample_rate))
        n_fft = min(n_fft, 32768, 2 ** int(math.floor(math.log2(duration_samples))))
        n_fft = max(256, n_fft)
        window_seconds = n_fft / self.sample_rate
        usable_start = start + window_seconds / 2
        usable_end = end - window_seconds / 2
        if usable_start >= usable_end:
            centres = np.asarray([(start + end) / 2], dtype=np.float64)
        else:
            available = max(1, int((end - start) / max(window_seconds / 4, 1e-6)))
            count = min(time_bins, max(2, available))
            centres = np.linspace(usable_start, usable_end, count)
        windows = self.read_windows(centres, n_fft)
        values, frequency_edges, df = _power_spectrogram(
            windows, self.sample_rate, low_hz, high_hz, frequency_bins
        )
        time_edges = np.linspace(start, end, len(centres) + 1)
        finite = values[np.isfinite(values)]
        color_low, color_high = np.quantile(finite, [0.03, 0.97])
        return {
            "kind": "recomputed",
            "deployment": self.deployment,
            "date": self.date,
            "time_edges_seconds": np.round(time_edges, 4).tolist(),
            "frequency_edges_hz": np.round(frequency_edges, 3).tolist(),
            "values": np.round(values, 1).tolist(),
            "unit": "dBFS/Hz",
            "scale": "linear",
            "color_domain": [round(float(color_low), 1), round(float(color_high), 1)],
            "resolution": {
                "time_step_seconds": round(float(np.median(np.diff(time_edges))), 4),
                "stft_window_seconds": round(window_seconds, 4),
                "frequency_resolution_hz": round(float(df), 4),
                "time_bins": len(centres),
                "frequency_bins": frequency_bins,
                "source": "raw 8 kHz FLAC",
            },
        }


def overview() -> dict:
    raw = json.loads(OVERVIEW_PATH.read_text(encoding="utf-8"))
    centres = np.asarray(raw["frequency_hz"], dtype=np.float64)
    ratios = centres[1:] / centres[:-1]
    ratio = float(np.median(ratios))
    edges = np.concatenate(([centres[0] / math.sqrt(ratio)], centres * math.sqrt(ratio)))
    values = np.asarray(raw["spectrum_dbfs"], dtype=np.float32)
    color_low, color_high = np.quantile(values, [0.03, 0.97])
    return {
        "kind": "overview",
        "deployment": raw["deployment"],
        "date": raw["date"],
        "time_edges_seconds": list(range(0, 24 * 3600 + 60, 60)),
        "frequency_edges_hz": np.round(edges, 3).tolist(),
        "values": raw["spectrum_dbfs"],
        "unit": "dBFS/band",
        "scale": "log",
        "color_domain": [round(float(color_low), 1), round(float(color_high), 1)],
        "resolution": {
            "time_step_seconds": 60,
            "stft_window_seconds": 60,
            "frequency_resolution_hz": None,
            "time_bins": len(values),
            "frequency_bins": len(centres),
            "source": "one-minute overview",
        },
    }


def make_handler(audio: DayAudio):
    cached_overview = overview()

    class Handler(BaseHTTPRequestHandler):
        server_version = "OceanNoiseViewer/1.0"

        def _headers(self, status: int, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:",
            )
            self.end_headers()

        def _json(self, status: int, value: dict) -> None:
            payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8")
            self.wfile.write(payload)

        def _binary(self, status: int, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "private, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/overview":
                self._json(200, cached_overview)
                return
            if parsed.path == "/api/detail":
                try:
                    query = parse_qs(parsed.query)
                    result = audio.detail(
                        round(float(query["start"][0]) * 1000),
                        round(float(query["end"][0]) * 1000),
                        round(float(query["low"][0]) * 10),
                        round(float(query["high"][0]) * 10),
                    )
                    self._json(200, result)
                except Exception as error:
                    self._json(400, {"error": str(error)})
                return
            if parsed.path == "/api/audio":
                try:
                    query = parse_qs(parsed.query)
                    payload = audio.wav_clip(
                        round(float(query["start"][0]) * 1000),
                        round(float(query.get("duration", ["30"])[0]) * 1000),
                    )
                    self._binary(200, "audio/wav", payload)
                except Exception as error:
                    self._json(400, {"error": str(error)})
                return
            routes = {
                "/": "index.html",
                "/index.html": "index.html",
                "/styles.css": "styles.css",
                "/app.js": "app.js",
            }
            filename = routes.get(parsed.path)
            if filename is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = WEB_ROOT / filename
            self._headers(200, CONTENT_TYPES[path.suffix])
            self.wfile.write(path.read_bytes())

        def log_message(self, format, *args) -> None:
            print(f"{self.address_string()} - {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", default="MB03_04")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    audio = DayAudio(args.deployment)
    url = f"http://{args.host}:{args.port}/"
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(audio))
    except OSError:
        if args.open:
            try:
                with urlopen(url + "api/overview", timeout=1) as response:
                    if response.status == 200:
                        webbrowser.open(url)
                        return
            except Exception:
                pass
        raise
    print(f"Vector ocean-noise viewer ready at {url}", flush=True)
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
