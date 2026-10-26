"""Train-day whale morphology calibration and soft-mask pseudo-real targets.

The event classifier is deliberately absent. All fitted contours and pseudo
stems come from DCLDE's train split; validation and test WAVs are never read.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks, hilbert, istft, stft

from .dclde import CLEAN_BACKGROUND_MANIFEST, MANIFEST, SAMPLE_RATE
from .dclde_oracle_residual_experiment import _hash_file
from .synthesis import draw_injection, render_injection


ROOT = Path("data/dclde2013")
WAVEFORMS = ROOT / "train_pseudo_whale_stems.f16.npy"
PROFILES = ROOT / "train_whale_morphology.npz"
META = ROOT / "train_target_pool_meta.json"
REPORT = Path("results/dclde_train_target_pool.json")
SCALAR_REPORT = Path("results/dclde_train_whale_morphology.csv")
SAMPLES = SAMPLE_RATE * 5
FFT = 512
HOP = 128
PROFILE_POINTS = 64


def read_clip(path: str, start_seconds: float) -> np.ndarray:
    with sf.SoundFile(path) as file:
        if file.samplerate != SAMPLE_RATE or file.channels != 1:
            raise ValueError("DCLDE target pool requires 2-kHz mono WAV")
        file.seek(round(start_seconds * SAMPLE_RATE))
        audio = file.read(SAMPLES, dtype="float32")
    if len(audio) != SAMPLES:
        raise ValueError("incomplete five-second event or background window")
    return audio


def spectrum(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return stft(audio, SAMPLE_RATE, nperseg=FFT, noverlap=FFT-HOP,
                boundary="zeros", padded=True)


def event_time_gate(intervals: list[list[float]], frames: np.ndarray) -> np.ndarray:
    gate = np.zeros(len(frames), dtype=np.float32)
    for left, right in intervals:
        if right <= left:
            continue
        inside = (frames >= max(0.0, left)) & (frames <= min(5.0, right))
        gate[inside] = 1.0
    return gaussian_filter1d(gate, 1.3).clip(0.0, 1.0)


def extract_pseudo_and_profile(event: np.ndarray, before: np.ndarray,
                               after: np.ndarray, intervals: list[list[float]]) -> tuple[np.ndarray, dict]:
    """Estimate local background from flanking clean WAV and retain excess.

    The mask is label-gated and limited to the whale-bearing 40--950 Hz band.
    This reduces background leakage but does not make a true isolated stem.
    """
    frequencies, times, mix = spectrum(event)
    _, _, pre = spectrum(before)
    _, _, post = spectrum(after)
    local_pre = np.median(np.abs(pre), axis=1)
    local_post = np.median(np.abs(post), axis=1)
    fraction = (times / 5.0).clip(0.0, 1.0)
    local_log = (1.0-fraction)[None, :] * np.log(local_pre[:, None] + 1e-7)
    local_log += fraction[None, :] * np.log(local_post[:, None] + 1e-7)
    excess_db = 20.0 * (np.log(np.abs(mix) + 1e-7) - local_log) / np.log(10.0)
    positive = np.maximum(excess_db, 0.0)
    time_gate = event_time_gate(intervals, times)
    freq_gate = ((frequencies >= 40.0) & (frequencies <= 950.0)).astype(np.float32)
    mask = np.clip((excess_db-2.0)/9.0, 0.0, .95)
    mask = gaussian_filter(mask, (1.0, 0.8))
    mask *= freq_gate[:, None] * time_gate[None, :]
    _, pseudo = istft(mask * mix, SAMPLE_RATE, nperseg=FFT,
                      noverlap=FFT-HOP, input_onesided=True, boundary=True)
    pseudo = np.pad(pseudo[:SAMPLES], (0, max(0, SAMPLES-len(pseudo))))
    event_region = time_gate > .5
    whale_band = (frequencies >= 40.0) & (frequencies <= 950.0)
    salience = positive * np.abs(mix) * mask
    salience = salience[whale_band]
    band_freq = frequencies[whale_band]
    frame_weight = salience.sum(axis=0)
    active = event_region & (frame_weight > np.quantile(frame_weight[event_region], .1))
    if active.sum() < 4:
        raise ValueError("event has too few active time-frequency frames")
    ridge = band_freq[np.argmax(salience, axis=0)]
    ridge = gaussian_filter1d(ridge, 1.0)
    track = ridge[active]
    weights = frame_weight[active]
    timeline = times[active]
    normalized_time = (timeline-timeline.min()) / max(np.ptp(timeline), 1e-6)
    line = np.polyfit(normalized_time, track, 1, w=np.sqrt(weights))
    curve = np.polyfit(normalized_time, track, 2, w=np.sqrt(weights))
    smooth_track = gaussian_filter1d(track, 2.0)
    jitter = float(np.std(track-smooth_track))
    band_total = salience[:, active].sum(axis=1)
    total = band_total.sum()
    band_centre = float(np.sum(band_freq*band_total)/max(total, 1e-9))
    bandwidth = float(np.sqrt(np.sum((band_freq-band_centre)**2*band_total)/max(total, 1e-9)))
    mean_spectrum = salience[:, active].mean(axis=1)
    prominent, _ = find_peaks(mean_spectrum, prominence=max(mean_spectrum.max()*.09, 1e-9),
                             distance=2)
    peaks = sorted(prominent, key=lambda k: mean_spectrum[k], reverse=True)[:10]
    peaks.sort(key=lambda k: band_freq[k])
    harmonic_count = max(1, min(len(peaks), 10))
    if len(peaks) >= 2:
        ranks = np.arange(1, len(peaks)+1)
        harmonic_decay = float(-np.polyfit(np.log(ranks),
                                          np.log(mean_spectrum[peaks]+1e-9), 1)[0])
    else:
        harmonic_decay = 1.4
    envelope = frame_weight[active]
    envelope = envelope / max(float(np.max(envelope)), 1e-9)
    peak_positions, _ = find_peaks(envelope, prominence=.22, distance=3)
    pulse_gap = (float(np.median(np.diff(timeline[peak_positions])))
                 if len(peak_positions) >= 2 else 0.0)
    support = np.flatnonzero(active)
    profile_x = np.linspace(timeline.min(), timeline.max(), PROFILE_POINTS)
    track_profile = np.interp(profile_x, timeline, track).astype(np.float32)
    env_profile = np.interp(profile_x, timeline, envelope).astype(np.float32)
    env_profile *= np.sin(np.linspace(0, np.pi, PROFILE_POINTS))**.5
    target_rms = float(np.sqrt(np.mean(pseudo**2, dtype=np.float64)))
    event_rms = float(np.sqrt(np.mean(event**2, dtype=np.float64)))
    contrast = float(np.quantile(excess_db[whale_band][:, event_region], .98))
    if target_rms < max(event_rms*.015, 1e-7) or contrast < 3.0:
        raise ValueError("acoustic excess too weak for a useful pseudo stem")
    pseudo = (pseudo / target_rms).astype(np.float32)
    return pseudo, {
        "duration_seconds": float(times[support[-1]]-times[support[0]]),
        "f_min_hz": float(np.quantile(track, .05)),
        "f_max_hz": float(np.quantile(track, .95)),
        "delta_f_hz": float(track[-1]-track[0]),
        "slope_hz_per_second": float(line[0]/max(np.ptp(timeline), 1e-6)),
        "curvature_hz": float(curve[0]),
        "bandwidth_hz": bandwidth,
        "harmonic_count": harmonic_count,
        "harmonic_decay": harmonic_decay,
        "envelope_rise_fraction": float(np.argmax(env_profile)/(PROFILE_POINTS-1)),
        "frequency_jitter_hz": jitter,
        "pulse_gap_seconds": pulse_gap,
        "event_count": len(intervals),
        "contrast_db": contrast,
        "track_hz": track_profile,
        "envelope": env_profile,
    }


def prepare_pool(max_flank_seconds: float = 90.0) -> dict:
    event_frame = pd.read_csv(MANIFEST)
    clean_frame = pd.read_csv(CLEAN_BACKGROUND_MANIFEST)
    events = event_frame.loc[event_frame.split.eq("train") & event_frame.label.eq(1)]
    clean = clean_frame.loc[clean_frame.split.eq("train") & clean_frame.label.eq(0)]
    if events.empty or clean.empty:
        raise ValueError("training events and annotation-clean references are required")
    signature = {"event_manifest_sha256": _hash_file(MANIFEST),
                 "clean_manifest_sha256": _hash_file(CLEAN_BACKGROUND_MANIFEST),
                 "max_flank_seconds": max_flank_seconds,
                 "mask": "positive log-spectral excess above 2 dB, 40--950 Hz, label gated"}
    by_path = {path: np.sort(group.start_seconds.to_numpy())
               for path, group in clean.groupby("path")}
    rows = []
    waveforms = []
    skipped = {"missing_flanks": 0, "low_acoustic_excess": 0}
    for row in events.itertuples():
        positions = by_path.get(row.path, np.empty(0))
        cursor = bisect.bisect_left(positions, row.start_seconds)
        if cursor == 0 or cursor == len(positions):
            skipped["missing_flanks"] += 1
            continue
        before_at = float(positions[cursor-1])
        after_at = float(positions[cursor])
        if (row.start_seconds-(before_at+5) > max_flank_seconds
                or after_at-(row.start_seconds+5) > max_flank_seconds):
            skipped["missing_flanks"] += 1
            continue
        intervals = json.loads(row.event_intervals)
        try:
            pseudo, profile = extract_pseudo_and_profile(
                read_clip(row.path, row.start_seconds),
                read_clip(row.path, before_at),
                read_clip(row.path, after_at), intervals)
        except ValueError:
            skipped["low_acoustic_excess"] += 1
            continue
        if row.split != "train":
            raise AssertionError("non-training event entered the target bank")
        profile.update({"day": row.day, "path": row.path,
                        "start_seconds": float(row.start_seconds),
                        "before_seconds": before_at, "after_seconds": after_at})
        rows.append(profile)
        waveforms.append(pseudo)
        if len(rows) % 250 == 0:
            print(json.dumps({"extracted": len(rows), "events_seen": len(rows)+sum(skipped.values())}),
                  flush=True)
    if len(rows) < 100:
        raise ValueError("too few usable training-day pseudo targets")
    WAVEFORMS.parent.mkdir(parents=True, exist_ok=True)
    maps = np.lib.format.open_memmap(WAVEFORMS, mode="w+", dtype=np.float16,
                                     shape=(len(waveforms), SAMPLES))
    for index, waveform in enumerate(waveforms):
        maps[index] = waveform
    maps.flush()
    fields = [key for key in rows[0] if key not in {"track_hz", "envelope", "path"}]
    arrays = {key: np.asarray([row[key] for row in rows]) for key in fields}
    arrays["track_hz"] = np.stack([row["track_hz"] for row in rows])
    arrays["envelope"] = np.stack([row["envelope"] for row in rows])
    np.savez_compressed(PROFILES, **arrays)
    numeric = [key for key in rows[0] if key not in {"day", "path", "start_seconds",
               "before_seconds", "after_seconds", "track_hz", "envelope"}]
    quantiles = {key: {str(q): float(np.quantile([row[key] for row in rows], q))
                       for q in (.05, .5, .95)} for key in numeric}
    days = sorted(set(row["day"] for row in rows))
    report = {"source": "DCLDE training days only; annotated event WAV plus flanking annotation-clean same-WAV windows",
              "train_days": days, "candidate_events": len(events), "accepted_events": len(rows),
              "skipped": skipped, "acoustic_parameters": quantiles,
              "method": signature, "pseudo_waveforms": str(WAVEFORMS),
              "empirical_profiles": str(PROFILES), "real_stem_gt": False,
              "evaluation_classifier_used_in_selection": False}
    META.write_text(json.dumps(signature | {"count": len(rows), "days": days}, indent=2), encoding="utf-8")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    export_scalar_profiles()
    return report


def export_scalar_profiles() -> Path:
    """Publish auditable scalar estimates without duplicating the WAV bank."""
    profiles = np.load(PROFILES)
    fields = [key for key in profiles.files if key not in {"track_hz", "envelope"}]
    SCALAR_REPORT.parent.mkdir(exist_ok=True)
    with SCALAR_REPORT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["profile_index", *fields])
        for index in range(len(profiles[fields[0]])):
            writer.writerow([index, *[profiles[key][index].item()
                                      for key in fields]])
    return SCALAR_REPORT


class EmpiricalTargetPool:
    def __init__(self, subset: str = "all"):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        manifest = pd.read_csv(MANIFEST)
        train_days = set(manifest.loc[manifest.split.eq("train"), "day"])
        if not set(report["train_days"]).issubset(train_days):
            raise ValueError("pool includes a non-training recording day")
        self.waveforms = np.load(WAVEFORMS, mmap_mode="r")
        self.profiles = np.load(PROFILES)
        if len(self.waveforms) != report["accepted_events"]:
            raise ValueError("pseudo waveform count does not match profile count")
        if subset not in {"all", "fit", "validation"}:
            raise ValueError("unknown empirical event subset")
        permutation = np.random.default_rng(20261026).permutation(len(self.waveforms))
        cutoff = int(round(.8 * len(permutation)))
        self.indices = {"all": np.arange(len(self.waveforms)),
                        "fit": permutation[:cutoff],
                        "validation": permutation[cutoff:]}[subset]
        self.subset = subset

    def procedural(self, seed: int) -> np.ndarray:
        """Render calibrated whale or retained vessel proxy with exact stem."""
        rng = np.random.default_rng(seed)
        # Half the final pool is pseudo-real whale. Four sevenths of this
        # procedural half keeps the original two-of-seven vessel share overall.
        if rng.random() < 4.0/7.0:
            family = ("vessel_tonal" if rng.random() < .5
                      else "vessel_cavitation")
            parameters = draw_injection(seed, snr_range=(0.0, 0.0),
                                        profile="marine", family=family)
            return render_injection(parameters, SAMPLE_RATE, SAMPLES)
        index = int(self.indices[int(rng.integers(len(self.indices)))])
        profile = self.profiles
        track = profile["track_hz"][index].astype(np.float64)
        envelope = profile["envelope"][index].astype(np.float64)
        target = np.zeros(SAMPLES, dtype=np.float32)
        repeats = int(np.clip(profile["event_count"][index], 1, 3))
        gap = float(np.clip(profile["pulse_gap_seconds"][index], .08, .6))
        duration = float(np.clip(profile["duration_seconds"][index] *
                                 rng.uniform(.85, 1.15), .15, 4.5))
        duration = min(duration, (4.8-(repeats-1)*gap)/repeats)
        length = min(SAMPLES, max(64, round(duration*SAMPLE_RATE)))
        gap_samples = round(gap*SAMPLE_RATE)
        full_length = repeats*length+(repeats-1)*gap_samples
        start = int(rng.integers(0, SAMPLES-full_length+1))
        harmonic_count = int(np.clip(profile["harmonic_count"][index], 1, 7))
        decay = float(np.clip(profile["harmonic_decay"][index], .4, 3.0))
        bandwidth = float(np.clip(profile["bandwidth_hz"][index], 20, 450))
        for event_number in range(repeats):
            u = np.linspace(0, 1, length)
            frequency = np.interp(u, np.linspace(0, 1, PROFILE_POINTS), track)
            frequency *= rng.uniform(.92, 1.08)
            jitter = float(np.clip(profile["frequency_jitter_hz"][index], 0, 100))
            noise = gaussian_filter1d(rng.standard_normal(length), SAMPLE_RATE*.012)
            noise /= max(float(np.std(noise)), 1e-6)
            frequency = np.clip(frequency + noise*jitter*rng.uniform(.2,.8), 35, 800)
            phase = 2*np.pi*np.cumsum(frequency)/SAMPLE_RATE + rng.uniform(-np.pi,np.pi)
            local = np.zeros(length)
            for harmonic in range(1, harmonic_count+1):
                if frequency.max()*harmonic >= SAMPLE_RATE*.49:
                    break
                local += harmonic**(-decay)*rng.uniform(.75,1.25)*np.sin(
                    phase*(harmonic+rng.normal(0,.002)) + rng.uniform(-np.pi,np.pi))
            amplitude = np.interp(u, np.linspace(0,1,PROFILE_POINTS), envelope)
            roughness = gaussian_filter1d(rng.standard_normal(length),
                                          max(1.0, SAMPLE_RATE/(bandwidth*2)))
            roughness /= max(float(np.std(roughness)), 1e-6)
            local = (local + roughness*np.clip(bandwidth/700, .03, .25))*amplitude
            offset = start+event_number*(length+gap_samples)
            target[offset:offset+length] += local.astype(np.float32)
        rms = float(np.sqrt(np.mean(target**2, dtype=np.float64)))
        return (target/max(rms,1e-8)).astype(np.float32)

    def pseudo_real(self, seed: int) -> np.ndarray:
        """Reuse an extracted train-day call with independent timing and phase."""
        rng = np.random.default_rng(seed)
        index = int(self.indices[int(rng.integers(len(self.indices)))])
        source = self.waveforms[index].astype(np.float32)
        angle = float(rng.uniform(-np.pi, np.pi))
        source = (np.cos(angle)*source + np.sin(angle)*np.imag(hilbert(source))).astype(np.float32)
        shift = int(rng.integers(-SAMPLE_RATE//2, SAMPLE_RATE//2+1))
        target = np.zeros_like(source)
        if shift >= 0:
            target[shift:] = source[:SAMPLES-shift]
        else:
            target[:shift] = source[-shift:]
        rms = float(np.sqrt(np.mean(target**2, dtype=np.float64)))
        return (target/max(rms,1e-8)).astype(np.float32)

    def sample(self, seed: int, route: str) -> np.ndarray:
        if route == "empirical_procedural":
            return self.procedural(seed)
        if route == "pseudo_real":
            return self.pseudo_real(seed)
        raise ValueError(f"unknown empirical target route: {route}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-flank-seconds", type=float, default=90.0)
    args = parser.parse_args()
    print(json.dumps(prepare_pool(args.max_flank_seconds), indent=2), flush=True)


if __name__ == "__main__":
    main()
