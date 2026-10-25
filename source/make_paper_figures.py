"""Generate reproducible architecture, signal, and diversity figures for the paper."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.signal import spectrogram
import torch

# IEEE PDF eXpress rejects Type 3 fonts.  Keep text in vector figures as
# embedded TrueType outlines instead of Matplotlib's Type 3 default.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from .data import DEFAULT_DATA_ROOT, SpectrumNormaliser, TOLMixtureDataset, load_experiment_frames
from .diversity_eval import load_checkpoint
from .metrics import si_sdr
from .real_audio_data import RealAudioMixtureDataset, load_manifest
from .dclde import CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE, SAMPLE_RATE as DCLDE_SAMPLE_RATE
from .dclde_separation_experiment import CausalFeatureStore, InjectedUpcallDataset, _make_model


COLORS = {
    "navy": "#15324A",
    "blue": "#2E6F9E",
    "cyan": "#49A6A6",
    "mint": "#8BC9B8",
    "gold": "#E0A458",
    "coral": "#D96C5F",
    "paper": "#F7F4ED",
    "ink": "#17212B",
    "muted": "#65727E",
}


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def architecture_figure(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(14.4, 5.5))
    axis.set_xlim(0, 16.0)
    axis.set_ylim(0, 6.1)
    axis.axis("off")

    def box(x, y, width, height, label, color, subtitle=""):
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.3, edgecolor=color, facecolor=color + "18",
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height * 0.62, label, ha="center", va="center",
                  fontsize=13, fontweight="bold", color=COLORS["ink"], linespacing=0.92)
        if subtitle:
            axis.text(x + width / 2, y + height * 0.18, subtitle, ha="center", va="center",
                      fontsize=9.5, color=COLORS["muted"])

    def arrow(start, end, color=COLORS["muted"], label=""):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                       linewidth=1.6, color=color))
        if label:
            axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.13,
                      label, ha="center", fontsize=8, color=color)

    axis.text(0.25, 5.80, "PRIMARY METHOD  ·  60-s event-masked history  →  next 5-s mixture", fontsize=15,
              fontweight="bold", color=COLORS["ink"])
    axis.text(0.25, 5.48, "128 equal-width frequency groups × 79 frames; all history ends strictly before the target window", fontsize=11,
              color=COLORS["muted"])

    box(0.25, 4.05, 2.15, 0.94, r"$H_t$", COLORS["cyan"], "12 clean maps")
    box(2.85, 4.05, 2.05, 0.94, "GRU / copy", COLORS["cyan"], "or persistence")
    box(5.35, 4.05, 2.15, 0.94, r"$\hat B_t$", COLORS["cyan"], "79 × 128 log map")
    arrow((2.42, 4.52), (2.84, 4.52), COLORS["cyan"])
    arrow((4.92, 4.52), (5.33, 4.52), COLORS["cyan"])

    box(0.25, 2.40, 2.15, 0.94, r"$x_t=N_t+s_t$", COLORS["blue"], "current mixture")
    box(2.85, 2.40, 2.05, 0.94, r"$X_t$", COLORS["blue"], "mixture STFT")
    box(5.35, 2.40, 2.15, 0.94, r"$\hat R_t$", COLORS["gold"], "innovation")
    arrow((2.42, 2.87), (2.84, 2.87), COLORS["blue"])
    arrow((4.92, 2.87), (5.33, 2.87), COLORS["blue"])
    arrow((6.42, 4.02), (6.42, 3.37), COLORS["cyan"])

    box(8.05, 3.22, 2.05, 1.03, r"$[X_t,\hat B_t,\hat R_t]$", COLORS["gold"], "aligned maps")
    box(10.55, 3.22, 2.05, 1.03, "Adapter", COLORS["gold"], "zero-init residual")
    box(13.05, 3.22, 2.15, 1.03, "BS-RoFormer", COLORS["blue"], "separator")
    arrow((7.52, 4.50), (8.03, 3.96), COLORS["cyan"])
    arrow((7.52, 2.90), (8.03, 3.53), COLORS["gold"])
    arrow((10.12, 3.73), (10.53, 3.73), COLORS["gold"])
    arrow((12.62, 3.73), (13.03, 3.73), COLORS["blue"])

    box(13.05, 1.55, 2.15, 0.94, r"$\hat s_t,\;\hat N_t$", COLORS["coral"], "event + ocean")
    arrow((14.12, 3.20), (14.12, 2.51), COLORS["coral"])
    axis.text(0.25, 1.05, "TRAINING: real clean $N_t$ + synthetic $s_t$ give exact stems; only $x_t$ and past history reach causal arms.",
              fontsize=11.2, color=COLORS["ink"])
    axis.text(0.25, 0.52, "CONTROLS: fixed-mean Zero · other-day Shuffled · last-map Persistence · GRU Forecast · current-background Oracle.",
              fontsize=11.2, color=COLORS["muted"])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    if path.suffix.lower() == ".pdf":
        figure.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _db_spectrogram(audio: np.ndarray, sample_rate: int):
    frequencies, times, values = spectrogram(
        audio, fs=sample_rate, window="hann", nperseg=256, noverlap=192, mode="magnitude"
    )
    db = 20 * np.log10(np.maximum(values, 1e-7))
    return frequencies, times, db


def signal_figure(
    path: Path,
    checkpoint_path: Path,
    diversity_csv: Path,
    data_root: Path,
    device_name: str,
) -> None:
    device = torch.device(device_name)
    model, config = load_checkpoint(checkpoint_path, device)
    normaliser = SpectrumNormaliser(
        mean=np.asarray(config["normaliser_mean"], dtype=np.float32),
        scale=np.asarray(config["normaliser_scale"], dtype=np.float32),
    )
    rows = list(csv.DictReader(diversity_csv.open(encoding="utf-8")))
    candidates = [row for row in rows if row["source_family"] == "band_noise"]
    selected = min(candidates, key=lambda row: abs(float(row["conditioned_gain_db"]) - 0.54))
    frame = load_experiment_frames(data_root)["MB03_04"]
    dataset = TOLMixtureDataset(
        frame, "MB03_04", normaliser=normaliser,
        context_hours=int(config["context_hours"]), sample_rate=int(config["sample_rate"]),
        seconds=float(config["seconds"]), seed=int(config["seed"]),
        snr_range=(0.0, 0.0), source_profile="broad", source_family="band_noise",
    )
    target_origin = int(selected["origin"])
    item_index = int(np.flatnonzero(dataset.origins == target_origin)[0])
    item = dataset[item_index]
    with torch.inference_mode():
        output = model(
            item["mixture"][None].to(device), item["history"][None].to(device)
        )
    estimate = output["estimate"][0].detach().cpu().numpy()
    length = min(len(estimate), len(item["source"]))
    source = item["source"].numpy()[:length]
    mixture = item["mixture"].numpy()[:length]
    ocean = item["ocean"].numpy()[:length]
    estimate = estimate[:length]
    residual = mixture - estimate
    predicted = output["prediction"][0].detach().cpu().numpy() * normaliser.scale + normaliser.mean
    history_db = item["history"].numpy() * normaliser.scale + normaliser.mean
    sample_rate = int(config["sample_rate"])

    figure = plt.figure(figsize=(12.2, 7.8), constrained_layout=True)
    grid = figure.add_gridspec(3, 4, height_ratios=(1.0, 1.15, 0.85))
    history_axis = figure.add_subplot(grid[0, :3])
    image = history_axis.imshow(history_db.T, origin="lower", aspect="auto", cmap="viridis")
    history_axis.set_title("Causal history: 24 hourly one-third-octave spectra", loc="left", fontweight="bold")
    history_axis.set_xlabel("Hours before current mixture")
    history_axis.set_ylabel("TOL band index")
    figure.colorbar(image, ax=history_axis, label="dB", pad=0.01)
    forecast_axis = figure.add_subplot(grid[0, 3])
    bands = np.arange(len(predicted))
    forecast_axis.plot(predicted, bands, color=COLORS["gold"], lw=2, label="forecast")
    current_db = item["target_spectrum"].numpy() * normaliser.scale + normaliser.mean
    forecast_axis.plot(current_db, bands, color=COLORS["navy"], lw=1.3, label="current truth")
    forecast_axis.set_title("Current ocean prior", loc="left", fontweight="bold")
    forecast_axis.set_xlabel("TOL (dB)")
    forecast_axis.set_yticks([])
    forecast_axis.legend(frameon=False, fontsize=8)

    signals = (
        (source, "Synthetic target"),
        (mixture, "Ocean + target"),
        (estimate, "Separated target"),
        (residual, "Residual ocean"),
    )
    for column, (audio, title) in enumerate(signals):
        axis = figure.add_subplot(grid[1, column])
        frequencies, times, values = _db_spectrogram(audio, sample_rate)
        axis.pcolormesh(times, frequencies / 1000, values, shading="auto", cmap="magma", vmin=-85, vmax=-20)
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.set_xlabel("Time (s)")
        if column == 0:
            axis.set_ylabel("Frequency (kHz)")
        else:
            axis.set_yticklabels([])

    spectrum_axis = figure.add_subplot(grid[2, :])
    frequencies = np.fft.rfftfreq(length, 1 / sample_rate)
    for audio, label, color in (
        (ocean, "ocean background", COLORS["cyan"]),
        (source, "synthetic target", COLORS["gold"]),
        (estimate, "separated target", COLORS["coral"]),
    ):
        power = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(audio * np.hanning(length))), 1e-6))
        spectrum_axis.plot(frequencies / 1000, power, lw=1.2, label=label, color=color, alpha=0.9)
    source_score = float(si_sdr(torch.from_numpy(estimate), torch.from_numpy(source)))
    mixture_score = float(si_sdr(torch.from_numpy(mixture), torch.from_numpy(source)))
    spectrum_axis.set_title(
        f"Power spectra · band-noise example · SI-SDRi {source_score - mixture_score:+.2f} dB",
        loc="left", fontweight="bold",
    )
    spectrum_axis.set_xlabel("Frequency (kHz)")
    spectrum_axis.set_ylabel("Relative magnitude (dB)")
    spectrum_axis.grid(alpha=0.18)
    spectrum_axis.legend(ncol=3, frameon=False, loc="upper right")
    save_figure(figure, path)


def real_signal_figure(
    path: Path,
    checkpoint_path: Path,
    baseline_checkpoint_path: Path,
    data_root: Path,
    device_name: str,
) -> None:
    """Compare baseline and FC-BSR on a deterministic weak-target 5-s example."""
    device = torch.device(device_name)
    model, config = load_checkpoint(checkpoint_path, device)
    baseline, baseline_config = load_checkpoint(baseline_checkpoint_path, device)
    if float(config["seconds"]) != 5.0 or float(baseline_config["seconds"]) != 5.0:
        raise ValueError("the paper real-audio visualization requires 5-second checkpoints")
    normaliser = SpectrumNormaliser(
        mean=np.asarray(config["normaliser_mean"], dtype=np.float32),
        scale=np.asarray(config["normaliser_scale"], dtype=np.float32),
    )
    frame = load_experiment_frames(data_root)["MB03_04"]
    dataset = RealAudioMixtureDataset(
        frame, "MB03_04", normaliser, load_manifest()["records"],
        condition="forecast", context_hours=int(config["context_hours"]),
        sample_rate=int(config["sample_rate"]), seconds=float(config["seconds"]),
        clips_per_hour=int(config["clips_per_hour"]), seed=int(config["seed"]),
        snr_range=(-12.0, -12.0), source_profile="broad",
    )
    candidate_indices = np.linspace(0, len(dataset) - 1, min(192, len(dataset)), dtype=int)
    candidate_rows = []
    for index in candidate_indices:
        candidate = dataset[int(index)]
        with torch.inference_mode():
            fc_estimate = model(candidate["mixture"][None].to(device), candidate["history"][None].to(device))["estimate"][0]
            base_estimate = baseline(candidate["mixture"][None].to(device), candidate["history"][None].to(device))["estimate"][0]
        length = min(len(candidate["source"]), len(fc_estimate), len(base_estimate))
        target = candidate["source"][:length].to(device)
        mixture_tensor = candidate["mixture"][:length].to(device)
        mixture_score = float(si_sdr(mixture_tensor, target))
        fc_gain = float(si_sdr(fc_estimate[:length], target)) - mixture_score
        base_gain = float(si_sdr(base_estimate[:length], target)) - mixture_score
        candidate_rows.append((int(index), fc_gain, fc_gain - base_gain))
    fc_median = float(np.median([row[1] for row in candidate_rows]))
    delta_median = float(np.median([row[2] for row in candidate_rows]))
    index = min(candidate_rows, key=lambda row: abs(row[1] - fc_median) + abs(row[2] - delta_median))[0]
    item = dataset[index]
    with torch.inference_mode():
        output = model(item["mixture"][None].to(device), item["history"][None].to(device))
        baseline_output = baseline(item["mixture"][None].to(device), item["history"][None].to(device))
    estimate = output["estimate"][0].detach().cpu().numpy()
    baseline_estimate = baseline_output["estimate"][0].detach().cpu().numpy()
    length = min(len(estimate), len(baseline_estimate), len(item["source"]))
    source = item["source"].numpy()[:length]
    mixture = item["mixture"].numpy()[:length]
    ocean = item["ocean"].numpy()[:length]
    estimate = estimate[:length]
    baseline_estimate = baseline_estimate[:length]
    predicted = output["prediction"][0].detach().cpu().numpy() * normaliser.scale + normaliser.mean
    history_db = item["history"].numpy() * normaliser.scale + normaliser.mean

    figure = plt.figure(figsize=(13.2, 7.8), constrained_layout=True)
    grid = figure.add_gridspec(3, 5, height_ratios=(1.0, 1.15, 0.85))
    history_axis = figure.add_subplot(grid[0, :4])
    image = history_axis.imshow(history_db.T, origin="lower", aspect="auto", cmap="viridis")
    history_axis.set_title("Past real soundscape: 24 hourly TOL spectra", loc="left", fontweight="bold")
    history_axis.set_xlabel("Hours before current recording")
    history_axis.set_ylabel("TOL band index")
    figure.colorbar(image, ax=history_axis, label="dB", pad=0.01)
    forecast_axis = figure.add_subplot(grid[0, 4])
    bands = np.arange(len(predicted))
    current_db = item["target_spectrum"].numpy() * normaliser.scale + normaliser.mean
    forecast_axis.plot(predicted, bands, color=COLORS["gold"], lw=2, label="forecast")
    forecast_axis.plot(current_db, bands, color=COLORS["navy"], lw=1.3, label="current truth")
    forecast_axis.set_title("Future background state", loc="left", fontweight="bold")
    forecast_axis.set_xlabel("TOL (dB)")
    forecast_axis.set_yticks([])
    forecast_axis.legend(frameon=False, fontsize=8)

    for column, (audio, title) in enumerate((
        (ocean, "Real current ocean WAV"),
        (mixture, f"Mixture (target at {item['snr_db']:+.0f} dB)"),
        (source, "Synthetic target GT"),
        (baseline_estimate, "Zero-condition FiLM target"),
        (estimate, "FC-BSR target"),
    )):
        axis = figure.add_subplot(grid[1, column])
        frequencies, times, values = _db_spectrogram(audio, int(config["sample_rate"]))
        axis.pcolormesh(times, frequencies / 1000, values, shading="auto", cmap="magma", vmin=-85, vmax=-20)
        axis.set_title(title, fontsize=9.5, fontweight="bold")
        axis.set_xlabel("Time (s)")
        if column == 0:
            axis.set_ylabel("Frequency (kHz)")
        else:
            axis.set_yticklabels([])

    spectrum_axis = figure.add_subplot(grid[2, :])
    frequencies = np.fft.rfftfreq(length, 1 / int(config["sample_rate"]))
    for audio, label, color in (
        (source, "synthetic target GT", COLORS["gold"]),
        (baseline_estimate, "Zero-condition FiLM", COLORS["navy"]),
        (estimate, "FC-BSR", COLORS["coral"]),
    ):
        power = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(audio * np.hanning(length))), 1e-6))
        spectrum_axis.plot(frequencies / 1000, power, lw=1.2, label=label, color=color, alpha=0.9)
    mixture_score = float(si_sdr(torch.from_numpy(mixture), torch.from_numpy(source)))
    improvement = float(si_sdr(torch.from_numpy(estimate), torch.from_numpy(source))) - mixture_score
    baseline_improvement = float(si_sdr(torch.from_numpy(baseline_estimate), torch.from_numpy(source))) - mixture_score
    spectrum_axis.set_title(
        f"Spectra · 5-s input at {item['snr_db']:+.1f} dB · SI-SDRi: Zero FiLM {baseline_improvement:+.2f}, FC-BSR {improvement:+.2f} dB",
        loc="left", fontweight="bold",
    )
    spectrum_axis.set_xlabel("Frequency (kHz)")
    spectrum_axis.set_ylabel("Relative magnitude (dB)")
    spectrum_axis.grid(alpha=0.18)
    spectrum_axis.legend(ncol=3, frameon=False, loc="upper right")
    save_figure(figure, path)


def clean_main_signal_figure(
    path: Path,
    checkpoint_path: Path,
    baseline_checkpoint_path: Path,
    device_name: str,
) -> None:
    """Show a typical labelled-clean DCLDE background with a synthetic event."""
    device = torch.device(device_name)
    model, config = load_checkpoint(checkpoint_path, device)
    if int(config["sample_rate"]) != DCLDE_SAMPLE_RATE or float(config["seconds"]) != 5.0:
        raise ValueError("the main-setting illustration requires the 5-s DCLDE checkpoint")
    baseline = _make_model("bsr", len(config["spectrum_columns"]), int(config["seed"]), device)
    baseline_checkpoint = torch.load(baseline_checkpoint_path, map_location=device, weights_only=False)
    baseline.load_state_dict(baseline_checkpoint["model"])
    baseline.eval()
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    if not np.allclose(store.mean, config["normaliser_mean"], atol=1e-5):
        raise ValueError("figure feature normalisation differs from the trained clean-background model")
    dataset = InjectedUpcallDataset(
        store, "test", examples=96, seed=20261024,
        source_family="whale_pulse_packet", snr_range=(-12.0, -12.0),
    )
    candidate_indices = np.linspace(0, len(store.indices("test", 0)) - 1, 96, dtype=int)
    candidate_rows = []
    for index in candidate_indices:
        item = dataset[int(index)]
        with torch.inference_mode():
            mixture = item["mixture"][None].to(device)
            history = item["history"][None].to(device)
            estimate = model(mixture, history)["estimate"][0]
            base_estimate = baseline(mixture, history)["estimate"][0]
        length = min(mixture.shape[-1], estimate.shape[-1], base_estimate.shape[-1], item["event"].shape[-1])
        target = item["event"].to(device)[:length]
        mixture_score = float(si_sdr(mixture[0, :length], target))
        estimate, base_estimate = estimate[:length], base_estimate[:length]
        fc_gain = float(si_sdr(estimate, target)) - mixture_score
        base_gain = float(si_sdr(base_estimate, target)) - mixture_score
        candidate_rows.append((int(index), fc_gain, fc_gain - base_gain))
    medians = np.median(np.asarray([row[1:] for row in candidate_rows]), axis=0)
    index = min(candidate_rows, key=lambda row: np.linalg.norm(np.asarray(row[1:]) - medians))[0]
    item = dataset[index]
    with torch.inference_mode():
        output = model(item["mixture"][None].to(device), item["history"][None].to(device))
        base_output = baseline(item["mixture"][None].to(device), item["history"][None].to(device))
    arrays = [
        item["background"].numpy(), item["mixture"].numpy(), item["event"].numpy(),
        base_output["estimate"][0].cpu().numpy(), output["estimate"][0].cpu().numpy(),
    ]
    shared_length = min(map(len, arrays))
    arrays = [audio[:shared_length] for audio in arrays]
    maps = []
    freq_ref = time_ref = None
    for audio in arrays:
        frequencies, times, values = spectrogram(
            audio, fs=DCLDE_SAMPLE_RATE, window="hann", nperseg=512,
            noverlap=480, mode="magnitude",
        )
        db = 20.0 * np.log10(np.maximum(values, 1e-8))
        keep = (frequencies >= 50.0) & (frequencies <= 450.0)
        maps.append(db[keep])
        freq_ref, time_ref = frequencies[keep], times
    # The absolute ocean/mixture pair uses one physical scale.  The lower row
    # uses event-to-ocean contrast, again with one common scale: at -12 dB
    # input SNR the stems are invisible on the ocean's absolute dB scale.
    ocean_magnitude = np.power(10.0, maps[0] / 20.0)
    contrast_maps = [
        20.0 * np.log10((np.power(10.0, values / 20.0) + 1e-8) / (ocean_magnitude + 1e-8))
        for values in maps[2:]
    ]
    absolute_vmin, absolute_vmax = np.percentile(maps[1], [2.0, 99.5])
    contrast_vmin, contrast_vmax = -30.0, 15.0

    figure = plt.figure(figsize=(12.6, 7.9), constrained_layout=True)
    grid = figure.add_gridspec(3, 6, height_ratios=(0.75, 1.0, 1.0))
    history_axis = figure.add_subplot(grid[0, :4])
    history_db = item["history"].numpy() * store.scale + store.mean
    history_image = history_axis.imshow(
        history_db.T, origin="lower", aspect="auto", cmap="viridis",
        extent=(12, 0, 0.5, store.bands + 0.5),
    )
    history_axis.set_title("H_t: 12 preceding event-masked 5-s windows", loc="left", fontweight="bold")
    history_axis.set_xlabel("5-s steps before target (60 s total)")
    history_axis.set_ylabel("Log-frequency band")
    figure.colorbar(history_image, ax=history_axis, label="Log band power (a.u.)", pad=0.01)
    forecast_axis = figure.add_subplot(grid[0, 4:])
    predicted = output["prediction"][0].cpu().numpy() * store.scale + store.mean
    actual = store.target[int(store.indices("test", 0)[index])]
    bands = np.arange(store.bands)
    forecast_axis.plot(actual, bands, color=COLORS["navy"], label="current N_t")
    forecast_axis.plot(predicted, bands, color=COLORS["gold"], label="GRU forecast S_hat_t")
    forecast_axis.set_title("Next background state", loc="left", fontweight="bold")
    forecast_axis.set_xlabel("Log band power")
    forecast_axis.set_yticks([])
    forecast_axis.legend(frameon=False, fontsize=8)

    absolute_axes = []
    for column, (name, values) in enumerate((
        ("Actual current ocean $N_t$", maps[0]),
        ("Mixture $x_t=N_t+s_t$", maps[1]),
    )):
        axis = figure.add_subplot(grid[1, 3 * column : 3 * (column + 1)])
        absolute_mesh = axis.pcolormesh(
            time_ref, freq_ref, values, shading="auto", cmap="magma",
            vmin=absolute_vmin, vmax=absolute_vmax,
        )
        axis.set_title(name, fontsize=10, fontweight="bold")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Frequency (Hz)" if column == 0 else "")
        axis.set_ylim(50, 450)
        absolute_axes.append(axis)
    figure.colorbar(absolute_mesh, ax=absolute_axes, label="Magnitude (dB)", pad=0.012, shrink=0.86)
    contrast_axes = []
    for column, (name, values) in enumerate(zip(
        ("Synthetic event GT", "BS-RoFormer event", "FC-BSR event"),
        contrast_maps, strict=True,
    )):
        axis = figure.add_subplot(grid[2, 2 * column : 2 * (column + 1)])
        contrast_mesh = axis.pcolormesh(
            time_ref, freq_ref, values, shading="auto", cmap="coolwarm",
            vmin=contrast_vmin, vmax=contrast_vmax,
        )
        axis.set_title(name, fontsize=10, fontweight="bold")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Frequency (Hz)" if column == 0 else "")
        axis.set_ylim(50, 450)
        contrast_axes.append(axis)
    figure.colorbar(contrast_mesh, ax=contrast_axes,
                    label="Event magnitude / true ocean (dB)", pad=0.012, shrink=0.86)
    mixture_score = float(si_sdr(torch.from_numpy(arrays[1]), torch.from_numpy(arrays[2])))
    base_gain = float(si_sdr(torch.from_numpy(arrays[3]), torch.from_numpy(arrays[2]))) - mixture_score
    fc_gain = float(si_sdr(torch.from_numpy(arrays[4]), torch.from_numpy(arrays[2]))) - mixture_score
    figure.suptitle(
        f"Preliminary 30-band FiLM · 5-s clean background + synthetic whale pulse packet · {item['snr_db']:+.0f} dB · "
        f"SI-SDRi: BSR {base_gain:+.2f} dB, FC-BSR {fc_gain:+.2f} dB",
        fontsize=12, fontweight="bold",
    )
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    save_figure(figure, path)


def strength_sweep_figure(path: Path, results_path: Path) -> None:
    """Plot all real-background model settings across fixed input SNR."""
    results = json.loads(results_path.read_text(encoding="utf-8"))
    variants = ("none_film", "oracle_film", "film", "shuffled_film")
    labels = ("Zero-condition FiLM", "Oracle-FC-BSR", "Forecast-FC-BSR", "Shuffled-FC-BSR")
    colors = (COLORS["navy"], COLORS["mint"], COLORS["coral"], COLORS["gold"])
    markers = ("o", "s", "D", "^")
    figure, axis = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    for variant, label, color, marker in zip(variants, labels, colors, markers, strict=True):
        sweep = results["conditions"][variant]["strength_sweep"]
        snr = np.asarray(sorted(float(key.removesuffix("_dB")) for key in sweep))
        rows = [sweep[f"{value:+.0f}_dB"] for value in snr]
        score = np.asarray([row["source_si_sdri_db"] for row in rows])
        interval = np.asarray([row["source_si_sdri_ci95_db"] for row in rows])
        axis.plot(snr, score, marker=marker, color=color, linewidth=1.8, label=label)
        axis.fill_between(snr, interval[:, 0], interval[:, 1], color=color, alpha=0.10)
    axis.set_xlabel("Synthetic-target input SNR (dB)")
    axis.set_ylabel("Target SI-SDR improvement (dB)")
    axis.set_title("Unlabeled SanctSound ablation · 5-s windows", loc="left", fontweight="bold")
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    save_figure(figure, path)


def clean_strength_figure(path: Path, results_path: Path) -> None:
    """Main DCLDE test: paired full-reference SI-SDRi by injected event SNR."""
    with results_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    edges = np.arange(-18.0, 15.0, 3.0)
    centers = edges[:-1] + 1.5
    figure, axis = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    rng = np.random.default_rng(20261024)
    for metric, label, color, marker in (
        ("baseline_si_sdri_db", "Zero-condition FiLM", COLORS["navy"], "o"),
        ("fc_bsr_si_sdri_db", "FC-BSR (GRU + FiLM)", COLORS["coral"], "D"),
    ):
        means, lows, highs = [], [], []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            subset = [row for row in rows if low <= float(row["snr_db"]) < high]
            by_day = np.asarray([
                np.mean([float(row[metric]) for row in subset if row["day"] == day])
                for day in sorted({row["day"] for row in subset})
            ])
            means.append(float(by_day.mean()))
            draws = rng.choice(by_day, size=(5000, len(by_day)), replace=True).mean(axis=1)
            lows.append(float(np.quantile(draws, 0.025)))
            highs.append(float(np.quantile(draws, 0.975)))
        axis.plot(centers, means, color=color, marker=marker, lw=2.1, label=label)
        axis.fill_between(centers, lows, highs, color=color, alpha=0.14, linewidth=0)
    axis.axhline(0.0, color=COLORS["ink"], lw=0.8, alpha=0.5)
    axis.set_xticks(centers, [f"{value:g}" for value in centers])
    axis.set_xlabel("Synthetic-event input SNR (dB; 3-dB bins)")
    axis.set_ylabel("Target SI-SDR improvement (dB)")
    axis.set_title("Annotation-clean DCLDE test: main 5-s setting", loc="left", fontweight="bold")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    save_figure(figure, path)


def main_oracle_strength_figure(path: Path, result_path: Path) -> None:
    """One matched DCLDE SNR sweep with all six main-domain conditions."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with result_path.with_name(result_path.stem + "_samples.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    snr = np.asarray([float(row["snr_db"]) for row in rows])
    styles = {
        "bsr": (COLORS["ink"], "o", "-"),
        "zero_film": (COLORS["muted"], "s", "--"),
        "shuffled_film": (COLORS["gold"], "^", "--"),
        "persistence_film": (COLORS["blue"], "v", "--"),
        "film": (COLORS["coral"], "D", "-"),
        "oracle_film": (COLORS["cyan"], "P", "-"),
    }
    edges = np.arange(-18.0, 15.0, 3.0)
    centers = edges[:-1] + 1.5
    figure, axis = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    for method in result["methods"]:
        variant = method["variant"]
        score = np.asarray([float(row[variant]) for row in rows])
        means = [float(score[(snr >= low) & (snr < high)].mean())
                 for low, high in zip(edges[:-1], edges[1:], strict=True)]
        color, marker, linestyle = styles[variant]
        axis.plot(centers, means, color=color, marker=marker, linestyle=linestyle,
                  lw=2 if variant in {"film", "oracle_film", "bsr"} else 1.4,
                  markersize=4.5, label=method["method"])
    axis.set_xticks(centers)
    axis.set_xlabel("Synthetic-event input SNR (dB; 3-dB bins)")
    axis.set_ylabel("Target SI-SDR improvement (dB)")
    axis.set_title("DCLDE annotation-clean test · 5-s · six matched models", loc="left", fontweight="bold")
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, fontsize=7.5)
    save_figure(figure, path)


def real_ablation_figure(path: Path, results_path: Path) -> None:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    variants = ["none_film", "oracle_film", "film", "shuffled_film"]
    labels = ["Zero-FiLM", "Oracle $S_t$", "Forecast $\\hat S_t$", "Shuffled $S_j$"]
    rows = [results["conditions"][name] for name in variants]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), constrained_layout=True)
    x = np.arange(len(rows))
    score = np.asarray([row["source_si_sdri_db"] for row in rows])
    score_intervals = np.asarray([row["source_si_sdri_ci95_db"] for row in rows])
    axes[0].bar(x, score, color=[COLORS["navy"], COLORS["mint"], COLORS["gold"], COLORS["coral"]])
    axes[0].errorbar(x, score, yerr=np.vstack((score-score_intervals[:, 0], score_intervals[:, 1]-score)),
                     fmt="none", ecolor=COLORS["ink"], capsize=3)
    axes[0].set_ylabel("SI-SDR improvement (dB)")
    axes[0].set_title("Real current-ocean waveform", loc="left", fontweight="bold")

    gain = np.asarray([row["paired_gain_over_baseline_db"] for row in rows])
    intervals = np.asarray([row["paired_gain_ci95_db"] for row in rows])
    axes[1].axhline(0, color=COLORS["muted"], lw=1)
    axes[1].errorbar(x, gain, yerr=np.vstack((gain-intervals[:, 0], intervals[:, 1]-gain)),
                     fmt="o", color=COLORS["coral"], capsize=4)
    axes[1].set_ylabel("Paired gain over baseline (dB)")
    axes[1].set_title("Does timing information matter?", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=12)
        axis.grid(alpha=0.16, axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, path)


def diversity_figure(path: Path, results_path: Path) -> None:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    quintiles = list(results["by_divergence_quintile"])
    js_rows = [results["by_divergence_quintile"][key] for key in quintiles]
    x = np.arange(1, 6)
    axes[0].plot(x, [row["baseline_si_sdri_db"] for row in js_rows], "o-", color=COLORS["navy"], label="no prior")
    axes[0].plot(x, [row["conditioned_si_sdri_db"] for row in js_rows], "o-", color=COLORS["coral"], label="forecast prior")
    axes[0].set_title("Separation vs spectral shift", loc="left", fontweight="bold")
    axes[0].set_xlabel("JS-divergence quintile")
    axes[0].set_ylabel("SI-SDR improvement (dB)")
    axes[0].set_xticks(x)
    axes[0].legend(frameon=False)

    entropy_rows = [results["by_spectral_entropy_quintile"][key] for key in quintiles]
    gain = np.asarray([row["paired_conditioned_gain_db"] for row in entropy_rows])
    low = gain - np.asarray([row["paired_gain_ci95_db"][0] for row in entropy_rows])
    high = np.asarray([row["paired_gain_ci95_db"][1] for row in entropy_rows]) - gain
    axes[1].axhline(0, color=COLORS["muted"], lw=1)
    axes[1].errorbar(x, gain, yerr=np.vstack([low, high]), fmt="o-", capsize=3, color=COLORS["gold"])
    axes[1].set_title("Prior gain vs spectral entropy", loc="left", fontweight="bold")
    axes[1].set_xlabel("Entropy quintile")
    axes[1].set_ylabel("Conditioned − baseline (dB)")
    axes[1].set_xticks(x)

    family_rows = results["by_family"]
    keys = list(family_rows)
    labels = [key.replace("frequency_modulated", "FM").replace("_", " ") for key in keys]
    family_gain = np.asarray([family_rows[key]["paired_conditioned_gain_db"] for key in keys])
    colors = [COLORS["mint"] if value > 0 else COLORS["coral"] for value in family_gain]
    axes[2].barh(np.arange(len(keys)), family_gain, color=colors)
    axes[2].axvline(0, color=COLORS["muted"], lw=1)
    axes[2].set_yticks(np.arange(len(keys)), labels, fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Conditioned − baseline (dB)")
    axes[2].set_title("Gain by source family", loc="left", fontweight="bold")
    for axis in axes:
        axis.grid(alpha=0.16, axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, path)


def conditioning_ablation_figure(path: Path, results_path: Path) -> None:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    variants = ["concat", "film", "cross_attention"]
    labels = ["Concatenation", "FiLM", "Cross-attention"]
    rows = [results["variants"][name] for name in variants]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    x = np.arange(len(variants))

    validation = [row["validation_si_sdri_db"] for row in rows]
    test = [row["test_si_sdri_db"] for row in rows]
    width = 0.34
    axes[0].bar(x - width / 2, validation, width, color=COLORS["cyan"], label="validation")
    axes[0].bar(x + width / 2, test, width, color=COLORS["navy"], label="held-out test")
    axes[0].set_ylabel("SI-SDR improvement (dB)")
    axes[0].set_title("Matched conditioning ablation", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    gain = np.asarray([row["fixed_0_db_paired_gain_over_unconditioned_db"] for row in rows])
    intervals = np.asarray([row["fixed_0_db_paired_gain_ci95_db"] for row in rows])
    errors = np.vstack((gain - intervals[:, 0], intervals[:, 1] - gain))
    axes[1].axhline(0, color=COLORS["muted"], lw=1)
    axes[1].errorbar(x, gain, yerr=errors, fmt="o", capsize=4, color=COLORS["coral"])
    axes[1].set_ylabel("Paired gain over no prior (dB)")
    axes[1].set_title("Fixed 0 dB, 768 paired examples", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=12)
        axis.grid(alpha=0.16, axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].annotate("selected", (1, validation[1]), xytext=(1, validation[1] + 0.18),
                     ha="center", color=COLORS["coral"], fontsize=8, fontweight="bold")
    save_figure(figure, path)


def window_resolution_figure(path: Path, results_path: Path) -> None:
    """Vector summary of sub-hour forecast selection and real-audio separation."""
    results = json.loads(results_path.read_text(encoding="utf-8"))["results"]
    labels = ["5 s", "30 s", "1 min", "10 min"]
    x = np.arange(len(results))
    selected_mae = np.asarray([
        row["forecast_pretrained"]["validation"]["mae_db"] for row in results
    ])
    persistence_mae = np.asarray([
        row["forecast_pretrained"]["validation"]["persistence_mae_db"] for row in results
    ])
    gain = np.asarray([row["paired_gain_over_baseline_db"] for row in results])
    intervals = np.asarray([row["paired_gain_ci95_db"] for row in results])
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    width = 0.34
    axes[0].bar(x - width / 2, persistence_mae, width, color=COLORS["muted"], label="persistence")
    axes[0].bar(x + width / 2, selected_mae, width, color=COLORS["gold"], label="selected forecast")
    axes[0].set_ylabel("Validation forecast MAE (dB)")
    axes[0].set_title("Forecast must beat persistence", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    errors = np.vstack((gain - intervals[:, 0], intervals[:, 1] - gain))
    axes[1].axhline(0, color=COLORS["muted"], lw=1)
    axes[1].errorbar(x, gain, yerr=errors, fmt="o", capsize=4, color=COLORS["coral"])
    axes[1].set_ylabel("Paired SI-SDRi gain (dB)")
    axes[1].set_title("Real waveform, 1,024 paired clips", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Prediction window")
        axis.grid(alpha=0.16, axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, path)


def reference_free_figure(path: Path, results_path: Path) -> None:
    """Vector plot of the real-event restoration intervention and controls."""
    results = json.loads(results_path.read_text(encoding="utf-8"))
    gains = np.asarray(results["protocol"]["restoration_gains_db"])
    conditions = results["conditions"]
    styles = {
        "bs_roformer": ("Zero-condition FiLM", COLORS["navy"], "o"),
        "fc_bsr": ("FC-BSR", COLORS["coral"], "D"),
        "identity": ("Identity (remove nothing)", COLORS["muted"], "s"),
        "all_to_event": ("All-to-event", COLORS["gold"], "^"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    for key, (label, color, marker) in styles.items():
        axes[0].plot(gains, conditions[key]["restoration_curve"], marker=marker,
                     color=color, label=label, linewidth=2)
    axes[0].set_xlabel("Restored event-stem gain (dB)")
    axes[0].set_ylabel("Frozen classifier: event probability")
    axes[0].set_title("Interventional restoration curve", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    labels = ["Zero-condition FiLM", "FC-BSR"]
    keys = ["bs_roformer", "fc_bsr"]
    leakage = [conditions[key]["leakage_outside_event"] for key in keys]
    off_error = [conditions[key]["off_event_relative_error"] for key in keys]
    x = np.arange(len(keys))
    width = 0.34
    axes[1].bar(x - width / 2, leakage, width, color=COLORS["gold"], label="stem energy outside label")
    axes[1].bar(x + width / 2, off_error, width, color=COLORS["cyan"], label="off-event change")
    axes[1].set_xticks(x, labels, rotation=10)
    axes[1].set_ylim(0, 0.7)
    axes[1].set_ylabel("Lower is better")
    axes[1].set_title("Matched model preservation", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.16, axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--conditioned-checkpoint", type=Path, default=Path("experiments/broad_film/best.pt"))
    parser.add_argument("--diversity-results", type=Path, default=Path("results/diversity_results.json"))
    parser.add_argument("--ablation-results", type=Path, default=Path("results/conditioning_ablation.json"))
    parser.add_argument("--real-results", type=Path, default=Path("results/real_audio_ablation.json"))
    parser.add_argument("--real-checkpoint", type=Path, default=Path("experiments/real_film/best.pt"))
    parser.add_argument("--real-baseline-checkpoint", type=Path, default=Path("experiments/real_none_film/best.pt"))
    parser.add_argument("--main-checkpoint", type=Path, default=Path("models/practical_film.pt"))
    parser.add_argument("--clean-baseline-checkpoint", type=Path, default=Path("experiments/dclde_true_bsr/best.pt"))
    parser.add_argument(
        "--window-results", type=Path, default=Path("results/window_resolution_results.json")
    )
    parser.add_argument(
        "--dclde-results", type=Path, default=Path("results/dclde_real_event_results.json")
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    architecture_figure(args.output_dir / "architecture.pdf")
    conditioning_ablation_figure(
        args.output_dir / "conditioning_ablation.png", args.ablation_results
    )
    diversity_figure(args.output_dir / "diversity_results.png", args.diversity_results)
    if args.window_results.exists():
        window_resolution_figure(args.output_dir / "window_resolution.pdf", args.window_results)
    if args.dclde_results.exists():
        reference_free_figure(args.output_dir / "dclde_reference_free.pdf", args.dclde_results)
    main_oracle_results = Path("results/dclde_main_oracle.json")
    clean_test_csv = Path("results/dclde_clean_synthetic_test.csv")
    if main_oracle_results.exists():
        main_oracle_strength_figure(args.output_dir / "clean_background_strength.pdf", main_oracle_results)
    elif clean_test_csv.exists():
        clean_strength_figure(args.output_dir / "clean_background_strength.pdf", clean_test_csv)
    signal_figure(
        args.output_dir / "signal_visualization.png",
        args.conditioned_checkpoint,
        args.diversity_results.with_suffix(".csv"),
        args.data_root,
        args.device,
    )
    if args.real_results.exists() and args.real_checkpoint.exists():
        real_ablation_figure(args.output_dir / "real_audio_ablation.png", args.real_results)
        strength_sweep_figure(args.output_dir / "real_strength_sweep.pdf", args.real_results)
    if args.main_checkpoint.exists() and args.clean_baseline_checkpoint.exists() and CLEAN_OCEAN_FEATURE_CACHE.exists():
        clean_main_signal_figure(
            args.output_dir / "real_audio_visualization.png", args.main_checkpoint,
            args.clean_baseline_checkpoint, args.device,
        )
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
