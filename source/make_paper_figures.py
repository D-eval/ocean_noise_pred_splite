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

from .data import DEFAULT_DATA_ROOT, SpectrumNormaliser, TOLMixtureDataset, load_experiment_frames
from .diversity_eval import load_checkpoint
from .metrics import si_sdr


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
    figure, axis = plt.subplots(figsize=(12.4, 4.4))
    axis.set_xlim(0, 12.4)
    axis.set_ylim(0, 4.4)
    axis.axis("off")

    def box(x, y, width, height, label, color, subtitle=""):
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.3, edgecolor=color, facecolor=color + "18",
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height * 0.62, label, ha="center", va="center",
                  fontsize=9.1, fontweight="bold", color=COLORS["ink"], linespacing=0.92)
        if subtitle:
            axis.text(x + width / 2, y + height * 0.18, subtitle, ha="center", va="center",
                      fontsize=7.2, color=COLORS["muted"])

    def arrow(start, end, color=COLORS["muted"], label=""):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                       linewidth=1.6, color=color))
        if label:
            axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.13,
                      label, ha="center", fontsize=8, color=color)

    axis.text(0.15, 4.12, "CAUSAL FORECAST PATH", fontsize=9, fontweight="bold", color=COLORS["cyan"])
    box(0.15, 2.75, 1.75, 0.92, "Past 24 h TOL", COLORS["cyan"], "history ending at t-1")
    box(2.35, 2.75, 1.6, 0.92, "GRU encoder", COLORS["cyan"], "causal latent zₜ")
    box(4.4, 2.75, 1.75, 0.92, "Spectrum decoder", COLORS["cyan"], "predicted background Ŝₜ")
    box(7.15, 2.64, 4.6, 1.08, "Joint objective", COLORS["coral"],
        "waveform + multi-resolution STFT + 0.2 × forecast MAE")
    arrow((1.9, 3.21), (2.35, 3.21), COLORS["cyan"])
    arrow((3.95, 3.21), (4.4, 3.21), COLORS["cyan"])

    axis.text(0.15, 1.93, "AUDIO SEPARATION PATH", fontsize=9, fontweight="bold", color=COLORS["blue"])
    box(0.15, 0.55, 1.75, 0.92, "Current mixture xₜ", COLORS["blue"], "ocean + unknown source")
    box(2.35, 0.55, 1.6, 0.92, "STFT + band split", COLORS["blue"], "frequency-band tokens")
    box(4.4, 0.55, 1.75, 0.92, "Condition injection", COLORS["gold"], "concat / FiLM / cross-attn")
    box(6.65, 0.55, 1.7, 0.92, "Axial Transformer", COLORS["blue"], "time × frequency attention")
    box(8.85, 0.55, 1.55, 0.92, "Complex mask", COLORS["blue"], "iSTFT reconstruction")
    box(10.85, 0.55, 1.4, 0.92, "Target ŝₜ", COLORS["coral"], "residual = ocean")
    for x1, x2 in ((1.9, 2.35), (3.95, 4.4), (6.15, 6.65), (8.35, 8.85), (10.4, 10.85)):
        arrow((x1, 1.01), (x2, 1.01), COLORS["blue"])
    arrow((5.28, 2.75), (5.28, 1.47), COLORS["gold"], "forecast prior")
    arrow((11.55, 1.48), (10.35, 2.64), COLORS["coral"], "separation loss")
    arrow((7.15, 3.18), (6.15, 3.18), COLORS["coral"])
    save_figure(figure, path)


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
    labels = ["harmonic", "chirp", "inharm.", "impulse", "FM", "band noise"]
    keys = list(family_rows)
    family_gain = np.asarray([family_rows[key]["paired_conditioned_gain_db"] for key in keys])
    colors = [COLORS["mint"] if value > 0 else COLORS["coral"] for value in family_gain]
    axes[2].barh(np.arange(len(keys)), family_gain, color=colors)
    axes[2].axvline(0, color=COLORS["muted"], lw=1)
    axes[2].set_yticks(np.arange(len(keys)), labels)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--conditioned-checkpoint", type=Path, default=Path("models/practical_film.pt"))
    parser.add_argument("--diversity-results", type=Path, default=Path("results/diversity_results.json"))
    parser.add_argument("--ablation-results", type=Path, default=Path("results/conditioning_ablation.json"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    architecture_figure(args.output_dir / "architecture.png")
    conditioning_ablation_figure(
        args.output_dir / "conditioning_ablation.png", args.ablation_results
    )
    diversity_figure(args.output_dir / "diversity_results.png", args.diversity_results)
    signal_figure(
        args.output_dir / "signal_visualization.png",
        args.conditioned_checkpoint,
        args.diversity_results.with_suffix(".csv"),
        args.data_root,
        args.device,
    )
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
