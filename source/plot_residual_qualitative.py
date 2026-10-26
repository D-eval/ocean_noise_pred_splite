"""Paired qualitative audit of the current 128-group causal residual separators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
import torch

from .dclde import CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE
from .dclde_causal_highres_experiment import (
    CausalResidualBSR, FORECAST_MAPS_PATH, HighResolutionStore,
)
from .dclde_separation_experiment import CausalFeatureStore, InjectedUpcallDataset, _make_model
from .metrics import si_sdr

plt.rcParams["pdf.fonttype"] = 42


def spectral_maps(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies, times, values = stft(waveform, fs=2000, window="hann",
                                      nperseg=512, noverlap=480)
    keep = (frequencies >= 50) & (frequencies <= 450)
    return frequencies[keep], times, abs(values[keep])


def load_models(store: CausalFeatureStore, device: torch.device,
                persistence_path: Path, forecast_path: Path):
    models = {}
    for name, path in (("persistence", persistence_path), ("forecast", forecast_path)):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("interface") != f"{name}_residual_128":
            raise ValueError(f"wrong residual interface: {path}")
        model = CausalResidualBSR(store.bands, 20261022, device,
                                  Path("models/practical_real.pt")).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        models[name] = model.eval()
    baseline = _make_model("bsr", store.bands, 20261022, device)
    baseline.load_state_dict(torch.load("experiments/dclde_true_bsr/best.pt",
                                       map_location=device, weights_only=False)["model"])
    return baseline.eval(), models


@torch.inference_mode()
def run(output: Path, device_name: str = "cuda",
        persistence_path: Path = Path("models/empirical_persistence_residual_128.pt"),
        forecast_path: Path = Path("models/empirical_forecast_residual_128.pt")) -> dict:
    device = torch.device(device_name)
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    cache = HighResolutionStore(store)
    baseline, models = load_models(store, device, persistence_path, forecast_path)
    dataset = InjectedUpcallDataset(store, "test", 96, 20261024,
                                    source_family="whale_pulse_packet",
                                    snr_range=(-12, -12))
    forecast_maps = np.load(FORECAST_MAPS_PATH, mmap_mode="r")
    positions = np.full(len(store.frame), -1, dtype=np.int32)
    positions[np.flatnonzero(store.eligible)] = np.arange(len(forecast_maps))
    candidates = []
    cohort_leakage = {name: [] for name in ("GT", "BSR", "persistence", "forecast")}
    for index in range(len(dataset)):
        item = dataset[index]
        manifest_index = int(dataset.rows[index])
        mixture = item["mixture"][None].to(device)
        target = item["source"][None].to(device)
        persistent = cache.current(int(cache.history_indices[manifest_index, -1]))
        forecast = np.asarray(forecast_maps[positions[manifest_index]], dtype=np.float32)
        conditions = {"persistence": persistent, "forecast": forecast}
        outputs = {"BSR": baseline(mixture, None)["estimate"].cpu().numpy()[0]}
        for name, model in models.items():
            condition = torch.from_numpy(conditions[name][None]).to(device)
            outputs[name] = model(mixture, condition).cpu().numpy()[0]
        gains = {name: float(si_sdr(torch.from_numpy(waveform)[None], target.cpu()[..., :len(waveform)]))
                 - float(si_sdr(mixture.cpu()[..., :len(waveform)], target.cpu()[..., :len(waveform)]))
                 for name, waveform in outputs.items()}
        shared = min(len(item["ocean"]), len(item["source"]), *(len(w) for w in outputs.values()))
        _, _, ocean_power = spectral_maps(item["ocean"].numpy()[:shared])
        _, _, source_power = spectral_maps(item["source"].numpy()[:shared])
        absent_bins = source_power < .1*np.maximum(ocean_power, 1e-8)
        for name, waveform in (("GT", item["source"].numpy()), *outputs.items()):
            _, _, estimate_power = spectral_maps(waveform[:shared])
            db = 20*np.log10((estimate_power+1e-8)/(ocean_power+1e-8))
            cohort_leakage[name].append(float(np.median(db[absent_bins])))
        candidates.append((item, outputs, gains))
    # Predetermined median across all 96 examples, not the most flattering case.
    gains_matrix = np.asarray([[row[2][name] for name in ("BSR", "persistence", "forecast")]
                              for row in candidates])
    median = np.median(gains_matrix, axis=0)
    chosen = int(np.argmin(np.linalg.norm(gains_matrix-median, axis=1)))
    item, outputs, gains = candidates[chosen]
    ocean = item["ocean"].numpy()
    mixture = item["mixture"].numpy()
    source = item["source"].numpy()
    length = min(len(ocean), len(mixture), len(source), *(len(w) for w in outputs.values()))
    ocean, mixture, source = ocean[:length], mixture[:length], source[:length]
    outputs = {name: waveform[:length] for name, waveform in outputs.items()}
    frequencies, times, ocean_mag = spectral_maps(ocean)
    _, _, mixture_mag = spectral_maps(mixture)
    _, _, source_mag = spectral_maps(source)
    stem_mags = {name: spectral_maps(waveform)[2] for name, waveform in outputs.items()}
    reference = np.maximum(ocean_mag, 1e-8)
    contrast = {"GT": 20*np.log10((source_mag+1e-8)/reference)}
    contrast.update({name: 20*np.log10((mag+1e-8)/reference)
                     for name, mag in stem_mags.items()})
    # Exclude only bins where target power clearly exceeds the local ocean.
    absent = source_mag < 0.1*reference
    leakage = {name: float(np.median(values[absent]))
               for name, values in contrast.items()}
    leakage_p90 = {name: float(np.quantile(values[absent], .9))
                   for name, values in contrast.items()}
    leakage_above_minus10 = {name: float(np.mean(values[absent] > -10))
                             for name, values in contrast.items()}
    figure = plt.figure(figsize=(13.4, 6.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(.85, 1))
    absolute = []
    for index, (title, magnitude) in enumerate((("Clean ocean $N_t$", ocean_mag),
                                                 ("Mixture $x_t$", mixture_mag))):
        axis = figure.add_subplot(grid[0, 2*index:2*index+2])
        image = axis.pcolormesh(times, frequencies, 20*np.log10(magnitude+1e-8),
                                cmap="magma", vmin=-80, vmax=-20, shading="auto")
        axis.set(title=title, xlabel="Time (s)", ylabel="Frequency (Hz)" if index == 0 else "")
        absolute.append(axis)
    figure.colorbar(image, ax=absolute, label="Magnitude (dB)", shrink=.8)
    lower = []
    labels = (("GT", "Synthetic event GT"), ("BSR", f"BSR · {gains['BSR']:+.1f} dB"),
              ("persistence", f"Persistence · {gains['persistence']:+.1f} dB"),
              ("forecast", f"Forecast · {gains['forecast']:+.1f} dB"))
    for index, (key, title) in enumerate(labels):
        axis = figure.add_subplot(grid[1, index])
        image = axis.pcolormesh(times, frequencies, contrast[key], cmap="coolwarm",
                                vmin=-40, vmax=20, shading="auto")
        axis.set(title=title, xlabel="Time (s)",
                 ylabel="Frequency (Hz)" if index == 0 else "")
        lower.append(axis)
    figure.colorbar(image, ax=lower, label="Event / true ocean (dB)", shrink=.8)
    figure.suptitle("Current 128-group residual models · unchanged 5-s held-out procedural test · $-12$ dB", fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    preview = output.with_name(output.stem + "_preview")
    subprocess.run(["pdftoppm", "-f", "1", "-l", "1", "-r", "160", "-png",
                    "-singlefile", str(output), str(preview)], check=True)
    report = {"example_id": item["example_id"], "sample_index": chosen,
              "selection": "nearest to joint median SI-SDRi among 96 fixed test examples",
              "si_sdri_db": gains, "median_event_to_ocean_db_where_gt_below_ocean_minus_20db": leakage,
              "p90_event_to_ocean_db_where_gt_below_ocean_minus_20db": leakage_p90,
              "fraction_above_minus10db_where_gt_below_ocean_minus_20db": leakage_above_minus10,
              "cohort_96_median_absent_event_to_ocean_db": {
                  name: float(np.median(values)) for name, values in cohort_leakage.items()},
              "figure": str(output), "checkpoint_paths": [str(persistence_path), str(forecast_path)]}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("paper/figures/real_audio_visualization.pdf"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--persistence", type=Path, default=Path("models/empirical_persistence_residual_128.pt"))
    parser.add_argument("--forecast", type=Path, default=Path("models/empirical_forecast_residual_128.pt"))
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.device, args.persistence, args.forecast), indent=2))


if __name__ == "__main__":
    main()
