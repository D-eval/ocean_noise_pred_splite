"""Oracle information ladder on the exact DCLDE main-domain mixtures.

The same frozen BSR and three-coefficient copy head are used at every rung.
Current-background waveform information is reduced in frequency, time, and
phase. This audits this particular copy interface; it is not a proof that
another network cannot use a compressed representation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dclde import CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE, SAMPLE_RATE
from .dclde_full_oracle_experiment import BASELINE_CHECKPOINT, _check_test_identity
from .dclde_separation_experiment import CausalFeatureStore, ExperimentConfig, InjectedUpcallDataset, _make_model
from .metrics import paired_bootstrap_interval, si_sdr


N_FFT = 512
HOP = 128
TIME_POOLS_SECONDS = (0.064, 0.25, 1.0, 5.0)
RUNGS = (
    "raw", "full_true", "full_mix",
    "full_250ms_true", "full_1s_true", "full_5s_true",
    "full_250ms_mix", "full_1s_mix", "full_5s_mix",
    "l128_frame_true", "l64_frame_true", "l32_frame_true",
    "l128_frame_mix", "l64_frame_mix", "l32_frame_mix",
    "b128_frame_true", "b64_frame_true", "b48_frame_true",
    "b128_frame_mix", "b64_frame_mix", "b48_frame_mix",
    "b30_frame_true", "b30_250ms_true",
    "b30_1s_true", "b30_5s_true", "b30_frame_mix", "b30_250ms_mix",
    "b30_1s_mix", "b30_5s_mix",
)


def _band_ids(device: torch.device, bands: int) -> torch.Tensor:
    frequencies = torch.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE, device=device)
    edges = torch.logspace(np.log10(20.0), np.log10(950.0), bands + 1, device=device)
    # Frequencies outside 20--950 Hz join the nearest edge band; every bin is
    # assigned to one group at every resolution.
    return torch.bucketize(frequencies, edges[1:-1]).long()


def _band_power(power: torch.Tensor, band_ids: torch.Tensor, bands: int) -> torch.Tensor:
    batch, _, frames = power.shape
    sums = power.new_zeros((batch, bands, frames))
    sums.scatter_add_(1, band_ids[None, :, None].expand(batch, -1, frames), power)
    counts = torch.bincount(band_ids, minlength=bands).to(power).clamp_min(1)
    return sums / counts[None, :, None]


def _time_pool(power: torch.Tensor, seconds: float) -> torch.Tensor:
    frames = power.shape[-1]
    samples_per_group = frames if seconds >= 5.0 else max(1, round(seconds * SAMPLE_RATE / HOP))
    if samples_per_group == 1:
        return power
    group_ids = torch.arange(frames, device=power.device) // samples_per_group
    groups = int(group_ids[-1]) + 1
    sums = power.new_zeros((*power.shape[:2], groups))
    sums.scatter_add_(2, group_ids[None, None, :].expand(power.shape[0], power.shape[1], -1), power)
    counts = torch.bincount(group_ids, minlength=groups).to(power)
    return (sums / counts[None, None, :]).index_select(2, group_ids)


@torch.inference_mode()
def oracle_conditions(ocean: torch.Tensor, mixture: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reconstruct progressively compressed N using true or mixture phase."""
    if ocean.shape != mixture.shape or ocean.ndim != 2:
        raise ValueError("expected matched [batch, audio-samples] tensors")
    length = ocean.shape[-1]
    window = torch.hann_window(N_FFT, device=ocean.device, dtype=ocean.dtype)
    n_spec = torch.stft(ocean, N_FFT, hop_length=HOP, window=window, return_complex=True)
    x_spec = torch.stft(mixture, N_FFT, hop_length=HOP, window=window, return_complex=True)
    n_phase = torch.polar(torch.ones_like(n_spec.real), torch.angle(n_spec))
    x_phase = torch.polar(torch.ones_like(x_spec.real), torch.angle(x_spec))

    def invert(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return torch.istft(magnitude * phase, N_FFT, hop_length=HOP,
                           window=window, length=length)

    out = {
        "raw": ocean,
        "full_true": invert(n_spec.abs(), n_phase),
        "full_mix": invert(n_spec.abs(), x_phase),
    }
    n_power = n_spec.abs().square()
    for label, seconds in (("250ms", 0.25), ("1s", 1.0), ("5s", 5.0)):
        magnitude = _time_pool(n_power, seconds).clamp_min(0).sqrt()
        out[f"full_{label}_true"] = invert(magnitude, n_phase)
        out[f"full_{label}_mix"] = invert(magnitude, x_phase)
    for bands in (128, 64, 32):
        ids = torch.arange(n_power.shape[1], device=ocean.device) * bands // n_power.shape[1]
        magnitude = _band_power(n_power, ids, bands).index_select(1, ids).clamp_min(0).sqrt()
        out[f"l{bands}_frame_true"] = invert(magnitude, n_phase)
        out[f"l{bands}_frame_mix"] = invert(magnitude, x_phase)
    for bands in (128, 64, 48):
        ids = _band_ids(ocean.device, bands)
        magnitude = _band_power(n_power, ids, bands).index_select(1, ids).clamp_min(0).sqrt()
        out[f"b{bands}_frame_true"] = invert(magnitude, n_phase)
        out[f"b{bands}_frame_mix"] = invert(magnitude, x_phase)
    band_ids = _band_ids(ocean.device, 30)
    band_power = _band_power(n_power, band_ids, 30)
    for label, seconds in zip(("frame", "250ms", "1s", "5s"), TIME_POOLS_SECONDS, strict=True):
        power = _time_pool(band_power, seconds)
        magnitude = power.index_select(1, band_ids).clamp_min(0).sqrt()
        out[f"b30_{label}_true"] = invert(magnitude, n_phase)
        out[f"b30_{label}_mix"] = invert(magnitude, x_phase)
    return out


@torch.inference_mode()
def fit(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, torch.Tensor]:
    gram = {name: torch.zeros((3, 3), dtype=torch.float64) for name in RUNGS}
    cross = {name: torch.zeros(3, dtype=torch.float64) for name in RUNGS}
    for batch in loader:
        mixture = batch["mixture"].to(device)
        source = batch["source"].to(device)
        ocean = batch["ocean"].to(device)
        base = model(mixture, batch["history"].to(device))["estimate"]
        length = base.shape[-1]
        mixture, source, ocean = mixture[..., :length], source[..., :length], ocean[..., :length]
        conditions = oracle_conditions(ocean, mixture)
        for name in RUNGS:
            design = torch.stack((base, mixture, conditions[name]), dim=-1).reshape(-1, 3).double()
            target = source.reshape(-1).double()
            gram[name] += (design.T @ design).cpu()
            cross[name] += (design.T @ target).cpu()
    return {name: torch.linalg.pinv(gram[name]) @ cross[name] for name in RUNGS}


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, weights: dict[str, torch.Tensor],
             device: torch.device) -> list[dict]:
    coefficients = {name: value.to(device=device, dtype=torch.float32) for name, value in weights.items()}
    rows = []
    for batch in loader:
        mixture = batch["mixture"].to(device)
        source = batch["source"].to(device)
        ocean = batch["ocean"].to(device)
        base = model(mixture, batch["history"].to(device))["estimate"]
        length = base.shape[-1]
        mixture, source, ocean = mixture[..., :length], source[..., :length], ocean[..., :length]
        conditions = oracle_conditions(ocean, mixture)
        mixture_score = si_sdr(mixture, source)
        base_gain = si_sdr(base, source) - mixture_score
        result = {}
        for name in RUNGS:
            estimate = (torch.stack((base, mixture, conditions[name]), dim=-1) @ coefficients[name])
            result[f"{name}_si_sdri_db"] = (si_sdr(estimate, source) - mixture_score).cpu().numpy()
            squared_error = (conditions[name] - ocean).square().sum(dim=-1)
            ocean_power = ocean.square().sum(dim=-1).clamp_min(1e-12)
            result[f"{name}_condition_snr_db"] = (10 * torch.log10(
                ocean_power / squared_error.clamp_min(1e-12))).cpu().numpy()
        for index, example_id in enumerate(batch["example_id"]):
            rows.append({
                "example_id": example_id, "day": example_id.split(":", 1)[0],
                "snr_db": float(batch["snr_db"][index]),
                "mixture_si_sdr_db": float(mixture_score[index]),
                "bsr_si_sdri_db": float(base_gain[index]),
                **{key: float(value[index]) for key, value in result.items()},
            })
    return rows


def summarize(rows: list[dict]) -> dict:
    days = np.asarray([row["day"] for row in rows])
    baseline = np.asarray([row["bsr_si_sdri_db"] for row in rows])
    out = {"bsr_mean_si_sdri_db": float(baseline.mean())}
    for name in RUNGS:
        scores = np.asarray([row[f"{name}_si_sdri_db"] for row in rows])
        delta = scores - baseline
        day_mean = np.asarray([delta[days == day].mean() for day in np.unique(days)])
        out[name] = {
            "mean_si_sdri_db": float(scores.mean()),
            "delta_vs_bsr_db": float(delta.mean()),
            "delta_vs_bsr_day_ci95_db": [float(x) for x in paired_bootstrap_interval(day_mean)],
            "mean_condition_snr_db": float(np.mean([row[f"{name}_condition_snr_db"] for row in rows])),
        }
    return out


def run(output: Path, device_name: str, train_examples: int) -> dict:
    config = ExperimentConfig(device=device_name)
    device = torch.device(device_name)
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    model = _make_model("bsr", store.bands, config.seed, device)
    model.load_state_dict(torch.load(BASELINE_CHECKPOINT, map_location=device, weights_only=False)["model"])
    model.eval()
    model.requires_grad_(False)
    train = InjectedUpcallDataset(store, "train", train_examples, config.seed, spread_rows=True)
    validation = InjectedUpcallDataset(store, "validation", 4096, config.seed + 10_000_000, cache_waveforms=True)
    test = InjectedUpcallDataset(store, "test", 4096, config.seed + 20_000_000, spread_rows=True)
    weights = fit(model, DataLoader(train, batch_size=16, num_workers=0), device)
    print(json.dumps({"stage": "fit", "train_examples": train_examples,
                      "raw_coefficients": weights["raw"].tolist()}), flush=True)
    validation_rows = evaluate(model, DataLoader(validation, batch_size=16, num_workers=0), weights, device)
    test_rows = evaluate(model, DataLoader(test, batch_size=16, num_workers=0), weights, device)
    _check_test_identity(test_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_path = output.with_name(output.stem + "_samples.csv")
    with sample_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    result = {
        "protocol": {
            "dataset": "same 4096 DCLDE main test mixtures, per-example matched to frozen BSR",
            "training_examples": train_examples, "validation_examples": 4096, "test_examples": 4096,
            "model": "one frozen BSR, one least-squares three-coefficient head per representation",
            "head": "w_b BSR(x) + w_x x + w_c reconstructed-condition-waveform",
            "transform": "512-point Hann STFT, 128-sample/64-ms hop, 2-kHz audio; 128/64/48/30 geometric 20--950-Hz bands with edge bins absorbed",
            "equal_width_control": "128/64/32 contiguous equal-width frequency-bin groups on the same 257-bin STFT",
            "time": "per-frame, 0.25-s, 1-s, or full-window pooled power, with or without frequency compression; all reconstructed with overlap-add",
            "phase": "true current N phase or mixture x phase, on an identical STFT grid",
            "note": "coarsest 30-band/5-s power is analogous to, but not numerically identical to, published E(N_t)",
            "interpretation": "fixed copy-interface sufficiency diagnostic, not a matched FiLM or information-theoretic limit",
            "interval": "paired bootstrap over three held-out recording-day means",
        },
        "rungs": list(RUNGS),
        "coefficients": {name: value.tolist() for name, value in weights.items()},
        "validation": summarize(validation_rows), "test": summarize(test_rows),
        "samples_csv": str(sample_path),
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-examples", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=Path("results/dclde_information_ladder.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.device, args.train_examples), indent=2), flush=True)


if __name__ == "__main__":
    main()
