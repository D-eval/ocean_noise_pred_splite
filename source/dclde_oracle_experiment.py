"""Six-way, matched-condition experiment on annotation-clean DCLDE audio.

Only Oracle receives the current event-free background spectrum E(N_t).
Every other condition is available without the current clean background, except
Shuffled, which receives E(N_j) from a different window in the same split.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .dclde import CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE
from .dclde_separation_experiment import CausalFeatureStore, ExperimentConfig, _make_model, train_separator
from .metrics import paired_bootstrap_interval


METHODS = (
    ("bsr", "BS-RoFormer", Path("experiments/dclde_true_bsr")),
    ("zero_film", "Zero-condition FiLM", Path("experiments/dclde_zero_film")),
    ("shuffled_film", "Shuffled-condition FiLM", Path("experiments/dclde_shuffled_film")),
    ("persistence_film", "Persistence-condition FiLM", Path("experiments/dclde_persistence_film")),
    ("film", "GRU-Forecast FiLM", Path("experiments/dclde_fc_bsr")),
    ("oracle_film", "Oracle-FiLM", Path("experiments/dclde_oracle_film")),
)
TRAIN_VARIANTS = {"bsr", "shuffled_film", "persistence_film", "oracle_film"}


def materialize_exact_zero_control(store: CausalFeatureStore, config: ExperimentConfig) -> None:
    """Attach the frozen GRU to the previously trained, equivalent zero-FiLM separator.

    The GRU is deliberately not called by `zero_film`, so its presence changes
    neither the old `none_film` waveform mapping nor any training gradient.
    """
    source = Path("experiments/dclde_bsr")
    destination = Path("experiments/dclde_zero_film")
    if all((destination / name).exists() for name in
           ("best.pt", "training.json", "result.json", "clean_synthetic_test.csv")):
        return
    old = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
    forecast = torch.load("experiments/dclde_forecaster/best.pt", map_location="cpu", weights_only=False)
    old_model = _make_model("none_film", store.bands, config.seed, torch.device("cpu"))
    old_model.load_state_dict(old["model"])
    zero = _make_model("zero_film", store.bands, config.seed, torch.device("cpu"))
    missing, unexpected = zero.load_state_dict(old["model"], strict=False)
    if unexpected or any(not key.startswith("forecaster.") for key in missing):
        raise ValueError("zero-FiLM separator weights are not compatible with the old checkpoint")
    zero.forecaster.load_state_dict(forecast["model"])
    for parameter in zero.forecaster.parameters():
        parameter.requires_grad_(False)
    old_model.eval()
    zero.eval()
    probe = torch.randn(1, 10000, generator=torch.Generator().manual_seed(91)) * 0.01
    history = torch.zeros(1, 12, store.bands)
    with torch.inference_mode():
        difference = (old_model(probe, history)["estimate"] -
                      zero(probe, history)["estimate"]).abs().max().item()
    if difference > 1e-6:
        raise ValueError(f"zero-FiLM equivalence failed: {difference:g}")
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({"model": zero.state_dict(), "config": old["config"],
                "variant": "zero_film", "step": old["step"]}, destination / "best.pt")
    for name in ("training.json", "clean_synthetic_test.csv"):
        shutil.copy2(source / name, destination / name)
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    result.update({
        "variant": "zero_film", "checkpoint": str(destination / "best.pt"),
        "equivalence": "same trained none_film separator; identical outputs; frozen forecast GRU attached but bypassed; c=0",
        "max_equivalence_error": difference,
    })
    (destination / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _blocked_interval(values: np.ndarray, days: np.ndarray) -> list[float]:
    day_means = np.asarray([values[days == day].mean() for day in np.unique(days)])
    return [float(value) for value in paired_bootstrap_interval(day_means)]


def summarize(output: Path = Path("results/dclde_main_oracle.json")) -> dict:
    sample_rows = {variant: _rows(folder / "clean_synthetic_test.csv") for variant, _, folder in METHODS}
    reference = sample_rows["bsr"]
    identifiers = [row["example_id"] for row in reference]
    if len(identifiers) != 4096 or len(set(identifiers)) != len(identifiers):
        raise ValueError("expected 4,096 unique held-out mixtures")
    days = np.asarray([row["day"] for row in reference])
    reference_snr = np.asarray([float(row["snr_db"]) for row in reference])
    mixture_scores = np.asarray([float(row["mixture_si_sdr_db"]) for row in reference])
    values = {}
    for variant, _, _ in METHODS:
        rows = sample_rows[variant]
        other_ids = [row["example_id"] for row in rows]
        if other_ids != identifiers:
            first = next((index for index, (left, right) in enumerate(zip(identifiers, other_ids))
                          if left != right), min(len(identifiers), len(other_ids)))
            raise ValueError(
                f"{variant} test examples differ at row {first}: "
                f"reference={identifiers[first] if first < len(identifiers) else '<missing>'}, "
                f"variant={other_ids[first] if first < len(other_ids) else '<missing>'}"
            )
        if not np.allclose([float(row["snr_db"]) for row in rows], reference_snr, atol=1e-6):
            raise ValueError(f"{variant} injected SNRs do not match")
        if not np.allclose([float(row["mixture_si_sdr_db"]) for row in rows], mixture_scores, atol=1e-5):
            raise ValueError(f"{variant} mixtures do not match")
        values[variant] = np.asarray([float(row["source_si_sdri_db"]) for row in rows])
    methods = []
    for variant, label, folder in METHODS:
        rows = sample_rows[variant]
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        history = json.loads((folder / "training.json").read_text(encoding="utf-8"))
        selected = next(row for row in history if row["step"] == result["best_step"])
        score = values[variant]
        vs_bsr = score - values["bsr"]
        vs_zero = score - values["zero_film"]
        methods.append({
            "variant": variant, "method": label, "checkpoint": str(folder / "best.pt"),
            "best_step": result["best_step"], "updates_run": result.get("updates_run", history[-1]["step"]),
            "best_validation_si_sdri_db": float(selected["validation_si_sdri_db"]),
            "mean_si_sdri_db": float(score.mean()),
            "mean_target_si_sdr_db": float(np.mean([float(row["source_si_sdr_db"]) for row in rows])),
            "mean_background_si_sdr_db": float(np.mean([float(row["background_si_sdr_db"]) for row in rows])),
            "delta_vs_bsr_db": float(vs_bsr.mean()),
            "delta_vs_bsr_day_ci95_db": _blocked_interval(vs_bsr, days),
            "delta_vs_zero_db": float(vs_zero.mean()),
            "delta_vs_zero_day_ci95_db": _blocked_interval(vs_zero, days),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": {
            "dataset": "DCLDE annotation-clean current 5-s real ocean WAV plus synthetic marine event",
            "examples": len(identifiers), "held_out_days": sorted(set(days)),
            "history": "12 immediately preceding event-free 5-s, 30-band features",
            "oracle": "30-band E(N_t) from the actual clean background waveform in each tuple",
            "shuffled": "E(N_j) from a half-cycle-shifted, different clean window within the same split",
            "persistence": "last historical 5-s spectrum S_(t-1)",
            "zero": "all-zero 30-band condition with FiLM parameters retained",
            "baseline": "band-split injector bypassed; no conditioning transformation",
            "training": "same waveform mixtures, seed, transferred separator initialization, optimizer, 10k-step maximum, 5k-step minimum, 100-step validation, patience 7",
            "interval": "paired 95% bootstrap over held-out recording day means",
        },
        "methods": methods,
        "decomposition": {
            name: {
                "mean_delta_db": float((values[left] - values[right]).mean()),
                "day_block_ci95_db": _blocked_interval(values[left] - values[right], days),
            }
            for name, left, right in (
                ("architecture_zero_minus_bsr", "zero_film", "bsr"),
                ("shuffled_information_minus_zero", "shuffled_film", "zero_film"),
                ("persistence_information_minus_zero", "persistence_film", "zero_film"),
                ("forecast_information_minus_zero", "film", "zero_film"),
                ("oracle_information_minus_zero", "oracle_film", "zero_film"),
                ("oracle_minus_forecast", "oracle_film", "film"),
            )
        },
    }
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[key for key in methods[0] if key != "checkpoint"])
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "checkpoint"} for row in methods)
    sample_path = output.with_name(output.stem + "_samples.csv")
    with sample_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["example_id", "day", "source_family", "snr_db", "mixture_si_sdr_db"]
        fields.extend(variant for variant, _, _ in METHODS)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(reference):
            writer.writerow({
                "example_id": row["example_id"], "day": row["day"],
                "source_family": row["source_family"], "snr_db": row["snr_db"],
                "mixture_si_sdr_db": row["mixture_si_sdr_db"],
                **{variant: float(values[variant][index]) for variant, _, _ in METHODS},
            })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    config = ExperimentConfig(device=args.device)
    materialize_exact_zero_control(store, config)
    if not args.summary_only:
        for variant, label, folder in METHODS:
            if variant not in TRAIN_VARIANTS:
                continue
            if (folder / "best.pt").exists() and (folder / "clean_synthetic_test.csv").exists():
                print(json.dumps({"method": label, "status": "existing result"}), flush=True)
                continue
            print(json.dumps({"method": label, "status": "training"}), flush=True)
            train_separator(config, store, variant, folder)
    print(json.dumps(summarize(), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
