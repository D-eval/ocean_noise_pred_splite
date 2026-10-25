"""Full-waveform current-background oracle as an information-sufficiency check.

The frozen mixture-only BS-RoFormer supplies one waveform. A three-coefficient
head combines that output, the mixture, and a background-condition waveform.
The same head is fitted with zero, wrong-time, and actual current background.
Least squares gives the global optimum for this head, so optimisation length
cannot explain a failure to copy the answer. The actual background is available
only in this controlled full-reference experiment, never at deployment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dclde import CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE
from .dclde_separation_experiment import (
    CausalFeatureStore, ExperimentConfig, InjectedUpcallDataset, _make_model,
)
from .metrics import paired_bootstrap_interval, si_sdr


CONDITIONS = ("zero", "shuffled", "current")
BASELINE_CSV = Path("experiments/dclde_true_bsr/clean_synthetic_test.csv")
BASELINE_CHECKPOINT = Path("experiments/dclde_true_bsr/best.pt")


class FullBackgroundDataset(InjectedUpcallDataset):
    """Extend the exact DCLDE tuple with a different clean waveform in its split."""

    def __getitem__(self, item: int) -> dict:
        value = super().__getitem__(item)
        local = (
            round(item * (len(self.rows) - 1) / max(1, self.examples - 1))
            if self.spread_rows and self.examples > 1 else item % len(self.rows)
        )
        wrong_local = (local + len(self.rows) // 2) % len(self.rows)
        wrong = self.base[wrong_local]["waveform"].numpy()
        value["wrong_ocean"] = torch.from_numpy(wrong.astype(np.float32))
        return value


def _features(base: torch.Tensor, mixture: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    return torch.stack((base, mixture, condition), dim=-1)


@torch.inference_mode()
def fit_heads(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, torch.Tensor]:
    """Solve the supervised waveform-MSE optimum from all training samples."""
    gram = {name: torch.zeros((3, 3), dtype=torch.float64) for name in CONDITIONS}
    cross = {name: torch.zeros(3, dtype=torch.float64) for name in CONDITIONS}
    examples = 0
    for batch in loader:
        mixture = batch["mixture"].to(device)
        source = batch["source"].to(device)
        ocean = batch["ocean"].to(device)
        history = batch["history"].to(device)
        base = model(mixture, history)["estimate"]
        length = base.shape[-1]
        mixture, source, ocean = mixture[..., :length], source[..., :length], ocean[..., :length]
        conditions = {"zero": torch.zeros_like(ocean),
                      "shuffled": batch["wrong_ocean"].to(device)[..., :length], "current": ocean}
        for name, condition in conditions.items():
            design = _features(base, mixture, condition).reshape(-1, 3).to(torch.float64)
            target = source.reshape(-1).to(torch.float64)
            gram[name] += (design.T @ design).cpu()
            cross[name] += (design.T @ target).cpu()
        examples += len(mixture)
    if examples != len(loader.dataset):
        raise ValueError("training pass did not cover every requested example")
    # pinv handles the intentionally rank-deficient zero-condition column.
    return {name: torch.linalg.pinv(gram[name]) @ cross[name] for name in CONDITIONS}


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module, loader: DataLoader,
    weights: dict[str, torch.Tensor], device: torch.device,
) -> list[dict]:
    rows = []
    coefficients = {name: value.to(device=device, dtype=torch.float32)
                    for name, value in weights.items()}
    for batch in loader:
        mixture = batch["mixture"].to(device)
        source = batch["source"].to(device)
        ocean = batch["ocean"].to(device)
        history = batch["history"].to(device)
        base = model(mixture, history)["estimate"]
        length = base.shape[-1]
        mixture, source, ocean = mixture[..., :length], source[..., :length], ocean[..., :length]
        wrong = batch["wrong_ocean"].to(device)[..., :length]
        mixed_score = si_sdr(mixture, source)
        estimates = {
            "bsr": base,
            "zero": _features(base, mixture, torch.zeros_like(ocean)) @ coefficients["zero"],
            "shuffled": _features(base, mixture, wrong) @ coefficients["shuffled"],
            "current": _features(base, mixture, ocean) @ coefficients["current"],
            "analytic": mixture - ocean,
        }
        gains = {name: (si_sdr(estimate, source) - mixed_score).cpu().numpy()
                 for name, estimate in estimates.items()}
        analytic_error = (estimates["analytic"] - source).abs().amax(dim=-1).cpu().numpy()
        for index, example_id in enumerate(batch["example_id"]):
            rows.append({
                "example_id": example_id, "day": example_id.split(":", 1)[0],
                "snr_db": float(batch["snr_db"][index]),
                "mixture_si_sdr_db": float(mixed_score[index]),
                **{f"{name}_si_sdri_db": float(score[index]) for name, score in gains.items()},
                "analytic_max_abs_error": float(analytic_error[index]),
            })
    return rows


def _check_test_identity(rows: list[dict]) -> None:
    with BASELINE_CSV.open(newline="", encoding="utf-8") as stream:
        reference = list(csv.DictReader(stream))
    if len(rows) != len(reference) or len(rows) != 4096:
        raise ValueError("expected the same 4,096 main-domain test mixtures")
    for index, (left, right) in enumerate(zip(rows, reference, strict=True)):
        if left["example_id"] != right["example_id"]:
            raise ValueError(f"test ID differs at row {index}")
        if abs(left["snr_db"] - float(right["snr_db"])) > 1e-5:
            raise ValueError(f"test SNR differs at row {index}")
        if abs(left["mixture_si_sdr_db"] - float(right["mixture_si_sdr_db"])) > 1e-4:
            raise ValueError(f"test mixture differs at row {index}")
        if abs(left["bsr_si_sdri_db"] - float(right["source_si_sdri_db"])) > 1e-3:
            raise ValueError(f"frozen BSR output differs at row {index}")


def _summary(rows: list[dict], baseline: str = "bsr") -> dict:
    days = np.asarray([row["day"] for row in rows])
    reference = np.asarray([row[f"{baseline}_si_sdri_db"] for row in rows])
    answer = {}
    for name in ("bsr", *CONDITIONS, "analytic"):
        score = np.asarray([row[f"{name}_si_sdri_db"] for row in rows])
        difference = score - reference
        day_means = np.asarray([difference[days == day].mean() for day in np.unique(days)])
        answer[name] = {
            "mean_si_sdri_db": float(score.mean()),
            "delta_vs_bsr_db": float(difference.mean()),
            "delta_vs_bsr_day_ci95_db": [float(x) for x in paired_bootstrap_interval(day_means)],
        }
    return answer


def run(output: Path, device_name: str, train_examples: int = 8192) -> dict:
    config = ExperimentConfig(device=device_name)
    device = torch.device(device_name)
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    model = _make_model("bsr", store.bands, config.seed, device)
    model.load_state_dict(torch.load(BASELINE_CHECKPOINT, map_location=device, weights_only=False)["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    datasets = {
        "train": FullBackgroundDataset(store, "train", train_examples, config.seed, spread_rows=True),
        "validation": FullBackgroundDataset(store, "validation", config.validation_examples,
                                            config.seed + 10_000_000, cache_waveforms=True),
        "test": FullBackgroundDataset(store, "test", config.validation_examples,
                                      config.seed + 20_000_000, spread_rows=True),
    }
    loaders = {name: DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0)
               for name, dataset in datasets.items()}
    weights = fit_heads(model, loaders["train"], device)
    print(json.dumps({"stage": "fit", "train_examples": train_examples,
                      "coefficients": {key: value.tolist() for key, value in weights.items()}}), flush=True)
    validation_rows = evaluate(model, loaders["validation"], weights, device)
    test_rows = evaluate(model, loaders["test"], weights, device)
    _check_test_identity(test_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_path = output.with_name(output.stem + "_samples.csv")
    with sample_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    result = {
        "protocol": {
            "dataset": "same annotation-clean DCLDE 5-s mixture test as dclde_main_oracle.json",
            "train_examples": train_examples, "validation_examples": len(validation_rows),
            "test_examples": len(test_rows), "held_out_days": sorted(set(row["day"] for row in test_rows)),
            "model": "frozen main-domain BSR plus a three-coefficient full-waveform fusion head",
            "head_input": "[BSR(x), x, condition waveform] at every audio sample",
            "initial_coefficients": [1.0, 0.0, 0.0],
            "fit": "global waveform-MSE least-squares optimum on training split; no test fitting",
            "conditions": "zero; half-cycle-shifted wrong-time clean waveform; actual current clean N_t",
            "analytic": "x_t - N_t, exact only in the controlled synthetic-mixture task",
            "interpretation": "information and copy-path sanity check; not an architecture-matched FiLM ablation or a deployable predictor",
            "interval": "paired 95% bootstrap over the three held-out recording-day means",
        },
        "coefficients": {key: value.tolist() for key, value in weights.items()},
        "validation": _summary(validation_rows), "test": _summary(test_rows),
        "analytic_max_abs_error": float(max(row["analytic_max_abs_error"] for row in test_rows)),
        "samples_csv": str(sample_path),
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-examples", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=Path("results/dclde_full_background_oracle.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.device, args.train_examples), indent=2), flush=True)


if __name__ == "__main__":
    main()
