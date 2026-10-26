"""Three independent separator seeds on one fixed DCLDE test set.

The published 20261022 run is retained as seed 1. Seeds 20261023 and
20261024 retrain all five main arms from the same transferred backbone.
The 128-band predictor is frozen, and all arms use 6000 updates, the same
per-seed training mixtures, validation selection, and held-out test mixtures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dclde import CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE
from .dclde_causal_highres_experiment import (
    ConditionedInjectedDataset, CausalResidualBSR, HighResolutionStore,
    separation_validation,
)
from .dclde_separation_experiment import (
    CausalFeatureStore, ExperimentConfig, InjectedUpcallDataset, _full_reference_test,
    _load_transfer, _make_model, _seed_everything, train_separator,
)
from .metrics import si_sdr
from .model import multi_resolution_loss


SEEDS = (20261022, 20261023, 20261024)
ARMS = ("bsr", "zero", "shuffled", "persistence", "forecast")
OUTPUT = Path("experiments/dclde_multiseed_main")
SUMMARY = Path("results/dclde_multiseed_main.json")
TEST_SEED = SEEDS[0] + 20_000_000
STEPS = 6000


class _MemoizedDataset(Dataset):
    """Avoid regenerating identical deterministic validation examples 60 times."""

    def __init__(self, base: Dataset):
        self.base = base
        self.cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> dict:
        if item not in self.cache:
            self.cache[item] = self.base[item]
        return self.cache[item]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def _test_residual(model: CausalResidualBSR, dataset: ConditionedInjectedDataset,
                   device: torch.device, arm: str) -> list[dict]:
    model.eval()
    rows = []
    for batch in DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0):
        mixture = batch["mixture"].to(device)
        source = batch["source"].to(device)
        estimate = model(mixture, batch["condition_map"].to(device))
        length = min(mixture.shape[-1], source.shape[-1], estimate.shape[-1])
        mixture, source, estimate = mixture[..., :length], source[..., :length], estimate[..., :length]
        mixed_score = si_sdr(mixture, source)
        improvement = si_sdr(estimate, source) - mixed_score
        for index, example_id in enumerate(batch["example_id"]):
            row = {"example_id": example_id, "day": example_id.split(":", 1)[0],
                   "snr_db": float(batch["snr_db"][index]),
                   "mixture_si_sdr_db": float(mixed_score[index]),
                   "source_si_sdri_db": float(improvement[index])}
            if arm == "shuffled":
                row["condition_day"] = batch["condition_day"][index]
                if row["condition_day"] == row["day"]:
                    raise ValueError("shuffled donor shares the test day")
            rows.append(row)
    return rows


def _train_bsr(store: CausalFeatureStore, seed: int, device_name: str) -> None:
    output = OUTPUT / str(seed) / "bsr"
    if (output / "result.json").exists() and (output / "best.pt").exists():
        return
    config = ExperimentConfig(seed=seed, device=device_name,
                              separation_minimum_steps=STEPS, patience=0)
    result = train_separator(config, store, "bsr", output,
                             cache_training_waveforms=True,
                             memoize_validation_examples=True)
    if result["updates_run"] != STEPS:
        raise ValueError("BSR update count differs from residual arms")


def _train_residual(cache: HighResolutionStore, seed: int, arm: str,
                    device_name: str) -> None:
    output = OUTPUT / str(seed) / arm
    if (output / "result.json").exists() and (output / "best.pt").exists():
        return
    config = ExperimentConfig(seed=seed, device=device_name)
    _seed_everything(seed)
    device = torch.device(device_name)
    model = CausalResidualBSR(cache.store.bands, seed, device,
                              Path(config.transfer_checkpoint)).to(device)
    train = ConditionedInjectedDataset(cache, "train", config.train_examples, seed, arm,
                                       cache_waveforms=True)
    validation = _MemoizedDataset(ConditionedInjectedDataset(
        cache, "validation", config.validation_examples, seed + 10_000_000,
        arm, cache_waveforms=True,
    ))
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    validation_loader = DataLoader(validation, batch_size=24, shuffle=False, num_workers=0)
    ordinary = _make_model("bsr", cache.store.bands, seed, device)
    transferred = _load_transfer(ordinary, Path(config.transfer_checkpoint))
    model.eval()
    ordinary.eval()
    probe = next(iter(DataLoader(validation, batch_size=2)))
    with torch.inference_mode():
        mixture = probe["mixture"].to(device)
        delta = (model(mixture, probe["condition_map"].to(device)) -
                 ordinary(mixture, None)["estimate"]).abs().max().item()
    del ordinary
    if delta > 1e-6 or transferred != model.transferred_tensors:
        raise ValueError(f"unmatched initialization: {delta}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.separation_steps)
    output.mkdir(parents=True, exist_ok=True)
    training_rows, best, best_step = [], -math.inf, 0
    iterator = iter(train_loader)
    for step in range(1, STEPS + 1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        mixture, source, condition_map = (batch[key].to(device) for key in
                                          ("mixture", "source", "condition_map"))
        loss = multi_resolution_loss(model(mixture, condition_map), source)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        scheduler.step()
        if step == 1 or step % config.separation_validate_every == 0:
            row = {"step": step, "train_loss": float(loss.detach())}
            if step % config.separation_validate_every == 0:
                score = separation_validation(model, validation_loader, device)
                row["validation_si_sdri_db"] = score
                if score > best + 1e-4:
                    best, best_step = score, step
                    torch.save({"model": model.state_dict(), "config": vars(config),
                                "step": step, "interface": f"{arm}_residual_128"},
                               output / "best.pt")
            training_rows.append(row)
            print(json.dumps({"seed": seed, "arm": arm, **row}), flush=True)
            (output / "training.json").write_text(json.dumps(training_rows, indent=2), encoding="utf-8")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = ConditionedInjectedDataset(cache, "test", config.validation_examples,
                                      TEST_SEED, arm, spread_rows=True)
    rows = _test_residual(model, test, device, arm)
    _write_csv(output / "test.csv", rows)
    result = {"seed": seed, "arm": arm, "updates_run": STEPS,
              "best_step": best_step, "best_validation_si_sdri_db": best,
              "mean_si_sdri_db": float(np.mean([r["source_si_sdri_db"] for r in rows])),
              "test_examples": len(rows), "test_seed": TEST_SEED,
              "initial_waveform_max_difference": delta,
              "transferred_tensors": transferred}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def _get_rows(seed: int, arm: str) -> list[dict]:
    if seed == SEEDS[0]:
        path = (Path("experiments/dclde_true_bsr/clean_synthetic_test.csv") if arm == "bsr"
                else Path(f"experiments/dclde_{arm}_residual_128/clean_synthetic_test.csv"))
    else:
        path = OUTPUT / str(seed) / arm / ("clean_synthetic_test.csv" if arm == "bsr" else "test.csv")
        if arm == "bsr":
            # The training routine evaluates its own-seed test draw. A separate
            # fixed-draw evaluation is written by _fixed_bsr_test below.
            path = OUTPUT / str(seed) / arm / "test_fixed.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def _fixed_bsr_test(store: CausalFeatureStore, seed: int, device_name: str) -> None:
    output = OUTPUT / str(seed) / "bsr" / "test_fixed.csv"
    if output.exists():
        return
    device = torch.device(device_name)
    model = _make_model("bsr", store.bands, seed, device)
    checkpoint = torch.load(output.parent / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    test = InjectedUpcallDataset(store, "test", 4096, TEST_SEED, spread_rows=True)
    _, rows = _full_reference_test(model, test, 24, device)
    _write_csv(output, rows)


def summarize() -> dict:
    samples = {(seed, arm): _get_rows(seed, arm) for seed in SEEDS for arm in ARMS}
    reference = samples[(SEEDS[0], "bsr")]
    ids = [row["example_id"] for row in reference]
    if len(ids) != 4096:
        raise ValueError("expected 4096 fixed test mixtures")
    result = {"seeds": SEEDS, "arms": ARMS, "test_seed": TEST_SEED,
              "test_examples": len(ids), "per_seed": {}, "contrasts": {},
              "contrast_by_seed_day": {}, "checkpoint_sha256": {},
              "fixed_test_csv_sha256": {},
              "protocol": "6000 updates/arm; 4096 validation mixtures every 100 steps; best validation SI-SDRi; common 4096 test mixtures; frozen original high-resolution GRU across separator seeds"}
    for seed in SEEDS:
        result["per_seed"][str(seed)] = {}
        result["checkpoint_sha256"][str(seed)] = {}
        result["fixed_test_csv_sha256"][str(seed)] = {}
        for arm in ARMS:
            rows = samples[(seed, arm)]
            checkpoint_path = (
                Path("experiments/dclde_true_bsr/best.pt") if seed == SEEDS[0] and arm == "bsr" else
                Path(f"experiments/dclde_{arm}_residual_128/best.pt") if seed == SEEDS[0] else
                OUTPUT / str(seed) / arm / "best.pt"
            )
            test_path = (
                Path("experiments/dclde_true_bsr/clean_synthetic_test.csv") if seed == SEEDS[0] and arm == "bsr" else
                Path(f"experiments/dclde_{arm}_residual_128/clean_synthetic_test.csv") if seed == SEEDS[0] else
                OUTPUT / str(seed) / arm / ("test_fixed.csv" if arm == "bsr" else "test.csv")
            )
            result["checkpoint_sha256"][str(seed)][arm] = _sha256(checkpoint_path)
            result["fixed_test_csv_sha256"][str(seed)][arm] = _sha256(test_path)
            if [row["example_id"] for row in rows] != ids:
                raise ValueError(f"test IDs differ for {seed}/{arm}")
            for old, new in zip(reference, rows, strict=True):
                for key in ("snr_db", "mixture_si_sdr_db"):
                    if abs(float(old[key]) - float(new[key])) > 1e-4:
                        raise ValueError(f"test mixture differs for {seed}/{arm}: {key}")
            values = np.asarray([float(row["source_si_sdri_db"] if arm == "bsr" or seed != SEEDS[0]
                                       else row[f"{arm}_si_sdri_db"]) for row in rows])
            result["per_seed"][str(seed)][arm] = float(values.mean())
        for left, right in (("persistence", "zero"), ("persistence", "shuffled"),
                            ("forecast", "persistence")):
            key = f"{left}_minus_{right}"
            result["contrasts"].setdefault(key, {})[str(seed)] = (
                result["per_seed"][str(seed)][left] - result["per_seed"][str(seed)][right]
            )
            days = np.asarray([row["day"] for row in samples[(seed, left)]])
            def score(arm: str) -> np.ndarray:
                return np.asarray([
                    float(row["source_si_sdri_db"] if seed != SEEDS[0]
                          else row[f"{arm}_si_sdri_db"])
                    for row in samples[(seed, arm)]
                ])
            differences = score(left) - score(right)
            result["contrast_by_seed_day"].setdefault(key, {})[str(seed)] = {
                str(day): float(differences[days == day].mean()) for day in np.unique(days)
            }
    result["mean_sd"] = {
        arm: {"mean": float(np.mean([result["per_seed"][str(seed)][arm] for seed in SEEDS])),
              "sd": float(np.std([result["per_seed"][str(seed)][arm] for seed in SEEDS], ddof=1))}
        for arm in ARMS
    }
    result["contrast_mean_sd"] = {
        key: {"mean": float(np.mean(list(by_seed.values()))),
              "sd": float(np.std(list(by_seed.values()), ddof=1)),
              "all_positive": bool(all(value > 0 for value in by_seed.values())),
              "all_seed_day_positive": bool(all(
                  value > 0 for day_values in result["contrast_by_seed_day"][key].values()
                  for value in day_values.values()))}
        for key, by_seed in result["contrasts"].items()
    }
    SUMMARY.parent.mkdir(exist_ok=True)
    SUMMARY.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.summary:
        print(json.dumps(summarize(), indent=2))
        return
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    cache = HighResolutionStore(store)
    for seed in ((args.seed,) if args.seed else SEEDS[1:]):
        for arm in ((args.arm,) if args.arm else ARMS):
            if arm == "bsr":
                _train_bsr(store, seed, args.device)
                _fixed_bsr_test(store, seed, args.device)
            else:
                _train_residual(cache, seed, arm, args.device)


if __name__ == "__main__":
    main()
