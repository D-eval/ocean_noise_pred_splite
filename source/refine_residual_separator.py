"""Small-set capacity check and train-day-only curriculum for causal Residual-BSR."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dclde import CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE
from .dclde_causal_highres_experiment import (
    CausalResidualBSR, ConditionedInjectedDataset, HighResolutionStore,
    separation_test_rows,
)
from .dclde_separation_experiment import (
    CausalFeatureStore, ExperimentConfig, _seed_everything,
    render_marine_training_target,
)
from .empirical_target_pool import EmpiricalTargetPool
from .metrics import si_sdr
from .model import multi_resolution_loss

OUT = Path("experiments/dclde_residual_refinement")


class CurriculumPool:
    def __init__(self, empirical: EmpiricalTargetPool, pseudo_fraction: float):
        if not 0 <= pseudo_fraction <= .5:
            raise ValueError("pseudo fraction must be in [0,.5]")
        self.empirical = empirical
        self.pseudo_fraction = pseudo_fraction

    def sample(self, seed: int, route: str) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if rng.random() < self.pseudo_fraction:
            return self.empirical.pseudo_real(seed)
        if self.pseudo_fraction == .5 and rng.random() < .5:
            return self.empirical.procedural(seed)
        return render_marine_training_target(10000, seed)[0]


def setup(device: torch.device):
    config = ExperimentConfig(device=str(device))
    _seed_everything(config.seed)
    store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
    cache = HighResolutionStore(store)
    model = CausalResidualBSR(store.bands, config.seed, device,
                              Path(config.transfer_checkpoint)).to(device)
    return config, cache, model


@torch.inference_mode()
def evaluate(model, loader, device) -> dict:
    model.eval()
    gains, losses, leak_db = [], [], []
    gains_by_family: dict[str, list[float]] = {}
    window = torch.hann_window(512, device=device)
    for batch in loader:
        mixture, target, ocean, condition = (batch[key].to(device) for key in
                                             ("mixture", "source", "ocean", "condition_map"))
        estimate = model(mixture, condition)
        length = min(estimate.shape[-1], target.shape[-1])
        estimate, target, ocean, mixture = (value[..., :length] for value in
                                             (estimate, target, ocean, mixture))
        batch_gains = (si_sdr(estimate, target)-si_sdr(mixture, target)).cpu().tolist()
        gains.extend(batch_gains)
        for family, gain in zip(batch["source_family"], batch_gains, strict=True):
            gains_by_family.setdefault(family, []).append(gain)
        losses.append(float(multi_resolution_loss(estimate, target)))
        est_spec, target_spec, ocean_spec = [torch.stft(value, 512, 64, window=window,
                                                        return_complex=True).abs()
                                             for value in (estimate, target, ocean)]
        frequencies = torch.arange(est_spec.shape[1], device=device)*2000/512
        band = (frequencies >= 50) & (frequencies <= 450)
        absent = (target_spec < .1*ocean_spec) & band[None, :, None]
        ratio = 20*torch.log10((est_spec+1e-8)/(ocean_spec+1e-8))
        leak_db.extend([float(ratio[i][absent[i]].median())
                        for i in range(len(ratio))])
    return {"examples": len(gains), "si_sdri_db": float(np.mean(gains)),
            "loss": float(np.mean(losses)),
            "median_absent_event_to_ocean_db": float(np.median(leak_db)),
            "si_sdri_by_source_family_db": {
                name: float(np.mean(values)) for name, values in gains_by_family.items()}}


def overfit(device: torch.device, steps: int = 2000, examples: int = 256,
            pool_name: str = "empirical") -> dict:
    config, cache, model = setup(device)
    start_path = ("models/empirical_persistence_residual_128.pt" if pool_name == "empirical"
                  else "experiments/dclde_persistence_residual_128/best.pt")
    checkpoint = torch.load(start_path,
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    pool = EmpiricalTargetPool("fit") if pool_name == "empirical" else None
    dataset = ConditionedInjectedDataset(cache, "train", examples, config.seed + 4700,
                                         "persistence", cache_waveforms=True,
                                         target_pool=pool)
    train = DataLoader(dataset, batch_size=4, shuffle=True,
                       generator=torch.Generator().manual_seed(config.seed), num_workers=0)
    score = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    history = [{"step": 0, **evaluate(model, score, device)}]
    print(json.dumps({"stage": "overfit", **history[0]}), flush=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0)
    iterator = iter(train)
    for step in range(1, steps+1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train)
            batch = next(iterator)
        mixture, source, condition = (batch[key].to(device) for key in
                                      ("mixture", "source", "condition_map"))
        estimate = model(mixture, condition)
        loss = multi_resolution_loss(estimate, source)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimiser.step()
        if step % 250 == 0 or step == steps:
            row = {"step": step, "batch_loss": float(loss.detach()),
                   **evaluate(model, score, device)}
            history.append(row)
            print(json.dumps({"stage": "overfit", **row}), flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"overfit_{pool_name}_training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            torch.save({"model": model.state_dict(), "step": step,
                        "fixed_examples": examples, "pool": pool_name,
                        "interface": "persistence_residual_128"},
                       OUT / f"overfit_{pool_name}_latest.pt")
    result = {"protocol": f"fixed {examples} train mixtures; warm start {start_path}; no test data",
              "steps": steps, "history": history}
    Path(f"results/dclde_residual_overfit_{pool_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def audit_overfit(device: torch.device, pool_name: str, examples: int) -> dict:
    config, cache, model = setup(device)
    initial = ("models/empirical_persistence_residual_128.pt" if pool_name == "empirical"
               else "experiments/dclde_persistence_residual_128/best.pt")
    target_pool = EmpiricalTargetPool("fit") if pool_name == "empirical" else None
    dataset = ConditionedInjectedDataset(cache, "train", examples, config.seed+4700,
                                         "persistence", cache_waveforms=True,
                                         target_pool=target_pool)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.load_state_dict(torch.load(initial, map_location=device,
                                     weights_only=False)["model"], strict=True)
    before = evaluate(model, loader, device)
    final = OUT / f"overfit_{pool_name}_latest.pt"
    checkpoint = torch.load(final, map_location=device, weights_only=False)
    if checkpoint["fixed_examples"] != examples:
        raise ValueError("overfit checkpoint uses a different fixed set size")
    model.load_state_dict(checkpoint["model"], strict=True)
    after = evaluate(model, loader, device)
    report = {"protocol": f"fixed {examples} training mixtures; diagnostic only",
              "steps": checkpoint["step"], "before": before, "after": after}
    Path(f"results/dclde_residual_overfit_{pool_name}_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def continue_overfit(device: torch.device, pool_name: str, examples: int,
                     max_steps: int = 10000) -> dict:
    config, cache, model = setup(device)
    path = OUT / f"overfit_{pool_name}_latest.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["fixed_examples"] != examples:
        raise ValueError("continuation must use the identical fixed set")
    model.load_state_dict(checkpoint["model"], strict=True)
    target_pool = EmpiricalTargetPool("fit") if pool_name == "empirical" else None
    dataset = ConditionedInjectedDataset(cache, "train", examples, config.seed+4700,
                                         "persistence", cache_waveforms=True,
                                         target_pool=target_pool)
    train = DataLoader(dataset, batch_size=4, shuffle=True,
                       generator=torch.Generator().manual_seed(config.seed+10000))
    score = DataLoader(dataset, batch_size=16, shuffle=False)
    history = [{"extra_step": 0, "total_step": checkpoint["step"],
                **evaluate(model, score, device)}]
    best, stale = history[0]["si_sdri_db"], 0
    print(json.dumps({"stage": "overfit_continue", **history[0]}), flush=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    iterator = iter(train)
    for step in range(1, max_steps+1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train)
            batch = next(iterator)
        mixture, source, condition = (batch[key].to(device) for key in
                                      ("mixture", "source", "condition_map"))
        estimate = model(mixture, condition)
        loss = multi_resolution_loss(estimate, source)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimiser.step()
        if step % 500 == 0:
            row = {"extra_step": step, "total_step": int(checkpoint["step"])+step,
                   "batch_loss": float(loss.detach()), **evaluate(model, score, device)}
            history.append(row)
            print(json.dumps({"stage": "overfit_continue", **row}), flush=True)
            if row["si_sdri_db"] > best + .05:
                best, stale = row["si_sdri_db"], 0
            else:
                stale += 1
            torch.save({"model": model.state_dict(), "step": row["total_step"],
                        "fixed_examples": examples, "pool": pool_name,
                        "interface": "persistence_residual_128"}, path)
            (OUT / f"overfit_{pool_name}_continue_training.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8")
            if step >= 5000 and stale >= 6:
                break
    result = {"protocol": "same fixed mixtures; continued diagnostic training, not a model-selection set",
              "initial_step": int(checkpoint["step"]), "extra_steps": history[-1]["extra_step"],
              "history": history}
    Path(f"results/dclde_residual_overfit_{pool_name}_continue.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def objective_probe(device: torch.device, examples: int = 128,
                    steps: int = 2000) -> dict:
    """Paired fixed-set check of the training objective, never used as a test score."""
    config, cache, model = setup(device)
    initial = torch.load(OUT / "overfit_empirical_latest.pt",
                         map_location=device, weights_only=False)
    if initial["fixed_examples"] != examples:
        raise ValueError("objective probe requires the identical fixed mixture set")
    dataset = ConditionedInjectedDataset(cache, "train", examples,
                                         config.seed+4700, "persistence",
                                         cache_waveforms=True,
                                         target_pool=EmpiricalTargetPool("fit"))
    score = DataLoader(dataset, batch_size=16, shuffle=False)
    arms = {}
    for name, sisdr_weight in (("original", 0.0), ("si_sdr_aux", 1e-4)):
        _seed_everything(config.seed+700)
        model.load_state_dict(initial["model"], strict=True)
        loader = DataLoader(dataset, batch_size=4, shuffle=True,
                            generator=torch.Generator().manual_seed(config.seed+700))
        optimiser = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
        iterator = iter(loader)
        history = [{"step": 0, **evaluate(model, score, device)}]
        for step in range(1, steps+1):
            model.train()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            mixture, source, condition = (batch[key].to(device) for key in
                                          ("mixture", "source", "condition_map"))
            estimate = model(mixture, condition)
            loss = multi_resolution_loss(estimate, source)
            if sisdr_weight:
                length = min(estimate.shape[-1], source.shape[-1])
                loss = loss-sisdr_weight*si_sdr(
                    estimate[..., :length], source[..., :length]).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimiser.step()
            if step % 500 == 0:
                row = {"step": step, "objective_loss": float(loss.detach()),
                       **evaluate(model, score, device)}
                history.append(row)
                print(json.dumps({"arm": name, **row}), flush=True)
        arms[name] = {"si_sdr_weight": sisdr_weight, "history": history}
    result = {"protocol": "same 128 fixed training mixtures, same 20k checkpoint, data order, 2e-5 learning rate and updates; training diagnostic only",
              "steps": steps, "arms": arms}
    Path("results/dclde_residual_objective_probe.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def curriculum(device: torch.device, max_steps: int = 20000,
               validate_every: int = 500, patience: int = 6) -> dict:
    config, cache, model = setup(device)
    init_path = Path("experiments/dclde_persistence_residual_128/best.pt")
    checkpoint = torch.load(init_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    train_pool = EmpiricalTargetPool("fit")
    heldout_pool = EmpiricalTargetPool("validation")
    old_val = ConditionedInjectedDataset(cache, "validation", 512,
                                         config.seed+10_000_000, "persistence",
                                         cache_waveforms=True)
    empirical_val = ConditionedInjectedDataset(cache, "validation", 512,
                                               config.seed+11_000_000, "persistence",
                                               cache_waveforms=True, target_pool=heldout_pool)
    validation = {"old_procedural": DataLoader(old_val, batch_size=24, num_workers=0),
                  "heldout_empirical": DataLoader(empirical_val, batch_size=24, num_workers=0)}
    def assess(step: int, phase: str) -> dict:
        old_score = evaluate(model, validation["old_procedural"], device)
        new_score = evaluate(model, validation["heldout_empirical"], device)
        return {"step": step, "phase": phase, "old": old_score, "empirical": new_score,
                "joint_si_sdri_db": (old_score["si_sdri_db"]+new_score["si_sdri_db"])/2}
    OUT.mkdir(parents=True, exist_ok=True)
    history = [assess(0, "initial")]
    print(json.dumps(history[-1]), flush=True)
    best, best_step, stale = history[0]["joint_si_sdri_db"], 0, 0
    torch.save({"model": model.state_dict(), "step": 0, "phase": "initial",
                "interface": "persistence_residual_128", "initial": str(init_path)}, OUT / "best.pt")
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)
    phase_bounds = [(min(2000, max_steps), 0.0),
                    (min(8000, max_steps), .25), (max_steps, .5)]
    start = 0
    stopped_early = False
    for phase_end, pseudo_fraction in phase_bounds:
        if phase_end <= start:
            continue
        dataset = ConditionedInjectedDataset(cache, "train", config.train_examples,
                                             config.seed+round(pseudo_fraction*1000),
                                             "persistence", target_pool=CurriculumPool(
                                                 train_pool, pseudo_fraction))
        loader = DataLoader(dataset, batch_size=4, shuffle=True,
                            generator=torch.Generator().manual_seed(config.seed+start),
                            num_workers=0)
        iterator = iter(loader)
        for step in range(start+1, phase_end+1):
            model.train()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            mixture, source, condition = (batch[key].to(device) for key in
                                          ("mixture", "source", "condition_map"))
            estimate = model(mixture, condition)
            loss = multi_resolution_loss(estimate, source)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimiser.step()
            if step % validate_every == 0 or step == phase_end:
                row = assess(step, f"pseudo_{pseudo_fraction:.2f}")
                row["batch_loss"] = float(loss.detach())
                history.append(row)
                print(json.dumps(row), flush=True)
                if row["joint_si_sdri_db"] > best + .01:
                    best, best_step, stale = row["joint_si_sdri_db"], step, 0
                    torch.save({"model": model.state_dict(), "step": step,
                                "phase": row["phase"], "interface": "persistence_residual_128",
                                "training_target_pool": "curriculum_legacy_calibrated_pseudo",
                                "pseudo_train_subset": "fit80pct"}, OUT / "best.pt")
                else:
                    stale += 1
                (OUT / "curriculum_training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                if step >= 8000 and stale >= patience:
                    stopped_early = True
                    break
        start = phase_end
        if stopped_early:
            break
    result = {"protocol": "legacy Persistence initialization; train-day pseudo 80/20 disjoint fit/validation; two fixed 512-example validation suites; no test selection",
              "max_steps": max_steps, "steps_run": history[-1]["step"],
              "best_step": best_step, "best_joint_si_sdri_db": best,
              "checkpoint": str(OUT / "best.pt"), "history": history}
    Path("results/dclde_residual_curriculum.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def resume_half_pseudo(device: torch.device, max_steps: int = 12000,
                       validate_every: int = 500,
                       sisdr_weight: float = 0.0) -> dict:
    """Give the final 50/50 phase an independent budget after the first curriculum."""
    config, cache, model = setup(device)
    prior = torch.load(OUT / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(prior["model"], strict=True)
    fit_pool = EmpiricalTargetPool("fit")
    heldout = EmpiricalTargetPool("validation")
    train = ConditionedInjectedDataset(cache, "train", config.train_examples,
                                       config.seed+500, "persistence",
                                       target_pool=CurriculumPool(fit_pool, .5))
    old_val = ConditionedInjectedDataset(cache, "validation", 512,
                                         config.seed+10_000_000, "persistence",
                                         cache_waveforms=True)
    new_val = ConditionedInjectedDataset(cache, "validation", 512,
                                         config.seed+11_000_000, "persistence",
                                         cache_waveforms=True, target_pool=heldout)
    loader = DataLoader(train, batch_size=4, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed+8000))
    checks = (DataLoader(old_val, batch_size=24), DataLoader(new_val, batch_size=24))
    def assess(step):
        old, empirical = (evaluate(model, check, device) for check in checks)
        return {"step": step, "phase": "pseudo_0.50_resume", "old": old,
                "empirical": empirical,
                "joint_si_sdri_db": (old["si_sdri_db"]+empirical["si_sdri_db"])/2}
    history = [assess(0)]
    best, best_step, stale = history[0]["joint_si_sdri_db"], 0, 0
    print(json.dumps(history[-1]), flush=True)
    suffix = "_sisdr" if sisdr_weight else ""
    output = OUT / f"resume{suffix}_best.pt"
    torch.save({"model": model.state_dict(), "step": int(prior["step"]),
                "phase": "resume_initial", "interface": "persistence_residual_128",
                "training_target_pool": "curriculum_legacy_calibrated_pseudo",
                "pseudo_train_subset": "fit80pct"}, output)
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)
    iterator = iter(loader)
    for step in range(1, max_steps+1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        mixture, source, condition = (batch[key].to(device) for key in
                                      ("mixture", "source", "condition_map"))
        estimate = model(mixture, condition)
        loss = multi_resolution_loss(estimate, source)
        if sisdr_weight:
            length = min(estimate.shape[-1], source.shape[-1])
            loss = loss-sisdr_weight*si_sdr(
                estimate[..., :length], source[..., :length]).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimiser.step()
        if step % validate_every == 0:
            row = assess(step)
            row["batch_loss"] = float(loss.detach())
            history.append(row)
            print(json.dumps(row), flush=True)
            if row["joint_si_sdri_db"] > best + .01:
                best, best_step, stale = row["joint_si_sdri_db"], step, 0
                torch.save({"model": model.state_dict(),
                            "step": int(prior["step"])+step,
                            "phase": "pseudo_0.50_resume",
                            "interface": "persistence_residual_128",
                            "training_target_pool": "curriculum_legacy_calibrated_pseudo",
                            "pseudo_train_subset": "fit80pct"}, output)
            else:
                stale += 1
            (OUT / f"resume{suffix}_training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if step >= 4000 and stale >= 8:
                break
    result = {"protocol": "resume from curriculum-selected 25% pseudo checkpoint; 50% stage, 80/20 disjoint train-day pseudo split; no test selection",
              "initial_checkpoint_step": int(prior["step"]), "max_steps": max_steps,
              "si_sdr_auxiliary_weight": sisdr_weight,
              "steps_run": history[-1]["step"], "best_step_in_resume": best_step,
              "best_joint_si_sdri_db": best, "checkpoint": str(output),
              "history": history}
    Path(f"results/dclde_residual_curriculum_resume{suffix}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def test_selected(device: torch.device, selected: Path = OUT / "best.pt") -> dict:
    config, cache, model = setup(device)
    checkpoint_path = selected
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = ConditionedInjectedDataset(cache, "test", 4096,
                                         config.seed+20_000_000, "persistence",
                                         spread_rows=True)
    loader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0)
    values = separation_test_rows(model, loader, device, "persistence")
    csv_path = Path("results/dclde_refined_persistence_legacy_test.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)
    refined = np.asarray([row["persistence_si_sdri_db"] for row in values])
    bsr = np.asarray([row["bsr_si_sdri_db"] for row in values])
    original_path = Path("experiments/dclde_persistence_residual_128/clean_synthetic_test.csv")
    with original_path.open(newline="", encoding="utf-8") as stream:
        original = list(csv.DictReader(stream))
    if [row["example_id"] for row in values] != [row["example_id"] for row in original]:
        raise ValueError("refined and old procedural test cohorts differ")
    old = np.asarray([float(row["persistence_si_sdri_db"]) for row in original])
    report = {"protocol": "one-shot unchanged 4096-mixture legacy test after dual-validation selection",
              "selected_step": checkpoint["step"], "selected_phase": checkpoint["phase"],
              "mean_si_sdri_db": float(refined.mean()),
              "original_persistence_si_sdri_db": float(old.mean()),
              "bsr_si_sdri_db": float(bsr.mean()),
              "delta_vs_original_persistence_db": float((refined-old).mean()),
              "delta_vs_bsr_db": float((refined-bsr).mean()),
              "csv": str(csv_path)}
    Path("results/dclde_refined_persistence_legacy_test.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(checkpoint_path, "models/refined_persistence_residual_128.pt")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("overfit", "continue-overfit", "audit-overfit", "objective-probe", "curriculum", "resume", "resume-sisdr", "test"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--pool", choices=("empirical", "legacy"), default="empirical")
    parser.add_argument("--checkpoint", type=Path, default=OUT / "best.pt")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.mode == "overfit":
        result = overfit(device, args.steps or 2000, args.examples, args.pool)
    elif args.mode == "continue-overfit":
        result = continue_overfit(device, args.pool, args.examples, args.steps or 10000)
    elif args.mode == "audit-overfit":
        result = audit_overfit(device, args.pool, args.examples)
    elif args.mode == "objective-probe":
        result = objective_probe(device, args.examples, args.steps or 2000)
    elif args.mode == "curriculum":
        result = curriculum(device, args.steps or 20000)
    elif args.mode == "resume":
        result = resume_half_pseudo(device, args.steps or 12000)
    elif args.mode == "resume-sisdr":
        result = resume_half_pseudo(device, args.steps or 4000,
                                   sisdr_weight=1e-4)
    else:
        result = test_selected(device, args.checkpoint)
    output_path = (f"results/dclde_residual_overfit_{args.pool}.json" if args.mode == "overfit"
                   else f"results/dclde_residual_overfit_{args.pool}_continue.json" if args.mode == "continue-overfit"
                   else f"results/dclde_residual_overfit_{args.pool}_audit.json" if args.mode == "audit-overfit"
                   else "results/dclde_residual_objective_probe.json" if args.mode == "objective-probe"
                   else "results/dclde_residual_curriculum.json" if args.mode == "curriculum"
                   else "results/dclde_residual_curriculum_resume.json" if args.mode == "resume"
                   else "results/dclde_residual_curriculum_resume_sisdr.json" if args.mode == "resume-sisdr"
                   else "results/dclde_refined_persistence_legacy_test.json")
    print(json.dumps({"mode": args.mode, "pool": args.pool,
                      "result_path": output_path,
                      "best": result.get("best_step")}), flush=True)


if __name__ == "__main__":
    main()
