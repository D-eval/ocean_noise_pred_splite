"""Vector summary of train-day target-pool acoustic and coverage audits."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .empirical_target_pool import PROFILES


OUTPUT = Path("paper/figures/dclde_empirical_target_acceptance.pdf")


def main() -> None:
    old = json.loads(Path("results/dclde_target_gap_diagnostic.json").read_text(encoding="utf-8"))
    new = json.loads(Path("results/dclde_empirical_target_gap_diagnostic.json").read_text(encoding="utf-8"))
    profiles = np.load(PROFILES)
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 2.7), layout="constrained")
    colors = ("#d36959", "#237c78")
    for entry, label, color in zip((old, new), ("Uniform procedural", "50/50 empirical"), colors):
        values = entry["coverage"]["equal_size_nn_median_across_repeats"]
        axes[0].hist(values, bins=np.linspace(.26, .68, 22), alpha=.62,
                     label=label, color=color)
    axes[0].axvline(old["coverage"]["real_to_real_loo_nn_median"],
                    color="black", ls="--", lw=1, label="real-to-real")
    axes[0].set(xlabel="Equal-size nearest distance", ylabel="Repeat count",
                title="Target coverage (32 repeats)")
    axes[0].legend(fontsize=7, frameon=False)

    names = ["Clean", "Old synth", "New 50/50", "Real calls"]
    probabilities = [new["classifier_probability"]["matched_same_WAV_clean_mean"],
                     old["classifier_probability"]["synthetic_whale_mixture_mean"],
                     new["classifier_probability"]["synthetic_whale_mixture_mean"],
                     new["classifier_probability"]["real_event_mean"]]
    axes[1].bar(np.arange(4), probabilities,
                color=["#64748b", colors[0], colors[1], "#30373c"])
    axes[1].set_xticks(np.arange(4), names, rotation=18, ha="right")
    axes[1].set(ylim=(0, 1.03), ylabel="Frozen classifier event probability",
                title="Detection response audit")
    for index, value in enumerate(probabilities):
        axes[1].text(index, value+.025, f"{value:.2f}", ha="center", fontsize=8)

    axes[2].hist(profiles["duration_seconds"], bins=np.linspace(0, 4, 25),
                 color="#5687aa", alpha=.85)
    axes[2].set(xlabel="Extracted call duration (s)", ylabel="Train events",
                title="Empirical morphology (n=2,616)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
