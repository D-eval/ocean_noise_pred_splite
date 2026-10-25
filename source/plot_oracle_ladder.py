"""Draw a vector summary of the DCLDE Oracle information ladder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(result: Path, output: Path) -> None:
    values = json.loads(result.read_text(encoding="utf-8"))["test"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25), sharey=True)
    colors = {"true": "#2463a7", "mix": "#d45d3e"}

    def panel(ax, title, prefix, labels, phase, names):
        xpos = np.arange(len(labels))
        for phase_name in phase:
            keys = names[phase_name]
            valid = [(index, values[key]) for index, key in enumerate(keys) if key is not None]
            ix = np.asarray([entry[0] for entry in valid])
            means = np.asarray([entry[1]["delta_vs_bsr_db"] for entry in valid])
            lows = np.asarray([entry[1]["delta_vs_bsr_day_ci95_db"][0] for entry in valid])
            highs = np.asarray([entry[1]["delta_vs_bsr_day_ci95_db"][1] for entry in valid])
            ax.errorbar(ix, means, yerr=[means-lows, highs-means], fmt="o-", lw=1.5,
                        ms=4, capsize=2.2, color=colors[phase_name],
                        label="True $N$ phase" if phase_name == "true" else "Mixture $x$ phase")
        ax.axhline(0, color="#555555", lw=0.8, ls="--")
        ax.set_xticks(xpos, labels)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.set_ylim(-2.0, 13.1)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if names["true"][0] is None:
            ax.annotate("+71.1", (0, 12.4), ha="center", color=colors["true"], fontsize=7,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        if prefix:
            ax.set_ylabel(r"SI-SDRi $\Delta$ vs BSR (dB)")

    panels = [
        (axes[0, 0], "A  Equal-width frequency groups · 64-ms hop", ["257", "128", "64", "32"],
         {"true": [None, "l128_frame_true", "l64_frame_true", "l32_frame_true"],
          "mix": ["full_mix", "l128_frame_mix", "l64_frame_mix", "l32_frame_mix"]}),
        (axes[0, 1], "B  Geometric frequency bands · 64-ms hop", ["257", "128", "64", "48", "30"],
         {"true": [None, "b128_frame_true", "b64_frame_true", "b48_frame_true", "b30_frame_true"],
          "mix": ["full_mix", "b128_frame_mix", "b64_frame_mix", "b48_frame_mix", "b30_frame_mix"]}),
        (axes[1, 0], "C  Time pooling · all 257 frequency bins", ["64ms", ".25s", "1s", "5s"],
         {"true": [None, "full_250ms_true", "full_1s_true", "full_5s_true"],
          "mix": ["full_mix", "full_250ms_mix", "full_1s_mix", "full_5s_mix"]}),
        (axes[1, 1], "D  Time pooling · 30 geometric bands", ["64ms", ".25s", "1s", "5s"],
         {"true": ["b30_frame_true", "b30_250ms_true", "b30_1s_true", "b30_5s_true"],
          "mix": ["b30_frame_mix", "b30_250ms_mix", "b30_1s_mix", "b30_5s_mix"]}),
    ]
    for index, (ax, title, labels, names) in enumerate(panels):
        panel(ax, title, index % 2 == 0, labels, ("true", "mix"), names)
    axes[0, 0].legend(loc="upper right", fontsize=7, frameon=False)
    axes[0, 0].set_xlabel("Frequency groups")
    axes[0, 1].set_xlabel("Nominal geometric bands")
    axes[1, 0].set_xlabel("Power averaging interval")
    axes[1, 1].set_xlabel("Power averaging interval")
    fig.text(0.5, 0.005, "Full-resolution true-phase Oracle: +71.1 dB, above plotted range; error bars are paired day-block 95% intervals.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.5, w_pad=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=Path("results/dclde_information_ladder.json"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures/oracle_information_ladder.pdf"))
    args = parser.parse_args()
    draw(args.result, args.output)


if __name__ == "__main__":
    main()
