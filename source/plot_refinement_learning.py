"""Vector diagnostic for separator convergence and the two-domain curriculum."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42


def rows(path: str) -> list[dict]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value["history"] if isinstance(value, dict) else value


def main() -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.55), constrained_layout=True)
    for condition, color in (("persistence", "#15324A"), ("forecast", "#D96C5F")):
        values = rows(f"experiments/dclde_empirical_{condition}_residual_128/training.json")
        axes[0].plot([row["step"] for row in values if "validation_si_sdri_db" in row],
                     [row["validation_si_sdri_db"] for row in values if "validation_si_sdri_db" in row],
                     marker="o", ms=3, label=condition.title(), color=color)
    axes[0].set(title="Original 50/50 runs", xlabel="Update", ylabel="Validation SI-SDRi (dB)")
    axes[0].legend(frameon=False)
    for name, color in (("legacy", "#15324A"), ("empirical", "#D96C5F")):
        values = rows(f"results/dclde_residual_overfit_{name}.json")
        continued = Path(f"results/dclde_residual_overfit_{name}_continue.json")
        if continued.exists():
            values.extend({"step": row["total_step"], "si_sdri_db": row["si_sdri_db"]}
                          for row in rows(str(continued))[1:])
        axes[1].plot([row["step"] for row in values],
                     [row["si_sdri_db"] for row in values], marker="o", ms=2,
                     label=f"{name.title()} fixed set", color=color)
    axes[1].set(title="Small-set capacity check", xlabel="Update", ylabel="Train SI-SDRi (dB)")
    axes[1].legend(frameon=False)
    curriculum = Path("results/dclde_residual_curriculum.json")
    if curriculum.exists():
        values = rows(str(curriculum))
        axes[2].plot([row["step"] for row in values],
                     [row["old"]["si_sdri_db"] for row in values],
                     marker="o", ms=2, label="Old procedural", color="#15324A")
        axes[2].plot([row["step"] for row in values],
                     [row["empirical"]["si_sdri_db"] for row in values],
                     marker="o", ms=2, label="Held-out empirical", color="#D96C5F")
        axes[2].axvline(2000, color="grey", lw=.7, ls="--")
        axes[2].axvline(8000, color="grey", lw=.7, ls="--")
        resumed = Path("results/dclde_residual_curriculum_resume.json")
        if resumed.exists():
            extra = rows(str(resumed))
            start = json.loads(resumed.read_text(encoding="utf-8"))["initial_checkpoint_step"]
            axes[2].plot([start+row["step"] for row in extra],
                         [row["old"]["si_sdri_db"] for row in extra],
                         ls=":", marker=".", ms=3, color="#15324A", label="Old · 50% continuation")
            axes[2].plot([start+row["step"] for row in extra],
                         [row["empirical"]["si_sdri_db"] for row in extra],
                         ls=":", marker=".", ms=3, color="#D96C5F", label="Empirical · 50% continuation")
        axes[2].legend(frameon=False)
    axes[2].set(title="Curriculum validation", xlabel="Update", ylabel="SI-SDRi (dB)")
    for axis in axes:
        axis.grid(alpha=.16)
        axis.spines[["top", "right"]].set_visible(False)
    output = Path("paper/figures/residual_training_diagnostic.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
