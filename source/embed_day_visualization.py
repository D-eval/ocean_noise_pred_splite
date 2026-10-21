"""Embed compact day-analysis data into the interactive HTML fragment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KEEP = (
    "deployment",
    "date",
    "timestamps",
    "frequency_hz",
    "spectrum_dbfs",
    "broadband_dbfs",
    "lag_statistics",
    "adjacent_js_p90_bits",
    "change_horizon_minutes",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/ocean_noise_day.json"))
    parser.add_argument(
        "--template", type=Path,
        default=Path("source/day_noise_visualization.template.html"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/visualizations/ocean-noise-day.html"),
    )
    args = parser.parse_args()
    analysis = json.loads(args.data.read_text(encoding="utf-8"))
    compact = {key: analysis[key] for key in KEEP}
    template = args.template.read_text(encoding="utf-8")
    if template.count("__DATA__") != 1:
        raise ValueError("template must contain exactly one __DATA__ placeholder")
    rendered = template.replace(
        "__DATA__", json.dumps(compact, ensure_ascii=True, separators=(",", ":"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
