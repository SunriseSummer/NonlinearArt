"""Phase 1 reference implementation: tune the traffic system to near-criticality.

Usage:
    python case1/solution/phase1_solution.py

Outputs:
    case1/figures/phase1_mean_size_vs_spill_prob.svg
    case1/figures/phase1_size_dist_compare.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly while reusing case1 shared modules.
CASE1_DIR = Path(__file__).resolve().parents[1]
if str(CASE1_DIR) not in sys.path:
    sys.path.insert(0, str(CASE1_DIR))

from plotting import log_hist, write_svg_line_plot
from traffic_model import TrafficCascadeSystem, TrafficParams

FIG_DIR = CASE1_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_phase1() -> None:
    """Tune spill probability and validate near-critical behavior."""
    p_values = [0.08 + 0.02 * i for i in range(15)]
    mean_sizes = []

    for p in p_values:
        params = TrafficParams(
            L=24,
            threshold=6,
            spill_prob=p,
            dissipation=0.20,
            steps=3500,
            warmup=800,
            seed=2026,
            adaptive=False,
        )
        res = TrafficCascadeSystem(params).run()
        mean_sizes.append(sum(res.avalanche_sizes) / max(len(res.avalanche_sizes), 1))

    write_svg_line_plot(
        FIG_DIR / "phase1_mean_size_vs_spill_prob.svg",
        series=[
            {
                "x": p_values,
                "y": mean_sizes,
                "label": "Mean avalanche size",
                "color": "#1f77b4",
            }
        ],
        title="Phase 1: Mean avalanche size when tuning spill probability",
        xlabel="Spill probability p",
        ylabel="Mean avalanche size",
        vline=0.26,
    )

    compare = [
        ("Subcritical p=0.12", 0.12, "#2ca02c"),
        ("Near-critical p=0.26", 0.26, "#ff7f0e"),
        ("Supercritical p=0.34", 0.34, "#d62728"),
    ]
    series = []
    for label, p, color in compare:
        params = TrafficParams(
            L=24,
            threshold=6,
            spill_prob=p,
            dissipation=0.20,
            steps=5500,
            warmup=1000,
            seed=2026,
            adaptive=False,
        )
        res = TrafficCascadeSystem(params).run()
        x, y = log_hist(res.avalanche_sizes)
        series.append({"x": x, "y": y, "label": label, "color": color})

    write_svg_line_plot(
        FIG_DIR / "phase1_size_dist_compare.svg",
        series=series,
        title="Phase 1: Avalanche-size distributions under different p",
        xlabel="Cascade size s",
        ylabel="P(s)",
        logx=True,
        logy=True,
    )


if __name__ == "__main__":
    run_phase1()
    print(f"Phase 1 done. Figures written to {FIG_DIR}")
