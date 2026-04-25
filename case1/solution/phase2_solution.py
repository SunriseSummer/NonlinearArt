"""Phase 2 reference implementation: add adaptation for self-organized criticality.

Usage:
    python case1/solution/phase2_solution.py

Outputs:
    case1/figures/phase2_density_and_spillprob.svg
    case1/figures/phase2_avalanche_dist.svg
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


def run_phase2() -> None:
    """Enable adaptation to let the system self-organize near criticality."""
    params = TrafficParams(
        L=24,
        threshold=6,
        spill_prob=0.10,
        dissipation=0.20,
        steps=7000,
        warmup=1000,
        seed=2026,
        adaptive=True,
        target_load=2.8,
        adapt_rate=0.020,
        spill_min=0.05,
        spill_max=0.45,
    )
    res = TrafficCascadeSystem(params).run()

    write_svg_line_plot(
        FIG_DIR / "phase2_density_and_spillprob.svg",
        series=[
            {
                "x": list(range(len(res.densities))),
                "y": res.densities,
                "label": "Mean load",
                "color": "#1f77b4",
            },
            {
                "x": list(range(len(res.spill_prob_series))),
                "y": res.spill_prob_series,
                "label": "Adaptive spill probability",
                "color": "#9467bd",
            },
        ],
        title="Phase 2: Self-organization of load and control parameter",
        xlabel="Simulation step",
        ylabel="Value",
        vline=params.warmup,
    )

    x1, y1 = log_hist(res.avalanche_sizes)
    x2, y2 = log_hist(res.avalanche_durations)
    write_svg_line_plot(
        FIG_DIR / "phase2_avalanche_dist.svg",
        series=[
            {"x": x1, "y": y1, "label": "Avalanche size", "color": "#e377c2"},
            {"x": x2, "y": y2, "label": "Avalanche duration", "color": "#8c564b"},
        ],
        title="Phase 2: Avalanche statistics after self-organization",
        xlabel="s or T",
        ylabel="P",
        logx=True,
        logy=True,
    )


if __name__ == "__main__":
    run_phase2()
    print(f"Phase 2 done. Figures written to {FIG_DIR}")
