"""Phase 2 reference implementation: add adaptation for self-organized criticality.

Instead of hand-tuning ``spill_prob`` (phase 1), here the model adjusts it
on-the-fly with a simple proportional controller targeting a steady mean
load. The system should drift to a regime where avalanche statistics are
heavy-tailed, mimicking SOC.

Usage::

    python case1/solution/phase2_solution.py

Outputs:
    case1/figures/phase2_density_and_spillprob.svg
    case1/figures/phase2_avalanche_dist.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly while reusing case1 base modules.
CASE1_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CASE1_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from plotting import log_hist, plot_dual_axis, plot_lines
from traffic_model import TrafficCascadeSystem, TrafficParams

FIG_DIR = CASE1_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_phase2() -> None:
    """Enable adaptation and let the system self-organize near criticality."""
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

    # Mean load and adaptive spill probability live on very different scales,
    # so we plot them on a twin y-axis.
    plot_dual_axis(
        FIG_DIR / "phase2_density_and_spillprob.svg",
        x=list(range(len(res.densities))),
        left={
            "y": res.densities,
            "label": "Mean load",
            "ylabel": "Mean load per intersection",
            "color": "#1f77b4",
        },
        right={
            "y": res.spill_prob_series,
            "label": "Adaptive spill probability",
            "ylabel": "Spill probability p(t)",
            "color": "#9467bd",
        },
        title="Phase 2: self-organization of load and control parameter",
        xlabel="Simulation step",
        vline=params.warmup,
    )

    x_size, y_size = log_hist(res.avalanche_sizes)
    x_dur, y_dur = log_hist(res.avalanche_durations)
    plot_lines(
        FIG_DIR / "phase2_avalanche_dist.svg",
        series=[
            {"x": x_size, "y": y_size, "label": "Avalanche size s",
             "color": "#e377c2", "marker": "o"},
            {"x": x_dur, "y": y_dur, "label": "Avalanche duration T",
             "color": "#8c564b", "marker": "s"},
        ],
        title="Phase 2: avalanche statistics after self-organization",
        xlabel="s or T",
        ylabel="P",
        logx=True,
        logy=True,
    )


if __name__ == "__main__":
    run_phase2()
    print(f"Phase 2 done. Figures written to {FIG_DIR}")
