"""Phase 1 reference implementation: tune the traffic system to near-criticality.

The script performs two studies on top of the base ``TrafficCascadeSystem``:

1. A sweep over ``spill_prob`` tracking the *mean* avalanche size — a classic
   order-parameter view that should grow rapidly near the critical point.
2. A side-by-side comparison of the avalanche-size distribution at
   sub-critical, near-critical and super-critical settings.

Usage::

    python case1/solution/phase1_solution.py

Outputs:
    case1/figures/phase1_mean_size_vs_spill_prob.svg
    case1/figures/phase1_size_dist_compare.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly while reusing case1 base modules.
CASE1_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CASE1_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from plotting import log_hist, plot_lines
from traffic_model import TrafficCascadeSystem, TrafficParams

FIG_DIR = CASE1_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_phase1() -> None:
    """Tune spill probability and validate near-critical behaviour."""
    # --- (1) order-parameter sweep -----------------------------------------
    p_values = [0.08 + 0.02 * i for i in range(15)]
    mean_sizes: list[float] = []

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
        mean_sizes.append(
            sum(res.avalanche_sizes) / max(len(res.avalanche_sizes), 1)
        )

    plot_lines(
        FIG_DIR / "phase1_mean_size_vs_spill_prob.svg",
        series=[
            {
                "x": p_values,
                "y": mean_sizes,
                "label": "Mean avalanche size",
                "color": "#1f77b4",
                "marker": "o",
            }
        ],
        title="Phase 1: mean avalanche size while tuning spill probability",
        xlabel="Spill probability p",
        ylabel="Mean avalanche size",
        vline=0.26,
    )

    # --- (2) three-regime distribution comparison --------------------------
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
        series.append({
            "x": x,
            "y": y,
            "label": label,
            "color": color,
            "marker": "o",
        })

    plot_lines(
        FIG_DIR / "phase1_size_dist_compare.svg",
        series=series,
        title="Phase 1: avalanche-size distributions under different p",
        xlabel="Cascade size s",
        ylabel="P(s)",
        logx=True,
        logy=True,
    )


if __name__ == "__main__":
    run_phase1()
    print(f"Phase 1 done. Figures written to {FIG_DIR}")
