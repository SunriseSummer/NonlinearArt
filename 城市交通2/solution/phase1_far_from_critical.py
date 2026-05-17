"""Phase 1 (far from critical): conservative dispatch, stable but inefficient.

Goal: let players *feel* the cost of being too cautious.

We compare two conservative configurations against the starter to show the
same qualitative pattern:

* Mean load stays well below the topple threshold.
* Cascades are tiny and rare; congestion-propagation range is essentially
  zero.
* Throughput is bounded by inflow — increasing inflow modestly does help,
  but per-intersection capacity is still mostly idle.

The figure produced is meant to be read together with the prompt in the
task book: "your network is stable but you've barely used the road
capacity — can you push throughput further without breaking it?"

Usage::

    python case1b/solution/phase1_far_from_critical.py

Outputs:
    case1b/figures/phase1_far_from_critical.svg
    case1b/figures/phase1_throughput_vs_inflow.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CASE_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from plotting import plot_bars, plot_lines, rolling_mean
from traffic_model import RushHourTrafficSystem, TrafficParams

FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _steady(values, warmup):
    tail = values[warmup:]
    return sum(tail) / max(len(tail), 1)


def run_phase1() -> None:
    common = dict(L=20, threshold=6, spill_prob=0.10, dissipation=0.25,
                  steps=4500, warmup=800, seed=2026)

    configs = [
        ("inflow=0.6 (very conservative)", 0.6, "#2ca02c"),
        ("inflow=1.0 (starter baseline)", 1.0, "#1f77b4"),
        ("inflow=1.4 (mild push)", 1.4, "#ff7f0e"),
    ]

    print("[phase1] conservative regime — utilisation stays low even when"
          " inflow grows:")
    print("  config                          <load>   <thrpt>   <cong_range>")

    series = []
    bar_categories = []
    bar_loads = []
    bar_thrpt = []
    for label, lam, color in configs:
        params = TrafficParams(inflow_rate=lam, **common)
        res = RushHourTrafficSystem(params).run()

        load_ss = _steady(res.densities, params.warmup)
        thrpt_ss = _steady(res.throughput, params.warmup)
        cong_ss = _steady(res.congestion_range, params.warmup)

        print(f"  {label:<32} {load_ss:6.3f}  {thrpt_ss:6.3f}    {cong_ss:6.3f}")

        series.append({
            "x": list(range(len(res.densities))),
            "y": rolling_mean(res.densities, 80),
            "label": f"{label} -> mean load",
            "color": color,
            "linewidth": 1.4,
        })
        bar_categories.append(label.split(" ")[0])
        bar_loads.append(load_ss)
        bar_thrpt.append(thrpt_ss)

    plot_lines(
        FIG_DIR / "phase1_far_from_critical.svg",
        series=series,
        title="Phase 1: conservative dispatch keeps load well below threshold",
        xlabel="Simulation step",
        ylabel="Mean load (rolling)",
        vlines=[(common["warmup"], "#b00000", f"warmup={common['warmup']}")],
    )

    plot_bars(
        FIG_DIR / "phase1_throughput_vs_inflow.svg",
        categories=bar_categories,
        values=bar_thrpt,
        title="Phase 1: throughput grows with inflow, but capacity stays under-used",
        ylabel="Steady-state throughput (vehicles / step)",
        colors=[c for _, _, c in configs],
    )


if __name__ == "__main__":
    run_phase1()
    print(f"Phase 1 done. Figures written to {FIG_DIR}")
