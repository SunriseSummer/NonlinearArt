"""Starter simulation for case1b (远离临界 baseline).

Run this script first. It uses the refactored :class:`RushHourTrafficSystem`
with a deliberately conservative ``spill_prob``: vehicles arrive steadily,
but the network is so under-utilised that almost nothing topples and the
average load stays low. Players should observe:

* high stability — no large cascades, very small congestion-propagation
  range;
* low resource utilisation — most road capacity is unused;
* throughput barely keeps up with inflow because nothing pressures the
  network into self-organising.

This figure pair sets the visual baseline that phase 2 / phase 3 will
contrast against.

Outputs (under ``case1b/figures/``):

* ``starter_load_throughput.svg`` — twin-axis load + throughput time series
* ``starter_congestion_range.svg`` — congestion-propagation range over time
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from plotting import plot_dual_axis, plot_lines, rolling_mean
from traffic_model import RushHourTrafficSystem, TrafficParams

CASE_DIR = BASE_DIR.parent
FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Conservative regime: low spill_prob and modest inflow → "远离临界".
    params = TrafficParams(
        L=20,
        threshold=6,
        inflow_rate=1.0,
        spill_prob=0.10,
        dissipation=0.25,
        steps=5000,
        warmup=800,
        seed=2026,
    )
    res = RushHourTrafficSystem(params).run()

    steps = list(range(len(res.densities)))
    smooth_throughput = rolling_mean(res.throughput, window=80)

    plot_dual_axis(
        FIG_DIR / "starter_load_throughput.svg",
        x=steps,
        left={
            "y": res.densities,
            "label": "Mean load",
            "ylabel": "Mean load per intersection",
            "color": "#1f77b4",
        },
        right={
            "y": smooth_throughput,
            "label": "Throughput (rolling mean)",
            "ylabel": "Vehicles served per step",
            "color": "#ff7f0e",
        },
        title="Starter (far from critical): low utilisation, stable but inefficient",
        xlabel="Simulation step",
        vlines=[(params.warmup, "#b00000", f"warmup={params.warmup}")],
    )

    plot_lines(
        FIG_DIR / "starter_congestion_range.svg",
        series=[
            {
                "x": steps,
                "y": rolling_mean(res.congestion_range, window=80),
                "label": "Congestion-propagation range (rolling mean)",
                "color": "#2ca02c",
            }
        ],
        title="Starter (far from critical): how many intersections are near-saturated",
        xlabel="Simulation step",
        ylabel="Number of high-load intersections",
        vlines=[(params.warmup, "#b00000", f"warmup={params.warmup}")],
    )

    n_av = len(res.avalanche_sizes)
    avg_load = sum(res.densities[params.warmup:]) / max(
        len(res.densities) - params.warmup, 1
    )
    avg_thrpt = sum(res.throughput[params.warmup:]) / max(
        len(res.throughput) - params.warmup, 1
    )
    print(f"[starter] non-trivial avalanches: {n_av}")
    print(f"[starter] steady-state mean load:  {avg_load:.3f}")
    print(f"[starter] steady-state throughput: {avg_thrpt:.3f} vehicles / step")
    print("Starter figures written to case1b/figures/starter_*.svg")


if __name__ == "__main__":
    main()
