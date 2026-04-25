"""Starter material for players.

This script models a concrete traffic-overload scene in a NON-critical regime.
Players should use this script as the baseline and develop phase-1/phase-2 experiments from it.
"""

from __future__ import annotations

from pathlib import Path

from plotting import log_hist, write_svg_line_plot
from traffic_model import TrafficCascadeSystem, TrafficParams

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Intentionally subcritical initial setup (small cascades, quick decay)
    params = TrafficParams(
        L=24,
        threshold=6,
        spill_prob=0.12,
        dissipation=0.20,
        steps=5000,
        warmup=1000,
        seed=2026,
        adaptive=False,
    )

    result = TrafficCascadeSystem(params).run()

    # Process figure: density time series
    write_svg_line_plot(
        FIG_DIR / "starter_density_timeseries.svg",
        series=[
            {
                "x": list(range(len(result.densities))),
                "y": result.densities,
                "label": "Starter mean load",
                "color": "#1f77b4",
            }
        ],
        title="Starter System (Non-critical): Mean load over time",
        xlabel="Simulation step",
        ylabel="Mean load per intersection",
        vline=params.warmup,
    )

    # Result figure: avalanche size distribution
    x, y = log_hist(result.avalanche_sizes)
    write_svg_line_plot(
        FIG_DIR / "starter_avalanche_distribution.svg",
        series=[{"x": x, "y": y, "label": "Avalanche size", "color": "#d62728"}],
        title="Starter System (Non-critical): Avalanche size distribution",
        xlabel="Cascade size s",
        ylabel="P(s)",
        logx=True,
        logy=True,
    )

    print("Starter simulation complete.")
    print("Subcritical baseline created at: case1/figures/starter_*.svg")


if __name__ == "__main__":
    main()
