"""Starter material for players (base scenario, NON-critical regime).

Run this script first — it produces two reference figures that show the
baseline behaviour of the traffic-cascade scenario before any tuning. From
here players are expected to:

* Phase 1 — keep the model identical and only tune parameters (most
  importantly ``spill_prob``) until the system shows tuned-criticality
  signatures (power-law cascade size distribution, growing mean cascade…).
* Phase 2 — add a feedback mechanism so the system *self-organizes* near
  the critical point without manual parameter sweeping.

Output figures are written under ``case1/figures/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the ``base`` package importable regardless of the current working
# directory the player runs the script from.
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from plotting import log_hist, plot_lines
from traffic_model import TrafficCascadeSystem, TrafficParams

CASE1_DIR = BASE_DIR.parent
FIG_DIR = CASE1_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Intentionally subcritical initial setup: small cascades, quick decay.
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

    # Process figure: density time series.
    plot_lines(
        FIG_DIR / "starter_density_timeseries.svg",
        series=[
            {
                "x": list(range(len(result.densities))),
                "y": result.densities,
                "label": "Starter mean load",
                "color": "#1f77b4",
            }
        ],
        title="Starter system (non-critical): mean load over time",
        xlabel="Simulation step",
        ylabel="Mean load per intersection",
        vline=params.warmup,
    )

    # Result figure: avalanche size distribution.
    x, y = log_hist(result.avalanche_sizes)
    plot_lines(
        FIG_DIR / "starter_avalanche_distribution.svg",
        series=[{
            "x": x,
            "y": y,
            "label": "Avalanche size",
            "color": "#d62728",
            "marker": "o",
            "linestyle": "-",
        }],
        title="Starter system (non-critical): avalanche size distribution",
        xlabel="Cascade size s",
        ylabel="P(s)",
        logx=True,
        logy=True,
    )

    print(f"Recorded {len(result.avalanche_sizes)} non-trivial avalanches.")
    print("Starter simulation complete.")
    print("Subcritical baseline figures written to case1/figures/starter_*.svg")


if __name__ == "__main__":
    main()
