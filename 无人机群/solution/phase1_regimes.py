"""Phase 1 (Case 6): three noise levels, three swarm fates.

We run the same Vicsek swarm at three noise amplitudes and compare phi(t)
together with quiver snapshots:

* low noise  (eta = 1.5): swarm becomes strongly polarised, phi ~ 0.9.
* medium     (eta = 3.0): swarm hovers near the order/disorder boundary.
* high noise (eta = 4.5): swarm is essentially disordered, phi ~ 0.1.

Phase 2 then sweeps eta to pin down the critical value.
"""

from __future__ import annotations

import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = CASE_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from vicsek_model import VicsekFlock, VicsekParams
from plotting import (mean, plot_bars, plot_lines, plot_swarm,
                      rolling_mean, variance)

FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


REGIMES = [
    ("low noise (eta=1.5)",   1.5, "#1f77b4"),
    ("medium (eta=3.0)",      3.0, "#9467bd"),
    ("high noise (eta=4.5)",  4.5, "#d62728"),
]


def run_regime(eta: float):
    params = VicsekParams(
        n_agents=400, box_size=10.0, speed=0.4, radius=1.0,
        eta=eta, steps=600, warmup=150, seed=2026,
    )
    return VicsekFlock(params).run(), params


def main() -> None:
    series_phi = []
    summary_phi = []
    summary_var = []
    cats: list[str] = []

    for label, eta, color in REGIMES:
        result, params = run_regime(eta)
        steps = list(range(params.steps))
        series_phi.append({
            "x": steps,
            "y": rolling_mean(result.phi, 15),
            "label": label,
            "color": color,
        })
        plot_swarm(
            FIG_DIR / f"phase1_swarm_eta{int(eta * 10):03d}.svg",
            result.final_positions,
            result.final_thetas,
            params.box_size,
            f"Phase 1 swarm: {label}",
        )
        tail = result.phi[params.warmup:]
        summary_phi.append(mean(tail))
        summary_var.append(variance(tail))
        cats.append(label)
        print(f"[phase1] {label}: <phi>={mean(tail):.3f}  Var(phi)={variance(tail):.4f}")

    plot_lines(
        FIG_DIR / "phase1_phi_timeseries.svg",
        series=series_phi,
        title="Phase 1: phi(t) at three noise levels (smoothed)",
        xlabel="time step",
        ylabel="phi = |<v>|/v0",
    )
    plot_bars(
        FIG_DIR / "phase1_steady_phi.svg",
        categories=cats,
        values=summary_phi,
        title="Phase 1: steady-state polarisation",
        ylabel="<phi>",
        colors=[r[2] for r in REGIMES],
    )
    plot_bars(
        FIG_DIR / "phase1_phi_variance.svg",
        categories=cats,
        values=summary_var,
        title="Phase 1: phi variance — biggest swings near the transition",
        ylabel="Var(phi)",
        colors=[r[2] for r in REGIMES],
    )
    print("Phase 1 figures written to case6/figures/phase1_*.svg")


if __name__ == "__main__":
    main()
