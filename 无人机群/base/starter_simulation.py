"""Starter simulation for Case 6: a noisy, near-disordered Vicsek swarm.

Run this first.  The default noise ``eta = 2.0`` rad is well above the
critical value for our default density, so the swarm cannot maintain a
collective heading: ``phi`` fluctuates near 0.2 and the quiver snapshot looks
like rainbow confetti.  Later phases sweep ``eta`` to locate the order /
disorder transition, then probe how a single leader's command propagates
through the swarm at different noise levels, and finally hand the swarm to a
feedback controller that self-organises near the critical edge.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from vicsek_model import VicsekFlock, VicsekParams
from plotting import plot_lines, plot_swarm, rolling_mean, mean

CASE_DIR = BASE_DIR.parent
FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    params = VicsekParams(
        n_agents=400,
        box_size=10.0,
        speed=0.4,
        radius=1.0,
        eta=4.5,
        steps=600,
        warmup=150,
        seed=2026,
    )
    result = VicsekFlock(params).run()
    steps = list(range(params.steps))

    plot_lines(
        FIG_DIR / "starter_phi_timeseries.svg",
        series=[{
            "x": steps,
            "y": rolling_mean(result.phi, 15),
            "label": "polarisation phi",
            "color": "#1f77b4",
        }],
        title="Starter (high-noise swarm): phi stays low, no consensus",
        xlabel="time step",
        ylabel="phi = |<v>|/v0",
        vlines=[(params.warmup, "#b00000", f"warmup={params.warmup}")],
    )

    plot_swarm(
        FIG_DIR / "starter_final_swarm.svg",
        result.final_positions,
        result.final_thetas,
        params.box_size,
        f"Starter swarm (eta={params.eta}, phi~={mean(result.phi[params.warmup:]):.2f})",
    )

    tail = result.phi[params.warmup:]
    print(f"[starter] N={params.n_agents}, eta={params.eta}, density={params.n_agents / params.box_size**2:.2f}")
    print(f"[starter] steady phi (after warmup) = {mean(tail):.3f}")
    print(f"[starter] phi range over tail       = [{min(tail):.3f}, {max(tail):.3f}]")
    print("Starter figures written to case6/figures/starter_*.svg")


if __name__ == "__main__":
    main()
