"""Phase 3 (Case 6): how a leader's command propagates at three noise levels.

We pin one agent to a fixed heading ``leader_theta = pi/2`` (north) and watch
how the *rest of the swarm* aligns with it over time.  The metric is

    A(t) = (1/N) * sum_i cos(theta_i(t) - leader_theta)

A(t) starts near 0 (random initial headings) and would saturate at 1 in a
perfectly aligned swarm.  We compare three regimes:

* sub-critical (eta=1.5): swarm is already very ordered, so it can be hard
  to *redirect* — once it locks onto a heading it tends to stay there.
* near-critical (eta=3.5): the swarm is responsive AND tracks the leader.
* super-critical (eta=4.5): noise drowns the leader's signal entirely.

To make the test fair, we initialise the swarm at the same random state, then
let it warm up briefly without a leader, then turn the leader on and measure
the alignment trajectory and a final swarm snapshot.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = CASE_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from vicsek_model import VicsekFlock, VicsekParams
from plotting import mean, plot_bars, plot_lines, plot_swarm

FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


REGIMES = [
    ("sub-critical (eta=1.5)",   1.5, "#1f77b4"),
    ("near-critical (eta=3.5)",  3.5, "#9467bd"),
    ("super-critical (eta=4.5)", 4.5, "#d62728"),
]


def run_with_leader(eta: float):
    params = VicsekParams(
        n_agents=400,
        box_size=10.0,
        speed=0.4,
        radius=1.0,
        eta=eta,
        steps=900,
        warmup=200,
        seed=2026,
        leader_index=0,
        leader_theta=math.pi / 2.0,  # "north"
    )
    return VicsekFlock(params).run(), params


def main() -> None:
    series_align = []
    final_align = []
    cats: list[str] = []
    for label, eta, color in REGIMES:
        result, params = run_with_leader(eta)
        steps = list(range(params.steps))
        series_align.append({
            "x": steps,
            "y": result.leader_alignment,
            "label": label,
            "color": color,
        })
        plot_swarm(
            FIG_DIR / f"phase3_swarm_eta{int(eta * 10):03d}.svg",
            result.final_positions,
            result.final_thetas,
            params.box_size,
            f"Phase 3 final swarm: {label} (leader -> north)",
            leader_index=params.leader_index,
        )
        tail_align = result.leader_alignment[-200:]
        final_align.append(mean(tail_align))
        cats.append(label)
        print(f"[phase3] {label}: final alignment with leader = {mean(tail_align):.3f}")

    plot_lines(
        FIG_DIR / "phase3_alignment_timeseries.svg",
        series=series_align,
        title="Phase 3: average alignment <cos(theta - theta_leader)> over time",
        xlabel="time step",
        ylabel="alignment with leader",
    )
    plot_bars(
        FIG_DIR / "phase3_alignment_summary.svg",
        categories=cats,
        values=final_align,
        title="Phase 3: final alignment with leader (last 200 steps)",
        ylabel="<cos(theta - theta_leader)>",
        colors=[r[2] for r in REGIMES],
    )

    # An external "wind gust" via a brief perturbation at three eta values.
    # We measure how big the phi response is to the same delta_theta.
    from vicsek_model import PerturbationSpec

    burst_results = []
    for label, eta, color in REGIMES:
        params = VicsekParams(
            n_agents=400, box_size=10.0, speed=0.4, radius=1.0,
            eta=eta, steps=600, warmup=150, seed=2026,
            perturbation=PerturbationSpec(start=300, duration=40, delta_theta=0.25),
        )
        res = VicsekFlock(params).run()
        burst_results.append((label, color, res.phi))

    plot_lines(
        FIG_DIR / "phase3_perturbation_response.svg",
        series=[{
            "x": list(range(len(phi))),
            "y": phi,
            "label": label,
            "color": color,
        } for label, color, phi in burst_results],
        title="Phase 3: phi response to a 40-step angular gust starting at t=300",
        xlabel="time step",
        ylabel="phi",
        vlines=[(300, "#555555", "gust on"), (340, "#555555", "gust off")],
    )

    print("Phase 3 figures written to case6/figures/phase3_*.svg")


if __name__ == "__main__":
    main()
