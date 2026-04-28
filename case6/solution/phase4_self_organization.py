"""Phase 4 (Case 6): self-organised near-criticality via adaptive noise.

We hand the swarm to a feedback controller that only knows the recent average
polarisation phi.  The rule is symmetric and entirely local:

    if  <phi>  >  target_phi  -> increase eta  (swarm too rigid, loosen it)
    if  <phi>  <  target_phi  -> decrease eta  (swarm too floppy, tighten it)

Starting from very different initial noise levels (eta_0 in {1.0, 3.5, 5.0})
we expect every trajectory to converge to a similar eta-band that places phi
near the target.  We then verify that the feedback swarm responds *better*
to a brief external gust than three fixed-eta controls (low/critical/high),
because feedback keeps the system on the responsive side of the transition
even if the operating environment changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = CASE_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from vicsek_model import PerturbationSpec, VicsekFlock, VicsekParams
from plotting import (mean, plot_bars, plot_lines, plot_swarm, rolling_mean,
                      variance)

FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PHI = 0.55
INITIAL_ETAS = [1.0, 3.5, 5.0]


def run_adaptive(eta0: float, seed: int = 2026, steps: int = 900,
                 perturbation: PerturbationSpec | None = None):
    params = VicsekParams(
        n_agents=400, box_size=10.0, speed=0.4, radius=1.0,
        eta=eta0, steps=steps, warmup=200, seed=seed,
        adaptive_noise=True, target_phi=TARGET_PHI,
        eta_gain=0.04, eta_min=0.05, eta_max=6.28,
        feedback_window=30,
        perturbation=perturbation,
    )
    return VicsekFlock(params).run(), params


def run_fixed(eta: float, seed: int = 2026, steps: int = 900,
              perturbation: PerturbationSpec | None = None):
    params = VicsekParams(
        n_agents=400, box_size=10.0, speed=0.4, radius=1.0,
        eta=eta, steps=steps, warmup=200, seed=seed,
        perturbation=perturbation,
    )
    return VicsekFlock(params).run(), params


def main() -> None:
    # -- 1) convergence from multiple initial eta_0
    series_eta = []
    series_phi = []
    final_etas = []
    final_phis = []
    cats: list[str] = []
    for eta0 in INITIAL_ETAS:
        result, params = run_adaptive(eta0)
        steps = list(range(params.steps))
        series_eta.append({
            "x": steps,
            "y": rolling_mean(result.eta_series, 15),
            "label": f"eta_0={eta0}",
        })
        series_phi.append({
            "x": steps,
            "y": rolling_mean(result.phi, 15),
            "label": f"eta_0={eta0}",
        })
        tail_eta = result.eta_series[-200:]
        tail_phi = result.phi[-200:]
        final_etas.append(mean(tail_eta))
        final_phis.append(mean(tail_phi))
        cats.append(f"eta_0={eta0}")
        print(f"[phase4] eta_0={eta0:.2f} -> "
              f"final eta={mean(tail_eta):.3f}, phi={mean(tail_phi):.3f}")

    plot_lines(
        FIG_DIR / "phase4_eta_convergence.svg",
        series=series_eta,
        title="Phase 4: noise eta(t) converges from three initial values",
        xlabel="time step",
        ylabel="eta",
    )
    plot_lines(
        FIG_DIR / "phase4_phi_convergence.svg",
        series=series_phi,
        title=f"Phase 4: phi(t) converges to target {TARGET_PHI:g}",
        xlabel="time step",
        ylabel="phi",
        hline=TARGET_PHI,
    )

    plot_bars(
        FIG_DIR / "phase4_final_eta.svg",
        categories=cats,
        values=final_etas,
        title="Phase 4: final eta after adaptive feedback (last 200 steps)",
        ylabel="<eta>",
    )
    plot_bars(
        FIG_DIR / "phase4_final_phi.svg",
        categories=cats,
        values=final_phis,
        title=f"Phase 4: final phi (target = {TARGET_PHI:g})",
        ylabel="<phi>",
    )

    # -- 2) gust response: feedback swarm vs three fixed-eta controls
    pulse = PerturbationSpec(start=400, duration=40, delta_theta=0.25)
    runs = []
    runs.append(("adaptive (eta_0=3.5)", "#9467bd",
                 run_adaptive(3.5, perturbation=pulse)[0]))
    runs.append(("fixed eta=1.5",        "#1f77b4",
                 run_fixed(1.5, perturbation=pulse)[0]))
    runs.append(("fixed eta=3.5",        "#2ca02c",
                 run_fixed(3.5, perturbation=pulse)[0]))
    runs.append(("fixed eta=4.5",        "#d62728",
                 run_fixed(4.5, perturbation=pulse)[0]))
    plot_lines(
        FIG_DIR / "phase4_gust_response.svg",
        series=[{
            "x": list(range(len(r.phi))),
            "y": r.phi,
            "label": label,
            "color": color,
        } for label, color, r in runs],
        title="Phase 4: same gust, four controllers",
        xlabel="time step",
        ylabel="phi",
        vlines=[(pulse.start, "#555555", "gust on"),
                (pulse.start + pulse.duration, "#555555", "gust off")],
    )

    # -- 3) susceptibility comparison: chi over the late-time window
    chi_vals = []
    chi_labels = []
    chi_colors = []
    for label, color, r in runs:
        tail = r.phi[300:]  # post warmup
        chi_vals.append(400 * variance(tail))
        chi_labels.append(label)
        chi_colors.append(color)
    plot_bars(
        FIG_DIR / "phase4_susceptibility_compare.svg",
        categories=chi_labels,
        values=chi_vals,
        title="Phase 4: late-time susceptibility chi = N*Var(phi)",
        ylabel="chi",
        colors=chi_colors,
    )
    print("[phase4] late-time chi (N=400, post-step 300):")
    for label, val in zip(chi_labels, chi_vals):
        print(f"  {label:>22}: chi={val:.3f}")

    # -- 4) snapshot of the adaptive swarm at end
    adaptive_main, adaptive_params = run_adaptive(3.5)
    plot_swarm(
        FIG_DIR / "phase4_swarm_adaptive.svg",
        adaptive_main.final_positions,
        adaptive_main.final_thetas,
        adaptive_params.box_size,
        "Phase 4: swarm under adaptive noise feedback",
    )

    print("Phase 4 figures written to case6/figures/phase4_*.svg")


if __name__ == "__main__":
    main()
