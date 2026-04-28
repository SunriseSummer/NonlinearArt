"""Phase 2 (Case 6): locate the order-disorder transition by scanning eta.

For each ``eta`` in a grid we run the Vicsek swarm long enough to forget
initial conditions and record phi(t) on the tail.  We then compute:

* the mean polarisation ``<phi>``: drops smoothly from ~1 to ~0;
* the susceptibility ``chi = N * Var(phi)``: peaks near the critical noise;
* the Binder cumulant ``U_4 = 1 - <phi^4>/(3 <phi^2>^2)``: curves at
  different N cross at the critical point.

We use two system sizes ``N = 200`` and ``N = 800`` (with the box size
adjusted to keep the *density* fixed) so that finite-size scaling is
visible.
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
from plotting import (binder_cumulant, mean, plot_lines, susceptibility)

FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_at_eta(n: int, eta: float, density: float) -> list[float]:
    box = math.sqrt(n / density)
    params = VicsekParams(
        n_agents=n, box_size=box, speed=0.4, radius=1.0,
        eta=eta, steps=900, warmup=300, seed=2026,
    )
    res = VicsekFlock(params).run()
    return res.phi[params.warmup:]


def sweep(n: int, etas: list[float], density: float):
    phi_means = []
    chi_vals = []
    binder = []
    for eta in etas:
        tail = run_at_eta(n, eta, density)
        phi_means.append(mean(tail))
        chi_vals.append(susceptibility(tail, n))
        binder.append(binder_cumulant(tail))
    return phi_means, chi_vals, binder


def main() -> None:
    density = 4.0  # particles per unit area
    etas = [round(0.5 + 0.25 * k, 3) for k in range(0, 21)]  # 0.5..5.5

    phi_small,  chi_small,  bind_small  = sweep(200, etas, density)
    phi_large,  chi_large,  bind_large  = sweep(800, etas, density)

    # 1) order parameter
    plot_lines(
        FIG_DIR / "phase2_order_parameter.svg",
        series=[
            {"x": etas, "y": phi_small, "label": "N=200",
             "color": "#1f77b4", "marker": "o"},
            {"x": etas, "y": phi_large, "label": "N=800",
             "color": "#d62728", "marker": "s"},
        ],
        title="Phase 2: polarisation phi(eta) at fixed density",
        xlabel="noise eta",
        ylabel="<phi>",
    )

    # 2) susceptibility
    plot_lines(
        FIG_DIR / "phase2_susceptibility.svg",
        series=[
            {"x": etas, "y": chi_small, "label": "N=200",
             "color": "#1f77b4", "marker": "o"},
            {"x": etas, "y": chi_large, "label": "N=800",
             "color": "#d62728", "marker": "s"},
        ],
        title="Phase 2: susceptibility chi(eta) — peak ~ critical noise",
        xlabel="noise eta",
        ylabel="chi = N * Var(phi)",
    )

    # 3) Binder cumulant
    plot_lines(
        FIG_DIR / "phase2_binder.svg",
        series=[
            {"x": etas, "y": bind_small, "label": "N=200",
             "color": "#1f77b4", "marker": "o"},
            {"x": etas, "y": bind_large, "label": "N=800",
             "color": "#d62728", "marker": "s"},
        ],
        title="Phase 2: Binder cumulant U_4(eta)",
        xlabel="noise eta",
        ylabel="U_4",
    )

    eta_star_small = etas[max(range(len(chi_small)), key=lambda i: chi_small[i])]
    eta_star_large = etas[max(range(len(chi_large)), key=lambda i: chi_large[i])]
    print(f"[phase2] N=200 chi peak at eta* = {eta_star_small:.3f}")
    print(f"[phase2] N=800 chi peak at eta* = {eta_star_large:.3f}")
    print(f"[phase2] density = {density:.2f}")
    print("Phase 2 figures written to case6/figures/phase2_*.svg")


if __name__ == "__main__":
    main()
