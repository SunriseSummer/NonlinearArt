"""Phase 2 (Case 5): locate the percolation threshold by scanning density.

For each ``density`` in a grid spanning the transition we do many independent
runs on a frozen forest and record:

* the mean and tail of the *fire size*;
* the *largest cluster* size before ignition (the standard percolation order
  parameter ``P_inf``);
* the *susceptibility-like* variance of the cluster size distribution.

We then repeat for two different system sizes ``L = 48`` and ``L = 80`` to
demonstrate finite-size scaling: in 2-D site percolation the critical density
is ``p_c ~= 0.5928`` for the infinite system, but on a finite lattice the peak
of the susceptibility shifts towards (and sharpens around) ``p_c`` as ``L``
increases.
"""

from __future__ import annotations

import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = CASE_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from forest_model import ForestFire, ForestParams
from plotting import (mean, percentile, plot_lines, variance)

FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

P_C = 0.5928  # 2-D site percolation threshold (literature value)


def sweep_one_size(L: int, densities: list[float], n_seeds: int):
    mean_size = []
    p99_size = []
    largest_cluster = []
    chi = []  # variance of largest cluster across seeds, normalised by L^2
    for d in densities:
        sizes = []
        clusters = []
        for seed in range(2026, 2026 + n_seeds):
            params = ForestParams(
                L=L, mode="static", density=d, ignition="center", seed=seed,
            )
            r = ForestFire(params).run()
            sizes.append(r.static_fire_size)
            clusters.append(r.static_largest_cluster)
        mean_size.append(mean(sizes))
        p99_size.append(percentile(sizes, 99))
        largest_cluster.append(mean(clusters) / (L * L))
        chi.append(variance(clusters) / (L * L))
    return mean_size, p99_size, largest_cluster, chi


def main() -> None:
    densities = [round(0.30 + 0.02 * k, 2) for k in range(0, 26)]  # 0.30 .. 0.80
    n_seeds = 50

    series_size_small, p99_small, p_inf_small, chi_small = sweep_one_size(48, densities, n_seeds)
    series_size_large, p99_large, p_inf_large, chi_large = sweep_one_size(80, densities, n_seeds)

    # 1) order parameter P_inf(p) = <largest cluster>/N
    plot_lines(
        FIG_DIR / "phase2_order_parameter.svg",
        series=[
            {"x": densities, "y": p_inf_small, "label": "L=48", "color": "#1f77b4", "marker": "o"},
            {"x": densities, "y": p_inf_large, "label": "L=80", "color": "#d62728", "marker": "s"},
        ],
        title="Phase 2: percolation order parameter P_inf(p)",
        xlabel="tree density p",
        ylabel="<largest cluster> / N",
        vlines=[(P_C, "#555555", f"p_c≈{P_C}")],
    )

    # 2) susceptibility-like variance of largest cluster
    plot_lines(
        FIG_DIR / "phase2_susceptibility.svg",
        series=[
            {"x": densities, "y": chi_small, "label": "L=48", "color": "#1f77b4", "marker": "o"},
            {"x": densities, "y": chi_large, "label": "L=80", "color": "#d62728", "marker": "s"},
        ],
        title="Phase 2: susceptibility-like variance of largest cluster",
        xlabel="tree density p",
        ylabel="Var(largest cluster) / N",
        vlines=[(P_C, "#555555", f"p_c≈{P_C}")],
    )

    # 3) mean fire size and 99th-percentile fire size
    plot_lines(
        FIG_DIR / "phase2_fire_size.svg",
        series=[
            {"x": densities, "y": series_size_large, "label": "<fire size> (L=80)",
             "color": "#2ca02c", "marker": "o"},
            {"x": densities, "y": p99_large, "label": "P99 fire size (L=80)",
             "color": "#d62728", "marker": "s"},
        ],
        title="Phase 2: fire size grows abruptly across p_c",
        xlabel="tree density p",
        ylabel="trees burnt",
        vlines=[(P_C, "#555555", f"p_c≈{P_C}")],
        logy=True,
    )

    # locate empirical critical density: peak of susceptibility (large size)
    p_star_idx = max(range(len(densities)), key=lambda i: chi_large[i])
    print(f"[phase2] L=80 susceptibility peak at p* = {densities[p_star_idx]:.3f}")
    print(f"[phase2] literature p_c   = {P_C}")
    print(f"[phase2] |p* - p_c| (L=80) = {abs(densities[p_star_idx] - P_C):.3f}")
    p_star_idx48 = max(range(len(chi_small)), key=lambda i: chi_small[i])
    print(f"[phase2] L=48 susceptibility peak at p* = {densities[p_star_idx48]:.3f}")
    print("Phase 2 figures written to case5/figures/phase2_*.svg")


if __name__ == "__main__":
    main()
