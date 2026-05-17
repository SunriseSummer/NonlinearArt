"""Starter simulation for Case 5: a sparse forest under a single ignition.

Run this first.  The default tree density ``p = 0.40`` is well below the 2-D
site-percolation threshold ``p_c ~= 0.5928``, so most lattices have no system
spanning cluster: a single ignition only burns a small isolated patch.  Later
phases sweep ``p`` to locate the percolation transition, then switch to the
slow-growth / lightning Drossel-Schwabl dynamics that self-organise to a
power-law fire-size distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from forest_model import ForestFire, ForestParams
from plotting import plot_forest, plot_lines

CASE_DIR = BASE_DIR.parent
FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # 1) take a "before" snapshot of the unburnt forest
    pre_params = ForestParams(
        L=64,
        mode="static",
        density=0.40,
        ignition="center",
        seed=2026,
    )
    pre_forest = ForestFire(pre_params)
    pre_grid = [row[:] for row in pre_forest.grid]
    plot_forest(
        FIG_DIR / "starter_initial_forest.svg",
        pre_grid,
        "Starter (sparse forest, p=0.40 < p_c)",
    )

    # 2) ignite it and let the fire run to completion
    fire_params = ForestParams(
        L=64,
        mode="static",
        density=0.40,
        ignition="center",
        seed=2026,
    )
    result = ForestFire(fire_params).run()
    plot_forest(
        FIG_DIR / "starter_post_fire.svg",
        result.final_grid,
        "Starter post-fire (small isolated burn)",
    )

    # 3) repeat across a few seeds to give a sense of variance
    sizes: list[int] = []
    durations: list[int] = []
    for seed in range(2026, 2026 + 30):
        params = ForestParams(
            L=64, mode="static", density=0.40, ignition="center", seed=seed,
        )
        r = ForestFire(params).run()
        sizes.append(r.static_fire_size)
        durations.append(r.static_fire_duration)

    plot_lines(
        FIG_DIR / "starter_size_per_seed.svg",
        series=[{
            "x": list(range(len(sizes))),
            "y": sizes,
            "label": "fire size",
            "color": "#d62728",
            "marker": "o",
            "linestyle": "-",
        }],
        title="Starter: sub-critical forest gives small, fizzling fires",
        xlabel="seed index",
        ylabel="trees burnt",
    )

    print(f"[starter] L={fire_params.L}, density={fire_params.density:.2f}, p_c~=0.5928")
    print(f"[starter] single fire: size={result.static_fire_size}, duration={result.static_fire_duration}")
    print(f"[starter] 30-seed mean fire size  = {sum(sizes) / len(sizes):.2f}")
    print(f"[starter] 30-seed max  fire size  = {max(sizes)}")
    print("Starter figures written to case5/figures/starter_*.svg")


if __name__ == "__main__":
    main()
