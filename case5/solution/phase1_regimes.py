"""Phase 1 (Case 5): three forest densities, three fates.

We compare three regimes for a single ignition on a frozen ``L x L`` forest:

* sub-critical  (``density = 0.40 < p_c``): trees are scattered, fires fizzle.
* near-critical (``density = 0.59 ~ p_c``): a fragile tipping point.
* super-critical(``density = 0.75 > p_c``): a spanning cluster, fires rage.

For each regime we run many independent seeds so the figures show the *full
distribution* of fire sizes, not a single anecdote.  This is the qualitative
"feel" of the percolation transition that phase 2 will pin down quantitatively.
"""

from __future__ import annotations

import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = CASE_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from forest_model import ForestFire, ForestParams
from plotting import plot_bars, plot_forest, plot_lines, mean, percentile

FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


REGIMES = [
    ("sub-critical (p=0.40)", 0.40, "#1f77b4"),
    ("near-critical (p=0.59)", 0.59, "#9467bd"),
    ("super-critical (p=0.75)", 0.75, "#d62728"),
]


def run_regime(density: float, n_seeds: int, L: int) -> list[int]:
    sizes: list[int] = []
    for seed in range(2026, 2026 + n_seeds):
        params = ForestParams(
            L=L, mode="static", density=density, ignition="center", seed=seed,
        )
        result = ForestFire(params).run()
        sizes.append(result.static_fire_size)
    return sizes


def main() -> None:
    L = 80
    n_seeds = 60
    all_sizes: dict[str, list[int]] = {}
    snapshots: dict[str, list[list[int]]] = {}

    for label, density, _ in REGIMES:
        all_sizes[label] = run_regime(density, n_seeds, L)
        # one representative snapshot per regime (post-fire)
        snap = ForestFire(
            ForestParams(L=L, mode="static", density=density,
                         ignition="center", seed=2026)
        ).run().final_grid
        snapshots[label] = snap
        plot_forest(
            FIG_DIR / f"phase1_post_fire_{int(density * 100):02d}.svg",
            snap,
            f"Phase 1 post-fire ({label})",
        )

    # 1) per-seed scatter, three regimes overlaid
    series = []
    for (label, density, color) in REGIMES:
        series.append({
            "x": list(range(n_seeds)),
            "y": all_sizes[label],
            "label": label,
            "color": color,
            "marker": "o",
            "linestyle": "",
            "alpha": 0.85,
        })
    plot_lines(
        FIG_DIR / "phase1_size_per_seed.svg",
        series=series,
        title="Phase 1: fire size per seed across three densities",
        xlabel="seed index",
        ylabel="trees burnt",
        logy=True,
    )

    # 2) summary bars: mean and 95th percentile fire size
    cats = [r[0] for r in REGIMES]
    means = [mean(all_sizes[r[0]]) for r in REGIMES]
    p95 = [percentile(all_sizes[r[0]], 95) for r in REGIMES]
    colors = [r[2] for r in REGIMES]
    plot_bars(
        FIG_DIR / "phase1_mean_fire_size.svg",
        categories=cats,
        values=means,
        title="Phase 1: mean fire size (60 seeds, L=80)",
        ylabel="<fire size>",
        colors=colors,
    )
    plot_bars(
        FIG_DIR / "phase1_p95_fire_size.svg",
        categories=cats,
        values=p95,
        title="Phase 1: 95th-percentile fire size — tail risk",
        ylabel="P95 fire size",
        colors=colors,
    )

    print("[phase1] regime summaries (mean, max, p95)")
    for label, _, _ in REGIMES:
        s = all_sizes[label]
        print(
            f"  {label:>26}: mean={mean(s):8.1f}  max={max(s):6d}  "
            f"p95={percentile(s, 95):8.1f}"
        )
    print("Phase 1 figures written to case5/figures/phase1_*.svg")


if __name__ == "__main__":
    main()
