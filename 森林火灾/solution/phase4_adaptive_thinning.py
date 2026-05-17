"""Phase 4 (Case 5): adaptive thinning — keep timber, cut tail risk.

We compare two SOC forests with identical drivers ``p_grow=0.02`` and
``p_lightning=1e-4`` but different management policies:

* **laissez-faire** (``adaptive_thinning=False``): the system is left to
  organise itself; we get the canonical SOC fire-size distribution with a
  long power-law tail and occasional system-spanning megafires.
* **adaptive thinning** (``adaptive_thinning=True``): a local crew clears a
  small fraction of trees in any neighbourhood whose density exceeds a
  threshold, mimicking real-world firebreaks.  The forest stays "almost
  critical" but the heavy tail is cut.

Verification figures focus on operational metrics a forester actually cares
about:

* total trees burnt over the simulation horizon (tail risk);
* total trees harvested by thinning (timber yield);
* P95 / P99 fire size (the rare-but-disastrous events).
"""

from __future__ import annotations

import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = CASE_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from forest_model import ForestFire, ForestParams
from plotting import (
    mean, percentile, plot_bars, plot_lines,
    plot_loglog_distribution, rolling_mean,
)

FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run(adaptive: bool, seed: int, steps: int = 6000, L: int = 64):
    params = ForestParams(
        L=L,
        mode="soc",
        seed=seed,
        steps=steps,
        warmup=steps // 4,
        p_grow=0.02,
        p_lightning=1.0e-4,
        adaptive_thinning=adaptive,
        thinning_radius=3,
        thinning_threshold=0.62,
        thinning_rate=0.05,
    )
    return ForestFire(params).run(), params


def main() -> None:
    seeds = [2026, 2027, 2028]
    laissez_sizes: list[int] = []
    managed_sizes: list[int] = []
    laissez_burnt = 0
    managed_burnt = 0
    managed_thinned = 0

    # use the first seed for time-series plots
    laissez0, lp = run(False, seed=seeds[0])
    managed0, mp = run(True, seed=seeds[0])
    steps_axis = list(range(lp.steps))

    plot_lines(
        FIG_DIR / "phase4_density_compare.svg",
        series=[
            {
                "x": steps_axis,
                "y": rolling_mean(laissez0.tree_density, 25),
                "label": "laissez-faire",
                "color": "#d62728",
            },
            {
                "x": steps_axis,
                "y": rolling_mean(managed0.tree_density, 25),
                "label": "adaptive thinning",
                "color": "#2ca02c",
            },
        ],
        title="Phase 4: tree density under two policies",
        xlabel="sweep",
        ylabel="tree density",
    )
    plot_lines(
        FIG_DIR / "phase4_burning_compare.svg",
        series=[
            {
                "x": steps_axis,
                "y": laissez0.burning,
                "label": "laissez-faire",
                "color": "#d62728",
                "linewidth": 1.0,
            },
            {
                "x": steps_axis,
                "y": managed0.burning,
                "label": "adaptive thinning",
                "color": "#2ca02c",
                "linewidth": 1.0,
            },
        ],
        title="Phase 4: burning cells over time (note suppressed spikes)",
        xlabel="sweep",
        ylabel="burning cells",
    )

    for seed in seeds:
        l_res, _ = run(False, seed=seed)
        m_res, _ = run(True, seed=seed)
        l_sizes = [f.size for f in l_res.fires if f.size > 0]
        m_sizes = [f.size for f in m_res.fires if f.size > 0]
        laissez_sizes.extend(l_sizes)
        managed_sizes.extend(m_sizes)
        laissez_burnt += sum(l_sizes)
        managed_burnt += sum(m_sizes)
        managed_thinned += sum(m_res.thinned)

    # 1) fire-size distribution comparison
    plot_loglog_distribution(
        FIG_DIR / "phase4_size_distribution_compare.svg",
        datasets=[
            {"data": laissez_sizes, "label": "laissez-faire", "color": "#d62728"},
            {"data": managed_sizes, "label": "adaptive thinning", "color": "#2ca02c"},
        ],
        title="Phase 4: fire-size PDF — adaptive thinning trims the heavy tail",
        xlabel="fire size s",
        ylabel="P(s)",
        reference=(-1.15, 0.0, "guide: s^{-1.15}"),
    )

    # 2) operational summary bars: tail metrics + timber yield
    cats = ["laissez-faire", "adaptive thinning"]
    p95 = [percentile(laissez_sizes, 95), percentile(managed_sizes, 95)]
    p99 = [percentile(laissez_sizes, 99), percentile(managed_sizes, 99)]
    max_sz = [max(laissez_sizes) if laissez_sizes else 0,
              max(managed_sizes) if managed_sizes else 0]
    colors = ["#d62728", "#2ca02c"]

    plot_bars(
        FIG_DIR / "phase4_tail_p95.svg",
        categories=cats,
        values=p95,
        title="Phase 4: P95 fire size (lower is safer)",
        ylabel="P95 fire size",
        colors=colors,
    )
    plot_bars(
        FIG_DIR / "phase4_tail_p99.svg",
        categories=cats,
        values=p99,
        title="Phase 4: P99 fire size (lower is safer)",
        ylabel="P99 fire size",
        colors=colors,
    )
    plot_bars(
        FIG_DIR / "phase4_max_fire.svg",
        categories=cats,
        values=max_sz,
        title="Phase 4: largest fire observed",
        ylabel="max fire size",
        colors=colors,
    )

    # 3) timber accounting: burnt vs harvested
    plot_bars(
        FIG_DIR / "phase4_loss_vs_yield.svg",
        categories=["burnt (laissez-faire)", "burnt (managed)", "harvested (managed)"],
        values=[laissez_burnt, managed_burnt, managed_thinned],
        title="Phase 4: total trees lost to fire vs. harvested by thinning",
        ylabel="trees",
        colors=["#d62728", "#ff7f0e", "#2ca02c"],
    )

    print(f"[phase4] # fires: laissez={len(laissez_sizes)}, managed={len(managed_sizes)}")
    print(f"[phase4] burnt    : laissez={laissez_burnt}, managed={managed_burnt}")
    print(f"[phase4] harvested: managed={managed_thinned}")
    print(f"[phase4] P95: laissez={p95[0]:.1f}, managed={p95[1]:.1f}")
    print(f"[phase4] P99: laissez={p99[0]:.1f}, managed={p99[1]:.1f}")
    print(f"[phase4] max: laissez={max_sz[0]}, managed={max_sz[1]}")
    print("Phase 4 figures written to case5/figures/phase4_*.svg")


if __name__ == "__main__":
    main()
