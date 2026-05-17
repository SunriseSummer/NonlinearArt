"""Phase 3 (Case 5): self-organised criticality via slow growth + lightning.

We switch to the Drossel--Schwabl forest fire model: every empty cell grows a
tree with probability ``p_grow``, every tree is hit by lightning with
probability ``p_lightning``.  When ``p_lightning << p_grow << 1`` the system
self-organises to a stationary state with a power-law fire-size distribution
``P(s) ~ s^{-tau}`` (no manual density tuning required).

We compare three regimes of the time-scale separation ``f/p`` ratio:

* large ratio  -> almost no separation, fires happen before forest can rebuild
* "SOC-like" intermediate ratio -> clean power-law tail
* tiny ratio   -> very slow driving, sparse but long-tailed fires

We also compare the SOC fire-size distribution against the static
super-critical fire-size distribution from phase 1: the SOC system reaches a
similar tail *without* anyone ever choosing a critical density.
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
    mean, plot_lines, plot_loglog_distribution, rolling_mean,
)

FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_soc(p_grow: float, p_lightning: float, steps: int, L: int, seed: int):
    params = ForestParams(
        L=L,
        mode="soc",
        seed=seed,
        steps=steps,
        warmup=steps // 4,
        p_grow=p_grow,
        p_lightning=p_lightning,
    )
    return ForestFire(params).run(), params


def main() -> None:
    L = 64
    steps = 4000

    # 1) main SOC run: clean time-scale separation
    soc_result, soc_params = run_soc(p_grow=0.02, p_lightning=1.0e-4,
                                     steps=steps, L=L, seed=2026)
    sizes_soc = [f.size for f in soc_result.fires if f.size > 0]
    durations_soc = [f.duration for f in soc_result.fires if f.duration > 0]
    print(f"[phase3] SOC run: {len(sizes_soc)} fires, "
          f"<size>={mean(sizes_soc):.1f}, max={max(sizes_soc) if sizes_soc else 0}")
    print(f"[phase3] steady tree density ~= "
          f"{mean(soc_result.tree_density[soc_params.warmup:]):.3f}")

    # 2) tree density and active-burning time series
    steps_axis = list(range(soc_params.steps))
    plot_lines(
        FIG_DIR / "phase3_density_timeseries.svg",
        series=[{
            "x": steps_axis,
            "y": rolling_mean(soc_result.tree_density, 25),
            "label": "tree density",
            "color": "#2ca02c",
        }],
        title="Phase 3 SOC run: tree density self-organises to a steady band",
        xlabel="sweep",
        ylabel="tree density",
        vlines=[(soc_params.warmup, "#b00000", f"warmup={soc_params.warmup}")],
    )
    plot_lines(
        FIG_DIR / "phase3_burning_timeseries.svg",
        series=[{
            "x": steps_axis,
            "y": soc_result.burning,
            "label": "burning cells",
            "color": "#d62728",
            "linewidth": 1.0,
        }],
        title="Phase 3 SOC run: bursty fire activity (note rare big spikes)",
        xlabel="sweep",
        ylabel="burning cells",
    )

    # 3) sweep f/p ratio: three SOC runs at different driving ratios
    runs = [
        ("f/p=5e-2 (poor separation)", 0.02, 1.0e-3, "#1f77b4"),
        ("f/p=5e-3 (SOC-like)",        0.02, 1.0e-4, "#9467bd"),
        ("f/p=5e-4 (very slow)",       0.02, 1.0e-5, "#d62728"),
    ]
    datasets = []
    for label, pg, pl, color in runs:
        result, _ = run_soc(p_grow=pg, p_lightning=pl, steps=6000, L=L, seed=2026)
        sizes = [f.size for f in result.fires if f.size > 0]
        datasets.append({"data": sizes, "label": label, "color": color})
        print(f"[phase3] {label}: n_fires={len(sizes)}, "
              f"max_size={max(sizes) if sizes else 0}, "
              f"<size>={mean(sizes):.1f}")

    plot_loglog_distribution(
        FIG_DIR / "phase3_size_distribution.svg",
        datasets=datasets,
        title="Phase 3: fire-size distribution at three f/p ratios",
        xlabel="fire size s",
        ylabel="P(s) (log-binned PDF)",
        reference=(-1.15, 0.0, "guide: s^{-1.15}"),
    )

    # 4) duration distribution for the main SOC run
    plot_loglog_distribution(
        FIG_DIR / "phase3_duration_distribution.svg",
        datasets=[{
            "data": durations_soc,
            "label": "fire duration (SOC)",
            "color": "#9467bd",
        }],
        title="Phase 3: fire-duration distribution (SOC)",
        xlabel="fire duration T",
        ylabel="P(T) (log-binned PDF)",
        reference=(-1.40, 0.0, "guide: T^{-1.40}"),
    )

    print("Phase 3 figures written to case5/figures/phase3_*.svg")


if __name__ == "__main__":
    main()
