"""Phase 2 (near critical): rising rush-hour demand drives the network to its capacity edge.

This phase tells the story directly from the operator's perspective.
``spill_prob`` is held fixed at a moderately aggressive routing policy and
we sweep ``inflow_rate`` (the demand on the network). Three things happen
in sequence as demand grows:

1. **Below capacity** — every vehicle is served, load is small, the network
   feels stable but most of the road capacity is unused.
2. **Near capacity** — load and congestion-propagation range start to grow
   non-linearly while throughput is still close to the inflow. The network
   is sensitive: small demand bumps trigger noticeable cascades.
3. **Above capacity** — load diverges, throughput plateaus near the
   network's "service rate". Adding more cars no longer moves more cars.

We then pick three operating points (sub / near / super critical) and show
their load + throughput trajectories side by side. This makes the trade-off
"high efficiency vs high fragility" tangible — exactly the lesson
``全新设计.md`` asks the player to discover.

Outputs:
    case1b/figures/phase2_demand_sweep.svg      (load + throughput vs inflow)
    case1b/figures/phase2_regime_compare.svg    (time series in three regimes)
    case1b/figures/phase2_avalanche_dist.svg    (cascade-size distributions)
"""

from __future__ import annotations

import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CASE_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from plotting import log_hist, plot_dual_axis, plot_lines, rolling_mean
from traffic_model import RushHourTrafficSystem, TrafficParams

FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


# Common parameters. ``spill_prob`` is fixed: the dispatcher has settled on
# a fairly aggressive routing policy and now demand keeps growing.
SPILL = 0.30
DISSIPATION = 0.18
L = 20
THRESHOLD = 6
STEPS = 4500
WARMUP = 800


def _sweep(inflows, seeds):
    loads, thrpts, congs, mean_sizes = [], [], [], []
    for lam in inflows:
        per_seed = []
        for seed in seeds:
            params = TrafficParams(
                L=L, threshold=THRESHOLD,
                inflow_rate=lam, spill_prob=SPILL, dissipation=DISSIPATION,
                steps=STEPS, warmup=WARMUP, seed=seed,
            )
            res = RushHourTrafficSystem(params).run()
            tail_d = res.densities[WARMUP:]
            tail_t = res.throughput[WARMUP:]
            tail_c = res.congestion_range[WARMUP:]
            mean_av = _mean(res.avalanche_sizes) if res.avalanche_sizes else 0.0
            per_seed.append((_mean(tail_d), _mean(tail_t), _mean(tail_c), mean_av))
        loads.append(_mean([s[0] for s in per_seed]))
        thrpts.append(_mean([s[1] for s in per_seed]))
        congs.append(_mean([s[2] for s in per_seed]))
        mean_sizes.append(_mean([s[3] for s in per_seed]))
    return loads, thrpts, congs, mean_sizes


def _detect_capacity(inflows, loads, thrpts, congs):
    """Return the inflow that sits closest to the network's stress edge.

    Operationally we define the stress edge as the smallest inflow whose
    average congestion-propagation range first exceeds 0.25 — i.e. at
    least one in four steps has a cell sitting at the topple threshold,
    which is when cascades become visible to a human dispatcher. Falling
    back to the highest swept inflow keeps the figure annotation sane on
    very calm sweeps.
    """
    for lam, cg in zip(inflows, congs):
        if cg >= 0.25:
            return lam
    return inflows[-1]


def run_phase2() -> None:
    inflows = [round(0.4 + 0.2 * i, 2) for i in range(11)]  # 0.4 .. 2.4
    seeds = [2026, 17, 991]
    loads, thrpts, congs, mean_sizes = _sweep(inflows, seeds)

    stress_edge = _detect_capacity(inflows, loads, thrpts, congs)

    print(f"[phase2] inflow sweep at spill_prob={SPILL}, dissipation={DISSIPATION},"
          f" averaged over {len(seeds)} seeds:")
    print("  inflow   <load>   <thrpt>   <cong_range>   <s>")
    for lam, ld, th, cg, ms in zip(inflows, loads, thrpts, congs, mean_sizes):
        marker = "  <-- stress edge" if lam == stress_edge else ""
        print(f"  {lam:0.2f}    {ld:6.3f}   {th:6.3f}    {cg:8.3f}     {ms:.2f}{marker}")
    print(f"[phase2] detected stress edge inflow ~= {stress_edge:.2f}")

    # --- (1) demand sweep figure -----------------------------------------
    plot_dual_axis(
        FIG_DIR / "phase2_demand_sweep.svg",
        x=inflows,
        left={
            "y": loads,
            "label": "Mean load",
            "ylabel": "Mean load per intersection",
            "color": "#1f77b4",
        },
        right={
            "y": thrpts,
            "label": "Throughput",
            "ylabel": "Throughput (vehicles / step)",
            "color": "#ff7f0e",
        },
        title=("Phase 2: demand vs network response - "
               f"stress edge ~= {stress_edge:.2f} cars/step"),
        xlabel="Inflow rate (cars / step)",
        vlines=[(stress_edge, "#444444", f"stress edge ~= {stress_edge:.2f}")],
    )

    # --- (2) three-regime time series ------------------------------------
    p_min, p_max = inflows[0], inflows[-1]
    inflow_sub = max(p_min, round(stress_edge - 0.6, 2))
    inflow_super = min(p_max, round(stress_edge + 0.6, 2))
    regimes = [
        (f"Subcritical inflow={inflow_sub:.2f}", inflow_sub, "#2ca02c"),
        (f"Near-critical inflow={stress_edge:.2f}", stress_edge, "#ff7f0e"),
        (f"Supercritical inflow={inflow_super:.2f}", inflow_super, "#d62728"),
    ]

    series_load, series_thr, series_dist = [], [], []
    for label, lam, color in regimes:
        params = TrafficParams(
            L=L, threshold=THRESHOLD,
            inflow_rate=lam, spill_prob=SPILL, dissipation=DISSIPATION,
            steps=6000, warmup=1000, seed=2026,
        )
        res = RushHourTrafficSystem(params).run()
        xs = list(range(len(res.densities)))
        series_load.append({
            "x": xs,
            "y": rolling_mean(res.densities, 80),
            "label": f"{label} -> load",
            "color": color, "linewidth": 1.4,
        })
        series_thr.append({
            "x": xs,
            "y": rolling_mean(res.throughput, 80),
            "label": f"{label} -> throughput",
            "color": color, "linestyle": "--", "linewidth": 1.2, "alpha": 0.85,
        })
        x, y = log_hist(res.avalanche_sizes)
        if x:
            series_dist.append({
                "x": x, "y": y,
                "label": label, "color": color, "marker": "o",
            })

    plot_lines(
        FIG_DIR / "phase2_regime_compare.svg",
        series=series_load + series_thr,
        title=("Phase 2: three regimes - efficiency vs fragility "
               "(solid = load, dashed = throughput)"),
        xlabel="Simulation step",
        ylabel="Load (solid) / throughput (dashed)",
        vlines=[(1000, "#b00000", "warmup=1000")],
    )

    if series_dist:
        plot_lines(
            FIG_DIR / "phase2_avalanche_dist.svg",
            series=series_dist,
            title="Phase 2: cascade-size distributions across the three regimes",
            xlabel="Cascade size s",
            ylabel="P(s)",
            logx=True, logy=True,
        )


if __name__ == "__main__":
    run_phase2()
    print(f"Phase 2 done. Figures written to {FIG_DIR}")
