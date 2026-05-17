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

Outputs (operational story):
    case1b/figures/phase2_demand_sweep.svg      (load + throughput vs inflow)
    case1b/figures/phase2_regime_compare.svg    (time series in three regimes)
    case1b/figures/phase2_avalanche_dist.svg    (cascade-size distributions)

Outputs (criticality diagnostics — answer the reviewer's question
"how do we *prove* this is critical?"):
    case1b/figures/phase2_susceptibility.svg     (chi(p) peak at p_c)
    case1b/figures/phase2_powerlaw_fit.svg       (P(s)~s^-tau, P(T)~T^-alpha)
    case1b/figures/phase2_finite_size_scaling.svg (cutoff grows with L)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CASE_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from plotting import (
    log_hist,
    plot_dual_axis,
    plot_lines,
    power_law_fit,
    rolling_mean,
)
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


# ---------------------------------------------------------------------------
# Criticality diagnostics
# ---------------------------------------------------------------------------
#
# The "operational" sweep above tells the rush-hour story but its strongly
# dissipative regime cannot, by itself, prove that the network is critical.
# The functions below switch the model into the BTW-like regime
# (``threshold=4``, ``spill_prob`` near 1, very small ``dissipation``) where
# the same simulator exhibits the textbook signatures of a critical phase
# transition:
#
#   1. A susceptibility peak ``chi(p) = var(s)/<s>`` at the critical
#      ``spill_prob`` ``p_c``.
#   2. A heavy-tailed avalanche-size distribution at ``p_c`` that is well
#      fitted by ``P(s) ~ s^(-tau)`` over multiple decades.
#   3. Finite-size scaling: as ``L`` grows, the cutoff of the avalanche
#      distribution moves to larger sizes. This is the standard finite-size
#      check that the heavy tail is *not* an artefact of one grid size.
#
# These diagnostics are what the reviewer asked for in PR comment
# #4328219231 — they go beyond "load and throughput look different in
# different regimes" and quantitatively show the system is at criticality.


CRIT_DISS = 0.05      # close to BTW limit, but a touch of dissipation makes
                      # the model run quickly to a steady state
CRIT_THRESHOLD = 4    # classic BTW (one unit of load shed per neighbour)
CRIT_INFLOW = 1.0     # one car per step, in line with the original BTW model


def _btw_run(*, L: int, p: float, seed: int, steps: int, warmup: int):
    return RushHourTrafficSystem(TrafficParams(
        L=L, threshold=CRIT_THRESHOLD,
        inflow_rate=CRIT_INFLOW,
        spill_prob=p, dissipation=CRIT_DISS,
        steps=steps, warmup=warmup, seed=seed,
    )).run()


def _susceptibility(sizes: list[int]) -> tuple[float, float]:
    """Return ``(<s>, chi)`` with ``chi = var(s) / <s>``."""
    if not sizes:
        return 0.0, 0.0
    n = len(sizes)
    m = sum(sizes) / n
    if m <= 0:
        return 0.0, 0.0
    var = sum((s - m) ** 2 for s in sizes) / n
    return m, var / m


def run_phase2_criticality_diagnostics() -> None:
    """Prove the network is critical — susceptibility peak + power-law fit
    + finite-size scaling. All three are run with the same simulator that
    powers the operational sweep, just with BTW-friendly parameters.
    """
    # -----------------------------------------------------------------
    # (A) Susceptibility curve: chi(p) = var(s)/<s>
    # -----------------------------------------------------------------
    p_grid = [round(0.50 + 0.05 * i, 2) for i in range(11)]  # 0.50 .. 1.00
    seeds = [2026, 17, 991]
    L_main = 24
    steps_main, warmup_main = 7000, 1500

    print("[phase2/diag] BTW-regime spill_prob sweep "
          f"(L={L_main}, threshold={CRIT_THRESHOLD}, dissipation={CRIT_DISS})")
    print("  p     <s>      chi=var/<s>   max_s")

    means, chis, max_sizes = [], [], []
    for p in p_grid:
        all_sizes = []
        for seed in seeds:
            res = _btw_run(L=L_main, p=p, seed=seed,
                           steps=steps_main, warmup=warmup_main)
            all_sizes.extend(res.avalanche_sizes)
        m, chi = _susceptibility(all_sizes)
        means.append(m)
        chis.append(chi)
        max_sizes.append(max(all_sizes) if all_sizes else 0)
        print(f"  {p:0.2f}  {m:6.2f}   {chi:8.3f}      {max_sizes[-1]:5d}")

    # p_c is where the susceptibility peaks (textbook definition).
    p_c = p_grid[max(range(len(chis)), key=lambda i: chis[i])]
    print(f"[phase2/diag] susceptibility peak at p_c = {p_c:.2f}")

    plot_dual_axis(
        FIG_DIR / "phase2_susceptibility.svg",
        x=p_grid,
        left={
            "y": means,
            "label": "Mean cascade size <s>",
            "ylabel": "<s>",
            "color": "#1f77b4",
        },
        right={
            "y": chis,
            "label": "Susceptibility chi = var(s)/<s>",
            "ylabel": "chi",
            "color": "#d62728",
        },
        title=("Phase 2 diagnostics: susceptibility peaks at the "
               f"critical spill probability p_c = {p_c:.2f}"),
        xlabel="Spill probability p",
        vlines=[(p_c, "#444444", f"p_c = {p_c:.2f}")],
    )

    # -----------------------------------------------------------------
    # (B) Power-law fit at p_c
    # -----------------------------------------------------------------
    # Pool many seeds at p_c for a statistically meaningful fit.
    crit_sizes: list[int] = []
    crit_durs: list[int] = []
    for seed in seeds + [4242, 123]:
        res = _btw_run(L=L_main, p=p_c, seed=seed,
                       steps=steps_main, warmup=warmup_main)
        crit_sizes.extend(res.avalanche_sizes)
        crit_durs.extend(res.avalanche_durations)

    xs, ys = log_hist(crit_sizes, bins=24)
    xd, yd = log_hist(crit_durs, bins=20)

    # Fit the body of the distribution (excluding the very first bin
    # which always carries the unit-cascade spike, and the last bin which
    # is finite-size-cutoff dominated).
    def _fit_range(xs_):
        if len(xs_) < 4:
            return None, None
        return xs_[1], xs_[-2]

    fit_lo, fit_hi = _fit_range(xs)
    tau, b, r2 = power_law_fit(xs, ys, x_min=fit_lo, x_max=fit_hi)
    fit_lo_d, fit_hi_d = _fit_range(xd)
    alpha, b_d, r2_d = power_law_fit(xd, yd, x_min=fit_lo_d, x_max=fit_hi_d)
    print(f"[phase2/diag] power-law fit at p_c={p_c:.2f}: "
          f"P(s)~s^-{tau:.2f}  (R^2={r2:.3f}),  "
          f"P(T)~T^-{alpha:.2f}  (R^2={r2_d:.3f})")

    series = []
    if xs:
        series.append({"x": xs, "y": ys,
                       "label": f"P(s) at p_c={p_c:.2f}",
                       "color": "#1f77b4", "marker": "o", "linestyle": ""})
    if xs and tau > 0 and fit_lo and fit_hi:
        fit_x = [fit_lo, fit_hi]
        fit_y = [10 ** (b - tau * math.log10(v)) for v in fit_x]
        series.append({"x": fit_x, "y": fit_y,
                       "label": f"fit: P(s)~s^-{tau:.2f}, R^2={r2:.2f}",
                       "color": "#1f77b4", "linestyle": "--", "linewidth": 1.5})
    if xd:
        series.append({"x": xd, "y": yd,
                       "label": f"P(T) at p_c={p_c:.2f}",
                       "color": "#e377c2", "marker": "s", "linestyle": ""})
    if xd and alpha > 0 and fit_lo_d and fit_hi_d:
        fit_x = [fit_lo_d, fit_hi_d]
        fit_y = [10 ** (b_d - alpha * math.log10(v)) for v in fit_x]
        series.append({"x": fit_x, "y": fit_y,
                       "label": f"fit: P(T)~T^-{alpha:.2f}, R^2={r2_d:.2f}",
                       "color": "#e377c2", "linestyle": "--", "linewidth": 1.5})

    plot_lines(
        FIG_DIR / "phase2_powerlaw_fit.svg",
        series=series,
        title=f"Phase 2 diagnostics: power-law avalanches at p_c = {p_c:.2f}",
        xlabel="Cascade size s or duration T",
        ylabel="Probability density",
        logx=True, logy=True,
    )

    # -----------------------------------------------------------------
    # (C) Finite-size scaling
    # -----------------------------------------------------------------
    # If P(s) really is power-law, increasing L should shift the cutoff
    # to larger s (cutoff ~ L^D with D some fractal dimension). Showing
    # three sizes overlaid is the canonical way to demonstrate this.
    sizes_by_L: dict[int, list[int]] = {}
    for L_v in (16, 24, 32):
        pool: list[int] = []
        for seed in seeds:
            res = _btw_run(L=L_v, p=p_c, seed=seed,
                           steps=steps_main, warmup=warmup_main)
            pool.extend(res.avalanche_sizes)
        sizes_by_L[L_v] = pool

    fs_series = []
    palette = {16: "#2ca02c", 24: "#ff7f0e", 32: "#9467bd"}
    print("[phase2/diag] finite-size scaling at p_c:")
    print("  L     N      <s>       max_s")
    for L_v in (16, 24, 32):
        pool = sizes_by_L[L_v]
        x, y = log_hist(pool, bins=20)
        m, _chi = _susceptibility(pool)
        print(f"  {L_v:2d}  {len(pool):5d}   {m:6.2f}    {max(pool) if pool else 0:5d}")
        if x:
            fs_series.append({
                "x": x, "y": y,
                "label": f"L={L_v}  (max s={max(pool)})",
                "color": palette[L_v], "marker": "o",
            })

    plot_lines(
        FIG_DIR / "phase2_finite_size_scaling.svg",
        series=fs_series,
        title=("Phase 2 diagnostics: finite-size scaling at p_c "
               "(cutoff grows with L)"),
        xlabel="Cascade size s",
        ylabel="P(s)",
        logx=True, logy=True,
    )


if __name__ == "__main__":
    run_phase2()
    run_phase2_criticality_diagnostics()
    print(f"Phase 2 done. Figures written to {FIG_DIR}")
