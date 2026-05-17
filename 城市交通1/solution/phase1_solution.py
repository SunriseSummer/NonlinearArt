"""Phase 1 reference implementation: tune the traffic system to near-criticality.

The script demonstrates the **full search procedure** for the critical
``spill_prob`` — the previous version simply hard-coded ``p_c = 0.26`` and the
three comparison points, which hid the most important step of the solution.
The current flow is:

1. **Sweep** ``spill_prob`` while collecting two diagnostics per parameter:
   the *mean* avalanche size (a textbook order parameter) and the
   *susceptibility* ``var(s)/mean(s)`` (a noise-tolerant indicator that peaks
   near the critical point). Each ``p`` is averaged over several seeds so the
   resulting curves are smooth enough to be read directly from the figure.
2. **Locate** ``p_c`` programmatically as the peak of the (lightly smoothed)
   susceptibility curve. The detected value drives both the reference line on
   the sweep figure and the choice of comparison points in step 3.
3. **Compare** avalanche-size distributions in three regimes — sub-critical,
   near-critical (= the detected ``p_c``) and super-critical — using offsets
   from the detected ``p_c`` so the comparison is data-driven.

Usage::

    python case1/solution/phase1_solution.py

Outputs:
    case1/figures/phase1_mean_size_vs_spill_prob.svg
    case1/figures/phase1_size_dist_compare.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly while reusing case1 base modules.
CASE1_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CASE1_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from plotting import log_hist, plot_dual_axis, plot_lines
from traffic_model import TrafficCascadeSystem, TrafficParams

FIG_DIR = CASE1_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


def _avalanche_stats(sizes: list[int]) -> tuple[float, float]:
    """Return ``(mean, susceptibility)`` for one run's avalanche-size list.

    Susceptibility is defined as ``var(s) / mean(s)`` — it is dimensionless,
    finite for sub-critical runs, and diverges at criticality in the
    thermodynamic limit. On a finite grid it shows a clear peak that we use as
    the search criterion.
    """
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0
    mean = sum(sizes) / n
    var = sum((s - mean) ** 2 for s in sizes) / n
    return mean, var / max(mean, 1e-9)


def _sweep_spill_prob(
    p_values: list[float], seeds: list[int]
) -> tuple[list[float], list[float]]:
    """Run a multi-seed sweep, returning ``(mean_sizes, susceptibilities)``."""
    mean_sizes: list[float] = []
    suscs: list[float] = []
    for p in p_values:
        per_seed_mean: list[float] = []
        per_seed_susc: list[float] = []
        for seed in seeds:
            params = TrafficParams(
                L=24,
                threshold=6,
                spill_prob=p,
                dissipation=0.20,
                steps=3500,
                warmup=800,
                seed=seed,
                adaptive=False,
            )
            res = TrafficCascadeSystem(params).run()
            m, chi = _avalanche_stats(res.avalanche_sizes)
            per_seed_mean.append(m)
            per_seed_susc.append(chi)
        mean_sizes.append(sum(per_seed_mean) / len(per_seed_mean))
        suscs.append(sum(per_seed_susc) / len(per_seed_susc))
    return mean_sizes, suscs


def _smooth(values: list[float]) -> list[float]:
    """3-point moving average, edges held fixed."""
    if len(values) < 3:
        return list(values)
    out = [values[0]]
    for i in range(1, len(values) - 1):
        out.append((values[i - 1] + values[i] + values[i + 1]) / 3.0)
    out.append(values[-1])
    return out


def _locate_critical(p_values: list[float], susc: list[float]) -> float:
    """Pick the ``p`` where the (smoothed) susceptibility is largest."""
    smoothed = _smooth(susc)
    idx = max(range(len(smoothed)), key=lambda i: smoothed[i])
    return p_values[idx]


def _pick_comparison(
    p_c: float, p_values: list[float]
) -> list[tuple[str, float, str]]:
    """Build the (label, p, color) triple around the detected ``p_c``.

    Offsets are clamped to the swept range so that the comparison points are
    always taken from parameters we actually have evidence for.
    """
    p_min, p_max = p_values[0], p_values[-1]
    p_sub = max(p_min, round(p_c - 0.12, 2))
    p_super = min(p_max, round(p_c + 0.08, 2))
    p_near = round(p_c, 2)
    return [
        (f"Subcritical p={p_sub:.2f}", p_sub, "#2ca02c"),
        (f"Near-critical p={p_near:.2f}", p_near, "#ff7f0e"),
        (f"Supercritical p={p_super:.2f}", p_super, "#d62728"),
    ]


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_phase1() -> None:
    """Search for the critical ``spill_prob`` and validate near-critical behaviour."""
    # --- (1) order-parameter sweep + susceptibility ------------------------
    p_values = [round(0.08 + 0.02 * i, 2) for i in range(15)]  # 0.08 .. 0.36
    seeds = [2026, 17, 991, 4242]
    mean_sizes, suscs = _sweep_spill_prob(p_values, seeds)

    # --- (2) locate the critical point from the sweep ----------------------
    p_c = _locate_critical(p_values, suscs)

    print("[phase1] spill_prob sweep (averaged over "
          f"{len(seeds)} seeds):")
    print("  p      mean_size   susceptibility")
    for p, m, chi in zip(p_values, mean_sizes, suscs):
        marker = "  <-- p_c" if p == p_c else ""
        print(f"  {p:0.2f}    {m:7.3f}      {chi:7.3f}{marker}")
    print(f"[phase1] detected critical spill probability p_c ≈ {p_c:.2f}")

    plot_dual_axis(
        FIG_DIR / "phase1_mean_size_vs_spill_prob.svg",
        x=p_values,
        left={
            "y": mean_sizes,
            "label": "Mean avalanche size <s>",
            "ylabel": "Mean avalanche size <s>",
            "color": "#1f77b4",
        },
        right={
            "y": suscs,
            "label": "Susceptibility var(s)/<s>",
            "ylabel": "Susceptibility var(s)/<s>",
            "color": "#9467bd",
        },
        title=("Phase 1: searching for criticality "
               f"(detected p_c ≈ {p_c:.2f})"),
        xlabel="Spill probability p",
        vline=p_c,
    )

    # --- (3) three-regime distribution comparison --------------------------
    compare = _pick_comparison(p_c, p_values)
    series = []
    for label, p, color in compare:
        params = TrafficParams(
            L=24,
            threshold=6,
            spill_prob=p,
            dissipation=0.20,
            steps=5500,
            warmup=1000,
            seed=2026,
            adaptive=False,
        )
        res = TrafficCascadeSystem(params).run()
        x, y = log_hist(res.avalanche_sizes)
        series.append({
            "x": x,
            "y": y,
            "label": label,
            "color": color,
            "marker": "o",
        })

    plot_lines(
        FIG_DIR / "phase1_size_dist_compare.svg",
        series=series,
        title=("Phase 1: avalanche-size distributions around "
               f"the detected p_c ≈ {p_c:.2f}"),
        xlabel="Cascade size s",
        ylabel="P(s)",
        logx=True,
        logy=True,
    )


if __name__ == "__main__":
    run_phase1()
    print(f"Phase 1 done. Figures written to {FIG_DIR}")
