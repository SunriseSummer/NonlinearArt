"""Phase 2 reference implementation: add adaptation for self-organized criticality.

Instead of hand-tuning ``spill_prob`` (phase 1), here the model adjusts it
on-the-fly with a simple proportional controller targeting a steady mean
load. The system should drift to a regime where avalanche statistics are
heavy-tailed, mimicking SOC.

The script produces three figures:

1. **Self-organization trajectory** — twin-axis time series of mean load and
   adaptive ``spill_prob`` for the canonical run.
2. **Avalanche statistics** — log-binned distributions of avalanche size and
   duration after the system has settled.
3. **Robustness check** — re-runs the same SOC mechanism with several
   ``(seed, initial spill_prob)`` combinations and overlays their mean-load
   trajectories, then prints the steady-state means so we can verify the
   system converges to (almost) the same operating point regardless of
   initial conditions. This piece used to be missing from the reference
   solution; without it the third validation requirement of phase 2
   (rule "鲁棒性") was only argued for in prose.

Usage::

    python case1/solution/phase2_solution.py

Outputs:
    case1/figures/phase2_density_and_spillprob.svg
    case1/figures/phase2_avalanche_dist.svg
    case1/figures/phase2_robustness.svg
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


# Canonical configuration used both for the main demonstration and as the
# template for the robustness sweep — only ``seed`` and ``spill_prob`` change
# between robustness runs.
CANONICAL = dict(
    L=24,
    threshold=6,
    dissipation=0.20,
    steps=7000,
    warmup=1000,
    adaptive=True,
    target_load=2.6,
    adapt_rate=0.020,
    spill_min=0.05,
    spill_max=0.45,
)


def _make_params(*, seed: int, spill_prob: float) -> TrafficParams:
    return TrafficParams(seed=seed, spill_prob=spill_prob, **CANONICAL)


def _steady_mean(values: list[float], warmup: int) -> float:
    """Mean of ``values`` after the warmup window — i.e. the steady state."""
    tail = values[warmup:]
    return sum(tail) / max(len(tail), 1)


def run_phase2() -> None:
    """Enable adaptation, demonstrate SOC, and validate robustness."""
    # --- (1) canonical SOC run -------------------------------------------
    params = _make_params(seed=2026, spill_prob=0.10)
    res = TrafficCascadeSystem(params).run()

    # Mean load and adaptive spill probability live on very different scales,
    # so we plot them on a twin y-axis.
    plot_dual_axis(
        FIG_DIR / "phase2_density_and_spillprob.svg",
        x=list(range(len(res.densities))),
        left={
            "y": res.densities,
            "label": "Mean load",
            "ylabel": "Mean load per intersection",
            "color": "#1f77b4",
        },
        right={
            "y": res.spill_prob_series,
            "label": "Adaptive spill probability",
            "ylabel": "Spill probability p(t)",
            "color": "#9467bd",
        },
        title="Phase 2: self-organization of load and control parameter",
        xlabel="Simulation step",
        vline=params.warmup,
    )

    x_size, y_size = log_hist(res.avalanche_sizes)
    x_dur, y_dur = log_hist(res.avalanche_durations)
    plot_lines(
        FIG_DIR / "phase2_avalanche_dist.svg",
        series=[
            {"x": x_size, "y": y_size, "label": "Avalanche size s",
             "color": "#e377c2", "marker": "o"},
            {"x": x_dur, "y": y_dur, "label": "Avalanche duration T",
             "color": "#8c564b", "marker": "s"},
        ],
        title="Phase 2: avalanche statistics after self-organization",
        xlabel="s or T",
        ylabel="P",
        logx=True,
        logy=True,
    )

    # --- (2) robustness sweep --------------------------------------------
    # Vary both the RNG seed and the *initial* spill_prob (well below and
    # well above target). If the proportional controller really self-
    # organises the system, every trajectory should converge to nearly the
    # same operating point — that is exactly what rule 3 ("鲁棒性") asks for.
    robustness_configs = [
        ("seed=2026, p0=0.10", 2026, 0.10, "#1f77b4"),
        ("seed=17,   p0=0.10", 17,   0.10, "#2ca02c"),
        ("seed=991,  p0=0.40", 991,  0.40, "#d62728"),
        ("seed=4242, p0=0.30", 4242, 0.30, "#9467bd"),
    ]
    print("[phase2] robustness sweep — steady-state means after warmup:")
    print("  config                 <load>     <spill_prob>")

    series = []
    for label, seed, p0, color in robustness_configs:
        rparams = _make_params(seed=seed, spill_prob=p0)
        rres = TrafficCascadeSystem(rparams).run()
        load_ss = _steady_mean(rres.densities, rparams.warmup)
        spill_ss = _steady_mean(rres.spill_prob_series, rparams.warmup)
        print(f"  {label:<22} {load_ss:7.3f}    {spill_ss:7.3f}")
        series.append({
            "x": list(range(len(rres.densities))),
            "y": rres.densities,
            "label": f"{label}  (load_ss={load_ss:.2f})",
            "color": color,
            "linewidth": 1.2,
        })

    plot_lines(
        FIG_DIR / "phase2_robustness.svg",
        series=series,
        title=("Phase 2: robustness — different (seed, initial p) "
               "converge to the same SOC state"),
        xlabel="Simulation step",
        ylabel="Mean load per intersection",
        vline=CANONICAL["warmup"],
    )


if __name__ == "__main__":
    run_phase2()
    print(f"Phase 2 done. Figures written to {FIG_DIR}")
