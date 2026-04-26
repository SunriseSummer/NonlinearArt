"""Phase 2 reference implementation: self-organized criticality via adaptive alpha.

Instead of hand-tuning ``alpha`` (Phase 1), here the model adjusts it
on-the-fly with a slow proportional controller targeting a fixed mean
cascade size. Combined with quenched threshold heterogeneity (which
provides the spatial irregularity needed for non-trivial avalanches), the
system spontaneously locks onto a self-organized state whose statistics
are insensitive to the initial ``alpha`` and the random seed.

Three verifications are produced:

1. **Steady state.** Mean stress fluctuates around a stationary value while
   the controller drives ``alpha`` to a self-consistent fixed point.
2. **Heavy-tailed avalanche statistics.** Both the size and the duration
   distributions show wide log–log support, with similar slopes to the
   Phase-1 near-critical regime — but achieved without any manual sweep.
3. **Robustness to initial conditions.** Three runs differing only in
   ``seed`` and initial ``alpha`` collapse onto nearly the same statistics
   and the same late-time mean ``alpha``.

Usage::

    python case2/solution/phase2_solution.py

Outputs (under ``case2/figures/``):
    phase2_stress_and_alpha.svg
    phase2_avalanche_dist.svg
    phase2_robustness.svg
    phase2_stress_field.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly while reusing case2 base modules.
CASE2_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CASE2_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fault_model import FaultParams, FaultStressSystem
from plotting import (
    ccdf_log,
    log_hist,
    plot_dual_axis,
    plot_heatmap,
    plot_lines,
    power_law_mle,
)

FIG_DIR = CASE2_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _make_adaptive_params(*, alpha0: float, seed: int) -> FaultParams:
    return FaultParams(
        L=32,
        alpha=alpha0,
        threshold=1.0,
        drive_steps=12000,
        warmup=3000,
        seed=seed,
        adaptive=True,
        target_size=4.0,
        adapt_rate=8e-4,
        activity_window=250,
        alpha_min=0.10,
        alpha_max=0.245,
        heterogeneity=0.20,
    )


def run_phase2() -> None:
    """Enable adaptation and let the system self-organise near criticality."""

    # --- (1) main self-organisation run -----------------------------------
    main_params = _make_adaptive_params(alpha0=0.10, seed=2026)
    main_res = FaultStressSystem(main_params).run()

    plot_dual_axis(
        FIG_DIR / "phase2_stress_and_alpha.svg",
        x=list(range(len(main_res.mean_stress))),
        left={
            "y": main_res.mean_stress,
            "label": "Mean stress",
            "ylabel": "Mean stress per cell",
            "color": "#1f77b4",
        },
        right={
            "y": main_res.alpha_series,
            "label": "Adaptive alpha",
            "ylabel": "Conservation parameter alpha(t)",
            "color": "#9467bd",
        },
        title="Phase 2: stress and alpha co-evolve to a self-organised state",
        xlabel="Macro-step",
        vline=main_params.warmup,
    )

    # --- (2) heavy-tailed cascade-size and duration distributions ---------
    x_size, y_size = log_hist(main_res.sizes)
    x_dur, y_dur = log_hist(main_res.durations)
    x_ccdf, y_ccdf = ccdf_log(main_res.sizes)
    plot_lines(
        FIG_DIR / "phase2_avalanche_dist.svg",
        series=[
            {"x": x_size, "y": y_size, "label": "PDF P(s)",
             "color": "#e377c2", "marker": "o", "linestyle": ""},
            {"x": x_dur,  "y": y_dur,  "label": "PDF P(T)",
             "color": "#8c564b", "marker": "s", "linestyle": ""},
            {"x": x_ccdf, "y": y_ccdf, "label": "CCDF P(S>=s)",
             "color": "#2ca02c", "marker": "", "linestyle": "--"},
        ],
        title="Phase 2: avalanche size & duration after self-organisation",
        xlabel="s or T",
        ylabel="Probability density / CCDF",
        logx=True,
        logy=True,
    )

    # --- (3) robustness across seeds and initial alpha --------------------
    robustness_runs = [
        dict(alpha0=0.10, seed=2026, color="#1f77b4"),
        dict(alpha0=0.18, seed=99,   color="#ff7f0e"),
        dict(alpha0=0.235, seed=7,   color="#2ca02c"),
    ]
    series_alpha: list[dict] = []
    summaries: list[str] = []
    for cfg in robustness_runs:
        params = _make_adaptive_params(alpha0=cfg["alpha0"], seed=cfg["seed"])
        res = FaultStressSystem(params).run()
        series_alpha.append({
            "x": list(range(len(res.alpha_series))),
            "y": res.alpha_series,
            "label": f"alpha0={cfg['alpha0']:.3f}, seed={cfg['seed']}",
            "color": cfg["color"],
            "linewidth": 1.4,
            "alpha": 0.9,
        })
        # Late-time statistics: drop the first 60 % so the controller has
        # equilibrated.
        cut = int(0.6 * len(res.alpha_series))
        late_alpha = sum(res.alpha_series[cut:]) / max(len(res.alpha_series[cut:]), 1)
        if res.sizes:
            late_mean = sum(res.sizes) / len(res.sizes)
            tau, n = power_law_mle(res.sizes, smin=4)
        else:
            late_mean = 0.0
            tau, n = float("nan"), 0
        summaries.append(
            f"  alpha0={cfg['alpha0']:.3f} seed={cfg['seed']:>4d}  "
            f"-> late <alpha>={late_alpha:.3f}  <s>={late_mean:.2f}  "
            f"tau~{tau:.2f} (n={n})"
        )

    plot_lines(
        FIG_DIR / "phase2_robustness.svg",
        series=series_alpha,
        title="Phase 2: alpha trajectories converge from different starts",
        xlabel="Macro-step",
        ylabel="Adaptive alpha(t)",
    )

    # --- (4) final stress field heatmap (visual flair) --------------------
    plot_heatmap(
        FIG_DIR / "phase2_stress_field.svg",
        main_res.final_field,
        title="Phase 2: stress field in the self-organised state",
        cmap="magma",
        vmin=0.0,
        vmax=main_params.threshold + main_params.heterogeneity,
    )

    print("Phase 2 done. Figures written to:", FIG_DIR)
    print("Robustness summary (target_size=4.0):")
    for line in summaries:
        print(line)


if __name__ == "__main__":
    run_phase2()
