"""Phase 1 reference implementation: tune the fault system to near-criticality.

Three coordinated experiments on top of the base ``FaultStressSystem``:

1. **Order-parameter sweep.** Scan ``alpha`` and track the mean cascade
   size. The mean cascade size diverges with system size at the OFC
   critical point — even on a finite 32×32 grid it grows by more than an
   order of magnitude as ``alpha`` approaches ``alpha_c``.
2. **Three-regime cascade-size distribution.** Compare the size CCDF at
   sub-critical, near-critical and super-critical settings. Only the
   near-critical curve shows the straight power-law tail predicted by
   Gutenberg–Richter; the player can also read off a magnitude exponent
   ``b`` via the Hill estimator (printed at the end of the run).
3. **Omori aftershock signal.** Stack the post-mainshock activity around
   the largest 5% of cascades and check that the rate decays roughly as
   ``1/Δt`` near criticality and is much flatter sub-critically.

Usage::

    python case2/solution/phase1_solution.py

Outputs (under ``case2/figures/``):
    phase1_mean_size_vs_alpha.svg
    phase1_size_dist_compare.svg
    phase1_omori.svg
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
from plotting import aftershock_rate, ccdf_log, plot_lines, power_law_mle

FIG_DIR = CASE2_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_phase1() -> None:
    """Tune the conservation parameter and validate near-critical behaviour."""

    # --- (1) order-parameter sweep ----------------------------------------
    alphas = [0.10 + 0.01 * i for i in range(15)]  # 0.10 .. 0.24 (15 points)
    mean_sizes: list[float] = []

    for a in alphas:
        params = FaultParams(
            L=32,
            alpha=a,
            threshold=1.0,
            drive_steps=4000,
            warmup=1000,
            seed=2026,
        )
        res = FaultStressSystem(params).run()
        if res.sizes:
            mean_sizes.append(sum(res.sizes) / len(res.sizes))
        else:
            mean_sizes.append(0.0)

    plot_lines(
        FIG_DIR / "phase1_mean_size_vs_alpha.svg",
        series=[
            {
                "x": alphas,
                "y": mean_sizes,
                "label": "Mean cascade size",
                "color": "#1f77b4",
                "marker": "o",
            }
        ],
        title="Phase 1: order parameter as we tune conservation alpha",
        xlabel="Conservation parameter alpha",
        ylabel="Mean cascade size",
        vline=0.22,
    )

    # --- (2) three-regime size distributions (CCDF on log-log) ------------
    compare = [
        ("Sub-critical alpha=0.12",  0.12, "#2ca02c"),
        ("Near-critical alpha=0.22", 0.22, "#ff7f0e"),
        ("Super-critical alpha=0.245", 0.245, "#d62728"),
    ]
    series: list[dict] = []
    fit_summaries: list[str] = []
    for label, a, color in compare:
        params = FaultParams(
            L=32,
            alpha=a,
            threshold=1.0,
            drive_steps=8000,
            warmup=2000,
            seed=2026,
        )
        res = FaultStressSystem(params).run()
        x, y = ccdf_log(res.sizes)
        series.append({
            "x": x,
            "y": y,
            "label": label,
            "color": color,
            "linestyle": "-",
            "marker": "",
            "linewidth": 1.6,
        })
        # Tail-only Hill estimator: only fit cascades larger than 4 cells —
        # i.e. genuinely "non-trivial" events. The Gutenberg–Richter b-value
        # is approximately ``tau - 1`` (in 2-D OFC tau is around 1.8–2.0).
        tau, n_used = power_law_mle(res.sizes, smin=4)
        fit_summaries.append(
            f"  {label}: tau ~ {tau:.2f}  (n={n_used} samples used)"
        )

    plot_lines(
        FIG_DIR / "phase1_size_dist_compare.svg",
        series=series,
        title="Phase 1: cascade-size CCDF — Gutenberg–Richter signature",
        xlabel="Cascade size s",
        ylabel="P(S >= s)",
        logx=True,
        logy=True,
    )

    # --- (3) Omori aftershock decay near criticality vs sub-critical ------
    omori_series: list[dict] = []
    for label, a, color in [
        ("Sub-critical alpha=0.12",  0.12, "#2ca02c"),
        ("Near-critical alpha=0.22", 0.22, "#ff7f0e"),
    ]:
        params = FaultParams(
            L=32,
            alpha=a,
            threshold=1.0,
            drive_steps=10000,
            warmup=2000,
            seed=2026,
        )
        res = FaultStressSystem(params).run()
        centers, rates, n_main = aftershock_rate(
            res.event_steps,
            res.sizes,
            mainshock_quantile=0.95,
            window=400,
            bins=20,
        )
        if centers:
            omori_series.append({
                "x": centers,
                "y": rates,
                "label": f"{label} (n_main={n_main})",
                "color": color,
                "marker": "o",
                "linestyle": "-",
            })

    if omori_series:
        plot_lines(
            FIG_DIR / "phase1_omori.svg",
            series=omori_series,
            title="Phase 1: stacked aftershock rate (Omori law check)",
            xlabel="Time after mainshock Δt (macro-steps)",
            ylabel="Stacked aftershock rate n(Δt)",
            logx=True,
            logy=True,
        )

    print("Phase 1 done. Figures written to:", FIG_DIR)
    print("Power-law tail fits (Hill estimator on sizes >= 4):")
    for line in fit_summaries:
        print(line)


if __name__ == "__main__":
    run_phase1()
