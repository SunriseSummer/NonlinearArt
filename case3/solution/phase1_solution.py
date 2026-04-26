"""Phase 1 reference implementation: tune the cortical network to criticality.

Five coordinated experiments on top of the base ``CorticalNetwork``:

1. **Order-parameter sweep.** Scan ``J`` and track the mean avalanche size.
   On a finite ``N=256, k=8`` graph the mean size grows by ~30x as J
   approaches J_c = 1/k = 0.125, identifying the critical point.

2. **Cascade-size distribution (Beggs–Plenz).** Compare CCDFs at
   sub-critical / near-critical / super-critical settings. The
   near-critical curve should follow ``P(s) ~ s^{-tau}`` with
   ``tau ≈ 3/2`` — the mean-field branching-process exponent.

3. **Cascade-duration distribution.** Same comparison for duration.
   Critical exponent ``alpha ≈ 2``.

4. **Mean size given duration ⟨s|T⟩.** Bin avalanches by duration and
   plot the conditional mean. Slope on log–log gives ``1/(sigma*nu*z)``,
   theoretical value ≈ 2 in the mean-field branching universality class.

5. **Crackling-noise scaling check.** Combine the three estimates and
   check the Sethna *et al.* (2001) relation
   ``(tau - 1) * sigma*nu*z = alpha - 1``.
   This is the deepest single-shot SOC test: three independently measured
   exponents must satisfy one equation. Players who pass it have done
   real statistical mechanics.

Usage::

    python case3/solution/phase1_solution.py

Outputs (under ``case3/figures/``):
    phase1_mean_size_vs_J.svg
    phase1_size_dist.svg
    phase1_duration_dist.svg
    phase1_mean_size_given_T.svg
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

CASE3_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CASE3_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from neural_model import CorticalNetwork, NeuralParams
from plotting import (
    ccdf_log,
    loglog_slope,
    mean_size_vs_duration,
    plot_lines,
    power_law_mle,
)

FIG_DIR = CASE3_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_phase1() -> None:
    # --- (1) order-parameter sweep ---------------------------------------
    Js = [0.06 + 0.01 * i for i in range(8)]  # 0.06 .. 0.13
    mean_sizes: list[float] = []
    for J in Js:
        params = NeuralParams(
            N=256, k=8, J=J,
            drive_steps=4000, warmup=1000, seed=2026,
            avalanche_size_cap=20000,
        )
        res = CorticalNetwork(params).run()
        if res.sizes:
            mean_sizes.append(sum(res.sizes) / len(res.sizes))
        else:
            mean_sizes.append(0.0)

    plot_lines(
        FIG_DIR / "phase1_mean_size_vs_J.svg",
        series=[
            {
                "x": Js, "y": mean_sizes,
                "label": "Mean avalanche size",
                "color": "#1f77b4", "marker": "o",
            }
        ],
        title="Phase 1: order parameter as we tune synaptic gain J",
        xlabel="Synaptic gain J",
        ylabel="Mean avalanche size",
        vline=0.125,  # J_c = 1/k
        logy=True,
    )

    # --- (2)/(3) three-regime distributions ------------------------------
    compare = [
        ("Sub-critical J=0.10",   0.10, "#2ca02c"),
        ("Near-critical J=0.125", 0.125, "#ff7f0e"),
        ("Super-critical J=0.135", 0.135, "#d62728"),
    ]

    runs: dict[str, object] = {}
    for label, J, _ in compare:
        params = NeuralParams(
            N=256, k=8, J=J,
            drive_steps=8000, warmup=2000, seed=2026,
            avalanche_size_cap=20000,
        )
        runs[label] = CorticalNetwork(params).run()

    # ---- size CCDF ----
    series_s: list[dict] = []
    for label, _, color in compare:
        x, y = ccdf_log(runs[label].sizes)
        series_s.append({"x": x, "y": y, "label": label, "color": color,
                         "linestyle": "-", "marker": "", "linewidth": 1.6})
    plot_lines(
        FIG_DIR / "phase1_size_dist.svg",
        series=series_s,
        title="Phase 1: avalanche-size CCDF (Beggs–Plenz tau ~ 3/2 at criticality)",
        xlabel="Avalanche size s",
        ylabel="P(S >= s)",
        logx=True, logy=True,
    )

    # ---- duration CCDF ----
    series_T: list[dict] = []
    for label, _, color in compare:
        x, y = ccdf_log(runs[label].durations)
        series_T.append({"x": x, "y": y, "label": label, "color": color,
                         "linestyle": "-", "marker": "", "linewidth": 1.6})
    plot_lines(
        FIG_DIR / "phase1_duration_dist.svg",
        series=series_T,
        title="Phase 1: avalanche-duration CCDF (alpha ~ 2 at criticality)",
        xlabel="Duration T",
        ylabel="P(T >= t)",
        logx=True, logy=True,
    )

    # ---- (4) mean size given duration <s|T> at critical ----
    crit_label = "Near-critical J=0.125"
    crit = runs[crit_label]
    Ts, means = mean_size_vs_duration(
        crit.sizes, crit.durations,
        min_count=10,
        max_duration=40,   # crop the noisy tail
    )
    fit_slope, fit_intercept = loglog_slope(Ts, means)

    # Build a fitted line for the plot
    fit_line = []
    if not math.isnan(fit_slope) and Ts:
        fit_line = [math.exp(fit_intercept + fit_slope * math.log(T)) for T in Ts]

    series_st: list[dict] = [
        {"x": Ts, "y": means, "label": crit_label,
         "color": "#ff7f0e", "marker": "o", "linestyle": ""}
    ]
    if fit_line:
        series_st.append({
            "x": Ts, "y": fit_line,
            "label": f"fit slope 1/(sigma*nu*z) ~ {fit_slope:.2f}",
            "color": "#1f77b4", "linestyle": "--", "linewidth": 1.4,
        })
    plot_lines(
        FIG_DIR / "phase1_mean_size_given_T.svg",
        series=series_st,
        title="Phase 1: <s|T> log-log -> 1/(sigma*nu*z) at criticality",
        xlabel="Duration T",
        ylabel="<s | T>",
        logx=True, logy=True,
    )

    # ---- exponent estimates and crackling-noise check ----
    fit_summaries: list[str] = []
    for label, _, _ in compare:
        res = runs[label]
        tau, n_tau = power_law_mle(res.sizes, smin=4)
        alpha, n_alpha = power_law_mle(res.durations, smin=3)
        fit_summaries.append(
            f"  {label}: tau~{tau:.2f} (n={n_tau})  "
            f"alpha~{alpha:.2f} (n={n_alpha})"
        )

    crit_tau, _ = power_law_mle(crit.sizes, smin=4)
    crit_alpha, _ = power_law_mle(crit.durations, smin=3)
    crackling_lhs = (crit_tau - 1.0) * fit_slope
    crackling_rhs = crit_alpha - 1.0

    print("Phase 1 done. Figures written to:", FIG_DIR)
    print("Power-law tail fits:")
    for line in fit_summaries:
        print(line)
    print()
    print("Crackling-noise scaling at criticality:")
    print(f"  tau            ~= {crit_tau:.3f}    (theory 3/2 = 1.500)")
    print(f"  alpha          ~= {crit_alpha:.3f}    (theory 2)")
    print(f"  1/(sigma nu z) ~= {fit_slope:.3f}    (theory 2)")
    print(f"  (tau - 1) * sigma_nu_z = {crackling_lhs:.3f}")
    print(f"  alpha - 1              = {crackling_rhs:.3f}")
    rel_err = abs(crackling_lhs - crackling_rhs) / max(abs(crackling_rhs), 1e-9)
    print(f"  relative error of scaling relation: {rel_err * 100:.1f} %")


if __name__ == "__main__":
    run_phase1()
