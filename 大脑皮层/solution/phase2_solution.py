"""Phase 2 reference implementation: SOC via dynamical synapses (Levina et al.).

Instead of hand-tuning ``J`` (Phase 1), here the model self-organizes onto
``sigma ≈ 1`` thanks to **synaptic resource depletion** (Tsodyks–Markram
dynamics, à la Levina, Herrmann & Geisel, *Nat. Phys.* 2007). Each spike
multiplies the firing neuron's outgoing synapses by ``(1 - epsilon)``;
resources recover linearly toward 1 with time constant ``tau_rec``. Even
when started well above the critical point (e.g. ``J = 0.20`` with nominal
sigma_0 = 1.6), the system settles into a stationary state where the
*effective* branching ratio ``sigma_eff = k * J * <u>`` fluctuates around
1.

> **Important physical note.** The depression mechanism only suppresses
> activity, so it can self-organize a *super-critical* network onto the
> critical line. Starting strictly below criticality, the network simply
> stays sub-critical: there are no spikes, so depression never engages.
> Phase 2 therefore demonstrates SOC by initialising ``J`` from above
> (and above-above) and showing that all initial conditions converge to
> the same statistics.

Three verifications are produced:

1. **Self-organisation trajectory.** Branching ratio and mean resource
   co-evolve to a stationary fixed point with ``sigma_eff ≈ 1``.
2. **Heavy-tailed avalanche statistics + Sethna scaling.** Sizes,
   durations and ⟨s|T⟩ all show power-law support; their exponents satisfy
   the crackling-noise relation ``(tau-1)*sigma*nu*z = (alpha-1)`` —
   without any manual tuning of ``J``.
3. **Avalanche shape collapse.** Average shapes from several duration
   bins, rescaled by ``T^{1 - 1/(sigma nu z)}``, fall on a single
   universal scaling function. This is the strongest one-figure SOC
   signature.

Usage::

    python case3/solution/phase2_solution.py

Outputs (under ``case3/figures/``):
    phase2_sigma_and_resource.svg
    phase2_avalanche_dist.svg
    phase2_robustness.svg
    phase2_shape_collapse.svg
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
    avalanche_profile,
    ccdf_log,
    collapsed_shape,
    log_hist,
    loglog_slope,
    mean_size_vs_duration,
    plot_dual_axis,
    plot_lines,
    power_law_mle,
)

FIG_DIR = CASE3_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _make_params(J0: float, seed: int) -> NeuralParams:
    return NeuralParams(
        N=256,
        k=8,
        J=J0,
        threshold=1.0,
        drive_kick=1.0,
        drive_steps=10000,
        warmup=2500,
        seed=seed,
        dynamical_synapses=True,
        epsilon=0.05,
        tau_rec=400.0,
        avalanche_size_cap=20000,
    )


def run_phase2() -> None:
    # --- (1) main self-organisation run ----------------------------------
    main_params = _make_params(J0=0.20, seed=2026)
    main_res = CorticalNetwork(main_params).run()

    plot_dual_axis(
        FIG_DIR / "phase2_sigma_and_resource.svg",
        x=list(range(len(main_res.branching_ratio))),
        left={
            "y": main_res.branching_ratio,
            "label": "Branching ratio sigma_eff(t)",
            "ylabel": "sigma_eff = k * J * <u>",
            "color": "#1f77b4",
        },
        right={
            "y": main_res.mean_resource,
            "label": "Mean synaptic resource <u>(t)",
            "ylabel": "<u>(t)",
            "color": "#9467bd",
        },
        title="Phase 2: depression locks sigma_eff onto criticality",
        xlabel="Macro-step",
        vline=main_params.warmup,
    )

    # --- (2) heavy-tailed avalanche statistics + Sethna scaling ----------
    sizes = main_res.sizes
    durs = main_res.durations
    Ts, means = mean_size_vs_duration(sizes, durs,
                                      min_count=10, max_duration=40)
    snz_inv, snz_intercept = loglog_slope(Ts, means)
    tau, _ = power_law_mle(sizes, smin=4)
    alpha, _ = power_law_mle(durs, smin=3)

    x_size, y_size = log_hist(sizes)
    x_dur, y_dur = log_hist(durs)
    x_ccdf, y_ccdf = ccdf_log(sizes)

    series_dist: list[dict] = [
        {"x": x_size, "y": y_size, "label": f"PDF P(s)  tau~{tau:.2f}",
         "color": "#e377c2", "marker": "o", "linestyle": ""},
        {"x": x_dur,  "y": y_dur,  "label": f"PDF P(T)  alpha~{alpha:.2f}",
         "color": "#8c564b", "marker": "s", "linestyle": ""},
        {"x": x_ccdf, "y": y_ccdf, "label": "CCDF P(S>=s)",
         "color": "#2ca02c", "marker": "", "linestyle": "--"},
    ]
    if Ts:
        # Add the <s|T> curve for visual reference, scaled to fit alongside.
        fit_line = [math.exp(snz_intercept + snz_inv * math.log(T)) for T in Ts]
        # Normalise to fit on the same axis nicely.
        norm = max(fit_line)
        scaled_means = [m / norm for m in means]
        scaled_fit = [f / norm for f in fit_line]
        series_dist.append({"x": Ts, "y": scaled_means,
                            "label": f"<s|T> (rescaled)  1/snz~{snz_inv:.2f}",
                            "color": "#1f77b4", "marker": "x", "linestyle": ""})
        series_dist.append({"x": Ts, "y": scaled_fit,
                            "label": "<s|T> fit",
                            "color": "#1f77b4", "linestyle": "--",
                            "linewidth": 1.2, "alpha": 0.8})

    plot_lines(
        FIG_DIR / "phase2_avalanche_dist.svg",
        series=series_dist,
        title="Phase 2: scale-free statistics under self-organised criticality",
        xlabel="s, T (rescaled curves dimensionless)",
        ylabel="Probability density / (rescaled <s|T>)",
        logx=True, logy=True,
    )

    # --- (3) robustness across initial J / seeds -------------------------
    robustness_runs = [
        dict(J0=0.16, seed=2026, color="#1f77b4"),
        dict(J0=0.20, seed=99,   color="#ff7f0e"),
        dict(J0=0.30, seed=7,    color="#2ca02c"),
    ]
    series_sigma: list[dict] = []
    summaries: list[str] = []
    for cfg in robustness_runs:
        params = _make_params(J0=cfg["J0"], seed=cfg["seed"])
        res = CorticalNetwork(params).run()
        series_sigma.append({
            "x": list(range(len(res.branching_ratio))),
            "y": res.branching_ratio,
            "label": f"J0={cfg['J0']:.2f}, seed={cfg['seed']}",
            "color": cfg["color"],
            "linewidth": 1.0,
            "alpha": 0.9,
        })
        cut = int(0.6 * len(res.branching_ratio))
        late_sigma = sum(res.branching_ratio[cut:]) / max(len(res.branching_ratio[cut:]), 1)
        late_size = sum(res.sizes) / max(len(res.sizes), 1)
        tau_r, n_r = power_law_mle(res.sizes, smin=4)
        summaries.append(
            f"  J0={cfg['J0']:.2f} seed={cfg['seed']:>4d}  "
            f"-> late <sigma_eff>={late_sigma:.3f}  <s>={late_size:.2f}  "
            f"tau~{tau_r:.2f} (n={n_r})"
        )
    plot_lines(
        FIG_DIR / "phase2_robustness.svg",
        series=series_sigma,
        title="Phase 2: sigma_eff trajectories converge from different starts",
        xlabel="Macro-step",
        ylabel="sigma_eff(t)",
        hline=1.0,
    )

    # --- (4) avalanche shape collapse ------------------------------------
    # Pick a few duration bins (with enough samples) and average their
    # profiles, then collapse by T^{1 - 1/sigma_nu_z}.
    target_durs = [4, 7, 11, 16, 22]
    profiles_by_T: dict[int, list[list[float]]] = {}
    for s, T, prof in zip(main_res.sizes, main_res.durations, main_res.profiles):
        if T in target_durs:
            profiles_by_T.setdefault(T, []).append(avalanche_profile(prof, T, n_bins=40))

    if all(len(profiles_by_T.get(T, [])) >= 5 for T in target_durs):
        collapsed = collapsed_shape(profiles_by_T, snz_inv, n_bins=40)
        # Two figures' worth of curves: raw profiles (averaged, not rescaled)
        # and collapsed curves. We put both on one figure with subplots.
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
        cmap = plt.cm.viridis
        for idx, T in enumerate(sorted(target_durs)):
            color = cmap(idx / max(len(target_durs) - 1, 1))
            avg = [sum(prof[b] for prof in profiles_by_T[T]) / len(profiles_by_T[T])
                   for b in range(40)]
            grid = [b / 39 for b in range(40)]
            ax1.plot(grid, avg, color=color, label=f"T={T}", linewidth=1.4)
            grid_c, ys_c = collapsed[T]
            ax2.plot(grid_c, ys_c, color=color, label=f"T={T}", linewidth=1.4)
        ax1.set_title("Raw average shapes")
        ax1.set_xlabel("t / T")
        ax1.set_ylabel("<n_spikes>(t/T)")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(fontsize=8, loc="best")
        ax2.set_title(
            f"Collapsed: shape * T^(1 - 1/snz),  1/snz={snz_inv:.2f}"
        )
        ax2.set_xlabel("t / T")
        ax2.set_ylabel("rescaled shape")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(fontsize=8, loc="best")
        fig.suptitle("Phase 2: avalanche shape collapse", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "phase2_shape_collapse.svg")
        plt.close(fig)

    # ---- console summary ----
    print("Phase 2 done. Figures written to:", FIG_DIR)
    print("Robustness summary (epsilon=0.05, tau_rec=400):")
    for line in summaries:
        print(line)
    print()
    print("Self-organised crackling-noise scaling (main run, J0=0.20):")
    print(f"  tau            ~= {tau:.3f}")
    print(f"  alpha          ~= {alpha:.3f}")
    print(f"  1/(sigma nu z) ~= {snz_inv:.3f}")
    print(f"  (tau - 1) * sigma_nu_z = {(tau - 1) * snz_inv:.3f}")
    print(f"  alpha - 1              = {alpha - 1:.3f}")
    rel_err = abs((tau - 1) * snz_inv - (alpha - 1)) / max(abs(alpha - 1), 1e-9)
    print(f"  relative error of scaling relation: {rel_err * 100:.1f} %")


if __name__ == "__main__":
    run_phase2()
