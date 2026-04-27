"""Starter material for Case 3 (cortical avalanches — sub-critical baseline).

Run this script first — it produces three reference figures of the baseline
network behaviour *before* any tuning. Defaults are deliberately well below
the branching-process critical point: ``J = 0.08`` while ``J_c = 1/k = 0.125``,
which gives a nominal branching ratio ``sigma = k * J = 0.64``. In that
regime:

* every external kick triggers at most a tiny offspring cascade (mean size
  of order 1 to a few),
* the size distribution decays exponentially fast with no power-law tail,
* the membrane potential trajectory wanders around a small mean.

From here players proceed to:

* Phase 1 — keep the model identical and only tune ``J`` until the
  cortical network shows tuned-criticality signatures (Beggs–Plenz
  ``tau ~= 3/2`` size law, Sethna crackling-noise scaling).
* Phase 2 — flip ``dynamical_synapses=True`` and let the system
  *self-organize* near criticality without sweeping ``J`` by hand.

Output figures are written under ``case3/figures/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from neural_model import CorticalNetwork, NeuralParams
from plotting import ccdf_log, log_hist, plot_lines

CASE3_DIR = BASE_DIR.parent
FIG_DIR = CASE3_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    params = NeuralParams(
        N=256,
        k=8,
        J=0.08,            # sigma_nom = 0.64, well sub-critical
        threshold=1.0,
        drive_kick=1.0,
        drive_steps=4000,
        warmup=1000,
        seed=2026,
    )
    result = CorticalNetwork(params).run()

    # --- (1) mean potential and instantaneous branching ratio ------------
    plot_lines(
        FIG_DIR / "starter_mean_potential.svg",
        series=[
            {
                "x": list(range(len(result.mean_potential))),
                "y": result.mean_potential,
                "label": "Mean membrane potential",
                "color": "#1f77b4",
            },
            {
                "x": list(range(len(result.branching_ratio))),
                "y": result.branching_ratio,
                "label": "Branching ratio sigma_eff(t)",
                "color": "#ff7f0e",
                "linewidth": 1.0,
                "alpha": 0.9,
            },
        ],
        title=f"Starter (sub-critical, J={params.J}): potentials and sigma_eff",
        xlabel="Macro-step (slow-drive index)",
        ylabel="<h>(t) / sigma_eff(t)",
        vline=params.warmup,
        hline=1.0,
    )

    # --- (2) avalanche-size distribution (log-binned PDF + CCDF) ---------
    x_pdf, y_pdf = log_hist(result.sizes)
    x_ccdf, y_ccdf = ccdf_log(result.sizes)
    series_dist: list[dict] = []
    if x_pdf:
        series_dist.append({
            "x": x_pdf, "y": y_pdf, "label": "log-binned PDF P(s)",
            "color": "#d62728", "marker": "o", "linestyle": "-",
        })
    if x_ccdf:
        series_dist.append({
            "x": x_ccdf, "y": y_ccdf, "label": "CCDF P(S>=s)",
            "color": "#2ca02c", "marker": "", "linestyle": "--",
        })
    plot_lines(
        FIG_DIR / "starter_avalanche_distribution.svg",
        series=series_dist,
        title="Starter (sub-critical): cascade-size distribution",
        xlabel="Avalanche size s",
        ylabel="Probability density / CCDF",
        logx=True,
        logy=True,
    )

    # --- (3) avalanche-duration distribution ------------------------------
    x_pdf_T, y_pdf_T = log_hist(result.durations)
    x_ccdf_T, y_ccdf_T = ccdf_log(result.durations)
    series_T: list[dict] = []
    if x_pdf_T:
        series_T.append({
            "x": x_pdf_T, "y": y_pdf_T, "label": "log-binned PDF P(T)",
            "color": "#9467bd", "marker": "s", "linestyle": "-",
        })
    if x_ccdf_T:
        series_T.append({
            "x": x_ccdf_T, "y": y_ccdf_T, "label": "CCDF P(T>=t)",
            "color": "#8c564b", "marker": "", "linestyle": "--",
        })
    plot_lines(
        FIG_DIR / "starter_duration_distribution.svg",
        series=series_T,
        title="Starter (sub-critical): cascade-duration distribution",
        xlabel="Duration T (synchronous generations)",
        ylabel="Probability density / CCDF",
        logx=True,
        logy=True,
    )

    n = len(result.sizes)
    avg = sum(result.sizes) / n if n else 0.0
    mx = max(result.sizes) if n else 0
    print(f"Recorded {n} non-trivial avalanches after warmup.")
    print(f"Mean avalanche size = {avg:.2f}, largest avalanche = {mx}.")
    print(f"Nominal branching ratio sigma = k * J = {params.k * params.J:.3f}"
          f" (critical at sigma=1, i.e. J_c=1/k={1/params.k:.3f}).")
    print("Sub-critical baseline figures written to case3/figures/starter_*.svg")


if __name__ == "__main__":
    main()
