"""Starter material for Case 2 (earthquake fault — clearly sub-critical baseline).

Run this script first — it produces three reference figures that show the
baseline behaviour of the fault-stress scenario *before* any tuning. The
defaults are deliberately chosen so the system sits well below the OFC
critical conservation level (``alpha = 0.10`` while criticality lives near
``alpha ~ 0.22``):

* the stress field stays "spotty" with frequent tiny ruptures,
* the cascade-size distribution decays exponentially fast on a log–log
  plot, with no power-law tail,
* there is essentially no Omori-style aftershock signal.

From here players are expected to:

* Phase 1 — keep the model identical and only tune parameters (most
  importantly ``alpha``) until the system shows tuned-criticality
  signatures (Gutenberg–Richter power-law tail, growing mean cascade…).
* Phase 2 — flip ``adaptive=True`` and add quenched heterogeneity so the
  system *self-organises* near the critical point without manual sweeps.

Output figures are written under ``case2/figures/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the ``base`` package importable regardless of where the script is run.
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fault_model import FaultParams, FaultStressSystem
from plotting import ccdf_log, log_hist, plot_heatmap, plot_lines

CASE2_DIR = BASE_DIR.parent
FIG_DIR = CASE2_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Intentionally sub-critical: small cascades, fast exponential decay.
    params = FaultParams(
        L=32,
        alpha=0.10,
        threshold=1.0,
        drive_steps=4000,
        warmup=1000,
        seed=2026,
    )

    result = FaultStressSystem(params).run()

    # --- (1) mean stress time series --------------------------------------
    plot_lines(
        FIG_DIR / "starter_mean_stress.svg",
        series=[
            {
                "x": list(range(len(result.mean_stress))),
                "y": result.mean_stress,
                "label": "Mean stress",
                "color": "#1f77b4",
            }
        ],
        title="Starter (sub-critical, alpha=0.10): mean stress over time",
        xlabel="Macro-step (slow-drive index)",
        ylabel="Mean stress per cell",
        vline=params.warmup,
    )

    # --- (2) avalanche-size distribution (log-binned PDF) -----------------
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
            "x": x_ccdf, "y": y_ccdf, "label": "CCDF P(S≥s)",
            "color": "#2ca02c", "marker": "", "linestyle": "--",
        })
    plot_lines(
        FIG_DIR / "starter_avalanche_distribution.svg",
        series=series_dist,
        title="Starter (sub-critical): cascade-size distribution",
        xlabel="Cascade size s",
        ylabel="Probability density / CCDF",
        logx=True,
        logy=True,
    )

    # --- (3) final stress field heatmap -----------------------------------
    plot_heatmap(
        FIG_DIR / "starter_stress_field.svg",
        result.final_field,
        title="Starter: final stress field (mostly low, no big asperities)",
        cmap="magma",
        vmin=0.0,
        vmax=params.threshold,
    )

    n = len(result.sizes)
    avg = sum(result.sizes) / n if n else 0.0
    mx = max(result.sizes) if n else 0
    print(f"Recorded {n} non-trivial cascades after warmup.")
    print(f"Mean cascade size = {avg:.2f}, largest cascade = {mx}.")
    print("Sub-critical baseline figures written to case2/figures/starter_*.svg")


if __name__ == "__main__":
    main()
