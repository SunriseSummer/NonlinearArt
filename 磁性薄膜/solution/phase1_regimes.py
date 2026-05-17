"""Phase 1: compare low/intermediate/high-temperature Ising regimes."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ising_model import IsingFilm, IsingParams
from plotting import plot_bars, plot_lines, plot_spin_field, rolling_mean

CASE_DIR = BASE_DIR.parent
FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REGIMES = [
    ("cold ordered", 1.60, "#1f77b4"),
    ("near edge", 2.30, "#d62728"),
    ("hot disordered", 3.60, "#ff7f0e"),
]


def main() -> None:
    series_m = []
    series_activity = []
    labels = []
    steady_m = []
    steady_activity = []

    for idx, (label, temp, color) in enumerate(REGIMES):
        params = IsingParams(L=28, temperature=temp, sweeps=1300, warmup=350, seed=2026 + idx)
        result = IsingFilm(params).run()
        steps = list(range(params.sweeps))
        series_m.append({
            "x": steps,
            "y": rolling_mean(result.abs_magnetisation, 30),
            "label": f"{label} (T={temp:.2f})",
            "color": color,
        })
        series_activity.append({
            "x": steps,
            "y": rolling_mean(result.accepted_flips, 30),
            "label": f"{label} (T={temp:.2f})",
            "color": color,
        })
        plot_spin_field(
            FIG_DIR / f"phase1_domains_{label.split()[0]}.svg",
            result.final_spins,
            f"Phase 1 final domains: {label}, T={temp:.2f}",
        )
        tail = slice(params.warmup, None)
        labels.append(label)
        steady_m.append(sum(result.abs_magnetisation[tail]) / len(result.abs_magnetisation[tail]))
        steady_activity.append(sum(result.accepted_flips[tail]) / len(result.accepted_flips[tail]))

    plot_lines(
        FIG_DIR / "phase1_magnetisation_regimes.svg",
        series=series_m,
        title="Phase 1: ordered, edge, and disordered regimes",
        xlabel="Monte Carlo sweep",
        ylabel="rolling |m|",
        vlines=[(350, "#777777", "warmup")],
    )
    plot_lines(
        FIG_DIR / "phase1_activity_regimes.svg",
        series=series_activity,
        title="Phase 1: spin-flip activity across regimes",
        xlabel="Monte Carlo sweep",
        ylabel="accepted flips / sweep (rolling)",
        vlines=[(350, "#777777", "warmup")],
    )
    plot_bars(
        FIG_DIR / "phase1_steady_abs_m.svg",
        labels,
        steady_m,
        "Phase 1: steady order parameter",
        "late-time mean |m|",
        colors=[c for _, _, c in REGIMES],
    )
    print("Phase 1 summary:")
    for label, m, a in zip(labels, steady_m, steady_activity):
        print(f"  {label:14s} |m|={m:.3f}, accepted flips/sweep={a:.1f}")


if __name__ == "__main__":
    main()
