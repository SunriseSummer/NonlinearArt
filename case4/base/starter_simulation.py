"""Starter simulation for Case 4: a hot paramagnetic Ising film.

Run this first.  The default bath temperature is deliberately above the 2-D
Ising critical point Tc ~= 2.269 (for J=1, k_B=1), so magnetic domains flicker
quickly and the net magnetisation stays near zero.  Later phases cool the film
toward Tc, probe critical amplification with a weak field pulse, and finally
add a feedback thermostat that self-organises near the transition.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ising_model import IsingFilm, IsingParams
from plotting import plot_dual_axis, plot_lines, plot_spin_field, rolling_mean

CASE_DIR = BASE_DIR.parent
FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    params = IsingParams(
        L=28,
        temperature=3.6,
        field=0.0,
        sweeps=1200,
        warmup=300,
        seed=2026,
        initial_state="random",
    )
    result = IsingFilm(params).run()
    steps = list(range(params.sweeps))

    plot_dual_axis(
        FIG_DIR / "starter_hot_film_timeseries.svg",
        x=steps,
        left={
            "y": rolling_mean(result.abs_magnetisation, 25),
            "label": "|magnetisation|",
            "ylabel": "|m|",
            "color": "#1f77b4",
        },
        right={
            "y": rolling_mean(result.accepted_flips, 25),
            "label": "accepted flips",
            "ylabel": "accepted flips / sweep",
            "color": "#ff7f0e",
        },
        title="Starter (hot paramagnet): noisy spins, no persistent order",
        xlabel="Monte Carlo sweep",
        vlines=[(params.warmup, "#b00000", f"warmup={params.warmup}")],
    )

    plot_lines(
        FIG_DIR / "starter_energy.svg",
        series=[{
            "x": steps,
            "y": rolling_mean(result.energy, 25),
            "label": "energy per spin",
            "color": "#2ca02c",
        }],
        title="Starter (hot paramagnet): energy fluctuates around a shallow basin",
        xlabel="Monte Carlo sweep",
        ylabel="E/N",
        vlines=[(params.warmup, "#b00000", f"warmup={params.warmup}")],
    )

    plot_spin_field(
        FIG_DIR / "starter_final_domains.svg",
        result.final_spins,
        "Starter final spin field (hot, short-range domains)",
    )

    tail = slice(params.warmup, None)
    mean_abs_m = sum(result.abs_magnetisation[tail]) / len(result.abs_magnetisation[tail])
    mean_accept = sum(result.accepted_flips[tail]) / len(result.accepted_flips[tail])
    print(f"[starter] T={params.temperature:.2f}, Tc≈2.269")
    print(f"[starter] steady |m|={mean_abs_m:.3f}")
    print(f"[starter] accepted flips/sweep={mean_accept:.1f}")
    print("Starter figures written to case4/figures/starter_*.svg")


if __name__ == "__main__":
    main()
