"""Phase 2: locate the Ising critical region by temperature sweeps."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ising_model import IsingFilm, IsingParams
from plotting import binder_cumulant, heat_capacity, plot_lines, susceptibility

CASE_DIR = BASE_DIR.parent
FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TEMPS = [1.60, 1.90, 2.10, 2.20, 2.25, 2.30, 2.35, 2.45, 2.60, 2.80, 3.20]
SIZES = [16, 24, 32]


def run_point(L: int, temp: float, seed: int) -> tuple[float, float, float, float]:
    params = IsingParams(
        L=L,
        temperature=temp,
        sweeps=1100,
        warmup=350,
        seed=seed,
        initial_state="up",
    )
    result = IsingFilm(params).run()
    tail = slice(params.warmup, None)
    m_tail = result.magnetisation[tail]
    abs_m_tail = result.abs_magnetisation[tail]
    e_tail = result.energy[tail]
    mean_abs_m = sum(abs_m_tail) / len(abs_m_tail)
    chi = susceptibility(m_tail, temp, L * L)
    c = heat_capacity(e_tail, temp, L * L)
    binder = binder_cumulant(m_tail)
    return mean_abs_m, chi, c, binder


def main() -> None:
    by_size: dict[int, dict[str, list[float]]] = {}
    for L in SIZES:
        vals = {"abs_m": [], "chi": [], "heat": [], "binder": []}
        for idx, temp in enumerate(TEMPS):
            mean_abs_m, chi, c, binder = run_point(L, temp, seed=2400 + 31 * L + idx)
            vals["abs_m"].append(mean_abs_m)
            vals["chi"].append(chi)
            vals["heat"].append(c)
            vals["binder"].append(binder)
            print(
                f"L={L:2d}, T={temp:.2f}: |m|={mean_abs_m:.3f}, "
                f"chi={chi:.2f}, C={c:.2f}, U4={binder:.3f}"
            )
        by_size[L] = vals

    colors = {16: "#1f77b4", 24: "#d62728", 32: "#2ca02c"}
    plot_lines(
        FIG_DIR / "phase2_temperature_sweep.svg",
        series=[{
            "x": TEMPS,
            "y": by_size[L]["abs_m"],
            "label": f"L={L}",
            "color": colors[L],
            "marker": "o",
        } for L in SIZES],
        title="Phase 2: order parameter drops around Tc",
        xlabel="Temperature T",
        ylabel="late-time mean |m|",
        vlines=[(2.269, "#555555", "exact Tc")],
    )
    plot_lines(
        FIG_DIR / "phase2_susceptibility.svg",
        series=[{
            "x": TEMPS,
            "y": by_size[L]["chi"],
            "label": f"L={L}",
            "color": colors[L],
            "marker": "o",
        } for L in SIZES],
        title="Phase 2: susceptibility peak marks the critical region",
        xlabel="Temperature T",
        ylabel="magnetic susceptibility chi",
        vlines=[(2.269, "#555555", "exact Tc")],
    )
    plot_lines(
        FIG_DIR / "phase2_finite_size.svg",
        series=[{
            "x": TEMPS,
            "y": by_size[L]["heat"],
            "label": f"heat proxy, L={L}",
            "color": colors[L],
            "marker": "s",
        } for L in SIZES],
        title="Phase 2: finite-size heat-capacity proxy",
        xlabel="Temperature T",
        ylabel="C proxy from energy fluctuations",
        vlines=[(2.269, "#555555", "exact Tc")],
    )
    plot_lines(
        FIG_DIR / "phase2_binder_cumulant.svg",
        series=[{
            "x": TEMPS,
            "y": by_size[L]["binder"],
            "label": f"L={L}",
            "color": colors[L],
            "marker": "o",
        } for L in SIZES],
        title="Phase 2: Binder cumulant curves cross near Tc",
        xlabel="Temperature T",
        ylabel="Binder cumulant U4 = 1 - <m^4>/(3<m^2>^2)",
        vlines=[(2.269, "#555555", "exact Tc")],
    )

    best_L = 32
    peak_idx = max(range(len(TEMPS)), key=lambda i: by_size[best_L]["chi"][i])
    print(f"Estimated critical window from L={best_L}: T≈{TEMPS[peak_idx]:.2f}")
    print("Binder cumulant crossing provides an independent finite-size check.")


if __name__ == "__main__":
    main()
