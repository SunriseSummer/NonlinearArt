"""Phase 4: feedback thermostat that self-organises near the Ising edge."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ising_model import IsingFilm, IsingParams
from plotting import log_hist, plot_bars, plot_lines, rolling_mean, susceptibility

CASE_DIR = BASE_DIR.parent
FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_TEMPS = [1.40, 3.20, 4.00]
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]
# L=12 has a broad finite-size critical region rather than a sharp singular
# point.  Phase 2's finite-size sweep places the useful high-fluctuation band
# around this interval, so phase 4 measures how often feedback stays inside it.
CRITICAL_BAND = (2.20, 2.80)


def run_adaptive(temp: float, seed: int):
    params = IsingParams(
        L=12,
        temperature=temp,
        sweeps=2500,
        warmup=500,
        seed=seed,
        adaptive_temperature=True,
        # For L=12, |m|≈0.45 keeps the controller in the finite-size critical
        # band instead of locking the small lattice into a fully ordered state.
        target_abs_m=0.45,
        temp_gain=0.06,
        feedback_window=35,
        temp_min=1.3,
        temp_max=4.2,
    )
    return params, IsingFilm(params).run()


def main() -> None:
    temp_series = []
    mag_series = []
    labels = []
    late_temps = []
    late_mags = []
    critical_occupancy = []
    adaptive_m: list[float] = []
    adaptive_t: list[float] = []
    adaptive_activity: list[int] = []

    for idx, temp in enumerate(INITIAL_TEMPS):
        params, result = run_adaptive(temp, 4400 + idx)
        steps = list(range(params.sweeps))
        label = f"T0={temp:.1f}"
        labels.append(label)
        temp_series.append({"x": steps, "y": result.temperature_series, "label": label, "color": COLORS[idx]})
        mag_series.append({"x": steps, "y": rolling_mean(result.abs_magnetisation, 35), "label": label, "color": COLORS[idx]})
        tail = slice(-400, None)
        late_temps.append(sum(result.temperature_series[tail]) / len(result.temperature_series[tail]))
        late_mags.append(sum(result.abs_magnetisation[tail]) / len(result.abs_magnetisation[tail]))
        late_t = result.temperature_series[tail]
        in_band = [
            1.0 if CRITICAL_BAND[0] <= value <= CRITICAL_BAND[1] else 0.0
            for value in late_t
        ]
        critical_occupancy.append(sum(in_band) / len(in_band))
        adaptive_m.extend(result.magnetisation[params.warmup:])
        adaptive_t.extend(result.temperature_series[params.warmup:])
        adaptive_activity.extend(result.accepted_flips[params.warmup:])
        print(f"{label}: late T={late_temps[-1]:.3f}, late |m|={late_mags[-1]:.3f}")

    plot_lines(
        FIG_DIR / "phase4_temperature_convergence.svg",
        series=temp_series,
        title="Phase 4: feedback thermostat converges toward the critical window",
        xlabel="Monte Carlo sweep",
        ylabel="temperature T(t)",
        hline=2.269,
    )
    plot_lines(
        FIG_DIR / "phase4_magnetisation_convergence.svg",
        series=mag_series,
        title="Phase 4: order parameter is held near the target band",
        xlabel="Monte Carlo sweep",
        ylabel="rolling |m|",
        hline=0.45,
    )

    # Static controls for activity distribution comparison.
    control_series = []
    susceptibility_labels = []
    susceptibility_values = []
    for label, temp, color in [
        ("fixed cold", 1.60, "#9467bd"),
        ("fixed near-Tc", 2.50, "#7f7f7f"),
        ("fixed hot", 3.60, "#8c564b"),
    ]:
        params = IsingParams(L=12, temperature=temp, sweeps=1400, warmup=400, seed=4700 + int(temp * 10))
        result = IsingFilm(params).run()
        xs, ys = log_hist(result.accepted_flips[params.warmup:])
        if xs:
            control_series.append({"x": xs, "y": ys, "label": label, "color": color, "marker": "o"})
        susceptibility_labels.append(label)
        susceptibility_values.append(
            susceptibility(result.magnetisation[params.warmup:], temp, params.L * params.L)
        )
    xs, ys = log_hist(adaptive_activity)
    if xs:
        control_series.append({"x": xs, "y": ys, "label": "adaptive near edge", "color": "#d62728", "marker": "s"})
    susceptibility_labels.append("adaptive")
    mean_adaptive_t = sum(adaptive_t) / len(adaptive_t)
    susceptibility_values.append(susceptibility(adaptive_m, mean_adaptive_t, 12 * 12))
    plot_lines(
        FIG_DIR / "phase4_activity_distribution.svg",
        series=control_series,
        title="Phase 4: activity distribution broadens under feedback",
        xlabel="accepted flips per sweep",
        ylabel="probability density",
        logx=True,
        logy=True,
    )
    plot_bars(
        FIG_DIR / "phase4_critical_occupancy.svg",
        labels,
        critical_occupancy,
        "Phase 4: late-time occupancy of the finite-size critical band",
        f"fraction of last 400 sweeps in T∈[{CRITICAL_BAND[0]:.1f},{CRITICAL_BAND[1]:.1f}]",
        colors=COLORS,
    )
    plot_bars(
        FIG_DIR / "phase4_susceptibility_compare.svg",
        susceptibility_labels,
        susceptibility_values,
        "Phase 4: adaptive state keeps high susceptibility versus static controls",
        "chi from late-time magnetisation fluctuations",
        colors=["#9467bd", "#7f7f7f", "#8c564b", "#d62728"],
    )
    plot_bars(
        FIG_DIR / "phase4_summary.svg",
        labels,
        late_temps,
        "Phase 4: late-time temperatures from different starts",
        "late-time mean T",
        colors=COLORS,
    )


if __name__ == "__main__":
    main()
