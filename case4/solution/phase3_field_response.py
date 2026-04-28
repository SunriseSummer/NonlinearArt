"""Phase 3: weak-field pulse response below, near, and above criticality."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ising_model import FieldPulseSpec, IsingFilm, IsingParams
from plotting import plot_bars, plot_lines, rolling_mean

CASE_DIR = BASE_DIR.parent
FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PULSE = FieldPulseSpec(start=500, duration=120, delta_h=0.035)
REGIMES = [
    ("ordered", 1.70, "#1f77b4"),
    ("near critical", 2.30, "#d62728"),
    ("disordered", 3.40, "#ff7f0e"),
]


def recovery_time(values: list[float], baseline: float, peak_delta: float, start: int) -> int:
    threshold = baseline + 0.25 * peak_delta
    for idx in range(start, len(values)):
        if abs(values[idx]) <= threshold:
            return idx - start
    return len(values) - start


def main() -> None:
    series_m = []
    series_activity = []
    labels = []
    response = []
    recovery = []

    for idx, (label, temp, color) in enumerate(REGIMES):
        params = IsingParams(
            L=28,
            temperature=temp,
            field=0.0,
            sweeps=1200,
            warmup=300,
            seed=3300 + idx,
            pulse=PULSE,
            initial_state="random",
        )
        result = IsingFilm(params).run()
        steps = list(range(params.sweeps))
        smoothed_m = rolling_mean(result.magnetisation, 20)
        smoothed_a = rolling_mean(result.accepted_flips, 20)
        series_m.append({"x": steps, "y": smoothed_m, "label": f"{label} T={temp:.2f}", "color": color})
        series_activity.append({"x": steps, "y": smoothed_a, "label": f"{label} T={temp:.2f}", "color": color})
        base_window = result.magnetisation[PULSE.start - 120:PULSE.start]
        pulse_window = result.magnetisation[PULSE.start:PULSE.start + PULSE.duration]
        baseline = sum(base_window) / len(base_window)
        delta = max(abs(v - baseline) for v in pulse_window)
        labels.append(label)
        response.append(delta)
        recovery.append(recovery_time(smoothed_m, abs(baseline), delta, PULSE.start + PULSE.duration))
        print(f"{label:13s}: response={delta:.3f}, recovery={recovery[-1]} sweeps")

    vlines = [
        (PULSE.start, "#555555", "pulse on"),
        (PULSE.start + PULSE.duration, "#777777", "pulse off"),
    ]
    plot_lines(
        FIG_DIR / "phase3_field_pulse_magnetisation.svg",
        series=series_m,
        title="Phase 3: the same weak field is amplified near criticality",
        xlabel="Monte Carlo sweep",
        ylabel="rolling magnetisation m",
        vlines=vlines,
    )
    plot_lines(
        FIG_DIR / "phase3_activity_burst.svg",
        series=series_activity,
        title="Phase 3: spin-flip activity during the pulse",
        xlabel="Monte Carlo sweep",
        ylabel="accepted flips / sweep (rolling)",
        vlines=vlines,
    )
    plot_bars(
        FIG_DIR / "phase3_pulse_response_summary.svg",
        labels,
        response,
        "Phase 3: pulse amplification by regime",
        "max |m - baseline| during pulse",
        colors=[c for _, _, c in REGIMES],
    )


if __name__ == "__main__":
    main()
