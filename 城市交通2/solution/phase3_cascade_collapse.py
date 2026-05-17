"""Phase 3 (cascade collapse): a small disturbance reveals fragility past the edge.

We hit three operating points (sub / near / super critical) with **the same**
disturbance — a single intersection frozen for 200 steps, mimicking a minor
accident — and observe how the network responds.

Expected lessons:

* In the **subcritical** regime almost nothing happens: the bottleneck
  drains slowly after recovery, throughput barely dips.
* **Near critical** the disturbance generates a noticeable surge in
  congestion-propagation range and a longer recovery, but the network
  eventually settles back near its previous operating point.
* In the **supercritical** regime the disturbance triggers a clear
  cascade: congestion spreads to many intersections, throughput collapses
  during the event, and recovery (return to pre-disturbance load) takes
  much longer — sometimes never within the simulation window.

This concretely shows why "running near capacity" is not a free lunch:
high efficiency comes with high fragility, and the dispatcher must keep
some slack for unexpected events.

Outputs:
    case1b/figures/phase3_disturbance_load.svg     (load vs t in 3 regimes)
    case1b/figures/phase3_congestion_spread.svg    (congestion range vs t)
    case1b/figures/phase3_recovery_summary.svg     (recovery-time bar chart)
"""

from __future__ import annotations

import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CASE_DIR / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from plotting import plot_bars, plot_lines, rolling_mean
from traffic_model import (
    DisturbanceSpec,
    RushHourTrafficSystem,
    TrafficParams,
)

FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Phase-2 detected operating points; we lock them in so phase 3 is
# reproducible and comparable to phase 2 visually.
SPILL = 0.30
DISSIPATION = 0.18
L = 20
THRESHOLD = 6
WARMUP = 1500
STEPS = 5500
EVENT_STEP = 3000   # disturbance triggers here, well after warmup
EVENT_LEN = 600
EVENT_RADIUS = 1    # 3x3 block of intersections frozen during the event

REGIMES = [
    ("Subcritical inflow=1.0", 1.0, "#2ca02c"),
    ("Near-critical inflow=1.6", 1.6, "#ff7f0e"),
    ("Supercritical inflow=2.2", 2.2, "#d62728"),
]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _recovery_time(metric_series: list[float], event_end: int,
                   pre_window: tuple[int, int],
                   peak_window: tuple[int, int]) -> int:
    """Steps after the disturbance ends until ``metric_series`` decays
    back to within 10% of the excess above the pre-event baseline.

    Concretely: if the pre-event mean is ``b`` and the peak during the
    event window is ``p``, the recovery target is ``b + 0.1*(p - b)`` —
    i.e. the system is "recovered" when only 10% of the added pressure
    remains. This makes recovery time comparable across regimes that
    sit at very different baselines."""
    base = _mean(metric_series[pre_window[0]:pre_window[1]])
    peak = max(metric_series[peak_window[0]:peak_window[1]])
    excess = max(0.0, peak - base)
    target = base + 0.1 * excess
    if excess < 1e-6:
        return 0
    for t in range(event_end, len(metric_series)):
        if metric_series[t] <= target:
            return t - event_end
    return len(metric_series) - event_end  # never recovered


def run_phase3() -> None:
    disturbance = DisturbanceSpec(
        start=EVENT_STEP,
        duration=EVENT_LEN,
        cell=(L // 2, L // 2),
        radius=EVENT_RADIUS,
    )

    print("[phase3] same disturbance, three regimes:")
    print("  regime                          peak_cong   recovery_steps   thrpt_drop")

    series_load, series_cong = [], []
    bar_labels, bar_peak = [], []
    bar_colors = []

    for label, inflow, color in REGIMES:
        params = TrafficParams(
            L=L, threshold=THRESHOLD,
            inflow_rate=inflow, spill_prob=SPILL, dissipation=DISSIPATION,
            steps=STEPS, warmup=WARMUP, seed=2026,
            disturbance=disturbance,
        )
        res = RushHourTrafficSystem(params).run()
        xs = list(range(len(res.densities)))

        # Smooth load + congestion-range time series for readable plots.
        smooth_load = rolling_mean(res.densities, 30)
        smooth_cong = rolling_mean(res.congestion_range, 30)

        # Recovery is measured from congestion-range (more sensitive than
        # the network-wide mean load to a localised accident). Time is
        # measured from the moment the accident clears.
        recovery = _recovery_time(
            smooth_cong,
            EVENT_STEP + EVENT_LEN,
            (EVENT_STEP - 1000, EVENT_STEP),
            (EVENT_STEP, EVENT_STEP + EVENT_LEN + 200),
        )
        peak_cong = max(smooth_cong[EVENT_STEP:EVENT_STEP + EVENT_LEN + 200])
        pre_thr = _mean(res.throughput[EVENT_STEP - 1000:EVENT_STEP])
        ev_thr = _mean(res.throughput[EVENT_STEP:EVENT_STEP + EVENT_LEN])
        thr_drop = max(0.0, pre_thr - ev_thr)

        print(f"  {label:<32} {peak_cong:5.2f}      {recovery:5d}            "
              f"{thr_drop:5.3f}")

        series_load.append({
            "x": xs, "y": smooth_load,
            "label": f"{label} (peak cong={peak_cong:.2f})",
            "color": color, "linewidth": 1.5,
        })
        series_cong.append({
            "x": xs, "y": smooth_cong,
            "label": f"{label} (peak cong={peak_cong:.2f})",
            "color": color, "linewidth": 1.4,
        })
        bar_labels.append(label.split()[0])
        bar_peak.append(peak_cong)
        bar_colors.append(color)

    event_marks = [
        (EVENT_STEP, "#444444", f"accident t={EVENT_STEP}"),
        (EVENT_STEP + EVENT_LEN, "#888888",
         f"clear t={EVENT_STEP + EVENT_LEN}"),
    ]

    plot_lines(
        FIG_DIR / "phase3_disturbance_load.svg",
        series=series_load,
        title="Phase 3: same accident, three regimes - load response",
        xlabel="Simulation step",
        ylabel="Mean load (rolling)",
        vlines=event_marks,
    )

    plot_lines(
        FIG_DIR / "phase3_congestion_spread.svg",
        series=series_cong,
        title="Phase 3: how far the congestion spreads after the accident",
        xlabel="Simulation step",
        ylabel="Congestion-propagation range (rolling)",
        vlines=event_marks,
    )

    plot_bars(
        FIG_DIR / "phase3_recovery_summary.svg",
        categories=bar_labels,
        values=bar_peak,
        title="Phase 3: how far the accident spreads (peak congestion range)",
        ylabel="Peak congestion-propagation range during the event",
        colors=bar_colors,
    )


if __name__ == "__main__":
    run_phase3()
    print(f"Phase 3 done. Figures written to {FIG_DIR}")
