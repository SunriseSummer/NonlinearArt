"""Phase 4: strong control vs. local-rule self-organisation.

The dispatcher faces the same difficult shift as phase 3 — high demand and a
mid-shift accident — but is given two very different policies:

* **Mode A (strong control)** — the controller manually picks an aggressive
  ``spill_prob`` that maximises throughput in steady state. There are no
  feedback rules. When the accident hits, the policy cannot react: it
  keeps pushing pressure into the affected area at the same rate.
* **Mode B (self-organising local rules)** — the controller installs two
  decentralised rules that case1b's model exposes:
  - ``inflow_feedback``: when the network mean load grows past
    ``target_load``, throttle the inflow gate (modelling on-ramp
    metering / TDM advice).
  - ``local_relief``: heavily loaded intersections discharge a few extra
    vehicles to side streets (modelling adaptive green time / VMS
    rerouting on overloaded corridors).

Both modes face the same inflow profile, the same disturbance, the same
RNG. We compare:

* mean throughput before / during / after the disturbance,
* peak congestion-propagation range during the disturbance,
* recovery time of the congestion-range metric.

The lesson: in a *static* high-throughput configuration, mode A wins on
throughput in calm weather. Once the system gets hit, mode B's
decentralised feedback contains the spread and recovers faster — exactly
the "good rules beat global hand-tuning under uncertainty" argument from
``全新设计.md``.

Outputs:
    case1b/figures/phase4_compare_load.svg
    case1b/figures/phase4_compare_congestion.svg
    case1b/figures/phase4_summary.svg
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


L = 20
THRESHOLD = 6
DISSIPATION = 0.18
WARMUP = 1500
STEPS = 5500
EVENT_STEP = 3000
EVENT_LEN = 600
EVENT_RADIUS = 1
INFLOW = 2.0  # demanding rush hour, well into the stressed regime


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _peak(values, lo, hi):
    sub = values[lo:hi]
    return max(sub) if sub else 0.0


def _recovery_steps(metric: list[float], event_end: int,
                    pre_window: tuple[int, int],
                    peak_window: tuple[int, int]) -> int:
    base = _mean(metric[pre_window[0]:pre_window[1]])
    peak = _peak(metric, peak_window[0], peak_window[1])
    excess = max(0.0, peak - base)
    if excess < 1e-6:
        return 0
    target = base + 0.1 * excess
    for t in range(event_end, len(metric)):
        if metric[t] <= target:
            return t - event_end
    return len(metric) - event_end


def _build_params(*, mode: str) -> TrafficParams:
    common = dict(
        L=L, threshold=THRESHOLD,
        inflow_rate=INFLOW, dissipation=DISSIPATION,
        steps=STEPS, warmup=WARMUP, seed=2026,
        disturbance=DisturbanceSpec(
            start=EVENT_STEP, duration=EVENT_LEN,
            cell=(L // 2, L // 2), radius=EVENT_RADIUS,
        ),
    )
    if mode == "A":  # strong control: hand-picked aggressive routing
        return TrafficParams(spill_prob=0.30, **common)
    if mode == "B":  # local rules: feedback + relief
        return TrafficParams(
            spill_prob=0.30,
            inflow_feedback=True,
            target_load=2.4,
            inflow_gain=0.8,
            local_relief=True,
            relief_extra=0.20,
            **common,
        )
    raise ValueError(mode)


def run_phase4() -> None:
    runs = []
    print("[phase4] strong control (A) vs local-rule self-organisation (B)")
    print("  mode          <thrpt_pre>  <thrpt_event>  <thrpt_post>  "
          "peak_cong  recovery")

    for label, mode, color in [
        ("Mode A: strong control", "A", "#1f77b4"),
        ("Mode B: local rules",    "B", "#2ca02c"),
    ]:
        params = _build_params(mode=mode)
        res = RushHourTrafficSystem(params).run()

        smooth_load = rolling_mean(res.densities, 30)
        smooth_cong = rolling_mean(res.congestion_range, 30)
        smooth_thr = rolling_mean(res.throughput, 80)

        thr_pre = _mean(res.throughput[EVENT_STEP - 1000:EVENT_STEP])
        thr_event = _mean(res.throughput[EVENT_STEP:EVENT_STEP + EVENT_LEN])
        thr_post = _mean(res.throughput[
            EVENT_STEP + EVENT_LEN:EVENT_STEP + EVENT_LEN + 1000
        ])
        peak_cong = _peak(smooth_cong, EVENT_STEP, EVENT_STEP + EVENT_LEN + 200)
        recovery = _recovery_steps(
            smooth_cong,
            EVENT_STEP + EVENT_LEN,
            (EVENT_STEP - 1000, EVENT_STEP),
            (EVENT_STEP, EVENT_STEP + EVENT_LEN + 200),
        )

        print(f"  {label:<18} {thr_pre:7.3f}      {thr_event:7.3f}      "
              f"{thr_post:7.3f}     {peak_cong:5.2f}    {recovery:5d}")

        runs.append({
            "label": label, "color": color,
            "load": smooth_load, "cong": smooth_cong, "thr": smooth_thr,
            "thr_pre": thr_pre, "thr_event": thr_event, "thr_post": thr_post,
            "peak_cong": peak_cong, "recovery": recovery,
            "n": len(res.densities),
        })

    xs = list(range(runs[0]["n"]))
    event_marks = [
        (EVENT_STEP, "#444444", f"accident t={EVENT_STEP}"),
        (EVENT_STEP + EVENT_LEN, "#888888",
         f"clear t={EVENT_STEP + EVENT_LEN}"),
    ]

    plot_lines(
        FIG_DIR / "phase4_compare_load.svg",
        series=[
            {"x": xs, "y": r["load"], "label": r["label"],
             "color": r["color"], "linewidth": 1.6}
            for r in runs
        ],
        title="Phase 4: load trajectory under the same accident",
        xlabel="Simulation step",
        ylabel="Mean load (rolling)",
        vlines=event_marks,
    )

    plot_lines(
        FIG_DIR / "phase4_compare_congestion.svg",
        series=[
            {"x": xs, "y": r["cong"],
             "label": f"{r['label']} (peak={r['peak_cong']:.2f}, "
                      f"recovery={r['recovery']})",
             "color": r["color"], "linewidth": 1.5}
            for r in runs
        ],
        title=("Phase 4: congestion-propagation range - "
               "local rules contain the spread"),
        xlabel="Simulation step",
        ylabel="Congestion-propagation range (rolling)",
        vlines=event_marks,
    )

    # Bar summary: peak congestion + throughput drop during the event.
    categories = [r["label"].split(":")[0] for r in runs]
    values = [r["peak_cong"] for r in runs]
    plot_bars(
        FIG_DIR / "phase4_summary.svg",
        categories=categories,
        values=values,
        title=("Phase 4: peak congestion spread under the same accident "
               "(lower is better)"),
        ylabel="Peak congestion-propagation range",
        colors=[r["color"] for r in runs],
    )


if __name__ == "__main__":
    run_phase4()
    print(f"Phase 4 done. Figures written to {FIG_DIR}")
