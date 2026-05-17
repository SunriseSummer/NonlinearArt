"""Refactored rush-hour traffic-cascade model used by case1b.

Compared with case1, this version is built around the "morning-rush dispatch"
narrative described in ``全新设计.md``. The mechanics are still in the
spirit of a Bak–Tang–Wiesenfeld sandpile (slow drive + threshold-triggered
cascading relaxation), but the model now exposes the metrics that a traffic
operator actually cares about and supports the experiments in all four
case1b phases:

* **Throughput** — vehicles served (i.e. dissipated) per simulation step.
* **Average delay** — proxied by the running mean load per intersection.
* **Congestion-propagation range** — number of intersections whose load is
  close to the topple threshold at a given step.
* **Disturbance event** — at a configurable step the model freezes one
  intersection for ``disturbance_duration`` steps, mimicking a minor
  accident that cannot relax pressure to neighbours.
* **Local feedback rules** — optional rules that emulate decentralised
  signal/flow control without globally retuning ``spill_prob``.

The model is intentionally kept in the standard library so that the
``solution/`` scripts can import it without extra dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Optional


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DisturbanceSpec:
    """A small accident that blocks a square region for a few steps.

    The blocked region is centred on ``cell`` (or grid centre if ``None``)
    and extends ``radius`` cells in every direction, so a ``radius=0``
    spec freezes a single intersection (1 cell) while ``radius=1`` freezes
    a 3x3 block — useful for modelling an accident that closes a small
    cluster of intersections.
    """

    start: int          # simulation step when the accident occurs
    duration: int       # how many steps the intersections remain frozen
    cell: Optional[tuple[int, int]] = None
    radius: int = 0


@dataclass
class TrafficParams:
    """All knobs the players are expected to tweak in case1b."""

    # --- Network geometry ---------------------------------------------------
    L: int = 20                 # grid side length
    threshold: int = 6          # load that triggers a topple

    # --- Demand & relaxation -----------------------------------------------
    inflow_rate: float = 1.0    # mean number of vehicles arriving per step
    spill_prob: float = 0.18    # probability that a topple unit reaches each neighbour
    dissipation: float = 0.20   # probability that a spill attempt exits the network

    # --- Simulation horizon -------------------------------------------------
    steps: int = 6000
    warmup: int = 1000
    seed: int = 2026

    # --- Disturbance --------------------------------------------------------
    disturbance: Optional[DisturbanceSpec] = None

    # --- Local self-organisation rules (Phase 4) ----------------------------
    local_relief: bool = False  # high-load cells get extra dissipation
    relief_extra: float = 0.20  # added to dissipation when load is high
    relief_load_frac: float = 0.8  # "high" means load >= relief_load_frac * threshold

    inflow_feedback: bool = False  # throttle inflow when network is overloaded
    target_load: float = 2.4    # desired steady-state mean load
    inflow_min_factor: float = 0.2  # inflow can drop to this fraction of nominal
    inflow_gain: float = 0.6    # how aggressively to throttle inflow

    # --- Strong-control adaptive (case1-style proportional spill control) ---
    adaptive_spill: bool = False
    adapt_rate: float = 0.020
    spill_min: float = 0.05
    spill_max: float = 0.45


@dataclass
class TrafficRunResult:
    densities: list[float] = field(default_factory=list)
    throughput: list[int] = field(default_factory=list)
    congestion_range: list[int] = field(default_factory=list)
    inflow_series: list[float] = field(default_factory=list)
    spill_prob_series: list[float] = field(default_factory=list)
    avalanche_sizes: list[int] = field(default_factory=list)
    avalanche_durations: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class RushHourTrafficSystem:
    """2-D grid traffic-cascade model with throughput / disturbance support."""

    def __init__(self, params: TrafficParams):
        self.params = params
        self.rng = random.Random(params.seed)
        self.grid = [[0 for _ in range(params.L)] for _ in range(params.L)]
        self.spill_prob = params.spill_prob
        self.inflow_rate = params.inflow_rate
        # Set of currently-frozen cells (accident block).
        self._block_left = 0
        self._block_cells: set[tuple[int, int]] = set()

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _mean_load(self) -> float:
        L = self.params.L
        return sum(sum(row) for row in self.grid) / (L * L)

    def _high_load_count(self) -> int:
        """Count intersections that are *at or above* the topple threshold —
        i.e. cells that are actively or imminently shedding pressure to
        their neighbours. This is the operationally meaningful definition of
        "congestion-propagation range"."""
        thr = self.params.threshold
        return sum(1 for row in self.grid for v in row if v >= thr)

    def _drive(self) -> list[tuple[int, int]]:
        """Inject ``inflow_rate`` vehicles (rounded stochastically)."""
        rate = max(0.0, self.inflow_rate)
        n_int = int(rate)
        if self.rng.random() < (rate - n_int):
            n_int += 1
        seeds: list[tuple[int, int]] = []
        L = self.params.L
        for _ in range(n_int):
            i = self.rng.randrange(L)
            j = self.rng.randrange(L)
            self.grid[i][j] += 1
            seeds.append((i, j))
        return seeds

    def _maybe_start_disturbance(self, t: int) -> None:
        d = self.params.disturbance
        if d is None or t != d.start:
            return
        cx, cy = d.cell if d.cell is not None else (self.params.L // 2, self.params.L // 2)
        L = self.params.L
        cells: set[tuple[int, int]] = set()
        for dx in range(-d.radius, d.radius + 1):
            for dy in range(-d.radius, d.radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < L and 0 <= y < L:
                    cells.add((x, y))
        self._block_cells = cells
        self._block_left = d.duration

    def _tick_disturbance(self) -> None:
        if self._block_left > 0:
            self._block_left -= 1
            if self._block_left == 0:
                self._block_cells = set()

    # -- relaxation ----------------------------------------------------------

    def _relax(self, seed_cells: list[tuple[int, int]]) -> tuple[int, int, int]:
        """Drain unstable cells; return (size, duration, served).

        ``served`` is the number of vehicles that left the network during the
        cascade — i.e. spill attempts that hit the dissipation outlet, plus
        spills that fall off the grid boundary. This is the operator-visible
        throughput contribution of the cascade.
        """
        L = self.params.L
        threshold = self.params.threshold
        relief_thr = threshold * self.params.relief_load_frac

        # The cascade only fires from cells that are already past threshold.
        frontier = {c for c in seed_cells if self.grid[c[0]][c[1]] >= threshold}
        if not frontier:
            return 0, 0, 0

        size = 0
        duration = 0
        served = 0

        while frontier:
            duration += 1
            unstable = list(frontier)
            frontier = set()
            size += len(unstable)

            for x, y in unstable:
                if self.grid[x][y] < threshold:
                    continue
                # Disturbance: a frozen cell accumulates load but cannot
                # topple — it acts as a hard bottleneck.
                if (x, y) in self._block_cells:
                    continue

                load_before = self.grid[x][y]
                self.grid[x][y] -= threshold

                # Local relief: heavy cells discharge a fraction of vehicles
                # straight out of the network (modelling diversion / extra
                # green time / VMS rerouting).
                local_diss = self.params.dissipation
                if self.params.local_relief and load_before >= relief_thr:
                    local_diss = min(1.0, local_diss + self.params.relief_extra)

                # Each topple removes ``threshold`` vehicles from the cell.
                # We make four spill attempts (one toward each neighbour);
                # any extra capacity (``threshold > 4``) is treated as a
                # fast local outlet and counts as served immediately.
                neighbours = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                served += max(0, threshold - len(neighbours))

                for nx, ny in neighbours:
                    if self.rng.random() < local_diss:
                        # Vehicle exits the network entirely.
                        served += 1
                        continue
                    if self.rng.random() > self.spill_prob:
                        # Spill failed: vehicle is absorbed somewhere
                        # downstream (not tracked, not counted as served).
                        # This is what makes the model non-conservative
                        # and lets the system show a critical transition.
                        continue
                    if 0 <= nx < L and 0 <= ny < L:
                        self.grid[nx][ny] += 1
                        if self.grid[nx][ny] >= threshold:
                            frontier.add((nx, ny))
                    else:
                        # Spill leaves the bounded grid → counts as served.
                        served += 1

                if self.grid[x][y] >= threshold:
                    frontier.add((x, y))

        return size, duration, served

    # -- public driver -------------------------------------------------------

    def run(self) -> TrafficRunResult:
        p = self.params
        result = TrafficRunResult()

        for t in range(p.steps):
            # Strong-control adaptive: classic case1-style proportional spill.
            if p.adaptive_spill:
                err = p.target_load - self._mean_load()
                self.spill_prob = self._clip(
                    self.spill_prob + p.adapt_rate * err,
                    p.spill_min,
                    p.spill_max,
                )

            # Inflow feedback: a self-organising rule that throttles arrivals
            # whenever the network gets dangerously full.
            if p.inflow_feedback:
                err = self._mean_load() - p.target_load
                # err > 0 → too full → reduce inflow; clip to [min, 1]*nominal.
                factor = max(p.inflow_min_factor, 1.0 - p.inflow_gain * max(err, 0.0))
                self.inflow_rate = p.inflow_rate * factor
            else:
                self.inflow_rate = p.inflow_rate

            self._maybe_start_disturbance(t)

            seeds = self._drive()
            # Sample congestion *before* relaxation: that is the moment of
            # peak load each step, when bottlenecks are most visible.
            result.congestion_range.append(self._high_load_count())
            size, duration, served = self._relax(seeds)

            self._tick_disturbance()

            result.densities.append(self._mean_load())
            result.throughput.append(served)
            result.inflow_series.append(self.inflow_rate)
            result.spill_prob_series.append(self.spill_prob)

            if t >= p.warmup and size > 0:
                result.avalanche_sizes.append(size)
                result.avalanche_durations.append(duration)

        return result
