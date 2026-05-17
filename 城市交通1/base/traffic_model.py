"""Core traffic-overload cascade model used by starter and challenge solutions.

This module is part of the **base materials** handed to players. It models a
2-D grid of intersections where:

* every step a random intersection receives a new vehicle (slow drive);
* once an intersection's load reaches ``threshold`` it topples — its load
  drops by ``threshold`` and each of the four neighbours independently
  receives a unit of pressure with probability ``spill_prob``;
* each spill attempt may fail with probability ``dissipation`` to model
  vehicles leaving the network entirely (energy loss).

When ``adaptive=True`` the system tunes ``spill_prob`` toward ``target_load``
via a simple proportional controller, which is the entry point for the
self-organized-criticality experiments in phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class TrafficParams:
    L: int = 24                 # grid side length
    threshold: int = 6          # load that triggers a topple
    spill_prob: float = 0.12    # probability that a topple unit reaches each neighbour
    dissipation: float = 0.15   # probability that a spill attempt is lost
    steps: int = 5000           # total simulation steps (drive + relax)
    warmup: int = 1000          # steps to discard before collecting statistics
    seed: int = 42

    # --- Optional adaptive controller (phase 2) -----------------------------
    adaptive: bool = False
    target_load: float = 2.7
    adapt_rate: float = 0.015
    spill_min: float = 0.05
    spill_max: float = 0.45


@dataclass
class TrafficRunResult:
    densities: list[float]
    avalanche_sizes: list[int]
    avalanche_durations: list[int]
    spill_prob_series: list[float]


class TrafficCascadeSystem:
    """2-D traffic pressure model with threshold-triggered local spillovers."""

    def __init__(self, params: TrafficParams):
        self.params = params
        self.rng = random.Random(params.seed)
        self.grid = [[0 for _ in range(params.L)] for _ in range(params.L)]
        self.spill_prob = params.spill_prob

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _mean_load(self) -> float:
        L = self.params.L
        return sum(sum(row) for row in self.grid) / (L * L)

    def _drive(self) -> tuple[int, int]:
        i = self.rng.randrange(self.params.L)
        j = self.rng.randrange(self.params.L)
        self.grid[i][j] += 1
        return i, j

    # --- main relaxation ----------------------------------------------------

    def _relax(self, seed_cell: tuple[int, int]) -> tuple[int, int]:
        """Drain unstable cells until the grid is stable again.

        Returns ``(size, duration)`` of the avalanche triggered from
        ``seed_cell``. ``size`` counts every topple event (a single cell may
        topple multiple times); ``duration`` is the number of synchronous
        relaxation rounds.
        """
        L = self.params.L
        threshold = self.params.threshold

        if self.grid[seed_cell[0]][seed_cell[1]] < threshold:
            return 0, 0

        frontier = {seed_cell}
        size = 0
        duration = 0

        while frontier:
            duration += 1
            unstable = list(frontier)
            frontier = set()
            size += len(unstable)

            for x, y in unstable:
                if self.grid[x][y] < threshold:
                    continue
                self.grid[x][y] -= threshold

                # Up/down/left/right spill attempts. Failed attempts dissipate.
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if self.rng.random() < self.params.dissipation:
                        continue
                    if self.rng.random() > self.spill_prob:
                        continue
                    if 0 <= nx < L and 0 <= ny < L:
                        self.grid[nx][ny] += 1
                        if self.grid[nx][ny] >= threshold:
                            frontier.add((nx, ny))

                # The just-toppled cell may itself still be unstable
                # (e.g. when load was much higher than ``threshold``).
                if self.grid[x][y] >= threshold:
                    frontier.add((x, y))

        return size, duration

    # --- public driver ------------------------------------------------------

    def run(self) -> TrafficRunResult:
        p = self.params
        densities: list[float] = []
        avalanche_sizes: list[int] = []
        avalanche_durations: list[int] = []
        spill_prob_series: list[float] = []

        for t in range(p.steps):
            if p.adaptive:
                # Proportional feedback: high load -> lower spill probability;
                # low load -> raise it. Clipped to a safe range.
                err = p.target_load - self._mean_load()
                self.spill_prob = self._clip(
                    self.spill_prob + p.adapt_rate * err,
                    p.spill_min,
                    p.spill_max,
                )

            cell = self._drive()
            size, duration = self._relax(cell)
            densities.append(self._mean_load())
            spill_prob_series.append(self.spill_prob)

            if t >= p.warmup and size > 0:
                avalanche_sizes.append(size)
                avalanche_durations.append(duration)

        return TrafficRunResult(
            densities=densities,
            avalanche_sizes=avalanche_sizes,
            avalanche_durations=avalanche_durations,
            spill_prob_series=spill_prob_series,
        )
