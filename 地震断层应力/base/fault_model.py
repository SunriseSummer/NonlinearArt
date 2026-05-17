"""Core fault-stress cascade model for the Case 2 earthquake SOC challenge.

This module is part of the **base materials** handed to players. It models a
2-D fault patch grid where every site stores a continuous stress level
``s[i][j]`` (think: shear stress accumulated on a small fault asperity).

Three time scales coexist — the SOC trademark:

* **Slow tectonic loading.** Each macro-step the whole grid is uniformly
  loaded by exactly the amount needed to push the most-stressed cell to its
  current strength threshold. This is the standard *extremal* OFC driving
  rule (Olami–Feder–Christensen, 1992) — equivalent to infinitely separating
  loading and rupture time scales.
* **Fast ruptures.** Once a cell reaches its threshold it slips: its stress
  drops to zero and a fraction ``alpha`` of the released stress is handed to
  each of its four neighbours. ``alpha`` is the **conservation parameter**:
  ``4 * alpha`` is the total fraction redistributed. ``alpha = 0.25``
  corresponds to perfect bulk conservation; smaller values dissipate stress
  per rupture (anelastic loss).
* **Intermediate frictional healing.** When ``healing=True`` each just-slipped
  cell becomes temporarily *strong*: its threshold relaxes from
  ``tau_static + heal_amp`` back to a heterogeneous quenched value
  ``tau_static[i][j]`` with characteristic time ``heal_time`` (in macro-steps).
  This is the entry point for the self-organized criticality experiments in
  phase 2: combined with quenched threshold heterogeneity the system locks
  itself near criticality without any manual tuning of ``alpha``.

Open boundaries (cells on the edge have <4 neighbours) are the second source
of dissipation and are essential for SOC: bulk conservation alone would
produce trivial extensive avalanches.

The model is intentionally written with the standard library only: every
expensive inner loop is a plain Python list comprehension or ``for`` loop,
which keeps the code readable for players who want to instrument it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class FaultParams:
    """All knobs the player is allowed to touch.

    Phase 1 typically only tunes ``alpha`` (the dimensionless conservation
    level, the order parameter of the OFC universality class). Phase 2
    flips ``healing=True`` and varies the heterogeneity / healing parameters
    instead, leaving ``alpha`` fixed at a clearly sub-critical value.
    """

    L: int = 32                  # grid side length (LxL fault patch)
    alpha: float = 0.18          # bulk conservation per neighbour (0..0.25)
    threshold: float = 1.0       # static rupture threshold (homogeneous baseline)
    drive_steps: int = 6000      # number of *loading* macro-steps to simulate
    warmup: int = 1500           # macro-steps to discard before collecting stats
    seed: int = 7

    # --- Phase-2 ingredients (off by default, on for SOC experiments) ------
    healing: bool = False        # enable frictional healing dynamics
    heal_amp: float = 0.45       # post-rupture threshold bump (frictional healing)
    heal_time: float = 80.0      # exponential healing time-scale (macro-steps)
    heterogeneity: float = 0.0   # half-width of quenched threshold disorder
                                 # (0.0 => uniform threshold)

    # --- Optional adaptive controller on the conservation parameter --------
    # When ``adaptive=True`` ``alpha`` is nudged on every macro-step toward
    # whatever value keeps the recent **mean cascade size** close to
    # ``target_size``. Mean cascade size is a clean order-parameter for the
    # OFC universality class — it is small in the sub-critical regime,
    # diverges with system size at criticality, and saturates above. By
    # locking onto a fixed ``target_size`` the controller drives alpha to
    # the value at which the *infinite-system* statistics are critical.
    # Combined with ``heterogeneity`` the resulting state is genuinely
    # self-organized (no manual sweep of ``alpha``).
    adaptive: bool = False
    target_size: float = 4.0        # desired rolling mean cascade size
    adapt_rate: float = 8e-4        # proportional gain on alpha
    activity_window: int = 250      # rolling window length used to estimate mean size
    alpha_min: float = 0.10
    alpha_max: float = 0.245

    # --- Stopping criteria for runaway avalanches --------------------------
    # OFC at alpha ~ 0.25 with no dissipation can blow up; we abort any
    # single avalanche that exceeds this many topples. The metric is still
    # recorded (clipped) so the histogram tail is visible.
    avalanche_size_cap: int = 200_000


@dataclass
class FaultRunResult:
    """Outputs collected over a full run."""

    # Step-resolved scalar diagnostics (length == drive_steps).
    mean_stress: list[float] = field(default_factory=list)
    max_stress: list[float] = field(default_factory=list)
    alpha_series: list[float] = field(default_factory=list)
    threshold_mean: list[float] = field(default_factory=list)

    # Per-avalanche records (only collected after warmup, only if size > 0).
    sizes: list[int] = field(default_factory=list)
    durations: list[int] = field(default_factory=list)
    waiting_times: list[int] = field(default_factory=list)  # macro-steps between
                                                            #   successive non-zero events
    event_steps: list[int] = field(default_factory=list)    # macro-step index of
                                                            #   each recorded event
    # Final stress field snapshot (useful for visualisations).
    final_field: list[list[float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class FaultStressSystem:
    """OFC-style continuous-stress earthquake model with optional healing.

    The public API is intentionally tiny — players normally only call
    :meth:`run`. The intermediate helpers (``_drive``, ``_relax``, ...) are
    exposed (single-underscore, not double) so subclasses or notebooks can
    override pieces of the dynamics without forking the whole file.
    """

    def __init__(self, params: FaultParams):
        self.params = params
        self.rng = random.Random(params.seed)

        L = params.L
        # Initial stress: uniform random below threshold so the first drive
        # step is non-degenerate.
        self.stress = [
            [self.rng.uniform(0.0, 0.5 * params.threshold) for _ in range(L)]
            for _ in range(L)
        ]

        # Quenched static threshold per cell. With heterogeneity=0 every cell
        # has the same threshold and the model reduces to vanilla OFC.
        h = params.heterogeneity
        self.tau_static = [
            [params.threshold + (self.rng.uniform(-h, h) if h > 0 else 0.0)
             for _ in range(L)]
            for _ in range(L)
        ]

        # Time since each cell last ruptured. ``+inf`` => never ruptured, so
        # healing contributes 0 (current threshold == tau_static).
        self.last_rupture = [[math.inf] * L for _ in range(L)]

        # Current effective threshold (recomputed lazily when healing is on).
        self.tau = [row[:] for row in self.tau_static]
        self.alpha = params.alpha

    # ------------------------------------------------------------------ #
    # Threshold bookkeeping
    # ------------------------------------------------------------------ #

    def _refresh_thresholds(self, step: int) -> None:
        """Recompute the current threshold of every cell from healing dynamics.

        With ``healing=False`` this is a no-op (saves a quadratic loop per
        macro-step). Otherwise: ``tau(t) = tau_static + heal_amp * exp(-dt / T)``
        where ``dt`` is the time since the last rupture of that cell.
        """
        p = self.params
        if not p.healing:
            return
        L = p.L
        amp = p.heal_amp
        T = max(p.heal_time, 1e-9)
        for i in range(L):
            row_static = self.tau_static[i]
            row_tau = self.tau[i]
            row_last = self.last_rupture[i]
            for j in range(L):
                last = row_last[j]
                if math.isinf(last):
                    # Cell has never ruptured — no healing bump active.
                    row_tau[j] = row_static[j]
                else:
                    dt = step - last
                    row_tau[j] = row_static[j] + amp * math.exp(-dt / T)

    # ------------------------------------------------------------------ #
    # One macro-step
    # ------------------------------------------------------------------ #

    def _drive(self) -> tuple[int, int, float]:
        """Apply uniform tectonic loading until the most-stressed cell trips.

        Returns ``(i, j, gap)`` where ``gap`` is how much stress was added
        globally. Equivalent to the extremal OFC driver: the moment a cell
        reaches its current threshold, the loading stops.
        """
        L = self.params.L
        # Find the cell with smallest "remaining capacity" tau - s.
        best_i = 0
        best_j = 0
        best_gap = math.inf
        for i in range(L):
            row_s = self.stress[i]
            row_t = self.tau[i]
            for j in range(L):
                gap = row_t[j] - row_s[j]
                if gap < best_gap:
                    best_gap = gap
                    best_i = i
                    best_j = j
        if best_gap < 0.0:
            best_gap = 0.0
        # Uniform loading
        for i in range(L):
            row = self.stress[i]
            for j in range(L):
                row[j] += best_gap
        return best_i, best_j, best_gap

    def _relax(self, seed_cell: tuple[int, int]) -> tuple[int, int, set[tuple[int, int]]]:
        """Run synchronous OFC ruptures until every cell is below threshold.

        ``size`` counts each rupture event (a single cell can rupture many
        times during one cascade); ``duration`` is the number of synchronous
        relaxation rounds; ``ruptured`` is the set of unique cells that
        slipped at least once during this avalanche (used by the healing
        bookkeeping in :meth:`run`).
        """
        L = self.params.L
        alpha = self.alpha
        cap = self.params.avalanche_size_cap

        si, sj = seed_cell
        if self.stress[si][sj] < self.tau[si][sj]:
            return 0, 0, set()

        frontier = {seed_cell}
        size = 0
        duration = 0
        ruptured: set[tuple[int, int]] = set()

        while frontier and size < cap:
            duration += 1
            unstable = list(frontier)
            frontier = set()
            size += len(unstable)

            # **Synchronous OFC update** — implemented in two passes to avoid
            # the classic bug where a cell that has already been zeroed in
            # this round receives a contribution and then gets zeroed *again*
            # while reading its (now inflated) value as ``s_old``.
            #
            # Pass 1: snapshot the stress that each unstable cell is about
            # to release, then zero it.
            releases: list[tuple[int, int, float]] = []
            for x, y in unstable:
                if self.stress[x][y] < self.tau[x][y]:
                    continue
                releases.append((x, y, self.stress[x][y]))
                self.stress[x][y] = 0.0
                ruptured.add((x, y))

            # Pass 2: redistribute alpha * s_old to each in-bounds neighbour.
            # Out-of-bounds shares are silently lost — that's the open-boundary
            # dissipation which keeps the system finite at alpha < 0.25.
            for x, y, s_old in releases:
                share = alpha * s_old
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < L and 0 <= ny < L:
                        self.stress[nx][ny] += share
                        if self.stress[nx][ny] >= self.tau[nx][ny]:
                            frontier.add((nx, ny))

        return size, duration, ruptured

    # ------------------------------------------------------------------ #
    # Public driver
    # ------------------------------------------------------------------ #

    def run(self) -> FaultRunResult:
        """Run ``drive_steps`` macro-steps and return all collected diagnostics."""
        p = self.params
        out = FaultRunResult()
        last_event_step = -1
        L = p.L

        # Rolling window of recent cascade sizes for the adaptive controller.
        # We only allocate it when adaptation is requested.
        recent_sizes: list[int] = []

        for t in range(p.drive_steps):
            # 1) Refresh healing-modulated thresholds.
            self._refresh_thresholds(t)

            # 2) Slow drive: load until the next cell trips.
            seed = self._drive()[:2]

            # 3) Fast relaxation: cascade until stable.
            size, duration, ruptured_cells = self._relax(seed)

            # 4) Bookkeeping for cells that just ruptured (healing reset).
            #    We use the exact rupture set returned by ``_relax`` rather
            #    than guessing from the post-cascade stress field, which would
            #    misidentify cells that merely received pushes from neighbours.
            if size > 0 and p.healing:
                for (i, j) in ruptured_cells:
                    self.last_rupture[i][j] = float(t)

            # 5) Slow adaptation of the conservation parameter (Phase 2).
            if p.adaptive:
                recent_sizes.append(size)
                if len(recent_sizes) > p.activity_window:
                    recent_sizes.pop(0)
                # Only act once we have a full window — avoids flailing on
                # the first few warmup steps.
                if len(recent_sizes) == p.activity_window:
                    rolling_mean = sum(recent_sizes) / p.activity_window
                    err = rolling_mean - p.target_size
                    # Too quiet (err < 0) -> push alpha up toward conservative.
                    # Too noisy (err > 0) -> pull alpha down (more dissipation).
                    self.alpha = self._clip(
                        self.alpha - p.adapt_rate * err,
                        p.alpha_min,
                        p.alpha_max,
                    )

            # 6) Diagnostics.
            mean_s = sum(sum(r) for r in self.stress) / (L * L)
            max_s = max(max(r) for r in self.stress)
            mean_tau = (
                sum(sum(r) for r in self.tau) / (L * L)
                if p.healing else p.threshold
            )
            out.mean_stress.append(mean_s)
            out.max_stress.append(max_s)
            out.alpha_series.append(self.alpha)
            out.threshold_mean.append(mean_tau)

            if t >= p.warmup and size > 0:
                out.sizes.append(size)
                out.durations.append(duration)
                out.event_steps.append(t)
                if last_event_step >= 0:
                    out.waiting_times.append(t - last_event_step)
                last_event_step = t

        out.final_field = [row[:] for row in self.stress]
        return out

    # ------------------------------------------------------------------ #
    # Small numerical helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value


__all__ = ["FaultParams", "FaultRunResult", "FaultStressSystem"]
