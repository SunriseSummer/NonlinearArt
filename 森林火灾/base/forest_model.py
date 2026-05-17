"""Forest-fire model used by the Case 5 criticality challenge.

The model represents a square forest patch on an ``L x L`` grid.  Each cell is
in one of three states::

    EMPTY = 0   # bare soil / firebreak
    TREE  = 1   # living tree
    FIRE  = 2   # tree currently burning

It supports two complementary modes that map to different parts of the
challenge:

* ``mode = "static"``: the forest is initialised once with tree density ``p``
  and a single ignition is dropped.  The fire then propagates by nearest
  neighbour contact until nothing is burning.  This is the classic
  **percolation / spreading** view used in phase 1 and phase 2 to locate the
  critical density ``p_c ~= 0.5928`` for site percolation on a 2-D square
  lattice.

* ``mode = "soc"``: the Drossel-Schwabl forest fire.  At every time step every
  empty cell becomes a tree with probability ``p_grow`` and every tree is hit
  by lightning with probability ``p_lightning``.  When ``p_lightning << p_grow
  << 1`` the system **self-organises** to a stationary state with a power-law
  fire-size distribution; this is one of the cleanest SOC models known.

Phase 4 adds an optional ``adaptive_thinning`` policy: when the local tree
density inside a cell's neighbourhood exceeds a threshold the cell is
preventively cleared, mimicking firebreaks / controlled thinning.  The policy
is deliberately simple so students can see the trade-off between timber yield
and tail-risk reduction.

Only the Python standard library is used, so the dynamics are easy to read
and modify.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random
from typing import Optional

EMPTY = 0
TREE = 1
FIRE = 2


@dataclass
class FireEvent:
    """Bookkeeping for a single fire (one ignition that has fully burnt out)."""

    start_step: int
    size: int           # total trees burnt
    duration: int       # number of sweeps the fire was alive
    max_front: int      # peak number of simultaneously burning cells


@dataclass
class ForestParams:
    """All knobs exposed to players in Case 5."""

    L: int = 64
    mode: str = "static"  # "static" or "soc"
    seed: int = 2026

    # ---- Static / percolation mode ----
    density: float = 0.55     # initial tree fraction
    ignition: str = "center"  # "center", "edge", "random"

    # ---- SOC / Drossel-Schwabl mode ----
    steps: int = 1500
    warmup: int = 300
    p_grow: float = 0.02       # per empty cell, per sweep
    p_lightning: float = 1.0e-4  # per tree, per sweep
    initial_density: float = 0.0

    # ---- Phase 4 adaptive thinning policy (optional) ----
    adaptive_thinning: bool = False
    thinning_radius: int = 2          # neighbourhood half-width for the local density
    thinning_threshold: float = 0.78  # local density above which cells are cleared
    thinning_rate: float = 0.02       # fraction of qualifying trees cleared per sweep


@dataclass
class ForestRunResult:
    # Time series (one entry per sweep, ``soc`` mode only)
    tree_density: list[float] = field(default_factory=list)
    burning: list[int] = field(default_factory=list)
    ignitions: list[int] = field(default_factory=list)
    thinned: list[int] = field(default_factory=list)
    timber_yield: list[int] = field(default_factory=list)  # trees harvested by thinning

    # Per-fire statistics (``soc`` mode)
    fires: list[FireEvent] = field(default_factory=list)

    # Static-mode summary
    static_fire_size: int = 0
    static_fire_duration: int = 0
    static_largest_cluster: int = 0

    # Final snapshot for plotting (0/1/2 cells)
    final_grid: list[list[int]] = field(default_factory=list)


class ForestFire:
    """2-D forest-fire simulator on a square lattice with periodic boundaries."""

    def __init__(self, params: ForestParams):
        self.params = params
        self.rng = random.Random(params.seed)
        L = params.L
        self.grid = [[EMPTY for _ in range(L)] for _ in range(L)]
        if params.mode == "static":
            self._seed_static()
        elif params.mode == "soc":
            self._seed_soc()
        else:
            raise ValueError("mode must be 'static' or 'soc'")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _seed_static(self) -> None:
        L = self.params.L
        density = self.params.density
        for i in range(L):
            for j in range(L):
                self.grid[i][j] = TREE if self.rng.random() < density else EMPTY

    def _seed_soc(self) -> None:
        L = self.params.L
        density = self.params.initial_density
        for i in range(L):
            for j in range(L):
                self.grid[i][j] = TREE if self.rng.random() < density else EMPTY

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _neighbours(self, i: int, j: int) -> list[tuple[int, int]]:
        L = self.params.L
        return [
            ((i - 1) % L, j),
            ((i + 1) % L, j),
            (i, (j - 1) % L),
            (i, (j + 1) % L),
        ]

    # ------------------------------------------------------------------
    # Static mode: a single fire on a frozen forest
    # ------------------------------------------------------------------
    def run_static(self) -> ForestRunResult:
        """Light a single fire on the frozen forest and let it burn out.

        Returns the size and duration of the fire, plus the size of the
        largest connected tree cluster that existed before the fire (a useful
        order-parameter for percolation studies).
        """
        out = ForestRunResult()
        out.static_largest_cluster = self._largest_cluster_size()
        ignite = self._pick_ignition()
        if ignite is None:
            out.final_grid = [row[:] for row in self.grid]
            return out
        i0, j0 = ignite
        self.grid[i0][j0] = FIRE
        front = deque([(i0, j0)])
        size = 1
        duration = 0
        while front:
            duration += 1
            next_front: deque[tuple[int, int]] = deque()
            while front:
                i, j = front.popleft()
                for ni, nj in self._neighbours(i, j):
                    if self.grid[ni][nj] == TREE:
                        self.grid[ni][nj] = FIRE
                        next_front.append((ni, nj))
                        size += 1
                # cell finishes burning -> empty
                self.grid[i][j] = EMPTY
            front = next_front
        out.static_fire_size = size
        out.static_fire_duration = duration
        out.final_grid = [row[:] for row in self.grid]
        return out

    def _pick_ignition(self) -> Optional[tuple[int, int]]:
        L = self.params.L
        kind = self.params.ignition
        if kind == "center":
            i, j = L // 2, L // 2
            if self.grid[i][j] == TREE:
                return (i, j)
            # fall back to nearest tree
        elif kind == "edge":
            for j in range(L):
                if self.grid[0][j] == TREE:
                    return (0, j)
        # random / fallback
        trees = [(i, j) for i in range(L) for j in range(L) if self.grid[i][j] == TREE]
        if not trees:
            return None
        return self.rng.choice(trees)

    def _largest_cluster_size(self) -> int:
        L = self.params.L
        seen = [[False] * L for _ in range(L)]
        best = 0
        for i in range(L):
            for j in range(L):
                if self.grid[i][j] != TREE or seen[i][j]:
                    continue
                stack = [(i, j)]
                seen[i][j] = True
                size = 0
                while stack:
                    ci, cj = stack.pop()
                    size += 1
                    for ni, nj in self._neighbours(ci, cj):
                        if self.grid[ni][nj] == TREE and not seen[ni][nj]:
                            seen[ni][nj] = True
                            stack.append((ni, nj))
                if size > best:
                    best = size
        return best

    # ------------------------------------------------------------------
    # SOC mode: slow growth + rare lightning
    # ------------------------------------------------------------------
    def run_soc(self) -> ForestRunResult:
        p = self.params
        L = p.L
        out = ForestRunResult()

        # Active fire fronts that span multiple sweeps.  Each entry is a list
        # of currently burning cells; we let one sweep equal "fire advances by
        # one ring" so phase-3 / phase-4 visualisations have a sensible
        # duration.  This is the standard Drossel-Schwabl synchronous update.
        burning: list[tuple[int, int]] = []
        current_fires: list[dict] = []  # list of {"start": t, "size": ..., "duration": .., "max_front": ..}

        for t in range(p.steps):
            # 1) regrowth on empty cells
            for i in range(L):
                row = self.grid[i]
                for j in range(L):
                    if row[j] == EMPTY and self.rng.random() < p.p_grow:
                        row[j] = TREE

            # 2) optional adaptive thinning (phase 4)
            thinned_now = 0
            if p.adaptive_thinning:
                thinned_now = self._adaptive_thinning_step()

            # 3) advance currently burning cells by one ring
            new_burning: list[tuple[int, int]] = []
            for (i, j) in burning:
                for ni, nj in self._neighbours(i, j):
                    if self.grid[ni][nj] == TREE:
                        self.grid[ni][nj] = FIRE
                        new_burning.append((ni, nj))
                self.grid[i][j] = EMPTY
            # update statistics for fires that are still alive
            if new_burning and current_fires:
                # group by fire-id is unnecessary because we treat the union
                # of simultaneously active fronts as one statistic; for this
                # SOC educational model this matches the standard literature
                pass
            # update per-fire bookkeeping: each currently-tracked fire grows
            # by however many of its descendants we just lit
            if current_fires:
                # All active fires merge their fronts; we treat them as a
                # single ongoing fire per ignition step for accounting.  To
                # keep statistics simple, accumulate to the most recent
                # active fire (the SOC literature reports the per-ignition
                # cluster size; we approximate with a single growing front).
                fire = current_fires[-1]
                fire["size"] += len(new_burning)
                fire["duration"] += 1 if (new_burning or burning) else 0
                fire["max_front"] = max(fire["max_front"], len(new_burning))
            burning = new_burning

            # 4) lightning strikes on remaining trees ignite NEW fires
            ignitions = 0
            new_strikes: list[tuple[int, int]] = []
            if p.p_lightning > 0.0:
                # Sample ignitions cell-wise to keep behaviour exact for any p
                for i in range(L):
                    row = self.grid[i]
                    for j in range(L):
                        if row[j] == TREE and self.rng.random() < p.p_lightning:
                            row[j] = FIRE
                            new_strikes.append((i, j))
                            ignitions += 1
            if new_strikes:
                # Close out the previous fire (if any) and start a new one for
                # statistics.
                if current_fires and current_fires[-1].get("open", False):
                    fire = current_fires[-1]
                    fire["open"] = False
                    out.fires.append(
                        FireEvent(
                            start_step=fire["start"],
                            size=fire["size"],
                            duration=fire["duration"],
                            max_front=fire["max_front"],
                        )
                    )
                current_fires.append({
                    "start": t,
                    "size": len(new_strikes),
                    "duration": 1,
                    "max_front": len(new_strikes),
                    "open": True,
                })
                burning.extend(new_strikes)

            # 5) close finished fires (no front and last fire still open)
            if not burning and current_fires and current_fires[-1].get("open", False):
                fire = current_fires[-1]
                fire["open"] = False
                out.fires.append(
                    FireEvent(
                        start_step=fire["start"],
                        size=fire["size"],
                        duration=fire["duration"],
                        max_front=fire["max_front"],
                    )
                )

            # 6) record per-step diagnostics
            n_trees = 0
            for row in self.grid:
                for v in row:
                    if v == TREE:
                        n_trees += 1
            out.tree_density.append(n_trees / (L * L))
            out.burning.append(len(burning))
            out.ignitions.append(ignitions)
            out.thinned.append(thinned_now)
            out.timber_yield.append(thinned_now)

        # Flush any still-open fire
        if current_fires and current_fires[-1].get("open", False):
            fire = current_fires[-1]
            out.fires.append(
                FireEvent(
                    start_step=fire["start"],
                    size=fire["size"],
                    duration=fire["duration"],
                    max_front=fire["max_front"],
                )
            )

        out.final_grid = [row[:] for row in self.grid]
        return out

    def _adaptive_thinning_step(self) -> int:
        """Clear trees in cells whose local density exceeds the threshold.

        The policy is intentionally local: it only looks at a small square
        neighbourhood and only clears a fraction of qualifying cells per
        sweep.  This represents real-world thinning crews that cannot clear an
        entire forest in one go but can systematically reduce risk hotspots.
        """
        p = self.params
        L = p.L
        r = max(1, p.thinning_radius)
        win = (2 * r + 1) ** 2
        thinned = 0
        for i in range(L):
            for j in range(L):
                if self.grid[i][j] != TREE:
                    continue
                count = 0
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        if self.grid[(i + di) % L][(j + dj) % L] == TREE:
                            count += 1
                local = count / win
                if local >= p.thinning_threshold and self.rng.random() < p.thinning_rate:
                    self.grid[i][j] = EMPTY
                    thinned += 1
        return thinned

    # ------------------------------------------------------------------
    # Convenience runner
    # ------------------------------------------------------------------
    def run(self) -> ForestRunResult:
        if self.params.mode == "static":
            return self.run_static()
        return self.run_soc()


__all__ = [
    "EMPTY",
    "TREE",
    "FIRE",
    "FireEvent",
    "ForestParams",
    "ForestRunResult",
    "ForestFire",
]
