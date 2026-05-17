"""Vicsek flocking model used by the Case 6 criticality challenge.

We simulate ``N`` self-propelled point agents (drones / fish / birds) moving
on a periodic 2-D box of side ``box_size``.  Each agent has a fixed speed
``v0`` and a heading angle ``theta``.  Once per time step every agent updates
its heading to the *average heading of all neighbours within radius* ``r``
plus a uniform random kick of width ``eta``::

    theta_i(t+1) = arg( sum_{j: |x_j-x_i|<r} e^{i theta_j(t)} ) + xi,
    xi ~ Uniform(-eta/2, +eta/2)
    x_i(t+1) = x_i(t) + v0 * (cos theta, sin theta)

This is the 1995 Vicsek model.  In the thermodynamic limit it shows a
continuous order-disorder phase transition driven by the noise amplitude
``eta``: at low noise the swarm is collectively polarised (order parameter
``phi = |<v>| / v0`` close to 1), at high noise the headings randomise (phi
near 0), and in between there is a critical noise ``eta_c(rho)`` where the
susceptibility-like fluctuations of phi peak.

The Case 6 challenge uses three orthogonal extensions of the basic model:

* ``leader_index`` / ``leader_theta``: pin one agent's heading to a fixed
  direction.  This is the "control input" used in phase 3 to measure how
  far / how fast a command propagates through the swarm at different noise
  levels.

* ``perturbation``: a brief external "wind" that adds a deterministic angle
  shift to every agent for a short window, used to probe critical
  amplification.

* ``adaptive_noise``: a feedback rule that nudges ``eta`` up when the swarm
  is too ordered and down when it is too disordered, illustrating
  self-organised near-criticality (phase 4).

Only the Python standard library is used, so the dynamics are easy to read
and modify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Optional


TWO_PI = 2.0 * math.pi


@dataclass
class PerturbationSpec:
    """A deterministic angular kick applied to every agent for a window."""

    start: int
    duration: int
    delta_theta: float  # radians


@dataclass
class VicsekParams:
    """All knobs exposed to players in Case 6."""

    n_agents: int = 400
    box_size: float = 10.0
    speed: float = 0.4
    radius: float = 1.0
    eta: float = 1.5            # noise amplitude in radians
    steps: int = 800
    warmup: int = 200
    seed: int = 2026

    # Phase-3 leader / perturbation controls
    leader_index: Optional[int] = None     # if set, agent[leader] is always
    leader_theta: float = 0.0              # forced to this heading
    perturbation: Optional[PerturbationSpec] = None

    # Phase-4 adaptive noise feedback
    adaptive_noise: bool = False
    target_phi: float = 0.55
    eta_gain: float = 0.04
    eta_min: float = 0.05
    eta_max: float = 2 * math.pi
    feedback_window: int = 30


@dataclass
class VicsekRunResult:
    phi: list[float] = field(default_factory=list)              # |<v>|/v0
    mean_heading: list[float] = field(default_factory=list)     # arg(<v>)
    eta_series: list[float] = field(default_factory=list)
    field_series: list[float] = field(default_factory=list)     # extra angle (perturbation)
    leader_alignment: list[float] = field(default_factory=list) # cos(theta_i - leader_theta), averaged
    final_positions: list[tuple[float, float]] = field(default_factory=list)
    final_thetas: list[float] = field(default_factory=list)


class VicsekFlock:
    """Self-propelled particles with Vicsek alignment on a periodic box."""

    def __init__(self, params: VicsekParams):
        self.params = params
        self.rng = random.Random(params.seed)
        self.eta = params.eta
        L = params.box_size
        self.x = [self.rng.random() * L for _ in range(params.n_agents)]
        self.y = [self.rng.random() * L for _ in range(params.n_agents)]
        self.theta = [self.rng.random() * TWO_PI for _ in range(params.n_agents)]
        if params.leader_index is not None:
            self.theta[params.leader_index] = params.leader_theta

    # ------------------------------------------------------------------
    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _build_cells(self) -> tuple[list[list[list[int]]], int, float]:
        """Return a grid index and its spacing for fast neighbour queries."""
        p = self.params
        # cell side >= radius keeps neighbour search local.
        side = p.radius
        ncells = max(1, int(p.box_size / side))
        # ensure cells_size * ncells >= box_size
        cell = p.box_size / ncells
        cells: list[list[list[int]]] = [[[] for _ in range(ncells)] for _ in range(ncells)]
        for i in range(p.n_agents):
            cx = int(self.x[i] / cell) % ncells
            cy = int(self.y[i] / cell) % ncells
            cells[cx][cy].append(i)
        return cells, ncells, cell

    def _step_headings(self, extra_field: float) -> None:
        """Update every heading using the mean of neighbours plus random noise."""
        p = self.params
        L = p.box_size
        r2 = p.radius * p.radius
        cells, ncells, cell = self._build_cells()
        new_theta = [0.0] * p.n_agents
        for i in range(p.n_agents):
            cx = int(self.x[i] / cell) % ncells
            cy = int(self.y[i] / cell) % ncells
            sx = 0.0
            sy = 0.0
            count = 0
            xi, yi = self.x[i], self.y[i]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx = (cx + dx) % ncells
                    ny = (cy + dy) % ncells
                    for j in cells[nx][ny]:
                        # periodic distance
                        ddx = self.x[j] - xi
                        if ddx > L * 0.5:
                            ddx -= L
                        elif ddx < -L * 0.5:
                            ddx += L
                        ddy = self.y[j] - yi
                        if ddy > L * 0.5:
                            ddy -= L
                        elif ddy < -L * 0.5:
                            ddy += L
                        if ddx * ddx + ddy * ddy <= r2:
                            sx += math.cos(self.theta[j])
                            sy += math.sin(self.theta[j])
                            count += 1
            if count == 0:
                base = self.theta[i]
            else:
                base = math.atan2(sy, sx)
            xi_noise = (self.rng.random() - 0.5) * self.eta
            new_theta[i] = base + xi_noise + extra_field
        self.theta = new_theta
        if p.leader_index is not None:
            self.theta[p.leader_index] = p.leader_theta

    def _move(self) -> None:
        p = self.params
        L = p.box_size
        v = p.speed
        for i in range(p.n_agents):
            self.x[i] = (self.x[i] + v * math.cos(self.theta[i])) % L
            self.y[i] = (self.y[i] + v * math.sin(self.theta[i])) % L

    def _polarisation(self) -> tuple[float, float]:
        """Return (phi, mean_angle).  phi = |<v>|/v0 in [0, 1]."""
        sx = 0.0
        sy = 0.0
        for th in self.theta:
            sx += math.cos(th)
            sy += math.sin(th)
        n = len(self.theta)
        magnitude = math.hypot(sx, sy) / n
        angle = math.atan2(sy, sx)
        return magnitude, angle

    def _leader_alignment(self) -> float:
        p = self.params
        if p.leader_index is None:
            return 0.0
        ref = p.leader_theta
        s = 0.0
        for th in self.theta:
            s += math.cos(th - ref)
        return s / len(self.theta)

    def _update_field(self, t: int) -> float:
        p = self.params
        if p.perturbation and p.perturbation.start <= t < p.perturbation.start + p.perturbation.duration:
            return p.perturbation.delta_theta
        return 0.0

    def _update_feedback_eta(self, recent_phi: list[float]) -> None:
        p = self.params
        if not p.adaptive_noise or len(recent_phi) < p.feedback_window:
            return
        observed = sum(recent_phi[-p.feedback_window:]) / p.feedback_window
        # too ordered (phi above target) -> add noise; too disordered -> reduce.
        error = observed - p.target_phi
        self.eta = self._clip(
            self.eta + p.eta_gain * error,
            p.eta_min,
            p.eta_max,
        )

    # ------------------------------------------------------------------
    def run(self) -> VicsekRunResult:
        p = self.params
        out = VicsekRunResult()
        recent_phi: list[float] = []
        for t in range(p.steps):
            extra = self._update_field(t)
            self._update_feedback_eta(recent_phi)
            self._step_headings(extra)
            self._move()
            phi, ang = self._polarisation()
            out.phi.append(phi)
            out.mean_heading.append(ang)
            out.eta_series.append(self.eta)
            out.field_series.append(extra)
            out.leader_alignment.append(self._leader_alignment())
            recent_phi.append(phi)
        out.final_positions = list(zip(self.x[:], self.y[:]))
        out.final_thetas = self.theta[:]
        return out


__all__ = [
    "PerturbationSpec",
    "VicsekParams",
    "VicsekRunResult",
    "VicsekFlock",
]
