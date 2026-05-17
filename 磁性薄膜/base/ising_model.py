"""Minimal 2-D Ising model used by the Case 4 criticality challenge.

The model represents a square ferromagnetic thin film in contact with a heat
bath.  It intentionally uses only the Python standard library so students can
read and modify every step of the dynamics.  The update rule is standard
single-spin Metropolis dynamics for the Hamiltonian

    E = -J * sum_<ij> s_i s_j - h * sum_i s_i,    s_i in {-1, +1}

with periodic boundaries.  The base model also contains two challenge-oriented
extensions used in later phases:

* a short external-field pulse, for probing critical amplification;
* an optional feedback thermostat that nudges the bath temperature according to
  the observed magnetisation, illustrating self-organised near-criticality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Optional


@dataclass
class FieldPulseSpec:
    """A weak magnetic-field pulse applied during a sweep interval."""

    start: int
    duration: int
    delta_h: float


@dataclass
class IsingParams:
    """All knobs exposed to players in Case 4."""

    L: int = 24
    J: float = 1.0
    temperature: float = 3.5
    field: float = 0.0
    sweeps: int = 1600
    warmup: int = 400
    seed: int = 2026
    initial_state: str = "random"  # random, up, down

    pulse: Optional[FieldPulseSpec] = None

    # Phase-4 feedback thermostat.  The controller is deliberately simple:
    # too-ordered film (|m| above target) -> heat it; too-disordered -> cool it.
    adaptive_temperature: bool = False
    target_abs_m: float = 0.55
    temp_gain: float = 0.08
    temp_min: float = 1.2
    temp_max: float = 4.5
    feedback_window: int = 30


@dataclass
class IsingRunResult:
    magnetisation: list[float] = field(default_factory=list)
    abs_magnetisation: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    temperature_series: list[float] = field(default_factory=list)
    field_series: list[float] = field(default_factory=list)
    accepted_flips: list[int] = field(default_factory=list)
    final_spins: list[list[int]] = field(default_factory=list)


class IsingFilm:
    """Square-lattice Ising film with Metropolis updates."""

    def __init__(self, params: IsingParams):
        self.params = params
        self.rng = random.Random(params.seed)
        self.temperature = params.temperature
        self.field = params.field
        L = params.L
        if params.initial_state == "up":
            self.spins = [[1 for _ in range(L)] for _ in range(L)]
        elif params.initial_state == "down":
            self.spins = [[-1 for _ in range(L)] for _ in range(L)]
        elif params.initial_state == "random":
            self.spins = [[1 if self.rng.random() < 0.5 else -1 for _ in range(L)] for _ in range(L)]
        else:
            raise ValueError("initial_state must be 'random', 'up', or 'down'")

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _local_sum(self, i: int, j: int) -> int:
        L = self.params.L
        return (
            self.spins[(i - 1) % L][j]
            + self.spins[(i + 1) % L][j]
            + self.spins[i][(j - 1) % L]
            + self.spins[i][(j + 1) % L]
        )

    def _sweep(self) -> int:
        """Attempt L*L random spin flips and return how many were accepted."""
        L = self.params.L
        accepted = 0
        beta = 1.0 / max(self.temperature, 1e-9)
        for _ in range(L * L):
            i = self.rng.randrange(L)
            j = self.rng.randrange(L)
            s = self.spins[i][j]
            # Energy change for flipping s -> -s.
            dE = 2.0 * s * (self.params.J * self._local_sum(i, j) + self.field)
            if dE <= 0.0 or self.rng.random() < math.exp(-beta * dE):
                self.spins[i][j] = -s
                accepted += 1
        return accepted

    def _magnetisation(self) -> float:
        L = self.params.L
        return sum(sum(row) for row in self.spins) / (L * L)

    def _energy_per_spin(self) -> float:
        L = self.params.L
        total = 0.0
        for i in range(L):
            for j in range(L):
                s = self.spins[i][j]
                # Count right and down bonds only once.
                total -= self.params.J * s * self.spins[(i + 1) % L][j]
                total -= self.params.J * s * self.spins[i][(j + 1) % L]
                total -= self.field * s
        return total / (L * L)

    def _update_field(self, t: int) -> None:
        p = self.params
        self.field = p.field
        if p.pulse and p.pulse.start <= t < p.pulse.start + p.pulse.duration:
            self.field += p.pulse.delta_h

    def _update_feedback_temperature(self, recent_abs_m: list[float]) -> None:
        p = self.params
        if not p.adaptive_temperature or len(recent_abs_m) < p.feedback_window:
            return
        observed = sum(recent_abs_m[-p.feedback_window:]) / p.feedback_window
        # Positive error means the film is too ordered; heat the bath.
        error = observed - p.target_abs_m
        self.temperature = self._clip(
            self.temperature + p.temp_gain * error,
            p.temp_min,
            p.temp_max,
        )

    def run(self) -> IsingRunResult:
        p = self.params
        out = IsingRunResult()
        recent_abs_m: list[float] = []

        for t in range(p.sweeps):
            self._update_field(t)
            self._update_feedback_temperature(recent_abs_m)
            accepted = self._sweep()
            m = self._magnetisation()
            abs_m = abs(m)
            e = self._energy_per_spin()

            out.magnetisation.append(m)
            out.abs_magnetisation.append(abs_m)
            out.energy.append(e)
            out.temperature_series.append(self.temperature)
            out.field_series.append(self.field)
            out.accepted_flips.append(accepted)
            recent_abs_m.append(abs_m)

        out.final_spins = [row[:] for row in self.spins]
        return out


__all__ = ["FieldPulseSpec", "IsingParams", "IsingRunResult", "IsingFilm"]
