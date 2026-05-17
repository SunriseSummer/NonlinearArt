"""Core cortical-network model for the Case 3 neural-avalanche SOC challenge.

This module is part of the **base materials** handed to players. It models a
sparse random network of ``N`` excitable, leaky-integrator neurons with a
synchronous spike-propagation rule. The dynamics belongs to the **mean-field
directed-percolation / branching-process universality class** (Beggs & Plenz,
*J. Neurosci.* 2003). On the same scaffold the player will, in two phases:

* **Phase 1 — tuned criticality.** Hand-tune the synaptic gain ``J`` so that
  the *branching ratio* ``sigma = k * J`` (mean number of post-synaptic
  spikes triggered by one pre-synaptic spike) equals 1, then verify the
  three independent critical exponents ``tau`` (size), ``alpha`` (duration)
  and ``1/(sigma*nu*z)`` (mean size given duration), and check that they
  satisfy the crackling-noise scaling relation
  ``(tau - 1) * sigma*nu*z = alpha - 1``  (Sethna *et al.* 2001).

* **Phase 2 — self-organized criticality.** Switch on **dynamical synapses**
  (Tsodyks–Markram resource depletion + slow recovery, à la Levina,
  Herrmann & Geisel, *Nat. Phys.* 2007). Each spike depletes the synapse
  by a factor ``(1 - epsilon)``; resources recover linearly with time
  constant ``tau_rec``. The system then locks itself onto ``sigma ≈ 1``
  irrespective of the initial ``J`` and irrespective of the random seed.

Three time scales coexist — the SOC trademark:

* **Slow drive** — a single neuron is loaded until it just crosses
  threshold (extremal driving, identical in spirit to Case 2).
* **Fast spike propagation** — synchronous discrete-time update; one
  "round" = one branching generation.
* **Intermediate synaptic recovery** — only active when dynamical
  synapses are enabled; ``tau_rec`` macro-steps to recover full strength.

The implementation uses only the Python standard library so players can
read every line and easily instrument or extend the dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class NeuralParams:
    """All knobs the player is allowed to touch.

    Phase 1 typically only tunes ``J`` — the global synaptic gain that
    controls the branching ratio ``sigma = k * J``. Critical point sits at
    ``J_c = 1 / k``. Phase 2 flips ``dynamical_synapses=True`` and lets
    the resource dynamics tune ``sigma`` to 1 from any initial ``J``.
    """

    # --- Network geometry -------------------------------------------------
    N: int = 256                 # number of neurons
    k: int = 8                   # out-degree of every neuron in the random graph
    threshold: float = 1.0       # spike threshold (units of membrane potential)

    # --- Synapses ---------------------------------------------------------
    J: float = 0.10              # synaptic gain. Critical at J_c = 1/k = 0.125.
    j_disorder: float = 0.0      # multiplicative half-width on per-edge weights
                                 # (0 => homogeneous J, harmless heterogeneity)

    # --- Time / drive -----------------------------------------------------
    drive_steps: int = 6000      # macro-steps to simulate (one avalanche each)
    warmup: int = 1500           # steps to discard before collecting statistics
    drive_kick: float = 1.0      # external input delivered to one random neuron
                                 # per macro-step. With drive_kick == threshold
                                 # every macro-step triggers a primary spike,
                                 # which makes the size statistics mean-field
                                 # branching-process clean.
    seed: int = 7

    # --- Phase-2 dynamical-synapse machinery (Tsodyks–Markram style) -----
    dynamical_synapses: bool = False
    epsilon: float = 0.05        # depression factor per spike, u <- u * (1 - epsilon)
    tau_rec: float = 400.0       # macro-steps to recover toward u = 1
    u_floor: float = 1e-3        # numerical safety floor on synaptic resources

    # --- Stopping criterion for runaway avalanches -----------------------
    # Above the absorbing-state critical point the branching process can
    # produce arbitrarily large bursts. We cap each avalanche so a runaway
    # super-critical ``J`` does not lock the simulation.
    avalanche_size_cap: int = 200_000


@dataclass
class NeuralRunResult:
    """Outputs collected over a full run."""

    # Step-resolved scalar diagnostics (length == drive_steps).
    mean_potential: list[float] = field(default_factory=list)
    branching_ratio: list[float] = field(default_factory=list)  # effective sigma
    mean_resource: list[float] = field(default_factory=list)    # <u_ij>(t)
    J_series: list[float] = field(default_factory=list)         # constant unless
                                                                # the player adapts it

    # Per-avalanche records (only collected after warmup, only if size > 0).
    sizes: list[int] = field(default_factory=list)
    durations: list[int] = field(default_factory=list)
    waiting_times: list[int] = field(default_factory=list)
    event_steps: list[int] = field(default_factory=list)
    # Per-avalanche temporal profile: ``profiles[k][r]`` is the number of
    # spikes that fired in synchronous round ``r`` of avalanche ``k``.
    # Sum equals ``sizes[k]``; length equals ``durations[k]``.
    profiles: list[list[int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class CorticalNetwork:
    """Sparse-graph excitable network with optional dynamical synapses.

    Public API is intentionally tiny: build with :class:`NeuralParams` then
    call :meth:`run`. Helpers ``_drive`` / ``_relax`` are exposed (single
    underscore) so notebooks can override pieces of the dynamics.
    """

    def __init__(self, params: NeuralParams):
        self.params = params
        self.rng = random.Random(params.seed)

        N = params.N
        k = params.k
        if k >= N:
            raise ValueError(f"k={k} must be < N={N}")

        # --- Random sparse out-graph: each neuron picks ``k`` distinct
        #     post-synaptic targets (excluding itself). The graph is fixed
        #     for the entire run (quenched topology).
        self.out_neighbours: list[list[int]] = []
        for i in range(N):
            others = list(range(N))
            others.remove(i)
            self.rng.shuffle(others)
            self.out_neighbours.append(others[:k])

        # --- Per-edge weights with optional small multiplicative disorder.
        #     Edge ``e`` of neuron ``i`` is the e-th in ``self.out_neighbours[i]``.
        if params.j_disorder > 0:
            self.weights: list[list[float]] = [
                [params.J * (1.0 + self.rng.uniform(-params.j_disorder, params.j_disorder))
                 for _ in row]
                for row in self.out_neighbours
            ]
        else:
            self.weights = [[params.J] * k for _ in range(N)]

        # --- Synaptic resources u in (0, 1]. With ``dynamical_synapses=False``
        #     they remain at 1 and do not affect propagation, but we keep
        #     the buffer always allocated to keep the inner loop branch-free.
        self.resources: list[list[float]] = [[1.0] * k for _ in range(N)]

        # --- Membrane potentials. Start at random small values strictly below
        #     threshold so the very first drive step is non-degenerate.
        self.h: list[float] = [self.rng.uniform(0.0, 0.5 * params.threshold)
                               for _ in range(N)]

        # Internal: rolling estimate of the *effective* branching ratio
        # ``sigma_eff = k * J * <u>``. Recomputed each macro-step.
        self.J = params.J

    # ------------------------------------------------------------------ #
    # One macro-step
    # ------------------------------------------------------------------ #

    def _drive(self) -> int:
        """Inject a single external kick into one randomly chosen neuron.

        This is the canonical branching-process protocol used in the Eurich
        and Levina–Herrmann–Geisel papers: each macro-step picks one neuron
        uniformly at random and raises its potential by ``drive_kick``. If
        the kick pushes the neuron over threshold, exactly one *primary*
        spike fires; subsequent spikes in the avalanche are then offspring
        of that primary spike, so the avalanche-size distribution mirrors
        the branching-process offspring statistics very directly. With
        ``drive_kick = threshold`` every macro-step triggers exactly one
        avalanche, which keeps the size and duration histograms clean.
        """
        i = self.rng.randrange(self.params.N)
        self.h[i] += self.params.drive_kick
        return i

    def _relax(self, seed_idx: int) -> tuple[int, int, list[int]]:
        """Run synchronous spike propagation until quiescence.

        Implements a clean **two-pass** synchronous update so a neuron that
        is reset to 0 in the current round cannot be reset *again* in the
        same round after receiving inputs from other neurons that fire in
        the same generation. ``size`` counts every individual spike,
        ``duration`` is the number of synchronous generations, and
        ``profile[r]`` is the number of spikes in generation ``r``
        (so ``sum(profile) == size`` and ``len(profile) == duration``).
        """
        N = self.params.N
        theta = self.params.threshold
        cap = self.params.avalanche_size_cap

        if self.h[seed_idx] < theta:
            return 0, 0, []

        frontier = {seed_idx}
        size = 0
        duration = 0
        profile: list[int] = []
        # Track whether resources need to be updated for this avalanche.
        dyn = self.params.dynamical_synapses
        eps = self.params.epsilon

        while frontier and size < cap:
            duration += 1
            firing = list(frontier)
            frontier = set()
            n_firing = len(firing)
            size += n_firing
            profile.append(n_firing)

            # --- Pass 1: collect post-synaptic kicks before resetting potentials.
            #     We accumulate in a temporary dict so we never mix freshly
            #     reset neurons with neurons that simply received a kick.
            kicks: dict[int, float] = {}
            for i in firing:
                if self.h[i] < theta:
                    # Could happen if a neuron was scheduled into the frontier
                    # by an earlier round but its potential has since been
                    # zeroed by overlapping kicks. Skip silently.
                    continue
                row_w = self.weights[i]
                row_u = self.resources[i]
                row_n = self.out_neighbours[i]
                for e in range(len(row_n)):
                    j = row_n[e]
                    kicks[j] = kicks.get(j, 0.0) + row_w[e] * row_u[e]
                # Synaptic depression: after spiking, every outgoing synapse
                # of i is depleted by (1 - eps). Recovery happens once per
                # macro-step in ``run`` so depletion accumulates within a
                # single avalanche.
                if dyn and eps > 0.0:
                    floor = self.params.u_floor
                    for e in range(len(row_u)):
                        new = row_u[e] * (1.0 - eps)
                        row_u[e] = new if new > floor else floor

            # --- Pass 2: apply resets and kicks. Resets first so a neuron
            #     that received a kick from itself's avalanche partners can
            #     still latch above threshold for the next generation.
            for i in firing:
                self.h[i] = 0.0
            for j, dh in kicks.items():
                self.h[j] += dh
                if self.h[j] >= theta:
                    frontier.add(j)

        return size, duration, profile

    # ------------------------------------------------------------------ #
    # Public driver
    # ------------------------------------------------------------------ #

    def run(self) -> NeuralRunResult:
        """Run ``drive_steps`` macro-steps and return all collected diagnostics."""
        p = self.params
        N = p.N
        k = p.k
        out = NeuralRunResult()
        last_event_step = -1

        # Pre-compute reciprocal recovery time-scale once.
        inv_tau_rec = 1.0 / max(p.tau_rec, 1e-9)

        for t in range(p.drive_steps):
            # 1) Extremal slow drive.
            seed_idx = self._drive()

            # 2) Fast synchronous propagation.
            size, duration, profile = self._relax(seed_idx)

            # 3) Slow synaptic recovery (only if dynamical synapses enabled).
            #    Linear recovery toward u = 1: u_new = u + (1 - u) / tau_rec.
            #    Done once per macro-step (not once per generation) so the
            #    recovery scale is genuinely slower than the propagation.
            if p.dynamical_synapses:
                for i in range(N):
                    row_u = self.resources[i]
                    for e in range(len(row_u)):
                        u = row_u[e]
                        row_u[e] = u + (1.0 - u) * inv_tau_rec

            # 4) Diagnostics.
            mean_h = sum(self.h) / N
            # Mean resource (1.0 unless dynamical synapses are on).
            if p.dynamical_synapses:
                total_u = 0.0
                count_u = 0
                for row in self.resources:
                    for u in row:
                        total_u += u
                        count_u += 1
                mean_u = total_u / count_u
            else:
                mean_u = 1.0
            out.mean_potential.append(mean_h)
            out.mean_resource.append(mean_u)
            # Effective branching ratio at this instant.
            out.branching_ratio.append(k * self.J * mean_u)
            out.J_series.append(self.J)

            if t >= p.warmup and size > 0:
                out.sizes.append(size)
                out.durations.append(duration)
                out.profiles.append(profile)
                out.event_steps.append(t)
                if last_event_step >= 0:
                    out.waiting_times.append(t - last_event_step)
                last_event_step = t

        return out


__all__ = ["NeuralParams", "NeuralRunResult", "CorticalNetwork"]
