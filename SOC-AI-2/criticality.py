"""Criticality probes for the SOC-AI-2 deep-MLP study.

These are the same four indicators used in SOC-AI, adapted to a feed-forward
ReLU MLP (no residual / no LayerNorm).  In this textbook setting the
classical edge-of-chaos predictions are expected to hold:

* branching ratio σ_b → 1 at the critical point,
* maximum Lyapunov exponent λ → 0 at the critical point,
* activation avalanche distribution → cleanest power law at the critical point,
* effective rank of representations → peak at the critical point.

All probes are evaluated **before any training step**, on a fixed batch
of test inputs, and therefore characterise the *architecture + init*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from model import DeepMLP


@dataclass
class CriticalityReport:
    branching_ratio: float
    lyapunov: float            # mean log singular value per layer
    power_law_tau: float       # avalanche exponent τ
    power_law_r2: float        # quality of power-law fit
    eff_rank: float            # participation ratio
    layer_norms: list          # per-layer ||h_l|| / sqrt(N)


# ---------------------------------------------------------------------------
@torch.no_grad()
def branching_ratio(activations: list) -> tuple[float, list]:
    """Geometric mean of consecutive normalised-activation norm ratios.

    We use ``||h_l|| / sqrt(numel(h_l))`` so different layer widths are
    compared on the same scale.  The activations list starts with the raw
    input (no ReLU applied), then one entry per hidden layer.
    """
    layer_norms = [a.norm().item() / np.sqrt(a.numel()) for a in activations]
    ratios = []
    for prev, cur in zip(layer_norms[:-1], layer_norms[1:]):
        if prev > 0 and cur > 0:
            ratios.append(cur / prev)
    if not ratios:
        return float("nan"), layer_norms
    return float(np.exp(np.mean(np.log(ratios)))), layer_norms


def lyapunov_exponent(model: DeepMLP, x: torch.Tensor,
                       n_probes: int = 16, eps: float = 1e-3) -> float:
    """Estimate λ = (1/L) E[ log(||f(x+u)-f(x)|| / ||u||) ] over the hidden
    stack (input → final pre-head hidden state).

    For a deterministic deep ReLU MLP this is exactly the depth-averaged
    maximum Lyapunov exponent of the forward map: the average exponential
    rate at which infinitesimal input perturbations are amplified through
    the L layers.  λ < 0 ⇒ ordered, λ = 0 ⇒ critical (edge of chaos),
    λ > 0 ⇒ chaotic.
    """
    model.eval()
    cfg = model.cfg
    # base hidden trajectory
    def stack_forward(inp: torch.Tensor) -> torch.Tensor:
        h = inp
        for layer in model.hidden:
            h = torch.relu(layer(h))
        return h
    base = stack_forward(x)
    log_ratios = []
    for _ in range(n_probes):
        u = torch.randn_like(x)
        u = u / u.norm() * eps
        delta = (stack_forward(x + u) - base).norm().item()
        if delta > 0:
            log_ratios.append(np.log(delta / eps))
    if not log_ratios:
        return float("nan")
    # divide by depth so the number is per-layer
    return float(np.mean(log_ratios)) / cfg.n_layer


@torch.no_grad()
def avalanche_powerlaw(activations: list, threshold_z: float = 1.0
                       ) -> tuple[float, float]:
    """Fit a power law to the activation avalanche-size distribution.

    For each hidden state h_l of shape (B, W) we threshold at
    ``θ = threshold_z · std(h_l)`` and count active units per batch sample.
    Sizes are aggregated across layers and fit on log-spaced bins.

    Returns (τ, R²).  ``τ`` is the power-law exponent (positive number);
    ``R²`` is the linear-regression coefficient of determination in log-log
    space.  R² close to 1 ⇒ good power-law evidence, which is the
    fingerprint of self-organised criticality.
    """
    sizes = []
    for h in activations[1:]:  # skip the raw input
        std = h.std().item() + 1e-12
        active = (h.abs() > threshold_z * std).float()
        s = active.sum(dim=-1).reshape(-1).cpu().numpy()
        sizes.append(s)
    sizes = np.concatenate(sizes)
    sizes = sizes[sizes >= 1]
    if sizes.size < 20:
        return float("nan"), float("nan")
    s_max = sizes.max()
    bins = np.unique(np.round(np.logspace(0, np.log10(max(s_max, 2)), 18)
                              ).astype(int))
    if bins.size < 4:
        return float("nan"), float("nan")
    counts, edges = np.histogram(sizes, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    mask = counts > 0
    if mask.sum() < 4:
        return float("nan"), float("nan")
    x = np.log(centers[mask])
    y = np.log(counts[mask] / widths[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(-slope), float(max(0.0, r2))


@torch.no_grad()
def effective_rank(activations: list) -> float:
    """Participation ratio of singular values of the final hidden state."""
    h = activations[-1]                       # (B, W)
    H = h.reshape(-1, h.shape[-1]).float()
    H = H - H.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(H)
    s2 = (s ** 2).clamp(min=1e-12)
    return float((s.sum() ** 2 / s2.sum()).item())


# ---------------------------------------------------------------------------
def measure(model: DeepMLP, x: torch.Tensor) -> CriticalityReport:
    model.eval()
    with torch.no_grad():
        _, activations = model(x, return_activations=True)
    br, layer_norms = branching_ratio(activations)
    tau, r2 = avalanche_powerlaw(activations)
    er = effective_rank(activations)
    lam = lyapunov_exponent(model, x)
    return CriticalityReport(
        branching_ratio=br,
        lyapunov=lam,
        power_law_tau=tau,
        power_law_r2=r2,
        eff_rank=er,
        layer_norms=layer_norms,
    )
