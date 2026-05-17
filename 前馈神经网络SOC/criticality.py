from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from model import DeepMLP


# --------------------------------------------------------------------------- #
@dataclass
class CriticalityReport:
    branching_ratio: float          # geometric mean of per-layer σ_b
    lyapunov: float                 # max Lyapunov exponent (per layer)
    eff_rank: float                 # participation ratio
    power_law_tau: float            # exponent from response-cascade fit
    power_law_r2: float             # R² of log-log linear fit
    per_layer_br: list = field(default_factory=list)   # σ_b per layer


# --------------------------------------------------------------------------- #
# Core criticality indicators
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _per_layer_branching(activations: list[torch.Tensor],
                         alpha: float = 0.5) -> list[float]:
    """Per-layer Beggs-Plenz branching ratio σ_b_l."""
    brs = []
    for prev, curr in zip(activations[:-1], activations[1:]):
        std_p = prev.std().clamp(min=1e-8).item()
        std_c = curr.std().clamp(min=1e-8).item()
        p_act = (prev.abs() > alpha * std_p).float().mean().item()
        c_act = (curr.abs() > alpha * std_c).float().mean().item()
        brs.append(c_act / max(p_act, 1e-6))
    return brs


@torch.no_grad()
def branching_ratio(activations: list[torch.Tensor]) -> float:
    """Geometric mean of per-layer branching ratios (σ_b)."""
    brs = _per_layer_branching(activations)
    valid = [b for b in brs if b > 0]
    if not valid:
        return float("nan")
    return float(np.exp(np.mean(np.log(valid))))


def lyapunov_exponent(model: DeepMLP, x: torch.Tensor,
                      n_probes: int = 8, eps: float = 1e-3) -> float:
    model.eval()

    def fwd(inp: torch.Tensor) -> torch.Tensor:
        h = inp
        for i, layer in enumerate(model.hidden):
            h = torch.relu(model.adaptive_gains[i] * layer(h))
        return h

    base = fwd(x)
    log_ratios = []
    for _ in range(n_probes):
        u = torch.randn_like(x)
        u = u / u.norm() * eps
        d = (fwd(x + u) - base).norm().item()
        if d > 0:
            log_ratios.append(np.log(d / eps))
    if not log_ratios:
        return float("nan")
    return float(np.mean(log_ratios)) / model.cfg.n_layer


@torch.no_grad()
def effective_rank(activations: list[torch.Tensor]) -> float:
    h = activations[-1]
    H = h.reshape(-1, h.shape[-1]).float()
    H = H - H.mean(0, keepdim=True)
    s = torch.linalg.svdvals(H)
    s2 = s.pow(2).clamp(min=1e-12)
    return float((s.sum() ** 2 / s2.sum()).item())


# --------------------------------------------------------------------------- #
# Response-cascade power-law measurement
# --------------------------------------------------------------------------- #

@torch.no_grad()
def response_cascade_sizes(model: DeepMLP, input_dim: int,
                           n_probes: int = 600,
                           rng_seed: int = 12345,
                           device: str = "cpu") -> np.ndarray:
    """Measure cascade sizes using log-uniformly scaled random probes.

    Rationale
    ---------
    Generate n_probes Gaussian inputs with scales drawn log-uniformly from
    [0.01, 10] (3 decades).  Cascade size = total active neurons across ALL
    hidden layers for one probe.  'Active' uses a GLOBAL threshold (0.5 × batch
    std of each layer).

    At σ_b ≈ 1 (critical):  signal ∝ scale  →  log-uniform scale  →  power law
      P(S) ~ S^{-1} with high R².
    At σ_b < 1 (ordered):   signal decays with depth  →  small cascades, steeper
      slope (τ > 1) or exponential cut-off.
    At σ_b > 1 (chaotic):   signal explodes → saturates at L·W  →  narrow, no
      clear power law.
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)
    log_scales = rng.uniform(np.log(0.01), np.log(10.0), n_probes).astype(np.float32)
    scales = np.exp(log_scales)

    probes = torch.randn(n_probes, input_dim, device=device)
    probes = probes * torch.tensor(scales, device=device).unsqueeze(1)

    _, acts = model(probes, return_activations=True)

    cascade = np.zeros(n_probes, dtype=float)
    for h in acts[1:]:                              # hidden layers only
        g_std = h.std().clamp(min=1e-8).item()      # global std over whole batch
        active = (h.abs() > 0.5 * g_std).float()   # (n_probes, width)
        cascade += active.sum(1).cpu().numpy()

    return cascade[cascade >= 1].astype(float)


def fit_powerlaw(sizes: np.ndarray, n_bins: int = 24) -> tuple[float, float]:
    """Fit P(S) ~ S^{-τ} via OLS in log-log space.  Returns (τ, R²)."""
    sizes = sizes[sizes >= 1]
    if sizes.size < 30:
        return float("nan"), float("nan")
    s_max = float(sizes.max())
    bins = np.unique(
        np.round(np.logspace(0, np.log10(max(s_max, 2)), n_bins)).astype(int)
    )
    if bins.size < 5:
        return float("nan"), float("nan")
    counts, edges = np.histogram(sizes, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    mask = counts > 0
    if mask.sum() < 5:
        return float("nan"), float("nan")
    x = np.log(centers[mask])
    y = np.log(counts[mask] / widths[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coef
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    r2 = float(max(0.0, 1.0 - ss_res / ss_tot))
    return float(abs(slope)), r2


# --------------------------------------------------------------------------- #
# Combined measurement
# --------------------------------------------------------------------------- #

@torch.no_grad()
def measure(model: DeepMLP, x: torch.Tensor,
            n_powerlaw_probes: int = 400) -> CriticalityReport:
    model.eval()
    _, acts = model(x, return_activations=True)
    br    = branching_ratio(acts)
    plbr  = _per_layer_branching(acts)
    lam   = lyapunov_exponent(model, x)
    er    = effective_rank(acts)

    # power-law from response cascades (independent of test batch)
    sizes = response_cascade_sizes(
        model, x.shape[1], n_probes=n_powerlaw_probes, rng_seed=42
    )
    tau, r2 = fit_powerlaw(sizes)

    return CriticalityReport(
        branching_ratio=br, lyapunov=lam, eff_rank=er,
        power_law_tau=tau, power_law_r2=r2, per_layer_br=plbr,
    )
