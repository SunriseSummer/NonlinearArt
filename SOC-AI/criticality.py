"""Criticality probes for the SOC-AI study.

For each value of the control parameter (init gain σ) we want to decide
whether the freshly-initialised Transformer is **ordered**, **critical**, or
**chaotic**.  We use four complementary probes; together they form the
"detection indicators / evidence for the critical state" requested by the
task.

1. **Branching ratio σ_b**
   The classical SOC quantity from branching-process theory.  We approximate
   it as the geometric mean over layers of ``||h_{l+1}|| / ||h_l||``, where
   ``h_l`` are the post-block hidden states from
   ``model.forward(..., return_activations=True)``.
   * σ_b ≪ 1 ⇒ subcritical (signals die out)
   * σ_b ≈ 1 ⇒ critical    (signals propagate without dying or blowing up)
   * σ_b ≫ 1 ⇒ supercritical/chaotic (signals explode)

2. **Mean log-Jacobian singular value (Lyapunov exponent λ)**
   The Jacobian of the full network maps small perturbations at the input
   embedding to small perturbations of the output.  Its mean log singular
   value is the maximum Lyapunov exponent of the deterministic forward
   dynamics.  λ < 0 ordered, λ > 0 chaotic, λ ≈ 0 critical / edge-of-chaos.
   We compute it via random perturbation rather than a full SVD (cheap and
   well-conditioned for our model size).

3. **Activation avalanche distribution**
   Define an "active site" as a hidden unit with ``|h_{l,t,i}| > θ``.  An
   "avalanche" is the count of active sites at one (layer, time) slot.
   At criticality the distribution of avalanche sizes follows a power law
   P(s) ∝ s^{-τ}.  We fit τ on a log-log histogram (least-squares on the
   binned counts) and report both τ and the residual R² of the power-law
   fit (R² close to 1 = good power-law evidence).

4. **Effective rank of representations**
   Critical networks lie on the boundary between ordered (low-rank,
   collapsing) and chaotic (high-entropy, decorrelated) representations.
   The participation ratio of the singular values of the final hidden state
   ``PR = (Σ s_i)^2 / Σ s_i^2`` peaks near criticality.

All probes are computed **before any training step**, on a batch of inputs
drawn from the corpus — so they characterise the dynamical regime of the
*architecture+init*, independent of optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from model import TinyTransformerLM


@dataclass
class CriticalityReport:
    branching_ratio: float
    lyapunov: float            # mean log singular value of Jacobian
    power_law_tau: float       # avalanche exponent τ
    power_law_r2: float        # quality of power-law fit
    eff_rank: float            # participation ratio
    layer_norms: list          # per-layer ||h_l||, useful for plotting


# ----------------------------------------------------------------------------
@torch.no_grad()
def branching_ratio(activations: list) -> tuple[float, list]:
    """Geometric mean over layers of ||h_{l+1}|| / ||h_l||."""
    layer_norms = [a.norm().item() / np.sqrt(a.numel()) for a in activations]
    ratios = []
    for prev, cur in zip(layer_norms[:-1], layer_norms[1:]):
        if prev > 0:
            ratios.append(cur / prev)
    if not ratios:
        return float("nan"), layer_norms
    return float(np.exp(np.mean(np.log(ratios)))), layer_norms


def lyapunov_exponent(model: TinyTransformerLM, idx: torch.Tensor,
                       n_probes: int = 8, eps: float = 1e-3) -> float:
    """Estimate λ = (1/L) E[ log ||J u|| / ||u|| ] using random perturbations
    of the **input embedding** (since input tokens are discrete).

    This is the analogue of the maximum Lyapunov exponent of a continuous
    dynamical system — the average exponential rate at which infinitesimal
    perturbations are amplified through the depth-L network.
    """
    model.eval()
    B, T = idx.shape
    cfg = model.cfg
    device = idx.device
    # base embedding
    pos = torch.arange(T, device=device).unsqueeze(0)
    base_emb = model.tok_emb(idx) + model.pos_emb(pos)
    # forward through stack only (skip embedding & final LN — the final LN
    # would normalise away the magnitude information we want to measure)
    def stack_forward(x):
        for blk in model.blocks:
            x = blk(x)
        return x
    base_out = stack_forward(base_emb)
    log_ratios = []
    for _ in range(n_probes):
        u = torch.randn_like(base_emb)
        u = u / u.norm() * eps
        perturbed = stack_forward(base_emb + u)
        delta = (perturbed - base_out).norm().item()
        if delta > 0:
            log_ratios.append(np.log(delta / eps))
    if not log_ratios:
        return float("nan")
    # divide by depth so the number is per-layer (a true Lyapunov exponent)
    return float(np.mean(log_ratios)) / cfg.n_layer


@torch.no_grad()
def avalanche_powerlaw(activations: list, threshold_z: float = 1.0
                       ) -> tuple[float, float]:
    """Fit a power law to the avalanche-size distribution.

    For each hidden state h_l of shape (B, T, C) we threshold at
    ``θ = threshold_z · std(h_l)`` and count the number of active units per
    (batch, time) slot — that is the "avalanche size" at that location.
    We aggregate sizes across layers and fit ``log N(s) = -τ log s + b``
    using least squares on log-spaced bins with s ≥ 1.
    """
    sizes = []
    for h in activations[1:]:  # skip pure embedding layer
        std = h.std().item() + 1e-12
        active = (h.abs() > threshold_z * std).float()
        # number of active units per (batch, time) site
        s = active.sum(dim=-1).reshape(-1).cpu().numpy()
        sizes.append(s)
    sizes = np.concatenate(sizes)
    sizes = sizes[sizes >= 1]
    if sizes.size < 20:
        return float("nan"), float("nan")
    # log-spaced bins
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
    y = np.log(counts[mask] / widths[mask])  # density to be scale-invariant
    # linear fit
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
    h = activations[-1]                       # (B, T, C)
    H = h.reshape(-1, h.shape[-1]).float()
    H = H - H.mean(dim=0, keepdim=True)
    # singular values via SVD (C is small ⇒ cheap)
    s = torch.linalg.svdvals(H)
    s2 = (s ** 2).clamp(min=1e-12)
    return float((s.sum() ** 2 / s2.sum()).item())


# ----------------------------------------------------------------------------
def measure(model: TinyTransformerLM, idx: torch.Tensor) -> CriticalityReport:
    model.eval()
    with torch.no_grad():
        _, activations = model(idx, return_activations=True)
    br, layer_norms = branching_ratio(activations)
    tau, r2 = avalanche_powerlaw(activations)
    er = effective_rank(activations)
    lam = lyapunov_exponent(model, idx)
    return CriticalityReport(
        branching_ratio=br,
        lyapunov=lam,
        power_law_tau=tau,
        power_law_r2=r2,
        eff_rank=er,
        layer_norms=layer_norms,
    )
