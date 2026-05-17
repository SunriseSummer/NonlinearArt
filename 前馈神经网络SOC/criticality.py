from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from model import DeepMLP


@dataclass
class CriticalityReport:
    branching_ratio: float
    lyapunov: float
    eff_rank: float
    power_law_tau: float
    power_law_r2: float


@torch.no_grad()
def branching_ratio(activations: list[torch.Tensor]) -> float:
    layer_norms = [a.norm().item() / np.sqrt(a.numel()) for a in activations]
    ratios = []
    for prev, cur in zip(layer_norms[:-1], layer_norms[1:]):
        if prev > 0 and cur > 0:
            ratios.append(cur / prev)
    if not ratios:
        return float("nan")
    return float(np.exp(np.mean(np.log(ratios))))


def lyapunov_exponent(model: DeepMLP, x: torch.Tensor,
                      n_probes: int = 8, eps: float = 1e-3) -> float:
    model.eval()
    cfg = model.cfg

    def stack_forward(inp: torch.Tensor) -> torch.Tensor:
        h = inp
        for i, layer in enumerate(model.hidden):
            h = torch.relu(model.adaptive_gains[i] * layer(h))
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
    return float(np.mean(log_ratios)) / cfg.n_layer


@torch.no_grad()
def effective_rank(activations: list[torch.Tensor]) -> float:
    h = activations[-1]
    H = h.reshape(-1, h.shape[-1]).float()
    H = H - H.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(H)
    s2 = (s ** 2).clamp(min=1e-12)
    return float((s.sum() ** 2 / s2.sum()).item())


@torch.no_grad()
def avalanche_sizes(activations: list[torch.Tensor], threshold_z: float = 1.0
                    ) -> np.ndarray:
    sizes = []
    for h in activations[1:]:
        std = h.std().item() + 1e-12
        active = (h.abs() > threshold_z * std).float()
        s = active.sum(dim=-1).reshape(-1).cpu().numpy()
        sizes.append(s)
    sizes = np.concatenate(sizes)
    return sizes[sizes >= 1]


@torch.no_grad()
def avalanche_powerlaw(activations: list[torch.Tensor], threshold_z: float = 1.0
                       ) -> tuple[float, float]:
    sizes = avalanche_sizes(activations, threshold_z=threshold_z)
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
    return float(abs(slope)), float(max(0.0, r2))


@torch.no_grad()
def measure(model: DeepMLP, x: torch.Tensor) -> CriticalityReport:
    model.eval()
    _, acts = model(x, return_activations=True)
    br = branching_ratio(acts)
    lam = lyapunov_exponent(model, x)
    er = effective_rank(acts)
    tau, r2 = avalanche_powerlaw(acts)
    return CriticalityReport(
        branching_ratio=br,
        lyapunov=lam,
        eff_rank=er,
        power_law_tau=tau,
        power_law_r2=r2,
    )
