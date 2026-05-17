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
def measure(model: DeepMLP, x: torch.Tensor) -> CriticalityReport:
    model.eval()
    _, acts = model(x, return_activations=True)
    br = branching_ratio(acts)
    lam = lyapunov_exponent(model, x)
    er = effective_rank(acts)
    return CriticalityReport(branching_ratio=br, lyapunov=lam, eff_rank=er)
