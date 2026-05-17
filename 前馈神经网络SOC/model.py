from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    input_dim: int = 784
    n_classes: int = 10
    width: int = 128
    n_layer: int = 20           # deeper: more extreme vanishing/exploding
    init_gain: float = 1.0
    soc_enabled: bool = False
    soc_target: float = 1.0     # target branching ratio (Beggs-Plenz)
    soc_eta: float = 0.05       # adaptive-gain learning rate
    soc_threshold: float = 0.5  # active-neuron threshold (× per-layer std)
    soc_min_gain: float = 0.2
    soc_max_gain: float = 6.0   # needs > 1/0.30 ≈ 3.33 for ordered start


class DeepMLP(nn.Module):
    """Deep ReLU MLP.  Optional local SOC adaptive gains maintain σ_b ≈ 1."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden = nn.ModuleList()
        in_dim = cfg.input_dim
        for _ in range(cfg.n_layer):
            self.hidden.append(nn.Linear(in_dim, cfg.width, bias=True))
            in_dim = cfg.width
        self.head = nn.Linear(cfg.width, cfg.n_classes, bias=True)
        self._init_weights(cfg.init_gain)
        # one adaptive gain per hidden layer, initialised to 1
        self.register_buffer("adaptive_gains", torch.ones(cfg.n_layer))

    # ------------------------------------------------------------------
    def _init_weights(self, gain: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                std = gain * math.sqrt(2.0 / fan_in)
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def local_soc_update(self, activations: list[torch.Tensor]) -> None:
        """Std-ratio SOC local rule (does not saturate for deep/chaotic nets).

        For each layer transition l → l+1:
          σ_b_l = std(h_{l+1}) / std(h_l)       ← signal propagation ratio
          log g_l  +=  η · clip( log(target / σ_b_l), -1.5, +1.5 )

        At criticality std(h_l) ≈ const for all l → σ_b = 1.
        The clipped log-update prevents overshooting from very disordered starts.
        """
        if not self.cfg.soc_enabled:
            return
        target = self.cfg.soc_target
        eta = self.cfg.soc_eta
        min_g, max_g = self.cfg.soc_min_gain, self.cfg.soc_max_gain

        for i in range(self.cfg.n_layer):
            std_prev = activations[i].std().clamp(min=1e-12).item()
            std_curr = activations[i + 1].std().clamp(min=1e-12).item()

            ratio = std_curr / std_prev          # ≈ gain_i × init_gain
            ratio = float(np.clip(ratio, 1e-8, 1e8))

            log_ratio = math.log(target / ratio)
            log_ratio = float(np.clip(log_ratio, -1.5, 1.5))   # stability
            delta = eta * log_ratio

            new_gain = float(self.adaptive_gains[i].item()) * math.exp(delta)
            self.adaptive_gains[i] = torch.tensor(
                float(np.clip(new_gain, min_g, max_g)), dtype=torch.float32
            )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_activations: bool = False):
        activations = [x.detach()] if return_activations else None
        h = x
        for i, layer in enumerate(self.hidden):
            z = layer(h)
            g = self.adaptive_gains[i]
            h = F.relu(g * z)
            if return_activations:
                activations.append(h.detach())
        logits = self.head(h)
        if return_activations:
            return logits, activations
        return logits

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
