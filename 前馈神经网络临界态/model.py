"""Classic deep feed-forward MLP for the SOC-AI-2 study.

Design goals
------------
* **No residual connections, no normalisation layers.**  Plain
  ``Linear → ReLU → Linear → ReLU → …``  This is the textbook setting where
  classical signal-propagation / edge-of-chaos theory is meant to apply.
* **Single control parameter ``init_gain`` (σ)**.  All hidden weight
  matrices are initialised as
  ``W ~ N(0, σ² · 2 / fan_in)``.
  At σ = 1 this reduces to standard He initialisation, which preserves
  the per-layer activation variance through ReLU — the canonical
  critical edge for a ReLU MLP.
* **Configurable depth ``n_layer``** so signal-propagation effects can be
  amplified by stacking many layers.
* Expose ``forward(..., return_activations=True)`` so the criticality
  probes can read per-layer hidden states directly.

Parameter count for the default config (L=12, width=128, in=784, out=10):
  784·128 + 11·128·128 + 128·10 + biases ≈ 285 k parameters (≪ 20 M).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    input_dim: int = 784
    n_classes: int = 10
    width: int = 128
    n_layer: int = 12         # depth of the hidden stack
    init_gain: float = 1.0    # σ — the SOC control parameter


class DeepMLP(nn.Module):
    """Deep ReLU MLP with a single ``init_gain`` knob."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Linear] = []
        in_dim = cfg.input_dim
        for _ in range(cfg.n_layer):
            layers.append(nn.Linear(in_dim, cfg.width, bias=True))
            in_dim = cfg.width
        self.hidden = nn.ModuleList(layers)
        self.head = nn.Linear(cfg.width, cfg.n_classes, bias=True)
        self._init_weights(cfg.init_gain)

    # ------------------------------------------------------------------
    def _init_weights(self, gain: float) -> None:
        """Scaled He initialisation.

        ``std = gain * sqrt(2 / fan_in)``.  For ReLU, gain = 1 is the
        critical value at which the per-layer activation variance is
        preserved through depth; gain < 1 ⇒ ordered (signal dies);
        gain > 1 ⇒ chaotic (signal explodes).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                std = gain * math.sqrt(2.0 / fan_in)
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
    ):
        # x: (B, input_dim)
        activations = [x.detach()] if return_activations else None
        h = x
        for layer in self.hidden:
            h = F.relu(layer(h))
            if return_activations:
                activations.append(h.detach())
        logits = self.head(h)
        if return_activations:
            return logits, activations
        return logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
