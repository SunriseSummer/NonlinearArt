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
    n_layer: int = 12
    init_gain: float = 1.0
    soc_enabled: bool = False
    soc_target: float = 1.0
    soc_eta: float = 0.03
    soc_min_gain: float = 0.25
    soc_max_gain: float = 4.0


class DeepMLP(nn.Module):
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

        # local adaptive gains g_l, one per hidden layer; updated by local rule
        self.register_buffer("adaptive_gains", torch.ones(cfg.n_layer))

    def _init_weights(self, gain: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                std = gain * math.sqrt(2.0 / fan_in)
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def local_soc_update(self, pre_activation_stds: list[torch.Tensor]) -> None:
        if not self.cfg.soc_enabled:
            return
        # local rule: keep each layer's propagation ratio close to target
        # r_l = std(z_l) / std(h_{l-1}), update log g_l += eta * log(target / r_l)
        target = self.cfg.soc_target
        eta = self.cfg.soc_eta
        min_g, max_g = self.cfg.soc_min_gain, self.cfg.soc_max_gain

        for i in range(1, len(pre_activation_stds)):
            prev_std = float(pre_activation_stds[i - 1].item())
            cur_std = float(pre_activation_stds[i].item())
            ratio = cur_std / max(prev_std, 1e-8)
            ratio = max(ratio, 1e-8)
            delta = eta * math.log(target / ratio)
            new_gain = float(self.adaptive_gains[i - 1].item()) * math.exp(delta)
            self.adaptive_gains[i - 1] = torch.tensor(min(max(new_gain, min_g), max_g))

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
        return_pre_stats: bool = False,
    ):
        activations = [x.detach()] if return_activations else None
        pre_stds = [] if return_pre_stats else None
        h = x
        for i, layer in enumerate(self.hidden):
            z = layer(h)
            if return_pre_stats:
                pre_stds.append(z.detach().std())
            g = self.adaptive_gains[i]
            h = F.relu(g * z)
            if return_activations:
                activations.append(h.detach())
        logits = self.head(h)

        out = (logits,)
        if return_activations:
            out += (activations,)
        if return_pre_stats:
            out += (pre_stds,)
        if len(out) == 1:
            return logits
        return out

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
