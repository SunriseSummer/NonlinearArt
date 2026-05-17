"""Tiny character-level Transformer used in the SOC-AI criticality study.

Design goals
------------
* Stay well under 20 M parameters (the default config is ~0.3 M).
* Expose a single ``init_gain`` knob that scales **all** trainable matrices
  multiplicatively at initialisation.  This is the control parameter we sweep
  to drive the network through the ordered / critical / chaotic regimes.
* Allow ``forward(..., return_activations=True)`` so the criticality probes
  in ``criticality.py`` can read per-layer hidden states without having to
  re-implement the model.

The architecture is a vanilla pre-norm Transformer decoder with weight tying.
We deliberately use ``F.scaled_dot_product_attention`` from PyTorch (no
attention dropout, no biases on the linear layers) to keep the Jacobian as
clean as possible — that makes the spectral-radius probe interpretable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 20
    block_size: int = 64
    n_layer: int = 6
    n_head: int = 4
    d_model: int = 96
    d_ff: int = 192
    # ``signal_scale`` (g) is the SOC control parameter.  It is a forward-time
    # multiplicative gain applied to the output of every residual block:
    #     x ← g · (x + sublayer(LN(x)))
    # With g = 1 we recover a standard pre-norm Transformer.
    #   g < 1 ⇒ signals decay exponentially with depth   (ordered phase)
    #   g = 1 ⇒ signals propagate without dying/exploding (critical edge)
    #   g > 1 ⇒ signals are amplified geometrically       (chaotic phase)
    # ``init_gain`` is kept at 1.0; we use He-style initialisation.
    signal_scale: float = 1.0
    init_gain: float = 1.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        # forward-time SOC control parameter (shared across layers).  Stored
        # as a buffer (not a parameter) — we never optimise it.
        self.register_buffer("signal_scale",
                             torch.tensor(float(cfg.signal_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + self.attn(self.ln1(x))) * self.signal_scale
        x = (x + self.mlp(self.ln2(x))) * self.signal_scale
        return x


class TinyTransformerLM(nn.Module):
    """Tiny pre-norm Transformer LM with a single ``init_gain`` knob.

    Parameter count for default config: ~0.3 M (≪ 20 M).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # weight tying with the token embedding ⇒ unique output projection
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights(cfg.init_gain)

    # ----- initialisation ---------------------------------------------------
    def _init_weights(self, gain: float) -> None:
        """He-style init for every Linear, then rescale by ``gain``.

        The variance of activations propagated through a layer scales as
        ``gain**2``.  This is exactly the SOC control parameter: at
        ``gain ≈ 1`` the per-layer activation norm is preserved (critical
        edge); ``gain < 1`` damps signals (ordered phase); ``gain > 1``
        amplifies them (chaotic phase).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                std = gain * math.sqrt(2.0 / fan_in)
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                # small embedding init; LM head shares this weight
                nn.init.normal_(m.weight, mean=0.0, std=0.02 * gain)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ----- forward ----------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        return_activations: bool = False,
    ):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence too long"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        activations = [x.detach()] if return_activations else None
        for blk in self.blocks:
            x = blk(x)
            if return_activations:
                activations.append(x.detach())
        x = self.ln_f(x)
        logits = self.head(x)
        if return_activations:
            return logits, activations
        return logits

    # ----- helpers ----------------------------------------------------------
    @torch.no_grad()
    def num_params(self) -> int:
        # head shares weight with tok_emb, count it only once
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total
