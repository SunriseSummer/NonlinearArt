"""Synthetic training/test corpus for the SOC-AI study.

Why synthetic?
--------------
* We need a task with a **known optimal cross-entropy** so that "learning
  efficiency" can be measured on an absolute, calibrated scale rather than
  relative to an unknown baseline.
* We need to be reproducible inside the sandbox without downloading any
  third-party corpus.

We therefore generate samples from a fixed **second-order Markov chain**
over a small alphabet (V = 20).  The conditional entropy
``H(X_t | X_{t-1}, X_{t-2})`` is computed analytically from the stationary
distribution and serves as the Bayes-optimal NLL per token.  A perfectly
trained model can approach this floor; the gap between final test loss and
this floor measures how much *learnable structure* the model has captured.

The 2nd-order chain is rich enough to require the Transformer to actually
use its attention over a >1 token window (a bigram model cannot reach
``H_2``), but simple enough that a 0.3 M-parameter model can come close
to optimal in a few hundred steps — exactly what we need for the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CorpusConfig:
    vocab_size: int = 20
    n_train_tokens: int = 60_000
    n_test_tokens: int = 10_000
    block_size: int = 64
    concentration: float = 0.35  # smaller ⇒ sparser / lower-entropy chain
    seed: int = 20260516


class MarkovCorpus:
    """Second-order Markov chain corpus with computed optimal entropy."""

    def __init__(self, cfg: CorpusConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        V = cfg.vocab_size
        # Dirichlet rows give a peaky next-token distribution; this creates
        # genuine structure for the model to learn.
        alpha = np.full(V, cfg.concentration)
        self.P = rng.dirichlet(alpha, size=(V, V)).astype(np.float64)  # P[a,b,c]
        # stationary distribution over pairs (a,b)
        self.pi_pair, self.H_opt = self._compute_stationary_entropy()
        # roll the two sequences once and cache as torch tensors
        self.train = torch.from_numpy(self._sample(cfg.n_train_tokens, rng)).long()
        self.test = torch.from_numpy(self._sample(cfg.n_test_tokens, rng)).long()

    # ------------------------------------------------------------------
    def _compute_stationary_entropy(self):
        """Stationary distribution over (a,b) pairs + conditional entropy."""
        V = self.cfg.vocab_size
        # Build transition matrix on the pair-state (a,b) -> (b,c)
        # M[(a*V+b), (b*V+c)] = P[a,b,c]
        M = np.zeros((V * V, V * V))
        for a in range(V):
            for b in range(V):
                for c in range(V):
                    M[a * V + b, b * V + c] = self.P[a, b, c]
        # Stationary distribution: power-iterate the transpose
        pi = np.full(V * V, 1.0 / (V * V))
        for _ in range(2000):
            pi_new = pi @ M
            pi_new /= pi_new.sum()
            if np.max(np.abs(pi_new - pi)) < 1e-12:
                pi = pi_new
                break
            pi = pi_new
        pi_pair = pi.reshape(V, V)
        # Conditional entropy
        H = 0.0
        for a in range(V):
            for b in range(V):
                row = self.P[a, b]
                # safe log
                with np.errstate(divide="ignore", invalid="ignore"):
                    logp = np.where(row > 0, np.log(row), 0.0)
                H -= pi_pair[a, b] * np.sum(row * logp)
        return pi_pair, float(H)

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        V = self.cfg.vocab_size
        out = np.empty(n, dtype=np.int64)
        # start: draw the first two tokens uniformly
        out[0] = rng.integers(V)
        out[1] = rng.integers(V)
        # vectorise: pre-build cumulative distributions
        cum = np.cumsum(self.P, axis=-1)
        for t in range(2, n):
            a, b = out[t - 2], out[t - 1]
            u = rng.random()
            out[t] = int(np.searchsorted(cum[a, b], u))
        return out

    # ------------------------------------------------------------------
    def batch(self, split: str, batch_size: int, rng: np.random.Generator
              ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.test
        T = self.cfg.block_size
        ix = rng.integers(0, len(data) - T - 1, size=batch_size)
        x = torch.stack([data[i : i + T] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + T] for i in ix])
        return x, y


if __name__ == "__main__":
    cfg = CorpusConfig()
    corpus = MarkovCorpus(cfg)
    print(f"H_opt = {corpus.H_opt:.4f} nats/token "
          f"({corpus.H_opt / np.log(2):.4f} bits)")
    print(f"H_uniform = {np.log(cfg.vocab_size):.4f} nats/token")
