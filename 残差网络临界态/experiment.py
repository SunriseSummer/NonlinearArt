"""SOC-AI main experiment.

We sweep the initialisation gain σ ∈ [0.3 … 2.5] and, at each value, build
a fresh tiny Transformer LM, measure four criticality probes on a frozen
batch, then train it for a **fixed compute budget** on the synthetic
second-order Markov corpus.  We record the test cross-entropy at every
checkpoint and the final test loss.

Outputs
-------
* ``SOC-AI/results.json`` — full numerical results
* ``SOC-AI/figures/01_param_count.svg``     — model sanity check
* ``SOC-AI/figures/02_criticality_probes.svg`` — branching ratio / λ /
  effective rank / power-law fit vs σ
* ``SOC-AI/figures/03_loss_curves.svg``    — train-time loss curves per σ
* ``SOC-AI/figures/04_loss_vs_sigma.svg``  — final test loss vs σ, with
  optimal entropy floor and critical band
* ``SOC-AI/figures/05_avalanche_distribution.svg`` — log-log avalanche
  histograms at three regimes (ordered / critical / chaotic)
* ``SOC-AI/figures/06_signal_propagation.svg`` — per-layer hidden-state
  norms at three regimes
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from criticality import (
    CriticalityReport,
    avalanche_powerlaw,
    measure,
)
from data import CorpusConfig, MarkovCorpus
from model import ModelConfig, TinyTransformerLM


# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SIGMAS = [0.55, 0.70, 0.82, 0.90, 0.97, 1.00, 1.03, 1.10, 1.25, 1.50, 2.00]
TRAIN_STEPS = 600
BATCH_SIZE = 32
EVAL_EVERY = 50
EVAL_BATCHES = 8
LR = 1.5e-3
SEED = 20260516

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")


# ---------------------------------------------------------------------------
def evaluate(model: TinyTransformerLM, corpus: MarkovCorpus,
             rng: np.random.Generator) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = corpus.batch("test", BATCH_SIZE, rng)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   y.reshape(-1))
            total += loss.item() * y.numel()
            count += y.numel()
    model.train()
    return total / count


def train_one(sigma: float, corpus: MarkovCorpus, seed: int):
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=corpus.cfg.vocab_size,
        block_size=corpus.cfg.block_size,
        n_layer=6, n_head=4, d_model=96, d_ff=192,
        signal_scale=sigma,
    )
    model = TinyTransformerLM(cfg).to(device)

    # criticality probes on a frozen batch *before* training
    probe_rng = np.random.default_rng(seed + 999)
    probe_x, _ = corpus.batch("test", BATCH_SIZE, probe_rng)
    probe_x = probe_x.to(device)
    report = measure(model, probe_x)
    # keep one set of activations for later figure 5/6 (regime exemplars)
    with torch.no_grad():
        _, exemplar_activations = model(probe_x, return_activations=True)

    # train ---------------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_rng = np.random.default_rng(seed + 1)
    eval_rng = np.random.default_rng(seed + 2)
    history = {"step": [], "loss": []}
    for step in range(1, TRAIN_STEPS + 1):
        x, y = corpus.batch("train", BATCH_SIZE, train_rng)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        # clip very aggressively — supercritical inits otherwise NaN out
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            tl = evaluate(model, corpus, eval_rng)
            history["step"].append(step)
            history["loss"].append(tl)
    final_loss = evaluate(model, corpus, eval_rng)
    return model, report, history, final_loss, exemplar_activations


# ---------------------------------------------------------------------------
def fmt_report(r: CriticalityReport) -> str:
    return (f"σ_b={r.branching_ratio:.3f} λ={r.lyapunov:+.3f} "
            f"τ={r.power_law_tau:.2f} R²={r.power_law_r2:.2f} "
            f"eff_rank={r.eff_rank:.1f}")


def main():
    t0 = time.time()
    corpus = MarkovCorpus(CorpusConfig())
    print(f"[corpus] H_opt = {corpus.H_opt:.4f} nats/token")
    print(f"[corpus] uniform baseline = {math.log(corpus.cfg.vocab_size):.4f}")

    # parameter sanity-check
    sanity = TinyTransformerLM(ModelConfig(vocab_size=corpus.cfg.vocab_size,
                                            block_size=corpus.cfg.block_size))
    n_params = sanity.num_params()
    print(f"[model] parameters = {n_params:,} (budget 20,000,000)")
    assert n_params < 20_000_000

    all_results = []
    histories = {}
    exemplars = {}  # σ -> activations for chosen regimes
    for sigma in SIGMAS:
        print(f"\n=== σ = {sigma:.2f} ===")
        model, report, history, final, acts = train_one(sigma, corpus,
                                                         seed=SEED)
        print(f"  init: {fmt_report(report)}")
        print(f"  final test NLL = {final:.4f} nats   "
              f"(gap to optimum = {final - corpus.H_opt:+.4f})")
        all_results.append({
            "sigma": sigma,
            "branching_ratio": report.branching_ratio,
            "lyapunov": report.lyapunov,
            "power_law_tau": report.power_law_tau,
            "power_law_r2": report.power_law_r2,
            "eff_rank": report.eff_rank,
            "layer_norms": report.layer_norms,
            "final_loss": final,
        })
        histories[sigma] = history
        exemplars[sigma] = [a.cpu() for a in acts]

    # pick three exemplar regimes for later detailed plots
    sigmas_arr = np.array([r["sigma"] for r in all_results])
    losses_arr = np.array([r["final_loss"] for r in all_results])
    critical_idx = int(np.argmin(losses_arr))
    ordered_idx = 0
    chaotic_idx = len(SIGMAS) - 1

    # ---------------- figures -----------------------------------------
    # Fig 1: parameter count
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["model"], [n_params / 1e6], color="#3a7bd5")
    ax.axhline(20, color="#d62728", linestyle="--",
               label="20 M task budget")
    ax.set_ylabel("parameters (M)")
    ax.set_title("Tiny Transformer LM — parameter count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "01_param_count.svg")
    plt.close(fig)

    # Fig 2: criticality probes
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    s = sigmas_arr
    axes[0, 0].plot(s, [r["branching_ratio"] for r in all_results],
                    "o-", color="#1f77b4")
    axes[0, 0].axhline(1.0, color="grey", linestyle="--", lw=1)
    axes[0, 0].set_ylabel("branching ratio  σ_b")
    axes[0, 0].set_title("(a) signal-propagation branching ratio")

    axes[0, 1].plot(s, [r["lyapunov"] for r in all_results],
                    "o-", color="#d62728")
    axes[0, 1].axhline(0.0, color="grey", linestyle="--", lw=1)
    axes[0, 1].set_ylabel("Lyapunov exponent  λ")
    axes[0, 1].set_title("(b) maximum Lyapunov exponent")

    axes[1, 0].plot(s, [r["power_law_r2"] for r in all_results],
                    "o-", color="#2ca02c", label="power-law R²")
    axes[1, 0].set_ylabel("power-law fit R²")
    axes[1, 0].set_xlabel("forward signal scale  g")
    axes[1, 0].set_title("(c) avalanche power-law quality")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].axhline(0.9, color="grey", linestyle=":", lw=1)

    axes[1, 1].plot(s, [r["eff_rank"] for r in all_results],
                    "o-", color="#9467bd")
    axes[1, 1].set_ylabel("effective rank (PR)")
    axes[1, 1].set_xlabel("forward signal scale  g")
    axes[1, 1].set_title("(d) participation ratio of representations")

    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("Four independent indicators of the critical regime",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "02_criticality_probes.svg")
    plt.close(fig)

    # Fig 3: loss curves
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.cm.viridis
    for i, sigma in enumerate(SIGMAS):
        h = histories[sigma]
        c = cmap(i / max(len(SIGMAS) - 1, 1))
        ax.plot(h["step"], h["loss"], "o-", color=c,
                label=f"σ={sigma:.2f}", markersize=3, lw=1.2)
    ax.axhline(corpus.H_opt, color="black", linestyle="--",
               label=f"optimal H = {corpus.H_opt:.3f}")
    ax.set_xlabel("training step")
    ax.set_ylabel("test NLL (nats / token)")
    ax.set_title("Learning curves at different init gains σ")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "03_loss_curves.svg")
    plt.close(fig)

    # Fig 4: final loss vs sigma
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(s, losses_arr, "o-", color="#e8702a", lw=2, markersize=7)
    ax.axhline(corpus.H_opt, color="black", linestyle="--",
               label=f"Bayes-optimal H = {corpus.H_opt:.3f} nats")
    ax.axhline(math.log(corpus.cfg.vocab_size), color="grey",
               linestyle=":", label="uniform baseline")
    # highlight critical band: where branching ratio is within ±0.15 of 1
    br = np.array([r["branching_ratio"] for r in all_results])
    band = s[np.abs(br - 1.0) < 0.20]
    if len(band) > 0:
        ax.axvspan(band.min() / 1.05, band.max() * 1.05,
                   color="#2ca02c", alpha=0.12,
                   label="critical band (σ_b ≈ 1)")
    ax.axvline(s[critical_idx], color="#d62728", linestyle="-.",
               label=f"loss-minimum σ = {s[critical_idx]:.2f}")
    ax.set_xlabel("forward signal scale  g  (control parameter)")
    ax.set_ylabel("final test NLL (nats / token)")
    ax.set_title("Learning efficiency vs control parameter g")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "04_loss_vs_sigma.svg")
    plt.close(fig)

    # Fig 5: avalanche distributions at three regimes
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, idx, title in zip(
        axes,
        [ordered_idx, critical_idx, chaotic_idx],
        ["ordered", "critical (loss min)", "chaotic"],
    ):
        acts = exemplars[SIGMAS[idx]]
        sizes = []
        for h in acts[1:]:
            std = h.std().item() + 1e-12
            s_ = (h.abs() > std).float().sum(dim=-1).reshape(-1).numpy()
            sizes.append(s_)
        sizes = np.concatenate(sizes)
        sizes = sizes[sizes >= 1]
        s_max = max(int(sizes.max()), 2)
        bins = np.unique(np.round(np.logspace(0, np.log10(s_max), 18)
                                  ).astype(int))
        counts, edges = np.histogram(sizes, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        mask = counts > 0
        ax.loglog(centers[mask], counts[mask] / widths[mask], "o",
                  color="#1f77b4")
        tau, r2 = avalanche_powerlaw(acts)
        ax.set_title(f"{title}\nσ={SIGMAS[idx]:.2f}, τ={tau:.2f}, R²={r2:.2f}")
        ax.set_xlabel("avalanche size  s")
        ax.grid(alpha=0.3, which="both")
        if mask.sum() >= 2:
            xs = np.array([centers[mask].min(), centers[mask].max()])
            ys = (counts[mask] / widths[mask])[0] * (xs / xs[0]) ** (-tau)
            ax.loglog(xs, ys, "--", color="#d62728", lw=1,
                      label=f"slope −{tau:.2f}")
            ax.legend(fontsize=8)
    axes[0].set_ylabel("density  P(s)")
    fig.suptitle("Activation avalanche distributions across regimes",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "05_avalanche_distribution.svg")
    plt.close(fig)

    # Fig 6: signal propagation across depth
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"ordered": "#1f77b4", "critical (loss min)": "#2ca02c",
              "chaotic": "#d62728"}
    for idx, name in zip([ordered_idx, critical_idx, chaotic_idx],
                          ["ordered", "critical (loss min)", "chaotic"]):
        norms = all_results[idx]["layer_norms"]
        ax.plot(range(len(norms)), norms, "o-",
                color=colors[name],
                label=f"{name}  (σ={SIGMAS[idx]:.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("layer index  ℓ")
    ax.set_ylabel("normalised hidden-state norm  ||h_ℓ||/√N")
    ax.set_title("Signal propagation through depth")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "06_signal_propagation.svg")
    plt.close(fig)

    # save results
    out = {
        "config": {
            "sigmas": SIGMAS, "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE, "lr": LR,
            "seed": SEED, "n_params": n_params,
            "H_opt": corpus.H_opt,
            "H_uniform": math.log(corpus.cfg.vocab_size),
        },
        "results": all_results,
        "histories": {str(k): v for k, v in histories.items()},
        "exemplar_indices": {"ordered": ordered_idx,
                              "critical": critical_idx,
                              "chaotic": chaotic_idx},
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] {time.time() - t0:.1f}s   "
          f"best σ = {SIGMAS[critical_idx]:.2f}  "
          f"loss = {losses_arr[critical_idx]:.4f}")


if __name__ == "__main__":
    main()
