"""SOC-AI-2 main experiment.

Sweep the weight-init gain σ ∈ [0.3 … 2.5] for a deep ReLU MLP trained on
Fashion-MNIST classification.  At each σ we:

1. build a fresh model;
2. compute four criticality indicators on a frozen batch of test images
   (branching ratio, Lyapunov exponent, avalanche power-law fit,
    effective rank of representations);
3. train for a fixed compute budget (Adam, 800 steps, batch 128);
4. record final test cross-entropy AND test accuracy.

For a non-residual ReLU MLP the classical edge-of-chaos prediction is that
σ = 1 (He scale) is critical: σ_b ≈ 1, λ ≈ 0, signal preserved across depth.
Whether this point also maximises learning efficiency is what the
experiment is here to test, *honestly* — including the case where the
match is partial or absent.

Outputs (under ``SOC-AI-2/``):
  * ``results.json``
  * ``figures/01_param_count.svg``
  * ``figures/02_criticality_probes.svg``
  * ``figures/03_loss_curves.svg``
  * ``figures/04_loss_vs_sigma.svg``
  * ``figures/05_avalanche_distribution.svg``
  * ``figures/06_signal_propagation.svg``
  * ``figures/07_accuracy_vs_sigma.svg``
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
from data import MNIST, MNISTConfig
from model import DeepMLP, ModelConfig


# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SIGMAS = [0.30, 0.50, 0.70, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30, 1.60, 2.00, 2.50]
TRAIN_STEPS = 800
BATCH_SIZE = 128
EVAL_EVERY = 80
EVAL_BATCHES = 16
LR = 1e-3
SEED = 20260516

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")


# ---------------------------------------------------------------------------
def evaluate(model: DeepMLP, corpus: MNIST,
             rng: np.random.Generator) -> tuple[float, float]:
    """Return (mean NLL nats/sample, top-1 accuracy)."""
    model.eval()
    nll_total, acc_total, n = 0.0, 0.0, 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = corpus.batch("test", BATCH_SIZE, rng)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            pred = logits.argmax(dim=-1)
            nll_total += loss.item()
            acc_total += (pred == y).float().sum().item()
            n += y.numel()
    model.train()
    return nll_total / n, acc_total / n


def train_one(sigma: float, corpus: MNIST, seed: int):
    torch.manual_seed(seed)
    cfg = ModelConfig(
        input_dim=corpus.input_dim,
        n_classes=corpus.num_classes,
        width=128, n_layer=12,
        init_gain=sigma,
    )
    model = DeepMLP(cfg).to(device)

    # criticality probes on a frozen batch *before* training
    probe_rng = np.random.default_rng(seed + 999)
    probe_x, _ = corpus.batch("test", BATCH_SIZE, probe_rng)
    probe_x = probe_x.to(device)
    report = measure(model, probe_x)
    with torch.no_grad():
        _, exemplar_activations = model(probe_x, return_activations=True)

    # train ---------------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_rng = np.random.default_rng(seed + 1)
    eval_rng = np.random.default_rng(seed + 2)
    history = {"step": [], "loss": [], "acc": []}
    for step in range(1, TRAIN_STEPS + 1):
        x, y = corpus.batch("train", BATCH_SIZE, train_rng)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            tl, ta = evaluate(model, corpus, eval_rng)
            history["step"].append(step)
            history["loss"].append(tl)
            history["acc"].append(ta)
    final_loss, final_acc = evaluate(model, corpus, eval_rng)
    return model, report, history, final_loss, final_acc, exemplar_activations


# ---------------------------------------------------------------------------
def fmt_report(r: CriticalityReport) -> str:
    return (f"σ_b={r.branching_ratio:.3f} λ={r.lyapunov:+.3f} "
            f"τ={r.power_law_tau:.2f} R²={r.power_law_r2:.2f} "
            f"eff_rank={r.eff_rank:.1f}")


def main():
    t0 = time.time()
    corpus = MNIST(MNISTConfig())
    H_uniform = math.log(corpus.num_classes)
    print(f"[corpus] Fashion-MNIST {corpus.train_x.shape[0]} train / "
          f"{corpus.test_x.shape[0]} test")
    print(f"[corpus] uniform NLL = {H_uniform:.4f} nats/sample "
          f"(accuracy 1/10 = 10%)")

    sanity = DeepMLP(ModelConfig(input_dim=corpus.input_dim,
                                  n_classes=corpus.num_classes))
    n_params = sanity.num_params()
    print(f"[model] parameters = {n_params:,} (budget 20,000,000)")
    assert n_params < 20_000_000

    all_results = []
    histories = {}
    exemplars = {}
    for sigma in SIGMAS:
        print(f"\n=== σ = {sigma:.2f} ===")
        model, report, history, final_loss, final_acc, acts = train_one(
            sigma, corpus, seed=SEED)
        print(f"  init: {fmt_report(report)}")
        print(f"  final test NLL = {final_loss:.4f} nats   "
              f"accuracy = {final_acc * 100:.2f}%")
        all_results.append({
            "sigma": sigma,
            "branching_ratio": report.branching_ratio,
            "lyapunov": report.lyapunov,
            "power_law_tau": report.power_law_tau,
            "power_law_r2": report.power_law_r2,
            "eff_rank": report.eff_rank,
            "layer_norms": report.layer_norms,
            "final_loss": final_loss,
            "final_acc": final_acc,
        })
        histories[sigma] = history
        exemplars[sigma] = [a.cpu() for a in acts]

    sigmas_arr = np.array([r["sigma"] for r in all_results])
    losses_arr = np.array([r["final_loss"] for r in all_results])
    accs_arr = np.array([r["final_acc"] for r in all_results])
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
    ax.set_title("Deep ReLU MLP — parameter count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "01_param_count.svg")
    plt.close(fig)

    # Fig 2: four criticality probes
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    s = sigmas_arr
    axes[0, 0].plot(s, [r["branching_ratio"] for r in all_results],
                    "o-", color="#1f77b4")
    axes[0, 0].axhline(1.0, color="grey", linestyle="--", lw=1,
                       label="critical σ_b = 1")
    axes[0, 0].set_ylabel("branching ratio σ_b")
    axes[0, 0].set_title("(a) signal-propagation branching ratio")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_yscale("log")

    axes[0, 1].plot(s, [r["lyapunov"] for r in all_results],
                    "o-", color="#d62728")
    axes[0, 1].axhline(0.0, color="grey", linestyle="--", lw=1,
                       label="critical λ = 0")
    axes[0, 1].set_ylabel("Lyapunov exponent λ")
    axes[0, 1].set_title("(b) maximum Lyapunov exponent (per layer)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(s, [r["power_law_r2"] for r in all_results],
                    "o-", color="#2ca02c")
    axes[1, 0].set_ylabel("power-law fit R²")
    axes[1, 0].set_xlabel("initialisation gain σ")
    axes[1, 0].set_title("(c) avalanche power-law quality")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].axhline(0.9, color="grey", linestyle=":", lw=1)

    axes[1, 1].plot(s, [r["eff_rank"] for r in all_results],
                    "o-", color="#9467bd")
    axes[1, 1].set_ylabel("effective rank (PR)")
    axes[1, 1].set_xlabel("initialisation gain σ")
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
    ax.axhline(H_uniform, color="grey", linestyle=":",
               label=f"uniform baseline = {H_uniform:.3f}")
    ax.set_xlabel("training step")
    ax.set_ylabel("test NLL (nats / sample)")
    ax.set_title("Learning curves at different init gains σ")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "03_loss_curves.svg")
    plt.close(fig)

    # Fig 4: final loss vs σ
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(s, losses_arr, "o-", color="#e8702a", lw=2, markersize=7,
            label="final test NLL")
    ax.axhline(H_uniform, color="grey", linestyle=":",
               label=f"uniform baseline = {H_uniform:.3f}")
    ax.axvline(1.0, color="#1f77b4", linestyle="--", lw=1.2,
               label="classical critical σ = 1")
    ax.axvline(s[critical_idx], color="#d62728", linestyle="-.",
               label=f"empirical loss-min σ = {s[critical_idx]:.2f}")
    ax.set_xlabel("initialisation gain  σ  (control parameter)")
    ax.set_ylabel("final test NLL (nats / sample)")
    ax.set_title("Learning efficiency vs control parameter σ (Fashion-MNIST, deep MLP)")
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
        if sizes.size == 0:
            ax.set_title(f"{title}\nσ={SIGMAS[idx]:.2f}  (no activity)")
            continue
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
        if mask.sum() >= 2 and np.isfinite(tau):
            xs = np.array([centers[mask].min(), centers[mask].max()])
            ys = (counts[mask] / widths[mask])[0] * (xs / xs[0]) ** (-tau)
            ax.loglog(xs, ys, "--", color="#d62728", lw=1,
                      label=f"slope −{tau:.2f}")
            ax.legend(fontsize=8)
    axes[0].set_ylabel("density P(s)")
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
    ax.set_xlabel("layer index ℓ  (0 = input, 12 = final hidden)")
    ax.set_ylabel("normalised activation norm  ||h_ℓ||/√N")
    ax.set_title("Signal propagation through depth")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "06_signal_propagation.svg")
    plt.close(fig)

    # Fig 7: accuracy vs σ
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(s, accs_arr * 100, "o-", color="#2ca02c", lw=2, markersize=7,
            label="final test accuracy")
    ax.axhline(10.0, color="grey", linestyle=":",
               label="chance level = 10%")
    ax.axvline(1.0, color="#1f77b4", linestyle="--", lw=1.2,
               label="classical critical σ = 1")
    best_acc_idx = int(np.argmax(accs_arr))
    ax.axvline(s[best_acc_idx], color="#d62728", linestyle="-.",
               label=f"empirical best-acc σ = {s[best_acc_idx]:.2f}")
    ax.set_xlabel("initialisation gain  σ")
    ax.set_ylabel("final test top-1 accuracy (%)")
    ax.set_title("Classification accuracy vs control parameter σ")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "07_accuracy_vs_sigma.svg")
    plt.close(fig)

    # save results
    out = {
        "config": {
            "sigmas": SIGMAS, "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE, "lr": LR,
            "seed": SEED, "n_params": n_params,
            "H_uniform": H_uniform,
        },
        "results": all_results,
        "histories": {str(k): v for k, v in histories.items()},
        "exemplar_indices": {"ordered": ordered_idx,
                              "critical": critical_idx,
                              "chaotic": chaotic_idx},
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] {time.time() - t0:.1f}s   "
          f"best σ (loss) = {SIGMAS[critical_idx]:.2f}  "
          f"loss = {losses_arr[critical_idx]:.4f}  "
          f"acc = {accs_arr[critical_idx]*100:.2f}%")


if __name__ == "__main__":
    main()
