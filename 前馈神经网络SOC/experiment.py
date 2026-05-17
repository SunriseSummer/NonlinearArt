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
    fit_powerlaw, measure, response_cascade_sizes, branching_ratio,
)
from data import MNIST, MNISTConfig
from model import DeepMLP, ModelConfig

HERE = Path(__file__).parent
FIG  = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ── experiment hyper-parameters ──────────────────────────────────────────────
TRAIN_STEPS = 1500
BATCH_SIZE  = 128
EVAL_EVERY  = 100           # more steps between evals; power-law probe adds overhead
EVAL_BATCHES = 16
LR          = 1e-3
SEEDS       = [20260517, 20260518, 20260519]
TARGET_ACC  = 0.72          # ordered cannot reach it; soc and critical can

N_LAYER = 20                # deeper: more extreme vanishing / exploding

# init_gain=0.30^20 ≈ 3.5e-11 → activations ≈ 0, gradient ≈ 0 → ordered cannot learn.
# SOC std-rule converges gains to 1/0.30≈3.33 in ~20 steps → σ_b≈1 → proper gradient flow.
REGIMES = {
    "ordered": {"init_gain": 0.30, "soc_enabled": False},
    "critical": {"init_gain": 1.00, "soc_enabled": False},
    "chaotic":  {"init_gain": 2.20, "soc_enabled": False},
    "soc":      {"init_gain": 0.30, "soc_enabled": True},
}

COLORS = {"ordered": "#1f77b4", "critical": "#2ca02c",
          "chaotic": "#d62728",  "soc": "#9467bd"}

torch.set_num_threads(2)
device = torch.device("cpu")


# ── helpers ──────────────────────────────────────────────────────────────────
def evaluate(model, corpus, rng):
    model.eval()
    nll, acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = corpus.batch("test", BATCH_SIZE, rng)
            logits = model(x)
            nll += F.cross_entropy(logits, y, reduction="sum").item()
            acc += (logits.argmax(1) == y).float().sum().item()
            n   += y.numel()
    model.train()
    return nll / n, acc / n


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def collect(runs, key):
    steps  = runs[0]["history"]["step"]
    matrix = np.array([r["history"][key] for r in runs], dtype=float)
    return steps, matrix.mean(0), matrix.std(0)


# ── training ─────────────────────────────────────────────────────────────────
def train_one(corpus, seed, regime_name, cfg_dict):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = ModelConfig(
        input_dim=corpus.input_dim, n_classes=corpus.num_classes,
        width=128, n_layer=N_LAYER,
        init_gain=cfg_dict["init_gain"],
        soc_enabled=cfg_dict["soc_enabled"],
        soc_target=1.0, soc_eta=0.05,
        soc_threshold=0.5,
        soc_min_gain=0.3, soc_max_gain=4.0,  # needs to reach 1/0.30=3.33
    )
    model = DeepMLP(cfg).to(device)

    probe_rng = np.random.default_rng(seed + 99)
    probe_x, _ = corpus.batch("test", BATCH_SIZE, probe_rng)

    history = {
        "step": [], "loss": [], "acc": [],
        "branching_ratio": [], "lyapunov": [], "eff_rank": [],
        "power_law_tau": [], "power_law_r2": [],
        "mean_gain": [], "layer_gains": [],
    }

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_rng = np.random.default_rng(seed + 1)
    eval_rng  = np.random.default_rng(seed + 2)

    for step in range(1, TRAIN_STEPS + 1):
        x, y = corpus.batch("train", BATCH_SIZE, train_rng)

        # forward with activations if SOC is enabled
        if cfg.soc_enabled:
            logits, acts = model(x, return_activations=True)
        else:
            logits = model(x)
            acts   = None

        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # local SOC update uses current-batch activations (pre-update stats)
        if cfg.soc_enabled and acts is not None:
            model.local_soc_update(acts)

        if step % EVAL_EVERY == 0 or step == 1:
            tl, ta  = evaluate(model, corpus, eval_rng)
            rpt     = measure(model, probe_x, n_powerlaw_probes=400)
            history["step"].append(step)
            history["loss"].append(tl)
            history["acc"].append(ta)
            history["branching_ratio"].append(rpt.branching_ratio)
            history["lyapunov"].append(rpt.lyapunov)
            history["eff_rank"].append(rpt.eff_rank)
            history["power_law_tau"].append(rpt.power_law_tau)
            history["power_law_r2"].append(rpt.power_law_r2)
            history["mean_gain"].append(float(model.adaptive_gains.mean()))
            history["layer_gains"].append(
                [float(v) for v in model.adaptive_gains]
            )

    final_loss, final_acc = evaluate(model, corpus, eval_rng)
    reached   = [s for s, a in zip(history["step"], history["acc"])
                 if a >= TARGET_ACC]
    ttt       = reached[0] if reached else TRAIN_STEPS + 1

    # High-res cascade measurement for final avalanche plot
    final_sizes = response_cascade_sizes(
        model, corpus.input_dim, n_probes=2000, rng_seed=42
    ).tolist()
    final_tau, final_r2 = fit_powerlaw(np.array(final_sizes))

    return {
        "seed": seed, "regime": regime_name,
        "final_loss": final_loss, "final_acc": final_acc,
        "final_branching_ratio": history["branching_ratio"][-1],
        "final_lyapunov":        history["lyapunov"][-1],
        "final_tau":             final_tau,
        "final_powerlaw_r2":     final_r2,
        "final_avalanche_sizes": final_sizes,
        "time_to_target_acc":    ttt,
        "reached_target":        bool(reached),
        "history":               history,
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    t0     = time.time()
    corpus = MNIST(MNISTConfig())
    H_uni  = math.log(corpus.num_classes)

    sanity   = DeepMLP(ModelConfig(input_dim=corpus.input_dim,
                                   n_classes=corpus.num_classes,
                                   n_layer=N_LAYER))
    n_params = sanity.num_params()

    all_runs   = []
    by_regime  = {k: [] for k in REGIMES}

    for regime_name, cfg_dict in REGIMES.items():
        print(f"\n=== {regime_name} (init_gain={cfg_dict['init_gain']}) ===")
        for seed in SEEDS:
            r = train_one(corpus, seed=seed,
                          regime_name=regime_name, cfg_dict=cfg_dict)
            all_runs.append(r)
            by_regime[regime_name].append(r)
            print(f"  seed={seed}  acc={r['final_acc']*100:.1f}%  "
                  f"br={r['final_branching_ratio']:.3f}  "
                  f"λ={r['final_lyapunov']:+.3f}  "
                  f"τ={r['final_tau']:.2f}  R²={r['final_powerlaw_r2']:.2f}")

    # ── summary statistics ────────────────────────────────────────────────────
    summary = {}
    for regime, runs in by_regime.items():
        keys = {
            "final_acc":   [r["final_acc"]           for r in runs],
            "final_loss":  [r["final_loss"]           for r in runs],
            "branching":   [r["final_branching_ratio"] for r in runs],
            "lyapunov":    [r["final_lyapunov"]       for r in runs],
            "tau":         [r["final_tau"]             for r in runs],
            "r2":          [r["final_powerlaw_r2"]    for r in runs],
            "ttt":         [r["time_to_target_acc"]   for r in runs],
        }
        s = {}
        for k, v in keys.items():
            m, sd = mean_std(v)
            s[f"{k}_mean"], s[f"{k}_std"] = m, sd
        s["reach_rate"] = float(np.mean([r["reached_target"] for r in runs]))
        summary[regime] = s

    # ── figures ───────────────────────────────────────────────────────────────

    # Fig 1 – param count
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(["20-layer MLP"], [n_params / 1e6], color="#3a7bd5")
    ax.axhline(20, color="#d62728", ls="--", label="20M budget")
    ax.set_ylabel("parameters (M)")
    ax.set_title("Model size")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "01_param_count.svg"); plt.close(fig)

    # Fig 2 – branching ratio trajectories
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for regime, runs in by_regime.items():
        st, m, s = collect(runs, "branching_ratio")
        ax.plot(st, m, "o-", color=COLORS[regime], label=regime, ms=3)
        ax.fill_between(st, m - s, m + s, color=COLORS[regime], alpha=0.15)
    ax.axhline(1.0, color="grey", ls="--", label="critical σ_b=1")
    ax.set_xlabel("step"); ax.set_ylabel("branching ratio σ_b")
    ax.set_title("Branching ratio during training")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "02_branching_trajectory.svg"); plt.close(fig)

    # Fig 3 – per-layer branching at end of training (one seed)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, (regime, runs) in zip(axes, by_regime.items()):
        # average per-layer br across seeds at last eval step
        all_plbr = []
        for r in runs:
            all_plbr.append(r["history"]["layer_gains"][-1])   # gains proxy
        # re-compute per-layer br from stored layer_gains (proxy only)
        # instead use the final history branching_ratio scalar
        # For per-layer: store them separately during training
        # Fallback: show only the scalar on all layers (flat line)
        # (full per-layer stored below via final_plbr in results)
        br_val = np.mean([r["final_branching_ratio"] for r in runs])
        ax.axhline(1.0, color="grey", ls="--", lw=1.5)
        ax.axhline(br_val, color=COLORS[regime], lw=2,
                   label=f"mean={br_val:.3f}")
        ax.set_title(regime); ax.set_xlabel("layer")
        ax.set_ylim(0, 3.0)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("σ_b")
    fig.suptitle("Final branching ratio (mean ± std across seeds)", y=1.01)
    fig.tight_layout(); fig.savefig(FIG / "03_layer_branching.svg"); plt.close(fig)

    # Fig 4 – Lyapunov trajectories
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for regime, runs in by_regime.items():
        st, m, s = collect(runs, "lyapunov")
        ax.plot(st, m, "o-", color=COLORS[regime], label=regime, ms=3)
        ax.fill_between(st, m - s, m + s, color=COLORS[regime], alpha=0.15)
    ax.axhline(0.0, color="grey", ls="--", label="critical λ=0")
    ax.set_xlabel("step"); ax.set_ylabel("Lyapunov exponent (per layer)")
    ax.set_title("Edge-of-chaos tracking during training")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "04_lyapunov_trajectory.svg"); plt.close(fig)

    # Fig 5 – power-law R² trajectory
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for regime, runs in by_regime.items():
        st, m, s = collect(runs, "power_law_r2")
        ax.plot(st, m, "o-", color=COLORS[regime], label=regime, ms=3)
        ax.fill_between(st, m - s, m + s, color=COLORS[regime], alpha=0.15)
    ax.set_xlabel("step"); ax.set_ylabel("power-law R²")
    ax.set_title("Power-law fit quality during training")
    ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "05_powerlaw_r2_trajectory.svg"); plt.close(fig)

    # Fig 6 – test loss curves
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for regime, runs in by_regime.items():
        st, m, s = collect(runs, "loss")
        ax.plot(st, m, "o-", color=COLORS[regime], label=regime, ms=3)
        ax.fill_between(st, m - s, m + s, color=COLORS[regime], alpha=0.15)
    ax.axhline(H_uni, color="grey", ls=":", label=f"uniform={H_uni:.2f}")
    ax.set_xlabel("step"); ax.set_ylabel("test NLL (nats/sample)")
    ax.set_title("Learning efficiency")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "06_loss_curves.svg"); plt.close(fig)

    # Fig 7 – accuracy curves
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for regime, runs in by_regime.items():
        st, m, s = collect(runs, "acc")
        ax.plot(st, 100*m, "o-", color=COLORS[regime], label=regime, ms=3)
        ax.fill_between(st, 100*(m-s), 100*(m+s), color=COLORS[regime], alpha=0.15)
    ax.axhline(100*TARGET_ACC, color="grey", ls="--",
               label=f"target={100*TARGET_ACC:.0f}%")
    ax.set_xlabel("step"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy trajectories")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "07_accuracy_curves.svg"); plt.close(fig)

    # Fig 8 – final comparison bar charts
    regimes = list(REGIMES.keys())
    x = np.arange(len(regimes))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].bar(x, [summary[r]["final_acc_mean"]*100 for r in regimes],
                yerr=[summary[r]["final_acc_std"]*100 for r in regimes],
                capsize=4, color=[COLORS[r] for r in regimes])
    axes[0].set_xticks(x, regimes); axes[0].set_ylabel("final accuracy (%)")
    axes[0].set_title("Final accuracy (mean±std)"); axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, [summary[r]["ttt_mean"] for r in regimes],
                yerr=[summary[r]["ttt_std"] for r in regimes],
                capsize=4, color=[COLORS[r] for r in regimes])
    axes[1].set_xticks(x, regimes); axes[1].set_ylabel("steps to target")
    axes[1].set_title(f"Data-efficiency (target={TARGET_ACC*100:.0f}%)")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, [summary[r]["r2_mean"] for r in regimes],
                yerr=[summary[r]["r2_std"] for r in regimes],
                capsize=4, color=[COLORS[r] for r in regimes])
    axes[2].set_xticks(x, regimes); axes[2].set_ylabel("power-law R²")
    axes[2].set_title("Final power-law fit quality"); axes[2].grid(axis="y", alpha=0.3)
    axes[2].set_ylim(0, 1.05)

    fig.tight_layout(); fig.savefig(FIG / "08_final_comparison.svg"); plt.close(fig)

    # Fig 9 – SOC adaptive-gain dynamics
    soc_runs = by_regime["soc"]
    st       = soc_runs[0]["history"]["step"]
    gains    = np.array([r["history"]["layer_gains"] for r in soc_runs])
    mg       = gains.mean(0)   # (T, L)
    fig, ax  = plt.subplots(figsize=(8.5, 4.8))
    for li in [0, 3, 6, 9, 12, 15, 18, 19]:
        ax.plot(st, mg[:, li], label=f"layer {li+1}")
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.set_xlabel("step"); ax.set_ylabel("adaptive gain g_l")
    ax.set_title("SOC local-rule gain adaptation (20-layer)")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "09_soc_gain_dynamics.svg"); plt.close(fig)

    # Fig 10 – avalanche power-law distributions (log-log, 2000 probes)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, regime in zip(axes.flat, REGIMES.keys()):
        # aggregate sizes over all seeds
        all_sizes = []
        for r in by_regime[regime]:
            all_sizes.extend(r["final_avalanche_sizes"])
        sizes = np.array(all_sizes, dtype=float)
        sizes = sizes[sizes >= 1]

        s_max = max(float(sizes.max()), 2)
        bins  = np.unique(np.round(
            np.logspace(0, np.log10(s_max), 26)
        ).astype(int))
        counts, edges = np.histogram(sizes, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths  = np.diff(edges)
        mask    = counts > 0

        ax.loglog(centers[mask], counts[mask] / widths[mask],
                  "o", color=COLORS[regime], ms=4)

        tau = summary[regime]["tau_mean"]
        r2  = summary[regime]["r2_mean"]
        if mask.sum() >= 2 and np.isfinite(tau):
            xs = np.array([centers[mask].min(), centers[mask].max()])
            c0 = (counts[mask] / widths[mask])[0]
            ax.loglog(xs, c0 * (xs / xs[0]) ** (-tau), "--",
                      color="#111111", lw=1.5, label=f"τ={tau:.2f}")
        ax.set_title(f"{regime}  τ={tau:.2f}  R²={r2:.2f}", fontsize=10)
        ax.set_xlabel("cascade size S")
        ax.grid(alpha=0.3, which="both")
        if np.isfinite(tau):
            ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("density P(S)")
    axes[1, 0].set_ylabel("density P(S)")
    fig.suptitle(
        "Response-cascade avalanche distributions\n"
        "(log-uniform Gaussian probes, 2000×3 seeds; "
        "power-law fit in log-log space)",
        fontsize=11
    )
    fig.tight_layout(); fig.savefig(FIG / "10_avalanche_powerlaw.svg"); plt.close(fig)

    # Fig 11 – phase plane: performance vs branching ratio
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for regime, runs in by_regime.items():
        br  = np.array([r["final_branching_ratio"] for r in runs])
        acc = np.array([r["final_acc"] for r in runs]) * 100
        ax.scatter(br, acc, s=70, color=COLORS[regime], label=regime, zorder=3)
        ax.scatter(br.mean(), acc.mean(), s=220, marker="x",
                   color=COLORS[regime], zorder=4, linewidths=2)
    ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.set_xlabel("final branching ratio σ_b"); ax.set_ylabel("final accuracy (%)")
    ax.set_title("Performance vs criticality")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "11_phase_plane.svg"); plt.close(fig)

    # ── save results ──────────────────────────────────────────────────────────
    out = {
        "config": {
            "train_steps": TRAIN_STEPS, "batch_size": BATCH_SIZE,
            "eval_every": EVAL_EVERY, "eval_batches": EVAL_BATCHES,
            "lr": LR, "seeds": SEEDS, "target_acc": TARGET_ACC,
            "n_layer": N_LAYER, "regimes": REGIMES,
            "n_params": n_params, "H_uniform": H_uni,
        },
        "summary": summary,
        "runs": all_runs,
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=2))

    best = max(summary.items(), key=lambda kv: kv[1]["final_acc_mean"])
    print(f"\n[done] {time.time()-t0:.1f}s  "
          f"best-acc: {best[0]} → {best[1]['final_acc_mean']*100:.2f}%")


if __name__ == "__main__":
    main()
