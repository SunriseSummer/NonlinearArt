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

from criticality import avalanche_powerlaw, avalanche_sizes, measure
from data import MNIST, MNISTConfig
from model import DeepMLP, ModelConfig

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TRAIN_STEPS = 700
BATCH_SIZE = 128
EVAL_EVERY = 50
EVAL_BATCHES = 16
LR = 1e-3
SEEDS = [20260517, 20260518, 20260519]
TARGET_ACC = 0.84

REGIMES = {
    "ordered": {"init_gain": 0.70, "soc_enabled": False},
    "critical": {"init_gain": 1.00, "soc_enabled": False},
    "chaotic": {"init_gain": 1.30, "soc_enabled": False},
    "soc": {"init_gain": 0.70, "soc_enabled": True},
}


torch.set_num_threads(2)
device = torch.device("cpu")


def evaluate(model: DeepMLP, corpus: MNIST,
             rng: np.random.Generator) -> tuple[float, float]:
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


def train_one(corpus: MNIST, seed: int, regime_name: str, cfg_dict: dict):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = ModelConfig(
        input_dim=corpus.input_dim,
        n_classes=corpus.num_classes,
        width=128,
        n_layer=12,
        init_gain=cfg_dict["init_gain"],
        soc_enabled=cfg_dict["soc_enabled"],
        soc_target=1.0,
        soc_eta=0.03,
        soc_min_gain=0.25,
        soc_max_gain=4.0,
    )
    model = DeepMLP(cfg).to(device)

    probe_rng = np.random.default_rng(seed + 99)
    probe_x, _ = corpus.batch("test", BATCH_SIZE, probe_rng)
    probe_x = probe_x.to(device)

    history = {
        "step": [], "loss": [], "acc": [], "branching_ratio": [],
        "lyapunov": [], "eff_rank": [], "power_law_tau": [], "power_law_r2": [],
        "mean_gain": [],
        "layer_gains": [],
    }

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_rng = np.random.default_rng(seed + 1)
    eval_rng = np.random.default_rng(seed + 2)

    for step in range(1, TRAIN_STEPS + 1):
        x, y = corpus.batch("train", BATCH_SIZE, train_rng)
        out = model(x, return_pre_stats=cfg.soc_enabled)
        if cfg.soc_enabled:
            logits, pre_stds = out
        else:
            logits = out
            pre_stds = None

        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if cfg.soc_enabled and pre_stds is not None:
            model.local_soc_update(pre_stds)

        if step % EVAL_EVERY == 0 or step == 1:
            tl, ta = evaluate(model, corpus, eval_rng)
            rpt = measure(model, probe_x)
            history["step"].append(step)
            history["loss"].append(tl)
            history["acc"].append(ta)
            history["branching_ratio"].append(rpt.branching_ratio)
            history["lyapunov"].append(rpt.lyapunov)
            history["eff_rank"].append(rpt.eff_rank)
            history["power_law_tau"].append(rpt.power_law_tau)
            history["power_law_r2"].append(rpt.power_law_r2)
            mg = float(model.adaptive_gains.mean().item())
            history["mean_gain"].append(mg)
            history["layer_gains"].append([float(v.item()) for v in model.adaptive_gains])

    final_loss, final_acc = evaluate(model, corpus, eval_rng)
    reached = [s for s, a in zip(history["step"], history["acc"]) if a >= TARGET_ACC]
    time_to_target = reached[0] if reached else TRAIN_STEPS + 1
    with torch.no_grad():
        _, final_acts = model(probe_x, return_activations=True)
    final_sizes = avalanche_sizes(final_acts).astype(float).tolist()
    final_tau, final_r2 = avalanche_powerlaw(final_acts)

    result = {
        "seed": seed,
        "regime": regime_name,
        "final_loss": final_loss,
        "final_acc": final_acc,
        "final_branching_ratio": history["branching_ratio"][-1],
        "final_lyapunov": history["lyapunov"][-1],
        "final_tau": final_tau,
        "final_powerlaw_r2": final_r2,
        "final_avalanche_sizes": final_sizes,
        "time_to_target_acc": time_to_target,
        "reached_target": bool(reached),
        "history": history,
    }
    return result


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def collect_step_stats(runs: list[dict], key: str):
    steps = runs[0]["history"]["step"]
    matrix = np.array([r["history"][key] for r in runs], dtype=float)
    return steps, matrix.mean(axis=0), matrix.std(axis=0)


def main():
    t0 = time.time()
    corpus = MNIST(MNISTConfig())
    H_uniform = math.log(corpus.num_classes)

    sanity = DeepMLP(ModelConfig(input_dim=corpus.input_dim, n_classes=corpus.num_classes))
    n_params = sanity.num_params()

    all_runs = []
    by_regime: dict[str, list[dict]] = {k: [] for k in REGIMES}

    for regime_name, cfg_dict in REGIMES.items():
        print(f"\n=== regime: {regime_name} ===")
        for seed in SEEDS:
            r = train_one(corpus, seed=seed, regime_name=regime_name, cfg_dict=cfg_dict)
            all_runs.append(r)
            by_regime[regime_name].append(r)
            print(f"seed={seed} loss={r['final_loss']:.4f} acc={r['final_acc']*100:.2f}% "
                  f"br={r['final_branching_ratio']:.3f} λ={r['final_lyapunov']:+.3f}")

    summary = {}
    for regime, runs in by_regime.items():
        m_loss, s_loss = mean_std([r["final_loss"] for r in runs])
        m_acc, s_acc = mean_std([r["final_acc"] for r in runs])
        m_br, s_br = mean_std([r["final_branching_ratio"] for r in runs])
        m_lam, s_lam = mean_std([r["final_lyapunov"] for r in runs])
        m_tau, s_tau = mean_std([r["final_tau"] for r in runs])
        m_r2, s_r2 = mean_std([r["final_powerlaw_r2"] for r in runs])
        ttt = [r["time_to_target_acc"] for r in runs]
        m_ttt, s_ttt = mean_std(ttt)
        reached_rate = float(np.mean([r["reached_target"] for r in runs]))

        summary[regime] = {
            "final_loss_mean": m_loss,
            "final_loss_std": s_loss,
            "final_acc_mean": m_acc,
            "final_acc_std": s_acc,
            "final_branching_mean": m_br,
            "final_branching_std": s_br,
            "final_lyapunov_mean": m_lam,
            "final_lyapunov_std": s_lam,
            "final_tau_mean": m_tau,
            "final_tau_std": s_tau,
            "final_powerlaw_r2_mean": m_r2,
            "final_powerlaw_r2_std": s_r2,
            "time_to_target_mean": m_ttt,
            "time_to_target_std": s_ttt,
            "target_reach_rate": reached_rate,
        }

    # Figure 1 parameter count
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["model"], [n_params / 1e6], color="#3a7bd5")
    ax.axhline(20, color="#d62728", linestyle="--", label="20M task budget")
    ax.set_ylabel("parameters (M)")
    ax.set_title("Deep ReLU MLP parameter count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "01_param_count.svg")
    plt.close(fig)

    colors = {"ordered": "#1f77b4", "critical": "#2ca02c", "chaotic": "#d62728", "soc": "#9467bd"}

    # Figure 2 branching trajectories
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for regime, runs in by_regime.items():
        steps, mean_line, std_line = collect_step_stats(runs, "branching_ratio")
        ax.plot(steps, mean_line, "o-", color=colors[regime], label=regime, markersize=3)
        ax.fill_between(steps, mean_line - std_line, mean_line + std_line, color=colors[regime], alpha=0.15)
    ax.axhline(1.0, color="grey", linestyle="--", label="critical branching=1")
    ax.set_xlabel("training step")
    ax.set_ylabel("branching ratio")
    ax.set_title("Criticality tracking during training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "02_branching_trajectory.svg")
    plt.close(fig)

    # Figure 3 lyapunov trajectories
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for regime, runs in by_regime.items():
        steps, mean_line, std_line = collect_step_stats(runs, "lyapunov")
        ax.plot(steps, mean_line, "o-", color=colors[regime], label=regime, markersize=3)
        ax.fill_between(steps, mean_line - std_line, mean_line + std_line, color=colors[regime], alpha=0.15)
    ax.axhline(0.0, color="grey", linestyle="--", label="critical λ=0")
    ax.set_xlabel("training step")
    ax.set_ylabel("Lyapunov exponent (per layer)")
    ax.set_title("Edge-of-chaos tracking during training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "03_lyapunov_trajectory.svg")
    plt.close(fig)

    # Figure 4 test loss curves
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for regime, runs in by_regime.items():
        steps, mean_line, std_line = collect_step_stats(runs, "loss")
        ax.plot(steps, mean_line, "o-", color=colors[regime], label=regime, markersize=3)
        ax.fill_between(steps, mean_line - std_line, mean_line + std_line, color=colors[regime], alpha=0.15)
    ax.axhline(H_uniform, color="grey", linestyle=":", label=f"uniform baseline={H_uniform:.3f}")
    ax.set_xlabel("training step")
    ax.set_ylabel("test NLL (nats/sample)")
    ax.set_title("Learning efficiency comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "04_loss_curves.svg")
    plt.close(fig)

    # Figure 5 accuracy curves
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for regime, runs in by_regime.items():
        steps, mean_line, std_line = collect_step_stats(runs, "acc")
        mean_p, std_p = 100 * mean_line, 100 * std_line
        ax.plot(steps, mean_p, "o-", color=colors[regime], label=regime, markersize=3)
        ax.fill_between(steps, mean_p - std_p, mean_p + std_p, color=colors[regime], alpha=0.15)
    ax.axhline(100 * TARGET_ACC, color="grey", linestyle="--", label=f"target={100*TARGET_ACC:.0f}%")
    ax.set_xlabel("training step")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy trajectories")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "05_accuracy_curves.svg")
    plt.close(fig)

    # Figure 6 final metrics bar plot
    regimes = list(REGIMES.keys())
    x = np.arange(len(regimes))
    acc_mean = [summary[r]["final_acc_mean"] * 100 for r in regimes]
    acc_std = [summary[r]["final_acc_std"] * 100 for r in regimes]
    ttt_mean = [summary[r]["time_to_target_mean"] for r in regimes]
    ttt_std = [summary[r]["time_to_target_std"] for r in regimes]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, acc_mean, yerr=acc_std, capsize=4, color=[colors[r] for r in regimes])
    axes[0].set_xticks(x, regimes)
    axes[0].set_ylabel("final test accuracy (%)")
    axes[0].set_title("Final accuracy (mean±std)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, ttt_mean, yerr=ttt_std, capsize=4, color=[colors[r] for r in regimes])
    axes[1].set_xticks(x, regimes)
    axes[1].set_ylabel("step to target accuracy")
    axes[1].set_title(f"Data-efficiency (target={TARGET_ACC*100:.0f}%)")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "06_final_comparison.svg")
    plt.close(fig)

    # Figure 7 SOC gain adaptation over layers
    soc_runs = by_regime["soc"]
    steps = soc_runs[0]["history"]["step"]
    gains = np.array([r["history"]["layer_gains"] for r in soc_runs], dtype=float)  # [seed, t, layer]
    mean_gains = gains.mean(axis=0)  # [t, layer]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    n_layer = mean_gains.shape[1]
    show_layers = [0, 2, 4, 6, 8, 10, 11]
    for li in show_layers:
        ax.plot(steps, mean_gains[:, li], label=f"layer {li+1}")
    ax.axhline(1.0, color="grey", linestyle="--", lw=1)
    ax.set_xlabel("training step")
    ax.set_ylabel("adaptive gain g_l")
    ax.set_title("SOC local-rule gain adaptation")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "07_soc_gain_dynamics.svg")
    plt.close(fig)

    # Figure 8 performance-criticality phase plane
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for regime, runs in by_regime.items():
        br = np.array([r["history"]["branching_ratio"][-1] for r in runs], dtype=float)
        acc = np.array([r["history"]["acc"][-1] for r in runs], dtype=float) * 100
        ax.scatter(br, acc, s=65, color=colors[regime], label=regime)
        ax.scatter(br.mean(), acc.mean(), s=180, marker="x", color=colors[regime])
    ax.axvline(1.0, color="grey", linestyle="--", lw=1)
    ax.set_xlabel("final branching ratio")
    ax.set_ylabel("final accuracy (%)")
    ax.set_title("Performance vs criticality (across seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "08_phase_plane.svg")
    plt.close(fig)

    # Figure 9 avalanche distributions and power-law fits
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharey=True)
    for ax, regime in zip(axes.flat, REGIMES.keys()):
        sample = by_regime[regime][0]
        sizes = np.array(sample["final_avalanche_sizes"], dtype=float)
        s_max = max(int(sizes.max()), 2)
        bins = np.unique(np.round(np.logspace(0, np.log10(s_max), 18)).astype(int))
        counts, edges = np.histogram(sizes, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        mask = counts > 0
        ax.loglog(centers[mask], counts[mask] / widths[mask], "o", color=colors[regime])
        tau, r2 = sample["final_tau"], sample["final_powerlaw_r2"]
        if mask.sum() >= 2 and np.isfinite(tau):
            xs = np.array([centers[mask].min(), centers[mask].max()])
            ys = (counts[mask] / widths[mask])[0] * (xs / xs[0]) ** (-tau)
            ax.loglog(xs, ys, "--", color="#111111", lw=1.2)
        ax.set_title(f"{regime}: τ={tau:.2f}, R²={r2:.2f}")
        ax.set_xlabel("avalanche size s")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel("density P(s)")
    axes[1, 0].set_ylabel("density P(s)")
    fig.suptitle("Avalanche power-law evidence across regimes", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "09_avalanche_powerlaw.svg")
    plt.close(fig)

    out = {
        "config": {
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "eval_every": EVAL_EVERY,
            "eval_batches": EVAL_BATCHES,
            "lr": LR,
            "seeds": SEEDS,
            "target_acc": TARGET_ACC,
            "regimes": REGIMES,
            "n_params": n_params,
            "H_uniform": H_uniform,
        },
        "summary": summary,
        "runs": all_runs,
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=2))

    best = sorted(summary.items(), key=lambda kv: kv[1]["final_acc_mean"], reverse=True)[0]
    print(f"\n[done] {time.time()-t0:.1f}s | best by final acc: {best[0]} -> {best[1]['final_acc_mean']*100:.2f}%")


if __name__ == "__main__":
    main()
