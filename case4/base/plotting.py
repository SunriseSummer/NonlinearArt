"""Plotting and small statistics helpers for the Case 4 Ising challenge."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 2:
        return [float(v) for v in values]
    out: list[float] = []
    buf: list[float] = []
    acc = 0.0
    for v in values:
        fv = float(v)
        buf.append(fv)
        acc += fv
        if len(buf) > window:
            acc -= buf.pop(0)
        out.append(acc / len(buf))
    return out


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


def susceptibility(m_values: Sequence[float], temperature: float, n_spins: int) -> float:
    return n_spins * variance(m_values) / max(temperature, 1e-9)


def heat_capacity(e_values: Sequence[float], temperature: float, n_spins: int) -> float:
    return n_spins * variance(e_values) / max(temperature * temperature, 1e-9)


def binder_cumulant(m_values: Sequence[float]) -> float:
    """Fourth-order Binder cumulant for the Ising magnetisation.

    Curves for different system sizes cross near the critical temperature,
    making this a stricter finite-size diagnostic than a single susceptibility
    peak.
    """
    if not m_values:
        return 0.0
    m2 = mean([m * m for m in m_values])
    if m2 <= 0.0:
        return 0.0
    m4 = mean([m ** 4 for m in m_values])
    return 1.0 - m4 / (3.0 * m2 * m2)


def log_hist(data: Iterable[int], bins: int = 28) -> tuple[list[float], list[float]]:
    values = [int(v) for v in data if v > 0]
    if len(values) < 2 or min(values) == max(values):
        return [], []
    dmin, dmax = min(values), max(values)
    lmin, lmax = math.log10(dmin), math.log10(dmax)
    edges = [10 ** (lmin + i * (lmax - lmin) / bins) for i in range(bins + 1)]
    counts = [0] * bins
    span = lmax - lmin
    for d in values:
        idx = int((math.log10(d) - lmin) / span * bins) if span > 0 else 0
        counts[min(max(idx, 0), bins - 1)] += 1
    total = sum(counts)
    xs, ys = [], []
    for i, c in enumerate(counts):
        if c == 0:
            continue
        width = edges[i + 1] - edges[i]
        xs.append(math.sqrt(edges[i] * edges[i + 1]))
        ys.append(c / total / width)
    return xs, ys


def plot_lines(
    out: Path,
    series: Sequence[dict],
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    logx: bool = False,
    logy: bool = False,
    hline: float | None = None,
    vlines: Sequence[tuple[float, str, str]] | None = None,
    figsize: tuple[float, float] = (9.0, 5.4),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    for s in series:
        ax.plot(
            s["x"],
            s["y"],
            label=s.get("label", ""),
            color=s.get("color"),
            marker=s.get("marker", ""),
            linestyle=s.get("linestyle", "-"),
            linewidth=s.get("linewidth", 1.7),
            alpha=s.get("alpha", 1.0),
        )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if hline is not None:
        ax.axhline(hline, color="#555555", linestyle="--", linewidth=1.2, label=f"y={hline:g}")
    if vlines:
        for x, color, label in vlines:
            ax.axvline(x, color=color, linestyle="--", linewidth=1.3, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    if any(s.get("label") for s in series) or hline is not None or vlines:
        ax.legend(loc="best", frameon=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_dual_axis(
    out: Path,
    x: Sequence[float],
    left: dict,
    right: dict,
    title: str,
    xlabel: str,
    *,
    vlines: Sequence[tuple[float, str, str]] | None = None,
    figsize: tuple[float, float] = (9.0, 5.4),
) -> None:
    fig, ax_l = plt.subplots(figsize=figsize)
    ax_r = ax_l.twinx()
    color_l = left.get("color", "#1f77b4")
    color_r = right.get("color", "#ff7f0e")
    line_l, = ax_l.plot(x, left["y"], color=color_l, label=left.get("label", "left"), linewidth=1.6)
    line_r, = ax_r.plot(x, right["y"], color=color_r, label=right.get("label", "right"), linewidth=1.6)
    ax_l.set_xlabel(xlabel)
    ax_l.set_ylabel(left.get("ylabel", left.get("label", "")), color=color_l)
    ax_r.set_ylabel(right.get("ylabel", right.get("label", "")), color=color_r)
    ax_l.tick_params(axis="y", colors=color_l)
    ax_r.tick_params(axis="y", colors=color_r)
    handles = [line_l, line_r]
    if vlines:
        for x_, color, label in vlines:
            handles.append(ax_l.axvline(x_, color=color, linestyle="--", linewidth=1.3, label=label))
    ax_l.set_title(title)
    ax_l.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax_l.legend(handles=handles, loc="best", frameon=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_bars(
    out: Path,
    categories: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    *,
    colors: Sequence[str] | None = None,
    figsize: tuple[float, float] = (8.0, 4.8),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    xs = list(range(len(categories)))
    ax.bar(xs, list(values), color=list(colors) if colors else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(categories))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    for x_, v in zip(xs, values):
        ax.text(x_, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_spin_field(out: Path, spins: Sequence[Sequence[int]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.imshow(spins, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
