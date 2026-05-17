"""Plotting and small statistics helpers for the Case 5 forest-fire challenge.

Only depends on ``matplotlib``.  The helpers mirror the spirit of
``case4/base/plotting.py`` but are tuned to forest-fire visualisations
(three-state grid snapshots, log-binned fire-size distributions, etc.).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---- small statistics ------------------------------------------------------

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


def log_hist(data: Iterable[int], bins: int = 28) -> tuple[list[float], list[float]]:
    """Logarithmically binned PDF estimator.  Returns ``(centres, density)``."""
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


def percentile(values: Sequence[float], q: float) -> float:
    """Return the ``q`` percentile (0..100) using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


# ---- generic line / bar plots ---------------------------------------------

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


# ---- forest-specific snapshots --------------------------------------------

# 0=empty (sand), 1=tree (green), 2=fire (red)
_FOREST_CMAP = ListedColormap(["#e8d6a8", "#2e8b57", "#d62728"])


def plot_forest(out: Path, grid: Sequence[Sequence[int]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.imshow(grid, cmap=_FOREST_CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_loglog_distribution(
    out: Path,
    datasets: Sequence[dict],
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    reference: tuple[float, float, str] | None = None,
    figsize: tuple[float, float] = (8.4, 5.4),
) -> None:
    """Log-binned PDF on log-log axes.

    Each dataset is ``{"data": [...], "label": "...", "color": "..."}``.
    Optional ``reference`` is ``(slope, intercept_log10, label)`` to draw a
    power-law guide ``y = 10**intercept * x**slope``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for ds in datasets:
        xs, ys = log_hist(ds["data"], bins=ds.get("bins", 28))
        if not xs:
            continue
        ax.plot(
            xs, ys,
            marker=ds.get("marker", "o"),
            linestyle=ds.get("linestyle", "-"),
            color=ds.get("color"),
            label=ds.get("label", ""),
            linewidth=1.5,
            markersize=4.5,
        )
    if reference is not None:
        slope, intercept, label = reference
        # build a reference line spanning two decades
        xref = [10.0, 1000.0]
        yref = [10.0 ** intercept * (x ** slope) for x in xref]
        ax.plot(xref, yref, color="#555555", linestyle="--", linewidth=1.2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(loc="best", frameon=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


__all__ = [
    "rolling_mean",
    "mean",
    "variance",
    "percentile",
    "log_hist",
    "plot_lines",
    "plot_bars",
    "plot_forest",
    "plot_loglog_distribution",
]
