"""Plotting helpers for the Case 1 traffic-cascade challenge.

All figures are produced with matplotlib so players can extend them freely
(add legends, annotations, additional axes, ...). The helpers below cover the
two recurring needs in this case:

* line / scatter plots with optional log scaling and a vertical reference line;
* logarithmically binned histograms used to inspect avalanche-size statistics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-friendly: no display required
import matplotlib.pyplot as plt


# --- Figure helpers ---------------------------------------------------------


def plot_lines(
    out: Path,
    series: Sequence[dict],
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    logx: bool = False,
    logy: bool = False,
    vline: float | None = None,
    figsize: tuple[float, float] = (9.0, 5.6),
) -> None:
    """Plot one or more line series to ``out``.

    Each entry of ``series`` is a dict with keys ``x``, ``y``, ``label`` and
    optionally ``color`` / ``marker`` / ``linestyle``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for s in series:
        ax.plot(
            s["x"],
            s["y"],
            label=s.get("label", ""),
            color=s.get("color"),
            marker=s.get("marker", ""),
            linestyle=s.get("linestyle", "-"),
            linewidth=s.get("linewidth", 1.8),
        )

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    if vline is not None:
        ax.axvline(vline, color="#b00000", linestyle="--", linewidth=1.5,
                   label=f"reference x={vline:g}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    if any(s.get("label") for s in series) or vline is not None:
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
    vline: float | None = None,
    figsize: tuple[float, float] = (9.0, 5.6),
) -> None:
    """Plot two series sharing an x-axis but with independent y-axes.

    ``left`` / ``right`` are dicts with keys ``y``, ``label``, ``ylabel`` and
    optionally ``color``.
    """
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    color_l = left.get("color", "#1f77b4")
    color_r = right.get("color", "#9467bd")

    line_l, = ax_left.plot(x, left["y"], color=color_l,
                           label=left.get("label", "left"), linewidth=1.6)
    line_r, = ax_right.plot(x, right["y"], color=color_r,
                            label=right.get("label", "right"), linewidth=1.6)

    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel(left.get("ylabel", left.get("label", "")), color=color_l)
    ax_right.set_ylabel(right.get("ylabel", right.get("label", "")), color=color_r)
    ax_left.tick_params(axis="y", colors=color_l)
    ax_right.tick_params(axis="y", colors=color_r)

    if vline is not None:
        ax_left.axvline(vline, color="#b00000", linestyle="--", linewidth=1.5)

    ax_left.set_title(title)
    ax_left.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax_left.legend(handles=[line_l, line_r], loc="best", frameon=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --- Statistics helpers -----------------------------------------------------


def log_hist(data: Iterable[int], bins: int = 36) -> tuple[list[float], list[float]]:
    """Return (bin_centers, density) using logarithmic bin edges.

    Returns two empty lists — so callers can plot safely — whenever the
    histogram cannot be defined: fewer than two positive samples, or all
    positive samples sharing the same value (zero-width log range).
    """
    values = [d for d in data if d > 0]
    if len(values) < 2 or min(values) == max(values):
        return [], []

    dmin, dmax = min(values), max(values)
    lmin, lmax = math.log10(dmin), math.log10(dmax)

    edges = [10 ** (lmin + i * (lmax - lmin) / bins) for i in range(bins + 1)]
    counts = [0] * bins
    span = lmax - lmin
    for d in values:
        ld = math.log10(d)
        idx = int((ld - lmin) / span * bins) if span > 0 else 0
        idx = min(max(idx, 0), bins - 1)
        counts[idx] += 1

    total = sum(counts)
    centers, probs = [], []
    for i, c in enumerate(counts):
        if c == 0:
            continue
        center = math.sqrt(edges[i] * edges[i + 1])
        width = edges[i + 1] - edges[i]
        centers.append(center)
        probs.append(c / total / width)
    return centers, probs
