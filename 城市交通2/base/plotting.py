"""Plotting helpers for the case1b rush-hour traffic challenge.

These helpers mirror the spirit of ``case1/base/plotting.py`` but add a few
small extras that the new metrics need (rolling means for noisy time series,
log-binned histograms with the same API, and a dual-axis plot).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- Statistics helpers -----------------------------------------------------


def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    """Causal rolling mean used to smooth throughput / congestion-range
    time series before plotting.

    The first ``window-1`` outputs reuse a shrinking prefix average so the
    series has the same length as the input — convenient for plotting on a
    shared x-axis.
    """
    if window <= 1 or len(values) < 2:
        return [float(v) for v in values]
    out: list[float] = []
    acc = 0.0
    buf: list[float] = []
    for v in values:
        buf.append(float(v))
        acc += float(v)
        if len(buf) > window:
            acc -= buf.pop(0)
        out.append(acc / len(buf))
    return out


def log_hist(data: Iterable[int], bins: int = 32) -> tuple[list[float], list[float]]:
    """Logarithmically binned probability density of positive integers."""
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


def power_law_fit(
    centers: Sequence[float],
    probs: Sequence[float],
    *,
    x_min: float | None = None,
    x_max: float | None = None,
) -> tuple[float, float, float]:
    """Fit ``log10(P) = -tau * log10(s) + b`` over the chosen range.

    Returns ``(tau, b, r2)`` where ``tau`` is the power-law exponent
    (positive number, so ``P(s) ~ s^(-tau)``), ``b`` is the intercept on
    log10 scale, and ``r2`` is the coefficient of determination of the
    log-log linear fit. Returns ``(0, 0, 0)`` if there are fewer than two
    valid points in the requested range — callers should treat that as
    "no fit available" and avoid drawing a line.

    The fit deliberately runs in log space (least squares on
    ``log10(centers)`` vs ``log10(probs)``) which is the standard textbook
    approach for assessing whether an empirical distribution is consistent
    with a power law over a chosen range.
    """
    xs, ys = [], []
    for x, p in zip(centers, probs):
        if x <= 0 or p <= 0:
            continue
        if x_min is not None and x < x_min:
            continue
        if x_max is not None and x > x_max:
            continue
        xs.append(math.log10(x))
        ys.append(math.log10(p))

    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0

    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0:
        return 0.0, 0.0, 0.0

    slope = sxy / sxx
    intercept = my - slope * mx
    syy = sum((y - my) ** 2 for y in ys)
    if syy <= 0:
        r2 = 1.0
    else:
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r2 = max(0.0, 1.0 - ss_res / syy)
    return -slope, intercept, r2


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
    vlines: Sequence[tuple[float, str, str]] | None = None,
    figsize: tuple[float, float] = (9.0, 5.4),
) -> None:
    """Plot one or more line series.

    ``vlines`` is a sequence of ``(x, color, label)`` triples — handy for
    marking warmup boundaries or disturbance events.
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
            linewidth=s.get("linewidth", 1.7),
            alpha=s.get("alpha", 1.0),
        )

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    if vlines:
        for x, color, label in vlines:
            ax.axvline(x, color=color, linestyle="--", linewidth=1.3, label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    if any(s.get("label") for s in series) or vlines:
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
    """Twin-y plot used to overlay quantities on very different scales
    (e.g. mean load vs throughput, or load vs adaptive spill_prob)."""
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

    handles = [line_l, line_r]
    if vlines:
        for x_, color, label in vlines:
            ln = ax_left.axvline(x_, color=color, linestyle="--", linewidth=1.3,
                                 label=label)
            handles.append(ln)

    ax_left.set_title(title)
    ax_left.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax_left.legend(handles=handles, loc="best", frameon=True)

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
    """Simple grouped-bar chart used for the regime-comparison summaries."""
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
