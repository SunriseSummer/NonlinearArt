"""Plotting helpers for the Case 2 earthquake-cascade challenge.

Compared with Case 1 we add two extras the player will need:

* :func:`ccdf_log` — complementary CDF on a logarithmic axis. CCDFs are far
  more stable than PDFs for fitting Gutenberg–Richter / power-law tails on
  finite samples, so the Phase-1 verification figures use them.
* :func:`plot_heatmap` — a quick stress-field snapshot with a fixed colour
  bar, used to visualise spatial organisation under SOC.

Everything else mirrors the Case 1 helpers so that players who already
worked through Case 1 do not have to relearn an API.
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
    hline: float | None = None,
    figsize: tuple[float, float] = (9.0, 5.6),
) -> None:
    """Plot one or more line series to ``out``.

    Each entry of ``series`` is a dict with keys ``x``, ``y``, ``label`` and
    optionally ``color`` / ``marker`` / ``linestyle`` / ``linewidth``.
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
            alpha=s.get("alpha", 1.0),
        )

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    if vline is not None:
        ax.axvline(vline, color="#b00000", linestyle="--", linewidth=1.4,
                   label=f"reference x={vline:g}")
    if hline is not None:
        ax.axhline(hline, color="#444444", linestyle=":", linewidth=1.2,
                   label=f"reference y={hline:g}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    if any(s.get("label") for s in series) or vline is not None or hline is not None:
        ax.legend(loc="best", frameon=True, fontsize=9)

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
    """Plot two series sharing an x-axis but with independent y-axes."""
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
        ax_left.axvline(vline, color="#b00000", linestyle="--", linewidth=1.4)

    ax_left.set_title(title)
    ax_left.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax_left.legend(handles=[line_l, line_r], loc="best", frameon=True, fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_heatmap(
    out: Path,
    field: Sequence[Sequence[float]],
    title: str,
    *,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (6.4, 5.4),
) -> None:
    """Render a 2-D scalar field as a heatmap with a colour bar.

    Used to visualise the spatial organisation of stress / activity. Values
    in ``field`` are taken as-is; the colour limits default to
    ``[min, max]`` of the data unless overridden.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        field,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Column j")
    ax.set_ylabel("Row i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --- Statistics helpers -----------------------------------------------------


def log_hist(data: Iterable[int], bins: int = 36) -> tuple[list[float], list[float]]:
    """Logarithmically binned probability density (same convention as Case 1)."""
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


def ccdf_log(data: Iterable[int]) -> tuple[list[float], list[float]]:
    """Empirical complementary CDF :math:`P(S \\geq s)` on positive integers.

    Returns two parallel lists ``(s, p)``. CCDFs are well-suited to power-law
    diagnosis because (a) they avoid the bin-width nuisance of histograms and
    (b) for ``P(s) \\propto s^{-\\tau}`` the tail satisfies
    ``P(S \\geq s) \\propto s^{-(\\tau - 1)}``, which still shows up as a
    straight line on a log–log plot. We return only ``s > 0`` and only
    distinct values so that subsequent log–log plotting does not break.
    """
    values = sorted(d for d in data if d > 0)
    if not values:
        return [], []
    n = len(values)
    out_s: list[float] = []
    out_p: list[float] = []
    last = None
    for i, v in enumerate(values):
        if v == last:
            continue
        out_s.append(float(v))
        out_p.append((n - i) / n)
        last = v
    return out_s, out_p


def power_law_mle(data: Iterable[int], smin: int = 1) -> tuple[float, int]:
    """Discrete Hill estimator for the exponent of ``P(s) ∝ s^{-tau}``.

    Returns ``(tau_hat, n_used)`` where ``n_used`` is the sample count above
    ``smin``. We use the continuous-tail approximation
    ``tau ≈ 1 + n / sum_i ln(s_i / (smin - 0.5))``,
    which is the standard estimator for binned-by-1 integer power-law data
    (Clauset, Shalizi, Newman 2009, Eq. B17). Returns ``(float('nan'), 0)``
    if there are not enough samples.
    """
    xs = [d for d in data if d >= smin]
    n = len(xs)
    if n < 5:
        return float("nan"), n
    base = max(smin - 0.5, 0.5)
    s_log = sum(math.log(x / base) for x in xs)
    if s_log <= 0:
        return float("nan"), n
    return 1.0 + n / s_log, n


def aftershock_rate(
    event_steps: Iterable[int],
    event_sizes: Iterable[int],
    *,
    mainshock_quantile: float = 0.95,
    window: int = 400,
    bins: int = 24,
) -> tuple[list[float], list[float], int]:
    """Stack post-mainshock activity into a log-binned aftershock-rate curve.

    A *mainshock* is defined as any event whose size exceeds the
    ``mainshock_quantile`` quantile of the supplied size distribution. For
    each mainshock we count subsequent events at relative time
    ``Δt = 1 .. window`` (in macro-steps) and average across mainshocks.
    The result is a log-binned rate ``n(Δt)`` which, in the OFC universality
    class, should approximately follow Omori's law ``n(Δt) ∝ Δt^{-p}`` with
    ``p ≈ 1`` near criticality.

    Returns ``(centers, rates, n_mainshocks)``. ``centers`` are bin centres in
    macro-steps, ``rates`` is the average number of aftershocks per
    mainshock per unit Δt within the bin.
    """
    steps = list(event_steps)
    sizes = list(event_sizes)
    if len(steps) < 20:
        return [], [], 0
    sorted_sizes = sorted(sizes)
    cutoff_index = int(mainshock_quantile * len(sorted_sizes))
    cutoff_index = min(max(cutoff_index, 0), len(sorted_sizes) - 1)
    cutoff = sorted_sizes[cutoff_index]
    main_idx = [i for i, s in enumerate(sizes) if s >= cutoff]
    if not main_idx:
        return [], [], 0

    # Log-spaced bins from 1 to ``window``.
    lmin, lmax = math.log10(1.0), math.log10(window)
    edges = [10 ** (lmin + k * (lmax - lmin) / bins) for k in range(bins + 1)]
    counts = [0] * bins

    n_mainshocks = 0
    for mi in main_idx:
        t0 = steps[mi]
        n_mainshocks += 1
        for j in range(mi + 1, len(steps)):
            dt = steps[j] - t0
            if dt <= 0:
                continue
            if dt > window:
                break
            ldt = math.log10(dt)
            k = int((ldt - lmin) / (lmax - lmin) * bins)
            if 0 <= k < bins:
                counts[k] += 1

    centers: list[float] = []
    rates: list[float] = []
    for k, c in enumerate(counts):
        if c == 0:
            continue
        width = edges[k + 1] - edges[k]
        center = math.sqrt(edges[k] * edges[k + 1])
        centers.append(center)
        rates.append(c / n_mainshocks / width)
    return centers, rates, n_mainshocks


__all__ = [
    "plot_lines",
    "plot_dual_axis",
    "plot_heatmap",
    "log_hist",
    "ccdf_log",
    "power_law_mle",
    "aftershock_rate",
]
