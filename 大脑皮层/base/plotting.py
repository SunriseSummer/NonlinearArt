"""Plotting helpers for the Case 3 cortical-avalanche challenge.

Builds on the Case 1 / Case 2 toolkit and adds the two extras the player
will need to verify the **crackling-noise scaling relation** for branching
neural avalanches:

* :func:`mean_size_vs_duration` — bin avalanches by their duration ``T``
  and compute the conditional mean size ``<s|T>``. The log–log slope of
  this curve estimates ``1/(sigma*nu*z)``.
* :func:`shape_collapse` — rescale individual avalanche shapes
  ``s(t/T) / T^{1/(sigma*nu*z) - 1}`` so all duration bins fall on a
  universal scaling function. A successful collapse is the strongest
  single-figure evidence that the system sits in the SOC universality
  class.

Everything else mirrors Case 2 helpers so players who have already worked
through Cases 1 / 2 can reuse the same API conventions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
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
    fig, ax = plt.subplots(figsize=figsize)
    for s in series:
        ax.plot(
            s["x"], s["y"],
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


# --- Statistics: distributions and exponent estimators ---------------------


def log_hist(data: Iterable[int], bins: int = 36) -> tuple[list[float], list[float]]:
    """Logarithmically binned probability density (same convention as cases 1/2)."""
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
    """Empirical CCDF P(S>=s) on positive integers."""
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
    """Discrete Hill estimator. Returns (tau_hat, n_used)."""
    xs = [d for d in data if d >= smin]
    n = len(xs)
    if n < 5:
        return float("nan"), n
    base = max(smin - 0.5, 0.5)
    s_log = sum(math.log(x / base) for x in xs)
    if s_log <= 0:
        return float("nan"), n
    return 1.0 + n / s_log, n


# --- Crackling-noise scaling -----------------------------------------------


def mean_size_vs_duration(
    sizes: Iterable[int],
    durations: Iterable[int],
    *,
    min_count: int = 6,
    max_duration: int | None = None,
) -> tuple[list[float], list[float]]:
    """Bin avalanches by ``T`` and return ``(T, <s|T>)``.

    Only durations for which we have at least ``min_count`` samples are
    returned, so the conditional mean is statistically meaningful.
    Cropping at ``max_duration`` is recommended when fitting the slope —
    finite-system cutoffs distort the largest bins.
    """
    buckets: dict[int, list[int]] = {}
    for s, T in zip(sizes, durations):
        if T <= 0:
            continue
        if max_duration is not None and T > max_duration:
            continue
        buckets.setdefault(T, []).append(s)
    Ts: list[float] = []
    means: list[float] = []
    for T in sorted(buckets):
        bucket = buckets[T]
        if len(bucket) >= min_count:
            Ts.append(float(T))
            means.append(sum(bucket) / len(bucket))
    return Ts, means


def loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Linear-regression slope of ``log y`` against ``log x``.

    Returns ``(slope, intercept)`` of the best-fit straight line in log–log
    space, or ``(NaN, NaN)`` if there are fewer than two points or any
    non-positive value. The intercept is in log-space so the line is
    ``log y = slope * log x + intercept``.
    """
    xs_pos = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(xs_pos) < 2:
        return float("nan"), float("nan")
    lx = [math.log(x) for x, _ in xs_pos]
    ly = [math.log(y) for _, y in xs_pos]
    n = len(lx)
    sx = sum(lx)
    sy = sum(ly)
    sxx = sum(x * x for x in lx)
    sxy = sum(a * b for a, b in zip(lx, ly))
    denom = n * sxx - sx * sx
    if denom == 0:
        return float("nan"), float("nan")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


# --- Shape-collapse helpers ------------------------------------------------


def avalanche_profile(
    sizes_per_round: Iterable[int],
    duration: int,
    n_bins: int = 40,
) -> list[float]:
    """Resample a single avalanche profile to a fixed grid of ``n_bins`` points.

    ``sizes_per_round[r]`` is the number of spikes in synchronous round ``r``
    of the avalanche (so its sum equals the avalanche's total size and its
    length equals ``duration``). We linearly resample (with simple bin
    averaging) onto ``n_bins`` time points spanning ``[0, 1]``.
    """
    arr = list(sizes_per_round)
    if duration <= 0 or not arr:
        return [0.0] * n_bins
    out = [0.0] * n_bins
    counts = [0] * n_bins
    for r, val in enumerate(arr):
        # map relative time r/(duration-1) (or r/duration for d=1) to bin
        t_rel = r / max(duration - 1, 1)
        b = int(t_rel * (n_bins - 1) + 0.5)
        if b < 0:
            b = 0
        if b >= n_bins:
            b = n_bins - 1
        out[b] += val
        counts[b] += 1
    # Average within each bin.
    for b in range(n_bins):
        if counts[b] > 0:
            out[b] /= counts[b]
    # Fill empty bins by linear interpolation between neighbouring filled bins.
    last_filled = -1
    for b in range(n_bins):
        if counts[b] > 0:
            if last_filled == -1:
                # extend left
                for q in range(b):
                    out[q] = out[b]
            elif b > last_filled + 1:
                lo = out[last_filled]
                hi = out[b]
                for q in range(last_filled + 1, b):
                    frac = (q - last_filled) / (b - last_filled)
                    out[q] = lo + (hi - lo) * frac
            last_filled = b
    if last_filled >= 0 and last_filled < n_bins - 1:
        for q in range(last_filled + 1, n_bins):
            out[q] = out[last_filled]
    return out


def collapsed_shape(
    profiles_by_duration: dict[int, list[list[float]]],
    sigma_nu_z_inv: float,
    *,
    n_bins: int = 40,
) -> dict[int, tuple[list[float], list[float]]]:
    """Collapse avalanche shapes across duration bins.

    ``profiles_by_duration`` maps each duration ``T`` to a list of profiles
    (each profile is the output of :func:`avalanche_profile` for one
    avalanche of that exact duration). For a self-affine SOC system the
    average shape ``s_T(t)`` should obey
    ``s_T(t) = T^{1/(sigma nu z) - 1} * F(t / T)`` for a *universal*
    scaling function ``F``. The collapse rescales each averaged profile by
    ``T^{1 - sigma_nu_z_inv}`` and returns the curves on the rescaled time
    axis ``t/T in [0, 1]``.
    """
    grid = [b / (n_bins - 1) for b in range(n_bins)]
    out: dict[int, tuple[list[float], list[float]]] = {}
    for T, profiles in profiles_by_duration.items():
        if not profiles:
            continue
        avg = [0.0] * n_bins
        for prof in profiles:
            for b, v in enumerate(prof):
                avg[b] += v
        for b in range(n_bins):
            avg[b] /= len(profiles)
        scale = T ** (1.0 - sigma_nu_z_inv)
        out[T] = (grid[:], [v * scale for v in avg])
    return out


__all__ = [
    "plot_lines",
    "plot_dual_axis",
    "log_hist",
    "ccdf_log",
    "power_law_mle",
    "mean_size_vs_duration",
    "loglog_slope",
    "avalanche_profile",
    "collapsed_shape",
]
