"""Plotting and small statistics helpers for the Case 6 Vicsek challenge."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

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


def susceptibility(phi_values: Sequence[float], n_agents: int) -> float:
    """Vicsek-style susceptibility: chi = N * Var(phi)."""
    return n_agents * variance(phi_values)


def binder_cumulant(phi_values: Sequence[float]) -> float:
    """Fourth-order Binder cumulant for a non-negative scalar order parameter.

    For the Vicsek scalar order parameter the standard finite-size diagnostic is

        U_4 = 1 - <phi^4> / (3 <phi^2>^2)

    Curves for different system sizes cross near eta_c.
    """
    if not phi_values:
        return 0.0
    phi2 = mean([v * v for v in phi_values])
    if phi2 <= 0.0:
        return 0.0
    phi4 = mean([v ** 4 for v in phi_values])
    return 1.0 - phi4 / (3.0 * phi2 * phi2)


# ---- generic plots --------------------------------------------------------

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
        ax.text(x_, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_swarm(
    out: Path,
    positions: Sequence[tuple[float, float]],
    thetas: Sequence[float],
    box_size: float,
    title: str,
    *,
    figsize: tuple[float, float] = (5.6, 5.4),
    leader_index: int | None = None,
) -> None:
    """Quiver plot of the swarm: arrow per agent, colour by heading angle."""
    fig, ax = plt.subplots(figsize=figsize)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    us = [math.cos(t) for t in thetas]
    vs = [math.sin(t) for t in thetas]
    # colour encodes heading angle in [-pi, pi]
    colors = [t % (2 * math.pi) for t in thetas]
    q = ax.quiver(
        xs, ys, us, vs, colors,
        cmap="hsv", scale=22, width=0.005, headwidth=3.5, alpha=0.9,
    )
    q.set_clim(0, 2 * math.pi)
    if leader_index is not None and 0 <= leader_index < len(xs):
        ax.scatter([xs[leader_index]], [ys[leader_index]],
                   s=80, facecolor="none", edgecolor="black", linewidths=1.6,
                   zorder=5, label="leader")
        ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


__all__ = [
    "rolling_mean",
    "mean",
    "variance",
    "susceptibility",
    "binder_cumulant",
    "plot_lines",
    "plot_dual_axis",
    "plot_bars",
    "plot_swarm",
]
