"""Tiny SVG plotting helpers (stdlib-only)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable


def _lin_scale(v: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    return (v - vmin) / (vmax - vmin)


def _log_scale(v: float, vmin: float, vmax: float) -> float:
    lv = math.log10(max(v, 1e-12))
    lmin = math.log10(max(vmin, 1e-12))
    lmax = math.log10(max(vmax, 1e-12))
    if lmax <= lmin:
        return 0.0
    return (lv - lmin) / (lmax - lmin)


def write_svg_line_plot(
    out: Path,
    series: list[dict],
    title: str,
    xlabel: str,
    ylabel: str,
    logx: bool = False,
    logy: bool = False,
    vline: float | None = None,
) -> None:
    width, height = 900, 560
    ml, mr, mt, mb = 90, 30, 70, 90
    pw, ph = width - ml - mr, height - mt - mb

    xs = [x for s in series for x in s["x"]]
    ys = [y for s in series for y in s["y"] if y > 0 or not logy]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    def to_px(x: float, y: float) -> tuple[float, float]:
        sx = _log_scale(x, xmin, xmax) if logx else _lin_scale(x, xmin, xmax)
        sy = _log_scale(y, ymin, ymax) if logy else _lin_scale(y, ymin, ymax)
        return ml + sx * pw, mt + (1 - sy) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Arial,sans-serif} .grid{stroke:#ddd;stroke-width:1} .axis{stroke:#222;stroke-width:2}</style>',
        f'<text x="{width/2}" y="35" font-size="24" text-anchor="middle">{title}</text>',
        f'<line class="axis" x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}"/>',
        f'<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}"/>',
        f'<text x="{width/2}" y="{height-25}" font-size="18" text-anchor="middle">{xlabel}</text>',
        f'<text x="25" y="{height/2}" font-size="18" text-anchor="middle" transform="rotate(-90 25,{height/2})">{ylabel}</text>',
    ]

    for i in range(6):
        gx = ml + i * pw / 5
        gy = mt + i * ph / 5
        parts.append(f'<line class="grid" x1="{gx}" y1="{mt}" x2="{gx}" y2="{mt+ph}"/>')
        parts.append(f'<line class="grid" x1="{ml}" y1="{gy}" x2="{ml+pw}" y2="{gy}"/>')

    if vline is not None and xmin <= vline <= xmax:
        vx, _ = to_px(vline, ymin)
        parts.append(f'<line x1="{vx}" y1="{mt}" x2="{vx}" y2="{mt+ph}" stroke="#b00" stroke-dasharray="6 4" stroke-width="2"/>')

    legend_y = mt + 10
    for s in series:
        pts = []
        for x, y in zip(s["x"], s["y"]):
            if logy and y <= 0:
                continue
            px, py = to_px(x, y)
            pts.append(f"{px:.2f},{py:.2f}")
        if len(pts) >= 2:
            parts.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="2" points="{" ".join(pts)}"/>')
        parts.append(f'<rect x="{ml+10}" y="{legend_y-12}" width="18" height="10" fill="{s["color"]}"/>')
        parts.append(f'<text x="{ml+35}" y="{legend_y-3}" font-size="14">{s["label"]}</text>')
        legend_y += 20

    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def log_hist(data: Iterable[int], bins: int = 36) -> tuple[list[float], list[float]]:
    values = [d for d in data if d > 0]
    dmin, dmax = min(values), max(values)
    lmin, lmax = math.log10(dmin), math.log10(dmax)

    edges = [10 ** (lmin + i * (lmax - lmin) / bins) for i in range(bins + 1)]
    counts = [0] * bins
    for d in values:
        ld = math.log10(d)
        idx = int((ld - lmin) / (lmax - lmin + 1e-12) * bins)
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
