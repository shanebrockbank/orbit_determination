"""Shared matplotlib styling for orbit_od plots.

Not a physics module -- centralizes chart chrome (colors, gridlines, ink)
so every plot in this project (progress screenshots, and later the
Module 6 visualization suite) looks like one consistent system rather
than ad hoc per-script styling.

Colors are the validated light-mode palette from the project's data-viz
skill reference: first three categorical slots (blue/orange/aqua), which
clear the all-pairs colorblind-safety check, plus chart chrome (gridlines,
axis, ink).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua

CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]


def style_axes(ax) -> None:
    """Apply consistent chrome to a 2D matplotlib Axes: recessive gridlines,
    muted spines/ticks, chart-surface background."""
    ax.set_facecolor(CHART_SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED_INK, labelsize=9)
    ax.xaxis.label.set_color(SECONDARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)
    ax.title.set_color(PRIMARY_INK)


def new_figure(figsize=(8, 5), nrows=1, ncols=1, **kwargs):
    """Figure/axes pair (or grid) pre-styled with the project's chart chrome."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=CHART_SURFACE, **kwargs)
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        style_axes(ax)
    return fig, axes
