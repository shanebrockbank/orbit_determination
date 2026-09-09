"""Shared matplotlib styling for orbit_od plots.

Not a physics module -- centralizes chart chrome (colors, gridlines, ink),
fixed canvas sizes, and series identity so every plot in this project
(progress screenshots, and later the Module 6 visualization suite) reads
as one designed system rather than N ad hoc scripts.

Colors are the validated light-mode palette from the project's data-viz
skill reference: first three categorical slots (blue/orange/aqua), which
clear the all-pairs colorblind-safety check, plus chart chrome (gridlines,
axis, ink).

Two fixed canvas conventions:

  - `new_trajectory_figure()` for trajectory-style plots (3D orbit path,
    ground track).
  - `new_timeseries_figure(nrows)` for time-series-style plots (altitude/
    speed, RAAN drift, per-axis position/velocity error, ...).

One fixed series identity, used everywhere a plot shows more than one of
these roles, so "truth" (or "measurement", or "estimate") never means a
different color in two different scripts:

  - `TRUTH_STYLE`: solid line, categorical slot 1 (blue).
  - `MEAS_STYLE`: unconnected markers, categorical slot 2 (orange) --
    measurements are discrete samples, never a connected line.
  - `EST_STYLE`: dashed line, categorical slot 3 (aqua).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua

CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]

DPI = 150
TRAJECTORY_FIGSIZE = (7.0, 6.5)
TIMESERIES_WIDTH = 7.5
TIMESERIES_ROW_HEIGHT = 2.5  # inches per stacked subplot row

TRUTH_STYLE = {
    "color": CATEGORICAL[0],
    "linestyle": "-",
    "linewidth": 2,
    "label": "Truth",
    "zorder": 3,
}
MEAS_STYLE = {
    "color": CATEGORICAL[1],
    "linestyle": "none",
    "marker": "o",
    "markersize": 4,
    "alpha": 0.7,
    "label": "GPS measurement",
    "zorder": 2,
}
EST_STYLE = {
    "color": CATEGORICAL[2],
    "linestyle": "--",
    "linewidth": 2,
    "label": "EKF estimate",
    "zorder": 4,
}


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
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, facecolor=CHART_SURFACE, dpi=DPI, **kwargs
    )
    for ax in axes.flat if hasattr(axes, "flat") else [axes]:
        style_axes(ax)
    return fig, axes


def new_timeseries_figure(nrows: int = 1, sharex: bool = True):
    """Fixed canvas convention for time-series-style plots: every module's
    time-series screenshot shares the same width, DPI, and per-row height."""
    figsize = (TIMESERIES_WIDTH, TIMESERIES_ROW_HEIGHT * nrows + 1.2)
    return new_figure(figsize=figsize, nrows=nrows, ncols=1, sharex=sharex)


def new_trajectory_figure():
    """Fixed canvas convention for trajectory-style plots: a pre-styled 3D
    Axes at the project's standard trajectory figure size/DPI."""
    fig = plt.figure(figsize=TRAJECTORY_FIGSIZE, facecolor=CHART_SURFACE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(CHART_SURFACE)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(CHART_SURFACE)
        pane.pane.set_edgecolor(GRIDLINE)
        pane._axinfo["grid"]["color"] = (0.88, 0.88, 0.85, 0.6)
    return fig, ax


def draw_earth_sphere(ax, r_eq_m: float) -> None:
    """Translucent Earth reference sphere for trajectory-style 3D plots.

    Args:
        ax: A 3D Axes, e.g. from `new_trajectory_figure()`.
        r_eq_m: Earth equatorial radius, meters (drawn in km for
            readability against typical LEO trajectory plot scales).
    """
    u, w = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 30j]
    re_km = r_eq_m / 1000.0
    xs = re_km * np.cos(u) * np.sin(w)
    ys = re_km * np.sin(u) * np.sin(w)
    zs = re_km * np.cos(w)
    ax.plot_surface(
        xs, ys, zs, color="#9ec5f4", alpha=0.35, linewidth=0, shade=True, antialiased=True
    )
