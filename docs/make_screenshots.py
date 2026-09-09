"""Regenerates the project's progress screenshots under docs/screenshots/.

These are dev-facing "here's what works so far" snapshots for the README's
progress log -- not the formal Module 6 deliverable (scripts/plot_results.py),
which will plot actual EKF estimation results once the filter exists. Run
this after any module lands that's worth a picture; add one function per
milestone rather than editing old ones, so the gallery is a history.

Usage: .venv/bin/python docs/make_screenshots.py
"""

from __future__ import annotations

import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from orbit_od.constants import J2, MU_EARTH, R_EARTH
from orbit_od.dynamics import propagate
from orbit_od.plotting import CATEGORICAL, SEQUENTIAL_BLUE, new_figure
from orbit_od.truth import propagate_tle

OUT_DIR = pathlib.Path(__file__).parent / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_ISS_TLE_LINE1 = "1 25544U 98067A   26251.46617104  .00001902  00000+0  42648-4 0  9994"
_ISS_TLE_LINE2 = "2 25544  51.6295 247.9233 0004929 112.8116 247.3394 15.49040812584669"


def module1_j2_raan_drift() -> None:
    """RAAN vs time, J2 on vs off -- visualizes what test_dynamics.py checks."""
    a = R_EARTH + 500e3
    inc = np.radians(51.6)
    v = np.sqrt(MU_EARTH / a)
    state0 = np.array([a, 0.0, 0.0, 0.0, v * np.cos(inc), v * np.sin(inc)])

    n = np.sqrt(MU_EARTH / a**3)
    period = 2.0 * np.pi / n
    n_orbits = 15
    t_eval = np.array([k * period for k in range(n_orbits + 1)])

    def raan_from_state(state):
        r, vv = state[:3], state[3:]
        h = np.cross(r, vv)
        node = np.cross([0.0, 0.0, 1.0], h)
        return np.arctan2(node[1], node[0]) % (2.0 * np.pi)

    fig, ax = new_figure(figsize=(7, 4.5))

    for j2_val, label, color in [(0.0, "J2 = 0 (pure two-body)", CATEGORICAL[0]),
                                  (J2, "J2 enabled", CATEGORICAL[1])]:
        result = propagate(state0, (0.0, n_orbits * period), t_eval=t_eval, j2=j2_val)
        raan = np.unwrap([raan_from_state(result.y[:, k]) for k in range(len(t_eval))])
        raan_deg_drift = np.degrees(raan - raan[0])
        ax.plot(t_eval / 3600.0, raan_deg_drift, color=color, linewidth=2, label=label,
                marker="o", markersize=4)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("RAAN drift (deg)")
    ax.set_title("J2 secular nodal regression: 500 km, 51.6° orbit")
    ax.legend(frameon=False, labelcolor="#52514e", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_dynamics_j2_raan_drift.png", dpi=150)


def module2_iss_truth_trajectory() -> None:
    """3D ISS truth trajectory over a 3-hour window, from a real TLE."""
    t_s, states = propagate_tle(
        _ISS_TLE_LINE1, _ISS_TLE_LINE2, datetime(2026, 9, 8, 12, 0, 0), 3 * 3600.0, 30.0
    )
    r = states[:, :3] / 1000.0  # km, for plot readability only

    fig = plt.figure(figsize=(7, 6.5), facecolor="#fcfcfb")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#fcfcfb")
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("#fcfcfb")
        pane.pane.set_edgecolor("#e1e0d9")
        pane._axinfo["grid"]["color"] = (0.88, 0.88, 0.85, 0.6)

    # Earth reference sphere.
    u, w = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 30j]
    re_km = R_EARTH / 1000.0
    xs = re_km * np.cos(u) * np.sin(w)
    ys = re_km * np.sin(u) * np.sin(w)
    zs = re_km * np.cos(w)
    ax.plot_surface(xs, ys, zs, color="#9ec5f4", alpha=0.35, linewidth=0, shade=True,
                     antialiased=True)

    # Trajectory, colored by elapsed time (sequential blue).
    n_seg = len(t_s) - 1
    n_colors = len(SEQUENTIAL_BLUE)
    for i in range(n_seg):
        color = SEQUENTIAL_BLUE[min(int(i / n_seg * n_colors), n_colors - 1)]
        ax.plot(r[i : i + 2, 0], r[i : i + 2, 1], r[i : i + 2, 2], color=color, linewidth=2)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("ISS truth trajectory, 3 h (SGP4, real TLE, TEME≈ECI)")
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_truth_iss_trajectory_3d.png", dpi=150)
    plt.close(fig)


def module2_iss_altitude_and_speed() -> None:
    """Altitude and speed vs time -- two stacked subplots, never dual-axis."""
    t_s, states = propagate_tle(
        _ISS_TLE_LINE1, _ISS_TLE_LINE2, datetime(2026, 9, 8, 12, 0, 0), 3 * 3600.0, 10.0
    )
    altitude_km = (np.linalg.norm(states[:, :3], axis=1) - R_EARTH) / 1000.0
    speed_km_s = np.linalg.norm(states[:, 3:], axis=1) / 1000.0
    t_hr = t_s / 3600.0

    fig, axes = new_figure(figsize=(7, 5.5), nrows=2, ncols=1, sharex=True)

    axes[0].plot(t_hr, altitude_km, color=CATEGORICAL[0], linewidth=2)
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("ISS truth trajectory: altitude and speed (SGP4, real TLE)")

    axes[1].plot(t_hr, speed_km_s, color=CATEGORICAL[1], linewidth=2)
    axes[1].set_ylabel("Speed (km/s)")
    axes[1].set_xlabel("Time (hours)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_truth_altitude_speed.png", dpi=150)


if __name__ == "__main__":
    module1_j2_raan_drift()
    module2_iss_truth_trajectory()
    module2_iss_altitude_and_speed()
    print(f"Wrote screenshots to {OUT_DIR}")
