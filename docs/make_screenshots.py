"""Regenerates the project's progress screenshots under docs/screenshots/.

These are dev-facing "here's what works so far" snapshots for the README's
progress log -- not the formal Module 6 deliverable (scripts/plot_results.py),
which will plot actual EKF estimation results once the filter exists. Add
one function per milestone rather than editing old ones, so the gallery is
a history; run this file directly to regenerate every screenshot in one
call, so a stale image from an earlier code version never sits in the
README after a later refactor.

Filename convention: docs/screenshots/<module_number>[<letter>]_<name>.png
-- the numeric prefix keeps files sorted in a file browser in the same
order they appear in the README; a letter suffix (02a, 02b, ...)
disambiguates multiple images from the same module while preserving that
order.

Usage: .venv/bin/python docs/make_screenshots.py
"""

from __future__ import annotations

import pathlib
from datetime import datetime

import numpy as np

from orbit_od.constants import J2, MU_EARTH, R_EARTH
from orbit_od.dynamics import propagate
from orbit_od.measurements import simulate_gps_measurements
from orbit_od.plotting import (
    CATEGORICAL,
    MEAS_STYLE,
    TRUTH_STYLE,
    draw_earth_sphere,
    new_timeseries_figure,
    new_trajectory_figure,
)
from orbit_od.truth import propagate_tle

OUT_DIR = pathlib.Path(__file__).parent / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_ISS_TLE_LINE1 = "1 25544U 98067A   26251.46617104  .00001902  00000+0  42648-4 0  9994"
_ISS_TLE_LINE2 = "2 25544  51.6295 247.9233 0004929 112.8116 247.3394 15.49040812584669"


def _iss_truth(duration_s: float, step_s: float):
    return propagate_tle(
        _ISS_TLE_LINE1, _ISS_TLE_LINE2, datetime(2026, 9, 8, 12, 0, 0), duration_s, step_s
    )


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

    fig, ax = new_timeseries_figure(nrows=1)

    for j2_val, label, color in [
        (0.0, "J2 = 0 (pure two-body)", CATEGORICAL[0]),
        (J2, "J2 enabled", CATEGORICAL[1]),
    ]:
        result = propagate(state0, (0.0, n_orbits * period), t_eval=t_eval, j2=j2_val)
        raan = np.unwrap([raan_from_state(result.y[:, k]) for k in range(len(t_eval))])
        raan_deg_drift = np.degrees(raan - raan[0])
        ax.plot(
            t_eval / 3600.0,
            raan_deg_drift,
            color=color,
            linewidth=2,
            label=label,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("RAAN drift (deg)")
    ax.set_title("J2 secular nodal regression: 500 km, 51.6° orbit")
    ax.legend(frameon=False, labelcolor="#52514e", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_j2_raan_drift.png")


def module2a_iss_truth_trajectory_3d() -> None:
    """3D ISS truth trajectory over a 3-hour window, from a real TLE."""
    _t_s, states = _iss_truth(duration_s=3 * 3600.0, step_s=30.0)
    r = states[:, :3] / 1000.0  # km, for plot readability only

    fig, ax = new_trajectory_figure()
    draw_earth_sphere(ax, R_EARTH)
    ax.plot(r[:, 0], r[:, 1], r[:, 2], **TRUTH_STYLE)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("ISS truth trajectory, 3 h (SGP4, real TLE, TEME≈ECI)")
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02a_iss_trajectory_3d.png")


def module2b_iss_altitude_and_speed() -> None:
    """Altitude and speed vs time -- two stacked subplots, never dual-axis."""
    t_s, states = _iss_truth(duration_s=3 * 3600.0, step_s=10.0)
    altitude_km = (np.linalg.norm(states[:, :3], axis=1) - R_EARTH) / 1000.0
    speed_km_s = np.linalg.norm(states[:, 3:], axis=1) / 1000.0
    t_hr = t_s / 3600.0

    fig, axes = new_timeseries_figure(nrows=2)

    axes[0].plot(t_hr, altitude_km, color=TRUTH_STYLE["color"], linewidth=2)
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("ISS truth trajectory: altitude and speed (SGP4, real TLE)")

    axes[1].plot(t_hr, speed_km_s, color=TRUTH_STYLE["color"], linewidth=2)
    axes[1].set_ylabel("Speed (km/s)")
    axes[1].set_xlabel("Time (hours)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "02b_altitude_speed.png")


def module3_gps_measurements() -> None:
    """Per-axis GPS measurement residual (z - truth) vs time, at the
    project's realistic noise level. A literal trajectory overlay can't
    show 10 m of noise against a ~7000 km orbit radius, so this plots the
    quantity that actually matters -- the residual itself, against its
    +/-1-sigma and +/-3-sigma bounds -- foreshadowing the innovation/NIS
    plots Module 4/6 will add on top of the EKF."""
    sigma_m = 10.0
    t_s, states = _iss_truth(duration_s=3 * 3600.0, step_s=10.0)
    t_meas, z, _R = simulate_gps_measurements(
        t_s, states, sample_interval_s=10.0, sigma_m=sigma_m, rng=np.random.default_rng(0)
    )
    truth_idx = np.searchsorted(t_s, t_meas)
    residual = z - states[truth_idx, :3]
    t_hr = t_meas / 3600.0

    fig, axes = new_timeseries_figure(nrows=3)
    labels = ["X residual (m)", "Y residual (m)", "Z residual (m)"]
    for i, ax in enumerate(axes):
        ax.axhline(3 * sigma_m, color="#c3c2b7", linewidth=1, linestyle=":")
        ax.axhline(-3 * sigma_m, color="#c3c2b7", linewidth=1, linestyle=":")
        ax.axhline(sigma_m, color="#c3c2b7", linewidth=1, linestyle="--")
        ax.axhline(-sigma_m, color="#c3c2b7", linewidth=1, linestyle="--")
        ax.plot(t_hr, residual[:, i], **MEAS_STYLE)
        ax.set_ylabel(labels[i])
    axes[0].set_title(f"Simulated GPS measurement residuals (σ={sigma_m:.0f} m, 10 s cadence)")
    axes[-1].set_xlabel("Time (hours)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_gps_measurements.png")


if __name__ == "__main__":
    module1_j2_raan_drift()
    module2a_iss_truth_trajectory_3d()
    module2b_iss_altitude_and_speed()
    module3_gps_measurements()
    print(f"Wrote screenshots to {OUT_DIR}")
