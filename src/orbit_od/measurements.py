"""Simulated noisy GPS position measurements sampled from a truth trajectory.

Measurements are direct 3D position fixes (as from a GPS receiver): a
noiseless truth position with zero-mean Gaussian noise added per axis.
Units are meters throughout, matching the rest of the package.
"""

from __future__ import annotations

import numpy as np


def simulate_gps_measurements(
    t_truth_s: np.ndarray,
    truth_states: np.ndarray,
    sample_interval_s: float = 10.0,
    sigma_m: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate GPS position fixes from a truth trajectory.

    Sampling is an exact subset of `t_truth_s` -- the index stride is
    `round(sample_interval_s / dt)` where dt is the truth grid's (assumed
    uniform) spacing -- not interpolated, so every returned timestamp and
    measurement corresponds exactly to a row of `truth_states`.

    Args:
        t_truth_s: shape (N,), seconds, uniformly spaced truth time grid.
        truth_states: shape (N,6), meters and m/s, [px,py,pz,vx,vy,vz] ECI.
        sample_interval_s: spacing between GPS fixes, seconds.
        sigma_m: per-axis 1-sigma Gaussian position noise, meters.
        rng: numpy Generator for reproducibility. If None, a fresh
            `np.random.default_rng()` is created -- this function never
            falls back to the global numpy random state.

    Returns:
        t_meas_s: shape (M,), seconds -- exact subset of t_truth_s.
        z: shape (M,3), meters -- noisy position measurements.
        R: shape (3,3), m^2 -- measurement noise covariance, constant and
            diagonal (sigma_m**2 * I) for this simulator.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = t_truth_s[1] - t_truth_s[0]
    stride = max(1, round(sample_interval_s / dt))

    idx = np.arange(0, len(t_truth_s), stride)
    t_meas_s = t_truth_s[idx]
    truth_pos = truth_states[idx, :3]

    noise = rng.normal(loc=0.0, scale=sigma_m, size=truth_pos.shape)
    z = truth_pos + noise

    R = (sigma_m**2) * np.eye(3)

    return t_meas_s, z, R
