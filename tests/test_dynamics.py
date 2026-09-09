"""Tests for orbit_od.dynamics: two-body + J2 propagation."""

import numpy as np
import pytest

from orbit_od.constants import J2, MU_EARTH, R_EARTH
from orbit_od.dynamics import propagate


def _circular_equatorial_state(a: float, mu: float = MU_EARTH) -> np.ndarray:
    """State for a circular, equatorial orbit at semi-major axis `a` (m)."""
    v = np.sqrt(mu / a)
    return np.array([a, 0.0, 0.0, 0.0, v, 0.0])


def _circular_inclined_state(a: float, inc: float, mu: float = MU_EARTH) -> np.ndarray:
    """State at the ascending node for a circular orbit of inclination `inc`
    (rad), semi-major axis `a` (m), with RAAN = 0 and argument of perigee = 0.
    """
    v = np.sqrt(mu / a)
    return np.array([a, 0.0, 0.0, 0.0, v * np.cos(inc), v * np.sin(inc)])


def _raan_from_state(state: np.ndarray) -> float:
    """Right ascension of the ascending node (rad, [0, 2*pi)) from a
    Cartesian state [px,py,pz,vx,vy,vz] (m, m/s)."""
    r = state[:3]
    v = state[3:]
    h = np.cross(r, v)
    node = np.cross(np.array([0.0, 0.0, 1.0]), h)
    raan = np.arctan2(node[1], node[0])
    return raan % (2.0 * np.pi)


def test_two_body_no_j2_matches_analytical_circular_orbit():
    a = R_EARTH + 500e3  # 500 km altitude, circular, equatorial
    state0 = _circular_equatorial_state(a)
    n = np.sqrt(MU_EARTH / a**3)
    period = 2.0 * np.pi / n

    t_eval = np.linspace(0.0, period, 9)
    result = propagate(state0, (0.0, period), t_eval=t_eval, j2=0.0)
    assert result.success

    theta = n * t_eval
    x_expected = a * np.cos(theta)
    y_expected = a * np.sin(theta)
    vx_expected = -a * n * np.sin(theta)
    vy_expected = a * n * np.cos(theta)

    np.testing.assert_allclose(result.y[0], x_expected, atol=5e-4)
    np.testing.assert_allclose(result.y[1], y_expected, atol=5e-4)
    np.testing.assert_allclose(result.y[2], 0.0, atol=1e-9)
    np.testing.assert_allclose(result.y[3], vx_expected, atol=1e-6)
    np.testing.assert_allclose(result.y[4], vy_expected, atol=1e-6)
    np.testing.assert_allclose(result.y[5], 0.0, atol=1e-9)


def test_j2_raan_drift_matches_analytical_secular_rate():
    a = R_EARTH + 500e3  # 500 km altitude, circular
    inc = np.radians(51.6)  # ISS-like inclination
    state0 = _circular_inclined_state(a, inc)

    n = np.sqrt(MU_EARTH / a**3)
    period = 2.0 * np.pi / n
    n_orbits = 10
    t_eval = np.array([k * period for k in range(n_orbits + 1)])

    result = propagate(state0, (0.0, n_orbits * period), t_eval=t_eval, j2=J2)
    assert result.success

    raan_samples = np.array([_raan_from_state(result.y[:, k]) for k in range(len(t_eval))])
    # Circular orbit, sampled once per (unperturbed) period: no wraparound
    # expected over this short a span, so a direct unwrap is safe.
    raan_samples = np.unwrap(raan_samples)

    slope, _ = np.polyfit(t_eval, raan_samples, 1)

    p = a  # semi-latus rectum for e = 0
    raan_dot_analytical = -1.5 * n * J2 * (R_EARTH / p) ** 2 * np.cos(inc)

    assert raan_dot_analytical != pytest.approx(0.0)
    np.testing.assert_allclose(slope, raan_dot_analytical, rtol=0.02)


def test_j2_disabled_gives_zero_raan_drift():
    a = R_EARTH + 500e3
    inc = np.radians(51.6)
    state0 = _circular_inclined_state(a, inc)
    n = np.sqrt(MU_EARTH / a**3)
    period = 2.0 * np.pi / n
    t_eval = np.array([0.0, 5 * period])

    result = propagate(state0, (0.0, 5 * period), t_eval=t_eval, j2=0.0)
    raan0 = _raan_from_state(result.y[:, 0])
    raan1 = _raan_from_state(result.y[:, 1])

    delta = (raan1 - raan0 + np.pi) % (2.0 * np.pi) - np.pi
    np.testing.assert_allclose(delta, 0.0, atol=1e-8)
