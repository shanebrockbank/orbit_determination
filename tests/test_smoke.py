"""Environment smoke tests: prove Python, deps, and the package all run."""

import sys

import numpy as np
import scipy

import orbit_determination as od


def test_python_version():
    assert sys.version_info >= (3, 10)


def test_scientific_stack_imports():
    assert np.__version__
    assert scipy.__version__


def test_numpy_actually_computes():
    a = np.array([1.0, 2.0, 3.0])
    assert np.isclose(a.sum(), 6.0)


def test_circular_orbit_speed():
    """For a circular orbit (r == a), vis-viva reduces to sqrt(mu/r)."""
    a = 7000e3
    assert np.isclose(od.vis_viva_speed(a, a), np.sqrt(od.MU_EARTH / a))


def test_leo_period_is_about_97_minutes():
    a = 7000e3  # ~622 km altitude
    minutes = od.orbital_period(a) / 60.0
    assert 96.0 < minutes < 98.0


def test_vis_viva_is_vectorized():
    a = 7000e3
    v = od.vis_viva_speed(np.array([a, a]), a)
    assert v.shape == (2,)


def test_hyperbolic_radius_beyond_apoapsis_is_nan():
    """r > 2a on a bound orbit is unreachable: expect a NaN, not a crash."""
    with np.errstate(invalid="ignore"):
        assert np.isnan(od.vis_viva_speed(1e12, 7000e3))
