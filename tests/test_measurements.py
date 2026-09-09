"""Tests for orbit_od.measurements: simulated noisy GPS position fixes."""

import numpy as np

from orbit_od.measurements import simulate_gps_measurements


def _synthetic_truth(n: int = 5000, dt: float = 1.0):
    t = np.arange(n) * dt
    pos = np.column_stack([t, 2.0 * t, 0.5 * t])
    vel = np.zeros((n, 3))
    return t, np.hstack([pos, vel])


def test_exact_subset_alignment():
    t_truth, states = _synthetic_truth(n=1000, dt=1.0)
    t_meas, z, _R = simulate_gps_measurements(
        t_truth, states, sample_interval_s=10.0, sigma_m=5.0, rng=np.random.default_rng(0)
    )

    # Every measurement timestamp is exactly one of the truth timestamps.
    assert np.all(np.isin(t_meas, t_truth))
    # Stride matches sample_interval_s / dt exactly (dt = 1.0 here).
    np.testing.assert_allclose(np.diff(t_meas), 10.0)
    assert t_meas[0] == t_truth[0]


def test_r_shape_and_value():
    t_truth, states = _synthetic_truth(n=100)
    sigma_m = 7.5
    _t_meas, _z, R = simulate_gps_measurements(
        t_truth, states, sample_interval_s=10.0, sigma_m=sigma_m, rng=np.random.default_rng(0)
    )

    assert R.shape == (3, 3)
    np.testing.assert_allclose(R, (sigma_m**2) * np.eye(3))


def test_noise_statistics_converge_to_zero_mean_and_sigma():
    n = 5000
    sigma_m = 10.0
    t_truth, states = _synthetic_truth(n=n, dt=1.0)

    t_meas, z, _R = simulate_gps_measurements(
        t_truth, states, sample_interval_s=1.0, sigma_m=sigma_m, rng=np.random.default_rng(42)
    )
    truth_idx = np.searchsorted(t_truth, t_meas)
    residual = z - states[truth_idx, :3]

    # Standard error of the mean here is sigma/sqrt(n) ~= 0.14 m; 1.0 m is a
    # generous multi-sigma margin against random seed variation.
    np.testing.assert_allclose(residual.mean(axis=0), 0.0, atol=1.0)
    # Sample std's relative error here is ~1%; 5% leaves ample margin.
    np.testing.assert_allclose(residual.std(axis=0), sigma_m, rtol=0.05)


def test_reproducibility_with_fixed_seed():
    t_truth, states = _synthetic_truth(n=500)

    _t1, z1, _r1 = simulate_gps_measurements(
        t_truth, states, sample_interval_s=10.0, sigma_m=10.0, rng=np.random.default_rng(123)
    )
    _t2, z2, _r2 = simulate_gps_measurements(
        t_truth, states, sample_interval_s=10.0, sigma_m=10.0, rng=np.random.default_rng(123)
    )

    np.testing.assert_array_equal(z1, z2)


def test_different_seeds_give_different_measurements():
    t_truth, states = _synthetic_truth(n=500)

    _t1, z1, _r1 = simulate_gps_measurements(
        t_truth, states, sample_interval_s=10.0, sigma_m=10.0, rng=np.random.default_rng(1)
    )
    _t2, z2, _r2 = simulate_gps_measurements(
        t_truth, states, sample_interval_s=10.0, sigma_m=10.0, rng=np.random.default_rng(2)
    )

    assert not np.array_equal(z1, z2)


def test_no_rng_never_uses_global_numpy_state():
    """Two calls with rng=None must not be forced into lockstep by the
    global numpy random state (e.g. via a hidden np.random.seed leak)."""
    t_truth, states = _synthetic_truth(n=500)

    _t1, z1, _r1 = simulate_gps_measurements(t_truth, states, sample_interval_s=10.0, sigma_m=10.0)
    _t2, z2, _r2 = simulate_gps_measurements(t_truth, states, sample_interval_s=10.0, sigma_m=10.0)

    assert not np.array_equal(z1, z2)
