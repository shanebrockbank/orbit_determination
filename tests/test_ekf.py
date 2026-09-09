"""Tests for orbit_od.ekf: hand-rolled EKF (finite-difference Jacobian)."""

import numpy as np
import pytest

from orbit_od.constants import J2, MU_EARTH, R_EARTH
from orbit_od.dynamics import propagate
from orbit_od.ekf import discrete_white_noise_q, predict, update


def _iss_like_circular_orbit_state():
    a = R_EARTH + 500e3
    v = np.sqrt(MU_EARTH / a)
    return np.array([a, 0.0, 0.0, 0.0, v, 0.0]), a


def test_update_matches_manual_position_block_formula():
    """Regression guard on H's construction inside update(): derive the
    expected gain/state/covariance purely by slicing P_pred (valid only
    if H really is [I3 | 0], since then S = P_pred[:3,:3] + R and
    P_pred @ H.T = P_pred[:, :3]) -- no H matrix is built in this test at
    all, so a fat-fingered H inside update() would show up as a mismatch."""
    rng = np.random.default_rng(7)
    x_pred = rng.normal(scale=1000.0, size=6)
    A = rng.normal(size=(6, 6))
    P_pred = A @ A.T + np.eye(6)  # symmetric positive-definite
    R = np.diag([25.0, 16.0, 36.0])
    z = x_pred[:3] + np.array([3.0, -2.0, 1.0])

    S_expected = P_pred[:3, :3] + R
    K_expected = P_pred[:, :3] @ np.linalg.inv(S_expected)
    innovation_expected = z - x_pred[:3]
    x_upd_expected = x_pred + K_expected @ innovation_expected

    KH = np.zeros((6, 6))
    KH[:, :3] = K_expected
    A_joseph = np.eye(6) - KH
    P_upd_expected = A_joseph @ P_pred @ A_joseph.T + K_expected @ R @ K_expected.T

    x_upd, P_upd, innovation = update(x_pred, P_pred, z, R)

    np.testing.assert_allclose(innovation, innovation_expected)
    np.testing.assert_allclose(x_upd, x_upd_expected, rtol=1e-10)
    np.testing.assert_allclose(P_upd, P_upd_expected, rtol=1e-10)


def test_update_keeps_covariance_symmetric_and_psd():
    rng = np.random.default_rng(3)
    x_pred = rng.normal(scale=1000.0, size=6)
    A = rng.normal(size=(6, 6))
    P_pred = A @ A.T + np.eye(6)
    R = np.diag([25.0, 25.0, 25.0])
    z = x_pred[:3] + rng.normal(scale=5.0, size=3)

    _x_upd, P_upd, _innov = update(x_pred, P_pred, z, R)

    np.testing.assert_allclose(P_upd, P_upd.T, atol=1e-9)
    eigvals = np.linalg.eigvalsh(P_upd)
    assert np.all(eigvals > -1e-9)


def test_discrete_white_noise_q_shape_and_blocks():
    dt = 5.0
    sigma_a = 1e-4
    Q = discrete_white_noise_q(dt, sigma_a)

    assert Q.shape == (6, 6)
    np.testing.assert_allclose(Q[0:3, 3:6], Q[3:6, 0:3].T)
    np.testing.assert_allclose(Q[0, 3], (dt**2 / 2.0) * sigma_a**2)
    np.testing.assert_allclose(np.diag(Q[0:3, 0:3]), np.full(3, (dt**3 / 3.0) * sigma_a**2))
    np.testing.assert_allclose(np.diag(Q[3:6, 3:6]), np.full(3, dt * sigma_a**2))


def test_predict_jacobian_mode_analytic_not_implemented():
    x0, _a = _iss_like_circular_orbit_state()
    with pytest.raises(NotImplementedError):
        predict(x0, np.eye(6), 10.0, np.zeros((6, 6)), jacobian_mode="analytic")


def test_predict_jacobian_mode_invalid_raises():
    x0, _a = _iss_like_circular_orbit_state()
    with pytest.raises(ValueError):
        predict(x0, np.eye(6), 10.0, np.zeros((6, 6)), jacobian_mode="bogus")


@pytest.mark.parametrize("j2_val", [0.0, J2])
def test_finite_difference_stm_is_symplectic(j2_val):
    """The two-body (and two-body+J2) state transition matrix must have
    determinant 1 (Liouville's theorem for conservative dynamics) -- an
    independent correctness check on the finite-difference Jacobian that
    doesn't rely on the rest of the filter. Checked via the public
    predict() API: with P=I and Q=0, P_pred = Phi @ Phi.T, so
    det(P_pred) = det(Phi)**2."""
    x0, _a = _iss_like_circular_orbit_state()
    _x_pred, P_pred = predict(x0, np.eye(6), 60.0, np.zeros((6, 6)), j2=j2_val)
    det_phi_squared = np.linalg.det(P_pred)
    np.testing.assert_allclose(det_phi_squared, 1.0, atol=1e-6)


def test_ekf_converges_within_one_orbit():
    """Filter seeded with +1 km position / +10 m/s velocity error must
    converge to < 20 m position error within one orbit. Truth is
    generated with the filter's own two-body+J2 model (dynamics.propagate)
    so this test isolates EKF correctness from model mismatch -- SGP4
    ground truth (with its unmodeled higher-order dynamics) is exercised
    in the Monte Carlo module instead."""
    x0_true, a = _iss_like_circular_orbit_state()
    n = np.sqrt(MU_EARTH / a**3)
    period = 2.0 * np.pi / n
    dt = 10.0
    n_steps = int(period // dt)
    t_grid = np.arange(n_steps + 1) * dt

    truth = propagate(x0_true, (0.0, t_grid[-1]), t_eval=t_grid, j2=J2).y.T

    sigma_m = 10.0
    R = (sigma_m**2) * np.eye(3)
    rng = np.random.default_rng(0)

    x_est = x0_true + np.array([1000.0, 0.0, 0.0, 0.0, 10.0, 0.0])
    P = np.diag([1000.0**2] * 3 + [10.0**2] * 3)
    Q = discrete_white_noise_q(dt, sigma_a=1e-6)

    for k in range(1, len(t_grid)):
        x_pred, P_pred = predict(x_est, P, dt, Q, j2=J2)
        z = truth[k, :3] + rng.normal(scale=sigma_m, size=3)
        x_est, P, _innovation = update(x_pred, P_pred, z, R)

    final_pos_error = np.linalg.norm(x_est[:3] - truth[-1, :3])
    assert final_pos_error < 20.0


def test_covariance_trace_shrinks_monotonically():
    """Total estimate uncertainty (trace(P), summed across all 6 states)
    must shrink monotonically as measurements are processed.

    Individual diagonal entries can wobble by a small amount step to step
    (observability of a given axis depends on where the satellite is in
    its orbit, and the finite-difference Jacobian carries its own small
    truncation error) -- that's expected behavior for a rotating state,
    not a bug. trace(P) is the robust, physically meaningful quantity
    that must never increase: it is the aggregate uncertainty the filter
    is supposed to be reducing every time it incorporates a measurement.
    """
    x0_true, a = _iss_like_circular_orbit_state()
    n = np.sqrt(MU_EARTH / a**3)
    period = 2.0 * np.pi / n
    dt = 10.0
    n_steps = int(period // dt)
    t_grid = np.arange(n_steps + 1) * dt

    truth = propagate(x0_true, (0.0, t_grid[-1]), t_eval=t_grid, j2=J2).y.T

    sigma_m = 10.0
    R = (sigma_m**2) * np.eye(3)
    rng = np.random.default_rng(0)

    x_est = x0_true + np.array([1000.0, 0.0, 0.0, 0.0, 10.0, 0.0])
    P = np.diag([1000.0**2] * 3 + [10.0**2] * 3)
    Q = discrete_white_noise_q(dt, sigma_a=1e-6)

    traces = [np.trace(P)]
    for k in range(1, len(t_grid)):
        x_pred, P_pred = predict(x_est, P, dt, Q, j2=J2)
        z = truth[k, :3] + rng.normal(scale=sigma_m, size=3)
        x_est, P, _innovation = update(x_pred, P_pred, z, R)
        traces.append(np.trace(P))

    traces = np.array(traces)
    assert np.all(np.diff(traces) <= 1e-9)
