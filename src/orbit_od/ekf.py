"""Hand-rolled Extended Kalman Filter for GPS-based orbit determination.

State vector convention matches the rest of the package:

    x = [px, py, pz, vx, vy, vz]^T, meters and m/s, ECI frame.

The state transition Jacobian F (equivalently, the discrete-time state
transition matrix Phi_k) is computed by finite differences of the
existing, already-tested two-body+J2 propagator in dynamics.py, rather
than a hand-derived analytic J2 Jacobian propagated via a separate
variational-equation integration. Concretely: perturb each of the 6
state components by a small step, propagate x+delta and x-delta each
over the full predict interval with dynamics.propagate(), and take the
central difference of the two propagated (discrete-time) states. This
gives Phi_k directly -- exactly what the covariance propagation step
needs -- without introducing and separately validating a second
integration scheme for Phi_dot = F @ Phi.

An `jacobian_mode="analytic"` option is reserved in predict()'s
signature for a future hand-derived J2 Jacobian (a good validation
target: assert it agrees with the finite-difference Phi_k to a tight
tolerance), but is not implemented yet -- it raises NotImplementedError.
"""

from __future__ import annotations

import numpy as np

from .constants import J2, MU_EARTH, R_EARTH
from .dynamics import propagate


def _propagate_state(
    x0: np.ndarray, dt: float, mu: float, j2: float, r_eq: float, rtol: float, atol: float
) -> np.ndarray:
    """Propagate a single state forward by dt and return the final state."""
    result = propagate(x0, (0.0, dt), t_eval=np.array([dt]), mu=mu, j2=j2, r_eq=r_eq,
                        rtol=rtol, atol=atol)
    return result.y[:, -1]


def predict(
    x: np.ndarray,
    P: np.ndarray,
    dt: float,
    Q: np.ndarray,
    mu: float = MU_EARTH,
    j2: float = J2,
    r_eq: float = R_EARTH,
    jacobian_mode: str = "finite_diff",
    pos_step_m: float = 1.0,
    vel_step_m_s: float = 1e-3,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict step: propagate the state and covariance forward by dt.

    Args:
        x: State estimate [px,py,pz,vx,vy,vz], meters and m/s, shape (6,).
        P: State covariance, shape (6,6), SI units.
        dt: Prediction interval, seconds.
        Q: Process noise covariance for this interval, shape (6,6), SI
            units (see discrete_white_noise_q() for a standard model).
        mu: Earth gravitational parameter, m^3/s^2.
        j2: J2 coefficient, dimensionless.
        r_eq: Earth equatorial radius, meters.
        jacobian_mode: "finite_diff" (default, implemented) or "analytic"
            (reserved for a future hand-derived J2 Jacobian; raises
            NotImplementedError for now).
        pos_step_m: Finite-difference perturbation for position states, m.
        vel_step_m_s: Finite-difference perturbation for velocity states, m/s.
        rtol: Relative tolerance for the internal propagations.
        atol: Absolute tolerance for the internal propagations.

    Returns:
        x_pred: Predicted state, shape (6,), meters and m/s.
        P_pred: Predicted covariance, shape (6,6), SI units.
            P_pred = Phi_k @ P @ Phi_k.T + Q, where Phi_k is the
            discrete-time state transition matrix from t to t+dt.
    """
    if jacobian_mode == "analytic":
        raise NotImplementedError(
            "Analytic J2 Jacobian is not implemented; use jacobian_mode='finite_diff'."
        )
    if jacobian_mode != "finite_diff":
        raise ValueError(f"Unknown jacobian_mode: {jacobian_mode!r}")

    x = np.asarray(x, dtype=float)
    steps = np.array([pos_step_m] * 3 + [vel_step_m_s] * 3)

    x_pred = _propagate_state(x, dt, mu, j2, r_eq, rtol, atol)

    Phi = np.zeros((6, 6))
    for i in range(6):
        dx = np.zeros(6)
        dx[i] = steps[i]
        x_plus = _propagate_state(x + dx, dt, mu, j2, r_eq, rtol, atol)
        x_minus = _propagate_state(x - dx, dt, mu, j2, r_eq, rtol, atol)
        Phi[:, i] = (x_plus - x_minus) / (2.0 * steps[i])

    P_pred = Phi @ P @ Phi.T + Q
    return x_pred, P_pred


def update(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    z: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measurement update step for direct 3D GPS position fixes.

    Args:
        x_pred: Predicted state [px,py,pz,vx,vy,vz], meters and m/s,
            shape (6,), ECI frame.
        P_pred: Predicted state covariance, shape (6,6), SI units.
        z: Measured position [zx,zy,zz], meters, shape (3,), ECI frame.
        R: Measurement noise covariance, shape (3,3), m^2.

    The measurement Jacobian H = [I_3x3 | 0_3x3] (shape (3,6)) is fixed
    and built fresh on every call, since this project's measurement model
    (direct position observation) never varies.

    Covariance is updated with the Joseph form rather than the textbook
    P_upd = (I - K H) P_pred, since Joseph form stays exactly symmetric
    and positive-semi-definite in floating point even when K is not the
    exact optimal gain (as in an EKF, where H is itself a linearization).

    Returns:
        x_upd: Updated state estimate, shape (6,), meters and m/s.
        P_upd: Updated covariance, shape (6,6), Joseph-form update.
        innovation: Pre-update residual z - H @ x_pred, shape (3,), meters.
    """
    H = np.zeros((3, 6))
    H[:, :3] = np.eye(3)

    innovation = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_upd = x_pred + K @ innovation

    I6 = np.eye(6)
    A = I6 - K @ H
    P_upd = A @ P_pred @ A.T + K @ R @ K.T

    return x_upd, P_upd, innovation


def discrete_white_noise_q(dt: float, sigma_a: float) -> np.ndarray:
    """Discrete white-noise-acceleration (DWNA) process noise covariance.

    Models unmodeled dynamics between predict steps (higher-order
    gravity, drag, SRP -- anything beyond the two-body+J2 model) as a
    continuous white-noise acceleration with 1-sigma magnitude sigma_a,
    held constant over one step of length dt.

    Args:
        dt: Time step, seconds.
        sigma_a: 1-sigma unmodeled acceleration, m/s^2, equal and
            uncorrelated across x/y/z.

    Returns:
        Q: Process noise covariance, shape (6,6), grouped to match this
            project's [px,py,pz,vx,vy,vz] state ordering (three 3x3
            sub-blocks, not interleaved per axis):

                Q[0:3, 0:3] = (dt**3 / 3) * sigma_a**2 * I_3
                Q[0:3, 3:6] = (dt**2 / 2) * sigma_a**2 * I_3
                Q[3:6, 0:3] = (dt**2 / 2) * sigma_a**2 * I_3
                Q[3:6, 3:6] =  dt        * sigma_a**2 * I_3
    """
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = (dt**3 / 3.0) * sigma_a**2 * np.eye(3)
    Q[0:3, 3:6] = (dt**2 / 2.0) * sigma_a**2 * np.eye(3)
    Q[3:6, 0:3] = (dt**2 / 2.0) * sigma_a**2 * np.eye(3)
    Q[3:6, 3:6] = dt * sigma_a**2 * np.eye(3)
    return Q
