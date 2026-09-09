"""Two-body + J2 orbital dynamics and numerical propagation.

State vector convention used throughout this module and the rest of the
package:

    x = [px, py, pz, vx, vy, vz]^T

with position in meters and velocity in meters/second, expressed in an
Earth-centered inertial (ECI) frame.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .constants import J2, MU_EARTH, R_EARTH


def two_body_j2_accel(
    r_eci: np.ndarray,
    mu: float = MU_EARTH,
    j2: float = J2,
    r_eq: float = R_EARTH,
) -> np.ndarray:
    """Two-body + J2 gravitational acceleration in the ECI frame.

    Args:
        r_eci: Position vector [x, y, z], meters, ECI frame.
        mu: Earth gravitational parameter, m^3/s^2.
        j2: J2 zonal harmonic coefficient, dimensionless. Pass 0.0 to
            disable the J2 term and get pure two-body acceleration.
        r_eq: Earth equatorial radius, meters.

    Returns:
        Acceleration vector [ax, ay, az], m/s^2, ECI frame.
    """
    r_eci = np.asarray(r_eci, dtype=float)
    x, y, z = r_eci
    r = np.linalg.norm(r_eci)

    a_two_body = -mu * r_eci / r**3

    z_r2 = (z / r) ** 2
    j2_factor = -1.5 * j2 * mu * r_eq**2 / r**5
    a_j2 = j2_factor * np.array(
        [
            x * (1.0 - 5.0 * z_r2),
            y * (1.0 - 5.0 * z_r2),
            z * (3.0 - 5.0 * z_r2),
        ]
    )

    return a_two_body + a_j2


def eom(
    t: float,
    state: np.ndarray,
    mu: float = MU_EARTH,
    j2: float = J2,
    r_eq: float = R_EARTH,
) -> np.ndarray:
    """Equations of motion (state derivative) for solve_ivp.

    Args:
        t: Time, seconds. Unused (dynamics are time-invariant) but
            required by the solve_ivp callback signature.
        state: [px, py, pz, vx, vy, vz], meters and m/s, ECI frame.
        mu: Earth gravitational parameter, m^3/s^2.
        j2: J2 coefficient, dimensionless (0.0 disables J2).
        r_eq: Earth equatorial radius, meters.

    Returns:
        d(state)/dt = [vx, vy, vz, ax, ay, az], m/s and m/s^2.
    """
    r_eci = state[:3]
    v_eci = state[3:]
    a_eci = two_body_j2_accel(r_eci, mu=mu, j2=j2, r_eq=r_eq)
    return np.concatenate([v_eci, a_eci])


def propagate(
    state0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray | None = None,
    mu: float = MU_EARTH,
    j2: float = J2,
    r_eq: float = R_EARTH,
    rtol: float = 1e-12,
    atol: float = 1e-12,
):
    """Propagate two-body + J2 dynamics with RK45.

    Args:
        state0: Initial [px, py, pz, vx, vy, vz], meters and m/s, ECI.
        t_span: (t0, tf), seconds.
        t_eval: Times at which to store the solution, seconds. If None,
            solve_ivp chooses its own output times.
        mu: Earth gravitational parameter, m^3/s^2.
        j2: J2 coefficient, dimensionless (0.0 disables J2).
        r_eq: Earth equatorial radius, meters.
        rtol: Relative tolerance passed to solve_ivp.
        atol: Absolute tolerance passed to solve_ivp.

    Returns:
        The scipy OdeResult. `.t` is shape (N,) seconds, `.y` is shape
        (6, N) with rows [px, py, pz, vx, vy, vz] in meters and m/s.
    """
    result = solve_ivp(
        eom,
        t_span,
        np.asarray(state0, dtype=float),
        method="RK45",
        t_eval=t_eval,
        args=(mu, j2, r_eq),
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    return result
