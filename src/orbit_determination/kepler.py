"""Two-body Keplerian relations."""

import numpy as np

from .constants import MU_EARTH


def vis_viva_speed(r, a, mu=MU_EARTH):
    """Orbital speed at radius `r` on an orbit of semi-major axis `a`.

    v = sqrt(mu * (2/r - 1/a))
    """
    r = np.asarray(r, dtype=float)
    a = np.asarray(a, dtype=float)
    return np.sqrt(mu * (2.0 / r - 1.0 / a))


def orbital_period(a, mu=MU_EARTH):
    """Keplerian period for semi-major axis `a`: T = 2*pi*sqrt(a^3/mu)."""
    a = np.asarray(a, dtype=float)
    return 2.0 * np.pi * np.sqrt(a**3 / mu)
