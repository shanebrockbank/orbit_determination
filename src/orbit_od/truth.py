"""Ground-truth trajectory generation via SGP4 propagation of a real TLE.

Frame handling (read before using this module's output):

SGP4 natively propagates in TEME (True Equator, Mean Equinox of date) --
a quasi-inertial frame whose equator tracks Earth's *true* (instantaneous)
equatorial plane, but whose x-axis (equinox) is defined by a low-precision
analytical theory baked into the SGP4 model itself. This is close to, but
not identical to, a standard ECI frame such as J2000/GCRF: the difference
is driven by precession and nutation, and grows with time since the TLE's
own reference epoch.

This project treats TEME as ECI without further rotation. That is a
deliberate, documented simplification, valid because:

  - The propagation windows used here are a few hours (a handful of ISS
    orbits), over which precession/nutation move the true equinox by at
    most a few arcseconds -- a few tens of meters of position error at
    LEO altitude at worst, one order of magnitude below the ~5-15 m GPS
    measurement noise this project's EKF is designed to handle.
  - dynamics.py's two-body + J2 model is itself most naturally expressed
    relative to the *true* equatorial bulge, which TEME already tracks --
    arguably a better match than rotating into a mean-equator-of-J2000
    frame would be, for this short-arc use case.

A production system, or one propagating over days rather than hours,
would apply an explicit IAU-76/FK5 (or IAU-2006/2000A) precession-nutation
rotation from TEME to GCRF/J2000. That is out of scope here and is not
implemented.

Units: SGP4 natively returns position in km and velocity in km/s. Both
are converted to the project's SI convention (m, m/s) before leaving this
module -- no other module in this package should ever see kilometers.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sgp4.api import Satrec, jday


def _to_utc_naive(dt: datetime) -> datetime:
    """Normalize a datetime to naive UTC (assumes naive input is already UTC)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def propagate_tle(
    line1: str,
    line2: str,
    start: datetime,
    duration_s: float,
    step_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a two-line element set with SGP4 over a fixed time grid.

    Args:
        line1: TLE line 1.
        line2: TLE line 2.
        start: UTC epoch of the first sample. Timezone-aware datetimes are
            converted to UTC; naive datetimes are assumed to already be UTC.
        duration_s: Total propagation span, seconds, from `start`.
        step_s: Sample spacing, seconds.

    Returns:
        t_s: shape (N,), seconds elapsed since `start` (0, step_s, 2*step_s, ...).
        states: shape (N, 6), columns [px, py, pz, vx, vy, vz], meters and
            m/s, in the TEME frame (treated as ECI -- see module docstring).

    Raises:
        RuntimeError: If SGP4 reports a propagation error at any sample
            (e.g. decayed orbit); the error identifies the offending
            sample indices and SGP4 error codes.
    """
    sat = Satrec.twoline2rv(line1, line2)

    start = _to_utc_naive(start)
    n = int(round(duration_s / step_s)) + 1
    t_s = np.arange(n) * step_s

    jd0, fr0 = jday(
        start.year,
        start.month,
        start.day,
        start.hour,
        start.minute,
        start.second + start.microsecond * 1e-6,
    )
    jds = np.full(n, jd0)
    frs = fr0 + t_s / 86400.0

    err, r_km, v_km_s = sat.sgp4_array(jds, frs)
    if np.any(err != 0):
        bad = np.nonzero(err)[0]
        raise RuntimeError(
            f"SGP4 propagation error at sample indices {bad.tolist()}: "
            f"codes {err[bad].tolist()}"
        )

    r_m = r_km * 1000.0
    v_m_s = v_km_s * 1000.0
    states = np.hstack([r_m, v_m_s])
    return t_s, states
