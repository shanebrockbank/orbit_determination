"""Physical constants used throughout orbit_od.

All modules import from here so that mu, J2, and Earth radius stay
consistent across dynamics, truth generation, and the EKF.

Units are strict SI (meters, seconds, radians) unless noted.
"""

#: Earth gravitational parameter GM, m^3/s^2 (WGS-84 / EGM96).
MU_EARTH = 3.986004418e14

#: Earth equatorial radius, m (WGS-84).
R_EARTH = 6378137.0

#: Earth's J2 zonal harmonic coefficient, dimensionless (WGS-84 / EGM96).
J2 = 1.08262668e-3

#: Earth rotation rate, rad/s (WGS-84), for TEME/ECEF frame conversions.
OMEGA_EARTH = 7.2921150e-5
