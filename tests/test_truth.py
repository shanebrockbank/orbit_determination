"""Tests for orbit_od.truth: SGP4 ground-truth generation from a real TLE."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from orbit_od.constants import R_EARTH
from orbit_od.truth import propagate_tle

# Close to the fixture TLE's own epoch (~2026-09-08 11:11 UTC) so SGP4 is
# evaluated near its reference point rather than extrapolated far from it.
_START = datetime(2026, 9, 8, 12, 0, 0)


def test_shape_and_time_grid(iss_tle):
    line1, line2 = iss_tle
    duration_s = 3 * 3600.0
    step_s = 60.0

    t_s, states = propagate_tle(line1, line2, _START, duration_s, step_s)

    expected_n = int(round(duration_s / step_s)) + 1
    assert t_s.shape == (expected_n,)
    assert states.shape == (expected_n, 6)
    assert t_s[0] == 0.0
    assert t_s[-1] == pytest.approx(duration_s)
    np.testing.assert_allclose(np.diff(t_s), step_s)


def test_altitude_within_iss_range(iss_tle):
    """ISS operates at roughly 400-420 km altitude; allow margin for the
    small eccentricity-driven variation and real (non-circular) telemetry."""
    line1, line2 = iss_tle
    t_s, states = propagate_tle(line1, line2, _START, 3 * 3600.0, 60.0)

    r = states[:, :3]
    altitude_m = np.linalg.norm(r, axis=1) - R_EARTH

    assert np.all(altitude_m > 395e3)
    assert np.all(altitude_m < 435e3)


def test_speed_matches_iss_orbital_velocity(iss_tle):
    line1, line2 = iss_tle
    t_s, states = propagate_tle(line1, line2, _START, 3 * 3600.0, 60.0)

    v = states[:, 3:]
    speed_m_s = np.linalg.norm(v, axis=1)

    assert np.all(speed_m_s > 7550.0)
    assert np.all(speed_m_s < 7750.0)


def test_units_are_meters_not_kilometers(iss_tle):
    """Sanity guard against a missed km->m conversion: LEO position
    components should be on the order of 1e6-1e7 m, not 1e3-1e4 km."""
    line1, line2 = iss_tle
    _, states = propagate_tle(line1, line2, _START, 60.0, 60.0)

    r = states[:, :3]
    assert np.all(np.abs(r) > 1e5)
    assert np.all(np.linalg.norm(r, axis=1) < 1e7)


def test_aware_and_naive_utc_datetimes_agree(iss_tle):
    """A tz-aware UTC datetime and the equivalent naive datetime must
    produce identical propagation -- guards the frame/time normalization
    in _to_utc_naive."""
    line1, line2 = iss_tle
    naive_start = _START
    aware_start = _START.replace(tzinfo=timezone.utc)

    _, states_naive = propagate_tle(line1, line2, naive_start, 600.0, 60.0)
    _, states_aware = propagate_tle(line1, line2, aware_start, 600.0, 60.0)

    np.testing.assert_array_equal(states_naive, states_aware)


def test_non_utc_timezone_is_converted_correctly(iss_tle):
    """A datetime in a non-UTC timezone should convert to the same instant
    as its UTC equivalent, not be treated as naive UTC."""
    line1, line2 = iss_tle
    utc_start = _START.replace(tzinfo=timezone.utc)
    offset_start = utc_start.astimezone(timezone(timedelta(hours=-5)))

    _, states_utc = propagate_tle(line1, line2, utc_start, 600.0, 60.0)
    _, states_offset = propagate_tle(line1, line2, offset_start, 600.0, 60.0)

    np.testing.assert_array_equal(states_utc, states_offset)


def test_sgp4_error_raises_runtime_error():
    """An orbit that has decayed (per the SGP4 model) should surface as a
    RuntimeError identifying the bad samples, not a silent bad state."""
    # Deliberately degenerate TLE: near-zero mean motion / decayed orbit,
    # causes SGP4 to return a non-zero error code (e.g. "mean motion less
    # than zero" or similar) rather than propagating successfully.
    line1 = "1 00005U 58002B   26251.00000000  .00000000  00000-0  00000-0 0  9998"
    line2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264  0.00000000000015"

    with pytest.raises(RuntimeError):
        propagate_tle(line1, line2, _START, 600.0, 60.0)
