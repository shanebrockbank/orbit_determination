"""Shared pytest fixtures."""

import pathlib

import pytest

_TLE_PATH = pathlib.Path(__file__).parent / "data" / "iss_tle.txt"


@pytest.fixture
def iss_tle() -> tuple[str, str]:
    """Real ISS (ZARYA) TLE, frozen offline so tests don't need network.

    Returns:
        (line1, line2) TLE lines. Epoch: 2026-09-08 ~11:11 UTC.
    """
    lines = _TLE_PATH.read_text().strip().splitlines()
    return lines[1].strip(), lines[2].strip()
