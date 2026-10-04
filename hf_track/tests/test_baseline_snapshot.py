"""Baseline test count snapshot.

Locks the test count at 619 (excluding this test itself) so that
regressions of the modular refactor — where tests may be split but not
silently removed — are immediately visible.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

# Baseline frozen on 2026-10-04 (plan 2026-10-04, step S01), after the
# collection abort in test_smooth_ticker.py was removed: that file's 18
# cases had never been counted, so re-measuring the now-collectable suite
# raised this from 386 to 619. Update this constant deliberately when
# adding or removing tests in bulk.
BASELINE_TEST_COUNT = 619


def _collect_test_count() -> int:
    """Run `pytest --collect-only -q` and return the parsed test count."""
    tests_dir = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=tests_dir.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"pytest collect-only failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    for line in result.stdout.splitlines():
        match = re.match(r"^(\d+) tests collected", line.strip())
        if match:
            return int(match.group(1))
    pytest.fail(f"Could not parse test count from:\n{result.stdout}")


def test_baseline_test_count() -> None:
    """The baseline test count (excluding this test) was 619.

    This test contributes 1 to the collected count, so we subtract 1 before
    comparing against the baseline.
    """
    total_collected = _collect_test_count()
    # Exclude this very test from the count.
    count_excluding_self = total_collected - 1
    assert count_excluding_self >= BASELINE_TEST_COUNT, (
        f"Test count regressed: {count_excluding_self} tests collected "
        f"(excluding this snapshot test), baseline was "
        f"{BASELINE_TEST_COUNT}. Update BASELINE_TEST_COUNT in "
        f"test_baseline_snapshot.py if this is an intentional change."
    )
