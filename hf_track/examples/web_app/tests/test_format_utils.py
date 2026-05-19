"""Tests for the Python reference formatting functions.

These validate the same logic that app.js implements on the frontend.
If you change the logic here, also update app.js.
"""
from __future__ import annotations

import math

from format_utils import format_bytes, format_eta, format_percentage, format_speed


class TestFormatBytes:
    def test_zero(self):
        assert format_bytes(0) == "0 B"

    def test_bytes(self):
        assert format_bytes(512) == "512 B"

    def test_kb(self):
        assert format_bytes(1536) == "1.5 KB"

    def test_mb(self):
        assert format_bytes(1048576) == "1.0 MB"

    def test_gb(self):
        assert format_bytes(1073741824) == "1.0 GB"

    def test_tb(self):
        assert format_bytes(1099511627776) == "1.0 TB"

    def test_fractional_mb(self):
        assert format_bytes(1572864) == "1.5 MB"


class TestFormatSpeed:
    def test_zero(self):
        assert format_speed(0) == "0 B/s"

    def test_kb_per_sec(self):
        assert format_speed(1536) == "1.5 KB/s"

    def test_mb_per_sec(self):
        assert format_speed(1048576) == "1.0 MB/s"


class TestFormatEta:
    def test_seconds(self):
        assert format_eta(45) == "45s"

    def test_minutes_seconds(self):
        assert format_eta(125) == "2m05s"

    def test_hours_minutes_seconds(self):
        assert format_eta(3665) == "1h01m05s"

    def test_negative(self):
        assert format_eta(-1) == "--:--"

    def test_infinity(self):
        assert format_eta(float("inf")) == "--:--"

    def test_nan(self):
        assert format_eta(float("nan")) == "--:--"

    def test_zero(self):
        assert format_eta(0) == "--:--"


class TestFormatPercentage:
    def test_zero(self):
        assert format_percentage(0) == "0.0%"

    def test_half(self):
        assert format_percentage(45.234) == "45.2%"

    def test_full(self):
        assert format_percentage(100) == "100.0%"

    def test_over_100(self):
        assert format_percentage(123.456) == "123.5%"
